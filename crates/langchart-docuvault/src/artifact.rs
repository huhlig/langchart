//! `ArtifactStore` implementation backed by a `docuvault` vault.
//!
//! # Threading model
//!
//! `docuvault::VaultHandle` is `Clone + Send + Sync` but its operations are
//! synchronous.  `VaultArtifactStore` calls them inside
//! `tokio::task::spawn_blocking` so they never block the async executor.
//!
//! # Artifact → vault entity mapping
//!
//! | `langchart` concept | `docuvault` concept |
//! |---|---|
//! | `ArtifactId` URI | `VaultEntityRef` (file / section / attachment) |
//! | `ArtifactVersion` tag `commit:…` | `CommitRef::Commit(CommitId)` |
//! | `ArtifactVersion` tag `session:…` | `CommitRef::SessionCursor { session_id, … }` |
//! | `propose()` | `ProposalRepository::create` → Draft |
//! | `commit()` | `ProposalRepository::validate` → `apply` |
//! | `list_proposals()` | `ProposalRepository::list` filtered by entity |
//!
//! # Session ownership
//!
//! Each langchart run owns at most one `SessionId`, stored in workflow data.
//! `VaultArtifactStore` accepts an optional `session_id` at construction time.
//! If absent, `propose` and `commit` will return `ArtifactError::Store` with a
//! clear message — the caller (`WorkflowHost`) is responsible for creating a
//! session before starting mutating runs.

use std::{io::Read, sync::Arc};

use async_trait::async_trait;
use langchart_adapters::artifact::{
    ArtifactContent, ArtifactError, ArtifactProposal, ArtifactStore, ProposalSummary,
};
use langchart_model::id::{ArtifactId, ArtifactVersion, ProposalId};
use tracing::instrument;

use docuvault::{
    model::{
        ids::{CheckpointId, CommitId, FileId, SectionId, SessionId},
        mutation::{Mutation, MutationRequest},
        proposal::ProposalStatus,
        provenance::{Actor, ActorType},
        section::SectionContentViewKind,
        selector::{CommitRef, FileSelector, SectionSelector},
    },
    proposal::ProposalRepository,
    read::{files, sections},
    store::VaultHandle,
};

use crate::ids::{VaultEntity, VaultEntityRef, VaultVersionRef};

// ── VaultArtifactStore ────────────────────────────────────────────────────────

/// A `langchart` `ArtifactStore` backed by a docuvault `VaultHandle`.
///
/// Construct one per vault run and share it via `Arc` across workflow activities
/// that operate on the same vault. Supply a `session_id` before any mutating
/// call.
#[derive(Clone)]
pub struct VaultArtifactStore {
    vault: Arc<VaultHandle>,
    /// Docuvault vault ID string — used as the default `vault_id` in ArtifactId URIs
    /// and for routing when multiple vaults are open.
    vault_id_str: String,
    /// Session ID for mutating operations. `None` → mutations return an error.
    session_id: Option<SessionId>,
}

impl VaultArtifactStore {
    /// Create a new store adapter.
    ///
    /// `vault` is the open vault handle. `session_id` may be supplied at
    /// construction or later via [`VaultArtifactStore::with_session`].
    pub fn new(vault: Arc<VaultHandle>, session_id: Option<SessionId>) -> Self {
        let vault_id_str = vault.manifest().vault_id.to_string();
        Self {
            vault,
            vault_id_str,
            session_id,
        }
    }

    /// Return a new store with `session_id` set.
    pub fn with_session(self, session_id: SessionId) -> Self {
        Self {
            session_id: Some(session_id),
            ..self
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Parse an `ArtifactId` URI and validate the vault_id matches this store.
    fn parse_entity(&self, id: &ArtifactId) -> Result<VaultEntityRef, ArtifactError> {
        let entity = VaultEntityRef::parse(&id.0).map_err(ArtifactError::Store)?;
        if entity.vault_id != self.vault_id_str {
            return Err(ArtifactError::Store(format!(
                "ArtifactId vault_id `{}` does not match this store's vault `{}`",
                entity.vault_id, self.vault_id_str
            )));
        }
        Ok(entity)
    }

    /// Convert a `VaultVersionRef` to a docuvault `CommitRef`.
    fn to_commit_ref(version: &VaultVersionRef) -> Result<CommitRef, ArtifactError> {
        match version {
            VaultVersionRef::Commit { commit_id } => {
                let id = commit_id.parse::<CommitId>().map_err(|e| {
                    ArtifactError::Store(format!("invalid commit_id `{commit_id}`: {e}"))
                })?;
                Ok(CommitRef::Commit(id))
            }
            VaultVersionRef::Session {
                session_id,
                operation_id,
            } => {
                let sid = session_id.parse::<SessionId>().map_err(|e| {
                    ArtifactError::Store(format!("invalid session_id `{session_id}`: {e}"))
                })?;
                let cursor = if operation_id.is_empty() {
                    None
                } else {
                    Some(operation_id.parse().map_err(|e| {
                        ArtifactError::Store(format!("invalid operation_id `{operation_id}`: {e}"))
                    })?)
                };
                Ok(CommitRef::SessionCursor {
                    session_id: sid,
                    operation_cursor: cursor,
                })
            }
            VaultVersionRef::Checkpoint { checkpoint_id } => checkpoint_id
                .parse::<CheckpointId>()
                .map(CommitRef::Checkpoint)
                .map_err(|e| {
                    ArtifactError::Store(format!("invalid checkpoint_id `{checkpoint_id}`: {e}"))
                }),
        }
    }

    /// Build a docuvault `Actor` for use in proposals and mutations.
    fn agent_actor(rationale: &str) -> Actor {
        Actor {
            actor_id: "langchart-agent".into(),
            actor_type: ActorType::Agent,
            display_name: Some(rationale.chars().take(80).collect()),
        }
    }

    fn mutation_targets_entity(entity: &VaultEntity, request: &MutationRequest) -> bool {
        match (entity, &request.mutation) {
            (
                VaultEntity::Section { section_id, .. },
                Mutation::ReplaceSectionBody {
                    section_id: actual, ..
                },
            ) => actual.to_string() == *section_id,
            (
                VaultEntity::Attachment { file_id, .. },
                Mutation::ReplaceAttachmentContent {
                    file_id: actual, ..
                },
            ) => actual.to_string() == *file_id,
            _ => false,
        }
    }

    fn proposal_touches_entity(entity: &VaultEntity, mutations: &[MutationRequest]) -> bool {
        mutations
            .iter()
            .any(|request| Self::mutation_targets_entity(entity, request))
    }

    fn proposal_exclusively_targets_entity(
        entity: &VaultEntity,
        mutations: &[MutationRequest],
    ) -> bool {
        !mutations.is_empty()
            && mutations
                .iter()
                .all(|request| Self::mutation_targets_entity(entity, request))
    }

    fn entity_exists_at_commit(
        vault: &VaultHandle,
        entity: &VaultEntity,
        commit_id: CommitId,
    ) -> Result<bool, ArtifactError> {
        let result = match entity {
            VaultEntity::File { file_id } => {
                let file_id = file_id.parse::<FileId>().map_err(|error| {
                    ArtifactError::Store(format!("invalid file_id `{file_id}`: {error}"))
                })?;
                files::get(
                    vault,
                    FileSelector::Id(file_id),
                    CommitRef::Commit(commit_id),
                    None,
                )
                .map(|_| true)
            }
            VaultEntity::Section {
                file_id,
                section_id,
            } => {
                let file_id = file_id.parse::<FileId>().map_err(|error| {
                    ArtifactError::Store(format!("invalid file_id `{file_id}`: {error}"))
                })?;
                let section_id = section_id.parse::<SectionId>().map_err(|error| {
                    ArtifactError::Store(format!("invalid section_id `{section_id}`: {error}"))
                })?;
                sections::get(
                    vault,
                    SectionSelector::EmbeddedId {
                        file: FileSelector::Id(file_id),
                        embedded_id: section_id.to_string(),
                    },
                    CommitRef::Commit(commit_id),
                    SectionContentViewKind::Body,
                )
                .map(|_| true)
            }
            VaultEntity::Attachment {
                file_id,
                attachment_path,
            } => {
                let file_id = file_id.parse::<FileId>().map_err(|error| {
                    ArtifactError::Store(format!("invalid file_id `{file_id}`: {error}"))
                })?;
                files::get(
                    vault,
                    FileSelector::Id(file_id),
                    CommitRef::Commit(commit_id),
                    None,
                )
                .map(|file| match file {
                    docuvault::model::file::File::Attachment(file) => {
                        file.path.to_string() == *attachment_path
                    }
                    docuvault::model::file::File::Markdown(_) => false,
                })
            }
        };

        match result {
            Ok(matches) => Ok(matches),
            Err(docuvault::model::VaultError::NotFound { .. }) => Ok(false),
            Err(error) => Err(ArtifactError::Store(error.to_string())),
        }
    }

    fn proposal_is_actionable(status: &ProposalStatus, validation_valid: Option<bool>) -> bool {
        match status {
            ProposalStatus::Draft => true,
            ProposalStatus::Validated => validation_valid == Some(true),
            ProposalStatus::Applied
            | ProposalStatus::Rejected
            | ProposalStatus::Conflicted
            | ProposalStatus::Expired => false,
        }
    }

    fn map_apply_error(
        session_id: SessionId,
        error: docuvault::model::VaultError,
    ) -> ArtifactError {
        match error {
            docuvault::model::VaultError::SessionCursorConflict {
                expected_cursor,
                actual_cursor,
                ..
            } => ArtifactError::VersionConflict {
                expected: ArtifactVersion::new(format!(
                    "session:{session_id}@{}",
                    expected_cursor.unwrap_or_default()
                )),
                actual: ArtifactVersion::new(format!(
                    "session:{session_id}@{}",
                    actual_cursor.unwrap_or_default()
                )),
            },
            other => ArtifactError::Store(other.to_string()),
        }
    }
}

#[async_trait]
impl ArtifactStore for VaultArtifactStore {
    /// Read an artifact at `version` (or `CommitRef::Published` if `None`).
    #[instrument(skip(self), fields(artifact_id = %id.0))]
    async fn read(
        &self,
        id: &ArtifactId,
        version: Option<&ArtifactVersion>,
    ) -> Result<ArtifactContent, ArtifactError> {
        let entity = self.parse_entity(id)?;

        let (commit_ref, artifact_version) = match version {
            None => {
                let vault = Arc::clone(&self.vault);
                let artifact_id = id.clone();
                let commit_id = tokio::task::spawn_blocking(move || {
                    vault
                        .refs()
                        .read_main()
                        .map_err(|error| ArtifactError::Store(error.to_string()))?
                        .map(|reference| reference.commit_id)
                        .ok_or(ArtifactError::NotFound(artifact_id))
                })
                .await
                .map_err(|error| ArtifactError::Store(error.to_string()))??;
                (
                    CommitRef::Commit(commit_id),
                    ArtifactVersion::new(format!("commit:{commit_id}")),
                )
            }
            Some(v) => {
                let vref = VaultVersionRef::parse(&v.0).map_err(ArtifactError::Store)?;
                (Self::to_commit_ref(&vref)?, v.clone())
            }
        };

        let vault = Arc::clone(&self.vault);
        let artifact_id = id.clone();

        match entity.entity {
            VaultEntity::File { file_id: fid_str } => {
                let fid = fid_str.parse::<FileId>().map_err(|e| {
                    ArtifactError::Store(format!("invalid file_id `{fid_str}`: {e}"))
                })?;
                let cr = commit_ref;
                let bytes = tokio::task::spawn_blocking(move || {
                    use docuvault::model::file::FileContentView;
                    let file = files::get(
                        &vault,
                        FileSelector::Id(fid),
                        cr,
                        Some(FileContentView::Content),
                    )
                    .map_err(|e| match e {
                        docuvault::model::VaultError::NotFound { .. } => {
                            ArtifactError::NotFound(artifact_id.clone())
                        }
                        other => ArtifactError::Store(other.to_string()),
                    })?;
                    use docuvault::model::file::File;
                    match file {
                        File::Markdown(f) => Ok(f.content.unwrap_or_default().into_bytes()),
                        File::Attachment(_) => Err(ArtifactError::Store(
                            "use the attachment path URI for attachment reads".into(),
                        )),
                    }
                })
                .await
                .map_err(|e| ArtifactError::Store(e.to_string()))??;

                Ok(ArtifactContent {
                    id: id.clone(),
                    version: artifact_version,
                    bytes,
                    content_type: "text/markdown".into(),
                })
            }

            VaultEntity::Section {
                file_id: fid_str,
                section_id: sid_str,
            } => {
                let fid = fid_str.parse::<FileId>().map_err(|e| {
                    ArtifactError::Store(format!("invalid file_id `{fid_str}`: {e}"))
                })?;
                let sid = sid_str.parse::<SectionId>().map_err(|e| {
                    ArtifactError::Store(format!("invalid section_id `{sid_str}`: {e}"))
                })?;
                let cr = commit_ref;
                let artifact_id2 = id.clone();
                let (bytes, content_type) = tokio::task::spawn_blocking(move || {
                    use docuvault::model::section::SectionContentView;
                    let (_, view) = sections::get(
                        &vault,
                        SectionSelector::EmbeddedId {
                            file: FileSelector::Id(fid),
                            embedded_id: sid.to_string(),
                        },
                        cr,
                        SectionContentViewKind::Body,
                    )
                    .map_err(|e| match e {
                        docuvault::model::VaultError::NotFound { .. } => {
                            ArtifactError::NotFound(artifact_id2.clone())
                        }
                        other => ArtifactError::Store(other.to_string()),
                    })?;
                    let body = match view {
                        SectionContentView::Body(b) => b,
                        SectionContentView::Subtree(s) => s,
                        _ => String::new(),
                    };
                    Ok::<_, ArtifactError>((body.into_bytes(), "text/markdown".to_owned()))
                })
                .await
                .map_err(|e| ArtifactError::Store(e.to_string()))??;

                Ok(ArtifactContent {
                    id: id.clone(),
                    version: artifact_version,
                    bytes,
                    content_type,
                })
            }

            VaultEntity::Attachment {
                file_id: fid_str,
                attachment_path,
            } => {
                let fid = fid_str.parse::<FileId>().map_err(|e| {
                    ArtifactError::Store(format!("invalid file_id `{fid_str}`: {e}"))
                })?;
                let cr = commit_ref;
                let artifact_id2 = id.clone();
                let (bytes, content_type) = tokio::task::spawn_blocking(move || {
                    let file = files::get(&vault, FileSelector::Id(fid), cr.clone(), None)
                        .map_err(|e| match e {
                            docuvault::model::VaultError::NotFound { .. } => {
                                ArtifactError::NotFound(artifact_id2.clone())
                            }
                            other => ArtifactError::Store(other.to_string()),
                        })?;
                    let docuvault::model::file::File::Attachment(file) = file else {
                        return Err(ArtifactError::NotFound(artifact_id2));
                    };
                    if file.path.to_string() != attachment_path {
                        return Err(ArtifactError::NotFound(artifact_id2));
                    }

                    let content_type = file.media_type;
                    let mut reader = files::open_attachment(&vault, FileSelector::Id(fid), cr)
                        .map_err(|e| match e {
                            docuvault::model::VaultError::NotFound { .. } => {
                                ArtifactError::NotFound(artifact_id2.clone())
                            }
                            other => ArtifactError::Store(other.to_string()),
                        })?;
                    let mut bytes = Vec::new();
                    reader
                        .read_to_end(&mut bytes)
                        .map_err(|e| ArtifactError::Store(e.to_string()))?;
                    reader
                        .finish()
                        .map_err(|e| ArtifactError::Store(e.to_string()))?;
                    Ok::<_, ArtifactError>((bytes, content_type))
                })
                .await
                .map_err(|e| ArtifactError::Store(e.to_string()))??;

                Ok(ArtifactContent {
                    id: id.clone(),
                    version: artifact_version,
                    bytes,
                    content_type,
                })
            }
        }
    }

    /// Create a mutation proposal in docuvault's `ProposalRepository`.
    #[instrument(skip(self), fields(artifact_id = %proposal.id.0))]
    async fn propose(&self, proposal: ArtifactProposal) -> Result<ProposalId, ArtifactError> {
        let session_id = self.session_id.ok_or_else(|| {
            ArtifactError::Store(
                "no session_id configured on VaultArtifactStore — call with_session() first".into(),
            )
        })?;

        let entity = self.parse_entity(&proposal.id)?;
        let artifact_id = proposal.id.clone();

        // Parse base_version → commit_id for the proposal base.
        let base_commit = {
            let vref =
                VaultVersionRef::parse(&proposal.base_version.0).map_err(ArtifactError::Store)?;
            match vref {
                VaultVersionRef::Commit { commit_id } => commit_id
                    .parse::<CommitId>()
                    .map_err(|e| ArtifactError::Store(format!("invalid base commit_id: {e}")))?,
                VaultVersionRef::Session { .. } | VaultVersionRef::Checkpoint { .. } => {
                    return Err(ArtifactError::Store(
                        "propose() requires a `commit:` base_version, not a session or checkpoint tag".into()
                    ));
                }
            }
        };

        // Build the docuvault `MutationRequest` from the proposal content.
        let mutation = match &entity.entity {
            VaultEntity::Section {
                section_id: sid_str,
                ..
            } => {
                let content = String::from_utf8(proposal.content.clone()).map_err(|_| {
                    ArtifactError::Store("section proposal content is not valid UTF-8".into())
                })?;
                let sid = sid_str.parse::<SectionId>().map_err(|e| {
                    ArtifactError::Store(format!("invalid section_id `{sid_str}`: {e}"))
                })?;
                Mutation::ReplaceSectionBody {
                    section_id: sid,
                    body: content,
                    patch: None,
                }
            }
            VaultEntity::File { .. } => {
                // Whole-file replacement is not a single Mutation variant in docuvault —
                // it would require a CreateFile or ReplaceFile mutation. For now, we require
                // section-level granularity for proposals.
                return Err(ArtifactError::Store(
                    "whole-file proposals require a section URI; file-level mutations are not yet supported via ArtifactStore".into()
                ));
            }
            VaultEntity::Attachment {
                file_id: fid_str, ..
            } => {
                let fid = fid_str.parse::<FileId>().map_err(|e| {
                    ArtifactError::Store(format!("invalid file_id `{fid_str}`: {e}"))
                })?;
                Mutation::ReplaceAttachmentContent {
                    file_id: fid,
                    media_type: Some(proposal.content_type.clone()),
                    bytes: proposal.content.clone(),
                }
            }
        };

        let mutation_id = docuvault::model::ids::MutationId::new();
        let mutation_request = MutationRequest {
            mutation_id,
            preconditions: docuvault::model::precondition::MutationPreconditions::default(),
            mutation,
        };

        let actor = Self::agent_actor(&proposal.rationale);
        let vault = Arc::clone(&self.vault);
        let _ = session_id; // consumed above

        let docuvault_proposal_id = tokio::task::spawn_blocking(move || {
            if !Self::entity_exists_at_commit(&vault, &entity.entity, base_commit)? {
                return Err(ArtifactError::NotFound(artifact_id));
            }
            let repo = ProposalRepository::from_vault(&vault);
            let p = repo
                .create(
                    base_commit,
                    None,
                    actor,
                    vec![mutation_request],
                    docuvault::model::provenance::Provenance::default(),
                )
                .map_err(|e| ArtifactError::Store(e.to_string()))?;
            Ok::<_, ArtifactError>(p.proposal_id)
        })
        .await
        .map_err(|e| ArtifactError::Store(e.to_string()))??;

        // Encode the docuvault ProposalId as a langchart ProposalId (same ULID string).
        Ok(ProposalId::new(docuvault_proposal_id.to_string()))
    }

    /// Validate and atomically apply a proposal; return the resulting session version.
    ///
    /// The docuvault flow (validate → apply) is collapsed into a single `commit()`
    /// call that auto-accepts a syntactically valid proposal.
    /// Semantic conflicts (e.g. concurrent session mutations) return `VersionConflict`.
    #[instrument(skip(self), fields(proposal_id = %proposal_id.0))]
    async fn commit(
        &self,
        artifact_id: &ArtifactId,
        proposal_id: &ProposalId,
        expected_base: &ArtifactVersion,
    ) -> Result<ArtifactVersion, ArtifactError> {
        let session_id = self.session_id.ok_or_else(|| {
            ArtifactError::Store(
                "no session_id configured on VaultArtifactStore — call with_session() first".into(),
            )
        })?;
        let entity = self.parse_entity(artifact_id)?.entity;
        let artifact_id = artifact_id.clone();

        // Parse and validate expected_base
        let expected_commit = {
            let vref = VaultVersionRef::parse(&expected_base.0).map_err(ArtifactError::Store)?;
            match vref {
                VaultVersionRef::Commit { commit_id } => {
                    commit_id.parse::<CommitId>().map_err(|e| {
                        ArtifactError::Store(format!("invalid expected commit_id: {e}"))
                    })?
                }
                _ => {
                    return Err(ArtifactError::Store(
                        "commit() requires a `commit:` expected_base version".into(),
                    ));
                }
            }
        };

        let pid_str = proposal_id.0.clone();
        let vault = Arc::clone(&self.vault);

        let session_ref = tokio::task::spawn_blocking(move || {
            let docuvault_pid = pid_str
                .parse::<docuvault::model::ids::ProposalId>()
                .map_err(|e| ArtifactError::Store(format!("invalid proposal_id: {e}")))?;
            let repo = ProposalRepository::from_vault(&vault);

            // Check all non-mutating preconditions before validation advances the
            // proposal from Draft to Validated. A rejected commit must remain
            // retryable with the correct artifact identity or base version.
            let proposal = repo
                .get(docuvault_pid)
                .map_err(|e| ArtifactError::Store(e.to_string()))?;

            if !Self::entity_exists_at_commit(&vault, &entity, proposal.base_commit)?
                || !Self::proposal_exclusively_targets_entity(&entity, &proposal.mutations)
            {
                return Err(ArtifactError::ProposalArtifactMismatch {
                    proposal_id: ProposalId::new(pid_str.clone()),
                    artifact_id,
                });
            }

            // Precondition check: base commit must match expected.
            if proposal.base_commit != expected_commit {
                return Err(ArtifactError::VersionConflict {
                    expected: ArtifactVersion::new(format!("commit:{}", expected_commit)),
                    actual: ArtifactVersion::new(format!("commit:{}", proposal.base_commit)),
                });
            }

            use docuvault::model::mutation::MutationTransaction;
            use docuvault::mutation::MutationEngine;
            use docuvault::session::SessionRepository;
            use std::collections::BTreeMap;

            let session_repo = SessionRepository::new(&vault);
            let session_ref = session_repo
                .get(session_id)
                .map_err(|e| ArtifactError::Store(e.to_string()))?;
            if session_ref.base_published_commit_id != proposal.base_commit {
                return Err(ArtifactError::VersionConflict {
                    expected: ArtifactVersion::new(format!("commit:{}", proposal.base_commit)),
                    actual: ArtifactVersion::new(format!(
                        "commit:{}",
                        session_ref.base_published_commit_id
                    )),
                });
            }

            // Step 1: validate drafts; proposals validated by another local
            // workflow may proceed only when their recorded result is valid.
            match &proposal.status {
                ProposalStatus::Draft => {
                    repo.validate(docuvault_pid, |_| vec![])
                        .map_err(|e| ArtifactError::Store(e.to_string()))?;
                }
                ProposalStatus::Validated
                    if proposal
                        .validation_result
                        .as_ref()
                        .is_some_and(|result| result.valid) => {}
                status => {
                    return Err(ArtifactError::Store(format!(
                        "proposal {docuvault_pid} is not valid and pending (status: {status:?})"
                    )));
                }
            }

            // Step 2: atomically apply via MutationEngine.
            repo.apply(docuvault_pid, |p| {
                let transaction = MutationTransaction {
                    base_commit_id: session_ref.base_published_commit_id,
                    session_id,
                    expected_cursor: session_ref.operation_cursor,
                    actor: p.actor.clone(),
                    message: p
                        .actor
                        .display_name
                        .clone()
                        .unwrap_or_else(|| "langchart proposal".into()),
                    mutations: p.mutations.clone(),
                    metadata: BTreeMap::new(),
                    provenance: p.provenance.clone(),
                };
                MutationEngine::new(&vault).apply(transaction).map(|_| ())
            })
            .map_err(|error| Self::map_apply_error(session_id, error))?;

            // Re-read the session to get the updated cursor for the returned ArtifactVersion.
            let updated_session = session_repo
                .get(session_id)
                .map_err(|e| ArtifactError::Store(e.to_string()))?;

            Ok::<_, ArtifactError>(updated_session)
        })
        .await
        .map_err(|e| ArtifactError::Store(e.to_string()))??;

        // Encode the result: session cursor tag so the caller can read the pending state.
        let op_id = session_ref
            .operation_cursor
            .map(|id| id.to_string())
            .unwrap_or_default();
        let version_tag = format!("session:{}@{}", session_id, op_id);

        Ok(ArtifactVersion::new(version_tag))
    }

    /// List pending proposals whose `base_commit` references the given artifact's vault entity.
    #[instrument(skip(self), fields(artifact_id = %artifact_id.0))]
    async fn list_proposals(
        &self,
        artifact_id: &ArtifactId,
    ) -> Result<Vec<ProposalSummary>, ArtifactError> {
        let entity = self.parse_entity(artifact_id)?;

        let vault = Arc::clone(&self.vault);
        let artifact_id_clone = artifact_id.clone();

        let summaries = tokio::task::spawn_blocking(move || {
            let repo = ProposalRepository::from_vault(&vault);
            let all = repo
                .list()
                .map_err(|e| ArtifactError::Store(e.to_string()))?;

            // Filter to actionable proposals whose mutations reference the
            // canonical entity at the proposal's base commit.
            let mut summaries = Vec::new();
            for proposal in all {
                if !Self::proposal_is_actionable(
                    &proposal.status,
                    proposal
                        .validation_result
                        .as_ref()
                        .map(|result| result.valid),
                ) || !Self::proposal_touches_entity(&entity.entity, &proposal.mutations)
                    || !Self::entity_exists_at_commit(&vault, &entity.entity, proposal.base_commit)?
                {
                    continue;
                }
                summaries.push(ProposalSummary {
                    proposal_id: ProposalId::new(proposal.proposal_id.to_string()),
                    artifact_id: artifact_id_clone.clone(),
                    base_version: ArtifactVersion::new(format!("commit:{}", proposal.base_commit)),
                    rationale: proposal.actor.display_name.clone().unwrap_or_default(),
                });
            }
            Ok::<_, ArtifactError>(summaries)
        })
        .await
        .map_err(|e| ArtifactError::Store(e.to_string()))??;

        Ok(summaries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use docuvault::{
        model::{
            ids::MutationId,
            precondition::MutationPreconditions,
            proposal::{DiagnosticSeverity, ProposalStatus, ValidationDiagnostic},
            vault::IndexedFile,
        },
        session::SessionRepository,
        store::{WorkingScope, detect, reconcile},
    };

    fn replace_section(section_id: SectionId) -> MutationRequest {
        MutationRequest {
            mutation_id: MutationId::new(),
            preconditions: MutationPreconditions::default(),
            mutation: Mutation::ReplaceSectionBody {
                section_id,
                body: "replacement".into(),
                patch: None,
            },
        }
    }

    #[test]
    fn commit_identity_requires_every_mutation_to_target_the_artifact() {
        let target = SectionId::new();
        let other = SectionId::new();
        let entity = VaultEntity::Section {
            file_id: FileId::new().to_string(),
            section_id: target.to_string(),
        };
        let matching = replace_section(target);
        let unrelated = replace_section(other);

        assert!(VaultArtifactStore::proposal_exclusively_targets_entity(
            &entity,
            std::slice::from_ref(&matching)
        ));
        assert!(!VaultArtifactStore::proposal_exclusively_targets_entity(
            &entity,
            &[matching, unrelated]
        ));
        assert!(!VaultArtifactStore::proposal_exclusively_targets_entity(
            &entity,
            &[]
        ));
    }

    #[test]
    fn session_cursor_conflict_preserves_artifact_conflict_semantics() {
        let session_id = SessionId::new();
        let error = docuvault::model::VaultError::SessionCursorConflict {
            session_id: session_id.to_string(),
            expected_revision: 1,
            expected_cursor: Some("expected".into()),
            actual_revision: Some(2),
            actual_cursor: Some("actual".into()),
        };

        let mapped = VaultArtifactStore::map_apply_error(session_id, error);
        assert!(matches!(
            mapped,
            ArtifactError::VersionConflict { expected, actual }
                if expected.0 == format!("session:{session_id}@expected")
                    && actual.0 == format!("session:{session_id}@actual")
        ));
    }

    #[tokio::test]
    async fn commit_applies_validated_proposal_to_session() {
        let root = tempfile::tempdir().unwrap();
        let vault = Arc::new(VaultHandle::create(root.path()).unwrap());
        std::fs::write(root.path().join("note.md"), "# Heading\nOriginal.\n").unwrap();
        std::fs::create_dir(root.path().join("assets")).unwrap();
        std::fs::write(
            root.path().join("assets").join("my % diagram-雪.bin"),
            b"asset",
        )
        .unwrap();
        let initial = reconcile(
            &vault,
            detect(&vault, WorkingScope::WholeVault).unwrap(),
            Vec::new(),
        )
        .unwrap();
        let (file_id, section_id) = vault
            .controls()
            .read_index()
            .unwrap()
            .files
            .iter()
            .find_map(|file| match file {
                IndexedFile::Markdown {
                    file_id, sections, ..
                } => Some((*file_id, sections[1].section_id)),
                IndexedFile::Attachment { .. } => None,
            })
            .unwrap();
        let (attachment_id, attachment_path) = vault
            .controls()
            .read_index()
            .unwrap()
            .files
            .iter()
            .find_map(|file| match file {
                IndexedFile::Attachment { file_id, path, .. } => Some((*file_id, path.to_string())),
                IndexedFile::Markdown { .. } => None,
            })
            .unwrap();
        let session = SessionRepository::new(&vault)
            .create(initial.commit_id)
            .unwrap();
        let store = VaultArtifactStore::new(Arc::clone(&vault), Some(session.session_id));
        let artifact_id = ArtifactId::new(format!(
            "vault://{}/file/{file_id}/section/{section_id}",
            vault.manifest().vault_id
        ));
        let latest = store.read(&artifact_id, None).await.unwrap();
        assert_eq!(latest.bytes, b"Original.\n");
        assert_eq!(latest.version.0, format!("commit:{}", initial.commit_id));
        let base_version = latest.version;

        let wrong_parent = ArtifactId::new(format!(
            "vault://{}/file/{}/section/{section_id}",
            vault.manifest().vault_id,
            FileId::new()
        ));
        assert!(matches!(
            store.read(&wrong_parent, None).await,
            Err(ArtifactError::NotFound(_))
        ));
        assert!(matches!(
            store
                .propose(ArtifactProposal {
                    id: wrong_parent.clone(),
                    base_version: base_version.clone(),
                    content: b"Wrong parent.\n".to_vec(),
                    content_type: "text/markdown".into(),
                    rationale: "wrong parent".into(),
                })
                .await,
            Err(ArtifactError::NotFound(_))
        ));

        let correct_attachment = VaultEntity::Attachment {
            file_id: attachment_id.to_string(),
            attachment_path: attachment_path.clone(),
        };
        let wrong_attachment_path = VaultEntity::Attachment {
            file_id: attachment_id.to_string(),
            attachment_path: format!("wrong-{attachment_path}"),
        };
        assert!(
            VaultArtifactStore::entity_exists_at_commit(
                &vault,
                &correct_attachment,
                initial.commit_id
            )
            .unwrap()
        );
        assert!(
            !VaultArtifactStore::entity_exists_at_commit(
                &vault,
                &wrong_attachment_path,
                initial.commit_id
            )
            .unwrap()
        );
        let attachment_artifact_id = ArtifactId::new(
            VaultEntityRef {
                vault_id: vault.manifest().vault_id.to_string(),
                entity: correct_attachment.clone(),
            }
            .to_uri(),
        );
        let published_attachment = store.read(&attachment_artifact_id, None).await.unwrap();
        assert_eq!(published_attachment.bytes, b"asset");
        assert_eq!(
            published_attachment.content_type,
            "application/octet-stream"
        );
        assert_eq!(published_attachment.version, base_version);
        let historical_attachment = store
            .read(&attachment_artifact_id, Some(&base_version))
            .await
            .unwrap();
        assert_eq!(historical_attachment.bytes, b"asset");

        let wrong_attachment_id = ArtifactId::new(
            VaultEntityRef {
                vault_id: vault.manifest().vault_id.to_string(),
                entity: wrong_attachment_path.clone(),
            }
            .to_uri(),
        );
        assert!(matches!(
            store.read(&wrong_attachment_id, Some(&base_version)).await,
            Err(ArtifactError::NotFound(_))
        ));

        let binary_content = vec![0x00, 0x80, 0xff, 0x01];
        let binary_proposal_id = store
            .propose(ArtifactProposal {
                id: attachment_artifact_id.clone(),
                base_version: base_version.clone(),
                content: binary_content.clone(),
                content_type: "application/octet-stream".into(),
                rationale: "binary replacement".into(),
            })
            .await
            .expect("binary attachment proposal");
        let binary_proposal = ProposalRepository::from_vault(&vault)
            .get(binary_proposal_id.0.parse().unwrap())
            .unwrap();
        assert!(matches!(
            binary_proposal.mutations.as_slice(),
            [MutationRequest {
                mutation: Mutation::ReplaceAttachmentContent { bytes, .. },
                ..
            }] if bytes == &binary_content
        ));
        assert_eq!(
            store
                .list_proposals(&attachment_artifact_id)
                .await
                .unwrap()
                .len(),
            1,
            "encoded attachment URI must match the canonical vault path"
        );
        let binary_session_version = store
            .commit(&attachment_artifact_id, &binary_proposal_id, &base_version)
            .await
            .expect("commit binary attachment proposal through encoded URI");
        assert!(
            binary_session_version
                .0
                .starts_with(&format!("session:{}@", session.session_id))
        );
        let pending_attachment = store
            .read(&attachment_artifact_id, Some(&binary_session_version))
            .await
            .unwrap();
        assert_eq!(pending_attachment.bytes, binary_content);
        assert_eq!(pending_attachment.content_type, "application/octet-stream");

        let binary_operation_id = match VaultVersionRef::parse(&binary_session_version.0).unwrap() {
            VaultVersionRef::Session { operation_id, .. } => operation_id.parse().unwrap(),
            other => panic!("expected session version, got {other:?}"),
        };
        let binary_operation = vault
            .objects()
            .read_checkpoint_operation(binary_operation_id)
            .unwrap();
        let attachment_checkpoint_id = binary_operation.attachment_checkpoint_ids[0];
        let attachment_checkpoint_version =
            ArtifactVersion::new(format!("checkpoint:{attachment_checkpoint_id}"));
        let checkpoint_attachment = store
            .read(
                &attachment_artifact_id,
                Some(&attachment_checkpoint_version),
            )
            .await
            .unwrap();
        assert_eq!(checkpoint_attachment.bytes, binary_content);
        assert!(
            store
                .list_proposals(&attachment_artifact_id)
                .await
                .unwrap()
                .is_empty(),
            "applied attachment proposal must not remain pending"
        );
        let proposal_id = store
            .propose(ArtifactProposal {
                id: artifact_id.clone(),
                base_version: base_version.clone(),
                content: b"Accepted body.\n".to_vec(),
                content_type: "text/markdown".into(),
                rationale: "test proposal".into(),
            })
            .await
            .unwrap();
        assert_eq!(store.list_proposals(&artifact_id).await.unwrap().len(), 1);
        assert!(
            store
                .list_proposals(&wrong_parent)
                .await
                .unwrap()
                .is_empty()
        );

        let wrong_artifact_id = ArtifactId::new(format!(
            "vault://{}/file/{file_id}",
            vault.manifest().vault_id
        ));
        let error = store
            .commit(&wrong_artifact_id, &proposal_id, &base_version)
            .await
            .expect_err("proposal ownership must be validated");
        assert!(matches!(
            error,
            ArtifactError::ProposalArtifactMismatch { .. }
        ));
        let error = store
            .commit(&wrong_parent, &proposal_id, &base_version)
            .await
            .expect_err("section parent file must be validated");
        assert!(matches!(
            error,
            ArtifactError::ProposalArtifactMismatch { .. }
        ));

        let docuvault_proposal_id = proposal_id
            .0
            .parse::<docuvault::model::ids::ProposalId>()
            .unwrap();
        ProposalRepository::from_vault(&vault)
            .validate(docuvault_proposal_id, |_| vec![])
            .unwrap();

        let session_version = store
            .commit(&artifact_id, &proposal_id, &base_version)
            .await
            .unwrap();

        assert!(
            session_version
                .0
                .starts_with(&format!("session:{}@", session.session_id))
        );
        let content = store
            .read(&artifact_id, Some(&session_version))
            .await
            .unwrap();
        assert_eq!(content.bytes, b"Accepted body.\n");
        let section_operation_id = match VaultVersionRef::parse(&session_version.0).unwrap() {
            VaultVersionRef::Session { operation_id, .. } => operation_id.parse().unwrap(),
            other => panic!("expected session version, got {other:?}"),
        };
        let section_operation = vault
            .objects()
            .read_checkpoint_operation(section_operation_id)
            .unwrap();
        let section_checkpoint_version = ArtifactVersion::new(format!(
            "checkpoint:{}",
            section_operation.section_checkpoint_ids[0]
        ));
        let checkpoint_content = store
            .read(&artifact_id, Some(&section_checkpoint_version))
            .await
            .unwrap();
        assert_eq!(checkpoint_content.bytes, b"Accepted body.\n");
        assert_eq!(
            ProposalRepository::from_vault(&vault)
                .get(docuvault_proposal_id)
                .unwrap()
                .status,
            ProposalStatus::Applied
        );
        assert!(
            store.list_proposals(&artifact_id).await.unwrap().is_empty(),
            "applied proposals must not remain in the pending list"
        );

        let invalid_id = store
            .propose(ArtifactProposal {
                id: artifact_id.clone(),
                base_version: base_version.clone(),
                content: b"Invalid body.\n".to_vec(),
                content_type: "text/markdown".into(),
                rationale: "invalid proposal".into(),
            })
            .await
            .unwrap()
            .0
            .parse::<docuvault::model::ids::ProposalId>()
            .unwrap();
        ProposalRepository::from_vault(&vault)
            .validate(invalid_id, |_| {
                vec![ValidationDiagnostic {
                    severity: DiagnosticSeverity::Error,
                    code: "invalid".into(),
                    message: "invalid proposal".into(),
                    mutation_id: None,
                    file_id: None,
                    section_id: None,
                    suggestion: None,
                }]
            })
            .unwrap();
        assert!(
            store.list_proposals(&artifact_id).await.unwrap().is_empty(),
            "failed validation must not appear actionable"
        );
    }
}
