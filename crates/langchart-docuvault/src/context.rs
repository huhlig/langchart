//! `ContextResolverStage` implementation: `VaultSearchStage`.
//!
//! Resolves `vault://`, `vault-search://`, and `vault-links://` selectors from
//! the `ContextPolicy` into `ContextItem` values backed by vault documents.
//!
//! ## Selector formats
//!
//! ```text
//! vault://<vault_id>/file/<file_id>                         — whole file
//! vault://<vault_id>/file/<file_id>/section/<section_id>    — one section
//! vault-search://<vault_id>?q=<query>&limit=5               — FTS search (requires docuvault "fts" feature)
//! vault-links://<vault_id>/file/<file_id>?depth=1           — link neighbourhood (stub)
//! ```
//!
//! ## Token budget enforcement (§4 of design doc)
//!
//! The stage stops adding items once `ctx.token_count` reaches 90% of the
//! `ContextPolicy::token_budget`.  The remaining 10% is reserved for the prompt
//! template and other downstream stages.
//!
//! ## Pinned source labels
//!
//! Every `ContextItem` carries a `source` label of the form:
//!
//! ```text
//! vault:<file_id>@head                                     — direct file read (head)
//! vault:<file_id>/<section_id>@<body_hash>                 — direct section read
//! vault-search:<query>                                     — search result
//! ```
//!
//! The `body_hash` from docuvault's `Section` is included so the observability
//! layer can record exactly which version of each section was in context for
//! each agent invocation, enabling deterministic replay.

use std::sync::Arc;

use async_trait::async_trait;
use langchart_adapters::context::{
    ContextAccumulator, ContextError, ContextItem, ContextResolverStage,
};
use langchart_model::id::RunId;
use langchart_model::policy::{ContextPolicy, ContextSource};
use tracing::instrument;

use docuvault::{
    model::{
        file::{File, FileContentView},
        ids::{CheckpointId, CommitId, FileId, SectionId, SessionId},
        section::SectionContentView,
        section::SectionContentViewKind,
        selector::{CommitRef, FileSelector, SectionSelector},
    },
    read::{files, sections},
    store::VaultHandle,
};

// ── VaultSearchStage ──────────────────────────────────────────────────────────

/// A `ContextResolverStage` that resolves vault entity selectors.
///
/// Construct one per open vault and register it in the `ContextResolverChain`.
/// Only selectors with a `vault://`, `vault-search://`, or `vault-links://`
/// URI scheme are processed; all others are passed through untouched.
#[derive(Clone)]
pub struct VaultSearchStage {
    vault: Arc<VaultHandle>,
    /// String form of the docuvault `VaultId` for this vault.
    vault_id_str: String,
}

impl VaultSearchStage {
    pub fn new(vault: Arc<VaultHandle>) -> Self {
        let vault_id_str = vault.manifest().vault_id.to_string();
        Self {
            vault,
            vault_id_str,
        }
    }

    /// Returns `true` if this selector should be handled by this stage.
    fn is_vault_selector(s: &str) -> bool {
        s.starts_with("vault://")
            || s.starts_with("vault-search://")
            || s.starts_with("vault-links://")
    }

    /// Returns `true` if the selector targets this vault (or the `*` wildcard).
    fn matches_this_vault(&self, selector: &str) -> bool {
        let rest = selector
            .strip_prefix("vault://")
            .or_else(|| selector.strip_prefix("vault-search://"))
            .or_else(|| selector.strip_prefix("vault-links://"));
        let Some(rest) = rest else { return false };
        let vault_id_in_selector = rest.split('/').next().unwrap_or("");
        vault_id_in_selector == self.vault_id_str || vault_id_in_selector == "*"
    }

    /// Estimate token count for a string (1 token ≈ 4 bytes).
    fn estimate_tokens(s: &str) -> u32 {
        ((s.len() as f32) / 4.0).ceil() as u32
    }

    /// Returns `true` when the budget is ≥ 90% exhausted.
    fn budget_exhausted(ctx: &ContextAccumulator, budget: Option<u32>) -> bool {
        let Some(budget) = budget else { return false };
        let threshold = budget.saturating_sub(budget / 10);
        ctx.token_count >= threshold
    }

    // ── vault:// — direct file/section reads ─────────────────────────────────

    fn resolve_vault_uri(
        vault: &VaultHandle,
        selector: &str,
        vault_id_str: &str,
        commit_ref: CommitRef,
        version_label: &str,
    ) -> Result<Option<ContextItem>, String> {
        use crate::ids::{VaultEntity, VaultEntityRef};

        let entity = VaultEntityRef::parse(selector)
            .map_err(|e| format!("failed to parse vault selector `{selector}`: {e}"))?;

        if entity.vault_id != vault_id_str && entity.vault_id != "*" {
            return Ok(None);
        }

        match entity.entity {
            VaultEntity::File { file_id: fid_str } => {
                let fid = fid_str
                    .parse::<FileId>()
                    .map_err(|e| format!("invalid file_id `{fid_str}`: {e}"))?;
                let file = files::get(
                    vault,
                    FileSelector::Id(fid),
                    commit_ref,
                    Some(FileContentView::Content),
                )
                .map_err(|e| e.to_string())?;

                let content = match file {
                    File::Markdown(f) => f.content.unwrap_or_default(),
                    File::Attachment(_) => return Ok(None),
                };
                let source = format!("vault:{fid_str}@{version_label}");
                let tokens = Self::estimate_tokens(&content);
                Ok(Some(ContextItem {
                    source,
                    content,
                    tokens,
                }))
            }

            VaultEntity::Section {
                file_id: fid_str,
                section_id: sid_str,
            } => {
                let fid = fid_str
                    .parse::<FileId>()
                    .map_err(|e| format!("invalid file_id `{fid_str}`: {e}"))?;
                let sid = sid_str
                    .parse::<SectionId>()
                    .map_err(|e| format!("invalid section_id `{sid_str}`: {e}"))?;

                let (section, view) = sections::get(
                    vault,
                    SectionSelector::Id(sid),
                    commit_ref,
                    SectionContentViewKind::Body,
                )
                .map_err(|e| e.to_string())?;

                if section.file_id != fid {
                    return Err(format!(
                        "section `{sid_str}` does not belong to file `{fid_str}`"
                    ));
                }

                let content = match view {
                    SectionContentView::Body(b) => b,
                    SectionContentView::Subtree(s) => s,
                    _ => String::new(),
                };

                // Pin the source label with the section's body_hash.
                let body_hash = section.body_hash.to_string();
                let source = format!("vault:{fid_str}/{sid_str}@{body_hash}");
                let tokens = Self::estimate_tokens(&content);
                Ok(Some(ContextItem {
                    source,
                    content,
                    tokens,
                }))
            }

            VaultEntity::Attachment { .. } => Ok(None),
        }
    }

    fn resolve_commit_ref(version: Option<&str>) -> Result<(CommitRef, String), String> {
        use crate::ids::VaultVersionRef;

        let Some(version) = version else {
            return Ok((CommitRef::Published, "published".into()));
        };
        let commit_ref = match VaultVersionRef::parse(version)? {
            VaultVersionRef::Commit { commit_id } => CommitRef::Commit(
                commit_id
                    .parse::<CommitId>()
                    .map_err(|e| format!("invalid commit_id `{commit_id}`: {e}"))?,
            ),
            VaultVersionRef::Session {
                session_id,
                operation_id,
            } => CommitRef::SessionCursor {
                session_id: session_id
                    .parse::<SessionId>()
                    .map_err(|e| format!("invalid session_id `{session_id}`: {e}"))?,
                operation_cursor: if operation_id.is_empty() {
                    None
                } else {
                    Some(
                        operation_id
                            .parse()
                            .map_err(|e| format!("invalid operation_id `{operation_id}`: {e}"))?,
                    )
                },
            },
            VaultVersionRef::Checkpoint { checkpoint_id } => CommitRef::Checkpoint(
                checkpoint_id
                    .parse::<CheckpointId>()
                    .map_err(|e| format!("invalid checkpoint_id `{checkpoint_id}`: {e}"))?,
            ),
        };
        Ok((commit_ref, version.to_owned()))
    }

    // ── vault-search:// — FTS search ─────────────────────────────────────────

    fn resolve_search_uri(
        _vault: &VaultHandle,
        selector: &str,
        _limit: u32,
    ) -> Result<Vec<ContextItem>, String> {
        // Parse: vault-search://<vault_id>?q=<query>&limit=5
        let after_scheme = selector
            .strip_prefix("vault-search://")
            .ok_or_else(|| format!("expected vault-search:// prefix in `{selector}`"))?;

        let query_str = after_scheme
            .split_once('?')
            .map(|(_, query)| query)
            .unwrap_or_default();
        let q = query_str
            .split('&')
            .find_map(|p| p.strip_prefix("q="))
            .unwrap_or_default();

        if q.is_empty() {
            return Ok(vec![]);
        }

        // TODO(collaborite-dhm): FTS requires the docuvault `fts` feature.
        // Wire `SearchCoordinator` here once the vault service layer exposes
        // an `FtsIndex` handle (tracked in collaborite-dhm follow-up).
        tracing::debug!(
            query = q,
            "vault-search:// FTS not yet wired; returning empty results (requires docuvault `fts` feature)"
        );
        Ok(vec![])
    }
}

#[async_trait]
impl ContextResolverStage for VaultSearchStage {
    #[instrument(skip(self, ctx), fields(vault_id = %self.vault_id_str))]
    async fn resolve(
        &self,
        policy: &ContextPolicy,
        _run_id: &RunId,
        ctx: &mut ContextAccumulator,
    ) -> Result<(), ContextError> {
        for source in &policy.sources {
            let (selector, version) = match source {
                ContextSource::Artifact { selector, version }
                    if Self::is_vault_selector(selector) =>
                {
                    (selector.as_str(), version.as_deref())
                }
                _ => continue,
            };

            if !self.matches_this_vault(selector) {
                return Err(ContextError::Stage {
                    stage: "VaultSearchStage",
                    message: format!(
                        "selector `{selector}` targets a different vault than `{}`",
                        self.vault_id_str
                    ),
                });
            }

            // Stop if token budget is exhausted.
            if Self::budget_exhausted(ctx, policy.token_budget) {
                tracing::debug!(
                    vault_id = %self.vault_id_str,
                    "token budget exhausted; stopping vault context resolution"
                );
                break;
            }

            let vault = Arc::clone(&self.vault);
            let vault_id_str = self.vault_id_str.clone();

            let items: Vec<ContextItem> = if selector.starts_with("vault-search://") {
                let selector = selector.to_owned();
                let search_limit = policy.token_budget.map(|b| (b / 500).max(5)).unwrap_or(10);
                tokio::task::spawn_blocking(move || {
                    Self::resolve_search_uri(&vault, &selector, search_limit)
                })
                .await
                .map_err(|e| ContextError::Stage {
                    stage: "VaultSearchStage",
                    message: e.to_string(),
                })?
                .map_err(|e| ContextError::Stage {
                    stage: "VaultSearchStage",
                    message: e,
                })?
            } else if selector.starts_with("vault-links://") {
                // Link neighbourhood — stub; returns empty until Phase 5.
                tracing::debug!(selector, "vault-links:// not yet implemented; skipping");
                vec![]
            } else {
                // Direct vault:// file or section read.
                let selector = selector.to_owned();
                let (commit_ref, version_label) =
                    Self::resolve_commit_ref(version).map_err(|e| ContextError::Stage {
                        stage: "VaultSearchStage",
                        message: e,
                    })?;
                let result = tokio::task::spawn_blocking(move || {
                    Self::resolve_vault_uri(
                        &vault,
                        &selector,
                        &vault_id_str,
                        commit_ref,
                        &version_label,
                    )
                })
                .await
                .map_err(|e| ContextError::Stage {
                    stage: "VaultSearchStage",
                    message: e.to_string(),
                })?
                .map_err(|e| ContextError::Stage {
                    stage: "VaultSearchStage",
                    message: e,
                })?;
                result.into_iter().collect()
            };

            // Push items respecting token budget.
            for item in items {
                if Self::budget_exhausted(ctx, policy.token_budget) {
                    break;
                }
                ctx.push(item);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::VaultVersionRef;
    use docuvault::model::ids::{CheckpointId, CommitId};

    #[test]
    fn artifact_commit_version_becomes_exact_commit_ref() {
        let commit_id = format!("commit_{}", "01".repeat(32))
            .parse::<CommitId>()
            .expect("valid test commit id");
        let version = VaultVersionRef::Commit {
            commit_id: commit_id.to_string(),
        }
        .to_tag();

        let (commit_ref, label) =
            VaultSearchStage::resolve_commit_ref(Some(&version)).expect("valid version");
        assert_eq!(commit_ref, CommitRef::Commit(commit_id));
        assert_eq!(label, version);
    }

    #[test]
    fn artifact_checkpoint_version_becomes_exact_checkpoint_ref() {
        let checkpoint_id = format!("checkpoint_{}", "02".repeat(32))
            .parse::<CheckpointId>()
            .expect("valid test checkpoint id");
        let version = format!("checkpoint:{checkpoint_id}");

        let (commit_ref, label) =
            VaultSearchStage::resolve_commit_ref(Some(&version)).expect("valid version");
        assert_eq!(commit_ref, CommitRef::Checkpoint(checkpoint_id));
        assert_eq!(label, version);
    }

    #[test]
    fn malformed_artifact_version_is_rejected() {
        let error = VaultSearchStage::resolve_commit_ref(Some("latest"))
            .expect_err("unversioned aliases must not silently read published content");
        assert!(error.contains("unknown version tag prefix"));
    }
}
