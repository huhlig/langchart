//! File-system backed [`ArtifactStore`] for langchart.
//!
//! # Layout
//!
//! ```text
//! <root>/
//!   <artifact_id>/
//!     latest.txt              — current version ULID (plain text)
//!     <version_ulid>.content  — raw content bytes
//!     <version_ulid>.meta.json — {"content_type":"…","created_at":"…"}
//!     <proposal_id>.proposal.json — serialised ArtifactProposal JSON
//! ```
//!
//! Writes are made atomic on a best-effort basis: content is written to a
//! `.tmp` file then renamed into place. `latest.txt` is updated last so a
//! crash mid-write leaves the previous version intact.
//!
//! # Example
//!
//! ```no_run
//! use langchart_artifact_fs::FsArtifactStore;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let store = FsArtifactStore::open("/var/lib/langchart/artifacts").await?;
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use langchart_adapters::artifact::{
    ArtifactContent, ArtifactError, ArtifactProposal, ArtifactStore, ProposalSummary,
};
use langchart_model::id::{ArtifactId, ArtifactVersion, ProposalId};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use tokio::io::AsyncWriteExt;
use ulid::Ulid;

const GENESIS_VERSION: &str = "none";

// ── Metadata sidecar ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct VersionMeta {
    content_type: String,
    created_at: String,
}

// ── FsArtifactStore ──────────────────────────────────────────────────────────

/// A file-system backed [`ArtifactStore`].
///
/// Create with [`FsArtifactStore::open`] — the root directory is created if it
/// does not already exist.
#[derive(Clone)]
pub struct FsArtifactStore {
    root: PathBuf,
}

impl FsArtifactStore {
    /// Open (or create) an artifact store rooted at `root_dir`.
    pub async fn open(root_dir: impl AsRef<Path>) -> Result<Self, ArtifactError> {
        tokio::fs::create_dir_all(root_dir.as_ref())
            .await
            .map_err(|e| ArtifactError::Store(format!("cannot create root dir: {e}")))?;
        let root = tokio::fs::canonicalize(root_dir.as_ref())
            .await
            .map_err(|e| ArtifactError::Store(format!("cannot resolve root dir: {e}")))?;
        Ok(Self { root })
    }

    fn artifact_dir(&self, id: &ArtifactId) -> Result<PathBuf, ArtifactError> {
        Ok(self.root.join(checked_component("artifact", &id.0)?))
    }

    fn content_path(
        &self,
        id: &ArtifactId,
        version: &ArtifactVersion,
    ) -> Result<PathBuf, ArtifactError> {
        let version = checked_component("artifact version", &version.0)?;
        Ok(self.artifact_dir(id)?.join(format!("{version}.content")))
    }

    fn meta_path(
        &self,
        id: &ArtifactId,
        version: &ArtifactVersion,
    ) -> Result<PathBuf, ArtifactError> {
        let version = checked_component("artifact version", &version.0)?;
        Ok(self.artifact_dir(id)?.join(format!("{version}.meta.json")))
    }

    fn latest_path(&self, id: &ArtifactId) -> Result<PathBuf, ArtifactError> {
        Ok(self.artifact_dir(id)?.join("latest.txt"))
    }

    fn proposal_path(
        &self,
        id: &ArtifactId,
        proposal_id: &ProposalId,
    ) -> Result<PathBuf, ArtifactError> {
        let proposal_id = checked_component("proposal", &proposal_id.0)?;
        Ok(self
            .artifact_dir(id)?
            .join(format!("{proposal_id}.proposal.json")))
    }

    async fn read_latest_version(&self, id: &ArtifactId) -> Result<ArtifactVersion, ArtifactError> {
        self.validate_artifact_dir(id).await?;
        let path = self.latest_path(id)?;
        let raw = tokio::fs::read_to_string(&path)
            .await
            .map_err(|error| artifact_read_error(id, "read latest version", error))?;
        Ok(ArtifactVersion(raw.trim().to_owned()))
    }

    async fn ensure_artifact_dir(&self, id: &ArtifactId) -> Result<(), ArtifactError> {
        let dir = self.artifact_dir(id)?;
        match tokio::fs::create_dir(&dir).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(ArtifactError::Store(format!(
                    "cannot create artifact dir: {error}"
                )));
            }
        }

        self.validate_artifact_dir(id).await?;
        Ok(())
    }

    async fn validate_artifact_dir(&self, id: &ArtifactId) -> Result<PathBuf, ArtifactError> {
        let dir = self.artifact_dir(id)?;
        let metadata = tokio::fs::symlink_metadata(&dir)
            .await
            .map_err(|error| artifact_read_error(id, "inspect artifact dir", error))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(ArtifactError::Store(
                "artifact path must be a real directory, not a link or file".into(),
            ));
        }

        let canonical_root = tokio::fs::canonicalize(&self.root)
            .await
            .map_err(|error| ArtifactError::Store(format!("resolve artifact root: {error}")))?;
        let canonical_dir = tokio::fs::canonicalize(&dir)
            .await
            .map_err(|error| ArtifactError::Store(format!("resolve artifact dir: {error}")))?;
        if !canonical_dir.starts_with(&canonical_root) {
            return Err(ArtifactError::Store(
                "artifact directory resolves outside the configured root".into(),
            ));
        }

        Ok(dir)
    }

    async fn acquire_commit_lock(&self) -> Result<std::fs::File, ArtifactError> {
        let lock_path = self.root.join(".commit.lock");
        tokio::task::spawn_blocking(move || {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(lock_path)
                .map_err(|e| ArtifactError::Store(format!("open commit lock: {e}")))?;
            fs2::FileExt::lock_exclusive(&file)
                .map_err(|e| ArtifactError::Store(format!("acquire commit lock: {e}")))?;
            Ok(file)
        })
        .await
        .map_err(|e| ArtifactError::Store(format!("commit lock task failed: {e}")))?
    }
}

#[async_trait]
impl ArtifactStore for FsArtifactStore {
    async fn read(
        &self,
        id: &ArtifactId,
        version: Option<&ArtifactVersion>,
    ) -> Result<ArtifactContent, ArtifactError> {
        self.validate_artifact_dir(id).await?;
        let ver = match version {
            Some(v) => v.clone(),
            None => self.read_latest_version(id).await?,
        };

        let bytes = tokio::fs::read(self.content_path(id, &ver)?)
            .await
            .map_err(|error| artifact_read_error(id, "read artifact content", error))?;

        let meta_raw = tokio::fs::read(self.meta_path(id, &ver)?)
            .await
            .map_err(|error| artifact_read_error(id, "read artifact metadata", error))?;

        let meta: VersionMeta = serde_json::from_slice(&meta_raw)
            .map_err(|e| ArtifactError::Store(format!("corrupt metadata: {e}")))?;

        Ok(ArtifactContent {
            id: id.clone(),
            version: ver,
            bytes,
            content_type: meta.content_type,
        })
    }

    async fn propose(&self, proposal: ArtifactProposal) -> Result<ProposalId, ArtifactError> {
        self.ensure_artifact_dir(&proposal.id).await?;

        let proposal_id = ProposalId(Ulid::generate().to_string());
        let path = self.proposal_path(&proposal.id, &proposal_id)?;

        let json = serde_json::to_vec(&proposal)
            .map_err(|e| ArtifactError::Store(format!("serialize proposal: {e}")))?;

        atomic_write(&path, &json).await?;

        Ok(proposal_id)
    }

    async fn commit(
        &self,
        artifact_id: &ArtifactId,
        proposal_id: &ProposalId,
        expected_base: &ArtifactVersion,
    ) -> Result<ArtifactVersion, ArtifactError> {
        let _commit_guard = self.acquire_commit_lock().await?;

        // Load the proposal. We need to know the artifact_id to find the file;
        // scan for a matching proposal file under any artifact directory.
        let proposal = self.find_proposal(proposal_id).await?;

        if proposal.id != *artifact_id {
            return Err(ArtifactError::ProposalArtifactMismatch {
                proposal_id: proposal_id.clone(),
                artifact_id: artifact_id.clone(),
            });
        }

        if proposal.base_version != *expected_base {
            return Err(ArtifactError::VersionConflict {
                expected: proposal.base_version,
                actual: expected_base.clone(),
            });
        }

        // Optimistic concurrency: check that the current latest version matches
        // the base the proposal was derived from.
        let current = self.read_latest_version(&proposal.id).await;
        match current {
            Ok(ref v) if v != &proposal.base_version => {
                return Err(ArtifactError::VersionConflict {
                    expected: proposal.base_version,
                    actual: v.clone(),
                });
            }
            Ok(_) => {}
            Err(ArtifactError::NotFound(_)) if proposal.base_version.0 == GENESIS_VERSION => {}
            Err(ArtifactError::NotFound(_)) => {
                return Err(ArtifactError::VersionConflict {
                    expected: proposal.base_version,
                    actual: ArtifactVersion(GENESIS_VERSION.into()),
                });
            }
            Err(e) => return Err(e),
        }

        let new_version = ArtifactVersion(Ulid::generate().to_string());
        self.ensure_artifact_dir(&proposal.id).await?;

        // Write content.
        atomic_write(
            &self.content_path(&proposal.id, &new_version)?,
            &proposal.content,
        )
        .await?;

        // Write metadata sidecar.
        let meta = VersionMeta {
            content_type: proposal.content_type.clone(),
            created_at: chrono_now(),
        };
        let meta_json = serde_json::to_vec(&meta)
            .map_err(|e| ArtifactError::Store(format!("serialize meta: {e}")))?;
        atomic_write(&self.meta_path(&proposal.id, &new_version)?, &meta_json).await?;

        // Update latest pointer last (so a crash leaves the previous version intact).
        atomic_write(&self.latest_path(&proposal.id)?, new_version.0.as_bytes()).await?;

        // Remove the committed proposal file.
        let _ = tokio::fs::remove_file(self.proposal_path(&proposal.id, proposal_id)?).await;

        Ok(new_version)
    }

    async fn list_proposals(
        &self,
        artifact_id: &ArtifactId,
    ) -> Result<Vec<ProposalSummary>, ArtifactError> {
        let dir = match self.validate_artifact_dir(artifact_id).await {
            Ok(dir) => dir,
            Err(ArtifactError::NotFound(_)) => return Ok(vec![]),
            Err(error) => return Err(error),
        };
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(error) => {
                return Err(ArtifactError::Store(format!(
                    "read artifact directory: {error}"
                )));
            }
        };

        let mut summaries = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| ArtifactError::Store(format!("read dir: {e}")))?
        {
            let fname = entry.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".proposal.json") {
                continue;
            }
            let proposal_id_str = fname.trim_end_matches(".proposal.json");
            let proposal_id = ProposalId(proposal_id_str.to_owned());

            let raw = tokio::fs::read(entry.path())
                .await
                .map_err(|e| ArtifactError::Store(format!("read proposal: {e}")))?;

            let proposal: ArtifactProposal = serde_json::from_slice(&raw)
                .map_err(|e| ArtifactError::Store(format!("corrupt proposal: {e}")))?;

            summaries.push(ProposalSummary {
                proposal_id,
                artifact_id: artifact_id.clone(),
                base_version: proposal.base_version,
                rationale: proposal.rationale,
            });
        }

        Ok(summaries)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

impl FsArtifactStore {
    /// Scan all artifact subdirectories for a proposal file matching `proposal_id`.
    async fn find_proposal(
        &self,
        proposal_id: &ProposalId,
    ) -> Result<ArtifactProposal, ArtifactError> {
        checked_component("proposal", &proposal_id.0)?;
        let mut entries = tokio::fs::read_dir(&self.root)
            .await
            .map_err(|e| ArtifactError::Store(format!("read root: {e}")))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| ArtifactError::Store(format!("read dir: {e}")))?
        {
            let file_type = entry
                .file_type()
                .await
                .map_err(|e| ArtifactError::Store(format!("read entry type: {e}")))?;
            if !file_type.is_dir() || file_type.is_symlink() {
                continue;
            }
            let artifact_id = ArtifactId(entry.file_name().to_string_lossy().to_string());
            let path = self.proposal_path(&artifact_id, proposal_id)?;
            if tokio::fs::metadata(&path).await.is_ok() {
                let raw = tokio::fs::read(&path)
                    .await
                    .map_err(|e| ArtifactError::Store(format!("read proposal: {e}")))?;
                return serde_json::from_slice(&raw)
                    .map_err(|e| ArtifactError::Store(format!("corrupt proposal: {e}")));
            }
        }

        Err(ArtifactError::NotFound(ArtifactId(proposal_id.0.clone())))
    }
}

fn checked_component<'a>(kind: &str, value: &'a str) -> Result<&'a str, ArtifactError> {
    let mut components = Path::new(value).components();
    let is_single_normal =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if value.is_empty()
        || value == "."
        || value == ".."
        || value
            .chars()
            .any(|character| matches!(character, '/' | '\\' | ':' | '\0'))
        || !is_single_normal
    {
        return Err(ArtifactError::Store(format!(
            "invalid {kind} id: expected one safe path component"
        )));
    }
    Ok(value)
}

fn artifact_read_error(id: &ArtifactId, operation: &str, error: std::io::Error) -> ArtifactError {
    if error.kind() == std::io::ErrorKind::NotFound {
        ArtifactError::NotFound(id.clone())
    } else {
        ArtifactError::Store(format!("{operation}: {error}"))
    }
}

/// Write `data` to `path` atomically: write to a `.tmp` sidecar first,
/// then rename into place.
async fn atomic_write(path: &Path, data: &[u8]) -> Result<(), ArtifactError> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ArtifactError::Store("invalid destination filename".into()))?;
    let tmp = path.with_file_name(format!(".{file_name}.{}.tmp", Ulid::generate()));
    let mut f = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .await
        .map_err(|e| ArtifactError::Store(format!("create tmp: {e}")))?;
    f.write_all(data)
        .await
        .map_err(|e| ArtifactError::Store(format!("write tmp: {e}")))?;
    f.sync_all()
        .await
        .map_err(|e| ArtifactError::Store(format!("sync tmp: {e}")))?;
    drop(f);
    if let Err(error) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(ArtifactError::Store(format!("rename: {error}")));
    }
    Ok(())
}

fn chrono_now() -> String {
    // Simple RFC 3339 timestamp without pulling in chrono.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as yyyy-mm-ddTHH:MM:SSZ (approximate — no sub-second precision).
    let secs_per_day = 86400u64;
    let days = secs / secs_per_day;
    let rem = secs % secs_per_day;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    // Gregorian calendar approximation (good enough for audit timestamps).
    let (y, mo, d) = days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Days since 1970-01-01.
    let mut y = 1970u64;
    loop {
        let leap = (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400);
        let dy = if leap { 366 } else { 365 };
        if days < dy {
            break;
        }
        days -= dy;
        y += 1;
    }
    let leap = (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mo = 1u64;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        mo += 1;
    }
    (y, mo, days + 1)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_store() -> (FsArtifactStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FsArtifactStore::open(dir.path()).await.expect("open");
        (store, dir)
    }

    fn art(id: &str) -> ArtifactId {
        ArtifactId(id.into())
    }
    fn ver(v: &str) -> ArtifactVersion {
        ArtifactVersion(v.into())
    }

    // ── read/write round-trip ────────────────────────────────────────────────

    /// Propose + commit → read latest returns the committed content.
    #[tokio::test]
    async fn propose_commit_read_round_trip() {
        let (store, _dir) = temp_store().await;
        let id = art("doc1");

        let proposal = ArtifactProposal {
            id: id.clone(),
            base_version: ver("none"),
            content: b"hello artifact".to_vec(),
            content_type: "text/plain".into(),
            rationale: "initial version".into(),
        };
        let pid = store.propose(proposal).await.expect("propose");
        let new_ver = store.commit(&id, &pid, &ver("none")).await.expect("commit");

        let content = store.read(&id, None).await.expect("read latest");
        assert_eq!(content.bytes, b"hello artifact");
        assert_eq!(content.content_type, "text/plain");
        assert_eq!(content.version, new_ver);
    }

    #[tokio::test]
    async fn commit_rejects_an_artifact_id_that_does_not_own_the_proposal() {
        let (store, _dir) = temp_store().await;
        let proposal_artifact = art("report");
        let proposal_id = store
            .propose(ArtifactProposal {
                id: proposal_artifact.clone(),
                base_version: ver(GENESIS_VERSION),
                content: b"content".to_vec(),
                content_type: "text/plain".into(),
                rationale: "test".into(),
            })
            .await
            .unwrap();

        let error = store
            .commit(&art("different"), &proposal_id, &ver(GENESIS_VERSION))
            .await
            .expect_err("artifact identity must be validated by the store");

        assert!(matches!(
            error,
            ArtifactError::ProposalArtifactMismatch { .. }
        ));
        assert_eq!(
            store
                .list_proposals(&proposal_artifact)
                .await
                .unwrap()
                .len(),
            1,
            "a mismatched commit must leave the proposal pending"
        );
    }

    /// Read a specific version by passing Some(version).
    #[tokio::test]
    async fn read_specific_version() {
        let (store, _dir) = temp_store().await;
        let id = art("doc2");

        let p1 = ArtifactProposal {
            id: id.clone(),
            base_version: ver("none"),
            content: b"v1".to_vec(),
            content_type: "text/plain".into(),
            rationale: "v1".into(),
        };
        let pid1 = store.propose(p1).await.unwrap();
        let v1 = store.commit(&id, &pid1, &ver("none")).await.unwrap();

        let p2 = ArtifactProposal {
            id: id.clone(),
            base_version: v1.clone(),
            content: b"v2".to_vec(),
            content_type: "text/plain".into(),
            rationale: "v2".into(),
        };
        let pid2 = store.propose(p2).await.unwrap();
        let _v2 = store.commit(&id, &pid2, &v1).await.unwrap();

        // Read v1 by specifying it explicitly.
        let content = store.read(&id, Some(&v1)).await.expect("read v1");
        assert_eq!(content.bytes, b"v1");
    }

    // ── list_proposals ───────────────────────────────────────────────────────

    /// Pending proposals are listed; committed proposals are removed from the list.
    #[tokio::test]
    async fn list_proposals_shows_pending_not_committed() {
        let (store, _dir) = temp_store().await;
        let id = art("doc3");

        let p = ArtifactProposal {
            id: id.clone(),
            base_version: ver("none"),
            content: b"x".to_vec(),
            content_type: "text/plain".into(),
            rationale: "pending".into(),
        };
        let pid = store.propose(p).await.unwrap();

        // Before commit: listed.
        let pending = store.list_proposals(&id).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].proposal_id, pid);
        assert_eq!(pending[0].rationale, "pending");

        // After commit: no longer listed.
        store.commit(&id, &pid, &ver("none")).await.unwrap();
        let after = store.list_proposals(&id).await.unwrap();
        assert!(after.is_empty(), "committed proposal must be removed");
    }

    // ── conflict detection ───────────────────────────────────────────────────

    /// Committing with a stale base version returns VersionConflict.
    #[tokio::test]
    async fn commit_with_stale_base_returns_conflict() {
        let (store, _dir) = temp_store().await;
        let id = art("doc4");

        // First commit.
        let p1 = ArtifactProposal {
            id: id.clone(),
            base_version: ver("none"),
            content: b"original".to_vec(),
            content_type: "text/plain".into(),
            rationale: "original".into(),
        };
        let pid1 = store.propose(p1).await.unwrap();
        let v1 = store.commit(&id, &pid1, &ver("none")).await.unwrap();

        // Two concurrent proposals both based on v1.
        let pa = ArtifactProposal {
            id: id.clone(),
            base_version: v1.clone(),
            content: b"branch-a".to_vec(),
            content_type: "text/plain".into(),
            rationale: "branch-a".into(),
        };
        let pb = ArtifactProposal {
            id: id.clone(),
            base_version: v1.clone(),
            content: b"branch-b".to_vec(),
            content_type: "text/plain".into(),
            rationale: "branch-b".into(),
        };
        let pida = store.propose(pa).await.unwrap();
        let pidb = store.propose(pb).await.unwrap();

        // First one wins.
        store
            .commit(&id, &pida, &v1)
            .await
            .expect("first commit ok");

        // Second one should conflict (current latest != v1 anymore).
        let result = store.commit(&id, &pidb, &v1).await;
        assert!(
            matches!(result, Err(ArtifactError::VersionConflict { .. })),
            "expected VersionConflict, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn rejects_ids_that_are_not_safe_path_components() {
        let (store, dir) = temp_store().await;

        for invalid in ["../escape", "..\\escape", "/absolute", "C:\\absolute", ""] {
            let proposal = ArtifactProposal {
                id: art(invalid),
                base_version: ver("none"),
                content: b"escape".to_vec(),
                content_type: "text/plain".into(),
                rationale: "invalid id".into(),
            };
            assert!(
                store.propose(proposal).await.is_err(),
                "accepted {invalid:?}"
            );
        }

        assert!(
            store
                .read(&art("safe"), Some(&ver("../version")))
                .await
                .is_err()
        );
        assert!(
            store
                .commit(&art("safe"), &ProposalId::new("../proposal"), &ver("none"),)
                .await
                .is_err()
        );

        assert!(!dir.path().join("escape").exists());
    }

    #[tokio::test]
    async fn concurrent_commits_allow_exactly_one_writer() {
        let (store, dir) = temp_store().await;
        let second_store = FsArtifactStore::open(dir.path()).await.unwrap();
        let id = art("concurrent");
        let initial = ArtifactProposal {
            id: id.clone(),
            base_version: ver("none"),
            content: b"initial".to_vec(),
            content_type: "text/plain".into(),
            rationale: "initial".into(),
        };
        let initial_id = store.propose(initial).await.unwrap();
        let base = store.commit(&id, &initial_id, &ver("none")).await.unwrap();

        let proposal = |content: &[u8]| ArtifactProposal {
            id: id.clone(),
            base_version: base.clone(),
            content: content.to_vec(),
            content_type: "text/plain".into(),
            rationale: "concurrent".into(),
        };
        let left = store.propose(proposal(b"left")).await.unwrap();
        let right = store.propose(proposal(b"right")).await.unwrap();

        let (left_result, right_result) = tokio::join!(
            store.commit(&id, &left, &base),
            second_store.commit(&id, &right, &base),
        );
        let results = [left_result, right_result];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(ArtifactError::VersionConflict { .. })))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn caller_cannot_rebase_a_stale_proposal() {
        let (store, _dir) = temp_store().await;
        let id = art("lineage");
        let initial_id = store
            .propose(ArtifactProposal {
                id: id.clone(),
                base_version: ver(GENESIS_VERSION),
                content: b"v1".to_vec(),
                content_type: "text/plain".into(),
                rationale: "initial".into(),
            })
            .await
            .unwrap();
        let v1 = store
            .commit(&id, &initial_id, &ver(GENESIS_VERSION))
            .await
            .unwrap();

        let proposal = |content: &[u8]| ArtifactProposal {
            id: id.clone(),
            base_version: v1.clone(),
            content: content.to_vec(),
            content_type: "text/plain".into(),
            rationale: "concurrent edit".into(),
        };
        let stale_id = store.propose(proposal(b"stale")).await.unwrap();
        let winning_id = store.propose(proposal(b"winner")).await.unwrap();
        let v2 = store.commit(&id, &winning_id, &v1).await.unwrap();

        let result = store.commit(&id, &stale_id, &v2).await;
        assert!(matches!(result, Err(ArtifactError::VersionConflict { .. })));
        assert_eq!(store.read(&id, None).await.unwrap().bytes, b"winner");
    }

    #[tokio::test]
    async fn first_commit_requires_the_genesis_base() {
        let (store, _dir) = temp_store().await;
        let id = art("new-artifact");
        let proposal_id = store
            .propose(ArtifactProposal {
                id: id.clone(),
                base_version: ver("invented"),
                content: b"content".to_vec(),
                content_type: "text/plain".into(),
                rationale: "invalid base".into(),
            })
            .await
            .unwrap();

        assert!(matches!(
            store.commit(&id, &proposal_id, &ver("invented")).await,
            Err(ArtifactError::VersionConflict { .. })
        ));
    }

    #[tokio::test]
    async fn list_proposals_preserves_non_not_found_io_errors() {
        let (store, dir) = temp_store().await;
        std::fs::write(dir.path().join("not-a-directory"), b"file").unwrap();

        assert!(matches!(
            store.list_proposals(&art("not-a-directory")).await,
            Err(ArtifactError::Store(_))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_symlinked_artifact_directories() {
        use std::os::unix::fs::symlink;

        let (store, dir) = temp_store().await;
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), dir.path().join("linked")).unwrap();

        assert!(matches!(
            store.read(&art("linked"), Some(&ver("version"))).await,
            Err(ArtifactError::Store(_))
        ));
        assert!(matches!(
            store.list_proposals(&art("linked")).await,
            Err(ArtifactError::Store(_))
        ));

        let result = store
            .propose(ArtifactProposal {
                id: art("linked"),
                base_version: ver(GENESIS_VERSION),
                content: b"escape".to_vec(),
                content_type: "text/plain".into(),
                rationale: "must stay confined".into(),
            })
            .await;
        assert!(matches!(result, Err(ArtifactError::Store(_))));
        assert_eq!(std::fs::read_dir(outside.path()).unwrap().count(), 0);
    }
}
