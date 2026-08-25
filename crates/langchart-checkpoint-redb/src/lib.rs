//! # langchart-checkpoint-redb
//!
//! Embedded [`CheckpointStore`] backed by [redb](https://github.com/cberner/redb).
//!
//! Suitable for single-process embedded use (desktop applications, Obsidian-style
//! plugins). For distributed or multi-process deployments, implement
//! `CheckpointStore` against a shared database.
//!
//! ## Usage
//!
//! ```text
//! let store = RedbCheckpointStore::open("./langchart.redb")?;
//! ```

use async_trait::async_trait;
use langchart_adapters::checkpoint::{CheckpointError, CheckpointStore, RunSnapshot};
use langchart_model::id::{CheckpointId, RunId};
use redb::{Database, TableDefinition};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use tracing::debug;

// Table: run_id (str) → JSON-encoded RunSnapshot (str)
const CHECKPOINTS: TableDefinition<&str, &str> = TableDefinition::new("checkpoints");

// ── Store ─────────────────────────────────────────────────────────────────────

/// A [`CheckpointStore`] backed by an embedded redb database.
#[derive(Clone)]
pub struct RedbCheckpointStore {
    db: Arc<Mutex<Database>>,
}

impl RedbCheckpointStore {
    /// Open or create a redb database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let db = Database::create(path.as_ref()).map_err(|e| StoreError::Open(e.to_string()))?;

        // Ensure table exists.
        let tx = db
            .begin_write()
            .map_err(|e| StoreError::Open(e.to_string()))?;
        tx.open_table(CHECKPOINTS)
            .map_err(|e| StoreError::Open(e.to_string()))?;
        tx.commit().map_err(|e| StoreError::Open(e.to_string()))?;

        Ok(Self {
            db: Arc::new(Mutex::new(db)),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("failed to open checkpoint store: {0}")]
    Open(String),
}

// ── CheckpointStore impl ──────────────────────────────────────────────────────

#[async_trait]
impl CheckpointStore for RedbCheckpointStore {
    async fn save(&self, snapshot: &RunSnapshot) -> Result<CheckpointId, CheckpointError> {
        let checkpoint_id = snapshot.checkpoint_id.clone();
        let key = snapshot.run_id.0.clone();
        let value =
            serde_json::to_string(snapshot).map_err(|e| CheckpointError::Store(e.to_string()))?;

        let db = self.db.lock().unwrap();
        let tx = db
            .begin_write()
            .map_err(|e| CheckpointError::Store(e.to_string()))?;
        {
            let mut table = tx
                .open_table(CHECKPOINTS)
                .map_err(|e| CheckpointError::Store(e.to_string()))?;
            table
                .insert(key.as_str(), value.as_str())
                .map_err(|e| CheckpointError::Store(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| CheckpointError::Store(e.to_string()))?;

        debug!(run = %snapshot.run_id, checkpoint = %checkpoint_id, "checkpoint saved");
        Ok(checkpoint_id)
    }

    async fn load(&self, run_id: &RunId) -> Result<Option<RunSnapshot>, CheckpointError> {
        let key = run_id.0.clone();
        let db = self.db.lock().unwrap();
        let tx = db
            .begin_read()
            .map_err(|e| CheckpointError::Store(e.to_string()))?;
        let table = tx
            .open_table(CHECKPOINTS)
            .map_err(|e| CheckpointError::Store(e.to_string()))?;

        match table
            .get(key.as_str())
            .map_err(|e| CheckpointError::Store(e.to_string()))?
        {
            Some(guard) => {
                let snapshot: RunSnapshot = serde_json::from_str(guard.value())
                    .map_err(|e| CheckpointError::Store(e.to_string()))?;
                Ok(Some(snapshot))
            }
            None => Ok(None),
        }
    }

    async fn latest(&self, run_id: &RunId) -> Result<Option<CheckpointId>, CheckpointError> {
        // Simple implementation: if a record exists the checkpoint_id is embedded in it.
        Ok(self.load(run_id).await?.map(|s| s.checkpoint_id))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::checkpoint::RunSnapshot;
    use tempfile::NamedTempFile;

    fn make_snapshot(run_id: &str) -> RunSnapshot {
        RunSnapshot {
            run_id: RunId::new(run_id),
            checkpoint_id: CheckpointId::new("cp-0"),
            payload: b"state-bytes".to_vec(),
        }
    }

    #[tokio::test]
    async fn save_and_load_round_trip() {
        let tmp = NamedTempFile::new().unwrap();
        let store = RedbCheckpointStore::open(tmp.path()).unwrap();

        let snap = make_snapshot("run-001");
        store.save(&snap).await.unwrap();

        let loaded = store.load(&RunId::new("run-001")).await.unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.run_id.0, "run-001");
        assert_eq!(loaded.payload, b"state-bytes");
    }

    #[tokio::test]
    async fn load_missing_returns_none() {
        let tmp = NamedTempFile::new().unwrap();
        let store = RedbCheckpointStore::open(tmp.path()).unwrap();

        let result = store.load(&RunId::new("no-such-run")).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn save_overwrites_previous() {
        let tmp = NamedTempFile::new().unwrap();
        let store = RedbCheckpointStore::open(tmp.path()).unwrap();

        let snap1 = make_snapshot("run-overwrite");
        let snap2 = RunSnapshot {
            run_id: RunId::new("run-overwrite"),
            checkpoint_id: CheckpointId::new("cp-1"),
            payload: b"new-state".to_vec(),
        };

        store.save(&snap1).await.unwrap();
        store.save(&snap2).await.unwrap();

        let loaded = store
            .load(&RunId::new("run-overwrite"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.payload, b"new-state");
    }

    #[tokio::test]
    async fn latest_returns_checkpoint_id() {
        let tmp = NamedTempFile::new().unwrap();
        let store = RedbCheckpointStore::open(tmp.path()).unwrap();

        let snap = make_snapshot("run-latest");
        let saved_id = store.save(&snap).await.unwrap();

        let latest = store.latest(&RunId::new("run-latest")).await.unwrap();
        assert_eq!(latest, Some(saved_id));
    }
}
