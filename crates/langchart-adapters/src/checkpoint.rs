//! Checkpoint store adapter: persist and recover workflow run snapshots.

use async_trait::async_trait;
use langchart_model::id::{CheckpointId, RunId};
use serde::{Deserialize, Serialize};

/// An opaque, serialized snapshot of a workflow run's complete state.
/// The runtime serializes this; the adapter stores and retrieves it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSnapshot {
    pub run_id: RunId,
    pub checkpoint_id: CheckpointId,
    /// Serialized run state (format is internal to the runtime).
    pub payload: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint for run `{0}` not found")]
    NotFound(RunId),
    #[error("checkpoint store error: {0}")]
    Store(String),
}

/// Persists and recovers run snapshots.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    async fn save(&self, snapshot: &RunSnapshot) -> Result<CheckpointId, CheckpointError>;
    async fn load(&self, run_id: &RunId) -> Result<Option<RunSnapshot>, CheckpointError>;
    async fn latest(&self, run_id: &RunId) -> Result<Option<CheckpointId>, CheckpointError>;
}
