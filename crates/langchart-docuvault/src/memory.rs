//! Optional `MemoryAdapter` backed by vault FTS search (feature = "vault-memory").
//!
//! Enables a vault to serve as a long-term knowledge store for langchart agents.
//! Only keyword / FTS queries are supported; semantic queries require a separate
//! embedding pipeline and are delegated to `langchart-memory-redb` instead.
//!
//! The recommended architecture is:
//! - `langchart-memory-redb`  — run-scoped scratch / semantic memory
//! - `langchart-docuvault` (this module) — vault-section long-term knowledge

use async_trait::async_trait;
use langchart_adapters::memory::{
    MemoryAdapter, MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult, QueryMode,
};

/// A `MemoryAdapter` that reads vault sections via FTS as long-term memory.
pub struct VaultMemoryAdapter {
    // TODO(langchart-docuvault): hold Arc<VaultHandle> or command channel.
    _vault_id: String,
}

impl VaultMemoryAdapter {
    pub fn new(vault_id: impl Into<String>) -> Self {
        Self {
            _vault_id: vault_id.into(),
        }
    }
}

#[async_trait]
impl MemoryAdapter for VaultMemoryAdapter {
    async fn store(&self, _record: MemoryRecord) -> Result<MemoryId, MemoryError> {
        // Vault sections are not created by the memory adapter — use ArtifactStore::propose.
        Err(MemoryError::Unsupported)
    }

    async fn search(&self, query: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError> {
        match &query.mode {
            QueryMode::Keyword { .. } => {
                // TODO: delegate to SearchCoordinator::fts
                Ok(vec![])
            }
            QueryMode::Semantic { .. } | QueryMode::Key { .. } => Err(MemoryError::Unsupported),
        }
    }

    async fn get(&self, _id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
        // Vault sections are addressed by section ID, not memory ID.
        Err(MemoryError::Unsupported)
    }

    async fn delete(&self, _id: &MemoryId) -> Result<(), MemoryError> {
        Err(MemoryError::Unsupported)
    }
}
