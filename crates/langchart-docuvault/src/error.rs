//! Unified error type for the langchart-docuvault adapter crate.

use langchart_adapters::artifact::ArtifactError;
use langchart_adapters::context::ContextError;

#[derive(Debug, thiserror::Error)]
pub enum DocuvaultAdapterError {
    #[error("id parse error: {0}")]
    Id(String),

    #[error("artifact error: {0}")]
    Artifact(#[from] ArtifactError),

    #[error("context error: {0}")]
    Context(#[from] ContextError),

    #[cfg(feature = "vault-memory")]
    #[error("memory error: {0}")]
    Memory(#[from] langchart_adapters::memory::MemoryError),

    #[error("internal error: {0}")]
    Internal(String),
}
