//! Memory adapter: abstract over long-term memory storage and retrieval.

use async_trait::async_trait;
use langchart_model::id::{AgentId, RunId, WorkflowId};
use serde::{Deserialize, Serialize};

/// The scope of a memory record: determines who can read and write it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// Visible only within the current run.
    Run(RunId),
    /// Visible to all runs of the same workflow.
    Workflow(WorkflowId),
    /// Visible to all uses of the same agent definition.
    Agent(AgentId),
    /// Host-application-wide visibility (subject to policy).
    Global,
}

/// A record stored in long-term memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub scope: MemoryScope,
    /// Free-form key for structured lookup.
    pub key: Option<String>,
    /// The content to store (plain text or JSON).
    pub content: String,
    /// Optional embedding vector (if pre-computed by the caller).
    pub embedding: Option<Vec<f32>>,
    /// Additional metadata as JSON.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A stable ID for a stored memory record (ULID).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MemoryId(pub String);

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Query parameters for memory retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub scope: MemoryScope,
    /// Mode determines how the store searches.
    pub mode: QueryMode,
    /// Maximum results to return.
    pub limit: u32,
    /// Minimum relevance score (0.0–1.0) for semantic queries.
    pub min_score: Option<f32>,
}

/// The search modality for a memory query.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryMode {
    /// Full-text / keyword search.
    Keyword { text: String },
    /// Semantic vector search (requires embedding support in the adapter).
    Semantic { text: String },
    /// Exact key lookup.
    Key { key: String },
}

/// One result from a memory query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    pub id: MemoryId,
    pub record: MemoryRecord,
    /// Relevance score (1.0 = exact match; not always populated).
    pub score: Option<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("memory record `{0}` not found")]
    NotFound(MemoryId),
    #[error("query mode not supported by this adapter")]
    Unsupported,
    #[error("memory store error: {0}")]
    Store(String),
}

/// Abstraction over long-term memory storage and retrieval.
#[async_trait]
pub trait MemoryAdapter: Send + Sync {
    async fn store(&self, record: MemoryRecord) -> Result<MemoryId, MemoryError>;
    async fn search(&self, query: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError>;
    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError>;
    async fn delete(&self, id: &MemoryId) -> Result<(), MemoryError>;
}
