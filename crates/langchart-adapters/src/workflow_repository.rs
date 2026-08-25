//! Workflow repository adapter: resolve workflow documents by reference.
//!
//! A `WorkflowRepository` lets the runtime look up a child workflow when
//! entering a `Subworkflow` state.  The host application registers compiled
//! workflows at startup; the runtime calls [`WorkflowRepository::get`] to
//! obtain the compiled document, then spawns a child [`WorkflowInstance`].
//!
//! # `workflow_ref` format
//!
//! The `workflow_ref` field on a `Subworkflow` state is a free-form string
//! whose interpretation is up to the host.  The built-in implementations use
//! the convention `"<id>@<version>"` (e.g. `"order-fulfillment@1.2.0"`),
//! but any convention works as long as the registered key matches.

use std::sync::Arc;

use async_trait::async_trait;
use langchart_model::validation::CompiledWorkflow;

/// Resolves a `workflow_ref` string to a compiled workflow document.
///
/// Implementors are stored behind `Arc<dyn WorkflowRepository>` and shared
/// across all concurrent runs.  All methods are async to support remote or
/// database-backed stores.
#[async_trait]
pub trait WorkflowRepository: Send + Sync {
    /// Return the compiled workflow for the given reference string, or `None`
    /// if no matching workflow is registered.
    async fn get(&self, workflow_ref: &str) -> Option<Arc<CompiledWorkflow>>;
}

// ── In-memory implementation ──────────────────────────────────────────────────

use std::collections::HashMap;

/// An in-memory [`WorkflowRepository`] backed by a [`HashMap`].
///
/// Suitable for tests and single-process deployments where all child workflows
/// are known at startup.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use langchart_adapters::workflow_repository::{InMemoryWorkflowRepository, WorkflowRepository};
///
/// let repo = InMemoryWorkflowRepository::new();
/// // register compiled workflows with repo.register(…)
/// let arc: Arc<dyn WorkflowRepository> = Arc::new(repo);
/// ```
pub struct InMemoryWorkflowRepository {
    workflows: HashMap<String, Arc<CompiledWorkflow>>,
}

impl InMemoryWorkflowRepository {
    /// Create an empty repository.
    pub fn new() -> Self {
        Self {
            workflows: HashMap::new(),
        }
    }

    /// Register a compiled workflow under the given reference key.
    ///
    /// Any existing entry for the same key is replaced.
    pub fn register(
        mut self,
        workflow_ref: impl Into<String>,
        workflow: Arc<CompiledWorkflow>,
    ) -> Self {
        self.workflows.insert(workflow_ref.into(), workflow);
        self
    }
}

impl Default for InMemoryWorkflowRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorkflowRepository for InMemoryWorkflowRepository {
    async fn get(&self, workflow_ref: &str) -> Option<Arc<CompiledWorkflow>> {
        self.workflows.get(workflow_ref).cloned()
    }
}
