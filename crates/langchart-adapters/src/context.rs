//! Context resolver stage adapter: one stage in the ContextResolverChain.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// An assembled, immutable snapshot of the information provided to one agent invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextView {
    /// Ordered list of included context items.
    pub items: Vec<ContextItem>,
    /// Total token count estimate across all items.
    pub token_count: u32,
    /// A stable hash of the assembled content for replay identification.
    pub content_hash: String,
}

/// One piece of resolved context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    /// Source label (e.g. `"artifact:draft-v1@3"`, `"memory:query=topic"`).
    pub source: String,
    /// The rendered text content.
    pub content: String,
    /// Token count estimate for this item.
    pub tokens: u32,
}

/// Accumulates context items during the resolver pipeline.
#[derive(Debug, Default)]
pub struct ContextAccumulator {
    pub items: Vec<ContextItem>,
    pub token_count: u32,
}

impl ContextAccumulator {
    pub fn push(&mut self, item: ContextItem) {
        self.token_count += item.tokens;
        self.items.push(item);
    }

    /// Finalise into an immutable `ContextView`.
    pub fn finish(self) -> ContextView {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        for item in &self.items {
            item.content.hash(&mut h);
        }
        let hash = format!("{:016x}", h.finish());
        ContextView {
            items: self.items,
            token_count: self.token_count,
            content_hash: hash,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("context resolution error in stage `{stage}`: {message}")]
    Stage {
        stage: &'static str,
        message: String,
    },
}

/// One composable stage in the context resolver pipeline.
///
/// Stages are called in order; each may add items to the `ContextAccumulator`.
/// Stages MUST be pure with respect to the accumulator — they add, never remove.
#[async_trait]
pub trait ContextResolverStage: Send + Sync {
    async fn resolve(
        &self,
        policy: &langchart_model::policy::ContextPolicy,
        run_id: &langchart_model::id::RunId,
        ctx: &mut ContextAccumulator,
    ) -> Result<(), ContextError>;
}

/// A complete context resolver that accepts a policy and run ID and returns a
/// fully assembled [`ContextView`].
///
/// The runtime depends on this trait, not on `ContextResolverChain` directly,
/// so that `langchart-runtime` does not need to depend on `langchart-context`.
/// `ContextResolverChain` (in `langchart-context`) implements this trait.
#[async_trait]
pub trait ContextResolver: Send + Sync {
    async fn resolve(
        &self,
        policy: &langchart_model::policy::ContextPolicy,
        run_id: &langchart_model::id::RunId,
    ) -> Result<ContextView, ContextError>;
}
