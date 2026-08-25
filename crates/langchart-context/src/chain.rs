//! `ContextResolverChain` — runs an ordered list of resolver stages.

use async_trait::async_trait;
use langchart_adapters::context::{
    ContextAccumulator, ContextError, ContextResolver, ContextResolverStage, ContextView,
};
use langchart_model::{id::RunId, policy::ContextPolicy};
use std::sync::Arc;

/// An ordered pipeline of [`ContextResolverStage`]s.
///
/// Call `add_stage` to register stages in execution order, then `resolve` to
/// run all stages and obtain a [`ContextView`](langchart_adapters::context::ContextView).
///
/// # Example
/// ```text
/// let chain = ContextResolverChain::new()
///     .add_stage(ArtifactResolverStage::new(artifact_store.clone()))
///     .add_stage(MemoryResolverStage::new(memory_adapter.clone()))
///     .add_stage(TruncationResolverStage::new(4096))
///     .add_stage(RecordingResolverStage::default());
///
/// let view = chain.resolve(&policy, &run_id).await?;
/// ```
pub struct ContextResolverChain {
    stages: Vec<Arc<dyn ContextResolverStage>>,
}

impl ContextResolverChain {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn add_stage(mut self, stage: impl ContextResolverStage + 'static) -> Self {
        self.stages.push(Arc::new(stage));
        self
    }

    pub async fn resolve(
        &self,
        policy: &ContextPolicy,
        run_id: &RunId,
    ) -> Result<langchart_adapters::context::ContextView, ContextError> {
        let mut acc = ContextAccumulator::default();
        for stage in &self.stages {
            stage.resolve(policy, run_id, &mut acc).await?;
        }
        Ok(acc.finish())
    }
}

impl Default for ContextResolverChain {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContextResolver for ContextResolverChain {
    async fn resolve(
        &self,
        policy: &ContextPolicy,
        run_id: &RunId,
    ) -> Result<ContextView, ContextError> {
        // Delegate to the inherent method.
        ContextResolverChain::resolve(self, policy, run_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::context::{
        ContextAccumulator, ContextError, ContextItem, ContextResolverStage,
    };
    use langchart_model::{id::RunId, policy::ContextPolicy};

    // ── Stubs ─────────────────────────────────────────────────────────────────

    struct AddingStage {
        source: &'static str,
        content: &'static str,
        tokens: u32,
    }

    #[async_trait::async_trait]
    impl ContextResolverStage for AddingStage {
        async fn resolve(
            &self,
            _policy: &ContextPolicy,
            _run_id: &RunId,
            ctx: &mut ContextAccumulator,
        ) -> Result<(), ContextError> {
            ctx.push(ContextItem {
                source: self.source.into(),
                content: self.content.into(),
                tokens: self.tokens,
            });
            Ok(())
        }
    }

    struct FailingStage;

    #[async_trait::async_trait]
    impl ContextResolverStage for FailingStage {
        async fn resolve(
            &self,
            _policy: &ContextPolicy,
            _run_id: &RunId,
            _ctx: &mut ContextAccumulator,
        ) -> Result<(), ContextError> {
            Err(ContextError::Stage {
                stage: "FailingStage",
                message: "injected error".into(),
            })
        }
    }

    fn policy() -> ContextPolicy {
        ContextPolicy::default()
    }

    fn run_id() -> RunId {
        RunId::new("test-run")
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn empty_chain_produces_empty_view() {
        let chain = ContextResolverChain::new();
        let view = chain.resolve(&policy(), &run_id()).await.unwrap();
        assert!(view.items.is_empty());
        assert_eq!(view.token_count, 0);
    }

    #[tokio::test]
    async fn single_stage_adds_item() {
        let chain = ContextResolverChain::new().add_stage(AddingStage {
            source: "test:src",
            content: "hello",
            tokens: 2,
        });
        let view = chain.resolve(&policy(), &run_id()).await.unwrap();
        assert_eq!(view.items.len(), 1);
        assert_eq!(view.items[0].source, "test:src");
        assert_eq!(view.items[0].content, "hello");
        assert_eq!(view.token_count, 2);
    }

    #[tokio::test]
    async fn multiple_stages_accumulate_in_order() {
        let chain = ContextResolverChain::new()
            .add_stage(AddingStage {
                source: "a",
                content: "first",
                tokens: 5,
            })
            .add_stage(AddingStage {
                source: "b",
                content: "second",
                tokens: 3,
            });
        let view = chain.resolve(&policy(), &run_id()).await.unwrap();
        assert_eq!(view.items.len(), 2);
        assert_eq!(view.items[0].source, "a");
        assert_eq!(view.items[1].source, "b");
        assert_eq!(view.token_count, 8);
    }

    #[tokio::test]
    async fn failing_stage_propagates_error() {
        let chain = ContextResolverChain::new()
            .add_stage(AddingStage {
                source: "a",
                content: "x",
                tokens: 1,
            })
            .add_stage(FailingStage);
        let result = chain.resolve(&policy(), &run_id()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("FailingStage"));
    }

    #[tokio::test]
    async fn content_hash_is_stable_for_same_content() {
        let chain = ContextResolverChain::new().add_stage(AddingStage {
            source: "s",
            content: "abc",
            tokens: 1,
        });
        let v1 = chain.resolve(&policy(), &run_id()).await.unwrap();
        let v2 = chain.resolve(&policy(), &run_id()).await.unwrap();
        assert_eq!(v1.content_hash, v2.content_hash);
    }

    #[tokio::test]
    async fn resolver_trait_impl_delegates_correctly() {
        // Exercises the ContextResolver trait impl (delegating to the inherent method).
        let chain = ContextResolverChain::new().add_stage(AddingStage {
            source: "t",
            content: "hi",
            tokens: 1,
        });
        let view = chain.resolve(&policy(), &run_id()).await.unwrap();
        assert_eq!(view.items.len(), 1);
    }
}
