//! Truncation stage: enforces the context token budget.
//!
//! Items are truncated from the end (lowest-priority items last) until the
//! total token count is within the budget. Items already in the accumulator
//! when this stage runs are treated as higher priority.

use async_trait::async_trait;
use langchart_adapters::context::{ContextAccumulator, ContextError, ContextResolverStage};
use langchart_model::{id::RunId, policy::ContextPolicy};

pub struct TruncationResolverStage {
    /// Maximum allowed tokens. Uses `policy.token_budget` if `None`.
    max_tokens: Option<u32>,
}

impl TruncationResolverStage {
    /// Use the token budget declared in the state's context policy.
    pub fn from_policy() -> Self {
        Self { max_tokens: None }
    }

    /// Override with a fixed maximum (useful in tests).
    pub fn with_max(max_tokens: u32) -> Self {
        Self {
            max_tokens: Some(max_tokens),
        }
    }
}

#[async_trait]
impl ContextResolverStage for TruncationResolverStage {
    async fn resolve(
        &self,
        policy: &ContextPolicy,
        _run_id: &RunId,
        ctx: &mut ContextAccumulator,
    ) -> Result<(), ContextError> {
        let budget = self.max_tokens.or(policy.token_budget).unwrap_or(u32::MAX);

        if ctx.token_count <= budget {
            return Ok(());
        }

        // Drop items from the end until within budget.
        while ctx.token_count > budget {
            if let Some(item) = ctx.items.pop() {
                ctx.token_count = ctx.token_count.saturating_sub(item.tokens);
            } else {
                break;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::context::ContextAccumulator;
    use langchart_model::{id::RunId, policy::ContextPolicy};

    fn run_id() -> RunId {
        RunId::new("run-1")
    }

    fn policy_with_budget(budget: u32) -> ContextPolicy {
        ContextPolicy {
            token_budget: Some(budget),
            ..Default::default()
        }
    }

    fn push_items(ctx: &mut ContextAccumulator, items: &[(&str, u32)]) {
        for (content, tokens) in items {
            ctx.push(langchart_adapters::context::ContextItem {
                source: "test".into(),
                content: content.to_string(),
                tokens: *tokens,
            });
        }
    }

    #[tokio::test]
    async fn no_op_when_under_budget() {
        let stage = TruncationResolverStage::with_max(100);
        let mut ctx = ContextAccumulator::default();
        push_items(&mut ctx, &[("a", 20), ("b", 30)]);
        stage
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.items.len(), 2);
        assert_eq!(ctx.token_count, 50);
    }

    #[tokio::test]
    async fn truncates_from_end_to_fit_budget() {
        let stage = TruncationResolverStage::with_max(5);
        let mut ctx = ContextAccumulator::default();
        push_items(&mut ctx, &[("keep", 3), ("drop", 4)]);
        stage
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.items.len(), 1);
        assert_eq!(ctx.items[0].content, "keep");
        assert_eq!(ctx.token_count, 3);
    }

    #[tokio::test]
    async fn uses_policy_budget_when_no_override() {
        let stage = TruncationResolverStage::from_policy();
        let mut ctx = ContextAccumulator::default();
        push_items(&mut ctx, &[("a", 10), ("b", 10), ("c", 10)]);
        let policy = policy_with_budget(15);
        stage.resolve(&policy, &run_id(), &mut ctx).await.unwrap();
        // "c" (last) should be dropped to fit within 15.
        assert!(ctx.token_count <= 15);
    }

    #[tokio::test]
    async fn unlimited_when_no_budget_set() {
        let stage = TruncationResolverStage::from_policy();
        let mut ctx = ContextAccumulator::default();
        push_items(&mut ctx, &[("a", 1000), ("b", 1000), ("c", 1000)]);
        // Default policy has no budget → no truncation.
        stage
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.items.len(), 3);
        assert_eq!(ctx.token_count, 3000);
    }

    #[tokio::test]
    async fn empty_accumulator_stays_empty_under_low_budget() {
        let stage = TruncationResolverStage::with_max(0);
        let mut ctx = ContextAccumulator::default();
        stage
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx)
            .await
            .unwrap();
        assert!(ctx.items.is_empty());
    }
}
