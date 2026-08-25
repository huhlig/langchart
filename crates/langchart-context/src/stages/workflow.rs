//! Workflow data resolver stage: injects workflow variables into context.

use async_trait::async_trait;
use langchart_adapters::context::{
    ContextAccumulator, ContextError, ContextItem, ContextResolverStage,
};
use langchart_model::{
    id::RunId,
    policy::{ContextPolicy, ContextSource},
};
use std::collections::HashMap;
use std::sync::Arc;

/// Resolves `WorkflowData` context sources by looking up named fields from
/// the current run's workflow data map.
pub struct WorkflowDataResolverStage {
    /// The current workflow data snapshot, serialized to JSON per field.
    data: Arc<HashMap<String, serde_json::Value>>,
}

impl WorkflowDataResolverStage {
    pub fn new(data: Arc<HashMap<String, serde_json::Value>>) -> Self {
        Self { data }
    }
}

#[async_trait]
impl ContextResolverStage for WorkflowDataResolverStage {
    async fn resolve(
        &self,
        policy: &ContextPolicy,
        _run_id: &RunId,
        ctx: &mut ContextAccumulator,
    ) -> Result<(), ContextError> {
        for source in &policy.sources {
            if let ContextSource::WorkflowData { expression } = source {
                // Simple field name lookup. Full expression evaluation (CEL)
                // is a Phase 3 enhancement.
                if let Some(value) = self.data.get(expression.as_str()) {
                    let content = serde_json::to_string_pretty(value).unwrap_or_default();
                    let tokens = (content.len() / 4).max(1) as u32;
                    ctx.push(ContextItem {
                        source: format!("workflow_data:{expression}"),
                        content,
                        tokens,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::context::ContextAccumulator;
    use langchart_model::{
        id::RunId,
        policy::{ContextPolicy, ContextSource},
    };
    use std::collections::HashMap;

    fn run_id() -> RunId {
        RunId::new("run-1")
    }

    fn policy_with_source(expression: &str) -> ContextPolicy {
        ContextPolicy {
            sources: vec![ContextSource::WorkflowData {
                expression: expression.into(),
            }],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn injects_known_field() {
        let mut data = HashMap::new();
        data.insert("topic".into(), serde_json::json!("climate change"));
        let stage = WorkflowDataResolverStage::new(Arc::new(data));
        let mut ctx = ContextAccumulator::default();
        let policy = policy_with_source("topic");
        stage.resolve(&policy, &run_id(), &mut ctx).await.unwrap();
        assert_eq!(ctx.items.len(), 1);
        assert!(ctx.items[0].source.contains("topic"));
        assert!(ctx.items[0].content.contains("climate change"));
        assert!(ctx.token_count > 0);
    }

    #[tokio::test]
    async fn missing_field_produces_no_item() {
        let stage = WorkflowDataResolverStage::new(Arc::new(HashMap::new()));
        let mut ctx = ContextAccumulator::default();
        let policy = policy_with_source("nonexistent");
        stage.resolve(&policy, &run_id(), &mut ctx).await.unwrap();
        assert!(ctx.items.is_empty());
    }

    #[tokio::test]
    async fn skips_non_workflow_data_sources() {
        let mut data = HashMap::new();
        data.insert("x".into(), serde_json::json!("value"));
        let stage = WorkflowDataResolverStage::new(Arc::new(data));
        let mut ctx = ContextAccumulator::default();
        // Memory source — should not be processed by WorkflowDataResolverStage.
        let policy = ContextPolicy {
            sources: vec![ContextSource::Memory {
                query: "q".into(),
                limit: 5,
            }],
            ..Default::default()
        };
        stage.resolve(&policy, &run_id(), &mut ctx).await.unwrap();
        assert!(ctx.items.is_empty());
    }

    #[tokio::test]
    async fn multiple_fields_all_injected() {
        let mut data = HashMap::new();
        data.insert("a".into(), serde_json::json!(1));
        data.insert("b".into(), serde_json::json!(true));
        let stage = WorkflowDataResolverStage::new(Arc::new(data));
        let mut ctx = ContextAccumulator::default();
        let policy = ContextPolicy {
            sources: vec![
                ContextSource::WorkflowData {
                    expression: "a".into(),
                },
                ContextSource::WorkflowData {
                    expression: "b".into(),
                },
            ],
            ..Default::default()
        };
        stage.resolve(&policy, &run_id(), &mut ctx).await.unwrap();
        assert_eq!(ctx.items.len(), 2);
    }
}
