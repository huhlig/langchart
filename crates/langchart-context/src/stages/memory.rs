//! Memory resolver stage: queries the MemoryAdapter for relevant records.

use async_trait::async_trait;
use langchart_adapters::{
    context::{ContextAccumulator, ContextError, ContextItem, ContextResolverStage},
    memory::{MemoryAdapter, MemoryQuery, MemoryScope, QueryMode},
};
use langchart_model::{
    id::RunId,
    policy::{ContextPolicy, ContextSource},
};
use std::sync::Arc;

pub struct MemoryResolverStage {
    memory: Arc<dyn MemoryAdapter>,
    /// Scope applied to all memory queries from this stage.
    scope: MemoryScope,
}

impl MemoryResolverStage {
    pub fn new(memory: Arc<dyn MemoryAdapter>, scope: MemoryScope) -> Self {
        Self { memory, scope }
    }
}

#[async_trait]
impl ContextResolverStage for MemoryResolverStage {
    async fn resolve(
        &self,
        policy: &ContextPolicy,
        _run_id: &RunId,
        ctx: &mut ContextAccumulator,
    ) -> Result<(), ContextError> {
        for source in &policy.sources {
            if let ContextSource::Memory { query, limit } = source {
                let results = self
                    .memory
                    .search(MemoryQuery {
                        scope: self.scope.clone(),
                        mode: QueryMode::Semantic {
                            text: query.clone(),
                        },
                        limit: *limit,
                        min_score: None,
                    })
                    .await
                    .map_err(|e| ContextError::Stage {
                        stage: "MemoryResolverStage",
                        message: e.to_string(),
                    })?;

                for result in results {
                    let tokens = estimate_tokens(&result.record.content);
                    ctx.push(ContextItem {
                        source: format!("memory:{}", result.id.0),
                        content: result.record.content,
                        tokens,
                    });
                }
            }
        }
        Ok(())
    }
}

fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4).max(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use langchart_adapters::{
        context::ContextAccumulator,
        memory::{MemoryAdapter, MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult},
    };
    use langchart_model::{
        id::RunId,
        policy::{ContextPolicy, ContextSource},
    };
    use std::sync::Arc;

    // ── Stub memory adapter ───────────────────────────────────────────────────

    struct StubMemory {
        results: Vec<MemoryResult>,
    }

    impl StubMemory {
        fn with_results(records: Vec<(&'static str, &'static str)>) -> Self {
            let results = records
                .into_iter()
                .enumerate()
                .map(|(i, (content, key))| MemoryResult {
                    id: MemoryId(format!("mem-{i}")),
                    record: MemoryRecord {
                        scope: MemoryScope::Global,
                        key: Some(key.into()),
                        content: content.into(),
                        embedding: None,
                        metadata: serde_json::Value::Null,
                    },
                    score: Some(1.0),
                })
                .collect();
            Self { results }
        }

        fn empty() -> Self {
            Self { results: vec![] }
        }
    }

    #[async_trait]
    impl MemoryAdapter for StubMemory {
        async fn store(&self, _r: MemoryRecord) -> Result<MemoryId, MemoryError> {
            Ok(MemoryId("stub".into()))
        }
        async fn search(&self, _q: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError> {
            Ok(self.results.clone())
        }
        async fn get(&self, _id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
            Ok(None)
        }
        async fn delete(&self, _id: &MemoryId) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    fn run_id() -> RunId {
        RunId::new("r")
    }

    fn policy_with_memory(query: &str, limit: u32) -> ContextPolicy {
        ContextPolicy {
            sources: vec![ContextSource::Memory {
                query: query.into(),
                limit,
            }],
            ..Default::default()
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn memory_results_appended_to_context() {
        let memory = Arc::new(StubMemory::with_results(vec![("result content", "key-a")]));
        let stage = MemoryResolverStage::new(memory, MemoryScope::Global);
        let mut ctx = ContextAccumulator::default();
        stage
            .resolve(&policy_with_memory("topic", 10), &run_id(), &mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.items.len(), 1);
        assert_eq!(ctx.items[0].content, "result content");
        assert!(ctx.items[0].source.starts_with("memory:"));
        assert!(ctx.token_count > 0);
    }

    #[tokio::test]
    async fn empty_results_produce_no_items() {
        let memory = Arc::new(StubMemory::empty());
        let stage = MemoryResolverStage::new(memory, MemoryScope::Global);
        let mut ctx = ContextAccumulator::default();
        stage
            .resolve(&policy_with_memory("nothing", 5), &run_id(), &mut ctx)
            .await
            .unwrap();
        assert!(ctx.items.is_empty());
    }

    #[tokio::test]
    async fn skips_non_memory_sources() {
        let memory = Arc::new(StubMemory::with_results(vec![("x", "k")]));
        let stage = MemoryResolverStage::new(memory, MemoryScope::Global);
        let mut ctx = ContextAccumulator::default();
        // WorkflowData source — should not be processed.
        let policy = ContextPolicy {
            sources: vec![langchart_model::policy::ContextSource::WorkflowData {
                expression: "field".into(),
            }],
            ..Default::default()
        };
        stage.resolve(&policy, &run_id(), &mut ctx).await.unwrap();
        assert!(ctx.items.is_empty());
    }

    #[tokio::test]
    async fn multiple_memory_sources_each_queried() {
        let memory = Arc::new(StubMemory::with_results(vec![("hit", "k")]));
        let stage = MemoryResolverStage::new(memory, MemoryScope::Global);
        let mut ctx = ContextAccumulator::default();
        let policy = ContextPolicy {
            sources: vec![
                ContextSource::Memory {
                    query: "q1".into(),
                    limit: 3,
                },
                ContextSource::Memory {
                    query: "q2".into(),
                    limit: 3,
                },
            ],
            ..Default::default()
        };
        stage.resolve(&policy, &run_id(), &mut ctx).await.unwrap();
        // Two queries × one result each = 2 items.
        assert_eq!(ctx.items.len(), 2);
    }
}
