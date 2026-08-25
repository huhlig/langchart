//! Artifact resolver stage: loads versioned artifact content into the context.

use async_trait::async_trait;
use langchart_adapters::{
    artifact::ArtifactStore,
    context::{ContextAccumulator, ContextError, ContextItem, ContextResolverStage},
};
use langchart_model::{
    id::RunId,
    policy::{ContextPolicy, ContextSource},
};
use std::sync::Arc;

pub struct ArtifactResolverStage {
    store: Arc<dyn ArtifactStore>,
}

impl ArtifactResolverStage {
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl ContextResolverStage for ArtifactResolverStage {
    async fn resolve(
        &self,
        policy: &ContextPolicy,
        _run_id: &RunId,
        ctx: &mut ContextAccumulator,
    ) -> Result<(), ContextError> {
        for source in &policy.sources {
            if let ContextSource::Artifact { selector, version } = source {
                let artifact_id = langchart_model::id::ArtifactId::new(selector.clone());
                let ver = version
                    .as_deref()
                    .map(langchart_model::id::ArtifactVersion::new);
                let content = self
                    .store
                    .read(&artifact_id, ver.as_ref())
                    .await
                    .map_err(|e| ContextError::Stage {
                        stage: "ArtifactResolverStage",
                        message: e.to_string(),
                    })?;
                let text = String::from_utf8_lossy(&content.bytes).into_owned();
                let tokens = estimate_tokens(&text);
                ctx.push(ContextItem {
                    source: format!("artifact:{}@{}", content.id, content.version),
                    content: text,
                    tokens,
                });
            }
        }
        Ok(())
    }
}

fn estimate_tokens(text: &str) -> u32 {
    // Rough estimate: 4 characters ≈ 1 token.
    (text.len() / 4).max(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use langchart_adapters::{
        artifact::{
            ArtifactContent, ArtifactError, ArtifactProposal, ArtifactStore, ProposalSummary,
        },
        context::ContextAccumulator,
    };
    use langchart_model::{
        id::{ArtifactId, ArtifactVersion, ProposalId, RunId},
        policy::{ContextPolicy, ContextSource},
    };
    use std::sync::Arc;

    // ── Stub artifact store ───────────────────────────────────────────────────

    struct StubArtifactStore {
        content: &'static str,
    }

    #[async_trait]
    impl ArtifactStore for StubArtifactStore {
        async fn read(
            &self,
            id: &ArtifactId,
            _version: Option<&ArtifactVersion>,
        ) -> Result<ArtifactContent, ArtifactError> {
            Ok(ArtifactContent {
                id: id.clone(),
                version: ArtifactVersion::new("v1"),
                bytes: self.content.as_bytes().to_vec(),
                content_type: "text/plain".into(),
            })
        }
        async fn propose(&self, _p: ArtifactProposal) -> Result<ProposalId, ArtifactError> {
            Ok(ProposalId::new("prop-1"))
        }
        async fn commit(
            &self,
            _artifact_id: &ArtifactId,
            _pid: &ProposalId,
            _base: &ArtifactVersion,
        ) -> Result<ArtifactVersion, ArtifactError> {
            Ok(ArtifactVersion::new("v2"))
        }
        async fn list_proposals(
            &self,
            _id: &ArtifactId,
        ) -> Result<Vec<ProposalSummary>, ArtifactError> {
            Ok(vec![])
        }
    }

    struct FailingArtifactStore;

    #[async_trait]
    impl ArtifactStore for FailingArtifactStore {
        async fn read(
            &self,
            id: &ArtifactId,
            _v: Option<&ArtifactVersion>,
        ) -> Result<ArtifactContent, ArtifactError> {
            Err(ArtifactError::NotFound(id.clone()))
        }
        async fn propose(&self, _p: ArtifactProposal) -> Result<ProposalId, ArtifactError> {
            unreachable!()
        }
        async fn commit(
            &self,
            _artifact_id: &ArtifactId,
            _pid: &ProposalId,
            _base: &ArtifactVersion,
        ) -> Result<ArtifactVersion, ArtifactError> {
            unreachable!()
        }
        async fn list_proposals(
            &self,
            _id: &ArtifactId,
        ) -> Result<Vec<ProposalSummary>, ArtifactError> {
            Ok(vec![])
        }
    }

    fn run_id() -> RunId {
        RunId::new("r")
    }

    fn policy_with_artifact(selector: &str) -> ContextPolicy {
        ContextPolicy {
            sources: vec![ContextSource::Artifact {
                selector: selector.into(),
                version: None,
            }],
            ..Default::default()
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn artifact_content_appended_to_context() {
        let store = Arc::new(StubArtifactStore {
            content: "Hello world content",
        });
        let stage = ArtifactResolverStage::new(store);
        let mut ctx = ContextAccumulator::default();
        stage
            .resolve(&policy_with_artifact("doc-1"), &run_id(), &mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.items.len(), 1);
        assert_eq!(ctx.items[0].content, "Hello world content");
        assert!(ctx.items[0].source.contains("artifact:"));
        assert!(ctx.token_count > 0);
    }

    #[tokio::test]
    async fn read_error_propagates() {
        let store = Arc::new(FailingArtifactStore);
        let stage = ArtifactResolverStage::new(store);
        let mut ctx = ContextAccumulator::default();
        let result = stage
            .resolve(&policy_with_artifact("missing"), &run_id(), &mut ctx)
            .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("ArtifactResolverStage")
        );
    }

    #[tokio::test]
    async fn skips_non_artifact_sources() {
        let store = Arc::new(StubArtifactStore {
            content: "should not appear",
        });
        let stage = ArtifactResolverStage::new(store);
        let mut ctx = ContextAccumulator::default();
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
    async fn token_estimate_is_nonzero_for_nonempty_content() {
        let store = Arc::new(StubArtifactStore { content: "1234" }); // 4 chars → 1 token
        let stage = ArtifactResolverStage::new(store);
        let mut ctx = ContextAccumulator::default();
        stage
            .resolve(&policy_with_artifact("a"), &run_id(), &mut ctx)
            .await
            .unwrap();
        assert!(ctx.token_count >= 1);
    }
}
