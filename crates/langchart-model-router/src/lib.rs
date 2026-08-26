//! # langchart-model-router
//!
//! Policy-driven model router that dispatches [`LlmRequest`]s to registered
//! [`LlmAdapter`] instances based on model profile routing rules.
//!
//! The router itself implements `LlmAdapter`, so it is a drop-in replacement
//! for any single adapter. The `CapabilityBroker` sees exactly one `LlmAdapter`
//! and delegates routing here.
//!
//! ## Routing rules (evaluated in order)
//!
//! 1. **Exact model name** — `"gpt-4o"` routes to adapter `"openai"`.
//! 2. **Prefix match** — `"claude-*"` routes to adapter `"anthropic"`.
//! 3. **Profile match** — `model_policy.profile == "high_quality"` → named adapter.
//! 4. **Fallback** — a default adapter used when no rule matches.
//!
//! ## Usage
//!
//! ```text
//! let router = ModelRouter::builder()
//!     .register("openai", Arc::new(openai_adapter))
//!     .register("anthropic", Arc::new(anthropic_adapter))
//!     .route(ModelRoute::Prefix { prefix: "gpt-".into(), adapter: "openai".into() })
//!     .route(ModelRoute::Prefix { prefix: "claude-".into(), adapter: "anthropic".into() })
//!     .fallback("openai")
//!     .build()?;
//! ```

use async_trait::async_trait;
use langchart_adapters::llm::{LlmAdapter, LlmError, LlmRequest, LlmResponse, ModelInfo};
use std::{collections::HashMap, sync::Arc};
use tracing::debug;

// ── Route definitions ─────────────────────────────────────────────────────────

/// One routing rule. Rules are evaluated in declaration order.
#[derive(Debug, Clone)]
pub enum ModelRoute {
    /// Route by exact model name (e.g. `"gpt-4o"`).
    Exact { model: String, adapter: String },
    /// Route by model name prefix (e.g. `"claude-"` → `"anthropic"`).
    Prefix { prefix: String, adapter: String },
    /// Route by `model_policy.profile` value (e.g. `"high_quality"` → `"openai"`).
    Profile { profile: String, adapter: String },
}

// ── Builder ───────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct ModelRouterBuilder {
    adapters: HashMap<String, Arc<dyn LlmAdapter>>,
    routes: Vec<ModelRoute>,
    fallback: Option<String>,
}

impl ModelRouterBuilder {
    /// Register a named adapter.
    pub fn register(mut self, name: impl Into<String>, adapter: Arc<dyn LlmAdapter>) -> Self {
        self.adapters.insert(name.into(), adapter);
        self
    }

    /// Add a routing rule (evaluated in declaration order).
    pub fn route(mut self, rule: ModelRoute) -> Self {
        self.routes.push(rule);
        self
    }

    /// Set the fallback adapter (used when no rule matches).
    pub fn fallback(mut self, name: impl Into<String>) -> Self {
        self.fallback = Some(name.into());
        self
    }

    pub fn build(self) -> Result<ModelRouter, BuildError> {
        if self.adapters.is_empty() {
            return Err(BuildError::NoAdapters);
        }
        if let Some(ref fb) = self.fallback
            && !self.adapters.contains_key(fb.as_str())
        {
            return Err(BuildError::UnknownAdapter(fb.clone()));
        }
        for rule in &self.routes {
            let name = match rule {
                ModelRoute::Exact { adapter, .. } => adapter,
                ModelRoute::Prefix { adapter, .. } => adapter,
                ModelRoute::Profile { adapter, .. } => adapter,
            };
            if !self.adapters.contains_key(name.as_str()) {
                return Err(BuildError::UnknownAdapter(name.clone()));
            }
        }
        Ok(ModelRouter {
            adapters: self.adapters,
            routes: self.routes,
            fallback: self.fallback,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("no adapters registered")]
    NoAdapters,
    #[error("unknown adapter name `{0}`")]
    UnknownAdapter(String),
    #[error("no fallback adapter set and no matching route for request")]
    NoRoute,
}

// ── Router ────────────────────────────────────────────────────────────────────

/// A policy-driven router that implements [`LlmAdapter`].
pub struct ModelRouter {
    adapters: HashMap<String, Arc<dyn LlmAdapter>>,
    routes: Vec<ModelRoute>,
    fallback: Option<String>,
}

impl ModelRouter {
    pub fn builder() -> ModelRouterBuilder {
        ModelRouterBuilder::default()
    }

    fn resolve(&self, request: &LlmRequest) -> Result<Arc<dyn LlmAdapter>, LlmError> {
        let model = request.model_policy.model.as_deref().unwrap_or("");
        let profile = request.model_policy.profile.as_deref().unwrap_or("");

        for rule in &self.routes {
            let adapter_name = match rule {
                ModelRoute::Exact { model: m, adapter } if m == model => Some(adapter),
                ModelRoute::Prefix { prefix, adapter } if model.starts_with(prefix.as_str()) => {
                    Some(adapter)
                }
                ModelRoute::Profile {
                    profile: p,
                    adapter,
                } if p == profile => Some(adapter),
                _ => None,
            };

            if let Some(name) = adapter_name
                && let Some(a) = self.adapters.get(name)
            {
                debug!(
                    model,
                    profile,
                    rule_adapter = name,
                    "model router: matched rule"
                );
                return Ok(a.clone());
            }
        }

        // Fallback.
        if let Some(ref fb) = self.fallback
            && let Some(a) = self.adapters.get(fb)
        {
            debug!(
                model,
                profile,
                fallback = fb,
                "model router: using fallback"
            );
            return Ok(a.clone());
        }

        Err(LlmError::Provider(format!(
            "model router: no route matched model={model:?} profile={profile:?}"
        )))
    }
}

#[async_trait]
impl LlmAdapter for ModelRouter {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        self.resolve(&request)?.complete(request).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        // Aggregate models from all registered adapters.
        let mut all = Vec::new();
        for adapter in self.adapters.values() {
            if let Ok(models) = adapter.list_models().await {
                all.extend(models);
            }
        }
        Ok(all)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::llm::{FinishReason, LlmRequest, LlmResponse, Message, TokenUsage};
    use langchart_model::policy::ModelPolicy;

    /// A fake adapter that records the model name it received.
    struct FakeAdapter(String);

    #[async_trait]
    impl LlmAdapter for FakeAdapter {
        async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
            Ok(LlmResponse {
                content: Some(format!("from:{}", self.0)),
                tool_calls: vec![],
                usage: TokenUsage::default(),
                finish_reason: FinishReason::Stop,
                refusal: None,
                model: format!("fake/{}", req.model_policy.model.unwrap_or_default()),
            })
        }
    }

    fn make_request(model: Option<&str>, profile: Option<&str>) -> LlmRequest {
        LlmRequest {
            model_policy: ModelPolicy {
                model: model.map(|s| s.to_string()),
                profile: profile.map(|s| s.to_string()),
                temperature: None,
                max_tokens: None,
            },
            messages: vec![Message::User {
                content: "hi".into(),
            }],
            tools: vec![],
            response_format: Default::default(),
        }
    }

    fn build_router() -> ModelRouter {
        ModelRouter::builder()
            .register("openai", Arc::new(FakeAdapter("openai".into())))
            .register("anthropic", Arc::new(FakeAdapter("anthropic".into())))
            .route(ModelRoute::Exact {
                model: "gpt-4o".into(),
                adapter: "openai".into(),
            })
            .route(ModelRoute::Prefix {
                prefix: "claude-".into(),
                adapter: "anthropic".into(),
            })
            .route(ModelRoute::Profile {
                profile: "fast".into(),
                adapter: "openai".into(),
            })
            .fallback("openai")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn exact_model_routes_correctly() {
        let router = build_router();
        let resp = router
            .complete(make_request(Some("gpt-4o"), None))
            .await
            .unwrap();
        assert_eq!(resp.content.unwrap(), "from:openai");
    }

    #[tokio::test]
    async fn prefix_routes_anthropic() {
        let router = build_router();
        let resp = router
            .complete(make_request(Some("claude-3-5-sonnet-20241022"), None))
            .await
            .unwrap();
        assert_eq!(resp.content.unwrap(), "from:anthropic");
    }

    #[tokio::test]
    async fn profile_routes_by_name() {
        let router = build_router();
        let resp = router
            .complete(make_request(None, Some("fast")))
            .await
            .unwrap();
        assert_eq!(resp.content.unwrap(), "from:openai");
    }

    #[tokio::test]
    async fn unknown_model_falls_back() {
        let router = build_router();
        let resp = router
            .complete(make_request(Some("mistral-7b"), None))
            .await
            .unwrap();
        // Falls back to openai adapter.
        assert_eq!(resp.content.unwrap(), "from:openai");
    }

    #[tokio::test]
    async fn no_fallback_no_route_returns_error() {
        let router = ModelRouter::builder()
            .register("only", Arc::new(FakeAdapter("only".into())))
            .route(ModelRoute::Exact {
                model: "exact-model".into(),
                adapter: "only".into(),
            })
            // No fallback set.
            .build()
            .unwrap();

        let result = router
            .complete(make_request(Some("other-model"), None))
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_unknown_adapter_in_route() {
        let result = ModelRouter::builder()
            .register("real", Arc::new(FakeAdapter("real".into())))
            .route(ModelRoute::Exact {
                model: "x".into(),
                adapter: "ghost".into(),
            })
            .build();
        assert!(matches!(result, Err(BuildError::UnknownAdapter(_))));
    }

    #[test]
    fn builder_rejects_empty_adapters() {
        let result = ModelRouter::builder().build();
        assert!(matches!(result, Err(BuildError::NoAdapters)));
    }

    #[tokio::test]
    async fn router_preserves_complete_response_format() {
        use langchart_adapters::llm::ResponseFormat;
        use std::sync::Mutex;

        struct CapturingAdapter(Arc<Mutex<Option<ResponseFormat>>>);

        #[async_trait]
        impl LlmAdapter for CapturingAdapter {
            async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
                *self.0.lock().unwrap() = Some(req.response_format);
                Ok(LlmResponse {
                    content: None,
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                    finish_reason: FinishReason::Stop,
                    refusal: None,
                    model: "captured".into(),
                })
            }
        }

        let captured = Arc::new(Mutex::new(None));
        let router = ModelRouter::builder()
            .register("capture", Arc::new(CapturingAdapter(captured.clone())))
            .fallback("capture")
            .build()
            .unwrap();
        let expected = ResponseFormat::JsonSchema {
            name: "review".into(),
            description: Some("Exact contract".into()),
            schema: serde_json::json!({
                "type": "object",
                "additionalProperties": true
            }),
            strict: false,
        };
        let mut request = make_request(Some("any-model"), None);
        request.response_format = expected.clone();

        router.complete(request).await.unwrap();

        assert_eq!(*captured.lock().unwrap(), Some(expected));
    }
}
