//! `langchart-llm-genai` — LlmAdapter backed by the `genai` crate.
//!
//! Covers cloud providers not reachable via the OpenAI-compatible wire format:
//! Gemini, Groq, Cohere, xAI (Grok), DeepSeek.
//!
//! OpenAI-compatible providers (OpenAI, Anthropic, Ollama, LM Studio, Lemonade,
//! Azure, vLLM) are handled by `langchart-llm-generic` instead.

pub mod bridge;
pub mod error;

use async_trait::async_trait;
use langchart_adapters::llm::{LlmAdapter, LlmError, LlmRequest, LlmResponse};

/// LLM adapter backed by the `genai` multi-provider client.
pub struct GenaiLlmAdapter {
    client: genai::Client,
}

impl GenaiLlmAdapter {
    pub fn new(client: genai::Client) -> Self {
        Self { client }
    }

    /// Construct from environment variables (`GENAI_API_KEY`, `GEMINI_API_KEY`, etc.).
    pub fn from_env() -> Self {
        Self {
            client: genai::Client::default(),
        }
    }

    /// Construct with a fixed API key injected via `AuthResolver`, bypassing
    /// process-global environment variables.  Safe to call concurrently from
    /// multiple vault contexts with different keys.
    pub fn with_key(api_key: impl Into<String> + Clone + Send + Sync + 'static) -> Self {
        let client = genai::Client::builder()
            .with_auth_resolver_fn(move |_model_iden| {
                Ok(Some(genai::resolver::AuthData::from_single(
                    api_key.clone(),
                )))
            })
            .build();
        Self { client }
    }
}

#[async_trait]
impl LlmAdapter for GenaiLlmAdapter {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let model = request
            .model_policy
            .model
            .as_deref()
            .unwrap_or("gemini-2.0-flash");
        let chat_req = bridge::to_genai_request(&request);
        let response = self
            .client
            .exec_chat(model, chat_req, None)
            .await
            .map_err(error::map)?;
        Ok(bridge::from_genai_response(response))
    }
}
