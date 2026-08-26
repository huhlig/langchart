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
use genai::{
    adapter::AdapterKind,
    chat::{ChatOptions, ChatResponseFormat},
};
use langchart_adapters::llm::{LlmAdapter, LlmError, LlmRequest, LlmResponse, ResponseFormat};

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
        let target = self
            .client
            .resolve_service_target(model)
            .await
            .map_err(error::map)?;
        let options = chat_options(&request, target.model.adapter_kind)?;
        let chat_req = bridge::to_genai_request(&request);
        let response = self
            .client
            .exec_chat(model, chat_req, Some(&options))
            .await
            .map_err(error::map)?;
        Ok(bridge::from_genai_response(response))
    }
}

fn chat_options(request: &LlmRequest, adapter_kind: AdapterKind) -> Result<ChatOptions, LlmError> {
    let mut options = ChatOptions::default();
    if let Some(temperature) = request.model_policy.temperature {
        options = options.with_temperature(f64::from(temperature));
    }
    if let Some(max_tokens) = request.model_policy.max_tokens {
        options = options.with_max_tokens(max_tokens);
    }

    match &request.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::JsonObject
            if matches!(
                adapter_kind,
                AdapterKind::OpenAI | AdapterKind::Groq | AdapterKind::Xai | AdapterKind::DeepSeek
            ) =>
        {
            options = options.with_response_format(ChatResponseFormat::JsonMode);
        }
        // genai 0.3.5 silently ignores JsonMode for other adapters.
        ResponseFormat::JsonObject => {
            return Err(LlmError::UnsupportedResponseFormat {
                adapter: format!("genai/{}", adapter_kind.as_lower_str()),
                requested: request.response_format.kind(),
            });
        }
        // JsonSpec cannot preserve this contract: genai 0.3.5 forces strict
        // mode and rewrites object schemas for every adapter that maps it.
        ResponseFormat::JsonSchema { .. } => {
            return Err(LlmError::UnsupportedResponseFormat {
                adapter: format!("genai/{}", adapter_kind.as_lower_str()),
                requested: request.response_format.kind(),
            });
        }
    }

    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::llm::Message;
    use langchart_model::policy::ModelPolicy;

    fn request(response_format: ResponseFormat) -> LlmRequest {
        LlmRequest {
            model_policy: ModelPolicy {
                temperature: Some(0.25),
                max_tokens: Some(321),
                ..Default::default()
            },
            messages: vec![Message::User {
                content: "respond".into(),
            }],
            tools: vec![],
            response_format,
        }
    }

    #[test]
    fn options_forward_temperature_tokens_and_json_mode() {
        let options =
            chat_options(&request(ResponseFormat::JsonObject), AdapterKind::Groq).unwrap();

        assert_eq!(options.temperature, Some(0.25));
        assert_eq!(options.max_tokens, Some(321));
        assert!(matches!(
            options.response_format,
            Some(ChatResponseFormat::JsonMode)
        ));
    }

    #[test]
    fn unsupported_provider_does_not_silently_ignore_json_mode() {
        assert!(matches!(
            chat_options(&request(ResponseFormat::JsonObject), AdapterKind::Gemini),
            Err(LlmError::UnsupportedResponseFormat { .. })
        ));
    }

    #[test]
    fn schema_is_rejected_because_genai_cannot_preserve_it() {
        let response_format = ResponseFormat::JsonSchema {
            name: "result".into(),
            description: Some("preserve me".into()),
            schema: serde_json::json!({"type": "object", "additionalProperties": true}),
            strict: false,
        };

        assert!(matches!(
            chat_options(&request(response_format), AdapterKind::OpenAI),
            Err(LlmError::UnsupportedResponseFormat { .. })
        ));
    }
}
