//! # langchart-llm-generic
//!
//! [`LlmAdapter`] implementation for the **OpenAI** Chat Completions API and the
//! **Anthropic** Messages API, plus any OpenAI-compatible endpoint (Azure OpenAI,
//! Ollama, vLLM, LM Studio, etc.).
//!
//! ## Provider detection
//!
//! The adapter inspects `model_policy.model` at call time:
//! - `claude-*` → Anthropic Messages API
//! - anything else (or `None`) → OpenAI Chat Completions API
//!
//! ## Usage
//!
//! ```text
//! let adapter = GenericLlmAdapter::builder()
//!     .openai_api_key("sk-...")
//!     .anthropic_api_key("sk-ant-...")
//!     .build()?;
//! ```
//!
//! For Azure / Ollama / vLLM:
//!
//! ```text
//! let adapter = GenericLlmAdapter::builder()
//!     .openai_api_key("...")
//!     .openai_base_url("http://localhost:11434/v1")  // Ollama
//!     .build()?;
//! ```

use async_trait::async_trait;
use langchart_adapters::llm::{
    FinishReason, LlmAdapter, LlmError, LlmRequest, LlmResponse, Message, ModelInfo,
    ResponseFormat, TokenUsage, ToolCall, ToolDefinition,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, warn};

// ── Constants ─────────────────────────────────────────────────────────────────

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_TIMEOUT_SECS: u64 = 120;

// ── Builder ───────────────────────────────────────────────────────────────────

/// Builder for [`GenericLlmAdapter`].
#[derive(Default)]
pub struct GenericLlmAdapterBuilder {
    openai_api_key: Option<String>,
    anthropic_api_key: Option<String>,
    openai_base_url: Option<String>,
    timeout: Option<Duration>,
}

impl GenericLlmAdapterBuilder {
    pub fn openai_api_key(mut self, key: impl Into<String>) -> Self {
        self.openai_api_key = Some(key.into());
        self
    }

    pub fn anthropic_api_key(mut self, key: impl Into<String>) -> Self {
        self.anthropic_api_key = Some(key.into());
        self
    }

    /// Override the OpenAI-compatible base URL (e.g. `http://localhost:11434/v1`
    /// for Ollama). Leave unset to use `api.openai.com`.
    pub fn openai_base_url(mut self, url: impl Into<String>) -> Self {
        self.openai_base_url = Some(url.into());
        self
    }

    pub fn timeout(mut self, t: Duration) -> Self {
        self.timeout = Some(t);
        self
    }

    pub fn build(self) -> Result<GenericLlmAdapter, BuildError> {
        let client = reqwest::Client::builder()
            .timeout(
                self.timeout
                    .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS)),
            )
            .build()
            .map_err(|e| BuildError::Client(e.to_string()))?;

        Ok(GenericLlmAdapter {
            client,
            openai_api_key: self.openai_api_key,
            anthropic_api_key: self.anthropic_api_key,
            openai_base_url: self
                .openai_base_url
                .unwrap_or_else(|| OPENAI_BASE_URL.to_string()),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("failed to build HTTP client: {0}")]
    Client(String),
}

// ── Adapter ───────────────────────────────────────────────────────────────────

/// A single adapter that covers OpenAI, Anthropic, and any OpenAI-compatible
/// endpoint. Register it with [`langchart_runtime::EngineAdapters`] as the
/// `llm` field.
pub struct GenericLlmAdapter {
    client: reqwest::Client,
    openai_api_key: Option<String>,
    anthropic_api_key: Option<String>,
    openai_base_url: String,
}

impl GenericLlmAdapter {
    pub fn builder() -> GenericLlmAdapterBuilder {
        GenericLlmAdapterBuilder::default()
    }

    /// Convenience constructor that reads API keys from environment variables.
    ///
    /// - `OPENAI_API_KEY`
    /// - `ANTHROPIC_API_KEY`
    pub fn from_env() -> Result<Self, BuildError> {
        Self::builder()
            .openai_api_key(std::env::var("OPENAI_API_KEY").unwrap_or_default())
            .anthropic_api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default())
            .build()
    }

    fn is_anthropic_model(model: &str) -> bool {
        model.starts_with("claude")
    }
}

#[async_trait]
impl LlmAdapter for GenericLlmAdapter {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let model = request
            .model_policy
            .model
            .clone()
            .unwrap_or_else(|| "gpt-4o-mini".to_string());

        if Self::is_anthropic_model(&model) {
            self.complete_anthropic(&model, &request).await
        } else {
            self.complete_openai(&model, &request).await
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        // Try OpenAI first if a key is configured.
        if self
            .openai_api_key
            .as_deref()
            .map(|k| !k.is_empty())
            .unwrap_or(false)
        {
            return self.list_openai_models().await;
        }
        Ok(vec![])
    }
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

/// Wire format for one OpenAI message.
#[derive(Serialize)]
struct OaiMessage {
    role: &'static str,
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OaiToolCall>>,
}

#[derive(Serialize, Deserialize)]
struct OaiToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: OaiFunction,
}

#[derive(Serialize, Deserialize)]
struct OaiFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OaiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OaiToolFunction,
}

#[derive(Serialize)]
struct OaiToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct OaiRequest {
    model: String,
    messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OaiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OaiResponseFormat>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OaiResponseFormat {
    JsonObject,
    JsonSchema { json_schema: OaiJsonSchema },
}

#[derive(Serialize)]
struct OaiJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    strict: bool,
}

#[derive(Deserialize)]
struct OaiResponse {
    model: String,
    choices: Vec<OaiChoice>,
    usage: OaiUsage,
}

#[derive(Deserialize)]
struct OaiChoice {
    message: OaiChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OaiChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OaiToolCall>,
}

#[derive(Deserialize)]
struct OaiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Deserialize)]
struct OaiModelsResponse {
    data: Vec<OaiModelData>,
}

#[derive(Deserialize)]
struct OaiModelData {
    id: String,
}

impl GenericLlmAdapter {
    async fn complete_openai(
        &self,
        model: &str,
        req: &LlmRequest,
    ) -> Result<LlmResponse, LlmError> {
        let api_key = self.openai_api_key.as_deref().filter(|k| !k.is_empty());

        let messages = req.messages.iter().map(msg_to_oai).collect::<Vec<_>>();
        let tools = req.tools.iter().map(tool_to_oai).collect::<Vec<_>>();

        let body = OaiRequest {
            model: model.to_string(),
            messages,
            tools,
            temperature: req.model_policy.temperature,
            max_tokens: req.model_policy.max_tokens,
            response_format: oai_response_format(&req.response_format),
        };

        debug!(model = model, "openai request");

        let url = format!("{}/chat/completions", self.openai_base_url);
        let mut req_builder = self.client.post(&url);

        // Only set bearer auth if API key is configured (not needed for local endpoints).
        if let Some(key) = api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let resp = req_builder
            .json(&body)
            .send()
            .await
            .map_err(|e| http_to_llm_err(&e))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(status_to_llm_err(status.as_u16(), &body));
        }

        let oai: OaiResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Provider(format!("OpenAI response parse error: {e}")))?;

        let choice = oai
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::Provider("OpenAI: empty choices".into()))?;

        let finish_reason = parse_oai_finish_reason(choice.finish_reason.as_deref());
        let tool_calls = choice
            .message
            .tool_calls
            .into_iter()
            .map(oai_tc_to_tool_call)
            .collect();

        Ok(LlmResponse {
            content: choice.message.content,
            tool_calls,
            usage: TokenUsage {
                prompt_tokens: oai.usage.prompt_tokens,
                completion_tokens: oai.usage.completion_tokens,
                total_tokens: oai.usage.total_tokens,
            },
            finish_reason,
            refusal: choice.message.refusal,
            model: oai.model,
        })
    }

    async fn list_openai_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let api_key = self.openai_api_key.as_deref().filter(|k| !k.is_empty());

        let url = format!("{}/models", self.openai_base_url);
        let mut req_builder = self.client.get(&url);

        // Only set bearer auth if API key is configured (not needed for local endpoints).
        if let Some(key) = api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let resp = req_builder.send().await.map_err(|e| http_to_llm_err(&e))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(status_to_llm_err(status.as_u16(), &body));
        }

        let models: OaiModelsResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Provider(format!("OpenAI models parse error: {e}")))?;

        Ok(models
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                description: None,
            })
            .collect())
    }
}

// ── Anthropic ─────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: serde_json::Value,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    model: String,
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl GenericLlmAdapter {
    async fn complete_anthropic(
        &self,
        model: &str,
        req: &LlmRequest,
    ) -> Result<LlmResponse, LlmError> {
        if !matches!(req.response_format, ResponseFormat::Text) {
            return Err(LlmError::UnsupportedResponseFormat {
                adapter: "anthropic".into(),
                requested: req.response_format.kind(),
            });
        }

        let api_key = self
            .anthropic_api_key
            .as_deref()
            .filter(|k| !k.is_empty())
            .ok_or_else(|| LlmError::Provider("ANTHROPIC_API_KEY not configured".into()))?;

        // Split system message out — Anthropic puts it at top-level, not in messages.
        let mut system: Option<String> = None;
        let mut messages: Vec<AnthropicMessage> = Vec::new();

        for msg in &req.messages {
            match msg {
                Message::System { content } => {
                    system = Some(content.clone());
                }
                Message::User { content } => {
                    messages.push(AnthropicMessage {
                        role: "user",
                        content: serde_json::Value::String(content.clone()),
                    });
                }
                Message::Assistant { content } => {
                    messages.push(AnthropicMessage {
                        role: "assistant",
                        content: serde_json::Value::String(content.clone()),
                    });
                }
                Message::Tool {
                    tool_call_id,
                    content,
                } => {
                    // Anthropic tool results go as user messages with content type "tool_result".
                    messages.push(AnthropicMessage {
                        role: "user",
                        content: serde_json::json!([{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content
                        }]),
                    });
                }
            }
        }

        let tools = req
            .tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            })
            .collect::<Vec<_>>();

        // Anthropic requires max_tokens; use policy or a safe default.
        let max_tokens = req.model_policy.max_tokens.unwrap_or(4096);

        let body = AnthropicRequest {
            model: model.to_string(),
            max_tokens,
            messages,
            system,
            tools,
            temperature: req.model_policy.temperature,
        };

        debug!(model = model, "anthropic request");

        let url = format!("{}/messages", ANTHROPIC_BASE_URL);
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| http_to_llm_err(&e))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(status_to_llm_err(status.as_u16(), &body));
        }

        let anthropic: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::Provider(format!("Anthropic response parse error: {e}")))?;

        let mut text_content: Option<String> = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in anthropic.content {
            match block {
                AnthropicContentBlock::Text { text } => {
                    text_content = Some(text);
                }
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: input,
                    });
                }
            }
        }

        let finish_reason = match anthropic.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("tool_use") => FinishReason::ToolCalls,
            Some("max_tokens") => FinishReason::Length,
            Some("stop_sequence") => FinishReason::Stop,
            Some(other) => {
                warn!(reason = other, "unknown anthropic stop_reason");
                FinishReason::Other(other.to_string())
            }
            None => FinishReason::Stop,
        };

        let total = anthropic.usage.input_tokens + anthropic.usage.output_tokens;
        Ok(LlmResponse {
            content: text_content,
            tool_calls,
            usage: TokenUsage {
                prompt_tokens: anthropic.usage.input_tokens,
                completion_tokens: anthropic.usage.output_tokens,
                total_tokens: total,
            },
            finish_reason,
            refusal: None,
            model: anthropic.model,
        })
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

fn msg_to_oai(msg: &Message) -> OaiMessage {
    match msg {
        Message::System { content } => OaiMessage {
            role: "system",
            content: serde_json::Value::String(content.clone()),
            tool_call_id: None,
            tool_calls: None,
        },
        Message::User { content } => OaiMessage {
            role: "user",
            content: serde_json::Value::String(content.clone()),
            tool_call_id: None,
            tool_calls: None,
        },
        Message::Assistant { content } => OaiMessage {
            role: "assistant",
            content: serde_json::Value::String(content.clone()),
            tool_call_id: None,
            tool_calls: None,
        },
        Message::Tool {
            tool_call_id,
            content,
        } => OaiMessage {
            role: "tool",
            content: serde_json::Value::String(content.clone()),
            tool_call_id: Some(tool_call_id.clone()),
            tool_calls: None,
        },
    }
}

fn tool_to_oai(t: &ToolDefinition) -> OaiTool {
    OaiTool {
        kind: "function",
        function: OaiToolFunction {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        },
    }
}

fn oai_response_format(format: &ResponseFormat) -> Option<OaiResponseFormat> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(OaiResponseFormat::JsonObject),
        ResponseFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
        } => Some(OaiResponseFormat::JsonSchema {
            json_schema: OaiJsonSchema {
                name: name.clone(),
                description: description.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        }),
    }
}

fn oai_tc_to_tool_call(tc: OaiToolCall) -> ToolCall {
    // OpenAI encodes arguments as a JSON string — parse it back.
    let arguments = serde_json::from_str(&tc.function.arguments)
        .unwrap_or(serde_json::Value::String(tc.function.arguments));
    ToolCall {
        id: tc.id,
        name: tc.function.name,
        arguments,
    }
}

fn parse_oai_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls") | Some("function_call") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some(other) => FinishReason::Other(other.to_string()),
        None => FinishReason::Stop,
    }
}

fn http_to_llm_err(e: &reqwest::Error) -> LlmError {
    if e.is_timeout() {
        LlmError::Timeout
    } else {
        LlmError::Provider(e.to_string())
    }
}

fn status_to_llm_err(status: u16, body: &str) -> LlmError {
    match status {
        429 => LlmError::RateLimited(body.chars().take(200).collect()),
        400 => {
            if body.contains("context_length_exceeded") || body.contains("context window") {
                LlmError::ContextLengthExceeded
            } else {
                LlmError::Provider(format!(
                    "HTTP 400: {}",
                    body.chars().take(200).collect::<String>()
                ))
            }
        }
        404 => LlmError::ModelNotFound {
            model: body.chars().take(100).collect(),
        },
        _ => LlmError::Provider(format!(
            "HTTP {status}: {}",
            body.chars().take(200).collect::<String>()
        )),
    }
}

// ── Unit tests (no network required) ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::llm::{Message, ResponseFormatKind};

    #[test]
    fn anthropic_model_detection() {
        assert!(GenericLlmAdapter::is_anthropic_model(
            "claude-3-5-sonnet-20241022"
        ));
        assert!(GenericLlmAdapter::is_anthropic_model("claude-opus-4"));
        assert!(!GenericLlmAdapter::is_anthropic_model("gpt-4o"));
        assert!(!GenericLlmAdapter::is_anthropic_model("o1-preview"));
        assert!(!GenericLlmAdapter::is_anthropic_model("mistral-7b"));
    }

    #[test]
    fn msg_to_oai_system() {
        let msg = Message::System {
            content: "You are helpful.".into(),
        };
        let oai = msg_to_oai(&msg);
        assert_eq!(oai.role, "system");
        assert_eq!(
            oai.content,
            serde_json::Value::String("You are helpful.".into())
        );
    }

    #[test]
    fn msg_to_oai_tool() {
        let msg = Message::Tool {
            tool_call_id: "call_1".into(),
            content: "result".into(),
        };
        let oai = msg_to_oai(&msg);
        assert_eq!(oai.role, "tool");
        assert_eq!(oai.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn oai_tc_arguments_parse() {
        let tc = OaiToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: OaiFunction {
                name: "search".into(),
                arguments: r#"{"query":"rust"}"#.into(),
            },
        };
        let call = oai_tc_to_tool_call(tc);
        assert_eq!(call.arguments["query"], "rust");
    }

    #[test]
    fn finish_reason_mapping() {
        assert_eq!(parse_oai_finish_reason(Some("stop")), FinishReason::Stop);
        assert_eq!(
            parse_oai_finish_reason(Some("tool_calls")),
            FinishReason::ToolCalls
        );
        assert_eq!(
            parse_oai_finish_reason(Some("length")),
            FinishReason::Length
        );
        assert_eq!(
            parse_oai_finish_reason(Some("content_filter")),
            FinishReason::ContentFilter
        );
        assert_eq!(parse_oai_finish_reason(None), FinishReason::Stop);
    }

    #[test]
    fn status_to_err_rate_limited() {
        let e = status_to_llm_err(429, "rate limit exceeded");
        assert!(matches!(e, LlmError::RateLimited(_)));
    }

    #[test]
    fn status_to_err_context_length() {
        let e = status_to_llm_err(400, "context_length_exceeded on this model");
        assert!(matches!(e, LlmError::ContextLengthExceeded));
    }

    #[test]
    fn builder_builds_without_keys() {
        // Should build fine even with no keys (errors come at call time).
        let adapter = GenericLlmAdapter::builder().build();
        assert!(adapter.is_ok());
    }

    #[test]
    fn oai_request_serialises_without_empty_tools() {
        let req = OaiRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            response_format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        // tools array should be absent (skip_serializing_if = Vec::is_empty)
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn oai_response_formats_have_exact_wire_shape() {
        assert!(oai_response_format(&ResponseFormat::Text).is_none());
        assert_eq!(
            serde_json::to_value(oai_response_format(&ResponseFormat::JsonObject).unwrap())
                .unwrap(),
            serde_json::json!({"type": "json_object"})
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": true
        });
        let format = ResponseFormat::JsonSchema {
            name: "answer".into(),
            description: Some("An answer".into()),
            schema: schema.clone(),
            strict: false,
        };
        assert_eq!(
            serde_json::to_value(oai_response_format(&format).unwrap()).unwrap(),
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "description": "An answer",
                    "schema": schema,
                    "strict": false
                }
            })
        );
    }

    #[test]
    fn openai_message_deserializes_refusal_separately() {
        let message: OaiChoiceMessage = serde_json::from_value(serde_json::json!({
            "content": null,
            "refusal": "I cannot comply"
        }))
        .unwrap();

        assert_eq!(message.content, None);
        assert_eq!(message.refusal.as_deref(), Some("I cannot comply"));
    }

    #[tokio::test]
    async fn anthropic_rejects_structured_formats_before_provider_io() {
        let adapter = GenericLlmAdapter::builder().build().unwrap();
        let request = LlmRequest {
            model_policy: langchart_model::policy::ModelPolicy {
                model: Some("claude-test".into()),
                ..Default::default()
            },
            messages: vec![],
            tools: vec![],
            response_format: ResponseFormat::JsonSchema {
                name: "result".into(),
                description: None,
                schema: serde_json::json!({"type": "object"}),
                strict: true,
            },
        };

        assert!(matches!(
            adapter.complete(request).await,
            Err(LlmError::UnsupportedResponseFormat {
                requested: ResponseFormatKind::JsonSchema,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn openai_http_body_preserves_schema_and_response_refusal() {
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::TcpListener,
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request_bytes = Vec::new();
            let mut buffer = [0_u8; 4096];
            let header_end = loop {
                let read = socket.read(&mut buffer).await.unwrap();
                assert!(read > 0, "client closed before sending HTTP headers");
                request_bytes.extend_from_slice(&buffer[..read]);
                if let Some(position) = request_bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    break position + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request_bytes[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            while request_bytes.len() < header_end + content_length {
                let read = socket.read(&mut buffer).await.unwrap();
                assert!(read > 0, "client closed before sending HTTP body");
                request_bytes.extend_from_slice(&buffer[..read]);
            }

            let body: serde_json::Value =
                serde_json::from_slice(&request_bytes[header_end..header_end + content_length])
                    .unwrap();
            let response_body = serde_json::json!({
                "model": "test-model",
                "choices": [{
                    "message": {"content": null, "refusal": "declined"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            body
        });

        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(format!("http://{address}"))
            .build()
            .unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": true
        });
        let response = adapter
            .complete(LlmRequest {
                model_policy: langchart_model::policy::ModelPolicy {
                    model: Some("test-model".into()),
                    ..Default::default()
                },
                messages: vec![Message::User {
                    content: "answer".into(),
                }],
                tools: vec![],
                response_format: ResponseFormat::JsonSchema {
                    name: "answer".into(),
                    description: Some("Exact answer".into()),
                    schema: schema.clone(),
                    strict: false,
                },
            })
            .await
            .unwrap();
        let captured = server.await.unwrap();

        assert_eq!(
            captured["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "description": "Exact answer",
                    "schema": schema,
                    "strict": false
                }
            })
        );
        assert_eq!(response.content, None);
        assert_eq!(response.refusal.as_deref(), Some("declined"));
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }
}
