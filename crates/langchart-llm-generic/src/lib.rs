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
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use langchart_adapters::llm::{
    FinishReason, LlmAdapter, LlmError, LlmEventStream, LlmRequest, LlmResponse, LlmStreamEvent,
    Message, ModelInfo, ResponseBodyMetadata, ResponseFormat, TokenUsage, ToolCall, ToolDefinition,
    TransportStage, buffered_response_stream,
};
use reqwest::header::HeaderMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::io::{Cursor, Read};
use std::pin::Pin;
use std::time::Duration;
use tokio::io::{AsyncRead, BufReader};
use tokio_util::io::{ReaderStream, StreamReader};
use tracing::{debug, warn};

// ── Constants ─────────────────────────────────────────────────────────────────

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_FIRST_BYTE_TIMEOUT_SECS: u64 = 300;
const DEFAULT_STREAM_IDLE_TIMEOUT_SECS: u64 = 120;
const DEFAULT_TOTAL_GENERATION_TIMEOUT_SECS: u64 = 900;
const DEFAULT_MAX_RESPONSE_BODY_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_ENCODED_RESPONSE_BODY_BYTES: usize = 16 * 1024 * 1024;
const ACCEPTED_CONTENT_ENCODINGS: &str = "gzip, br, deflate, zstd";

// ── Builder ───────────────────────────────────────────────────────────────────

/// Builder for [`GenericLlmAdapter`].
#[derive(Default)]
pub struct GenericLlmAdapterBuilder {
    openai_api_key: Option<String>,
    anthropic_api_key: Option<String>,
    openai_base_url: Option<String>,
    total_generation_timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
    first_byte_timeout: Option<Duration>,
    stream_idle_timeout: Option<Duration>,
    max_response_body_bytes: Option<usize>,
    max_encoded_response_body_bytes: Option<usize>,
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
        self.total_generation_timeout = Some(t);
        self
    }

    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    pub fn first_byte_timeout(mut self, timeout: Duration) -> Self {
        self.first_byte_timeout = Some(timeout);
        self
    }

    pub fn stream_idle_timeout(mut self, timeout: Duration) -> Self {
        self.stream_idle_timeout = Some(timeout);
        self
    }

    pub fn total_generation_timeout(mut self, timeout: Duration) -> Self {
        self.total_generation_timeout = Some(timeout);
        self
    }

    /// Set the maximum decoded JSON response size. The default is 16 MiB.
    pub fn max_response_body_bytes(mut self, bytes: usize) -> Self {
        self.max_response_body_bytes = Some(bytes);
        self
    }

    /// Set the maximum encoded response size read from the network. The default
    /// is 16 MiB. This is enforced independently of the decoded body limit.
    pub fn max_encoded_response_body_bytes(mut self, bytes: usize) -> Self {
        self.max_encoded_response_body_bytes = Some(bytes);
        self
    }

    pub fn build(self) -> Result<GenericLlmAdapter, BuildError> {
        let connect_timeout = self
            .connect_timeout
            .unwrap_or(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS));
        let client = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .build()
            .map_err(|e| BuildError::Client(e.to_string()))?;

        Ok(GenericLlmAdapter {
            client,
            openai_api_key: self.openai_api_key,
            anthropic_api_key: self.anthropic_api_key,
            openai_base_url: self
                .openai_base_url
                .unwrap_or_else(|| OPENAI_BASE_URL.to_string()),
            max_response_body_bytes: self
                .max_response_body_bytes
                .unwrap_or(DEFAULT_MAX_RESPONSE_BODY_BYTES),
            max_encoded_response_body_bytes: self
                .max_encoded_response_body_bytes
                .unwrap_or(DEFAULT_MAX_ENCODED_RESPONSE_BODY_BYTES),
            first_byte_timeout: self
                .first_byte_timeout
                .unwrap_or(Duration::from_secs(DEFAULT_FIRST_BYTE_TIMEOUT_SECS)),
            stream_idle_timeout: self
                .stream_idle_timeout
                .unwrap_or(Duration::from_secs(DEFAULT_STREAM_IDLE_TIMEOUT_SECS)),
            total_generation_timeout: self
                .total_generation_timeout
                .unwrap_or(Duration::from_secs(DEFAULT_TOTAL_GENERATION_TIMEOUT_SECS)),
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
    max_response_body_bytes: usize,
    max_encoded_response_body_bytes: usize,
    first_byte_timeout: Duration,
    stream_idle_timeout: Duration,
    total_generation_timeout: Duration,
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
            tokio::time::timeout(
                self.total_generation_timeout,
                self.complete_anthropic(&model, &request),
            )
            .await
            .map_err(|_| generation_timeout_error("total generation deadline exceeded"))?
        } else {
            collect_completed_response(self.complete_openai_stream(&model, &request).await?).await
        }
    }

    async fn complete_stream(&self, request: LlmRequest) -> Result<LlmEventStream, LlmError> {
        let model = request
            .model_policy
            .model
            .clone()
            .unwrap_or_else(|| "gpt-4o-mini".to_string());

        if Self::is_anthropic_model(&model) {
            let response = tokio::time::timeout(
                self.total_generation_timeout,
                self.complete_anthropic(&model, &request),
            )
            .await
            .map_err(|_| generation_timeout_error("total generation deadline exceeded"))??;
            Ok(buffered_response_stream(response))
        } else {
            self.complete_openai_stream(&model, &request).await
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
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OaiStreamOptions>,
}

#[derive(Serialize)]
struct OaiStreamOptions {
    include_usage: bool,
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
    #[serde(default)]
    model: Option<String>,
    choices: Vec<OaiChoice>,
    #[serde(default)]
    usage: Option<OaiUsage>,
}

#[derive(Deserialize)]
struct OaiChoice {
    message: OaiChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OaiChoiceMessage {
    #[serde(default)]
    content: Option<OaiContent>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OaiToolCall>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OaiContentPart {
    Text(String),
    Object {
        #[serde(default)]
        text: Option<String>,
    },
}

impl OaiContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .filter_map(|part| match part {
                    OaiContentPart::Text(text) => Some(text),
                    OaiContentPart::Object { text } => text,
                })
                .collect(),
        }
    }
}

#[derive(Deserialize, Default)]
struct OaiUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

impl OaiUsage {
    fn into_token_usage(self) -> TokenUsage {
        let prompt_tokens = self.prompt_tokens.unwrap_or(0);
        let completion_tokens = self.completion_tokens.unwrap_or(0);
        TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: self
                .total_tokens
                .unwrap_or_else(|| prompt_tokens.saturating_add(completion_tokens)),
        }
    }
}

#[derive(Deserialize)]
struct OaiStreamChunk {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OaiStreamChoice>,
    #[serde(default)]
    usage: Option<OaiUsage>,
}

#[derive(Deserialize)]
struct OaiStreamChoice {
    #[serde(default)]
    delta: OaiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct OaiStreamDelta {
    #[serde(default)]
    content: Option<OaiContent>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OaiStreamToolCall>,
}

#[derive(Deserialize)]
struct OaiStreamToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OaiStreamFunction>,
}

#[derive(Deserialize)]
struct OaiStreamFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
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
    async fn complete_openai_stream(
        &self,
        model: &str,
        req: &LlmRequest,
    ) -> Result<LlmEventStream, LlmError> {
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
            stream: true,
            stream_options: Some(OaiStreamOptions {
                include_usage: true,
            }),
        };

        debug!(model = model, "openai request");

        let url = format!("{}/chat/completions", self.openai_base_url);
        let mut req_builder = self
            .client
            .post(&url)
            .header(reqwest::header::ACCEPT_ENCODING, ACCEPTED_CONTENT_ENCODINGS);

        // Only set bearer auth if API key is configured (not needed for local endpoints).
        if let Some(key) = api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let total_deadline = tokio::time::Instant::now() + self.total_generation_timeout;
        let headers_deadline = std::cmp::min(
            total_deadline,
            tokio::time::Instant::now() + self.first_byte_timeout,
        );
        let response = tokio::time::timeout_at(headers_deadline, req_builder.json(&body).send())
            .await
            .map_err(|_| generation_timeout_error("response headers deadline exceeded"))?
            .map_err(|error| http_to_llm_err(&error))?;

        if !response.status().is_success() {
            return match tokio::time::timeout_at(
                total_deadline,
                self.decode_response::<serde_json::Value>(response),
            )
            .await
            {
                Ok(Err(error)) => Err(error),
                Ok(Ok(_)) => Err(LlmError::Provider(
                    "unexpected successful error response decode".into(),
                )),
                Err(_) => Err(generation_timeout_error(
                    "total generation deadline exceeded while reading error response",
                )),
            };
        }

        let request_id = response_request_id(response.headers());
        let metadata = response_metadata(response.headers());
        let byte_stream = decoded_streaming_body(
            response,
            self.max_encoded_response_body_bytes,
            self.max_response_body_bytes,
            self.first_byte_timeout,
            self.stream_idle_timeout,
            total_deadline,
        )?;
        let event_stream = byte_stream.eventsource();
        Ok(openai_event_stream(
            event_stream,
            model.to_string(),
            req.response_format.clone(),
            request_id,
            metadata,
        ))
    }

    async fn list_openai_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let api_key = self.openai_api_key.as_deref().filter(|k| !k.is_empty());

        let url = format!("{}/models", self.openai_base_url);
        let mut req_builder = self
            .client
            .get(&url)
            .header(reqwest::header::ACCEPT_ENCODING, ACCEPTED_CONTENT_ENCODINGS);

        // Only set bearer auth if API key is configured (not needed for local endpoints).
        if let Some(key) = api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let resp = req_builder.send().await.map_err(|e| http_to_llm_err(&e))?;

        let models: OaiModelsResponse = self.decode_response(resp).await?;

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
            .header(reqwest::header::ACCEPT_ENCODING, ACCEPTED_CONTENT_ENCODINGS)
            .json(&body)
            .send()
            .await
            .map_err(|e| http_to_llm_err(&e))?;

        let anthropic: AnthropicResponse = self.decode_response(resp).await?;

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
            model: model.to_string(),
            reported_model: Some(anthropic.model),
        })
    }

    async fn decode_response<T: DeserializeOwned>(
        &self,
        response: reqwest::Response,
    ) -> Result<T, LlmError> {
        let status = response.status().as_u16();
        let retry_after = parse_retry_after(response.headers());
        let (encoded_body, mut metadata) =
            read_bounded_body(response, self.max_encoded_response_body_bytes).await?;

        if !(200..300).contains(&status) {
            return Err(LlmError::Http {
                status,
                retry_after,
                body_metadata: Box::new(metadata),
            });
        }

        let body = decode_content(
            encoded_body,
            metadata.content_encoding.as_deref(),
            self.max_response_body_bytes,
        )
        .map_err(|error| LlmError::Decode {
            status,
            body_metadata: Box::new(metadata.clone()),
            json_path: None,
            line: None,
            column: None,
            cause: error.to_string(),
            likely_truncated: metadata
                .content_length
                .is_some_and(|declared| declared > metadata.body_len as u64),
        })?;
        populate_decoded_body_fingerprint(&mut metadata, &body);

        let value: serde_json::Value = serde_json::from_slice(&body).map_err(|error| {
            let likely_truncated = json_error_likely_truncated(&error, &metadata);
            LlmError::Decode {
                status,
                body_metadata: Box::new(metadata.clone()),
                json_path: None,
                line: Some(error.line()),
                column: Some(error.column()),
                cause: error.to_string(),
                likely_truncated,
            }
        })?;

        // Some imperfect OpenAI-compatible servers send provider errors with 200.
        if value.get("error").is_some() {
            return Err(LlmError::Http {
                status,
                retry_after,
                body_metadata: Box::new(metadata),
            });
        }

        let mut deserializer = serde_json::Deserializer::from_slice(&body);
        serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
            let inner = error.inner();
            let likely_truncated = json_error_likely_truncated(inner, &metadata);
            LlmError::Decode {
                status,
                body_metadata: Box::new(metadata),
                json_path: Some(error.path().to_string()),
                line: Some(inner.line()),
                column: Some(inner.column()),
                cause: inner.to_string(),
                likely_truncated,
            }
        })
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

async fn read_bounded_body(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<(Vec<u8>, ResponseBodyMetadata), LlmError> {
    let status = response.status().as_u16();
    let mut metadata = response_metadata(response.headers());
    let capture_limit = max_bytes.saturating_add(1);
    let mut body = Vec::with_capacity(
        metadata
            .content_length
            .and_then(|length| usize::try_from(length).ok())
            .unwrap_or(0)
            .min(capture_limit),
    );

    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let remaining = capture_limit.saturating_sub(body.len());
                body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if body.len() > max_bytes || (remaining == 0 && !chunk.is_empty()) {
                    populate_body_fingerprint(&mut metadata, &body);
                    return Err(LlmError::Decode {
                        status,
                        body_metadata: Box::new(metadata),
                        json_path: None,
                        line: None,
                        column: None,
                        cause: format!(
                            "encoded response body exceeds configured limit of {max_bytes} bytes"
                        ),
                        likely_truncated: true,
                    });
                }
            }
            Ok(None) => break,
            Err(error) => {
                populate_body_fingerprint(&mut metadata, &body);
                return Err(LlmError::Decode {
                    status,
                    body_metadata: Box::new(metadata),
                    json_path: None,
                    line: None,
                    column: None,
                    cause: format!("response body read failed: {error}"),
                    likely_truncated: true,
                });
            }
        }
    }

    populate_body_fingerprint(&mut metadata, &body);
    Ok((body, metadata))
}

fn response_metadata(headers: &HeaderMap) -> ResponseBodyMetadata {
    ResponseBodyMetadata {
        content_type: header_string(headers, reqwest::header::CONTENT_TYPE.as_str()),
        content_encoding: header_string(headers, reqwest::header::CONTENT_ENCODING.as_str()),
        content_length: header_string(headers, reqwest::header::CONTENT_LENGTH.as_str())
            .and_then(|value| value.parse().ok()),
        request_id: ["x-request-id", "request-id", "x-correlation-id", "cf-ray"]
            .into_iter()
            .find_map(|name| header_string(headers, name)),
        body_len: 0,
        body_hash: sha256_hex(&[]),
        decoded_body_len: None,
        decoded_body_hash: None,
    }
}

fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn populate_body_fingerprint(metadata: &mut ResponseBodyMetadata, body: &[u8]) {
    metadata.body_len = body.len();
    metadata.body_hash = sha256_hex(body);
}

fn populate_decoded_body_fingerprint(metadata: &mut ResponseBodyMetadata, body: &[u8]) {
    metadata.decoded_body_len = Some(body.len());
    metadata.decoded_body_hash = Some(sha256_hex(body));
}

#[derive(Debug, thiserror::Error)]
enum ContentDecodeError {
    #[error("unsupported content encoding `{0}`")]
    Unsupported(String),
    #[error("invalid {encoding} response body: {cause}")]
    Invalid { encoding: String, cause: String },
    #[error("decoded response body exceeds configured limit of {limit} bytes")]
    TooLarge { limit: usize },
}

fn decode_content(
    mut body: Vec<u8>,
    content_encoding: Option<&str>,
    max_decoded_bytes: usize,
) -> Result<Vec<u8>, ContentDecodeError> {
    let encodings = content_encoding
        .into_iter()
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("identity"))
        .collect::<Vec<_>>();

    if encodings.is_empty() && body.len() > max_decoded_bytes {
        return Err(ContentDecodeError::TooLarge {
            limit: max_decoded_bytes,
        });
    }

    for encoding in encodings.into_iter().rev() {
        body = decode_one_content_encoding(body, encoding, max_decoded_bytes)?;
    }
    Ok(body)
}

fn decode_one_content_encoding(
    body: Vec<u8>,
    encoding: &str,
    max_decoded_bytes: usize,
) -> Result<Vec<u8>, ContentDecodeError> {
    if encoding.eq_ignore_ascii_case("gzip") || encoding.eq_ignore_ascii_case("x-gzip") {
        return read_decoded(
            flate2::read::GzDecoder::new(Cursor::new(body)),
            "gzip",
            max_decoded_bytes,
        );
    }
    if encoding.eq_ignore_ascii_case("br") {
        return read_decoded(
            brotli::Decompressor::new(Cursor::new(body), 4096),
            "br",
            max_decoded_bytes,
        );
    }
    if encoding.eq_ignore_ascii_case("deflate") {
        match read_decoded(
            flate2::read::ZlibDecoder::new(Cursor::new(body.clone())),
            "deflate",
            max_decoded_bytes,
        ) {
            Err(ContentDecodeError::Invalid { .. }) => {
                return read_decoded(
                    flate2::read::DeflateDecoder::new(Cursor::new(body)),
                    "deflate",
                    max_decoded_bytes,
                );
            }
            result => return result,
        }
    }
    if encoding.eq_ignore_ascii_case("zstd") {
        let decoder = zstd::stream::read::Decoder::new(Cursor::new(body)).map_err(|error| {
            ContentDecodeError::Invalid {
                encoding: "zstd".into(),
                cause: error.to_string(),
            }
        })?;
        return read_decoded(decoder, "zstd", max_decoded_bytes);
    }

    Err(ContentDecodeError::Unsupported(encoding.to_owned()))
}

fn read_decoded(
    reader: impl Read,
    encoding: &str,
    max_decoded_bytes: usize,
) -> Result<Vec<u8>, ContentDecodeError> {
    let mut decoded = Vec::with_capacity(max_decoded_bytes.min(64 * 1024));
    reader
        .take(max_decoded_bytes.saturating_add(1) as u64)
        .read_to_end(&mut decoded)
        .map_err(|error| ContentDecodeError::Invalid {
            encoding: encoding.to_owned(),
            cause: error.to_string(),
        })?;
    if decoded.len() > max_decoded_bytes {
        return Err(ContentDecodeError::TooLarge {
            limit: max_decoded_bytes,
        });
    }
    Ok(decoded)
}

fn sha256_hex(body: &[u8]) -> String {
    let digest = Sha256::digest(body);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    header_string(headers, reqwest::header::RETRY_AFTER.as_str())
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
}

fn json_error_likely_truncated(error: &serde_json::Error, metadata: &ResponseBodyMetadata) -> bool {
    error.classify() == serde_json::error::Category::Eof
        || metadata
            .content_length
            .is_some_and(|declared| declared > metadata.body_len as u64)
}

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
    LlmError::Transport {
        stage: if e.is_connect() {
            TransportStage::Connect
        } else {
            TransportStage::Send
        },
        retryable: e.is_connect() || e.is_timeout(),
        cause: e.to_string(),
    }
}

// ── Unit tests (no network required) ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::llm::{Message, ResponseFormatKind};
    use std::io::Write as IoWrite;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        task::JoinHandle,
    };

    async fn serve_once(
        status: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> (String, JoinHandle<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let mut response = format!(
            "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n",
            body.len()
        )
        .into_bytes();
        for (name, value) in headers {
            response.extend_from_slice(format!("{name}: {value}\r\n").as_bytes());
        }
        response.extend_from_slice(b"\r\n");
        response.extend_from_slice(body);

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                assert!(read > 0, "client closed before sending HTTP headers");
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            socket.write_all(&response).await.unwrap();
            request
        });

        (format!("http://{address}"), server)
    }

    fn openai_request(model: &str) -> LlmRequest {
        LlmRequest {
            model_policy: langchart_model::policy::ModelPolicy {
                model: Some(model.into()),
                ..Default::default()
            },
            messages: vec![Message::User {
                content: "answer".into(),
            }],
            tools: vec![],
            response_format: ResponseFormat::Text,
        }
    }

    fn encode_fixture(body: &[u8], encoding: &str) -> Vec<u8> {
        match encoding {
            "gzip" => {
                let mut encoder =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(body).unwrap();
                encoder.finish().unwrap()
            }
            "br" => {
                let mut encoded = Vec::new();
                {
                    let mut encoder = brotli::CompressorWriter::new(&mut encoded, 4096, 5, 22);
                    encoder.write_all(body).unwrap();
                }
                encoded
            }
            "deflate" => {
                let mut encoder =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(body).unwrap();
                encoder.finish().unwrap()
            }
            "zstd" => zstd::stream::encode_all(Cursor::new(body), 3).unwrap(),
            other => panic!("unsupported fixture encoding {other}"),
        }
    }

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
    fn builder_builds_without_keys() {
        // Should build fine even with no keys (errors come at call time).
        let adapter = GenericLlmAdapter::builder().build();
        assert!(adapter.is_ok());
    }

    #[test]
    fn supported_content_encodings_decode_with_stacked_order() {
        let body = b"decoded response";
        for encoding in ["gzip", "br", "deflate", "zstd"] {
            let encoded = encode_fixture(body, encoding);
            assert_eq!(decode_content(encoded, Some(encoding), 1024).unwrap(), body);
        }

        let gzip_then_brotli = encode_fixture(&encode_fixture(body, "gzip"), "br");
        assert_eq!(
            decode_content(gzip_then_brotli, Some("gzip, br"), 1024).unwrap(),
            body
        );
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

        assert!(message.content.is_none());
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

    #[tokio::test]
    async fn openai_accepts_missing_usage_model_and_inaccurate_content_type() {
        let body = br#"{
            "choices": [{
                "message": {
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "provider_extension"},
                        "world"
                    ]
                },
                "finish_reason": "stop"
            }]
        }"#;
        let (base_url, server) =
            serve_once("200 OK", &[("content-type", "text/plain")], body).await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let response = adapter
            .complete(openai_request("resolved-model"))
            .await
            .unwrap();
        let request = String::from_utf8_lossy(&server.await.unwrap()).to_ascii_lowercase();

        assert_eq!(response.content.as_deref(), Some("hello world"));
        assert_eq!(response.usage.total_tokens, 0);
        assert_eq!(response.model, "resolved-model");
        assert_eq!(response.reported_model, None);
        assert!(request.contains("accept-encoding: gzip, br, deflate, zstd"));
    }

    #[tokio::test]
    async fn truncated_json_has_safe_structured_diagnostics() {
        let body = br#"{"choices":[{"message":{"content":"partial"}"#;
        let (base_url, server) = serve_once(
            "200 OK",
            &[
                ("content-type", "application/json"),
                ("x-request-id", "req-123"),
            ],
            body,
        )
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        server.await.unwrap();

        match error {
            LlmError::Decode {
                status,
                body_metadata,
                likely_truncated,
                line,
                column,
                ..
            } => {
                assert_eq!(status, 200);
                assert_eq!(
                    body_metadata.content_type.as_deref(),
                    Some("application/json")
                );
                assert_eq!(body_metadata.request_id.as_deref(), Some("req-123"));
                assert_eq!(body_metadata.body_len, body.len());
                assert_eq!(body_metadata.body_hash, sha256_hex(body));
                assert!(likely_truncated);
                assert!(line.is_some());
                assert!(column.is_some());
            }
            other => panic!("expected decode error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn disconnect_mid_body_is_classified_as_likely_truncated() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let request_len = socket.read(&mut request).await.unwrap();
            assert!(request_len > 0);
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 100\r\nconnection: close\r\n\r\n{\"choices\":[",
                )
                .await
                .unwrap();
        });
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(format!("http://{address}"))
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        server.await.unwrap();

        match error {
            LlmError::Decode {
                body_metadata,
                likely_truncated,
                ..
            } => {
                assert_eq!(body_metadata.content_length, Some(100));
                assert!(body_metadata.body_len < 100);
                assert!(likely_truncated);
            }
            other => panic!("expected decode error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn incompatible_shape_reports_json_path() {
        let body = br#"{
            "choices": [{
                "message": {"content": {"unexpected": true}},
                "finish_reason": "stop"
            }]
        }"#;
        let (base_url, server) = serve_once("200 OK", &[], body).await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        server.await.unwrap();

        assert!(matches!(
            error,
            LlmError::Decode {
                json_path: Some(path),
                likely_truncated: false,
                ..
            } if path.contains("choices[0].message.content")
        ));
    }

    #[tokio::test]
    async fn oversized_body_is_rejected_at_configured_limit() {
        let body = br#"{"choices":[]}"#;
        let (base_url, server) = serve_once("200 OK", &[], body).await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .max_encoded_response_body_bytes(8)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        server.await.unwrap();

        match error {
            LlmError::Decode {
                body_metadata,
                likely_truncated,
                ..
            } => {
                assert_eq!(body_metadata.body_len, 9);
                assert!(likely_truncated);
            }
            other => panic!("expected decode error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn decoded_body_limit_rejects_compression_expansion() {
        let decoded = vec![b'a'; 4096];
        let body = encode_fixture(&decoded, "gzip");
        let wire_len = body.len();
        let (base_url, server) = serve_once("200 OK", &[("content-encoding", "gzip")], &body).await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .max_response_body_bytes(128)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        server.await.unwrap();

        match error {
            LlmError::Decode {
                body_metadata,
                cause,
                likely_truncated,
                ..
            } => {
                assert_eq!(body_metadata.body_len, wire_len);
                assert_eq!(body_metadata.decoded_body_len, None);
                assert!(cause.contains("decoded response body exceeds"));
                assert!(!likely_truncated);
            }
            other => panic!("expected decode error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_error_preserves_retry_and_request_metadata() {
        let body = br#"{"error":{"message":"busy"}}"#;
        let (base_url, server) = serve_once(
            "429 Too Many Requests",
            &[("retry-after", "12"), ("x-correlation-id", "corr-1")],
            body,
        )
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        server.await.unwrap();

        match error {
            LlmError::Http {
                status: 429,
                retry_after: Some(delay),
                body_metadata,
            } => {
                assert_eq!(delay, Duration::from_secs(12));
                assert_eq!(body_metadata.request_id.as_deref(), Some("corr-1"));
            }
            other => panic!("expected HTTP 429 error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn successful_status_with_error_payload_is_classified_as_http_error() {
        let body = br#"{"error":{"message":"provider failed"}}"#;
        let (base_url, server) = serve_once("200 OK", &[], body).await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        server.await.unwrap();

        assert!(matches!(error, LlmError::Http { status: 200, .. }));
    }

    #[tokio::test]
    async fn gzip_compressed_response_is_decoded() {
        let decoded = br#"{
            "model": "reported-model",
            "choices": [{
                "message": {"content": "compressed response"},
                "finish_reason": "stop"
            }]
        }"#;
        let body = encode_fixture(decoded, "gzip");
        let (base_url, server) = serve_once(
            "200 OK",
            &[
                ("content-encoding", "gzip"),
                ("content-type", "application/json"),
            ],
            &body,
        )
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let response = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap();
        let request = String::from_utf8_lossy(&server.await.unwrap()).to_ascii_lowercase();

        assert_eq!(response.content.as_deref(), Some("compressed response"));
        assert_eq!(response.reported_model.as_deref(), Some("reported-model"));
        assert!(request.contains("accept-encoding: gzip, br, deflate, zstd"));
    }

    #[tokio::test]
    async fn corrupt_compression_has_safe_wire_diagnostics() {
        let body = b"not a gzip stream";
        let (base_url, server) = serve_once(
            "200 OK",
            &[("content-encoding", "gzip"), ("x-request-id", "req-gzip")],
            body,
        )
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        server.await.unwrap();

        match error {
            LlmError::Decode {
                body_metadata,
                likely_truncated,
                ..
            } => {
                assert_eq!(body_metadata.content_encoding.as_deref(), Some("gzip"));
                assert_eq!(body_metadata.request_id.as_deref(), Some("req-gzip"));
                assert_eq!(body_metadata.body_len, body.len());
                assert_eq!(body_metadata.body_hash, sha256_hex(body));
                assert_eq!(body_metadata.decoded_body_len, None);
                assert!(!likely_truncated);
            }
            other => panic!("expected decode error, got {other:?}"),
        }
    }
}
