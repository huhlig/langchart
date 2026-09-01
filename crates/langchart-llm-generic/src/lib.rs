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

// The public adapter trait returns the shared, diagnostic-rich LlmError by value.
#![allow(clippy::result_large_err)]
//!
//! For Azure / Ollama / vLLM:
//!
//! ```text
//! let adapter = GenericLlmAdapter::builder()
//!     .openai_api_key("...")
//!     .openai_base_url("http://localhost:11434/v1")  // Ollama
//!     .build()?;
//! ```

use async_compression::tokio::bufread::{BrotliDecoder, GzipDecoder, ZlibDecoder, ZstdDecoder};
use async_trait::async_trait;
use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::TryStreamExt;
use futures::stream::{self, Stream, StreamExt};
use langchart_adapters::llm::{
    FinishReason, LlmAdapter, LlmError, LlmEventStream, LlmRequest, LlmResponse, LlmStreamEvent,
    Message, ModelInfo, ResponseBodyMetadata, ResponseFormat, TokenUsage, ToolCall, ToolDefinition,
    TransportStage,
};
use reqwest::header::HeaderMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::io::{Cursor, Read};
use std::pin::Pin;
use std::time::Duration;
use tokio::io::{AsyncBufRead, BufReader};
use tokio_util::io::{ReaderStream, StreamReader};
use tracing::{debug, warn};

// ── Constants ─────────────────────────────────────────────────────────────────

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";
#[allow(dead_code)]
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_FIRST_BYTE_TIMEOUT_SECS: u64 = 300;
const DEFAULT_STREAM_IDLE_TIMEOUT_SECS: u64 = 120;
const DEFAULT_TOTAL_GENERATION_TIMEOUT_SECS: u64 = 900;
const DEFAULT_MAX_RETRIES: u32 = 3;
const DEFAULT_MAX_RESPONSE_BODY_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_ENCODED_RESPONSE_BODY_BYTES: usize = 16 * 1024 * 1024;
const ACCEPTED_CONTENT_ENCODINGS: &str = "gzip, br, deflate, zstd";
const MAX_EVENT_QUEUE_DEPTH: usize = 64;
type ByteResultStream = Pin<Box<dyn Stream<Item = Result<Bytes, LlmError>> + Send>>;

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
    max_retries: Option<u32>,
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

    /// Maximum number of retries for transient (retryable) errors. Default: 3.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
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
            max_retries: self.max_retries.unwrap_or(DEFAULT_MAX_RETRIES),
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
    max_retries: u32,
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

    /// Retry a fallible async operation with exponential backoff.
    ///
    /// Retries on errors where [`LlmError::is_retryable`] returns `true`.
    /// Respects `retry_after` hints and the total generation deadline.
    async fn with_retry<F, Fut, T>(
        &self,
        total_deadline: tokio::time::Instant,
        mut f: F,
    ) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, LlmError>>,
    {
        let base_delay = Duration::from_millis(500);
        let max_delay = Duration::from_secs(30);

        for attempt in 0..=self.max_retries {
            if tokio::time::Instant::now() >= total_deadline {
                return Err(generation_timeout_error(
                    "total generation deadline exceeded before retry",
                ));
            }

            match f().await {
                Err(e) if e.is_retryable() && attempt < self.max_retries => {
                    let delay = e.retry_after().unwrap_or_else(|| {
                        let exponential = base_delay * 2u32.saturating_pow(attempt);
                        let capped = exponential.min(max_delay);
                        // Simple jitter: 1–25% based on current time nanos.
                        let nanos = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0);
                        let jitter_pct = (nanos as u32 % 25) + 1;
                        capped + capped * jitter_pct / 100
                    });

                    // Don't wait past the deadline.
                    let remaining =
                        total_deadline.saturating_duration_since(tokio::time::Instant::now());
                    let wait = delay.min(remaining);

                    if wait.is_zero() && tokio::time::Instant::now() >= total_deadline {
                        return Err(generation_timeout_error(
                            "total generation deadline exceeded during retry backoff",
                        ));
                    }

                    warn!(
                        attempt = attempt + 1,
                        max = self.max_retries,
                        delay_ms = wait.as_millis(),
                        error = %e,
                        "retrying after transient error"
                    );
                    if !wait.is_zero() {
                        tokio::time::sleep(wait).await;
                    }
                }
                Err(e) => return Err(e),
                Ok(response) => return Ok(response),
            }
        }
        unreachable!("loop always returns")
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
            let total_deadline = tokio::time::Instant::now() + self.total_generation_timeout;
            let model = model.clone();
            self.with_retry(total_deadline, || {
                let model = model.clone();
                let request = request.clone();
                async move { self.complete_anthropic(&model, &request).await }
            })
            .await
        } else {
            let total_deadline = tokio::time::Instant::now() + self.total_generation_timeout;
            let model = model.clone();
            self.with_retry(total_deadline, || {
                let model = model.clone();
                let request = request.clone();
                async move { self.complete_openai(&model, &request).await }
            })
            .await
        }
    }

    async fn complete_stream(&self, request: LlmRequest) -> Result<LlmEventStream, LlmError> {
        let model = request
            .model_policy
            .model
            .clone()
            .unwrap_or_else(|| "gpt-4o-mini".to_string());

        if Self::is_anthropic_model(&model) {
            let total_deadline = tokio::time::Instant::now() + self.total_generation_timeout;
            let model = model.clone();
            self.with_retry(total_deadline, || {
                let model = model.clone();
                let request = request.clone();
                async move {
                    self.complete_anthropic_stream(&model, &request, total_deadline)
                        .await
                }
            })
            .await
        } else {
            let total_deadline = tokio::time::Instant::now() + self.total_generation_timeout;
            let model = model.clone();
            self.with_retry(total_deadline, || {
                let model = model.clone();
                let request = request.clone();
                async move { self.complete_openai_stream(&model, &request).await }
            })
            .await
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

#[derive(Clone, Serialize, Deserialize)]
struct OaiToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: OaiFunction,
}

#[derive(Clone, Serialize, Deserialize)]
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

    #[allow(dead_code)]
    fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(_) => None,
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
            stream: false,
            stream_options: None,
        };

        debug!(model = model, "openai request");

        let url = format!("{}/chat/completions", self.openai_base_url);
        let mut req_builder = self
            .client
            .post(&url)
            .header(reqwest::header::ACCEPT_ENCODING, ACCEPTED_CONTENT_ENCODINGS);

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

        let oai: OaiResponse =
            tokio::time::timeout_at(total_deadline, self.decode_response(response))
                .await
                .map_err(|_| generation_timeout_error("total generation deadline exceeded"))??;

        let mut text_content: Option<String> = None;
        let mut refusal: Option<String> = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for choice in &oai.choices {
            if let Some(content) = &choice.message.content {
                let text = match content {
                    OaiContent::Text(t) => t.clone(),
                    OaiContent::Parts(parts) => parts
                        .iter()
                        .filter_map(|part| match part {
                            OaiContentPart::Text(t) => Some(t.as_str()),
                            OaiContentPart::Object { text } => text.as_deref(),
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                };
                if !text.is_empty() {
                    text_content = Some(text);
                }
            }
            if let Some(r) = &choice.message.refusal {
                refusal = Some(r.clone());
            }
            for tc in &choice.message.tool_calls {
                tool_calls.push(oai_tc_to_tool_call(tc.clone()));
            }
        }

        let finish_reason = oai
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
            .map(|r| parse_oai_finish_reason(Some(r)))
            .unwrap_or(FinishReason::Stop);

        let usage = oai.usage.map(|u| u.into_token_usage()).unwrap_or_default();

        Ok(LlmResponse {
            content: text_content,
            tool_calls,
            usage,
            finish_reason,
            refusal,
            model: model.to_string(),
            reported_model: oai.model,
        })
    }

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
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
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
            stream: false,
        };

        debug!(model = model, "anthropic request");

        let url = format!("{}/messages", ANTHROPIC_BASE_URL);
        let total_deadline = tokio::time::Instant::now() + self.total_generation_timeout;
        let headers_deadline = std::cmp::min(
            total_deadline,
            tokio::time::Instant::now() + self.first_byte_timeout,
        );
        let resp = tokio::time::timeout_at(
            headers_deadline,
            self.client
                .post(&url)
                .header("x-api-key", api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header(reqwest::header::ACCEPT_ENCODING, ACCEPTED_CONTENT_ENCODINGS)
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| generation_timeout_error("response headers deadline exceeded"))?
        .map_err(|error| http_to_llm_err(&error))?;

        let anthropic: AnthropicResponse =
            tokio::time::timeout_at(total_deadline, self.decode_response(resp))
                .await
                .map_err(|_| generation_timeout_error("total generation deadline exceeded"))??;

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

    async fn complete_anthropic_stream(
        &self,
        model: &str,
        req: &LlmRequest,
        total_deadline: tokio::time::Instant,
    ) -> Result<LlmEventStream, LlmError> {
        if !matches!(req.response_format, ResponseFormat::Text) {
            return Err(LlmError::UnsupportedResponseFormat {
                adapter: "anthropic".into(),
                requested: req.response_format.kind(),
            });
        }
        let api_key = self
            .anthropic_api_key
            .as_deref()
            .filter(|key| !key.is_empty())
            .ok_or_else(|| LlmError::Provider("ANTHROPIC_API_KEY not configured".into()))?;

        let mut system = None;
        let mut messages = Vec::new();
        for message in &req.messages {
            match message {
                Message::System { content } => system = Some(content.clone()),
                Message::User { content } => messages.push(AnthropicMessage {
                    role: "user",
                    content: serde_json::Value::String(content.clone()),
                }),
                Message::Assistant { content } => messages.push(AnthropicMessage {
                    role: "assistant",
                    content: serde_json::Value::String(content.clone()),
                }),
                Message::Tool {
                    tool_call_id,
                    content,
                } => messages.push(AnthropicMessage {
                    role: "user",
                    content: serde_json::json!([{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content
                    }]),
                }),
            }
        }
        let body = AnthropicRequest {
            model: model.to_string(),
            max_tokens: req.model_policy.max_tokens.unwrap_or(4096),
            messages,
            system,
            tools: req
                .tools
                .iter()
                .map(|tool| AnthropicTool {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    input_schema: tool.parameters.clone(),
                })
                .collect(),
            temperature: req.model_policy.temperature,
            stream: true,
        };

        let headers_deadline = std::cmp::min(
            total_deadline,
            tokio::time::Instant::now() + self.first_byte_timeout,
        );
        let response = tokio::time::timeout_at(
            headers_deadline,
            self.client
                .post(format!("{ANTHROPIC_BASE_URL}/messages"))
                .header("x-api-key", api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .header(reqwest::header::ACCEPT_ENCODING, ACCEPTED_CONTENT_ENCODINGS)
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| generation_timeout_error("response headers deadline exceeded"))?
        .map_err(|error| http_to_llm_err(&error))?;

        if !response.status().is_success() {
            return self
                .decode_response::<serde_json::Value>(response)
                .await
                .and_then(|_| Err(LlmError::Provider("unexpected error response".into())));
        }
        let request_id = response_request_id(response.headers());
        let metadata = response_metadata(response.headers());
        let bytes = decoded_streaming_body(
            response,
            self.max_encoded_response_body_bytes,
            self.max_response_body_bytes,
            self.first_byte_timeout,
            self.stream_idle_timeout,
            total_deadline,
        )?;
        Ok(anthropic_event_stream(
            bytes.eventsource(),
            model.to_string(),
            request_id,
            metadata,
        ))
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
                request_id: metadata.request_id.clone(),
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
            content_type: metadata.content_type.clone(),
            content_encoding: metadata.content_encoding.clone(),
            body_len: metadata.body_len,
            body_hash: metadata.body_hash.clone(),
            json_path: None,
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
                content_type: metadata.content_type.clone(),
                content_encoding: metadata.content_encoding.clone(),
                body_len: metadata.body_len,
                body_hash: metadata.body_hash.clone(),
                json_path: None,
                cause: error.to_string(),
                likely_truncated,
            }
        })?;

        // Some imperfect OpenAI-compatible servers send provider errors with 200.
        if value.get("error").is_some() {
            return Err(LlmError::Http {
                status,
                retry_after,
                request_id: metadata.request_id.clone(),
                body_metadata: Box::new(metadata),
            });
        }

        let mut deserializer = serde_json::Deserializer::from_slice(&body);
        serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
            let inner = error.inner();
            let likely_truncated = json_error_likely_truncated(inner, &metadata);
            LlmError::Decode {
                status,
                content_type: metadata.content_type.clone(),
                content_encoding: metadata.content_encoding.clone(),
                body_len: metadata.body_len,
                body_hash: metadata.body_hash.clone(),
                json_path: Some(error.path().to_string()),
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
                        content_type: metadata.content_type.clone(),
                        content_encoding: metadata.content_encoding.clone(),
                        body_len: metadata.body_len,
                        body_hash: metadata.body_hash.clone(),
                        json_path: None,
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
                    content_type: metadata.content_type.clone(),
                    content_encoding: metadata.content_encoding.clone(),
                    body_len: metadata.body_len,
                    body_hash: metadata.body_hash.clone(),
                    json_path: None,
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

fn is_false(value: &bool) -> bool {
    !*value
}

fn generation_timeout_error(message: &str) -> LlmError {
    warn!("{message}");
    LlmError::Timeout
}

fn response_request_id(headers: &HeaderMap) -> Option<String> {
    ["x-request-id", "request-id", "x-correlation-id", "cf-ray"]
        .into_iter()
        .find_map(|name| header_string(headers, name))
}

fn decoded_streaming_body(
    response: reqwest::Response,
    max_encoded_bytes: usize,
    max_decoded_bytes: usize,
    first_byte_timeout: Duration,
    idle_timeout: Duration,
    total_deadline: tokio::time::Instant,
) -> Result<ByteResultStream, LlmError> {
    let status = response.status().as_u16();
    let metadata = response_metadata(response.headers());
    let content_encoding = metadata.content_encoding.clone();

    let raw = response
        .bytes_stream()
        .map_err(|error| LlmError::Transport {
            stage: TransportStage::Body,
            retryable: true,
            cause: error.to_string(),
        });

    // Wrap the raw stream to count encoded (wire) bytes before decompression.
    let encoded_bytes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let encoded_counter = encoded_bytes.clone();
    let raw = raw.map(move |result| {
        result.inspect(|bytes| {
            encoded_counter.fetch_add(bytes.len(), std::sync::atomic::Ordering::Relaxed);
        })
    });

    let stream_reader = StreamReader::new(raw.map_err(std::io::Error::other));
    let mut reader: Pin<Box<dyn AsyncBufRead + Send + Unpin>> =
        Box::pin(BufReader::new(stream_reader));

    if let Some(ref encoding) = content_encoding {
        let encodings: Vec<&str> = encoding.split(',').map(str::trim).collect();
        for enc in encodings.iter().rev() {
            if enc.eq_ignore_ascii_case("identity") || enc.is_empty() {
                continue;
            }
            // `prev` becomes the input to the decoder; the replacement is a
            // no-op BufReader (its content doesn't matter since it will be
            // overwritten or dropped on the next iteration / after the loop).
            let prev = std::mem::replace(&mut reader, Box::pin(BufReader::new(tokio::io::empty())));
            reader = match *enc {
                e if e.eq_ignore_ascii_case("gzip") || e.eq_ignore_ascii_case("x-gzip") => {
                    Box::pin(BufReader::new(GzipDecoder::new(BufReader::new(prev))))
                }
                e if e.eq_ignore_ascii_case("br") => {
                    Box::pin(BufReader::new(BrotliDecoder::new(BufReader::new(prev))))
                }
                e if e.eq_ignore_ascii_case("deflate") => {
                    Box::pin(BufReader::new(ZlibDecoder::new(BufReader::new(prev))))
                }
                e if e.eq_ignore_ascii_case("zstd") => {
                    Box::pin(BufReader::new(ZstdDecoder::new(BufReader::new(prev))))
                }
                _ => {
                    return Err(LlmError::Decode {
                        status,
                        content_type: metadata.content_type,
                        content_encoding: Some(encoding.to_string()),
                        body_len: 0,
                        body_hash: String::new(),
                        json_path: None,
                        cause: format!("unsupported content encoding: {enc}"),
                        likely_truncated: false,
                    });
                }
            };
        }
    }

    let decoded_stream = ReaderStream::new(reader).map_err(move |e| LlmError::Decode {
        status,
        content_type: None,
        content_encoding: content_encoding.clone(),
        body_len: 0,
        body_hash: String::new(),
        json_path: None,
        cause: format!("content decoding failed: {e}"),
        likely_truncated: false,
    });

    let initial_wait = first_byte_timeout;
    let idle = idle_timeout;

    struct DecodeState<S> {
        stream: S,
        encoded_bytes: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        decoded: usize,
        seen_first: bool,
        done: bool,
    }

    let init = DecodeState {
        stream: decoded_stream,
        encoded_bytes,
        decoded: 0,
        seen_first: false,
        done: false,
    };

    let result = stream::unfold(init, move |mut st| async move {
        if st.done {
            return None;
        }
        if tokio::time::Instant::now() >= total_deadline {
            st.done = true;
            return Some((
                Err(generation_timeout_error(
                    "total generation deadline exceeded",
                )),
                st,
            ));
        }
        let wait = if st.seen_first { idle } else { initial_wait };
        let deadline = std::cmp::min(total_deadline, tokio::time::Instant::now() + wait);
        match tokio::time::timeout_at(deadline, st.stream.next()).await {
            Err(_elapsed) => {
                st.done = true;
                Some((Err(LlmError::Timeout), st))
            }
            Ok(None) => None,
            Ok(Some(Err(e))) => {
                st.done = true;
                Some((Err(e), st))
            }
            Ok(Some(Ok(bytes))) => {
                st.seen_first = true;
                let encoded = st.encoded_bytes.load(std::sync::atomic::Ordering::Relaxed);
                if encoded > max_encoded_bytes {
                    st.done = true;
                    Some((
                        Err(LlmError::Decode {
                            status,
                            content_type: None,
                            content_encoding: None,
                            body_len: encoded,
                            body_hash: String::new(),
                            json_path: None,
                            cause: format!(
                                "encoded response body exceeds configured limit of \
                                 {max_encoded_bytes} bytes"
                            ),
                            likely_truncated: true,
                        }),
                        st,
                    ))
                } else {
                    st.decoded += bytes.len();
                    if st.decoded > max_decoded_bytes {
                        st.done = true;
                        Some((
                            Err(LlmError::Decode {
                                status,
                                content_type: None,
                                content_encoding: None,
                                body_len: st.decoded,
                                body_hash: String::new(),
                                json_path: None,
                                cause: format!(
                                    "decoded response body exceeds configured limit of \
                                     {max_decoded_bytes} bytes"
                                ),
                                likely_truncated: false,
                            }),
                            st,
                        ))
                    } else {
                        Some((Ok(bytes), st))
                    }
                }
            }
        }
    });

    Ok(Box::pin(result))
}

fn openai_event_stream<S>(
    stream: S,
    model: String,
    response_format: ResponseFormat,
    request_id: Option<String>,
    metadata: ResponseBodyMetadata,
) -> LlmEventStream
where
    S: Stream<
            Item = Result<
                eventsource_stream::Event,
                eventsource_stream::EventStreamError<LlmError>,
            >,
        > + Send
        + Unpin
        + 'static,
{
    struct OaiStreamState<S> {
        stream: S,
        queue: VecDeque<Result<LlmStreamEvent, LlmError>>,
        started: bool,
        done: bool,
        content: String,
        refusal: String,
        tool_calls: Vec<(String, String, String)>,
        open_tool: Option<usize>,
        usage: Option<TokenUsage>,
        finish: Option<FinishReason>,
        reported_model: Option<String>,
        received_bytes: usize,
        finish_event_seen: bool,
        hasher: Sha256,
        model: String,
        request_id: Option<String>,
        metadata: ResponseBodyMetadata,
        response_format: ResponseFormat,
    }

    impl<S> OaiStreamState<S>
    where
        S: Stream<
                Item = Result<
                    eventsource_stream::Event,
                    eventsource_stream::EventStreamError<LlmError>,
                >,
            > + Send
            + Unpin
            + 'static,
    {
        fn start(&mut self) {
            self.started = true;
            self.queue.push_back(Ok(LlmStreamEvent::ResponseStarted {
                request_id: self.request_id.clone(),
                reported_model: self.reported_model.clone(),
            }));
        }

        fn apply_chunk(&mut self, chunk: OaiStreamChunk) {
            if let Some(m) = chunk.model {
                self.reported_model = Some(m);
            }
            if let Some(usage) = chunk.usage {
                let token_usage = usage.into_token_usage();
                self.usage = Some(token_usage.clone());
                self.queue
                    .push_back(Ok(LlmStreamEvent::UsageUpdate { usage: token_usage }));
            }
            for choice in chunk.choices {
                if let Some(content) = choice.delta.content {
                    let text = content.into_text();
                    if !text.is_empty() {
                        self.content.push_str(&text);
                        self.queue
                            .push_back(Ok(LlmStreamEvent::TextDelta { delta: text }));
                    }
                }
                if let Some(reasoning) = choice.delta.reasoning_content {
                    self.queue
                        .push_back(Ok(LlmStreamEvent::ReasoningDelta { delta: reasoning }));
                }
                if let Some(refusal) = choice.delta.refusal {
                    self.refusal.push_str(&refusal);
                    self.queue
                        .push_back(Ok(LlmStreamEvent::RefusalDelta { delta: refusal }));
                }
                for tc in choice.delta.tool_calls {
                    let slot = if tc.index >= self.tool_calls.len() {
                        self.tool_calls
                            .resize(tc.index + 1, (String::new(), String::new(), String::new()));
                        self.tool_calls.last_mut().unwrap()
                    } else {
                        &mut self.tool_calls[tc.index]
                    };
                    let is_start = tc.id.is_some() || (slot.0.is_empty() && slot.1.is_empty());
                    if is_start {
                        if let Some(open) = self.open_tool.take()
                            && open != tc.index
                        {
                            self.queue
                                .push_back(Ok(LlmStreamEvent::ToolCallEnd { index: open }));
                        }
                        if let Some(id) = tc.id {
                            slot.0 = id;
                        }
                        if let Some(function) = &tc.function
                            && let Some(name) = &function.name
                        {
                            slot.1 = name.clone();
                        }
                        self.open_tool = Some(tc.index);
                        self.queue.push_back(Ok(LlmStreamEvent::ToolCallStart {
                            index: tc.index,
                            id: (!slot.0.is_empty()).then(|| slot.0.clone()),
                            name: (!slot.1.is_empty()).then(|| slot.1.clone()),
                        }));
                    }
                    if let Some(function) = tc.function
                        && let Some(args) = function.arguments
                    {
                        slot.2.push_str(&args);
                        self.queue
                            .push_back(Ok(LlmStreamEvent::ToolCallArgumentsDelta {
                                index: tc.index,
                                delta: args,
                            }));
                    }
                }
                if let Some(finish_reason) = choice.finish_reason {
                    let reason = parse_oai_finish_reason(Some(&finish_reason));
                    self.finish = Some(reason.clone());
                    self.queue
                        .push_back(Ok(LlmStreamEvent::FinishReason { reason }));
                }
            }
        }

        fn finalize(&mut self) {
            if let Some(open) = self.open_tool.take() {
                self.queue
                    .push_back(Ok(LlmStreamEvent::ToolCallEnd { index: open }));
            }
            let content = if self.content.is_empty() {
                None
            } else {
                Some(self.content.clone())
            };
            let mut tool_calls = Vec::with_capacity(self.tool_calls.len());
            for (id, name, args) in &self.tool_calls {
                let arguments = match serde_json::from_str(args) {
                    Ok(arguments) => arguments,
                    Err(error) => {
                        self.done = true;
                        self.queue.push_back(Err(
                            self.decode_error(format!("invalid streamed tool arguments: {error}"))
                        ));
                        return;
                    }
                };
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments,
                });
            }
            if let Some(content_text) = content.as_ref()
                && matches!(
                    self.response_format,
                    ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. }
                )
                && let Err(e) = serde_json::from_str::<serde_json::Value>(content_text)
            {
                self.done = true;
                self.queue.push_back(Err(LlmError::Decode {
                    status: 200,
                    content_type: None,
                    content_encoding: None,
                    body_len: self.received_bytes,
                    body_hash: String::new(),
                    json_path: None,
                    cause: format!("structured response content is not valid JSON: {e}"),
                    likely_truncated: matches!(self.finish, Some(FinishReason::Length)),
                }));
                return;
            }
            self.done = true;
            self.queue.push_back(Ok(LlmStreamEvent::ResponseCompleted {
                response: LlmResponse {
                    content,
                    tool_calls,
                    usage: self.usage.clone().unwrap_or_default(),
                    finish_reason: self.finish.clone().unwrap_or(FinishReason::Stop),
                    refusal: if self.refusal.is_empty() {
                        None
                    } else {
                        Some(self.refusal.clone())
                    },
                    model: self.model.clone(),
                    reported_model: self.reported_model.clone(),
                },
            }));
        }

        fn decode_error(&self, cause: String) -> LlmError {
            LlmError::Decode {
                status: 200,
                content_type: self.metadata.content_type.clone(),
                content_encoding: self.metadata.content_encoding.clone(),
                body_len: self.received_bytes,
                body_hash: String::new(),
                json_path: None,
                cause,
                likely_truncated: false,
            }
        }
    }

    let state = OaiStreamState {
        stream,
        queue: VecDeque::new(),
        started: false,
        done: false,
        content: String::new(),
        refusal: String::new(),
        tool_calls: Vec::new(),
        open_tool: None,
        usage: None,
        finish: None,
        reported_model: None,
        received_bytes: 0,
        finish_event_seen: false,
        hasher: Sha256::new(),
        model,
        request_id,
        metadata,
        response_format,
    };

    Box::pin(stream::unfold(state, |mut st| async move {
        loop {
            // Backpressure: drain queued events before reading more from upstream.
            if st.queue.len() >= MAX_EVENT_QUEUE_DEPTH
                && let Some(item) = st.queue.pop_front()
            {
                if item.is_err() {
                    st.done = true;
                }
                return Some((item, st));
            }

            if let Some(item) = st.queue.pop_front() {
                if item.is_err() {
                    st.done = true;
                }
                return Some((item, st));
            }
            if st.done {
                return None;
            }
            match st.stream.next().await {
                None => {
                    if st.finish_event_seen {
                        return None;
                    }
                    st.done = true;
                    return Some((
                        Err(LlmError::IncompleteStream {
                            received_bytes: st.received_bytes,
                            finish_event_seen: false,
                        }),
                        st,
                    ));
                }
                Some(Err(error)) => {
                    st.done = true;
                    let mapped = match error {
                        eventsource_stream::EventStreamError::Transport(e) => e,
                        eventsource_stream::EventStreamError::Utf8(e) => {
                            st.decode_error(format!("stream is not valid UTF-8: {e}"))
                        }
                        eventsource_stream::EventStreamError::Parser(e) => {
                            st.decode_error(format!("invalid event stream: {e}"))
                        }
                    };
                    return Some((Err(mapped), st));
                }
                Some(Ok(event)) => {
                    st.received_bytes += event.data.len();
                    st.hasher.update(event.data.as_bytes());
                    if !st.started {
                        st.start();
                    }
                    let data = event.data.trim();
                    if data == "[DONE]" {
                        st.finish_event_seen = true;
                        st.finalize();
                        continue;
                    }
                    if data.is_empty() {
                        continue;
                    }
                    if event.event.as_str() == "error" {
                        st.done = true;
                        return Some((
                            Err(LlmError::Provider(format!("provider error event: {data}"))),
                            st,
                        ));
                    }
                    let mut de = serde_json::Deserializer::from_str(data);
                    match serde_path_to_error::deserialize::<_, OaiStreamChunk>(&mut de) {
                        Ok(chunk) => st.apply_chunk(chunk),
                        Err(e) => {
                            st.done = true;
                            return Some((
                                Err(st.decode_error(format!(
                                    "failed to parse SSE data: {}",
                                    e.inner()
                                ))),
                                st,
                            ));
                        }
                    }
                }
            }
        }
    }))
}

fn anthropic_event_stream<S>(
    stream: S,
    model: String,
    request_id: Option<String>,
    metadata: ResponseBodyMetadata,
) -> LlmEventStream
where
    S: Stream<
            Item = Result<
                eventsource_stream::Event,
                eventsource_stream::EventStreamError<LlmError>,
            >,
        > + Send
        + Unpin
        + 'static,
{
    struct State<S> {
        stream: S,
        queue: VecDeque<Result<LlmStreamEvent, LlmError>>,
        model: String,
        request_id: Option<String>,
        metadata: ResponseBodyMetadata,
        reported_model: Option<String>,
        content: String,
        tools: BTreeMap<usize, (String, String, String)>,
        usage: TokenUsage,
        finish: Option<FinishReason>,
        received_bytes: usize,
        started: bool,
        stopped: bool,
        done: bool,
    }

    impl<S> State<S> {
        fn decode_error(&self, cause: String) -> LlmError {
            LlmError::Decode {
                status: 200,
                content_type: self.metadata.content_type.clone(),
                content_encoding: self.metadata.content_encoding.clone(),
                body_len: self.received_bytes,
                body_hash: String::new(),
                json_path: None,
                cause,
                likely_truncated: false,
            }
        }

        fn start(&mut self) {
            if !self.started {
                self.started = true;
                self.queue.push_back(Ok(LlmStreamEvent::ResponseStarted {
                    request_id: self.request_id.clone(),
                    reported_model: self.reported_model.clone(),
                }));
            }
        }

        fn apply(&mut self, value: &serde_json::Value) -> Result<(), LlmError> {
            let kind = value
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            match kind {
                "message_start" => {
                    let message = &value["message"];
                    self.reported_model = message["model"].as_str().map(str::to_owned);
                    if let Some(input) = message["usage"]["input_tokens"].as_u64() {
                        self.usage.prompt_tokens = u32::try_from(input).unwrap_or(u32::MAX);
                    }
                    if let Some(output) = message["usage"]["output_tokens"].as_u64() {
                        self.usage.completion_tokens = u32::try_from(output).unwrap_or(u32::MAX);
                    }
                    self.usage.total_tokens = self
                        .usage
                        .prompt_tokens
                        .saturating_add(self.usage.completion_tokens);
                    self.start();
                }
                "content_block_start" => {
                    self.start();
                    let index = value["index"].as_u64().unwrap_or(0) as usize;
                    let block = &value["content_block"];
                    if block["type"].as_str() == Some("tool_use") {
                        let id = block["id"].as_str().unwrap_or_default().to_owned();
                        let name = block["name"].as_str().unwrap_or_default().to_owned();
                        self.tools
                            .insert(index, (id.clone(), name.clone(), String::new()));
                        self.queue.push_back(Ok(LlmStreamEvent::ToolCallStart {
                            index,
                            id: (!id.is_empty()).then_some(id),
                            name: (!name.is_empty()).then_some(name),
                        }));
                    }
                }
                "content_block_delta" => {
                    self.start();
                    let index = value["index"].as_u64().unwrap_or(0) as usize;
                    let delta = &value["delta"];
                    match delta["type"].as_str() {
                        Some("text_delta") => {
                            if let Some(text) = delta["text"].as_str() {
                                self.content.push_str(text);
                                self.queue.push_back(Ok(LlmStreamEvent::TextDelta {
                                    delta: text.to_owned(),
                                }));
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(thinking) = delta["thinking"].as_str() {
                                self.queue.push_back(Ok(LlmStreamEvent::ReasoningDelta {
                                    delta: thinking.to_owned(),
                                }));
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(fragment) = delta["partial_json"].as_str() {
                                let Some(tool) = self.tools.get_mut(&index) else {
                                    return Err(self.decode_error(format!(
                                        "tool argument delta for unknown content block {index}"
                                    )));
                                };
                                tool.2.push_str(fragment);
                                self.queue
                                    .push_back(Ok(LlmStreamEvent::ToolCallArgumentsDelta {
                                        index,
                                        delta: fragment.to_owned(),
                                    }));
                            }
                        }
                        _ => {}
                    }
                }
                "content_block_stop" => {
                    let index = value["index"].as_u64().unwrap_or(0) as usize;
                    if self.tools.contains_key(&index) {
                        self.queue
                            .push_back(Ok(LlmStreamEvent::ToolCallEnd { index }));
                    }
                }
                "message_delta" => {
                    if let Some(output) = value["usage"]["output_tokens"].as_u64() {
                        self.usage.completion_tokens = u32::try_from(output).unwrap_or(u32::MAX);
                        self.usage.total_tokens = self
                            .usage
                            .prompt_tokens
                            .saturating_add(self.usage.completion_tokens);
                        self.queue.push_back(Ok(LlmStreamEvent::UsageUpdate {
                            usage: self.usage.clone(),
                        }));
                    }
                    if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                        let reason = parse_anthropic_finish_reason(reason);
                        self.finish = Some(reason.clone());
                        self.queue
                            .push_back(Ok(LlmStreamEvent::FinishReason { reason }));
                    }
                }
                "message_stop" => {
                    self.stopped = true;
                    let mut tool_calls = Vec::with_capacity(self.tools.len());
                    for (id, name, arguments) in self.tools.values() {
                        let arguments = serde_json::from_str(arguments).map_err(|error| {
                            self.decode_error(format!("invalid streamed tool arguments: {error}"))
                        })?;
                        tool_calls.push(ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            arguments,
                        });
                    }
                    self.queue.push_back(Ok(LlmStreamEvent::ResponseCompleted {
                        response: LlmResponse {
                            content: (!self.content.is_empty()).then(|| self.content.clone()),
                            tool_calls,
                            usage: self.usage.clone(),
                            finish_reason: self.finish.clone().unwrap_or(FinishReason::Stop),
                            refusal: None,
                            model: self.model.clone(),
                            reported_model: self.reported_model.clone(),
                        },
                    }));
                    self.done = true;
                }
                "error" => return Err(LlmError::Provider("anthropic stream error event".into())),
                "ping" => {}
                _ => {}
            }
            Ok(())
        }
    }

    let state = State {
        stream,
        queue: VecDeque::new(),
        model,
        request_id,
        metadata,
        reported_model: None,
        content: String::new(),
        tools: BTreeMap::new(),
        usage: TokenUsage::default(),
        finish: None,
        received_bytes: 0,
        started: false,
        stopped: false,
        done: false,
    };

    Box::pin(stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.queue.pop_front() {
                return Some((event, state));
            }
            if state.done {
                return None;
            }
            match state.stream.next().await {
                None => {
                    state.done = true;
                    return Some((
                        Err(LlmError::IncompleteStream {
                            received_bytes: state.received_bytes,
                            finish_event_seen: state.stopped,
                        }),
                        state,
                    ));
                }
                Some(Err(error)) => {
                    state.done = true;
                    let error = match error {
                        eventsource_stream::EventStreamError::Transport(error) => error,
                        eventsource_stream::EventStreamError::Utf8(error) => {
                            state.decode_error(format!("stream is not valid UTF-8: {error}"))
                        }
                        eventsource_stream::EventStreamError::Parser(error) => {
                            state.decode_error(format!("invalid event stream: {error}"))
                        }
                    };
                    return Some((Err(error), state));
                }
                Some(Ok(event)) => {
                    state.received_bytes += event.data.len();
                    if event.data.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<serde_json::Value>(&event.data) {
                        Ok(value) => {
                            if let Err(error) = state.apply(&value) {
                                state.done = true;
                                return Some((Err(error), state));
                            }
                        }
                        Err(error) => {
                            state.done = true;
                            let error = state.decode_error(format!(
                                "failed to parse Anthropic SSE data: {error}"
                            ));
                            return Some((Err(error), state));
                        }
                    }
                }
            }
        }
    }))
}

fn parse_anthropic_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" | "pause_turn" => FinishReason::Stop,
        "tool_use" => FinishReason::ToolCalls,
        "max_tokens" => FinishReason::Length,
        "refusal" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

#[allow(dead_code)]
async fn collect_completed_response(stream: LlmEventStream) -> Result<LlmResponse, LlmError> {
    let mut stream = stream;
    while let Some(item) = stream.next().await {
        if let LlmStreamEvent::ResponseCompleted { response } = item? {
            return Ok(response);
        }
    }
    Err(LlmError::IncompleteStream {
        received_bytes: 0,
        finish_event_seen: false,
    })
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

    type TestResponse<'a> = (&'a str, &'a [(&'a str, &'a str)], &'a [u8]);
    type OwnedTestResponse = (String, Vec<(String, String)>, Vec<u8>);

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
            stream: false,
            stream_options: None,
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
                content_type,
                body_len,
                body_hash,
                likely_truncated,
                ..
            } => {
                assert_eq!(status, 200);
                assert_eq!(content_type.as_deref(), Some("application/json"));
                assert_eq!(body_len, body.len());
                assert_eq!(body_hash, sha256_hex(body));
                assert!(likely_truncated);
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
                body_len,
                content_encoding: _,
                likely_truncated,
                ..
            } => {
                assert!(body_len < 100);
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
                body_len,
                likely_truncated,
                ..
            } => {
                assert_eq!(body_len, 9);
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
                body_len,
                content_encoding: _,
                cause,
                likely_truncated,
                ..
            } => {
                assert_eq!(body_len, wire_len);
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
            .max_retries(0)
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
                request_id,
                body_metadata: _,
            } => {
                assert_eq!(delay, Duration::from_secs(12));
                assert_eq!(request_id.as_deref(), Some("corr-1"));
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
                content_encoding,
                body_len,
                body_hash,
                likely_truncated,
                ..
            } => {
                assert_eq!(content_encoding.as_deref(), Some("gzip"));
                assert_eq!(body_len, body.len());
                assert_eq!(body_hash, sha256_hex(body));
                assert!(!likely_truncated);
            }
            other => panic!("expected decode error, got {other:?}"),
        }
    }

    // ── Retry tests ────────────────────────────────────────────────────────────

    /// Server that responds to N sequential connections with the given
    /// (status, headers, body) triples, then panics.
    async fn serve_multi(responses: &[TestResponse<'_>]) -> (String, JoinHandle<Vec<usize>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let responses: Vec<OwnedTestResponse> = responses
            .iter()
            .map(|(s, h, b)| {
                (
                    s.to_string(),
                    h.iter()
                        .map(|(n, v)| (n.to_string(), v.to_string()))
                        .collect(),
                    b.to_vec(),
                )
            })
            .collect();

        let server = tokio::spawn(async move {
            let mut request_lengths = Vec::new();
            for (status, headers, body) in &responses {
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
                request_lengths.push(request.len());

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
                socket.write_all(&response).await.unwrap();
            }
            request_lengths
        });

        (format!("http://{address}"), server)
    }

    #[tokio::test]
    async fn retry_on_500_then_succeeds() {
        let success_body = br#"{
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }"#;
        let error_body = br#"{"error":{"message":"internal error"}}"#;
        let (base_url, server) = serve_multi(&[
            ("500 Internal Server Error", &[], error_body),
            ("200 OK", &[], success_body),
        ])
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .max_retries(3)
            .build()
            .unwrap();

        let response = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap();
        let request_counts = server.await.unwrap();

        assert_eq!(response.content.as_deref(), Some("ok"));
        // Server handled 2 connections: 1 failed + 1 succeeded
        assert_eq!(request_counts.len(), 2);
    }

    #[tokio::test]
    async fn retry_on_429_then_succeeds() {
        let success_body = br#"{
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }"#;
        let error_body = br#"{"error":{"message":"rate limited"}}"#;
        let (base_url, server) = serve_multi(&[
            ("429 Too Many Requests", &[("retry-after", "0")], error_body),
            ("200 OK", &[], success_body),
        ])
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .max_retries(3)
            .build()
            .unwrap();

        let response = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap();
        let request_counts = server.await.unwrap();

        assert_eq!(response.content.as_deref(), Some("ok"));
        assert_eq!(request_counts.len(), 2);
    }

    #[tokio::test]
    async fn no_retry_on_400() {
        let error_body = br#"{"error":{"message":"bad request"}}"#;
        let (base_url, server) = serve_multi(&[("400 Bad Request", &[], error_body)]).await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .max_retries(3)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        let request_counts = server.await.unwrap();

        assert!(matches!(error, LlmError::Http { status: 400, .. }));
        assert_eq!(request_counts.len(), 1);
    }

    #[tokio::test]
    async fn retry_exhausted_returns_last_error() {
        let error_body = br#"{"error":{"message":"still broken"}}"#;
        let (base_url, server) = serve_multi(&[
            ("500 Internal Server Error", &[], error_body),
            ("500 Internal Server Error", &[], error_body),
            ("500 Internal Server Error", &[], error_body),
        ])
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .max_retries(2)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        let request_counts = server.await.unwrap();

        assert!(matches!(error, LlmError::Http { status: 500, .. }));
        // 1 initial + 2 retries = 3 connections
        assert_eq!(request_counts.len(), 3);
    }

    #[tokio::test]
    async fn retry_respects_max_retries_zero() {
        let error_body = br#"{"error":{"message":"fail"}}"#;
        let (base_url, server) =
            serve_multi(&[("500 Internal Server Error", &[], error_body)]).await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .max_retries(0)
            .build()
            .unwrap();

        let error = adapter
            .complete(openai_request("test-model"))
            .await
            .unwrap_err();
        let request_counts = server.await.unwrap();

        assert!(matches!(error, LlmError::Http { status: 500, .. }));
        assert_eq!(request_counts.len(), 1);
    }

    // ── LlmError::is_retryable / retry_after tests ─────────────────────────────

    #[test]
    fn transport_retryable_is_retryable() {
        let err = LlmError::Transport {
            stage: TransportStage::Connect,
            retryable: true,
            cause: "connection refused".into(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn transport_non_retryable_is_not_retryable() {
        let err = LlmError::Transport {
            stage: TransportStage::Body,
            retryable: false,
            cause: "body read error".into(),
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn http_429_is_retryable() {
        let err = LlmError::Http {
            status: 429,
            retry_after: Some(Duration::from_secs(5)),
            request_id: None,
            body_metadata: Box::new(ResponseBodyMetadata {
                content_type: None,
                content_encoding: None,
                content_length: None,
                request_id: None,
                body_len: 0,
                body_hash: String::new(),
                decoded_body_len: None,
                decoded_body_hash: None,
            }),
        };
        assert!(err.is_retryable());
        assert_eq!(err.retry_after(), Some(Duration::from_secs(5)));
    }

    #[test]
    fn http_500_is_retryable() {
        let err = LlmError::Http {
            status: 500,
            retry_after: None,
            request_id: None,
            body_metadata: Box::new(ResponseBodyMetadata {
                content_type: None,
                content_encoding: None,
                content_length: None,
                request_id: None,
                body_len: 0,
                body_hash: String::new(),
                decoded_body_len: None,
                decoded_body_hash: None,
            }),
        };
        assert!(err.is_retryable());
        assert_eq!(err.retry_after(), None);
    }

    #[test]
    fn http_400_is_not_retryable() {
        let err = LlmError::Http {
            status: 400,
            retry_after: None,
            request_id: None,
            body_metadata: Box::new(ResponseBodyMetadata {
                content_type: None,
                content_encoding: None,
                content_length: None,
                request_id: None,
                body_len: 0,
                body_hash: String::new(),
                decoded_body_len: None,
                decoded_body_hash: None,
            }),
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn decode_error_is_not_retryable() {
        let err = LlmError::Decode {
            status: 200,
            content_type: None,
            content_encoding: None,
            body_len: 0,
            body_hash: String::new(),
            json_path: None,
            cause: "invalid json".into(),
            likely_truncated: false,
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn timeout_error_is_not_retryable() {
        assert!(!LlmError::Timeout.is_retryable());
    }

    #[test]
    fn incomplete_stream_is_not_retryable() {
        let err = LlmError::IncompleteStream {
            received_bytes: 0,
            finish_event_seen: false,
        };
        assert!(!err.is_retryable());
    }

    // ── Backpressure tests ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn streaming_delivers_all_events_with_backpressure() {
        // Build an SSE response with many chunks.
        let mut sse_body = String::new();
        for i in 0..100 {
            let chunk = serde_json::json!({
                "choices": [{
                    "delta": {"content": format!("token-{i}")},
                    "finish_reason": null
                }]
            });
            sse_body.push_str(&format!("data: {}\n\n", chunk));
        }
        sse_body.push_str("data: [DONE]\n\n");

        let (base_url, server) = serve_once(
            "200 OK",
            &[("content-type", "text/event-stream")],
            sse_body.as_bytes(),
        )
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let mut stream = adapter
            .complete_stream(openai_request("test-model"))
            .await
            .unwrap();

        let mut text_deltas = Vec::new();
        let mut got_completed = false;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                LlmStreamEvent::TextDelta { delta } => text_deltas.push(delta),
                LlmStreamEvent::ResponseCompleted { .. } => {
                    got_completed = true;
                    break;
                }
                _ => {}
            }
        }

        server.await.unwrap();
        assert!(got_completed);
        assert_eq!(text_deltas.len(), 100);
        assert_eq!(text_deltas[0], "token-0");
        assert_eq!(text_deltas[99], "token-99");
    }

    #[tokio::test]
    async fn streaming_tool_call_events_are_correctly_assembled() {
        let sse_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\
             \"function\":{\"name\":\"search\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\
             \"function\":{\"arguments\":\"{\\\"q\\\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\
             \"function\":{\"arguments\":\":\\\"rust\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let (base_url, server) = serve_once(
            "200 OK",
            &[("content-type", "text/event-stream")],
            sse_body.as_bytes(),
        )
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let mut stream = adapter
            .complete_stream(openai_request("test-model"))
            .await
            .unwrap();

        let mut tool_starts = Vec::new();
        let mut tool_args = Vec::new();
        let mut tool_ends = Vec::new();
        let mut got_finish = false;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                LlmStreamEvent::ToolCallStart { index, id, name } => {
                    tool_starts.push((index, id, name));
                }
                LlmStreamEvent::ToolCallArgumentsDelta { index, delta } => {
                    tool_args.push((index, delta));
                }
                LlmStreamEvent::ToolCallEnd { index } => {
                    tool_ends.push(index);
                }
                LlmStreamEvent::FinishReason { .. } => {
                    got_finish = true;
                }
                LlmStreamEvent::ResponseCompleted { .. } => break,
                _ => {}
            }
        }

        server.await.unwrap();
        assert!(got_finish);
        assert_eq!(tool_starts.len(), 1);
        assert_eq!(tool_starts[0].0, 0);
        assert_eq!(tool_starts[0].1.as_deref(), Some("call_1"));
        assert_eq!(tool_starts[0].2.as_deref(), Some("search"));
        assert_eq!(tool_args.len(), 2);
        assert_eq!(tool_ends, vec![0]);
    }

    #[tokio::test]
    async fn streaming_gzip_compressed_sse_is_decoded() {
        let decoded_sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"compressed\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let body = encode_fixture(decoded_sse.as_bytes(), "gzip");
        let (base_url, server) = serve_once(
            "200 OK",
            &[
                ("content-type", "text/event-stream"),
                ("content-encoding", "gzip"),
            ],
            &body,
        )
        .await;
        let adapter = GenericLlmAdapter::builder()
            .openai_base_url(base_url)
            .build()
            .unwrap();

        let mut stream = adapter
            .complete_stream(openai_request("test-model"))
            .await
            .unwrap();

        let mut text = String::new();
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                LlmStreamEvent::TextDelta { delta } => text.push_str(&delta),
                LlmStreamEvent::ResponseCompleted { .. } => break,
                _ => {}
            }
        }

        server.await.unwrap();
        assert_eq!(text, "compressed");
    }

    #[tokio::test]
    async fn anthropic_stream_assembles_text_tools_usage_and_terminal_event() {
        let data = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-test\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"lookup\",\"input\":{}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"rust\\\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":4}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let source = stream::iter([Ok::<Bytes, LlmError>(Bytes::from_static(data.as_bytes()))]);
        let mut events = anthropic_event_stream(
            source.eventsource(),
            "claude-resolved".to_owned(),
            Some("req-1".to_owned()),
            ResponseBodyMetadata {
                content_type: Some("text/event-stream".to_owned()),
                content_encoding: None,
                content_length: None,
                request_id: Some("req-1".to_owned()),
                body_len: 0,
                body_hash: String::new(),
                decoded_body_len: None,
                decoded_body_hash: None,
            },
        );
        let mut completed = None;
        while let Some(event) = events.next().await {
            if let LlmStreamEvent::ResponseCompleted { response } = event.unwrap() {
                completed = Some(response);
            }
        }
        let response = completed.expect("message_stop must complete the response");
        assert_eq!(response.content.as_deref(), Some("hello"));
        assert_eq!(
            response.tool_calls[0].arguments,
            serde_json::json!({"q": "rust"})
        );
        assert_eq!(response.usage.total_tokens, 7);
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.reported_model.as_deref(), Some("claude-test"));
    }
}
