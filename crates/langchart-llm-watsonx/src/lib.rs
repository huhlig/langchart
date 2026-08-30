//! IBM watsonx.ai transport for Langchart.
//!
//! [`WatsonxAdapter`] implements Langchart's [`LlmAdapter`] contract using the
//! watsonx text-chat API. IBM Cloud API keys are exchanged for short-lived IAM
//! tokens and cached in memory until shortly before expiry.

#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic)]
// The public adapter trait returns the shared, diagnostic-rich LlmError by value.
#![allow(clippy::result_large_err)]

use async_compression::tokio::bufread::{BrotliDecoder, GzipDecoder, ZlibDecoder, ZstdDecoder};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt, TryStreamExt, stream};
use langchart_adapters::llm::{
    FinishReason, LlmAdapter, LlmError, LlmEventStream, LlmRequest, LlmResponse, LlmStreamEvent,
    Message, ResponseFormat, TokenUsage, TransportStage,
};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufRead, BufReader};
use tokio::sync::Mutex;
use tokio_util::io::{ReaderStream, StreamReader};
use url::Url;

const IAM_TOKEN_URL: &str = "https://iam.cloud.ibm.com/identity/token";
const IAM_GRANT_TYPE: &str = "urn:ibm:params:oauth:grant-type:apikey";
const TOKEN_REFRESH_MARGIN: Duration = Duration::from_secs(60);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_FIRST_BYTE_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_TOTAL_GENERATION_TIMEOUT: Duration = Duration::from_mins(15);

/// A watsonx deployment scope. Exactly one project or space ID is required.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WatsonxScope {
    Project(String),
    Space(String),
}

/// Connection settings for an IBM watsonx.ai service instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WatsonxConfig {
    pub service_url: String,
    pub api_version: String,
    pub scope: WatsonxScope,
}

/// Independent deadlines for connection establishment and long generations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WatsonxTimeouts {
    pub connect: Duration,
    pub first_byte: Duration,
    pub stream_idle: Duration,
    pub total_generation: Duration,
}

impl Default for WatsonxTimeouts {
    fn default() -> Self {
        Self {
            connect: DEFAULT_CONNECT_TIMEOUT,
            first_byte: DEFAULT_FIRST_BYTE_TIMEOUT,
            stream_idle: DEFAULT_STREAM_IDLE_TIMEOUT,
            total_generation: DEFAULT_TOTAL_GENERATION_TIMEOUT,
        }
    }
}

impl WatsonxConfig {
    pub fn new(
        service_url: impl Into<String>,
        api_version: impl Into<String>,
        scope: WatsonxScope,
    ) -> Self {
        Self {
            service_url: service_url.into(),
            api_version: api_version.into(),
            scope,
        }
    }
}

/// Authentication material. This type deliberately does not implement `Debug`.
#[derive(Clone)]
pub enum WatsonxCredentials {
    /// An IBM Cloud API key exchanged for a cached IAM bearer token.
    ApiKey(String),
    /// A caller-managed IAM bearer token.
    BearerToken(String),
}

/// Configuration failure detected before a provider call is attempted.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("invalid watsonx configuration: {0}")]
    InvalidConfig(String),
    #[error("failed to build watsonx HTTP client: {0}")]
    Client(String),
}

/// Langchart LLM adapter for the watsonx.ai text-chat API.
pub struct WatsonxAdapter {
    client: Client,
    config: WatsonxConfig,
    credentials: WatsonxCredentials,
    cached_token: Mutex<Option<CachedToken>>,
    timeouts: WatsonxTimeouts,
}

struct CachedToken {
    value: String,
    refresh_at: Instant,
}

impl WatsonxAdapter {
    /// Creates an adapter with phase-aware defaults suitable for long generations.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the endpoint, API version, scope, or
    /// credentials are malformed, or when the HTTP client cannot be built.
    pub fn new(config: WatsonxConfig, credentials: WatsonxCredentials) -> Result<Self, BuildError> {
        Self::new_with_timeouts(config, credentials, WatsonxTimeouts::default())
    }

    /// Creates an adapter with caller-provided phase deadlines.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] for invalid configuration or client construction.
    pub fn new_with_timeouts(
        config: WatsonxConfig,
        credentials: WatsonxCredentials,
        timeouts: WatsonxTimeouts,
    ) -> Result<Self, BuildError> {
        validate_config(&config, &credentials)?;
        let client = Client::builder()
            .connect_timeout(timeouts.connect)
            .build()
            .map_err(|error| BuildError::Client(error.to_string()))?;
        Ok(Self {
            client,
            config,
            credentials,
            cached_token: Mutex::new(None),
            timeouts,
        })
    }

    async fn bearer_token(&self) -> Result<String, LlmError> {
        match &self.credentials {
            WatsonxCredentials::BearerToken(token) => Ok(token.clone()),
            WatsonxCredentials::ApiKey(api_key) => {
                let mut cached = self.cached_token.lock().await;
                if let Some(token) = cached.as_ref()
                    && Instant::now() < token.refresh_at
                {
                    return Ok(token.value.clone());
                }

                let response = self
                    .client
                    .post(IAM_TOKEN_URL)
                    .header("Accept", "application/json")
                    .form(&[("grant_type", IAM_GRANT_TYPE), ("apikey", api_key)])
                    .send()
                    .await
                    .map_err(|error| map_reqwest_error(&error))?;
                let response = checked_response(response, "watsonx IAM").await?;
                let token: IamTokenResponse = response.json().await.map_err(|error| {
                    LlmError::Provider(format!("invalid watsonx IAM response: {error}"))
                })?;
                validate_wire_secret("watsonx IAM access token", &token.access_token)?;

                let lifetime = Duration::from_secs(token.expires_in);
                let margin = TOKEN_REFRESH_MARGIN.min(lifetime / 10);
                let refresh_at = Instant::now() + lifetime.saturating_sub(margin);
                *cached = Some(CachedToken {
                    value: token.access_token.clone(),
                    refresh_at,
                });
                Ok(token.access_token)
            }
        }
    }

    fn chat_stream_url(&self) -> String {
        format!(
            "{}/ml/v1/text/chat_stream?version={}",
            self.config.service_url, self.config.api_version
        )
    }
}

#[async_trait]
impl LlmAdapter for WatsonxAdapter {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        collect_completed_response(self.complete_stream(request).await?).await
    }

    async fn complete_stream(&self, request: LlmRequest) -> Result<LlmEventStream, LlmError> {
        if !request.tools.is_empty() {
            return Err(LlmError::Provider(
                "watsonx adapter does not support tool calls".to_owned(),
            ));
        }
        let model =
            request.model_policy.model.clone().ok_or_else(|| {
                LlmError::Provider("watsonx requires an explicit model".to_owned())
            })?;
        let body = WatsonxChatRequest::from_request(&model, &self.config, request)?;
        let total_deadline = tokio::time::Instant::now() + self.timeouts.total_generation;
        let headers_deadline = std::cmp::min(
            total_deadline,
            tokio::time::Instant::now() + self.timeouts.first_byte,
        );
        let response = tokio::time::timeout_at(
            headers_deadline,
            self.client
                .post(self.chat_stream_url())
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .header(reqwest::header::ACCEPT_ENCODING, "gzip, br, deflate, zstd")
                .bearer_auth(self.bearer_token().await?)
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| LlmError::Transport {
            stage: TransportStage::Headers,
            retryable: true,
            cause: "watsonx response headers deadline exceeded".to_owned(),
        })?
        .map_err(|error| map_reqwest_error(&error))?;
        let response = checked_response(response, "watsonx chat").await?;
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let stream = timed_body_stream(
            decoded_watsonx_body(response)?,
            self.timeouts.first_byte,
            self.timeouts.stream_idle,
            total_deadline,
        );
        Ok(watsonx_event_stream(
            stream.eventsource(),
            model,
            request_id,
        ))
    }
}

#[derive(Serialize)]
struct WatsonxChatRequest {
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    space_id: Option<String>,
    messages: Vec<WatsonxMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<WatsonxResponseFormat>,
}

impl WatsonxChatRequest {
    fn from_request(
        model: &str,
        config: &WatsonxConfig,
        request: LlmRequest,
    ) -> Result<Self, LlmError> {
        let (project_id, space_id) = match &config.scope {
            WatsonxScope::Project(value) => (Some(value.clone()), None),
            WatsonxScope::Space(value) => (None, Some(value.clone())),
        };
        let messages = request
            .messages
            .into_iter()
            .map(WatsonxMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let response_format_kind = request.response_format.kind();
        let response_format = match request.response_format {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(WatsonxResponseFormat {
                kind: "json_object",
            }),
            ResponseFormat::JsonSchema { .. } => {
                return Err(LlmError::UnsupportedResponseFormat {
                    adapter: "watsonx".into(),
                    requested: response_format_kind,
                });
            }
        };

        Ok(Self {
            model_id: model.to_owned(),
            project_id,
            space_id,
            messages,
            temperature: request.model_policy.temperature,
            max_tokens: request.model_policy.max_tokens,
            response_format,
        })
    }
}

#[derive(Serialize)]
struct WatsonxMessage {
    role: &'static str,
    content: String,
}

impl TryFrom<Message> for WatsonxMessage {
    type Error = LlmError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        match message {
            Message::System { content } => Ok(Self {
                role: "system",
                content,
            }),
            Message::User { content } => Ok(Self {
                role: "user",
                content,
            }),
            Message::Assistant { content } => Ok(Self {
                role: "assistant",
                content,
            }),
            Message::Tool { .. } => Err(LlmError::Provider(
                "watsonx adapter does not support tool messages".to_owned(),
            )),
        }
    }
}

#[derive(Serialize)]
struct WatsonxResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Deserialize)]
struct IamTokenResponse {
    access_token: String,
    expires_in: u64,
}

#[cfg(test)]
#[derive(Deserialize)]
struct WatsonxChatResponse {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    model_version: Option<String>,
    choices: Vec<WatsonxChoice>,
    #[serde(default)]
    usage: WatsonxUsage,
}

#[cfg(test)]
impl WatsonxChatResponse {
    fn into_llm_response(self) -> Result<LlmResponse, LlmError> {
        let choice = self
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::Provider("watsonx returned no choices".to_owned()))?;
        let prompt_tokens = self.usage.prompt();
        let completion_tokens = self.usage.completion();
        let resolved_model = self.model_id.or(self.model).unwrap_or_default();
        let reported_model = self.model_version.unwrap_or_else(|| resolved_model.clone());
        Ok(LlmResponse {
            content: Some(choice.message.content),
            tool_calls: Vec::new(),
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: self
                    .usage
                    .total()
                    .unwrap_or(prompt_tokens.saturating_add(completion_tokens)),
            },
            finish_reason: map_finish_reason(choice.finish_reason.as_deref()),
            refusal: None,
            model: resolved_model,
            reported_model: Some(reported_model),
        })
    }
}

#[cfg(test)]
#[derive(Deserialize)]
struct WatsonxChoice {
    message: WatsonxResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WatsonxResponseMessage {
    content: String,
}

#[allow(clippy::struct_field_names)]
#[derive(Default, Deserialize)]
struct WatsonxUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    generated_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

impl WatsonxUsage {
    fn prompt(&self) -> u32 {
        self.prompt_tokens.or(self.input_tokens).unwrap_or(0)
    }

    fn completion(&self) -> u32 {
        self.completion_tokens.or(self.generated_tokens).unwrap_or(0)
    }

    fn total(&self) -> Option<u32> {
        self.total_tokens
    }
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop" | "eos_token") | None => FinishReason::Stop,
        Some("length" | "max_tokens") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some(other) => FinishReason::Other(other.to_owned()),
    }
}

#[derive(Deserialize)]
struct WatsonxStreamChunk {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<WatsonxStreamChoice>,
    usage: Option<WatsonxUsage>,
}

impl WatsonxStreamChunk {
    fn model(&self) -> Option<&str> {
        self.model_id.as_deref().or(self.model.as_deref())
    }
}

#[derive(Deserialize)]
struct WatsonxStreamChoice {
    #[serde(default)]
    delta: WatsonxStreamDelta,
    message: Option<WatsonxResponseMessage>,
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct WatsonxStreamDelta {
    content: Option<String>,
}

type WatsonxByteStream = Pin<Box<dyn Stream<Item = Result<bytes::Bytes, LlmError>> + Send>>;

fn decoded_watsonx_body(response: reqwest::Response) -> Result<WatsonxByteStream, LlmError> {
    let encoding = response
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let raw = response.bytes_stream().map_err(std::io::Error::other);
    let reader = StreamReader::new(raw);
    let mut reader: Pin<Box<dyn AsyncBufRead + Send + Unpin>> = Box::pin(BufReader::new(reader));

    if let Some(ref encoding) = encoding {
        for item in encoding
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>()
            .iter()
            .rev()
        {
            if item.is_empty() || item.eq_ignore_ascii_case("identity") {
                continue;
            }
            let previous =
                std::mem::replace(&mut reader, Box::pin(BufReader::new(tokio::io::empty())));
            reader = if item.eq_ignore_ascii_case("gzip") || item.eq_ignore_ascii_case("x-gzip") {
                Box::pin(BufReader::new(GzipDecoder::new(BufReader::new(previous))))
            } else if item.eq_ignore_ascii_case("br") {
                Box::pin(BufReader::new(BrotliDecoder::new(BufReader::new(previous))))
            } else if item.eq_ignore_ascii_case("deflate") {
                Box::pin(BufReader::new(ZlibDecoder::new(BufReader::new(previous))))
            } else if item.eq_ignore_ascii_case("zstd") {
                Box::pin(BufReader::new(ZstdDecoder::new(BufReader::new(previous))))
            } else {
                return Err(LlmError::Provider(format!(
                    "unsupported watsonx content encoding `{item}`"
                )));
            };
        }
    }

    Ok(Box::pin(ReaderStream::new(reader).map_err(|error| {
        LlmError::Transport {
            stage: TransportStage::Body,
            retryable: true,
            cause: error.to_string(),
        }
    })))
}

fn timed_body_stream<S>(
    stream: S,
    first_byte_timeout: Duration,
    idle_timeout: Duration,
    total_deadline: tokio::time::Instant,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<bytes::Bytes, LlmError>> + Send>>
where
    S: Stream<Item = Result<bytes::Bytes, LlmError>> + Send + Unpin + 'static,
{
    struct State<S> {
        stream: S,
        seen_first: bool,
        done: bool,
    }
    Box::pin(stream::unfold(
        State {
            stream,
            seen_first: false,
            done: false,
        },
        move |mut state| async move {
            if state.done {
                return None;
            }
            let phase_deadline = tokio::time::Instant::now()
                + if state.seen_first {
                    idle_timeout
                } else {
                    first_byte_timeout
                };
            let deadline = std::cmp::min(total_deadline, phase_deadline);
            match tokio::time::timeout_at(deadline, state.stream.next()).await {
                Ok(Some(result)) => {
                    state.seen_first = true;
                    if result.is_err() {
                        state.done = true;
                    }
                    Some((result, state))
                }
                Ok(None) => None,
                Err(_) => {
                    state.done = true;
                    let cause = if tokio::time::Instant::now() >= total_deadline {
                        "watsonx total generation deadline exceeded"
                    } else if state.seen_first {
                        "watsonx stream idle deadline exceeded"
                    } else {
                        "watsonx first response byte deadline exceeded"
                    };
                    Some((
                        Err(LlmError::Transport {
                            stage: TransportStage::Body,
                            retryable: true,
                            cause: cause.to_owned(),
                        }),
                        state,
                    ))
                }
            }
        },
    ))
}

#[allow(clippy::too_many_lines)]
fn watsonx_event_stream<S>(stream: S, model: String, request_id: Option<String>) -> LlmEventStream
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
        queue: std::collections::VecDeque<Result<LlmStreamEvent, LlmError>>,
        model: String,
        request_id: Option<String>,
        reported_model: Option<String>,
        content: String,
        usage: TokenUsage,
        finish: Option<FinishReason>,
        received_bytes: usize,
        started: bool,
        terminal: bool,
        done: bool,
    }

    impl<S> State<S> {
        fn start(&mut self) {
            if !self.started {
                self.started = true;
                self.queue.push_back(Ok(LlmStreamEvent::ResponseStarted {
                    request_id: self.request_id.clone(),
                    reported_model: self.reported_model.clone(),
                }));
            }
        }

        fn finalize(&mut self) {
            self.terminal = true;
            self.done = true;
            self.queue.push_back(Ok(LlmStreamEvent::ResponseCompleted {
                response: LlmResponse {
                    content: (!self.content.is_empty()).then(|| self.content.clone()),
                    tool_calls: Vec::new(),
                    usage: self.usage.clone(),
                    finish_reason: self.finish.clone().unwrap_or(FinishReason::Stop),
                    refusal: None,
                    model: self.model.clone(),
                    reported_model: self.reported_model.clone(),
                },
            }));
        }

        fn apply(&mut self, chunk: WatsonxStreamChunk) {
            if let Some(reported_model) = chunk.model() {
                self.reported_model = Some(reported_model.to_owned());
            }
            self.start();
            if let Some(usage) = chunk.usage {
                self.usage = TokenUsage {
                    prompt_tokens: usage.prompt(),
                    completion_tokens: usage.completion(),
                    total_tokens: usage
                        .total()
                        .unwrap_or_else(|| usage.prompt().saturating_add(usage.completion())),
                };
                self.queue.push_back(Ok(LlmStreamEvent::UsageUpdate {
                    usage: self.usage.clone(),
                }));
            }
            for choice in chunk.choices {
                let text = choice
                    .delta
                    .content
                    .or_else(|| choice.message.map(|message| message.content));
                if let Some(text) = text
                    && !text.is_empty()
                {
                    self.content.push_str(&text);
                    self.queue
                        .push_back(Ok(LlmStreamEvent::TextDelta { delta: text }));
                }
                if let Some(reason) = choice.finish_reason {
                    let reason = map_finish_reason(Some(&reason));
                    self.finish = Some(reason.clone());
                    self.queue
                        .push_back(Ok(LlmStreamEvent::FinishReason { reason }));
                }
            }
        }
    }

    let state = State {
        stream,
        queue: std::collections::VecDeque::new(),
        model,
        request_id,
        reported_model: None,
        content: String::new(),
        usage: TokenUsage::default(),
        finish: None,
        received_bytes: 0,
        started: false,
        terminal: false,
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
                    if state.started {
                        state.finalize();
                        if let Some(event) = state.queue.pop_front() {
                            return Some((event, state));
                        }
                        return None;
                    }
                    return Some((
                        Err(LlmError::IncompleteStream {
                            received_bytes: state.received_bytes,
                            finish_event_seen: state.terminal,
                        }),
                        state,
                    ));
                }
                Some(Err(error)) => {
                    state.done = true;
                    let error = match error {
                        eventsource_stream::EventStreamError::Transport(error) => error,
                        eventsource_stream::EventStreamError::Utf8(error) => LlmError::Provider(
                            format!("watsonx stream is not valid UTF-8: {error}"),
                        ),
                        eventsource_stream::EventStreamError::Parser(error) => {
                            LlmError::Provider(format!("invalid watsonx event stream: {error}"))
                        }
                    };
                    return Some((Err(error), state));
                }
                Some(Ok(event)) => {
                    state.received_bytes += event.data.len();
                    let data = event.data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    if data == "[DONE]" || event.event == "message_stop" {
                        state.finalize();
                        continue;
                    }
                    match serde_json::from_str::<WatsonxStreamChunk>(data) {
                        Ok(chunk) => state.apply(chunk),
                        Err(error) => {
                            state.done = true;
                            return Some((
                                Err(LlmError::Provider(format!(
                                    "invalid watsonx stream item: {error}"
                                ))),
                                state,
                            ));
                        }
                    }
                }
            }
        }
    }))
}

async fn collect_completed_response(mut stream: LlmEventStream) -> Result<LlmResponse, LlmError> {
    while let Some(event) = stream.next().await {
        if let LlmStreamEvent::ResponseCompleted { response } = event? {
            return Ok(response);
        }
    }
    Err(LlmError::IncompleteStream {
        received_bytes: 0,
        finish_event_seen: false,
    })
}

async fn checked_response(
    response: reqwest::Response,
    operation: &str,
) -> Result<reqwest::Response, LlmError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    let detail: String = body.chars().take(512).collect();
    match status {
        StatusCode::TOO_MANY_REQUESTS => Err(LlmError::RateLimited(detail)),
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => Err(LlmError::Timeout),
        StatusCode::NOT_FOUND => Err(LlmError::ModelNotFound {
            model: "configured watsonx model".to_owned(),
        }),
        _ => Err(LlmError::Provider(format!(
            "{operation} failed with HTTP {status}: {detail}"
        ))),
    }
}

fn map_reqwest_error(error: &reqwest::Error) -> LlmError {
    if error.is_timeout() {
        LlmError::Timeout
    } else {
        LlmError::Provider(error.to_string())
    }
}

fn validate_config(
    config: &WatsonxConfig,
    credentials: &WatsonxCredentials,
) -> Result<(), BuildError> {
    validate_endpoint(&config.service_url)?;
    validate_api_version(&config.api_version)?;
    validate_scope(&config.scope)?;
    match credentials {
        WatsonxCredentials::ApiKey(value) => validate_secret("watsonx API key", value),
        WatsonxCredentials::BearerToken(value) => validate_secret("watsonx bearer token", value),
    }
}

fn validate_endpoint(value: &str) -> Result<(), BuildError> {
    if value.trim() != value || value.ends_with('/') {
        return Err(BuildError::InvalidConfig(
            "service URL must be normalized without a trailing slash".to_owned(),
        ));
    }
    let endpoint = Url::parse(value)
        .map_err(|error| BuildError::InvalidConfig(format!("invalid service URL: {error}")))?;
    if endpoint.scheme() != "https" {
        return Err(BuildError::InvalidConfig(
            "service URL must use HTTPS".to_owned(),
        ));
    }
    if !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(BuildError::InvalidConfig(
            "service URL cannot contain credentials, query, or fragment".to_owned(),
        ));
    }
    Ok(())
}

fn validate_api_version(value: &str) -> Result<(), BuildError> {
    let bytes = value.as_bytes();
    if bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
    {
        Ok(())
    } else {
        Err(BuildError::InvalidConfig(
            "API version must use YYYY-MM-DD".to_owned(),
        ))
    }
}

fn validate_scope(scope: &WatsonxScope) -> Result<(), BuildError> {
    let (name, value) = match scope {
        WatsonxScope::Project(value) => ("watsonx project ID", value),
        WatsonxScope::Space(value) => ("watsonx space ID", value),
    };
    validate_secret(name, value)
}

fn validate_secret(name: &str, value: &str) -> Result<(), BuildError> {
    if value.trim().is_empty() || value.trim() != value {
        Err(BuildError::InvalidConfig(format!(
            "{name} must be non-empty and normalized"
        )))
    } else {
        Ok(())
    }
}

fn validate_wire_secret(name: &str, value: &str) -> Result<(), LlmError> {
    if value.trim().is_empty() || value.trim() != value {
        Err(LlmError::Provider(format!("{name} was empty or malformed")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use langchart_model::policy::ModelPolicy;
    use serde_json::json;
    use std::io::Write;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn config(scope: WatsonxScope) -> WatsonxConfig {
        WatsonxConfig::new("https://us-south.ml.cloud.ibm.com", "2024-05-31", scope)
    }

    fn request() -> LlmRequest {
        LlmRequest {
            model_policy: ModelPolicy {
                profile: None,
                model: Some("ibm/granite-4-h-small".to_owned()),
                temperature: Some(0.0),
                max_tokens: Some(512),
            },
            messages: vec![Message::User {
                content: "Return JSON".to_owned(),
            }],
            tools: Vec::new(),
            response_format: ResponseFormat::Text,
        }
    }

    #[test]
    fn configuration_rejects_unsafe_endpoints_versions_and_secrets() {
        assert!(
            WatsonxAdapter::new(
                config(WatsonxScope::Project("project-1".to_owned())),
                WatsonxCredentials::ApiKey("secret".to_owned()),
            )
            .is_ok()
        );
        let mut insecure = config(WatsonxScope::Project("project-1".to_owned()));
        insecure.service_url = "http://us-south.ml.cloud.ibm.com".to_owned();
        assert!(
            WatsonxAdapter::new(insecure, WatsonxCredentials::ApiKey("secret".to_owned())).is_err()
        );
        let mut invalid_version = config(WatsonxScope::Space("space-1".to_owned()));
        invalid_version.api_version = "latest".to_owned();
        assert!(
            WatsonxAdapter::new(
                invalid_version,
                WatsonxCredentials::BearerToken("token".to_owned())
            )
            .is_err()
        );
    }

    #[test]
    fn custom_timeouts_are_preserved() {
        let timeouts = WatsonxTimeouts {
            connect: Duration::from_secs(2),
            first_byte: Duration::from_mins(10),
            stream_idle: Duration::from_mins(3),
            total_generation: Duration::from_mins(45),
        };
        let adapter = WatsonxAdapter::new_with_timeouts(
            config(WatsonxScope::Project("project-1".to_owned())),
            WatsonxCredentials::BearerToken("token".to_owned()),
            timeouts,
        )
        .unwrap();
        assert_eq!(adapter.timeouts, timeouts);
    }

    #[test]
    fn text_request_serialization_omits_response_format() {
        let body = WatsonxChatRequest::from_request(
            "ibm/granite-4-h-small",
            &config(WatsonxScope::Project("project-1".to_owned())),
            request(),
        )
        .unwrap();
        let value = serde_json::to_value(body).unwrap();
        assert_eq!(value["project_id"], "project-1");
        assert!(value.get("space_id").is_none());
        assert!(value.get("response_format").is_none());
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["max_tokens"], 512);
    }

    #[test]
    fn json_object_request_serialization_includes_response_format() {
        let mut request = request();
        request.response_format = ResponseFormat::JsonObject;
        let body = WatsonxChatRequest::from_request(
            "ibm/granite-4-h-small",
            &config(WatsonxScope::Project("project-1".to_owned())),
            request,
        )
        .unwrap();

        assert_eq!(
            serde_json::to_value(body).unwrap()["response_format"],
            json!({"type": "json_object"})
        );
    }

    #[test]
    fn json_schema_is_rejected_explicitly() {
        let mut request = request();
        request.response_format = ResponseFormat::JsonSchema {
            name: "result".into(),
            description: None,
            schema: json!({"type": "object"}),
            strict: true,
        };

        assert!(matches!(
            WatsonxChatRequest::from_request(
                "ibm/granite-4-h-small",
                &config(WatsonxScope::Project("project-1".to_owned())),
                request,
            ),
            Err(LlmError::UnsupportedResponseFormat { .. })
        ));
    }

    #[test]
    fn response_parsing_preserves_model_usage_and_finish_reason() {
        let response: WatsonxChatResponse = serde_json::from_value(json!({
            "model_id": "ibm/granite-4-h-small",
            "model_version": "4.0.0",
            "choices": [{
                "message": {"content": "{\"result\":\"pass\"}"},
                "finish_reason": "max_tokens"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        }))
        .unwrap();
        let response = response.into_llm_response().unwrap();
        assert_eq!(response.model, "ibm/granite-4-h-small");
        assert_eq!(response.reported_model.as_deref(), Some("4.0.0"));
        assert_eq!(response.usage.total_tokens, 25);
        assert_eq!(response.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn stream_assembles_deltas_usage_and_done_marker() {
        let body = concat!(
            "data: {\"model_id\":\"ibm/granite-test\",\"choices\":[{\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        );
        let source = stream::iter([Ok::<bytes::Bytes, LlmError>(bytes::Bytes::from_static(
            body.as_bytes(),
        ))]);
        let mut events = watsonx_event_stream(
            source.eventsource(),
            "ibm/granite-resolved".to_owned(),
            Some("req-1".to_owned()),
        );
        let mut completed = None;
        while let Some(event) = events.next().await {
            if let LlmStreamEvent::ResponseCompleted { response } = event.unwrap() {
                completed = Some(response);
            }
        }
        let response = completed.expect("[DONE] must complete the response");
        assert_eq!(response.content.as_deref(), Some("hello world"));
        assert_eq!(response.usage.total_tokens, 5);
        assert_eq!(response.reported_model.as_deref(), Some("ibm/granite-test"));
    }

    #[tokio::test]
    async fn gzip_compressed_stream_is_decoded() {
        let expected =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"compressed\"}}]}\n\ndata: [DONE]\n\n";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(expected).unwrap();
        let compressed = encoder.finish().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            let headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-encoding: gzip\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                compressed.len()
            );
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(&compressed).await.unwrap();
        });
        let response = Client::builder()
            .build()
            .unwrap()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap();
        let decoded = decoded_watsonx_body(response).unwrap();
        let bytes = decoded
            .try_fold(Vec::new(), |mut all, chunk| async move {
                all.extend_from_slice(&chunk);
                Ok::<_, LlmError>(all)
            })
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(bytes, expected);
    }

    #[tokio::test]
    async fn stream_handles_duplicate_model_keys_and_eof_without_done() {
        let body = "data: {\"model_id\":\"meta-llama/llama-3-3-70b-instruct\",\"model\":\"meta-llama/llama-3-3-70b-instruct\",\"choices\":[{\"delta\":{\"content\":\"WatsonX works!\"},\"finish_reason\":\"stop\"}],\"usage\":{\"input_tokens\":10,\"generated_tokens\":5,\"total_tokens\":15}}\n\n";
        let source = stream::iter([Ok::<bytes::Bytes, LlmError>(bytes::Bytes::from_static(
            body.as_bytes(),
        ))]);
        let mut events = watsonx_event_stream(
            source.eventsource(),
            "meta-llama/llama-3-3-70b-instruct".to_owned(),
            Some("req-2".to_owned()),
        );
        let mut completed = None;
        while let Some(event) = events.next().await {
            if let LlmStreamEvent::ResponseCompleted { response } = event.unwrap() {
                completed = Some(response);
            }
        }
        let response = completed.expect("stream EOF must complete the response");
        assert_eq!(response.content.as_deref(), Some("WatsonX works!"));
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.completion_tokens, 5);
        assert_eq!(response.usage.total_tokens, 15);
        assert_eq!(
            response.reported_model.as_deref(),
            Some("meta-llama/llama-3-3-70b-instruct")
        );
    }
}

