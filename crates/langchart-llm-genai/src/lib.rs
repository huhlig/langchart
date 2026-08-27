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
use futures::{Stream, StreamExt, stream};
use genai::{
    adapter::AdapterKind,
    chat::{ChatOptions, ChatResponseFormat, ChatStreamEvent, MessageContent},
};
use langchart_adapters::llm::{
    FinishReason, LlmAdapter, LlmError, LlmEventStream, LlmRequest, LlmResponse, LlmStreamEvent,
    ResponseFormat, TokenUsage, ToolCall,
};
use std::collections::VecDeque;

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

    async fn complete_stream(&self, request: LlmRequest) -> Result<LlmEventStream, LlmError> {
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
        let mut options = chat_options(&request, target.model.adapter_kind)?;
        options.capture_usage = Some(true);
        options.capture_content = Some(true);
        options.capture_reasoning_content = Some(true);
        let chat_req = bridge::to_genai_request(&request);
        let response = self
            .client
            .exec_chat_stream(model, chat_req, Some(&options))
            .await
            .map_err(error::map)?;
        let resolved_model = response.model_iden.model_name.to_string();
        Ok(normalize_stream(response.stream, resolved_model))
    }
}

fn normalize_stream<S>(stream: S, model: String) -> LlmEventStream
where
    S: Stream<Item = genai::Result<ChatStreamEvent>> + Send + Unpin + 'static,
{
    struct State<S> {
        stream: S,
        queue: VecDeque<Result<LlmStreamEvent, LlmError>>,
        model: String,
        content: String,
        reasoning: String,
        done: bool,
    }

    let state = State {
        stream,
        queue: VecDeque::new(),
        model,
        content: String::new(),
        reasoning: String::new(),
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
                Some(Ok(ChatStreamEvent::Start)) => {
                    state.queue.push_back(Ok(LlmStreamEvent::ResponseStarted {
                        request_id: None,
                        reported_model: None,
                    }));
                }
                Some(Ok(ChatStreamEvent::Chunk(chunk))) => {
                    state.content.push_str(&chunk.content);
                    state.queue.push_back(Ok(LlmStreamEvent::TextDelta {
                        delta: chunk.content,
                    }));
                }
                Some(Ok(ChatStreamEvent::ReasoningChunk(chunk))) => {
                    state.reasoning.push_str(&chunk.content);
                    state.queue.push_back(Ok(LlmStreamEvent::ReasoningDelta {
                        delta: chunk.content,
                    }));
                }
                Some(Ok(ChatStreamEvent::End(end))) => {
                    let usage = end.captured_usage.map(normalize_usage).unwrap_or_default();
                    let mut tool_calls = Vec::new();
                    if let Some(MessageContent::ToolCalls(calls)) = end.captured_content {
                        for (index, call) in calls.into_iter().enumerate() {
                            let arguments = serde_json::to_string(&call.fn_arguments)
                                .unwrap_or_else(|_| "null".to_owned());
                            state.queue.push_back(Ok(LlmStreamEvent::ToolCallStart {
                                index,
                                id: Some(call.call_id.clone()),
                                name: Some(call.fn_name.clone()),
                            }));
                            state
                                .queue
                                .push_back(Ok(LlmStreamEvent::ToolCallArgumentsDelta {
                                    index,
                                    delta: arguments,
                                }));
                            state
                                .queue
                                .push_back(Ok(LlmStreamEvent::ToolCallEnd { index }));
                            tool_calls.push(ToolCall {
                                id: call.call_id,
                                name: call.fn_name,
                                arguments: call.fn_arguments,
                            });
                        }
                    }
                    let finish_reason = if tool_calls.is_empty() {
                        FinishReason::Stop
                    } else {
                        FinishReason::ToolCalls
                    };
                    state.queue.push_back(Ok(LlmStreamEvent::UsageUpdate {
                        usage: usage.clone(),
                    }));
                    state.queue.push_back(Ok(LlmStreamEvent::FinishReason {
                        reason: finish_reason.clone(),
                    }));
                    state.queue.push_back(Ok(LlmStreamEvent::ResponseCompleted {
                        response: LlmResponse {
                            content: (!state.content.is_empty()).then(|| state.content.clone()),
                            tool_calls,
                            usage,
                            finish_reason,
                            refusal: None,
                            model: state.model.clone(),
                            reported_model: None,
                        },
                    }));
                    state.done = true;
                }
                Some(Err(error)) => {
                    state.done = true;
                    return Some((Err(error::map(error)), state));
                }
                None => {
                    state.done = true;
                    return Some((
                        Err(LlmError::IncompleteStream {
                            received_bytes: state.content.len() + state.reasoning.len(),
                            finish_event_seen: false,
                        }),
                        state,
                    ));
                }
            }
        }
    }))
}

fn normalize_usage(usage: genai::chat::Usage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: usage.prompt_tokens.unwrap_or(0).max(0) as u32,
        completion_tokens: usage.completion_tokens.unwrap_or(0).max(0) as u32,
        total_tokens: usage.total_tokens.unwrap_or(0).max(0) as u32,
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

    #[tokio::test]
    async fn native_stream_is_normalized_and_completed() {
        let usage = genai::chat::Usage {
            prompt_tokens: Some(2),
            completion_tokens: Some(3),
            total_tokens: Some(5),
            ..Default::default()
        };
        let upstream = stream::iter([
            Ok::<_, genai::Error>(ChatStreamEvent::Start),
            Ok(ChatStreamEvent::Chunk(genai::chat::StreamChunk {
                content: "hello".into(),
            })),
            Ok(ChatStreamEvent::ReasoningChunk(genai::chat::StreamChunk {
                content: "because".into(),
            })),
            Ok(ChatStreamEvent::End(genai::chat::StreamEnd {
                captured_usage: Some(usage),
                captured_content: Some(MessageContent::Text("hello".into())),
                captured_reasoning_content: Some("because".into()),
            })),
        ]);
        let events = normalize_stream(upstream, "test-model".into())
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            &events[0],
            Ok(LlmStreamEvent::ResponseStarted {
                reported_model: None,
                ..
            })
        ));
        assert!(matches!(
            &events[1],
            Ok(LlmStreamEvent::TextDelta { delta }) if delta == "hello"
        ));
        assert!(matches!(
            &events[2],
            Ok(LlmStreamEvent::ReasoningDelta { delta }) if delta == "because"
        ));
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(LlmStreamEvent::UsageUpdate { usage }) if usage.total_tokens == 5
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(LlmStreamEvent::ResponseCompleted { response })
                if response.content.as_deref() == Some("hello")
                    && response.usage.total_tokens == 5
                    && response.model == "test-model"
        )));
    }

    #[tokio::test]
    async fn native_stream_preserves_captured_tool_calls() {
        let call = genai::chat::ToolCall {
            call_id: "call-1".into(),
            fn_name: "weather".into(),
            fn_arguments: serde_json::json!({ "city": "Paris" }),
        };
        let upstream = stream::iter([
            Ok::<_, genai::Error>(ChatStreamEvent::Start),
            Ok(ChatStreamEvent::End(genai::chat::StreamEnd {
                captured_usage: None,
                captured_content: Some(MessageContent::ToolCalls(vec![call])),
                captured_reasoning_content: None,
            })),
        ]);
        let events = normalize_stream(upstream, "test-model".into())
            .collect::<Vec<_>>()
            .await;

        assert!(events.iter().any(|event| matches!(
            event,
            Ok(LlmStreamEvent::ToolCallStart { id, name, .. })
                if id.as_deref() == Some("call-1") && name.as_deref() == Some("weather")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(LlmStreamEvent::ResponseCompleted { response })
                if response.tool_calls.len() == 1
                    && response.finish_reason == FinishReason::ToolCalls
        )));
    }
}
