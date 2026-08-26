//! LLM adapter: abstract over model providers.

use async_trait::async_trait;
use langchart_model::policy::ModelPolicy;
use serde::{Deserialize, Serialize};

/// A tool definition exposed to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the tool's parameters.
    pub parameters: serde_json::Value,
}

/// A single message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// A tool call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: serde_json::Value,
}

/// The response representation required from an LLM provider.
///
/// Structured variants require native provider support. Adapters must return
/// [`LlmError::UnsupportedResponseFormat`] rather than emulate or downgrade
/// them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: serde_json::Value,
        strict: bool,
    },
}

/// Non-sensitive response-format identifier used in errors and observability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatKind {
    #[default]
    Text,
    JsonObject,
    JsonSchema,
}

impl ResponseFormat {
    pub fn kind(&self) -> ResponseFormatKind {
        match self {
            Self::Text => ResponseFormatKind::Text,
            Self::JsonObject => ResponseFormatKind::JsonObject,
            Self::JsonSchema { .. } => ResponseFormatKind::JsonSchema,
        }
    }
}

impl std::fmt::Display for ResponseFormatKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Text => "text",
            Self::JsonObject => "json_object",
            Self::JsonSchema => "json_schema",
        })
    }
}

/// A request to the LLM adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    /// Resolved model policy (may be overridden by the router).
    pub model_policy: ModelPolicy,
    /// Conversation history including the system prompt as the first message.
    pub messages: Vec<Message>,
    /// Tools available to the model for this call (from the capability envelope).
    pub tools: Vec<ToolDefinition>,
    /// Required provider-native response representation.
    #[serde(default)]
    pub response_format: ResponseFormat,
}

/// Token usage for one LLM call.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// The reason the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Other(String),
}

/// A response from the LLM adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Text content, if any (may be empty when `tool_calls` is non-empty).
    pub content: Option<String>,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
    pub finish_reason: FinishReason,
    /// Provider refusal, separate from ordinary response content.
    #[serde(default)]
    pub refusal: Option<String>,
    /// Raw model name returned by the provider.
    pub model: String,
}

/// A model available from a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub description: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("rate limited by provider: {0}")]
    RateLimited(String),
    #[error("model `{model}` not found")]
    ModelNotFound { model: String },
    #[error("context length exceeded")]
    ContextLengthExceeded,
    #[error("content filtered by provider")]
    ContentFiltered,
    #[error("adapter `{adapter}` does not support response format `{requested}`")]
    UnsupportedResponseFormat {
        adapter: String,
        requested: ResponseFormatKind,
    },
    #[error("LLM provider error: {0}")]
    Provider(String),
    #[error("request timed out")]
    Timeout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_response_format_deserializes_as_text() {
        let request: LlmRequest = serde_json::from_value(serde_json::json!({
            "model_policy": {},
            "messages": [],
            "tools": []
        }))
        .unwrap();

        assert_eq!(request.response_format, ResponseFormat::Text);
    }

    #[test]
    fn response_format_kinds_do_not_include_schema_material() {
        let format = ResponseFormat::JsonSchema {
            name: "result".into(),
            description: Some("sensitive description".into()),
            schema: serde_json::json!({"secret": "value"}),
            strict: true,
        };

        assert_eq!(serde_json::to_value(format.kind()).unwrap(), "json_schema");
    }

    #[test]
    fn missing_refusal_deserializes_as_none() {
        let response: LlmResponse = serde_json::from_value(serde_json::json!({
            "content": "ok",
            "tool_calls": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "finish_reason": "stop",
            "model": "test"
        }))
        .unwrap();

        assert_eq!(response.refusal, None);
    }
}

/// Abstraction over a language model provider.
#[async_trait]
pub trait LlmAdapter: Send + Sync {
    /// Perform one completion call.
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError>;

    /// List models available from this provider. Optional — returns empty vec
    /// if the provider does not support enumeration.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        Ok(vec![])
    }
}
