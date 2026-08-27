//! Policy types: capability envelopes, context policies, model policies, and retry policies.

use crate::id::{SecretRef, ServerId, ToolName};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ── Model policy ──────────────────────────────────────────────────────────────

/// Selects which LLM model or model profile to use for an agent invocation.
/// The model router resolves a `ModelPolicy` to a concrete `LlmAdapter` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPolicy {
    /// A named profile (e.g. `"high_quality"`, `"fast"`, `"local"`).
    /// The model router maps profiles to concrete models.
    pub profile: Option<String>,
    /// An explicit model name that overrides the profile
    /// (e.g. `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`).
    pub model: Option<String>,
    /// Sampling temperature. Provider-default if absent.
    pub temperature: Option<f32>,
    /// Maximum output tokens. Provider-default if absent.
    pub max_tokens: Option<u32>,
}

impl Default for ModelPolicy {
    fn default() -> Self {
        Self {
            profile: Some("default".into()),
            model: None,
            temperature: None,
            max_tokens: None,
        }
    }
}

// ── Execution limits ──────────────────────────────────────────────────────────

/// Resource and iteration limits for a single agent invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLimits {
    /// Maximum number of LLM turns (model calls) in one invocation.
    pub max_turns: u32,
    /// Maximum number of MCP tool calls in one invocation.
    pub max_tool_calls: u32,
    /// Wall-clock timeout for the entire invocation.
    #[serde(with = "duration_secs")]
    pub timeout: Duration,
    /// Maximum total input + output tokens across all LLM calls.
    pub max_tokens_total: Option<u32>,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_turns: 10,
            max_tool_calls: 20,
            timeout: Duration::from_secs(600),
            max_tokens_total: None,
        }
    }
}

// ── Retry policy ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of attempts, including the first. Default: 1 (no retry).
    pub max_attempts: u32,
    /// Initial delay before the first retry.
    #[serde(with = "duration_secs")]
    pub delay: Duration,
    /// Backoff strategy applied to subsequent delays.
    pub backoff: BackoffStrategy,
    /// Event types or error classes that permit a retry.
    /// If empty, all failures are retryable.
    pub retryable_on: Vec<String>,
    /// Alternative model profile to use on retry attempts.
    pub fallback_model: Option<String>,
    /// State to transition to when all attempts are exhausted.
    /// If absent the workflow falls to the default failure path.
    pub on_exhausted: Option<String>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            delay: Duration::from_secs(1),
            backoff: BackoffStrategy::Exponential,
            retryable_on: vec![],
            fallback_model: None,
            on_exhausted: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackoffStrategy {
    Fixed,
    Linear,
    Exponential,
}

// ── Context policy ────────────────────────────────────────────────────────────

/// Describes what information an agent is allowed and expected to receive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextPolicy {
    /// Ordered list of context sources to resolve.
    pub sources: Vec<ContextSource>,
    /// Maximum number of tokens the assembled context may consume.
    pub token_budget: Option<u32>,
    /// Fields or sections to explicitly exclude from the context view.
    pub exclude: Vec<String>,
}

/// One source of information in the context resolution pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextSource {
    /// A specific versioned artifact (or the current version if unspecified).
    Artifact {
        selector: String,
        #[serde(default)]
        version: Option<String>,
    },
    /// A memory query (keyword, semantic, or structured filter).
    Memory {
        query: String,
        #[serde(default = "default_memory_limit")]
        limit: u32,
    },
    /// An inline reference to a workflow data expression.
    WorkflowData { expression: String },
}

fn default_memory_limit() -> u32 {
    10
}

// ── Redaction policy ─────────────────────────────────────────────────────────

/// Controls which fields of runtime events are redacted before they leave the
/// engine boundary.
///
/// Redaction is applied by `RedactingEventSink` (in `langchart-adapters`) as
/// a wrapper around any `EventSink`. Sensitive values are replaced with a
/// fixed placeholder string (e.g. `"[REDACTED]"`).
///
/// The primary use-case is production audit logging where tool arguments or
/// LLM prompt content must not appear in plain text in the event log.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RedactionPolicy {
    /// Redact the prompt / request content of [`LlmRequest`] events.
    #[serde(default)]
    pub redact_llm_prompts: bool,

    /// Redact tool-call arguments on [`ToolRequest`] events.
    #[serde(default)]
    pub redact_tool_arguments: bool,

    /// Redact tool-call result content if it is added to [`ToolResponse`]
    /// events in a future schema. Current events contain metadata only and
    /// never persist raw tool results.
    #[serde(default)]
    pub redact_tool_results: bool,

    /// Redact memory record content on [`MemoryStored`] events.
    #[serde(default)]
    pub redact_memory_content: bool,

    /// Redact the query preview on [`MemorySearched`] events.
    #[serde(default)]
    pub redact_memory_queries: bool,

    /// A custom list of JSON pointer paths (e.g. `"/data/password"`) whose
    /// values should be scrubbed from the event `payload` before the event is
    /// forwarded. Applied after all other redaction rules.
    #[serde(default)]
    pub scrub_paths: Vec<String>,
}

// ── Capability policy ─────────────────────────────────────────────────────────

/// The set of permissions granted to a state or agent.
/// Effective capabilities are computed as the intersection of all policy layers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityPolicy {
    /// Per-MCP-server tool and resource allowlists.
    #[serde(default)]
    pub mcp: HashMap<ServerId, McpServerPolicy>,
    /// Artifact operations permitted for this invocation.
    #[serde(default)]
    pub artifact_operations: Vec<OperationClass>,
    /// Whether the agent may write to long-term memory.
    #[serde(default)]
    pub memory_write: bool,
    /// Whether this policy explicitly elevates above the parent.
    /// Flagged by the validator.
    #[serde(default)]
    pub elevate: bool,
}

/// Restrictions on one MCP server's tools and resources.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerPolicy {
    /// Explicitly allowed tool names. Empty means none allowed.
    pub allow: Vec<ToolName>,
    /// Allowed resource URI patterns. Plain strings use glob syntax (`*` and
    /// `?`); regular expressions must use the explicit `regex:` prefix.
    #[serde(default)]
    pub resource_patterns: Vec<String>,
    /// Operation classes permitted on resources.
    #[serde(default)]
    pub operations: Vec<OperationClass>,
    /// Maximum number of tool calls to this server in one invocation.
    pub call_budget: Option<u32>,
    /// Named secret references whose values are injected into calls to this server.
    #[serde(default)]
    pub credentials: Vec<SecretRef>,
    /// If true, the broker emits a `human_confirmation_required` event and
    /// rejects the call without consuming its budget. The host may obtain
    /// approval and retry with a confirmation-cleared effective policy.
    #[serde(default)]
    pub require_human_confirmation: bool,
}

/// The class of operation being performed on an artifact or resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationClass {
    Read,
    Propose,
    Commit,
    Publish,
    Delete,
}

// ── Duration serialization helper ─────────────────────────────────────────────

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        d.as_secs().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(Duration::from_secs(secs))
    }
}
