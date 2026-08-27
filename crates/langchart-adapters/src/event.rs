//! Runtime event sink and source adapters.

use crate::llm::ResponseFormatKind;
use async_trait::async_trait;
use futures::Stream;
use langchart_model::id::{EventId, RegionId, RunId, StateId};
use serde::{Deserialize, Serialize};

/// Every observable runtime action produces one of these records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvent {
    pub event_id: EventId,
    pub run_id: RunId,
    /// RFC 3339 timestamp as a string (avoids needing time's serde features).
    pub timestamp: String,
    pub payload: RuntimeEventPayload,
}

/// The typed payload of a runtime event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeEventPayload {
    // Run lifecycle
    RunStarted,
    RunSuspended,
    RunResumed,
    RunCompleted,
    RunFailed {
        message: String,
    },
    RunCancelled,

    // State lifecycle
    StateEntered {
        state_id: StateId,
    },
    StateExited {
        state_id: StateId,
    },

    // Transitions
    TransitionSelected {
        from: StateId,
        to: StateId,
        event_type: String,
        /// Payload of the causal event, retained so the transition can be replayed.
        #[serde(default)]
        event_payload: serde_json::Value,
    },

    // Activities
    ActivityStarted {
        state_id: StateId,
    },
    ActivityCompleted {
        state_id: StateId,
    },
    ActivityFailed {
        state_id: StateId,
        message: String,
    },
    ActivityCancelled {
        state_id: StateId,
    },
    ActivityRetried {
        state_id: StateId,
        attempt: u32,
    },

    // Model calls
    LlmRequest {
        state_id: StateId,
        model: String,
        prompt_tokens: u32,
        #[serde(default)]
        response_format: ResponseFormatKind,
    },
    LlmResponse {
        state_id: StateId,
        model: String,
        output_tokens: u32,
        latency_ms: u64,
    },

    // Tool calls
    ToolRequest {
        state_id: StateId,
        server_id: String,
        tool_name: String,
    },
    ToolResponse {
        state_id: StateId,
        server_id: String,
        tool_name: String,
        latency_ms: u64,
    },
    ToolRejected {
        state_id: StateId,
        server_id: String,
        tool_name: String,
        reason: String,
    },

    // Memory
    MemoryStored {
        scope: String,
    },
    MemorySearched {
        query_preview: String,
    },

    // Context
    ContextResolved {
        state_id: StateId,
        token_count: u32,
        content_hash: String,
    },

    // Proposals
    ProposalCreated {
        artifact_id: String,
        proposal_id: String,
    },
    ProposalAccepted {
        artifact_id: String,
        proposal_id: String,
    },
    ProposalRejected {
        artifact_id: String,
        proposal_id: String,
        reason: String,
    },
    ProposalCommitted {
        artifact_id: String,
        proposal_id: String,
        new_version: String,
    },
    ProposalConflicted {
        artifact_id: String,
        proposal_id: String,
    },

    // Checkpoints
    CheckpointSaved,

    // Budgets
    BudgetWarning {
        budget_type: String,
        used: u32,
        limit: u32,
    },
    BudgetExhausted {
        budget_type: String,
    },

    // Human
    HumanInputRequested {
        state_id: StateId,
    },
    HumanInputReceived {
        state_id: StateId,
    },
    HumanConfirmationRequired {
        state_id: StateId,
        server_id: String,
        tool_name: String,
    },

    // Parallel regions
    ParallelRegionEntered {
        parallel_id: StateId,
        region_id: RegionId,
    },
    ParallelRegionCompleted {
        parallel_id: StateId,
        region_id: RegionId,
    },
    ParallelCompleted {
        parallel_id: StateId,
    },

    // History
    HistorySaved {
        state_id: StateId,
    },
    HistoryRestored {
        state_id: StateId,
    },

    // Subworkflow
    SubworkflowStarted {
        state_id: StateId,
    },
    SubworkflowCompleted {
        state_id: StateId,
    },
    SubworkflowFailed {
        state_id: StateId,
        message: String,
    },

    // Actions (on_entry / on_exit)
    ActionStarted {
        state_id: StateId,
        action_id: String,
    },
    ActionCompleted {
        state_id: StateId,
        action_id: String,
    },
    ActionFailed {
        state_id: StateId,
        action_id: String,
        message: String,
    },

    // Errors
    EventUnhandled {
        event_type: String,
    },
    ActivityInvalidOutput {
        state_id: StateId,
        event_type: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum EventSinkError {
    #[error("event sink error: {0}")]
    Sink(String),
}

/// Append-only sink for observable runtime events.
#[async_trait]
pub trait EventSink: Send + Sync {
    async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError>;
}

/// Subscribe to a stream of runtime events for a run.
pub trait EventSource: Send + Sync {
    fn subscribe(&self, run_id: &RunId) -> Box<dyn Stream<Item = RuntimeEvent> + Send + Unpin>;
}

// ── RedactingEventSink ────────────────────────────────────────────────────────

/// A wrapper around any [`EventSink`] that applies a [`RedactionPolicy`] before
/// forwarding each event.
///
/// Sensitive information (LLM prompts, tool arguments, memory query previews,
/// etc.) is replaced with `"[REDACTED]"` according to the active policy. This
/// ensures that the downstream sink — which may be a database, file, or remote
/// collector — never receives raw credential or PII data.
///
/// # Example
///
/// ```text
/// let redacting = RedactingEventSink::new(inner_sink, RedactionPolicy {
///     redact_tool_arguments: true,
///     redact_memory_queries: true,
///     ..Default::default()
/// });
/// ```
pub struct RedactingEventSink {
    inner: std::sync::Arc<dyn EventSink>,
    policy: langchart_model::policy::RedactionPolicy,
}

impl RedactingEventSink {
    /// Wrap `inner` with the given `policy`.
    pub fn new(
        inner: std::sync::Arc<dyn EventSink>,
        policy: langchart_model::policy::RedactionPolicy,
    ) -> Self {
        Self { inner, policy }
    }

    /// Apply the policy to a mutable event before forwarding.
    fn redact(&self, event: &mut RuntimeEvent) {
        match &mut event.payload {
            RuntimeEventPayload::LlmRequest { .. } => {
                // `prompt_tokens` is a count — safe to log. The redaction flag
                // covers the *content* of the prompt, which is not currently a
                // field in `LlmRequest` event payloads (content lives in the
                // LlmAdapter call, not the observable event). Nothing to scrub
                // here in the current schema; flag reserved for future expansion.
                let _ = self.policy.redact_llm_prompts;
            }
            RuntimeEventPayload::ToolRequest { tool_name, .. } => {
                if self.policy.redact_tool_arguments {
                    // The observable event only carries the tool name, not the
                    // raw arguments — those are passed through the broker and
                    // never serialised into the event. Scrub the tool name when
                    // redact_tool_arguments is set, replacing it with a token
                    // that identifies the call class but not the operation.
                    *tool_name = "[REDACTED]".into();
                }
            }
            RuntimeEventPayload::MemorySearched { query_preview } => {
                if self.policy.redact_memory_queries {
                    *query_preview = "[REDACTED]".into();
                }
            }
            RuntimeEventPayload::MemoryStored { scope } if self.policy.redact_memory_content => {
                *scope = "[REDACTED]".into();
            }
            _ => {}
        }
    }
}

#[async_trait]
impl EventSink for RedactingEventSink {
    async fn append(&self, mut event: RuntimeEvent) -> Result<(), EventSinkError> {
        self.redact(&mut event);
        self.inner.append(event).await
    }
}

#[cfg(test)]
mod redaction_tests {
    use super::*;
    use langchart_model::{
        id::{EventId, RunId},
        policy::RedactionPolicy,
    };
    use std::sync::{Arc, Mutex};

    /// A simple in-memory sink used only for testing.
    #[derive(Default, Clone)]
    struct VecSink(Arc<Mutex<Vec<RuntimeEvent>>>);

    #[async_trait]
    impl EventSink for VecSink {
        async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
            self.0.lock().unwrap().push(event);
            Ok(())
        }
    }

    fn make_event(payload: RuntimeEventPayload) -> RuntimeEvent {
        RuntimeEvent {
            event_id: EventId::new("test-event"),
            run_id: RunId::new("test-run"),
            timestamp: "2024-01-01T00:00:00Z".into(),
            payload,
        }
    }

    #[tokio::test]
    async fn tool_arguments_redacted_when_policy_set() {
        let inner = Arc::new(VecSink::default());
        let sink = RedactingEventSink::new(
            inner.clone(),
            RedactionPolicy {
                redact_tool_arguments: true,
                ..Default::default()
            },
        );

        let state = langchart_model::id::StateId::new("s");
        let event = make_event(RuntimeEventPayload::ToolRequest {
            state_id: state,
            server_id: "srv".into(),
            tool_name: "my_tool".into(),
        });

        sink.append(event).await.unwrap();
        let stored = inner.0.lock().unwrap();
        match &stored[0].payload {
            RuntimeEventPayload::ToolRequest { tool_name, .. } => {
                assert_eq!(tool_name, "[REDACTED]");
            }
            _ => panic!("wrong payload"),
        }
    }

    #[tokio::test]
    async fn memory_query_redacted_when_policy_set() {
        let inner = Arc::new(VecSink::default());
        let sink = RedactingEventSink::new(
            inner.clone(),
            RedactionPolicy {
                redact_memory_queries: true,
                ..Default::default()
            },
        );

        let event = make_event(RuntimeEventPayload::MemorySearched {
            query_preview: "find documents about passwords".into(),
        });
        sink.append(event).await.unwrap();

        let stored = inner.0.lock().unwrap();
        match &stored[0].payload {
            RuntimeEventPayload::MemorySearched { query_preview } => {
                assert_eq!(query_preview, "[REDACTED]");
            }
            _ => panic!("wrong payload"),
        }
    }

    #[tokio::test]
    async fn no_redaction_when_policy_empty() {
        let inner = Arc::new(VecSink::default());
        let sink = RedactingEventSink::new(inner.clone(), RedactionPolicy::default());

        let event = make_event(RuntimeEventPayload::MemorySearched {
            query_preview: "sensitive query".into(),
        });
        sink.append(event).await.unwrap();

        let stored = inner.0.lock().unwrap();
        match &stored[0].payload {
            RuntimeEventPayload::MemorySearched { query_preview } => {
                assert_eq!(query_preview, "sensitive query");
            }
            _ => panic!("wrong payload"),
        }
    }

    #[test]
    fn llm_request_event_records_only_response_format_kind() {
        let payload = RuntimeEventPayload::LlmRequest {
            state_id: StateId::new("s"),
            model: "test-model".into(),
            prompt_tokens: 0,
            response_format: ResponseFormatKind::JsonSchema,
        };

        let value = serde_json::to_value(payload).unwrap();
        assert_eq!(value["response_format"], "json_schema");
        assert!(value.get("schema").is_none());
    }

    #[test]
    fn old_transition_event_without_payload_defaults_to_null() {
        let payload: RuntimeEventPayload = serde_json::from_value(serde_json::json!({
            "kind": "transition_selected",
            "from": "source",
            "to": "target",
            "event_type": "continue"
        }))
        .unwrap();

        assert!(matches!(
            payload,
            RuntimeEventPayload::TransitionSelected { event_payload, .. }
                if event_payload.is_null()
        ));
    }
}
