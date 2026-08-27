//! `AgentActor` trait, `StateAction` trait, and test doubles.
//!
//! The `AgentActor` and `StateAction` traits live here (in `langchart-runtime`)
//! because they receive a reference to `CapabilityBroker`, which is a runtime
//! type.  Agent and action implementors take a dependency on `langchart-runtime`.

use crate::broker::{CapabilityBroker, CapabilityEnvelope};
use async_trait::async_trait;
use langchart_adapters::context::ContextView;
use langchart_model::{
    id::{InvocationId, RunId, StateId},
    policy::ExecutionLimits,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

// ── Agent invocation context ──────────────────────────────────────────────────

/// Everything the runtime provides to an agent actor when starting it.
#[derive(Debug)]
pub struct AgentInvocation {
    pub run_id: RunId,
    pub state_id: StateId,
    pub invocation_id: InvocationId,
    /// Resolved system instructions + task prompt.
    pub instructions: ResolvedInstructions,
    /// Immutable context snapshot assembled by the ContextResolverChain.
    pub context_view: ContextView,
    /// Input data resolved from the state's `input:` expressions (RON).
    pub input: ron::Value,
    /// Declared output event schemas the actor MUST emit one of.
    pub output_event_types: Vec<String>,
    /// Resource and iteration limits.
    pub limits: ExecutionLimits,
}

/// Resolved prompt instructions for one invocation.
#[derive(Debug, Clone)]
pub struct ResolvedInstructions {
    pub system: String,
    pub task: Option<String>,
}

// ── Output event envelope ─────────────────────────────────────────────────────

/// The typed event an agent actor emits when it completes (or fails).
///
/// The runtime validates `event_type` against `AgentInvocation::output_event_types`
/// before accepting it. Undeclared types produce `activity.invalid_output`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutputEvent {
    /// Must be one of the declared `output_event_types`.
    pub event_type: String,
    /// Structured payload. Validated against the declared event schema.
    pub payload: serde_json::Value,
}

/// Errors returned by an agent actor.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("turn limit exhausted")]
    TurnLimitExhausted,

    #[error("tool call limit exhausted")]
    ToolCallLimitExhausted,

    #[error("agent internal error: {0}")]
    Internal(String),

    #[error("actor was cancelled")]
    Cancelled,
}

// ── AgentActor trait ──────────────────────────────────────────────────────────

/// An opaque, async unit of agent execution.
///
/// The runtime starts the actor and awaits its completion. The actor is
/// responsible for its own internal loop (multi-turn ReAct, single-shot, etc.)
/// and MUST eventually emit exactly one `AgentOutputEvent`.
///
/// The runtime enforces limits via `CapabilityBroker`; actors that attempt
/// to exceed their budget will receive `BrokerError::TurnLimitExhausted` or
/// `BrokerError::ToolCallLimitExhausted` from the broker.
#[async_trait]
pub trait AgentActor: Send + Sync {
    async fn run(
        &self,
        invocation: AgentInvocation,
        envelope: CapabilityEnvelope,
        broker: Arc<CapabilityBroker>,
    ) -> Result<AgentOutputEvent, AgentError>;
}

// ── ScriptedAgentActor ────────────────────────────────────────────────────────

/// A deterministic test double for `AgentActor`.
///
/// Emits a pre-configured event after an optional simulated delay.
/// Used in model-free runtime tests (transitions, retries, parallel regions,
/// timers, suspension, recovery, artifact conflicts — all without an LLM).
///
/// # Example
/// ```rust
/// use langchart_runtime::instance::ScriptedAgentActor;
/// use serde_json::json;
///
/// let actor = ScriptedAgentActor::emit("analysis.completed", json!({"confidence": 0.9}));
/// ```
pub struct ScriptedAgentActor {
    event_type: String,
    payload: serde_json::Value,
    /// Optional simulated delay before emitting (useful for timeout tests).
    delay: Option<std::time::Duration>,
    /// If Some, the actor returns this error instead of emitting an event.
    fail_with: Option<String>,
}

impl ScriptedAgentActor {
    /// Emit a successful event with the given type and payload.
    pub fn emit(event_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            event_type: event_type.into(),
            payload,
            delay: None,
            fail_with: None,
        }
    }

    /// Emit after a simulated delay.
    pub fn emit_after(
        event_type: impl Into<String>,
        payload: serde_json::Value,
        delay: std::time::Duration,
    ) -> Self {
        Self {
            delay: Some(delay),
            ..Self::emit(event_type, payload)
        }
    }

    /// Fail with an internal error (exercises the failure path).
    pub fn fail(message: impl Into<String>) -> Self {
        Self {
            event_type: "activity.failed".into(),
            payload: serde_json::Value::Null,
            delay: None,
            fail_with: Some(message.into()),
        }
    }
}

#[async_trait]
impl AgentActor for ScriptedAgentActor {
    async fn run(
        &self,
        _invocation: AgentInvocation,
        _envelope: CapabilityEnvelope,
        _broker: Arc<CapabilityBroker>,
    ) -> Result<AgentOutputEvent, AgentError> {
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        if let Some(msg) = &self.fail_with {
            return Err(AgentError::Internal(msg.clone()));
        }
        Ok(AgentOutputEvent {
            event_type: self.event_type.clone(),
            payload: self.payload.clone(),
        })
    }
}

// ── StateAction trait ─────────────────────────────────────────────────────────

/// Context provided to a [`StateAction`] when it runs.
#[derive(Debug)]
pub struct ActionContext {
    pub run_id: RunId,
    pub state_id: StateId,
    pub action_id: String,
    pub trigger: ActionTrigger,
}

/// Whether the action is running on state entry, state exit, or a transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionTrigger {
    Entry,
    Exit,
    Transition,
}

/// An error returned by a [`StateAction`].
#[derive(Debug, thiserror::Error)]
#[error("action `{action_id}` failed: {message}")]
pub struct ActionError {
    pub action_id: String,
    pub message: String,
}

/// A synchronous side-effect attached to a state or transition.
///
/// Actions are identified by string ID in the workflow document
/// (`on_entry: ["log_started"]` or transition `actions: ["audit"]`) and
/// registered with the `ActionRegistry` when constructing the `WorkflowInstance`.
///
/// Actions must complete quickly. Long-running work belongs in an
/// `AgentActor`, not a `StateAction`.
#[async_trait]
pub trait StateAction: Send + Sync {
    async fn run(
        &self,
        ctx: ActionContext,
        broker: Arc<CapabilityBroker>,
    ) -> Result<(), ActionError>;
}

/// Registry mapping action IDs (from the workflow document) to concrete
/// [`StateAction`] implementations.
///
/// ```rust
/// use langchart_runtime::instance::{ActionRegistry, NoopAction};
///
/// let registry = ActionRegistry::new()
///     .register("log_started", NoopAction);
/// ```
#[derive(Default)]
pub struct ActionRegistry {
    actions: HashMap<String, Arc<dyn StateAction>>,
}

impl ActionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a `StateAction` under `id`.
    pub fn register(mut self, id: impl Into<String>, action: impl StateAction + 'static) -> Self {
        self.actions.insert(id.into(), Arc::new(action));
        self
    }

    /// Look up an action by ID.
    pub fn get(&self, id: &str) -> Option<Arc<dyn StateAction>> {
        self.actions.get(id).cloned()
    }

    pub fn into_map(self) -> HashMap<String, Arc<dyn StateAction>> {
        self.actions
    }
}

// ── NoopAction ────────────────────────────────────────────────────────────────

/// A `StateAction` that does nothing. Useful for testing or stubbing.
pub struct NoopAction;

#[async_trait]
impl StateAction for NoopAction {
    async fn run(
        &self,
        _ctx: ActionContext,
        _broker: Arc<CapabilityBroker>,
    ) -> Result<(), ActionError> {
        Ok(())
    }
}
