//! State model: state types, lifecycle configuration, and compiled state graph.

use crate::{
    id::{AgentId, AgentVersion, RegionId, StateId},
    policy::{CapabilityPolicy, ContextPolicy, ExecutionLimits, ModelPolicy, RetryPolicy},
    workflow::EventSchema,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ── State type ────────────────────────────────────────────────────────────────

/// The fundamental kind of a state, determining its execution behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateType {
    /// Runs a deterministic activity or waits for an event.
    Atomic,
    /// Starts an agent actor within a capability and context envelope.
    Agentic,
    /// Contains a nested statechart with an initial child state.
    Compound,
    /// Activates two or more orthogonal regions concurrently.
    Parallel,
    /// Suspends until an authorized human supplies a decision or data.
    Human,
    /// Invokes a separately versioned workflow through typed ports.
    Subworkflow,
    /// Marks completion of a region or workflow.
    Final,
}

// ── Agent reference ───────────────────────────────────────────────────────────

/// A versioned reference to an agent definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRef {
    pub id: AgentId,
    pub version: AgentVersion,
}

// ── Subworkflow port binding ───────────────────────────────────────────────────

/// Maps caller workflow-data expressions to child workflow input fields,
/// and child output event fields back to caller workflow data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PortBinding {
    /// `child_field_name → workflow-data expression` evaluated at invocation time.
    pub input: HashMap<String, String>,
    /// Maps the child's final-transition event type to field-level bindings.
    /// Mapped fields are written to caller data and emitted as
    /// `subworkflow.<child-event-type>`.
    pub output: HashMap<String, HashMap<String, String>>,
}

// ── Parallel completion mode ──────────────────────────────────────────────────

/// Determines when a parallel state is considered complete.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ParallelCompletion {
    /// All regions must reach a final state.
    #[default]
    All,
    /// The first region to reach a final state completes the parallel state.
    Any,
    /// Exactly N regions must complete.
    Quorum { n: usize },
    /// A CEL expression evaluated after each region completion.
    Guard { expr: String },
    /// Requires an explicit external termination event.
    Manual,
}

// ── History mode ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryMode {
    /// Remember only the most recently active direct child.
    Shallow,
    /// Remember the full active configuration recursively.
    Deep,
}

// ── State definition ─────────────────────────────────────────────────────────

/// A state as declared in the workflow document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDefinition {
    /// Stable slug identity. Never changes even when the display name does.
    pub id: StateId,
    /// Human-readable display name for the editor.
    pub name: String,
    /// The fundamental state kind.
    #[serde(rename = "type")]
    pub state_type: StateType,

    // ── Agentic fields ──
    /// Agent to invoke (agentic states only).
    pub agent: Option<AgentRef>,
    /// Static task prompt appended to the agent's system instructions.
    pub prompt: Option<String>,
    /// Workflow-data expressions mapped to agent input fields.
    #[serde(default)]
    pub input: HashMap<String, String>,
    /// Context policy for this state (overrides/narrows agent defaults).
    pub context: Option<ContextPolicy>,
    /// Model policy for this state (overrides agent defaults).
    pub model: Option<ModelPolicy>,
    /// Capability policy for this state.
    pub capabilities: Option<CapabilityPolicy>,
    /// Execution resource limits.
    pub limits: Option<ExecutionLimits>,

    // ── Compound / parallel fields ──
    /// Children states (compound and parallel).
    #[serde(default)]
    pub states: Vec<StateDefinition>,
    /// Parallel regions (parallel states only; alternative to `states` for named regions).
    #[serde(default)]
    pub regions: Vec<ParallelRegion>,
    /// Parallel completion mode (parallel states only).
    pub completion: Option<ParallelCompletion>,
    /// History mode (compound and parallel states only).
    pub history: Option<HistoryMode>,
    /// ID of the initial child state (compound states only).
    pub initial: Option<StateId>,

    // ── Subworkflow fields ──
    /// `workflow_id@version` reference (subworkflow states only).
    pub workflow_ref: Option<String>,
    /// Input/output port bindings (subworkflow states only).
    pub ports: Option<PortBinding>,

    // ── Human fields ──
    /// The role or identity required to fulfill a human state.
    #[serde(default)]
    pub authorized_roles: Vec<String>,

    // ── Lifecycle ──
    /// Registered action IDs executed in order when the state is entered.
    #[serde(default)]
    pub on_entry: Vec<String>,
    /// Registered action IDs executed in order when the state is exited.
    #[serde(default)]
    pub on_exit: Vec<String>,
    /// Retry policy for this state's activity.
    pub retry: Option<RetryPolicy>,
    /// Wall-clock timeout on the state's activity (overrides `limits.timeout`).
    #[serde(default, with = "option_duration_secs")]
    pub timeout: Option<Duration>,

    // ── Transitions ──
    /// Declared outbound transitions keyed by event type.
    /// Multiple specs per event are allowed; they are evaluated in priority order
    /// (lowest integer = highest priority) with guards as discriminators.
    #[serde(default)]
    pub on: HashMap<String, Vec<TransitionSpec>>,
    /// Optional payload schemas for output events emitted by this state.
    /// Keys are event type names; values declare required fields and their
    /// JSON types. An absent entry means the event type is unconstrained.
    #[serde(default)]
    pub output_schemas: HashMap<String, EventSchema>,

    // ── Editor metadata ──
    /// Non-semantic visual layout data. MUST NOT affect execution.
    #[serde(default)]
    pub _editor: serde_json::Value,
}

// ── Parallel region ───────────────────────────────────────────────────────────

/// A named orthogonal region within a parallel state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelRegion {
    pub id: RegionId,
    pub name: String,
    /// Initial state of this region.
    pub initial: StateId,
    /// States contained in this region.
    pub states: Vec<StateDefinition>,
}

// ── Transition spec ───────────────────────────────────────────────────────────

/// An outbound transition as declared in a state's `on:` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionSpec {
    /// Target state ID.
    pub target: StateId,
    /// CEL guard expression. Absence means always-true.
    pub guard: Option<String>,
    /// Transition priority — lower integer = higher priority. Default 0.
    #[serde(default)]
    pub priority: i32,
    /// Registered action IDs executed after state exit and before target entry.
    #[serde(default)]
    pub actions: Vec<String>,
    /// Transition kind. Default: External.
    #[serde(default)]
    pub kind: TransitionKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TransitionKind {
    #[default]
    External,
    Internal,
    Local,
}

// ── Option<Duration> serde helper ──────────────────────────────────────────────

mod option_duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(opt: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        opt.map(|d| d.as_secs()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        let opt = Option::<u64>::deserialize(d)?;
        Ok(opt.map(Duration::from_secs))
    }
}
