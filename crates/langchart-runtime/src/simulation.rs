//! Simulation mode for testing and evaluation.
//!
//! `WorkflowSimulator` provides a high-level harness for running a workflow
//! deterministically without an LLM. It:
//!
//! - accepts a `ScriptedActorMap` (`state_id → ScriptedAgentActor`) so every
//!   agentic state is driven by a pre-configured script;
//! - exposes `run_to_completion`, `step`, and `inject` methods;
//! - captures all emitted `RuntimeEvent`s for assertion in tests;
//! - returns the final `RunStatus` and full event log.
//!
//! # Example
//!
//! ```rust
//! # use langchart_runtime::simulation::WorkflowSimulator;
//! # use langchart_runtime::instance::ScriptedAgentActor;
//! # use langchart_model::validation::compile;
//! # // (document construction elided for brevity)
//! ```
//!
//! This module is **not** a replacement for the production `RuntimeEngine`.
//! It is a testing / authoring utility.

use crate::{
    broker::CapabilityBroker,
    instance::{AgentActor, ScriptedAgentActor},
    run::{RunStatus, WorkflowInstance},
};
use async_trait::async_trait;
use langchart_adapters::{
    event::{EventSink, EventSinkError, RuntimeEvent, RuntimeEventPayload},
    llm::{FinishReason, LlmAdapter, LlmError, LlmRequest, LlmResponse, TokenUsage},
    mcp::{McpAdapter, McpError, ResourceContent, ToolDefinition as McpToolDef},
    memory::{MemoryAdapter, MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult},
    secrets::{HostMapSecretsAdapter, SecretsAdapter},
};
use langchart_model::{
    id::{IdempotencyKey, RunId, ServerId, StateId, ToolName},
    validation::CompiledWorkflow,
};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

// ── No-op adapters for simulation ────────────────────────────────────────────

struct NoopLlm;

#[async_trait]
impl LlmAdapter for NoopLlm {
    async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse, LlmError> {
        // Scripted actors never call the broker LLM in tests; this should
        // never be reached in a correctly configured simulation.
        Ok(LlmResponse {
            content: Some("(simulation noop)".into()),
            tool_calls: vec![],
            usage: TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            finish_reason: FinishReason::Stop,
            refusal: None,
            model: "simulation/noop".into(),
            reported_model: None,
        })
    }
}

struct NoopMcp;

#[async_trait]
impl McpAdapter for NoopMcp {
    async fn call_tool(
        &self,
        _server_id: &ServerId,
        _tool: &ToolName,
        _args: serde_json::Value,
        _credentials: &[langchart_adapters::mcp::McpCredential],
        _key: Option<&IdempotencyKey>,
    ) -> Result<serde_json::Value, McpError> {
        Err(McpError::Call("simulation: no MCP server wired".into()))
    }

    async fn list_tools(&self, _server_id: &ServerId) -> Result<Vec<McpToolDef>, McpError> {
        Ok(vec![])
    }

    async fn read_resource(
        &self,
        _server_id: &ServerId,
        _uri: &str,
    ) -> Result<ResourceContent, McpError> {
        Err(McpError::Call("simulation: no MCP server wired".into()))
    }
}

struct NoopMemory;

#[async_trait]
impl MemoryAdapter for NoopMemory {
    async fn store(&self, _r: MemoryRecord) -> Result<MemoryId, MemoryError> {
        Ok(MemoryId("sim-noop".into()))
    }
    async fn search(&self, _q: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError> {
        Ok(vec![])
    }
    async fn get(&self, _id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
        Ok(None)
    }
    async fn delete(&self, _id: &MemoryId) -> Result<(), MemoryError> {
        Ok(())
    }
}

// ── Capturing event sink ──────────────────────────────────────────────────────

/// An `EventSink` that records all events for later inspection.
#[derive(Default, Clone)]
pub struct CapturingSink {
    events: Arc<Mutex<Vec<RuntimeEvent>>>,
}

#[async_trait]
impl EventSink for CapturingSink {
    async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
        self.events.lock().await.push(event);
        Ok(())
    }
}

impl CapturingSink {
    /// Clone the recorded events without clearing the buffer.
    /// **Prefer [`take`] in production code** — this allocates a full copy.
    pub async fn drain(&self) -> Vec<RuntimeEvent> {
        self.events.lock().await.clone()
    }

    /// Move the recorded events out of the buffer, leaving it empty.
    ///
    /// More efficient than [`drain`] because it avoids a clone; suitable for
    /// high-frequency use in production observers.
    pub async fn take(&self) -> Vec<RuntimeEvent> {
        let mut guard = self.events.lock().await;
        std::mem::take(&mut *guard)
    }

    /// Return all captured payloads.
    pub async fn payloads(&self) -> Vec<RuntimeEventPayload> {
        self.events
            .lock()
            .await
            .iter()
            .map(|e| e.payload.clone())
            .collect()
    }
}

// ── Scripted actor map ────────────────────────────────────────────────────────

/// A map from `StateId` to a scripted actor.
///
/// Build with `SimActorMap::new()` + `.add(state_id, actor)`.
#[derive(Default)]
pub struct SimActorMap {
    actors: HashMap<StateId, Arc<dyn AgentActor>>,
}

impl SimActorMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a scripted actor for the given state.
    pub fn add(mut self, state_id: impl Into<StateId>, actor: ScriptedAgentActor) -> Self {
        self.actors.insert(state_id.into(), Arc::new(actor));
        self
    }

    fn into_actors(self) -> HashMap<StateId, Arc<dyn AgentActor>> {
        self.actors
    }
}

// ── Simulation result ─────────────────────────────────────────────────────────

/// The result of a completed simulation run.
pub struct SimulationResult {
    pub status: RunStatus,
    pub events: Vec<RuntimeEvent>,
}

impl SimulationResult {
    /// Count events by payload kind name (for assertions).
    pub fn count_kind(&self, kind: &str) -> usize {
        self.events
            .iter()
            .filter(|e| {
                let s = format!("{:?}", e.payload);
                s.starts_with(kind)
            })
            .count()
    }

    /// Return true if any event payload matches the predicate.
    pub fn has_payload(&self, f: impl Fn(&RuntimeEventPayload) -> bool) -> bool {
        self.events.iter().any(|e| f(&e.payload))
    }

    /// Return all payloads.
    pub fn payloads(&self) -> Vec<&RuntimeEventPayload> {
        self.events.iter().map(|e| &e.payload).collect()
    }
}

// ── WorkflowSimulator ─────────────────────────────────────────────────────────

/// High-level simulation harness.
///
/// ```text
/// let result = WorkflowSimulator::new(compiled)
///     .with_actors(SimActorMap::new()
///         .add("analyze", ScriptedAgentActor::emit("analysis.done", json!({}))))
///     .inject("start.ready", serde_json::Value::Null)
///     .run()
///     .await;
///
/// assert_eq!(result.status, RunStatus::Completed);
/// ```
pub struct WorkflowSimulator {
    workflow: Arc<CompiledWorkflow>,
    actors: HashMap<StateId, Arc<dyn AgentActor>>,
    initial_events: Vec<(String, serde_json::Value)>,
    /// Maximum RTC steps before giving up.  Default 10 000.
    step_limit: usize,
}

impl WorkflowSimulator {
    pub fn new(workflow: Arc<CompiledWorkflow>) -> Self {
        Self {
            workflow,
            actors: HashMap::new(),
            initial_events: vec![],
            step_limit: 10_000,
        }
    }

    /// Set the scripted actor map.
    pub fn with_actors(mut self, map: SimActorMap) -> Self {
        self.actors = map.into_actors();
        self
    }

    /// Add an event that will be injected immediately after `start()`.
    pub fn inject(mut self, event_type: impl Into<String>, payload: serde_json::Value) -> Self {
        self.initial_events.push((event_type.into(), payload));
        self
    }

    /// Override the maximum step budget used by [`run`].
    ///
    /// The default is 10 000, which is sufficient for workflows with hundreds
    /// of micro-transitions. Raise this if your workflow is known to require
    /// more steps; lower it to fail-fast in tests that expect quick termination.
    pub fn with_step_limit(mut self, n: usize) -> Self {
        self.step_limit = n;
        self
    }

    /// Run the workflow to completion and return the result.
    ///
    /// The loop is bounded to the configured step limit (default 10 000) so
    /// that a stuck / blocked workflow returns `RunStatus::Running` rather than
    /// blocking forever.
    pub async fn run(self) -> Result<SimulationResult, crate::engine::EngineError> {
        let limit = self.step_limit;
        self.run_bounded(limit).await
    }

    /// Like `run()` but with an explicit step budget.
    pub async fn run_bounded(
        self,
        max_steps: usize,
    ) -> Result<SimulationResult, crate::engine::EngineError> {
        let sink = Arc::new(CapturingSink::default());
        let broker = build_sim_broker(sink.clone());
        let run_id = RunId::new(ulid::Ulid::generate().to_string());

        let mut instance =
            WorkflowInstance::new(run_id, self.workflow, broker, sink.clone(), self.actors);

        instance.start().await?;

        for (event_type, payload) in self.initial_events {
            instance.send(event_type, payload);
        }

        // Drive with a bounded step loop so stuck workflows don't deadlock.
        for _ in 0..max_steps {
            let done = instance.step().await?;
            if done {
                break;
            }
        }

        let status = instance.status.clone();
        let events = sink.drain().await;

        Ok(SimulationResult { status, events })
    }
}

pub(crate) fn build_sim_broker_pub(sink: Arc<CapturingSink>) -> Arc<CapabilityBroker> {
    build_sim_broker(sink)
}

fn build_sim_broker(sink: Arc<CapturingSink>) -> Arc<CapabilityBroker> {
    let llm: Arc<dyn LlmAdapter> = Arc::new(NoopLlm);
    let mcp: Arc<dyn McpAdapter> = Arc::new(NoopMcp);
    let memory: Arc<dyn MemoryAdapter> = Arc::new(NoopMemory);
    let secrets: Arc<dyn SecretsAdapter> = Arc::new(HostMapSecretsAdapter::empty());
    Arc::new(CapabilityBroker::new(llm, mcp, memory, secrets, sink))
}

// ── Fixture format ────────────────────────────────────────────────────────────

/// A test fixture that drives a simulation run.
///
/// Fixtures are serialised as JSON and can be stored alongside workflow
/// documents for regression testing and evaluation.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimFixture {
    /// Human-readable description.
    pub description: String,
    /// Events to inject after start (in order).
    pub inject: Vec<FixtureEvent>,
    /// Scripts for agentic states.
    pub scripts: Vec<FixtureScript>,
    /// Expected final status.
    pub expected_status: String,
    /// Payload kinds that MUST appear in the event log.
    pub expect_events: Vec<String>,
    /// Payload kinds that MUST NOT appear in the event log.
    pub reject_events: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct FixtureEvent {
    pub event_type: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct FixtureScript {
    pub state_id: String,
    pub emit_event_type: String,
    pub emit_payload: serde_json::Value,
}

impl SimFixture {
    /// Build a simulator from this fixture.
    pub fn into_simulator(self, workflow: Arc<CompiledWorkflow>) -> WorkflowSimulator {
        let mut map = SimActorMap::new();
        for script in self.scripts {
            map = map.add(
                script.state_id,
                ScriptedAgentActor::emit(script.emit_event_type, script.emit_payload),
            );
        }

        let mut sim = WorkflowSimulator::new(workflow).with_actors(map);
        for ev in self.inject {
            sim = sim.inject(ev.event_type, ev.payload);
        }
        sim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_model::{
        id::AgentId,
        state::{AgentRef, StateDefinition, StateType, TransitionSpec},
        validation::compile,
        workflow::WorkflowDocument,
    };

    fn minimal_doc_with_agent() -> WorkflowDocument {
        let mut agentic_on = std::collections::HashMap::new();
        agentic_on.insert(
            "task.done".into(),
            vec![TransitionSpec {
                target: "end".into(),
                guard: None,
                priority: 0,
                actions: vec![],
                kind: Default::default(),
            }],
        );

        WorkflowDocument {
            schema_version: "1.0.0".into(),
            id: "sim-test".into(),
            version: "0.1.0".into(),
            name: "Sim Test".into(),
            description: None,
            inputs: vec![],
            outputs: vec![],
            data_schema: Default::default(),
            policy: Default::default(),
            agents: vec![],
            states: vec![
                StateDefinition {
                    id: "work".into(),
                    name: "Work".into(),
                    state_type: StateType::Agentic,
                    agent: Some(AgentRef {
                        id: AgentId::new("sim-agent"),
                        version: langchart_model::id::AgentVersion::new("0.1.0"),
                    }),
                    prompt: Some("Do the work.".into()),
                    on: agentic_on,
                    input: Default::default(),
                    context: None,
                    model: None,
                    capabilities: None,
                    limits: None,
                    states: vec![],
                    regions: vec![],
                    completion: None,
                    history: None,
                    initial: None,
                    workflow_ref: None,
                    ports: None,
                    authorized_roles: vec![],
                    on_entry: vec![],
                    on_exit: vec![],
                    retry: None,
                    timeout: None,
                    output_schemas: Default::default(),
                    _editor: serde_json::Value::Null,
                },
                StateDefinition {
                    id: "end".into(),
                    name: "End".into(),
                    state_type: StateType::Final,
                    agent: None,
                    prompt: None,
                    on: Default::default(),
                    input: Default::default(),
                    context: None,
                    model: None,
                    capabilities: None,
                    limits: None,
                    states: vec![],
                    regions: vec![],
                    completion: None,
                    history: None,
                    initial: None,
                    workflow_ref: None,
                    ports: None,
                    authorized_roles: vec![],
                    on_entry: vec![],
                    on_exit: vec![],
                    retry: None,
                    timeout: None,
                    output_schemas: Default::default(),
                    _editor: serde_json::Value::Null,
                },
            ],
            initial: "work".into(),
            _editor: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn simulator_completes_with_scripted_actor() {
        let doc = minimal_doc_with_agent();
        let compiled = Arc::new(compile(doc).expect("compile"));

        let result = WorkflowSimulator::new(compiled)
            .with_actors(SimActorMap::new().add(
                "work",
                ScriptedAgentActor::emit("task.done", serde_json::json!({})),
            ))
            .run()
            .await
            .expect("run");

        assert_eq!(result.status, RunStatus::Completed);
        assert!(result.has_payload(|p| matches!(p, RuntimeEventPayload::ActivityCompleted { .. })));
        assert!(result.has_payload(|p| matches!(p, RuntimeEventPayload::RunCompleted)));
    }

    #[tokio::test]
    async fn simulator_failure_actor() {
        let doc = minimal_doc_with_agent();
        let compiled = Arc::new(compile(doc).expect("compile"));

        // A failing actor with no failure transition fails the run.
        let result = WorkflowSimulator::new(compiled)
            .with_actors(
                SimActorMap::new()
                    .add("work", ScriptedAgentActor::fail("intentional test failure")),
            )
            .run_bounded(100)
            .await
            .expect("run");

        assert_eq!(result.status, RunStatus::Failed);
        assert!(result.has_payload(|p| matches!(p, RuntimeEventPayload::ActivityFailed { .. })));
        assert!(result.has_payload(|p| matches!(p, RuntimeEventPayload::RunFailed { .. })));
    }

    #[tokio::test]
    async fn fixture_round_trip() {
        let fixture_json = r#"{
            "description": "basic simulation fixture",
            "inject": [],
            "scripts": [{"state_id":"work","emit_event_type":"task.done","emit_payload":{}}],
            "expected_status": "completed",
            "expect_events": ["RunCompleted"],
            "reject_events": ["RunFailed"]
        }"#;

        let fixture: SimFixture = serde_json::from_str(fixture_json).expect("parse fixture");
        let doc = minimal_doc_with_agent();
        let compiled = Arc::new(compile(doc).expect("compile"));

        let result = fixture.into_simulator(compiled).run().await.expect("run");

        assert_eq!(result.status, RunStatus::Completed);
    }
}
