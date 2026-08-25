//! Trace replay and fork-from-snapshot utilities.
//!
//! # `TraceReplayer`
//!
//! Takes a captured [`RuntimeEvent`] log (e.g. from a [`CapturingSink`]) and
//! replays the *causal* events (those that actually drove transitions) through a
//! fresh [`WorkflowInstance`]. This is useful for:
//!
//! - **Debugging** — reproduce a live run in a controlled environment.
//! - **Regression testing** — store a trace as a fixture, assert replayed
//!   behaviour matches.
//! - **Audit** — verify that a recorded trace can be reconstructed from
//!   scratch.
//!
//! Only [`RuntimeEventPayload::TransitionSelected`] events are replayed; all
//! other payload kinds are observational and are not injected.
//!
//! # `ForkRequest` / `RuntimeEngine::fork_from_snapshot`
//!
//! Creates a new run starting from a frozen [`RunSnapshot`] — the same active
//! state configuration — rather than from the workflow's initial state. Useful
//! for:
//!
//! - **What-if branching** — explore alternative continuations from a
//!   checkpoint.
//! - **Hot-reload** — resume a suspended run with a new actor implementation
//!   after a code change.
//!
//! [`CapturingSink`]: crate::simulation::CapturingSink

use crate::{
    broker::CapabilityBroker,
    engine::{EngineError, RunSnapshot},
    instance::AgentActor,
    run::{RunStatus, WorkflowInstance},
    simulation::CapturingSink,
};
use langchart_adapters::event::{EventSink, RuntimeEvent, RuntimeEventPayload};
use langchart_model::{
    id::{RunId, StateId},
    validation::CompiledWorkflow,
};
use std::{collections::HashMap, sync::Arc};
use tracing::debug;
use ulid::Ulid;

// ── TraceReplayer ─────────────────────────────────────────────────────────────

/// Replays the causal event sequence from a captured trace through a fresh
/// `WorkflowInstance`.
///
/// # Example
///
/// ```text
/// let replayed = TraceReplayer::new(compiled.clone(), trace)
///     .with_actors(actors)
///     .replay()
///     .await?;
///
/// assert_eq!(replayed.final_status, RunStatus::Completed);
/// ```
pub struct TraceReplayer {
    workflow: Arc<CompiledWorkflow>,
    trace: Vec<RuntimeEvent>,
    actors: HashMap<StateId, Arc<dyn AgentActor>>,
}

impl TraceReplayer {
    /// Create a new replayer for `workflow` using events from `trace`.
    pub fn new(workflow: Arc<CompiledWorkflow>, trace: Vec<RuntimeEvent>) -> Self {
        Self {
            workflow,
            trace,
            actors: HashMap::new(),
        }
    }

    /// Register actors for agentic states that may be re-entered during replay.
    pub fn with_actors(mut self, actors: HashMap<StateId, Arc<dyn AgentActor>>) -> Self {
        self.actors = actors;
        self
    }

    /// Run the replay and return a [`ReplayResult`].
    pub async fn replay(self) -> Result<ReplayResult, EngineError> {
        let sink = Arc::new(CapturingSink::default());
        let broker = build_noop_broker(sink.clone());
        let run_id = RunId::new(Ulid::generate().to_string());

        let mut instance = WorkflowInstance::new(
            run_id.clone(),
            self.workflow,
            broker,
            sink.clone(),
            self.actors,
        );

        instance.start().await?;

        // Extract transition events from the trace in order.
        let causal_events: Vec<(String, serde_json::Value)> = self
            .trace
            .iter()
            .filter_map(|e| {
                if let RuntimeEventPayload::TransitionSelected { event_type, .. } = &e.payload {
                    // Skip internal synthetic events produced by the runtime itself.
                    if event_type.starts_with("parallel.completed") {
                        return None;
                    }
                    Some((event_type.clone(), serde_json::Value::Null))
                } else {
                    None
                }
            })
            .collect();

        debug!(
            run = %run_id,
            events = causal_events.len(),
            "replaying trace"
        );

        for (event_type, payload) in causal_events {
            instance.send(event_type, payload);
        }

        // Step until done or 10 000 iterations.
        for _ in 0..10_000 {
            let done = instance.step().await?;
            if done {
                break;
            }
        }

        let final_status = instance.status.clone();
        let events = sink.drain().await;

        Ok(ReplayResult {
            final_status,
            events,
        })
    }
}

/// The result of a [`TraceReplayer::replay`] call.
pub struct ReplayResult {
    /// The status of the run at replay end.
    pub final_status: RunStatus,
    /// All events emitted during the replay.
    pub events: Vec<RuntimeEvent>,
}

impl ReplayResult {
    /// Return true if any emitted payload matches the predicate.
    pub fn has_payload(&self, f: impl Fn(&RuntimeEventPayload) -> bool) -> bool {
        self.events.iter().any(|e| f(&e.payload))
    }
}

// ── fork_from_snapshot ────────────────────────────────────────────────────────

/// Parameters for forking a run from an existing [`RunSnapshot`].
pub struct ForkRequest {
    /// The snapshot to fork from. Active states are restored into the new run.
    pub snapshot: RunSnapshot,
    /// Actors for agentic states.
    pub actors: HashMap<StateId, Arc<dyn AgentActor>>,
    /// Optional override event sink for the forked run. If `None`, the engine's
    /// shared sink is used.
    pub event_sink: Option<Arc<dyn EventSink>>,
}

impl ForkRequest {
    pub fn new(snapshot: RunSnapshot) -> Self {
        Self {
            snapshot,
            actors: HashMap::new(),
            event_sink: None,
        }
    }

    pub fn with_actors(mut self, actors: HashMap<StateId, Arc<dyn AgentActor>>) -> Self {
        self.actors = actors;
        self
    }

    pub fn with_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.event_sink = Some(sink);
        self
    }
}

/// Fork a workflow run from a snapshot without calling `start()`.
///
/// The new instance starts with the active state configuration copied from
/// `snapshot`, bypassing the initial-state entry logic. Activities for those
/// states are re-started immediately.
///
/// Returns the new `WorkflowInstance` *before* it is driven by a run task,
/// giving callers full control over how to schedule it.
pub async fn fork_instance(
    workflow: Arc<CompiledWorkflow>,
    broker: Arc<CapabilityBroker>,
    request: ForkRequest,
) -> Result<WorkflowInstance, EngineError> {
    let sink: Arc<dyn EventSink> = request
        .event_sink
        .unwrap_or_else(|| broker.event_sink_ref());

    let new_run_id = RunId::new(Ulid::generate().to_string());

    let mut instance =
        WorkflowInstance::new(new_run_id.clone(), workflow, broker, sink, request.actors);

    // Restore active states from the snapshot (skip re-entering — we're
    // forking from a mid-run point, not from the initial state).
    for state_id in &request.snapshot.active_states {
        instance.active_states.push(state_id.clone());
    }

    // Re-start activities for every active agentic state.
    let states: Vec<StateId> = instance.active_states.clone();
    for state_id in states {
        instance.start_activity_if_needed_pub(&state_id).await?;
    }

    Ok(instance)
}

// ── no-op broker for replay ───────────────────────────────────────────────────

fn build_noop_broker(sink: Arc<CapturingSink>) -> Arc<CapabilityBroker> {
    crate::simulation::build_sim_broker_pub(sink)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instance::ScriptedAgentActor,
        simulation::{SimActorMap, WorkflowSimulator},
    };
    use langchart_model::{
        id::AgentId,
        state::{AgentRef, StateDefinition, StateType, TransitionSpec},
        validation::compile,
        workflow::WorkflowDocument,
    };

    fn minimal_agentic_doc() -> WorkflowDocument {
        let mut on = std::collections::HashMap::new();
        on.insert(
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
            id: "replay-test".into(),
            version: "0.1.0".into(),
            name: "Replay Test".into(),
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
                        id: AgentId::new("replay-agent"),
                        version: langchart_model::id::AgentVersion::new("0.1.0"),
                    }),
                    prompt: Some("Do the work.".into()),
                    on,
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
    async fn trace_replayer_reproduces_completion() {
        let doc = minimal_agentic_doc();
        let compiled = Arc::new(compile(doc).expect("compile"));

        // First run — capture the trace.
        let original = WorkflowSimulator::new(compiled.clone())
            .with_actors(SimActorMap::new().add(
                "work",
                ScriptedAgentActor::emit("task.done", serde_json::json!({})),
            ))
            .run()
            .await
            .expect("original run");

        assert_eq!(original.status, RunStatus::Completed);

        // Replay needs the same actor so the agentic state can be re-entered.
        let mut actors: HashMap<StateId, Arc<dyn AgentActor>> = HashMap::new();
        actors.insert(
            StateId::new("work"),
            Arc::new(ScriptedAgentActor::emit("task.done", serde_json::json!({}))),
        );

        // Now replay the captured trace.
        let replayed = TraceReplayer::new(compiled, original.events)
            .with_actors(actors)
            .replay()
            .await
            .expect("replay");

        assert_eq!(replayed.final_status, RunStatus::Completed);
        assert!(replayed.has_payload(|p| matches!(p, RuntimeEventPayload::RunCompleted)));
    }
}
