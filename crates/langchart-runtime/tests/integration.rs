//! Integration tests for `WorkflowInstance`.
//!
//! Uses `ScriptedAgentActor` as a deterministic stand-in for a real LLM agent
//! so the entire test suite is model-free and completes in milliseconds.
//!
//! # Scenarios
//!
//! 1. `atomic_to_final`          — two atomic states; manual event → completes
//! 2. `agentic_to_final`         — atomic → agentic (ScriptedActor) → final
//! 3. `guard_blocks_transition`  — CEL guard that should NOT fire
//! 4. `guard_passes_transition`  — CEL guard with matching payload fires
//! 5. `actor_failure`            — actor returns Err → ActivityFailed event emitted
//! 6. `suspend_then_cancel`      — suspend then cancel, run never completes

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use langchart_adapters::{
    event::{EventSink, EventSinkError, RuntimeEvent, RuntimeEventPayload},
    llm::{LlmAdapter, LlmError, LlmRequest, LlmResponse},
    mcp::{McpAdapter, McpCredential, McpError, ResourceContent, ToolDefinition as McpToolDef},
    memory::{MemoryAdapter, MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult},
    secrets::{HostMapSecretsAdapter, SecretsAdapter},
};
use langchart_model::{
    id::{AgentId, AgentVersion, IdempotencyKey, RunId, ServerId, StateId, ToolName},
    policy::ModelPolicy,
    state::{AgentRef, StateDefinition, StateType, TransitionKind, TransitionSpec},
    validation::compile,
    workflow::{AgentDefinition, WorkflowDocument},
};
use langchart_runtime::{
    RuntimeEngine,
    broker::CapabilityBroker,
    engine::EngineAdapters,
    instance::{AgentActor, ScriptedAgentActor},
    run::{RunStatus, WorkflowInstance},
};
use tokio::sync::Mutex;

// ── No-op adapter stubs ───────────────────────────────────────────────────────

struct NoopLlm;

#[async_trait]
impl LlmAdapter for NoopLlm {
    async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse, LlmError> {
        Err(LlmError::Provider("noop".into()))
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
        _credentials: &[McpCredential],
        _key: Option<&IdempotencyKey>,
    ) -> Result<serde_json::Value, McpError> {
        Err(McpError::Call("noop".into()))
    }

    async fn list_tools(&self, _server_id: &ServerId) -> Result<Vec<McpToolDef>, McpError> {
        Ok(vec![])
    }

    async fn read_resource(
        &self,
        _server_id: &ServerId,
        _uri: &str,
    ) -> Result<ResourceContent, McpError> {
        Err(McpError::Call("noop".into()))
    }
}

struct NoopMemory;

#[async_trait]
impl MemoryAdapter for NoopMemory {
    async fn store(&self, _r: MemoryRecord) -> Result<MemoryId, MemoryError> {
        Ok(MemoryId("noop".into()))
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

// ── Event recording sink ──────────────────────────────────────────────────────

#[derive(Default)]
struct VecSink {
    events: Mutex<Vec<RuntimeEventPayload>>,
}

#[async_trait]
impl EventSink for VecSink {
    async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
        self.events.lock().await.push(event.payload);
        Ok(())
    }
}

impl VecSink {
    async fn payloads(&self) -> Vec<RuntimeEventPayload> {
        self.events.lock().await.clone()
    }
}

// ── Broker factory ────────────────────────────────────────────────────────────

/// Build a `CapabilityBroker` backed by no-op stubs. Sufficient for scripted tests.
fn bare_broker(sink: Arc<dyn EventSink>) -> Arc<CapabilityBroker> {
    let llm: Arc<dyn LlmAdapter> = Arc::new(NoopLlm);
    let mcp: Arc<dyn McpAdapter> = Arc::new(NoopMcp);
    let memory: Arc<dyn MemoryAdapter> = Arc::new(NoopMemory);
    let secrets: Arc<dyn SecretsAdapter> = Arc::new(HostMapSecretsAdapter::empty());
    Arc::new(CapabilityBroker::new(llm, mcp, memory, secrets, sink))
}

// ── Workflow builders ─────────────────────────────────────────────────────────

fn base_doc(id: &str, states: Vec<StateDefinition>, initial: &str) -> WorkflowDocument {
    WorkflowDocument {
        schema_version: "1.0.0".into(),
        id: id.into(),
        version: "0.1.0".into(),
        name: id.into(),
        description: None,
        inputs: vec![],
        outputs: vec![],
        data_schema: Default::default(),
        policy: Default::default(),
        agents: vec![],
        states,
        initial: initial.into(),
        _editor: serde_json::Value::Null,
    }
}

fn leaf_state(id: &str, state_type: StateType) -> StateDefinition {
    StateDefinition {
        id: id.into(),
        name: id.into(),
        state_type,
        agent: None,
        prompt: None,
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
        on: Default::default(),
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    }
}

fn with_transition(
    mut state: StateDefinition,
    event: &str,
    target: &str,
    guard: Option<&str>,
) -> StateDefinition {
    state.on.insert(
        event.into(),
        vec![TransitionSpec {
            target: target.into(),
            guard: guard.map(Into::into),
            priority: 0,
            actions: vec![],
            kind: Default::default(),
        }],
    );
    state
}

fn agentic_state(id: &str, output_event: &str, next_state: &str) -> StateDefinition {
    let mut s = leaf_state(id, StateType::Agentic);
    s.agent = Some(AgentRef {
        id: AgentId::new("test-agent"),
        version: AgentVersion::new("0.1.0"),
    });
    s.on.insert(
        output_event.into(),
        vec![TransitionSpec {
            target: next_state.into(),
            guard: None,
            priority: 0,
            actions: vec![],
            kind: Default::default(),
        }],
    );
    s
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// 1. Two atomic states; caller sends one event → run completes.
#[tokio::test]
async fn atomic_to_final() {
    let states = vec![
        with_transition(leaf_state("idle", StateType::Atomic), "done", "end", None),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-atomic", states, "idle");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r1"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("done", serde_json::Value::Null);
    let status = inst.run_to_completion().await.expect("run");

    assert_eq!(status, RunStatus::Completed);

    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunStarted))
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted))
    );
}

#[tokio::test]
async fn idle_step_remains_non_blocking_for_simulation() {
    let states = vec![leaf_state("idle", StateType::Atomic)];
    let doc = base_doc("wf-idle-wait", states, "idle");
    let compiled = Arc::new(compile(doc).expect("compile"));
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-idle-wait"),
        compiled,
        broker,
        sink,
        HashMap::new(),
    );
    inst.start().await.expect("start");

    assert!(!inst.step().await.expect("step"));
}

/// 2. Atomic → Agentic (ScriptedActor) → Final. Full agent lifecycle exercised.
#[tokio::test]
async fn agentic_to_final() {
    let states = vec![
        with_transition(
            leaf_state("prepare", StateType::Atomic),
            "prepare.done",
            "analyze",
            None,
        ),
        agentic_state("analyze", "analysis.completed", "end"),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-agentic", states, "prepare");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let actor: Arc<dyn AgentActor> = Arc::new(ScriptedAgentActor::emit(
        "analysis.completed",
        serde_json::json!({ "confidence": 0.95 }),
    ));
    let actors = HashMap::from([(StateId::new("analyze"), actor)]);

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(RunId::new("r2"), compiled, broker, sink.clone(), actors);

    inst.start().await.expect("start");
    inst.send("prepare.done", serde_json::Value::Null);
    let status = inst.run_to_completion().await.expect("run");

    assert_eq!(status, RunStatus::Completed);

    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::ActivityStarted { .. }))
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::ActivityCompleted { .. }))
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted))
    );
}

#[tokio::test]
async fn queued_activity_event_is_ignored_after_its_state_exits() {
    let mut work = agentic_state("work", "agent.done", "correct");
    work = with_transition(work, "leave", "other", None);
    let other = with_transition(
        leaf_state("other", StateType::Atomic),
        "agent.done",
        "wrong",
        None,
    );
    let doc = base_doc(
        "wf-stale-queued-activity",
        vec![
            work,
            other,
            leaf_state("correct", StateType::Final),
            leaf_state("wrong", StateType::Final),
        ],
        "work",
    );
    let compiled = Arc::new(compile(doc).expect("compile"));
    let actor: Arc<dyn AgentActor> = Arc::new(ScriptedAgentActor::emit(
        "agent.done",
        serde_json::json!({}),
    ));
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-stale-queued-activity"),
        compiled,
        broker,
        sink,
        HashMap::from([(StateId::new("work"), actor)]),
    );

    inst.start().await.expect("start");
    inst.send("leave", serde_json::Value::Null);
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    inst.step().await.expect("process leave");
    assert!(inst.active_states.contains(&StateId::new("other")));

    inst.step().await.expect("discard stale activity event");
    assert_eq!(inst.status, RunStatus::Running);
    assert!(inst.active_states.contains(&StateId::new("other")));
    assert!(!inst.active_states.contains(&StateId::new("wrong")));
}

#[tokio::test]
async fn broker_rejects_a_forged_capability_envelope() {
    use langchart_model::policy::{CapabilityPolicy, McpServerPolicy, OperationClass};
    use langchart_runtime::broker::{BrokerError, CapabilityEnvelope};

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let server = ServerId::new("vault");
    let envelope = CapabilityEnvelope::new(
        CapabilityPolicy {
            mcp: HashMap::from([(
                server.clone(),
                McpServerPolicy {
                    resource_patterns: vec!["vault://docs/*".into()],
                    operations: vec![OperationClass::Read],
                    ..Default::default()
                },
            )]),
            ..Default::default()
        },
        RunId::new("r-resource-policy"),
        langchart_model::id::InvocationId::new("inv-resource-policy"),
        StateId::new("work"),
        1,
        1,
    );

    let error = broker
        .read_resource(&envelope, &server, "vault://private/secret.md")
        .await
        .expect_err("publicly constructed envelopes must not authorize broker calls");
    assert!(matches!(error, BrokerError::InvalidCapabilityEnvelope));
}

/// 3. CEL guard blocks the transition when the payload field is false.
#[tokio::test]
async fn guard_blocks_transition() {
    let states = vec![
        with_transition(
            leaf_state("idle", StateType::Atomic),
            "done",
            "end",
            Some("approved == true"),
        ),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-guard-block", states, "idle");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r3"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("done", serde_json::json!({ "approved": false }));

    // Drive one step — guard should block, run stays Running.
    let terminal = inst.step().await.expect("step");
    assert!(!terminal, "run should still be running after blocked guard");

    let payloads = sink.payloads().await;
    assert!(
        !payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::TransitionSelected { .. })),
        "no transition should have fired"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::EventUnhandled { .. })),
        "EventUnhandled should have been emitted"
    );
}

/// 4. CEL guard passes when the payload matches.
#[tokio::test]
async fn guard_passes_transition() {
    let states = vec![
        with_transition(
            leaf_state("idle", StateType::Atomic),
            "done",
            "end",
            Some("approved == true"),
        ),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-guard-pass", states, "idle");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r4"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("done", serde_json::json!({ "approved": true }));
    let status = inst.run_to_completion().await.expect("run");

    assert_eq!(status, RunStatus::Completed);
    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::TransitionSelected { .. }))
    );
}

/// The runtime executes the guard program captured during workflow compilation,
/// rather than recompiling the mutable source document at transition time.
#[tokio::test]
async fn runtime_uses_precompiled_guard() {
    let states = vec![
        with_transition(
            leaf_state("idle", StateType::Atomic),
            "done",
            "end",
            Some("approved == true"),
        ),
        leaf_state("end", StateType::Final),
    ];
    let mut compiled = compile(base_doc("wf-compiled-guard", states, "idle")).expect("compile");
    compiled.document.states[0].on.get_mut("done").unwrap()[0].guard = Some("false".into());

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut instance = WorkflowInstance::new(
        RunId::new("r-compiled-guard"),
        Arc::new(compiled),
        broker,
        sink,
        HashMap::new(),
    );
    instance.start().await.expect("start");
    instance.send("done", serde_json::json!({ "approved": true }));

    assert_eq!(
        instance.run_to_completion().await.expect("run"),
        RunStatus::Completed
    );
}

/// 5. Actor that fails produces an `ActivityFailed` observable event.
#[tokio::test]
async fn actor_failure_produces_activity_failed() {
    // Workflow: start immediately in the agentic state.
    let states = vec![
        agentic_state("process", "process.done", "end"),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-fail", states, "process");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let actor: Arc<dyn AgentActor> = Arc::new(ScriptedAgentActor::fail("simulated failure"));
    let actors = HashMap::from([(StateId::new("process"), actor)]);

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(RunId::new("r5"), compiled, broker, sink.clone(), actors);

    inst.start().await.expect("start");

    // Drive several steps to allow the spawned task to complete and its result
    // to be flushed back through the activity channel.
    for _ in 0..30 {
        let _ = inst.step().await.expect("step");
        tokio::task::yield_now().await;
    }

    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::ActivityFailed { .. })),
        "expected ActivityFailed in events;\ngot:\n{payloads:#?}"
    );
    assert_eq!(inst.status, RunStatus::Failed);
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunFailed { .. })),
        "unhandled activity failure must terminate the run"
    );
}

/// 6. Run can be suspended then cancelled; it never reaches Completed.
#[tokio::test]
async fn suspend_then_cancel() {
    let states = vec![
        with_transition(leaf_state("idle", StateType::Atomic), "done", "end", None),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-suspend-cancel", states, "idle");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r6"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.suspend().await.expect("suspend");
    assert_eq!(inst.status, RunStatus::Suspended);

    inst.cancel().await.expect("cancel");
    assert_eq!(inst.status, RunStatus::Cancelled);

    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunSuspended))
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCancelled))
    );
    assert!(
        !payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted))
    );
}

// ── Phase 4 tests ─────────────────────────────────────────────────────────────

/// 7. Parallel state with `All` completion: both regions must finish.
///
/// Topology:
///   [start] --start.ready--> [parallel_work] (parallel, All)
///     region_a: [task_a] --(task_a.done)--> [final_a] (final)
///     region_b: [task_b] --(task_b.done)--> [final_b] (final)
///   parallel_work --parallel.completed--> [done] (final)
#[tokio::test]
async fn parallel_all_completion() {
    use langchart_model::{
        id::RegionId,
        state::{ParallelCompletion, ParallelRegion},
    };

    let region_a = ParallelRegion {
        id: RegionId::new("region_a"),
        name: "Region A".into(),
        initial: StateId::new("task_a"),
        states: vec![
            with_transition(
                leaf_state("task_a", StateType::Atomic),
                "task_a.done",
                "final_a",
                None,
            ),
            leaf_state("final_a", StateType::Final),
        ],
    };

    let region_b = ParallelRegion {
        id: RegionId::new("region_b"),
        name: "Region B".into(),
        initial: StateId::new("task_b"),
        states: vec![
            with_transition(
                leaf_state("task_b", StateType::Atomic),
                "task_b.done",
                "final_b",
                None,
            ),
            leaf_state("final_b", StateType::Final),
        ],
    };

    // Build parallel state.
    let parallel = StateDefinition {
        id: "parallel_work".into(),
        name: "Parallel Work".into(),
        state_type: StateType::Parallel,
        agent: None,
        prompt: None,
        input: Default::default(),
        context: None,
        model: None,
        capabilities: None,
        limits: None,
        states: vec![],
        regions: vec![region_a, region_b],
        completion: Some(ParallelCompletion::All),
        history: None,
        initial: None,
        workflow_ref: None,
        ports: None,
        authorized_roles: vec![],
        on_entry: vec![],
        on_exit: vec![],
        retry: None,
        timeout: None,
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "parallel.completed".into(),
                vec![TransitionSpec {
                    target: "done".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m
        },
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    };

    let states = vec![
        with_transition(
            leaf_state("start", StateType::Atomic),
            "start.ready",
            "parallel_work",
            None,
        ),
        parallel,
        leaf_state("done", StateType::Final),
    ];

    let doc = base_doc("wf-parallel-all", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r7"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("start.ready", serde_json::Value::Null);
    // Only one region done → should NOT complete yet.
    inst.send("task_a.done", serde_json::Value::Null);
    // Second region done → should complete.
    inst.send("task_b.done", serde_json::Value::Null);

    let status = inst.run_to_completion().await.expect("run");
    assert_eq!(status, RunStatus::Completed, "expected Completed");

    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::ParallelRegionEntered { .. })),
        "expected ParallelRegionEntered"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::ParallelCompleted { .. })),
        "expected ParallelCompleted"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted)),
        "expected RunCompleted"
    );
}

/// 8. Parallel state with `Any` completion: first region to finish wins.
#[tokio::test]
async fn parallel_any_completion() {
    use langchart_model::{
        id::RegionId,
        state::{ParallelCompletion, ParallelRegion},
    };

    let region_a = ParallelRegion {
        id: RegionId::new("region_a"),
        name: "Region A".into(),
        initial: StateId::new("task_a2"),
        states: vec![
            with_transition(
                leaf_state("task_a2", StateType::Atomic),
                "task_a2.done",
                "final_a2",
                None,
            ),
            leaf_state("final_a2", StateType::Final),
        ],
    };

    let region_b = ParallelRegion {
        id: RegionId::new("region_b"),
        name: "Region B".into(),
        initial: StateId::new("task_b2"),
        states: vec![
            // task_b2 has NO transitions — would never complete on its own.
            leaf_state("task_b2", StateType::Atomic),
            leaf_state("final_b2", StateType::Final),
        ],
    };

    let parallel = StateDefinition {
        id: "par_any".into(),
        name: "Par Any".into(),
        state_type: StateType::Parallel,
        completion: Some(ParallelCompletion::Any),
        regions: vec![region_a, region_b],
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "parallel.completed".into(),
                vec![TransitionSpec {
                    target: "done2".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m
        },
        agent: None,
        prompt: None,
        input: Default::default(),
        context: None,
        model: None,
        capabilities: None,
        limits: None,
        states: vec![],
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
    };

    let states = vec![
        with_transition(leaf_state("s0", StateType::Atomic), "go", "par_any", None),
        parallel,
        leaf_state("done2", StateType::Final),
    ];
    let doc = base_doc("wf-parallel-any", states, "s0");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r8"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("go", serde_json::Value::Null);
    // Only region_a fires — `Any` mode means this is enough.
    inst.send("task_a2.done", serde_json::Value::Null);

    let status = inst.run_to_completion().await.expect("run");
    assert_eq!(
        status,
        RunStatus::Completed,
        "parallel Any should complete after one region"
    );
}

/// 9. Subworkflow stub: emits `SubworkflowFailed` and the workflow
///    handles it via `subworkflow.failed` transition.
#[tokio::test]
async fn subworkflow_failure_handled() {
    use langchart_model::state::StateDefinition;

    let subwf_state = StateDefinition {
        id: "child_wf".into(),
        name: "Child".into(),
        state_type: StateType::Subworkflow,
        workflow_ref: Some("some-workflow@1.0".into()),
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "subworkflow.failed".into(),
                vec![TransitionSpec {
                    target: "recovered".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m
        },
        agent: None,
        prompt: None,
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
        ports: None,
        authorized_roles: vec![],
        on_entry: vec![],
        on_exit: vec![],
        retry: None,
        timeout: None,
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    };

    let states = vec![subwf_state, leaf_state("recovered", StateType::Final)];
    let doc = base_doc("wf-subwf-fail", states, "child_wf");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r9"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");

    // Drive until the spawned subworkflow placeholder emits its failure.
    for _ in 0..30 {
        let terminal = inst.step().await.expect("step");
        tokio::task::yield_now().await;
        if terminal {
            break;
        }
    }

    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "subworkflow failure should transition to recovered (final)"
    );

    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::SubworkflowStarted { .. })),
        "expected SubworkflowStarted"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::SubworkflowFailed { .. })),
        "expected SubworkflowFailed"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted)),
        "expected RunCompleted after recovery"
    );
}

// ── B1: on_entry / on_exit action tests ──────────────────────────────────────

/// 10. on_entry action fires when a state is entered; on_exit fires on exit.
#[tokio::test]
async fn on_entry_and_on_exit_actions_fire() {
    use langchart_runtime::instance::{ActionContext, ActionError, ActionRegistry, StateAction};
    use std::sync::atomic::{AtomicU32, Ordering};

    // Shared counters to verify each action fires exactly once.
    let entry_count = Arc::new(AtomicU32::new(0));
    let exit_count = Arc::new(AtomicU32::new(0));

    struct CountAction(Arc<AtomicU32>);
    #[async_trait]
    impl StateAction for CountAction {
        async fn run(
            &self,
            _ctx: ActionContext,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<(), ActionError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());

    // Build a simple atomic → final workflow where the atomic state declares
    // on_entry and on_exit actions.
    let mut atomic = leaf_state("start", StateType::Atomic);
    atomic.on_entry = vec!["count_entry".into()];
    atomic.on_exit = vec!["count_exit".into()];
    let atomic = with_transition(atomic, "go", "end", None);

    let doc = base_doc(
        "action-test",
        vec![atomic, leaf_state("end", StateType::Final)],
        "start",
    );
    let compiled = Arc::new(compile(doc).unwrap());

    let registry = ActionRegistry::new()
        .register("count_entry", CountAction(entry_count.clone()))
        .register("count_exit", CountAction(exit_count.clone()))
        .into_map();

    let mut inst = WorkflowInstance::with_actions(
        RunId::new("r-action"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
        registry,
    );
    inst.start().await.unwrap();

    // on_entry for "start" must have fired during start().
    assert_eq!(
        entry_count.load(Ordering::SeqCst),
        1,
        "on_entry should fire on state entry"
    );
    assert_eq!(
        exit_count.load(Ordering::SeqCst),
        0,
        "on_exit should NOT fire yet"
    );

    inst.send("go", serde_json::Value::Null);
    // Drive one step — processes the "go" event, exits "start", enters "end".
    for _ in 0..50 {
        if inst.step().await.unwrap() {
            break;
        }
    }

    assert_eq!(
        exit_count.load(Ordering::SeqCst),
        1,
        "on_exit should fire when state exits"
    );
    assert_eq!(inst.status, RunStatus::Completed);

    let payloads = sink.payloads().await;
    assert!(payloads.iter().any(|p| matches!(p, RuntimeEventPayload::ActionStarted { action_id, .. } if action_id == "count_entry")));
    assert!(payloads.iter().any(|p| matches!(p, RuntimeEventPayload::ActionCompleted { action_id, .. } if action_id == "count_exit")));
}

/// 11. on_entry action that fails emits ActionFailed but does NOT abort the run.
#[tokio::test]
async fn on_entry_action_failure_does_not_abort_run() {
    use langchart_runtime::instance::{ActionContext, ActionError, ActionRegistry, StateAction};

    struct FailingAction;
    #[async_trait]
    impl StateAction for FailingAction {
        async fn run(
            &self,
            ctx: ActionContext,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<(), ActionError> {
            Err(ActionError {
                action_id: ctx.action_id.clone(),
                message: "deliberate failure".into(),
            })
        }
    }

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());

    let mut atomic = leaf_state("start", StateType::Atomic);
    atomic.on_entry = vec!["will_fail".into()];
    let atomic = with_transition(atomic, "go", "end", None);

    let doc = base_doc(
        "action-fail-test",
        vec![atomic, leaf_state("end", StateType::Final)],
        "start",
    );
    let compiled = Arc::new(compile(doc).unwrap());

    let registry = ActionRegistry::new()
        .register("will_fail", FailingAction)
        .into_map();

    let mut inst = WorkflowInstance::with_actions(
        RunId::new("r-action-fail"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
        registry,
    );
    inst.start().await.unwrap();
    inst.send("go", serde_json::Value::Null);
    for _ in 0..50 {
        if inst.step().await.unwrap() {
            break;
        }
    }

    // Run still completes despite the action failure.
    assert_eq!(inst.status, RunStatus::Completed);

    let payloads = sink.payloads().await;
    assert!(
        payloads.iter().any(|p| matches!(p, RuntimeEventPayload::ActionFailed { action_id, .. } if action_id == "will_fail")),
        "expected ActionFailed event"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted))
    );
}

// ── B2: Retry policy tests ────────────────────────────────────────────────────

/// A test actor that fails a configurable number of times before succeeding.
/// Uses an atomic counter so the `Arc<dyn AgentActor>` can be cloned freely.
struct FailThenSucceedActor {
    /// How many times to fail before emitting a success event.
    fail_count: u32,
    calls: Arc<std::sync::atomic::AtomicU32>,
    success_event: String,
}

impl FailThenSucceedActor {
    fn new(fail_count: u32, success_event: impl Into<String>) -> Self {
        Self {
            fail_count,
            calls: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            success_event: success_event.into(),
        }
    }
}

#[async_trait]
impl AgentActor for FailThenSucceedActor {
    async fn run(
        &self,
        _inv: langchart_runtime::instance::AgentInvocation,
        _env: langchart_runtime::broker::CapabilityEnvelope,
        _broker: Arc<CapabilityBroker>,
    ) -> Result<
        langchart_runtime::instance::AgentOutputEvent,
        langchart_runtime::instance::AgentError,
    > {
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n < self.fail_count {
            Err(langchart_runtime::instance::AgentError::Internal(format!(
                "deliberate failure #{}",
                n + 1
            )))
        } else {
            Ok(langchart_runtime::instance::AgentOutputEvent {
                event_type: self.success_event.clone(),
                payload: serde_json::json!({ "attempt": n + 1 }),
            })
        }
    }
}

/// 12. Retry policy: actor fails twice, succeeds on third attempt.
///     - max_attempts = 3, Fixed delay = 0s (instant retries in tests).
///     - Expects: 2 × ActivityRetried + 1 × ActivityCompleted + RunCompleted.
#[tokio::test]
async fn retry_succeeds_on_third_attempt() {
    use langchart_model::policy::{BackoffStrategy, RetryPolicy};

    let actor: Arc<dyn AgentActor> = Arc::new(FailThenSucceedActor::new(2, "work.done"));

    // Build state: agentic with retry(max=3, delay=0) → final on success.
    let mut work = leaf_state("work", StateType::Agentic);
    work.agent = Some(AgentRef {
        id: AgentId::new("test-agent"),
        version: AgentVersion::new("0.1.0"),
    });
    work.retry = Some(RetryPolicy {
        max_attempts: 3,
        delay: std::time::Duration::ZERO,
        backoff: BackoffStrategy::Fixed,
        retryable_on: vec!["internal".into()],
        fallback_model: None,
        on_exhausted: None,
    });
    work.on.insert(
        "work.done".into(),
        vec![TransitionSpec {
            target: "end".into(),
            guard: None,
            priority: 0,
            actions: vec![],
            kind: Default::default(),
        }],
    );

    let states = vec![work, leaf_state("end", StateType::Final)];
    let doc = base_doc("wf-retry-ok", states, "work");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let actors = HashMap::from([(StateId::new("work"), actor)]);
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-retry-ok"),
        compiled,
        broker,
        sink.clone(),
        actors,
    );

    inst.start().await.expect("start");

    // Drive steps — retry sleeps 0s so they resolve quickly.
    for _ in 0..100 {
        let terminal = inst.step().await.expect("step");
        tokio::task::yield_now().await;
        if terminal {
            break;
        }
    }

    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "should complete after successful retry"
    );

    let payloads = sink.payloads().await;

    let retry_count = payloads
        .iter()
        .filter(|p| matches!(p, RuntimeEventPayload::ActivityRetried { .. }))
        .count();
    assert_eq!(retry_count, 2, "expected exactly 2 ActivityRetried events");

    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::ActivityCompleted { .. })),
        "expected ActivityCompleted after retries succeed",
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted)),
        "expected RunCompleted",
    );
}

#[tokio::test(start_paused = true)]
async fn exiting_a_state_cancels_its_delayed_retry() {
    use langchart_model::policy::{BackoffStrategy, RetryPolicy};

    let actor = Arc::new(FailThenSucceedActor::new(u32::MAX, "never"));
    let calls = actor.calls.clone();
    let actor_for_runtime: Arc<dyn AgentActor> = actor;
    let mut work = leaf_state("work", StateType::Agentic);
    work.agent = Some(AgentRef {
        id: AgentId::new("test-agent"),
        version: AgentVersion::new("0.1.0"),
    });
    work.retry = Some(RetryPolicy {
        max_attempts: 3,
        delay: std::time::Duration::from_secs(60),
        backoff: BackoffStrategy::Fixed,
        retryable_on: vec!["internal".into()],
        fallback_model: None,
        on_exhausted: None,
    });
    work = with_transition(work, "leave", "parked", None);
    let doc = base_doc(
        "wf-retry-cancelled-on-exit",
        vec![work, leaf_state("parked", StateType::Atomic)],
        "work",
    );
    let compiled = Arc::new(compile(doc).expect("compile"));
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-retry-cancelled-on-exit"),
        compiled,
        broker,
        sink.clone(),
        HashMap::from([(StateId::new("work"), actor_for_runtime)]),
    );
    inst.start().await.expect("start");

    for _ in 0..10 {
        inst.step().await.expect("step");
        tokio::task::yield_now().await;
        if sink
            .payloads()
            .await
            .iter()
            .any(|payload| matches!(payload, RuntimeEventPayload::ActivityRetried { .. }))
        {
            break;
        }
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    inst.send("leave", serde_json::Value::Null);
    inst.step().await.expect("leave step");
    assert!(inst.active_states.contains(&StateId::new("parked")));

    tokio::time::advance(std::time::Duration::from_secs(61)).await;
    tokio::task::yield_now().await;
    for _ in 0..3 {
        inst.step().await.expect("post-retry step");
    }
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the exited state's retry must not restart its actor"
    );
}

/// 13. Retry exhaustion: actor always fails; `on_exhausted` target transitions to a
///     `failed_state` (Final). Expects: (max-1) × ActivityRetried + RunCompleted via
///     the exhausted path.
#[tokio::test]
async fn retry_exhausted_transitions_to_on_exhausted_state() {
    use langchart_model::policy::{BackoffStrategy, RetryPolicy};

    // Actor always fails.
    let actor: Arc<dyn AgentActor> = Arc::new(ScriptedAgentActor::fail("always fails"));

    let mut work = leaf_state("work", StateType::Agentic);
    work.agent = Some(AgentRef {
        id: AgentId::new("test-agent"),
        version: AgentVersion::new("0.1.0"),
    });
    work.retry = Some(RetryPolicy {
        max_attempts: 2,
        delay: std::time::Duration::ZERO,
        backoff: BackoffStrategy::Fixed,
        retryable_on: vec![],
        fallback_model: None,
        on_exhausted: Some("failed_state".into()),
    });

    let states = vec![work, leaf_state("failed_state", StateType::Final)];
    let doc = base_doc("wf-retry-exhausted", states, "work");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let actors = HashMap::from([(StateId::new("work"), actor)]);
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-retry-exh"),
        compiled,
        broker,
        sink.clone(),
        actors,
    );

    inst.start().await.expect("start");

    for _ in 0..100 {
        let terminal = inst.step().await.expect("step");
        tokio::task::yield_now().await;
        if terminal {
            break;
        }
    }

    // `on_exhausted` = "failed_state" (Final) → run should be Completed.
    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "exhausted path via Final state should complete"
    );

    let payloads = sink.payloads().await;

    // 2 attempts total: 1 initial + 1 retry → 1 ActivityRetried.
    let retry_count = payloads
        .iter()
        .filter(|p| matches!(p, RuntimeEventPayload::ActivityRetried { .. }))
        .count();
    assert_eq!(
        retry_count, 1,
        "max_attempts=2 → exactly 1 retry before exhaustion"
    );

    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted)),
        "expected RunCompleted via on_exhausted Final state",
    );
}

// ── B3: Subworkflow resolution tests ─────────────────────────────────────────

/// 14. A subworkflow state resolves its child from an `InMemoryWorkflowRepository`
///     and the parent transitions via `subworkflow.completed` to a Final state.
#[tokio::test]
async fn subworkflow_resolves_and_completes() {
    use langchart_adapters::workflow_repository::{InMemoryWorkflowRepository, WorkflowRepository};

    // Build the child workflow: a trivially self-completing atomic→final doc.
    let child_states = vec![
        with_transition(
            leaf_state("child_step", StateType::Atomic),
            "proceed",
            "child_end",
            None,
        ),
        leaf_state("child_end", StateType::Final),
    ];
    let child_doc = base_doc("child-wf", child_states, "child_step");
    let child_compiled = Arc::new(compile(child_doc).expect("compile child"));

    // Register in a repository (not used directly — superseded by repo2 below).
    let _repo: Arc<dyn WorkflowRepository> =
        Arc::new(InMemoryWorkflowRepository::new().register("child-wf@1.0", child_compiled));

    // Parent workflow: enter subworkflow state, complete via subworkflow.completed.
    let subwf_state = StateDefinition {
        id: "run_child".into(),
        name: "Run Child".into(),
        state_type: StateType::Subworkflow,
        workflow_ref: Some("child-wf@1.0".into()),
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "subworkflow.completed".into(),
                vec![TransitionSpec {
                    target: "done".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m
        },
        agent: None,
        prompt: None,
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
        ports: None,
        authorized_roles: vec![],
        on_entry: vec![],
        on_exit: vec![],
        retry: None,
        timeout: None,
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    };

    // NOTE: the child workflow is self-contained but needs an event to drive
    // "child_step" → "child_end". We pre-send "proceed" to the child via the
    // child's own run loop, which happens inside the spawned task. Since the
    // child is created with no external events, it will sit in Running state.
    // We need to inject "proceed" into the child. The easiest test approach is
    // to use a child workflow whose initial state is directly a Final state.
    //
    // Rebuild child as: initial = "child_end" (Final) → completes immediately.
    let child_states_imm = vec![leaf_state("child_end_imm", StateType::Final)];
    let child_doc_imm = base_doc("child-wf-imm", child_states_imm, "child_end_imm");
    let child_compiled_imm = Arc::new(compile(child_doc_imm).expect("compile child imm"));

    let repo2: Arc<dyn WorkflowRepository> =
        Arc::new(InMemoryWorkflowRepository::new().register("child-wf@1.0", child_compiled_imm));

    let parent_states = vec![subwf_state, leaf_state("done", StateType::Final)];
    let parent_doc = base_doc("wf-b3-subwf", parent_states, "run_child");
    let parent_compiled = Arc::new(compile(parent_doc).expect("compile parent"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-b3-subwf"),
        parent_compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    )
    .with_workflow_repo(repo2);

    inst.start().await.expect("start");

    // Drive — child runs synchronously in a spawned task and completes
    // immediately (its initial state is Final).
    for _ in 0..100 {
        let terminal = inst.step().await.expect("step");
        tokio::task::yield_now().await;
        if terminal {
            break;
        }
    }

    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "parent should complete after child workflow finishes"
    );

    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::SubworkflowStarted { .. })),
        "expected SubworkflowStarted"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::SubworkflowCompleted { .. })),
        "expected SubworkflowCompleted"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted)),
        "expected parent RunCompleted"
    );
}

/// 15. A subworkflow state with no repository still fails gracefully via
///     `SubworkflowFailed` (regression guard for the existing stub behaviour).
#[tokio::test]
async fn subworkflow_no_repo_emits_failed() {
    // Same test as the existing `subworkflow_failure_handled` but explicit that
    // "no repo" is the trigger.  Uses the test that already covers this path —
    // just verify the message differs from the old hard-coded string.
    let subwf_state = StateDefinition {
        id: "child_wf2".into(),
        name: "Child2".into(),
        state_type: StateType::Subworkflow,
        workflow_ref: Some("missing-wf@1.0".into()),
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "subworkflow.failed".into(),
                vec![TransitionSpec {
                    target: "recovered2".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m
        },
        agent: None,
        prompt: None,
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
        ports: None,
        authorized_roles: vec![],
        on_entry: vec![],
        on_exit: vec![],
        retry: None,
        timeout: None,
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    };

    let states = vec![subwf_state, leaf_state("recovered2", StateType::Final)];
    let doc = base_doc("wf-b3-no-repo", states, "child_wf2");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    // No with_workflow_repo → repo is None.
    let mut inst = WorkflowInstance::new(
        RunId::new("r-b3-no-repo"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");

    for _ in 0..30 {
        let terminal = inst.step().await.expect("step");
        tokio::task::yield_now().await;
        if terminal {
            break;
        }
    }

    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "recovery state is Final → Completed"
    );
    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::SubworkflowFailed { .. })),
        "expected SubworkflowFailed when no repo"
    );
}

// ── B4: History pseudo-state tests ───────────────────────────────────────────

/// 16. History pseudo-state fallback: first visit to `"compound.history"` with
///     no prior history recorded falls back to the compound's `initial` child.
///
/// Topology:
///   [start] --go--> [compound.history]
///   compound: initial=step_a
///     [step_a] --done--> [end_final]
///   [end_final] (Final)
#[tokio::test]
async fn history_pseudo_fallback_uses_initial() {
    // We build a *flat* workflow where the compound is simulated as:
    // start → (via "compound.history") → step_a → end.
    // No prior history exists so it must fall back to step_a.
    //
    // The compound state here is a simple Compound parent wrapping step_a.
    // For simplicity we test the pseudo-state routing on a flat compound:
    // the target state "compound_state.history" is the transition target.

    use langchart_model::state::StateDefinition;

    // step_a → end (child of compound_state)
    let step_a = with_transition(
        leaf_state("step_a", StateType::Atomic),
        "advance",
        "the_end",
        None,
    );

    // compound_state: Compound, initial = "step_a"
    let compound_state = StateDefinition {
        id: "compound_state".into(),
        name: "Compound".into(),
        state_type: StateType::Compound,
        initial: Some(StateId::new("step_a")),
        states: vec![step_a],
        history: Some(langchart_model::state::HistoryMode::Shallow),
        agent: None,
        prompt: None,
        input: Default::default(),
        context: None,
        model: None,
        capabilities: None,
        limits: None,
        regions: vec![],
        completion: None,
        workflow_ref: None,
        ports: None,
        authorized_roles: vec![],
        on_entry: vec![],
        on_exit: vec![],
        retry: None,
        timeout: None,
        on: Default::default(),
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    };

    // start --go--> compound_state.history (pseudo-state)
    let start = with_transition(
        leaf_state("start", StateType::Atomic),
        "go",
        "compound_state.history",
        None,
    );

    let states = vec![
        start,
        compound_state,
        leaf_state("the_end", StateType::Final),
    ];
    let doc = base_doc("wf-b4-hist-fallback", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-b4-fallback"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("go", serde_json::Value::Null);

    // After "go": history pseudo-state → no history → enter step_a (initial).
    // step_a is Atomic so we need another event to drive it to the_end.
    for _ in 0..20 {
        if inst.step().await.expect("step") {
            break;
        }
    }
    // Now in step_a; send "advance" → the_end (Final) → Completed.
    inst.send("advance", serde_json::Value::Null);
    for _ in 0..20 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "should complete via history fallback → step_a → the_end"
    );

    let payloads = sink.payloads().await;
    // The pseudo-state "compound_state.history" should emit StateEntered.
    assert!(
        payloads.iter().any(|p| matches!(
            p,
            RuntimeEventPayload::StateEntered { state_id } if state_id.0 == "compound_state.history"
        )),
        "expected StateEntered for history pseudo-state"
    );
    // step_a should also have been entered (via fallback).
    assert!(
        payloads.iter().any(|p| matches!(
            p,
            RuntimeEventPayload::StateEntered { state_id } if state_id.0 == "step_a"
        )),
        "expected StateEntered for step_a via fallback"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted))
    );
}

/// 17. History restore: exit compound, then re-enter via pseudo-state to confirm
///     the last-active child is restored rather than the initial.
///
/// Topology:
///   [hub] --to_b--> [compound_h] --go_back--> [hub]
///   compound_h: initial=a, history=Shallow
///     [a] --pick_b--> [b]
///     [b] (Atomic — stays active until exited)
///   After: [hub] --to_hist--> [compound_h.history]  (restores b)
///          [b] --finish--> [done]
#[tokio::test]
async fn history_pseudo_restores_last_active_child() {
    use langchart_model::state::StateDefinition;

    // Inside compound_h: a --pick_b--> b; b has a transition out to hub.
    let state_a = with_transition(leaf_state("ha", StateType::Atomic), "pick_b", "hb", None);
    let state_b = StateDefinition {
        id: "hb".into(),
        name: "B".into(),
        state_type: StateType::Atomic,
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "go_back".into(),
                vec![TransitionSpec {
                    target: "hub".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m.insert(
                "finish".into(),
                vec![TransitionSpec {
                    target: "done".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m
        },
        agent: None,
        prompt: None,
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
    };

    let compound_h = StateDefinition {
        id: "compound_h".into(),
        name: "Compound H".into(),
        state_type: StateType::Compound,
        initial: Some(StateId::new("ha")),
        states: vec![state_a, state_b],
        history: Some(langchart_model::state::HistoryMode::Shallow),
        on: Default::default(),
        agent: None,
        prompt: None,
        input: Default::default(),
        context: None,
        model: None,
        capabilities: None,
        limits: None,
        regions: vec![],
        completion: None,
        workflow_ref: None,
        ports: None,
        authorized_roles: vec![],
        on_entry: vec![],
        on_exit: vec![],
        retry: None,
        timeout: None,
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    };

    // hub transitions
    let hub = StateDefinition {
        id: "hub".into(),
        name: "Hub".into(),
        state_type: StateType::Atomic,
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "to_compound".into(),
                vec![TransitionSpec {
                    target: "compound_h".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m.insert(
                "to_hist".into(),
                vec![TransitionSpec {
                    target: "compound_h.history".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m
        },
        agent: None,
        prompt: None,
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
    };

    let states = vec![hub, compound_h, leaf_state("done", StateType::Final)];
    let doc = base_doc("wf-b4-hist-restore", states, "hub");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-b4-restore"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");

    // Step 1: hub → compound_h → enters ha (initial).
    inst.send("to_compound", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    // Step 2: ha → hb (pick_b).
    inst.send("pick_b", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    // Step 3: hb → hub (go_back); this exits compound_h → saves history: [hb].
    inst.send("go_back", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    // Step 4: hub → compound_h.history → restores hb (not ha).
    inst.send("to_hist", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    // Step 5: hb → done (finish).
    inst.send("finish", serde_json::Value::Null);
    for _ in 0..20 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "should complete via restored history → hb → done"
    );

    let payloads = sink.payloads().await;

    // HistoryRestored event should have been emitted.
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::HistoryRestored { .. })),
        "expected HistoryRestored event"
    );
    // hb should have been entered twice: once directly, once via history restore.
    let hb_enters = payloads
        .iter()
        .filter(
            |p| matches!(p, RuntimeEventPayload::StateEntered { state_id } if state_id.0 == "hb"),
        )
        .count();
    assert_eq!(
        hb_enters, 2,
        "hb should have been entered twice (direct + history restore)"
    );

    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunCompleted))
    );
}

// ── B5: CEL context variables in guards ──────────────────────────────────────

/// 18. Guard using `workflow_id` context variable passes or blocks based on
///     the workflow's own identity — not the event payload.
///
/// - Transition guarded by `workflow_id == "wf-cel-ctx"` → passes.
/// - Transition guarded by `workflow_id == "wrong-id"` → blocked.
#[tokio::test]
async fn cel_guard_uses_workflow_id_context() {
    // Two transitions: one that should fire (correct workflow_id), one blocked.
    // We test the passing case via run_to_completion.
    let states = vec![
        with_transition(
            leaf_state("start", StateType::Atomic),
            "go",
            "end",
            Some("workflow_id == \"wf-cel-ctx\""),
        ),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-cel-ctx", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-cel-ctx"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("go", serde_json::Value::Null);
    let status = inst.run_to_completion().await.expect("run");

    assert_eq!(
        status,
        RunStatus::Completed,
        "guard on correct workflow_id should pass"
    );
    let payloads = sink.payloads().await;
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::TransitionSelected { .. }))
    );
}

/// 19. Guard using `run_id` context variable.  The guard checks that `run_id`
///     starts with `"r-"` — a prefix we know the test uses.
#[tokio::test]
async fn cel_guard_uses_run_id_context() {
    let states = vec![
        with_transition(
            leaf_state("start", StateType::Atomic),
            "go",
            "end",
            Some("run_id.startsWith(\"r-\")"),
        ),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-run-id-guard", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-cel-run"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("go", serde_json::Value::Null);
    let status = inst.run_to_completion().await.expect("run");

    assert_eq!(
        status,
        RunStatus::Completed,
        "guard on run_id prefix should pass"
    );
}

/// 20. Guard mixing `workflow_id` (context) with payload field — both must pass.
#[tokio::test]
async fn cel_guard_combines_context_and_payload() {
    let states = vec![
        with_transition(
            leaf_state("start", StateType::Atomic),
            "go",
            "end",
            Some("workflow_id == \"wf-combined\" && approved == true"),
        ),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-combined", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-combined"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("go", serde_json::json!({ "approved": true }));
    let status = inst.run_to_completion().await.expect("run");

    assert_eq!(
        status,
        RunStatus::Completed,
        "combined guard (context + payload) should pass"
    );
}

// ── C1: ContextResolverChain integration test ─────────────────────────────────

/// 21. When a `ContextResolverChain` is injected, the agent invocation receives
///     non-empty `context_view` items from the stage.
///
/// We wire a single `ConstantContextStage` that always adds one item, then
/// capture it in a `CapturingActor` and assert the item arrived.
#[tokio::test]
async fn context_resolver_chain_populates_invocation() {
    use langchart_adapters::context::{
        ContextAccumulator, ContextError, ContextItem, ContextResolverStage, ContextView,
    };
    use langchart_context::chain::ContextResolverChain;
    use langchart_model::policy::ContextPolicy;
    use tokio::sync::Mutex as TokioMutex;

    // ── Stage: always injects a fixed item ──────────────────────────────────
    struct ConstantContextStage {
        captured_policy: Arc<TokioMutex<Option<ContextPolicy>>>,
    }

    #[async_trait]
    impl ContextResolverStage for ConstantContextStage {
        async fn resolve(
            &self,
            policy: &ContextPolicy,
            _run_id: &langchart_model::id::RunId,
            ctx: &mut ContextAccumulator,
        ) -> Result<(), ContextError> {
            *self.captured_policy.lock().await = Some(policy.clone());
            ctx.push(ContextItem {
                source: "test-stage".into(),
                content: "injected-content".into(),
                tokens: 3,
            });
            Ok(())
        }
    }

    // ── Actor: captures the context_view for assertion ─────────────────────
    struct CapturingActor {
        captured: Arc<TokioMutex<Option<ContextView>>>,
        captured_memory_write: Arc<TokioMutex<Option<bool>>>,
    }

    #[async_trait]
    impl AgentActor for CapturingActor {
        async fn run(
            &self,
            inv: langchart_runtime::instance::AgentInvocation,
            env: langchart_runtime::broker::CapabilityEnvelope,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<
            langchart_runtime::instance::AgentOutputEvent,
            langchart_runtime::instance::AgentError,
        > {
            *self.captured.lock().await = Some(inv.context_view);
            *self.captured_memory_write.lock().await = Some(env.policy().memory_write);
            Ok(langchart_runtime::instance::AgentOutputEvent {
                event_type: "work.done".into(),
                payload: serde_json::json!({}),
            })
        }
    }

    let captured: Arc<TokioMutex<Option<ContextView>>> = Arc::new(TokioMutex::new(None));
    let captured_policy = Arc::new(TokioMutex::new(None));
    let captured_memory_write = Arc::new(TokioMutex::new(None));

    let actor: Arc<dyn AgentActor> = Arc::new(CapturingActor {
        captured: captured.clone(),
        captured_memory_write: captured_memory_write.clone(),
    });

    let mut work = leaf_state("work", StateType::Agentic);
    work.agent = Some(AgentRef {
        id: AgentId::new("test-agent"),
        version: AgentVersion::new("0.1.0"),
    });
    work.on.insert(
        "work.done".into(),
        vec![TransitionSpec {
            target: "end".into(),
            guard: None,
            priority: 0,
            actions: vec![],
            kind: Default::default(),
        }],
    );

    let states = vec![work, leaf_state("end", StateType::Final)];
    let mut doc = base_doc("wf-c1-ctx", states, "work");
    doc.policy.max_capabilities.memory_write = true;
    doc.agents = vec![AgentDefinition {
        id: AgentId::new("test-agent"),
        version: AgentVersion::new("0.1.0"),
        description: "test agent defaults".into(),
        system_prompt: "test".into(),
        model_policy: ModelPolicy::default(),
        default_context_policy: ContextPolicy {
            sources: vec![langchart_model::policy::ContextSource::WorkflowData {
                expression: "data.topic".into(),
            }],
            token_budget: Some(64),
            exclude: Vec::new(),
        },
        default_capabilities: langchart_model::policy::CapabilityPolicy {
            memory_write: true,
            ..Default::default()
        },
        output_events: vec!["work.done".into()],
    }];
    let compiled = Arc::new(compile(doc).expect("compile"));

    let chain: Arc<dyn langchart_adapters::context::ContextResolver> =
        Arc::new(ContextResolverChain::new().add_stage(ConstantContextStage {
            captured_policy: captured_policy.clone(),
        }));

    let actors = HashMap::from([(StateId::new("work"), actor)]);
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst =
        WorkflowInstance::new(RunId::new("r-c1"), compiled, broker, sink.clone(), actors)
            .with_context_resolver(chain);

    inst.start().await.expect("start");

    for _ in 0..50 {
        let terminal = inst.step().await.expect("step");
        tokio::task::yield_now().await;
        if terminal {
            break;
        }
    }

    assert_eq!(inst.status, RunStatus::Completed, "run should complete");

    let view = captured
        .lock()
        .await
        .take()
        .expect("actor must have been called");
    assert_eq!(view.items.len(), 1, "should have exactly one context item");
    assert_eq!(view.items[0].source, "test-stage");
    let policy = captured_policy
        .lock()
        .await
        .take()
        .expect("policy captured");
    assert_eq!(policy.token_budget, Some(64));
    assert_eq!(*captured_memory_write.lock().await, Some(true));
    assert_eq!(view.items[0].content, "injected-content");
    assert_eq!(view.token_count, 3);
}

#[tokio::test]
async fn configured_unhandled_events_fail_the_run() {
    let mut doc = base_doc(
        "wf-unhandled-fails",
        vec![leaf_state("idle", StateType::Atomic)],
        "idle",
    );
    doc.policy.unhandled_event_is_failure = true;
    let compiled = Arc::new(compile(doc).expect("compile"));
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-unhandled-fails"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send("unknown.event", serde_json::Value::Null);
    assert!(inst.step().await.expect("step"));
    assert_eq!(inst.status, RunStatus::Failed);

    let payloads = sink.payloads().await;
    assert!(payloads.iter().any(|payload| matches!(
        payload,
        RuntimeEventPayload::EventUnhandled { event_type }
            if event_type == "unknown.event"
    )));
    assert!(payloads.iter().any(|payload| matches!(
        payload,
        RuntimeEventPayload::RunFailed { message }
            if message.contains("unknown.event")
    )));
}

#[tokio::test]
async fn unhandled_integration_broadcasts_do_not_fail_strict_runs() {
    let mut doc = base_doc(
        "wf-unhandled-broadcast",
        vec![leaf_state("idle", StateType::Atomic)],
        "idle",
    );
    doc.policy.unhandled_event_is_failure = true;
    let compiled = Arc::new(compile(doc).expect("compile"));
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-unhandled-broadcast"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );

    inst.start().await.expect("start");
    inst.send_broadcast("unrelated.integration.event", serde_json::Value::Null);
    inst.step().await.expect("step");
    assert_eq!(inst.status, RunStatus::Running);

    let payloads = sink.payloads().await;
    assert!(payloads.iter().any(|payload| matches!(
        payload,
        RuntimeEventPayload::EventUnhandled { event_type }
            if event_type == "unrelated.integration.event"
    )));
    assert!(
        !payloads
            .iter()
            .any(|payload| matches!(payload, RuntimeEventPayload::RunFailed { .. }))
    );
}

#[tokio::test]
async fn recovering_a_suspended_run_waits_for_resume_before_restarting_activity() {
    use langchart_adapters::{
        checkpoint::{CheckpointError, CheckpointStore, RunSnapshot},
        workflow_repository::{InMemoryWorkflowRepository, WorkflowRepository},
    };
    use langchart_model::id::CheckpointId;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct MemCheckpointStore {
        snapshot: Mutex<Option<RunSnapshot>>,
    }

    #[async_trait]
    impl CheckpointStore for MemCheckpointStore {
        async fn save(&self, snapshot: &RunSnapshot) -> Result<CheckpointId, CheckpointError> {
            *self.snapshot.lock().await = Some(snapshot.clone());
            Ok(snapshot.checkpoint_id.clone())
        }

        async fn load(&self, run_id: &RunId) -> Result<Option<RunSnapshot>, CheckpointError> {
            Ok(self
                .snapshot
                .lock()
                .await
                .clone()
                .filter(|snapshot| &snapshot.run_id == run_id))
        }

        async fn latest(&self, run_id: &RunId) -> Result<Option<CheckpointId>, CheckpointError> {
            Ok(self
                .load(run_id)
                .await?
                .map(|snapshot| snapshot.checkpoint_id))
        }
    }

    struct PendingActor {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AgentActor for PendingActor {
        async fn run(
            &self,
            _invocation: langchart_runtime::instance::AgentInvocation,
            _envelope: langchart_runtime::broker::CapabilityEnvelope,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<
            langchart_runtime::instance::AgentOutputEvent,
            langchart_runtime::instance::AgentError,
        > {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::future::pending().await
        }
    }

    let document = base_doc(
        "wf-suspended-recovery",
        vec![
            agentic_state("work", "work.done", "end"),
            leaf_state("end", StateType::Final),
        ],
        "work",
    );
    let compiled = Arc::new(compile(document.clone()).expect("compile"));
    let repository: Arc<dyn WorkflowRepository> = Arc::new(
        InMemoryWorkflowRepository::new().register("wf-suspended-recovery@0.1.0", compiled.clone()),
    );
    let store: Arc<dyn CheckpointStore> = Arc::new(MemCheckpointStore::default());
    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let calls = Arc::new(AtomicUsize::new(0));
    let actor: Arc<dyn AgentActor> = Arc::new(PendingActor {
        calls: calls.clone(),
    });
    let actors = HashMap::from([(StateId::new("work"), actor.clone())]);
    let run_id = RunId::new("r-suspended-recovery");

    let mut original = WorkflowInstance::new(
        run_id.clone(),
        compiled,
        broker,
        sink.clone(),
        actors.clone(),
    )
    .with_checkpoint_store(store.clone());
    original.start().await.expect("start original");
    tokio::task::yield_now().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    original.suspend().await.expect("suspend original");

    let engine = RuntimeEngine::new(EngineAdapters {
        llm: Arc::new(NoopLlm),
        mcp: Arc::new(NoopMcp),
        memory: Arc::new(NoopMemory),
        secrets: Arc::new(HostMapSecretsAdapter::empty()),
        event_sink: sink,
        checkpoint_store: Some(store),
        workflow_repo: Some(repository),
        event_source: None,
        artifact_store: None,
    });
    engine
        .recover_run(&run_id, actors)
        .await
        .expect("recover suspended run");
    tokio::task::yield_now().await;

    assert_eq!(
        engine.inspect(&run_id).await.expect("inspect").status,
        RunStatus::Suspended
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "recovery must stay quiescent"
    );

    engine.resume(&run_id).await.expect("resume recovered run");
    tokio::task::yield_now().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "resume restarts the activity once"
    );
    engine.cancel(&run_id).await.expect("cancel recovered run");

    let error = engine
        .recover_run(&run_id, HashMap::new())
        .await
        .expect_err("terminal checkpoints must not be recovered");
    assert!(matches!(
        error,
        langchart_runtime::engine::EngineError::Checkpoint(message)
            if message.contains("cannot recover terminal run")
    ));
}

#[tokio::test]
async fn agent_state_input_bindings_are_resolved_from_workflow_data() {
    use tokio::sync::Mutex as TokioMutex;

    struct InputCapturingActor {
        captured: Arc<TokioMutex<Option<ron::Value>>>,
    }

    #[async_trait]
    impl AgentActor for InputCapturingActor {
        async fn run(
            &self,
            invocation: langchart_runtime::instance::AgentInvocation,
            _envelope: langchart_runtime::broker::CapabilityEnvelope,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<
            langchart_runtime::instance::AgentOutputEvent,
            langchart_runtime::instance::AgentError,
        > {
            *self.captured.lock().await = Some(invocation.input);
            Ok(langchart_runtime::instance::AgentOutputEvent {
                event_type: "work.done".into(),
                payload: serde_json::json!({}),
            })
        }
    }

    let captured = Arc::new(TokioMutex::new(None));
    let mut work = leaf_state("work", StateType::Agentic);
    work.agent = Some(AgentRef {
        id: AgentId::new("test-agent"),
        version: AgentVersion::new("0.1.0"),
    });
    work.input
        .insert("topic".into(), "${workflow.topic}".into());
    work.input.insert("mode".into(), "careful".into());
    work.on.insert(
        "work.done".into(),
        vec![TransitionSpec {
            target: "end".into(),
            guard: None,
            priority: 0,
            actions: vec![],
            kind: Default::default(),
        }],
    );

    let compiled = Arc::new(
        compile(base_doc(
            "wf-agent-input",
            vec![work, leaf_state("end", StateType::Final)],
            "work",
        ))
        .expect("compile"),
    );
    let actor: Arc<dyn AgentActor> = Arc::new(InputCapturingActor {
        captured: captured.clone(),
    });
    let sink = Arc::new(VecSink::default());
    let mut instance = WorkflowInstance::new(
        RunId::new("r-agent-input"),
        compiled,
        bare_broker(sink.clone()),
        sink,
        HashMap::from([(StateId::new("work"), actor)]),
    )
    .with_workflow_data(
        serde_json::from_value(serde_json::json!({
            "topic": "migration"
        }))
        .unwrap(),
    );

    instance.start().await.expect("start");
    for _ in 0..50 {
        if instance.step().await.expect("step") {
            break;
        }
        tokio::task::yield_now().await;
    }

    let input = captured.lock().await.take().expect("actor was invoked");
    let input_json = serde_json::to_value(input).unwrap();
    assert_eq!(
        input_json,
        serde_json::json!({
            "topic": "migration",
            "mode": "careful"
        })
    );
}

// ── D: Checkpoint save / restore ─────────────────────────────────────────────

/// 22. `take_checkpoint` captures the mutable state; `restore_from_checkpoint`
///     rehydrates a new instance with the same configuration.
///
/// Flow:
///   start ─[go]─> middle ─[done]─> end
///
/// We start the run, pause in `middle` (by not sending `done`), take a
/// checkpoint, build a fresh instance, restore from the checkpoint, then
/// verify that the active state is `middle` and that sending `done` completes
/// the new instance.
#[tokio::test]
async fn checkpoint_save_and_restore() {
    use langchart_runtime::run::InstanceCheckpoint;

    let states = vec![
        with_transition(leaf_state("start", StateType::Atomic), "go", "middle", None),
        with_transition(leaf_state("middle", StateType::Atomic), "done", "end", None),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-ck", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());

    // ── Phase 1: advance to "middle" ─────────────────────────────────────────
    let mut orig = WorkflowInstance::new(
        RunId::new("r-ck"),
        compiled.clone(),
        broker.clone(),
        sink.clone(),
        HashMap::new(),
    );
    orig.start().await.expect("start");
    orig.send("go", serde_json::Value::Null);
    for _ in 0..20 {
        if orig.step().await.expect("step") {
            break;
        }
    }

    assert!(
        orig.active_states.contains(&StateId::new("middle")),
        "should be in middle"
    );

    // ── Phase 2: take checkpoint ──────────────────────────────────────────────
    let ck: InstanceCheckpoint = orig.take_checkpoint();
    assert_eq!(ck.active_states, vec![StateId::new("middle")]);
    assert_eq!(ck.status, RunStatus::Running);

    // ── Phase 3: restore into a fresh instance ────────────────────────────────
    let sink2 = Arc::new(VecSink::default());
    let broker2 = bare_broker(sink2.clone());
    let mut restored = WorkflowInstance::new(
        RunId::new("r-ck"),
        compiled.clone(),
        broker2.clone(),
        sink2.clone(),
        HashMap::new(),
    );
    restored.restore_from_checkpoint(&ck);

    // Active states should match the checkpoint.
    assert_eq!(restored.active_states, vec![StateId::new("middle")]);
    assert_eq!(restored.status, RunStatus::Running);

    // ── Phase 4: send `done` and verify completion ────────────────────────────
    restored.send("done", serde_json::Value::Null);
    for _ in 0..20 {
        if restored.step().await.expect("step") {
            break;
        }
    }

    assert_eq!(
        restored.status,
        RunStatus::Completed,
        "restored run should complete"
    );
}

#[tokio::test]
async fn checkpoint_preserves_workflow_data_used_by_guards() {
    use langchart_runtime::run::InstanceCheckpoint;

    let mut decision = leaf_state("decision", StateType::Atomic);
    decision.on.insert(
        "decide".into(),
        vec![
            TransitionSpec {
                target: "approved".into(),
                guard: Some("data.approved == true".into()),
                priority: 0,
                actions: vec![],
                kind: Default::default(),
            },
            TransitionSpec {
                target: "rejected".into(),
                guard: Some("data.approved == false".into()),
                priority: 1,
                actions: vec![],
                kind: Default::default(),
            },
        ],
    );
    let mut document = base_doc(
        "wf-checkpoint-data",
        vec![
            decision,
            leaf_state("approved", StateType::Final),
            leaf_state("rejected", StateType::Final),
        ],
        "decision",
    );
    document
        .data_schema
        .fields
        .insert("approved".into(), "bool".into());
    let compiled = Arc::new(compile(document).expect("compile"));
    let sink = Arc::new(VecSink::default());
    let mut original = WorkflowInstance::new(
        RunId::new("r-checkpoint-data"),
        compiled.clone(),
        bare_broker(sink.clone()),
        sink,
        HashMap::new(),
    )
    .with_workflow_data(ron::from_str(r#"{"approved": true}"#).expect("workflow data"));
    original.start().await.expect("start original");

    let encoded = serde_json::to_vec(&original.take_checkpoint()).expect("serialize checkpoint");
    let checkpoint: InstanceCheckpoint =
        serde_json::from_slice(&encoded).expect("deserialize checkpoint");
    assert!(checkpoint.workflow_data.is_some());
    let mut legacy_json: serde_json::Value =
        serde_json::from_slice(&encoded).expect("checkpoint JSON");
    legacy_json
        .as_object_mut()
        .expect("checkpoint object")
        .remove("workflow_data");
    let legacy: InstanceCheckpoint =
        serde_json::from_value(legacy_json).expect("legacy checkpoint without workflow data");
    assert_eq!(legacy.workflow_data, None);

    let restored_sink = Arc::new(VecSink::default());
    let mut restored = WorkflowInstance::new(
        RunId::new("r-checkpoint-data"),
        compiled,
        bare_broker(restored_sink.clone()),
        restored_sink.clone(),
        HashMap::new(),
    );
    restored.restore_from_checkpoint(&checkpoint);
    restored.send("decide", serde_json::Value::Null);
    assert_eq!(
        restored.run_to_completion().await.expect("run"),
        RunStatus::Completed
    );

    assert!(restored_sink.payloads().await.iter().any(
        |payload| matches!(payload, RuntimeEventPayload::TransitionSelected { to, .. } if to.0 == "approved")
    ));
}

#[tokio::test]
async fn checkpoint_preserves_queued_activity_completion_and_ownership() {
    use langchart_runtime::run::InstanceCheckpoint;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingPendingActor(Arc<AtomicUsize>);

    #[async_trait]
    impl AgentActor for CountingPendingActor {
        async fn run(
            &self,
            _invocation: langchart_runtime::instance::AgentInvocation,
            _envelope: langchart_runtime::broker::CapabilityEnvelope,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<
            langchart_runtime::instance::AgentOutputEvent,
            langchart_runtime::instance::AgentError,
        > {
            self.0.fetch_add(1, Ordering::SeqCst);
            std::future::pending().await
        }
    }

    let states = vec![
        agentic_state("work", "work.done", "end"),
        leaf_state("end", StateType::Final),
    ];
    let compiled = Arc::new(compile(base_doc("wf-queued-ck", states, "work")).expect("compile"));
    let sink = Arc::new(VecSink::default());
    let mut original = WorkflowInstance::new(
        RunId::new("r-queued-ck"),
        compiled.clone(),
        bare_broker(sink.clone()),
        sink,
        HashMap::from([(
            StateId::new("work"),
            Arc::new(ScriptedAgentActor::emit(
                "work.done",
                serde_json::Value::Null,
            )) as Arc<dyn AgentActor>,
        )]),
    );

    original.start().await.expect("start");
    original.send("unhandled", serde_json::Value::Null);
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    original.step().await.expect("queue activity completion");

    let checkpoint: InstanceCheckpoint = original.take_checkpoint();
    assert_eq!(checkpoint.event_queue.len(), 1);
    assert_eq!(checkpoint.event_queue[0].event_type, "work.done");
    assert!(
        checkpoint
            .queued_activity_invocations
            .contains_key(&StateId::new("work"))
    );

    let restarted = Arc::new(AtomicUsize::new(0));
    let sink = Arc::new(VecSink::default());
    let mut restored = WorkflowInstance::new(
        RunId::new("r-queued-ck"),
        compiled,
        bare_broker(sink.clone()),
        sink,
        HashMap::from([(
            StateId::new("work"),
            Arc::new(CountingPendingActor(restarted.clone())) as Arc<dyn AgentActor>,
        )]),
    );
    restored.restore_from_checkpoint(&checkpoint);
    restored
        .start_activity_if_needed_pub(&StateId::new("work"))
        .await
        .expect("restore activity state");
    assert_eq!(restarted.load(Ordering::SeqCst), 0);

    assert!(restored.step().await.expect("process restored completion"));
    assert_eq!(restored.status, RunStatus::Completed);
}

/// 23. `save_checkpoint` + `CheckpointStore` integration:
///     after suspension the checkpoint round-trips through the in-memory store.
#[tokio::test]
async fn checkpoint_round_trips_through_store() {
    use langchart_adapters::checkpoint::{
        CheckpointError, CheckpointStore, RunSnapshot as CkSnapshot,
    };
    use langchart_model::id::CheckpointId;
    use langchart_runtime::run::InstanceCheckpoint;
    use std::sync::Mutex as StdMutex;

    // ── Minimal in-memory store ───────────────────────────────────────────────
    #[derive(Default)]
    struct MemStore {
        inner: StdMutex<HashMap<String, CkSnapshot>>,
    }
    #[async_trait]
    impl CheckpointStore for MemStore {
        async fn save(&self, snap: &CkSnapshot) -> Result<CheckpointId, CheckpointError> {
            let id = snap.checkpoint_id.clone();
            self.inner
                .lock()
                .unwrap()
                .insert(snap.run_id.0.clone(), snap.clone());
            Ok(id)
        }
        async fn load(
            &self,
            run_id: &langchart_model::id::RunId,
        ) -> Result<Option<CkSnapshot>, CheckpointError> {
            Ok(self.inner.lock().unwrap().get(&run_id.0).cloned())
        }
        async fn latest(
            &self,
            run_id: &langchart_model::id::RunId,
        ) -> Result<Option<CheckpointId>, CheckpointError> {
            Ok(self
                .inner
                .lock()
                .unwrap()
                .get(&run_id.0)
                .map(|s| s.checkpoint_id.clone()))
        }
    }

    let store: Arc<dyn CheckpointStore> = Arc::new(MemStore::default());

    let states = vec![
        with_transition(leaf_state("start", StateType::Atomic), "go", "middle", None),
        with_transition(leaf_state("middle", StateType::Atomic), "done", "end", None),
        leaf_state("end", StateType::Final),
    ];
    let doc = base_doc("wf-ck2", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let run_id = RunId::new("r-ck2");

    let mut inst = WorkflowInstance::new(
        run_id.clone(),
        compiled.clone(),
        broker.clone(),
        sink.clone(),
        HashMap::new(),
    )
    .with_checkpoint_store(store.clone());

    inst.start().await.expect("start");
    inst.send("go", serde_json::Value::Null);
    for _ in 0..20 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    // Suspend → should auto-save checkpoint.
    inst.suspend().await.expect("suspend");

    // Verify checkpoint was saved.
    let snap = store
        .load(&run_id)
        .await
        .expect("load")
        .expect("should be present");
    let ck: InstanceCheckpoint = serde_json::from_slice(&snap.payload).expect("deserialize");

    assert_eq!(ck.active_states, vec![StateId::new("middle")]);
    assert_eq!(ck.status, RunStatus::Suspended);
}

// ── F3: Internal / Local transition kinds ────────────────────────────────────

/// 24. Internal transition does NOT re-run on_entry / on_exit.
///
/// Workflow: `start` (atomic, on_entry increments counter) --internal--> same
/// target `start`.  After firing, `active_states` still contains `start` and
/// the on_entry counter must still be 1 (not 2).
#[tokio::test]
async fn internal_transition_does_not_rerun_on_entry() {
    use langchart_runtime::instance::{ActionContext, ActionError, ActionRegistry, StateAction};
    use std::sync::atomic::{AtomicU32, Ordering};

    let entry_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

    struct CountEntry(Arc<AtomicU32>);
    #[async_trait]
    impl StateAction for CountEntry {
        async fn run(
            &self,
            _ctx: ActionContext,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<(), ActionError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());

    // `start` has an on_entry counter and an *internal* self-transition on "ping".
    let mut start = leaf_state("start", StateType::Atomic);
    start.on_entry = vec!["count_entry".into()];
    start.on.insert(
        "ping".into(),
        vec![TransitionSpec {
            target: "start".into(), // same state — internal
            guard: None,
            priority: 0,
            actions: vec![],
            kind: TransitionKind::Internal,
        }],
    );
    // Also need a way to exit cleanly.
    start.on.insert(
        "finish".into(),
        vec![TransitionSpec {
            target: "end".into(),
            guard: None,
            priority: 0,
            actions: vec![],
            kind: TransitionKind::External,
        }],
    );

    let doc = base_doc(
        "wf-f3-internal",
        vec![start, leaf_state("end", StateType::Final)],
        "start",
    );
    let compiled = Arc::new(compile(doc).expect("compile"));

    let registry = ActionRegistry::new()
        .register("count_entry", CountEntry(entry_count.clone()))
        .into_map();

    let mut inst = WorkflowInstance::with_actions(
        RunId::new("r-f3-internal"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
        registry,
    );
    inst.start().await.expect("start");

    // on_entry fires once on initial entry.
    assert_eq!(
        entry_count.load(Ordering::SeqCst),
        1,
        "on_entry fires once on entry"
    );

    // Fire the internal transition — state stays active, on_entry must NOT fire again.
    inst.send("ping", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    assert_eq!(
        entry_count.load(Ordering::SeqCst),
        1,
        "internal transition must NOT re-run on_entry"
    );
    assert_eq!(
        inst.status,
        RunStatus::Running,
        "run still active after internal transition"
    );

    // Verify state is still start (not exited).
    let payloads = sink.payloads().await;
    let state_exited_start = payloads.iter().any(
        |p| matches!(p, RuntimeEventPayload::StateExited { state_id } if state_id.0 == "start"),
    );
    assert!(
        !state_exited_start,
        "start must NOT have emitted StateExited for internal transition"
    );

    // Clean exit.
    inst.send("finish", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }
    assert_eq!(inst.status, RunStatus::Completed);
}

/// 25. Local transition from a compound state to a child descendant does NOT
///     exit/re-enter the compound state itself.
///
/// Workflow: compound `outer` (initial=`child_a`, with on_entry counter)
/// declares a *local* transition on "next" targeting `child_b`.
/// `child_a` has no "next" transition so the event bubbles to `outer`.
/// (In this test we put the transition directly on `outer` to isolate F3.)
/// After firing, `outer` stays in `active_states` and its on_entry counter
/// must NOT increase (no re-entry), while `child_b` becomes active.
#[tokio::test]
async fn local_transition_stays_in_compound() {
    use langchart_runtime::instance::{ActionContext, ActionError, ActionRegistry, StateAction};
    use std::sync::atomic::{AtomicU32, Ordering};

    let outer_entry_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

    struct CountOuter(Arc<AtomicU32>);
    #[async_trait]
    impl StateAction for CountOuter {
        async fn run(
            &self,
            _ctx: ActionContext,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<(), ActionError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    // Build: outer (compound, initial=child_a) has a LOCAL transition
    //        on "next" targeting child_b.
    //        child_a is a plain leaf with no transitions.
    //        child_b has an external "done" → end.
    let child_a = leaf_state("child_a", StateType::Atomic);
    let child_b = {
        let mut s = leaf_state("child_b", StateType::Atomic);
        s.on.insert(
            "done".into(),
            vec![TransitionSpec {
                target: "end".into(),
                guard: None,
                priority: 0,
                actions: vec![],
                kind: TransitionKind::External,
            }],
        );
        s
    };

    let outer = StateDefinition {
        id: "outer".into(),
        name: "Outer".into(),
        state_type: StateType::Compound,
        initial: Some(StateId::new("child_a")),
        states: vec![child_a, child_b],
        on_entry: vec!["count_outer".into()],
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "next".into(),
                vec![TransitionSpec {
                    target: "child_b".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: TransitionKind::Local,
                }],
            );
            m
        },
        agent: None,
        prompt: None,
        input: Default::default(),
        context: None,
        model: None,
        capabilities: None,
        limits: None,
        regions: vec![],
        completion: None,
        history: None,
        workflow_ref: None,
        ports: None,
        authorized_roles: vec![],
        on_exit: vec![],
        retry: None,
        timeout: None,
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    };

    let states = vec![outer, leaf_state("end", StateType::Final)];
    let doc = base_doc("wf-f3-local", states, "outer");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let registry = ActionRegistry::new()
        .register("count_outer", CountOuter(outer_entry_count.clone()))
        .into_map();

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::with_actions(
        RunId::new("r-f3-local"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
        registry,
    );
    inst.start().await.expect("start");

    // on_entry for outer fires once on initial entry.
    assert_eq!(
        outer_entry_count.load(Ordering::SeqCst),
        1,
        "outer on_entry fires once on entry"
    );

    // Initial: outer + child_a active.
    assert!(inst.active_states.contains(&StateId::new("outer")));
    assert!(inst.active_states.contains(&StateId::new("child_a")));

    // Fire the LOCAL transition on "next" — outer has it, targeting child_b.
    inst.send("next", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    // outer on_entry must NOT have fired again (no re-entry of compound).
    assert_eq!(
        outer_entry_count.load(Ordering::SeqCst),
        1,
        "local transition must NOT re-run outer on_entry"
    );

    // `outer` must still be active (local transition did not exit it).
    assert!(
        inst.active_states.contains(&StateId::new("outer")),
        "outer compound state must remain active after local transition"
    );
    // child_b is now active.
    assert!(
        inst.active_states.contains(&StateId::new("child_b")),
        "child_b must be active after local transition"
    );
    // child_a is no longer active.
    assert!(
        !inst.active_states.contains(&StateId::new("child_a")),
        "child_a must no longer be active"
    );

    // Verify outer did NOT emit StateExited during the local transition.
    let payloads = sink.payloads().await;
    let outer_exited_count = payloads
        .iter()
        .filter(
            |p| matches!(p, RuntimeEventPayload::StateExited { state_id } if state_id.0 == "outer"),
        )
        .count();
    assert_eq!(
        outer_exited_count, 0,
        "outer compound must NOT have emitted StateExited during the local transition"
    );

    // Clean exit.
    inst.send("done", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }
    assert_eq!(inst.status, RunStatus::Completed);
}

// ── F4: Event bubbling ────────────────────────────────────────────────────────

/// 26. Event handled by compound parent when active leaf has no matching transition.
///
/// Workflow: compound `outer` (initial=`inner`) → `outer` has "escape" → end.
/// `inner` has NO "escape" transition.
/// Sending "escape" must bubble up to `outer`, which handles it.
#[tokio::test]
async fn event_bubbles_to_compound_ancestor() {
    let inner = leaf_state("inner", StateType::Atomic); // no "escape" transition

    let outer = StateDefinition {
        id: "outer".into(),
        name: "Outer".into(),
        state_type: StateType::Compound,
        initial: Some(StateId::new("inner")),
        states: vec![inner],
        on: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "escape".into(),
                vec![TransitionSpec {
                    target: "end".into(),
                    guard: None,
                    priority: 0,
                    actions: vec![],
                    kind: Default::default(),
                }],
            );
            m
        },
        agent: None,
        prompt: None,
        input: Default::default(),
        context: None,
        model: None,
        capabilities: None,
        limits: None,
        regions: vec![],
        completion: None,
        history: None,
        workflow_ref: None,
        ports: None,
        authorized_roles: vec![],
        on_entry: vec![],
        on_exit: vec![],
        retry: None,
        timeout: None,
        output_schemas: Default::default(),
        _editor: serde_json::Value::Null,
    };

    let states = vec![outer, leaf_state("end", StateType::Final)];
    let doc = base_doc("wf-f4-bubble", states, "outer");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-f4-bubble"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    );
    inst.start().await.expect("start");

    // Active: outer + inner.
    assert!(inst.active_states.contains(&StateId::new("outer")));
    assert!(inst.active_states.contains(&StateId::new("inner")));

    // "escape" is not on `inner` — must bubble to `outer`.
    inst.send("escape", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "run should complete via bubbled event"
    );

    // Verify TransitionSelected was emitted from `outer` (not from `inner`).
    let payloads = sink.payloads().await;
    let selected = payloads.iter().find(|p| {
        matches!(p, RuntimeEventPayload::TransitionSelected { from, event_type, .. }
            if from.0 == "outer" && event_type == "escape")
    });
    assert!(
        selected.is_some(),
        "TransitionSelected must be emitted with from=outer for bubbled event"
    );

    // Verify EventUnhandled was NOT emitted.
    let unhandled = payloads
        .iter()
        .any(|p| matches!(p, RuntimeEventPayload::EventUnhandled { event_type } if event_type == "escape"));
    assert!(
        !unhandled,
        "EventUnhandled must NOT be emitted when ancestor handles the event"
    );
}

// ── F1: WorkflowData in CEL guards ───────────────────────────────────────────

/// 27. Guard expression reads `data.field_name` from workflow data.
///
/// Workflow: start --[data.approved == true]--> approved
///                 --[data.approved == false]--> rejected
/// With `approved = true` in workflow data → transitions to `approved`.
#[tokio::test]
async fn cel_guard_reads_workflow_data() {
    use langchart_runtime::run::WorkflowInstance;

    let start = StateDefinition {
        id: "start".into(),
        name: "Start".into(),
        state_type: StateType::Atomic,
        on: {
            let mut m = std::collections::HashMap::new();
            // Higher priority (lower number) transition with data.approved == true guard.
            m.insert(
                "decide".into(),
                vec![
                    TransitionSpec {
                        target: "approved".into(),
                        guard: Some("data.approved == true".into()),
                        priority: 0,
                        actions: vec![],
                        kind: Default::default(),
                    },
                    TransitionSpec {
                        target: "rejected".into(),
                        guard: Some("data.approved == false".into()),
                        priority: 1,
                        actions: vec![],
                        kind: Default::default(),
                    },
                ],
            );
            m
        },
        agent: None,
        prompt: None,
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
    };

    let states = vec![
        start,
        leaf_state("approved", StateType::Final),
        leaf_state("rejected", StateType::Final),
    ];
    let doc = base_doc("wf-f1-data", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());

    // Workflow data: approved = true
    let workflow_data = ron::Value::Map(ron::Map::from_iter([(
        ron::Value::String("approved".into()),
        ron::Value::Bool(true),
    )]));

    let mut inst = WorkflowInstance::new(
        RunId::new("r-f1-data"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    )
    .with_workflow_data(workflow_data);

    inst.start().await.expect("start");

    inst.send("decide", serde_json::Value::Null);
    for _ in 0..10 {
        if inst.step().await.expect("step") {
            break;
        }
    }

    assert_eq!(inst.status, RunStatus::Completed);

    // The `approved` guard should have passed — verify the transition went to `approved`.
    let payloads = sink.payloads().await;
    let went_to_approved = payloads.iter().any(
        |p| matches!(p, RuntimeEventPayload::TransitionSelected { to, .. } if to.0 == "approved"),
    );
    assert!(
        went_to_approved,
        "workflow data guard data.approved==true should route to `approved`"
    );
}

// ── F2: Timer checkpoint persistence ─────────────────────────────────────────

/// 28. A timer scheduled before a checkpoint is preserved and fires after recovery.
///
/// Scenario:
///   1. Build a two-state workflow: `start` --timer.fired--> `end`.
///   2. Start the instance, manually schedule a 30 ms timer on `start`.
///   3. Take a checkpoint immediately (timer not yet fired).
///   4. Construct a fresh instance and call `restore_from_checkpoint`.
///   5. Drive the new instance's step loop — the restored timer must fire and
///      inject "timer.fired", causing the run to complete.
#[tokio::test(start_paused = true)]
async fn timer_survives_checkpoint_and_fires_after_recovery() {
    use langchart_model::id::StateId;
    use langchart_runtime::run::{InstanceCheckpoint, WorkflowInstance};
    use std::time::Duration;

    // Build start --timer.fired--> end
    let start = with_transition(
        leaf_state("start", StateType::Atomic),
        "timer.fired",
        "end",
        None,
    );
    let states = vec![start, leaf_state("end", StateType::Final)];
    let doc = base_doc("wf-f2-timer", states, "start");
    let compiled = Arc::new(compile(doc).expect("compile"));

    let sink1 = Arc::new(VecSink::default());
    let broker1 = bare_broker(sink1.clone());
    let mut inst1 = WorkflowInstance::new(
        RunId::new("r-f2-timer"),
        compiled.clone(),
        broker1,
        sink1.clone(),
        HashMap::new(),
    );

    inst1.start().await.expect("start");

    // Schedule a 30 ms timer on the "start" state.
    inst1.schedule_timer(
        StateId::new("start"),
        "timer.fired",
        Duration::from_millis(30),
    );

    // Take the checkpoint while the timer is still pending.
    let checkpoint: InstanceCheckpoint = inst1.take_checkpoint();
    assert_eq!(
        checkpoint.pending_timers.len(),
        1,
        "checkpoint must contain the pending timer"
    );

    // --- Recovery on a fresh instance ---
    let sink2 = Arc::new(VecSink::default());
    let broker2 = bare_broker(sink2.clone());
    let mut inst2 = WorkflowInstance::new(
        RunId::new("r-f2-timer"),
        compiled.clone(),
        broker2,
        sink2.clone(),
        HashMap::new(),
    );
    // Restore BEFORE advancing time so that the re-armed timer's sleep(remaining_ms)
    // fires when we subsequently advance the virtual clock past the fire point.
    inst2.restore_from_checkpoint(&checkpoint);
    // After restore the timer is re-armed; status = Running.
    assert_eq!(inst2.status, RunStatus::Running);

    // Yield once so the newly spawned timer task can register its sleep with the
    // Tokio time driver. Without this, `advance` may not see the pending sleep.
    tokio::task::yield_now().await;

    // Advance virtual time past the 30 ms fire point (timer was scheduled for 30 ms).
    tokio::time::advance(Duration::from_millis(50)).await;
    // Yield to allow the timer task that was woken by `advance` to run and send
    // on the channel before we poll with `try_recv` in `step`.
    tokio::task::yield_now().await;

    // Drive the step loop until completion (timer fires → "timer.fired" → transition).
    for _ in 0..50 {
        if inst2.step().await.expect("step") {
            break;
        }
        tokio::task::yield_now().await;
    }

    assert_eq!(
        inst2.status,
        RunStatus::Completed,
        "run must complete after restored timer fires"
    );

    // Verify "timer.fired" produced a TransitionSelected event.
    let payloads = sink2.payloads().await;
    let routed = payloads.iter().any(|p| {
        matches!(p, RuntimeEventPayload::TransitionSelected { event_type, to, .. }
            if event_type == "timer.fired" && to.0 == "end")
    });
    assert!(
        routed,
        "TransitionSelected for timer.fired → end must be present"
    );
}

// ── F6: Agent output_events validation ───────────────────────────────────────

/// 29. An agentic state with a declared AgentDefinition rejects an event_type
///     that is not listed in the agent's `output_events`.
///
/// Scenario:
///   1. Build a workflow with an inline AgentDefinition whose output_events is
///      `["analysis.completed"]`.
///   2. The agentic state references that agent.
///   3. A `ScriptedAgentActor` emits `"undeclared.event"` — not in output_events.
///   4. The instance must emit `ActivityInvalidOutput` and NOT transition.
#[tokio::test]
async fn agent_undeclared_output_triggers_invalid_output() {
    let agent_id = AgentId::new("checker-agent");
    let agent_ver = AgentVersion::new("1.0.0");

    // Inline agent definition with a single declared output event.
    let agent_def = AgentDefinition {
        id: agent_id.clone(),
        version: agent_ver.clone(),
        description: "test agent".into(),
        system_prompt: "You are a test agent.".into(),
        model_policy: ModelPolicy::default(),
        default_context_policy: Default::default(),
        default_capabilities: Default::default(),
        output_events: vec!["analysis.completed".into()],
    };

    // Agentic state that references the agent and has a transition for the
    // declared event (to show a transition *would* fire if the event were valid).
    let mut work_state = leaf_state("work", StateType::Agentic);
    work_state.agent = Some(AgentRef {
        id: agent_id.clone(),
        version: agent_ver.clone(),
    });
    work_state.on.insert(
        "analysis.completed".into(),
        vec![TransitionSpec {
            target: "end".into(),
            guard: None,
            priority: 0,
            actions: vec![],
            kind: Default::default(),
        }],
    );

    // Also add a transition for the undeclared event — the check is on
    // output_events, NOT on the transition table, so this must NOT fire.
    work_state.on.insert(
        "undeclared.event".into(),
        vec![TransitionSpec {
            target: "end".into(),
            guard: None,
            priority: 0,
            actions: vec![],
            kind: Default::default(),
        }],
    );

    let states = vec![work_state, leaf_state("end", StateType::Final)];

    let mut doc = base_doc("wf-f6-output-validation", states, "work");
    doc.agents = vec![agent_def];

    let compiled = Arc::new(compile(doc).expect("compile"));

    // Actor emits an event type NOT in output_events.
    let actor: Arc<dyn AgentActor> = Arc::new(ScriptedAgentActor::emit(
        "undeclared.event",
        serde_json::Value::Null,
    ));
    let actors = HashMap::from([(StateId::new("work"), actor)]);

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst =
        WorkflowInstance::new(RunId::new("r-f6"), compiled, broker, sink.clone(), actors);

    inst.start().await.expect("start");
    // Drive a few steps — the activity completes immediately via ScriptedAgentActor.
    for _ in 0..20 {
        if inst.step().await.expect("step") {
            break;
        }
        tokio::task::yield_now().await;
    }

    assert_eq!(
        inst.status,
        RunStatus::Failed,
        "undeclared output must fail instead of leaving the run stuck"
    );

    let payloads = sink.payloads().await;

    // ActivityInvalidOutput must have been emitted.
    let invalid_output_emitted = payloads.iter().any(|p| {
        matches!(p, RuntimeEventPayload::ActivityInvalidOutput {
            state_id, event_type
        } if state_id.0 == "work" && event_type == "undeclared.event")
    });
    assert!(
        invalid_output_emitted,
        "ActivityInvalidOutput must be emitted for undeclared event type"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunFailed { .. })),
        "undeclared output without a retry/failure transition must fail the run"
    );

    // No transition to `end` should have fired.
    let transitioned_to_end = payloads
        .iter()
        .any(|p| matches!(p, RuntimeEventPayload::TransitionSelected { to, .. } if to.0 == "end"));
    assert!(
        !transitioned_to_end,
        "transition to end must NOT fire for undeclared event"
    );
}

// ── E4: WorkflowData RON round-trip tests ────────────────────────────────────

/// 30. RON → ron::Value → with_workflow_data → CEL guard reads bool field.
///
/// The `data.approved` field is `true` in RON; the guard `data.approved == true`
/// must route the workflow to `approved`, not `rejected`.
#[tokio::test]
async fn ron_bool_field_routes_correctly() {
    // Two guarded transitions on the same event, in priority order.
    let mut start = leaf_state("start", StateType::Atomic);
    start.on.insert(
        "decide".into(),
        vec![
            TransitionSpec {
                target: "approved".into(),
                guard: Some("data.approved == true".into()),
                priority: 0,
                actions: vec![],
                kind: Default::default(),
            },
            TransitionSpec {
                target: "rejected".into(),
                guard: Some("data.approved == false".into()),
                priority: 1,
                actions: vec![],
                kind: Default::default(),
            },
        ],
    );
    let mut doc = base_doc(
        "wf-e4-bool",
        vec![
            start,
            leaf_state("approved", StateType::Final),
            leaf_state("rejected", StateType::Final),
        ],
        "start",
    );
    doc.data_schema
        .fields
        .insert("approved".into(), "bool".into());
    let compiled = Arc::new(compile(doc).expect("compile"));

    // Build workflow data: approved = true
    let data: ron::Value = ron::from_str(r#"{"approved": true}"#).expect("ron parse");

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-e4-bool"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    )
    .with_workflow_data(data);

    inst.start().await.expect("start");
    inst.send("decide", serde_json::Value::Null);
    let status = inst.run_to_completion().await.expect("run");

    assert_eq!(status, RunStatus::Completed);

    let payloads = sink.payloads().await;
    let went_to_approved = payloads.iter().any(
        |p| matches!(p, RuntimeEventPayload::TransitionSelected { to, .. } if to.0 == "approved"),
    );
    assert!(
        went_to_approved,
        "data.approved==true should route to `approved`"
    );
}

/// 31. RON map with integer and string fields are readable in CEL guards.
///
/// `data.score >= 80` must route to `pass` when score is 95.
#[tokio::test]
async fn ron_int_and_string_fields_readable_in_guard() {
    let mut start = leaf_state("start", StateType::Atomic);
    start.on.insert(
        "check".into(),
        vec![
            TransitionSpec {
                target: "pass".into(),
                guard: Some("data.score >= 80".into()),
                priority: 0,
                actions: vec![],
                kind: Default::default(),
            },
            TransitionSpec {
                target: "fail".into(),
                guard: Some("data.score < 80".into()),
                priority: 1,
                actions: vec![],
                kind: Default::default(),
            },
        ],
    );
    let mut doc = base_doc(
        "wf-e4-int-str",
        vec![
            start,
            leaf_state("pass", StateType::Final),
            leaf_state("fail", StateType::Final),
        ],
        "start",
    );
    doc.data_schema.fields.insert("score".into(), "u32".into());
    doc.data_schema
        .fields
        .insert("tier".into(), "String".into());
    let compiled = Arc::new(compile(doc).expect("compile"));

    let data: ron::Value = ron::from_str(r#"{"score": 95, "tier": "gold"}"#).expect("ron parse");

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-e4-int"),
        compiled,
        broker,
        sink.clone(),
        HashMap::new(),
    )
    .with_workflow_data(data);

    inst.start().await.expect("start");
    inst.send("check", serde_json::Value::Null);
    let status = inst.run_to_completion().await.expect("run");

    assert_eq!(status, RunStatus::Completed);

    let payloads = sink.payloads().await;
    let went_to_pass = payloads
        .iter()
        .any(|p| matches!(p, RuntimeEventPayload::TransitionSelected { to, .. } if to.0 == "pass"));
    assert!(
        went_to_pass,
        "data.score>=80 should route to `pass` when score=95"
    );
}

/// 32. RON round-trip: ron::Value serialises to JSON and back without data loss.
///
/// This is a pure serialisation test: `ron::Value` → `serde_json::Value` → back
/// to `ron::Value` (via JSON). Verifies the serde_json bridge used in
/// `evaluate_guard` preserves field values.
#[test]
fn ron_value_round_trips_through_json() {
    // Map with bool, integer, and string values.
    let original: ron::Value =
        ron::from_str(r#"{"active": true, "count": 42, "label": "hello"}"#).expect("ron parse");

    // Simulate the path taken by evaluate_guard.
    let json_val = serde_json::to_value(&original).expect("ron→json");
    let back: ron::Value = serde_json::from_value(json_val).expect("json→ron");

    // Field values must survive the round-trip.
    if let ron::Value::Map(map) = &back {
        let find = |key: &str| -> Option<&ron::Value> {
            map.iter()
                .find(|(k, _)| matches!(k, ron::Value::String(s) if s == key))
                .map(|(_, v)| v)
        };
        assert_eq!(
            find("active"),
            Some(&ron::Value::Bool(true)),
            "bool field must survive"
        );
        assert_eq!(
            find("label"),
            Some(&ron::Value::String("hello".into())),
            "string field must survive"
        );
        // Integer 42 may round-trip as Number::Integer(42) or Number::Float(42.0)
        // depending on the JSON deserializer; into_f64() covers both variants.
        let count_val = find("count").expect("count field must be present");
        if let ron::Value::Number(n) = *count_val {
            let f = n.into_f64();
            assert!((f - 42.0).abs() < 1e-9, "count must equal 42, got {f}");
        } else {
            panic!("count must be a Number, got {count_val:?}");
        }
    } else {
        panic!("expected ron::Value::Map after round-trip, got: {back:?}");
    }
}

// ── F5: Event payload schema validation ──────────────────────────────────────

/// Build an agentic state + matching AgentDefinition for F5 tests.
/// The state declares an `output_schema` requiring the given fields, and the
/// AgentDefinition lists `output_event` so `is_output_event_declared` passes.
fn f5_agentic_state(
    state_id: &str,
    output_event: &str,
    next_state: &str,
    schema_fields: std::collections::HashMap<String, String>,
) -> (StateDefinition, AgentDefinition) {
    use langchart_model::workflow::EventSchema;

    let agent_id = AgentId::new(format!("f5-agent-{state_id}"));
    let agent_ver = AgentVersion::new("1.0.0");

    let agent_def = AgentDefinition {
        id: agent_id.clone(),
        version: agent_ver.clone(),
        description: "f5 test agent".into(),
        system_prompt: "".into(),
        model_policy: ModelPolicy::default(),
        default_context_policy: Default::default(),
        default_capabilities: Default::default(),
        output_events: vec![output_event.into()],
    };

    let mut state = leaf_state(state_id, StateType::Agentic);
    state.agent = Some(AgentRef {
        id: agent_id,
        version: agent_ver,
    });
    state.on.insert(
        output_event.into(),
        vec![TransitionSpec {
            target: next_state.into(),
            guard: None,
            priority: 0,
            actions: vec![],
            kind: Default::default(),
        }],
    );
    state.output_schemas.insert(
        output_event.into(),
        EventSchema {
            fields: schema_fields,
        },
    );

    (state, agent_def)
}

/// 33. An agentic state declares `output_schemas` for its output event.
///     A ScriptedAgentActor emits the event with a **valid** payload (all
///     required fields present and correctly typed). The event must be queued
///     and the run must complete.
///
/// Schema: `{"status": "string", "score": "number"}`
/// Payload: `{"status": "ok", "score": 42}`  ← valid
#[tokio::test]
async fn event_valid_payload_passes_schema() {
    let mut schema_fields = std::collections::HashMap::new();
    schema_fields.insert("status".into(), "string".into());
    schema_fields.insert("score".into(), "number".into());

    let (work, agent_def) = f5_agentic_state("work", "result", "end", schema_fields);

    let states = vec![work, leaf_state("end", StateType::Final)];
    let mut doc = base_doc("wf-f5-valid", states, "work");
    doc.agents = vec![agent_def];
    let compiled = Arc::new(compile(doc).expect("compile"));

    let valid_payload = serde_json::json!({"status": "ok", "score": 42});
    let actor: Arc<dyn AgentActor> = Arc::new(ScriptedAgentActor::emit("result", valid_payload));
    let actors = HashMap::from([(StateId::new("work"), actor)]);

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-f5-valid"),
        compiled,
        broker,
        sink.clone(),
        actors,
    );

    inst.start().await.expect("start");
    for _ in 0..20 {
        if inst.step().await.expect("step") {
            break;
        }
        tokio::task::yield_now().await;
    }

    assert_eq!(
        inst.status,
        RunStatus::Completed,
        "valid payload should let the run complete"
    );

    // No ActivityInvalidOutput should have been emitted.
    let payloads = sink.payloads().await;
    let had_invalid = payloads
        .iter()
        .any(|p| matches!(p, RuntimeEventPayload::ActivityInvalidOutput { .. }));
    assert!(
        !had_invalid,
        "valid payload must not emit ActivityInvalidOutput"
    );
}

/// 34. A state with an output schema rejects a payload that has an incorrect
///     field type. The run must NOT complete and ActivityInvalidOutput is emitted.
///
/// Schema: `{"count": "number"}`
/// Payload: `{"count": "not-a-number"}`  ← wrong type
#[tokio::test]
async fn event_wrong_field_type_rejected_by_schema() {
    let mut schema_fields = std::collections::HashMap::new();
    schema_fields.insert("count".into(), "number".into());

    let (work, agent_def) = f5_agentic_state("work", "done", "end", schema_fields);

    let states = vec![work, leaf_state("end", StateType::Final)];
    let mut doc = base_doc("wf-f5-badtype", states, "work");
    doc.agents = vec![agent_def];
    let compiled = Arc::new(compile(doc).expect("compile"));

    // Payload has `count` as a string, not a number.
    let bad_payload = serde_json::json!({"count": "not-a-number"});
    let actor: Arc<dyn AgentActor> = Arc::new(ScriptedAgentActor::emit("done", bad_payload));
    let actors = HashMap::from([(StateId::new("work"), actor)]);

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-f5-badtype"),
        compiled,
        broker,
        sink.clone(),
        actors,
    );

    inst.start().await.expect("start");
    for _ in 0..20 {
        if inst.step().await.expect("step") {
            break;
        }
        tokio::task::yield_now().await;
    }

    assert_eq!(
        inst.status,
        RunStatus::Failed,
        "invalid payload must fail instead of leaving the run stuck"
    );

    let payloads = sink.payloads().await;
    let invalid_emitted = payloads.iter().any(|p| {
        matches!(p, RuntimeEventPayload::ActivityInvalidOutput {
            state_id, event_type
        } if state_id.0 == "work" && event_type == "done")
    });
    assert!(
        invalid_emitted,
        "ActivityInvalidOutput must be emitted for wrong field type"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunFailed { .. })),
        "invalid payload without a recovery transition must fail the run"
    );

    let transitioned = payloads
        .iter()
        .any(|p| matches!(p, RuntimeEventPayload::TransitionSelected { to, .. } if to.0 == "end"));
    assert!(
        !transitioned,
        "transition to end must NOT fire for payload with wrong field type"
    );
}

/// 35. A state with an output schema rejects a payload that is missing a
///     required field. ActivityInvalidOutput is emitted and the run fails.
///
/// Schema: `{"name": "string", "age": "number"}`
/// Payload: `{"name": "Alice"}`  ← missing `age`
#[tokio::test]
async fn event_missing_required_field_rejected_by_schema() {
    let mut schema_fields = std::collections::HashMap::new();
    schema_fields.insert("name".into(), "string".into());
    schema_fields.insert("age".into(), "number".into());

    let (work, agent_def) = f5_agentic_state("work", "user.created", "end", schema_fields);

    let states = vec![work, leaf_state("end", StateType::Final)];
    let mut doc = base_doc("wf-f5-missingfield", states, "work");
    doc.agents = vec![agent_def];
    let compiled = Arc::new(compile(doc).expect("compile"));

    // Missing `age` field.
    let partial_payload = serde_json::json!({"name": "Alice"});
    let actor: Arc<dyn AgentActor> =
        Arc::new(ScriptedAgentActor::emit("user.created", partial_payload));
    let actors = HashMap::from([(StateId::new("work"), actor)]);

    let sink = Arc::new(VecSink::default());
    let broker = bare_broker(sink.clone());
    let mut inst = WorkflowInstance::new(
        RunId::new("r-f5-missing"),
        compiled,
        broker,
        sink.clone(),
        actors,
    );

    inst.start().await.expect("start");
    for _ in 0..20 {
        if inst.step().await.expect("step") {
            break;
        }
        tokio::task::yield_now().await;
    }

    assert_eq!(
        inst.status,
        RunStatus::Failed,
        "missing-field payload must fail instead of leaving the run stuck"
    );

    let payloads = sink.payloads().await;
    let invalid_emitted = payloads.iter().any(|p| {
        matches!(p, RuntimeEventPayload::ActivityInvalidOutput {
            state_id, event_type
        } if state_id.0 == "work" && event_type == "user.created")
    });
    assert!(
        invalid_emitted,
        "ActivityInvalidOutput must be emitted for missing required field"
    );
    assert!(
        payloads
            .iter()
            .any(|p| matches!(p, RuntimeEventPayload::RunFailed { .. })),
        "missing required output must fail the run"
    );

    let transitioned = payloads
        .iter()
        .any(|p| matches!(p, RuntimeEventPayload::TransitionSelected { to, .. } if to.0 == "end"));
    assert!(
        !transitioned,
        "transition to end must NOT fire for payload missing required field"
    );
}

#[tokio::test]
async fn engine_step_error_emits_terminal_run_failure() {
    use langchart_adapters::context::{ContextError, ContextResolver, ContextView};

    struct FailingContextResolver;

    #[async_trait]
    impl ContextResolver for FailingContextResolver {
        async fn resolve(
            &self,
            _policy: &langchart_model::policy::ContextPolicy,
            _run_id: &RunId,
        ) -> Result<ContextView, ContextError> {
            Err(ContextError::Stage {
                stage: "test",
                message: "context unavailable".into(),
            })
        }
    }

    let states = vec![
        with_transition(leaf_state("idle", StateType::Atomic), "go", "work", None),
        agentic_state("work", "work.done", "end"),
        leaf_state("end", StateType::Final),
    ];
    let sink = Arc::new(VecSink::default());
    let engine = RuntimeEngine::new(EngineAdapters {
        llm: Arc::new(NoopLlm),
        mcp: Arc::new(NoopMcp),
        memory: Arc::new(NoopMemory),
        secrets: Arc::new(HostMapSecretsAdapter::empty()),
        event_sink: sink.clone(),
        checkpoint_store: None,
        workflow_repo: None,
        event_source: None,
        artifact_store: None,
    })
    .with_context_resolver(Arc::new(FailingContextResolver));
    let actors = HashMap::from([(
        StateId::new("work"),
        Arc::new(ScriptedAgentActor::emit(
            "work.done",
            serde_json::Value::Null,
        )) as Arc<dyn AgentActor>,
    )]);
    let run_id = engine
        .start(base_doc("wf-step-error", states, "idle"), actors)
        .await
        .expect("start");

    engine
        .send(&run_id, "go", serde_json::Value::Null)
        .await
        .expect("queue event");

    for _ in 0..50 {
        if sink.payloads().await.iter().any(|payload| {
            matches!(
                payload,
                RuntimeEventPayload::RunFailed { message }
                    if message.contains("context unavailable")
            )
        }) {
            return;
        }
        tokio::task::yield_now().await;
    }

    panic!("a step error must emit RunFailed before the engine removes the run");
}

#[tokio::test]
async fn dropping_instance_aborts_a_live_activity() {
    use std::sync::atomic::{AtomicBool, Ordering};

    struct DropAwareActor {
        started: Arc<AtomicBool>,
        cancelled: Arc<AtomicBool>,
    }

    struct CancellationGuard(Arc<AtomicBool>);

    impl Drop for CancellationGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl AgentActor for DropAwareActor {
        async fn run(
            &self,
            _invocation: langchart_runtime::instance::AgentInvocation,
            _envelope: langchart_runtime::broker::CapabilityEnvelope,
            _broker: Arc<CapabilityBroker>,
        ) -> Result<
            langchart_runtime::instance::AgentOutputEvent,
            langchart_runtime::instance::AgentError,
        > {
            let _guard = CancellationGuard(self.cancelled.clone());
            self.started.store(true, Ordering::SeqCst);
            std::future::pending().await
        }
    }

    let started = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::new(AtomicBool::new(false));
    let sink = Arc::new(VecSink::default());
    let compiled = Arc::new(
        compile(base_doc(
            "wf-drop-aborts",
            vec![
                agentic_state("work", "work.done", "end"),
                leaf_state("end", StateType::Final),
            ],
            "work",
        ))
        .expect("compile"),
    );
    let mut instance = WorkflowInstance::new(
        RunId::new("r-drop-aborts"),
        compiled,
        bare_broker(sink.clone()),
        sink,
        HashMap::from([(
            StateId::new("work"),
            Arc::new(DropAwareActor {
                started: started.clone(),
                cancelled: cancelled.clone(),
            }) as Arc<dyn AgentActor>,
        )]),
    );

    instance.start().await.expect("start");
    for _ in 0..50 {
        if started.load(Ordering::SeqCst) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(started.load(Ordering::SeqCst), "activity did not start");

    drop(instance);
    for _ in 0..50 {
        if cancelled.load(Ordering::SeqCst) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        cancelled.load(Ordering::SeqCst),
        "dropping a run must not detach its activity"
    );
}

#[tokio::test]
async fn actor_spawned_task_loses_broker_authority_after_invocation() {
    use langchart_adapters::memory::{MemoryScope, QueryMode};
    use langchart_runtime::broker::BrokerError;
    use tokio::sync::{Notify, oneshot};

    struct EscapingActor {
        gate: Arc<Notify>,
        result: Mutex<Option<oneshot::Sender<bool>>>,
    }

    #[async_trait]
    impl AgentActor for EscapingActor {
        async fn run(
            &self,
            _invocation: langchart_runtime::instance::AgentInvocation,
            envelope: langchart_runtime::broker::CapabilityEnvelope,
            broker: Arc<CapabilityBroker>,
        ) -> Result<
            langchart_runtime::instance::AgentOutputEvent,
            langchart_runtime::instance::AgentError,
        > {
            let gate = self.gate.clone();
            let result = self.result.lock().await.take().expect("result sender");
            tokio::spawn(async move {
                gate.notified().await;
                let error = broker
                    .memory_search(
                        &envelope,
                        MemoryQuery {
                            scope: MemoryScope::Global,
                            mode: QueryMode::Keyword {
                                text: "after completion".into(),
                            },
                            limit: 1,
                            min_score: None,
                        },
                    )
                    .await
                    .expect_err("completed invocation must lose authority");
                let _ = result.send(matches!(error, BrokerError::ExpiredCapabilityEnvelope));
            });

            Ok(langchart_runtime::instance::AgentOutputEvent {
                event_type: "work.done".into(),
                payload: serde_json::Value::Null,
            })
        }
    }

    let gate = Arc::new(Notify::new());
    let (result_tx, result_rx) = oneshot::channel();
    let actor: Arc<dyn AgentActor> = Arc::new(EscapingActor {
        gate: gate.clone(),
        result: Mutex::new(Some(result_tx)),
    });
    let sink = Arc::new(VecSink::default());
    let compiled = Arc::new(
        compile(base_doc(
            "wf-expiring-envelope",
            vec![
                agentic_state("work", "work.done", "end"),
                leaf_state("end", StateType::Final),
            ],
            "work",
        ))
        .expect("compile"),
    );
    let mut instance = WorkflowInstance::new(
        RunId::new("r-expiring-envelope"),
        compiled,
        bare_broker(sink.clone()),
        sink,
        HashMap::from([(StateId::new("work"), actor)]),
    );

    instance.start().await.expect("start");
    for _ in 0..50 {
        if instance.step().await.expect("step") {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(instance.status, RunStatus::Completed);

    gate.notify_one();
    let rejected = tokio::time::timeout(std::time::Duration::from_secs(1), result_rx)
        .await
        .expect("spawned task result timeout")
        .expect("spawned task result channel");
    assert!(rejected, "expired envelope returned the wrong broker error");
}

#[tokio::test]
async fn suspend_waits_for_an_admitted_detached_broker_call_to_cancel() {
    use langchart_adapters::memory::{MemoryScope, QueryMode};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct BlockingMemory {
        started: Arc<AtomicBool>,
        cancelled: Arc<AtomicBool>,
    }

    struct CancellationGuard(Arc<AtomicBool>);

    impl Drop for CancellationGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl MemoryAdapter for BlockingMemory {
        async fn store(&self, _record: MemoryRecord) -> Result<MemoryId, MemoryError> {
            Ok(MemoryId("unused".into()))
        }

        async fn search(&self, _query: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError> {
            let _guard = CancellationGuard(self.cancelled.clone());
            self.started.store(true, Ordering::SeqCst);
            std::future::pending().await
        }

        async fn get(&self, _id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
            Ok(None)
        }

        async fn delete(&self, _id: &MemoryId) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    struct DetachedSearchActor;

    #[async_trait]
    impl AgentActor for DetachedSearchActor {
        async fn run(
            &self,
            _invocation: langchart_runtime::instance::AgentInvocation,
            envelope: langchart_runtime::broker::CapabilityEnvelope,
            broker: Arc<CapabilityBroker>,
        ) -> Result<
            langchart_runtime::instance::AgentOutputEvent,
            langchart_runtime::instance::AgentError,
        > {
            tokio::spawn(async move {
                let _ = broker
                    .memory_search(
                        &envelope,
                        MemoryQuery {
                            scope: MemoryScope::Global,
                            mode: QueryMode::Keyword {
                                text: "block until revoked".into(),
                            },
                            limit: 1,
                            min_score: None,
                        },
                    )
                    .await;
            });
            std::future::pending().await
        }
    }

    let started = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::new(AtomicBool::new(false));
    let sink = Arc::new(VecSink::default());
    let broker = Arc::new(CapabilityBroker::new(
        Arc::new(NoopLlm),
        Arc::new(NoopMcp),
        Arc::new(BlockingMemory {
            started: started.clone(),
            cancelled: cancelled.clone(),
        }),
        Arc::new(HostMapSecretsAdapter::empty()),
        sink.clone(),
    ));
    let compiled = Arc::new(
        compile(base_doc(
            "wf-suspend-drains",
            vec![
                with_transition(
                    agentic_state("work", "work.done", "end"),
                    "leave",
                    "end",
                    None,
                ),
                leaf_state("end", StateType::Final),
            ],
            "work",
        ))
        .expect("compile"),
    );
    let mut instance = WorkflowInstance::new(
        RunId::new("r-suspend-drains"),
        compiled,
        broker,
        sink,
        HashMap::from([(
            StateId::new("work"),
            Arc::new(DetachedSearchActor) as Arc<dyn AgentActor>,
        )]),
    );

    instance.start().await.expect("start");
    for _ in 0..50 {
        if started.load(Ordering::SeqCst) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        started.load(Ordering::SeqCst),
        "detached call did not start"
    );

    instance.suspend().await.expect("suspend");
    assert!(
        cancelled.load(Ordering::SeqCst),
        "suspend returned before the admitted broker call was cancelled"
    );

    started.store(false, Ordering::SeqCst);
    cancelled.store(false, Ordering::SeqCst);
    instance.resume().await.expect("resume");
    for _ in 0..50 {
        if started.load(Ordering::SeqCst) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(started.load(Ordering::SeqCst), "resumed call did not start");

    instance.send("leave", serde_json::Value::Null);
    instance.step().await.expect("process forced state exit");
    assert_eq!(instance.status, RunStatus::Completed);
    assert!(
        cancelled.load(Ordering::SeqCst),
        "state exit completed before the admitted broker call was cancelled"
    );
}
