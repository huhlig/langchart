//! Benchmark: Parallel state entry time — N=10 and N=100 regions.
//!
//! Measures the time to enter a parallel state with N orthogonal regions.
//! Each region contains a single atomic state. The benchmark covers only
//! the state-entry path; no events are processed after entry.
//!
//! Run:  cargo bench --bench parallel_entry -p langchart-runtime

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use langchart_adapters::{
    event::{EventSink, EventSinkError, RuntimeEvent},
    llm::{LlmAdapter, LlmError, LlmRequest, LlmResponse},
    mcp::{McpAdapter, McpError, ResourceContent, ToolDefinition as McpToolDef},
    memory::{MemoryAdapter, MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult},
    secrets::HostMapSecretsAdapter,
};
use langchart_model::{
    id::{IdempotencyKey, RegionId, RunId, ServerId, StateId, ToolName},
    state::{ParallelCompletion, ParallelRegion, StateDefinition, StateType, TransitionSpec},
    validation::compile,
    workflow::WorkflowDocument,
};
use langchart_runtime::{broker::CapabilityBroker, run::WorkflowInstance};
use std::{collections::HashMap, sync::Arc};

// ── No-op stubs ───────────────────────────────────────────────────────────────

struct DevNull;
#[async_trait::async_trait]
impl EventSink for DevNull {
    async fn append(&self, _: RuntimeEvent) -> Result<(), EventSinkError> {
        Ok(())
    }
}

struct NoopLlm;
#[async_trait::async_trait]
impl LlmAdapter for NoopLlm {
    async fn complete(&self, _: LlmRequest) -> Result<LlmResponse, LlmError> {
        Err(LlmError::Provider("noop".into()))
    }
}

struct NoopMcp;
#[async_trait::async_trait]
impl McpAdapter for NoopMcp {
    async fn call_tool(
        &self,
        _: &ServerId,
        _: &ToolName,
        _: serde_json::Value,
        _: &[langchart_adapters::mcp::McpCredential],
        _: Option<&IdempotencyKey>,
    ) -> Result<serde_json::Value, McpError> {
        Err(McpError::Call("noop".into()))
    }
    async fn list_tools(&self, _: &ServerId) -> Result<Vec<McpToolDef>, McpError> {
        Ok(vec![])
    }
    async fn read_resource(&self, _: &ServerId, _: &str) -> Result<ResourceContent, McpError> {
        Err(McpError::Call("noop".into()))
    }
}

struct NoopMemory;
#[async_trait::async_trait]
impl MemoryAdapter for NoopMemory {
    async fn store(&self, _: MemoryRecord) -> Result<MemoryId, MemoryError> {
        Ok(MemoryId("noop".into()))
    }
    async fn search(&self, _: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError> {
        Ok(vec![])
    }
    async fn get(&self, _: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
        Ok(None)
    }
    async fn delete(&self, _: &MemoryId) -> Result<(), MemoryError> {
        Ok(())
    }
}

fn make_broker() -> Arc<CapabilityBroker> {
    let sink: Arc<dyn EventSink> = Arc::new(DevNull);
    Arc::new(CapabilityBroker::new(
        Arc::new(NoopLlm),
        Arc::new(NoopMcp),
        Arc::new(NoopMemory),
        Arc::new(HostMapSecretsAdapter::empty()),
        sink,
    ))
}

// ── Workflow builder ──────────────────────────────────────────────────────────

fn make_leaf(id: &str) -> StateDefinition {
    StateDefinition {
        id: id.into(),
        name: id.into(),
        state_type: StateType::Atomic,
        on: Default::default(),
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
    }
}

/// Build a workflow with a single parallel state containing `n` regions.
/// Each region has one atomic state. The workflow starts by entering the
/// parallel state immediately (no prior atomic state).
fn parallel_doc(n: usize) -> WorkflowDocument {
    let regions: Vec<ParallelRegion> = (0..n)
        .map(|i| ParallelRegion {
            id: RegionId::new(format!("r{i}")),
            name: format!("Region {i}"),
            initial: StateId::new(format!("task_{i}")),
            states: vec![make_leaf(&format!("task_{i}"))],
        })
        .collect();

    let parallel = StateDefinition {
        id: "par".into(),
        name: "par".into(),
        state_type: StateType::Parallel,
        regions,
        completion: Some(ParallelCompletion::All),
        on: {
            let mut m = HashMap::new();
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

    let done = make_leaf("done");
    let mut done_final = done;
    done_final.state_type = StateType::Final;

    WorkflowDocument {
        schema_version: "1.0.0".into(),
        id: format!("bench-par-{n}").into(),
        version: "0.1.0".into(),
        name: format!("bench-par-{n}"),
        description: None,
        inputs: vec![],
        outputs: vec![],
        data_schema: Default::default(),
        policy: Default::default(),
        agents: vec![],
        states: vec![parallel, done_final],
        initial: "par".into(),
        _editor: serde_json::Value::Null,
    }
}

// ── Benchmark ────────────────────────────────────────────────────────────────

fn parallel_entry(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("parallel_state_entry");

    for &n in &[10_usize, 100_usize] {
        let compiled = Arc::new(compile(parallel_doc(n)).expect("compile"));
        let broker = make_broker();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.to_async(&rt).iter(|| async {
                let sink: Arc<dyn EventSink> = Arc::new(DevNull);
                let mut inst = WorkflowInstance::new(
                    RunId::new("bench-par"),
                    compiled.clone(),
                    broker.clone(),
                    sink,
                    HashMap::new(),
                );
                // start() enters the initial state (the parallel state), spawning
                // all N regions. This is what we are measuring.
                inst.start().await.expect("start");
            })
        });
    }

    group.finish();
}

criterion_group!(benches, parallel_entry);
criterion_main!(benches);
