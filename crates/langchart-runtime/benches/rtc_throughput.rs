//! Benchmark: RTC loop throughput — atomic state transitions per second.
//!
//! Measures how many complete single-transition runs (start → final) the
//! engine can execute per second with no I/O, no LLM, and no actors.
//!
//! Run:  cargo bench --bench rtc_throughput -p langchart-runtime

use criterion::{Criterion, criterion_group, criterion_main};
use langchart_adapters::{
    event::{EventSink, EventSinkError, RuntimeEvent},
    llm::{LlmAdapter, LlmError, LlmRequest, LlmResponse},
    mcp::{McpAdapter, McpError, ResourceContent, ToolDefinition as McpToolDef},
    memory::{MemoryAdapter, MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult},
    secrets::HostMapSecretsAdapter,
};
use langchart_model::{
    id::{IdempotencyKey, RunId, ServerId, ToolName},
    state::{StateDefinition, StateType, TransitionSpec},
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

// ── Helpers ───────────────────────────────────────────────────────────────────

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

fn two_state_doc() -> WorkflowDocument {
    let start = StateDefinition {
        id: "start".into(),
        name: "start".into(),
        state_type: StateType::Atomic,
        on: {
            let mut m = HashMap::new();
            m.insert(
                "go".into(),
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
    let end = StateDefinition {
        id: "end".into(),
        name: "end".into(),
        state_type: StateType::Final,
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
    };
    WorkflowDocument {
        schema_version: "1.0.0".into(),
        id: "bench-rtc".into(),
        version: "0.1.0".into(),
        name: "bench-rtc".into(),
        description: None,
        inputs: vec![],
        outputs: vec![],
        data_schema: Default::default(),
        policy: Default::default(),
        agents: vec![],
        states: vec![start, end],
        initial: "start".into(),
        _editor: serde_json::Value::Null,
    }
}

// ── Benchmark ────────────────────────────────────────────────────────────────

fn rtc_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let compiled = Arc::new(compile(two_state_doc()).expect("compile"));
    let broker = make_broker();

    c.bench_function("rtc_one_transition", |b| {
        b.to_async(&rt).iter(|| async {
            let sink: Arc<dyn EventSink> = Arc::new(DevNull);
            let mut inst = WorkflowInstance::new(
                RunId::new("b1"),
                compiled.clone(),
                broker.clone(),
                sink,
                HashMap::new(),
            );
            inst.start().await.expect("start");
            inst.send("go", serde_json::Value::Null);
            inst.run_to_completion().await.expect("run")
        })
    });
}

criterion_group!(benches, rtc_throughput);
criterion_main!(benches);
