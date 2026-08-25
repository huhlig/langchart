//! Benchmark: CapabilityBroker budget enforcement overhead.
//!
//! Measures the cost of the budget-check path in `call_llm`: creating an
//! envelope, enforcing the turn budget, and receiving the no-op LLM error.
//! I/O is excluded — the LLM adapter returns `Err` immediately.
//!
//! Run:  cargo bench --bench broker_overhead -p langchart-runtime

use criterion::{Criterion, criterion_group, criterion_main};
use langchart_adapters::{
    event::{EventSink, EventSinkError, RuntimeEvent},
    llm::{LlmAdapter, LlmError, LlmRequest, LlmResponse, Message},
    mcp::{McpAdapter, McpError, ResourceContent, ToolDefinition as McpToolDef},
    memory::{MemoryAdapter, MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult},
    secrets::HostMapSecretsAdapter,
};
use langchart_model::{
    id::{IdempotencyKey, InvocationId, RunId, ServerId, StateId, ToolName},
    policy::CapabilityPolicy,
};
use langchart_runtime::broker::{CapabilityBroker, CapabilityEnvelope};
use std::sync::Arc;

// ── No-op stubs ───────────────────────────────────────────────────────────────

struct DevNull;
#[async_trait::async_trait]
impl EventSink for DevNull {
    async fn append(&self, _: RuntimeEvent) -> Result<(), EventSinkError> {
        Ok(())
    }
}

/// LLM that returns an error synchronously without any async work.
struct InstantErrorLlm;
#[async_trait::async_trait]
impl LlmAdapter for InstantErrorLlm {
    async fn complete(&self, _: LlmRequest) -> Result<LlmResponse, LlmError> {
        Err(LlmError::Provider("bench-noop".into()))
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

// ── Benchmark ────────────────────────────────────────────────────────────────

fn broker_overhead(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let sink: Arc<dyn EventSink> = Arc::new(DevNull);
    let broker = Arc::new(CapabilityBroker::new(
        Arc::new(InstantErrorLlm),
        Arc::new(NoopMcp),
        Arc::new(NoopMemory),
        Arc::new(HostMapSecretsAdapter::empty()),
        sink,
    ));

    let run_id = RunId::new("bench-broker");
    let request = LlmRequest {
        model_policy: Default::default(),
        messages: vec![Message::User {
            content: "hello".into(),
        }],
        tools: vec![],
    };

    c.bench_function("broker_rejects_unissued_envelope", |b| {
        b.to_async(&rt).iter(|| async {
            // Publicly constructed envelopes are intentionally inert. Issued
            // envelopes can only be created inside the runtime invocation path.
            let mut envelope = CapabilityEnvelope::new(
                CapabilityPolicy::default(),
                run_id.clone(),
                InvocationId::new("bench-inv"),
                StateId::new("bench-state"),
                1, // max_turns
                0, // max_tool_calls (unused here)
            );
            // Exercise the broker's authority check without adapter I/O.
            let _ = broker.call_llm(&mut envelope, request.clone()).await;
        })
    });
}

criterion_group!(benches, broker_overhead);
criterion_main!(benches);
