# Embedding Guide

**langchart v0.1.0**
**Audience:** Application developers integrating langchart into a host application
**Status:** Current (reflects implementation as of spec §0.4)

---

## Overview

`langchart` is a library — it does not run as a standalone server. You embed it inside your Rust application and provide concrete implementations of its adapter traits. This guide walks through every step: wiring adapters, writing agents, defining workflows, launching runs, and operating a running system.

---

## 1. Dependency Setup

Add the following to your `Cargo.toml`:

```toml
[dependencies]
langchart          = { path = "../crates/langchart" }         # re-export facade
langchart-runtime  = { path = "../crates/langchart-runtime" } # engine + broker
langchart-adapters = { path = "../crates/langchart-adapters" } # adapter traits
langchart-model    = { path = "../crates/langchart-model" }   # types

# Tokio is required; the engine runs on Tokio.
tokio = { version = "1", features = ["full"] }
```

If you need WASM compilation, add `langchart-wasm` for the browser-facing validation API.

---

## 2. Adapter Traits

Every external concern is abstracted through a trait. You implement (or select) a concrete adapter for each:

| Trait | Location | Abstracts |
|---|---|---|
| `LlmAdapter` | `langchart_adapters::llm` | LLM completions (OpenAI, Anthropic, local) |
| `McpAdapter` | `langchart_adapters::mcp` | MCP tool and resource calls |
| `MemoryAdapter` | `langchart_adapters::memory` | Long-term memory storage / retrieval |
| `CheckpointStore` | `langchart_adapters::checkpoint` | Run snapshot persistence |
| `ArtifactStore` | `langchart_adapters::artifact` | Versioned artifact reads and proposals |
| `EventSink` | `langchart_adapters::event` | Observable runtime events (audit log) |
| `SecretsAdapter` | `langchart_adapters::secrets` | Credential resolution (never logged) |

The engine requires `LlmAdapter`, `McpAdapter`, `MemoryAdapter`, `SecretsAdapter`, and `EventSink`. The rest are optional and can be no-ops.

### Minimal no-op sink (for development)

```rust
use async_trait::async_trait;
use langchart_adapters::event::{EventSink, EventSinkError, RuntimeEvent};

pub struct NoopSink;

#[async_trait]
impl EventSink for NoopSink {
    async fn append(&self, _event: RuntimeEvent) -> Result<(), EventSinkError> {
        Ok(())
    }
}
```

### Redacting sensitive events before logging

Wrap any `EventSink` with `RedactingEventSink` to apply a `RedactionPolicy`:

```rust
use std::sync::Arc;
use langchart_adapters::event::RedactingEventSink;
use langchart_model::policy::RedactionPolicy;

let inner: Arc<dyn EventSink> = Arc::new(MyDatabaseSink::new(...));
let sink = Arc::new(RedactingEventSink::new(
    inner,
    RedactionPolicy {
        redact_tool_arguments: true,
        redact_memory_queries: true,
        ..Default::default()
    },
));
```

Redaction happens in the adapter layer — the downstream sink never sees raw tool arguments or memory queries.

---

## 3. Wiring the Engine

```rust
use std::{collections::HashMap, sync::Arc};
use langchart_runtime::{EngineAdapters, RuntimeEngine};

let engine = RuntimeEngine::new(EngineAdapters {
    llm:        Arc::new(MyLlmAdapter::new(...)),
    mcp:        Arc::new(MyMcpClient::new(...)),
    memory:     Arc::new(MyMemoryStore::new(...)),
    secrets:    Arc::new(langchart_adapters::secrets::HostMapSecretsAdapter::from_env()),
    event_sink: Arc::new(MyEventSink::new(...)),
});
```

`RuntimeEngine` is `Send + Sync` and may be shared freely across threads (it uses `Arc<Mutex<...>>` internally for the run registry).

### HostMapSecretsAdapter

The built-in `HostMapSecretsAdapter` resolves secret references from an in-process `HashMap`. Use it to inject environment variables or vault-loaded credentials without a full `SecretsAdapter` implementation:

```rust
use langchart_adapters::secrets::HostMapSecretsAdapter;

let secrets = HostMapSecretsAdapter::from_map(HashMap::from([
    ("openai_api_key".into(), std::env::var("OPENAI_API_KEY").unwrap()),
]));
```

Secrets are **never** serialized to checkpoints, event logs, or workflow state.

---

## 4. Defining Workflow Documents

A workflow document is a `WorkflowDocument` struct (or YAML/JSON that deserializes to one):

```json
{
  "schema_version": "1.0.0",
  "id": "content-review",
  "version": "0.1.0",
  "name": "Content Review",
  "initial": "extract",
  "states": [
    {
      "id": "extract",
      "name": "Extract",
      "type": "agentic",
      "agent": { "id": "extractor", "version": "0.1.0" },
      "prompt": "Extract key facts from the provided document.",
      "on": {
        "extraction.done": [{ "target": "review", "priority": 0, "actions": [] }]
      }
    },
    {
      "id": "review",
      "name": "Review",
      "type": "human",
      "on": {
        "review.approved": [{ "target": "end", "priority": 0, "actions": [] }],
        "review.rejected":  [{ "target": "extract", "priority": 0, "actions": [] }]
      }
    },
    {
      "id": "end",
      "name": "End",
      "type": "final",
      "on": {}
    }
  ]
}
```

### Validation and compilation

Always validate before starting a run. This catches bad transition targets, unreachable states, missing agent refs, and policy violations:

```rust
use langchart_model::validation::compile;

let compiled = compile(workflow_document)?;
// `compiled` is a `CompiledWorkflow` — all references are verified.
```

The WASM bindings (`langchart-wasm`) expose the same validation API for browser-side authoring tools.

---

## 5. Implementing Agents

An agent is any type implementing the `AgentActor` trait:

```rust
use async_trait::async_trait;
use langchart_runtime::instance::{AgentActor, AgentError, AgentInvocation, AgentOutputEvent};
use langchart_runtime::broker::{CapabilityBroker, CapabilityEnvelope};

pub struct MyAgent;

#[async_trait]
impl AgentActor for MyAgent {
    async fn run(
        &self,
        invocation: AgentInvocation,
        mut envelope: CapabilityEnvelope,
        broker: std::sync::Arc<CapabilityBroker>,
    ) -> Result<AgentOutputEvent, AgentError> {
        // Call the LLM through the broker — budget enforcement is automatic.
        let response = broker.call_llm(
            &mut envelope,
            langchart_adapters::llm::LlmRequest {
                messages: vec![/* ... */],
                model_policy: invocation.instructions.into(),
                tools: vec![],
            },
        ).await?;

        // Produce the output event that drives the statechart forward.
        Ok(AgentOutputEvent {
            event_type: "extraction.done".into(),
            payload: serde_json::json!({ "facts": [] }),
        })
    }
}
```

**Key rules:**
- All LLM, MCP, memory, and artifact calls **must** go through `broker`. Direct adapter or HTTP calls bypass all policy enforcement.
- `envelope` carries the remaining budget; the broker decrements it on each call and rejects over-budget calls.
- Return exactly one `AgentOutputEvent` whose `event_type` must match a declared `on` transition in the state definition. Undeclared event types are logged as `ActivityInvalidOutput` and silently dropped.

---

## 6. Starting and Driving Runs

```rust
use std::collections::HashMap;
use langchart_model::id::StateId;
use langchart_runtime::instance::AgentActor;
use std::sync::Arc;

let mut actors: HashMap<StateId, Arc<dyn AgentActor>> = HashMap::new();
actors.insert(StateId::new("extract"), Arc::new(MyAgent));

let run_id = engine.start(workflow_document, actors).await?;

// Send an external event (e.g., human approval):
engine.send(&run_id, "review.approved", serde_json::json!({})).await?;

// Inspect the current state:
let snapshot = engine.inspect(&run_id).await?;
println!("active states: {:?}", snapshot.active_states);

// Suspend / resume:
engine.suspend(&run_id).await?;
engine.resume(&run_id).await?;

// Cancel:
engine.cancel(&run_id).await?;
```

---

## 7. Token and Resource Budgets

Token budgets are configured per-state via `ExecutionLimits`:

```json
{
  "id": "extract",
  "type": "agentic",
  "limits": {
    "max_turns": 5,
    "max_tool_calls": 10,
    "timeout": 120,
    "max_tokens_total": 50000
  }
}
```

The broker enforces these automatically. When a budget is at 80% utilization it emits `BudgetWarning`; when exhausted it emits `BudgetExhausted` and returns `BrokerError::TokenBudgetExhausted`.

---

## 8. Parallel States

Parallel states launch multiple orthogonal regions concurrently. Each region's initial state is entered at the same time:

```json
{
  "id": "analysis",
  "type": "parallel",
  "completion": { "mode": "all" },
  "regions": [
    {
      "id": "text-region",
      "initial": "text-analysis",
      "states": [...]
    },
    {
      "id": "image-region",
      "initial": "image-analysis",
      "states": [...]
    }
  ],
  "on": {
    "parallel.completed": { "target": "synthesis" }
  }
}
```

Completion modes:
- `All` — all regions must reach a final state (default)
- `Any` — first region to complete triggers exit
- `Quorum { n }` — at least `n` regions must complete
- `Guard { expr }` — a CEL expression evaluated with `completed` and `total` variables
- `Manual` — only an explicit external event causes exit

---

## 9. History States

Add `"history": "Shallow"` or `"history": "Deep"` to a compound or parallel state to save and restore its configuration on re-entry:

```json
{
  "id": "editing",
  "type": "compound",
  "history": "deep",
  "initial": "idle",
  "states": [...]
}
```

Call `instance.restore_history(&state_id)` to explicitly restore (this happens automatically when a `History` pseudo-state is the transition target).

---

## 10. Testing and Simulation

Use `WorkflowSimulator` from `langchart_runtime::simulation` for deterministic in-process testing:

```rust
use std::sync::Arc;
use langchart_runtime::simulation::{WorkflowSimulator, SimActorMap};
use langchart_runtime::instance::ScriptedAgentActor;
use langchart_model::validation::compile;

let compiled = Arc::new(compile(document)?);

let result = WorkflowSimulator::new(compiled)
    .with_actors(
        SimActorMap::new()
            .add("extract", ScriptedAgentActor::emit("extraction.done", json!({})))
            .add("other-state", ScriptedAgentActor::fail("intentional failure"))
    )
    .run()
    .await?;

assert_eq!(result.status, RunStatus::Completed);
assert!(result.has_payload(|p| matches!(p, RunCompleted)));
```

`ScriptedAgentActor::emit` always emits the specified event.  
`ScriptedAgentActor::fail` simulates an actor failure, which emits `ActivityFailed` and injects an `activity.failed` event into the queue.

For stuck workflows (failure with no failure transition), use `run_bounded(n)` to avoid infinite loops.

### Trace replay

Capture a trace and replay it for regression testing:

```rust
use langchart_runtime::replay::TraceReplayer;

let replayed = TraceReplayer::new(compiled.clone(), original_run_events)
    .with_actors(actors)
    .replay()
    .await?;

assert_eq!(replayed.final_status, RunStatus::Completed);
```

### What-if forking

Fork from a snapshot to explore alternative continuations:

```rust
use langchart_runtime::replay::{fork_instance, ForkRequest};

let mut forked = fork_instance(compiled, broker, ForkRequest::new(snapshot)).await?;
forked.send("approval.overridden", json!({}));
// drive the forked instance...
```

---

## 11. Observable Events

Every action the engine takes emits a typed `RuntimeEvent` to the `EventSink`. Subscribe to these for monitoring, audit, and UI updates:

| Payload kind | When emitted |
|---|---|
| `RunStarted` / `RunCompleted` / `RunFailed` | Run lifecycle |
| `StateEntered` / `StateExited` | State transitions |
| `TransitionSelected` | A transition fires |
| `ActivityStarted` / `ActivityCompleted` / `ActivityFailed` | Agent invocations |
| `LlmRequest` / `LlmResponse` | Model calls (via broker) |
| `ToolRequest` / `ToolResponse` / `ToolRejected` | MCP calls (via broker) |
| `BudgetWarning` / `BudgetExhausted` | Budget enforcement |
| `ParallelRegionEntered` / `ParallelCompleted` | Parallel states |
| `HistorySaved` / `HistoryRestored` | History pseudo-states |
| `SubworkflowStarted` / `SubworkflowCompleted` | Nested workflows |
| `HumanInputRequested` | Human states waiting for input |
| `EventUnhandled` | Events with no matching transition |

---

## 12. Security Model

The `CapabilityBroker` is the security kernel. Key guarantees:

1. **No bypass path.** Every LLM, MCP, and memory call must go through the broker. Direct calls bypass enforcement.
2. **Allowlist-only tool access.** Tools not in a state's `CapabilityPolicy.mcp` allowlist are rejected with `ToolRejected` events.
3. **Secrets never logged.** `SecretsAdapter.get()` returns an opaque value; the broker never passes secret values to the event sink.
4. **Budget caps respected.** Turn and token budgets are enforced pre-call; exceeding them returns a typed error immediately.
5. **No capability elevation.** A child state's policy is always the intersection with the parent's; a state cannot grant itself permissions the workflow-level policy doesn't allow.

---

## 13. Production Checklist

Before deploying to production:

- [ ] Replace `HostMapSecretsAdapter` with a vault-backed implementation.
- [ ] Implement `CheckpointStore` so runs can be recovered after process restart.
- [ ] Configure `RedactingEventSink` with appropriate `RedactionPolicy` for your compliance requirements.
- [ ] Set `max_tokens_total`, `max_turns`, and `timeout` on every agentic state.
- [ ] Review `CapabilityPolicy.mcp` allowlists — grant only the tools each state actually needs.
- [ ] Wire the `EventSink` to a durable store (database, message bus) for audit.
- [ ] Test all failure paths using `ScriptedAgentActor::fail` in simulation.
- [ ] Run `cargo audit` and `cargo deny` before every release.

---

*Made with IBM Bob*
