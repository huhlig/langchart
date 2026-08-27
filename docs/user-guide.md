# langchart User Guide

**langchart v0.1.0**
**Audience:** Workflow authors, integration developers, and anyone using the library for the first time
**Status:** Current (reflects implementation as of spec §0.4)

---

## What is langchart?

`langchart` is a Rust library for defining, validating, executing, and observing long-running **agentic workflows** as hierarchical statecharts. It combines deterministic statechart semantics with bounded agent autonomy.

- A **workflow** is a set of states, transitions, and events encoded in a JSON or YAML document.
- An **agent** (or any asynchronous actor) runs inside a state with a bounded capability envelope: it can only use the tools, memory, and LLM calls you explicitly allow.
- The **runtime** drives state transitions, enforces limits, checkpoints state, and emits observable events for every action.

`langchart` is an embeddable library — it runs inside your Rust application. You provide the LLM, MCP server connections, memory store, and any other adapters your workflow needs.

---

## Quick Start

### 1. Add the dependency

In your workspace or crate `Cargo.toml`:

```toml
[dependencies]
langchart         = { path = "../crates/langchart" }    # re-export facade
langchart-runtime = { path = "../crates/langchart-runtime" }
langchart-adapters = { path = "../crates/langchart-adapters" }
langchart-model   = { path = "../crates/langchart-model" }
tokio = { version = "1", features = ["full"] }
```

### 2. Write a workflow document

Save this as `my-workflow.json`:

```json
{
  "schema_version": "1.0.0",
  "id": "hello-world",
  "version": "0.1.0",
  "name": "Hello World",
  "initial": "greet",
  "states": [
    {
      "id": "greet",
      "name": "Greet",
      "type": "agentic",
      "agent": { "id": "greeter", "version": "0.1.0" },
      "prompt": "Say hello and confirm you are ready.",
      "on": {
        "greeter.done": [{ "target": "done", "priority": 0, "actions": [] }]
      }
    },
    {
      "id": "done",
      "name": "Done",
      "type": "final",
      "on": {}
    }
  ]
}
```

### 3. Validate the workflow

```bash
langchart validate my-workflow.json
```

Or from Rust:

```rust
use langchart_model::validation::{validate, compile};

let doc: langchart_model::workflow::WorkflowDocument =
    serde_json::from_str(include_str!("my-workflow.json"))?;

let diagnostics = validate(&doc);
for d in &diagnostics {
    if d.is_error() { eprintln!("error [{}]: {}", d.code, d.message); }
}

let compiled = compile(doc)?;  // fails if there are errors
```

### 4. Implement an agent

```rust
use async_trait::async_trait;
use langchart_runtime::instance::{AgentActor, AgentError, AgentInvocation, AgentOutputEvent};
use langchart_runtime::broker::CapabilityBroker;
use std::sync::Arc;

pub struct Greeter;

#[async_trait]
impl AgentActor for Greeter {
    async fn run(
        &self,
        _invocation: AgentInvocation,
        _broker: Arc<CapabilityBroker>,
    ) -> Result<AgentOutputEvent, AgentError> {
        println!("Hello! I'm ready.");
        Ok(AgentOutputEvent {
            event_type: "greeter.done".into(),
            payload: serde_json::json!({}),
        })
    }
}
```

### 5. Run the workflow

```rust
use std::{collections::HashMap, sync::Arc};
use langchart_model::id::StateId;
use langchart_runtime::instance::AgentActor;
use langchart_runtime::{EngineAdapters, RuntimeEngine};
use langchart_adapters::secrets::HostMapSecretsAdapter;

// Provide no-op adapters for a quick test.
let engine = RuntimeEngine::new(EngineAdapters {
    llm:        Arc::new(MyLlm::new()),
    mcp:        Arc::new(langchart_runtime::simulation::NoopMcp),  // or your adapter
    memory:     Arc::new(langchart_runtime::simulation::NoopMemory),
    secrets:    Arc::new(HostMapSecretsAdapter::empty()),
    event_sink: Arc::new(NoopSink),
});

let mut actors: HashMap<StateId, Arc<dyn AgentActor>> = HashMap::new();
actors.insert(StateId::new("greet"), Arc::new(Greeter));

let run_id = engine.start(compiled, actors).await?;
```

---

## Workflow Document Reference

### State types

| Type | When to use |
|---|---|
| `atomic` | Deterministic step; waits for an external event before transitioning |
| `agentic` | Runs an agent actor; transitions on the emitted output event |
| `compound` | Contains nested states; provides a hierarchical structure |
| `parallel` | Runs multiple orthogonal regions concurrently |
| `human` | Suspends until a human decision event arrives |
| `subworkflow` | Invokes a separately versioned workflow via port bindings |
| `final` | End state; marks completion of a region or workflow |

### Transitions

Each state's `on` block maps event type names to transition spec arrays:

```json
"on": {
  "event.type": [
    {
      "target": "next-state-id",
      "priority": 0,
      "guard": "event.payload.score >= 0.7",
      "actions": [],
      "kind": "external"
    }
  ]
}
```

- **`priority`** — lower number = higher priority. Required when multiple transitions share the same event type.
- **`guard`** — CEL expression. Evaluated against `event`, `workflow`, and `state` variables. Must be pure (no I/O).
- **`kind`** — `"external"` (default), `"internal"` (no exit/entry actions), or `"local"` (stays inside compound state if target is a descendant).

### Execution limits

Configure resource limits per agentic state:

```json
{
  "id": "analyze",
  "type": "agentic",
  "limits": {
    "max_turns": 8,
    "max_tool_calls": 20,
    "timeout": 300,
    "max_tokens_total": 100000
  }
}
```

The `CapabilityBroker` enforces these automatically. When a budget reaches 80% it emits `BudgetWarning`; at 100% it returns an error and emits `BudgetExhausted`.

### CEL guards

Guards use the [Common Expression Language](https://cel.dev). Available variables:

| Variable | Contents |
|---|---|
| `event.type` | Event type string |
| `event.payload` | Event payload as a map |
| `workflow.id` | Workflow ID |
| `workflow.version` | Workflow version |
| `run.id` | Run ID |
| `data.<field>` | Workflow data fields (if `with_workflow_data` was called) |

Example guards:

```cel
event.payload.confidence >= 0.8
event.payload.status == "approved"
data.retry_count < 3
```

### Retry policy

```json
{
  "id": "risky-step",
  "type": "agentic",
  "retry": {
    "max_attempts": 3,
    "delay": 2,
    "backoff": "exponential",
    "on_exhausted": "recovery-state"
  }
}
```

When all attempts are exhausted the workflow transitions to `on_exhausted`. Each retry attempt creates a distinct observable event record.

### History

Add `"history": "shallow"` or `"history": "deep"` to a compound or parallel state to restore the last active configuration on re-entry:

```json
{
  "id": "editing",
  "type": "compound",
  "history": "shallow",
  "initial": "idle",
  "states": [...]
}
```

---

## Parallel States

Parallel states activate all regions simultaneously:

```json
{
  "id": "multi-analysis",
  "type": "parallel",
  "completion": { "mode": "all" },
  "regions": [
    {
      "id": "text",
      "initial": "analyze-text",
      "states": [
        { "id": "analyze-text", "type": "agentic", "agent": { "id": "text-agent", "version": "1.0.0" },
          "on": { "text.done": [{ "target": "text-done", "priority": 0, "actions": [] }] } },
        { "id": "text-done", "type": "final", "on": {} }
      ]
    },
    {
      "id": "image",
      "initial": "analyze-image",
      "states": [
        { "id": "analyze-image", "type": "agentic", "agent": { "id": "image-agent", "version": "1.0.0" },
          "on": { "image.done": [{ "target": "image-done", "priority": 0, "actions": [] }] } },
        { "id": "image-done", "type": "final", "on": {} }
      ]
    }
  ],
  "on": {
    "parallel.completed": [{ "target": "synthesize", "priority": 0, "actions": [] }]
  }
}
```

Completion modes: `all`, `any`, `quorum` (`{ "mode": "quorum", "n": 2 }`), `guard` (`{ "mode": "guard", "expr": "completed >= 2" }`), `manual`.

---

## Subworkflows

A `subworkflow` state invokes another versioned workflow document resolved through the `WorkflowRepository`:

```json
{
  "id": "run-review",
  "type": "subworkflow",
  "workflow_ref": "content-review@1.0.0",
  "ports": {
    "input": {
      "draft_version": "${workflow.current_draft_version}"
    },
    "output": {
      "approved": {
        "issues": "${event.payload.issues}",
        "approved": "${event.payload.approved}"
      }
    }
  },
  "on": {
    "subworkflow.approved":  [{ "target": "finalize", "priority": 0, "actions": [] }],
    "subworkflow.failed":    [{ "target": "recovery", "priority": 0, "actions": [] }]
  }
}
```

Input bindings initialize the child workflow's data. The event that transitions the child into its top-level final
state selects the matching output map. Mapped fields are merged into the parent's workflow data and delivered in a
`subworkflow.<child-event-type>` event. A subworkflow without output mappings retains the compatibility event
`subworkflow.completed` with an empty payload.

Wire the repository into the engine:

```rust
use langchart_adapters::workflow_repository::InMemoryWorkflowRepository;

let repo = InMemoryWorkflowRepository::new();
repo.store(review_workflow_doc).await?;

let engine = RuntimeEngine::new(adapters).with_workflow_repository(Arc::new(repo));
```

---

## Checkpointing and Recovery

Checkpoints let runs survive process restarts. Add a `CheckpointStore` to the engine:

```rust
use langchart_checkpoint_redb::RedbCheckpointStore;

let store = RedbCheckpointStore::open("checkpoints.redb")?;

let engine = RuntimeEngine::new(adapters)
    .with_checkpoint_store(Arc::new(store));
```

The engine saves a checkpoint automatically on suspend, cancel, and run completion. Recover a run after restart:

```rust
let run_id = RunId::new("the-run-id-you-saved");
engine.recover_run(compiled, actors, &run_id).await?;
```

Inspect a checkpoint from the command line:

```bash
langchart inspect checkpoints.redb --run-id 01JXXXXXXXXXXXXXXXX
```

---

## Context Resolution

Wire a `ContextResolverChain` into the engine to control what information is assembled for each agent invocation:

```rust
use langchart_context::{ContextResolverChain, stages::{
    ArtifactResolverStage, MemoryResolverStage, WorkflowDataResolverStage,
    TruncationResolverStage, RecordingResolverStage,
}};

let chain = ContextResolverChain::new()
    .add_stage(ArtifactResolverStage::new(artifact_store.clone()))
    .add_stage(MemoryResolverStage::new(memory_adapter.clone(), MemoryScope::Global))
    .add_stage(WorkflowDataResolverStage::new(Arc::new(workflow_data_map)))
    .add_stage(TruncationResolverStage::from_policy())  // honours state token_budget
    .add_stage(RecordingResolverStage::new(event_sink.clone(), state_id));

let engine = RuntimeEngine::new(adapters)
    .with_context_resolver(Arc::new(chain));
```

Stages run in order. Each adds items to a `ContextAccumulator`. `TruncationResolverStage` drops items from the end if the token count exceeds the budget. `RecordingResolverStage` emits a `ContextResolved` event with the content hash so the view is reproducible and inspectable.

---

## Capability Resolution

Configure an optional deployment-wide ceiling on the engine:

```rust
let engine = RuntimeEngine::new(adapters)
    .with_deployment_capabilities(deployment_policy);
```

For each agent invocation, the runtime intersects the deployment ceiling (when
present), workflow maximum, every ancestor-state policy, agent defaults, and the
state policy. An omitted deployment or ancestor policy adds no restriction; a
present empty policy denies all capabilities. `elevate: true` is diagnostic only
and does not widen authority.

---

## Artifacts

The `ArtifactStore` trait manages versioned, proposal-based artifacts. Use `langchart-artifact-fs` for a quick file-system store:

```rust
use langchart_artifact_fs::FsArtifactStore;

let store = FsArtifactStore::open("artifacts/")?;

let engine = RuntimeEngine::new(adapters)
    .with_artifact_store(Arc::new(store));
```

Agent actors propose changes via the broker:

```rust
let proposal_id = broker.propose_artifact(&envelope, ArtifactProposal {
    id: ArtifactId::new("report"),
    base_version: current_version,
    content: new_content.into_bytes(),
    content_type: "text/markdown".into(),
    rationale: "Updated introduction based on findings.".into(),
}).await?;

// Later, in an authorized state:
let new_version = broker.commit_artifact(
    &envelope, &artifact_id, &proposal_id, &expected_base
).await?;
```

Artifact access is deny-by-default. The effective `CapabilityPolicy` must
include the corresponding `artifact_operations` entry (`read`, `propose`, or
`commit`) at every intersected policy layer.

---

## Observable Events

Every engine action emits a `RuntimeEvent` to the `EventSink`. Wire a durable sink for audit logs:

```rust
use langchart_adapters::event::{EventSink, RuntimeEvent, EventSinkError};

pub struct DatabaseSink { /* your database connection */ }

#[async_trait]
impl EventSink for DatabaseSink {
    async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
        // persist event.event_id, event.run_id, event.timestamp, event.payload
        Ok(())
    }
}
```

To receive live events from a running workflow, use `BroadcastEventSink`:

```rust
use langchart_adapters::broadcast::BroadcastEventSink;

let broadcast = Arc::new(BroadcastEventSink::new(128));  // channel capacity
let engine = RuntimeEngine::new(EngineAdapters { event_sink: broadcast.clone(), .. })
    .with_event_source(broadcast.clone());

// Subscribe to events for a specific run:
let mut stream = engine.subscribe(&run_id).unwrap();
while let Some(event) = stream.next().await {
    println!("{:?}", event.payload);
}
```

---

## Testing Workflows

### WorkflowSimulator — in-process deterministic testing

```rust
use langchart_runtime::simulation::{WorkflowSimulator, SimActorMap};
use langchart_runtime::instance::ScriptedAgentActor;

let compiled = Arc::new(compile(doc)?);

let result = WorkflowSimulator::new(compiled)
    .with_actors(
        SimActorMap::new()
            .add("extract", ScriptedAgentActor::emit("extraction.done", json!({})))
            .add("review",  ScriptedAgentActor::emit("review.approved", json!({})))
    )
    .run()
    .await?;

assert_eq!(result.status, RunStatus::Completed);
```

Test failure paths:

```rust
let result = WorkflowSimulator::new(compiled)
    .with_actors(
        SimActorMap::new()
            .add("extract", ScriptedAgentActor::fail("timeout"))
    )
    .run()
    .await?;

// The run should transition to the retry/recovery state, not Completed.
assert_ne!(result.status, RunStatus::Completed);
```

### TraceReplayer — regression testing

Capture a run's events, then replay to verify the same transitions occur:

```rust
use langchart_runtime::replay::TraceReplayer;

let replayed = TraceReplayer::new(compiled.clone(), original_events)
    .with_actors(actors)
    .replay()
    .await?;

assert_eq!(replayed.final_status, RunStatus::Completed);
```

### WASM simulation — browser-side testing

The `editor/` simulation panel uses `simulateWorkflow` for quick stateless path-tracing without a Tokio runtime. Configure actor scripts and inject initial events directly in the browser.

---

## CLI Reference

The `langchart` binary provides four subcommands:

```
langchart validate <workflow>
    Validate a workflow document (JSON or YAML). Exits 1 if there are errors.
    Prints warnings and errors with diagnostic codes.

langchart run <workflow>
    Compile and run a workflow with no agent actors wired.
    Useful for testing pure-transition workflows or as a smoke test.

langchart replay <workflow> <events-json>
    Replay a previously recorded event trace through a compiled workflow.

langchart inspect <checkpoint-db> --run-id <id>
    Open a redb checkpoint file and print the latest checkpoint
    for the given run: active states, history, retry counts.
```

---

## Model Routing

`langchart-model-router` dispatches LLM calls to different adapter implementations based on model policy:

```rust
use langchart_model_router::{ModelRouter, Route};

let router = ModelRouter::builder()
    .add_adapter("openai", Arc::new(openai_adapter))
    .add_adapter("anthropic", Arc::new(anthropic_adapter))
    .exact_route("gpt-4o", "openai")
    .prefix_route("claude-", "anthropic")
    .profile_route("high_quality", "anthropic")
    .fallback("openai")
    .build()?;
```

Wire it as your `LlmAdapter`:

```rust
let engine = RuntimeEngine::new(EngineAdapters {
    llm: Arc::new(router),
    ..
});
```

---

## MCP Connections

`langchart-mcp-client` connects to MCP servers via stdio child processes:

```rust
use langchart_mcp_client::{McpClientRegistry, LangchartMcpAdapter};
use langchart_model::id::ServerId;

let registry = McpClientRegistry::new();

registry.connect_stdio(
    ServerId::new("file-tools"),
    "uvx",
    &["my-mcp-server", "--config", "config.json"],
).await?;

let mcp = Arc::new(LangchartMcpAdapter::new(Arc::new(registry)));
```

Then allow specific tools in your workflow document's capability policy:

```json
{
  "id": "write-file",
  "type": "agentic",
  "capabilities": {
    "mcp": {
      "file-tools": {
        "allow": ["read_file", "write_file", "list_directory"]
      }
    }
  }
}
```

---

## Security Checklist

Before going to production:

- [ ] Replace `HostMapSecretsAdapter` with a vault-backed `SecretsAdapter` implementation.
- [ ] Implement `CheckpointStore` (or use `RedbCheckpointStore`) so runs survive restarts.
- [ ] Wrap your `EventSink` in `RedactingEventSink` with a `RedactionPolicy` to prevent tool arguments and memory queries from appearing in audit logs.
- [ ] Set `max_turns`, `max_tool_calls`, and `timeout` on every agentic state.
- [ ] Audit `CapabilityPolicy.mcp` allowlists — grant each state only the tools it actually needs.
- [ ] Wire `EventSink` to a durable store (database or message bus) for audit.
- [ ] Test all failure paths using `ScriptedAgentActor::fail` in `WorkflowSimulator`.
- [ ] Never call an LLM, MCP tool, or memory adapter directly from an agent — always go through `CapabilityBroker`. The broker is the only enforcement point.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `E003` — "Initial state not declared" | The `initial` field references a state ID that doesn't exist |
| `E011` — "Priority tie" | Two transitions on the same `(state, event)` have the same priority; add distinct priorities |
| `ActivityInvalidOutput` in event log | The agent emitted an event type not declared in the state's `on` block or agent's `output_events` |
| Workflow stuck at `running` | No event is available in an atomic or human state; inject the expected event |
| `BudgetExhausted` immediately | `max_turns` or `max_tool_calls` set too low for your workload |
| Checkpoint not restoring timers | Upgrade to the current version; timer state is now persisted in `InstanceCheckpoint.pending_timers` |

---

*Made with IBM Bob*
