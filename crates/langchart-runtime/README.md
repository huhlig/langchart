# langchart-runtime

The asynchronous statechart execution engine for Langchart.

## What's in this crate

- **`RuntimeEngine`** — creates and manages workflow run instances
- **Workflow instances** — per-run state machines that drive state transitions
- **Actor and action contracts** — Rust traits implemented by host code to handle agent turns
- **Capability broker** — enforces the per-state tool, memory, and LLM allowlists from the workflow definition
- **Timers** — state-entry timeouts and scheduled transitions
- **Outbox handling** — buffered event emission with at-least-once delivery semantics
- **Checkpoints** — serialized run snapshots written via the `CheckpointStore` adapter
- **Simulation and replay** — deterministic re-execution from a saved event log

## Setup

The host constructs `EngineAdapters` from concrete adapter implementations and passes it to `RuntimeEngine::new`:

```rust
use langchart_runtime::{EngineAdapters, RuntimeEngine};

fn build(adapters: EngineAdapters) -> RuntimeEngine {
    RuntimeEngine::new(adapters)
}
```

`EngineAdapters` requires at minimum an `LlmAdapter`, `McpAdapter`, `MemoryAdapter`, `ArtifactStore`, `CheckpointStore`,
and `EventSink`. See the repository [embedding guide](../../docs/embedding-guide.md) for a complete wiring example that
registers actors and starts a run.

## Adapter crates

This crate depends only on `langchart-model` and `langchart-adapters`. Concrete adapter implementations are in their own
crates:

- [`langchart-llm-generic`](../langchart-llm-generic) — OpenAI / Anthropic / compatible endpoints
- [`langchart-llm-genai`](../langchart-llm-genai) — `genai`-backed multi-provider adapter
- [`langchart-llm-watsonx`](../langchart-llm-watsonx) — IBM watsonx.ai
- [`langchart-llm-bedrock`](../langchart-llm-bedrock) — AWS Bedrock Converse API
- [`langchart-mcp-client`](../langchart-mcp-client) — child-process MCP servers
- [`langchart-artifact-fs`](../langchart-artifact-fs) — file-system artifact store
- [`langchart-checkpoint-redb`](../langchart-checkpoint-redb) — embedded redb checkpoint store
- [`langchart-memory-redb`](../langchart-memory-redb) — embedded redb memory adapter
- [`langchart-context`](../langchart-context) — composable context resolver pipeline

## License

Licensed under MIT or Apache-2.0.
