# Langchart

Langchart is a Rust workspace for defining, validating, and running governed, durable agentic workflows as hierarchical
statecharts. It combines deterministic statechart semantics with bounded agent autonomy: each agent can only use the
tools, memory, and LLM calls explicitly allowed by the workflow definition.

## Getting started

Requires a recent stable Rust toolchain.

**Validate the included example workflow:**

```console
cargo run -p langchart-cli -- validate examples/hello-world.json
```

**Run the workspace test suite:**

```console
cargo test --workspace
```

**Build the desktop workflow editor:**

```console
cargo tauri build --config crates/langchart-editor-tauri/tauri.conf.json
```

## Embedding Langchart

Add the facade crate and Tokio to your `Cargo.toml`:

```toml
[dependencies]
langchart = { path = "path/to/langchart/crates/langchart" }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
```

Parse and validate a workflow document:

```rust
use langchart::model::{validation, workflow::WorkflowDocument};

fn load(source: &str) -> Result<WorkflowDocument, Box<dyn std::error::Error>> {
    let workflow: WorkflowDocument = serde_json::from_str(source)?;

    for diagnostic in validation::validate(&workflow) {
        println!("{}: {}", diagnostic.code, diagnostic.message);
    }

    validation::compile(workflow.clone())?;
    Ok(workflow)
}
```

The runtime requires host-provided adapter implementations for LLM, MCP, memory, secrets, artifacts, checkpoints, and
events. See the [facade crate README](crates/langchart/README.md) and the [embedding guide](docs/embedding-guide.md) for
complete runtime wiring.

## Workspace crates

| Crate                                                           | Purpose                                                                              |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`langchart`](crates/langchart)                                 | Public facade — re-exports model, adapters, runtime, and context                     |
| [`langchart-model`](crates/langchart-model)                     | Workflow schema, typed IDs, validation, and CEL guard compilation (WASM-compatible)  |
| [`langchart-adapters`](crates/langchart-adapters)               | Integration trait contracts for LLM, MCP, memory, artifacts, checkpoints, and events |
| [`langchart-runtime`](crates/langchart-runtime)                 | Async statechart execution engine with capability broker, timers, and replay         |
| [`langchart-context`](crates/langchart-context)                 | Composable `ContextResolverChain` with built-in pipeline stages                      |
| [`langchart-cli`](crates/langchart-cli)                         | `langchart` CLI binary — validate, run, replay, and inspect commands                 |
| [`langchart-wasm`](crates/langchart-wasm)                       | WebAssembly bindings for in-browser workflow validation and inspection               |
| [`langchart-llm-generic`](crates/langchart-llm-generic)         | `LlmAdapter` for OpenAI, Anthropic, and any OpenAI-compatible endpoint               |
| [`langchart-llm-genai`](crates/langchart-llm-genai)             | `LlmAdapter` backed by `genai` (Gemini, Groq, Cohere, xAI, DeepSeek)                 |
| [`langchart-llm-watsonx`](crates/langchart-llm-watsonx)         | IBM watsonx.ai `LlmAdapter` with IAM authentication                                  |
| [`langchart-mcp-client`](crates/langchart-mcp-client)           | `McpAdapter` over child-process MCP servers via `rmcp`                               |
| [`langchart-artifact-fs`](crates/langchart-artifact-fs)         | File-system `ArtifactStore` with atomic writes and optimistic concurrency            |
| [`langchart-checkpoint-redb`](crates/langchart-checkpoint-redb) | Embedded redb-backed `CheckpointStore`                                               |
| [`langchart-memory-redb`](crates/langchart-memory-redb)         | Embedded redb-backed `MemoryAdapter`                                                 |
| [`langchart-model-router`](crates/langchart-model-router)       | Policy-driven `LlmAdapter` router for multi-provider deployments                     |
| [`langchart-editor-tauri`](crates/langchart-editor-tauri)       | Standalone Tauri desktop editor for authoring workflows                              |
| [`langchart-docuvault`](crates/langchart-docuvault)             | Optional bridge onto Docuvault artifact, memory, and context APIs                    |

### Optional crate: langchart-docuvault

Docuvault support is excluded from default workspace builds because it depends on a Git source. Build it explicitly:

```console
cargo check -p langchart-docuvault
```

Or enable the facade feature to bring it into your application:

```toml
langchart = { path = "path/to/langchart/crates/langchart", features = ["docuvault"] }
```

## Documentation

- [User guide](docs/user-guide.md)
- [Embedding guide](docs/embedding-guide.md)
- [Workflow JSON Schema](docs/workflow-schema.json)
- [Library specification](docs/agentic-statechart-library-spec.md)

## License

Licensed under MIT or Apache-2.0.
