# Langchart

Langchart is a Rust workspace for defining, validating, and running governed,
durable agentic workflows as hierarchical statecharts.

The [`langchart`](crates/langchart) crate is the main public API. The workspace
also contains the model, runtime, adapter contracts, concrete integrations,
CLI, and WebAssembly bindings.

## Getting started

You need a recent stable Rust toolchain. Clone the repository, then validate
the included workflow:

```console
cargo run -p langchart-cli -- validate examples/hello-world.json
```

To run the sample workflow editor:

```console
cargo tauri build --config crates/langchart-editor-tauri/tauri.conf.json
```

Run the workspace tests with:

```console
cargo test --workspace
```

To embed Langchart in another crate, add the facade and Tokio:

```toml
[dependencies]
langchart = { path = "path/to/langchart/crates/langchart" }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
```

Parse and validate a workflow before passing it to the runtime:

```rust
use langchart::model::{validation, workflow::WorkflowDocument};

fn load(source: &str) -> Result<WorkflowDocument, Box<dyn std::error::Error>> {
    let workflow: WorkflowDocument = serde_json::from_str(source)?;
    let diagnostics = validation::validate(&workflow);

    for diagnostic in &diagnostics {
        println!("{}: {}", diagnostic.code, diagnostic.message);
    }

    validation::compile(workflow.clone())?;
    Ok(workflow)
}
```

Runtime applications supply implementations of the adapter traits for LLMs,
MCP, memory, secrets, artifacts, checkpoints, and events. See the
[facade crate README](crates/langchart/README.md) for the next steps and
[embedding guide](docs/embedding-guide.md) for complete runtime wiring.

## Workspace crates

| Crate | Purpose |
| --- | --- |
| [`langchart`](crates/langchart) | Public facade and re-exports |
| [`langchart-model`](crates/langchart-model) | Workflow schema, IDs, validation, and CEL guards |
| [`langchart-adapters`](crates/langchart-adapters) | External integration contracts |
| [`langchart-runtime`](crates/langchart-runtime) | Async execution engine |
| [`langchart-context`](crates/langchart-context) | Composable context resolution |
| [`langchart-cli`](crates/langchart-cli) | Validate, run, replay, and inspect commands |
| [`langchart-wasm`](crates/langchart-wasm) | Browser/editor validation bindings |
| [`langchart-llm-watsonx`](crates/langchart-llm-watsonx) | IBM watsonx.ai LLM adapter |
| [`langchart-docuvault`](crates/langchart-docuvault) | Optional Docuvault adapter bridge |

The remaining crates provide concrete artifact, checkpoint, memory, LLM, MCP,
document-vault, and model-routing integrations.

Docuvault support is excluded from default workspace builds. Build it directly
with `cargo check -p langchart-docuvault`, or expose it through the facade with:

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
