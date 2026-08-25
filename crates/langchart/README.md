# langchart

The main public crate for Langchart, an embeddable engine for governed and
durable agentic workflows built on hierarchical statecharts.

This facade re-exports the workspace's four core crates as `model`, `adapters`,
`runtime`, and `context`, so most applications can begin with one dependency.

Docuvault support is optional. Enable the `docuvault` feature to re-export the
bridge as `langchart::docuvault`:

```toml
langchart = { path = "../langchart/crates/langchart", features = ["docuvault"] }
```

## Getting started

Add Langchart and the dependencies used by your host application:

```toml
[dependencies]
langchart = { path = "../langchart/crates/langchart" }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
```

### Parse and validate a workflow

```rust
use langchart::model::{validation, workflow::WorkflowDocument};

fn compile_workflow(source: &str) -> Result<(), Box<dyn std::error::Error>> {
    let document: WorkflowDocument = serde_json::from_str(source)?;

    for diagnostic in validation::validate(&document) {
        println!("{}: {}", diagnostic.code, diagnostic.message);
    }

    let compiled = validation::compile(document)?;
    println!("compiled {}", compiled.document.id);
    Ok(())
}
```

Validation reports all diagnostics; compilation rejects documents containing
errors and produces the representation consumed by the runtime.

### Use a concrete adapter

Adapter traits are available through `langchart::adapters`. Concrete
implementations live in focused crates. For example, an embedded checkpoint
store can be opened with:

```rust,no_run
use langchart_checkpoint_redb::RedbCheckpointStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoints = RedbCheckpointStore::open("./langchart.redb")?;
    // Supply `checkpoints` while wiring your runtime host.
    Ok(())
}
```

### Run a workflow

The runtime requires host-provided LLM, MCP, memory, secrets, and event
adapters. Construct `langchart::runtime::EngineAdapters`, create a
`langchart::runtime::RuntimeEngine`, register an actor, and start the compiled
workflow. A complete, copyable walkthrough is maintained in the repository's
[embedding guide](../../docs/embedding-guide.md).

For a quick validation smoke test from the repository root:

```console
cargo run -p langchart-cli -- validate examples/hello-world.json
```

## Modules

- `langchart::model`: workflow data structures, IDs, policies, guards, and validation.
- `langchart::adapters`: integration traits and shared request/response types.
- `langchart::runtime`: engine, actors, capability broker, timers, and replay.
- `langchart::context`: context resolver chains and built-in stages.

## License

Licensed under MIT or Apache-2.0.
