# langchart

The main public crate for Langchart — an embeddable Rust engine for governed, durable agentic workflows built on
hierarchical statecharts.

This facade re-exports the four core workspace crates as top-level modules, so most applications need only one
dependency:

| Module                | Source crate                                                    |
|-----------------------|-----------------------------------------------------------------|
| `langchart::model`    | `langchart-model` — workflow types, IDs, validation, CEL guards |
| `langchart::adapters` | `langchart-adapters` — integration trait contracts              |
| `langchart::runtime`  | `langchart-runtime` — async execution engine                    |
| `langchart::context`  | `langchart-context` — composable context resolver pipeline      |

## Adding to your project

```toml
[dependencies]
langchart = { path = "../langchart/crates/langchart" }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
```

### Optional: Docuvault integration

Enable the `docuvault` feature to re-export the Docuvault bridge as `langchart::docuvault`:

```toml
langchart = { path = "../langchart/crates/langchart", features = ["docuvault"] }
```

Building with this feature requires Git and network access to clone the Docuvault source dependency on first build.

## Usage

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

`validation::validate` reports all diagnostics without stopping. `validation::compile` rejects documents with errors and
returns the compiled representation consumed by the runtime.

### Open an embedded checkpoint store

```rust,no_run
use langchart_checkpoint_redb::RedbCheckpointStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoints = RedbCheckpointStore::open("./langchart.redb")?;
    // Supply `checkpoints` while wiring EngineAdapters.
    Ok(())
}
```

### Run a workflow

Construct `langchart::runtime::EngineAdapters` from host implementations of the adapter traits, create a
`langchart::runtime::RuntimeEngine`, register an actor, and start the compiled workflow. A complete, copyable
walkthrough is in the repository's [embedding guide](../../docs/embedding-guide.md).

For a quick validation smoke test from the repository root:

```console
cargo run -p langchart-cli -- validate examples/hello-world.json
```

## License

Licensed under MIT or Apache-2.0.
