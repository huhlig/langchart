# langchart-model

Pure, WebAssembly-compatible workflow types and validation for Langchart.

This crate contains the complete workflow schema, strongly-typed IDs, state and transition definitions, policies, CEL
guard compilation, and the validation pipeline. It performs only pure computation and deliberately avoids async
runtimes, file-system access, threads, and networking — making it safe to compile to WASM.

## What's in this crate

- **`workflow`** — `WorkflowDocument`, state types, transition definitions, and the full serialization schema (JSON,
  YAML, and RON)
- **`id`** — `WorkflowId`, `StateId`, `TransitionId`, `ArtifactId`, `ArtifactVersion`, and other typed ULID-based
  identifiers
- **`policy`** — `ModelPolicy` and capability envelope definitions
- **`guard`** — CEL expression compilation and evaluation
- **`validation`** — diagnostic reporting and compiled workflow construction

## Usage

```rust
use langchart_model::{validation, workflow::WorkflowDocument};

fn validate_json(json: &str) -> Result<(), Box<dyn std::error::Error>> {
    let document: WorkflowDocument = serde_json::from_str(json)?;

    let diagnostics = validation::validate(&document);
    for d in &diagnostics {
        println!("{}: {}", d.code, d.message);
    }

    // compile() rejects documents with errors
    let compiled = validation::compile(document)?;
    println!("states: {}", compiled.states().count());
    Ok(())
}
```

## Validation vs compilation

- `validation::validate` collects all diagnostics (errors and warnings) without failing.
- `validation::compile` rejects any document that contains at least one error diagnostic and returns the compiled
  representation consumed by `langchart-runtime`.

## License

Licensed under MIT or Apache-2.0.
