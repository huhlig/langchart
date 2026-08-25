# langchart-model

Pure, WebAssembly-compatible workflow types and validation for Langchart.

The crate contains the workflow schema, strongly typed IDs, state and
transition definitions, policies, CEL guard compilation, and validation.

```rust
use langchart_model::{validation, workflow::WorkflowDocument};

fn validate_json(json: &str) -> Result<(), Box<dyn std::error::Error>> {
    let document: WorkflowDocument = serde_json::from_str(json)?;
    let diagnostics = validation::validate(&document);
    assert!(diagnostics.iter().all(|item| !item.is_error()));
    validation::compile(document)?;
    Ok(())
}
```

This layer performs pure computation and deliberately avoids async runtimes,
file-system access, threads, and networking.

Licensed under MIT or Apache-2.0.
