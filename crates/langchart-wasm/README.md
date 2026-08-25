# langchart-wasm

WebAssembly bindings for Langchart workflow validation and inspection.

The exported JavaScript API accepts and returns JSON strings. It includes
`schemaVersion`, `validateWorkflow`, `compileWorkflow`, `listStateIds`, and
state-inspection helpers for browser and editor integrations.

```javascript
const diagnostics = JSON.parse(validateWorkflow(workflowJson));
const errors = diagnostics.filter((item) => item.severity === "error");
```

This crate depends only on the pure model layer and intentionally excludes the
Tokio-based runtime and external adapters.

Licensed under MIT or Apache-2.0.
