# langchart-wasm

WebAssembly bindings for Langchart workflow validation and inspection, for use in browser and editor environments.

This crate wraps `langchart-model` with `wasm-bindgen` exports. It depends only on the pure model layer and deliberately
excludes the Tokio-based runtime and all external adapter crates.

## Exported JavaScript API

All functions accept and return JSON strings.

| Function                 | Description                                                |
|--------------------------|------------------------------------------------------------|
| `schemaVersion()`        | Returns the supported workflow schema version string       |
| `validateWorkflow(json)` | Returns a JSON array of diagnostic objects                 |
| `compileWorkflow(json)`  | Validates and compiles; returns an error string on failure |
| `listStateIds(json)`     | Returns a JSON array of state ID strings                   |
| State-inspection helpers | Query state metadata from a compiled workflow              |

## Example

```javascript
import init, {validateWorkflow, compileWorkflow} from "./langchart_wasm.js";

await init();

const diagnostics = JSON.parse(validateWorkflow(workflowJson));
const errors = diagnostics.filter((d) => d.severity === "error");

if (errors.length === 0) {
    compileWorkflow(workflowJson); // throws on error
}
```

## Building

From the repository root, build with `wasm-pack`:

```console
wasm-pack build crates/langchart-wasm --target web
```

Run the exported API integration tests in Node's WebAssembly runtime:

```console
wasm-pack test --node crates/langchart-wasm
```

The WASM package is consumed by [`langchart-editor-tauri`](../langchart-editor-tauri) for in-editor validation.

## License

Licensed under MIT or Apache-2.0.
