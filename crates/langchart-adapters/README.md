# langchart-adapters

Integration contracts for Langchart. This crate defines traits and shared
types; it intentionally contains no provider-specific implementations.

The modules cover LLM completion, MCP calls, memory, artifact and checkpoint
storage, event sinks and sources, secrets, context resolution, broadcasting,
and workflow repositories.

```rust
use langchart_adapters::llm::LlmAdapter;

fn accept_adapter(_adapter: &dyn LlmAdapter) {}
```

Implement these contracts in a host application or use one of the concrete
`langchart-*` adapter crates in this workspace.

Licensed under MIT or Apache-2.0.
