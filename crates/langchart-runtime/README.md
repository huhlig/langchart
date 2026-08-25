# langchart-runtime

The asynchronous statechart execution engine for Langchart.

It provides `RuntimeEngine`, workflow instances, actor and action contracts,
the capability broker, timers, outbox handling, checkpoints, simulation, and
replay. Hosts construct `EngineAdapters` from implementations of the contracts
in `langchart-adapters`, then run a compiled `langchart-model` workflow.

See the repository [embedding guide](../../docs/embedding-guide.md) for a full
example that wires adapters, registers actors, and starts a run.

```rust
use langchart_runtime::{EngineAdapters, RuntimeEngine};

fn build(adapters: EngineAdapters) -> RuntimeEngine {
    RuntimeEngine::new(adapters)
}
```

Licensed under MIT or Apache-2.0.
