# langchart-context

Composable context resolution for Langchart agents.

`ContextResolverChain` applies an ordered pipeline of stages to build the context window passed to an agent at each
turn. Stages are drawn from the adapter contracts in `langchart-adapters::context` and assembled by the host
application.

## Built-in stages

| Stage         | Description                                                       |
|---------------|-------------------------------------------------------------------|
| Workflow data | Injects the current workflow and state metadata                   |
| Artifacts     | Fetches named artifact content from an `ArtifactStore`            |
| Memory        | Queries the `MemoryAdapter` for relevant prior memory entries     |
| Recording     | Appends a formatted turn history from previous agent interactions |
| Truncation    | Enforces a token budget by trimming lower-priority context slices |

## Usage

Assemble a resolver from the stages your host needs and pass it into the runtime:

```rust
use langchart_context::ContextResolverChain;
use langchart_adapters::context::ContextResolver;

fn build_resolver() -> impl ContextResolver {
    ContextResolverChain::builder()
        // .add_stage(WorkflowDataStage::new())
        // .add_stage(MemoryStage::new(memory_adapter))
        // .add_stage(TruncationStage::new(token_limit))
        .build()
}
```

See the [embedding guide](../../docs/embedding-guide.md) for a complete example showing context assembly alongside the
runtime.

## License

Licensed under MIT or Apache-2.0.
