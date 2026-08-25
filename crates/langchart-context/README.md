# langchart-context

Composable context resolution for Langchart agents.

`ContextResolverChain` applies ordered stages to build the context passed to an
agent. Built-in stages can incorporate workflow data, artifacts, memory,
recording, and truncation behavior.

Use the traits and request types from `langchart-adapters::context`, assemble a
chain from the stages needed by your host, and pass the resulting resolver into
runtime integration code.

Licensed under MIT or Apache-2.0.
