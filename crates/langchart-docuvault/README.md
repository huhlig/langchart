# langchart-docuvault

An optional bridge between Langchart adapter contracts and Docuvault APIs.

It provides artifact, context, memory, ID-conversion, and event-bridge modules
for hosts that use Docuvault as their document and knowledge backend. Enable
the `vault-memory` feature when the memory integration is required.

This crate currently consumes Docuvault directly from its Git repository, so
building it requires Git and network access unless the dependency is already
cached.

It is excluded from default workspace builds. Build it explicitly with
`cargo check -p langchart-docuvault`, or enable the facade's opt-in feature:

```toml
langchart = { path = "../langchart", features = ["docuvault"] }
```

Licensed under MIT or Apache-2.0.
