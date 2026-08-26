# langchart-docuvault

Optional bridge between Langchart adapter contracts and [Docuvault](https://github.com/huhlig/docuvault) APIs.

This crate provides adapter implementations for hosts that use Docuvault as their document and knowledge backend. It is
excluded from the default workspace build because it depends on the Docuvault Git repository directly.

## Provided adapters

| Module     | Adapter                                                               |
|------------|-----------------------------------------------------------------------|
| `artifact` | `ArtifactStore` backed by Docuvault documents                         |
| `context`  | `ContextResolver` stage that fetches content from Docuvault           |
| `memory`   | `MemoryAdapter` backed by Docuvault (requires `vault-memory` feature) |
| `id`       | Conversion utilities between Langchart and Docuvault IDs              |
| `event`    | Event bridge — forwards Langchart runtime events to Docuvault         |

## Features

- `vault-memory` — enables the Docuvault-backed `MemoryAdapter`

## Adding to your project

Enable the facade's `docuvault` feature to pull in this crate:

```toml
[dependencies]
langchart = { path = "../langchart/crates/langchart", features = ["docuvault"] }
```

Or depend on this crate directly:

```toml
[dependencies]
langchart-docuvault = { path = "../langchart/crates/langchart-docuvault" }
```

## Building

This crate depends on Docuvault via a Git source. Building it for the first time requires Git and network access:

```console
cargo check -p langchart-docuvault
```

Subsequent builds use the cached source from `cargo`'s local registry.

## License

Licensed under MIT or Apache-2.0.
