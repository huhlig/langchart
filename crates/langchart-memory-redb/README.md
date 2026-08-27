# langchart-memory-redb

Embedded [redb](https://www.redb.org/)-backed `MemoryAdapter` implementation for Langchart.

`RedbMemoryAdapter` persists agent memory entries locally in a redb database file. It implements the shared request,
query, and result types from `langchart-adapters::memory` and is intended for embedded and single-node hosts that need
durable agent memory without an external service.

## Usage

```rust,no_run
use langchart_memory_redb::RedbMemoryAdapter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = RedbMemoryAdapter::open("./langchart-memory.redb")?;
    // Supply `memory` to EngineAdapters as the memory backend.
    Ok(())
}
```

## Notes

- Memory entries use ULID primary keys plus scope and exact-key indexes. Existing databases are indexed automatically
  when first opened by a newer adapter.
- Keyword queries scan only the requested scope; exact-key queries use the secondary index. Semantic queries fall back
  to scoped keyword matching with a warning.
- For multi-process or distributed deployments, use a shared memory backend instead of a local file.

## License

Licensed under MIT or Apache-2.0.
