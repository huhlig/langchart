# langchart-memory-redb

An embedded redb-backed implementation of Langchart's `MemoryAdapter`.

The adapter persists agent memory locally and implements the shared request,
query, and result types from `langchart-adapters`. It is intended for embedded
and single-node hosts that need durable memory without an external service.

Open the store with `RedbMemoryAdapter::open`, then provide it wherever a
`langchart_adapters::memory::MemoryAdapter` is required.

Licensed under MIT or Apache-2.0.
