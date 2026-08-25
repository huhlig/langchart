# langchart-checkpoint-redb

An embedded [redb](https://www.redb.org/)-backed implementation of Langchart's
`CheckpointStore` contract.

```rust,no_run
use langchart_checkpoint_redb::RedbCheckpointStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = RedbCheckpointStore::open("./langchart.redb")?;
    Ok(())
}
```

The store keeps the latest serialized snapshot for each run. It is suited to
single-process embedded applications; use shared storage for distributed or
multi-process deployments.

Licensed under MIT or Apache-2.0.
