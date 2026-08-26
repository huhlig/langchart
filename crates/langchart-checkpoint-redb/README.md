# langchart-checkpoint-redb

Embedded [redb](https://www.redb.org/)-backed `CheckpointStore` implementation for Langchart.

`RedbCheckpointStore` persists the latest serialized workflow-run snapshot for each run ID in a local redb database
file. It is suitable for single-process embedded applications and development use.

## Usage

```rust,no_run
use langchart_checkpoint_redb::RedbCheckpointStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = RedbCheckpointStore::open("./langchart.redb")?;
    // Supply `store` to EngineAdapters as the checkpoint backend.
    Ok(())
}
```

## Notes

- One `.redb` file can be shared by multiple `RedbCheckpointStore` instances within the same process; redb handles
  internal locking.
- For multi-process or distributed deployments, use a shared checkpoint backend instead of a local file.
- The store keeps only the most recent snapshot per run ID; prior snapshots are overwritten on each checkpoint.

## License

Licensed under MIT or Apache-2.0.
