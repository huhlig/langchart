# langchart-artifact-fs

File-system backed [`ArtifactStore`][adapters] implementation for the
[`langchart`] agentic statechart library.

## Overview

`FsArtifactStore` stores versioned artifacts in a flat directory layout on any
local file system. It supports atomic writes, optimistic concurrency control
(version-precondition commits), and proposal staging — making it suitable for
development, testing, and single-node production deployments.

## Layout

Given a root directory `<root>`, each artifact gets its own subdirectory:

```
<root>/
  <artifact_id>/
    <ulid>.content        — raw content bytes
    <ulid>.meta.json      — {"content_type":"...","created_at":"..."}
    <ulid>.proposal.json  — pending ArtifactProposal (before commit)
    latest.txt            — current committed version ULID
```

Writes use an atomic rename (`write to <ulid>.tmp`, then `rename`) so partial
writes are never visible to readers.

## Usage

```rust
use langchart_artifact_fs::FsArtifactStore;
use langchart_adapters::artifact::{ArtifactStore, ArtifactProposal};
use langchart_model::id::{ArtifactId, ArtifactVersion};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = FsArtifactStore::open("./artifacts").await?;

    let id = ArtifactId::new("my-doc");
    let proposal = ArtifactProposal {
        id: id.clone(),
        content: b"Hello, world!".to_vec(),
        content_type: "text/plain".to_string(),
        base_version: ArtifactVersion::new("none"),
        rationale: "initial version".to_string(),
    };

    let proposal_id = store.propose(proposal).await?;
    let initial_version = ArtifactVersion::new("none");
    let new_version = store.commit(&id, &proposal_id, &initial_version).await?;

    let content = store.read(&id, Some(&new_version)).await?;
    assert_eq!(content.bytes, b"Hello, world!");
    Ok(())
}
```

## Concurrency

`FsArtifactStore` uses optimistic concurrency: `commit` requires the caller's
`expected_base` to match the proposal's stored `base_version`, then checks that
`latest.txt` still matches that base before writing. Initial proposals use the
sentinel version `"none"`. Conflicts return `ArtifactError::VersionConflict`.

Multiple `FsArtifactStore` instances pointing at the same root directory serialize
commits through an OS-backed lock and use atomic `rename` for file replacement.

## Feature flags

None. The crate has no optional features.

## License

Licensed under MIT or Apache-2.0.
