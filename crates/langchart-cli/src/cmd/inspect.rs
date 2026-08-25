//! `langchart inspect <checkpoint-db> --run-id <id>`
//!
//! Opens a redb checkpoint store and pretty-prints the latest `InstanceCheckpoint`
//! for the given run ID.

use anyhow::{Context, Result};
use clap::Args;
use langchart_adapters::checkpoint::CheckpointStore;
use langchart_checkpoint_redb::RedbCheckpointStore;
use langchart_model::id::RunId;
use langchart_runtime::run::InstanceCheckpoint;
use std::path::PathBuf;

/// Inspect the latest checkpoint stored in a redb checkpoint file.
#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Path to the redb checkpoint database file.
    pub checkpoint_db: PathBuf,

    /// Run ID to inspect.
    #[arg(long)]
    pub run_id: String,
}

pub async fn execute(args: InspectArgs) -> Result<()> {
    let store = RedbCheckpointStore::open(&args.checkpoint_db).with_context(|| {
        format!(
            "cannot open checkpoint store `{}`",
            args.checkpoint_db.display()
        )
    })?;

    let run_id = RunId::new(args.run_id.clone());

    let snap = store
        .load(&run_id)
        .await
        .map_err(|e| anyhow::anyhow!("checkpoint store error: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("no checkpoint found for run `{}`", args.run_id))?;

    let ck: InstanceCheckpoint = serde_json::from_slice(&snap.payload)
        .context("checkpoint payload is not a valid InstanceCheckpoint")?;

    println!("Checkpoint for run `{}`", ck.run_id);
    println!(
        "  workflow:      {}@{}",
        ck.workflow_id, ck.workflow_version
    );
    println!("  status:        {:?}", ck.status);
    println!("  checkpoint_id: {}", snap.checkpoint_id);
    println!();
    println!("Active states ({}):", ck.active_states.len());
    for s in &ck.active_states {
        println!("  - {}", s);
    }

    if !ck.history.is_empty() {
        println!();
        println!("History ({} entries):", ck.history.len());
        for (parent, children) in &ck.history {
            let children: Vec<_> = children.iter().map(|c| c.0.as_str()).collect();
            println!("  {} → [{}]", parent, children.join(", "));
        }
    }

    if !ck.attempt_counts.is_empty() {
        println!();
        println!("Retry attempt counts:");
        for (state, count) in &ck.attempt_counts {
            println!("  {} → {}", state, count);
        }
    }

    Ok(())
}
