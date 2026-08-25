//! `langchart replay <workflow> <trace>`
//!
//! Loads a workflow document and a captured event trace (JSON array of
//! `RuntimeEvent`), then replays the causal events through a fresh instance
//! and reports the final status.
//!
//! The trace file is a JSON array as produced by serializing a `Vec<RuntimeEvent>`.

use anyhow::{Context, Result};
use clap::Args;
use langchart_adapters::event::RuntimeEvent;
use langchart_model::validation::compile;
use langchart_runtime::{replay::TraceReplayer, run::RunStatus};
use std::{path::PathBuf, sync::Arc};

use super::validate::load_workflow;

/// Replay a captured event trace through a fresh workflow instance.
#[derive(Debug, Args)]
pub struct ReplayArgs {
    /// Path to the workflow document (`.json` or `.yaml` / `.yml`).
    pub workflow: PathBuf,

    /// Path to the trace file (JSON array of `RuntimeEvent`).
    pub trace: PathBuf,
}

pub async fn execute(args: ReplayArgs) -> Result<()> {
    let doc = load_workflow(&args.workflow)?;
    let workflow_id = doc.id.0.clone();
    let compiled = Arc::new(compile(doc).map_err(|e| anyhow::anyhow!("compile error: {e}"))?);

    let trace_src = std::fs::read_to_string(&args.trace)
        .with_context(|| format!("cannot read trace `{}`", args.trace.display()))?;
    let trace: Vec<RuntimeEvent> = serde_json::from_str(&trace_src)
        .with_context(|| format!("invalid trace JSON in `{}`", args.trace.display()))?;

    let event_count = trace.len();
    println!("▶ replaying workflow `{workflow_id}` ({event_count} events) …");

    let result = TraceReplayer::new(compiled, trace)
        .replay()
        .await
        .map_err(|e| anyhow::anyhow!("replay error: {e}"))?;

    println!("  causal events replayed: {}", result.events.len());
    println!("  final status: {:?}", result.final_status);

    match result.final_status {
        RunStatus::Completed => {
            println!("✓ replay completed");
            Ok(())
        }
        other => {
            anyhow::bail!("replay ended with status {:?}", other)
        }
    }
}
