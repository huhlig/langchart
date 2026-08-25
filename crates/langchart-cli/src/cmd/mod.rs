//! CLI argument types and module declarations.

use clap::{Parser, Subcommand};

pub mod inspect;
pub mod replay;
pub mod run;
pub mod validate;

// ── Top-level CLI ─────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(
    name = "langchart",
    about = "Agentic statechart engine — CLI tools",
    version,
    propagate_version = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Validate a workflow document (JSON or YAML). Exits 1 on errors.
    Validate(validate::ValidateArgs),
    /// Run a workflow headlessly with scripted actors. Exits 1 on failure.
    Run(run::RunArgs),
    /// Replay a captured event trace through a fresh workflow instance.
    Replay(replay::ReplayArgs),
    /// Inspect the latest checkpoint stored in a redb checkpoint file.
    Inspect(inspect::InspectArgs),
}
