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

#[cfg(test)]
pub(crate) mod test_support {
    use std::io::Write;
    use tempfile::NamedTempFile;

    pub const SIMPLE_WORKFLOW: &str = r#"{
        "schema_version": "1.0.0",
        "id": "cli-test",
        "version": "1.0.0",
        "name": "CLI test",
        "initial": "start",
        "states": [
            {
                "id": "start",
                "name": "Start",
                "type": "atomic",
                "on": { "go": [{ "target": "done", "priority": 0, "actions": [] }] }
            },
            { "id": "done", "name": "Done", "type": "final", "on": {} }
        ]
    }"#;

    pub fn write_json(content: &str) -> NamedTempFile {
        let mut file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }
}
