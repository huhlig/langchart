//! # langchart CLI
//!
//! Command-line interface to the langchart engine.
//!
//! ## Subcommands
//!
//! ```text
//! langchart validate <workflow>          Validate a workflow document (JSON or YAML).
//! langchart run      <workflow> [opts]   Run a workflow headlessly with scripted actors.
//! langchart replay   <workflow> <trace>  Replay a captured event trace.
//! langchart inspect  <checkpoint-db>     Inspect the latest checkpoint in a redb file.
//! ```

mod cmd;

use clap::Parser;
use cmd::{Cli, Command};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("langchart=info".parse().unwrap()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Command::Validate(args) => cmd::validate::execute(args).await,
        Command::Run(args) => cmd::run::execute(args).await,
        Command::Replay(args) => cmd::replay::execute(args).await,
        Command::Inspect(args) => cmd::inspect::execute(args).await,
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
