//! `langchart run <workflow> [--actor state=script.json ...]`
//!
//! Runs a workflow headlessly. Each agentic state that needs an actor must
//! be covered by `--actor <state_id>=<path>` where the file is a JSON object:
//!
//! ```json
//! { "event_type": "work.done", "payload": { "result": "ok" } }
//! ```
//!
//! Or to simulate failure:
//!
//! ```json
//! { "fail": "simulated failure message" }
//! ```

use anyhow::{Context, Result, bail};
use clap::Args;
use langchart_model::{id::StateId, validation::compile};
use langchart_runtime::{
    instance::ScriptedAgentActor,
    run::RunStatus,
    simulation::{SimActorMap, WorkflowSimulator},
};
use std::{path::PathBuf, sync::Arc};

use super::validate::load_workflow;

/// Run a workflow headlessly with scripted actors. Exits 1 on failure.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// Path to the workflow document (`.json` or `.yaml` / `.yml`).
    pub workflow: PathBuf,

    /// Scripted actor specification: `<state_id>=<path-to-script.json>`.
    /// Repeat for each agentic state. Each file must be a JSON object with
    /// either `{"event_type":"…","payload":{…}}` for success or
    /// `{"fail":"…"}` for a simulated failure.
    #[arg(long = "actor", value_name = "STATE=FILE", value_parser = parse_actor_spec)]
    pub actors: Vec<(String, PathBuf)>,

    /// Event to inject immediately after startup: `<event_type>[=<json_payload>]`.
    /// Repeat for multiple initial events (processed in order).
    #[arg(long = "inject", value_name = "EVENT[=JSON]", value_parser = parse_event_spec)]
    pub inject: Vec<(String, serde_json::Value)>,

    /// Maximum RTC steps before giving up (default: 10 000).
    #[arg(long, default_value_t = 10_000)]
    pub step_limit: usize,
}

pub async fn execute(args: RunArgs) -> Result<()> {
    let doc = load_workflow(&args.workflow)?;
    let workflow_id = doc.id.0.clone();
    let compiled = Arc::new(compile(doc).map_err(|e| anyhow::anyhow!("compile error: {e}"))?);

    // Build scripted actor map from --actor flags.
    let mut sim_map = SimActorMap::new();
    for (state_id, script_path) in &args.actors {
        let script_src = std::fs::read_to_string(script_path)
            .with_context(|| format!("cannot read actor script `{}`", script_path.display()))?;
        let spec: ActorScript = serde_json::from_str(&script_src)
            .with_context(|| format!("invalid actor script JSON in `{}`", script_path.display()))?;
        let actor = match spec {
            ActorScript::Emit {
                event_type,
                payload,
            } => ScriptedAgentActor::emit(event_type, payload),
            ActorScript::Fail { fail } => ScriptedAgentActor::fail(fail),
        };
        sim_map = sim_map.add(StateId::new(state_id.clone()), actor);
    }

    // Build simulator.
    let mut sim = WorkflowSimulator::new(compiled)
        .with_actors(sim_map)
        .with_step_limit(args.step_limit);
    for (event_type, payload) in args.inject {
        sim = sim.inject(event_type, payload);
    }

    println!("▶ running workflow `{workflow_id}` …");
    let result = sim
        .run()
        .await
        .map_err(|e| anyhow::anyhow!("engine error: {e}"))?;

    // Print event summary.
    println!("  events emitted: {}", result.events.len());
    for event in &result.events {
        println!("  · {:?}", event.payload);
    }

    match result.status {
        RunStatus::Completed => {
            println!("✓ run completed");
            Ok(())
        }
        RunStatus::Running => {
            bail!("run did not complete within {} steps", args.step_limit)
        }
        other => {
            bail!("run ended with status {:?}", other)
        }
    }
}

// ── Actor script shape ────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum ActorScript {
    Emit {
        event_type: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    Fail {
        fail: String,
    },
}

// ── Argument parsers ──────────────────────────────────────────────────────────

fn parse_actor_spec(s: &str) -> Result<(String, PathBuf), String> {
    let (state, path) = s
        .split_once('=')
        .ok_or_else(|| format!("expected `STATE=FILE`, got `{s}`"))?;
    Ok((state.to_owned(), PathBuf::from(path)))
}

fn parse_event_spec(s: &str) -> Result<(String, serde_json::Value), String> {
    match s.split_once('=') {
        Some((event, json)) => {
            let payload: serde_json::Value = serde_json::from_str(json)
                .map_err(|e| format!("invalid JSON payload for event `{event}`: {e}"))?;
            Ok((event.to_owned(), payload))
        }
        None => Ok((s.to_owned(), serde_json::Value::Null)),
    }
}
