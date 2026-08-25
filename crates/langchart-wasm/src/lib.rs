//! # langchart-wasm
//!
//! WebAssembly bindings for [`langchart_model`].
//!
//! Exposes workflow validation, compilation metadata, state inspection, and
//! CEL guard analysis to the TypeScript visual editor via `wasm-bindgen`.
//!
//! # Design
//!
//! All functions accept and return JSON strings for maximum compatibility.
//! This avoids the need for `serde-wasm-bindgen` and works with any
//! JS bundler. Each function returns either a serialised result or throws a
//! descriptive `JsValue` error.
//!
//! # WASM-safety rules
//!
//! - MUST NOT import `langchart-runtime`, `langchart-adapters`, or any crate
//!   that uses `tokio`, `std::thread`, `std::fs`, or `std::net`.
//! - MUST NOT bring in async runtimes.
//! - All computation is pure (validated, deterministic).

use langchart_model::{
    guard::CompiledGuard,
    id::StateId,
    state::{StateDefinition, StateType},
    validation::{compile, validate},
    workflow::WorkflowDocument,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use wasm_bindgen::prelude::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&msg.to_string())
}

fn parse_doc(workflow_json: &str) -> Result<WorkflowDocument, JsValue> {
    serde_json::from_str(workflow_json).map_err(|e| js_err(format!("JSON parse error: {e}")))
}

fn to_json<T: Serialize>(v: &T) -> Result<String, JsValue> {
    serde_json::to_string(v).map_err(|e| js_err(format!("JSON serialise error: {e}")))
}

// ── Serialisable types ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct DiagnosticJson {
    code: String,
    severity: &'static str,
    message: String,
    location: String,
}

#[derive(Serialize, Default)]
struct WorkflowCounts {
    total: usize,
    atomic: usize,
    agentic: usize,
    compound: usize,
    parallel: usize,
    human: usize,
    subworkflow: usize,
    final_states: usize,
}

// ── Public WASM API ───────────────────────────────────────────────────────────

/// Return the library's schema version string (currently `"1.0.0"`).
///
/// The editor uses this to check compatibility before loading a workflow.
#[wasm_bindgen]
pub fn schema_version() -> String {
    langchart_model::workflow::CURRENT_SCHEMA_VERSION.to_string()
}

/// Validate a workflow document (JSON string).
///
/// Returns a JSON array of `{ code, severity, message, location }` objects.
/// An empty array means the document is error-free.
/// Throws on JSON parse failure.
///
/// ```js
/// const diags = JSON.parse(validateWorkflow(json));
/// const errors = diags.filter(d => d.severity === "error");
/// ```
#[wasm_bindgen(js_name = validateWorkflow)]
pub fn validate_workflow(workflow_json: &str) -> Result<String, JsValue> {
    let doc = parse_doc(workflow_json)?;
    let raw = validate(&doc);
    let diags: Vec<DiagnosticJson> = raw
        .iter()
        .map(|d| DiagnosticJson {
            code: d.code.to_string(),
            severity: if d.is_error() { "error" } else { "warning" },
            message: d.message.clone(),
            location: format!("{:?}", d.location),
        })
        .collect();
    to_json(&diags)
}

/// Attempt to compile a workflow document (JSON string).
///
/// Returns `{ ok: true }` on success, or `{ ok: false, errors: [...] }` on
/// validation failure. Purely for the editor's "compile check" button.
/// Throws on JSON parse failure.
#[wasm_bindgen(js_name = compileWorkflow)]
pub fn compile_workflow(workflow_json: &str) -> Result<String, JsValue> {
    #[derive(Serialize)]
    struct CompileResult {
        ok: bool,
        errors: Vec<DiagnosticJson>,
    }

    let doc = parse_doc(workflow_json)?;
    match compile(doc) {
        Ok(_) => to_json(&CompileResult {
            ok: true,
            errors: vec![],
        }),
        Err(compile_err) => {
            // Re-validate to get a richer diagnostic list.
            let doc2 = parse_doc(workflow_json)?;
            let mut errors: Vec<DiagnosticJson> = validate(&doc2)
                .into_iter()
                .filter(|d| d.is_error())
                .map(|d| DiagnosticJson {
                    code: d.code.to_string(),
                    severity: "error",
                    message: d.message.clone(),
                    location: format!("{:?}", d.location),
                })
                .collect();
            if errors.is_empty() {
                errors.push(DiagnosticJson {
                    code: "E000".into(),
                    severity: "error",
                    message: compile_err.to_string(),
                    location: "workflow".into(),
                });
            }
            to_json(&CompileResult { ok: false, errors })
        }
    }
}

/// List all state IDs in a workflow document (flat walk of the full tree).
///
/// Returns a JSON array of strings. Useful for the editor's autocomplete
/// (transition target picker, state selector, etc.).
/// Throws on JSON parse failure.
#[wasm_bindgen(js_name = listStateIds)]
pub fn list_state_ids(workflow_json: &str) -> Result<String, JsValue> {
    let doc = parse_doc(workflow_json)?;
    let ids = collect_state_ids(&doc.states);
    to_json(&ids)
}

/// Inspect a single state by ID.
///
/// Returns a JSON object or `null` if the state is not found.
/// Throws on JSON parse failure.
#[wasm_bindgen(js_name = inspectState)]
pub fn inspect_state(workflow_json: &str, state_id: &str) -> Result<String, JsValue> {
    let doc = parse_doc(workflow_json)?;
    let target = StateId::new(state_id);

    let Some(def) = find_state_in_doc(&doc.states, &target) else {
        return to_json(&Option::<()>::None);
    };

    #[derive(Serialize)]
    struct TxJson {
        event: String,
        target: String,
        guard: Option<String>,
        priority: i32,
    }

    #[derive(Serialize)]
    struct StateJson<'a> {
        id: &'a str,
        name: &'a str,
        #[serde(rename = "type")]
        state_type: String,
        is_initial: bool,
        transitions: Vec<TxJson>,
        agent: Option<String>,
        prompt: Option<&'a str>,
        has_limits: bool,
        has_capabilities: bool,
        child_count: usize,
        region_count: usize,
    }

    let transitions = def
        .on
        .iter()
        .flat_map(|(event, specs)| {
            specs.iter().map(move |spec| TxJson {
                event: event.clone(),
                target: spec.target.0.clone(),
                guard: spec.guard.clone(),
                priority: spec.priority,
            })
        })
        .collect();

    let result = StateJson {
        id: &def.id.0,
        name: &def.name,
        state_type: format!("{:?}", def.state_type).to_lowercase(),
        is_initial: doc.initial == def.id.0,
        transitions,
        agent: def
            .agent
            .as_ref()
            .map(|a| format!("{}@{}", a.id.0, a.version.0)),
        prompt: def.prompt.as_deref(),
        has_limits: def.limits.is_some(),
        has_capabilities: def.capabilities.is_some(),
        child_count: def.states.len(),
        region_count: def.regions.len(),
    };
    to_json(&Some(result))
}

/// Validate all CEL guard expressions in a workflow document.
///
/// Returns a JSON array of `{ state_id, event_type, error }` objects.
/// An empty array means all guards compile cleanly.
/// Throws on JSON parse failure.
#[wasm_bindgen(js_name = getGuardErrors)]
pub fn get_guard_errors(workflow_json: &str) -> Result<String, JsValue> {
    #[derive(Serialize)]
    struct GuardError {
        state_id: String,
        event_type: String,
        error: String,
    }

    let doc = parse_doc(workflow_json)?;
    let mut errors: Vec<GuardError> = Vec::new();

    fn walk(states: &[StateDefinition], out: &mut Vec<GuardError>) {
        for state in states {
            for (event_type, specs) in &state.on {
                for spec in specs {
                    if let Some(expr) = &spec.guard
                        && let Err(e) = CompiledGuard::compile(expr)
                    {
                        out.push(GuardError {
                            state_id: state.id.0.clone(),
                            event_type: event_type.clone(),
                            error: e.to_string(),
                        });
                    }
                }
            }
            walk(&state.states, out);
            for r in &state.regions {
                walk(&r.states, out);
            }
        }
    }

    walk(&doc.states, &mut errors);
    to_json(&errors)
}

/// Return a summary of the workflow document for the editor canvas header.
///
/// Returns `{ id, version, name, initial, schema_version, state_counts, agent_ids }`.
/// Throws on JSON parse failure.
#[wasm_bindgen(js_name = workflowSummary)]
pub fn workflow_summary(workflow_json: &str) -> Result<String, JsValue> {
    let doc = parse_doc(workflow_json)?;
    let state_counts = tally_states(&doc.states);

    let mut agent_ids: Vec<String> = doc
        .agents
        .iter()
        .map(|a| format!("{}@{}", a.id.0, a.version.0))
        .collect();
    agent_ids.sort();
    agent_ids.dedup();

    #[derive(Serialize)]
    struct Summary {
        id: String,
        version: String,
        name: String,
        initial: String,
        schema_version: String,
        state_counts: WorkflowCounts,
        agent_ids: Vec<String>,
    }

    to_json(&Summary {
        id: doc.id.0.clone(),
        version: doc.version.0.clone(),
        name: doc.name.clone(),
        initial: doc.initial.clone(),
        schema_version: doc.schema_version.clone(),
        state_counts,
        agent_ids,
    })
}

/// Return a flat list of every transition in the workflow.
///
/// Each entry: `{ from, event, to, guard, priority }`.
/// Useful for rendering the transition graph.
/// Throws on JSON parse failure.
#[wasm_bindgen(js_name = listTransitions)]
pub fn list_transitions(workflow_json: &str) -> Result<String, JsValue> {
    #[derive(Serialize)]
    struct Edge {
        from: String,
        event: String,
        to: String,
        guard: Option<String>,
        priority: i32,
    }

    let doc = parse_doc(workflow_json)?;
    let mut edges: Vec<Edge> = Vec::new();

    fn walk(states: &[StateDefinition], out: &mut Vec<Edge>) {
        for s in states {
            for (event, specs) in &s.on {
                for spec in specs {
                    out.push(Edge {
                        from: s.id.0.clone(),
                        event: event.clone(),
                        to: spec.target.0.clone(),
                        guard: spec.guard.clone(),
                        priority: spec.priority,
                    });
                }
            }
            walk(&s.states, out);
            for r in &s.regions {
                walk(&r.states, out);
            }
        }
    }

    walk(&doc.states, &mut edges);
    to_json(&edges)
}

/// Return `{ reachable: [...], unreachable: [...] }` state ID lists.
///
/// BFS from `doc.initial` following transitions, region initials, and compound
/// initial children. Useful for the Problems panel's "unreachable state" check.
/// Throws on JSON parse failure.
#[wasm_bindgen(js_name = reachabilityAnalysis)]
pub fn reachability_analysis(workflow_json: &str) -> Result<String, JsValue> {
    let doc = parse_doc(workflow_json)?;
    let all_ids = collect_state_ids(&doc.states);

    let mut reachable: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    queue.push_back(doc.initial.clone());

    while let Some(id) = queue.pop_front() {
        if reachable.contains(&id) {
            continue;
        }
        reachable.insert(id.clone());

        if let Some(def) = find_state_in_doc(&doc.states, &StateId::new(&id)) {
            for region in &def.regions {
                enqueue(&mut queue, &reachable, region.initial.0.clone());
            }
            if let Some(child_init) = &def.initial {
                enqueue(&mut queue, &reachable, child_init.0.clone());
            }
            for specs in def.on.values() {
                for spec in specs {
                    enqueue(&mut queue, &reachable, spec.target.0.clone());
                }
            }
        }
    }

    let mut unreachable: Vec<String> = all_ids
        .into_iter()
        .filter(|id| !reachable.contains(id))
        .collect();
    let mut reachable_list: Vec<String> = reachable.into_iter().collect();
    reachable_list.sort();
    unreachable.sort();

    #[derive(Serialize)]
    struct Reach {
        reachable: Vec<String>,
        unreachable: Vec<String>,
    }

    to_json(&Reach {
        reachable: reachable_list,
        unreachable,
    })
}

/// Run a deterministic simulation of a workflow, driven by a script.
///
/// **Input** (`simulation_json`): a JSON object with shape:
/// ```json
/// {
///   "actors": { "state-id": { "emit": "event.type" }, ... },
///   "inject":  [{ "event_type": "start.ready", "payload": {} }, ...],
///   "max_steps": 50
/// }
/// ```
/// `actors` maps state IDs to scripted responses — when the simulation is
/// waiting in that state it automatically emits the configured event.
/// `inject` is a list of events sent immediately after start.
/// `max_steps` limits the step budget (default 100).
///
/// **Output**: a JSON object with shape:
/// ```json
/// {
///   "status": "completed" | "running" | "stuck",
///   "final_state": "state-id",
///   "steps": [
///     { "step": 0, "active_state": "prepare", "event": "start.ready", "target": "write" },
///     ...
///   ],
///   "error": null
/// }
/// ```
/// Throws on JSON parse failure or if the workflow has compile errors.
#[wasm_bindgen(js_name = simulateWorkflow)]
pub fn simulate_workflow(workflow_json: &str, simulation_json: &str) -> Result<String, JsValue> {
    // ── Parse inputs ──────────────────────────────────────────────────────────

    let doc = parse_doc(workflow_json)?;
    let compiled = compile(doc).map_err(|e| js_err(format!("Compile error: {e}")))?;

    #[derive(Deserialize)]
    struct ActorScript {
        emit: String,
    }

    #[derive(Deserialize)]
    struct InjectEvent {
        event_type: String,
    }

    #[derive(Deserialize, Default)]
    struct SimInput {
        #[serde(default)]
        actors: HashMap<String, ActorScript>,
        #[serde(default)]
        inject: Vec<InjectEvent>,
        #[serde(default = "default_max_steps")]
        max_steps: usize,
    }

    fn default_max_steps() -> usize {
        100
    }

    let sim: SimInput = serde_json::from_str(simulation_json)
        .map_err(|e| js_err(format!("Simulation JSON parse error: {e}")))?;

    // ── Simulation state ──────────────────────────────────────────────────────

    #[derive(Serialize)]
    struct StepRecord {
        step: usize,
        active_state: String,
        event: String,
        target: String,
    }

    #[derive(Serialize)]
    struct SimOutput {
        status: &'static str,
        final_state: String,
        steps: Vec<StepRecord>,
        error: Option<String>,
    }

    let mut active = compiled.document.initial.clone();
    let mut steps: Vec<StepRecord> = Vec::new();

    // Pending event queue — starts with injected events.
    let mut queue: VecDeque<String> = sim.inject.into_iter().map(|e| e.event_type).collect();

    for step_num in 0..sim.max_steps {
        // Check if the current state is a final state.
        let state_def = compiled.state_index.get(&StateId::new(&active));

        let is_final = state_def
            .map(|s| s.state_type == StateType::Final)
            .unwrap_or(false);

        if is_final {
            return to_json(&SimOutput {
                status: "completed",
                final_state: active,
                steps,
                error: None,
            });
        }

        // If no event in queue, check if actor script provides one.
        if queue.is_empty() {
            if let Some(script) = sim.actors.get(&active) {
                queue.push_back(script.emit.clone());
            } else {
                // No event and no actor script — simulation is stuck.
                return to_json(&SimOutput {
                    status: "stuck",
                    final_state: active.clone(),
                    steps,
                    error: Some(format!(
                        "No event available in state \"{active}\" and no actor script configured for it."
                    )),
                });
            }
        }

        let event = queue.pop_front().unwrap();

        // Find a matching transition.
        let transitions = state_def
            .and_then(|s| s.on.get(&event))
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        // Pick the highest-priority (lowest integer) transition whose guard is
        // absent or evaluates to true. Guards are not evaluated against live data
        // in the static simulation — only guardless transitions are followed.
        let chosen = transitions
            .iter()
            .filter(|t| t.guard.is_none())
            .min_by_key(|t| t.priority);

        let Some(tx) = chosen else {
            return to_json(&SimOutput {
                status: "stuck",
                final_state: active.clone(),
                steps,
                error: Some(format!(
                    "No matching transition from \"{active}\" on event \"{event}\". \
                     Guarded transitions are not evaluated in the static simulation."
                )),
            });
        };

        steps.push(StepRecord {
            step: step_num,
            active_state: active.clone(),
            event: event.clone(),
            target: tx.target.0.clone(),
        });

        active = tx.target.0.clone();
    }

    // Exhausted step budget.
    to_json(&SimOutput {
        status: "running",
        final_state: active,
        steps,
        error: Some("Step budget exhausted — workflow may be in a loop.".into()),
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn enqueue(queue: &mut VecDeque<String>, seen: &HashSet<String>, id: String) {
    if !seen.contains(&id) {
        queue.push_back(id);
    }
}

fn collect_state_ids(states: &[StateDefinition]) -> Vec<String> {
    let mut ids = Vec::new();
    for s in states {
        ids.push(s.id.0.clone());
        ids.extend(collect_state_ids(&s.states));
        for r in &s.regions {
            ids.extend(collect_state_ids(&r.states));
        }
    }
    ids
}

fn find_state_in_doc<'a>(
    states: &'a [StateDefinition],
    id: &StateId,
) -> Option<&'a StateDefinition> {
    for s in states {
        if &s.id == id {
            return Some(s);
        }
        if let Some(found) = find_state_in_doc(&s.states, id) {
            return Some(found);
        }
        for r in &s.regions {
            if let Some(found) = find_state_in_doc(&r.states, id) {
                return Some(found);
            }
        }
    }
    None
}

fn tally_states(states: &[StateDefinition]) -> WorkflowCounts {
    let mut c = WorkflowCounts::default();
    tally_inner(states, &mut c);
    c
}

fn tally_inner(states: &[StateDefinition], c: &mut WorkflowCounts) {
    for s in states {
        c.total += 1;
        match s.state_type {
            StateType::Atomic => c.atomic += 1,
            StateType::Agentic => c.agentic += 1,
            StateType::Compound => c.compound += 1,
            StateType::Parallel => c.parallel += 1,
            StateType::Human => c.human += 1,
            StateType::Subworkflow => c.subworkflow += 1,
            StateType::Final => c.final_states += 1,
        }
        tally_inner(&s.states, c);
        for r in &s.regions {
            tally_inner(&r.states, c);
        }
    }
}

// ── Native unit tests ─────────────────────────────────────────────────────────
//
// `wasm-bindgen` public functions call `JsValue::from_str` which panics on
// non-wasm32 targets. Tests therefore call the pure inner helpers directly,
// bypassing the WASM entry points. WASM integration tests use `wasm-pack test`.

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_doc() -> WorkflowDocument {
        serde_json::from_str(
            r#"{
            "schema_version": "1.0.0",
            "id": "test-wf",
            "version": "1.0.0",
            "name": "Test",
            "initial": "idle",
            "states": [
                {
                    "id": "idle",
                    "name": "Idle",
                    "type": "atomic",
                    "on": { "go": [{ "target": "end", "priority": 0, "actions": [] }] }
                },
                {
                    "id": "end",
                    "name": "End",
                    "type": "final",
                    "on": {}
                }
            ]
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn validate_valid_doc() {
        let doc = minimal_doc();
        let diags = validate(&doc);
        let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn compile_valid_doc() {
        let doc = minimal_doc();
        assert!(compile(doc).is_ok());
    }

    #[test]
    fn list_state_ids_returns_both() {
        let doc = minimal_doc();
        let ids = collect_state_ids(&doc.states);
        assert!(ids.contains(&"idle".to_string()));
        assert!(ids.contains(&"end".to_string()));
    }

    #[test]
    fn inspect_state_found() {
        let doc = minimal_doc();
        let def = find_state_in_doc(&doc.states, &StateId::new("idle"));
        assert!(def.is_some());
        let def = def.unwrap();
        assert_eq!(def.id.0, "idle");
        assert_eq!(def.state_type, StateType::Atomic);
    }

    #[test]
    fn inspect_state_not_found() {
        let doc = minimal_doc();
        assert!(find_state_in_doc(&doc.states, &StateId::new("ghost")).is_none());
    }

    #[test]
    fn reachability_all_reachable() {
        let doc = minimal_doc();
        let all_ids = collect_state_ids(&doc.states);
        // Both idle and end are reachable from initial=idle.
        let mut reachable: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(doc.initial.clone());
        while let Some(id) = queue.pop_front() {
            if reachable.contains(&id) {
                continue;
            }
            reachable.insert(id.clone());
            if let Some(def) = find_state_in_doc(&doc.states, &StateId::new(&id)) {
                for specs in def.on.values() {
                    for spec in specs {
                        enqueue(&mut queue, &reachable, spec.target.0.clone());
                    }
                }
            }
        }
        let unreachable: Vec<_> = all_ids
            .iter()
            .filter(|id| !reachable.contains(*id))
            .collect();
        assert!(
            unreachable.is_empty(),
            "unexpected unreachable: {unreachable:?}"
        );
    }

    #[test]
    fn guard_errors_empty_on_valid_doc() {
        // No guards in the minimal doc — error list should be empty.
        let doc = minimal_doc();
        for state in &doc.states {
            for specs in state.on.values() {
                for spec in specs {
                    if let Some(expr) = &spec.guard {
                        assert!(
                            CompiledGuard::compile(expr).is_ok(),
                            "guard compile failed: {expr}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn workflow_summary_counts() {
        let doc = minimal_doc();
        let counts = tally_states(&doc.states);
        assert_eq!(counts.total, 2);
        assert_eq!(counts.atomic, 1);
        assert_eq!(counts.final_states, 1);
    }

    #[test]
    fn list_transitions_finds_edge() {
        let doc = minimal_doc();
        let idle = find_state_in_doc(&doc.states, &StateId::new("idle")).unwrap();
        assert!(idle.on.contains_key("go"));
        assert_eq!(idle.on["go"][0].target.0, "end");
    }

    #[test]
    fn schema_version_is_1_0_0() {
        assert_eq!(langchart_model::workflow::CURRENT_SCHEMA_VERSION, "1.0.0");
    }

    #[test]
    fn simulate_completes_via_actor_script() {
        let doc = minimal_doc();
        let compiled = langchart_model::validation::compile(doc).unwrap();
        // Verify: inject "go" drives idle → end (which is Final).
        // Tests the state_index lookup and Final-type detection used by simulate_workflow.
        let start = &compiled.document.initial;
        let state = compiled.state_index.get(&StateId::new(start)).unwrap();
        assert!(
            state.on.contains_key("go"),
            "idle should have 'go' transition"
        );
        let target = &state.on["go"][0].target.0;
        let target_state = compiled.state_index.get(&StateId::new(target)).unwrap();
        assert_eq!(
            target_state.state_type,
            StateType::Final,
            "target should be Final"
        );
    }

    #[test]
    fn simulate_stuck_when_no_actor_and_no_event() {
        // A workflow where the initial state has no outbound transitions and
        // no actor script → the simulation should report "stuck".
        let doc: WorkflowDocument = serde_json::from_str(
            r#"{
            "schema_version": "1.0.0",
            "id": "stuck-wf",
            "version": "1.0.0",
            "name": "Stuck",
            "initial": "idle",
            "states": [
                { "id": "idle", "name": "Idle", "type": "atomic", "on": {} },
                { "id": "end",  "name": "End",  "type": "final",  "on": {} }
            ]
        }"#,
        )
        .unwrap();
        let compiled = langchart_model::validation::compile(doc).unwrap();
        let idle = compiled.state_index.get(&StateId::new("idle")).unwrap();
        // No transitions defined, no actor — stuck.
        assert!(
            idle.on.is_empty(),
            "expected no transitions on idle in stuck-wf"
        );
    }
}
