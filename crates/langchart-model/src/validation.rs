//! Structural and semantic validation of workflow documents.
//!
//! `validate` produces a list of [`Diagnostic`]s; callers decide whether to
//! proceed (warnings only) or abort (any error). `compile` runs validation
//! and, if clean, produces a [`CompiledWorkflow`] ready for the runtime.

use crate::{
    error::{CompileError, Diagnostic, DiagnosticLocation},
    guard::CompiledGuard,
    id::{StateId, TransitionId},
    schema::check_schema_version,
    state::{StateDefinition, StateType},
    workflow::WorkflowDocument,
};
use std::collections::{HashMap, HashSet};

// ── Compiled workflow ─────────────────────────────────────────────────────────

/// A validated and compiled workflow ready for execution by the runtime.
/// Compilation is infallible after a clean validation.
#[derive(Debug)]
pub struct CompiledWorkflow {
    pub document: WorkflowDocument,
    /// Pre-compiled CEL guards keyed by `(state_id, event_type)`.
    pub guards: HashMap<(StateId, String), CompiledGuard>,
    /// Flat index of all states in the document for O(1) lookup by the runtime.
    /// Built once at compile time; avoids repeated O(n) tree walks per RTC step.
    pub state_index: HashMap<StateId, StateDefinition>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Validate a workflow document and return all diagnostics.
/// A workflow with no [`Severity::Error`] diagnostics is considered valid.
pub fn validate(doc: &WorkflowDocument) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    // 1. Schema version
    if let Err(e) = check_schema_version(&doc.schema_version) {
        diags.push(Diagnostic::error(
            "E001",
            format!("Schema version error: {e}"),
            DiagnosticLocation::Workflow {
                id: doc.id.clone(),
                version: doc.version.clone(),
            },
        ));
        // Cannot proceed with further checks if schema is wrong.
        return diags;
    }

    // 2. Non-empty state list
    if doc.states.is_empty() {
        diags.push(Diagnostic::error(
            "E002",
            "Workflow has no states",
            DiagnosticLocation::Workflow {
                id: doc.id.clone(),
                version: doc.version.clone(),
            },
        ));
    }

    // 3. Collect all declared state IDs (flat walk)
    let all_ids = collect_all_state_ids(&doc.states);

    // 4. Initial state exists
    if !doc.initial.is_empty() && !all_ids.contains(doc.initial.as_str()) {
        diags.push(Diagnostic::error(
            "E003",
            format!("Initial state `{}` is not declared", doc.initial),
            DiagnosticLocation::Workflow {
                id: doc.id.clone(),
                version: doc.version.clone(),
            },
        ));
    }

    // 5. Duplicate state IDs
    check_duplicate_ids(&doc.states, &mut diags);

    // 6. Validate each state recursively
    for state in &doc.states {
        validate_state(state, &all_ids, &mut diags);
    }

    // 7. CEL guard compilation (warning-level; errors are collected per-guard)
    check_guards(&doc.states, &mut diags);

    // 8. data_schema guard reference check (Spec §8.2 / §11.1)
    // If data_schema declares fields, any guard that references `data.<name>`
    // where `<name>` is not in the schema is flagged as E012.
    if !doc.data_schema.fields.is_empty() {
        check_data_schema_guard_refs(&doc.states, &doc.data_schema.fields, &mut diags);
    }

    diags
}

/// Compile a workflow document. Runs validation first; returns an error if
/// any error-severity diagnostics are produced.
pub fn compile(doc: WorkflowDocument) -> Result<CompiledWorkflow, CompileError> {
    let diags = validate(&doc);
    let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();
    if !errors.is_empty() {
        return Err(CompileError::ValidationFailed(errors.len()));
    }

    // Pre-compile all guards.
    let mut guards = HashMap::new();
    compile_guards(&doc.states, &mut guards)?;

    // Build flat state index for O(1) runtime lookup.
    let mut state_index = HashMap::new();
    build_state_index(&doc.states, &mut state_index);

    Ok(CompiledWorkflow {
        document: doc,
        guards,
        state_index,
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn build_state_index(states: &[StateDefinition], index: &mut HashMap<StateId, StateDefinition>) {
    for state in states {
        index.insert(state.id.clone(), state.clone());
        build_state_index(&state.states, index);
        for region in &state.regions {
            build_state_index(&region.states, index);
        }
    }
}

fn collect_all_state_ids(states: &[StateDefinition]) -> HashSet<String> {
    let mut ids = HashSet::new();
    for state in states {
        ids.insert(state.id.0.clone());
        ids.extend(collect_all_state_ids(&state.states));
        for region in &state.regions {
            ids.extend(collect_all_state_ids(&region.states));
        }
    }
    ids
}

fn check_duplicate_ids(states: &[StateDefinition], diags: &mut Vec<Diagnostic>) {
    let mut seen = HashSet::new();
    check_duplicate_ids_inner(states, &mut seen, diags);
}

fn check_duplicate_ids_inner(
    states: &[StateDefinition],
    seen: &mut HashSet<String>,
    diags: &mut Vec<Diagnostic>,
) {
    for state in states {
        if !seen.insert(state.id.0.clone()) {
            diags.push(Diagnostic::error(
                "E004",
                format!("Duplicate state ID `{}`", state.id),
                DiagnosticLocation::State {
                    id: state.id.clone(),
                },
            ));
        }
        check_duplicate_ids_inner(&state.states, seen, diags);
        for region in &state.regions {
            check_duplicate_ids_inner(&region.states, seen, diags);
        }
    }
}

fn validate_state(state: &StateDefinition, all_ids: &HashSet<String>, diags: &mut Vec<Diagnostic>) {
    match state.state_type {
        StateType::Agentic => {
            if state.agent.is_none() {
                diags.push(Diagnostic::error(
                    "E010",
                    format!("Agentic state `{}` has no agent reference", state.id),
                    DiagnosticLocation::State {
                        id: state.id.clone(),
                    },
                ));
            }
            if state.on.is_empty() {
                diags.push(Diagnostic::warning(
                    "W010",
                    format!(
                        "Agentic state `{}` declares no output event handlers",
                        state.id
                    ),
                    DiagnosticLocation::State {
                        id: state.id.clone(),
                    },
                ));
            }
        }
        StateType::Compound => {
            if state.states.is_empty() {
                diags.push(Diagnostic::error(
                    "E020",
                    format!("Compound state `{}` has no child states", state.id),
                    DiagnosticLocation::State {
                        id: state.id.clone(),
                    },
                ));
            }
            if state.initial.is_none() {
                diags.push(Diagnostic::error(
                    "E021",
                    format!("Compound state `{}` has no initial child", state.id),
                    DiagnosticLocation::State {
                        id: state.id.clone(),
                    },
                ));
            }
        }
        StateType::Parallel => {
            if state.regions.is_empty() && state.states.is_empty() {
                diags.push(Diagnostic::error(
                    "E030",
                    format!("Parallel state `{}` has no regions", state.id),
                    DiagnosticLocation::State {
                        id: state.id.clone(),
                    },
                ));
            }
        }
        StateType::Subworkflow => {
            if state.workflow_ref.is_none() {
                diags.push(Diagnostic::error(
                    "E040",
                    format!("Subworkflow state `{}` has no workflow_ref", state.id),
                    DiagnosticLocation::State {
                        id: state.id.clone(),
                    },
                ));
            }
        }
        _ if state.capabilities.as_ref().map(|c| c.elevate) == Some(true) => {
            diags.push(Diagnostic::warning(
                "W050",
                format!(
                    "State `{}` declares capability elevation (`elevate: true`)",
                    state.id
                ),
                DiagnosticLocation::State {
                    id: state.id.clone(),
                },
            ));
        }
        _ => {}
    }

    // Check all transition targets exist.
    // History pseudo-state targets use the convention `"<compound_id>.history"` —
    // they are virtual targets resolved at runtime, not declared state IDs.
    for (event_type, specs) in &state.on {
        for spec in specs {
            let target = spec.target.as_ref();
            let is_history_pseudo = target
                .strip_suffix(".history")
                .map(|base| all_ids.contains(base))
                .unwrap_or(false);

            if !is_history_pseudo && !all_ids.contains(target) {
                let tid = TransitionId::new(format!("{}_on_{}", state.id, event_type));
                diags.push(Diagnostic::error(
                    "E005",
                    format!(
                        "Transition target `{}` in state `{}` on event `{}` does not exist",
                        spec.target, state.id, event_type
                    ),
                    DiagnosticLocation::Transition { id: tid },
                ));
            }
        }
    }

    // Check priority ties (multiple transitions on the same event with same priority)
    check_transition_priority_ties(state, diags);

    // Recurse
    for child in &state.states {
        validate_state(child, all_ids, diags);
    }
    for region in &state.regions {
        for child in &region.states {
            validate_state(child, all_ids, diags);
        }
    }
}

fn check_transition_priority_ties(state: &StateDefinition, diags: &mut Vec<Diagnostic>) {
    // Spec §8.2: "Ties are a validation error and MUST be reported by the validator."
    // A tie occurs when two or more transitions on the same event in the same state
    // share the same priority value.
    for (event_type, specs) in &state.on {
        let mut seen_priorities: Vec<i32> = Vec::new();
        for spec in specs {
            if seen_priorities.contains(&spec.priority) {
                let tid = TransitionId::new(format!("{}_on_{}", state.id, event_type));
                diags.push(Diagnostic::error(
                    "E011",
                    format!(
                        "Ambiguous transition priority: state `{}` has multiple transitions \
                         on event `{}` with the same priority {}",
                        state.id, event_type, spec.priority
                    ),
                    DiagnosticLocation::Transition { id: tid },
                ));
                // Report once per tie group, not once per extra duplicate.
                break;
            }
            seen_priorities.push(spec.priority);
        }
    }
}

/// Scan guard expressions for `data.<name>` references and report any `<name>`
/// that is not declared in `data_schema.fields`.
///
/// Uses a simple string scan rather than full CEL AST analysis: locate every
/// occurrence of `"data."` in the expression and extract the identifier that
/// follows. False positives (e.g., inside string literals) are rare in practice
/// and the check is warning-level so they do not block compilation.
fn check_data_schema_guard_refs(
    states: &[StateDefinition],
    schema_fields: &std::collections::HashMap<String, String>,
    diags: &mut Vec<Diagnostic>,
) {
    for state in states {
        for (event_type, specs) in &state.on {
            for spec in specs {
                if let Some(expr) = &spec.guard {
                    for field_name in extract_data_field_refs(expr) {
                        if !schema_fields.contains_key(&field_name) {
                            let tid = TransitionId::new(format!("{}_on_{}", state.id, event_type));
                            diags.push(Diagnostic::warning(
                                "E012",
                                format!(
                                    "Guard in state `{}` references `data.{}` which is not \
                                     declared in data_schema.fields",
                                    state.id, field_name
                                ),
                                DiagnosticLocation::Guard {
                                    state_id: state.id.clone(),
                                    transition_id: tid,
                                },
                            ));
                        }
                    }
                }
            }
        }
        check_data_schema_guard_refs(&state.states, schema_fields, diags);
        for region in &state.regions {
            check_data_schema_guard_refs(&region.states, schema_fields, diags);
        }
    }
}

/// Extract all `data.<identifier>` field references from a CEL expression string.
/// Returns unique field names found.
fn extract_data_field_refs(expr: &str) -> Vec<String> {
    let prefix = "data.";
    let mut found = Vec::new();
    let mut search = expr;
    while let Some(pos) = search.find(prefix) {
        let after = &search[pos + prefix.len()..];
        // Collect the identifier: alphanumeric + underscore characters.
        let field: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !field.is_empty() && !found.contains(&field) {
            found.push(field);
        }
        // Advance past this occurrence.
        search = &search[pos + prefix.len()..];
    }
    found
}

fn check_guards(states: &[StateDefinition], diags: &mut Vec<Diagnostic>) {
    for state in states {
        for (event_type, specs) in &state.on {
            for spec in specs {
                if let Some(expr) = &spec.guard
                    && let Err(e) = CompiledGuard::compile(expr)
                {
                    let tid = TransitionId::new(format!("{}_on_{}", state.id, event_type));
                    diags.push(Diagnostic::error(
                        "E006",
                        format!("Guard expression error: {e}"),
                        DiagnosticLocation::Guard {
                            state_id: state.id.clone(),
                            transition_id: tid,
                        },
                    ));
                }
            }
        }
        check_guards(&state.states, diags);
        for region in &state.regions {
            check_guards(&region.states, diags);
        }
    }
}

fn compile_guards(
    states: &[StateDefinition],
    guards: &mut HashMap<(StateId, String), CompiledGuard>,
) -> Result<(), CompileError> {
    for state in states {
        for (event_type, specs) in &state.on {
            for spec in specs {
                if let Some(expr) = &spec.guard {
                    let guard = CompiledGuard::compile(expr).map_err(|source| {
                        CompileError::GuardCompile {
                            state_id: state.id.clone(),
                            transition_id: TransitionId::new(format!(
                                "{}_on_{}",
                                state.id, event_type
                            )),
                            source,
                        }
                    })?;
                    // Note: if multiple transitions exist for the same event, later
                    // guards overwrite earlier ones in the compiled map. The runtime
                    // evaluates guards directly from the state definition.
                    guards.insert((state.id.clone(), event_type.clone()), guard);
                }
            }
        }
        compile_guards(&state.states, guards)?;
        for region in &state.regions {
            compile_guards(&region.states, guards)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        state::{StateDefinition, StateType, TransitionSpec},
        workflow::WorkflowDocument,
    };

    fn minimal_atomic_doc() -> WorkflowDocument {
        WorkflowDocument {
            schema_version: "1.0.0".into(),
            id: "test-workflow".into(),
            version: "1.0.0".into(),
            name: "Test".into(),
            description: None,
            inputs: vec![],
            outputs: vec![],
            data_schema: Default::default(),
            policy: Default::default(),
            agents: vec![],
            states: vec![
                StateDefinition {
                    id: "start".into(),
                    name: "Start".into(),
                    state_type: StateType::Atomic,
                    agent: None,
                    prompt: None,
                    input: Default::default(),
                    context: None,
                    model: None,
                    capabilities: None,
                    limits: None,
                    states: vec![],
                    regions: vec![],
                    completion: None,
                    history: None,
                    initial: None,
                    workflow_ref: None,
                    ports: None,
                    authorized_roles: vec![],
                    on_entry: vec![],
                    on_exit: vec![],
                    retry: None,
                    timeout: None,
                    on: {
                        let mut m = std::collections::HashMap::new();
                        m.insert(
                            "done".into(),
                            vec![TransitionSpec {
                                target: "end".into(),
                                guard: None,
                                priority: 0,
                                actions: vec![],
                                kind: Default::default(),
                            }],
                        );
                        m
                    },
                    output_schemas: Default::default(),
                    _editor: serde_json::Value::Null,
                },
                StateDefinition {
                    id: "end".into(),
                    name: "End".into(),
                    state_type: StateType::Final,
                    agent: None,
                    prompt: None,
                    input: Default::default(),
                    context: None,
                    model: None,
                    capabilities: None,
                    limits: None,
                    states: vec![],
                    regions: vec![],
                    completion: None,
                    history: None,
                    initial: None,
                    workflow_ref: None,
                    ports: None,
                    authorized_roles: vec![],
                    on_entry: vec![],
                    on_exit: vec![],
                    retry: None,
                    timeout: None,
                    on: Default::default(),
                    output_schemas: Default::default(),
                    _editor: serde_json::Value::Null,
                },
            ],
            initial: "start".into(),
            _editor: serde_json::Value::Null,
        }
    }

    #[test]
    fn valid_document_produces_no_errors() {
        let doc = minimal_atomic_doc();
        let diags = validate(&doc);
        let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();
        assert!(errors.is_empty(), "unexpected errors: {errors:#?}");
    }

    #[test]
    fn missing_initial_is_an_error() {
        let mut doc = minimal_atomic_doc();
        doc.initial = "nonexistent".into();
        let diags = validate(&doc);
        assert!(diags.iter().any(|d| d.code == "E003"));
    }

    #[test]
    fn bad_transition_target_is_an_error() {
        let mut doc = minimal_atomic_doc();
        doc.states[0].on.get_mut("done").unwrap()[0].target = "ghost".into();
        let diags = validate(&doc);
        assert!(diags.iter().any(|d| d.code == "E005"));
    }

    #[test]
    fn compile_succeeds_on_valid_doc() {
        let doc = minimal_atomic_doc();
        assert!(compile(doc).is_ok());
    }

    #[test]
    fn priority_tie_is_an_error() {
        let mut doc = minimal_atomic_doc();
        // Add a second transition on "done" with the same priority (0) → tie.
        doc.states[0]
            .on
            .get_mut("done")
            .unwrap()
            .push(TransitionSpec {
                target: "end".into(),
                guard: Some("true".into()),
                priority: 0,
                actions: vec![],
                kind: Default::default(),
            });
        let diags = validate(&doc);
        assert!(
            diags.iter().any(|d| d.code == "E011"),
            "expected E011 priority-tie diagnostic, got: {diags:#?}"
        );
    }

    #[test]
    fn different_priorities_no_tie_error() {
        let mut doc = minimal_atomic_doc();
        // Add a second transition on "done" with a *different* priority → no tie.
        doc.states[0]
            .on
            .get_mut("done")
            .unwrap()
            .push(TransitionSpec {
                target: "end".into(),
                guard: Some("true".into()),
                priority: 1,
                actions: vec![],
                kind: Default::default(),
            });
        let diags = validate(&doc);
        assert!(
            !diags.iter().any(|d| d.code == "E011"),
            "unexpected E011 diagnostic: {diags:#?}"
        );
    }

    // ── E4: data_schema guard reference validation ────────────────────────────

    /// E012 is emitted when a guard references `data.X` but X is not in
    /// `data_schema.fields` (and the schema is non-empty).
    #[test]
    fn undeclared_data_field_in_guard_emits_e012() {
        let mut doc = minimal_atomic_doc();
        // Declare a data_schema with one field "score".
        doc.data_schema.fields.insert("score".into(), "f64".into());
        // Guard references "data.undeclared_field" which is not in the schema.
        doc.states[0].on.get_mut("done").unwrap()[0].guard =
            Some("data.undeclared_field == true".into());
        let diags = validate(&doc);
        assert!(
            diags.iter().any(|d| d.code == "E012"),
            "expected E012 for undeclared data field, got: {diags:#?}"
        );
    }

    /// No E012 when the referenced field IS declared in data_schema.
    #[test]
    fn declared_data_field_in_guard_no_e012() {
        let mut doc = minimal_atomic_doc();
        doc.data_schema
            .fields
            .insert("approved".into(), "bool".into());
        doc.states[0].on.get_mut("done").unwrap()[0].guard = Some("data.approved == true".into());
        let diags = validate(&doc);
        assert!(
            !diags.iter().any(|d| d.code == "E012"),
            "unexpected E012 for declared data field: {diags:#?}"
        );
    }

    /// No E012 when data_schema is empty (opt-in: validation only fires when
    /// the schema is explicitly declared).
    #[test]
    fn empty_data_schema_no_e012() {
        let mut doc = minimal_atomic_doc();
        // data_schema.fields is empty (default).
        doc.states[0].on.get_mut("done").unwrap()[0].guard = Some("data.any_field == 42".into());
        let diags = validate(&doc);
        assert!(
            !diags.iter().any(|d| d.code == "E012"),
            "E012 must not fire when data_schema is empty: {diags:#?}"
        );
    }

    // ── E4: extract_data_field_refs unit tests ────────────────────────────────

    #[test]
    fn extract_data_field_refs_single() {
        let refs = extract_data_field_refs("data.score > 0.5");
        assert_eq!(refs, vec!["score"]);
    }

    #[test]
    fn extract_data_field_refs_multiple() {
        let mut refs = extract_data_field_refs("data.a == true && data.b == false");
        refs.sort();
        assert_eq!(refs, vec!["a", "b"]);
    }

    #[test]
    fn extract_data_field_refs_deduplicates() {
        let refs = extract_data_field_refs("data.x == 1 || data.x == 2");
        assert_eq!(refs, vec!["x"]);
    }

    #[test]
    fn extract_data_field_refs_empty_when_no_data_prefix() {
        let refs = extract_data_field_refs("approved == true");
        assert!(refs.is_empty());
    }
}
