#![cfg(target_arch = "wasm32")]

use langchart_wasm::{
    compile_workflow, get_guard_errors, inspect_state, list_state_ids, list_transitions,
    reachability_analysis, schema_version, simulate_workflow, validate_workflow, workflow_summary,
};
use serde_json::Value;
use wasm_bindgen_test::wasm_bindgen_test;

const WORKFLOW: &str = r#"{
    "schema_version": "1.0.0",
    "id": "wasm-test",
    "version": "1.0.0",
    "name": "WASM test",
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

fn json(result: Result<String, wasm_bindgen::JsValue>) -> Value {
    serde_json::from_str(&result.expect("WASM export failed"))
        .expect("export returned invalid JSON")
}

#[wasm_bindgen_test]
fn validation_and_inspection_exports_execute_in_wasm() {
    assert_eq!(schema_version(), "1.0.0");
    assert_eq!(json(validate_workflow(WORKFLOW)), serde_json::json!([]));
    assert_eq!(json(compile_workflow(WORKFLOW))["ok"], true);
    assert_eq!(
        json(list_state_ids(WORKFLOW)),
        serde_json::json!(["start", "done"])
    );
    assert_eq!(json(inspect_state(WORKFLOW, "start"))["id"], "start");
    assert_eq!(json(get_guard_errors(WORKFLOW)), serde_json::json!([]));
    assert_eq!(json(workflow_summary(WORKFLOW))["state_counts"]["total"], 2);
    assert_eq!(
        json(list_transitions(WORKFLOW)).as_array().unwrap().len(),
        1
    );
    assert_eq!(
        json(reachability_analysis(WORKFLOW))["unreachable"],
        serde_json::json!([])
    );
}

#[wasm_bindgen_test]
fn simulation_export_executes_in_wasm() {
    let result = json(simulate_workflow(
        WORKFLOW,
        r#"{"inject":[{"event_type":"go","payload":{"source":"test"}}]}"#,
    ));

    assert_eq!(result["status"], "completed");
    assert_eq!(result["final_state"], "done");
    assert_eq!(result["steps"].as_array().unwrap().len(), 1);
}
