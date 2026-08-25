//! `langchart validate <workflow>` — validate a workflow document.

use anyhow::{Context, Result, bail};
use clap::Args;
use langchart_model::validation::validate;
use std::path::PathBuf;

/// Validate a workflow document (JSON or YAML). Exits 1 on errors.
#[derive(Debug, Args)]
pub struct ValidateArgs {
    /// Path to the workflow document (`.json` or `.yaml` / `.yml`).
    pub workflow: PathBuf,
}

pub async fn execute(args: ValidateArgs) -> Result<()> {
    let doc = load_workflow(&args.workflow)?;

    let diagnostics = validate(&doc);
    let errors: Vec<_> = diagnostics.iter().filter(|d| d.is_error()).collect();
    let warnings: Vec<_> = diagnostics.iter().filter(|d| !d.is_error()).collect();

    for w in &warnings {
        eprintln!("warning [{}]: {}", w.code, w.message);
    }
    for e in &errors {
        eprintln!("error   [{}]: {}", e.code, e.message);
    }

    if errors.is_empty() {
        println!(
            "✓ workflow `{}` is valid  ({} warning{})",
            doc.id,
            warnings.len(),
            if warnings.len() == 1 { "" } else { "s" },
        );
        Ok(())
    } else {
        bail!(
            "workflow `{}` has {} error{}",
            doc.id,
            errors.len(),
            if errors.len() == 1 { "" } else { "s" },
        )
    }
}

// ── Shared helper: load a workflow document from JSON or YAML ─────────────────

pub(crate) fn load_workflow(path: &PathBuf) -> Result<langchart_model::workflow::WorkflowDocument> {
    let src = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read `{}`", path.display()))?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "json" => serde_json::from_str(&src)
            .with_context(|| format!("invalid JSON in `{}`", path.display())),
        "yaml" | "yml" => serde_yaml::from_str(&src)
            .with_context(|| format!("invalid YAML in `{}`", path.display())),
        other => bail!(
            "unrecognised workflow file extension `.{other}` — use `.json`, `.yaml`, or `.yml`"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const VALID_JSON: &str = r#"{
        "schema_version": "1.0.0",
        "id": "test-wf",
        "version": "1.0.0",
        "name": "Test",
        "initial": "start",
        "states": [
            { "id": "start", "name": "Start", "type": "atomic",
              "on": { "go": [{ "target": "done", "priority": 0, "actions": [] }] }
            },
            { "id": "done", "name": "Done", "type": "final", "on": {} }
        ]
    }"#;

    const INVALID_JSON: &str = r#"{ not valid json"#;

    const VALID_YAML: &str = "
schema_version: \"1.0.0\"
id: yaml-wf
version: \"1.0.0\"
name: YAML Test
initial: s1
states:
  - id: s1
    name: S1
    type: atomic
    on:
      done:
        - target: s2
          priority: 0
          actions: []
  - id: s2
    name: S2
    type: final
    on: {}
";

    fn write_temp(content: &str, ext: &str) -> NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(ext).tempfile().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn load_valid_json_workflow() {
        let f = write_temp(VALID_JSON, ".json");
        let p = f.path().to_path_buf();
        let doc = load_workflow(&p).unwrap();
        assert_eq!(doc.id.0, "test-wf");
        assert_eq!(doc.states.len(), 2);
    }

    #[test]
    fn load_valid_yaml_workflow() {
        let f = write_temp(VALID_YAML, ".yaml");
        let p = f.path().to_path_buf();
        let doc = load_workflow(&p).unwrap();
        assert_eq!(doc.id.0, "yaml-wf");
    }

    #[test]
    fn load_invalid_json_returns_error() {
        let f = write_temp(INVALID_JSON, ".json");
        let p = f.path().to_path_buf();
        assert!(load_workflow(&p).is_err());
    }

    #[test]
    fn load_unknown_extension_returns_error() {
        let f = write_temp(VALID_JSON, ".toml");
        let p = f.path().to_path_buf();
        let err = load_workflow(&p).unwrap_err();
        assert!(err.to_string().contains("unrecognised"));
    }

    #[test]
    fn load_missing_file_returns_error() {
        let path = std::path::PathBuf::from("/nonexistent/path/workflow.json");
        assert!(load_workflow(&path).is_err());
    }

    #[tokio::test]
    async fn execute_valid_workflow_succeeds() {
        let f = write_temp(VALID_JSON, ".json");
        let args = ValidateArgs {
            workflow: f.path().to_owned(),
        };
        assert!(execute(args).await.is_ok());
    }

    #[tokio::test]
    async fn execute_bad_json_returns_error() {
        let f = write_temp(INVALID_JSON, ".json");
        let args = ValidateArgs {
            workflow: f.path().to_owned(),
        };
        assert!(execute(args).await.is_err());
    }
}
