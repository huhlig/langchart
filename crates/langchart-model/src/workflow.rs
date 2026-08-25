//! Workflow document: the top-level structure loaded from JSON or YAML.

use crate::{
    id::{AgentId, AgentVersion, WorkflowId, WorkflowVersion},
    policy::{CapabilityPolicy, ContextPolicy, ModelPolicy},
    state::StateDefinition,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Schema version ────────────────────────────────────────────────────────────

/// The schema version embedded in every workflow document.
/// Follows semver; the runtime must reject documents whose major version
/// it does not understand.
pub const CURRENT_SCHEMA_VERSION: &str = "1.0.0";

// ── Agent definition ──────────────────────────────────────────────────────────

/// A reusable agent definition. May be declared inline in the workflow
/// document or referenced by `id@version` from an external registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    pub id: AgentId,
    pub version: AgentVersion,
    pub description: String,
    /// Path (or inline content) of the system prompt template.
    pub system_prompt: String,
    /// Default model policy. States may narrow (never widen) this.
    pub model_policy: ModelPolicy,
    /// Default context policy. States may narrow (never widen) this.
    #[serde(default)]
    pub default_context_policy: ContextPolicy,
    /// Default capability policy. States may narrow (never widen) this.
    #[serde(default)]
    pub default_capabilities: CapabilityPolicy,
    /// Declared output event type names this agent may emit.
    pub output_events: Vec<String>,
}

// ── Workflow input / output ports ─────────────────────────────────────────────

/// A typed input port declared on the workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputPort {
    pub name: String,
    /// RON type signature string (e.g. `"String"`, `"Option<u32>"`, `"MyEnum"`).
    pub type_sig: String,
    pub required: bool,
    pub description: Option<String>,
}

/// A typed output port declared on the workflow (populated from the final state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputPort {
    pub name: String,
    pub type_sig: String,
    pub description: Option<String>,
}

// ── Workflow data schema ──────────────────────────────────────────────────────

/// Declared type schema for workflow data fields.
/// Enables static CEL guard type-checking and runtime RON deserialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkflowDataSchema {
    /// `field_name → RON type signature`.
    pub fields: HashMap<String, String>,
}

// ── Event schema ──────────────────────────────────────────────────────────────

/// A minimal payload schema for a single event type.
///
/// Each key is a required field name; the value is a JSON type name
/// (`"string"`, `"number"`, `"boolean"`, `"array"`, `"object"`, `"null"`).
///
/// An event payload is **valid** if every declared field is present in the
/// JSON object and has the declared type. Extra fields are allowed (open
/// schema). Absent schema (`EventSchema { fields: {} }`) means any payload
/// is accepted.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventSchema {
    /// `field_name → JSON type name`.
    pub fields: HashMap<String, String>,
}

impl EventSchema {
    /// Validate a JSON object payload against this schema.
    ///
    /// Returns `Ok(())` if all required fields are present with the correct
    /// type, or an `Err` with the first violation message.
    pub fn validate(&self, payload: &serde_json::Value) -> Result<(), String> {
        if self.fields.is_empty() {
            return Ok(());
        }
        let obj = match payload.as_object() {
            Some(o) => o,
            None => {
                return Err(format!(
                    "expected a JSON object payload but got {}",
                    json_type_name(payload)
                ));
            }
        };
        for (field, expected_type) in &self.fields {
            match obj.get(field) {
                None => {
                    return Err(format!("required field '{field}' is missing from payload"));
                }
                Some(val) => {
                    let actual = json_type_name(val);
                    if actual != expected_type.as_str() {
                        return Err(format!(
                            "field '{field}': expected type '{expected_type}' but got '{actual}'"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Return the JSON type name for a value (matches JSON Schema type names).
fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

// ── Workflow policy ───────────────────────────────────────────────────────────

/// Workflow-level policies that apply to all states unless narrowed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkflowPolicy {
    /// Maximum allowed capability policy across all states.
    #[serde(default)]
    pub max_capabilities: CapabilityPolicy,
    /// Whether `event.unhandled` at the workflow level is a failure.
    #[serde(default)]
    pub unhandled_event_is_failure: bool,
}

// ── Workflow document ─────────────────────────────────────────────────────────

/// The canonical workflow document loaded from JSON or YAML.
/// This is the complete, denormalized representation as stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDocument {
    /// Schema version string — must be parseable as semver.
    pub schema_version: String,

    /// Unique stable identifier for this workflow definition.
    pub id: WorkflowId,
    /// Semantic version of this workflow definition.
    pub version: WorkflowVersion,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,

    /// Declared input ports.
    #[serde(default)]
    pub inputs: Vec<InputPort>,
    /// Declared output ports.
    #[serde(default)]
    pub outputs: Vec<OutputPort>,

    /// Typed schema for workflow data variables.
    #[serde(default)]
    pub data_schema: WorkflowDataSchema,

    /// Workflow-level policy.
    #[serde(default)]
    pub policy: WorkflowPolicy,

    /// Inline agent definitions. States reference these by `id@version`.
    #[serde(default)]
    pub agents: Vec<AgentDefinition>,

    /// The flat list of top-level states. The statechart is hierarchical;
    /// compound and parallel states embed their children.
    pub states: Vec<StateDefinition>,

    /// ID of the initial top-level state.
    pub initial: String,

    /// Non-semantic editor metadata. MUST NOT affect execution.
    #[serde(default)]
    pub _editor: serde_json::Value,
}

impl WorkflowDocument {
    /// Parse a workflow document from JSON.
    pub fn from_json(s: &str) -> Result<Self, crate::error::LoadError> {
        Ok(serde_json::from_str(s)?)
    }

    /// Parse a workflow document from YAML.
    pub fn from_yaml(s: &str) -> Result<Self, crate::error::LoadError> {
        Ok(serde_yaml::from_str(s)?)
    }

    /// Serialize the document to pretty-printed JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
