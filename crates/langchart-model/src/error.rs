//! Model-layer error types.

use crate::id::{StateId, TransitionId, WorkflowId, WorkflowVersion};
use thiserror::Error;

/// Errors produced during workflow document loading and normalization.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("unsupported schema version `{found}` (expected `{expected}`)")]
    UnsupportedVersion { found: String, expected: String },

    #[error("missing required field `{field}` in {context}")]
    MissingField {
        field: &'static str,
        context: String,
    },
}

/// A single validation diagnostic — may be an error, warning, or hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: &'static str,
    pub message: String,
    /// The state, transition, or other element the diagnostic refers to.
    pub location: Option<DiagnosticLocation>,
}

impl Diagnostic {
    pub fn error(
        code: &'static str,
        message: impl Into<String>,
        location: impl Into<Option<DiagnosticLocation>>,
    ) -> Self {
        Self {
            severity: Severity::Error,
            code,
            message: message.into(),
            location: location.into(),
        }
    }

    pub fn warning(
        code: &'static str,
        message: impl Into<String>,
        location: impl Into<Option<DiagnosticLocation>>,
    ) -> Self {
        Self {
            severity: Severity::Warning,
            code,
            message: message.into(),
            location: location.into(),
        }
    }

    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Hint,
}

/// Where in the workflow document a diagnostic originates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticLocation {
    Workflow {
        id: WorkflowId,
        version: WorkflowVersion,
    },
    State {
        id: StateId,
    },
    Transition {
        id: TransitionId,
    },
    Guard {
        state_id: StateId,
        transition_id: TransitionId,
    },
}

/// Errors produced when compiling a validated workflow document into an
/// executable representation.
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("validation failed with {0} error(s)")]
    ValidationFailed(usize),
}

/// Errors from CEL guard compilation or evaluation.
#[derive(Debug, Error)]
pub enum GuardError {
    #[error("CEL parse error: {0}")]
    Parse(String),

    #[error("CEL program error: {0}")]
    Program(String),

    #[error("CEL extension function `{name}` is not in the approved whitelist")]
    DisallowedExtension { name: String },

    #[error("guard evaluation error: {0}")]
    Eval(String),

    #[error("guard did not return a boolean value")]
    NotBoolean,
}
