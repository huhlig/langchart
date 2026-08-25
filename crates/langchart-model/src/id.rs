//! Stable identity types used throughout the system.
//!
//! Every ID is a newtype over a `String` (ULID or slug). Newtypes prevent
//! accidental interchange (passing a `RunId` where a `StateId` is required).
//! IDs are serialized as plain strings.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! id_type {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self { Self(s) }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self { Self(s.to_owned()) }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str { &self.0 }
        }
    };
}

id_type!(
    /// Identifies a workflow definition (e.g. `"content-review"`).
    WorkflowId
);

id_type!(
    /// Identifies a specific version of a workflow definition (semver string).
    WorkflowVersion
);

id_type!(
    /// Identifies a live workflow run instance (ULID).
    RunId
);

id_type!(
    /// Identifies a state within a workflow (stable slug, e.g. `"draft_scene"`).
    StateId
);

id_type!(
    /// Identifies a transition within a workflow.
    TransitionId
);

id_type!(
    /// Identifies a region within a parallel state.
    RegionId
);

id_type!(
    /// Identifies a reusable agent definition (e.g. `"content_analyst"`).
    AgentId
);

id_type!(
    /// Identifies a specific version of an agent definition (semver string).
    AgentVersion
);

id_type!(
    /// Identifies a single agent invocation within a run (ULID).
    InvocationId
);

id_type!(
    /// Identifies a durable artifact (e.g. `"draft-v1"`).
    ArtifactId
);

id_type!(
    /// Identifies a specific immutable version of an artifact (ULID or content hash).
    ArtifactVersion
);

id_type!(
    /// Identifies a change proposal on an artifact (ULID).
    ProposalId
);

id_type!(
    /// Identifies a checkpoint of a run snapshot (ULID).
    CheckpointId
);

id_type!(
    /// Identifies a runtime event record (ULID — lexicographically sortable by time).
    EventId
);

id_type!(
    /// Identifies an MCP server registered with the capability broker.
    ServerId
);

id_type!(
    /// Identifies an MCP tool name within a server.
    ToolName
);

id_type!(
    /// An idempotency key for external (MCP/artifact) calls to prevent duplicate
    /// execution on checkpoint recovery.
    IdempotencyKey
);

id_type!(
    /// A reference to a named secret declared in the workflow document.
    /// Never contains the secret value.
    SecretRef
);
