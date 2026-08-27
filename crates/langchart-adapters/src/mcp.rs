//! MCP adapter: abstract over Model Context Protocol server calls.

use crate::secrets::SecretValue;
use async_trait::async_trait;
use langchart_model::id::{IdempotencyKey, SecretRef, ServerId, ToolName};
use serde::{Deserialize, Serialize};

/// Metadata about one tool exposed by an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: ToolName,
    pub description: String,
    /// JSON Schema object describing the tool's input parameters.
    pub input_schema: serde_json::Value,
}

/// The content of a resource read from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    pub uri: String,
    pub content_type: String,
    /// Raw bytes of the resource.
    pub bytes: Vec<u8>,
}

/// A credential resolved immediately before an MCP call.
///
/// Adapters must inject these into transport-level request metadata, never
/// ordinary tool arguments. The value's `Debug` implementation is redacted.
#[derive(Clone, Debug)]
pub struct McpCredential {
    pub secret_ref: SecretRef,
    pub value: SecretValue,
}

/// Namespaced MCP `_meta` key used for broker-resolved credentials.
pub const CREDENTIALS_META_KEY: &str = "langchart/credentials";

/// Namespaced MCP `_meta` key used to forward tool-call idempotency keys.
pub const IDEMPOTENCY_KEY_META_KEY: &str = "langchart/idempotency-key";

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("MCP server `{server_id}` not found")]
    ServerNotFound { server_id: ServerId },
    #[error("tool `{tool}` not found on server `{server_id}`")]
    ToolNotFound { server_id: ServerId, tool: ToolName },
    #[error("resource `{uri}` not found on server `{server_id}`")]
    ResourceNotFound { server_id: ServerId, uri: String },
    #[error("MCP call error: {0}")]
    Call(String),
    #[error("MCP transport error: {0}")]
    Transport(String),
}

/// Abstraction over MCP server tool and resource access.
///
/// The `CapabilityBroker` wraps calls to this adapter with policy enforcement
/// before forwarding. Implementors do NOT need to enforce capability policies.
#[async_trait]
pub trait McpAdapter: Send + Sync {
    /// Call a tool on an MCP server.
    /// Call a tool on an MCP server.
    ///
    /// `credentials` are short-lived values resolved by the broker and must be
    /// injected into protocol/transport metadata rather than tool arguments.
    /// `idempotency_key` should be passed to the underlying server when supported.
    /// A server must explicitly honor the key before retries can be considered safe.
    async fn call_tool(
        &self,
        server_id: &ServerId,
        tool_name: &ToolName,
        arguments: serde_json::Value,
        credentials: &[McpCredential],
        idempotency_key: Option<&IdempotencyKey>,
    ) -> Result<serde_json::Value, McpError>;

    /// List the tools exposed by an MCP server.
    async fn list_tools(&self, server_id: &ServerId) -> Result<Vec<ToolDefinition>, McpError>;

    /// Read a resource from an MCP server.
    async fn read_resource(
        &self,
        server_id: &ServerId,
        uri: &str,
    ) -> Result<ResourceContent, McpError>;
}
