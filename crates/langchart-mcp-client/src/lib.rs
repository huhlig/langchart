//! # langchart-mcp-client
//!
//! [`McpAdapter`] implementation backed by [`rmcp`] client connections.
//!
//! ## Architecture
//!
//! - **[`LangchartMcpAdapter`]** implements `McpAdapter`; it holds a shared
//!   registry of connected MCP servers.
//! - **[`McpClientRegistry`]** maps `ServerId → RunningService` (the live
//!   rmcp client handle).  Connections are added by the host at startup via
//!   [`McpClientRegistry::connect_stdio`].
//!
//! ## Quick-start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use langchart_model::id::ServerId;
//! use langchart_mcp_client::{McpClientRegistry, LangchartMcpAdapter};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let registry = McpClientRegistry::new();
//!
//! // Connect to a stdio MCP server (spawns a child process).
//! registry.connect_stdio(
//!     ServerId::new("my-tools"),
//!     "uvx",
//!     &["my-mcp-server"],
//! ).await?;
//!
//! let adapter: Arc<dyn langchart_adapters::mcp::McpAdapter> =
//!     Arc::new(LangchartMcpAdapter::new(Arc::new(registry)));
//! # Ok(())
//! # }
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use langchart_adapters::mcp::{
    CREDENTIALS_META_KEY, McpAdapter, McpCredential, McpError, ResourceContent, ToolDefinition,
};
use langchart_model::id::{IdempotencyKey, ServerId, ToolName};
use rmcp::{
    ServiceExt,
    model::{
        CallToolRequestParams, Meta, PaginatedRequestParams, ReadResourceRequestParams,
        ResourceContents,
    },
    service::{RoleClient, RunningService},
    transport::child_process::TokioChildProcess,
};
use tokio::process::Command;
use tracing::debug;

// ── Minimal no-op client handler ─────────────────────────────────────────────
//
// `impl ClientHandler for ()` is provided by rmcp: uses `ClientInfo::default()`
// and silently ignores all server-initiated callbacks.  We use `()` to avoid
// any struct construction of `#[non_exhaustive]` rmcp types.

// ── Server connection handle ──────────────────────────────────────────────────

/// A live connection to one MCP server.
///
/// Holds the `RunningService` which owns the background I/O task and the
/// child process lifetime.  `RunningService` derefs to `Peer<RoleClient>` for
/// sending requests.
struct McpConnection {
    /// The running service — kept alive so the child process is not dropped.
    running: RunningService<RoleClient, ()>,
}

// ── Registry ──────────────────────────────────────────────────────────────────

/// Thread-safe registry mapping [`ServerId`]s to live MCP connections.
///
/// Connections are established by the host at startup via
/// [`McpClientRegistry::connect_stdio`] and then shared read-only across
/// concurrent requests.
pub struct McpClientRegistry {
    connections: RwLock<HashMap<ServerId, Arc<McpConnection>>>,
}

impl McpClientRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
        }
    }

    /// Connect to a stdio-based MCP server by spawning a child process.
    ///
    /// `program` is the binary to execute; `args` are its command-line
    /// arguments.  The child's stdout/stdin are used as the MCP transport.
    ///
    /// # Errors
    ///
    /// Returns an error if the child process cannot be spawned, the transport
    /// cannot be constructed, or the MCP handshake fails.
    pub async fn connect_stdio(
        &self,
        server_id: ServerId,
        program: impl AsRef<std::ffi::OsStr>,
        args: &[impl AsRef<std::ffi::OsStr>],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        let mut cmd = Command::new(program);
        cmd.args(args);

        // `TokioChildProcess::new` spawns the child and wires its stdio.
        // `()` implements `ClientHandler` with all-default callbacks.
        let transport = TokioChildProcess::new(cmd)?;

        let running = ()
            .serve(transport)
            .await
            .map_err(|e| format!("MCP handshake failed for `{server_id}`: {e}"))?;

        debug!(server_id = %server_id, "MCP server connected");

        let conn = Arc::new(McpConnection { running });
        self.connections.write().unwrap().insert(server_id, conn);
        Ok(())
    }

    /// Return the connection for `server_id`, if any.
    fn get(&self, server_id: &ServerId) -> Option<Arc<McpConnection>> {
        self.connections.read().unwrap().get(server_id).cloned()
    }
}

impl Default for McpClientRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── McpAdapter implementation ─────────────────────────────────────────────────

/// [`McpAdapter`] implementation that routes requests to the appropriate
/// connected MCP server via [`McpClientRegistry`].
pub struct LangchartMcpAdapter {
    registry: Arc<McpClientRegistry>,
}

impl LangchartMcpAdapter {
    /// Create a new adapter wrapping `registry`.
    pub fn new(registry: Arc<McpClientRegistry>) -> Self {
        Self { registry }
    }
}

fn build_call_tool_params(
    tool: &ToolName,
    args: serde_json::Value,
    credentials: &[McpCredential],
) -> CallToolRequestParams {
    let arguments = match args {
        serde_json::Value::Object(m) => Some(m),
        serde_json::Value::Null => None,
        other => {
            // Wrap non-object args under an "input" key.
            let mut map = serde_json::Map::new();
            map.insert("input".into(), other);
            Some(map)
        }
    };

    let mut params = CallToolRequestParams::new(tool.0.clone());
    params.arguments = arguments;
    if !credentials.is_empty() {
        let values = credentials
            .iter()
            .map(|credential| {
                (
                    credential.secret_ref.0.clone(),
                    serde_json::Value::String(credential.value.0.clone()),
                )
            })
            .collect();
        let mut meta = serde_json::Map::new();
        meta.insert(
            CREDENTIALS_META_KEY.to_owned(),
            serde_json::Value::Object(values),
        );
        params.meta = Some(Meta(meta));
    }
    params
}

#[async_trait]
impl McpAdapter for LangchartMcpAdapter {
    async fn call_tool(
        &self,
        server_id: &ServerId,
        tool: &ToolName,
        args: serde_json::Value,
        credentials: &[McpCredential],
        _key: Option<&IdempotencyKey>,
    ) -> Result<serde_json::Value, McpError> {
        let conn = self
            .registry
            .get(server_id)
            .ok_or_else(|| McpError::ServerNotFound {
                server_id: server_id.clone(),
            })?;

        let params = build_call_tool_params(tool, args, credentials);

        let result = conn
            .running
            .call_tool(params)
            .await
            .map_err(|e| McpError::Call(e.to_string()))?;

        if result.is_error == Some(true) {
            let msg = result
                .content
                .iter()
                .filter_map(|c| c.as_text())
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(McpError::Call(msg));
        }

        // Fold content items into a JSON array.
        let items: Vec<serde_json::Value> = result
            .content
            .iter()
            .filter_map(|c| serde_json::to_value(c).ok())
            .collect();

        Ok(serde_json::Value::Array(items))
    }

    async fn list_tools(&self, server_id: &ServerId) -> Result<Vec<ToolDefinition>, McpError> {
        let conn = self
            .registry
            .get(server_id)
            .ok_or_else(|| McpError::ServerNotFound {
                server_id: server_id.clone(),
            })?;

        let result = conn
            .running
            .list_tools(Some(PaginatedRequestParams::default()))
            .await
            .map_err(|e| McpError::Call(e.to_string()))?;

        let tools = result
            .tools
            .into_iter()
            .map(|t| ToolDefinition {
                name: ToolName::new(t.name.to_string()),
                description: t.description.unwrap_or_default().to_string(),
                input_schema: serde_json::to_value(&t.input_schema).unwrap_or_default(),
            })
            .collect();

        Ok(tools)
    }

    async fn read_resource(
        &self,
        server_id: &ServerId,
        uri: &str,
    ) -> Result<ResourceContent, McpError> {
        let conn = self
            .registry
            .get(server_id)
            .ok_or_else(|| McpError::ServerNotFound {
                server_id: server_id.clone(),
            })?;

        let params = ReadResourceRequestParams::new(uri);

        let result = conn
            .running
            .read_resource(params)
            .await
            .map_err(|e| McpError::Call(e.to_string()))?;

        // Take the first content item and map it to our `ResourceContent` type.
        let first =
            result
                .contents
                .into_iter()
                .next()
                .ok_or_else(|| McpError::ResourceNotFound {
                    server_id: server_id.clone(),
                    uri: uri.to_owned(),
                })?;

        let content = match first {
            ResourceContents::TextResourceContents {
                text, mime_type, ..
            } => ResourceContent {
                uri: uri.to_owned(),
                content_type: mime_type.unwrap_or_else(|| "text/plain".to_owned()),
                bytes: text.into_bytes(),
            },
            ResourceContents::BlobResourceContents {
                blob, mime_type, ..
            } => {
                // `blob` is a base64-encoded string in MCP.
                let decoded = decode_base64(&blob).unwrap_or_else(|_| blob.into_bytes());
                ResourceContent {
                    uri: uri.to_owned(),
                    content_type: mime_type
                        .unwrap_or_else(|| "application/octet-stream".to_owned()),
                    bytes: decoded,
                }
            }
        };

        Ok(content)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Decode a base64-encoded string. Returns `Err(())` on invalid input.
fn decode_base64(s: &str) -> Result<Vec<u8>, ()> {
    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4 + 1);
    let mut buf = 0u32;
    let mut bits = 0u8;

    for ch in s.bytes() {
        let val = match ch {
            b'A'..=b'Z' => (ch - b'A') as u32,
            b'a'..=b'z' => (ch - b'a' + 26) as u32,
            b'0'..=b'9' => (ch - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(()),
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::mcp::McpError;
    use langchart_adapters::secrets::SecretValue;
    use langchart_model::id::{SecretRef, ServerId, ToolName};
    use std::sync::Arc;

    // ── Registry ──────────────────────────────────────────────────────────────

    #[test]
    fn new_registry_is_empty() {
        let registry = McpClientRegistry::new();
        assert!(registry.get(&ServerId::new("any")).is_none());
    }

    #[test]
    fn default_registry_is_empty() {
        let registry = McpClientRegistry::default();
        assert!(registry.get(&ServerId::new("any")).is_none());
    }

    #[test]
    fn credentials_are_protocol_metadata_not_tool_arguments() {
        let params = build_call_tool_params(
            &ToolName::new("write_note"),
            serde_json::json!({ "body": "hello" }),
            &[McpCredential {
                secret_ref: SecretRef::new("token"),
                value: SecretValue("secret-value".into()),
            }],
        );

        assert_eq!(
            params.arguments,
            Some(serde_json::Map::from_iter([(
                "body".into(),
                serde_json::Value::String("hello".into()),
            )]))
        );
        assert_eq!(
            params
                .meta
                .expect("credential metadata")
                .0
                .get(CREDENTIALS_META_KEY),
            Some(&serde_json::json!({ "token": "secret-value" }))
        );
    }

    // ── Adapter: unknown server errors ────────────────────────────────────────

    #[tokio::test]
    async fn call_tool_unknown_server_returns_not_found() {
        let registry = Arc::new(McpClientRegistry::new());
        let adapter = LangchartMcpAdapter::new(registry);
        let err = adapter
            .call_tool(
                &ServerId::new("missing"),
                &ToolName::new("some_tool"),
                serde_json::json!({}),
                &[],
                None,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, McpError::ServerNotFound { .. }));
    }

    #[tokio::test]
    async fn list_tools_unknown_server_returns_not_found() {
        let registry = Arc::new(McpClientRegistry::new());
        let adapter = LangchartMcpAdapter::new(registry);
        let err = adapter
            .list_tools(&ServerId::new("gone"))
            .await
            .unwrap_err();
        assert!(matches!(err, McpError::ServerNotFound { .. }));
    }

    #[tokio::test]
    async fn read_resource_unknown_server_returns_not_found() {
        let registry = Arc::new(McpClientRegistry::new());
        let adapter = LangchartMcpAdapter::new(registry);
        let err = adapter
            .read_resource(&ServerId::new("nope"), "file:///test.txt")
            .await
            .unwrap_err();
        assert!(matches!(err, McpError::ServerNotFound { .. }));
    }

    // ── Base64 decoder ────────────────────────────────────────────────────────

    #[test]
    fn decode_base64_hello_world() {
        // "Hello" in base64 is "SGVsbG8="
        let decoded = decode_base64("SGVsbG8=").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn decode_base64_empty_string() {
        let decoded = decode_base64("").unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn decode_base64_invalid_chars_returns_err() {
        assert!(decode_base64("!!!invalid!!!").is_err());
    }

    #[test]
    fn decode_base64_no_padding_works() {
        // "abc" → "YWJj" (no padding needed)
        let decoded = decode_base64("YWJj").unwrap();
        assert_eq!(decoded, b"abc");
    }
}
