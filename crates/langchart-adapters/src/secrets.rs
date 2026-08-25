//! Secrets adapter: resolve named secret references to opaque values.
//!
//! Secrets are NEVER serialized to checkpoints, event records, or logs.
//! The broker resolves them in-flight and discards the resolved value
//! immediately after use.

use async_trait::async_trait;
use langchart_model::id::SecretRef;
use std::collections::HashMap;

/// An opaque resolved secret value.
///
/// **Never log, serialize, or clone this value beyond its immediate use.**
#[derive(Clone)]
pub struct SecretValue(pub String);

impl std::fmt::Debug for SecretValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretValue(***)")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SecretsError {
    #[error("secret `{0}` not found")]
    NotFound(SecretRef),
    #[error("secrets backend error: {0}")]
    Backend(String),
}

/// Resolves named secret references to their current values.
///
/// The built-in implementation is [`HostMapSecretsAdapter`].
/// Host applications may implement this trait to delegate to a vault,
/// OS keychain, AWS Secrets Manager, or any other secrets backend.
#[async_trait]
pub trait SecretsAdapter: Send + Sync {
    async fn get(&self, secret_ref: &SecretRef) -> Result<SecretValue, SecretsError>;
}

// ── Built-in: host-provided map ───────────────────────────────────────────────

/// The default `SecretsAdapter` backed by a `HashMap` provided by the host
/// application at run start.
///
/// ```rust
/// use langchart_adapters::secrets::{HostMapSecretsAdapter, SecretValue};
/// use langchart_model::id::SecretRef;
/// use std::collections::HashMap;
///
/// let mut map = HashMap::new();
/// map.insert("openai_key".to_string(), SecretValue("sk-...".to_string()));
/// let adapter = HostMapSecretsAdapter::new(map);
/// ```
pub struct HostMapSecretsAdapter {
    map: HashMap<String, SecretValue>,
}

impl HostMapSecretsAdapter {
    pub fn new(map: HashMap<String, SecretValue>) -> Self {
        Self { map }
    }

    pub fn empty() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}

#[async_trait]
impl SecretsAdapter for HostMapSecretsAdapter {
    async fn get(&self, secret_ref: &SecretRef) -> Result<SecretValue, SecretsError> {
        self.map
            .get(&secret_ref.0)
            .cloned()
            .ok_or_else(|| SecretsError::NotFound(secret_ref.clone()))
    }
}
