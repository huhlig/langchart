//! Idempotent external call outbox.
//!
//! Before forwarding any external call (MCP tool call, artifact commit),
//! the runtime records the call with a stable idempotency key. On checkpoint
//! recovery, already-confirmed calls are not re-executed.
//!
//! This is the minimum safe outbox: in-memory for now, designed so that
//! Phase 6 can replace it with a durable database-backed outbox.

use langchart_model::id::{IdempotencyKey, InvocationId};
use std::collections::HashMap;
use ulid::Ulid;

// ── Call record ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallStatus {
    /// Call recorded but not yet confirmed by the external system.
    Pending,
    /// Call completed successfully; result is stored.
    Confirmed,
    /// Call failed permanently.
    Failed,
}

#[derive(Debug, Clone)]
pub struct OutboxEntry {
    pub key: IdempotencyKey,
    pub invocation_id: InvocationId,
    pub description: String,
    pub status: CallStatus,
    /// Serialized result value (JSON), populated on confirmation.
    pub result: Option<serde_json::Value>,
}

// ── Outbox ────────────────────────────────────────────────────────────────────

/// Tracks in-flight and confirmed external calls for idempotent recovery.
#[derive(Default)]
pub struct Outbox {
    entries: HashMap<String, OutboxEntry>,
}

impl Outbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a new idempotency key for a call.
    pub fn new_key() -> IdempotencyKey {
        IdempotencyKey::new(Ulid::generate().to_string())
    }

    /// Record a pending call. Returns the key to pass to the adapter.
    pub fn record_pending(
        &mut self,
        invocation_id: InvocationId,
        description: impl Into<String>,
    ) -> IdempotencyKey {
        let key = Self::new_key();
        self.entries.insert(
            key.0.clone(),
            OutboxEntry {
                key: key.clone(),
                invocation_id,
                description: description.into(),
                status: CallStatus::Pending,
                result: None,
            },
        );
        key
    }

    /// Mark a call as confirmed with its result.
    pub fn confirm(&mut self, key: &IdempotencyKey, result: serde_json::Value) {
        if let Some(entry) = self.entries.get_mut(&key.0) {
            entry.status = CallStatus::Confirmed;
            entry.result = Some(result);
        }
    }

    /// Mark a call as permanently failed.
    pub fn fail(&mut self, key: &IdempotencyKey) {
        if let Some(entry) = self.entries.get_mut(&key.0) {
            entry.status = CallStatus::Failed;
        }
    }

    /// Check if a call was already confirmed (safe to skip on recovery).
    pub fn is_confirmed(&self, key: &IdempotencyKey) -> Option<&serde_json::Value> {
        self.entries.get(&key.0).and_then(|e| {
            if e.status == CallStatus::Confirmed {
                e.result.as_ref()
            } else {
                None
            }
        })
    }

    /// All pending entries (for checkpoint persistence).
    pub fn pending_entries(&self) -> Vec<&OutboxEntry> {
        self.entries
            .values()
            .filter(|e| e.status == CallStatus::Pending)
            .collect()
    }
}
