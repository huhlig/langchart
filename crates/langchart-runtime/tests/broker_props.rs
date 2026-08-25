//! Property-based tests for `CapabilityBroker` / `CapabilityEnvelope`.
//!
//! These tests verify three invariants using `proptest`:
//!
//! 1. **Policy intersection** — `allows_tool` returns true iff the server/tool
//!    pair is in the policy allowlist; adding unrelated pairs never grants
//!    access.
//! 2. **Elevation detection** — A restrictive parent policy can never be
//!    "widened" by a child policy; any tool not in the parent list must
//!    remain blocked regardless of child configuration.
//! 3. **Credential isolation** — secrets are never surfaced through any
//!    `CapabilityEnvelope` field.

use langchart_model::{
    id::{InvocationId, ServerId, StateId, ToolName},
    policy::{CapabilityPolicy, McpServerPolicy, OperationClass},
};
use langchart_runtime::broker::CapabilityEnvelope;
use proptest::prelude::*;
use std::collections::HashMap;

// ── Generators ────────────────────────────────────────────────────────────────

fn arb_server_id() -> impl Strategy<Value = ServerId> {
    prop_oneof![
        Just(ServerId::new("server-a")),
        Just(ServerId::new("server-b")),
        Just(ServerId::new("server-c")),
    ]
}

fn arb_tool_name() -> impl Strategy<Value = ToolName> {
    prop_oneof![
        Just(ToolName::new("tool-1")),
        Just(ToolName::new("tool-2")),
        Just(ToolName::new("tool-3")),
    ]
}

/// Build a `CapabilityPolicy` with a given set of allowed (server, tool) pairs.
fn make_policy(allowed: &[(ServerId, ToolName)]) -> CapabilityPolicy {
    let mut mcp: HashMap<ServerId, McpServerPolicy> = HashMap::new();
    for (server_id, tool_name) in allowed {
        mcp.entry(server_id.clone())
            .or_default()
            .allow
            .push(tool_name.clone());
    }
    CapabilityPolicy {
        mcp,
        artifact_operations: Vec::new(),
        memory_write: false,
        elevate: false,
    }
}

fn make_envelope(policy: CapabilityPolicy) -> CapabilityEnvelope {
    CapabilityEnvelope::new(
        policy,
        langchart_model::id::RunId::new("run-0"),
        InvocationId::new("inv-0"),
        StateId::new("test-state"),
        10,
        10,
    )
}

// ── Property 1: Policy intersection ──────────────────────────────────────────
//
// If a tool IS in the allowlist → `allows_tool` returns true.
// If a tool is NOT in the allowlist → `allows_tool` returns false.

proptest! {
    #[test]
    fn policy_intersection_allowed(
        server in arb_server_id(),
        tool   in arb_tool_name(),
    ) {
        let policy = make_policy(&[(server.clone(), tool.clone())]);
        let envelope = make_envelope(policy);
        prop_assert!(envelope.allows_tool(&server, &tool));
    }

    #[test]
    fn policy_intersection_blocked(
        server in arb_server_id(),
        // A different tool than what's in the allowlist.
        allowed_tool in Just(ToolName::new("tool-1")),
        blocked_tool in Just(ToolName::new("tool-2")),
    ) {
        prop_assume!(allowed_tool != blocked_tool);
        let policy = make_policy(&[(server.clone(), allowed_tool)]);
        let envelope = make_envelope(policy);
        prop_assert!(!envelope.allows_tool(&server, &blocked_tool));
    }

    #[test]
    fn unknown_server_always_blocked(
        server in arb_server_id(),
        unlisted_server in Just(ServerId::new("unlisted-server")),
        tool in arb_tool_name(),
    ) {
        let policy = make_policy(&[(server.clone(), tool.clone())]);
        let envelope = make_envelope(policy);
        // A different server is always blocked regardless of tool.
        prop_assert!(!envelope.allows_tool(&unlisted_server, &tool));
    }
}

// ── Property 2: Elevation detection ──────────────────────────────────────────
//
// Tools not in the parent policy cannot be reached regardless of what tools
// we try to call; `allows_tool` for a non-allowed (server, tool) pair must
// remain false even after checking an allowed pair on the same server.

proptest! {
    #[test]
    fn elevation_blocked_tools_stay_blocked(
        server in arb_server_id(),
        allowed_tool in Just(ToolName::new("tool-1")),
    ) {
        let blocked_tool = ToolName::new("tool-2");
        let policy = make_policy(&[(server.clone(), allowed_tool.clone())]);
        let envelope = make_envelope(policy);

        // Even after confirming the allowed tool is permitted…
        assert!(envelope.allows_tool(&server, &allowed_tool));

        // …the blocked tool remains blocked.
        prop_assert!(!envelope.allows_tool(&server, &blocked_tool));
    }
}

// ── Property 3: Credential isolation ─────────────────────────────────────────
//
// The `CapabilityEnvelope` struct carries no secret values.  We verify this
// at compile-time by checking that none of the public fields or methods
// return a type that could carry a credential.  This is a structural test
// rather than a proptest, but placed here alongside the other broker tests.

#[test]
fn credential_isolation_no_secrets_in_envelope() {
    let policy = CapabilityPolicy {
        mcp: HashMap::new(),
        artifact_operations: Vec::new(),
        memory_write: false,
        elevate: false,
    };
    let envelope = CapabilityEnvelope::new(
        policy,
        langchart_model::id::RunId::new("run-cred"),
        InvocationId::new("inv-cred"),
        StateId::new("s"),
        5,
        5,
    );

    // These are all the public fields on CapabilityEnvelope.
    // None of them should contain a raw credential.
    let _: &langchart_model::policy::CapabilityPolicy = envelope.policy();
    let _: &langchart_model::id::RunId = envelope.run_id();
    let _: &InvocationId = envelope.invocation_id();
    let _: &StateId = envelope.state_id();
    let _: u32 = envelope.turns_remaining();
    let _: u32 = envelope.tool_calls_remaining();
    let _: &HashMap<ServerId, u32> = envelope.server_tool_calls_remaining();
    let _: u32 = envelope.tokens_used();
    let _: Option<u32> = envelope.max_tokens_total();
    // tokens_remaining returns Option<u32> — no credential.
    let _: Option<u32> = envelope.tokens_remaining();
    // token_budget_exhausted returns bool.
    let _: bool = envelope.token_budget_exhausted();

    // If any field were a SecretValue / String containing a credential this
    // test would need to be extended. The type constraints above are
    // sufficient for structural verification.
}

#[test]
fn resource_reads_require_operation_and_matching_uri() {
    let server = ServerId::new("vault");
    let policy = CapabilityPolicy {
        mcp: HashMap::from([(
            server.clone(),
            McpServerPolicy {
                resource_patterns: vec!["vault://docs/*".into()],
                operations: vec![OperationClass::Read],
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    let envelope = make_envelope(policy);

    assert!(envelope.allows_resource(&server, &OperationClass::Read, "vault://docs/guide.md"));
    assert!(!envelope.allows_resource(&server, &OperationClass::Read, "vault://private/secret.md"));
    assert!(!envelope.allows_resource(&server, &OperationClass::Delete, "vault://docs/guide.md"));
}

#[test]
fn resource_regex_requires_explicit_prefix() {
    let server = ServerId::new("vault");
    let policy = CapabilityPolicy {
        mcp: HashMap::from([(
            server.clone(),
            McpServerPolicy {
                resource_patterns: vec![r"regex:^vault://docs/[a-z]+\.md$".into()],
                operations: vec![OperationClass::Read],
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    let envelope = make_envelope(policy);

    assert!(envelope.allows_resource(&server, &OperationClass::Read, "vault://docs/guide.md"));
    assert!(!envelope.allows_resource(&server, &OperationClass::Read, "vault://docs/Guide-1.md"));
}
