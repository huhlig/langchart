//! Policy-enforcing gateway between agent actors and all external systems.
//!
//! The `CapabilityBroker` is the security kernel of the runtime. Every call
//! from an agent actor to an LLM, MCP server, or memory system passes through
//! the broker. The broker:
//!
//! 1. Checks the active `CapabilityEnvelope` before forwarding.
//! 2. Records a `RuntimeEvent` for every call (before forwarding).
//! 3. Enforces turn and tool-call budgets.
//! 4. Resolves and injects secrets (never logs the resolved values).
//! 5. Rejects any call that is not permitted by the envelope.
//!
//! There is no "trusted" bypass path. If a call does not go through the broker,
//! it does not happen.

use crate::{
    artifact::{ArtifactContent, ArtifactError, ArtifactProposal, ArtifactStore},
    event::{EventSink, RuntimeEvent, RuntimeEventPayload},
    llm::{LlmAdapter, LlmRequest, LlmResponse},
    mcp::{McpAdapter, McpCredential, ResourceContent},
    memory::{MemoryAdapter, MemoryQuery, MemoryResult},
    secrets::SecretsAdapter,
};
use langchart_model::{
    id::{
        ArtifactId, ArtifactVersion, EventId, IdempotencyKey, InvocationId, ProposalId, RunId,
        ServerId, StateId, ToolName,
    },
    policy::{CapabilityPolicy, OperationClass},
};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing::{instrument, warn};
use ulid::Ulid;

const ARTIFACT_COMPLETION_EVENT_TIMEOUT: Duration = Duration::from_secs(1);

// ── Capability envelope ───────────────────────────────────────────────────────

/// The resolved, effective capability set for a single agent invocation.
/// Computed once at state entry from all layered policies.
///
/// Envelopes created through [`CapabilityEnvelope::new`] are inert. The
/// runtime binds them to the invocation's broker before handing them to an
/// actor; broker calls reject any envelope that was constructed by a caller.
#[derive(Debug)]
pub struct CapabilityEnvelope {
    /// The policy this envelope was computed from.
    policy: CapabilityPolicy,
    /// The run this invocation belongs to. Broker events derive attribution
    /// from this sealed value rather than actor-controlled input.
    run_id: RunId,
    /// The invocation this envelope is bound to.
    invocation_id: InvocationId,
    /// The state this invocation is executing within.
    state_id: StateId,
    /// Remaining LLM turns allowed.
    turns_remaining: u32,
    /// Remaining tool calls allowed.
    tool_calls_remaining: u32,
    /// Remaining calls for servers with a policy-level call budget.
    server_tool_calls_remaining: HashMap<ServerId, u32>,
    /// Cumulative token count across all LLM calls in this invocation.
    tokens_used: u32,
    /// Maximum total tokens (input + output) for this invocation, if set.
    max_tokens_total: Option<u32>,
    /// Broker-specific authority. Publicly constructed envelopes are inert
    /// until the runtime binds them to the broker for their invocation.
    authority: Option<EnvelopeAuthority>,
}

#[derive(Debug)]
struct EnvelopeAuthority {
    broker: Arc<()>,
    lease: Arc<LeaseState>,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct InvocationLease {
    state: Arc<LeaseState>,
}

/// Opaque authority held by the runtime to bind capability envelopes.
///
/// Agent actors receive the broker but never this token, so they cannot turn a
/// publicly constructed envelope into an authorized one.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct BrokerRuntimeAuthority {
    broker: Arc<()>,
}

#[derive(Debug)]
struct LeaseState {
    inner: StdMutex<LeaseStateInner>,
    changed: Notify,
}

#[derive(Debug)]
struct LeaseStateInner {
    active: bool,
    in_flight: usize,
}

impl LeaseState {
    async fn cancelled(&self) {
        loop {
            let notified = self.changed.notified();
            if !self.inner.lock().unwrap_or_else(|e| e.into_inner()).active {
                return;
            }
            notified.await;
        }
    }
}

impl InvocationLease {
    #[doc(hidden)]
    pub fn revoke(&self) {
        let mut inner = self.state.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.active = false;
        drop(inner);
        self.state.changed.notify_waiters();
    }

    #[doc(hidden)]
    pub fn guard(&self) -> InvocationLeaseGuard {
        InvocationLeaseGuard(self.clone())
    }

    async fn wait_for_idle(&self) {
        loop {
            let notified = self.state.changed.notified();
            if self
                .state
                .inner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .in_flight
                == 0
            {
                return;
            }
            notified.await;
        }
    }

    #[doc(hidden)]
    pub async fn revoke_and_wait(&self) {
        self.revoke();
        self.wait_for_idle().await;
    }
}

#[doc(hidden)]
pub struct InvocationLeaseGuard(InvocationLease);

impl InvocationLeaseGuard {
    #[doc(hidden)]
    pub async fn revoke_and_wait(&self) {
        self.0.revoke_and_wait().await;
    }
}

impl Drop for InvocationLeaseGuard {
    fn drop(&mut self) {
        self.0.revoke();
    }
}

struct OperationPermit {
    state: Arc<LeaseState>,
}

impl OperationPermit {
    async fn cancelled(&self) {
        self.state.cancelled().await;
    }
}

impl Drop for OperationPermit {
    fn drop(&mut self) {
        let mut inner = self.state.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.in_flight = inner.in_flight.saturating_sub(1);
        let idle = inner.in_flight == 0;
        drop(inner);
        if idle {
            self.state.changed.notify_waiters();
        }
    }
}

impl CapabilityEnvelope {
    pub fn new(
        policy: CapabilityPolicy,
        run_id: RunId,
        invocation_id: InvocationId,
        state_id: StateId,
        max_turns: u32,
        max_tool_calls: u32,
    ) -> Self {
        let server_tool_calls_remaining = policy
            .mcp
            .iter()
            .filter_map(|(server, policy)| {
                policy.call_budget.map(|budget| (server.clone(), budget))
            })
            .collect();
        Self {
            policy,
            run_id,
            invocation_id,
            state_id,
            turns_remaining: max_turns,
            tool_calls_remaining: max_tool_calls,
            server_tool_calls_remaining,
            tokens_used: 0,
            max_tokens_total: None,
            authority: None,
        }
    }

    /// Builder: set a token budget (from `ExecutionLimits.max_tokens_total`).
    /// Calling this on an issued envelope invalidates its broker authority.
    pub fn with_token_budget(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens_total = max_tokens;
        // Any public transformation invalidates a previously issued envelope.
        self.authority = None;
        self
    }

    pub fn policy(&self) -> &CapabilityPolicy {
        &self.policy
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn invocation_id(&self) -> &InvocationId {
        &self.invocation_id
    }

    pub fn state_id(&self) -> &StateId {
        &self.state_id
    }

    pub fn turns_remaining(&self) -> u32 {
        self.turns_remaining
    }

    pub fn tool_calls_remaining(&self) -> u32 {
        self.tool_calls_remaining
    }

    pub fn server_tool_calls_remaining(&self) -> &HashMap<ServerId, u32> {
        &self.server_tool_calls_remaining
    }

    pub fn tokens_used(&self) -> u32 {
        self.tokens_used
    }

    pub fn max_tokens_total(&self) -> Option<u32> {
        self.max_tokens_total
    }

    /// Returns true if the named tool on the given server is permitted.
    pub fn allows_tool(&self, server_id: &ServerId, tool_name: &ToolName) -> bool {
        self.policy
            .mcp
            .get(server_id)
            .map(|p| p.allow.iter().any(|t| t == tool_name))
            .unwrap_or(false)
    }

    /// Returns true if `operation` is allowed for `uri` on `server_id`.
    pub fn allows_resource(
        &self,
        server_id: &ServerId,
        operation: &OperationClass,
        uri: &str,
    ) -> bool {
        self.policy.mcp.get(server_id).is_some_and(|policy| {
            policy.operations.contains(operation)
                && policy
                    .resource_patterns
                    .iter()
                    .any(|pattern| resource_pattern_matches(pattern, uri))
        })
    }

    /// Returns true if the agent may write to memory.
    pub fn allows_memory_write(&self) -> bool {
        self.policy.memory_write
    }

    /// Returns true if the named artifact operation is permitted.
    pub fn allows_artifact(&self, operation: &OperationClass) -> bool {
        self.policy.artifact_operations.contains(operation)
    }

    /// Return how many tokens remain in the budget (`None` = unlimited).
    pub fn tokens_remaining(&self) -> Option<u32> {
        self.max_tokens_total
            .map(|max| max.saturating_sub(self.tokens_used))
    }

    /// Returns true if the token budget is exhausted.
    pub fn token_budget_exhausted(&self) -> bool {
        self.max_tokens_total
            .map(|max| self.tokens_used >= max)
            .unwrap_or(false)
    }
}

// ── Broker errors ─────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum BrokerError {
    #[error("tool `{tool}` on server `{server}` is not permitted by the capability envelope")]
    ToolNotPermitted { server: ServerId, tool: ToolName },

    #[error("memory write is not permitted by the capability envelope")]
    MemoryWriteNotPermitted,

    #[error("resource `{uri}` on server `{server}` is not permitted by the capability envelope")]
    ResourceNotPermitted { server: ServerId, uri: String },

    #[error("artifact store not configured")]
    ArtifactStoreNotConfigured,

    #[error("artifact operation `{operation:?}` is not permitted by the capability envelope")]
    ArtifactOperationNotPermitted { operation: OperationClass },

    #[error("turn limit exhausted (max_turns reached)")]
    TurnLimitExhausted,

    #[error("tool call limit exhausted (max_tool_calls reached)")]
    ToolCallLimitExhausted,

    #[error("tool call budget exhausted for MCP server `{server}`")]
    ServerToolCallLimitExhausted { server: ServerId },

    #[error("tool `{tool}` on server `{server}` requires human confirmation")]
    HumanConfirmationRequired { server: ServerId, tool: ToolName },

    #[error("token budget exhausted ({used} / {limit} tokens used)")]
    TokenBudgetExhausted { used: u32, limit: u32 },

    #[error("LLM error: {0}")]
    Llm(#[from] crate::llm::LlmError),

    #[error("MCP error: {0}")]
    Mcp(#[from] crate::mcp::McpError),

    #[error("memory error: {0}")]
    Memory(#[from] crate::memory::MemoryError),

    #[error("artifact error: {0}")]
    Artifact(#[from] crate::artifact::ArtifactError),

    #[error("secrets error: {0}")]
    Secrets(#[from] crate::secrets::SecretsError),

    #[error("event sink error: {0}")]
    EventSink(#[from] crate::event::EventSinkError),

    #[error("capability envelope was not issued by this broker")]
    InvalidCapabilityEnvelope,

    #[error("capability envelope is no longer active")]
    ExpiredCapabilityEnvelope,
}

// ── Broker ────────────────────────────────────────────────────────────────────

/// The central policy-enforcement and observability gateway.
pub struct CapabilityBroker {
    llm: Arc<dyn LlmAdapter>,
    mcp: Arc<dyn McpAdapter>,
    memory: Arc<dyn MemoryAdapter>,
    secrets: Arc<dyn SecretsAdapter>,
    event_sink: Arc<dyn EventSink>,
    /// Optional artifact store. When `None`, artifact broker methods return
    /// `BrokerError::ArtifactStoreNotConfigured`.
    artifact_store: Option<Arc<dyn ArtifactStore>>,
    envelope_authority: Arc<()>,
    claimable_runtime_authority: StdMutex<Option<BrokerRuntimeAuthority>>,
}

impl CapabilityBroker {
    pub fn new(
        llm: Arc<dyn LlmAdapter>,
        mcp: Arc<dyn McpAdapter>,
        memory: Arc<dyn MemoryAdapter>,
        secrets: Arc<dyn SecretsAdapter>,
        event_sink: Arc<dyn EventSink>,
    ) -> Self {
        let envelope_authority = Arc::new(());
        Self {
            llm,
            mcp,
            memory,
            secrets,
            event_sink,
            artifact_store: None,
            claimable_runtime_authority: StdMutex::new(Some(BrokerRuntimeAuthority {
                broker: envelope_authority.clone(),
            })),
            envelope_authority,
        }
    }

    /// Claim the runtime authority for this broker. This succeeds once; the
    /// runtime must retain and clone the returned opaque token for its runs.
    #[doc(hidden)]
    pub fn claim_runtime_authority(&self) -> Option<BrokerRuntimeAuthority> {
        self.claimable_runtime_authority
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
    }

    /// Bind an envelope using authority claimed when the runtime was created.
    #[doc(hidden)]
    pub fn authorize_envelope(
        &self,
        authority: &BrokerRuntimeAuthority,
        envelope: &mut CapabilityEnvelope,
    ) -> Result<InvocationLease, BrokerError> {
        if !Arc::ptr_eq(&authority.broker, &self.envelope_authority) {
            return Err(BrokerError::InvalidCapabilityEnvelope);
        }
        Ok(self.bind_envelope(envelope))
    }

    fn bind_envelope(&self, envelope: &mut CapabilityEnvelope) -> InvocationLease {
        let lease = InvocationLease {
            state: Arc::new(LeaseState {
                inner: StdMutex::new(LeaseStateInner {
                    active: true,
                    in_flight: 0,
                }),
                changed: Notify::new(),
            }),
        };
        envelope.authority = Some(EnvelopeAuthority {
            broker: self.envelope_authority.clone(),
            lease: lease.state.clone(),
        });
        lease
    }

    fn begin_operation(
        &self,
        envelope: &CapabilityEnvelope,
    ) -> Result<OperationPermit, BrokerError> {
        let authority = envelope
            .authority
            .as_ref()
            .filter(|authority| Arc::ptr_eq(&authority.broker, &self.envelope_authority))
            .ok_or(BrokerError::InvalidCapabilityEnvelope)?;
        let mut inner = authority
            .lease
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !inner.active {
            return Err(BrokerError::ExpiredCapabilityEnvelope);
        }
        inner.in_flight += 1;
        drop(inner);
        Ok(OperationPermit {
            state: authority.lease.clone(),
        })
    }

    async fn await_active<T, E>(
        &self,
        permit: &OperationPermit,
        future: impl Future<Output = Result<T, E>>,
    ) -> Result<T, BrokerError>
    where
        BrokerError: From<E>,
    {
        tokio::select! {
            biased;
            _ = permit.cancelled() => Err(BrokerError::ExpiredCapabilityEnvelope),
            result = future => result.map_err(BrokerError::from),
        }
    }

    async fn emit_best_effort(
        &self,
        permit: &OperationPermit,
        run_id: &RunId,
        payload: RuntimeEventPayload,
    ) {
        if let Err(error) = self.await_active(permit, self.emit(run_id, payload)).await {
            warn!(%error, "external operation succeeded but its completion event was not recorded");
        }
    }

    async fn await_tracked_artifact<T>(
        lease: Arc<LeaseState>,
        mut task: JoinHandle<Result<T, ArtifactError>>,
    ) -> Result<T, BrokerError>
    where
        T: Send + 'static,
    {
        tokio::select! {
            biased;
            _ = lease.cancelled() => Err(BrokerError::ExpiredCapabilityEnvelope),
            result = &mut task => match result {
                Ok(result) => result.map_err(BrokerError::Artifact),
                Err(error) => Err(BrokerError::Artifact(ArtifactError::Store(
                    format!("tracked artifact operation failed: {error}"),
                ))),
            },
        }
    }

    async fn emit_to_sink_best_effort(
        event_sink: Arc<dyn EventSink>,
        run_id: RunId,
        payload: RuntimeEventPayload,
    ) {
        let event = RuntimeEvent {
            event_id: EventId::new(Ulid::generate().to_string()),
            run_id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            payload,
        };
        match timeout(ARTIFACT_COMPLETION_EVENT_TIMEOUT, event_sink.append(event)).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                warn!(%error, "external operation succeeded but its completion event was not recorded");
            }
            Err(_) => {
                warn!("external operation succeeded but recording its completion event timed out");
            }
        }
    }

    /// Attach an artifact store. Returns `self` for builder-style chaining.
    pub fn with_artifact_store(mut self, store: Arc<dyn ArtifactStore>) -> Self {
        self.artifact_store = Some(store);
        self
    }

    /// Return a clone of the event sink for use by `WorkflowInstance`.
    pub fn event_sink_ref(&self) -> Arc<dyn EventSink> {
        self.event_sink.clone()
    }

    // ── LLM ──────────────────────────────────────────────────────────────────

    /// Call the LLM.
    ///
    /// Enforcements (in order):
    /// 1. Token budget: if `max_tokens_total` is set and already exhausted, reject.
    /// 2. Turn budget: if `turns_remaining == 0`, reject.
    /// 3. Forward to adapter, accumulate token usage, emit budget events.
    #[instrument(skip(self, envelope, request), fields(state = %envelope.state_id))]
    pub async fn call_llm(
        &self,
        envelope: &mut CapabilityEnvelope,
        request: LlmRequest,
    ) -> Result<LlmResponse, BrokerError> {
        let permit = self.begin_operation(envelope)?;
        let run_id = envelope.run_id.clone();
        // Pre-call: token budget check.
        if let Some(limit) = envelope.max_tokens_total
            && envelope.tokens_used >= limit
        {
            self.await_active(
                &permit,
                self.emit(
                    &run_id,
                    RuntimeEventPayload::BudgetExhausted {
                        budget_type: "tokens".into(),
                    },
                ),
            )
            .await?;
            return Err(BrokerError::TokenBudgetExhausted {
                used: envelope.tokens_used,
                limit,
            });
        }

        // Pre-call: turn budget check.
        if envelope.turns_remaining == 0 {
            self.await_active(
                &permit,
                self.emit(
                    &run_id,
                    RuntimeEventPayload::BudgetExhausted {
                        budget_type: "turns".into(),
                    },
                ),
            )
            .await?;
            return Err(BrokerError::TurnLimitExhausted);
        }
        envelope.turns_remaining -= 1;

        let model = request.model_policy.model.clone().unwrap_or_default();
        let response_format = request.response_format.kind();
        self.await_active(
            &permit,
            self.emit(
                &run_id,
                RuntimeEventPayload::LlmRequest {
                    state_id: envelope.state_id.clone(),
                    model: model.clone(),
                    prompt_tokens: 0, // real value arrives in the response
                    response_format,
                },
            ),
        )
        .await?;

        let start = std::time::Instant::now();
        let response = self
            .await_active(&permit, self.llm.complete(request))
            .await?;
        let latency_ms = start.elapsed().as_millis() as u64;

        // Accumulate token usage.
        let call_tokens = response.usage.prompt_tokens + response.usage.completion_tokens;
        envelope.tokens_used = envelope.tokens_used.saturating_add(call_tokens);

        self.emit_best_effort(
            &permit,
            &run_id,
            RuntimeEventPayload::LlmResponse {
                state_id: envelope.state_id.clone(),
                model: response.model.clone(),
                output_tokens: response.usage.completion_tokens,
                latency_ms,
            },
        )
        .await;

        // Post-call: emit budget warning/exhausted if threshold crossed.
        if let Some(limit) = envelope.max_tokens_total {
            let used = envelope.tokens_used;
            if used >= limit {
                self.emit_best_effort(
                    &permit,
                    &run_id,
                    RuntimeEventPayload::BudgetExhausted {
                        budget_type: "tokens".into(),
                    },
                )
                .await;
            } else if used * 10 / limit >= 8 {
                // Warn at 80 % utilisation.
                self.emit_best_effort(
                    &permit,
                    &run_id,
                    RuntimeEventPayload::BudgetWarning {
                        budget_type: "tokens".into(),
                        used,
                        limit,
                    },
                )
                .await;
            }
        }

        Ok(response)
    }

    // ── MCP ───────────────────────────────────────────────────────────────────

    /// Call an MCP tool. Checks tool allowlist, decrements the tool-call
    /// budget, logs the call, and forwards to the adapter.
    #[instrument(skip(self, envelope, arguments), fields(state = %envelope.state_id))]
    pub async fn call_tool(
        &self,
        envelope: &mut CapabilityEnvelope,
        server_id: &ServerId,
        tool_name: &ToolName,
        arguments: serde_json::Value,
        idempotency_key: Option<&IdempotencyKey>,
    ) -> Result<serde_json::Value, BrokerError> {
        let permit = self.begin_operation(envelope)?;
        let run_id = envelope.run_id.clone();
        if !envelope.allows_tool(server_id, tool_name) {
            warn!(
                server = %server_id,
                tool = %tool_name,
                "tool call rejected by capability envelope"
            );
            self.await_active(
                &permit,
                self.emit(
                    &run_id,
                    RuntimeEventPayload::ToolRejected {
                        state_id: envelope.state_id.clone(),
                        server_id: server_id.0.clone(),
                        tool_name: tool_name.0.clone(),
                        reason: "not in capability allowlist".into(),
                    },
                ),
            )
            .await?;
            return Err(BrokerError::ToolNotPermitted {
                server: server_id.clone(),
                tool: tool_name.clone(),
            });
        }

        let server_policy = envelope
            .policy
            .mcp
            .get(server_id)
            .expect("allowed tool must have a server policy");

        if server_policy.require_human_confirmation {
            self.await_active(
                &permit,
                self.emit(
                    &run_id,
                    RuntimeEventPayload::HumanConfirmationRequired {
                        state_id: envelope.state_id.clone(),
                        server_id: server_id.0.clone(),
                        tool_name: tool_name.0.clone(),
                    },
                ),
            )
            .await?;
            return Err(BrokerError::HumanConfirmationRequired {
                server: server_id.clone(),
                tool: tool_name.clone(),
            });
        }

        if envelope.tool_calls_remaining == 0 {
            return Err(BrokerError::ToolCallLimitExhausted);
        }

        if envelope
            .server_tool_calls_remaining
            .get(server_id)
            .is_some_and(|remaining| *remaining == 0)
        {
            self.await_active(
                &permit,
                self.emit(
                    &run_id,
                    RuntimeEventPayload::BudgetExhausted {
                        budget_type: format!("mcp_server:{server_id}"),
                    },
                ),
            )
            .await?;
            return Err(BrokerError::ServerToolCallLimitExhausted {
                server: server_id.clone(),
            });
        }

        let credential_refs = server_policy.credentials.clone();
        let mut credentials = Vec::with_capacity(credential_refs.len());
        for secret_ref in credential_refs {
            credentials.push(McpCredential {
                value: self
                    .await_active(&permit, self.secrets.get(&secret_ref))
                    .await?,
                secret_ref,
            });
        }

        envelope.tool_calls_remaining -= 1;
        if let Some(remaining) = envelope.server_tool_calls_remaining.get_mut(server_id) {
            *remaining -= 1;
        }

        self.await_active(
            &permit,
            self.emit(
                &run_id,
                RuntimeEventPayload::ToolRequest {
                    state_id: envelope.state_id.clone(),
                    server_id: server_id.0.clone(),
                    tool_name: tool_name.0.clone(),
                },
            ),
        )
        .await?;

        let start = std::time::Instant::now();
        let result = self
            .await_active(
                &permit,
                self.mcp.call_tool(
                    server_id,
                    tool_name,
                    arguments,
                    &credentials,
                    idempotency_key,
                ),
            )
            .await?;
        let latency_ms = start.elapsed().as_millis() as u64;

        self.emit_best_effort(
            &permit,
            &run_id,
            RuntimeEventPayload::ToolResponse {
                state_id: envelope.state_id.clone(),
                server_id: server_id.0.clone(),
                tool_name: tool_name.0.clone(),
                latency_ms,
            },
        )
        .await;

        Ok(result)
    }

    /// Read a resource from an MCP server (no budget decrement — reads are free).
    pub async fn read_resource(
        &self,
        envelope: &CapabilityEnvelope,
        server_id: &ServerId,
        uri: &str,
    ) -> Result<ResourceContent, BrokerError> {
        let permit = self.begin_operation(envelope)?;
        if !envelope.allows_resource(server_id, &OperationClass::Read, uri) {
            warn!(server = %server_id, uri, "resource read rejected by capability envelope");
            self.await_active(
                &permit,
                self.emit(
                    &envelope.run_id,
                    RuntimeEventPayload::ToolRejected {
                        state_id: envelope.state_id.clone(),
                        server_id: server_id.0.clone(),
                        tool_name: format!("resource:{uri}"),
                        reason: "resource URI or read operation not permitted".into(),
                    },
                ),
            )
            .await?;
            return Err(BrokerError::ResourceNotPermitted {
                server: server_id.clone(),
                uri: uri.to_owned(),
            });
        }

        self.await_active(
            &permit,
            self.emit(
                &envelope.run_id,
                RuntimeEventPayload::ToolRequest {
                    state_id: envelope.state_id.clone(),
                    server_id: server_id.0.clone(),
                    tool_name: format!("resource:{uri}"),
                },
            ),
        )
        .await?;

        self.await_active(&permit, self.mcp.read_resource(server_id, uri))
            .await
    }

    // ── Memory ────────────────────────────────────────────────────────────────

    /// Search memory. Always permitted (controlled by context policy, not capability).
    pub async fn memory_search(
        &self,
        envelope: &CapabilityEnvelope,
        query: MemoryQuery,
    ) -> Result<Vec<MemoryResult>, BrokerError> {
        let permit = self.begin_operation(envelope)?;
        let preview = format!("{:?}", query.mode)
            .chars()
            .take(60)
            .collect::<String>();
        self.await_active(
            &permit,
            self.emit(
                &envelope.run_id,
                RuntimeEventPayload::MemorySearched {
                    query_preview: preview,
                },
            ),
        )
        .await?;
        let _ = envelope; // read is always permitted via context policy
        self.await_active(&permit, self.memory.search(query)).await
    }

    /// Store a memory record. Requires `memory_write` capability.
    pub async fn memory_store(
        &self,
        envelope: &CapabilityEnvelope,
        record: crate::memory::MemoryRecord,
    ) -> Result<crate::memory::MemoryId, BrokerError> {
        let permit = self.begin_operation(envelope)?;
        if !envelope.allows_memory_write() {
            return Err(BrokerError::MemoryWriteNotPermitted);
        }
        let scope = format!("{:?}", record.scope);
        let id = self
            .await_active(&permit, self.memory.store(record))
            .await?;
        self.emit_best_effort(
            &permit,
            &envelope.run_id,
            RuntimeEventPayload::MemoryStored { scope },
        )
        .await;
        Ok(id)
    }

    // ── Artifacts ─────────────────────────────────────────────────────────────

    /// Read an artifact from the store.
    ///
    /// Emits no observable event (reads are non-mutating and potentially high
    /// frequency). Returns `Err(ArtifactStoreNotConfigured)` if no store is set.
    pub async fn read_artifact(
        &self,
        envelope: &CapabilityEnvelope,
        id: &ArtifactId,
        version: Option<&ArtifactVersion>,
    ) -> Result<ArtifactContent, BrokerError> {
        let permit = self.begin_operation(envelope)?;
        self.ensure_artifact_allowed(envelope, OperationClass::Read)?;
        let store = self
            .artifact_store
            .as_ref()
            .ok_or(BrokerError::ArtifactStoreNotConfigured)?;
        self.await_active(&permit, store.read(id, version)).await
    }

    /// Submit a proposal to modify an artifact.
    ///
    /// Emits `ProposalCreated` on success.
    pub async fn propose_artifact(
        &self,
        envelope: &CapabilityEnvelope,
        proposal: ArtifactProposal,
    ) -> Result<ProposalId, BrokerError> {
        let permit = self.begin_operation(envelope)?;
        self.ensure_artifact_allowed(envelope, OperationClass::Propose)?;
        let store = self
            .artifact_store
            .as_ref()
            .ok_or(BrokerError::ArtifactStoreNotConfigured)?
            .clone();
        let artifact_id = proposal.id.0.clone();
        let lease = permit.state.clone();
        let event_sink = self.event_sink.clone();
        let run_id = envelope.run_id.clone();
        let task = tokio::spawn(async move {
            let result = store.propose(proposal).await;
            if let Ok(proposal_id) = &result {
                Self::emit_to_sink_best_effort(
                    event_sink,
                    run_id,
                    RuntimeEventPayload::ProposalCreated {
                        artifact_id,
                        proposal_id: proposal_id.0.clone(),
                    },
                )
                .await;
            }
            drop(permit);
            result
        });
        Self::await_tracked_artifact(lease, task).await
    }

    /// Commit a pending proposal.
    ///
    /// Emits `ProposalCommitted` on success or `ProposalConflicted` on a
    /// version conflict (which is then returned as an error to the caller).
    pub async fn commit_artifact(
        &self,
        envelope: &CapabilityEnvelope,
        artifact_id: &ArtifactId,
        proposal_id: &ProposalId,
        expected_base: &ArtifactVersion,
    ) -> Result<ArtifactVersion, BrokerError> {
        let permit = self.begin_operation(envelope)?;
        self.ensure_artifact_allowed(envelope, OperationClass::Commit)?;
        let store = self
            .artifact_store
            .as_ref()
            .ok_or(BrokerError::ArtifactStoreNotConfigured)?
            .clone();
        let lease = permit.state.clone();
        let event_sink = self.event_sink.clone();
        let run_id = envelope.run_id.clone();
        let artifact_id = artifact_id.clone();
        let proposal_id = proposal_id.clone();
        let expected_base = expected_base.clone();
        let task = tokio::spawn(async move {
            let result = store
                .commit(&artifact_id, &proposal_id, &expected_base)
                .await;
            match &result {
                Ok(new_version) => {
                    Self::emit_to_sink_best_effort(
                        event_sink,
                        run_id,
                        RuntimeEventPayload::ProposalCommitted {
                            artifact_id: artifact_id.0.clone(),
                            proposal_id: proposal_id.0.clone(),
                            new_version: new_version.0.clone(),
                        },
                    )
                    .await;
                }
                Err(ArtifactError::VersionConflict { .. }) => {
                    Self::emit_to_sink_best_effort(
                        event_sink,
                        run_id,
                        RuntimeEventPayload::ProposalConflicted {
                            artifact_id: artifact_id.0.clone(),
                            proposal_id: proposal_id.0.clone(),
                        },
                    )
                    .await;
                }
                Err(_) => {}
            }
            drop(permit);
            result
        });
        Self::await_tracked_artifact(lease, task).await
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn ensure_artifact_allowed(
        &self,
        envelope: &CapabilityEnvelope,
        operation: OperationClass,
    ) -> Result<(), BrokerError> {
        if envelope.allows_artifact(&operation) {
            Ok(())
        } else {
            Err(BrokerError::ArtifactOperationNotPermitted { operation })
        }
    }

    async fn emit(&self, run_id: &RunId, payload: RuntimeEventPayload) -> Result<(), BrokerError> {
        let event = RuntimeEvent {
            event_id: EventId::new(Ulid::generate().to_string()),
            run_id: run_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            payload,
        };
        self.event_sink.append(event).await?;
        Ok(())
    }
}

fn resource_pattern_matches(pattern: &str, value: &str) -> bool {
    if let Some(expression) = pattern.strip_prefix("regex:") {
        return regex::Regex::new(expression)
            .map(|regex| regex.is_match(value))
            .unwrap_or(false);
    }
    glob_matches(pattern, value)
}

fn glob_matches(pattern: &str, value: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let value: Vec<char> = value.chars().collect();
    let mut previous = vec![false; value.len() + 1];
    previous[0] = true;

    for token in pattern {
        let mut current = vec![false; value.len() + 1];
        if token == '*' {
            current[0] = previous[0];
        }
        for (index, character) in value.iter().enumerate() {
            current[index + 1] = match token {
                '*' => previous[index + 1] || current[index],
                '?' => previous[index],
                literal => previous[index] && literal == *character,
            };
        }
        previous = current;
    }
    previous[value.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{ArtifactError, ProposalSummary},
        event::EventSinkError,
        llm::LlmError,
        mcp::{McpError, ToolDefinition},
        memory::{MemoryError, MemoryId, MemoryRecord, MemoryScope},
        secrets::{HostMapSecretsAdapter, SecretValue},
    };
    use async_trait::async_trait;
    use langchart_model::{
        id::SecretRef,
        policy::{McpServerPolicy, OperationClass},
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    struct NoopLlm;

    #[async_trait]
    impl LlmAdapter for NoopLlm {
        async fn complete(&self, _request: LlmRequest) -> Result<LlmResponse, LlmError> {
            Err(LlmError::Provider("unused".into()))
        }
    }

    struct NoopMemory;

    #[async_trait]
    impl MemoryAdapter for NoopMemory {
        async fn store(&self, _record: MemoryRecord) -> Result<MemoryId, MemoryError> {
            Ok(MemoryId("unused".into()))
        }

        async fn search(&self, _query: MemoryQuery) -> Result<Vec<MemoryResult>, MemoryError> {
            Ok(Vec::new())
        }

        async fn get(&self, _id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
            Ok(None)
        }

        async fn delete(&self, _id: &MemoryId) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<RuntimeEvent>>);

    #[async_trait]
    impl EventSink for RecordingSink {
        async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
            self.0.lock().await.push(event);
            Ok(())
        }
    }

    #[derive(Default)]
    struct BlockingCompletionSink {
        completion_started: AtomicBool,
        released: AtomicBool,
        changed: Notify,
    }

    impl BlockingCompletionSink {
        fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.changed.notify_waiters();
        }
    }

    #[async_trait]
    impl EventSink for BlockingCompletionSink {
        async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
            if matches!(event.payload, RuntimeEventPayload::ProposalCreated { .. }) {
                self.completion_started.store(true, Ordering::SeqCst);
                self.changed.notify_waiters();
                while !self.released.load(Ordering::SeqCst) {
                    self.changed.notified().await;
                }
            }
            Ok(())
        }
    }

    struct CompletionFailingSink;

    #[async_trait]
    impl EventSink for CompletionFailingSink {
        async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
            if matches!(
                event.payload,
                RuntimeEventPayload::LlmResponse { .. }
                    | RuntimeEventPayload::ToolResponse { .. }
                    | RuntimeEventPayload::MemoryStored { .. }
                    | RuntimeEventPayload::ProposalCreated { .. }
                    | RuntimeEventPayload::ProposalCommitted { .. }
                    | RuntimeEventPayload::ProposalConflicted { .. }
            ) {
                Err(EventSinkError::Sink("completion sink failure".into()))
            } else {
                Ok(())
            }
        }
    }

    #[derive(Default)]
    struct RecordingMcp {
        calls: AtomicUsize,
        credentials: Mutex<Vec<(String, String)>>,
    }

    #[derive(Default)]
    struct RecordingArtifactStore {
        reads: AtomicUsize,
        proposals: AtomicUsize,
        commits: AtomicUsize,
        proposal_artifact: Mutex<Option<ArtifactId>>,
    }

    #[async_trait]
    impl ArtifactStore for RecordingArtifactStore {
        async fn read(
            &self,
            id: &ArtifactId,
            version: Option<&ArtifactVersion>,
        ) -> Result<ArtifactContent, ArtifactError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(ArtifactContent {
                id: id.clone(),
                version: version
                    .cloned()
                    .unwrap_or_else(|| ArtifactVersion::new("v1")),
                bytes: b"content".to_vec(),
                content_type: "text/plain".into(),
            })
        }

        async fn propose(&self, proposal: ArtifactProposal) -> Result<ProposalId, ArtifactError> {
            self.proposals.fetch_add(1, Ordering::SeqCst);
            *self.proposal_artifact.lock().await = Some(proposal.id);
            Ok(ProposalId::new("proposal-1"))
        }

        async fn commit(
            &self,
            artifact_id: &ArtifactId,
            proposal_id: &ProposalId,
            _expected_base: &ArtifactVersion,
        ) -> Result<ArtifactVersion, ArtifactError> {
            if self.proposal_artifact.lock().await.as_ref() != Some(artifact_id) {
                return Err(ArtifactError::ProposalArtifactMismatch {
                    proposal_id: proposal_id.clone(),
                    artifact_id: artifact_id.clone(),
                });
            }
            self.commits.fetch_add(1, Ordering::SeqCst);
            Ok(ArtifactVersion::new("v2"))
        }

        async fn list_proposals(
            &self,
            _artifact_id: &ArtifactId,
        ) -> Result<Vec<ProposalSummary>, ArtifactError> {
            Ok(Vec::new())
        }
    }

    #[async_trait]
    impl McpAdapter for RecordingMcp {
        async fn call_tool(
            &self,
            _server_id: &ServerId,
            _tool_name: &ToolName,
            _arguments: serde_json::Value,
            credentials: &[McpCredential],
            _idempotency_key: Option<&IdempotencyKey>,
        ) -> Result<serde_json::Value, McpError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.credentials.lock().await.extend(
                credentials.iter().map(|credential| {
                    (credential.secret_ref.0.clone(), credential.value.0.clone())
                }),
            );
            Ok(serde_json::json!({ "ok": true }))
        }

        async fn list_tools(&self, _server_id: &ServerId) -> Result<Vec<ToolDefinition>, McpError> {
            Ok(Vec::new())
        }

        async fn read_resource(
            &self,
            _server_id: &ServerId,
            uri: &str,
        ) -> Result<ResourceContent, McpError> {
            Ok(ResourceContent {
                uri: uri.into(),
                content_type: "text/plain".into(),
                bytes: b"content".to_vec(),
            })
        }
    }

    #[derive(Default)]
    struct BlockingMcp {
        started: Notify,
        release: Notify,
        cancelled: AtomicBool,
    }

    struct BlockingArtifactStore {
        started: Arc<AtomicBool>,
        finished: Arc<AtomicBool>,
        release: Arc<(StdMutex<bool>, std::sync::Condvar)>,
    }

    impl Default for BlockingArtifactStore {
        fn default() -> Self {
            Self {
                started: Arc::new(AtomicBool::new(false)),
                finished: Arc::new(AtomicBool::new(false)),
                release: Arc::new((StdMutex::new(false), std::sync::Condvar::new())),
            }
        }
    }

    impl BlockingArtifactStore {
        fn release(&self) {
            let (lock, changed) = &*self.release;
            *lock.lock().unwrap_or_else(|error| error.into_inner()) = true;
            changed.notify_all();
        }
    }

    #[async_trait]
    impl ArtifactStore for BlockingArtifactStore {
        async fn read(
            &self,
            _id: &ArtifactId,
            _version: Option<&ArtifactVersion>,
        ) -> Result<ArtifactContent, ArtifactError> {
            Err(ArtifactError::Store("unused".into()))
        }

        async fn propose(&self, _proposal: ArtifactProposal) -> Result<ProposalId, ArtifactError> {
            let started = self.started.clone();
            let finished = self.finished.clone();
            let release = self.release.clone();
            tokio::task::spawn_blocking(move || {
                started.store(true, Ordering::SeqCst);
                let (lock, changed) = &*release;
                let mut released = lock.lock().unwrap_or_else(|error| error.into_inner());
                while !*released {
                    released = changed
                        .wait(released)
                        .unwrap_or_else(|error| error.into_inner());
                }
                finished.store(true, Ordering::SeqCst);
                ProposalId::new("blocking-proposal")
            })
            .await
            .map_err(|error| ArtifactError::Store(error.to_string()))
        }

        async fn commit(
            &self,
            _artifact_id: &ArtifactId,
            _proposal_id: &ProposalId,
            _expected_base: &ArtifactVersion,
        ) -> Result<ArtifactVersion, ArtifactError> {
            Err(ArtifactError::Store("unused".into()))
        }

        async fn list_proposals(
            &self,
            _artifact_id: &ArtifactId,
        ) -> Result<Vec<ProposalSummary>, ArtifactError> {
            Ok(Vec::new())
        }
    }

    struct CancellationMarker<'a>(&'a AtomicBool);

    impl Drop for CancellationMarker<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl McpAdapter for BlockingMcp {
        async fn call_tool(
            &self,
            _server_id: &ServerId,
            _tool_name: &ToolName,
            _arguments: serde_json::Value,
            _credentials: &[McpCredential],
            _idempotency_key: Option<&IdempotencyKey>,
        ) -> Result<serde_json::Value, McpError> {
            let _marker = CancellationMarker(&self.cancelled);
            self.started.notify_one();
            self.release.notified().await;
            Ok(serde_json::json!({ "ok": true }))
        }

        async fn list_tools(&self, _server_id: &ServerId) -> Result<Vec<ToolDefinition>, McpError> {
            Ok(Vec::new())
        }

        async fn read_resource(
            &self,
            _server_id: &ServerId,
            _uri: &str,
        ) -> Result<ResourceContent, McpError> {
            unreachable!()
        }
    }

    fn broker(
        secrets: HashMap<String, SecretValue>,
    ) -> (CapabilityBroker, Arc<RecordingMcp>, Arc<RecordingSink>) {
        let mcp = Arc::new(RecordingMcp::default());
        let sink = Arc::new(RecordingSink::default());
        let broker = CapabilityBroker::new(
            Arc::new(NoopLlm),
            mcp.clone(),
            Arc::new(NoopMemory),
            Arc::new(HostMapSecretsAdapter::new(secrets)),
            sink.clone(),
        );
        (broker, mcp, sink)
    }

    fn envelope(policy: CapabilityPolicy) -> CapabilityEnvelope {
        CapabilityEnvelope::new(
            policy,
            RunId::new("run-test"),
            InvocationId::new("inv-test"),
            StateId::new("work"),
            2,
            3,
        )
    }

    #[test]
    fn runtime_authority_is_single_claim_and_broker_scoped() {
        let (first, _, _) = broker(HashMap::new());
        let (second, _, _) = broker(HashMap::new());
        let authority = first
            .claim_runtime_authority()
            .expect("first claim succeeds");
        assert!(first.claim_runtime_authority().is_none());

        let mut envelope = envelope(CapabilityPolicy::default());
        let error = second
            .authorize_envelope(&authority, &mut envelope)
            .expect_err("authority from another broker must be rejected");
        assert!(matches!(error, BrokerError::InvalidCapabilityEnvelope));
    }

    #[tokio::test]
    async fn server_budget_and_credentials_are_enforced_inside_broker() {
        let server = ServerId::new("files");
        let tool = ToolName::new("write");
        let secret_ref = SecretRef::new("files-token");
        let (broker, mcp, _) = broker(HashMap::from([(
            secret_ref.0.clone(),
            SecretValue("sensitive".into()),
        )]));
        let mut envelope = envelope(CapabilityPolicy {
            mcp: HashMap::from([(
                server.clone(),
                McpServerPolicy {
                    allow: vec![tool.clone()],
                    call_budget: Some(1),
                    credentials: vec![secret_ref.clone()],
                    ..Default::default()
                },
            )]),
            ..Default::default()
        });
        let _lease = broker.bind_envelope(&mut envelope);

        broker
            .call_tool(&mut envelope, &server, &tool, serde_json::Value::Null, None)
            .await
            .expect("first call");
        let error = broker
            .call_tool(&mut envelope, &server, &tool, serde_json::Value::Null, None)
            .await
            .expect_err("server budget must stop the second call");

        assert!(matches!(
            error,
            BrokerError::ServerToolCallLimitExhausted { server: exhausted }
                if exhausted == server
        ));
        assert_eq!(mcp.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            *mcp.credentials.lock().await,
            vec![(secret_ref.0, "sensitive".into())]
        );
        assert_eq!(envelope.tool_calls_remaining(), 2);
        assert_eq!(
            envelope.server_tool_calls_remaining().get(&server),
            Some(&0)
        );
    }

    #[tokio::test]
    async fn human_confirmation_rejection_does_not_consume_budget() {
        let server = ServerId::new("dangerous");
        let tool = ToolName::new("delete");
        let (broker, mcp, sink) = broker(HashMap::new());
        let mut envelope = envelope(CapabilityPolicy {
            mcp: HashMap::from([(
                server.clone(),
                McpServerPolicy {
                    allow: vec![tool.clone()],
                    call_budget: Some(1),
                    require_human_confirmation: true,
                    ..Default::default()
                },
            )]),
            ..Default::default()
        });
        let _lease = broker.bind_envelope(&mut envelope);

        let error = broker
            .call_tool(&mut envelope, &server, &tool, serde_json::Value::Null, None)
            .await
            .expect_err("confirmation must block the call");

        assert!(matches!(
            error,
            BrokerError::HumanConfirmationRequired { .. }
        ));
        assert_eq!(mcp.calls.load(Ordering::SeqCst), 0);
        assert_eq!(envelope.tool_calls_remaining(), 3);
        assert_eq!(
            envelope.server_tool_calls_remaining().get(&server),
            Some(&1)
        );
        assert!(sink.0.lock().await.iter().any(|event| matches!(
            event.payload,
            RuntimeEventPayload::HumanConfirmationRequired { .. }
        )));
    }

    #[tokio::test]
    async fn resource_reads_require_broker_authority_and_matching_policy() {
        let server = ServerId::new("vault");
        let (broker, _, _) = broker(HashMap::new());
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
        let forged = envelope(policy.clone());
        let error = broker
            .read_resource(&forged, &server, "vault://docs/readme.md")
            .await
            .expect_err("an unissued envelope must be inert");
        assert!(matches!(error, BrokerError::InvalidCapabilityEnvelope));

        let mut authorized = envelope(policy);
        let _lease = broker.bind_envelope(&mut authorized);
        broker
            .read_resource(&authorized, &server, "vault://docs/readme.md")
            .await
            .expect("matching resource");
        let error = broker
            .read_resource(&authorized, &server, "vault://private/secret.md")
            .await
            .expect_err("non-matching resource");
        assert!(matches!(error, BrokerError::ResourceNotPermitted { .. }));
    }

    #[tokio::test]
    async fn artifact_operations_require_explicit_permissions_and_use_bound_run_id() {
        let (broker, _, sink) = broker(HashMap::new());
        let store = Arc::new(RecordingArtifactStore::default());
        let broker = broker.with_artifact_store(store.clone());
        let artifact_id = ArtifactId::new("report");
        let base = ArtifactVersion::new("v1");
        let proposal = ArtifactProposal {
            id: artifact_id.clone(),
            base_version: base.clone(),
            content: b"new content".to_vec(),
            content_type: "text/plain".into(),
            rationale: "test".into(),
        };

        let mut denied = envelope(CapabilityPolicy::default());
        let _lease = broker.bind_envelope(&mut denied);
        let error = broker
            .propose_artifact(&denied, proposal.clone())
            .await
            .expect_err("artifact proposal must be denied by default");
        assert!(matches!(
            error,
            BrokerError::ArtifactOperationNotPermitted {
                operation: OperationClass::Propose
            }
        ));
        assert_eq!(store.proposals.load(Ordering::SeqCst), 0);

        let mut allowed = envelope(CapabilityPolicy {
            artifact_operations: vec![
                OperationClass::Read,
                OperationClass::Propose,
                OperationClass::Commit,
            ],
            ..Default::default()
        });
        let _lease = broker.bind_envelope(&mut allowed);
        broker
            .read_artifact(&allowed, &artifact_id, Some(&base))
            .await
            .expect("read");
        let proposal_id = broker
            .propose_artifact(&allowed, proposal)
            .await
            .expect("propose");
        let error = broker
            .commit_artifact(
                &allowed,
                &ArtifactId::new("wrong-artifact"),
                &proposal_id,
                &base,
            )
            .await
            .expect_err("store must validate proposal ownership");
        assert!(matches!(
            error,
            BrokerError::Artifact(ArtifactError::ProposalArtifactMismatch { .. })
        ));
        broker
            .commit_artifact(&allowed, &artifact_id, &proposal_id, &base)
            .await
            .expect("commit");

        assert_eq!(store.reads.load(Ordering::SeqCst), 1);
        assert_eq!(store.proposals.load(Ordering::SeqCst), 1);
        assert_eq!(store.commits.load(Ordering::SeqCst), 1);
        assert!(
            sink.0
                .lock()
                .await
                .iter()
                .all(|event| event.run_id == RunId::new("run-test")),
            "broker events must derive their run ID from the sealed envelope"
        );
    }

    #[tokio::test]
    async fn completion_event_failure_does_not_turn_successful_side_effects_into_errors() {
        let server = ServerId::new("files");
        let tool = ToolName::new("write");
        let mcp = Arc::new(RecordingMcp::default());
        let artifacts = Arc::new(RecordingArtifactStore::default());
        let broker = CapabilityBroker::new(
            Arc::new(NoopLlm),
            mcp.clone(),
            Arc::new(NoopMemory),
            Arc::new(HostMapSecretsAdapter::empty()),
            Arc::new(CompletionFailingSink),
        )
        .with_artifact_store(artifacts.clone());
        let mut envelope = envelope(CapabilityPolicy {
            mcp: HashMap::from([(
                server.clone(),
                McpServerPolicy {
                    allow: vec![tool.clone()],
                    ..Default::default()
                },
            )]),
            artifact_operations: vec![OperationClass::Propose, OperationClass::Commit],
            memory_write: true,
            ..Default::default()
        });
        let _lease = broker.bind_envelope(&mut envelope);

        broker
            .call_tool(&mut envelope, &server, &tool, serde_json::Value::Null, None)
            .await
            .expect("tool result must survive response-event failure");
        broker
            .memory_store(
                &envelope,
                MemoryRecord {
                    scope: MemoryScope::Global,
                    key: Some("key".into()),
                    content: "content".into(),
                    embedding: None,
                    metadata: serde_json::Value::Null,
                },
            )
            .await
            .expect("memory ID must survive completion-event failure");

        let artifact_id = ArtifactId::new("report");
        let base = ArtifactVersion::new("v1");
        let proposal_id = broker
            .propose_artifact(
                &envelope,
                ArtifactProposal {
                    id: artifact_id.clone(),
                    base_version: base.clone(),
                    content: b"new content".to_vec(),
                    content_type: "text/plain".into(),
                    rationale: "test".into(),
                },
            )
            .await
            .expect("proposal ID must survive completion-event failure");
        broker
            .commit_artifact(&envelope, &artifact_id, &proposal_id, &base)
            .await
            .expect("commit version must survive completion-event failure");

        assert_eq!(mcp.calls.load(Ordering::SeqCst), 1);
        assert_eq!(artifacts.proposals.load(Ordering::SeqCst), 1);
        assert_eq!(artifacts.commits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn revoked_invocation_lease_rejects_broker_calls() {
        let (broker, _, _) = broker(HashMap::new());
        let mut envelope = envelope(CapabilityPolicy::default());
        let lease = broker.bind_envelope(&mut envelope);
        lease.revoke();

        let error = broker
            .memory_search(
                &envelope,
                MemoryQuery {
                    scope: crate::memory::MemoryScope::Global,
                    mode: crate::memory::QueryMode::Keyword {
                        text: "test".into(),
                    },
                    limit: 1,
                    min_score: None,
                },
            )
            .await
            .expect_err("revoked authority must be inert");
        assert!(matches!(error, BrokerError::ExpiredCapabilityEnvelope));
    }

    #[tokio::test]
    async fn revocation_waits_for_artifact_mutation_and_completion_event() {
        let store = Arc::new(BlockingArtifactStore::default());
        let sink = Arc::new(BlockingCompletionSink::default());
        let broker = Arc::new(
            CapabilityBroker::new(
                Arc::new(NoopLlm),
                Arc::new(RecordingMcp::default()),
                Arc::new(NoopMemory),
                Arc::new(HostMapSecretsAdapter::empty()),
                sink.clone(),
            )
            .with_artifact_store(store.clone()),
        );
        let mut envelope = envelope(CapabilityPolicy {
            artifact_operations: vec![OperationClass::Propose],
            ..Default::default()
        });
        let lease = broker.bind_envelope(&mut envelope);
        let call_broker = broker.clone();
        let call = tokio::spawn(async move {
            call_broker
                .propose_artifact(
                    &envelope,
                    ArtifactProposal {
                        id: ArtifactId::new("blocking"),
                        base_version: ArtifactVersion::new("v1"),
                        content: Vec::new(),
                        content_type: "application/octet-stream".into(),
                        rationale: "test".into(),
                    },
                )
                .await
        });

        for _ in 0..100 {
            if store.started.load(Ordering::SeqCst) {
                break;
            }
            tokio::task::yield_now().await;
        }
        if !store.started.load(Ordering::SeqCst) {
            store.release();
            panic!("mutation did not start");
        }

        lease.revoke();
        assert!(matches!(
            call.await.expect("broker call task"),
            Err(BrokerError::ExpiredCapabilityEnvelope)
        ));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(25), lease.wait_for_idle())
                .await
                .is_err(),
            "lease drained while spawn_blocking mutation was still active"
        );

        store.release();
        for _ in 0..100 {
            if sink.completion_started.load(Ordering::SeqCst) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(sink.completion_started.load(Ordering::SeqCst));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(25), lease.wait_for_idle())
                .await
                .is_err(),
            "lease drained before the completion event was recorded"
        );
        sink.release();
        tokio::time::timeout(std::time::Duration::from_secs(1), lease.wait_for_idle())
            .await
            .expect("tracked artifact operation did not drain");
        assert!(store.finished.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn revocation_cancels_an_admitted_in_flight_call() {
        let server = ServerId::new("blocking");
        let tool = ToolName::new("mutate");
        let mcp = Arc::new(BlockingMcp::default());
        let sink = Arc::new(RecordingSink::default());
        let broker = Arc::new(CapabilityBroker::new(
            Arc::new(NoopLlm),
            mcp.clone(),
            Arc::new(NoopMemory),
            Arc::new(HostMapSecretsAdapter::empty()),
            sink,
        ));
        let mut envelope = envelope(CapabilityPolicy {
            mcp: HashMap::from([(
                server.clone(),
                McpServerPolicy {
                    allow: vec![tool.clone()],
                    ..Default::default()
                },
            )]),
            ..Default::default()
        });
        let lease = broker.bind_envelope(&mut envelope);
        let call_broker = broker.clone();
        let call = tokio::spawn(async move {
            call_broker
                .call_tool(&mut envelope, &server, &tool, serde_json::Value::Null, None)
                .await
        });

        mcp.started.notified().await;
        lease.revoke();
        tokio::time::timeout(std::time::Duration::from_secs(1), lease.wait_for_idle())
            .await
            .expect("revoked call did not drain");
        let error = call
            .await
            .expect("call task")
            .expect_err("revoked in-flight call must fail");

        assert!(matches!(error, BrokerError::ExpiredCapabilityEnvelope));
        assert!(mcp.cancelled.load(Ordering::SeqCst));
    }
}
