//! `WorkflowInstance` — the live execution state of one workflow run.
//!
//! Owns the active state configuration, event queue, spawned activity tasks,
//! and timer registry for a single run. Executes the run-to-completion (RTC)
//! macro-step on each dequeued event.
//!
//! # RTC macro-step
//! 1. Dequeue one event from the external queue.
//! 2. Find enabled transitions from the active configuration.
//! 3. Select deterministically (priority, then declaration order).
//! 4. Exit states inner-to-outer, execute transition actions.
//! 5. Enter target states outer-to-inner.
//! 6. Start activities for newly entered states.
//! 7. Checkpoint if policy requires.
//! 8. Publish observable events.
//!
//! # Phase 4 additions
//! - **Parallel states:** entering a `Parallel` state enters all region
//!   initials concurrently. Each region tracks its own final state. Completion
//!   mode (`All`, `Any`, `Quorum(n)`, `Guard`, `Manual`) determines when the
//!   parallel state itself completes and synthesises a `parallel.completed`
//!   internal event.
//! - **History:** on exit of a compound or parallel state, the active child
//!   configuration is saved to `history`. On subsequent entry via a history
//!   pseudo-state, the saved configuration is restored.
//! - **Subworkflow:** a `Subworkflow` state spawns a child `WorkflowInstance`
//!   as an async task. When the child completes, a `subworkflow.completed`
//!   event (with the child's final output) is injected into the parent's queue.

use crate::{
    broker::{CapabilityBroker, CapabilityEnvelope, InvocationLease},
    engine::EngineError,
    instance::{
        ActionContext, ActionTrigger, AgentActor, AgentError, AgentOutputEvent,
        ResolvedInstructions, StateAction,
    },
    outbox::Outbox,
    timer::{TimerFired, TimerRegistry},
};
use langchart_adapters::checkpoint::CheckpointStore;
use langchart_adapters::{
    context::{ContextResolver, ContextView},
    event::{EventSink, RuntimeEvent, RuntimeEventPayload},
    workflow_repository::WorkflowRepository,
};
use langchart_model::{
    id::{EventId, InvocationId, RegionId, RunId, StateId},
    policy::{CapabilityPolicy, ContextPolicy, McpServerPolicy},
    state::{ParallelCompletion, StateDefinition, StateType, TransitionKind, TransitionSpec},
    validation::{CompiledWorkflow, GuardKey},
    workflow::AgentDefinition,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::Arc,
};
use tokio::{sync::mpsc, task::JoinHandle, time::timeout};
use tracing::{debug, info, warn};
use ulid::Ulid;

// ── Checkpoint ───────────────────────────────────────────────────────────────

/// Serializable snapshot of the recoverable state of a [`WorkflowInstance`].
///
/// Captures all state needed to resume a suspended run: active states,
/// queued events, history, attempt counts, parallel completion flags, and
/// pending timers.
/// Ephemeral state (in-flight activity tasks, channels) is not captured —
/// on recovery the engine re-enters `active_states` to restart activities,
/// and calls `TimerRegistry::restore` to re-arm pending timers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceCheckpoint {
    pub run_id: RunId,
    pub workflow_id: String,
    pub workflow_version: String,
    pub status: RunStatus,
    pub active_states: Vec<StateId>,
    /// Run-scoped workflow data used by guards and input bindings.
    /// Defaults to absent when loading checkpoints written by older versions.
    #[serde(default)]
    pub workflow_data: Option<ron::Value>,
    /// Events acknowledged by the runtime but not yet processed by the RTC loop.
    #[serde(default)]
    pub event_queue: VecDeque<QueuedEvent>,
    /// Activity invocation that owns each queued activity completion event.
    /// This prevents a recovered stale completion from being applied to a new
    /// invocation of the same state.
    #[serde(default)]
    pub queued_activity_invocations: HashMap<StateId, InvocationId>,
    /// Compound/parallel state → last-active child configuration.
    pub history: HashMap<StateId, Vec<StateId>>,
    /// Per-state attempt count for retry policy tracking.
    pub attempt_counts: HashMap<StateId, u32>,
    /// Per-parallel-state completion flags: outer key = parallel StateId (as String),
    /// inner key = RegionId (as String).
    pub parallel_regions_done: HashMap<String, HashMap<String, bool>>,
    /// Pending timer entries. On recovery these are passed to
    /// `TimerRegistry::restore` which re-arms them with their remaining delay.
    /// Defaults to empty for forward-compatibility with older checkpoints.
    #[serde(default)]
    pub pending_timers: Vec<crate::timer::TimerEntry>,
}

// ── Run status ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Suspended,
    Completed,
    Failed,
    Cancelled,
}

// ── Queued event ──────────────────────────────────────────────────────────────

/// An event enqueued for processing by the RTC loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedEvent {
    pub event_type: String,
    pub payload: serde_json::Value,
    pub source: EventSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSource {
    /// Emitted by an agent actor completing its run.
    Activity {
        state_id: StateId,
        invocation_id: InvocationId,
    },
    /// Fired by a durable timer.
    Timer { timer_id: crate::timer::TimerId },
    /// Delivered externally (human input, integration, etc.).
    External,
    /// Broadcast by an integration. An unhandled broadcast is observable but
    /// does not fail a workflow whose strict unhandled-event policy is enabled.
    ExternalBroadcast,
    /// Synthesised by the runtime (cancellation, timeout, parallel completion, etc.).
    Internal,
}

// ── Activity result ───────────────────────────────────────────────────────────

/// Result of a spawned activity task, sent back on the internal channel.
enum ActivityResult {
    Completed {
        state_id: StateId,
        #[allow(dead_code)]
        invocation_id: InvocationId,
        event: AgentOutputEvent,
    },
    Failed {
        state_id: StateId,
        #[allow(dead_code)]
        invocation_id: InvocationId,
        error: AgentError,
    },
    Cancelled {
        state_id: StateId,
        #[allow(dead_code)]
        invocation_id: InvocationId,
    },
    /// A child subworkflow completed with a final output event.
    SubworkflowCompleted {
        state_id: StateId,
        invocation_id: InvocationId,
        output_event_type: String,
        output_payload: serde_json::Value,
    },
    /// A child subworkflow failed.
    SubworkflowFailed {
        state_id: StateId,
        invocation_id: InvocationId,
        message: String,
    },
    /// Retry timer fired — re-start the activity for `state_id`.
    RetryReady {
        state_id: StateId,
        /// Original error message (for tracing).
        #[allow(dead_code)]
        message: String,
    },
}

// ── WorkflowInstance ──────────────────────────────────────────────────────────

/// The live execution state of a single workflow run.
pub struct WorkflowInstance {
    pub run_id: RunId,
    pub status: RunStatus,

    // Compiled workflow (immutable for the run's lifetime).
    workflow: Arc<CompiledWorkflow>,

    // Active state IDs (flat set — parallel regions add multiple entries).
    pub active_states: Vec<StateId>,

    // External event queue (dequeued one at a time by the RTC loop).
    event_queue: VecDeque<QueuedEvent>,

    // Channel for activity tasks to report completion back to the instance.
    activity_tx: mpsc::UnboundedSender<ActivityResult>,
    activity_rx: mpsc::UnboundedReceiver<ActivityResult>,
    pending_activity_results: VecDeque<ActivityResult>,

    // Timer channel (fires become queued events).
    #[allow(dead_code)]
    timer_tx: mpsc::UnboundedSender<TimerFired>,
    timer_rx: mpsc::UnboundedReceiver<TimerFired>,

    // Live activity handles (cancelled on state exit).
    activities: HashMap<StateId, JoinHandle<()>>,

    // Invocation currently owned by each activity state. Results from older
    // invocations are ignored after exit, suspend/resume, or re-entry.
    active_invocations: HashMap<StateId, InvocationId>,

    // Revocable authority retained separately from the actor-owned envelope.
    invocation_leases: HashMap<StateId, InvocationLease>,

    // Completed activities whose derived event is still waiting in the event
    // queue. Ownership is revalidated when the event is dequeued.
    queued_activity_invocations: HashMap<StateId, InvocationId>,

    // Delayed retry handles are state-owned so exiting a state cancels them.
    retry_tasks: HashMap<StateId, JoinHandle<()>>,

    // Durable timer registry.
    timers: TimerRegistry,

    // Idempotent outbox (Phase 4+: checkpoint recovery re-delivery).
    #[allow(dead_code)]
    outbox: Outbox,

    // Adapters.
    broker: Arc<CapabilityBroker>,
    event_sink: Arc<dyn EventSink>,

    // Per-state agent actor registry.
    actors: HashMap<StateId, Arc<dyn AgentActor>>,

    // Action registry (on_entry / on_exit action IDs → implementations).
    action_registry: HashMap<String, Arc<dyn StateAction>>,

    // ── Phase 4: Parallel completion tracking ────────────────────────────────
    //
    // For each active parallel state: maps region_id → whether its final
    // state has been reached. Entries are inserted when the parallel state
    // is entered and removed when it exits.
    parallel_regions_done: HashMap<StateId, HashMap<RegionId, bool>>,

    // ── Phase 4: History ─────────────────────────────────────────────────────
    //
    // Compound / parallel state → last active leaf configuration at exit time.
    // Shallow: only direct-child states. Deep: full leaf set.
    history: HashMap<StateId, Vec<StateId>>,

    // ── B2: Retry tracking ────────────────────────────────────────────────────
    //
    // Number of attempts made so far per state (including the initial attempt).
    // Reset to 0 when the state is successfully exited.
    attempt_counts: HashMap<StateId, u32>,

    // ── B3: Subworkflow repository ────────────────────────────────────────────
    //
    // Optional repository for resolving child workflows by `workflow_ref`.
    // When `None` the subworkflow stub emits `SubworkflowFailed`.
    workflow_repo: Option<Arc<dyn WorkflowRepository>>,

    // ── C1: Context resolver ──────────────────────────────────────────────────
    //
    // Optional pipeline that resolves a `ContextView` before each agent
    // invocation.  When `None` the invocation receives an empty context.
    context_resolver: Option<Arc<dyn ContextResolver>>,

    // ── D: Checkpoint store ───────────────────────────────────────────────────
    //
    // When set, the engine saves a checkpoint on suspend / complete / fail.
    pub(crate) checkpoint_store: Option<Arc<dyn CheckpointStore>>,

    // ── F1: Workflow data ──────────────────────────────────────────────────────
    //
    // Optional run-time workflow data (RON-typed). When present, top-level
    // fields are exposed as `data.<field>` variables in CEL guard expressions.
    workflow_data: Option<ron::Value>,

    // The causal event that transitioned this run into a top-level final state.
    completion_event: Option<AgentOutputEvent>,
}

impl WorkflowInstance {
    pub fn new(
        run_id: RunId,
        workflow: Arc<CompiledWorkflow>,
        broker: Arc<CapabilityBroker>,
        event_sink: Arc<dyn EventSink>,
        actors: HashMap<StateId, Arc<dyn AgentActor>>,
    ) -> Self {
        Self::with_actions(run_id, workflow, broker, event_sink, actors, HashMap::new())
    }

    /// Like `new` but also accepts an action registry for on_entry/on_exit hooks.
    pub fn with_actions(
        run_id: RunId,
        workflow: Arc<CompiledWorkflow>,
        broker: Arc<CapabilityBroker>,
        event_sink: Arc<dyn EventSink>,
        actors: HashMap<StateId, Arc<dyn AgentActor>>,
        action_registry: HashMap<String, Arc<dyn StateAction>>,
    ) -> Self {
        let (activity_tx, activity_rx) = mpsc::unbounded_channel();
        let (timer_tx, timer_rx) = mpsc::unbounded_channel();
        let timers = TimerRegistry::new(run_id.clone(), timer_tx.clone());

        Self {
            run_id,
            status: RunStatus::Running,
            workflow,
            active_states: Vec::new(),
            event_queue: VecDeque::new(),
            activity_tx,
            activity_rx,
            pending_activity_results: VecDeque::new(),
            timer_tx,
            timer_rx,
            activities: HashMap::new(),
            active_invocations: HashMap::new(),
            invocation_leases: HashMap::new(),
            queued_activity_invocations: HashMap::new(),
            retry_tasks: HashMap::new(),
            timers,
            outbox: Outbox::new(),
            broker,
            event_sink,
            actors,
            action_registry,
            parallel_regions_done: HashMap::new(),
            history: HashMap::new(),
            attempt_counts: HashMap::new(),
            workflow_repo: None,
            context_resolver: None,
            checkpoint_store: None,
            workflow_data: None,
            completion_event: None,
        }
    }

    /// Like `with_actions` but also injects a [`WorkflowRepository`] so that
    /// `Subworkflow` states can resolve and spawn real child instances.
    pub fn with_workflow_repo(mut self, repo: Arc<dyn WorkflowRepository>) -> Self {
        self.workflow_repo = Some(repo);
        self
    }

    /// Inject a [`ContextResolver`] so agent invocations receive a real context
    /// view instead of an empty placeholder.
    pub fn with_context_resolver(mut self, resolver: Arc<dyn ContextResolver>) -> Self {
        self.context_resolver = Some(resolver);
        self
    }

    /// Inject a [`CheckpointStore`] so the engine saves a snapshot on suspend /
    /// complete / fail.
    pub fn with_checkpoint_store(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.checkpoint_store = Some(store);
        self
    }

    /// Provide run-time workflow data (RON-typed).
    ///
    /// When set, top-level fields of the RON value are exposed as `data.<field>`
    /// variables inside CEL guard expressions (Spec §8.2, §11.1).
    pub fn with_workflow_data(mut self, data: ron::Value) -> Self {
        self.workflow_data = Some(data);
        self
    }

    // ── Checkpointing ─────────────────────────────────────────────────────────

    /// Capture the recoverable runtime state into an [`InstanceCheckpoint`].
    ///
    /// This does NOT include ephemeral state (in-flight async tasks and
    /// channels). Recovery re-enters active states from scratch, re-starts
    /// activities without queued completions, and re-arms timers.
    pub fn take_checkpoint(&self) -> InstanceCheckpoint {
        InstanceCheckpoint {
            run_id: self.run_id.clone(),
            workflow_id: self.workflow.document.id.0.clone(),
            workflow_version: self.workflow.document.version.0.clone(),
            status: self.status.clone(),
            active_states: self.active_states.clone(),
            workflow_data: self.workflow_data.clone(),
            event_queue: self.event_queue.clone(),
            queued_activity_invocations: self.queued_activity_invocations.clone(),
            history: self.history.clone(),
            attempt_counts: self.attempt_counts.clone(),
            parallel_regions_done: self
                .parallel_regions_done
                .iter()
                .map(|(k, v)| {
                    (
                        k.0.clone(),
                        v.iter().map(|(r, b)| (r.0.clone(), *b)).collect(),
                    )
                })
                .collect(),
            // Spec §8.4: A checkpoint MUST include all pending timer state.
            pending_timers: self.timers.active_entries(),
        }
    }

    /// Save the current checkpoint to the store (if one is configured).
    /// Errors are logged but do not abort the run.
    pub async fn save_checkpoint(&self) {
        let Some(store) = &self.checkpoint_store else {
            return;
        };
        let ck = self.take_checkpoint();
        let payload = match serde_json::to_vec(&ck) {
            Ok(b) => b,
            Err(e) => {
                warn!(run = %self.run_id, error = %e, "checkpoint serialization failed");
                return;
            }
        };
        use langchart_adapters::checkpoint::RunSnapshot as CkSnapshot;
        use langchart_model::id::CheckpointId;
        use ulid::Ulid;
        let snap = CkSnapshot {
            run_id: self.run_id.clone(),
            checkpoint_id: CheckpointId::new(Ulid::generate().to_string()),
            payload,
        };
        if let Err(e) = store.save(&snap).await {
            warn!(run = %self.run_id, error = %e, "checkpoint save failed");
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Enter the initial state and start the run.
    pub async fn start(&mut self) -> Result<(), EngineError> {
        let initial_id = StateId::new(self.workflow.document.initial.clone());
        info!(run = %self.run_id, initial = %initial_id, "run starting");
        self.emit(RuntimeEventPayload::RunStarted).await?;
        self.enter_state(&initial_id).await
    }

    /// Schedule a durable timer that will inject `event_type` into the workflow's
    /// event queue after `delay`. Returns the timer ID for optional cancellation.
    ///
    /// The timer is associated with `state_id` so it is automatically cancelled
    /// if that state exits. It is captured in the next checkpoint and re-armed on
    /// `restore_from_checkpoint`, so it survives suspend/recover cycles.
    pub fn schedule_timer(
        &mut self,
        state_id: StateId,
        event_type: impl Into<String>,
        delay: std::time::Duration,
    ) -> crate::timer::TimerId {
        self.timers.schedule(state_id, event_type, delay)
    }

    /// Enqueue an external event for processing.
    pub fn send(&mut self, event_type: impl Into<String>, payload: serde_json::Value) {
        self.event_queue.push_back(QueuedEvent {
            event_type: event_type.into(),
            payload,
            source: EventSource::External,
        });
    }

    /// Enqueue an integration broadcast that may be irrelevant to the current
    /// state. Unlike directed external input, an unhandled broadcast never
    /// fails the run.
    pub fn send_broadcast(&mut self, event_type: impl Into<String>, payload: serde_json::Value) {
        self.event_queue.push_back(QueuedEvent {
            event_type: event_type.into(),
            payload,
            source: EventSource::ExternalBroadcast,
        });
    }

    /// Suspend the run. Activities are cancelled; timers remain armed.
    pub async fn suspend(&mut self) -> Result<(), EngineError> {
        if self.status != RunStatus::Running {
            return Err(EngineError::AlreadySuspended);
        }
        self.cancel_all_activities().await;
        self.status = RunStatus::Suspended;
        self.emit(RuntimeEventPayload::RunSuspended).await?;
        self.save_checkpoint().await;
        info!(run = %self.run_id, "run suspended");
        Ok(())
    }

    /// Resume a suspended run; re-enters active states.
    pub async fn resume(&mut self) -> Result<(), EngineError> {
        if self.status != RunStatus::Suspended {
            return Err(EngineError::NotSuspended);
        }
        self.status = RunStatus::Running;
        self.emit(RuntimeEventPayload::RunResumed).await?;
        // Re-start activities for all currently active states.
        let states: Vec<StateId> = self.active_states.clone();
        for state_id in states {
            if !self.queued_activity_invocations.contains_key(&state_id) {
                self.start_activity_if_needed(&state_id).await?;
            }
        }
        info!(run = %self.run_id, "run resumed");
        Ok(())
    }

    /// Cancel the run immediately. All activities are aborted.
    pub async fn cancel(&mut self) -> Result<(), EngineError> {
        self.cancel_all_activities().await;
        self.status = RunStatus::Cancelled;
        self.emit(RuntimeEventPayload::RunCancelled).await?;
        self.save_checkpoint().await;
        info!(run = %self.run_id, "run cancelled");
        Ok(())
    }

    /// Transition the run to a failed terminal state and stop every activity.
    ///
    /// State and cleanup are applied before observability so a failing event
    /// sink cannot leave work running behind a non-terminal instance.
    pub(crate) async fn fail(&mut self, message: String) -> Result<(), EngineError> {
        self.cancel_all_activities().await;
        self.status = RunStatus::Failed;
        let emit_result = self.emit(RuntimeEventPayload::RunFailed { message }).await;
        self.save_checkpoint().await;
        emit_result
    }

    // ── Main drive loop ───────────────────────────────────────────────────────

    /// Drive the instance until it reaches a terminal state.
    pub async fn run_to_completion(&mut self) -> Result<RunStatus, EngineError> {
        loop {
            if !matches!(self.status, RunStatus::Running) {
                return Ok(self.status.clone());
            }

            if !self.has_immediate_work() {
                self.wait_for_work().await;
            }
            self.step().await?;
        }
    }

    /// Single-step version used by the run task's `tokio::select!` loop.
    pub async fn step(&mut self) -> Result<bool, EngineError> {
        match self.status {
            RunStatus::Running => {}
            RunStatus::Suspended => return Ok(false),
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled => return Ok(true),
        }

        // Drain timers.
        while let Ok(fired) = self.timer_rx.try_recv() {
            self.event_queue.push_back(QueuedEvent {
                event_type: fired.event_type,
                payload: serde_json::Value::Null,
                source: EventSource::Timer {
                    timer_id: fired.timer_id,
                },
            });
        }

        // Handle a result received by `wait_for_work` before draining the
        // channel. Keeping receipt and processing separate makes the wait
        // future safe to cancel from the engine's command select loop.
        while let Some(result) = self.pending_activity_results.pop_front() {
            self.handle_activity_result(result).await?;
        }

        // Drain activity completions.
        while let Ok(result) = self.activity_rx.try_recv() {
            self.handle_activity_result(result).await?;
        }

        // Process one event if available.
        if let Some(event) = self.event_queue.pop_front() {
            self.process_event(event).await?;
        } else {
            tokio::task::yield_now().await;
        }

        Ok(!matches!(self.status, RunStatus::Running))
    }

    pub(crate) fn has_immediate_work(&self) -> bool {
        matches!(
            self.status,
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
        ) || !self.event_queue.is_empty()
            || !self.pending_activity_results.is_empty()
            || !self.activity_rx.is_empty()
            || !self.timer_rx.is_empty()
    }

    /// Wait for work without processing it. Receiving from Tokio's MPSC is
    /// cancellation-safe; the selected item is synchronously retained before
    /// this future can yield again.
    pub(crate) async fn wait_for_work(&mut self) {
        if self.has_immediate_work() {
            return;
        }
        tokio::select! {
            Some(result) = self.activity_rx.recv() => {
                self.pending_activity_results.push_back(result);
            }
            Some(fired) = self.timer_rx.recv() => {
                self.event_queue.push_back(QueuedEvent {
                    event_type: fired.event_type,
                    payload: serde_json::Value::Null,
                    source: EventSource::Timer { timer_id: fired.timer_id },
                });
            }
        }
    }

    // ── RTC step ──────────────────────────────────────────────────────────────

    async fn process_event(&mut self, event: QueuedEvent) -> Result<(), EngineError> {
        debug!(run = %self.run_id, event = %event.event_type, "processing event");

        if let EventSource::Activity {
            state_id,
            invocation_id,
        } = &event.source
        {
            let is_current = self.active_states.contains(state_id)
                && self.queued_activity_invocations.get(state_id) == Some(invocation_id);
            if !is_current {
                debug!(
                    run = %self.run_id,
                    state = %state_id,
                    invocation = %invocation_id,
                    "ignoring stale queued activity event"
                );
                return Ok(());
            }
            self.queued_activity_invocations.remove(state_id);
        }

        if let EventSource::Timer { timer_id } = &event.source {
            let Some(entry) = self.timers.consume_fired(timer_id) else {
                debug!(run = %self.run_id, timer = %timer_id.0, "ignoring cancelled timer event");
                return Ok(());
            };
            if !self.active_states.contains(&entry.state_id) {
                debug!(
                    run = %self.run_id,
                    timer = %timer_id.0,
                    state = %entry.state_id,
                    "ignoring timer event for inactive state"
                );
                return Ok(());
            }
        }

        // Handle parallel.completed synthetic event specially.
        if event.event_type.starts_with("parallel.completed:") {
            let parallel_id_str = &event.event_type["parallel.completed:".len()..];
            let parallel_id = StateId::new(parallel_id_str);
            return self.exit_parallel_state(&parallel_id).await;
        }

        // Collect all enabled transitions across all active states.
        // For parallel states, each region's active state processes independently.
        let transitions = self.find_all_transitions(&event);

        if transitions.is_empty() {
            self.emit(RuntimeEventPayload::EventUnhandled {
                event_type: event.event_type.clone(),
            })
            .await?;
            debug!(run = %self.run_id, event = %event.event_type, "event unhandled");
            if self.workflow.document.policy.unhandled_event_is_failure
                && !matches!(event.source, EventSource::ExternalBroadcast)
            {
                let message = format!("unhandled event `{}`", event.event_type);
                self.fail(message).await?;
            }
            return Ok(());
        }

        // Execute all enabled transitions (one per orthogonal region).
        // In a non-parallel workflow this is at most one transition.
        for (source_state_id, spec) in transitions {
            let target_id = spec.target.clone();

            self.emit(RuntimeEventPayload::TransitionSelected {
                from: source_state_id.clone(),
                to: target_id.clone(),
                event_type: event.event_type.clone(),
                event_payload: event.payload.clone(),
            })
            .await?;

            match spec.kind {
                // ── Internal ─────────────────────────────────────────────────
                // Spec §8.3: handles event without exiting/re-entering the
                // source state.  No on_exit or on_entry actions run.  The
                // state configuration is unchanged.
                TransitionKind::Internal => {
                    debug!(
                        run    = %self.run_id,
                        source = %source_state_id,
                        target = %target_id,
                        "internal transition — skipping exit/enter"
                    );
                }

                // ── Local ─────────────────────────────────────────────────
                // Spec §8.3: remains within the compound state hierarchy.
                // If the source is a compound state and the target is a strict
                // descendant, exit only the active inner leaf states (not the
                // compound parent itself) and enter the target.
                // If the target is NOT a descendant, behave like External.
                TransitionKind::Local => {
                    let source_def = self.find_state_def(&source_state_id);
                    let source_descendants = source_def
                        .map(collect_all_descendant_ids)
                        .unwrap_or_default();

                    if !source_descendants.is_empty() && source_descendants.contains(&target_id) {
                        // Source is a compound/parallel state and target is
                        // inside it — exit only the active descendant leaves,
                        // keep the source compound active.
                        debug!(
                            run    = %self.run_id,
                            source = %source_state_id,
                            target = %target_id,
                            "local transition — target is descendant, keeping compound active"
                        );
                        // Exit every currently-active state that is a descendant
                        // of source (inner-to-outer order already maintained by
                        // active_states ordering).
                        let leaves_to_exit: Vec<StateId> = self
                            .active_states
                            .iter()
                            .filter(|s| *s != &source_state_id && source_descendants.contains(*s))
                            .cloned()
                            .collect();
                        for leaf in leaves_to_exit {
                            self.exit_state(&leaf).await?;
                        }
                        self.enter_state(&target_id).await?;
                    } else {
                        // Target is outside the source boundary — fall back to
                        // External semantics.
                        debug!(
                            run    = %self.run_id,
                            source = %source_state_id,
                            target = %target_id,
                            "local transition — target not a descendant, using external semantics"
                        );
                        self.exit_state(&source_state_id).await?;
                        self.enter_state(&target_id).await?;
                    }
                }

                // ── External (default) ────────────────────────────────────
                TransitionKind::External => {
                    self.exit_state(&source_state_id).await?;
                    self.enter_state(&target_id).await?;
                }
            }

            // If the run completed during enter_state, stop processing further transitions.
            if !matches!(self.status, RunStatus::Running) {
                break;
            }
        }

        if self.status == RunStatus::Completed && self.completion_event.is_none() {
            self.completion_event = Some(AgentOutputEvent {
                event_type: event.event_type,
                payload: event.payload,
            });
        }

        Ok(())
    }

    /// Find all enabled transitions for `event` across the active configuration.
    ///
    /// For non-parallel workflows this returns at most one `(source, spec)`.
    /// For parallel states it may return one per orthogonal region.
    fn find_all_transitions(&self, event: &QueuedEvent) -> Vec<(StateId, TransitionSpec)> {
        // Group active states by their containing parallel region (if any).
        // States not in a parallel region are in the "root" group.
        //
        // Algorithm: for each active state, find its highest-priority enabled
        // transition. Within a parallel region, each region may independently
        // fire one transition. At the root level, only the highest-priority
        // transition fires.
        //
        // Since active_states is ordered, parallel region states appear after
        // their active parallel parent. We group them by region membership.

        // Build a map: active_state → (region_of_parallel_parent, or None)
        let region_groups = self.group_active_states_by_region();

        // For each group, pick the best enabled transition (if any).
        let mut result: Vec<(StateId, TransitionSpec)> = Vec::new();

        // root group
        if let Some(root_states) = region_groups.get(&None)
            && let Some(t) = self.best_transition_from(root_states, event)
        {
            // A transition selected at the root configuration exits the whole
            // parallel state and therefore preempts region-local transitions.
            return vec![t];
        }

        // parallel region groups
        let region_keys: Vec<Option<(StateId, RegionId)>> = region_groups
            .keys()
            .filter(|k| k.is_some())
            .cloned()
            .collect();

        for key in region_keys {
            if let Some(states) = region_groups.get(&key)
                && let Some(t) = self.best_transition_from(states, event)
            {
                result.push(t);
            }
        }

        result
    }

    /// For a slice of state IDs, return the single best enabled transition
    /// (lowest priority number, then earliest in `active_states` as tiebreaker).
    ///
    /// Implements Spec §8.5 event bubbling: if no transition is found on the
    /// leaf state itself, the event is propagated upward through compound
    /// ancestors until one handles it.
    fn best_transition_from(
        &self,
        states: &[StateId],
        event: &QueuedEvent,
    ) -> Option<(StateId, TransitionSpec)> {
        let mut best: Option<(i32, usize, StateId, TransitionSpec)> = None;

        for state_id in states {
            // Build the probe chain: [leaf, parent, grandparent, ...].
            // Check each level in order; the first matching level wins for this leaf.
            let ancestors = find_ancestor_chain(&self.workflow.document.states, state_id);
            let probe_chain = std::iter::once(state_id.clone()).chain(ancestors);

            for probe_id in probe_chain {
                if let Some(state_def) = self.find_state_def(&probe_id)
                    && let Some(specs) = state_def.on.get(&event.event_type)
                {
                    // Found a transition at this level — evaluate guards and
                    // stop climbing for this leaf (first matching ancestor wins).
                    for (transition_index, spec) in specs.iter().enumerate() {
                        let guard_ok = if spec.guard.is_some() {
                            let key = GuardKey {
                                state_id: probe_id.clone(),
                                event_type: event.event_type.clone(),
                                transition_index,
                            };
                            self.workflow
                                .guards
                                .get(&key)
                                .map(|guard| self.evaluate_guard(guard, event))
                                .unwrap_or(false)
                        } else {
                            true
                        };

                        if guard_ok {
                            // Position in active_states for stable tiebreaking.
                            // Use the original leaf's position (not the ancestor's).
                            let pos = self
                                .active_states
                                .iter()
                                .position(|s| s == state_id)
                                .unwrap_or(usize::MAX);

                            let is_better = best
                                .as_ref()
                                .map(|(bp, bpos, _, _)| {
                                    spec.priority < *bp || (spec.priority == *bp && pos < *bpos)
                                })
                                .unwrap_or(true);

                            if is_better {
                                // The source is the ancestor that handled it,
                                // not the leaf — so that process_event exits
                                // the right states.
                                best = Some((spec.priority, pos, probe_id.clone(), spec.clone()));
                            }
                        }
                    }
                    // Stop climbing for this leaf once any matching level is found.
                    break;
                }
            }
        }

        best.map(|(_, _, sid, spec)| (sid, spec))
    }

    /// Group the active states by `(parallel_state_id, region_id)` key.
    /// States that are not inside a parallel region map to `None`.
    fn group_active_states_by_region(&self) -> HashMap<Option<(StateId, RegionId)>, Vec<StateId>> {
        let mut groups: HashMap<Option<(StateId, RegionId)>, Vec<StateId>> = HashMap::new();

        for state_id in &self.active_states {
            let key = self.find_parallel_region_key(state_id);
            groups.entry(key).or_default().push(state_id.clone());
        }

        groups
    }

    /// Walk the state tree to find if `state_id` lives inside a parallel
    /// region, returning `Some((parallel_state_id, region_id))` if so.
    fn find_parallel_region_key(&self, target: &StateId) -> Option<(StateId, RegionId)> {
        find_parallel_region_key_in(&self.workflow.document.states, target)
    }

    fn evaluate_guard(
        &self,
        guard: &langchart_model::guard::CompiledGuard,
        event: &QueuedEvent,
    ) -> bool {
        let mut ctx = langchart_model::guard::evaluation_context();

        // ── B5: Runtime context variables ─────────────────────────────────────
        // Inject well-known runtime variables so guards can inspect run state:
        //   `run_id`           — the current run's identifier string
        //   `workflow_id`      — the workflow document's `id` field
        //   `workflow_version` — the workflow document's `version` field
        //
        // These complement the event payload fields (see below) and allow
        // routing / branching logic without extra data in every event.
        let _ = ctx.add_variable("run_id", self.run_id.0.as_str());
        let _ = ctx.add_variable("workflow_id", self.workflow.document.id.0.as_str());
        let _ = ctx.add_variable(
            "workflow_version",
            self.workflow.document.version.0.as_str(),
        );

        // ── F1: Workflow data fields ───────────────────────────────────────────
        // Expose top-level workflow data fields as `data.<field>` in CEL.
        // The ron::Value is first round-tripped through serde_json so we can use
        // cel_interpreter::to_value for the actual CEL conversion.
        if let Some(wd) = &self.workflow_data
            && let Ok(json_val) = serde_json::to_value(wd)
            && let Ok(cel_val) = cel_interpreter::to_value(&json_val)
        {
            let _ = ctx.add_variable("data", cel_val);
        }

        // Event payload fields: each top-level key becomes a variable.
        if let serde_json::Value::Object(map) = &event.payload {
            for (k, v) in map {
                if let Ok(cel_val) = cel_interpreter::to_value(v) {
                    let _ = ctx.add_variable(k, cel_val);
                }
            }
        }

        guard.evaluate(&ctx).unwrap_or(false)
    }

    fn resolve_agent_input(&self, state_id: &StateId) -> Result<ron::Value, EngineError> {
        let bindings = self
            .find_state_def(state_id)
            .map(|state| state.input.clone())
            .unwrap_or_default();
        self.resolve_workflow_bindings(state_id, &bindings)
    }

    fn resolve_workflow_bindings(
        &self,
        state_id: &StateId,
        bindings: &HashMap<String, String>,
    ) -> Result<ron::Value, EngineError> {
        use cel_interpreter::{Context, Program};

        let workflow_json = self
            .workflow_data
            .as_ref()
            .and_then(|data| serde_json::to_value(data).ok())
            .unwrap_or_else(|| serde_json::json!({}));
        let mut resolved = serde_json::Map::new();

        for (field, binding) in bindings {
            let value = if let Some(expression) = binding
                .strip_prefix("${")
                .and_then(|value| value.strip_suffix('}'))
            {
                let expression = expression
                    .strip_prefix("workflow.")
                    .map(|field| format!("data.{field}"))
                    .unwrap_or_else(|| expression.to_owned());
                let program = Program::compile(&expression).map_err(|error| {
                    EngineError::Activity(format!(
                        "invalid input binding `{binding}` for state `{state_id}`: {error}"
                    ))
                })?;
                let mut context = Context::default();
                let cel_data = cel_interpreter::to_value(&workflow_json).map_err(|error| {
                    EngineError::Serialization(format!(
                        "workflow data is not CEL-compatible: {error}"
                    ))
                })?;
                context.add_variable("data", cel_data).map_err(|error| {
                    EngineError::Activity(format!("could not bind workflow data: {error}"))
                })?;
                program
                    .execute(&context)
                    .map_err(|error| {
                        EngineError::Activity(format!(
                            "input binding `{binding}` failed for state `{state_id}`: {error}"
                        ))
                    })?
                    .json()
                    .map_err(|error| EngineError::Serialization(error.to_string()))?
            } else if let Some(value) = workflow_json.get(binding) {
                value.clone()
            } else {
                serde_json::Value::String(binding.clone())
            };
            resolved.insert(field.clone(), value);
        }

        serde_json::from_value(serde_json::Value::Object(resolved))
            .map_err(|error| EngineError::Serialization(error.to_string()))
    }

    // ── State entry/exit ──────────────────────────────────────────────────────

    // `enter_state` is recursive (via `enter_parallel_state` → each region's
    // initial state). Rust requires explicit `Box::pin` for recursive async fns.
    fn enter_state<'a>(
        &'a mut self,
        state_id: &'a StateId,
    ) -> Pin<Box<dyn Future<Output = Result<(), EngineError>> + Send + 'a>> {
        Box::pin(async move {
            // ── B4: History pseudo-state ─────────────────────────────────────
            // A transition target of `"<compound_id>.history"` means
            // "restore the last-known active configuration of that compound
            // state, falling back to its declared `initial` if no history
            // has been recorded yet."
            if let Some(compound_id_str) = state_id.0.strip_suffix(".history") {
                let compound_id = StateId::new(compound_id_str.to_string());
                debug!(
                    run   = %self.run_id,
                    state = %compound_id,
                    "entering via history pseudo-state"
                );
                self.emit(RuntimeEventPayload::StateEntered {
                    state_id: state_id.clone(),
                })
                .await?;
                let had_history = self.restore_history(&compound_id).await?;
                if !had_history {
                    // No history yet — fall back to the compound state's initial.
                    let fallback = self
                        .find_state_def(&compound_id)
                        .and_then(|d| d.initial.clone())
                        .unwrap_or(compound_id.clone());
                    debug!(
                        run      = %self.run_id,
                        state    = %compound_id,
                        fallback = %fallback,
                        "no history recorded; using initial fallback"
                    );
                    self.enter_state(&fallback).await?;
                }
                return Ok(());
            }

            let state_type = self
                .find_state_def(state_id)
                .map(|d| d.state_type.clone())
                .unwrap_or(StateType::Atomic);

            match state_type {
                StateType::Parallel => self.enter_parallel_state(state_id).await,
                StateType::Compound => {
                    // Compound state: emit StateEntered, run on_entry actions,
                    // add to active_states, then automatically enter the initial
                    // child state (SCXML semantics).
                    debug!(run = %self.run_id, state = %state_id, "entering compound state");
                    self.run_on_entry_actions(state_id).await?;
                    self.active_states.push(state_id.clone());
                    self.emit(RuntimeEventPayload::StateEntered {
                        state_id: state_id.clone(),
                    })
                    .await?;
                    let initial = self
                        .find_state_def(state_id)
                        .and_then(|d| d.initial.clone());
                    if let Some(init_id) = initial {
                        self.enter_state(&init_id).await?;
                    }
                    Ok(())
                }
                StateType::Final => {
                    debug!(run = %self.run_id, state = %state_id, "entering final state");
                    self.run_on_entry_actions(state_id).await?;
                    self.active_states.push(state_id.clone());
                    self.emit(RuntimeEventPayload::StateEntered {
                        state_id: state_id.clone(),
                    })
                    .await?;

                    // Check if this Final is inside a parallel region.
                    if let Some((parallel_id, region_id)) = self.find_parallel_region_key(state_id)
                    {
                        self.mark_region_done(&parallel_id, &region_id, state_id)
                            .await?;
                    } else {
                        // Top-level final state → run completed.
                        self.status = RunStatus::Completed;
                        self.emit(RuntimeEventPayload::RunCompleted).await?;
                        self.save_checkpoint().await;
                        info!(run = %self.run_id, "run completed");
                    }
                    Ok(())
                }
                _ => {
                    debug!(run = %self.run_id, state = %state_id, "entering state");
                    self.run_on_entry_actions(state_id).await?;
                    self.active_states.push(state_id.clone());
                    self.emit(RuntimeEventPayload::StateEntered {
                        state_id: state_id.clone(),
                    })
                    .await?;
                    self.start_activity_if_needed(state_id).await
                }
            }
        })
    }

    async fn enter_parallel_state(&mut self, parallel_id: &StateId) -> Result<(), EngineError> {
        debug!(run = %self.run_id, state = %parallel_id, "entering parallel state");
        self.run_on_entry_actions(parallel_id).await?;
        self.active_states.push(parallel_id.clone());
        self.emit(RuntimeEventPayload::StateEntered {
            state_id: parallel_id.clone(),
        })
        .await?;

        // Initialise completion tracking for this parallel state.
        let region_ids: Vec<(RegionId, StateId)> = self
            .find_state_def(parallel_id)
            .map(|def| {
                def.regions
                    .iter()
                    .map(|r| (r.id.clone(), r.initial.clone()))
                    .collect()
            })
            .unwrap_or_default();

        if region_ids.is_empty() {
            warn!(
                run = %self.run_id,
                state = %parallel_id,
                "parallel state has no regions; completing immediately"
            );
            self.synthesise_parallel_completed(parallel_id);
            return Ok(());
        }

        let mut done_map: HashMap<RegionId, bool> = HashMap::new();
        for (rid, _) in &region_ids {
            done_map.insert(rid.clone(), false);
        }
        self.parallel_regions_done
            .insert(parallel_id.clone(), done_map);

        // Enter each region's initial state.
        for (region_id, initial_id) in &region_ids {
            self.emit(RuntimeEventPayload::ParallelRegionEntered {
                parallel_id: parallel_id.clone(),
                region_id: region_id.clone(),
            })
            .await?;
            // Box to avoid borrow issues with &self.
            let initial_id = initial_id.clone();
            self.enter_state(&initial_id).await?;
        }

        Ok(())
    }

    /// Mark a region as done, then check the parallel completion mode.
    async fn mark_region_done(
        &mut self,
        parallel_id: &StateId,
        region_id: &RegionId,
        _final_state_id: &StateId,
    ) -> Result<(), EngineError> {
        if let Some(map) = self.parallel_regions_done.get_mut(parallel_id) {
            map.insert(region_id.clone(), true);
        }

        self.emit(RuntimeEventPayload::ParallelRegionCompleted {
            parallel_id: parallel_id.clone(),
            region_id: region_id.clone(),
        })
        .await?;

        // Evaluate completion mode.
        let completion_mode = self
            .find_state_def(parallel_id)
            .and_then(|d| d.completion.clone())
            .unwrap_or_default(); // Default::All

        let done_map = self
            .parallel_regions_done
            .get(parallel_id)
            .cloned()
            .unwrap_or_default();
        let total = done_map.len();
        let completed = done_map.values().filter(|&&v| v).count();

        let satisfied = match &completion_mode {
            ParallelCompletion::All => completed == total,
            ParallelCompletion::Any => completed >= 1,
            ParallelCompletion::Quorum { n } => completed >= *n,
            ParallelCompletion::Guard { expr } => {
                // Evaluate CEL with `completed` and `total` as variables.
                use cel_interpreter::{Context, Program};
                Program::compile(expr)
                    .ok()
                    .and_then(|prog| {
                        let mut ctx = Context::default();
                        let _ = ctx.add_variable("completed", completed as i64);
                        let _ = ctx.add_variable("total", total as i64);
                        prog.execute(&ctx).ok()
                    })
                    .map(|v| matches!(v, cel_interpreter::objects::Value::Bool(true)))
                    .unwrap_or(false)
            }
            ParallelCompletion::Manual => false, // Requires explicit external event.
        };

        if satisfied {
            self.synthesise_parallel_completed(parallel_id);
        }

        Ok(())
    }

    fn synthesise_parallel_completed(&mut self, parallel_id: &StateId) {
        // Use a namespaced event type so the RTC loop can dispatch it.
        let event_type = format!("parallel.completed:{}", parallel_id.0);
        self.event_queue.push_front(QueuedEvent {
            event_type,
            payload: serde_json::Value::Null,
            source: EventSource::Internal,
        });
    }

    /// Exit all active states belonging to a parallel state, then enter its
    /// transition target (if any) or complete the run.
    async fn exit_parallel_state(&mut self, parallel_id: &StateId) -> Result<(), EngineError> {
        self.emit(RuntimeEventPayload::ParallelCompleted {
            parallel_id: parallel_id.clone(),
        })
        .await?;

        self.exit_parallel_configuration(parallel_id).await?;

        // If the parallel state has a `parallel.completed` transition, fire it.
        // Otherwise check if parallel state was the root and complete the run.
        let transition = self.find_state_def(parallel_id).and_then(|d| {
            d.on.get("parallel.completed")
                .and_then(|v| v.first())
                .cloned()
        });

        if let Some(spec) = transition {
            let target = spec.target.clone();
            self.emit(RuntimeEventPayload::TransitionSelected {
                from: parallel_id.clone(),
                to: target.clone(),
                event_type: "parallel.completed".into(),
                event_payload: serde_json::Value::Null,
            })
            .await?;
            self.enter_state(&target).await?;
        } else {
            // No transition; treat parallel completion as a run completion
            // if the parallel state is at the top level.
            let is_top_level = self
                .workflow
                .document
                .states
                .iter()
                .any(|s| &s.id == parallel_id);

            if is_top_level {
                self.status = RunStatus::Completed;
                self.emit(RuntimeEventPayload::RunCompleted).await?;
                self.save_checkpoint().await;
                info!(run = %self.run_id, "run completed via parallel");
            }
        }

        Ok(())
    }

    async fn exit_parallel_configuration(
        &mut self,
        parallel_id: &StateId,
    ) -> Result<(), EngineError> {
        // Collect all active states that belong to regions of this parallel.
        let region_states: Vec<StateId> = self
            .active_states
            .iter()
            .filter(|s| {
                self.find_parallel_region_key(s)
                    .map(|(pid, _)| &pid == parallel_id)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        // Exit region states (inner to outer).
        for s in region_states {
            self.exit_state_silent(&s).await?;
        }

        // Clean up completion tracking.
        self.parallel_regions_done.remove(parallel_id);
        self.active_states.retain(|state| state != parallel_id);

        // Run lifecycle hooks and emit StateExited for the parallel state itself.
        self.run_on_exit_actions(parallel_id).await?;
        self.emit(RuntimeEventPayload::StateExited {
            state_id: parallel_id.clone(),
        })
        .await?;
        Ok(())
    }

    /// Exit a state without doing completion-checking side-effects (used when
    /// bulk-exiting parallel region states).
    async fn exit_state_silent(&mut self, state_id: &StateId) -> Result<(), EngineError> {
        self.stop_invocation(state_id).await;
        if let Some(handle) = self.retry_tasks.remove(state_id) {
            handle.abort();
        }
        self.queued_activity_invocations.remove(state_id);
        let timer_ids: Vec<_> = self
            .timers
            .active_entries()
            .into_iter()
            .filter(|e| &e.state_id == state_id)
            .map(|e| e.id)
            .collect();
        for tid in timer_ids {
            self.timers.cancel(&tid);
        }
        self.active_states.retain(|s| s != state_id);
        self.run_on_exit_actions(state_id).await?;
        self.emit(RuntimeEventPayload::StateExited {
            state_id: state_id.clone(),
        })
        .await?;
        Ok(())
    }

    async fn exit_state(&mut self, state_id: &StateId) -> Result<(), EngineError> {
        debug!(run = %self.run_id, state = %state_id, "exiting state");

        if self
            .find_state_def(state_id)
            .is_some_and(|state| state.state_type == StateType::Parallel)
        {
            return self.exit_parallel_configuration(state_id).await;
        }

        // Save history on the exiting state itself (no-op if it has no history mode).
        self.save_history(state_id);

        // Also save history on all ancestor compound/parallel states that have
        // a `history` mode configured.  This is the common case: a leaf state
        // (e.g. `hb`) exits and its parent compound (`compound_h`) must record
        // the current active-child configuration so it can be restored later.
        let ancestors = find_ancestors_with_history(&self.workflow.document.states, state_id);
        for ancestor_id in ancestors {
            self.save_history(&ancestor_id);
        }

        self.stop_invocation(state_id).await;
        if let Some(handle) = self.retry_tasks.remove(state_id) {
            handle.abort();
        }
        self.queued_activity_invocations.remove(state_id);

        let timer_ids: Vec<_> = self
            .timers
            .active_entries()
            .into_iter()
            .filter(|e| &e.state_id == state_id)
            .map(|e| e.id)
            .collect();
        for tid in timer_ids {
            self.timers.cancel(&tid);
        }

        self.active_states.retain(|s| s != state_id);
        self.run_on_exit_actions(state_id).await?;
        self.emit(RuntimeEventPayload::StateExited {
            state_id: state_id.clone(),
        })
        .await?;
        Ok(())
    }

    // ── History ───────────────────────────────────────────────────────────────

    /// Save the current active-child configuration for `state_id` if it is a
    /// compound or parallel state that has a `history` mode configured.
    fn save_history(&mut self, state_id: &StateId) {
        let Some(def) = self.find_state_def(state_id) else {
            return;
        };
        if def.history.is_none() {
            return;
        }

        let snapshot: Vec<StateId> = match def.history {
            Some(langchart_model::state::HistoryMode::Shallow) => {
                // Direct children only.
                let child_ids: HashSet<StateId> = def
                    .states
                    .iter()
                    .map(|s| s.id.clone())
                    .chain(
                        def.regions
                            .iter()
                            .flat_map(|r| r.states.iter().map(|s| s.id.clone())),
                    )
                    .collect();
                self.active_states
                    .iter()
                    .filter(|s| child_ids.contains(s))
                    .cloned()
                    .collect()
            }
            Some(langchart_model::state::HistoryMode::Deep) | None => {
                // All active descendants.
                let all_descendants = collect_all_descendant_ids(def);
                self.active_states
                    .iter()
                    .filter(|s| all_descendants.contains(s))
                    .cloned()
                    .collect()
            }
        };

        if !snapshot.is_empty() {
            self.history.insert(state_id.clone(), snapshot);
        }
    }

    /// Restore historical configuration for `state_id`, entering saved states.
    /// Returns `true` if history was available and restored.
    pub async fn restore_history(&mut self, state_id: &StateId) -> Result<bool, EngineError> {
        let snapshot = match self.history.get(state_id).cloned() {
            Some(s) if !s.is_empty() => s,
            _ => return Ok(false),
        };

        self.emit(RuntimeEventPayload::HistoryRestored {
            state_id: state_id.clone(),
        })
        .await?;

        for sid in snapshot {
            self.active_states.push(sid.clone());
            self.emit(RuntimeEventPayload::StateEntered {
                state_id: sid.clone(),
            })
            .await?;
            self.start_activity_if_needed(&sid).await?;
        }
        Ok(true)
    }

    // ── Action execution ──────────────────────────────────────────────────────

    /// Run all `on_entry` actions declared on `state_id` in order.
    async fn run_on_entry_actions(&mut self, state_id: &StateId) -> Result<(), EngineError> {
        let action_ids: Vec<String> = self
            .find_state_def(state_id)
            .map(|d| d.on_entry.clone())
            .unwrap_or_default();
        self.run_actions(state_id, &action_ids, ActionTrigger::Entry)
            .await
    }

    /// Run all `on_exit` actions declared on `state_id` in order.
    async fn run_on_exit_actions(&mut self, state_id: &StateId) -> Result<(), EngineError> {
        let action_ids: Vec<String> = self
            .find_state_def(state_id)
            .map(|d| d.on_exit.clone())
            .unwrap_or_default();
        self.run_actions(state_id, &action_ids, ActionTrigger::Exit)
            .await
    }

    async fn run_actions(
        &mut self,
        state_id: &StateId,
        action_ids: &[String],
        trigger: ActionTrigger,
    ) -> Result<(), EngineError> {
        for action_id in action_ids {
            let action = match self.action_registry.get(action_id.as_str()).cloned() {
                Some(a) => a,
                None => {
                    warn!(
                        run = %self.run_id,
                        state = %state_id,
                        action_id = %action_id,
                        "on_entry/on_exit action not found in registry; skipping"
                    );
                    continue;
                }
            };

            self.emit(RuntimeEventPayload::ActionStarted {
                state_id: state_id.clone(),
                action_id: action_id.clone(),
            })
            .await?;

            let ctx = ActionContext {
                run_id: self.run_id.clone(),
                state_id: state_id.clone(),
                action_id: action_id.clone(),
                trigger: trigger.clone(),
            };

            match action.run(ctx, self.broker.clone()).await {
                Ok(()) => {
                    self.emit(RuntimeEventPayload::ActionCompleted {
                        state_id: state_id.clone(),
                        action_id: action_id.clone(),
                    })
                    .await?;
                }
                Err(e) => {
                    self.emit(RuntimeEventPayload::ActionFailed {
                        state_id: state_id.clone(),
                        action_id: action_id.clone(),
                        message: e.message.clone(),
                    })
                    .await?;
                    // Action failure does not abort the run; it logs and continues.
                    // To make an action failure fatal, implement retry / error transitions
                    // in the workflow document.
                    warn!(
                        run = %self.run_id,
                        state = %state_id,
                        action_id = %action_id,
                        error = %e,
                        "on_entry/on_exit action failed; continuing"
                    );
                }
            }
        }
        Ok(())
    }

    // ── Activity dispatch ─────────────────────────────────────────────────────

    /// Public wrapper for `start_activity_if_needed` — used by `replay::fork_instance`
    /// and `engine::recover_run`.
    pub async fn start_activity_if_needed_pub(
        &mut self,
        state_id: &StateId,
    ) -> Result<(), EngineError> {
        if self.queued_activity_invocations.contains_key(state_id) {
            return Ok(());
        }
        self.start_activity_if_needed(state_id).await
    }

    // ── Checkpoint restore ────────────────────────────────────────────────────

    /// Restore mutable runtime state from a previously captured [`InstanceCheckpoint`].
    ///
    /// This overwrites recoverable state, including queued events and their
    /// activity ownership. Ephemeral tasks and channels are **not** restored —
    /// the caller is responsible for re-starting activities that do not already
    /// have a queued completion.
    pub fn restore_from_checkpoint(&mut self, ck: &InstanceCheckpoint) {
        use langchart_model::id::RegionId;
        self.active_states = ck.active_states.clone();
        self.workflow_data = ck.workflow_data.clone();
        self.event_queue = ck.event_queue.clone();
        self.queued_activity_invocations = ck.queued_activity_invocations.clone();
        self.history = ck.history.clone();
        self.attempt_counts = ck.attempt_counts.clone();
        self.status = ck.status.clone();
        self.parallel_regions_done = ck
            .parallel_regions_done
            .iter()
            .map(|(k, v)| {
                let state_id = StateId::new(k.clone());
                let region_map = v
                    .iter()
                    .map(|(r, b)| (RegionId::new(r.clone()), *b))
                    .collect();
                (state_id, region_map)
            })
            .collect();
        // Spec §8.4: re-arm pending timers with their remaining delay.
        if !ck.pending_timers.is_empty() {
            self.timers.restore(ck.pending_timers.clone());
        }
    }

    async fn start_activity_if_needed(&mut self, state_id: &StateId) -> Result<(), EngineError> {
        let state_type = self
            .find_state_def(state_id)
            .map(|d| d.state_type.clone())
            .unwrap_or(StateType::Atomic);

        match state_type {
            StateType::Agentic => self.start_agent_activity(state_id).await,
            StateType::Subworkflow => self.start_subworkflow_activity(state_id).await,
            StateType::Human => {
                self.emit(RuntimeEventPayload::HumanInputRequested {
                    state_id: state_id.clone(),
                })
                .await?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    async fn start_agent_activity(&mut self, state_id: &StateId) -> Result<(), EngineError> {
        let actor = match self.actors.get(state_id).cloned() {
            Some(a) => a,
            None => {
                warn!(run = %self.run_id, state = %state_id, "no actor registered for agentic state");
                return Err(EngineError::Activity(format!(
                    "no actor registered for state `{state_id}`"
                )));
            }
        };

        let invocation_id = InvocationId::new(Ulid::generate().to_string());
        let limits = self
            .find_state_def(state_id)
            .and_then(|d| d.limits.clone())
            .unwrap_or_default();

        let capability_policy = self.effective_capability_policy(state_id);

        let mut envelope = CapabilityEnvelope::new(
            capability_policy,
            self.run_id.clone(),
            invocation_id.clone(),
            state_id.clone(),
            limits.max_turns,
            limits.max_tool_calls,
        )
        .with_token_budget(limits.max_tokens_total);
        let lease = self.broker.authorize_envelope(&mut envelope);

        // ── C1: Context resolution ────────────────────────────────────────────
        // If a ContextResolver was injected, call it now to assemble the
        // context items for this invocation. Resolution errors fail the run;
        // only the absence of a configured resolver yields an empty view.
        let context_policy = self.effective_context_policy(state_id);

        let context_view = match &self.context_resolver {
            Some(resolver) => resolver
                .resolve(&context_policy, &self.run_id)
                .await
                .map_err(|error| {
                    EngineError::Activity(format!(
                        "context resolution failed for state `{state_id}`: {error}"
                    ))
                })?,
            None => ContextView {
                items: vec![],
                token_count: 0,
                content_hash: "empty".into(),
            },
        };
        let input = self.resolve_agent_input(state_id)?;

        let invocation = crate::instance::AgentInvocation {
            run_id: self.run_id.clone(),
            state_id: state_id.clone(),
            invocation_id: invocation_id.clone(),
            instructions: ResolvedInstructions {
                system: self
                    .find_state_def(state_id)
                    .and_then(|d| d.prompt.clone())
                    .unwrap_or_default(),
                task: None,
            },
            context_view,
            input,
            output_event_types: self
                .find_state_def(state_id)
                .map(|d| d.on.keys().cloned().collect())
                .unwrap_or_default(),
            limits,
        };

        let broker = self.broker.clone();
        let tx = self.activity_tx.clone();
        let sid = state_id.clone();
        let iid = invocation_id.clone();
        let wall_timeout = invocation.limits.timeout;

        // Track attempt count (first attempt = 1).
        let _attempt = {
            let count = self.attempt_counts.entry(state_id.clone()).or_insert(0);
            *count += 1;
            *count
        };

        self.emit(RuntimeEventPayload::ActivityStarted {
            state_id: state_id.clone(),
        })
        .await?;

        self.active_invocations
            .insert(state_id.clone(), invocation_id.clone());
        self.invocation_leases
            .insert(state_id.clone(), lease.clone());
        let lease_guard = lease.guard();

        let handle = tokio::spawn(async move {
            let lease_guard = lease_guard;
            let result = timeout(wall_timeout, actor.run(invocation, envelope, broker)).await;
            let activity_result = match result {
                Ok(Ok(event)) => ActivityResult::Completed {
                    state_id: sid.clone(),
                    invocation_id: iid.clone(),
                    event,
                },
                Ok(Err(e)) => ActivityResult::Failed {
                    state_id: sid.clone(),
                    invocation_id: iid.clone(),
                    error: e,
                },
                Err(_elapsed) => ActivityResult::Cancelled {
                    state_id: sid.clone(),
                    invocation_id: iid.clone(),
                },
            };
            lease_guard.revoke_and_wait().await;
            let _ = tx.send(activity_result);
        });

        self.activities.insert(state_id.clone(), handle);
        Ok(())
    }

    // ── Subworkflow ───────────────────────────────────────────────────────────

    async fn start_subworkflow_activity(&mut self, state_id: &StateId) -> Result<(), EngineError> {
        let Some(def) = self.find_state_def(state_id) else {
            return Err(EngineError::Activity(format!(
                "subworkflow state `{state_id}` definition not found"
            )));
        };

        let workflow_ref = match &def.workflow_ref {
            Some(r) => r.clone(),
            None => {
                return Err(EngineError::Activity(format!(
                    "subworkflow state `{state_id}` has no workflow_ref"
                )));
            }
        };
        let child_input = def
            .ports
            .as_ref()
            .map(|ports| self.resolve_workflow_bindings(state_id, &ports.input))
            .transpose()?;

        self.emit(RuntimeEventPayload::SubworkflowStarted {
            state_id: state_id.clone(),
        })
        .await?;

        let invocation_id = InvocationId::new(Ulid::generate().to_string());
        self.active_invocations
            .insert(state_id.clone(), invocation_id.clone());

        let tx = self.activity_tx.clone();
        let sid = state_id.clone();
        let wref = workflow_ref.clone();
        let broker = self.broker.clone();
        let event_sink = self.event_sink.clone();
        let parent_run_id = self.run_id.clone();
        let repo = self.workflow_repo.clone();
        let actors = self.actors.clone();
        let iid = invocation_id;

        let handle = tokio::spawn(async move {
            // Resolve the child workflow from the repository if one is present.
            let child_compiled = match &repo {
                Some(r) => r.get(&wref).await,
                None => None,
            };

            let Some(child_wf) = child_compiled else {
                // No repository or workflow not found — emit failure.
                warn!(
                    state = %sid,
                    workflow_ref = %wref,
                    "subworkflow not found in repository; emitting SubworkflowFailed"
                );
                let _ = tx.send(ActivityResult::SubworkflowFailed {
                    state_id: sid,
                    invocation_id: iid.clone(),
                    message: format!("subworkflow `{wref}` not found in repository"),
                });
                return;
            };

            // Derive a child run-id so it is traceable back to the parent.
            let child_run_id =
                langchart_model::id::RunId::new(format!("{}::{}", parent_run_id, sid));

            let mut child = crate::run::WorkflowInstance::new(
                child_run_id,
                child_wf.clone(),
                broker,
                event_sink,
                actors,
            );
            let input_json = child_input
                .as_ref()
                .and_then(|input| serde_json::to_value(input).ok())
                .unwrap_or_else(|| serde_json::json!({}));
            if let Some(required) = child_wf
                .document
                .inputs
                .iter()
                .find(|port| port.required && input_json.get(&port.name).is_none())
            {
                let _ = tx.send(ActivityResult::SubworkflowFailed {
                    state_id: sid,
                    invocation_id: iid.clone(),
                    message: format!(
                        "required child input port `{}` has no binding",
                        required.name
                    ),
                });
                return;
            }
            if let Some(input) = child_input {
                child = child.with_workflow_data(input);
            }

            if let Err(e) = child.start().await {
                let _ = tx.send(ActivityResult::SubworkflowFailed {
                    state_id: sid,
                    invocation_id: iid.clone(),
                    message: e.to_string(),
                });
                return;
            }

            let result = child.run_to_completion().await;

            match result {
                Ok(crate::run::RunStatus::Completed) => {
                    let output = child.completion_event.take().unwrap_or(AgentOutputEvent {
                        event_type: "completed".into(),
                        payload: serde_json::json!({}),
                    });
                    let _ = tx.send(ActivityResult::SubworkflowCompleted {
                        state_id: sid,
                        invocation_id: iid.clone(),
                        output_event_type: output.event_type,
                        output_payload: output.payload,
                    });
                }
                Ok(other) => {
                    let _ = tx.send(ActivityResult::SubworkflowFailed {
                        state_id: sid,
                        invocation_id: iid.clone(),
                        message: format!("child run ended with status {:?}", other),
                    });
                }
                Err(e) => {
                    let _ = tx.send(ActivityResult::SubworkflowFailed {
                        state_id: sid,
                        invocation_id: iid.clone(),
                        message: e.to_string(),
                    });
                }
            }
        });

        self.activities.insert(state_id.clone(), handle);
        Ok(())
    }

    async fn handle_activity_failure(
        &mut self,
        state_id: StateId,
        invocation_id: InvocationId,
        error: AgentError,
    ) -> Result<(), EngineError> {
        let message = error.to_string();
        self.emit(RuntimeEventPayload::ActivityFailed {
            state_id: state_id.clone(),
            message: message.clone(),
        })
        .await?;

        let retry = self.find_state_def(&state_id).and_then(|d| d.retry.clone());
        let attempts_so_far = self.attempt_counts.get(&state_id).copied().unwrap_or(1);

        if let Some(ref policy) = retry {
            let error_class = agent_error_class(&error);
            let retryable = policy.retryable_on.is_empty()
                || policy
                    .retryable_on
                    .iter()
                    .any(|entry| entry == error_class || entry == &message);

            if retryable && attempts_so_far < policy.max_attempts {
                let delay = compute_retry_delay(policy, attempts_so_far);
                let attempt = attempts_so_far + 1;
                self.emit(RuntimeEventPayload::ActivityRetried {
                    state_id: state_id.clone(),
                    attempt,
                })
                .await?;

                let tx = self.activity_tx.clone();
                let retry_state_id = state_id.clone();
                let handle = tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    let _ = tx.send(ActivityResult::RetryReady {
                        state_id: retry_state_id,
                        message,
                    });
                });
                if let Some(previous) = self.retry_tasks.insert(state_id, handle) {
                    previous.abort();
                }
                return Ok(());
            }

            if let Some(ref target) = policy.on_exhausted {
                let target_id = StateId::new(target.clone());
                self.emit(RuntimeEventPayload::TransitionSelected {
                    from: state_id.clone(),
                    to: target_id.clone(),
                    event_type: "retry.exhausted".into(),
                    event_payload: serde_json::json!({ "error": message }),
                })
                .await?;
                self.exit_state(&state_id).await?;
                self.attempt_counts.remove(&state_id);
                self.enter_state(&target_id).await?;
                return Ok(());
            }
        }

        self.attempt_counts.remove(&state_id);
        let failure_event = QueuedEvent {
            event_type: "activity.failed".into(),
            payload: serde_json::json!({ "error": message }),
            source: EventSource::Activity {
                state_id: state_id.clone(),
                invocation_id: invocation_id.clone(),
            },
        };
        if self
            .best_transition_from(std::slice::from_ref(&state_id), &failure_event)
            .is_some()
        {
            self.queued_activity_invocations
                .insert(state_id, invocation_id);
            self.event_queue.push_back(failure_event);
        } else {
            self.fail(message).await?;
        }
        Ok(())
    }

    async fn handle_activity_result(&mut self, result: ActivityResult) -> Result<(), EngineError> {
        match result {
            ActivityResult::Completed {
                state_id,
                invocation_id,
                event,
            } => {
                if !self.take_active_invocation(&state_id, &invocation_id) {
                    return Ok(());
                }
                // Spec §9.4: For agentic states, validate the emitted event_type
                // against the agent's declared output_events list.
                // For non-agentic states, fall back to checking for a transition.
                let declared = self.is_output_event_declared(&state_id, &event.event_type);

                if !declared {
                    self.emit(RuntimeEventPayload::ActivityInvalidOutput {
                        state_id: state_id.clone(),
                        event_type: event.event_type.clone(),
                    })
                    .await?;
                    self.handle_activity_failure(
                        state_id,
                        invocation_id,
                        AgentError::Internal(format!(
                            "undeclared output event `{}`",
                            event.event_type
                        )),
                    )
                    .await?;
                } else {
                    // Spec §8.1: Validate the payload against the state's declared
                    // output_schema for this event type (if one exists).
                    let schema_err = self
                        .find_state_def(&state_id)
                        .and_then(|def| def.output_schemas.get(&event.event_type))
                        .and_then(|schema| schema.validate(&event.payload).err());

                    if let Some(reason) = schema_err {
                        warn!(
                            run = %self.run_id,
                            state = %state_id,
                            event_type = %event.event_type,
                            %reason,
                            "event payload failed schema validation"
                        );
                        self.emit(RuntimeEventPayload::ActivityInvalidOutput {
                            state_id: state_id.clone(),
                            event_type: event.event_type.clone(),
                        })
                        .await?;
                        self.handle_activity_failure(
                            state_id,
                            invocation_id,
                            AgentError::Internal(format!(
                                "invalid payload for `{}`: {reason}",
                                event.event_type
                            )),
                        )
                        .await?;
                    } else {
                        self.attempt_counts.remove(&state_id);
                        self.emit(RuntimeEventPayload::ActivityCompleted {
                            state_id: state_id.clone(),
                        })
                        .await?;
                        self.queue_activity_event(
                            state_id,
                            invocation_id,
                            event.event_type,
                            event.payload,
                        );
                    }
                }
            }
            ActivityResult::Failed {
                state_id,
                invocation_id,
                error,
            } => {
                if !self.take_active_invocation(&state_id, &invocation_id) {
                    return Ok(());
                }
                self.handle_activity_failure(state_id, invocation_id, error)
                    .await?;
            }
            ActivityResult::Cancelled {
                state_id,
                invocation_id,
            } => {
                if !self.take_active_invocation(&state_id, &invocation_id) {
                    return Ok(());
                }
                self.emit(RuntimeEventPayload::ActivityCancelled {
                    state_id: state_id.clone(),
                })
                .await?;
                self.queue_activity_event(
                    state_id,
                    invocation_id,
                    "activity.cancelled".into(),
                    serde_json::Value::Null,
                );
            }
            ActivityResult::SubworkflowCompleted {
                state_id,
                invocation_id,
                output_event_type,
                output_payload,
            } => {
                if !self.take_active_invocation(&state_id, &invocation_id) {
                    return Ok(());
                }
                match self.map_subworkflow_output(&state_id, &output_event_type, &output_payload) {
                    Ok((parent_event_type, parent_payload)) => {
                        self.emit(RuntimeEventPayload::SubworkflowCompleted {
                            state_id: state_id.clone(),
                        })
                        .await?;
                        self.queue_activity_event(
                            state_id,
                            invocation_id,
                            parent_event_type,
                            parent_payload,
                        );
                    }
                    Err(error) => {
                        let message = error.to_string();
                        self.emit(RuntimeEventPayload::SubworkflowFailed {
                            state_id: state_id.clone(),
                            message: message.clone(),
                        })
                        .await?;
                        self.queue_activity_event(
                            state_id,
                            invocation_id,
                            "subworkflow.failed".into(),
                            serde_json::json!({ "error": message }),
                        );
                    }
                }
            }
            ActivityResult::SubworkflowFailed {
                state_id,
                invocation_id,
                message,
            } => {
                if !self.take_active_invocation(&state_id, &invocation_id) {
                    return Ok(());
                }
                self.emit(RuntimeEventPayload::SubworkflowFailed {
                    state_id: state_id.clone(),
                    message: message.clone(),
                })
                .await?;
                self.queue_activity_event(
                    state_id,
                    invocation_id,
                    "subworkflow.failed".into(),
                    serde_json::json!({ "error": message }),
                );
            }
            ActivityResult::RetryReady { state_id, .. } => {
                self.retry_tasks.remove(&state_id);
                if !self.active_states.contains(&state_id) {
                    debug!(run = %self.run_id, state = %state_id, "ignoring retry for inactive state");
                    return Ok(());
                }
                debug!(run = %self.run_id, state = %state_id, "retry timer fired; re-starting activity");
                self.start_agent_activity(&state_id).await?;
            }
        }
        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn map_subworkflow_output(
        &mut self,
        state_id: &StateId,
        child_event_type: &str,
        child_payload: &serde_json::Value,
    ) -> Result<(String, serde_json::Value), EngineError> {
        use cel_interpreter::{Context, Program};

        let output_ports = self
            .find_state_def(state_id)
            .and_then(|state| state.ports.as_ref())
            .map(|ports| ports.output.clone())
            .unwrap_or_default();
        if output_ports.is_empty() {
            return Ok(("subworkflow.completed".into(), serde_json::json!({})));
        }

        let bindings = output_ports.get(child_event_type).ok_or_else(|| {
            EngineError::Activity(format!(
                "subworkflow state `{state_id}` has no output mapping for child event `{child_event_type}`"
            ))
        })?;
        let workflow_json = self
            .workflow_data
            .as_ref()
            .and_then(|data| serde_json::to_value(data).ok())
            .unwrap_or_else(|| serde_json::json!({}));
        let event_json = serde_json::json!({ "payload": child_payload });
        let mut mapped = serde_json::Map::new();

        for (field, binding) in bindings {
            let value = if let Some(expression) = binding
                .strip_prefix("${")
                .and_then(|value| value.strip_suffix('}'))
            {
                let program = Program::compile(expression).map_err(|error| {
                    EngineError::Activity(format!(
                        "invalid output binding `{binding}` for state `{state_id}`: {error}"
                    ))
                })?;
                let mut context = Context::default();
                context
                    .add_variable(
                        "event",
                        cel_interpreter::to_value(&event_json)
                            .map_err(|error| EngineError::Serialization(error.to_string()))?,
                    )
                    .map_err(|error| EngineError::Activity(error.to_string()))?;
                context
                    .add_variable(
                        "workflow",
                        cel_interpreter::to_value(&workflow_json)
                            .map_err(|error| EngineError::Serialization(error.to_string()))?,
                    )
                    .map_err(|error| EngineError::Activity(error.to_string()))?;
                program
                    .execute(&context)
                    .map_err(|error| {
                        EngineError::Activity(format!(
                            "output binding `{binding}` failed for state `{state_id}`: {error}"
                        ))
                    })?
                    .json()
                    .map_err(|error| EngineError::Serialization(error.to_string()))?
            } else {
                serde_json::Value::String(binding.clone())
            };
            mapped.insert(field.clone(), value);
        }

        let mut parent_data = workflow_json;
        let parent_object = parent_data.as_object_mut().ok_or_else(|| {
            EngineError::Activity("parent workflow data must be an object".into())
        })?;
        parent_object.extend(mapped.clone());
        self.workflow_data = Some(
            serde_json::from_value(parent_data)
                .map_err(|error| EngineError::Serialization(error.to_string()))?,
        );

        Ok((
            format!("subworkflow.{child_event_type}"),
            serde_json::Value::Object(mapped),
        ))
    }

    fn queue_activity_event(
        &mut self,
        state_id: StateId,
        invocation_id: InvocationId,
        event_type: String,
        payload: serde_json::Value,
    ) {
        self.queued_activity_invocations
            .insert(state_id.clone(), invocation_id.clone());
        self.event_queue.push_back(QueuedEvent {
            event_type,
            payload,
            source: EventSource::Activity {
                state_id,
                invocation_id,
            },
        });
    }

    fn take_active_invocation(&mut self, state_id: &StateId, invocation_id: &InvocationId) -> bool {
        let is_current = self.active_states.contains(state_id)
            && self.active_invocations.get(state_id) == Some(invocation_id);
        if !is_current {
            debug!(
                run = %self.run_id,
                state = %state_id,
                invocation = %invocation_id,
                "ignoring stale activity result"
            );
            return false;
        }
        self.revoke_invocation(state_id);
        self.activities.remove(state_id);
        true
    }

    fn revoke_invocation(&mut self, state_id: &StateId) {
        if let Some(lease) = self.invocation_leases.remove(state_id) {
            lease.revoke();
        }
        self.active_invocations.remove(state_id);
    }

    async fn stop_invocation(&mut self, state_id: &StateId) {
        let lease = self.invocation_leases.remove(state_id);
        if let Some(lease) = &lease {
            lease.revoke();
        }
        self.active_invocations.remove(state_id);
        if let Some(handle) = self.activities.remove(state_id) {
            handle.abort();
        }
        if let Some(lease) = lease {
            lease.revoke_and_wait().await;
        }
    }

    /// Check whether `event_type` is a declared output for the activity at `state_id`.
    ///
    /// For **agentic** states with a resolvable `AgentDefinition`, the check is
    /// against `AgentDefinition.output_events` (Spec §9.4).
    /// For all other states the fallback is: does the state have an `on` transition
    /// for the event type?
    fn is_output_event_declared(&self, state_id: &StateId, event_type: &str) -> bool {
        let Some(state_def) = self.find_state_def(state_id) else {
            return false;
        };
        if state_def.state_type == StateType::Agentic
            && let Some(agent_ref) = &state_def.agent
            && let Some(agent_def) = self.find_agent_def(agent_ref)
        {
            return agent_def.output_events.iter().any(|e| e == event_type);
        }
        // Fallback for non-agentic activities: any event with a declared transition.
        state_def.on.contains_key(event_type)
    }

    /// Look up an `AgentDefinition` by its `AgentRef` in the workflow document.
    fn find_agent_def<'a>(
        &'a self,
        agent_ref: &langchart_model::state::AgentRef,
    ) -> Option<&'a langchart_model::workflow::AgentDefinition> {
        self.workflow
            .document
            .agents
            .iter()
            .find(|a| a.id == agent_ref.id && a.version == agent_ref.version)
    }

    fn agent_def_for_state(&self, state_id: &StateId) -> Option<&AgentDefinition> {
        let state = self.find_state_def(state_id)?;
        self.find_agent_def(state.agent.as_ref()?)
    }

    fn effective_capability_policy(&self, state_id: &StateId) -> CapabilityPolicy {
        let workflow_max = &self.workflow.document.policy.max_capabilities;
        let agent_default = self
            .agent_def_for_state(state_id)
            .map(|agent| &agent.default_capabilities)
            .cloned()
            .unwrap_or_default();
        let inherited = intersect_capability_policies(workflow_max, &agent_default);
        match self
            .find_state_def(state_id)
            .and_then(|state| state.capabilities.as_ref())
        {
            Some(state_policy) => intersect_capability_policies(&inherited, state_policy),
            None => inherited,
        }
    }

    fn effective_context_policy(&self, state_id: &StateId) -> ContextPolicy {
        self.find_state_def(state_id)
            .and_then(|state| state.context.clone())
            .or_else(|| {
                self.agent_def_for_state(state_id)
                    .map(|agent| agent.default_context_policy.clone())
            })
            .unwrap_or_default()
    }

    fn find_state_def<'a>(&'a self, id: &StateId) -> Option<&'a StateDefinition> {
        // O(1) lookup via the pre-built index (built during compile).
        // Fall back to the recursive walk only for pseudo-state IDs (e.g.
        // `"foo.history"`) which are not in the index.
        self.workflow.state_index.get(id)
    }

    async fn cancel_all_activities(&mut self) {
        let leases: Vec<_> = self
            .invocation_leases
            .drain()
            .map(|(_, lease)| lease)
            .collect();
        for lease in &leases {
            lease.revoke();
        }
        for (_, handle) in self.activities.drain() {
            handle.abort();
        }
        for (_, handle) in self.retry_tasks.drain() {
            handle.abort();
        }
        self.active_invocations.clear();
        for lease in leases {
            lease.revoke_and_wait().await;
        }
    }

    async fn emit(&self, payload: RuntimeEventPayload) -> Result<(), EngineError> {
        let event = RuntimeEvent {
            event_id: EventId::new(Ulid::generate().to_string()),
            run_id: self.run_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            payload,
        };
        self.event_sink
            .append(event)
            .await
            .map_err(|e| EngineError::Activity(e.to_string()))
    }
}

impl Drop for WorkflowInstance {
    fn drop(&mut self) {
        for (_, lease) in self.invocation_leases.drain() {
            lease.revoke();
        }
        for (_, handle) in self.activities.drain() {
            handle.abort();
        }
        for (_, handle) in self.retry_tasks.drain() {
            handle.abort();
        }
    }
}

fn intersect_capability_policies(
    parent: &CapabilityPolicy,
    child: &CapabilityPolicy,
) -> CapabilityPolicy {
    let mcp = parent
        .mcp
        .iter()
        .filter_map(|(server_id, parent_server)| {
            child.mcp.get(server_id).map(|child_server| {
                (
                    server_id.clone(),
                    intersect_mcp_server_policies(parent_server, child_server),
                )
            })
        })
        .collect();

    CapabilityPolicy {
        mcp,
        artifact_operations: intersect_vec(&parent.artifact_operations, &child.artifact_operations),
        memory_write: parent.memory_write && child.memory_write,
        // Elevation is a validation signal, never an effective permission.
        elevate: false,
    }
}

fn intersect_mcp_server_policies(
    parent: &McpServerPolicy,
    child: &McpServerPolicy,
) -> McpServerPolicy {
    McpServerPolicy {
        allow: intersect_vec(&parent.allow, &child.allow),
        resource_patterns: intersect_resource_patterns(
            &parent.resource_patterns,
            &child.resource_patterns,
        ),
        operations: intersect_vec(&parent.operations, &child.operations),
        call_budget: match (parent.call_budget, child.call_budget) {
            (Some(parent), Some(child)) => Some(parent.min(child)),
            (Some(limit), None) | (None, Some(limit)) => Some(limit),
            (None, None) => None,
        },
        credentials: intersect_vec(&parent.credentials, &child.credentials),
        require_human_confirmation: parent.require_human_confirmation
            || child.require_human_confirmation,
    }
}

fn intersect_vec<T: Clone + PartialEq>(parent: &[T], child: &[T]) -> Vec<T> {
    parent
        .iter()
        .filter(|item| child.contains(item))
        .cloned()
        .collect()
}

fn intersect_resource_patterns(parent: &[String], child: &[String]) -> Vec<String> {
    let mut intersection = Vec::new();
    for child_pattern in child {
        for parent_pattern in parent {
            let narrower = if resource_pattern_contains(parent_pattern, child_pattern) {
                Some(child_pattern)
            } else if resource_pattern_contains(child_pattern, parent_pattern) {
                Some(parent_pattern)
            } else {
                None
            };
            if let Some(pattern) = narrower
                && !intersection.contains(pattern)
            {
                intersection.push(pattern.clone());
            }
        }
    }
    intersection
}

/// Return whether every URI matched by `candidate` is also matched by
/// `container`. Besides equality, this recognizes the common unambiguous glob
/// form with one trailing `*`. Regex-like patterns are only intersected when
/// exactly equal because proving arbitrary regex containment is out of scope.
fn resource_pattern_contains(container: &str, candidate: &str) -> bool {
    if container == candidate || container == "*" {
        return true;
    }
    let Some(prefix) = container.strip_suffix('*') else {
        return false;
    };
    if prefix.contains('*') || looks_regex_like(container) || looks_regex_like(candidate) {
        return false;
    }
    candidate.starts_with(prefix)
}

fn looks_regex_like(pattern: &str) -> bool {
    pattern.starts_with("regex:") || pattern.chars().any(|ch| "|()^$+{}\\".contains(ch))
}

fn agent_error_class(error: &AgentError) -> &'static str {
    match error {
        AgentError::TurnLimitExhausted => "turn_limit_exhausted",
        AgentError::ToolCallLimitExhausted => "tool_call_limit_exhausted",
        AgentError::Internal(_) => "internal",
        AgentError::Cancelled => "cancelled",
    }
}

/// Walk the tree to find if `target` lives in a parallel region, returning
/// `Some((parallel_state_id, region_id))` if found.
fn find_parallel_region_key_in(
    states: &[StateDefinition],
    target: &StateId,
) -> Option<(StateId, RegionId)> {
    for state in states {
        if state.state_type == StateType::Parallel {
            for region in &state.regions {
                if region_contains(&region.states, target) {
                    return Some((state.id.clone(), region.id.clone()));
                }
                // Also check nested in non-parallel descendants inside the region.
                if let Some(k) = find_parallel_region_key_in(&region.states, target) {
                    return Some(k);
                }
            }
            // Parallel states may also use `states` instead of `regions`.
            for child in &state.states {
                if &child.id == target {
                    // Direct child of a parallel state without named regions
                    // — treat state.id as region id for completion tracking.
                    return Some((state.id.clone(), RegionId::new(child.id.0.clone())));
                }
            }
        }
        // Recurse into non-parallel compound children.
        if let Some(k) = find_parallel_region_key_in(&state.states, target) {
            return Some(k);
        }
        for region in &state.regions {
            if let Some(k) = find_parallel_region_key_in(&region.states, target) {
                return Some(k);
            }
        }
    }
    None
}

fn region_contains(states: &[StateDefinition], target: &StateId) -> bool {
    for s in states {
        if &s.id == target {
            return true;
        }
        if region_contains(&s.states, target) {
            return true;
        }
        for r in &s.regions {
            if region_contains(&r.states, target) {
                return true;
            }
        }
    }
    false
}

/// Collect all descendant state IDs (for deep history).
fn collect_all_descendant_ids(def: &StateDefinition) -> HashSet<StateId> {
    let mut ids = HashSet::new();
    collect_descendants_inner(def, &mut ids);
    ids
}

fn collect_descendants_inner(def: &StateDefinition, ids: &mut HashSet<StateId>) {
    for child in &def.states {
        ids.insert(child.id.clone());
        collect_descendants_inner(child, ids);
    }
    for region in &def.regions {
        for child in &region.states {
            ids.insert(child.id.clone());
            collect_descendants_inner(child, ids);
        }
    }
}

// ── History helpers ────────────────────────────────────────────────────────────

/// Return the IDs of all ancestor states of `target` that have a `history`
/// mode configured.  Used in `exit_state` so that history snapshots are
/// saved on the parent compound/parallel states, not just the exiting leaf.
fn find_ancestors_with_history(states: &[StateDefinition], target: &StateId) -> Vec<StateId> {
    let mut result = Vec::new();
    find_ancestors_with_history_inner(states, target, &mut result);
    result
}

fn find_ancestors_with_history_inner(
    states: &[StateDefinition],
    target: &StateId,
    out: &mut Vec<StateId>,
) -> bool {
    for state in states {
        let found_in_direct = state.states.iter().any(|c| &c.id == target)
            || state
                .regions
                .iter()
                .any(|r| r.states.iter().any(|c| &c.id == target));

        if found_in_direct {
            if state.history.is_some() {
                out.push(state.id.clone());
            }
            return true;
        }

        // Recurse into children.
        let found_deeper = find_ancestors_with_history_inner(&state.states, target, out)
            || state
                .regions
                .iter()
                .any(|r| find_ancestors_with_history_inner(&r.states, target, out));

        if found_deeper {
            if state.history.is_some() {
                out.push(state.id.clone());
            }
            return true;
        }
    }
    false
}

// ── Ancestor chain helper (for event bubbling, F4) ────────────────────────────

/// Return the ancestor chain of `target` from nearest (direct parent) to
/// farthest (root), for use in event bubbling (Spec §8.5).
fn find_ancestor_chain(states: &[StateDefinition], target: &StateId) -> Vec<StateId> {
    let mut chain = Vec::new();
    find_ancestor_chain_inner(states, target, &mut chain);
    chain
}

fn find_ancestor_chain_inner(
    states: &[StateDefinition],
    target: &StateId,
    out: &mut Vec<StateId>,
) -> bool {
    for state in states {
        // Check direct children (states and region states).
        let found_direct = state.states.iter().any(|c| &c.id == target)
            || state
                .regions
                .iter()
                .any(|r| r.states.iter().any(|c| &c.id == target));

        if found_direct {
            out.push(state.id.clone());
            return true;
        }

        // Recurse into children.
        let found_deeper = find_ancestor_chain_inner(&state.states, target, out)
            || state
                .regions
                .iter()
                .any(|r| find_ancestor_chain_inner(&r.states, target, out));

        if found_deeper {
            out.push(state.id.clone());
            return true;
        }
    }
    false
}

// ── Retry delay computation ────────────────────────────────────────────────────

fn compute_retry_delay(
    policy: &langchart_model::policy::RetryPolicy,
    attempt: u32,
) -> std::time::Duration {
    use langchart_model::policy::BackoffStrategy;
    let base = policy.delay;
    match policy.backoff {
        BackoffStrategy::Fixed => base,
        BackoffStrategy::Linear => base * attempt,
        BackoffStrategy::Exponential => {
            // 2^(attempt-1) * base, capped at 5 minutes.
            let shift = attempt.saturating_sub(1).min(62);
            let factor = 1u64 << shift;
            let nanos = (base.as_nanos() as u64).saturating_mul(factor);
            std::time::Duration::from_nanos(nanos.min(300_000_000_000u64)) // 5 min cap
        }
    }
}

#[cfg(test)]
mod capability_policy_tests {
    use super::*;
    use langchart_model::{
        id::{SecretRef, ServerId, ToolName},
        policy::OperationClass,
    };

    #[test]
    fn capability_intersection_never_widens_parent() {
        let server = ServerId::new("vault");
        let parent = CapabilityPolicy {
            mcp: HashMap::from([(
                server.clone(),
                McpServerPolicy {
                    allow: vec![ToolName::new("read"), ToolName::new("write")],
                    resource_patterns: vec!["vault://docs/*".into()],
                    operations: vec![OperationClass::Read],
                    call_budget: Some(3),
                    credentials: vec![SecretRef::new("reader")],
                    require_human_confirmation: false,
                },
            )]),
            artifact_operations: vec![OperationClass::Read],
            memory_write: false,
            elevate: false,
        };
        let child = CapabilityPolicy {
            mcp: HashMap::from([(
                server.clone(),
                McpServerPolicy {
                    allow: vec![ToolName::new("write"), ToolName::new("delete")],
                    resource_patterns: vec![
                        "vault://docs/public/*".into(),
                        "vault://secret/*".into(),
                    ],
                    operations: vec![OperationClass::Read, OperationClass::Delete],
                    call_budget: Some(10),
                    credentials: vec![SecretRef::new("reader"), SecretRef::new("admin")],
                    require_human_confirmation: true,
                },
            )]),
            artifact_operations: vec![OperationClass::Read, OperationClass::Commit],
            memory_write: true,
            elevate: true,
        };

        let effective = intersect_capability_policies(&parent, &child);
        let effective_server = effective.mcp.get(&server).expect("shared server");
        assert_eq!(effective_server.allow, vec![ToolName::new("write")]);
        assert_eq!(
            effective_server.resource_patterns,
            vec!["vault://docs/public/*"]
        );
        assert_eq!(effective_server.operations, vec![OperationClass::Read]);
        assert_eq!(effective.artifact_operations, vec![OperationClass::Read]);
        assert_eq!(effective_server.call_budget, Some(3));
        assert_eq!(effective_server.credentials, vec![SecretRef::new("reader")]);
        assert!(effective_server.require_human_confirmation);
        assert!(!effective.memory_write);
        assert!(!effective.elevate);
    }
}
