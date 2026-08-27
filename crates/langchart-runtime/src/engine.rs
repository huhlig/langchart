//! `RuntimeEngine` — the public API for creating and managing workflow runs.
//!
//! The engine is the host application's entry point. Wire up adapters,
//! register actors, then call `start` to launch a run.
//!
//! # Architecture
//! Each run is a `WorkflowInstance` running in its own tokio task. The engine
//! communicates with it through a per-run command channel. This keeps the
//! engine thread-safe and allows many runs to proceed concurrently.

use crate::{
    broker::CapabilityBroker,
    run::{InstanceCheckpoint, RunStatus, WorkflowInstance},
};
use langchart_adapters::{
    artifact::ArtifactStore,
    checkpoint::CheckpointStore,
    context::ContextResolver,
    event::{EventSink, EventSource, RuntimeEvent},
    secrets::SecretsAdapter,
    workflow_repository::WorkflowRepository,
};
use langchart_model::{
    id::{RunId, StateId},
    validation::compile,
    workflow::WorkflowDocument,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::sync::{mpsc, oneshot};
use tracing::info;
use ulid::Ulid;

// ── Commands sent to a run task ───────────────────────────────────────────────

enum RunCommand {
    Send {
        event_type: String,
        payload: serde_json::Value,
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
    SendBroadcast {
        event_type: String,
        payload: serde_json::Value,
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
    Suspend {
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
    Resume {
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
    Cancel {
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
    Inspect {
        reply: oneshot::Sender<RunSnapshot>,
    },
}

// ── Run snapshot ──────────────────────────────────────────────────────────────

/// A point-in-time view of a run's observable state.
#[derive(Debug, Clone)]
pub struct RunSnapshot {
    pub run_id: RunId,
    pub status: RunStatus,
    pub active_states: Vec<StateId>,
}

// ── Engine adapters bundle ────────────────────────────────────────────────────

/// All adapters needed to create a `RuntimeEngine`.
pub struct EngineAdapters {
    pub llm: Arc<dyn langchart_adapters::llm::LlmAdapter>,
    pub mcp: Arc<dyn langchart_adapters::mcp::McpAdapter>,
    pub memory: Arc<dyn langchart_adapters::memory::MemoryAdapter>,
    pub secrets: Arc<dyn SecretsAdapter>,
    pub event_sink: Arc<dyn EventSink>,
    /// Optional: persist run snapshots on suspend / complete / fail.
    pub checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    /// Optional: resolve child workflows by reference for `Subworkflow` states.
    pub workflow_repo: Option<Arc<dyn WorkflowRepository>>,
    /// Optional: subscribe to live event streams by run_id.
    /// If set, `RuntimeEngine::subscribe` will delegate to this source.
    /// A `BroadcastEventSink` can serve as both the event_sink and event_source.
    pub event_source: Option<Arc<dyn EventSource>>,
    /// Optional: artifact store for versioned artifact read/propose/commit.
    /// Exposed via `broker.read_artifact()`, `broker.propose_artifact()`,
    /// and `broker.commit_artifact()`.
    pub artifact_store: Option<Arc<dyn ArtifactStore>>,
}

// ── RuntimeEngine ─────────────────────────────────────────────────────────────

/// The primary host-application interface to the langchart engine.
///
/// ```text
/// let engine = RuntimeEngine::new(adapters);
/// let run_id = engine.start(workflow_doc, actors).await?;
/// engine.send(&run_id, "user.approved", serde_json::json!({})).await?;
/// ```
pub struct RuntimeEngine {
    broker: Arc<CapabilityBroker>,
    /// Optional checkpoint store — shared by all runs created by this engine.
    checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    /// Optional workflow repository for subworkflow resolution.
    workflow_repo: Option<Arc<dyn WorkflowRepository>>,
    /// Optional event source for live subscription.
    event_source: Option<Arc<dyn EventSource>>,
    /// Optional context resolver shared by runs created by this engine.
    context_resolver: Option<Arc<dyn ContextResolver>>,
    /// Live run handles: run_id → command sender.
    runs: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<RunCommand>>>>,
}

impl RuntimeEngine {
    pub fn new(adapters: EngineAdapters) -> Self {
        let broker = if let Some(art) = adapters.artifact_store {
            Arc::new(
                CapabilityBroker::new(
                    adapters.llm,
                    adapters.mcp,
                    adapters.memory,
                    adapters.secrets,
                    adapters.event_sink,
                )
                .with_artifact_store(art),
            )
        } else {
            Arc::new(CapabilityBroker::new(
                adapters.llm,
                adapters.mcp,
                adapters.memory,
                adapters.secrets,
                adapters.event_sink,
            ))
        };
        Self {
            broker,
            checkpoint_store: adapters.checkpoint_store,
            workflow_repo: adapters.workflow_repo,
            event_source: adapters.event_source,
            context_resolver: None,
            runs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Configure context resolution for every run created or recovered by this engine.
    pub fn with_context_resolver(mut self, resolver: Arc<dyn ContextResolver>) -> Self {
        self.context_resolver = Some(resolver);
        self
    }

    // ── Run lifecycle ─────────────────────────────────────────────────────────

    /// Compile and start a new workflow run.
    ///
    /// `actors` maps state IDs (for agentic states) to their `AgentActor`
    /// implementations. States with no entry in `actors` that are of type
    /// `agentic` will fail at runtime when entered.
    pub async fn start(
        &self,
        document: WorkflowDocument,
        actors: HashMap<StateId, Arc<dyn crate::instance::AgentActor>>,
    ) -> Result<RunId, EngineError> {
        let run_id = RunId::new(Ulid::generate().to_string());
        self.start_with_run_id(document, actors, run_id, None).await
    }

    /// Compile and start a workflow using a caller-provided run ID and
    /// optional run-time workflow data.
    pub async fn start_with_run_id(
        &self,
        document: WorkflowDocument,
        actors: HashMap<StateId, Arc<dyn crate::instance::AgentActor>>,
        run_id: RunId,
        workflow_data: Option<ron::Value>,
    ) -> Result<RunId, EngineError> {
        let compiled =
            compile(document).map_err(|e| EngineError::ValidationFailed(e.to_string()))?;
        let compiled = Arc::new(compiled);

        info!(run = %run_id, workflow = %compiled.document.id, "starting run");

        // Each run gets its own event sink wrapper that also records to the
        // engine's observable sink.
        let event_sink = self.broker.event_sink_ref();
        let mut instance = WorkflowInstance::new(
            run_id.clone(),
            compiled,
            self.broker.clone(),
            event_sink,
            actors,
        );
        if let Some(store) = &self.checkpoint_store {
            instance = instance.with_checkpoint_store(store.clone());
        }
        if let Some(repo) = &self.workflow_repo {
            instance = instance.with_workflow_repo(repo.clone());
        }
        if let Some(resolver) = &self.context_resolver {
            instance = instance.with_context_resolver(resolver.clone());
        }
        if let Some(data) = workflow_data {
            instance = instance.with_workflow_data(data);
        }
        instance.start().await?;

        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        self.runs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(run_id.0.clone(), cmd_tx);

        // Spawn the run task.
        let runs = self.runs.clone();
        let rid = run_id.clone();
        tokio::spawn(async move {
            run_task(instance, cmd_rx).await;
            runs.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&rid.0);
        });

        Ok(run_id)
    }

    /// Send an event into a running workflow.
    pub async fn send(
        &self,
        run_id: &RunId,
        event_type: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_cmd(
            run_id,
            RunCommand::Send {
                event_type: event_type.into(),
                payload,
                reply: reply_tx,
            },
        )?;
        reply_rx
            .await
            .map_err(|_| EngineError::RunNotFound(run_id.clone()))?
    }

    /// Broadcast an integration event into a run.
    ///
    /// If the current state does not handle the event it is still observable as
    /// `EventUnhandled`, but it does not trigger `unhandled_event_is_failure`.
    pub async fn send_broadcast(
        &self,
        run_id: &RunId,
        event_type: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_cmd(
            run_id,
            RunCommand::SendBroadcast {
                event_type: event_type.into(),
                payload,
                reply: reply_tx,
            },
        )?;
        reply_rx
            .await
            .map_err(|_| EngineError::RunNotFound(run_id.clone()))?
    }

    /// Suspend a running workflow.
    pub async fn suspend(&self, run_id: &RunId) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_cmd(run_id, RunCommand::Suspend { reply: reply_tx })?;
        reply_rx
            .await
            .map_err(|_| EngineError::RunNotFound(run_id.clone()))?
    }

    /// Resume a suspended workflow.
    pub async fn resume(&self, run_id: &RunId) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_cmd(run_id, RunCommand::Resume { reply: reply_tx })?;
        reply_rx
            .await
            .map_err(|_| EngineError::RunNotFound(run_id.clone()))?
    }

    /// Cancel a workflow run.
    pub async fn cancel(&self, run_id: &RunId) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_cmd(run_id, RunCommand::Cancel { reply: reply_tx })?;
        reply_rx
            .await
            .map_err(|_| EngineError::RunNotFound(run_id.clone()))?
    }

    /// Recover a running or suspended run from the latest checkpoint.
    ///
    /// Requires a `checkpoint_store` and a `workflow_repo` to be configured.
    /// Re-creates the `WorkflowInstance` from the saved [`InstanceCheckpoint`],
    /// restarts active states only for a running checkpoint, then spawns the
    /// run task. Suspended checkpoints remain quiescent until resumed.
    ///
    /// Returns `Err(EngineError::Checkpoint(_))` if no checkpoint is found or
    /// if the latest checkpoint is already terminal.
    pub async fn recover_run(
        &self,
        run_id: &RunId,
        actors: HashMap<StateId, Arc<dyn crate::instance::AgentActor>>,
    ) -> Result<RunId, EngineError> {
        let store = self
            .checkpoint_store
            .as_ref()
            .ok_or_else(|| EngineError::Checkpoint("no checkpoint store configured".into()))?;

        let snap = store
            .load(run_id)
            .await
            .map_err(|e| EngineError::Checkpoint(e.to_string()))?
            .ok_or_else(|| EngineError::Checkpoint(format!("no checkpoint for run {run_id}")))?;

        let ck: InstanceCheckpoint = serde_json::from_slice(&snap.payload)
            .map_err(|e| EngineError::Serialization(e.to_string()))?;

        if matches!(
            ck.status,
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
        ) {
            return Err(EngineError::Checkpoint(format!(
                "cannot recover terminal run {run_id} with status {:?}",
                ck.status
            )));
        }

        // Resolve the workflow document from the repository.
        let repo = self
            .workflow_repo
            .as_ref()
            .ok_or_else(|| EngineError::Checkpoint("no workflow_repo configured".into()))?;

        let workflow_ref = format!("{}@{}", ck.workflow_id, ck.workflow_version);
        let compiled = repo.get(&workflow_ref).await.ok_or_else(|| {
            EngineError::Checkpoint(format!("workflow `{workflow_ref}` not found in repo"))
        })?;

        info!(run = %run_id, workflow = %ck.workflow_id, "recovering run from checkpoint");

        let event_sink = self.broker.event_sink_ref();
        let mut instance = WorkflowInstance::new(
            ck.run_id.clone(),
            compiled,
            self.broker.clone(),
            event_sink,
            actors,
        );
        if let Some(ck_store) = &self.checkpoint_store {
            instance = instance.with_checkpoint_store(ck_store.clone());
        }
        if let Some(wf_repo) = &self.workflow_repo {
            instance = instance.with_workflow_repo(wf_repo.clone());
        }
        if let Some(resolver) = &self.context_resolver {
            instance = instance.with_context_resolver(resolver.clone());
        }

        // Restore mutable state from the checkpoint.
        instance.restore_from_checkpoint(&ck);

        // Only running checkpoints restart activities. A suspended run must
        // remain quiescent until an explicit Resume command arrives.
        if ck.status == RunStatus::Running {
            let active = ck.active_states.clone();
            for state_id in active {
                instance.start_activity_if_needed_pub(&state_id).await?;
            }
        }

        let restored_run_id = ck.run_id.clone();
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        self.runs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(restored_run_id.0.clone(), cmd_tx);

        let runs = self.runs.clone();
        let rid = restored_run_id.clone();
        tokio::spawn(async move {
            run_task(instance, cmd_rx).await;
            runs.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&rid.0);
        });

        Ok(restored_run_id)
    }

    /// Inspect the current state of a run.
    pub async fn inspect(&self, run_id: &RunId) -> Result<RunSnapshot, EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_cmd(run_id, RunCommand::Inspect { reply: reply_tx })?;
        reply_rx
            .await
            .map_err(|_| EngineError::RunNotFound(run_id.clone()))
    }

    /// Subscribe to the live event stream for a run.
    ///
    /// Returns `Some(stream)` when an `EventSource` was configured in
    /// `EngineAdapters`, `None` otherwise.
    ///
    /// Events are filtered to `run_id`; subscribers joining mid-run will not
    /// receive past events (broadcast semantics, not replay).
    pub fn subscribe(
        &self,
        run_id: &RunId,
    ) -> Option<Box<dyn futures::Stream<Item = RuntimeEvent> + Send + Unpin>> {
        self.event_source.as_ref().map(|src| src.subscribe(run_id))
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    fn send_cmd(&self, run_id: &RunId, cmd: RunCommand) -> Result<(), EngineError> {
        let runs = self
            .runs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let tx = runs
            .get(&run_id.0)
            .ok_or_else(|| EngineError::RunNotFound(run_id.clone()))?;
        tx.send(cmd)
            .map_err(|_| EngineError::RunNotFound(run_id.clone()))
    }
}

// ── Per-run task ──────────────────────────────────────────────────────────────

async fn run_task(mut instance: WorkflowInstance, mut cmd_rx: mpsc::UnboundedReceiver<RunCommand>) {
    loop {
        if instance.status == RunStatus::Suspended {
            let Some(cmd) = cmd_rx.recv().await else {
                return;
            };
            if handle_run_command(&mut instance, cmd).await {
                return;
            }
            continue;
        }

        // Control commands stay responsive even when internal events keep
        // producing more immediate work.
        if let Ok(cmd) = cmd_rx.try_recv()
            && handle_run_command(&mut instance, cmd).await
        {
            return;
        }

        if instance.has_immediate_work() {
            match instance.step().await {
                Ok(true) => {
                    while let Ok(cmd) = cmd_rx.try_recv() {
                        reply_error(cmd, EngineError::Cancelled);
                    }
                    return;
                }
                Ok(false) => continue,
                Err(e) => {
                    tracing::error!(run = %instance.run_id, error = %e, "run failed");
                    let failure_message = e.to_string();
                    if let Err(report_error) = instance.fail(failure_message.clone()).await {
                        tracing::error!(
                            run = %instance.run_id,
                            error = %report_error,
                            "failed to report terminal run failure"
                        );
                    }
                    while let Ok(cmd) = cmd_rx.try_recv() {
                        reply_error(cmd, EngineError::Activity(failure_message.clone()));
                    }
                    return;
                }
            }
        }

        tokio::select! {
            // Only wait for work here; processing happens outside `select!` so
            // command arrival cannot cancel a half-applied activity result.
            _ = instance.wait_for_work() => {}

            // Handle control commands.
            Some(cmd) = cmd_rx.recv() => {
                if handle_run_command(&mut instance, cmd).await {
                    return;
                }
            }
        }
    }
}

async fn handle_run_command(instance: &mut WorkflowInstance, cmd: RunCommand) -> bool {
    match cmd {
        RunCommand::Send {
            event_type,
            payload,
            reply,
        } => {
            instance.send(event_type, payload);
            let _ = reply.send(Ok(()));
        }
        RunCommand::SendBroadcast {
            event_type,
            payload,
            reply,
        } => {
            instance.send_broadcast(event_type, payload);
            let _ = reply.send(Ok(()));
        }
        RunCommand::Suspend { reply } => {
            let _ = reply.send(instance.suspend().await);
        }
        RunCommand::Resume { reply } => {
            let _ = reply.send(instance.resume().await);
        }
        RunCommand::Cancel { reply } => {
            let _ = reply.send(instance.cancel().await);
            return true;
        }
        RunCommand::Inspect { reply } => {
            let _ = reply.send(RunSnapshot {
                run_id: instance.run_id.clone(),
                status: instance.status.clone(),
                active_states: instance.active_states.clone(),
            });
        }
    }
    false
}

fn reply_error(cmd: RunCommand, error: EngineError) {
    match cmd {
        RunCommand::Send { reply, .. } => {
            let _ = reply.send(Err(error));
        }
        RunCommand::SendBroadcast { reply, .. } => {
            let _ = reply.send(Err(error));
        }
        RunCommand::Suspend { reply } => {
            let _ = reply.send(Err(error));
        }
        RunCommand::Resume { reply } => {
            let _ = reply.send(Err(error));
        }
        RunCommand::Cancel { reply } => {
            let _ = reply.send(Err(error));
        }
        RunCommand::Inspect { .. } => {} // drop
    }
}

// ── EngineError ───────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("run `{0}` not found")]
    RunNotFound(RunId),

    #[error("workflow validation failed: {0}")]
    ValidationFailed(String),

    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    #[error("broker error: {0}")]
    Broker(String),

    #[error("activity error: {0}")]
    Activity(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("run is already suspended")]
    AlreadySuspended,

    #[error("run is not suspended")]
    NotSuspended,

    #[error("run is cancelled")]
    Cancelled,
}
