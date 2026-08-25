//! Observability bridge: translates docuvault `VaultEvent`s into langchart `RuntimeEvent`s.
//!
//! ## Event mapping (design doc §5)
//!
//! | `docuvault` `VaultEvent`                       | `langchart` `RuntimeEventPayload`            |
//! |------------------------------------------------|----------------------------------------------|
//! | `VaultCommitPublished { commit_id }`           | `ProposalCommitted` (best fit; see notes)    |
//! | `SessionCommitCreated { session_id, commit_id}`| `ProposalCreated` (session boundary)         |
//! | `ProposalApplied { proposal_id, commit_id }`   | `ProposalCommitted`                          |
//! | `ProposalConflicted { proposal_id }`           | `ProposalConflicted`                         |
//! | `MergeConflicted { merge_id }`                 | `ProposalConflicted` (merge variant)         |
//! | `BufferOpened / BufferClosed / BufferPushed …` | **Dropped** — non-durable, §29.1             |
//!
//! ## Notes on mapping
//!
//! `langchart_adapters::event::RuntimeEventPayload` does not have a vault-specific
//! variant. The bridge uses the closest semantic match from the existing payload
//! enum: proposals are the closest analog to vault mutations, and commit events
//! map to their commit/conflict counterparts.
//!
//! ## Threading
//!
//! `VaultEventBridge` wraps a `tokio::sync::broadcast::Receiver<EventEnvelope>`
//! (the emitter side is held by the vault service layer in `collaborite-core`).
//! The `subscribe()` method returns a stream that translates each received
//! `EventEnvelope` to a `RuntimeEvent`.
//!
//! `EventSource` is an outward subscription interface; it does not enqueue
//! workflow inputs. Hosts that want vault events to select transitions must
//! forward the corresponding event name and payload through `RuntimeEngine::send`.

use std::sync::Arc;

use futures::Stream;
use langchart_adapters::event::{EventSource, RuntimeEvent, RuntimeEventPayload};
use langchart_model::id::{EventId, RunId};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use docuvault::model::event::{EventEnvelope, VaultEvent};

// ── VaultEventBridge ──────────────────────────────────────────────────────────

/// Translates docuvault `VaultEvent`s into langchart `RuntimeEvent`s.
///
/// Construct one per vault when translated vault events should be observable
/// through the runtime's `EventSource` API. The bridge filters and translates
/// only durable event variants; buffer events are explicitly dropped.
#[derive(Clone)]
pub struct VaultEventBridge {
    /// The broadcast sender; held to allow new `subscribe()` calls without
    /// requiring a mutable reference.  Each `subscribe()` call clones the Sender
    /// to obtain a new Receiver.
    sender: Arc<broadcast::Sender<EventEnvelope>>,
}

impl VaultEventBridge {
    /// Create a new bridge.
    ///
    /// `sender` is the broadcast channel that the vault service (in
    /// `collaborite-core`'s `WatchEmitter` impl) emits `EventEnvelope`s on.
    pub fn new(sender: Arc<broadcast::Sender<EventEnvelope>>) -> Self {
        Self { sender }
    }

    /// Translate a docuvault `EventEnvelope` to a langchart `RuntimeEventPayload`.
    ///
    /// Returns `None` for events that MUST be dropped (buffer events and
    /// events without a meaningful langchart analog).
    fn translate(envelope: &EventEnvelope) -> Option<RuntimeEventPayload> {
        match &envelope.payload {
            // ── Durable commit events ─────────────────────────────────────────
            VaultEvent::VaultCommitPublished { commit_id } => {
                // A published commit means some set of mutations are now on `main`.
                // We surface this as ProposalCommitted with the commit_id as the
                // new_version so workflow states can trigger on vault publish.
                Some(RuntimeEventPayload::ProposalCommitted {
                    artifact_id: String::new(), // vault-level event; no single artifact
                    proposal_id: String::new(),
                    new_version: format!("commit:{commit_id}"),
                })
            }

            VaultEvent::SessionCommitCreated {
                session_id,
                commit_id,
            } => {
                // A session durable boundary. Surfaces as ProposalCreated so workflows
                // can trigger on "a session boundary was packed."
                Some(RuntimeEventPayload::ProposalCreated {
                    artifact_id: format!("session:{session_id}"),
                    proposal_id: format!("commit:{commit_id}"),
                })
            }

            VaultEvent::ProposalApplied {
                proposal_id,
                commit_id,
            } => Some(RuntimeEventPayload::ProposalCommitted {
                artifact_id: String::new(),
                proposal_id: proposal_id.to_string(),
                new_version: format!("commit:{commit_id}"),
            }),

            VaultEvent::ProposalConflicted { proposal_id } => {
                Some(RuntimeEventPayload::ProposalConflicted {
                    artifact_id: String::new(),
                    proposal_id: proposal_id.to_string(),
                })
            }

            VaultEvent::MergeConflicted { merge_id } => {
                // No langchart merge concept; map to ProposalConflicted with merge context.
                Some(RuntimeEventPayload::ProposalConflicted {
                    artifact_id: String::new(),
                    proposal_id: format!("merge:{merge_id}"),
                })
            }

            // ── Buffer events (§29.1) — explicitly dropped ────────────────────
            VaultEvent::BufferOpened { .. }
            | VaultEvent::BufferClosed { .. }
            | VaultEvent::BufferPushed { .. }
            | VaultEvent::BufferProposalDrafted { .. }
            | VaultEvent::BufferConflictDetected { .. }
            | VaultEvent::BufferFlushed { .. } => {
                // Non-durable. Design doc §5 explicitly says "Not forwarded".
                None
            }

            // ── All other vault events — not bridged in this release ──────────
            _ => None,
        }
    }
}

impl EventSource for VaultEventBridge {
    fn subscribe(&self, run_id: &RunId) -> Box<dyn Stream<Item = RuntimeEvent> + Send + Unpin> {
        let receiver = self.sender.subscribe();
        let stream_run_id = run_id.clone();

        let stream = BroadcastStream::new(receiver)
            // Discard channel lag errors — they indicate the consumer was too slow;
            // the missed events are gone (broadcast channel is not persistent).
            .filter_map(|result| result.ok())
            .filter_map(move |envelope| {
                let payload = Self::translate(&envelope)?;
                let timestamp = envelope.occurred_at.to_string();
                Some(RuntimeEvent {
                    event_id: EventId::new(envelope.event_id.clone()),
                    run_id: stream_run_id.clone(),
                    timestamp,
                    payload,
                })
            });

        Box::new(stream)
    }
}
