//! Durable timer registry.
//!
//! Timers are persisted with the run snapshot so that a restart does not
//! silently discard pending timeouts. Each timer is identified by a stable
//! `TimerId` and fires by sending a typed event into the workflow's event queue.
//!
//! The initial implementation is in-process (tokio). For distributed execution
//! (Phase 6), the registry would delegate to a durable scheduler.

use langchart_model::id::{RunId, StateId};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{debug, warn};
use ulid::Ulid;

// ── Timer ID ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct TimerId(pub String);

impl TimerId {
    pub fn new() -> Self {
        Self(Ulid::generate().to_string())
    }
}

impl Default for TimerId {
    fn default() -> Self {
        Self::new()
    }
}

// ── Timer event ───────────────────────────────────────────────────────────────

/// Sent on the event channel when a timer fires.
#[derive(Debug, Clone)]
pub struct TimerFired {
    pub timer_id: TimerId,
    pub run_id: RunId,
    pub state_id: StateId,
    /// The event type to inject into the workflow event queue.
    pub event_type: String,
}

// ── Timer entry (persisted) ───────────────────────────────────────────────────

/// The persisted representation of a pending timer.
///
/// `remaining_ms` is the number of milliseconds remaining from the moment the
/// checkpoint was taken to when the timer should fire. On restore it is used
/// directly as the new delay (saturating at zero for already-elapsed timers).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimerEntry {
    pub id: TimerId,
    pub run_id: RunId,
    pub state_id: StateId,
    pub event_type: String,
    /// Remaining delay in milliseconds at the time the checkpoint was taken.
    pub remaining_ms: u64,
}

// ── Registry ──────────────────────────────────────────────────────────────────

/// Manages in-process timers for a workflow run.
///
/// On checkpoint save, `active_entries()` returns the persisted timer state.
/// On checkpoint restore, `restore()` re-arms all timers with their remaining
/// delay using the stored `remaining_ms`.
pub struct TimerRegistry {
    run_id: RunId,
    tx: mpsc::UnboundedSender<TimerFired>,
    /// Active timer handles, keyed by `TimerId`.
    handles: HashMap<TimerId, JoinHandle<()>>,
    /// Tokio instants at which each timer should fire, kept in sync with handles.
    fire_at: HashMap<TimerId, Instant>,
    /// Persisted metadata (event type, state, etc.) kept in sync with handles.
    entries: HashMap<TimerId, TimerEntry>,
}

impl TimerRegistry {
    pub fn new(run_id: RunId, tx: mpsc::UnboundedSender<TimerFired>) -> Self {
        Self {
            run_id,
            tx,
            handles: HashMap::new(),
            fire_at: HashMap::new(),
            entries: HashMap::new(),
        }
    }

    /// Schedule a new timer. Returns the `TimerId` for cancellation.
    pub fn schedule(
        &mut self,
        state_id: StateId,
        event_type: impl Into<String>,
        delay: Duration,
    ) -> TimerId {
        let id = TimerId::new();
        let event_type = event_type.into();
        let fire_instant = Instant::now() + delay;

        // Store a placeholder entry; remaining_ms is filled by active_entries().
        let entry = TimerEntry {
            id: id.clone(),
            run_id: self.run_id.clone(),
            state_id: state_id.clone(),
            event_type: event_type.clone(),
            remaining_ms: delay.as_millis() as u64,
        };
        self.entries.insert(id.clone(), entry);
        self.fire_at.insert(id.clone(), fire_instant);

        let tx = self.tx.clone();
        let fired = TimerFired {
            timer_id: id.clone(),
            run_id: self.run_id.clone(),
            state_id,
            event_type,
        };
        let handle = tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            if tx.send(fired).is_err() {
                warn!("timer fired but receiver was dropped");
            }
        });
        self.handles.insert(id.clone(), handle);
        debug!(timer = %id.0, delay_ms = delay.as_millis(), "timer scheduled");
        id
    }

    /// Cancel a pending timer. No-op if already fired or unknown.
    pub fn cancel(&mut self, id: &TimerId) {
        if let Some(handle) = self.handles.remove(id) {
            handle.abort();
        }
        self.fire_at.remove(id);
        self.entries.remove(id);
        debug!(timer = %id.0, "timer cancelled");
    }

    /// Returns all pending timer entries for checkpoint persistence.
    ///
    /// Each entry's `remaining_ms` reflects the time left from *now* (using
    /// Tokio's virtual clock) so the value is valid under `start_paused` tests.
    pub fn active_entries(&self) -> Vec<TimerEntry> {
        let now = Instant::now();
        self.entries
            .values()
            .map(|e| {
                let remaining_ms = self
                    .fire_at
                    .get(&e.id)
                    .and_then(|&fi| fi.checked_duration_since(now))
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                TimerEntry {
                    remaining_ms,
                    ..e.clone()
                }
            })
            .collect()
    }

    /// Restore timers from checkpoint entries.
    ///
    /// Each entry's `remaining_ms` is used directly as the new sleep delay.
    /// Timers that were already past their fire time (remaining_ms == 0) fire
    /// on the next scheduler tick.
    pub fn restore(&mut self, entries: Vec<TimerEntry>) {
        for entry in entries {
            let delay = Duration::from_millis(entry.remaining_ms);
            // Re-use the existing timer ID so checkpoint references remain valid.
            let id = entry.id.clone();
            let fire_instant = Instant::now() + delay;
            let tx = self.tx.clone();
            let fired = TimerFired {
                timer_id: id.clone(),
                run_id: entry.run_id.clone(),
                state_id: entry.state_id.clone(),
                event_type: entry.event_type.clone(),
            };
            let handle = tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                if tx.send(fired).is_err() {
                    warn!("restored timer fired but receiver was dropped");
                }
            });
            self.handles.insert(id.clone(), handle);
            self.fire_at.insert(id.clone(), fire_instant);
            self.entries.insert(id, entry);
        }
    }
}

impl Drop for TimerRegistry {
    fn drop(&mut self) {
        for (_, handle) in self.handles.drain() {
            handle.abort();
        }
        self.fire_at.clear();
    }
}
