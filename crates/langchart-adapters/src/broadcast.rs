//! `BroadcastEventSink` — in-memory fanout for runtime events.
//!
//! Implements both [`EventSink`] and [`EventSource`]:
//!
//! - `append` broadcasts each event to all live receivers.
//! - `subscribe` returns a filtered [`Stream`] that yields only events
//!   belonging to the requested `run_id`.
//!
//! Intended for in-process observability (tests, dashboards, simulation).
//! For distributed / persistent event storage use a dedicated adapter.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use langchart_adapters::broadcast::BroadcastEventSink;
//!
//! let bus = Arc::new(BroadcastEventSink::new(128));
//! // Use `bus` as an EventSink when building the engine, and `bus` as an
//! // EventSource to subscribe to events from a specific run.
//! ```

use crate::event::{EventSink, EventSinkError, EventSource, RuntimeEvent};
use async_trait::async_trait;
use futures::{Stream, stream};
use langchart_model::id::RunId;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

/// A broadcast-channel backed event bus.
///
/// Clone the `Arc<BroadcastEventSink>` freely — all clones share the same
/// underlying channel so subscribers created from any clone receive all events.
pub struct BroadcastEventSink {
    tx: broadcast::Sender<RuntimeEvent>,
}

impl BroadcastEventSink {
    /// Create a new bus with the given channel capacity.
    ///
    /// `capacity` is the number of events that can be buffered before the
    /// oldest is dropped for lagging receivers. 128–1024 is typical.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Return the number of active receivers (subscribers).
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

#[async_trait]
impl EventSink for BroadcastEventSink {
    async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
        // `send` returns Err only when there are no receivers; that is not an
        // error — events are just dropped when nobody is listening.
        let _ = self.tx.send(event);
        Ok(())
    }
}

impl EventSource for BroadcastEventSink {
    fn subscribe(&self, run_id: &RunId) -> Box<dyn Stream<Item = RuntimeEvent> + Send + Unpin> {
        let rx = self.tx.subscribe();
        let run_id = run_id.clone();

        // BroadcastStream converts the tokio Receiver into a Stream.
        // We filter to the requested run_id and discard lagged errors.
        let filtered = stream::unfold(BroadcastStream::new(rx), move |mut bstream| {
            let rid = run_id.clone();
            async move {
                loop {
                    match futures::StreamExt::next(&mut bstream).await {
                        None => return None,
                        Some(Err(_lagged)) => continue, // skip lagged errors
                        Some(Ok(event)) if event.run_id == rid => {
                            return Some((event, bstream));
                        }
                        Some(Ok(_)) => continue, // different run_id
                    }
                }
            }
        });

        Box::new(Box::pin(filtered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::RuntimeEventPayload;
    use futures::StreamExt;
    use langchart_model::id::EventId;
    use std::sync::Arc;

    fn make_event(run_id: &str, payload: RuntimeEventPayload) -> RuntimeEvent {
        RuntimeEvent {
            event_id: EventId::new("e1"),
            run_id: RunId::new(run_id),
            timestamp: "2024-01-01T00:00:00Z".into(),
            payload,
        }
    }

    /// A subscriber receives only events for its run_id, not for other runs.
    #[tokio::test]
    async fn subscribe_filters_by_run_id() {
        let bus = Arc::new(BroadcastEventSink::new(64));
        let mut sub = bus.subscribe(&RunId::new("run-a"));

        bus.append(make_event("run-b", RuntimeEventPayload::RunStarted))
            .await
            .unwrap();
        bus.append(make_event("run-a", RuntimeEventPayload::RunStarted))
            .await
            .unwrap();
        bus.append(make_event("run-a", RuntimeEventPayload::RunCompleted))
            .await
            .unwrap();

        let ev1 = sub.next().await.expect("first event");
        assert_eq!(ev1.run_id.0, "run-a");
        assert!(matches!(ev1.payload, RuntimeEventPayload::RunStarted));

        let ev2 = sub.next().await.expect("second event");
        assert!(matches!(ev2.payload, RuntimeEventPayload::RunCompleted));
    }

    /// Two independent subscribers both receive the same event.
    #[tokio::test]
    async fn multi_subscriber_fanout() {
        let bus = Arc::new(BroadcastEventSink::new(64));
        let mut sub1 = bus.subscribe(&RunId::new("run-x"));
        let mut sub2 = bus.subscribe(&RunId::new("run-x"));

        bus.append(make_event("run-x", RuntimeEventPayload::RunStarted))
            .await
            .unwrap();

        let e1 = sub1.next().await.expect("sub1 event");
        let e2 = sub2.next().await.expect("sub2 event");
        assert!(matches!(e1.payload, RuntimeEventPayload::RunStarted));
        assert!(matches!(e2.payload, RuntimeEventPayload::RunStarted));
    }

    /// A subscriber that joins after events have been sent does not receive
    /// the missed events (broadcast semantics, not replay).
    #[tokio::test]
    async fn subscriber_joining_mid_run_misses_past_events() {
        let bus = Arc::new(BroadcastEventSink::new(64));

        // Send one event before anyone subscribes.
        bus.append(make_event("run-y", RuntimeEventPayload::RunStarted))
            .await
            .unwrap();

        // Subscribe after the event.
        let mut sub = bus.subscribe(&RunId::new("run-y"));

        // Send a second event that the subscriber should see.
        bus.append(make_event("run-y", RuntimeEventPayload::RunCompleted))
            .await
            .unwrap();

        let ev = sub.next().await.expect("subscriber must see RunCompleted");
        assert!(matches!(ev.payload, RuntimeEventPayload::RunCompleted));
    }
}
