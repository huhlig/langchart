//! Recording stage: finalises and records the context view for observability.
//!
//! This stage is always last in the chain. It computes the content hash and
//! emits a `ContextResolved` event so the view is observable and replayable.

use async_trait::async_trait;
use langchart_adapters::{
    context::{ContextAccumulator, ContextError, ContextResolverStage},
    event::{EventSink, RuntimeEvent, RuntimeEventPayload},
};
use langchart_model::id::{EventId, RunId, StateId};
use std::sync::Arc;
use ulid::Ulid;

/// Records the assembled context view to the event sink.
///
/// Must be the last stage; downstream stages that add items after this will
/// not be included in the recorded hash.
pub struct RecordingResolverStage {
    event_sink: Arc<dyn EventSink>,
    state_id: StateId,
}

impl RecordingResolverStage {
    pub fn new(event_sink: Arc<dyn EventSink>, state_id: StateId) -> Self {
        Self {
            event_sink,
            state_id,
        }
    }
}

#[async_trait]
impl ContextResolverStage for RecordingResolverStage {
    async fn resolve(
        &self,
        _policy: &langchart_model::policy::ContextPolicy,
        run_id: &RunId,
        ctx: &mut ContextAccumulator,
    ) -> Result<(), ContextError> {
        // Compute the hash that will end up in the ContextView.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        for item in &ctx.items {
            item.content.hash(&mut h);
        }
        let content_hash = format!("{:016x}", h.finish());

        let event = RuntimeEvent {
            event_id: EventId::new(Ulid::generate().to_string()),
            run_id: run_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            payload: RuntimeEventPayload::ContextResolved {
                state_id: self.state_id.clone(),
                token_count: ctx.token_count,
                content_hash,
            },
        };
        self.event_sink
            .append(event)
            .await
            .map_err(|e| ContextError::Stage {
                stage: "RecordingResolverStage",
                message: e.to_string(),
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use langchart_adapters::{
        context::{ContextAccumulator, ContextItem},
        event::{EventSink, EventSinkError, RuntimeEvent, RuntimeEventPayload},
    };
    use langchart_model::{id::RunId, policy::ContextPolicy};
    use std::sync::{Arc, Mutex};

    // ── Capturing event sink ──────────────────────────────────────────────────

    #[derive(Default, Clone)]
    struct CapturingSink(Arc<Mutex<Vec<RuntimeEvent>>>);

    #[async_trait]
    impl EventSink for CapturingSink {
        async fn append(&self, event: RuntimeEvent) -> Result<(), EventSinkError> {
            self.0.lock().unwrap().push(event);
            Ok(())
        }
    }

    fn run_id() -> RunId {
        RunId::new("rec-run")
    }

    fn stage_id() -> StateId {
        StateId::new("my-state")
    }

    fn push_item(ctx: &mut ContextAccumulator, content: &str, tokens: u32) {
        ctx.push(ContextItem {
            source: "t".into(),
            content: content.into(),
            tokens,
        });
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn emits_context_resolved_event() {
        let sink = CapturingSink::default();
        let stage = RecordingResolverStage::new(Arc::new(sink.clone()), stage_id());
        let mut ctx = ContextAccumulator::default();
        push_item(&mut ctx, "hello", 2);
        stage
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx)
            .await
            .unwrap();

        let events = sink.0.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0].payload,
            RuntimeEventPayload::ContextResolved { .. }
        ));
    }

    #[tokio::test]
    async fn event_carries_correct_token_count() {
        let sink = CapturingSink::default();
        let stage = RecordingResolverStage::new(Arc::new(sink.clone()), stage_id());
        let mut ctx = ContextAccumulator::default();
        push_item(&mut ctx, "a", 7);
        push_item(&mut ctx, "b", 3);
        stage
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx)
            .await
            .unwrap();

        let events = sink.0.lock().unwrap();
        if let RuntimeEventPayload::ContextResolved { token_count, .. } = &events[0].payload {
            assert_eq!(*token_count, 10);
        } else {
            panic!("unexpected payload");
        }
    }

    #[tokio::test]
    async fn content_hash_differs_for_different_content() {
        let sink1 = CapturingSink::default();
        let sink2 = CapturingSink::default();
        let stage1 = RecordingResolverStage::new(Arc::new(sink1.clone()), stage_id());
        let stage2 = RecordingResolverStage::new(Arc::new(sink2.clone()), stage_id());

        let mut ctx1 = ContextAccumulator::default();
        push_item(&mut ctx1, "content A", 1);
        let mut ctx2 = ContextAccumulator::default();
        push_item(&mut ctx2, "content B", 1);

        stage1
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx1)
            .await
            .unwrap();
        stage2
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx2)
            .await
            .unwrap();

        let hash1 = match &sink1.0.lock().unwrap()[0].payload {
            RuntimeEventPayload::ContextResolved { content_hash, .. } => content_hash.clone(),
            _ => panic!(),
        };
        let hash2 = match &sink2.0.lock().unwrap()[0].payload {
            RuntimeEventPayload::ContextResolved { content_hash, .. } => content_hash.clone(),
            _ => panic!(),
        };
        assert_ne!(hash1, hash2);
    }

    #[tokio::test]
    async fn event_run_id_matches() {
        let sink = CapturingSink::default();
        let stage = RecordingResolverStage::new(Arc::new(sink.clone()), stage_id());
        let mut ctx = ContextAccumulator::default();
        stage
            .resolve(&ContextPolicy::default(), &run_id(), &mut ctx)
            .await
            .unwrap();

        let events = sink.0.lock().unwrap();
        assert_eq!(events[0].run_id.0, "rec-run");
    }
}
