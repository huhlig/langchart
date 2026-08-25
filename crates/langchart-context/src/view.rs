//! Helper for assembling a `ContextView` programmatically (used in tests
//! and by stages that build views without the full chain).

use langchart_adapters::context::{ContextAccumulator, ContextItem, ContextView};

pub struct ContextViewBuilder(ContextAccumulator);

impl ContextViewBuilder {
    pub fn new() -> Self {
        Self(ContextAccumulator::default())
    }

    pub fn push(
        mut self,
        source: impl Into<String>,
        content: impl Into<String>,
        tokens: u32,
    ) -> Self {
        self.0.push(ContextItem {
            source: source.into(),
            content: content.into(),
            tokens,
        });
        self
    }

    pub fn build(self) -> ContextView {
        self.0.finish()
    }
}

impl Default for ContextViewBuilder {
    fn default() -> Self {
        Self::new()
    }
}
