//! # langchart-context
//!
//! Composable context resolver chain and built-in stages.

pub mod chain;
pub mod stages;
pub mod view;

pub use chain::ContextResolverChain;
pub use view::ContextViewBuilder;
