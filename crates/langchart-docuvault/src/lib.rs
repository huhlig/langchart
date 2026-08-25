//! # langchart-docuvault
//!
//! Implements `langchart` adapter traits on top of `docuvault` APIs, bridging
//! the two independent libraries so that agentic workflows can read, propose
//! changes to, and commit vault documents without `langchart-runtime` having
//! any direct dependency on `docuvault`.
//!
//! ## Modules
//!
//! | Module | Provides |
//! |---|---|
//! | [`artifact`] | `ArtifactStore` → `VaultArtifactStore` |
//! | [`context`] | `ContextResolverStage` → `VaultSearchStage` |
//! | [`event_bridge`] | `EventSource` → `VaultEventBridge` |
//! | [`memory`] | `MemoryAdapter` → `VaultMemoryAdapter` (feature `vault-memory`) |
//! | [`ids`] | `VaultEntityRef`, `VaultVersionRef`, `VaultRef` |
//! | [`error`] | `DocuvaultAdapterError` |
//!
//! ## Dependency constraint
//!
//! This crate MUST NOT depend on `langchart-runtime`. It sits at the adapter
//! layer: `langchart-model ← langchart-adapters ← langchart-docuvault`.

pub mod artifact;
pub mod context;
pub mod error;
pub mod event_bridge;
pub mod ids;

#[cfg(feature = "vault-memory")]
pub mod memory;

pub use artifact::VaultArtifactStore;
pub use context::VaultSearchStage;
pub use error::DocuvaultAdapterError;
pub use event_bridge::VaultEventBridge;
pub use ids::VaultRef;
