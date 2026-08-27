//! # langchart-adapters
//!
//! Adapter contracts and the policy-enforcing capability broker for external
//! integrations. Concrete external-system implementations live in separate crates.
//!
//! Every external system the engine touches is abstracted through one of
//! these traits. The runtime depends only on these traits; concrete adapter
//! crates depend on this crate and implement the traits.
//!
//! ## Adapter traits
//!
//! | Trait | Abstracts |
//! |---|---|
//! | [`llm::LlmAdapter`] | Language model completion |
//! | [`mcp::McpAdapter`] | MCP tool and resource calls |
//! | [`memory::MemoryAdapter`] | Long-term memory storage and retrieval |
//! | [`artifact::ArtifactStore`] | Versioned artifact reads, proposals, commits |
//! | [`checkpoint::CheckpointStore`] | Run snapshot persistence and recovery |
//! | [`event::EventSink`] | Observable runtime event appending |
//! | [`event::EventSource`] | Observable runtime event subscription |
//! | [`context::ContextResolverStage`] | One stage in the context resolver chain |

pub mod artifact;
pub mod broadcast;
pub mod broker;
pub mod checkpoint;
pub mod context;
pub mod event;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod secrets;
pub mod workflow_repository;
