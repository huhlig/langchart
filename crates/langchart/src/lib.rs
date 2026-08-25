//! # langchart
//!
//! **Agentic statechart engine** — governed, durable agentic workflows built
//! on hierarchical statechart semantics.
//!
//! This crate is the public API surface. Add `langchart` to your `Cargo.toml`
//! and implement the adapter traits to connect your LLM, MCP servers, memory
//! system, artifact store, and checkpoint store.
//!
//! ## Quick orientation
//!
//! ```text
//! langchart
//!   ├── model     — workflow types, validation, CEL guards (re-exported)
//!   ├── adapters  — integration trait definitions (re-exported)
//!   ├── runtime   — execution engine and capability broker (re-exported)
//!   └── context   — context resolver chain (re-exported)
//! ```
//!
//! Enable the optional `docuvault` feature to expose the Docuvault adapter
//! bridge as [`docuvault`].
//!
//! ## Getting started
//!
//! 1. Implement the adapter traits for your environment:
//!    [`adapters::llm::LlmAdapter`], [`adapters::mcp::McpAdapter`],
//!    [`adapters::memory::MemoryAdapter`], [`adapters::artifact::ArtifactStore`],
//!    [`adapters::checkpoint::CheckpointStore`], [`adapters::event::EventSink`].
//!
//! 2. Author a workflow in JSON or YAML and validate it:
//!    ```text
//!    let doc = langchart::model::workflow::WorkflowDocument::from_yaml(src)?;
//!    let diagnostics = langchart::model::validation::validate(&doc);
//!    ```
//!
//! 3. Compile and start a run:
//!    ```text
//!    let compiled = langchart::model::validation::compile(doc)?;
//!    let engine = langchart::runtime::engine::RuntimeEngine::new(adapters);
//!    let run_id = engine.start(compiled, input).await?;
//!    ```

pub use langchart_adapters as adapters;
pub use langchart_context as context;
pub use langchart_model as model;
pub use langchart_runtime as runtime;

#[cfg(feature = "docuvault")]
pub use langchart_docuvault as docuvault;
