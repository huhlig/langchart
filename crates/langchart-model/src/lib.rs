//! # langchart-model
//!
//! Core statechart types, schema, validation, and CEL guard compilation.
//!
//! **WASM-compatible.** This crate MUST NOT use `std::fs`, `std::thread`,
//! `std::net`, or any async runtime. All data is passed in; this crate is
//! pure computation.
//!
//! ## Modules
//!
//! - [`types`] — canonical statechart type definitions
//! - [`workflow`] — workflow document and identity types
//! - [`state`] — state, state type, and lifecycle types
//! - [`transition`] — transition, event envelope, and guard types
//! - [`policy`] — capability, context, and model policy types
//! - [`validation`] — structural and semantic validation
//! - [`guard`] — CEL guard compilation and evaluation
//! - [`schema`] — JSON Schema versioning and migration support
//! - [`id`] — stable identity types (WorkflowId, StateId, RunId, etc.)
//! - [`error`] — model-layer error types

pub mod error;
pub mod guard;
pub mod id;
pub mod policy;
pub mod schema;
pub mod state;
pub mod transition;
pub mod validation;
pub mod workflow;
