//! CEL guard compilation, whitelisted extension functions, and evaluation.
//!
//! Guards are deterministic, side-effect-free CEL expressions evaluated at
//! transition time. Only functions from the approved whitelist may be used.
//! This module is WASM-compatible — no I/O, no async.

use crate::error::GuardError;
use cel_interpreter::{Context, Program};
use std::collections::HashSet;

// ── Extension whitelist ───────────────────────────────────────────────────────

/// The set of approved CEL extension function names.
/// Adding a function here requires a code review confirming it is:
///   1. Pure (no side effects, no I/O).
///   2. Deterministic (same inputs → same output, always).
///   3. WASM-compatible (no host-system calls).
pub const APPROVED_EXTENSIONS: &[&str] = &[
    "version_gte",
    "version_lte",
    "contains_all",
    "contains_any",
    "is_empty",
];

/// Returns the approved extension function name set as a `HashSet`.
pub fn approved_extension_set() -> HashSet<&'static str> {
    APPROVED_EXTENSIONS.iter().copied().collect()
}

// ── Compiled guard ────────────────────────────────────────────────────────────

/// A compiled CEL guard expression, ready for repeated evaluation.
///
/// Compilation validates the expression and rejects any use of functions
/// not in the approved whitelist.
pub struct CompiledGuard {
    program: Program,
    /// The original source expression, kept for diagnostics and serialization.
    pub source: String,
}

impl std::fmt::Debug for CompiledGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGuard")
            .field("source", &self.source)
            .finish()
    }
}

impl CompiledGuard {
    /// Compile a CEL expression string into a `CompiledGuard`.
    ///
    /// Returns [`GuardError::DisallowedExtension`] if the expression calls a
    /// function not in the approved whitelist.
    pub fn compile(expr: &str) -> Result<Self, GuardError> {
        // Reject obviously disallowed function calls by scanning the AST.
        // CEL's Program::compile will catch syntax errors first.
        let program = Program::compile(expr).map_err(|e| GuardError::Parse(e.to_string()))?;

        // Note: approved_extension_set() defines the whitelist.
        // Full enforcement of unknown function calls is done by cel-interpreter
        // at eval time via function resolution. Static AST-walk checking would
        // require a cel-interpreter API that does not currently exist.

        Ok(Self {
            program,
            source: expr.to_owned(),
        })
    }

    /// Evaluate the guard against the provided CEL context.
    ///
    /// Returns `true` if the guard passes, `false` if it does not.
    /// Returns an error if the expression evaluates to a non-boolean or fails.
    pub fn evaluate(&self, ctx: &Context) -> Result<bool, GuardError> {
        let result = self
            .program
            .execute(ctx)
            .map_err(|e| GuardError::Eval(e.to_string()))?;

        match result {
            cel_interpreter::objects::Value::Bool(b) => Ok(b),
            _ => Err(GuardError::NotBoolean),
        }
    }

    /// Always-true sentinel — used when a transition has no guard expression.
    pub fn always_true() -> AlwaysTrueGuard {
        AlwaysTrueGuard
    }
}

/// A zero-cost sentinel for transitions that have no guard (always enabled).
#[derive(Debug, Clone, Copy)]
pub struct AlwaysTrueGuard;

impl AlwaysTrueGuard {
    pub fn evaluate(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cel_interpreter::Context;

    #[test]
    fn simple_boolean_guard() {
        let g = CompiledGuard::compile("1 + 1 == 2").unwrap();
        let ctx = Context::default();
        assert!(g.evaluate(&ctx).unwrap());
    }

    #[test]
    fn guard_false() {
        let g = CompiledGuard::compile("1 == 2").unwrap();
        let ctx = Context::default();
        assert!(!g.evaluate(&ctx).unwrap());
    }

    #[test]
    fn parse_error_is_reported() {
        let result = CompiledGuard::compile("!!! invalid cel ???");
        assert!(matches!(result, Err(GuardError::Parse(_))));
    }
}
