//! CEL guard compilation, whitelisted extension functions, and evaluation.
//!
//! Guards are deterministic, side-effect-free CEL expressions evaluated at
//! transition time. Only functions from the approved whitelist may be used.
//! This module is WASM-compatible — no I/O, no async.

use crate::error::GuardError;
use cel_interpreter::{Context, Program, objects::Value};
use std::collections::HashSet;
use std::sync::Arc;

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

/// Pure functions supplied by `cel-interpreter`'s default context. These are
/// part of the CEL execution environment rather than Langchart extensions.
const CEL_BUILTINS: &[&str] = &[
    "contains",
    "size",
    "has",
    "map",
    "filter",
    "all",
    "max",
    "min",
    "startsWith",
    "endsWith",
    "string",
    "bytes",
    "double",
    "exists",
    "exists_one",
    "int",
    "uint",
    "matches",
    "duration",
    "timestamp",
    "getFullYear",
    "getMonth",
    "getDayOfYear",
    "getDayOfMonth",
    "getDate",
    "getDayOfWeek",
    "getHours",
    "getMinutes",
    "getSeconds",
    "getMilliseconds",
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
        let parsed = cel_parser::parse(expr).map_err(|e| GuardError::Parse(e.to_string()))?;
        let builtins: HashSet<_> = CEL_BUILTINS.iter().copied().collect();
        let extensions = approved_extension_set();
        if let Some(name) = parsed
            .references()
            .functions()
            .into_iter()
            .find(|name| !builtins.contains(name) && !extensions.contains(name))
        {
            return Err(GuardError::DisallowedExtension {
                name: name.to_owned(),
            });
        }

        let program = Program::compile(expr).map_err(|e| GuardError::Parse(e.to_string()))?;

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

/// Build the deterministic CEL context used for guard evaluation.
pub fn evaluation_context() -> Context<'static> {
    let mut context = Context::default();
    context.add_function("version_gte", version_gte);
    context.add_function("version_lte", version_lte);
    context.add_function("contains_all", contains_all);
    context.add_function("contains_any", contains_any);
    context.add_function("is_empty", is_empty);
    context
}

fn version_gte(left: Arc<String>, right: Arc<String>) -> bool {
    compare_versions(&left, &right).is_some_and(|ordering| !ordering.is_lt())
}

fn version_lte(left: Arc<String>, right: Arc<String>) -> bool {
    compare_versions(&left, &right).is_some_and(|ordering| !ordering.is_gt())
}

fn compare_versions(left: &str, right: &str) -> Option<std::cmp::Ordering> {
    Some(
        semver::Version::parse(left)
            .ok()?
            .cmp(&semver::Version::parse(right).ok()?),
    )
}

fn contains_all(haystack: Value, needles: Value) -> bool {
    match (haystack, needles) {
        (Value::List(haystack), Value::List(needles)) => {
            needles.iter().all(|needle| haystack.contains(needle))
        }
        _ => false,
    }
}

fn contains_any(haystack: Value, needles: Value) -> bool {
    match (haystack, needles) {
        (Value::List(haystack), Value::List(needles)) => {
            needles.iter().any(|needle| haystack.contains(needle))
        }
        _ => false,
    }
}

fn is_empty(value: Value) -> bool {
    match value {
        Value::List(value) => value.is_empty(),
        Value::Map(value) => value.map.is_empty(),
        Value::String(value) => value.is_empty(),
        Value::Bytes(value) => value.is_empty(),
        _ => false,
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

    #[test]
    fn unknown_extension_is_rejected_at_compile_time() {
        let result = CompiledGuard::compile("read_file('/secret') == true");
        assert!(matches!(
            result,
            Err(GuardError::DisallowedExtension { name }) if name == "read_file"
        ));
    }

    #[test]
    fn builtins_and_approved_extensions_are_executable() {
        let context = evaluation_context();
        assert!(
            CompiledGuard::compile("size([1, 2]) == 2")
                .unwrap()
                .evaluate(&context)
                .unwrap()
        );
        assert!(
            CompiledGuard::compile("version_gte('1.2.0', '1.1.9') && contains_all([1, 2], [2])")
                .unwrap()
                .evaluate(&context)
                .unwrap()
        );
    }
}
