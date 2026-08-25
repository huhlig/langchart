/**
 * @langchart/editor — library entry point.
 *
 * Re-exports the public API for embedding the editor in host applications.
 * Import from this module when using the editor as a package.
 *
 * Usage:
 *   import { Editor, initWasm, wasm } from "@langchart/editor";
 */

// ── Core ──────────────────────────────────────────────────────────────────────
export { Editor } from "./editor.js";
export type { EditorConfig, EditorMode } from "./editor.js";

export { EditorState } from "./editor-state.js";
export type { EditorEvent, RunSnapshot, RunEventEntry, RunEventKind } from "./editor-state.js";

// ── WASM ──────────────────────────────────────────────────────────────────────
export { initWasm, wasm } from "./wasm-loader.js";
export type {
  Diagnostic,
  CompileResult,
  StateInspection,
  TransitionInfo,
  WorkflowSummary,
  StateCounts,
  TransitionEdge,
  ReachabilityResult,
  GuardError,
  SimulationStepRecord,
  SimulationResult,
  SimulationInput,
} from "./wasm-loader.js";

// ── Panels ────────────────────────────────────────────────────────────────────
export { Panel } from "./panels/panel-base.js";
export { WorkflowCanvas } from "./panels/workflow-canvas.js";
export { StateInspector } from "./panels/state-inspector.js";
export { ProblemsPanel } from "./panels/problems-panel.js";
export { RunInspector } from "./panels/run-inspector.js";
export {
  ContextInspector,
  CapabilityInspector,
  ArtifactReview,
  TraceTimeline,
  SimulationPanel,
} from "./panels/secondary-panels.js";

// ── Utilities ─────────────────────────────────────────────────────────────────
export { escapeHtml } from "./html.js";
