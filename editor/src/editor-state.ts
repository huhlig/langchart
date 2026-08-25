/**
 * EditorState — the central in-memory model for one editing session.
 *
 * Responsible for:
 * - holding the current workflow JSON string and parsed summary
 * - tracking which state is selected in the canvas
 * - tracking the live run snapshot (if a run is connected)
 * - emitting change events so panels can re-render independently
 */

import { wasm, WorkflowSummary, Diagnostic, TransitionEdge, ReachabilityResult } from "./wasm-loader.js";

// ── Events emitted by EditorState ─────────────────────────────────────────────

export type EditorEvent =
  | { type: "workflow-changed"; json: string }
  | { type: "state-selected"; stateId: string | null }
  | { type: "diagnostics-updated"; diagnostics: Diagnostic[] }
  | { type: "run-snapshot-updated"; snapshot: RunSnapshot | null }
  | { type: "run-event-appended"; entry: RunEventEntry };

// ── Run event log entry ───────────────────────────────────────────────────────

export type RunEventKind = "lifecycle" | "state" | "error" | "budget" | "other";

export interface RunEventEntry {
  timestamp: number; // Date.now()
  kind: RunEventKind;
  label: string;
  detail?: string;
}

// ── Live run snapshot (from the host application's event stream) ──────────────

export interface RunSnapshot {
  runId: string;
  status: "running" | "suspended" | "completed" | "failed" | "cancelled";
  activeStates: string[];
  eventQueueDepth: number;
  activities: string[];
}

// ── Editor state ──────────────────────────────────────────────────────────────

export class EditorState {
  private _json: string = "";
  private _selectedStateId: string | null = null;
  private _runSnapshot: RunSnapshot | null = null;
  private _runEvents: RunEventEntry[] = [];
  private _listeners: Array<(event: EditorEvent) => void> = [];

  // Derived (recomputed on each workflow change)
  summary: WorkflowSummary | null = null;
  diagnostics: Diagnostic[] = [];
  transitions: TransitionEdge[] = [];
  reachability: ReachabilityResult = { reachable: [], unreachable: [] };

  // ── Subscriptions ──────────────────────────────────────────────────────────

  on(listener: (event: EditorEvent) => void): () => void {
    this._listeners.push(listener);
    return () => {
      this._listeners = this._listeners.filter((l) => l !== listener);
    };
  }

  private emit(event: EditorEvent): void {
    for (const l of this._listeners) {
      try { l(event); } catch (_) { /* isolate panel errors */ }
    }
  }

  // ── Workflow document ──────────────────────────────────────────────────────

  get json(): string {
    return this._json;
  }

  loadJson(json: string): void {
    this._json = json;
    this._recompute();
    this.emit({ type: "workflow-changed", json });
    this.emit({ type: "diagnostics-updated", diagnostics: this.diagnostics });
  }

  private _recompute(): void {
    if (!this._json) return;
    try {
      this.diagnostics = wasm.validateWorkflow(this._json);
      this.summary = wasm.workflowSummary(this._json);
      this.transitions = wasm.listTransitions(this._json);
      this.reachability = wasm.reachabilityAnalysis(this._json);
    } catch (e) {
      this.summary = null;
      this.transitions = [];
      this.reachability = { reachable: [], unreachable: [] };
      this.diagnostics = [
        {
          code: "E000",
          severity: "error",
          message: String(e),
          location: "workflow",
        },
      ];
    }
  }

  // ── State selection ────────────────────────────────────────────────────────

  get selectedStateId(): string | null {
    return this._selectedStateId;
  }

  selectState(id: string | null): void {
    if (this._selectedStateId === id) return;
    this._selectedStateId = id;
    this.emit({ type: "state-selected", stateId: id });
  }

  // ── Live run ───────────────────────────────────────────────────────────────

  get runSnapshot(): RunSnapshot | null {
    return this._runSnapshot;
  }

  updateRunSnapshot(snapshot: RunSnapshot | null): void {
    this._runSnapshot = snapshot;
    this.emit({ type: "run-snapshot-updated", snapshot });
  }

  // ── Run event log ──────────────────────────────────────────────────────────

  get runEvents(): readonly RunEventEntry[] {
    return this._runEvents;
  }

  appendRunEvent(entry: RunEventEntry): void {
    this._runEvents.push(entry);
    this.emit({ type: "run-event-appended", entry });
  }

  clearRunEvents(): void {
    this._runEvents = [];
  }

  isStateActive(stateId: string): boolean {
    return this._runSnapshot?.activeStates.includes(stateId) ?? false;
  }
}
