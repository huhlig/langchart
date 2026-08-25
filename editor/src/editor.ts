/**
 * Editor — configurable orchestrator for the langchart visual statechart editor.
 *
 * Wraps EditorState, panel mounting, and file I/O into a single reusable class.
 * Host applications create an Editor instance, mount it into a container element,
 * and use its API to load/export workflows. The editor does NOT own the toolbar —
 * the host app wires UI controls to Editor methods.
 *
 * Usage:
 *   const editor = new Editor({ container: document.getElementById("app")! });
 *   await editor.init();
 *   editor.loadJson(workflowJson);
 *   const json = editor.getJson();
 */

import { EditorState } from "./editor-state.js";
import { initWasm } from "./wasm-loader.js";
import { WorkflowCanvas } from "./panels/workflow-canvas.js";
import { StateInspector } from "./panels/state-inspector.js";
import { ProblemsPanel } from "./panels/problems-panel.js";
import { RunInspector } from "./panels/run-inspector.js";
import {
  ContextInspector,
  CapabilityInspector,
  ArtifactReview,
  TraceTimeline,
  SimulationPanel,
} from "./panels/secondary-panels.js";
import { Panel } from "./panels/panel-base.js";

// ── Types ─────────────────────────────────────────────────────────────────────

export type EditorMode = "edit" | "engine";

export interface EditorConfig {
  /** Container element to mount the editor into. */
  container: HTMLElement;
  /**
   * Editor mode.
   * - "edit": hide skeleton panels (Context, Artifact, Trace, Run). Default.
   * - "engine": show all panels (requires host app to push run snapshots).
   */
  mode?: EditorMode;
  /**
   * Explicit panel IDs to show. Overrides mode-based defaults.
   * If provided, only the listed panels are mounted.
   * Valid IDs: "canvas", "inspector", "problems", "run", "context",
   *            "capability", "artifact", "trace", "simulation", "source"
   */
  visiblePanels?: string[];
  /** Whether to attempt WASM initialization. Default: true. */
  wasmEnabled?: boolean;
  /** Called when dirty state changes. */
  onDirtyChange?: (dirty: boolean) => void;
  /** Called when the file name changes. */
  onFileChange?: (fileName: string) => void;
}

export interface EditorEvents {
  "dirty-change": boolean;
  "file-change": string;
  "json-change": string;
  "state-select": string | null;
}

// ── Panel ID → class mapping ──────────────────────────────────────────────────

const PANEL_FACTORIES: Record<string, new (el: HTMLElement, state: EditorState) => Panel> = {
  canvas:     WorkflowCanvas,
  inspector:  StateInspector,
  problems:   ProblemsPanel,
  run:        RunInspector,
  context:    ContextInspector,
  capability: CapabilityInspector,
  artifact:   ArtifactReview,
  trace:      TraceTimeline,
  simulation: SimulationPanel,
};

/** Default panel layout by mode. */
const EDIT_MODE_PANELS = ["canvas", "inspector", "problems", "simulation", "source"];
const ENGINE_MODE_PANELS = [
  "canvas", "inspector", "problems", "run",
  "context", "capability", "artifact", "trace", "simulation", "source",
];

// ── Editor ────────────────────────────────────────────────────────────────────

export class Editor {
  private readonly _config: Required<EditorConfig>;
  private readonly _container: HTMLElement;
  private _state!: EditorState;
  private _panels: Panel[] = [];
  private _panelContainers: Map<string, HTMLElement> = new Map();
  private _dirty = false;
  private _fileName = "untitled.langchart";
  private _listeners: Map<string, Array<(value: unknown) => void>> = new Map();
  private _canvas: WorkflowCanvas | null = null;
  private _sourceTimer: number | undefined;
  private _sourceTextarea: HTMLTextAreaElement | null = null;
  private _initialised = false;

  constructor(config: EditorConfig) {
    this._container = config.container;
    this._config = {
      container: config.container,
      mode: config.mode ?? "edit",
      visiblePanels: config.visiblePanels ?? [],
      wasmEnabled: config.wasmEnabled ?? true,
      onDirtyChange: config.onDirtyChange ?? (() => {}),
      onFileChange: config.onFileChange ?? (() => {}),
    };
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  /** Initialise WASM and mount all panels. Call once after construction. */
  async init(): Promise<void> {
    if (this._initialised) return;

    if (this._config.wasmEnabled) {
      await initWasm();
    }

    this._state = new EditorState();
    this._state.on((event) => {
      if (event.type === "workflow-changed") {
        this._syncSourceTextarea(event.json);
        this._emit("json-change", event.json);
      }
      if (event.type === "state-selected") {
        this._emit("state-select", event.stateId);
      }
    });

    this._container.innerHTML = "";
    this._container.style.cssText = "display:flex;flex-direction:column;height:100%;overflow:hidden;";

    const visiblePanels = this._resolveVisiblePanels();

    // Build layout structure
    const layout = this._buildLayout(visiblePanels);
    this._container.appendChild(layout);

    // Mount panels into their containers
    for (const id of visiblePanels) {
      const container = this._panelContainers.get(id);
      if (!container) continue;

      if (id === "source") {
        this._mountSourcePanel(container);
        continue;
      }

      const Factory = PANEL_FACTORIES[id];
      if (!Factory) continue;

      const panel = new Factory(container, this._state);
      this._panels.push(panel);

      if (id === "canvas" && panel instanceof WorkflowCanvas) {
        this._canvas = panel;
        this._canvas.onEditCommit = () => {
          this._dirty = true;
          this._emit("dirty-change", true);
        };
      }

      if (id === "inspector" && panel instanceof StateInspector) {
        panel.onEditCommit = () => {
          this._dirty = true;
          this._emit("dirty-change", true);
        };
      }
    }

    this._initialised = true;
  }

  /** Destroy the editor and clean up all panels. */
  destroy(): void {
    for (const panel of this._panels) {
      panel.destroy();
    }
    this._panels = [];
    this._panelContainers.clear();
    this._canvas = null;
    this._sourceTextarea = null;
    this._container.innerHTML = "";
    this._initialised = false;
  }

  // ── Document API ───────────────────────────────────────────────────────────

  /** Load a workflow from a JSON string. */
  loadJson(json: string, fileName?: string, markDirty = false): void {
    this._ensureInit();
    if (fileName !== undefined) {
      this._fileName = normaliseFileName(fileName);
      this._emit("file-change", this._fileName);
    }
    this._dirty = markDirty;
    this._emit("dirty-change", markDirty);
    this._state.loadJson(json);
    this._syncSourceTextarea(json);
  }

  /** Get the current workflow as a JSON string. */
  getJson(): string {
    this._ensureInit();
    return this._state.json;
  }

  /** Get the current file name. */
  getFileName(): string {
    return this._fileName;
  }

  /** Check if the document has unsaved changes. */
  isDirty(): boolean {
    return this._dirty;
  }

  /** Get the underlying EditorState (for advanced use). */
  getState(): EditorState {
    this._ensureInit();
    return this._state;
  }

  /** Get the WASM summary (if available). */
  getSummary() {
    this._ensureInit();
    return this._state.summary;
  }

  // ── File I/O helpers ───────────────────────────────────────────────────────

  /**
   * Export the current workflow as a .langchart file download.
   * Works in browser context (creates a blob URL and triggers download).
   */
  exportFile(): void {
    const json = this.getJson();
    if (!json) return;
    const output = prettyJson(json);
    const blob = new Blob([output + "\n"], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = this._exportFileName();
    a.click();
    URL.revokeObjectURL(url);
    this._fileName = a.download;
    this._dirty = false;
    this._emit("dirty-change", false);
    this._emit("file-change", this._fileName);
  }

  /**
   * Open a file picker dialog and load the selected file.
   * Works in browser context.
   */
  importFile(): Promise<void> {
    return new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".langchart,.json,application/json";
      input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) { resolve(); return; }
        const text = await file.text();
        this.loadJson(text, file.name);
        resolve();
      };
      input.click();
    });
  }

  /** Create a new empty workflow. */
  newWorkflow(): void {
    this.loadJson(newWorkflowJson(), "untitled.langchart", true);
  }

  // ── Source textarea sync ───────────────────────────────────────────────────

  private _syncSourceTextarea(json: string): void {
    if (this._sourceTextarea && document.activeElement !== this._sourceTextarea) {
      this._sourceTextarea.value = prettyJson(json);
    }
  }

  private _mountSourcePanel(container: HTMLElement): void {
    container.innerHTML = "";
    container.style.cssText =
      "display:flex;flex-direction:column;background:#1e1e2e;overflow:hidden;";

    const header = document.createElement("div");
    header.style.cssText =
      "font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em;" +
      "color:#6c7086;padding:6px 10px;border-bottom:1px solid #44445a;flex-shrink:0;display:flex;justify-content:space-between;";
    const titleSpan = document.createElement("span");
    titleSpan.textContent = "Document source";
    const hintSpan = document.createElement("span");
    hintSpan.textContent = "JSON";
    hintSpan.style.fontWeight = "400";
    header.appendChild(titleSpan);
    header.appendChild(hintSpan);
    container.appendChild(header);

    const textarea = document.createElement("textarea");
    textarea.className = "workflow-source";
    textarea.spellcheck = false;
    textarea.setAttribute("aria-label", "Workflow JSON source");
    textarea.style.cssText =
      "flex:1;min-height:0;width:100%;resize:none;border:0;outline:0;padding:10px 12px;" +
      "background:#1e1e2e;color:#cdd6f4;font:12px/1.5 ui-monospace,'Cascadia Code',Consolas,monospace;tab-size:2;";
    textarea.addEventListener("input", () => {
      this._dirty = true;
      this._emit("dirty-change", true);
      window.clearTimeout(this._sourceTimer);
      this._sourceTimer = window.setTimeout(() => {
        this._state.loadJson(textarea.value);
      }, 180);
    });
    container.appendChild(textarea);
    this._sourceTextarea = textarea;
  }

  // ── Layout ─────────────────────────────────────────────────────────────────

  private _resolveVisiblePanels(): string[] {
    if (this._config.visiblePanels.length > 0) {
      return this._config.visiblePanels;
    }
    return this._config.mode === "engine" ? ENGINE_MODE_PANELS : EDIT_MODE_PANELS;
  }

  private _buildLayout(panelIds: string[]): HTMLElement {
    const has = (id: string) => panelIds.includes(id);

    // Three-column layout matching the original HTML structure
    const layout = document.createElement("main");
    layout.className = "layout";
    layout.style.cssText = "flex:1;display:flex;gap:8px;padding:8px;overflow:hidden;";

    // Left column: canvas + source + trace
    const colMain = document.createElement("div");
    colMain.className = "col col--main";
    colMain.style.cssText = "display:flex;flex-direction:column;gap:8px;flex:3;min-width:0;overflow:hidden;";

    if (has("canvas")) {
      const p = this._makePanelSlot("canvas");
      p.style.flex = "3";
      colMain.appendChild(p);
    }
    if (has("source")) {
      const p = this._makePanelSlot("source");
      p.style.flex = "1.35";
      colMain.appendChild(p);
    }
    if (has("trace")) {
      const p = this._makePanelSlot("trace");
      p.style.flex = "1";
      colMain.appendChild(p);
    }
    layout.appendChild(colMain);

    // Right column: inspector + problems + run
    const colRight = document.createElement("div");
    colRight.className = "col col--right";
    colRight.style.cssText = "display:flex;flex-direction:column;gap:8px;flex:1.4;min-width:220px;overflow:hidden;";

    for (const id of ["inspector", "problems", "run"]) {
      if (has(id)) {
        colRight.appendChild(this._makePanelSlot(id));
      }
    }
    if (colRight.children.length > 0) layout.appendChild(colRight);

    // Far-right column: capability + context + artifact + simulation
    const colFar = document.createElement("div");
    colFar.className = "col col--far-right";
    colFar.style.cssText = "display:flex;flex-direction:column;gap:8px;flex:1.2;min-width:200px;overflow:hidden;";

    for (const id of ["capability", "context", "artifact", "simulation"]) {
      if (has(id)) {
        colFar.appendChild(this._makePanelSlot(id));
      }
    }
    if (colFar.children.length > 0) layout.appendChild(colFar);

    return layout;
  }

  private _makePanelSlot(id: string): HTMLElement {
    const el = document.createElement("section");
    el.className = "panel";
    el.id = `panel-${id}`;
    el.style.cssText = "border:1px solid #e5e7eb;border-radius:6px;background:#fff;overflow:auto;flex:1;min-height:0;";
    this._panelContainers.set(id, el);
    return el;
  }

  // ── Internal helpers ───────────────────────────────────────────────────────

  private _ensureInit(): void {
    if (!this._initialised) {
      throw new Error("Editor not initialised. Call editor.init() first.");
    }
  }

  private _exportFileName(): string {
    const id = this._state.summary?.id?.trim();
    return normaliseFileName(id ? `${id}.langchart` : this._fileName);
  }

  // ── Event emitter ──────────────────────────────────────────────────────────

  on<K extends keyof EditorEvents>(event: K, listener: (value: EditorEvents[K]) => void): () => void {
    const list = this._listeners.get(event) ?? [];
    list.push(listener as (value: unknown) => void);
    this._listeners.set(event, list);
    return () => {
      const l = this._listeners.get(event);
      if (l) {
        const idx = l.indexOf(listener as (value: unknown) => void);
        if (idx >= 0) l.splice(idx, 1);
      }
    };
  }

  private _emit<K extends keyof EditorEvents>(event: K, value: EditorEvents[K]): void {
    for (const listener of this._listeners.get(event) ?? []) {
      try { listener(value); } catch (_) { /* isolate */ }
    }
    // Also fire the callback from config
    if (event === "dirty-change") this._config.onDirtyChange(value as boolean);
    if (event === "file-change") this._config.onFileChange(value as string);
  }
}

// ── Utility functions ─────────────────────────────────────────────────────────

function normaliseFileName(name: string): string {
  return name.replace(/\.json$/i, ".langchart") || "untitled.langchart";
}

function prettyJson(json: string): string {
  try { return JSON.stringify(JSON.parse(json), null, 2); } catch { return json; }
}

function newWorkflowJson(): string {
  return JSON.stringify({
    schema_version: "1.0.0",
    id: "untitled",
    version: "0.1.0",
    name: "Untitled workflow",
    initial: "start",
    states: [
      {
        id: "start",
        name: "Start",
        type: "atomic",
        on: { done: [{ target: "complete", priority: 0, actions: [] }] },
      },
      {
        id: "complete",
        name: "Complete",
        type: "final",
        on: {},
      },
    ],
  }, null, 2);
}
