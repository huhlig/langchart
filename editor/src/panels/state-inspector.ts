/**
 * StateInspector — Panel 2
 *
 * Shows configuration for the currently selected state.
 * Uses inline styles throughout — no dependency on an external stylesheet.
 */

import { Panel } from "./panel-base.js";
import { EditorEvent } from "../editor-state.js";
import { wasm, StateInspection } from "../wasm-loader.js";

// ── Colours (matches Collaborite dark palette) ────────────────────────────────
const C = {
  bg:         "#1e1e2e",
  surface:    "#2a2a3d",
  border:     "#3a3a54",
  text:       "#cdd6f4",
  muted:      "#6c7086",
  accent:     "#89b4fa",
  headerBg:   "#22223a",
};

type Tab = "basic" | "advanced" | "source";

/** Raw editable state node fields (subset of the schema we expose in the UI). */
interface RawStateNode {
  id: string;
  name?: string;
  type?: string;
  agent?: { id?: string; version?: string };
  prompt?: string;
  model?: {
    profile?: string;
    model?: string;
    temperature?: number;
    max_tokens?: number;
  };
  [k: string]: unknown;
}

export class StateInspector extends Panel {
  /** Fires after the user edits a property (inspector committed new JSON). */
  onEditCommit?: (json: string) => void;

  private _activeTab: Tab = "basic";

  protected render(): void {
    this.root.innerHTML = "";
    this.root.style.cssText =
      `background:${C.bg};color:${C.text};font-family:system-ui,sans-serif;` +
      `font-size:13px;display:flex;flex-direction:column;height:100%;overflow:hidden;`;

    // Header
    const header = document.createElement("div");
    header.textContent = "State Inspector";
    header.style.cssText =
      `font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em;` +
      `color:${C.muted};padding:6px 10px;border-bottom:1px solid ${C.border};` +
      `background:${C.headerBg};flex-shrink:0;`;
    this.root.appendChild(header);

    const id = this.state.selectedStateId;
    if (!id) {
      const empty = document.createElement("p");
      empty.textContent = "Select a state on the canvas.";
      empty.style.cssText = `padding:12px 10px;color:${C.muted};font-size:12px;`;
      this.root.appendChild(empty);
      return;
    }

    const info: StateInspection | null = this.state.json
      ? wasm.inspectState(this.state.json, id)
      : null;

    if (!info) {
      const empty = document.createElement("p");
      empty.textContent = `State "${id}" not found.`;
      empty.style.cssText = `padding:12px 10px;color:${C.muted};font-size:12px;`;
      this.root.appendChild(empty);
      return;
    }

    // ── Tab bar ──────────────────────────────────────────────────────────────
    const tabBar = document.createElement("div");
    tabBar.style.cssText =
      `display:flex;border-bottom:1px solid ${C.border};` +
      `background:${C.headerBg};flex-shrink:0;`;

    const tabs: Tab[] = ["basic", "advanced", "source"];
    for (const tab of tabs) {
      const isActive = this._activeTab === tab;
      const btn = document.createElement("button");
      btn.textContent = tab.charAt(0).toUpperCase() + tab.slice(1);
      btn.style.cssText =
        `padding:5px 12px;font-size:11px;font-family:system-ui,sans-serif;` +
        `border:none;border-bottom:${isActive ? `2px solid ${C.accent}` : "2px solid transparent"};` +
        `background:transparent;cursor:pointer;` +
        `color:${isActive ? C.accent : C.muted};` +
        `font-weight:${isActive ? "600" : "400"};` +
        `transition:color 0.1s;`;
      btn.addEventListener("mouseenter", () => {
        if (!isActive) btn.style.color = C.text;
      });
      btn.addEventListener("mouseleave", () => {
        if (!isActive) btn.style.color = C.muted;
      });
      btn.addEventListener("click", () => {
        this._activeTab = tab;
        this.refresh();
      });
      tabBar.appendChild(btn);
    }
    this.root.appendChild(tabBar);

    // ── Scrollable content area ───────────────────────────────────────────────
    const content = document.createElement("div");
    content.style.cssText = `flex:1;min-height:0;overflow:auto;`;
    this.root.appendChild(content);

    if (this._activeTab === "basic") {
      this._renderBasic(content, info);
    } else if (this._activeTab === "advanced") {
      this._renderAdvanced(content, info);
    } else {
      this._renderSource(content, id);
    }
  }

  private _row(parent: HTMLElement, label: string, value: string): void {
    const div = document.createElement("div");
    div.style.cssText =
      `display:flex;gap:8px;padding:4px 10px;border-bottom:1px solid ${C.border};`;
    const lbl = document.createElement("span");
    lbl.textContent = label;
    lbl.style.cssText = `color:${C.muted};min-width:90px;flex-shrink:0;font-size:12px;`;
    const val = document.createElement("span");
    val.textContent = value;
    val.style.cssText = `color:${C.text};font-size:12px;word-break:break-all;`;
    div.appendChild(lbl);
    div.appendChild(val);
    parent.appendChild(div);
  }

  /**
   * Editable property row: label + input (or textarea). Commits on Enter or
   * blur when the value changed; Escape reverts.
   */
  private _editRow(
    parent: HTMLElement,
    label: string,
    value: string,
    onCommit: (v: string) => void,
    opts: { multiline?: boolean; placeholder?: string; numeric?: boolean } = {},
  ): void {
    const div = document.createElement("div");
    div.style.cssText =
      `display:flex;gap:8px;padding:4px 10px;border-bottom:1px solid ${C.border};`;
    div.dataset.label = label;
    const lbl = document.createElement("span");
    lbl.textContent = label;
    lbl.style.cssText =
      `color:${C.muted};min-width:90px;flex-shrink:0;font-size:12px;padding-top:3px;`;
    const input = opts.multiline
      ? document.createElement("textarea")
      : document.createElement("input");
    input.value = value;
    if (opts.placeholder) input.placeholder = opts.placeholder;
    if (opts.numeric) (input as HTMLInputElement).type = "number";
    input.style.cssText =
      `flex:1;min-width:0;background:${C.bg};color:${C.text};font-size:12px;` +
      `font-family:system-ui,sans-serif;border:1px solid ${C.border};` +
      `border-radius:3px;padding:2px 6px;outline:none;` +
      (opts.multiline ? `resize:vertical;min-height:44px;` : `height:22px;`);
    input.addEventListener("focus", () => { input.style.borderColor = C.accent; });
    input.addEventListener("blur",  () => { input.style.borderColor = C.border; });

const commit = () => {
      const v = input.value;
      if (v !== value) onCommit(v);
    };
    input.addEventListener("blur", commit);
    div.appendChild(lbl);
    div.appendChild(input);
    parent.appendChild(div);
  }

  /** Apply a mutation to the selected state's raw JSON node and commit. */
  private _commitEdit(stateId: string, mutate: (node: RawStateNode) => void): void {
    if (!this.state.json) return;
    try {
      const doc = JSON.parse(this.state.json) as { states?: unknown[] };
      const found = findStateInDoc(doc.states ?? [], stateId);
      if (found === null) return;
      mutate(found as RawStateNode);
      const json = JSON.stringify(doc, null, 2);
      this.state.loadJson(json);
      this.onEditCommit?.(json);
    } catch {
      // JSON parse/serialize error — leave the document untouched
    }
  }

  private _sectionLabel(parent: HTMLElement, text: string): void {
    const h = document.createElement("div");
    h.textContent = text;
    h.style.cssText =
      `font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;` +
      `color:${C.muted};padding:8px 10px 3px;`;
    parent.appendChild(h);
  }

  private _renderBasic(parent: HTMLElement, info: StateInspection): void {
    this._row(parent, "ID",      info.id);
    this._row(parent, "Type",    info.type);
    this._row(parent, "Initial", info.is_initial ? "yes" : "no");

    // Raw node drives the editable fields (works with or without WASM)
    const node = this._rawNode(info.id);

    this._editRow(parent, "Name", info.name, (v) => {
      this._commitEdit(info.id, (n) => { n.name = v; });
    });

    if (info.type === "agentic" && node) {
      this._sectionLabel(parent, "Agent & model");
      this._editRow(parent, "Agent ID", node.agent?.id ?? "", (v) => {
        this._commitEdit(info.id, (n) => this._setAgentField(n, "id", v));
      }, { placeholder: "analyst" });
      this._editRow(parent, "Agent ver.", node.agent?.version ?? "", (v) => {
        this._commitEdit(info.id, (n) => this._setAgentField(n, "version", v));
      }, { placeholder: "1.0.0" });
      this._editRow(parent, "Prompt", node.prompt ?? "", (v) => {
        this._commitEdit(info.id, (n) => {
          if (v) n.prompt = v; else delete n.prompt;
        });
      }, { multiline: true, placeholder: "Task prompt appended to the agent's system instructions" });
      this._editRow(parent, "Model alias", node.model?.profile ?? "", (v) => {
        this._commitEdit(info.id, (n) => this._setModelField(n, "profile", v));
      }, { placeholder: "fast / high_quality / local" });
      this._editRow(parent, "Model name", node.model?.model ?? "", (v) => {
        this._commitEdit(info.id, (n) => this._setModelField(n, "model", v));
      }, { placeholder: "explicit model, overrides the alias" });
      this._editRow(parent, "Temperature", node.model?.temperature?.toString() ?? "", (v) => {
        this._commitEdit(info.id, (n) => this._setModelNum(n, "temperature", v));
      }, { numeric: true, placeholder: "0.0 – 2.0" });
      this._editRow(parent, "Max tokens", node.model?.max_tokens?.toString() ?? "", (v) => {
        this._commitEdit(info.id, (n) => this._setModelNum(n, "max_tokens", v));
      }, { numeric: true, placeholder: "output token cap" });
    }

    if (info.transitions.length > 0) {
      this._sectionLabel(parent, "Transitions");
      for (const tx of info.transitions) {
        const row = document.createElement("div");
        row.style.cssText =
          `padding:3px 10px;border-bottom:1px solid ${C.border};font-size:11px;` +
          `display:flex;gap:6px;align-items:baseline;`;
        const evt = document.createElement("span");
        evt.textContent = tx.event;
        evt.style.cssText = `color:${C.accent};font-family:monospace;`;
        const arrow = document.createElement("span");
        arrow.textContent = "→";
        arrow.style.cssText = `color:${C.muted};`;
        const tgt = document.createElement("span");
        tgt.textContent = tx.target;
        tgt.style.cssText = `color:${C.text};font-family:monospace;`;
        row.appendChild(evt);
        row.appendChild(arrow);
        row.appendChild(tgt);
        if (tx.guard) {
          const guard = document.createElement("span");
          guard.textContent = `[${tx.guard}]`;
          guard.style.cssText = `color:${C.muted};font-style:italic;margin-left:4px;`;
          row.appendChild(guard);
        }
        parent.appendChild(row);
      }
    }
  }

  /** The selected state's raw JSON node, or null. */
  private _rawNode(id: string): RawStateNode | null {
    if (!this.state.json) return null;
    try {
      const doc = JSON.parse(this.state.json) as { states?: unknown[] };
      const found = findStateInDoc(doc.states ?? [], id);
      return found === null ? null : (found as RawStateNode);
    } catch {
      return null;
    }
  }

  /** Set/clear a string field of `node.agent`, dropping the record when empty. */
  private _setAgentField(node: RawStateNode, field: "id" | "version", v: string): void {
    if (v) {
      (node.agent ??= {})[field] = v;
    } else if (node.agent) {
      delete node.agent[field];
      if (!node.agent.id && !node.agent.version) delete node.agent;
    }
  }

  /** Set/clear a string field of `node.model`, dropping the record when empty. */
  private _setModelField(node: RawStateNode, field: "profile" | "model", v: string): void {
    if (v) {
      (node.model ??= {})[field] = v;
    } else if (node.model) {
      delete node.model[field];
      if (!this._modelHasFields(node.model)) delete node.model;
    }
  }

  /** Set/clear a numeric field of `node.model` (invalid input is ignored). */
  private _setModelNum(node: RawStateNode, field: "temperature" | "max_tokens", v: string): void {
    const n = v === "" ? undefined : Number(v);
    if (v !== "" && (n === undefined || Number.isNaN(n))) return;
    if (n !== undefined) {
      (node.model ??= {})[field] = n;
    } else if (node.model) {
      delete node.model[field];
      if (!this._modelHasFields(node.model)) delete node.model;
    }
  }

  private _modelHasFields(model: NonNullable<RawStateNode["model"]>): boolean {
    return model.profile !== undefined || model.model !== undefined ||
           model.temperature !== undefined || model.max_tokens !== undefined;
  }

  private _renderAdvanced(parent: HTMLElement, info: StateInspection): void {
    this._row(parent, "Limits",       info.has_limits       ? "configured" : "defaults");
    this._row(parent, "Capabilities", info.has_capabilities ? "configured" : "inherited");
    this._row(parent, "Children",     `${info.child_count}`);
    this._row(parent, "Regions",      `${info.region_count}`);
  }

  private _renderSource(parent: HTMLElement, stateId: string): void {
    if (!this.state.json) return;
    let fragment = "null";
    try {
      const doc = JSON.parse(this.state.json) as { states?: unknown[] };
      const found = findStateInDoc(doc.states ?? [], stateId);
      fragment = found !== null
        ? JSON.stringify(found, null, 2)
        : `// State "${stateId}" not found in document`;
    } catch {
      fragment = `// JSON parse error`;
    }
    const pre = document.createElement("pre");
    pre.textContent = fragment;
    pre.style.cssText =
      `font-family:monospace;font-size:11px;padding:10px;white-space:pre-wrap;` +
      `word-break:break-all;line-height:1.5;color:${C.text};`;
    parent.appendChild(pre);
  }

  protected override onEditorEvent(event: EditorEvent): void {
    if (event.type === "state-selected" || event.type === "workflow-changed") {
      this.refresh();
    }
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function findStateInDoc(states: unknown[], id: string): unknown | null {
  for (const s of states) {
    if (typeof s !== "object" || s === null) continue;
    const state = s as Record<string, unknown>;
    if (state["id"] === id) return state;
    const child = findStateInDoc((state["states"] as unknown[]) ?? [], id);
    if (child !== null) return child;
    for (const region of (state["regions"] as Array<{ states?: unknown[] }>) ?? []) {
      const r = findStateInDoc(region.states ?? [], id);
      if (r !== null) return r;
    }
  }
  return null;
}
