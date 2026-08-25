/**
 * WorkflowCanvas — Panel 1
 *
 * Renders the statechart graph hierarchically:
 *   - Compound states → rounded container box with children inside
 *   - Parallel states → container with region lane boxes
 *   - All nodes and containers are independently draggable
 *   - Edges always redraw from current box positions
 *   - Transitions are clickable: select (click), edit (double-click or
 *     right-click menu), delete (Delete key or menu)
 *   - Right-drag from a node onto another node creates a transition
 *   - New states can be created as any supported type (atomic, agentic,
 *     compound, parallel, human, final) — top-level or as children of
 *     compound/parallel containers
 *
 * Design principle: the layout tree stores *base* positions (from the layout
 * engine). Per-node drag deltas are kept in `dragOffsets`. Effective position
 * is always `base + delta`. The layout tree is NEVER mutated — `applyOffsets`
 * is replaced by a pure `effectivePos()` read at draw time.
 */

import { Panel } from "./panel-base.js";
import { EditorEvent } from "../editor-state.js";

// ── Layout constants ──────────────────────────────────────────────────────────
const NODE_W     = 144;
const NODE_H     = 44;
const H_GAP      = 32;
const CPD_PAD    = 20;
const REGION_GAP = 12;

// ── Colours ───────────────────────────────────────────────────────────────────
const C = {
  bg:              "#1e1e2e",
  nodeFill:        "#2a2a3d",
  nodeFillInitial: "#25304a",
  nodeFillActive:  "#1e3a2f",
  nodeFillFinal:   "#2a2333",
  nodeFillHuman:   "#2a3028",
  nodeStroke:      "#3a3a54",
  nodeStrokeSel:   "#89b4fa",
  nodeStrokeAct:   "#a6e3a1",
  nodeText:        "#cdd6f4",
  nodeTextMuted:   "#6c7086",
  compoundFill:    "#22223a",
  compoundStroke:  "#44445a",
  regionFill:      "#1e1e30",
  regionStroke:    "#33334a",
  regionLabel:     "#6c7086",
  edgeStroke:      "#6c7086",
  edgeLabelText:   "#6c7086",
  headerText:      "#6c7086",
};

// ── Schema types ──────────────────────────────────────────────────────────────

interface RawTransitionSpec { target: string; guard?: string; priority?: number; }
interface RawRegion { id: string; name?: string; initial?: string; states: RawStateNode[]; }
interface RawStateNode {
  id: string; name?: string; type?: string; initial?: string;
  on?: Record<string, RawTransitionSpec[]>;
  states?: RawStateNode[];
  regions?: RawRegion[];
}
interface RawWorkflow { id?: string; version?: string; name?: string; initial?: string; states?: RawStateNode[]; }

/** Identifies one transition (from-state, event name, target state). */
interface EdgeRef { from: string; event: string; to: string; }

/** State types offered when creating states (subworkflow needs a workflow ref — JSON only). */
const STATE_TYPES = [
  { label: "Add atomic state…",   type: "atomic"   },
  { label: "Add agentic state…",  type: "agentic"  },
  { label: "Add compound state…", type: "compound" },
  { label: "Add parallel state…", type: "parallel" },
  { label: "Add human state…",    type: "human"    },
  { label: "Add final state…",    type: "final"    },
] as const;

/** Help-popup content: state types — structural class and purpose. */
const HELP_STATE_TYPES: Array<{ type: string; kind: string; desc: string }> = [
  {
    type: "atomic", kind: "leaf · no activity",
    desc: "A pure event sink: on entry the workflow simply waits until an event fires one of its transitions. Use it for routing, joins, and waiting on external events — all behaviour lives in the transitions.",
  },
  {
    type: "agentic", kind: "leaf · agent activity",
    desc: "On entry an LLM agent is started inside a capability envelope (limits, retry, fallback model — configured via the agent / prompt / model properties below). Use it when the state's work is done by a model; the agent's emitted event (e.g. analysis.completed) typically drives the outbound transition.",
  },
  {
    type: "human", kind: "leaf · human activity",
    desc: "On entry a HumanInputRequested event is emitted and the workflow suspends until someone holding an authorized_roles role supplies a decision or data. Structurally identical to atomic — the type just names who does the work: a person instead of a model. Use it for approvals and reviews.",
  },
  {
    type: "subworkflow", kind: "leaf · child-workflow activity",
    desc: "On entry a separately versioned workflow is invoked through typed port bindings (workflow_ref, ports). Use it to compose and version sub-processes independently. Created via JSON — it needs a workflow reference.",
  },
  {
    type: "compound", kind: "container",
    desc: "A nested statechart drawn as a box: its initial field selects the first child state, and children are laid out inside. Use it to group a phase of work or to give a set of children one shared transition surface.",
  },
  {
    type: "parallel", kind: "container · concurrent",
    desc: "A container whose two or more orthogonal regions run concurrently. Completion modes: all (default), any, quorum, guard, manual. Use it when the work genuinely branches into independent tracks.",
  },
  {
    type: "final", kind: "leaf · terminal",
    desc: "No outgoing transitions, no activity. Marks completion of a region — or of the whole workflow when it is top-level.",
  },
];

// ── Layout types ──────────────────────────────────────────────────────────────

interface NodeBox {
  id: string; type: string; name: string;
  /** Base position — never mutated after layout. */
  bx: number; by: number;
  w: number; h: number;
  children: NodeBox[];
  regions?: RegionBox[];
  isInitial: boolean;
}

interface RegionBox {
  id: string; name: string;
  /** Base position — relative to parent's base. Offset by parent drag delta. */
  bx: number; by: number;
  w: number; h: number;
  children: NodeBox[];
}

/** Union rect of all node effective positions. */
type Bounds = { minX: number; minY: number; maxX: number; maxY: number };

// ── WorkflowCanvas ────────────────────────────────────────────────────────────

export class WorkflowCanvas extends Panel {
  private readonly NS = "http://www.w3.org/2000/svg";

  // All fields are initialized lazily inside render() because Panel's base
  // constructor calls render() before subclass field initializers run.
  private dragOffsets!: Map<string, { dx: number; dy: number }>;
  private svg!:            SVGSVGElement | null;
  private containerLayer!: SVGGElement   | null;
  private edgeLayer!:      SVGGElement   | null;
  private nodeLayer!:      SVGGElement   | null;
  private groupMap!: Map<string, SVGGElement>;
  private boxMap!:   Map<string, NodeBox>;
  private allBoxList!: NodeBox[];
  private topStates!: RawStateNode[];
  /** Raw workflow envelope (id/name/version/initial/etc.) persisted across edits. */
  private _docEnvelope!: RawWorkflow;
  /** True when the loaded workflow is read-only (builtin). */
  private _readOnly: boolean = false;
  /** Suppresses dragOffsets.clear() on the next workflow-changed event (our own commit). */
  private _suppressNextDragClear: boolean = false;
  /**
   * Grows the SVG when the canvas viewport resizes. `declare` — like the lazy
   * fields above, it must survive the base-constructor render() call (a real
   * field initializer would define it as undefined after render() ran).
   */
  declare private _resizeObs: ResizeObserver | undefined;
  /** Currently selected transition (edge), if any. */
  private _selectedEdge: EdgeRef | null = null;
  /** Set right after a completed right-button node drag so the trailing contextmenu is swallowed. */
  private _suppressNextContextMenu: boolean = false;
  /** Callback invoked whenever the user edits the workflow JSON. */
  onEditCommit?: (json: string) => void;

  /** Mark this canvas read-only (builtins). */
  setReadOnly(value: boolean): void { this._readOnly = value; }

  // ── Edit helpers ──────────────────────────────────────────────────────────

  /** Commit the current topStates back into the EditorState and notify the host. */
  private commitEdit(): void {
    const json = JSON.stringify({ ...this._docEnvelope, states: this.topStates });
    this._suppressNextDragClear = true;
    this.state.loadJson(json);
    this.onEditCommit?.(json);
  }

  /** Show a brief read-only toast and return true if edits are blocked. */
  private guardReadOnly(): boolean {
    if (!this._readOnly) return false;
    this.showToast("This is a built-in workflow — fork to Global to edit.");
    return true;
  }

  /** Display a small toast message in the canvas. */
  private showToast(msg: string): void {
    const existing = this.root.querySelector<HTMLElement>(".wf-toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.className = "wf-toast";
    toast.textContent = msg;
    toast.style.cssText =
      `position:absolute;bottom:12px;left:50%;transform:translateX(-50%);` +
      `background:#45475a;color:#cdd6f4;font-size:11px;padding:4px 12px;` +
      `border-radius:4px;pointer-events:none;z-index:100;white-space:nowrap;`;
    this.root.style.position = "relative";
    this.root.appendChild(toast);
    setTimeout(() => toast.remove(), 2200);
  }

  /** Show a lightweight context menu near the given SVG point. */
  private showContextMenu(
    items: Array<{ label: string; action: () => void } | { separator: true }>,
    clientX: number,
    clientY: number,
  ): void {
    // Remove any existing menu
    document.querySelector(".wf-ctx-menu")?.remove();
    const menu = document.createElement("div");
    menu.className = "wf-ctx-menu";
    menu.style.cssText =
      `position:fixed;left:${clientX}px;top:${clientY}px;` +
      `background:#2a2a3d;border:1px solid #3a3a54;border-radius:5px;` +
      `box-shadow:0 4px 12px #00000066;z-index:9999;min-width:160px;overflow:hidden;`;
    for (const item of items) {
      if ("separator" in item) {
        const sep = document.createElement("div");
        sep.style.cssText = `height:1px;background:#3a3a54;margin:4px 0;`;
        menu.appendChild(sep);
        continue;
      }
      const btn = document.createElement("div");
      btn.textContent = item.label;
      btn.style.cssText =
        `padding:6px 14px;font-size:12px;color:#cdd6f4;cursor:pointer;` +
        `font-family:system-ui,sans-serif;`;
      btn.addEventListener("mouseenter", () => { btn.style.background = "#3a3a54"; });
      btn.addEventListener("mouseleave", () => { btn.style.background = ""; });
      btn.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        menu.remove();
        item.action();
      });
      menu.appendChild(btn);
    }
    document.body.appendChild(menu);
    // Close on any outside click
    const close = (e: MouseEvent) => {
      if (!menu.contains(e.target as Node)) { menu.remove(); document.removeEventListener("mousedown", close, true); }
    };
    setTimeout(() => document.addEventListener("mousedown", close, true), 0);
  }

  /** Prompt user inline with a browser prompt (minimal, avoids native dialog dependency). */
  private promptValue(label: string, defaultVal = ""): string | null {
    // Use native prompt — sufficient for desktop Tauri context
    return window.prompt(label, defaultVal);
  }

  // ── Help popup ──────────────────────────────────────────────────────────────

  /** Open the canvas help modal (operations, state types, state properties). */
  private showHelp(): void {
    document.querySelector(".wf-help-overlay")?.remove();
    const overlay = document.createElement("div");
    overlay.className = "wf-help-overlay";
    overlay.style.cssText =
      `position:fixed;inset:0;background:#000000aa;z-index:10001;` +
      `display:flex;align-items:center;justify-content:center;`;

    const dlg = document.createElement("div");
    dlg.style.cssText =
      `width:min(700px,92vw);max-height:86vh;overflow:auto;background:#1e1e2e;` +
      `border:1px solid #3a3a54;border-radius:8px;padding:16px 20px 20px;` +
      `font-family:system-ui,sans-serif;font-size:12.5px;color:#cdd6f4;` +
      `box-shadow:0 8px 32px #000000aa;`;

    const h = (text: string) => {
      const el = document.createElement("div");
      el.textContent = text;
      el.style.cssText =
        `font-weight:600;font-size:13px;color:#89b4fa;margin:16px 0 6px;` +
        `text-transform:uppercase;letter-spacing:.04em;`;
      dlg.appendChild(el);
    };
    const p = (html: string) => {
      const el = document.createElement("div");
      el.innerHTML = html;
      el.style.cssText = `line-height:1.55;margin:4px 0;`;
      dlg.appendChild(el);
      return el;
    };
    const li = (html: string, list: HTMLElement) => {
      const el = document.createElement("li");
      el.innerHTML = html;
      el.style.cssText = `line-height:1.5;margin:3px 0;`;
      list.appendChild(el);
    };
    const pre = (text: string) => {
      const el = document.createElement("pre");
      el.textContent = text;
      el.style.cssText =
        `font-family:monospace;font-size:11px;line-height:1.5;background:#181825;` +
        `border:1px solid #313145;border-radius:6px;padding:10px;margin:8px 0;` +
        `white-space:pre;overflow:auto;`;
      dlg.appendChild(el);
    };

    const title = document.createElement("div");
    title.textContent = "Workflow Canvas Help";
    title.style.cssText =
      `font-weight:600;font-size:15px;margin-bottom:4px;flex:1;`;
    const titleRow = document.createElement("div");
    titleRow.style.cssText = `display:flex;align-items:center;`;
    titleRow.appendChild(title);

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    closeBtn.style.cssText =
      `background:transparent;border:none;color:#6c7086;font-size:14px;` +
      `cursor:pointer;padding:2px 6px;`;
    closeBtn.addEventListener("mouseenter", () => { closeBtn.style.color = "#cdd6f4"; });
    closeBtn.addEventListener("mouseleave", () => { closeBtn.style.color = "#6c7086"; });
    titleRow.appendChild(closeBtn);
    dlg.appendChild(titleRow);

    h("Basic operations");
    const ops = document.createElement("ul");
    ops.style.cssText = `padding-left:18px;margin:6px 0;`;
    li("<b>Move</b> — drag any node or container with the left mouse button; containers carry their children.", ops);
    li("<b>Select</b> — click a node or a transition (line). The Delete or Backspace key removes the current selection (a selected transition takes precedence).", ops);
    li("<b>Rename</b> — double-click a node, or right-click → Rename…", ops);
    li("<b>Add states</b> — right-click empty canvas and pick a type. On compound/parallel containers: right-click → Add child state…", ops);
    li("<b>Create a transition</b> — <b>right-drag</b> from a node onto another node, then name the event. (Alternative: right-click a node → Add transition from here…)", ops);
    li("<b>Edit a transition</b> — click to select it; double-click to rename its event; right-click for Change target… / Edit guard… / Delete.", ops);
    li("<b>Scroll</b> — the canvas grows automatically as you drag content around; nothing gets cut off.", ops);
    dlg.appendChild(ops);

    h("State types");
    p(
      "Two independent axes define a state: <b>structure</b> and <b>activity</b>. " +
      "Structurally, <b>compound</b> and <b>parallel</b> are containers (they hold child states); every other type is a <b>leaf</b>. " +
      "On entry, a leaf may start an <b>activity</b> — nothing (atomic), an agent (agentic), a human decision (human), or a child workflow (subworkflow). " +
      "Agentic and human really are atomic leaves — the type just names who does the work. " +
      "Visual cues: the initial state has a dashed border; the active state is green.",
    );
    const types = document.createElement("ul");
    types.style.cssText = `padding-left:18px;margin:6px 0;`;
    for (const t of HELP_STATE_TYPES) {
      li(
        `<b><span style="font-family:monospace">${t.type}</span></b>` +
        ` <span style="color:#6c7086">(${t.kind})</span> — ${t.desc}`,
        types,
      );
    }
    dlg.appendChild(types);

    h("State properties (agent, prompt, model alias…)");
    p("Properties beyond the basics (agent, prompt, model, limits, …) are set on the state's JSON object. View a state's JSON in the <b>State Inspector → Source</b> tab, and edit it in the workflow JSON (the <b>Source</b> panel or the <code>.json</code> file).");
    pre(JSON.stringify({
      id: "analyze",
      name: "Analyze",
      type: "agentic",
      agent: { id: "analyst", version: "1.0.0" },
      prompt: "Summarise the input, then emit analysis.completed",
      model: { profile: "fast", model: "gpt-4o", temperature: 0.2, max_tokens: 1024 },
      input: { text: "data.request_text" },
      on: { "analysis.completed": [{ target: "review" }] },
    }, null, 2));
    const props = document.createElement("ul");
    props.style.cssText = `padding-left:18px;margin:6px 0;`;
    li("<b>agent</b> — <code>{ \"id\", \"version\" }</code> reference to the agent to invoke (agentic states).", props);
    li("<b>prompt</b> — static task prompt appended to the agent's system instructions.", props);
    li("<b>model</b> — model selection: <code>profile</code> is a <b>model alias</b> resolved by the model router (e.g. <code>\"fast\"</code>, <code>\"high_quality\"</code>), <code>model</code> is an explicit model name that overrides the profile; <code>temperature</code> and <code>max_tokens</code> are optional.", props);
    li("<b>input</b> — maps workflow-data expressions to agent input fields.", props);
    li("Agent-level defaults (system prompt, default model policy) live in the workflow's top-level <code>agents</code> array; a state's <code>model</code> narrows them.", props);
    dlg.appendChild(props);

    overlay.appendChild(dlg);

    const close = () => {
      overlay.remove();
      document.removeEventListener("keydown", onKey);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
    closeBtn.addEventListener("click", close);
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
  }

  /** Generate a collision-free state id. */
  private newStateId(base: string): string {
    const all = new Set<string>();
    for (const b of (this.allBoxList ?? [])) all.add(b.id);
    let candidate = base.toLowerCase().replace(/\s+/g, "_");
    let i = 0;
    while (all.has(candidate)) candidate = `${base.toLowerCase().replace(/\s+/g, "_")}_${++i}`;
    return candidate;
  }

  protected render(): void {
    // Lazily initialize all instance state — safe to call on every render
    // because Panel's base constructor calls render() before field initializers.
    if (!this.dragOffsets) this.dragOffsets = new Map();
    if (!this.groupMap)    this.groupMap    = new Map();
    if (!this.boxMap)      this.boxMap      = new Map();
    this.groupMap.clear();
    this.boxMap.clear();
    this.allBoxList = [];
    this.topStates  = [];
    this.svg             = null;
    this.containerLayer  = null;
    this.edgeLayer       = null;
    this.nodeLayer       = null;

    this.root.innerHTML = "";
    this.root.style.cssText =
      `background:${C.bg};display:flex;flex-direction:column;flex:1;min-width:0;height:100%;overflow:hidden;`;

    // Header
    const header = document.createElement("div");
    header.style.cssText =
      `display:flex;align-items:center;gap:8px;font-weight:600;font-size:11px;` +
      `text-transform:uppercase;letter-spacing:.05em;color:${C.headerText};` +
      `padding:6px 10px;border-bottom:1px solid ${C.compoundStroke};` +
      `background:${C.bg};flex-shrink:0;`;
    const title = document.createElement("span");
    title.textContent = "Workflow Canvas";
    title.style.flex = "1";
    header.appendChild(title);

    const helpBtn = document.createElement("button");
    helpBtn.textContent = "?";
    helpBtn.title = "Canvas help";
    helpBtn.style.cssText =
      `width:20px;height:20px;border-radius:50%;border:1px solid ${C.compoundStroke};` +
      `background:transparent;color:${C.headerText};font-size:11px;line-height:1;` +
      `cursor:pointer;padding:0;font-family:system-ui,sans-serif;text-transform:none;`;
    helpBtn.addEventListener("mouseenter", () => { helpBtn.style.color = C.nodeStrokeSel; });
    helpBtn.addEventListener("mouseleave", () => { helpBtn.style.color = C.headerText; });
    helpBtn.addEventListener("click", () => this.showHelp());
    header.appendChild(helpBtn);
    this.root.appendChild(header);

    if (!this.state.json) {
      const empty = document.createElement("p");
      empty.textContent = "No workflow loaded.";
      empty.style.cssText = `padding:16px 12px;color:${C.headerText};font-size:13px;`;
      this.root.appendChild(empty);
      return;
    }

    let doc: RawWorkflow;
    try { doc = JSON.parse(this.state.json) as RawWorkflow; }
    catch { doc = {}; }

    // Persist envelope across edits — only update when NOT re-rendering after our own commitEdit
    if (!this._suppressNextDragClear) {
      const { states: _s, ...envelope } = doc as RawWorkflow & { states?: RawStateNode[] };
      this._docEnvelope = envelope as RawWorkflow;
    }
    this._suppressNextDragClear = false;

    this.topStates = doc.states ?? [];
    const initial  = doc.initial ?? "";
    const summary  = this.state.summary;

    // Info bar
    const infoBar = document.createElement("div");
    infoBar.textContent = summary
      ? `${summary.name} v${summary.version} — ${summary.state_counts.total} states`
      : (doc.name ?? "");
    infoBar.style.cssText =
      `padding:3px 10px;font-size:11px;color:${C.headerText};` +
      `border-bottom:1px solid ${C.compoundStroke};background:${C.bg};flex-shrink:0;`;
    this.root.appendChild(infoBar);

    const wrapper = document.createElement("div");
    wrapper.style.cssText = `flex:1;min-height:0;overflow:auto;`;
    this.root.appendChild(wrapper);

    // ── Layout ───────────────────────────────────────────────────────────────

    const layout = layoutStates(this.topStates, initial, CPD_PAD, CPD_PAD + 8);

    // Populate lookup maps (immutable base positions)
    for (const b of walkAll(layout)) {
      this.boxMap.set(b.id, b);
      this.allBoxList.push(b);
    }

    // ── SVG ──────────────────────────────────────────────────────────────────

    // Re-anchor content that sits at negative coordinates (pure offset shift,
    // applied before first paint so nothing jumps).
    this.normalizeNegativeOffsets();

    const svg = document.createElementNS(this.NS, "svg") as SVGSVGElement;
    // Placeholder size — growCanvas() (called once mounted) sizes the SVG to
    // at least the visible wrapper area plus all content bounds.
    svg.setAttribute("width",  "300");
    svg.setAttribute("height", "200");
    svg.style.cssText = `display:block;background:${C.bg};`;
    this.svg = svg;

    // Arrow marker
    const defs   = document.createElementNS(this.NS, "defs");
    const marker = document.createElementNS(this.NS, "marker");
    marker.setAttribute("id", "wf-arrow");
    marker.setAttribute("markerWidth", "7"); marker.setAttribute("markerHeight", "7");
    marker.setAttribute("refX", "5");        marker.setAttribute("refY", "3");
    marker.setAttribute("orient", "auto");
    const ap = document.createElementNS(this.NS, "path");
    ap.setAttribute("d", "M0,0 L0,6 L7,3 z");
    ap.setAttribute("fill", C.edgeStroke);
    marker.appendChild(ap);
    defs.appendChild(marker);
    svg.appendChild(defs);

    this.containerLayer = document.createElementNS(this.NS, "g") as SVGGElement;
    this.edgeLayer      = document.createElementNS(this.NS, "g") as SVGGElement;
    this.nodeLayer      = document.createElementNS(this.NS, "g") as SVGGElement;
    svg.appendChild(this.containerLayer);
    svg.appendChild(this.edgeLayer);
    svg.appendChild(this.nodeLayer);

    // Render everything
    this.renderContainerLayer(layout);
    this.renderNodeLayer(layout);
    this.renderEdgeLayer();

    // ── SVG background right-click → Add state (choose type) ─────────────────
    svg.addEventListener("contextmenu", (e: MouseEvent) => {
      // Only fire if the click is directly on the SVG (not a node group)
      if ((e.target as Element) !== svg) return;
      e.preventDefault();
      if (this._suppressNextContextMenu) { this._suppressNextContextMenu = false; return; }
      if (this.guardReadOnly()) return;
      this.showContextMenu(
        [
          ...STATE_TYPES.map(t => ({ label: t.label, action: () => this.addTopState(t.type) })),
          { separator: true },
          { label: "Help…", action: () => this.showHelp() },
        ],
        e.clientX, e.clientY,
      );
    });

    // Click on empty canvas background deselects the selected transition
    svg.addEventListener("click", (e: MouseEvent) => {
      if ((e.target as Element) === svg && this._selectedEdge) this.selectEdge(null);
    });

    // ── Delete key on selected transition or state ───────────────────────────
    // We attach to the wrapper so it captures keyboard when the canvas is focused
    wrapper.setAttribute("tabindex", "-1");
    wrapper.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      if (this.guardReadOnly()) return;
      // A selected transition takes precedence over the selected state
      if (this._selectedEdge) {
        e.preventDefault();
        this.deleteTransition(this._selectedEdge);
        return;
      }
      const selId = this.state.selectedStateId;
      if (!selId) return;
      e.preventDefault();
      this.deleteState(selId);
    });

    wrapper.appendChild(svg);

    // Fill the visible canvas area (at minimum) and fit all content; keep the
    // SVG filling the wrapper when the window resizes.
    this.growCanvas();
    if (!this._resizeObs) this._resizeObs = new ResizeObserver(() => this.growCanvas());
    this._resizeObs.disconnect();
    this._resizeObs.observe(wrapper);
  }

  /** Remove a state (and all transitions targeting it) from topStates. */
  private deleteState(id: string): void {
    const removeFrom = (states: RawStateNode[]): RawStateNode[] =>
      states
        .filter(s => s.id !== id)
        .map(s => ({
          ...s,
          on: s.on
            ? Object.fromEntries(
                Object.entries(s.on).map(([ev, specs]) => [
                  ev,
                  specs.filter(sp => sp.target !== id),
                ]).filter(([, specs]) => (specs as RawTransitionSpec[]).length > 0),
              )
            : undefined,
          states: s.states ? removeFrom(s.states) : undefined,
        })) as RawStateNode[];
    this.topStates = removeFrom(this.topStates);
    this.commitEdit();
  }

  // ── Transition (edge) editing ──────────────────────────────────────────────

  /** Select (or deselect) a transition and redraw the edge layer. */
  private selectEdge(ref: EdgeRef | null): void {
    this._selectedEdge = ref;
    this.renderEdgeLayer();
  }

  /** Find a state node anywhere in the tree by id. */
  private findStateInTree(states: RawStateNode[], id: string): RawStateNode | null {
    for (const s of states) {
      if (s.id === id) return s;
      if (s.states) { const f = this.findStateInTree(s.states, id); if (f) return f; }
      if (s.regions) for (const r of s.regions) {
        const f = this.findStateInTree(r.states, id);
        if (f) return f;
      }
    }
    return null;
  }

  /** Locate the spec array + index of a transition, or null if it no longer exists. */
  private findTransition(ref: EdgeRef): { state: RawStateNode; specs: RawTransitionSpec[]; index: number } | null {
    const state = this.findStateInTree(this.topStates, ref.from);
    const specs = state?.on?.[ref.event];
    if (!state || !specs) return null;
    const index = specs.findIndex(sp => sp.target === ref.to);
    return index < 0 ? null : { state, specs, index };
  }

  /** Remove the event key (and the `on` record when it becomes empty). */
  private removeEventKey(state: RawStateNode, event: string, specs: RawTransitionSpec[]): void {
    if (specs.length) return;
    if (!state.on) return;
    delete state.on[event];
    if (!Object.keys(state.on).length) delete state.on;
  }

  /** Remove the transition identified by (from, event, to). */
  private deleteTransition(ref: EdgeRef): void {
    const t = this.findTransition(ref);
    if (!t) return;
    t.specs.splice(t.index, 1);
    this.removeEventKey(t.state, ref.event, t.specs);
    this._selectedEdge = null;
    this.commitEdit();
  }

  /** Prompt to rename the event of a transition. */
  private promptEditEvent(ref: EdgeRef): void {
    const name = this.promptValue("Event name:", ref.event);
    if (!name || name === ref.event) return;
    const t = this.findTransition(ref);
    if (!t) return;
    const [spec] = t.specs.splice(t.index, 1);
    this.removeEventKey(t.state, ref.event, t.specs);
    (t.state.on ??= {})[name] ??= [];
    t.state.on[name]!.push(spec);
    this._selectedEdge = null;
    this.commitEdit();
  }

  /** Prompt to change the target state of a transition. */
  private promptRetarget(ref: EdgeRef): void {
    const target = this.promptValue("Target state id:", ref.to);
    if (!target || target === ref.to) return;
    if (!this.boxMap.has(target)) { this.showToast(`Unknown state id "${target}"`); return; }
    const t = this.findTransition(ref);
    if (!t) return;
    t.specs[t.index]!.target = target;
    this._selectedEdge = null;
    this.commitEdit();
  }

  /** Prompt to set or clear the guard expression of a transition. */
  private promptEditGuard(ref: EdgeRef): void {
    const t = this.findTransition(ref);
    if (!t) return;
    const guard = this.promptValue("Guard expression (empty to clear):", t.specs[t.index]!.guard ?? "");
    if (guard === null) return;
    const spec = t.specs[t.index]!;
    if (guard) spec.guard = guard; else delete spec.guard;
    this.commitEdit();
  }

  // ── State creation ─────────────────────────────────────────────────────────

  /** Prompt for a name and build a new state node of the given type. */
  private promptNewState(type: string): RawStateNode | null {
    if (this.guardReadOnly()) return null;
    const name = this.promptValue(`${type} state name:`);
    if (!name) return null;
    const id = this.newStateId(name);
    const node: RawStateNode = { id, name, type };
    if (type === "compound") node.states = [];
    if (type === "parallel") node.regions = [{ id: `${id}_r1`, name: "Region 1", states: [] }];
    return node;
  }

  /** Add a new state of the given type at the top level. */
  private addTopState(type: string): void {
    const node = this.promptNewState(type);
    if (!node) return;
    this.topStates.push(node);
    this.commitEdit();
  }

  /** Show the state-type submenu for adding a child under a compound/parallel state. */
  private beginAddChild(parentId: string, clientX: number, clientY: number): void {
    this.showContextMenu(
      STATE_TYPES.map(t => ({ label: t.label, action: () => this.addChildState(parentId, t.type) })),
      clientX, clientY,
    );
  }

  /** Add a new state of the given type as a child of the given compound/parallel state. */
  private addChildState(parentId: string, type: string): void {
    const node = this.promptNewState(type);
    if (!node) return;
    const parent = this.findStateInTree(this.topStates, parentId);
    if (!parent) return;
    if (parent.type === "parallel") {
      parent.regions ??= [];
      if (!parent.regions.length) parent.regions.push({ id: `${parent.id}_r1`, name: "Region 1", states: [] });
      parent.regions[0]!.states.push(node);
    } else {
      parent.states ??= [];
      parent.states.push(node);
    }
    this.commitEdit();
  }

  /** Find the node id whose group contains the element at the given viewport point. */
  private nodeGroupAt(cx: number, cy: number, excludeId?: string): string | null {
    const el = document.elementFromPoint(cx, cy);
    if (!el) return null;
    for (const [nid, g] of this.groupMap) {
      if (nid === excludeId) continue;
      if (g === el || g.contains(el)) return nid;
    }
    return null;
  }

  // ── Effective position ────────────────────────────────────────────────────
  // Returns the screen position of a box = base + its own drag delta.
  // Parent drag deltas are handled by the SVG group transform on the parent <g>,
  // so children inherit the visual shift without needing to know about it.

  private ex(b: NodeBox): number { return b.bx + (this.dragOffsets.get(b.id)?.dx ?? 0); }
  private ey(b: NodeBox): number { return b.by + (this.dragOffsets.get(b.id)?.dy ?? 0); }

  // ── Canvas sizing ─────────────────────────────────────────────────────────
  // The SVG must cover the whole visible wrapper area (so the entire "blue
  // window" is drawable) AND all content — including nodes moved by drags —
  // with padding, so nothing is ever clipped inside the visible area.

  /**
   * Union rect of all node effective positions (base + committed drag offset).
   * `extra` applies the in-flight drag delta to the given descendant ids —
   * during a compound drag their offsets are not committed yet.
   */
  private contentBounds(extra?: { ids: Set<string>; dx: number; dy: number }): Bounds {
    let minX = 0, minY = 0, maxX = 0, maxY = 0;
    for (const b of this.allBoxList) {
      const o = this.dragOffsets.get(b.id);
      let x = b.bx + (o?.dx ?? 0);
      let y = b.by + (o?.dy ?? 0);
      if (extra?.ids.has(b.id)) { x += extra.dx; y += extra.dy; }
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + b.w);
      maxY = Math.max(maxY, y + b.h);
    }
    return { minX, minY, maxX, maxY };
  }

  /** Grow the SVG (never shrink) to cover the visible area and all content. */
  private growCanvas(extra?: { ids: Set<string>; dx: number; dy: number }): void {
    const svg = this.svg;
    if (!svg) return;
    const wrapper = svg.parentElement;
    const cw = wrapper instanceof HTMLElement ? wrapper.clientWidth  : 0;
    const ch = wrapper instanceof HTMLElement ? wrapper.clientHeight : 0;
    const b  = this.contentBounds(extra);
    const w  = Math.max(300, cw, Math.ceil(b.maxX) + CPD_PAD * 2);
    const h  = Math.max(200, ch, Math.ceil(b.maxY) + CPD_PAD * 2);
    if (w > parseFloat(svg.getAttribute("width")  ?? "0")) svg.setAttribute("width",  `${w}`);
    if (h > parseFloat(svg.getAttribute("height") ?? "0")) svg.setAttribute("height", `${h}`);
  }

  /**
   * Shift every drag offset so no node sits at negative coordinates (SVG
   * content left of/above the origin is unreachable by scrolling). Pure map
   * mutation — callers refresh transforms/edges as needed.
   */
  private normalizeNegativeOffsets(): { gx: number; gy: number } {
    const b = this.contentBounds();
    const gx = b.minX < 0 ? CPD_PAD - b.minX : 0;
    const gy = b.minY < 0 ? CPD_PAD - b.minY : 0;
    if (gx || gy) {
      for (const box of this.allBoxList) {
        const o = this.dragOffsets.get(box.id) ?? { dx: 0, dy: 0 };
        this.dragOffsets.set(box.id, { dx: o.dx + gx, dy: o.dy + gy });
      }
    }
    return { gx, gy };
  }

  /** After a drag ends: fit the canvas and re-anchor content that crossed the top/left edge. */
  private finalizeDrag(): void {
    const { gx, gy } = this.normalizeNegativeOffsets();
    if (gx || gy) {
      for (const b of this.allBoxList) {
        const o = this.dragOffsets.get(b.id)!;
        this.groupMap.get(b.id)?.setAttribute("transform", `translate(${o.dx},${o.dy})`);
      }
    }
    this.growCanvas();
    if (gx || gy) {
      // Content moved right/down by (gx,gy) — shift the scroll view by the
      // same amount so the visible content stays put.
      const wrapper = this.svg?.parentElement;
      if (wrapper instanceof HTMLElement) {
        wrapper.scrollLeft += gx;
        wrapper.scrollTop  += gy;
      }
      this.renderEdgeLayer();
    }
  }

  // ── Container layer ───────────────────────────────────────────────────────

  private renderContainerLayer(boxes: NodeBox[]): void {
    this.containerLayer!.innerHTML = "";
    this.renderContainers(this.containerLayer!, boxes);
  }

  private renderContainers(layer: SVGGElement, boxes: NodeBox[]): void {
    for (const b of boxes) {
      if (b.type !== "compound" && b.type !== "parallel") continue;

      // Compound/parallel outer box — wrapped in a <g> so the whole thing moves together
      const g = document.createElementNS(this.NS, "g") as SVGGElement;
      g.style.cursor = "grab";
      this.groupMap.set(b.id, g);

      const rect = document.createElementNS(this.NS, "rect");
      rect.setAttribute("x",      `${b.bx}`);
      rect.setAttribute("y",      `${b.by}`);
      rect.setAttribute("width",  `${b.w}`);
      rect.setAttribute("height", `${b.h}`);
      rect.setAttribute("rx", "8");
      rect.setAttribute("fill",   C.compoundFill);
      rect.setAttribute("stroke", C.compoundStroke);
      rect.setAttribute("stroke-width", "1.5");
      g.appendChild(rect);

      const title = document.createElementNS(this.NS, "text");
      title.setAttribute("x", `${b.bx + 10}`);
      title.setAttribute("y", `${b.by + 15}`);
      title.setAttribute("fill", C.regionLabel);
      title.setAttribute("font-size", "10");
      title.setAttribute("font-family", "system-ui, sans-serif");
      title.setAttribute("font-weight", "600");
      title.textContent = b.name;
      g.appendChild(title);

      // Type badge, top-right of the container (every node states its type)
      const cBadge = document.createElementNS(this.NS, "text");
      cBadge.setAttribute("x", `${b.bx + b.w - 8}`);
      cBadge.setAttribute("y", `${b.by + 15}`);
      cBadge.setAttribute("text-anchor", "end");
      cBadge.setAttribute("fill", C.regionLabel);
      cBadge.setAttribute("font-size", "9");
      cBadge.setAttribute("font-family", "system-ui, sans-serif");
      cBadge.setAttribute("pointer-events", "none");
      cBadge.textContent = b.type;
      g.appendChild(cBadge);

      // Region lanes
      for (const r of b.regions ?? []) {
        const rRect = document.createElementNS(this.NS, "rect");
        // Region base position is absolute
        rRect.setAttribute("x",      `${r.bx}`);
        rRect.setAttribute("y",      `${r.by}`);
        rRect.setAttribute("width",  `${r.w}`);
        rRect.setAttribute("height", `${r.h}`);
        rRect.setAttribute("rx", "5");
        rRect.setAttribute("fill",   C.regionFill);
        rRect.setAttribute("stroke", C.regionStroke);
        rRect.setAttribute("stroke-width", "1");
        g.appendChild(rRect);

        const rLabel = document.createElementNS(this.NS, "text");
        rLabel.setAttribute("x", `${r.bx + 8}`);
        rLabel.setAttribute("y", `${r.by + 14}`);
        rLabel.setAttribute("fill", C.regionLabel);
        rLabel.setAttribute("font-size", "9");
        rLabel.setAttribute("font-family", "system-ui, sans-serif");
        rLabel.textContent = r.name;
        g.appendChild(rLabel);
      }

      // Apply current drag offset as a transform on the group
      const dx = this.dragOffsets.get(b.id)?.dx ?? 0;
      const dy = this.dragOffsets.get(b.id)?.dy ?? 0;
      if (dx !== 0 || dy !== 0) g.setAttribute("transform", `translate(${dx},${dy})`);

      g.addEventListener("click", () => {
        if (this._selectedEdge) this.selectEdge(null);
        this.state.selectState(b.id);
      });

      // ── Right-click on compound/parallel → Rename / Add child / Delete ─────
      g.addEventListener("contextmenu", (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (this._suppressNextContextMenu) { this._suppressNextContextMenu = false; return; }
        if (this.guardReadOnly()) return;
        const { clientX, clientY } = e;
        this.showContextMenu([
          { label: "Rename…",                   action: () => this.beginInlineRename(b.id) },
          { label: "Add child state…",          action: () => this.beginAddChild(b.id, clientX, clientY) },
          { label: "Add transition from here…", action: () => this.beginAddTransition(b.id) },
          { label: "Delete state",              action: () => this.deleteState(b.id) },
        ], clientX, clientY);
      });

      this.attachDrag(b.id, g);
      layer.appendChild(g);

      // Recurse into compound children (they get their own containers)
      this.renderContainers(layer, b.children);
      for (const r of b.regions ?? []) this.renderContainers(layer, r.children);
    }
  }

  // ── Node layer ────────────────────────────────────────────────────────────

  private renderNodeLayer(boxes: NodeBox[]): void {
    this.nodeLayer!.innerHTML = "";
    this.renderNodes(this.nodeLayer!, boxes);
  }

  private renderNodes(layer: SVGGElement, boxes: NodeBox[]): void {
    for (const b of boxes) {
      if (b.type === "compound" || b.type === "parallel") {
        // Recurse — their children may be leaf nodes
        this.renderNodes(layer, b.children);
        for (const r of b.regions ?? []) this.renderNodes(layer, r.children);
        continue;
      }

      const isActive   = this.state.isStateActive(b.id);
      const isSelected = this.state.selectedStateId === b.id;

      const fill = isActive    ? C.nodeFillActive  :
                   b.isInitial ? C.nodeFillInitial  :
                   b.type === "final" ? C.nodeFillFinal  :
                   b.type === "human" ? C.nodeFillHuman  : C.nodeFill;
      const stroke = isSelected ? C.nodeStrokeSel :
                     isActive   ? C.nodeStrokeAct  : C.nodeStroke;
      const sw   = isSelected ? "2.5" : "1.5";
      const dash = b.isInitial && !isSelected ? "4 2" : "none";

      const g = document.createElementNS(this.NS, "g") as SVGGElement;
      g.style.cursor = "grab";
      this.groupMap.set(b.id, g);

      const rect = document.createElementNS(this.NS, "rect");
      rect.setAttribute("x",      `${b.bx}`);
      rect.setAttribute("y",      `${b.by}`);
      rect.setAttribute("width",  `${b.w}`);
      rect.setAttribute("height", `${b.h}`);
      rect.setAttribute("rx", "6");
      rect.setAttribute("fill",         fill);
      rect.setAttribute("stroke",       stroke);
      rect.setAttribute("stroke-width", sw);
      if (dash !== "none") rect.setAttribute("stroke-dasharray", dash);

      const text = document.createElementNS(this.NS, "text");
      text.setAttribute("x",           `${b.bx + b.w / 2}`);
      text.setAttribute("y",           `${b.by + b.h / 2 + 4}`);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("fill",        C.nodeText);
      text.setAttribute("font-size",   "12");
      text.setAttribute("font-family", "system-ui, sans-serif");
      text.setAttribute("pointer-events", "none");
      text.textContent = b.name;
      g.appendChild(rect);
      g.appendChild(text);

      // Type badge (every node states its type on the canvas)
      const badge = document.createElementNS(this.NS, "text");
      badge.setAttribute("x",           `${b.bx + b.w / 2}`);
      badge.setAttribute("y",           `${b.by + b.h / 2 + 16}`);
      badge.setAttribute("text-anchor", "middle");
      badge.setAttribute("fill",        C.nodeTextMuted);
      badge.setAttribute("font-size",   "9");
      badge.setAttribute("font-family", "system-ui, sans-serif");
      badge.setAttribute("pointer-events", "none");
      badge.textContent = b.type;
      g.appendChild(badge);

      // Apply current drag offset
      const dx = this.dragOffsets.get(b.id)?.dx ?? 0;
      const dy = this.dragOffsets.get(b.id)?.dy ?? 0;
      if (dx !== 0 || dy !== 0) g.setAttribute("transform", `translate(${dx},${dy})`);

      g.addEventListener("click", () => {
        if (this._selectedEdge) this.selectEdge(null);
        this.state.selectState(b.id);
      });

      // ── Right-click on leaf node → Delete / Rename / Add transition ─────────
      g.addEventListener("contextmenu", (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (this._suppressNextContextMenu) { this._suppressNextContextMenu = false; return; }
        if (this.guardReadOnly()) return;
        this.showContextMenu([
          { label: "Rename…",                 action: () => this.beginInlineRename(b.id) },
          { label: "Add transition from here…", action: () => this.beginAddTransition(b.id) },
          { label: "Delete state",            action: () => this.deleteState(b.id) },
        ], e.clientX, e.clientY);
      });

      // ── Double-click → inline rename ─────────────────────────────────────
      g.addEventListener("dblclick", (e: MouseEvent) => {
        e.stopPropagation();
        if (this.guardReadOnly()) return;
        this.beginInlineRename(b.id);
      });

      this.attachDrag(b.id, g);
      layer.appendChild(g);
    }
  }

  // ── Edge layer ────────────────────────────────────────────────────────────

  private renderEdgeLayer(): void {
    const layer = this.edgeLayer!;
    layer.innerHTML = "";
    const edges: EdgeRef[] = [];
    collectEdges(this.topStates, edges);

    let selectionFound = false;
    for (const edge of edges) {
      const srcBox = this.boxMap.get(edge.from);
      const dstBox = this.boxMap.get(edge.to);
      if (!srcBox || !dstBox) continue;
      const sel = this._selectedEdge != null &&
        this._selectedEdge.from === edge.from &&
        this._selectedEdge.event === edge.event &&
        this._selectedEdge.to === edge.to;
      if (sel) selectionFound = true;
      this.drawEdge(layer, srcBox, dstBox, edge, edge.from === edge.to, sel);
    }
    // Drop selection if the edge no longer exists (deleted / edited away)
    if (!selectionFound && this._selectedEdge) this._selectedEdge = null;
  }

  private drawEdge(
    layer: SVGGElement,
    src: NodeBox, dst: NodeBox,
    edge: EdgeRef,
    isSelf: boolean,
    selected: boolean,
  ): void {
    // Use effective positions (base + drag delta)
    const sx = this.ex(src), sy = this.ey(src);
    const dx = this.ex(dst), dy = this.ey(dst);

    let d: string; let lx: number; let ly: number;
    if (isSelf) {
      const cx = sx + src.w / 2;
      const cy = sy;
      d = `M ${cx - 18} ${cy} Q ${cx} ${cy - 36} ${cx + 18} ${cy}`;
      lx = cx; ly = cy - 26;
    } else {
      const x1 = sx + src.w / 2;
      const y1 = sy + src.h;
      const x2 = dx + dst.w / 2;
      const y2 = dy;

      const goesUp = y2 <= y1 + NODE_H / 2;
      if (goesUp) {
        const ex = Math.max(sx + src.w, dx + dst.w) + H_GAP;
        const ty = dy + dst.h / 2;
        d = `M ${x1} ${y1} L ${x1} ${y1 + 10} L ${ex} ${y1 + 10} L ${ex} ${ty} L ${x2} ${ty}`;
        lx = (x1 + Math.max(sx + src.w, dx + dst.w) + H_GAP) / 2;
        ly = y1 + 6;
      } else {
        const my = (y1 + y2) / 2;
        d = `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;
        lx = (x1 + x2) / 2;
        ly = my - 4;
      }
    }

    const path = document.createElementNS(this.NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", selected ? C.nodeStrokeSel : C.edgeStroke);
    path.setAttribute("stroke-width", selected ? "2.2" : "1.2");
    path.setAttribute("marker-end", "url(#wf-arrow)");
    path.setAttribute("pointer-events", "none");
    layer.appendChild(path);

    // Wide invisible hit area so the transition can be selected / edited
    const hit = document.createElementNS(this.NS, "path");
    hit.setAttribute("class", "wf-edge-hit");
    hit.setAttribute("d", d);
    hit.setAttribute("fill", "none");
    hit.setAttribute("stroke", "transparent");
    hit.setAttribute("stroke-width", "12");
    hit.setAttribute("pointer-events", "stroke");
    hit.style.cursor = "pointer";
    layer.appendChild(hit);

    const ref: EdgeRef = { from: edge.from, event: edge.event, to: edge.to };
    hit.addEventListener("click", (e: MouseEvent) => {
      e.stopPropagation();
      this.selectEdge(ref);
    });
    hit.addEventListener("dblclick", (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (this.guardReadOnly()) return;
      this.promptEditEvent(ref);
    });
    hit.addEventListener("contextmenu", (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (this._suppressNextContextMenu) { this._suppressNextContextMenu = false; return; }
      if (this.guardReadOnly()) return;
      this.selectEdge(ref);
      this.showContextMenu([
        { label: "Edit event…",       action: () => this.promptEditEvent(ref) },
        { label: "Change target…",    action: () => this.promptRetarget(ref) },
        { label: "Edit guard…",       action: () => this.promptEditGuard(ref) },
        { label: "Delete transition", action: () => this.deleteTransition(ref) },
      ], e.clientX, e.clientY);
    });

    const label = document.createElementNS(this.NS, "text");
    label.setAttribute("x", `${lx}`);
    label.setAttribute("y", `${ly}`);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("fill", selected ? C.nodeStrokeSel : C.edgeLabelText);
    label.setAttribute("font-size", "9");
    label.setAttribute("font-family", "system-ui, sans-serif");
    label.setAttribute("pointer-events", "none");
    label.textContent = edge.event;
    layer.appendChild(label);
  }

  // ── Drag ──────────────────────────────────────────────────────────────────

  private attachDrag(id: string, g: SVGGElement): void {
    if (!this.svg) return;
    const svg = this.svg;

    g.addEventListener("mousedown", (e: MouseEvent) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      g.style.cursor = "grabbing";

      const startPt  = toSvgPoint(svg, e);
      const startOff = { ...(this.dragOffsets.get(id) ?? { dx: 0, dy: 0 }) };

      // Collect all descendant IDs so we can shift them together with the parent
      const box = this.boxMap.get(id);
      const descendantIds  = box ? collectDescendantIds(box) : [];
      const descendantSet  = new Set(descendantIds);

      const onMove = (e: MouseEvent) => {
        const pt  = toSvgPoint(svg, e);
        const ddx = pt.x - startPt.x;
        const ddy = pt.y - startPt.y;
        const newOff = { dx: startOff.dx + ddx, dy: startOff.dy + ddy };

        // Update this node's offset
        this.dragOffsets.set(id, newOff);

        // Move the SVG group via transform (base coords in attrs stay the same)
        g.setAttribute("transform", `translate(${newOff.dx},${newOff.dy})`);

        // Move each descendant's SVG group by the same delta
        for (const did of descendantIds) {
          const dg = this.groupMap.get(did);
          if (!dg) continue;
          // Move the group visually using the parent delta on top of own offset
          const ownOff = this.dragOffsets.get(did) ?? { dx: 0, dy: 0 };
          dg.setAttribute("transform", `translate(${ownOff.dx + ddx},${ownOff.dy + ddy})`);
        }

        // Redraw edges using updated effective positions
        this.renderEdgeLayer();

        // Grow the canvas if the drag extends beyond the current SVG bounds
        this.growCanvas({ ids: descendantSet, dx: ddx, dy: ddy });
      };

      const onUp = () => {
        g.style.cursor = "grab";

        // Commit the visual positions of descendants into their offsets
        if (box) {
          const finalOff = this.dragOffsets.get(id) ?? { dx: 0, dy: 0 };
          const delta = { dx: finalOff.dx - startOff.dx, dy: finalOff.dy - startOff.dy };
          for (const did of descendantIds) {
            const prev = this.dragOffsets.get(did) ?? { dx: 0, dy: 0 };
            this.dragOffsets.set(did, { dx: prev.dx + delta.dx, dy: prev.dy + delta.dy });
            // Ensure group transform matches committed offset
            const dg = this.groupMap.get(did);
            const co = this.dragOffsets.get(did)!;
            dg?.setAttribute("transform", `translate(${co.dx},${co.dy})`);
          }
        }

        // Fit the canvas and re-anchor any content dragged past the top/left edge
        this.finalizeDrag();

        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    // ── Right-drag → create a transition to the node under the cursor ────────
    // A plain right-click (no movement) still shows the context menu via the
    // contextmenu event; a completed drag suppresses it.
    g.addEventListener("mousedown", (e: MouseEvent) => {
      if (e.button !== 2) return;
      const start = { x: e.clientX, y: e.clientY };
      const sourceBox = this.boxMap.get(id);
      if (!sourceBox) return;

      let overlay: SVGPathElement | null = null;
      let moved = false;
      let hoverId: string | null = null;

      const clearHover = () => {
        if (!hoverId) return;
        const rect = this.groupMap.get(hoverId)?.querySelector("rect");
        if (rect) { rect.style.stroke = ""; rect.style.strokeWidth = ""; }
        hoverId = null;
      };

      const onMove = (ev: MouseEvent) => {
        if (!(ev.buttons & 2)) return;
        if (!moved && Math.hypot(ev.clientX - start.x, ev.clientY - start.y) < 5) return;
        moved = true;
        if (!overlay) {
          overlay = document.createElementNS(this.NS, "path");
          overlay.setAttribute("fill", "none");
          overlay.setAttribute("stroke", C.nodeStrokeSel);
          overlay.setAttribute("stroke-width", "1.5");
          overlay.setAttribute("stroke-dasharray", "5 3");
          overlay.setAttribute("pointer-events", "none");
          svg.appendChild(overlay);
          document.body.style.cursor = "crosshair";
        }
        const pt = toSvgPoint(svg, ev);
        const sx = this.ex(sourceBox) + sourceBox.w;
        const sy = this.ey(sourceBox) + sourceBox.h / 2;
        overlay.setAttribute("d", `M ${sx} ${sy} L ${pt.x} ${pt.y}`);

        // Highlight the potential drop target under the cursor
        const targetId = this.nodeGroupAt(ev.clientX, ev.clientY, id);
        if (targetId !== hoverId) {
          clearHover();
          hoverId = targetId;
          const rect = targetId ? this.groupMap.get(targetId)?.querySelector("rect") : null;
          if (rect) { rect.style.stroke = C.nodeStrokeSel; rect.style.strokeWidth = "2.5"; }
        }
      };

      const onUp = (ev: MouseEvent) => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        clearHover();
        document.body.style.cursor = "";
        overlay?.remove();
        if (!moved) return; // plain right-click → context menu (contextmenu event)

        // Swallow the contextmenu event that follows a completed right-drag
        this._suppressNextContextMenu = true;
        setTimeout(() => { this._suppressNextContextMenu = false; }, 100);

        const targetId = this.nodeGroupAt(ev.clientX, ev.clientY, id);
        if (!targetId) return;
        const eventName = this.promptValue(`Event for transition ${id} → ${targetId}:`, "done");
        if (!eventName) return;
        this.addTransitionInTree(this.topStates, id, eventName, targetId);
        this.commitEdit();
      };

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });
  }

  // ── Inline rename ─────────────────────────────────────────────────────────

  /** Overlay a text <input> over the node label for in-place rename. */
  private beginInlineRename(id: string): void {
    const box = this.boxMap.get(id);
    const svgEl = this.svg;
    if (!box || !svgEl) return;

    // Get the SVG element's bounding rect relative to viewport
    const svgRect = svgEl.getBoundingClientRect();
    const dx = this.dragOffsets.get(id)?.dx ?? 0;
    const dy = this.dragOffsets.get(id)?.dy ?? 0;

    // Scale factor between SVG logical units and CSS pixels
    const scaleX = svgRect.width  / parseFloat(svgEl.getAttribute("width")  ?? "1");
    const scaleY = svgRect.height / parseFloat(svgEl.getAttribute("height") ?? "1");

    const inputLeft = svgRect.left + (box.bx + dx + 4)       * scaleX;
    const inputTop  = svgRect.top  + (box.by + dy + box.h / 2 - 10) * scaleY;
    const inputW    = (box.w - 8) * scaleX;

    const input = document.createElement("input");
    input.type = "text";
    input.value = box.name;
    input.style.cssText =
      `position:fixed;left:${inputLeft}px;top:${inputTop}px;width:${inputW}px;` +
      `height:20px;font-size:12px;font-family:system-ui,sans-serif;` +
      `background:#2a2a3d;color:#cdd6f4;border:1px solid #89b4fa;` +
      `border-radius:3px;padding:0 4px;z-index:9999;box-sizing:border-box;outline:none;`;
    document.body.appendChild(input);
    input.focus();
    input.select();

    const commit = () => {
      const newName = input.value.trim();
      input.remove();
      if (!newName || newName === box.name) return;
      // Patch the state node in topStates
      this.renameStateInTree(this.topStates, id, newName);
      this.commitEdit();
    };

    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter")  { e.preventDefault(); commit(); }
      if (e.key === "Escape") { input.removeEventListener("blur", commit); input.remove(); }
    });
  }

  private renameStateInTree(states: RawStateNode[], id: string, newName: string): void {
    for (const s of states) {
      if (s.id === id) { s.name = newName; return; }
      if (s.states) this.renameStateInTree(s.states, id, newName);
      if (s.regions) for (const r of s.regions) this.renameStateInTree(r.states, id, newName);
    }
  }

  // ── Add transition ────────────────────────────────────────────────────────

  /** Start the "click target node" interaction for adding a transition. */
  private beginAddTransition(fromId: string): void {
    const svgEl = this.svg;
    if (!svgEl) return;
    this.showToast(`Click target state for transition from "${fromId}"… (Esc to cancel)`);

    // Highlight all node groups as potential targets
    for (const [nid, g] of this.groupMap) {
      if (nid === fromId) continue;
      g.style.outline = "1px dashed #89b4fa";
    }

    const cleanup = () => {
      for (const g of this.groupMap.values()) g.style.outline = "";
      svgEl.removeEventListener("click", onTargetClick);
      document.removeEventListener("keydown", onEsc);
    };

    const onTargetClick = (e: MouseEvent) => {
      e.stopPropagation();
      // Find which group was clicked
      let el: Element | null = e.target as Element;
      let toId: string | null = null;
      while (el && el !== svgEl) {
        for (const [nid, g] of this.groupMap) {
          if (g === el || g.contains(el)) { toId = nid; break; }
        }
        if (toId) break;
        el = el.parentElement;
      }
      cleanup();
      if (!toId || toId === fromId) return;

      const eventName = this.promptValue("Event name for transition:", "done");
      if (!eventName) return;

      // Patch the from-state's `on` record
      this.addTransitionInTree(this.topStates, fromId, eventName, toId);
      this.commitEdit();
    };

    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") cleanup();
    };

    // Attach to svg with a short delay so the current contextmenu click doesn't fire it
    setTimeout(() => {
      svgEl.addEventListener("click", onTargetClick);
      document.addEventListener("keydown", onEsc);
    }, 100);
  }

  private addTransitionInTree(
    states: RawStateNode[],
    fromId: string,
    eventName: string,
    toId: string,
  ): void {
    for (const s of states) {
      if (s.id === fromId) {
        if (!s.on) s.on = {};
        if (!s.on[eventName]) s.on[eventName] = [];
        s.on[eventName].push({ target: toId });
        return;
      }
      if (s.states) this.addTransitionInTree(s.states, fromId, eventName, toId);
      if (s.regions) for (const r of s.regions) this.addTransitionInTree(r.states, fromId, eventName, toId);
    }
  }

  protected override onEditorEvent(event: EditorEvent): void {
    if (event.type === "workflow-changed" && !this._suppressNextDragClear) {
      this.dragOffsets?.clear();
    }
    if (
      event.type === "workflow-changed" ||
      event.type === "run-snapshot-updated" ||
      event.type === "state-selected"
    ) {
      this.refresh();
    }
  }
}

// ── Layout engine ─────────────────────────────────────────────────────────────

/**
 * Layout a list of sibling states.
 * Top-level states (depth=0) flow top-to-bottom so tall compound/parallel
 * containers don't produce an impossibly wide horizontal scroll.
 * Nested states (inside a compound or parallel region) flow left-to-right
 * since containers constrain their width.
 */
function layoutStates(
  states: RawStateNode[], initial: string, ox: number, oy: number,
  horizontal = false,
): NodeBox[] {
  const boxes: NodeBox[] = [];
  let curX = ox, curY = oy;
  for (const s of states) {
    const b = layoutOne(s, s.id === initial, curX, curY);
    boxes.push(b);
    if (horizontal) {
      curX += b.w + H_GAP;
    } else {
      curY += b.h + H_GAP;
    }
  }
  return boxes;
}

function layoutOne(s: RawStateNode, isInitial: boolean, x: number, y: number): NodeBox {
  const type = s.type ?? "atomic";
  const name = s.name ?? s.id;

  if (type === "compound" && s.states?.length) {
    const childInit = s.initial ?? s.states[0]?.id ?? "";
    // Children inside a compound always flow left-to-right
    const children  = layoutStates(s.states, childInit, x + CPD_PAD, y + CPD_PAD + 24, true);
    const iW = innerWidth(children);
    const iH = innerHeight(children);
    return { id: s.id, type, name,
      bx: x, by: y,
      w: Math.max(NODE_W + CPD_PAD * 2, iW + CPD_PAD * 2),
      h: iH + CPD_PAD * 2 + 24,
      children, isInitial };
  }

  if (type === "parallel" && s.regions?.length) {
    return layoutParallel(s, isInitial, x, y);
  }

  return { id: s.id, type, name, bx: x, by: y, w: NODE_W, h: NODE_H, children: [], isInitial };
}

function layoutParallel(s: RawStateNode, isInitial: boolean, x: number, y: number): NodeBox {
  const name = s.name ?? s.id;
  const regions = s.regions ?? [];
  let curX = x + CPD_PAD;
  const regBoxes: RegionBox[] = [];

  for (const r of regions) {
    const rInit    = r.initial ?? r.states[0]?.id ?? "";
    const children = layoutStates(r.states, rInit, curX + CPD_PAD, y + CPD_PAD + 24 + 20);
    const rW = Math.max(NODE_W + CPD_PAD * 2, innerWidth(children) + CPD_PAD * 2);
    const rH = innerHeight(children) + CPD_PAD * 2 + 20;
    regBoxes.push({ id: r.id, name: r.name ?? r.id, bx: curX, by: y + CPD_PAD + 24, w: rW, h: rH, children });
    curX += rW + REGION_GAP;
  }

  const w = curX - x - REGION_GAP + CPD_PAD;
  const h = (regBoxes[0]?.h ?? NODE_H) + CPD_PAD * 2 + 24;
  const children = regBoxes.flatMap(r => r.children);
  return { id: s.id, type: "parallel", name, bx: x, by: y, w, h, children, regions: regBoxes, isInitial };
}

function innerWidth(boxes: NodeBox[]): number {
  if (!boxes.length) return NODE_W;
  // boxes may be horizontal (left-to-right) or vertical (top-to-bottom).
  // Width = max right edge relative to leftmost box.
  const left = boxes[0]!.bx;
  return boxes.reduce((m, b) => Math.max(m, b.bx - left + b.w), NODE_W);
}

function innerHeight(boxes: NodeBox[]): number {
  if (!boxes.length) return NODE_H;
  // boxes may be horizontal or vertical.
  // Height = max bottom edge relative to topmost box.
  const top = boxes[0]!.by;
  return boxes.reduce((m, b) => Math.max(m, b.by - top + b.h), NODE_H);
}

function* walkAll(boxes: NodeBox[]): Iterable<NodeBox> {
  for (const b of boxes) {
    yield b;
    yield* walkAll(b.children);
    for (const r of b.regions ?? []) yield* walkAll(r.children);
  }
}

function collectDescendantIds(box: NodeBox): string[] {
  const ids: string[] = [];
  function walk(b: NodeBox) {
    for (const c of b.children) { ids.push(c.id); walk(c); }
    for (const r of b.regions ?? []) for (const c of r.children) { ids.push(c.id); walk(c); }
  }
  walk(box);
  return ids;
}

function collectEdges(
  states: RawStateNode[],
  out: EdgeRef[],
): void {
  for (const s of states) {
    for (const [event, specs] of Object.entries(s.on ?? {})) {
      for (const spec of specs) out.push({ from: s.id, event, to: spec.target });
    }
    if (s.states)  collectEdges(s.states, out);
    if (s.regions) for (const r of s.regions) collectEdges(r.states, out);
  }
}

function toSvgPoint(svg: SVGSVGElement, e: MouseEvent): { x: number; y: number } {
  const pt  = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  const ctm = svg.getScreenCTM();
  if (ctm) { const t = pt.matrixTransform(ctm.inverse()); return { x: t.x, y: t.y }; }
  return { x: e.clientX, y: e.clientY };
}
