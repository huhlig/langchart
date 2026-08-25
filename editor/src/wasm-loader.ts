/**
 * langchart-wasm loader.
 *
 * Initialises the WASM module once and exposes a typed API surface.
 * Import this module before calling any WASM functions.
 *
 * Build the WASM package first:
 *   npm run wasm:build
 *
 * Then the `src/wasm/` directory will contain the generated bindings.
 * In dev/CI without the WASM build, all functions return empty stubs so
 * the editor shell renders without requiring a Rust toolchain.
 */

// ── Types returned by the WASM API ───────────────────────────────────────────

export interface Diagnostic {
  code: string;
  severity: "error" | "warning";
  message: string;
  location: string;
}

export interface CompileResult {
  ok: boolean;
  errors: Diagnostic[];
}

export interface StateInspection {
  id: string;
  name: string;
  type:
    | "atomic"
    | "agentic"
    | "compound"
    | "parallel"
    | "human"
    | "subworkflow"
    | "final";
  is_initial: boolean;
  transitions: TransitionInfo[];
  agent?: string;
  prompt?: string;
  has_limits: boolean;
  has_capabilities: boolean;
  child_count: number;
  region_count: number;
}

export interface TransitionInfo {
  event: string;
  target: string;
  guard?: string;
  priority: number;
}

export interface WorkflowSummary {
  id: string;
  version: string;
  name: string;
  initial: string;
  schema_version: string;
  state_counts: StateCounts;
  agent_ids: string[];
}

export interface StateCounts {
  total: number;
  atomic: number;
  agentic: number;
  compound: number;
  parallel: number;
  human: number;
  subworkflow: number;
  final_states: number;
}

export interface TransitionEdge {
  from: string;
  event: string;
  to: string;
  guard?: string;
  priority: number;
}

export interface ReachabilityResult {
  reachable: string[];
  unreachable: string[];
}

export interface GuardError {
  state_id: string;
  event_type: string;
  error: string;
}

export interface SimulationStepRecord {
  step: number;
  active_state: string;
  event: string;
  target: string;
}

export interface SimulationResult {
  status: "completed" | "running" | "stuck";
  final_state: string;
  steps: SimulationStepRecord[];
  error: string | null;
}

export interface SimulationInput {
  actors: Record<string, { emit: string }>;
  inject: Array<{ event_type: string }>;
  max_steps?: number;
}

// ── WASM module interface ─────────────────────────────────────────────────────

interface WasmModule {
  schema_version(): string;
  validateWorkflow(json: string): string;
  compileWorkflow(json: string): string;
  listStateIds(json: string): string;
  inspectState(json: string, state_id: string): string;
  getGuardErrors(json: string): string;
  workflowSummary(json: string): string;
  listTransitions(json: string): string;
  reachabilityAnalysis(json: string): string;
  simulateWorkflow(workflow_json: string, simulation_json: string): string;
}

// ── Pure-TypeScript fallback (used when WASM is not built yet) ────────────────
//
// These functions parse the workflow JSON directly so the canvas renders even
// without running `npm run wasm:build`.

interface RawState {
  id: string;
  name?: string;
  type?: string;
  initial?: string;
  agent?: { id: string; version: string };
  on?: Record<string, Array<{ target: string; guard?: string; priority?: number }>>;
  states?: RawState[];
  regions?: Array<{ id: string; states: RawState[] }>;
}

interface RawWorkflow {
  schema_version?: string;
  id?: string;
  version?: string;
  name?: string;
  initial?: string;
  agents?: Array<{ id: string }>;
  states?: RawState[];
}

/** Collect all state IDs recursively (top-level + compound children + parallel regions). */
function collectStateIds(states: RawState[]): string[] {
  const ids: string[] = [];
  for (const s of states) {
    ids.push(s.id);
    if (s.states) ids.push(...collectStateIds(s.states));
    if (s.regions) {
      for (const r of s.regions) ids.push(...collectStateIds(r.states));
    }
  }
  return ids;
}

/** Find a state by id anywhere in the tree. */
function findState(states: RawState[], id: string): RawState | null {
  for (const s of states) {
    if (s.id === id) return s;
    if (s.states) { const f = findState(s.states, id); if (f) return f; }
    if (s.regions) { for (const r of s.regions) { const f = findState(r.states, id); if (f) return f; } }
  }
  return null;
}

/** Return all direct child IDs of a compound or parallel state. */
function childStateIds(s: RawState): string[] {
  const ids: string[] = [];
  if (s.states) for (const c of s.states) ids.push(c.id);
  if (s.regions) for (const r of s.regions) for (const c of r.states) ids.push(c.id);
  return ids;
}

/** Collect all transition edges recursively. */
function collectTransitions(states: RawState[]): TransitionEdge[] {
  const edges: TransitionEdge[] = [];
  for (const s of states) {
    for (const [event, specs] of Object.entries(s.on ?? {})) {
      for (const spec of specs) {
        const edge: TransitionEdge = {
          from: s.id,
          event,
          to: spec.target,
          priority: spec.priority ?? 0,
        };
        if (spec.guard !== undefined) edge.guard = spec.guard;
        edges.push(edge);
      }
    }
    if (s.states) edges.push(...collectTransitions(s.states));
    if (s.regions) {
      for (const r of s.regions) edges.push(...collectTransitions(r.states));
    }
  }
  return edges;
}

function countStates(states: RawState[]): StateCounts {
  const c: StateCounts = { total: 0, atomic: 0, agentic: 0, compound: 0, parallel: 0, human: 0, subworkflow: 0, final_states: 0 };
  for (const s of states) {
    c.total++;
    switch (s.type) {
      case "atomic":      c.atomic++;      break;
      case "agentic":     c.agentic++;     break;
      case "compound":    c.compound++;    break;
      case "parallel":    c.parallel++;    break;
      case "human":       c.human++;       break;
      case "subworkflow": c.subworkflow++; break;
      case "final":       c.final_states++; break;
    }
    if (s.states)   { const cc = countStates(s.states);   for (const k of Object.keys(c) as (keyof StateCounts)[]) c[k] += cc[k]; }
    if (s.regions)  { for (const r of s.regions) { const cc = countStates(r.states); for (const k of Object.keys(c) as (keyof StateCounts)[]) c[k] += cc[k]; } }
  }
  return c;
}

function parseWorkflow(json: string): RawWorkflow {
  try { return JSON.parse(json) as RawWorkflow; } catch { return {}; }
}

const stub: WasmModule = {
  schema_version: () => "1.0.0",

  validateWorkflow: () => "[]",

  compileWorkflow: () => '{"ok":true,"errors":[]}',

  listStateIds: (json) => {
    const doc = parseWorkflow(json);
    return JSON.stringify(collectStateIds(doc.states ?? []));
  },

  inspectState: (json, stateId) => {
    const doc = parseWorkflow(json);
    const s = findState(doc.states ?? [], stateId);
    if (!s) return "null";
    const transitions = Object.entries(s.on ?? {}).flatMap(([event, specs]) =>
      specs.map(spec => {
        const info: TransitionInfo = {
          event,
          target: spec.target,
          priority: spec.priority ?? 0,
        };
        if (spec.guard !== undefined) info.guard = spec.guard;
        return info;
      })
    );
    const info: StateInspection = {
      id:   s.id,
      name: s.name ?? s.id,
      type: (s.type ?? "atomic") as StateInspection["type"],
      is_initial: doc.initial === s.id,
      transitions,
      has_limits:       false,
      has_capabilities: false,
      child_count:      s.states?.length ?? 0,
      region_count:     s.regions?.length ?? 0,
    };
    if (s.agent?.id !== undefined) info.agent = s.agent.id;
    return JSON.stringify(info);
  },

  getGuardErrors: () => "[]",

  workflowSummary: (json) => {
    const doc = parseWorkflow(json);
    const states = doc.states ?? [];
    const counts = countStates(states);
    const summary: WorkflowSummary = {
      id:             doc.id             ?? "",
      version:        doc.version        ?? "",
      name:           doc.name           ?? "",
      initial:        doc.initial        ?? "",
      schema_version: doc.schema_version ?? "1.0.0",
      state_counts:   counts,
      agent_ids:      (doc.agents ?? []).map(a => a.id),
    };
    return JSON.stringify(summary);
  },

  listTransitions: (json) => {
    const doc = parseWorkflow(json);
    return JSON.stringify(collectTransitions(doc.states ?? []));
  },

  reachabilityAnalysis: (json) => {
    const doc = parseWorkflow(json);
    const allStates = doc.states ?? [];
    const allIds = collectStateIds(allStates);
    const transitions = collectTransitions(allStates);
    const reachable = new Set<string>();
    if (doc.initial) {
      const queue = [doc.initial];
      while (queue.length) {
        const cur = queue.shift()!;
        if (reachable.has(cur)) continue;
        reachable.add(cur);
        // When a compound/parallel state is reachable, its children are too —
        // the runtime enters their initial sub-state automatically.
        const s = findState(allStates, cur);
        if (s) {
          for (const childId of childStateIds(s)) {
            if (!reachable.has(childId)) queue.push(childId);
          }
        }
        for (const t of transitions) {
          if (t.from === cur && !reachable.has(t.to)) queue.push(t.to);
        }
      }
    }
    const result: ReachabilityResult = {
      reachable:   allIds.filter(id =>  reachable.has(id)),
      unreachable: allIds.filter(id => !reachable.has(id)),
    };
    return JSON.stringify(result);
  },

  simulateWorkflow: () =>
    '{"status":"stuck","final_state":"","steps":[],"error":"WASM not built — run npm run wasm:build"}',
};

// ── Module singleton ──────────────────────────────────────────────────────────

let wasmModule: WasmModule = stub;
let initialised = false;

/**
 * Initialise the WASM module. Call once before using any API functions.
 * Safe to call multiple times — subsequent calls are no-ops.
 */
export async function initWasm(): Promise<void> {
  if (initialised) return;
  try {
    // Dynamic import — will fail gracefully if wasm:build hasn't been run.
    // Keep the path non-literal so Vite does not require the optional generated
    // module during ordinary desktop builds.
    const modulePath = "./wasm/langchart_wasm.js";
    const wasm = await import(/* @vite-ignore */ modulePath);
    await (wasm as { default?: () => Promise<void> }).default?.();
    wasmModule = wasm as unknown as WasmModule;
    console.info(
      `[langchart-wasm] initialised (schema ${wasmModule.schema_version()})`
    );
  } catch (e) {
    console.warn(
      "[langchart-wasm] WASM module not found; using stub. Run `npm run wasm:build` to enable full validation.",
      e
    );
  }
  initialised = true;
}

// ── Typed API wrappers ────────────────────────────────────────────────────────

function parse<T>(raw: string): T {
  return JSON.parse(raw) as T;
}

export const wasm = {
  schemaVersion(): string {
    return wasmModule.schema_version();
  },

  validateWorkflow(json: string): Diagnostic[] {
    return parse<Diagnostic[]>(wasmModule.validateWorkflow(json));
  },

  compileWorkflow(json: string): CompileResult {
    return parse<CompileResult>(wasmModule.compileWorkflow(json));
  },

  listStateIds(json: string): string[] {
    return parse<string[]>(wasmModule.listStateIds(json));
  },

  inspectState(json: string, stateId: string): StateInspection | null {
    return parse<StateInspection | null>(wasmModule.inspectState(json, stateId));
  },

  getGuardErrors(json: string): GuardError[] {
    return parse<GuardError[]>(wasmModule.getGuardErrors(json));
  },

  workflowSummary(json: string): WorkflowSummary {
    return parse<WorkflowSummary>(wasmModule.workflowSummary(json));
  },

  listTransitions(json: string): TransitionEdge[] {
    return parse<TransitionEdge[]>(wasmModule.listTransitions(json));
  },

  reachabilityAnalysis(json: string): ReachabilityResult {
    return parse<ReachabilityResult>(wasmModule.reachabilityAnalysis(json));
  },

  simulateWorkflow(workflowJson: string, input: SimulationInput): SimulationResult {
    return parse<SimulationResult>(
      wasmModule.simulateWorkflow(workflowJson, JSON.stringify(input))
    );
  },
};
