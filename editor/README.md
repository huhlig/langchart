# langchart editor

Visual statechart editor for langchart agentic workflows. It can be embedded as
a set of TypeScript panels or run as a standalone, local-first browser editor.

## Overview

A TypeScript/vanilla-DOM component that renders and edits langchart workflow
documents. Uses the `langchart-wasm` WASM module for **continuous validation,
compilation checks, guard expression analysis, and reachability analysis** —
all without a network round-trip.

The editor is independently optional. The Rust runtime operates without it.

## Architecture

```
editor/
  src/
    index.html          — shell with 8 panel mount points + toolbar
    styles.css          — layout, panel, canvas, inspector styles
    main.ts             — bootstrap: init WASM, mount panels, wire toolbar
    wasm-loader.ts      — WASM module initialiser + typed API wrappers
    editor-state.ts     — central model (workflow JSON, selection, run snapshot)
    panels/
      panel-base.ts         — abstract Panel: subscribe/emit, DOM helpers
      workflow-canvas.ts    — Panel 1: statechart graph (nodes + edges)
      state-inspector.ts    — Panel 2: selected state — Basic / Advanced / Source
      problems-panel.ts     — Panel 3: diagnostics + unreachable-state warnings
      run-inspector.ts      — Panel 4: live run — status, active states, activities
      secondary-panels.ts   — Panels 5–8: Context, Capability, Artifact, Trace
```

## Panels

| # | Panel | Data source |
|---|-------|-------------|
| 1 | Workflow Canvas | `listStateIds`, `listTransitions`, run snapshot |
| 2 | State Inspector | `inspectState`, selected state ID |
| 3 | Problems | `validateWorkflow`, `reachabilityAnalysis` |
| 4 | Run Inspector | host app `RunSnapshot` |
| 5 | Context Inspector | host app `ContextResolvedEvent` |
| 6 | Capability Inspector | `inspectState`, compiled workflow |
| 7 | Artifact Review | host app `ArtifactStore` events |
| 8 | Trace Timeline | host app runtime event stream |

## Prerequisites

- Node.js ≥ 20
- Rust + `wasm-pack` (only for building the WASM module)

```bash
cargo install wasm-pack
```

## Getting started

```bash
# 1. Build the WASM module from the Rust crate
npm run wasm:build

# 2. Install JS dependencies
npm install

# 3. Start the dev server
npm run dev
```

The standalone editor is served at `http://localhost:5173`. It supports new,
open, drag-and-drop, source editing, automatic local draft recovery, keyboard
shortcuts, and export to portable `.langchart` files. A `.langchart` file is a
UTF-8 JSON `WorkflowDocument` and can also be consumed anywhere workflow JSON
is accepted.
Append `#example` to the URL to load a built-in example workflow.

Without running `wasm:build`, the editor still starts but uses stub
implementations that always return empty diagnostic lists. All panels render
correctly; validation is disabled.

## Build for production

```bash
npm run build
```

Output is in `dist/`.

## WASM API

The `langchart-wasm` crate exposes these functions (all accept/return JSON):

| Function | Description |
|----------|-------------|
| `schema_version()` | Library schema version string |
| `validateWorkflow(json)` | Structural + semantic diagnostics |
| `compileWorkflow(json)` | Compile check — `{ok, errors}` |
| `listStateIds(json)` | All state IDs in the document tree |
| `inspectState(json, id)` | Single-state details |
| `getGuardErrors(json)` | CEL guard compilation errors |
| `workflowSummary(json)` | Document summary + state type counts |
| `listTransitions(json)` | All transitions as flat edge list |
| `reachabilityAnalysis(json)` | Reachable / unreachable state sets |

## Host application integration

The editor is designed for embedding. Mount it in your application:

```typescript
import { initWasm, wasm } from "@langchart/editor/wasm-loader";
import { EditorState } from "@langchart/editor/editor-state";
import { WorkflowCanvas } from "@langchart/editor/panels/workflow-canvas";

await initWasm();

const state = new EditorState();
const canvas = new WorkflowCanvas(document.getElementById("canvas")!, state);

// Load a workflow.
state.loadJson(workflowJsonString);

// Push a live run snapshot from your runtime event stream.
state.updateRunSnapshot({
  runId: "run_01JXX",
  status: "running",
  activeStates: ["write"],
  eventQueueDepth: 2,
  activities: ["write/invocation_01JXX"],
});
```

## Phase 5 status

- [x] WASM module: all 9 validation + inspection functions implemented and tested
- [x] TypeScript types: full interface definitions for all WASM return types
- [x] Editor state: central model with subscription / event emission
- [x] Panel base: abstract DOM component with lifecycle management
- [x] Panel 1 — Workflow Canvas: SVG node/edge graph with active state highlighting
- [x] Panel 2 — State Inspector: Basic + Advanced + transition table
- [x] Panel 3 — Problems: diagnostics + reachability warnings, live update
- [x] Panel 4 — Run Inspector: live run snapshot display
- [x] Panel 5 — Context Inspector: skeleton (host app integration point)
- [x] Panel 6 — Capability Inspector: static config summary
- [x] Panel 7 — Artifact Review: skeleton (host app integration point)
- [x] Panel 8 — Trace Timeline: skeleton (host app integration point)
- [ ] ELK automatic layout (elkjs dependency is declared; async layout call is TODO)
- [ ] Compound state drill-down (Phase 5 full)
- [ ] Drag-to-rearrange canvas nodes
- [ ] YAML import (requires host app Rust layer for parse)
- [ ] wasm-pack test integration in CI
