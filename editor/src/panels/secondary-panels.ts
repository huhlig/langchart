/**
 * ContextInspector — Panel 5
 * CapabilityInspector — Panel 6
 * ArtifactReview — Panel 7
 * TraceTimeline — Panel 8
 * SimulationPanel — (replaces the simulation stub described in spec §17)
 *
 * Secondary panels: context, capability, artifact review, trace timeline,
 * and workflow simulation.
 */

import { Panel } from "./panel-base.js";
import { escapeHtml } from "../html.js";
import { EditorEvent } from "../editor-state.js";
import { wasm, SimulationInput, SimulationResult } from "../wasm-loader.js";

export class ContextInspector extends Panel {
  protected render(): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.el("div", { class: "panel-header", text: "Context Inspector" }));
    this.root.appendChild(
      this.el("p", {
        class: "panel-empty",
        text: "Context details are populated from live run events (§11, ContextResolverChain). Connect a run to inspect.",
      })
    );
  }

  protected override onEditorEvent(_event: EditorEvent): void {}
}

/**
 * CapabilityInspector — Panel 6
 *
 * Shows the effective capability envelope for the selected state:
 * permitted MCP tools, resource URIs, operation classes, and budget limits.
 * Derived from the compiled workflow's static capability calculation.
 */
export class CapabilityInspector extends Panel {
  protected render(): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.el("div", { class: "panel-header", text: "Capability Inspector" }));

    const id = this.state.selectedStateId;
    if (!id) {
      this.root.appendChild(
        this.el("p", { class: "panel-empty", text: "Select an agentic state to inspect its capability envelope." })
      );
      return;
    }

    if (!this.state.json) return;

    const info = wasm.inspectState(this.state.json, id);
    if (!info) {
      this.root.appendChild(this.el("p", { class: "panel-empty", text: `State "${id}" not found.` }));
      return;
    }

    const row = (l: string, v: string) => {
      const div = this.el("div", { class: "inspector-row" });
      div.innerHTML = `<span class="inspector-label">${escapeHtml(l)}</span><span class="inspector-value">${escapeHtml(v)}</span>`;
      this.root.appendChild(div);
    };

    row("State", id);
    row("Type", info.type);
    row("Has capability config", info.has_capabilities ? "yes" : "no (inherits workflow defaults)");
    row("Has limits config", info.has_limits ? "yes" : "no (inherits workflow defaults)");
    row("Agent", info.agent ?? "—");

    this.root.appendChild(
      this.el("p", {
        class: "panel-note",
        text: "Full MCP tool list and budget numbers require the compiled workflow from the Rust runtime.",
      })
    );
  }

  protected override onEditorEvent(event: EditorEvent): void {
    if (event.type === "state-selected" || event.type === "workflow-changed") {
      this.refresh();
    }
  }
}

/**
 * ArtifactReview — Panel 7
 *
 * Displays proposals, diffs, conflicts, approval status, and committed artifact
 * versions. Populated by the host application's artifact store adapter.
 */
export class ArtifactReview extends Panel {
  protected render(): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.el("div", { class: "panel-header", text: "Artifact Review" }));
    this.root.appendChild(
      this.el("p", {
        class: "panel-empty",
        text: "Artifact proposals appear here during a run. Implement ArtifactStore to enable this panel.",
      })
    );
  }

  protected override onEditorEvent(_event: EditorEvent): void {}
}

/**
 * TraceTimeline — Panel 8
 *
 * Shows a chronological timeline of observable runtime events: transitions,
 * model calls, tool calls, costs, latency, and errors.
 * Populated from the host application's event stream.
 */
export class TraceTimeline extends Panel {
  protected render(): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.el("div", { class: "panel-header", text: "Trace Timeline" }));
    this.root.appendChild(
      this.el("p", {
        class: "panel-empty",
        text: "Runtime events appear here during a live run. Connect a run to see the trace.",
      })
    );
  }

  protected override onEditorEvent(event: EditorEvent): void {
    if (event.type === "run-snapshot-updated" && event.snapshot) {
      // A full implementation would stream RuntimeEvent records here and render
      // them as a scrollable timeline with swimlanes per state.
      this.refresh();
    }
  }
}

/**
 * SimulationPanel — Panel 9 (or secondary tab)
 *
 * Allows the user to configure a deterministic simulation script:
 * - Inject initial events
 * - Configure scripted actor responses per state
 * - Run the simulation and view the resulting step trace
 *
 * Backed by the `simulateWorkflow` WASM binding (pure synchronous model
 * simulation — no LLM calls).
 */
export class SimulationPanel extends Panel {
  // Per-state actor scripts: state_id → emit event_type
  private _actors: Record<string, string> = {};
  // Events to inject at start
  private _injectEvents: string[] = [];
  private _result: SimulationResult | null = null;

  protected render(): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.el("div", { class: "panel-header", text: "Simulation" }));

    if (!this.state.json) {
      this.root.appendChild(
        this.el("p", { class: "panel-empty", text: "Load a workflow to run a simulation." })
      );
      return;
    }

    const stateIds = wasm.listStateIds(this.state.json);

    // ── Actor scripts form ────────────────────────────────────────────────────
    const scriptSection = this.el("div", { class: "sim-section" });
    scriptSection.appendChild(this.el("h3", { class: "inspector-section", text: "Actor Scripts" }));

    const note = this.el("p", { class: "sim-note" });
    note.textContent =
      "Configure which event each state emits when the simulation reaches it. " +
      "Leave blank to treat the state as terminal (no auto-advance).";
    scriptSection.appendChild(note);

    const grid = this.el("div", { class: "sim-grid" });
    for (const id of stateIds) {
      const label = this.el("label", { class: "sim-actor-label", text: id });
      const input = this.el("input", { class: "sim-actor-input" }) as HTMLInputElement;
      input.type = "text";
      input.placeholder = "emit event type…";
      input.value = this._actors[id] ?? "";
      input.addEventListener("change", () => {
        const v = input.value.trim();
        if (v) {
          this._actors[id] = v;
        } else {
          delete this._actors[id];
        }
      });
      const row = this.el("div", { class: "sim-actor-row" });
      row.appendChild(label);
      row.appendChild(input);
      grid.appendChild(row);
    }
    scriptSection.appendChild(grid);
    this.root.appendChild(scriptSection);

    // ── Inject events ─────────────────────────────────────────────────────────
    const injectSection = this.el("div", { class: "sim-section" });
    injectSection.appendChild(this.el("h3", { class: "inspector-section", text: "Inject Events" }));

    const injectNote = this.el("p", { class: "sim-note" });
    injectNote.textContent =
      "Comma-separated list of event types to inject immediately after start.";
    injectSection.appendChild(injectNote);

    const injectInput = this.el("input", { class: "sim-inject-input" }) as HTMLInputElement;
    injectInput.type = "text";
    injectInput.placeholder = "e.g. start.ready, prepare.done";
    injectInput.value = this._injectEvents.join(", ");
    injectInput.addEventListener("change", () => {
      this._injectEvents = injectInput.value
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    });
    injectSection.appendChild(injectInput);
    this.root.appendChild(injectSection);

    // ── Run button ────────────────────────────────────────────────────────────
    const runBtn = this.el("button", { class: "btn btn--primary sim-run-btn", text: "▶ Run Simulation" });
    runBtn.addEventListener("click", () => this._runSimulation());
    this.root.appendChild(runBtn);

    // ── Results ───────────────────────────────────────────────────────────────
    if (this._result) {
      this._renderResult(this._result);
    }
  }

  private _runSimulation(): void {
    if (!this.state.json) return;

    const input: SimulationInput = {
      actors: Object.fromEntries(
        Object.entries(this._actors).map(([id, emit]) => [id, { emit }])
      ),
      inject: this._injectEvents.map((e) => ({ event_type: e })),
      max_steps: 100,
    };

    try {
      this._result = wasm.simulateWorkflow(this.state.json, input);
    } catch (e) {
      this._result = {
        status: "stuck",
        final_state: "",
        steps: [],
        error: String(e),
      };
    }
    this.refresh();
  }

  private _renderResult(result: SimulationResult): void {
    const section = this.el("div", { class: "sim-section sim-result" });

    const statusClass = `sim-status sim-status--${result.status}`;
    const statusBadge = this.el("span", { class: statusClass });
    statusBadge.textContent = result.status.toUpperCase();

    const header = this.el("div", { class: "sim-result-header" });
    header.appendChild(statusBadge);
    if (result.final_state) {
      const fs = this.el("span", { class: "sim-final-state" });
      fs.textContent = ` → ${result.final_state}`;
      header.appendChild(fs);
    }
    section.appendChild(header);

    if (result.error) {
      const errDiv = this.el("div", { class: "sim-error" });
      errDiv.textContent = result.error;
      section.appendChild(errDiv);
    }

    if (result.steps.length > 0) {
      section.appendChild(this.el("h3", { class: "inspector-section", text: "Step Trace" }));
      const table = this.el("table", { class: "inspector-table sim-trace-table" });
      table.innerHTML = `
        <thead><tr>
          <th>#</th><th>State</th><th>Event</th><th>→ Target</th>
        </tr></thead>`;
      const tbody = this.el("tbody");
      for (const s of result.steps) {
        const tr = this.el("tr");
        tr.innerHTML = `<td>${s.step}</td><td>${escapeHtml(s.active_state)}</td>
          <td class="sim-event">${escapeHtml(s.event)}</td><td>${escapeHtml(s.target)}</td>`;
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      section.appendChild(table);
    } else if (!result.error) {
      section.appendChild(
        this.el("p", { class: "panel-empty", text: "No transitions taken." })
      );
    }

    this.root.appendChild(section);
  }

  protected override onEditorEvent(event: EditorEvent): void {
    if (event.type === "workflow-changed") {
      // Reset actors and result when workflow changes.
      this._actors = {};
      this._result = null;
      this.refresh();
    }
  }
}
