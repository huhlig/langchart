/**
 * ProblemsPanel — Panel 3
 *
 * Displays structural, schema, guard, and reachability diagnostics.
 * Updates continuously as the workflow document changes.
 * Clicking a diagnostic that references a state selects it in the canvas.
 */

import { Panel } from "./panel-base.js";
import { escapeHtml } from "../html.js";
import { EditorEvent } from "../editor-state.js";

export class ProblemsPanel extends Panel {
  protected render(): void {
    this.root.innerHTML = "";
    const errorCount = this.state.diagnostics.filter((d) => d.severity === "error").length;
    const warnCount = this.state.diagnostics.filter((d) => d.severity === "warning").length;

    const header = this.el("div", { class: "panel-header" });
    header.textContent = `Problems (${errorCount} errors, ${warnCount} warnings)`;
    this.root.appendChild(header);

    // Reachability warnings.
    const unreachable = this.state.reachability.unreachable;
    const allDiags = [
      ...this.state.diagnostics,
      ...unreachable.map((id) => ({
        code: "W100",
        severity: "warning" as const,
        message: `State "${id}" is unreachable from the initial state`,
        location: `state:${id}`,
      })),
    ];

    if (allDiags.length === 0) {
      this.root.appendChild(
        this.el("p", { class: "problems-ok", text: "✓ No problems detected." })
      );
      return;
    }

    const list = this.el("ul", { class: "problems-list" });
    for (const d of allDiags) {
      const stateId = extractStateId(d.location);

      const li = this.el("li", {
        class: `problems-item problems-item--${d.severity}${stateId ? " problems-item--clickable" : ""}`,
      });
      li.innerHTML = `<span class="problems-code">${escapeHtml(d.code)}</span>
        <span class="problems-message">${escapeHtml(d.message)}</span>
        <span class="problems-location">${escapeHtml(d.location)}</span>`;

      if (stateId) {
        li.title = `Click to select state "${stateId}"`;
        li.addEventListener("click", () => this.state.selectState(stateId));
      }

      list.appendChild(li);
    }
    this.root.appendChild(list);
  }

  protected override onEditorEvent(event: EditorEvent): void {
    if (event.type === "diagnostics-updated" || event.type === "workflow-changed") {
      this.refresh();
    }
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Extract a state ID from a diagnostic location string.
 * Handles formats: "state:my-state-id", "state/my-state-id", "my-state-id".
 * Returns null if the location doesn't reference a specific state.
 */
function extractStateId(location: string): string | null {
  if (!location) return null;
  // "state:some-id" or "state/some-id"
  const m = location.match(/^state[:/](.+)$/);
  if (m) return m[1] ?? null;
  return null;
}
