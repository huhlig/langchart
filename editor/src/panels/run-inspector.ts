/**
 * RunInspector — Panel 4
 *
 * Displays the active run snapshot: active states, event queue depth,
 * running activities, and status. Also shows a scrolling colour-coded
 * event log populated via `EditorState.appendRunEvent()`.
 *
 * Populated by the host application via `EditorState.updateRunSnapshot()`
 * and `EditorState.appendRunEvent()`.
 */

import { Panel } from "./panel-base.js";
import { escapeHtml } from "../html.js";
import { EditorEvent, RunEventEntry } from "../editor-state.js";

// Colour classes per event kind — mapped in styles.css
const KIND_CLASS: Record<string, string> = {
  lifecycle: "run-event--lifecycle",
  state:     "run-event--state",
  error:     "run-event--error",
  budget:    "run-event--budget",
  other:     "run-event--other",
};

export class RunInspector extends Panel {
  protected render(): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.el("div", { class: "panel-header", text: "Run Inspector" }));

    const snap = this.state.runSnapshot;
    if (!snap) {
      this.root.appendChild(
        this.el("p", { class: "panel-empty", text: "No active run. Press ▶ Run in the toolbar to start one." })
      );
      // Still render any accumulated events even without an active snapshot.
      this._renderEventLog();
      return;
    }

    const row = (label: string, value: string) => {
      const div = this.el("div", { class: "inspector-row" });
      div.innerHTML = `<span class="inspector-label">${escapeHtml(label)}</span><span class="inspector-value">${escapeHtml(value)}</span>`;
      this.root.appendChild(div);
    };

    const statusClass = `run-status run-status--${snap.status}`;
    const statusDiv = this.el("div", { class: statusClass, text: snap.status.toUpperCase() });
    this.root.appendChild(statusDiv);

    row("Run ID", snap.runId);
    row("Queue depth", `${snap.eventQueueDepth}`);

    if (snap.activeStates.length > 0) {
      this.root.appendChild(this.el("h3", { class: "inspector-section", text: "Active States" }));
      const ul = this.el("ul", { class: "run-active-list" });
      for (const s of snap.activeStates) {
        ul.appendChild(this.el("li", { text: s }));
      }
      this.root.appendChild(ul);
    }

    if (snap.activities.length > 0) {
      this.root.appendChild(this.el("h3", { class: "inspector-section", text: "Running Activities" }));
      const ul = this.el("ul", { class: "run-activity-list" });
      for (const a of snap.activities) {
        ul.appendChild(this.el("li", { text: a }));
      }
      this.root.appendChild(ul);
    }

    this._renderEventLog();
  }

  private _renderEventLog(): void {
    const events = this.state.runEvents;
    if (events.length === 0) return;

    this.root.appendChild(this.el("h3", { class: "inspector-section", text: "Event Log" }));

    const log = this.el("div", { class: "run-event-log" });
    // Render newest-first for scrollability.
    for (let i = events.length - 1; i >= 0; i--) {
      log.appendChild(this._eventRow(events[i]!));
    }
    this.root.appendChild(log);
  }

  private _eventRow(entry: RunEventEntry): HTMLElement {
    const kindClass = KIND_CLASS[entry.kind] ?? KIND_CLASS["other"];
    const row = this.el("div", { class: `run-event ${kindClass}` });

    const ts = new Date(entry.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    row.innerHTML =
      `<span class="run-event-time">${escapeHtml(ts)}</span>` +
      `<span class="run-event-kind">${escapeHtml(entry.kind)}</span>` +
      `<span class="run-event-label">${escapeHtml(entry.label)}</span>` +
      (entry.detail ? `<span class="run-event-detail">${escapeHtml(entry.detail)}</span>` : "");

    return row;
  }

  protected override onEditorEvent(event: EditorEvent): void {
    if (event.type === "run-snapshot-updated") {
      this.refresh();
    } else if (event.type === "run-event-appended") {
      // Append a single row to the log without full re-render.
      const log = this.root.querySelector<HTMLElement>(".run-event-log");
      if (log) {
        log.insertBefore(this._eventRow(event.entry), log.firstChild);
      } else {
        // Log section doesn't exist yet — full re-render to create it.
        this.refresh();
      }
    }
  }
}
