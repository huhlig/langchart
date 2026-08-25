/**
 * langchart editor — standalone bootstrap.
 *
 * Creates an Editor instance, wires toolbar buttons, keyboard shortcuts,
 * drag-drop, and localStorage draft persistence.
 * When running inside Tauri, uses native file dialogs and menu bar.
 *
 * This file is the standalone app shell. For library usage, import from
 * "@langchart/editor" instead.
 */

import { Editor } from "./editor.js";
import { wasm } from "./wasm-loader.js";
import {
  isTauri,
  openFileDialog,
  saveFileDialog,
  onMenuAction,
} from "./tauri-adapter.js";

// ── Bootstrap ─────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const root = document.getElementById("editor-root")!;
  const status = document.getElementById("document-status");
  const schemaEl = document.getElementById("schema-version");

  const editor = new Editor({
    container: root,
    mode: isTauri() ? "engine" : "edit",
  });
  await editor.init();

  // ── Status bar ───────────────────────────────────────────────────────────

  let fileName = editor.getFileName();
  let dirty = editor.isDirty();

  const setStatus = (message?: string): void => {
    if (!status) return;
    status.textContent = message ?? `${fileName}${dirty ? " • Unsaved" : ""}`;
    status.classList.toggle("document-status--dirty", dirty);
  };

  editor.on("dirty-change", (d) => { dirty = d; setStatus(); });
  editor.on("file-change", (n) => { fileName = n; setStatus(); });

  if (schemaEl) schemaEl.textContent = `schema ${wasm.schemaVersion()}`;

  // ── File operations (browser or Tauri) ──────────────────────────────────

  const doOpen = async (): Promise<void> => {
    if (isTauri()) {
      const result = await openFileDialog();
      if (result) editor.loadJson(result.content, result.path);
    } else {
      await editor.importFile();
    }
  };

  const doSave = (): void => {
    editor.exportFile();
    setStatus("Exported");
    window.setTimeout(() => setStatus(), 1200);
  };

  // ── Toolbar wiring ─────────────────────────────────────────────────────

  document.getElementById("btn-new")?.addEventListener("click", () => {
    if (!confirmDiscard(dirty)) return;
    editor.newWorkflow();
    setStatus();
  });

  document.getElementById("btn-import")?.addEventListener("click", doOpen);
  document.getElementById("btn-export")?.addEventListener("click", doSave);

  document.getElementById("btn-compile")?.addEventListener("click", () => {
    const json = editor.getJson();
    if (!json) return;
    const result = wasm.compileWorkflow(json);
    const msg = result.ok
      ? "✓ Workflow compiled successfully."
      : `✗ ${result.errors.length} compile error(s):\n${result.errors.map((e) => `  [${e.code}] ${e.message}`).join("\n")}`;
    alert(msg);
  });

  // ── Tauri native menu integration ──────────────────────────────────────

  if (isTauri()) {
    await onMenuAction(async (action) => {
      switch (action) {
        case "new":
          if (!confirmDiscard(dirty)) return;
          editor.newWorkflow();
          setStatus();
          break;
        case "open":
          await doOpen();
          break;
        case "save":
          doSave();
          break;
        case "save-as": {
          const json = editor.getJson();
          if (json) await saveFileDialog(json);
          break;
        }
      }
    });

    // Hide HTML toolbar when running in Tauri (native menu is used)
    document.querySelector(".toolbar")?.setAttribute("style", "display:none");
  }

  // ── Keyboard shortcuts ──────────────────────────────────────────────────

  window.addEventListener("keydown", (event) => {
    if (!(event.ctrlKey || event.metaKey)) return;
    const key = event.key.toLowerCase();
    if (key === "s") { event.preventDefault(); doSave(); }
    if (key === "o") { event.preventDefault(); doOpen(); }
    if (key === "n") { event.preventDefault(); document.getElementById("btn-new")?.click(); }
  });

  // ── Drag and drop ──────────────────────────────────────────────────────

  document.body.addEventListener("dragover", (event) => event.preventDefault());
  document.body.addEventListener("drop", async (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files[0];
    if (!file || !/\.(langchart|json)$/i.test(file.name)) return;
    const text = await file.text();
    editor.loadJson(text, file.name);
  });

  // ── Draft persistence ──────────────────────────────────────────────────

  const draft = readDraft();
  const hash = location.hash.slice(1);

  if (hash === "example") {
    editor.loadJson(EXAMPLE_WORKFLOW, "example.langchart");
  } else if (draft) {
    editor.loadJson(draft.json, draft.fileName, true);
  } else {
    editor.newWorkflow();
  }
  setStatus();

  // ── Beforeunload guard ──────────────────────────────────────────────────

  window.addEventListener("beforeunload", (event) => {
    if (!editor.isDirty()) return;
    event.preventDefault();
  });

  // ── Persist on change ───────────────────────────────────────────────────

  editor.on("json-change", (json) => {
    persistDraft(json, editor.getFileName());
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const DRAFT_KEY = "langchart.standalone.draft.v1";

function confirmDiscard(isDirty: boolean): boolean {
  return !isDirty || window.confirm("Discard the current unsaved workflow?");
}

function persistDraft(json: string, name: string): void {
  try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ json, fileName: name })); } catch { /* storage may be disabled */ }
}

function readDraft(): { json: string; fileName: string } | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as { json?: unknown; fileName?: unknown };
    return typeof value.json === "string" && typeof value.fileName === "string"
      ? { json: value.json, fileName: value.fileName } : null;
  } catch { return null; }
}

// ── Example workflow ──────────────────────────────────────────────────────────

const EXAMPLE_WORKFLOW = JSON.stringify(
  {
    schema_version: "1.0.0",
    id: "content-draft",
    version: "1.0.0",
    name: "Content Draft",
    initial: "prepare",
    states: [
      {
        id: "prepare",
        name: "Prepare",
        type: "atomic",
        on: { "prepare.done": { target: "write", priority: 0, actions: [] } },
      },
      {
        id: "write",
        name: "Write",
        type: "agentic",
        agent: { id: "writer-agent", version: "1.0.0" },
        prompt: "Draft the content section.",
        on: {
          "draft.ready": { target: "review", priority: 0, actions: [] },
          "draft.failed": { target: "prepare", priority: 1, actions: [] },
        },
      },
      {
        id: "review",
        name: "Review",
        type: "human",
        on: {
          "review.approved": { target: "publish", priority: 0, actions: [] },
          "review.rejected": { target: "write", priority: 0, actions: [] },
        },
      },
      {
        id: "publish",
        name: "Publish",
        type: "atomic",
        on: { "publish.done": { target: "done", priority: 0, actions: [] } },
      },
      { id: "done", name: "Done", type: "final", on: {} },
    ],
  },
  null,
  2
);

main().catch(console.error);
