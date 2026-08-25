/**
 * Tauri adapter for the langchart editor.
 *
 * Detects if running inside Tauri and provides native file I/O via IPC.
 * Falls back to browser APIs when not in Tauri.
 */

declare global {
  interface Window {
    __TAURI__?: {
      invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
      event: {
        listen: (event: string, handler: (payload: { payload: unknown }) => void) => Promise<() => void>;
      };
    };
  }
}

export interface FilePayload {
  path: string;
  content: string;
}

/** Check if running inside Tauri. */
export function isTauri(): boolean {
  return typeof window !== "undefined" && window.__TAURI__ !== undefined;
}

/** Invoke a Tauri command. */
async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!window.__TAURI__) throw new Error("Not running in Tauri");
  return window.__TAURI__.invoke(cmd, args) as Promise<T>;
}

/** Show native open dialog and return file content. */
export async function openFileDialog(): Promise<FilePayload | null> {
  if (!isTauri()) return null;
  return invoke<FilePayload | null>("open_file_dialog");
}

/** Show native save dialog and write file. */
export async function saveFileDialog(content: string): Promise<string | null> {
  if (!isTauri()) return null;
  return invoke<string | null>("save_file_dialog", { content });
}

/** Read a file by path. */
export async function readFile(path: string): Promise<string> {
  return invoke<string>("read_workflow_file", { path });
}

/** Write content to a file by path. */
export async function writeFile(path: string, content: string): Promise<void> {
  return invoke<void>("write_workflow_file", { path, content });
}

/** Get the current file path from Tauri state. */
export async function getCurrentFilePath(): Promise<string | null> {
  if (!isTauri()) return null;
  return invoke<string | null>("get_current_file_path");
}

/** Listen for menu actions from the native menu bar. */
export async function onMenuAction(
  handler: (action: "new" | "open" | "save" | "save-as") => void
): Promise<() => void> {
  if (!isTauri()) return () => {};
  const tauri = window.__TAURI__;
  if (!tauri) return () => {};
  return tauri.event.listen("menu-action", (event: { payload: unknown }) => {
    handler(event.payload as "new" | "open" | "save" | "save-as");
  });
}
