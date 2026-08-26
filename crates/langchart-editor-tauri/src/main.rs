//! langchart-editor-tauri — Standalone desktop editor for langchart workflows.
//!
//! Wraps the TypeScript editor in a native shell with file system access,
//! native menus, and optional runtime integration for simulation and execution.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;
use tauri::Emitter;
use tauri::Manager;

// ── File I/O commands ─────────────────────────────────────────────────────────

/// Read a workflow file from disk.
#[tauri::command]
fn read_workflow_file(path: String) -> Result<String, String> {
    fs::read_to_string(&path).map_err(|e| format!("Failed to read {path}: {e}"))
}

/// Write a workflow file to disk.
#[tauri::command]
fn write_workflow_file(path: String, content: String) -> Result<(), String> {
    // Ensure parent directory exists
    if let Some(parent) = PathBuf::from(&path).parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {e}"))?;
    }
    fs::write(&path, content).map_err(|e| format!("Failed to write {path}: {e}"))
}

/// Show a native open file dialog and return the selected path + content.
#[tauri::command]
async fn open_file_dialog(app: tauri::AppHandle) -> Result<Option<FilePayload>, String> {
    use rfd::FileDialog;

    let dialog = FileDialog::new()
        .add_filter("Langchart Workflow", &["langchart", "json"])
        .add_filter("JSON", &["json"])
        .set_title("Open Workflow");

    let path = dialog
        .pick_file()
        .ok_or_else(|| "No file selected".to_string())?;

    let content =
        fs::read_to_string(&path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    // Store the path in app state for future saves
    if let Some(state) = app.try_state::<CurrentFilePath>() {
        *state.0.lock().map_err(|e| e.to_string())? = Some(path.display().to_string());
    }

    Ok(Some(FilePayload {
        path: path.display().to_string(),
        content,
    }))
}

/// Show a native save file dialog and write the content.
#[tauri::command]
async fn save_file_dialog(
    app: tauri::AppHandle,
    content: String,
) -> Result<Option<String>, String> {
    use rfd::FileDialog;

    let dialog = FileDialog::new()
        .add_filter("Langchart Workflow", &["langchart", "json"])
        .set_title("Save Workflow");

    let path = dialog
        .save_file()
        .ok_or_else(|| "No file selected".to_string())?;

    let path_str = path.display().to_string();

    fs::write(&path, content).map_err(|e| format!("Failed to write {path_str}: {e}"))?;

    // Update stored path
    if let Some(state) = app.try_state::<CurrentFilePath>() {
        *state.0.lock().map_err(|e| e.to_string())? = Some(path_str.clone());
    }

    Ok(Some(path_str))
}

/// Get the currently active file path (if any).
#[tauri::command]
fn get_current_file_path(app: tauri::AppHandle) -> Result<Option<String>, String> {
    let state = app
        .try_state::<CurrentFilePath>()
        .ok_or_else(|| "State not initialized".to_string())?;
    let path = state.0.lock().map_err(|e| e.to_string())?.clone();
    Ok(path)
}

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FilePayload {
    path: String,
    content: String,
}

/// Shared state for the current file path.
struct CurrentFilePath(std::sync::Mutex<Option<String>>);

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(CurrentFilePath(std::sync::Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            read_workflow_file,
            write_workflow_file,
            open_file_dialog,
            save_file_dialog,
            get_current_file_path,
        ])
        .setup(|app| {
            // Set up native menu
            #[cfg(desktop)]
            {
                use tauri::menu::{MenuBuilder, MenuItemBuilder, SubmenuBuilder};

                let file_menu = SubmenuBuilder::new(app, "File")
                    .item(
                        &MenuItemBuilder::new("New")
                            .id("menu-new")
                            .accelerator("CmdOrCtrl+N")
                            .build(app)?,
                    )
                    .item(
                        &MenuItemBuilder::new("Open...")
                            .id("menu-open")
                            .accelerator("CmdOrCtrl+O")
                            .build(app)?,
                    )
                    .item(
                        &MenuItemBuilder::new("Save")
                            .id("menu-save")
                            .accelerator("CmdOrCtrl+S")
                            .build(app)?,
                    )
                    .item(
                        &MenuItemBuilder::new("Save As...")
                            .id("menu-save-as")
                            .accelerator("CmdOrCtrl+Shift+S")
                            .build(app)?,
                    )
                    .separator()
                    .quit()
                    .build()?;

                let edit_menu = SubmenuBuilder::new(app, "Edit")
                    .undo()
                    .redo()
                    .separator()
                    .cut()
                    .copy()
                    .paste()
                    .select_all()
                    .build()?;

                let menu = MenuBuilder::new(app)
                    .item(&file_menu)
                    .item(&edit_menu)
                    .build()?;

                app.set_menu(menu)?;
            }

            Ok(())
        })
        .on_menu_event(|app, event| {
            let id = event.id().as_ref();
            if let Some(win) = app.get_webview_window("main") {
                match id {
                    "menu-new" => {
                        let _ = win.emit("menu-action", "new");
                    }
                    "menu-open" => {
                        let _ = win.emit("menu-action", "open");
                    }
                    "menu-save" => {
                        let _ = win.emit("menu-action", "save");
                    }
                    "menu-save-as" => {
                        let _ = win.emit("menu-action", "save-as");
                    }
                    _ => {}
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
