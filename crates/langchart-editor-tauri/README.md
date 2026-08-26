# langchart-editor-tauri

Standalone Tauri desktop application for authoring and validating Langchart workflow documents.

## Features

- Open, edit, and save workflow documents (JSON or YAML)
- In-editor workflow validation using the `langchart-wasm` WASM bindings
- Native file-picker dialogs via `rfd`
- System tray integration

## Building

From the repository root, with a [Tauri v2 build environment](https://v2.tauri.app/start/prerequisites/) installed:

```console
cargo tauri build --config crates/langchart-editor-tauri/tauri.conf.json
```

For development with hot-reload:

```console
cargo tauri dev --config crates/langchart-editor-tauri/tauri.conf.json
```

## Prerequisites

- Rust stable toolchain
- Node.js (for the Tauri CLI and frontend bundler)
- Platform-specific Tauri build dependencies — see
  the [Tauri prerequisites guide](https://v2.tauri.app/start/prerequisites/) for your operating system

## Architecture

The Tauri backend is a Rust binary (`langchart-editor`) that embeds the Langchart model and runtime. The frontend is a
web-based editor interface served via Tauri's custom-protocol feature. Workflow validation is performed in the backend
using `langchart-model` directly, keeping all workflow logic in Rust.

## License

Licensed under MIT or Apache-2.0.
