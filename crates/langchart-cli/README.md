# langchart-cli

Command-line tools for validating, running, replaying, and inspecting Langchart workflows.

The installed binary is named `langchart`. Workflow input files may be JSON or YAML.

## Commands

| Command           | Description                                            |
|-------------------|--------------------------------------------------------|
| `validate <file>` | Validate a workflow document and print any diagnostics |
| `run <file>`      | Run a workflow to completion using a no-op actor       |
| `replay <file>`   | Re-execute a workflow from a saved event log           |
| `--help`          | Show available commands and options                    |

## Running from the workspace

```console
# Validate a workflow
cargo run -p langchart-cli -- validate examples/hello-world.json

# Run a no-op workflow
cargo run -p langchart-cli -- run examples/noop-actor.json

# Show all options
cargo run -p langchart-cli -- --help
```

## Installing the binary

```console
cargo install --path crates/langchart-cli
langchart validate my-workflow.json
```

## Notes

- Exit code `0` means validation passed with no errors. Warnings are printed but do not change the exit code.
- The `run` command uses `langchart-checkpoint-redb` for local checkpoint storage.
- Use `--log-level trace` (or set `RUST_LOG`) for verbose tracing output.

## License

Licensed under MIT or Apache-2.0.
