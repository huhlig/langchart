# langchart-cli

Command-line tools for validating, running, replaying, and inspecting Langchart
workflows.

From the workspace root:

```console
cargo run -p langchart-cli -- validate examples/hello-world.json
cargo run -p langchart-cli -- run examples/noop-actor.json
cargo run -p langchart-cli -- --help
```

The installed binary is named `langchart`. Workflow inputs may be JSON or YAML;
use the command-specific `--help` output for available arguments.

Licensed under MIT or Apache-2.0.
