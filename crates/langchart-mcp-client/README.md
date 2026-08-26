# langchart-mcp-client

`McpAdapter` implementation for Langchart that connects to Model Context Protocol servers over child-process transports.

## What this crate does

It uses the [`rmcp`](https://crates.io/crates/rmcp) client library to launch and communicate with MCP servers as child
processes. Langchart tool-call and resource-access requests are translated into MCP protocol calls so the runtime can
use any MCP-compatible server through the standard adapter interface.

## Usage

```rust,no_run
use langchart_mcp_client::McpClientAdapter;

// Spawn an MCP server by command
let adapter = McpClientAdapter::spawn("uvx", &["mcp-server-filesystem", "/data"]).await?;
```

The host is responsible for:

1. Choosing which MCP server commands to run
2. Passing safe, controlled environment variables to each server process
3. Ensuring the server command is available on `PATH` or providing an absolute path

## Security

MCP servers run as child processes with the permissions of the parent process. Only launch trusted MCP server commands.
Never pass secret credentials through untrusted server arguments or environment variables.

## License

Licensed under MIT or Apache-2.0.
