# langchart-mcp-client

An MCP client implementation of Langchart's `McpAdapter` contract.

The crate uses `rmcp` to connect to Model Context Protocol servers over child
process transports. It translates Langchart tool and resource requests into
MCP calls so the runtime can access an MCP server through the standard adapter
interface.

The host is responsible for selecting and launching trusted server commands
and for supplying their environment safely.

Licensed under MIT or Apache-2.0.
