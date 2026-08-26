# langchart-adapters

Integration trait contracts for Langchart. This crate defines the traits and shared request/response types that the
runtime depends on; it contains no provider-specific implementations.

## Modules

| Module       | Traits / types                                                                                                                                                                                  |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `llm`        | `LlmAdapter`, `LlmRequest`, `LlmResponse`, `LlmStreamEvent`, `LlmEventStream`, `Message`, `ToolDefinition`, `ToolCall`, `ResponseFormat`, `TokenUsage`, `FinishReason`, `ModelInfo`, `LlmError` |
| `mcp`        | `McpAdapter` — tool and resource access via the Model Context Protocol                                                                                                                          |
| `memory`     | `MemoryAdapter` — agent memory read/write/query                                                                                                                                                 |
| `artifact`   | `ArtifactStore` — versioned artifact proposal, commit, and retrieval                                                                                                                            |
| `checkpoint` | `CheckpointStore` — serialized workflow-run state snapshots                                                                                                                                     |
| `event`      | `EventSink`, `EventSource` — observable workflow events                                                                                                                                         |
| `secrets`    | `SecretsAdapter` — runtime secret resolution                                                                                                                                                    |
| `context`    | `ContextResolver` — context assembly pipeline contracts                                                                                                                                         |
| `broadcast`  | `BroadcastSink` — fan-out event delivery                                                                                                                                                        |
| `workflow`   | `WorkflowRepository` — workflow document storage and retrieval                                                                                                                                  |

## Using adapter traits

```rust
use langchart_adapters::llm::LlmAdapter;

fn accept_llm(_adapter: &dyn LlmAdapter) {}
```

Implement a trait in your host application, or use one of the concrete adapter crates in this workspace:

- [`langchart-llm-generic`](../langchart-llm-generic) — OpenAI, Anthropic, and compatible endpoints
- [`langchart-llm-genai`](../langchart-llm-genai) — Gemini, Groq, Cohere, xAI, DeepSeek
- [`langchart-llm-watsonx`](../langchart-llm-watsonx) — IBM watsonx.ai
- [`langchart-mcp-client`](../langchart-mcp-client) — child-process MCP servers
- [`langchart-artifact-fs`](../langchart-artifact-fs) — file-system artifact store
- [`langchart-checkpoint-redb`](../langchart-checkpoint-redb) — embedded checkpoint store
- [`langchart-memory-redb`](../langchart-memory-redb) — embedded memory adapter
- [`langchart-model-router`](../langchart-model-router) — multi-provider LLM router

## LLM streaming

`LlmAdapter::complete_stream` returns an `LlmEventStream` — a pinned, boxed `Stream` of `LlmStreamEvent` values.
Adapters that only implement `complete` inherit a two-event fallback via `buffered_response_stream`. Streaming adapters
override `complete_stream` directly.

The `ResponseCompleted` event is the only event that carries a durable, usable response. All earlier delta events are
provisional.

## License

Licensed under MIT or Apache-2.0.
