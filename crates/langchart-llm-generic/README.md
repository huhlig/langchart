# langchart-llm-generic

`LlmAdapter` implementations for OpenAI, Anthropic, and any OpenAI-compatible HTTP endpoint.

## Supported providers

| Provider          | Model prefix  | Notes                                       |
|-------------------|---------------|---------------------------------------------|
| OpenAI            | any (default) | Chat Completions API with SSE streaming     |
| Anthropic         | `claude-*`    | Messages API; text responses only           |
| OpenAI-compatible | any           | Azure OpenAI, Ollama, vLLM, LM Studio, etc. |

Provider routing is automatic: model names starting with `claude` route to the Anthropic Messages API; everything else
uses the OpenAI Chat Completions API.

## Features

- `openai` — enables the OpenAI / compatible path (on by default)
- `anthropic` — enables the Anthropic path (on by default)

Disable default features to include only the provider your host requires:

```toml
langchart-llm-generic = { path = "...", default-features = false, features = ["openai"] }
```

## Usage

```rust,no_run
use langchart_llm_generic::GenericLlmAdapter;

// Read API keys from OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables
let adapter = GenericLlmAdapter::from_env()?;

// Or configure explicitly with the builder
let adapter = GenericLlmAdapter::builder()
    .openai_api_key("sk-...")
    .anthropic_api_key("sk-ant-...")
    .build()?;
```

### OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, Azure, …)

```rust,no_run
use langchart_llm_generic::GenericLlmAdapter;

let adapter = GenericLlmAdapter::builder()
    .openai_api_key("...")                                   // omit for unauthenticated local endpoints
    .openai_base_url("http://localhost:11434/v1")            // Ollama
    .build()?;
```

### Timeout configuration

All timeouts default to safe values and can be overridden on the builder:

| Builder method                    | Default | Description                                       |
|-----------------------------------|---------|---------------------------------------------------|
| `connect_timeout`                 | 10 s    | TCP / TLS connect deadline                        |
| `first_byte_timeout`              | 300 s   | Deadline from request send to first response byte |
| `stream_idle_timeout`             | 120 s   | Maximum gap between SSE events                    |
| `total_generation_timeout`        | 900 s   | Wall-clock cap for the entire generation          |
| `max_response_body_bytes`         | 16 MiB  | Maximum decoded JSON response body                |
| `max_encoded_response_body_bytes` | 16 MiB  | Maximum encoded wire body                         |

## Security

Never commit API keys. Resolve credentials in the host application (e.g. from environment variables or a secrets
manager) and pass them to the adapter at startup. Credentials are held in memory and excluded from `Debug` output.

## License

Licensed under MIT or Apache-2.0.
