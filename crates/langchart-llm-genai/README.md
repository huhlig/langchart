# langchart-llm-genai

`LlmAdapter` backed by the [`genai`](https://crates.io/crates/genai) crate, giving Langchart access to a wide range of
hosted model providers through a single dependency.

## Supported providers

Any provider supported by `genai`, including:

- Google Gemini
- Groq
- Cohere
- xAI (Grok)
- DeepSeek

Refer to the `genai` documentation for the full, up-to-date provider list and the environment variables used for API key
configuration.

## Usage

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
langchart-llm-genai = { path = "../langchart-llm-genai" }
```

Construct the adapter and supply it to `EngineAdapters`:

```rust,no_run
use langchart_llm_genai::GenaiLlmAdapter;

let adapter = GenaiLlmAdapter::new();
```

API credentials are resolved from environment variables by `genai`'s own provider-detection logic.

## Integration tests

The `integration` feature enables tests that call live provider APIs:

```console
cargo test -p langchart-llm-genai --features integration
```

These tests require provider credentials in the environment and are not intended for CI.

## License

Licensed under MIT or Apache-2.0.
