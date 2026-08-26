# langchart-llm-watsonx

IBM watsonx.ai implementation of Langchart's `LlmAdapter` contract.

## Features

- **Project and deployment-space scopes** — select between `WatsonxScope::Project` and `WatsonxScope::DeploymentSpace`
- **Caller-supplied bearer tokens** — pass a pre-fetched token via `WatsonxCredentials::Bearer`
- **IBM Cloud API keys** — exchanged through IBM IAM; the resulting short-lived bearer token is cached in memory and
  refreshed automatically before expiry
- **TLS via rustls** — no OpenSSL dependency

## Usage

```rust,no_run
use langchart_llm_watsonx::{
    WatsonxAdapter, WatsonxConfig, WatsonxCredentials, WatsonxScope,
};

let adapter = WatsonxAdapter::new(
    WatsonxConfig::new(
        "https://us-south.ml.cloud.ibm.com",
        "2024-05-31",
        WatsonxScope::Project("your-project-id".to_owned()),
    ),
    WatsonxCredentials::ApiKey("your-api-key".to_owned()),
)?;
```

## Security

Credentials are held in memory and are intentionally excluded from `Debug` output. Never embed API keys in source code;
resolve them at runtime from environment variables or a secrets manager.

## Notes

- Model enumeration (`list_models`) is not currently implemented and returns an empty result.
- The Anthropic `claude-*` model names are not routed to this adapter; they belong to `langchart-llm-generic`.
- For multi-provider deployments, combine this adapter with [`langchart-model-router`](../langchart-model-router).

## License

Licensed under MIT or Apache-2.0.
