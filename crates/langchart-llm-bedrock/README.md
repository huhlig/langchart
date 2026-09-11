# langchart-llm-bedrock

AWS Bedrock Converse API implementation of Langchart's `LlmAdapter` contract.

## Features

- **Uniform Bedrock Converse API** — supports Claude, Nova, Llama, and other foundation models on AWS Bedrock
- **Flexible AWS Credentials** — supports environment variables, AWS profiles, SSO, IAM roles, or explicit static credentials
- **TLS via rustls** — no OpenSSL dependency
- **Model enumeration** — built-in discovery for major foundation models

## Usage

```rust,no_run
use langchart_llm_bedrock::{BedrockAdapter, BedrockConfig, BedrockCredentials};

let adapter = BedrockAdapter::new(
    BedrockConfig::new("us-east-1"),
    BedrockCredentials::EnvironmentOrProfile,
)?;
```

### Bearer token credentials

```rust,no_run
use langchart_llm_bedrock::{BedrockAdapter, BedrockConfig, BedrockCredentials};

let adapter = BedrockAdapter::new(
    BedrockConfig::new("us-east-1"),
    BedrockCredentials::BearerToken("secret-bearer-token".to_owned()),
)?;
```

### Using Inference Profiles as Model IDs

You can supply an AWS Bedrock inference profile ID (regional `us.*`/`eu.*`/`apac.*`, or global `global.*`) or ARN directly anywhere a model ID is accepted—no environment variable is required:

```rust,no_run
use langchart_adapters::llm::LlmRequest;

let mut request = LlmRequest { /* ... */ };
// Treat regional or global inference profile directly as the model ID
request.model_policy.model = Some("global.anthropic.claude-3-7-sonnet-20250219-v1:0".to_owned());
```

### Registering Custom Models & Profiles

To ensure downstream model registries or health checks recognize custom models or newly released models:

```rust,no_run
use langchart_llm_bedrock::{BedrockAdapter, BedrockConfig, BedrockCredentials};

let config = BedrockConfig::new("us-east-1")
    .with_model("anthropic.claude-opus-4:0")
    .with_models(["global.anthropic.claude-3-7-sonnet-20250219-v1:0"]);
```

When a foundation model without a regional prefix (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`) is requested and AWS Bedrock denies direct invocation, the adapter automatically detects the denial and falls back to invoking the matching inference profile.

### Dynamic Control-Plane Discovery (Optional)

By default, `langchart-llm-bedrock` targets the AWS Bedrock data plane (`bedrock-runtime`) to remain lightweight and compatible with bearer tokens. If dynamic catalog queries against AWS Bedrock's Control Plane are desired and AWS IAM credentials with `bedrock:List*` permissions are available, enable the `control-plane` feature:

```toml
[dependencies]
langchart-llm-bedrock = { version = "0.1", features = ["control-plane"] }
```

## License

Licensed under MIT or Apache-2.0.
