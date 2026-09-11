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

You can supply an AWS Bedrock inference profile ID or ARN directly anywhere a model ID is accepted—no environment variable is required:

```rust,no_run
use langchart_adapters::llm::LlmRequest;

let mut request = LlmRequest { /* ... */ };
// Treat inference profile directly as the model ID
request.model_policy.model = Some("us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_owned());
```

Alternatively, you can configure an inference profile on `BedrockConfig`:

```rust,no_run
use langchart_llm_bedrock::{BedrockAdapter, BedrockConfig, BedrockCredentials};

let config = BedrockConfig::new("us-east-1")
    .with_inference_profile("us.anthropic.claude-3-7-sonnet-20250219-v1:0");
```

When a foundation model without a regional prefix (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`) is requested and AWS Bedrock denies direct invocation, the adapter automatically detects the denial and falls back to invoking the matching inference profile.

## License

Licensed under MIT or Apache-2.0.
