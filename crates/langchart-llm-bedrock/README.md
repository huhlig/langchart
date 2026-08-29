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

### Static credentials

```rust,no_run
use langchart_llm_bedrock::{BedrockAdapter, BedrockConfig, BedrockCredentials};

let adapter = BedrockAdapter::new(
    BedrockConfig::new("us-east-1"),
    BedrockCredentials::Static {
        access_key_id: "AKIA...".to_owned(),
        secret_access_key: "secret...".to_owned(),
        session_token: None,
    },
)?;
```

## License

Licensed under MIT or Apache-2.0.
