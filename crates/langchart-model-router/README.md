# langchart-model-router

Policy-driven `LlmAdapter` that routes completion requests to registered model providers.

`ModelRouter` implements `LlmAdapter` itself, so it can be supplied to `EngineAdapters` as the single LLM backend even
when the deployment uses multiple providers. It selects a downstream adapter at call time based on the request's
explicit model name or model profile.

## How routing works

1. The request's `model_policy.model` field is matched against registered provider names.
2. If no explicit model is given, the request's model profile is used to select the provider.
3. The selected adapter handles the request; the result is returned unchanged.
4. `list_models` aggregates results from all registered adapters.

## Usage

```rust,no_run
use langchart_model_router::ModelRouter;
use langchart_llm_generic::GenericLlmAdapter;
use langchart_llm_watsonx::{WatsonxAdapter, WatsonxConfig, WatsonxCredentials, WatsonxScope};

let openai = GenericLlmAdapter::from_env()?;
let watsonx = WatsonxAdapter::new(config, credentials)?;

let router = ModelRouter::builder()
    .register("gpt-4o", openai)
    .register("ibm/granite-3-8b-instruct", watsonx)
    .build();

// Pass `router` to EngineAdapters as the llm field.
```

## License

Licensed under MIT or Apache-2.0.
