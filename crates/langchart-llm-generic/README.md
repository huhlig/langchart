# langchart-llm-generic

Langchart `LlmAdapter` implementations for OpenAI, Anthropic, and compatible
HTTP APIs.

The default feature set enables both `openai` and `anthropic`; disable default
features when a host only needs one provider. Credentials and endpoints are
provided through the adapter configuration rather than workflow documents.

Never commit API keys. Resolve credentials in the host and pass them to the
adapter at runtime.

Licensed under MIT or Apache-2.0.
