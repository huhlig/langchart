# langchart-llm-genai

A `genai`-backed implementation of Langchart's `LlmAdapter`.

This bridge provides access to providers supported by the `genai` dependency,
including Gemini, Groq, Cohere, xAI, and DeepSeek, while exposing Langchart's
provider-neutral request and response types.

The `integration` feature enables tests that call live APIs. Those tests require
the appropriate provider credentials and are not intended for ordinary CI.

Licensed under MIT or Apache-2.0.
