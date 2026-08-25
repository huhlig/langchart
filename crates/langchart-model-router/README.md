# langchart-model-router

A policy-driven `LlmAdapter` that routes requests to registered model
providers.

`ModelRouter` selects an adapter from the request's explicit model or model
profile, forwards completion requests, and aggregates model enumeration across
registered providers. Use it when one Langchart runtime needs to address
multiple LLM backends behind the common adapter contract.

Licensed under MIT or Apache-2.0.
