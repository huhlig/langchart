// Copyright 2026 Hans W. Uhlig
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use langchart_adapters::llm::{LlmAdapter, LlmRequest, Message, ResponseFormat};
use langchart_llm_bedrock::{BedrockAdapter, BedrockConfig, BedrockCredentials};

#[tokio::test]
async fn test_live_bedrock_bearer_token() {
    let token = match std::env::var("AWS_BEARER_TOKEN_BEDROCK") {
        Ok(t) if !t.is_empty() => t,
        _ => {
            println!("Skipping live test: AWS_BEARER_TOKEN_BEDROCK not set");
            return;
        }
    };

    let adapter = BedrockAdapter::new(
        BedrockConfig::new("us-east-1"),
        BedrockCredentials::BearerToken(token),
    )
    .unwrap();

    let mut request = LlmRequest {
        model_policy: Default::default(),
        messages: vec![Message::User {
            content: "Say 'Bearer token authentication succeeded!' and nothing else.".to_owned(),
        }],
        tools: vec![],
        response_format: ResponseFormat::Text,
    };
    let target_model = std::env::var("AWS_BEDROCK_INFERENCE_PROFILE")
        .or_else(|_| std::env::var("AWS_BEDROCK_INFERENCE_PROFILE_ID"))
        .or_else(|_| std::env::var("AWS_BEDROCK_MODEL"))
        .unwrap_or_else(|_| "us.meta.llama3-3-70b-instruct-v1:0".to_owned());

    request.model_policy.model = Some(target_model);
    request.model_policy.max_tokens = Some(30);

    let result = adapter.complete(request).await;
    println!("Bedrock complete result: {:?}", result);
    match result {
        Ok(response) => {
            println!("Live Response Content: {:?}", response.content);
            assert!(response.content.is_some());
        }
        Err(e) => {
            panic!("Live Bedrock call failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_live_bedrock_environment_bearer_token() {
    if std::env::var("AWS_BEARER_TOKEN_BEDROCK").is_err() {
        println!("Skipping live env test: AWS_BEARER_TOKEN_BEDROCK not set");
        return;
    }

    let adapter = BedrockAdapter::new(
        BedrockConfig::new("us-east-1"),
        BedrockCredentials::EnvironmentOrProfile,
    )
    .unwrap();

    let mut request = LlmRequest {
        model_policy: Default::default(),
        messages: vec![Message::User {
            content: "Say 'Env bearer token authentication succeeded!' and nothing else.".to_owned(),
        }],
        tools: vec![],
        response_format: ResponseFormat::Text,
    };
    let target_model = std::env::var("AWS_BEDROCK_INFERENCE_PROFILE")
        .or_else(|_| std::env::var("AWS_BEDROCK_INFERENCE_PROFILE_ID"))
        .or_else(|_| std::env::var("AWS_BEDROCK_MODEL"))
        .unwrap_or_else(|_| "us.meta.llama3-3-70b-instruct-v1:0".to_owned());

    request.model_policy.model = Some(target_model);
    request.model_policy.max_tokens = Some(30);

    let result = adapter.complete(request).await;
    println!("Bedrock env complete result: {:?}", result);
    match result {
        Ok(response) => {
            println!("Live Env Response Content: {:?}", response.content);
            assert!(response.content.is_some());
        }
        Err(e) => {
            panic!("Live Bedrock env call failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_live_bedrock_automatic_inference_profile_fallback() {
    let token = match std::env::var("AWS_BEARER_TOKEN_BEDROCK") {
        Ok(t) if !t.is_empty() => t,
        _ => {
            println!("Skipping live fallback test: AWS_BEARER_TOKEN_BEDROCK not set");
            return;
        }
    };

    let adapter = BedrockAdapter::new(
        BedrockConfig::new("us-east-1"),
        BedrockCredentials::BearerToken(token),
    )
    .unwrap();

    let mut request = LlmRequest {
        model_policy: Default::default(),
        messages: vec![Message::User {
            content: "Say 'Fallback succeeded!' and nothing else.".to_owned(),
        }],
        tools: vec![],
        response_format: ResponseFormat::Text,
    };
    // Passing a base foundation model without regional prefix.
    // In us-east-1, Bedrock requires an inference profile for Meta Llama / Claude.
    // BedrockAdapter will automatically catch the denial and fall back to "us.meta.llama3-3-70b-instruct-v1:0".
    request.model_policy.model = Some("meta.llama3-3-70b-instruct-v1:0".to_owned());
    request.model_policy.max_tokens = Some(30);

    let result = adapter.complete(request).await;
    println!("Bedrock fallback complete result: {:?}", result);
    match result {
        Ok(response) => {
            println!("Live Fallback Response Content: {:?}", response.content);
            println!("Reported model fallback: {:?}", response.reported_model);
            assert!(response.content.is_some());
        }
        Err(e) => {
            panic!("Live Bedrock fallback call failed: {:?}", e);
        }
    }
}
