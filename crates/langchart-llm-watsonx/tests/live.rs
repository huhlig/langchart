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

use futures::StreamExt;
use langchart_adapters::llm::{LlmAdapter, LlmRequest, Message, ResponseFormat};
use langchart_llm_watsonx::{WatsonxAdapter, WatsonxConfig, WatsonxCredentials, WatsonxScope};

#[tokio::test]
async fn test_live_watsonx() {
    let api_key = match std::env::var("WATSONX_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            println!("Skipping live test: WATSONX_API_KEY not set");
            return;
        }
    };
    let project_id = match std::env::var("WATSONX_PROJECT_ID") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            println!("Skipping live test: WATSONX_PROJECT_ID not set");
            return;
        }
    };

    let service_url = std::env::var("WATSONX_SERVICE_URL")
        .unwrap_or_else(|_| "https://us-south.ml.cloud.ibm.com".to_string());

    let adapter = WatsonxAdapter::new(
        WatsonxConfig::new(
            service_url,
            "2024-05-31",
            WatsonxScope::Project(project_id),
        ),
        WatsonxCredentials::ApiKey(api_key),
    )
    .unwrap();

    let mut request = LlmRequest {
        model_policy: Default::default(),
        messages: vec![Message::User {
            content: "Say 'WatsonX test succeeded!' and nothing else.".to_owned(),
        }],
        tools: vec![],
        response_format: ResponseFormat::Text,
    };
    request.model_policy.model = Some("meta-llama/llama-3-3-70b-instruct".to_owned());
    request.model_policy.max_tokens = Some(30);

    println!("Testing complete_stream...");
    let stream_res = adapter.complete_stream(request.clone()).await.unwrap();
    let mut stream = stream_res;
    let mut collected_deltas = String::new();
    while let Some(event) = stream.next().await {
        println!("Stream event: {:?}", event);
        let event = event.expect("Stream event must succeed");
        if let langchart_adapters::llm::LlmStreamEvent::TextDelta { delta } = event {
            collected_deltas.push_str(&delta);
        }
    }
    println!("Collected stream text: {}", collected_deltas);
    assert!(!collected_deltas.is_empty());

    println!("Testing complete...");
    let result = adapter.complete(request).await;
    println!("WatsonX complete result: {:?}", result);
    let response = result.expect("complete() should succeed");
    println!("Live Response Content: {:?}", response.content);
    assert!(response.content.is_some());
}
