use langchart_adapters::llm::{
    FinishReason, LlmRequest, LlmResponse, Message, TokenUsage, ToolCall,
};

/// Convert an [`LlmRequest`] to a `genai::chat::ChatRequest`.
pub fn to_genai_request(req: &LlmRequest) -> genai::chat::ChatRequest {
    let mut messages: Vec<genai::chat::ChatMessage> = Vec::new();
    let mut system: Option<String> = None;

    for msg in &req.messages {
        match msg {
            Message::System { content } => {
                // genai takes system prompt as a separate field, not a message
                system = Some(content.clone());
            }
            Message::User { content } => {
                messages.push(genai::chat::ChatMessage::user(content));
            }
            Message::Assistant { content } => {
                messages.push(genai::chat::ChatMessage::assistant(content));
            }
            Message::Tool {
                tool_call_id,
                content,
            } => {
                messages.push(genai::chat::ToolResponse::new(tool_call_id, content).into());
            }
        }
    }

    let mut chat_req = genai::chat::ChatRequest::new(messages);
    if let Some(sys) = system {
        chat_req = chat_req.with_system(sys);
    }
    if !req.tools.is_empty() {
        chat_req = chat_req.with_tools(
            req.tools
                .iter()
                .map(|tool| {
                    genai::chat::Tool::new(&tool.name)
                        .with_description(&tool.description)
                        .with_schema(tool.parameters.clone())
                })
                .collect(),
        );
    }
    chat_req
}

/// Convert a `genai::chat::ChatResponse` to an [`LlmResponse`].
pub fn from_genai_response(resp: genai::chat::ChatResponse) -> LlmResponse {
    let model = resp.model_iden.model_name.to_string();

    let tool_calls: Vec<ToolCall> = match &resp.content {
        Some(genai::chat::MessageContent::ToolCalls(calls)) => calls
            .iter()
            .map(|tc| ToolCall {
                id: tc.call_id.clone(),
                name: tc.fn_name.clone(),
                arguments: tc.fn_arguments.clone(),
            })
            .collect(),
        _ => vec![],
    };

    let content = match &resp.content {
        Some(genai::chat::MessageContent::Text(t)) => Some(t.clone()),
        _ => None,
    };

    let usage = TokenUsage {
        prompt_tokens: resp.usage.prompt_tokens.unwrap_or(0).max(0) as u32,
        completion_tokens: resp.usage.completion_tokens.unwrap_or(0).max(0) as u32,
        total_tokens: resp.usage.total_tokens.unwrap_or(0).max(0) as u32,
    };

    let finish_reason = if !tool_calls.is_empty() {
        FinishReason::ToolCalls
    } else {
        FinishReason::Stop
    };

    LlmResponse {
        content,
        tool_calls,
        usage,
        finish_reason,
        refusal: None,
        reported_model: Some(model.clone()),
        model,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use langchart_adapters::llm::ToolDefinition;
    use langchart_model::policy::ModelPolicy;

    fn make_request(messages: Vec<Message>) -> LlmRequest {
        LlmRequest {
            model_policy: ModelPolicy {
                model: Some("gemini-2.0-flash".into()),
                ..Default::default()
            },
            messages,
            tools: vec![],
            response_format: Default::default(),
        }
    }

    // ── to_genai_request ─────────────────────────────────────────────────────

    #[test]
    fn system_message_goes_to_system_field() {
        let req = make_request(vec![
            Message::System {
                content: "You are helpful.".into(),
            },
            Message::User {
                content: "Hello".into(),
            },
        ]);
        let gr = to_genai_request(&req);
        assert_eq!(gr.system.as_deref(), Some("You are helpful."));
        assert_eq!(gr.messages.len(), 1);
    }

    #[test]
    fn user_and_assistant_messages_preserved() {
        let req = make_request(vec![
            Message::User {
                content: "ping".into(),
            },
            Message::Assistant {
                content: "pong".into(),
            },
        ]);
        let gr = to_genai_request(&req);
        assert_eq!(gr.messages.len(), 2);
        assert!(gr.system.is_none());
    }

    #[test]
    fn tool_result_encoded_as_native_tool_response() {
        let req = make_request(vec![Message::Tool {
            tool_call_id: "call_1".into(),
            content: "42".into(),
        }]);
        let gr = to_genai_request(&req);
        assert_eq!(gr.messages.len(), 1);
        assert!(matches!(gr.messages[0].role, genai::chat::ChatRole::Tool));
        match &gr.messages[0].content {
            genai::chat::MessageContent::ToolResponses(responses) => {
                assert_eq!(responses[0].call_id, "call_1");
                assert_eq!(responses[0].content, "42");
            }
            other => panic!("expected native tool response, got {other:?}"),
        }
    }

    #[test]
    fn tool_definitions_are_forwarded() {
        let mut req = make_request(vec![Message::User {
            content: "check weather".into(),
        }]);
        req.tools.push(ToolDefinition {
            name: "weather".into(),
            description: "Look up weather".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        });

        let gr = to_genai_request(&req);

        let tools = gr.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "weather");
        assert_eq!(tools[0].description.as_deref(), Some("Look up weather"));
        assert_eq!(
            tools[0].schema.as_ref().unwrap()["required"][0],
            serde_json::json!("city")
        );
    }

    #[test]
    fn no_system_message_leaves_system_none() {
        let req = make_request(vec![Message::User {
            content: "hi".into(),
        }]);
        let gr = to_genai_request(&req);
        assert!(gr.system.is_none());
    }

    // ── from_genai_response ──────────────────────────────────────────────────

    fn make_response(
        model: &str,
        content: Option<genai::chat::MessageContent>,
        usage: genai::chat::Usage,
    ) -> genai::chat::ChatResponse {
        use genai::{ModelIden, adapter::AdapterKind};
        let model_iden = ModelIden::new(AdapterKind::Gemini, model.to_string());
        genai::chat::ChatResponse {
            content,
            reasoning_content: None,
            model_iden: model_iden.clone(),
            provider_model_iden: model_iden,
            usage,
        }
    }

    #[test]
    fn text_content_extracted() {
        let resp = make_response(
            "gemini-2.0-flash",
            Some(genai::chat::MessageContent::Text("hello".into())),
            genai::chat::Usage::default(),
        );
        let r = from_genai_response(resp);
        assert_eq!(r.content.as_deref(), Some("hello"));
        assert!(r.tool_calls.is_empty());
        assert_eq!(r.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn tool_calls_extracted() {
        let tc = genai::chat::ToolCall {
            call_id: "call_abc".into(),
            fn_name: "my_tool".into(),
            fn_arguments: serde_json::json!({ "x": 1 }),
        };
        let resp = make_response(
            "gemini-2.0-flash",
            Some(genai::chat::MessageContent::ToolCalls(vec![tc])),
            genai::chat::Usage::default(),
        );
        let r = from_genai_response(resp);
        assert!(r.content.is_none());
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].id, "call_abc");
        assert_eq!(r.tool_calls[0].name, "my_tool");
        assert_eq!(r.tool_calls[0].arguments["x"], 1);
        assert_eq!(r.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn usage_fields_mapped() {
        let usage = genai::chat::Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
            ..Default::default()
        };
        let resp = make_response("gemini-2.0-flash", None, usage);
        let r = from_genai_response(resp);
        assert_eq!(r.usage.prompt_tokens, 10);
        assert_eq!(r.usage.completion_tokens, 20);
        assert_eq!(r.usage.total_tokens, 30);
    }

    #[test]
    fn model_name_preserved() {
        let resp = make_response("gemini-2.0-flash", None, genai::chat::Usage::default());
        let r = from_genai_response(resp);
        assert_eq!(r.model, "gemini-2.0-flash");
    }
}
