use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{
    AnthropicFallbackModel, AnthropicMessagesCompatibility, AnthropicOptions, Api,
    ApiStreamOptions, AssistantContent, AssistantMessage, AssistantToolCall, CacheRetention,
    Context, InputContent, Message, ModelCompatibility, ModelCost, ModelCostRates,
    OpenAiCodexResponsesOptions, OpenAiResponsesOptions, Provider, ProviderId, StopReason,
    StreamOptions, Tool, ToolResultMessage, Transport, Usage, builtin_model,
};
use serde_json::{Value, json};

#[tokio::test]
async fn routes_openai_specific_options() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let options = ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            temperature: Some(0.25),
            max_tokens: Some(4096),
            ..Default::default()
        },
        reasoning_effort: Some(ds_ai::openai::ReasoningEffort::High),
        reasoning_summary: Some(ds_ai::openai::ReasoningSummary::Detailed),
        service_tier: Some(ds_ai::openai::ServiceTier::Flex),
        tool_choice: Some(ds_ai::openai::ToolChoice::Required),
    });

    provider
        .stream(&model, &Context::new([Message::user("Hello")]), &options)
        .result()
        .await
        .unwrap();

    let request = server.requests().await.pop().unwrap();
    let payload = request_json(&request);
    assert_eq!(payload["max_output_tokens"], 4096);
    assert_eq!(payload["temperature"], 0.25);
    assert_eq!(
        payload["reasoning"],
        json!({"effort": "high", "summary": "detailed"})
    );
    assert_eq!(payload["service_tier"], "flex");
    assert_eq!(payload["tool_choice"], "required");
}

#[tokio::test]
async fn routes_named_openai_tool_choices() {
    for (tool_choice, expected) in [
        (
            ds_ai::openai::ToolChoice::Function("search".into()),
            json!({"type": "function", "name": "search"}),
        ),
        (
            ds_ai::openai::ToolChoice::Custom("query".into()),
            json!({"type": "custom", "name": "query"}),
        ),
    ] {
        let server = serve([Reply::sse(openai_done())]).await;
        let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
        model.base_url = server.base_url.clone();
        let provider = ds_ai::openai::Provider::new([model.clone()]);
        let options = ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            tool_choice: Some(tool_choice),
            ..Default::default()
        });

        provider
            .stream(&model, &Context::new([Message::user("Hello")]), &options)
            .result()
            .await
            .unwrap();

        let request = server.requests().await.pop().unwrap();
        assert_eq!(request_json(&request)["tool_choice"], expected);
    }
}

#[tokio::test]
async fn prices_openai_service_tiers_from_response_or_request() {
    for (response_tier, request_tier, multiplier) in [
        (Some("priority"), ds_ai::openai::ServiceTier::Priority, 2.5),
        (None, ds_ai::openai::ServiceTier::Flex, 0.5),
    ] {
        let service_tier = response_tier
            .map(|tier| format!(",\"service_tier\":\"{tier}\""))
            .unwrap_or_default();
        let sse = format!(
            "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_1\",\"status\":\"completed\"{service_tier},\"usage\":{{\"input_tokens\":100000,\"output_tokens\":100000,\"total_tokens\":200000}}}}}}\n\n"
        );
        let server = serve([Reply::sse(sse)]).await;
        let mut model = builtin_model("openai", "gpt-5.5").unwrap();
        model.base_url = server.base_url.clone();
        let provider = ds_ai::openai::Provider::new([model.clone()]);
        let result = provider
            .stream(
                &model,
                &Context::new([Message::user("Hello")]),
                &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                    stream: StreamOptions {
                        api_key: Some("test-key".into()),
                        ..Default::default()
                    },
                    service_tier: Some(request_tier),
                    ..Default::default()
                }),
            )
            .result()
            .await
            .unwrap();
        let mut expected = ds_ai::Usage {
            input: 100_000,
            output: 100_000,
            total_tokens: 200_000,
            reasoning: Some(0),
            ..Default::default()
        };
        model.calculate_cost(&mut expected);

        assert_eq!(result.usage.cost.input, expected.cost.input * multiplier);
        assert_eq!(result.usage.cost.output, expected.cost.output * multiplier);
        server.requests().await;
    }
}

#[tokio::test]
async fn preserves_openai_replay_data_in_public_messages() {
    let first = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Need lookup\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Need lookup\"}],\"encrypted_content\":\"secret\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Working\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Working\"}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"value\\\":\\\"hello\\\"}\",\"namespace\":\"dynamic_tools\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(first), Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let options = ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    });
    let tool = Tool::new(
        "lookup",
        "Look up a value",
        json!({"type": "object", "properties": {"value": {"type": "string"}}}),
    );
    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Hello")]).with_tools([tool.clone()]),
            &options,
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.response_id.as_deref(), Some("resp_1"));
    let AssistantContent::Thinking(thinking) = &result.content[0] else {
        panic!("missing thinking content");
    };
    assert_eq!(
        serde_json::from_str::<Value>(thinking.thinking_signature.as_deref().unwrap()).unwrap(),
        json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Need lookup"}],
            "encrypted_content": "secret"
        })
    );
    let AssistantContent::Text(text) = &result.content[1] else {
        panic!("missing text content");
    };
    assert_eq!(
        serde_json::from_str::<Value>(text.text_signature.as_deref().unwrap()).unwrap(),
        json!({"v": 1, "id": "msg_1", "phase": "final_answer"})
    );
    let AssistantContent::ToolCall(call) = &result.content[2] else {
        panic!("missing tool call");
    };
    assert_eq!(call.id, "call_1|fc_1");
    assert_eq!(call.namespace.as_deref(), Some("dynamic_tools"));

    provider
        .stream(
            &model,
            &Context::new([
                Message::assistant(result),
                Message::tool_result(ToolResultMessage::new(
                    "call_1|fc_1",
                    "lookup",
                    [InputContent::text("done")],
                )),
            ])
            .with_tools([tool]),
            &options,
        )
        .result()
        .await
        .unwrap();

    let requests = server.requests().await;
    let replay = request_json(&requests[1]);
    assert_eq!(replay["input"][0]["id"], "rs_1");
    assert_eq!(replay["input"][1]["id"], "msg_1");
    assert_eq!(replay["input"][2]["id"], "fc_1");
    assert_eq!(replay["input"][2]["namespace"], "dynamic_tools");
    assert_eq!(replay["input"][3]["output"], "done");
}

#[tokio::test]
async fn prices_codex_with_the_requested_tier_when_the_response_echoes_default() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"service_tier\":\"default\",\"usage\":{\"input_tokens\":100000,\"output_tokens\":100000,\"total_tokens\":200000}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("openai-codex", "gpt-5.5").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::codex::Provider::new([model.clone()]);
    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Hello")]),
            &ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
                stream: StreamOptions {
                    api_key: Some(token()),
                    transport: Some(Transport::Sse),
                    ..Default::default()
                },
                service_tier: Some(ds_ai::codex::ServiceTier::Priority),
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();
    let mut expected = ds_ai::Usage {
        input: 100_000,
        output: 100_000,
        total_tokens: 200_000,
        reasoning: Some(0),
        ..Default::default()
    };
    model.calculate_cost(&mut expected);

    assert_eq!(result.usage.cost.input, expected.cost.input * 2.5);
    assert_eq!(result.usage.cost.output, expected.cost.output * 2.5);
    server.request_bytes().await;
}

#[tokio::test]
async fn routes_anthropic_specific_options() {
    let server = serve([Reply::sse(anthropic_done())]).await;
    let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);
    let options = ApiStreamOptions::AnthropicMessages(AnthropicOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            temperature: Some(0.25),
            max_tokens: Some(4096),
            ..Default::default()
        },
        thinking_enabled: Some(true),
        thinking_budget_tokens: Some(2048),
        effort: None,
        thinking_display: Some(ds_ai::anthropic::ThinkingDisplay::Omitted),
        interleaved_thinking: Some(true),
        tool_choice: Some(ds_ai::anthropic::ToolChoice::Tool("search".into())),
    });

    provider
        .stream(&model, &Context::new([Message::user("Hello")]), &options)
        .result()
        .await
        .unwrap();

    let request = server.requests().await.pop().unwrap();
    assert!(request.contains("anthropic-beta: interleaved-thinking-2025-05-14\r\n"));
    let payload = request_json(&request);
    assert_eq!(payload["max_tokens"], 4096);
    assert_eq!(
        payload["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048, "display": "omitted"})
    );
    assert_eq!(
        payload["tool_choice"],
        json!({"type": "tool", "name": "search"})
    );
    assert!(payload.get("temperature").is_none());
}

#[tokio::test]
async fn applies_anthropic_model_compatibility() {
    let server = serve([Reply::sse(anthropic_done()), Reply::sse(anthropic_done())]).await;
    let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::Anthropic(
        AnthropicMessagesCompatibility {
            supports_eager_tool_input_streaming: Some(false),
            supports_long_cache_retention: Some(false),
            send_session_affinity_headers: Some(true),
            supports_cache_control_on_tools: Some(false),
            supports_temperature: Some(false),
            force_adaptive_thinking: Some(true),
            allow_empty_signature: Some(true),
            supports_strict_tools: Some(true),
            supports_tool_references: Some(false),
            ..Default::default()
        },
    ));
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);
    let context = Context::new([Message::user("Hello")])
        .with_system("System")
        .with_tools([Tool::new(
            "lookup",
            "Look up a value",
            json!({"type": "object", "properties": {"value": {"type": "string"}}}),
        )
        .with_strict()]);
    for thinking_enabled in [Some(true), None] {
        provider
            .stream(
                &model,
                &context,
                &ApiStreamOptions::AnthropicMessages(AnthropicOptions {
                    stream: StreamOptions {
                        api_key: Some("test-key".into()),
                        cache_retention: CacheRetention::Long,
                        session_id: Some("session-1".into()),
                        temperature: Some(0.25),
                        ..Default::default()
                    },
                    thinking_enabled,
                    ..Default::default()
                }),
            )
            .result()
            .await
            .unwrap();
    }

    let requests = server.requests().await;
    for request in &requests {
        assert!(request.contains("anthropic-beta: fine-grained-tool-streaming-2025-05-14\r\n"));
        assert!(request.contains("x-session-affinity: session-1\r\n"));
        assert!(!request.contains("interleaved-thinking-2025-05-14"));
        let payload = request_json(request);
        assert_eq!(
            payload["system"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert!(payload["tools"][0].get("eager_input_streaming").is_none());
        assert_eq!(payload["tools"][0]["strict"], true);
        assert!(payload["tools"][0].get("cache_control").is_none());
        assert!(payload.get("temperature").is_none());
    }
    assert_eq!(
        request_json(&requests[0])["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert!(request_json(&requests[1]).get("thinking").is_none());
}

#[tokio::test]
async fn uses_anthropic_fallback_models_and_pricing() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fallback\",\"model\":\"claude-fallback\",\"usage\":{\"input_tokens\":100000,\"output_tokens\":0,\"cache_read_input_tokens\":300000,\"cache_creation_input_tokens\":400000,\"cache_creation\":{\"ephemeral_1h_input_tokens\":250000}}}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":200000}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::Anthropic(
        AnthropicMessagesCompatibility {
            allowed_fallback_models: vec![AnthropicFallbackModel {
                provider: ProviderId::new("anthropic"),
                model: "claude-fallback".into(),
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 2.0,
                        output: 4.0,
                        cache_read: 1.0,
                        cache_write: 3.0,
                    },
                    tiers: Vec::new(),
                },
            }],
            ..Default::default()
        },
    ));
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);
    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Hello")]),
            &ApiStreamOptions::AnthropicMessages(AnthropicOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.model, model.id);
    assert_eq!(result.response_model.as_deref(), Some("claude-fallback"));
    assert_eq!(result.response_id.as_deref(), Some("msg_fallback"));
    assert_eq!(result.usage.total_tokens, 1_000_000);
    assert_eq!(result.usage.cache_write_1h, Some(250_000));
    assert_eq!(result.usage.cost.input, 0.2);
    assert_eq!(result.usage.cost.output, 0.8);
    assert_eq!(result.usage.cost.cache_read, 0.3);
    assert_eq!(result.usage.cost.cache_write, 1.45);
    assert_eq!(result.usage.cost.total, 2.75);

    let request = server.requests().await.pop().unwrap();
    assert!(request.contains("anthropic-beta: server-side-fallback-2026-07-01\r\n"));
    assert_eq!(
        request_json(&request)["fallbacks"],
        json!([{"model": "claude-fallback"}])
    );
}

#[tokio::test]
async fn shapes_anthropic_oauth_requests_and_tool_names() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-5\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_2\",\"name\":\"Read\",\"input\":{}}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(AssistantToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: json!({"command": "pwd"}),
            thought_signature: None,
            namespace: None,
        })],
        api: Api::AnthropicMessages,
        provider: ProviderId::new("anthropic"),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: Some("tool_use".into()),
        end_turn: None,
        timestamp: 1,
    };
    let mut tool_result = ToolResultMessage::new("call_1", "bash", [InputContent::text("done")]);
    tool_result.added_tool_names = Some(vec!["websearch".into()]);
    let context = Context::new([
        Message::assistant(assistant),
        Message::tool_result(tool_result),
    ])
    .with_system("System")
    .with_tools([
        Tool::new("read", "Read", json!({"type": "object"})),
        Tool::new("bash", "Run", json!({"type": "object"})),
        Tool::new("websearch", "Search", json!({"type": "object"})),
    ]);
    let result = provider
        .stream(
            &model,
            &context,
            &ApiStreamOptions::AnthropicMessages(AnthropicOptions {
                stream: StreamOptions {
                    api_key: Some("sk-ant-oat-test".into()),
                    ..Default::default()
                },
                tool_choice: Some(ds_ai::anthropic::ToolChoice::Tool("read".into())),
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let AssistantContent::ToolCall(call) = &result.content[0] else {
        panic!("missing tool call");
    };
    assert_eq!(call.name, "read");

    let request = server.requests().await.pop().unwrap();
    assert!(request.contains("authorization: Bearer sk-ant-oat-test\r\n"));
    assert!(!request.contains("x-api-key:"));
    assert!(request.contains("user-agent: claude-cli/2.1.75\r\n"));
    assert!(request.contains("x-app: cli\r\n"));
    assert!(request.contains("anthropic-beta: claude-code-20250219,oauth-2025-04-20\r\n"));
    let payload = request_json(&request);
    assert_eq!(
        payload["system"][0]["text"],
        "You are Claude Code, Anthropic's official CLI for Claude."
    );
    assert_eq!(payload["system"][1]["text"], "System");
    assert_eq!(payload["tools"][0]["name"], "Read");
    assert_eq!(payload["tools"][1]["name"], "Bash");
    assert_eq!(payload["tools"][2]["name"], "WebSearch");
    assert_eq!(payload["tools"][2]["defer_loading"], true);
    assert_eq!(
        payload["tool_choice"],
        json!({"type": "tool", "name": "Read"})
    );
    assert_eq!(payload["messages"][0]["content"][0]["name"], "Bash");
    assert_eq!(
        payload["messages"][1]["content"][0]["content"][0],
        json!({"type": "tool_reference", "tool_name": "WebSearch"})
    );
}

#[tokio::test]
async fn accepts_anthropic_header_owned_auth_without_oauth_shaping() {
    let server = serve([Reply::sse(anthropic_done())]).await;
    let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);
    provider
        .stream(
            &model,
            &Context::new([Message::user("Hello")]).with_system("System"),
            &ApiStreamOptions::AnthropicMessages(AnthropicOptions {
                stream: StreamOptions {
                    headers: [("Authorization".into(), Some("Bearer gateway-token".into()))].into(),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let request = server.requests().await.pop().unwrap();
    assert!(request.contains("authorization: Bearer gateway-token\r\n"));
    assert!(!request.contains("x-api-key:"));
    assert!(!request.contains("oauth-2025-04-20"));
    let payload = request_json(&request);
    assert_eq!(payload["system"].as_array().unwrap().len(), 1);
    assert_eq!(payload["system"][0]["text"], "System");
}

#[tokio::test]
async fn routes_codex_specific_options() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::codex::Provider::new([model.clone()]);
    let options = ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
        stream: StreamOptions {
            api_key: Some(token()),
            temperature: Some(0.25),
            transport: Some(Transport::Sse),
            ..Default::default()
        },
        reasoning_effort: Some(ds_ai::codex::ReasoningEffort::High),
        reasoning_summary: Some(ds_ai::codex::ReasoningSummary::Detailed),
        service_tier: Some(ds_ai::codex::ServiceTier::Priority),
        text_verbosity: Some(ds_ai::codex::TextVerbosity::High),
        tool_choice: Some(ds_ai::codex::ToolChoice::Required),
    });

    provider
        .stream(&model, &Context::new([Message::user("Hello")]), &options)
        .result()
        .await
        .unwrap();

    let request = server.request_bytes().await.pop().unwrap();
    let split = request
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap()
        + 4;
    let payload: Value =
        serde_json::from_slice(&zstd::stream::decode_all(&request[split..]).unwrap()).unwrap();
    assert_eq!(payload["temperature"], 0.25);
    assert_eq!(
        payload["reasoning"],
        json!({"effort": "high", "summary": "detailed"})
    );
    assert_eq!(payload["service_tier"], "priority");
    assert_eq!(payload["text"], json!({"verbosity": "high"}));
    assert_eq!(payload["tool_choice"], "required");
}

#[tokio::test]
async fn rejects_options_for_a_different_api() {
    let model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let result = provider
        .stream(
            &model,
            &Context::new([]),
            &ApiStreamOptions::AnthropicMessages(Default::default()),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("invalid request: OpenAI Responses options are required")
    );
}

fn request_json(request: &str) -> Value {
    serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

fn openai_done() -> &'static str {
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n"
}

fn anthropic_done() -> &'static str {
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

fn token() -> String {
    let payload = BASE64_URL_SAFE_NO_PAD.encode(
        json!({"https://api.openai.com/auth": {"chatgpt_account_id": "account"}}).to_string(),
    );
    format!("header.{payload}.signature")
}
