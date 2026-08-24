use crate::support::{Reply, serve};
use ds_ai::{
    AnthropicOptions, Api, ApiStreamOptions, AssistantContent, AssistantMessage, AssistantToolCall,
    CacheRetention, Context, Event, InputContent, Message, OpenAiResponsesOptions, Provider,
    ProviderId, StopReason, StreamOptions, TextContent, ThinkingContent, ToolResultMessage, Usage,
    anthropic, builtin_model, openai,
};
use futures_util::StreamExt;
use serde_json::{Value, json};

#[tokio::test]
async fn normalizes_a_cross_provider_tool_transcript() {
    let source_sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Need tools\"}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Running\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Running\",\"annotations\":[]}]}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1|fc_1\",\"name\":\"read\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":2,\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1|fc_1\",\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":3,\"item\":{\"id\":\"fc_2\",\"type\":\"function_call\",\"call_id\":\"call_2|fc_2\",\"name\":\"shell\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":3,\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":3,\"item\":{\"id\":\"fc_2\",\"type\":\"function_call\",\"call_id\":\"call_2|fc_2\",\"name\":\"shell\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_source\",\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":1,\"output_tokens_details\":{}}}}\n\n",
    ]
    .concat();
    let source_server = serve([Reply::sse(source_sse)]).await;
    let source_model = openai::Model::new("gpt-5.6").with_base_url(&source_server.base_url);
    let source_events = openai::stream(
        &source_model,
        &Context::new([Message::user("Run")]),
        &openai::Options::new("test-key"),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;
    let source_response = done(&source_events).clone();
    source_server.requests().await;

    let target_sse = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let target_server = serve([Reply::sse(target_sse)]).await;
    let target_model =
        anthropic::Model::new("claude-sonnet-4-5").with_base_url(&target_server.base_url);
    let target_context = Context::new([
        Message::user("Run"),
        Message::assistant(source_response),
        Message::tool_result(ToolResultMessage::new(
            "call_1|fc_1",
            "read",
            [InputContent::text("done")],
        )),
    ]);

    anthropic::stream(
        &target_model,
        &target_context,
        &anthropic::Options::new("test-key").with_cache_retention(CacheRetention::None),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    let request = target_server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["messages"],
        json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Run"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Need tools"},
                    {"type": "text", "text": "Running"},
                    {
                        "type": "tool_use",
                        "id": "call_1_fc_1",
                        "name": "read",
                        "input": {"path": "README.md"}
                    },
                    {
                        "type": "tool_use",
                        "id": "call_2_fc_2",
                        "name": "shell",
                        "input": {"command": "pwd"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1_fc_1",
                        "content": [{"type": "text", "text": "done"}],
                        "is_error": false
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_2_fc_2",
                        "content": [{"type": "text", "text": "No result provided"}],
                        "is_error": true
                    }
                ]
            }
        ])
    );
}

#[tokio::test]
async fn replays_an_anthropic_transcript_to_openai() {
    let source_sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_source\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Working\"}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu/1|foreign\",\"name\":\"inspect\",\"input\":{\"path\":\"README.md\"}}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let source_server = serve([Reply::sse(source_sse)]).await;
    let source_model =
        anthropic::Model::new("claude-sonnet-4-5").with_base_url(&source_server.base_url);
    let source = anthropic::stream(
        &source_model,
        &Context::new([Message::user("Run")]),
        &anthropic::Options::new("test-key").with_cache_retention(CacheRetention::None),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;
    let source = done(&source).clone();
    source_server.requests().await;

    let target_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_target\",\"usage\":{}}}\n\n";
    let target_server = serve([Reply::sse(target_sse)]).await;
    let target_model = openai::Model::new("gpt-5.6").with_base_url(&target_server.base_url);
    openai::stream(
        &target_model,
        &Context::new([
            Message::user("Run"),
            Message::assistant(source),
            Message::tool_result(ToolResultMessage::new(
                "toolu/1|foreign",
                "inspect",
                [InputContent::text("done")],
            )),
            Message::user("Continue"),
        ]),
        &openai::Options::new("test-key").with_cache_retention(CacheRetention::None),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    let request = target_server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["input"],
        json!([
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Run"}]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Working", "annotations": []}],
                "status": "completed",
                "id": "msg_ds_1_0"
            },
            {
                "type": "function_call",
                "id": "fc_smumt218o8l4c",
                "call_id": "toolu_1",
                "name": "inspect",
                "arguments": "{\"path\":\"README.md\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "toolu_1",
                "output": "done"
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn normalizes_openai_handoffs_across_models_and_providers() {
    let server = serve([Reply::sse(openai_done()), Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = openai::Provider::new([model.clone()]);
    let options = ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    });

    for source in [
        AssistantSource {
            api: Api::OpenAiResponses,
            provider: ProviderId::new("openai"),
            model: "gpt-5.5".into(),
        },
        AssistantSource {
            api: Api::OpenAiCodexResponses,
            provider: ProviderId::new("openai-codex"),
            model: model.id.clone(),
        },
    ] {
        let first_id = "call/first|item+first";
        let second_id = "call/second|item+second";
        let context = Context::new([
            Message::user("Start"),
            Message::assistant(AssistantMessage {
                content: vec![AssistantContent::Thinking(ThinkingContent {
                    thinking: "Discarded".into(),
                    thinking_signature: Some(
                        json!({"type": "reasoning", "id": "rs_aborted"}).to_string(),
                    ),
                    redacted: None,
                })],
                stop_reason: StopReason::Aborted,
                ..assistant(&source)
            }),
            Message::assistant(AssistantMessage {
                content: vec![
                    AssistantContent::Thinking(ThinkingContent {
                        thinking: "Reasoned".into(),
                        thinking_signature: Some(
                            json!({"type": "reasoning", "id": "rs_foreign"}).to_string(),
                        ),
                        redacted: None,
                    }),
                    AssistantContent::Text(TextContent {
                        text: "Visible".into(),
                        text_signature: Some(
                            json!({"v": 1, "id": "msg_foreign", "phase": "final_answer"})
                                .to_string(),
                        ),
                    }),
                    tool_call(first_id, "first"),
                    tool_call(second_id, "second"),
                ],
                stop_reason: StopReason::ToolUse,
                ..assistant(&source)
            }),
            Message::tool_result(ToolResultMessage::new(
                first_id,
                "first",
                [InputContent::text("done")],
            )),
            Message::user("Continue"),
        ]);

        provider
            .stream(&model, &context, &options)
            .result()
            .await
            .unwrap();
    }

    for (request, foreign) in server.requests().await.iter().zip([false, true]) {
        let input = request_json(request)["input"].as_array().unwrap().to_vec();
        assert!(!input.iter().any(|item| item["id"] == "rs_aborted"));
        assert!(!input.iter().any(|item| item["type"] == "reasoning"));
        let messages = input
            .iter()
            .filter(|item| item["type"] == "message" && item["role"] == "assistant")
            .collect::<Vec<_>>();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"][0]["text"], "Reasoned");
        assert_eq!(messages[1]["content"][0]["text"], "Visible");
        assert_ne!(messages[1]["id"], "msg_foreign");

        let calls = input
            .iter()
            .filter(|item| item["type"] == "function_call")
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert!(calls.iter().all(|call| call.get("namespace").is_none()));
        if foreign {
            assert!(calls.iter().all(|call| {
                call["id"]
                    .as_str()
                    .is_some_and(|id| id.starts_with("fc_") && id.len() <= 64)
            }));
        } else {
            assert!(calls.iter().all(|call| call.get("id").is_none()));
        }
        assert_eq!(calls[0]["call_id"], "call_first");
        assert_eq!(calls[1]["call_id"], "call_second");

        let outputs = input
            .iter()
            .filter(|item| item["type"] == "function_call_output")
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0]["call_id"], "call_first");
        assert_eq!(outputs[0]["output"], "done");
        assert_eq!(outputs[1]["call_id"], "call_second");
        assert_eq!(outputs[1]["output"], "No result provided");
    }
}

#[tokio::test]
async fn drops_foreign_anthropic_replay_data_when_model_ids_match() {
    let server = serve([Reply::sse(anthropic_done())]).await;
    let mut model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let provider = anthropic::Provider::new([model.clone()]);
    let source = AssistantSource {
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: model.id.clone(),
    };
    let context = Context::new([
        Message::user("Start"),
        Message::assistant(AssistantMessage {
            content: vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "Reasoned".into(),
                    thinking_signature: Some("foreign-signature".into()),
                    redacted: None,
                }),
                AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some("foreign-redacted".into()),
                    redacted: Some(true),
                }),
                AssistantContent::Text(TextContent {
                    text: "Visible".into(),
                    text_signature: Some("foreign-text-signature".into()),
                }),
            ],
            stop_reason: StopReason::Stop,
            ..assistant(&source)
        }),
    ]);

    provider
        .stream(
            &model,
            &context,
            &ApiStreamOptions::AnthropicMessages(AnthropicOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    cache_retention: CacheRetention::None,
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let payload = request_json(&server.requests().await.pop().unwrap());
    assert_eq!(
        payload["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "Start"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reasoned"},
                    {"type": "text", "text": "Visible"}
                ]
            }
        ])
    );
}

struct AssistantSource {
    api: Api,
    provider: ProviderId,
    model: String,
}

fn assistant(source: &AssistantSource) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: source.api.clone(),
        provider: source.provider.clone(),
        model: source.model.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

fn tool_call(id: &str, name: &str) -> AssistantContent {
    AssistantContent::ToolCall(AssistantToolCall {
        id: id.into(),
        name: name.into(),
        arguments: json!({"value": name}),
        thought_signature: Some("discarded".into()),
        namespace: Some("dynamic_tools".into()),
    })
}

fn request_json(request: &str) -> Value {
    serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

fn openai_done() -> &'static str {
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_done\",\"status\":\"completed\",\"usage\":{}}}\n\n"
}

fn anthropic_done() -> &'static str {
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

fn done(events: &[Result<Event, ds_ai::Error>]) -> &ds_ai::Response {
    match events.last() {
        Some(Ok(Event::Done(response))) => response,
        _ => panic!("stream did not complete"),
    }
}
