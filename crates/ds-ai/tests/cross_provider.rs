use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{
    AnthropicOptions, Api, ApiStreamOptions, AssistantContent, AssistantMessage,
    AssistantMessageEvent, AssistantMessageEventStream, AssistantToolCall, CacheRetention, Context,
    InputContent, Message, OpenAiCodexResponsesOptions, OpenAiResponsesOptions, Provider,
    ProviderId, StopReason, StreamOptions, TextContent, ThinkingContent, ToolResultMessage,
    Transport, Usage, anthropic, builtin_model, codex, openai,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

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
    let mut source_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    source_model.base_url = source_server.base_url.clone();
    let mut source_stream = openai::stream(
        &source_model,
        &Context::new([Message::user("Run")]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let source_response = source_stream.result().await.unwrap();
    source_server.requests().await;

    let target_sse = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let target_server = serve([Reply::sse(target_sse)]).await;
    let mut target_model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    target_model.base_url = target_server.base_url.clone();
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
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cache_retention: CacheRetention::None,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();

    let request = target_server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["messages"],
        json!([
            {
                "role": "user",
                "content": "Run"
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
                        "content": "done",
                        "is_error": false
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_2_fc_2",
                        "content": "No result provided",
                        "is_error": true
                    }
                ]
            }
        ])
    );
}

#[tokio::test]
async fn repairs_a_direct_openai_tool_call_without_a_result() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = openai::Provider::new([model.clone()]);
    let assistant = AssistantMessage {
        content: vec![tool_call("call_orphan", "lookup")],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    };
    let context = Context::new([
        Message::user("Use lookup"),
        Message::assistant(assistant),
        Message::user("Never mind"),
    ]);

    provider
        .stream(
            &model,
            &context,
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
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

    let input = request_json(&server.requests().await[0])["input"]
        .as_array()
        .unwrap()
        .to_vec();
    assert_eq!(
        input
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .unwrap(),
        &json!({
            "type": "function_call_output",
            "call_id": "call_orphan",
            "output": "No result provided"
        })
    );
}

#[tokio::test]
async fn preserves_emoji_in_an_openai_tool_result() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = openai::Provider::new([model.clone()]);
    let assistant = AssistantMessage {
        content: vec![tool_call("call_emoji", "lookup")],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    };
    let context = Context::new([
        Message::user("Use lookup"),
        Message::assistant(assistant),
        Message::tool_result(ToolResultMessage::new(
            "call_emoji",
            "lookup",
            [InputContent::text("🙈 👍 ❤️ 🤔 🚀 こんにちは 你好")],
        )),
    ]);

    provider
        .stream(
            &model,
            &context,
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
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

    let input = request_json(&server.requests().await[0])["input"]
        .as_array()
        .unwrap()
        .to_vec();
    assert_eq!(
        input
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .unwrap()["output"],
        "🙈 👍 ❤️ 🤔 🚀 こんにちは 你好"
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
    let mut source_model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    source_model.base_url = source_server.base_url.clone();
    let source = anthropic::stream(
        &source_model,
        &Context::new([Message::user("Run")]),
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cache_retention: CacheRetention::None,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    source_server.requests().await;

    let target_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_target\",\"usage\":{}}}\n\n";
    let target_server = serve([Reply::sse(target_sse)]).await;
    let mut target_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    target_model.base_url = target_server.base_url.clone();
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
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cache_retention: CacheRetention::None,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();

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
                "id": "msg_pi_1"
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
            {"role": "user", "content": "Start"},
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

#[tokio::test]
async fn accepts_empty_turns_for_all_selected_provider_apis() {
    let openai_server = serve((0..4).map(|_| Reply::sse(openai_done()))).await;
    let mut openai_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    openai_model.base_url = openai_server.base_url.clone();
    let openai_source = AssistantSource {
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: openai_model.id.clone(),
    };
    for context in empty_contexts(&openai_source) {
        openai::stream(
            &openai_model,
            &context,
            &OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    cache_retention: CacheRetention::None,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .result()
        .await
        .unwrap();
    }
    let openai_requests = openai_server.requests().await;
    assert_eq!(openai_requests.len(), 4);
    let expected_openai_input = [
        json!([]),
        json!([{
            "role": "user",
            "content": [{"type": "input_text", "text": ""}]
        }]),
        json!([{
            "role": "user",
            "content": [{"type": "input_text", "text": "   \n\t  "}]
        }]),
        json!([
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello, how are you?"}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Please respond this time."}]
            }
        ]),
    ];
    for (request, expected) in openai_requests.iter().zip(expected_openai_input.iter()) {
        let body = request_json(request);
        assert_eq!(&body["input"], expected);
    }

    let anthropic_server = serve((0..4).map(|_| Reply::sse(anthropic_done()))).await;
    let mut anthropic_model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    anthropic_model.base_url = anthropic_server.base_url.clone();
    let anthropic_source = AssistantSource {
        api: Api::AnthropicMessages,
        provider: ProviderId::new("anthropic"),
        model: anthropic_model.id.clone(),
    };
    for context in empty_contexts(&anthropic_source) {
        anthropic::stream(
            &anthropic_model,
            &context,
            &AnthropicOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    cache_retention: CacheRetention::None,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .result()
        .await
        .unwrap();
    }
    let anthropic_requests = anthropic_server.requests().await;
    assert_eq!(anthropic_requests.len(), 4);
    let expected_anthropic_messages = [
        json!([]),
        json!([]),
        json!([]),
        json!([
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "Please respond this time."}
        ]),
    ];
    for (request, expected) in anthropic_requests.iter().zip(expected_anthropic_messages) {
        let body = request_json(request);
        assert_eq!(&body["messages"], &expected);
    }

    let codex_server = serve((0..4).map(|_| Reply::sse(codex_done()))).await;
    let mut codex_model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    codex_model.base_url = codex_server.base_url.clone();
    let codex_source = AssistantSource {
        api: Api::OpenAiCodexResponses,
        provider: ProviderId::new("openai-codex"),
        model: codex_model.id.clone(),
    };
    for context in empty_contexts(&codex_source) {
        codex::stream(
            &codex_model,
            &context,
            &OpenAiCodexResponsesOptions {
                stream: StreamOptions {
                    api_key: Some(codex_token("acc_empty")),
                    cache_retention: CacheRetention::None,
                    transport: Some(Transport::Sse),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .result()
        .await
        .unwrap();
    }
    let codex_requests = codex_server.request_bytes().await;
    assert_eq!(codex_requests.len(), 4);
    for (request, expected) in codex_requests.iter().zip(expected_openai_input.iter()) {
        let body = codex_request_json(request);
        assert_eq!(&body["input"], expected);
    }
}

#[tokio::test]
async fn cancels_selected_provider_streams_and_preserves_provider_usage() {
    let openai_server = serve([Reply::open_sse(openai_partial())]).await;
    let mut openai_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    openai_model.base_url = openai_server.base_url.clone();
    let openai_cancellation = CancellationToken::new();
    let mut openai_stream = openai::stream(
        &openai_model,
        &Context::new([Message::user("Cancel")]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cancellation: openai_cancellation.clone(),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let openai_error = cancel_after_text(&mut openai_stream, &openai_cancellation).await;
    assert_eq!(openai_error.stop_reason, StopReason::Aborted);
    assert_eq!(openai_error.usage, Usage::default());
    assert_eq!(
        openai_error.error_message.as_deref(),
        Some("OpenAI Responses stream ended before a terminal response event")
    );
    drop(openai_stream);
    openai_server.request_bytes().await;

    let anthropic_server = serve([Reply::open_sse(anthropic_partial())]).await;
    let mut anthropic_model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    anthropic_model.base_url = anthropic_server.base_url.clone();
    let anthropic_cancellation = CancellationToken::new();
    let mut anthropic_stream = anthropic::stream(
        &anthropic_model,
        &Context::new([Message::user("Cancel")]),
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cancellation: anthropic_cancellation.clone(),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let anthropic_error = cancel_after_text(&mut anthropic_stream, &anthropic_cancellation).await;
    assert_eq!(anthropic_error.stop_reason, StopReason::Aborted);
    assert_eq!(anthropic_error.usage.input, 11);
    assert_eq!(anthropic_error.usage.cache_read, 2);
    assert_eq!(anthropic_error.usage.cache_write, 3);
    assert_eq!(anthropic_error.usage.total_tokens, 16);
    assert_eq!(
        anthropic_error.error_message.as_deref(),
        Some("Request was aborted")
    );
    drop(anthropic_stream);
    anthropic_server.request_bytes().await;

    let codex_server = serve([Reply::open_sse(codex_partial())]).await;
    let mut codex_model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    codex_model.base_url = codex_server.base_url.clone();
    let codex_cancellation = CancellationToken::new();
    let mut codex_stream = codex::stream(
        &codex_model,
        &Context::new([Message::user("Cancel")]),
        &OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(codex_token("acc_cancel")),
                cancellation: codex_cancellation.clone(),
                transport: Some(Transport::Sse),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let codex_error = cancel_after_text(&mut codex_stream, &codex_cancellation).await;
    assert_eq!(codex_error.stop_reason, StopReason::Aborted);
    assert_eq!(codex_error.usage, Usage::default());
    drop(codex_stream);
    codex_server.request_bytes().await;
}

#[tokio::test]
async fn rejects_already_cancelled_requests_without_connecting() {
    let openai_server = serve(std::iter::empty::<Reply>()).await;
    let mut openai_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    openai_model.base_url = openai_server.base_url.clone();
    let openai_cancellation = CancellationToken::new();
    openai_cancellation.cancel();
    let openai_response = openai::stream(
        &openai_model,
        &Context::new([Message::user("Cancel")]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cancellation: openai_cancellation,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_eq!(openai_response.stop_reason, StopReason::Aborted);
    assert_eq!(
        openai_response.error_message.as_deref(),
        Some("Request aborted")
    );
    assert!(openai_server.requests().await.is_empty());

    let anthropic_server = serve(std::iter::empty::<Reply>()).await;
    let mut anthropic_model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    anthropic_model.base_url = anthropic_server.base_url.clone();
    let anthropic_cancellation = CancellationToken::new();
    anthropic_cancellation.cancel();
    let anthropic_response = anthropic::stream(
        &anthropic_model,
        &Context::new([Message::user("Cancel")]),
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cancellation: anthropic_cancellation,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_eq!(anthropic_response.stop_reason, StopReason::Aborted);
    assert_eq!(
        anthropic_response.error_message.as_deref(),
        Some("Request was aborted")
    );
    assert!(anthropic_server.requests().await.is_empty());

    let codex_server = serve(std::iter::empty::<Reply>()).await;
    let mut codex_model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    codex_model.base_url = codex_server.base_url.clone();
    let codex_cancellation = CancellationToken::new();
    codex_cancellation.cancel();
    let codex_response = codex::stream(
        &codex_model,
        &Context::new([Message::user("Cancel")]),
        &OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(codex_token("acc_pre_cancel")),
                cancellation: codex_cancellation,
                transport: Some(Transport::Sse),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_eq!(codex_response.stop_reason, StopReason::Aborted);
    assert_eq!(
        codex_response.error_message.as_deref(),
        Some("Request was aborted")
    );
    assert!(codex_server.requests().await.is_empty());
}

#[tokio::test]
async fn completes_a_follow_up_after_cancelling_each_provider() {
    let openai_server = serve([Reply::open_sse(openai_partial())]).await;
    let mut openai_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    openai_model.base_url = openai_server.base_url.clone();
    let openai_cancellation = CancellationToken::new();
    let mut openai_stream = openai::stream(
        &openai_model,
        &Context::new([Message::user("Start")]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cancellation: openai_cancellation.clone(),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let openai_aborted = cancel_after_text(&mut openai_stream, &openai_cancellation).await;
    drop(openai_stream);
    openai_server.request_bytes().await;
    let openai_followup_server = serve([Reply::sse(openai_done())]).await;
    openai_model.base_url = openai_followup_server.base_url.clone();
    let openai_followup = openai::stream(
        &openai_model,
        &Context::new([
            Message::user("Start"),
            Message::assistant(openai_aborted),
            Message::user("Continue"),
        ]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_eq!(openai_followup.stop_reason, StopReason::Stop);
    let openai_followup_request = openai_followup_server.requests().await.pop().unwrap();
    assert_eq!(
        request_json(&openai_followup_request)["input"],
        json!([
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Start"}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );

    let anthropic_server = serve([Reply::open_sse(anthropic_partial())]).await;
    let mut anthropic_model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    anthropic_model.base_url = anthropic_server.base_url.clone();
    let anthropic_cancellation = CancellationToken::new();
    let mut anthropic_stream = anthropic::stream(
        &anthropic_model,
        &Context::new([Message::user("Start")]),
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cancellation: anthropic_cancellation.clone(),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let anthropic_aborted = cancel_after_text(&mut anthropic_stream, &anthropic_cancellation).await;
    drop(anthropic_stream);
    anthropic_server.request_bytes().await;
    let anthropic_followup_server = serve([Reply::sse(anthropic_followup_done())]).await;
    anthropic_model.base_url = anthropic_followup_server.base_url.clone();
    let anthropic_followup = anthropic::stream(
        &anthropic_model,
        &Context::new([
            Message::user("Start"),
            Message::assistant(anthropic_aborted),
            Message::user("Continue"),
        ]),
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_eq!(anthropic_followup.stop_reason, StopReason::Stop);
    let anthropic_followup_request = anthropic_followup_server.requests().await.pop().unwrap();
    assert_eq!(
        request_json(&anthropic_followup_request)["messages"],
        json!([
            {"role": "user", "content": "Start"},
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Continue",
                    "cache_control": {"type": "ephemeral"}
                }]
            }
        ])
    );

    let codex_server = serve([Reply::open_sse(codex_partial())]).await;
    let mut codex_model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    codex_model.base_url = codex_server.base_url.clone();
    let codex_cancellation = CancellationToken::new();
    let mut codex_stream = codex::stream(
        &codex_model,
        &Context::new([Message::user("Start")]),
        &OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(codex_token("acc_followup")),
                cancellation: codex_cancellation.clone(),
                transport: Some(Transport::Sse),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let codex_aborted = cancel_after_text(&mut codex_stream, &codex_cancellation).await;
    drop(codex_stream);
    codex_server.request_bytes().await;
    let codex_followup_server = serve([Reply::sse(codex_done())]).await;
    codex_model.base_url = codex_followup_server.base_url.clone();
    let codex_followup = codex::stream(
        &codex_model,
        &Context::new([
            Message::user("Start"),
            Message::assistant(codex_aborted),
            Message::user("Continue"),
        ]),
        &OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(codex_token("acc_followup")),
                transport: Some(Transport::Sse),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_eq!(codex_followup.stop_reason, StopReason::Stop);
    let codex_followup_request = codex_followup_server.request_bytes().await.pop().unwrap();
    assert_eq!(
        codex_request_json(&codex_followup_request)["input"],
        json!([
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Start"}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn preserves_totals_and_cache_accounting_across_consecutive_calls() {
    let openai_server = serve([
        Reply::sse(openai_usage_sse("resp_cache_write", 100, 0, 80, 2, 102)),
        Reply::sse(openai_usage_sse("resp_cache_read", 100, 80, 0, 3, 103)),
    ])
    .await;
    let mut openai_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    openai_model.base_url = openai_server.base_url.clone();
    let openai_context = Context::new([Message::user("Cache this context")]);
    let openai_first = openai::stream(
        &openai_model,
        &openai_context,
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cache_retention: CacheRetention::Short,
                session_id: Some("cache-session".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_usage_totals(&openai_first.usage, 20, 2, 0, 80, 102);
    let openai_second = openai::stream(
        &openai_model,
        &openai_context,
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cache_retention: CacheRetention::Short,
                session_id: Some("cache-session".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_usage_totals(&openai_second.usage, 20, 3, 80, 0, 103);
    let openai_requests = openai_server.requests().await;
    assert_eq!(openai_requests.len(), 2);
    for request in openai_requests {
        assert_eq!(request_json(&request)["prompt_cache_key"], "cache-session");
    }

    let anthropic_server = serve([
        Reply::sse(anthropic_usage_sse("msg_cache_write", 20, 0, 80, 2)),
        Reply::sse(anthropic_usage_sse("msg_cache_read", 20, 80, 0, 3)),
    ])
    .await;
    let mut anthropic_model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    anthropic_model.base_url = anthropic_server.base_url.clone();
    let anthropic_context = Context::new([Message::user("Cache this context")]);
    let anthropic_first = anthropic::stream(
        &anthropic_model,
        &anthropic_context,
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cache_retention: CacheRetention::Short,
                session_id: Some("cache-session".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_usage_totals(&anthropic_first.usage, 20, 2, 0, 80, 102);
    let anthropic_second = anthropic::stream(
        &anthropic_model,
        &anthropic_context,
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                cache_retention: CacheRetention::Short,
                session_id: Some("cache-session".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_usage_totals(&anthropic_second.usage, 20, 3, 80, 0, 103);
    assert_eq!(anthropic_server.requests().await.len(), 2);

    let codex_server = serve([
        Reply::sse(codex_usage_sse("resp_cache_write", 100, 0, 80, 2, 102)),
        Reply::sse(codex_usage_sse("resp_cache_read", 100, 80, 0, 3, 103)),
    ])
    .await;
    let mut codex_model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    codex_model.base_url = codex_server.base_url.clone();
    let codex_context = Context::new([Message::user("Cache this context")]);
    let codex_first = codex::stream(
        &codex_model,
        &codex_context,
        &OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(codex_token("acc_cache")),
                cache_retention: CacheRetention::Short,
                session_id: Some("cache-session".into()),
                transport: Some(Transport::Sse),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_usage_totals(&codex_first.usage, 20, 2, 0, 80, 102);
    let codex_second = codex::stream(
        &codex_model,
        &codex_context,
        &OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(codex_token("acc_cache")),
                cache_retention: CacheRetention::Short,
                session_id: Some("cache-session".into()),
                transport: Some(Transport::Sse),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_usage_totals(&codex_second.usage, 20, 3, 80, 0, 103);
    assert_eq!(codex_server.request_bytes().await.len(), 2);
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

fn empty_contexts(source: &AssistantSource) -> Vec<Context> {
    let mut empty_assistant = assistant(source);
    empty_assistant.stop_reason = StopReason::Stop;
    vec![
        Context::new([Message::user_content([])]),
        Context::new([Message::user("")]),
        Context::new([Message::user("   \n\t  ")]),
        Context::new([
            Message::user("Hello, how are you?"),
            Message::assistant(empty_assistant),
            Message::user("Please respond this time."),
        ]),
    ]
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

fn codex_request_json(request: &[u8]) -> Value {
    let body_start = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap()
        + 4;
    serde_json::from_slice(&zstd::stream::decode_all(&request[body_start..]).unwrap()).unwrap()
}

fn openai_done() -> &'static str {
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_done\",\"status\":\"completed\",\"usage\":{}}}\n\n"
}

fn codex_done() -> &'static str {
    openai_done()
}

fn anthropic_done() -> &'static str {
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

fn anthropic_followup_done() -> &'static str {
    concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_followup\",\"usage\":{}}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    )
}

fn openai_partial() -> &'static str {
    concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_cancel\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_cancel\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
    )
}

fn anthropic_partial() -> &'static str {
    concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_cancel\",\"usage\":{\"input_tokens\":11,\"cache_read_input_tokens\":2,\"cache_creation_input_tokens\":3}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Visible\"}}\n\n",
    )
}

fn codex_partial() -> &'static str {
    openai_partial()
}

fn openai_usage_sse(
    response_id: &str,
    input: u64,
    cache_read: u64,
    cache_write: u64,
    output: u64,
    total: u64,
) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "type": "response.completed",
            "response": {
                "id": response_id,
                "status": "completed",
                "usage": {
                    "input_tokens": input,
                    "input_tokens_details": {
                        "cached_tokens": cache_read,
                        "cache_write_tokens": cache_write
                    },
                    "output_tokens": output,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": total
                }
            }
        })
    )
}

fn codex_usage_sse(
    response_id: &str,
    input: u64,
    cache_read: u64,
    cache_write: u64,
    output: u64,
    total: u64,
) -> String {
    openai_usage_sse(response_id, input, cache_read, cache_write, output, total)
}

fn anthropic_usage_sse(
    message_id: &str,
    input: u64,
    cache_read: u64,
    cache_write: u64,
    output: u64,
) -> String {
    let start = json!({
        "type": "message_start",
        "message": {
            "id": message_id,
            "usage": {
                "input_tokens": input,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write
            }
        }
    });
    let delta = json!({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": output}
    });
    format!(
        "event: message_start\ndata: {start}\n\nevent: message_delta\ndata: {delta}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
    )
}

fn assert_usage_totals(
    usage: &Usage,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    total: u64,
) {
    assert_eq!(usage.input, input);
    assert_eq!(usage.output, output);
    assert_eq!(usage.cache_read, cache_read);
    assert_eq!(usage.cache_write, cache_write);
    assert_eq!(usage.total_tokens, total);
    assert_eq!(
        usage.total_tokens,
        usage.input + usage.output + usage.cache_read + usage.cache_write
    );
}

async fn cancel_after_text(
    stream: &mut AssistantMessageEventStream,
    cancellation: &CancellationToken,
) -> AssistantMessage {
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::TextDelta { .. } => cancellation.cancel(),
            AssistantMessageEvent::Error { error, .. } => return error,
            _ => {}
        }
    }
    panic!("stream ended without a cancellation error");
}

fn codex_token(account_id: &str) -> String {
    let payload = BASE64_URL_SAFE_NO_PAD.encode(
        serde_json::to_vec(&json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id}
        }))
        .unwrap(),
    );
    format!("aaa.{payload}.bbb")
}
