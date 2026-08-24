use crate::support::{Reply, serve};
use ds_ai::{
    AnthropicMessagesCompatibility, AnthropicOptions, AssistantContent, AssistantMessage,
    AssistantMessageEvent, AssistantToolCall, CacheRetention, Context, InputContent, Message,
    ModelCompatibility, ResponseHook, StopReason, StreamOptions, TextContent, ThinkingContent,
    Tool, ToolResultMessage, anthropic, builtin_model,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn streams_anthropic_text_until_message_stop() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":12,\"output_tokens\":0,\"cache_read_input_tokens\":2,\"cache_creation_input_tokens\":3}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)
        .with_header("request-id", "req_anthropic")
        .with_header("anthropic-ratelimit-requests-limit", "100")
        .with_header("anthropic-ratelimit-requests-remaining", "90")
        .with_header("anthropic-ratelimit-requests-reset", "2026-08-24T12:00:00Z")
        .with_header("anthropic-ratelimit-tokens-limit", "10000")
        .with_header("anthropic-ratelimit-tokens-remaining", "9000")
        .with_header("anthropic-ratelimit-tokens-reset", "2026-08-24T12:01:00Z")])
    .await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]).with_system("Be brief");
    let options = options(|stream| stream.max_tokens = Some(1024));

    let events = events(&model, &context, &options).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Hello"
    )));
    let response = done(&events);
    assert_eq!(response.response_id.as_deref(), Some("msg_1"));
    assert_eq!(response.content, [text("Hello")]);
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(response.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(response.usage.input, 12);
    assert_eq!(response.usage.output, 5);
    assert_eq!(response.usage.cache_read, 2);
    assert_eq!(response.usage.cache_write, 3);
    assert_eq!(response.usage.cache_write_1h, Some(0));
    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /v1/messages HTTP/1.1\r\n"));
    assert!(request.contains("x-api-key: test-key\r\n"));
    assert!(request.contains("anthropic-version: 2023-06-01\r\n"));
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "claude-sonnet-4-5",
            "system": [{
                "type": "text",
                "text": "Be brief",
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Hello",
                    "cache_control": {"type": "ephemeral"}
                }]
            }],
            "max_tokens": 1024,
            "stream": true
        })
    );
}

#[tokio::test]
async fn exposes_anthropic_error_headers_to_the_response_hook() {
    let server = serve([Reply::json(
        429,
        json!({"error": {"type": "rate_limit_error", "message": "Too many requests"}}),
    )
    .with_header("request-id", "req_anthropic_failure")
    .with_header("anthropic-ratelimit-requests-limit", "100")
    .with_header("anthropic-ratelimit-requests-remaining", "0")
    .with_header("anthropic-ratelimit-tokens-limit", "10000")
    .with_header("anthropic-ratelimit-tokens-remaining", "200")])
    .await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let responses = Arc::new(Mutex::new(Vec::new()));
    let captured = responses.clone();
    let mut options = options(|_| {});
    options.stream.on_response = Some(ResponseHook::new(move |response, _| {
        let captured = captured.clone();
        async move {
            captured.lock().unwrap().push(response);
            Ok(())
        }
    }));
    let context = Context::new([Message::user("Hello")]);

    let events = events(&model, &context, &options).await;
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider returned HTTP 429: Too many requests")
    );
    let response = responses.lock().unwrap()[0].clone();
    assert_eq!(response.status, 429);
    assert_eq!(
        response.headers.get("request-id").map(String::as_str),
        Some("req_anthropic_failure")
    );
    assert_eq!(
        response
            .headers
            .get("anthropic-ratelimit-requests-remaining")
            .map(String::as_str),
        Some("0")
    );
    assert_eq!(
        response
            .headers
            .get("anthropic-ratelimit-tokens-remaining")
            .map(String::as_str),
        Some("200")
    );
    server.requests().await;
}

#[tokio::test]
async fn streams_and_replays_anthropic_thinking_and_tool_calls() {
    let first_sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_thinking\",\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"I\",\"signature\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" think\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_1\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"encrypted\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"edit\",\"input\":{}}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"README.md\\\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"}\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8,\"output_tokens_details\":{\"thinking_tokens\":3}}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(first_sse)]).await;
    let first_model = model("claude-sonnet-4-5", &first_server.base_url);
    let first_context = Context::new([Message::user("Edit")]);
    let options = options(|_| {});

    let first_events = events(&first_model, &first_context, &options).await;

    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial,
        } if matches!(
            partial.content.first(),
            Some(AssistantContent::Thinking(ThinkingContent { thinking, .. })) if thinking == "I"
        )
    )));
    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta,
            ..
        } if delta == " think"
    )));
    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ToolCallDelta {
            content_index: 2,
            delta,
            ..
        } if delta == "{\"path\":\"README.md\""
    )));
    let response = done(&first_events);
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.usage.reasoning, Some(3));
    assert_eq!(
        response.content,
        [
            AssistantContent::Thinking(ThinkingContent {
                thinking: "I think".into(),
                thinking_signature: Some("sig_1".into()),
                redacted: None,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "[Reasoning redacted]".into(),
                thinking_signature: Some("encrypted".into()),
                redacted: Some(true),
            }),
            AssistantContent::ToolCall(AssistantToolCall {
                id: "toolu_1".into(),
                name: "edit".into(),
                arguments: json!({"path": "README.md"}),
                thought_signature: None,
                namespace: None,
            }),
        ]
    );
    let restored: AssistantMessage =
        serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap();
    first_server.requests().await;

    let second_sse = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model = model("claude-sonnet-4-5", &second_server.base_url);
    let second_context = Context::new([
        Message::assistant(restored),
        Message::tool_result(ToolResultMessage::new(
            "toolu_1",
            "edit",
            [InputContent::text("done")],
        )),
    ]);

    let second_events = events(&second_model, &second_context, &options).await;
    done(&second_events);

    let request = second_server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["messages"],
        json!([
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I think", "signature": "sig_1"},
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "edit",
                        "input": {"path": "README.md"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "done"}],
                    "is_error": false,
                    "cache_control": {"type": "ephemeral"}
                }]
            }
        ])
    );
}

#[tokio::test]
async fn preserves_anthropic_start_content_and_refusal_details() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_refusal\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Blocked\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"type\":\"refusal\",\"category\":\"policy\",\"explanation\":\"Request denied\"}},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-fable-5", &server.base_url);
    let context = Context::new([Message::user("Blocked request")]);
    let events = events(&model, &context, &options(|_| {})).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial,
        } if partial.content == [text("Blocked")]
    )));
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.raw_stop_reason.as_deref(), Some("refusal"));
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider response failed: Request denied")
    );
    assert_eq!(error.content, [text("Blocked")]);
}

#[tokio::test]
async fn rejects_sensitive_and_unknown_anthropic_stop_reasons() {
    for (reason, expected) in [
        ("sensitive", "Provider stopped with: sensitive"),
        ("new_reason", "Unhandled stop reason: new_reason"),
    ] {
        let sse = [
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"",
            reason,
            "\"},\"usage\":{}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]
        .concat();
        let server = serve([Reply::sse(sse)]).await;
        let model = model("claude-haiku-4-5", &server.base_url);
        let context = Context::new([Message::user("Blocked request")]);
        let events = events(&model, &context, &options(|_| {})).await;
        let error = failed(&events);

        assert_eq!(error.stop_reason, StopReason::Error);
        assert_eq!(error.raw_stop_reason.as_deref(), Some(reason));
        assert_eq!(
            error.error_message.as_deref(),
            Some(format!("provider response failed: {expected}").as_str())
        );
    }
}

#[tokio::test]
async fn retries_anthropic_before_streaming_starts() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([
        Reply::json(
            529,
            json!({"error": {"type": "overloaded_error", "message": "busy"}}),
        )
        .with_header("retry-after-ms", "0"),
        Reply::sse(completed),
    ])
    .await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|stream| stream.max_retries = Some(1));

    let events = events(&model, &context, &options).await;

    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test]
async fn cancels_an_active_anthropic_stream_with_partial_content() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_cancel\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Visible\"}}\n\n",
    ]
    .concat();
    let server = serve([Reply::open_sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|stream| stream.cancellation = cancellation.clone());
    let mut response = anthropic::stream(&model, &context, &options);

    while !matches!(
        response.next().await,
        Some(AssistantMessageEvent::TextDelta { .. })
    ) {}
    cancellation.cancel();

    match response.next().await {
        Some(AssistantMessageEvent::Error { reason, error }) => {
            assert_eq!(reason, StopReason::Aborted);
            assert_eq!(error.stop_reason, StopReason::Aborted);
            assert_eq!(error.error_message.as_deref(), Some("request cancelled"));
            assert_eq!(error.content, [text("Visible")]);
        }
        event => panic!("unexpected cancellation event: {event:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn times_out_an_anthropic_stream_before_its_first_event() {
    let server = serve([Reply::open_sse(": keepalive\n\n")]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|stream| {
        stream.timeout = Some(std::time::Duration::from_secs(5));
    });
    let mut response = anthropic::stream(&model, &context, &options);
    let next = tokio::spawn(async move { response.next().await });

    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_secs(5)).await;

    match next.await.unwrap() {
        Some(AssistantMessageEvent::Error { reason, error }) => {
            assert_eq!(reason, StopReason::Error);
            assert_eq!(error.stop_reason, StopReason::Error);
            assert_eq!(
                error.error_message.as_deref(),
                Some("provider timed out during Overall")
            );
        }
        event => panic!("unexpected timeout event: {event:?}"),
    }
}

#[tokio::test]
async fn maps_anthropic_pause_turn_to_stop() {
    let sse = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"pause_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let response = done(&events);
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(response.raw_stop_reason.as_deref(), Some("pause_turn"));
}

#[tokio::test]
async fn rejects_anthropic_stream_closure_before_message_stop() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_partial\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Partial\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let error = failed(&events);
    assert_eq!(error.response_id.as_deref(), Some("msg_partial"));
    assert_eq!(error.content, [text("Partial")]);
    assert_eq!(error.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider stream ended before a terminal event")
    );
}

#[tokio::test]
async fn rejects_an_anthropic_error_event_with_partial_content() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_error\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Visible\"}}\n\n",
        "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.raw_stop_reason.as_deref(),
        Some("error.overloaded_error")
    );
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider response failed: busy")
    );
    assert_eq!(error.content, [text("Visible")]);
}

#[tokio::test]
async fn encodes_anthropic_generation_thinking_and_cache_options() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    let model = model("claude-opus-4-8", &server.base_url);
    let context = Context::new([Message::user_content([
        InputContent::text("Inspect"),
        InputContent::image("image/png", "iVBORw0KGgo="),
    ])])
    .with_system("Be brief")
    .with_tools([Tool::new(
        "inspect",
        "Inspect the input",
        json!({"type": "object", "properties": {}}),
    )]);
    let mut options = options(|stream| {
        stream.temperature = Some(0.2);
        stream.max_tokens = Some(4096);
        stream.cache_retention = CacheRetention::Long;
        stream.metadata.insert("user_id".into(), json!("user_1"));
    });
    options.thinking_enabled = Some(true);
    options.effort = Some(anthropic::Effort::High);
    options.thinking_display = Some(anthropic::ThinkingDisplay::Summarized);
    options.tool_choice = Some(anthropic::ToolChoice::Tool("inspect".into()));

    let response = events(&model, &context, &options).await;
    done(&response);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "claude-opus-4-8",
            "system": [{
                "type": "text",
                "text": "Be brief",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    }
                ]
            }],
            "tools": [{
                "name": "inspect",
                "description": "Inspect the input",
                "eager_input_streaming": true,
                "input_schema": {"type": "object", "properties": {}, "required": []},
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }],
            "max_tokens": 4096,
            "stream": true,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"},
            "metadata": {"user_id": "user_1"},
            "tool_choice": {"type": "tool", "name": "inspect"}
        })
    );
    assert!(body.get("temperature").is_none());
}

#[tokio::test]
async fn keeps_anthropic_temperature_with_disabled_thinking_and_cache() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    let mut model = model("claude-sonnet-4-5", &server.base_url);
    anthropic_compat(&mut model).supports_eager_tool_input_streaming = Some(false);
    let context = Context::new([Message::user("Hello")]).with_system("Be brief");
    let mut options = options(|stream| {
        stream.temperature = Some(0.0);
        stream.cache_retention = CacheRetention::None;
    });
    options.thinking_enabled = Some(false);

    let response = events(&model, &context, &options).await;
    done(&response);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("output_config").is_none());
    assert!(!request.contains("cache_control"));
    assert!(request.contains("anthropic-beta: interleaved-thinking-2025-05-14\r\n"));
}

#[tokio::test]
async fn encodes_legacy_tool_streaming_and_strict_schemas() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    let mut model = model("claude-opus-4-8", &server.base_url);
    let compat = anthropic_compat(&mut model);
    compat.supports_eager_tool_input_streaming = Some(false);
    compat.supports_strict_tools = Some(true);
    let context = Context::new([Message::user("Look up")]).with_tools([Tool::new(
        "lookup",
        "Look up a value",
        json!({
            "type": "object",
            "title": "LookupInput",
            "properties": {
                "value": {"type": "string"},
                "optional": {"type": "number"}
            },
            "required": ["value"]
        }),
    )
    .with_strict()]);

    let response = events(&model, &context, &options(|_| {})).await;
    done(&response);

    let request = server.requests().await.pop().unwrap();
    assert!(request.contains("anthropic-beta: fine-grained-tool-streaming-2025-05-14\r\n"));
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["tools"],
        json!([{
            "name": "lookup",
            "description": "Look up a value",
            "strict": true,
            "input_schema": {
                "type": "object",
                "title": "LookupInput",
                "properties": {
                    "value": {"type": "string"},
                    "optional": {"anyOf": [{"type": "number"}, {"type": "null"}]}
                },
                "required": ["value", "optional"],
                "additionalProperties": false
            },
            "cache_control": {"type": "ephemeral"}
        }])
    );
}

#[tokio::test]
async fn replays_empty_signature_thinking_as_text_unless_enabled() {
    let thinking = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_unsigned\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"unsigned\",\"signature\":\"\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"signed\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(thinking)]).await;
    let first_model = model("claude-sonnet-4-5", &first_server.base_url);
    let context = Context::new([Message::user("Think")]);
    let first_events = events(&first_model, &context, &options(|_| {})).await;
    let response = done(&first_events).clone();

    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let default_server = serve([Reply::sse(completed.clone())]).await;
    let default_model = model("claude-sonnet-4-5", &default_server.base_url);
    let default_events = events(
        &default_model,
        &Context::new([Message::assistant(response.clone())]),
        &options(|_| {}),
    )
    .await;
    done(&default_events);
    let default_request = default_server.requests().await.pop().unwrap();
    let default_body: Value =
        serde_json::from_str(default_request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        default_body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "unsigned"},
            {"type": "thinking", "thinking": "", "signature": "signed"}
        ])
    );

    let enabled_server = serve([Reply::sse(completed)]).await;
    let mut enabled_model = model("claude-sonnet-4-5", &enabled_server.base_url);
    anthropic_compat(&mut enabled_model).allow_empty_signature = Some(true);
    let enabled_events = events(
        &enabled_model,
        &Context::new([Message::assistant(response)]),
        &options(|_| {}),
    )
    .await;
    done(&enabled_events);
    let enabled_request = enabled_server.requests().await.pop().unwrap();
    let enabled_body: Value =
        serde_json::from_str(enabled_request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        enabled_body["messages"][0]["content"],
        json!([
            {"type": "thinking", "thinking": "unsigned", "signature": ""},
            {"type": "thinking", "thinking": "", "signature": "signed"}
        ])
    );
}

#[tokio::test]
async fn repairs_malformed_anthropic_event_and_tool_json() {
    let malformed = r#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1	col2\",\"unicode\":\"\u12xz\"}"}}

"#;
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_repair\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_repair\",\"name\":\"edit\",\"input\":{}}}\n\n",
        malformed,
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Edit")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let response = done(&events);
    assert_eq!(
        response.content,
        [AssistantContent::ToolCall(AssistantToolCall {
            id: "toolu_repair".into(),
            name: "edit".into(),
            arguments: json!({"path": "A\\H", "text": "col1\tcol2", "unicode": "\\u12xz"}),
            thought_signature: None,
            namespace: None,
        })]
    );
}

fn model(id: &str, base_url: &str) -> ds_ai::Model {
    let mut model = builtin_model("anthropic", id).unwrap();
    model.base_url = base_url.into();
    model
}

fn anthropic_compat(model: &mut ds_ai::Model) -> &mut AnthropicMessagesCompatibility {
    if !matches!(model.compat, Some(ModelCompatibility::Anthropic(_))) {
        model.compat = Some(ModelCompatibility::Anthropic(Default::default()));
    }
    let Some(ModelCompatibility::Anthropic(compat)) = &mut model.compat else {
        unreachable!()
    };
    compat
}

fn options(configure: impl FnOnce(&mut StreamOptions)) -> AnthropicOptions {
    let mut stream = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    configure(&mut stream);
    AnthropicOptions {
        stream,
        ..Default::default()
    }
}

async fn events(
    model: &ds_ai::Model,
    context: &Context,
    options: &AnthropicOptions,
) -> Vec<AssistantMessageEvent> {
    anthropic::stream(model, context, options).collect().await
}

fn text(value: &str) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: value.into(),
        text_signature: None,
    })
}

fn done(events: &[AssistantMessageEvent]) -> &AssistantMessage {
    match events.last() {
        Some(AssistantMessageEvent::Done { message, .. }) => message,
        _ => panic!("stream did not complete"),
    }
}

fn failed(events: &[AssistantMessageEvent]) -> &AssistantMessage {
    match events.last() {
        Some(AssistantMessageEvent::Error { error, .. }) => error,
        event => panic!("stream did not fail: {event:?}"),
    }
}
