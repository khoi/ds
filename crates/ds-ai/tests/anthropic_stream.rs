use crate::support::{Reply, serve};
use ds_ai::{Context, Event, InputContent, Message, StopReason, ToolCall, ToolResult, anthropic};
use futures_util::StreamExt;
use serde_json::{Value, json};

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
    let server = serve([Reply::sse(sse).with_header("request-id", "req_anthropic")]).await;
    let model = anthropic::Model::new("claude-sonnet-4-5").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]).with_system("Be brief");
    let options = anthropic::Options::new("test-key").with_max_tokens(1024);

    let events = anthropic::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        events.first(),
        Some(&Ok(Event::TextDelta {
            content_index: 0,
            delta: "Hello".into(),
        }))
    );
    let response = done(&events);
    assert_eq!(response.id.as_deref(), Some("msg_1"));
    assert_eq!(response.content, [ds_ai::Content::Text("Hello".into())]);
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(response.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(response.usage.input, 12);
    assert_eq!(response.usage.output, 5);
    assert_eq!(response.usage.cache_read, 2);
    assert_eq!(response.usage.cache_write, 3);
    assert_eq!(
        response.metadata.request_id.as_deref(),
        Some("req_anthropic")
    );

    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /v1/messages HTTP/1.1\r\n"));
    assert!(request.contains("x-api-key: test-key\r\n"));
    assert!(request.contains("anthropic-version: 2023-06-01\r\n"));
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "claude-sonnet-4-5",
            "system": "Be brief",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }],
            "max_tokens": 1024,
            "stream": true
        })
    );
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
    let first_model =
        anthropic::Model::new("claude-sonnet-4-5").with_base_url(&first_server.base_url);
    let first_context = Context::new([Message::user("Edit")]);
    let options = anthropic::Options::new("test-key");

    let first_events = anthropic::stream(&first_model, &first_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        first_events.first(),
        Some(&Ok(Event::ReasoningDelta {
            content_index: 0,
            delta: " think".into(),
        }))
    );
    assert_eq!(
        first_events.get(1),
        Some(&Ok(Event::ToolCallDelta {
            content_index: 2,
            delta: "{\"path\":\"README.md\"".into(),
        }))
    );
    let response = done(&first_events);
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.usage.reasoning, 3);
    assert_eq!(
        response.content,
        [
            ds_ai::Content::Reasoning("I think".into()),
            ds_ai::Content::Reasoning("[Reasoning redacted]".into()),
            ds_ai::Content::ToolCall(ToolCall {
                id: "toolu_1".into(),
                name: "edit".into(),
                arguments: json!({"path": "README.md"}),
            }),
        ]
    );
    let restored: ds_ai::Response =
        serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap();
    first_server.requests().await;

    let second_sse = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model =
        anthropic::Model::new("claude-sonnet-4-5").with_base_url(&second_server.base_url);
    let second_context = Context::new([
        Message::assistant(restored),
        Message::tool_result(ToolResult::new(
            "toolu_1",
            "edit",
            [InputContent::text("done")],
        )),
    ]);

    anthropic::stream(&second_model, &second_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

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
                    "is_error": false
                }]
            }
        ])
    );
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
    let model = anthropic::Model::new("claude-sonnet-4-5").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = anthropic::Options::new("test-key").with_max_retries(1);

    let events = anthropic::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert!(matches!(events.as_slice(), [Ok(Event::Done(_))]));
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
    let model = anthropic::Model::new("claude-sonnet-4-5").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = anthropic::Options::new("test-key").with_cancellation(cancellation.clone());
    let mut response = anthropic::stream(&model, &context, &options).await.unwrap();

    assert!(matches!(
        response.next().await,
        Some(Ok(Event::TextDelta { .. }))
    ));
    cancellation.cancel();

    match response.next().await {
        Some(Err(ds_ai::Error::Cancelled {
            partial: Some(partial),
        })) => {
            assert_eq!(partial.stop_reason, StopReason::Aborted);
            assert_eq!(partial.content, [ds_ai::Content::Text("Visible".into())]);
        }
        event => panic!("unexpected cancellation event: {event:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn times_out_an_anthropic_stream_before_its_first_event() {
    let server = serve([Reply::open_sse(": keepalive\n\n")]).await;
    let model = anthropic::Model::new("claude-sonnet-4-5").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = anthropic::Options::new("test-key")
        .with_first_event_timeout(std::time::Duration::from_secs(5));
    let mut response = anthropic::stream(&model, &context, &options).await.unwrap();
    let next = tokio::spawn(async move { response.next().await });

    tokio::time::advance(std::time::Duration::from_secs(5)).await;

    match next.await.unwrap() {
        Some(Err(ds_ai::Error::Timeout {
            phase,
            partial: Some(partial),
        })) => {
            assert_eq!(phase, ds_ai::TimeoutPhase::FirstEvent);
            assert_eq!(partial.stop_reason, StopReason::Error);
        }
        event => panic!("unexpected timeout event: {event:?}"),
    }
}

fn done(events: &[Result<Event, ds_ai::Error>]) -> &ds_ai::Response {
    match events.last() {
        Some(Ok(Event::Done(response))) => response,
        _ => panic!("stream did not complete"),
    }
}
