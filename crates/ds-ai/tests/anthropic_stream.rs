use crate::support::{Reply, serve};
use ds_ai::{Context, Event, Message, StopReason, anthropic};
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

fn done(events: &[Result<Event, ds_ai::Error>]) -> &ds_ai::Response {
    match events.last() {
        Some(Ok(Event::Done(response))) => response,
        _ => panic!("stream did not complete"),
    }
}
