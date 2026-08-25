use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{
    AnthropicOptions, AssistantContent, AssistantMessageEvent, Context, ErrorReason, Message,
    OpenAiCodexResponsesOptions, OpenAiResponsesOptions, StreamOptions, Transport, anthropic,
    builtin_model, codex, openai,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use tokio::{net::TcpListener, sync::oneshot};
use tokio_tungstenite::{accept_async, tungstenite::Message as WebSocketMessage};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn exposes_progressive_openai_tool_arguments() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"weather\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"city\\\":\\\"Lo\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"London\\\"}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.id = "gpt-5.6".into();
    model.name = "gpt-5.6".into();
    model.base_url = server.base_url.clone();
    let options = OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };

    let events = openai::stream(
        &model.typed::<OpenAiResponsesOptions>().unwrap(),
        &Context::new([Message::user("Weather")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    let partial = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::ToolCallDelta { partial, .. } => Some(partial),
            _ => None,
        })
        .unwrap();
    let [AssistantContent::ToolCall(tool_call)] = partial.content.as_slice() else {
        panic!("expected one tool call");
    };
    assert_eq!(tool_call.arguments, json!({"city": "Lo"}));
    server.requests().await;
}

#[tokio::test]
async fn openai_keeps_one_timestamp_without_synthesizing_text_end() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let options = OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };

    let events = openai::stream(
        &model.typed::<OpenAiResponsesOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    assert_open_content_without_synthesized_end(&events);
    assert_one_timestamp(&events);
    server.requests().await;
}

#[tokio::test]
async fn openai_rechecks_cancellation_before_done() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let cancellation = CancellationToken::new();
    let options = OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            cancellation: cancellation.clone(),
            ..Default::default()
        },
        ..Default::default()
    };
    let stream = openai::stream(
        &model.typed::<OpenAiResponsesOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    );

    let events = cancel_after_text_end(stream, &cancellation).await;

    assert_cancelled_terminal(&events);
    server.requests().await;
}

#[tokio::test]
async fn exposes_progressive_anthropic_tool_arguments() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"weather\",\"input\":{}}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Lo\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let options = AnthropicOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };

    let events = anthropic::stream(
        &model.typed::<AnthropicOptions>().unwrap(),
        &Context::new([Message::user("Weather")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    assert_progressive_arguments(&events);
    server.requests().await;
}

#[tokio::test]
async fn anthropic_keeps_one_timestamp_without_synthesizing_text_end() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let options = AnthropicOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };

    let events = anthropic::stream(
        &model.typed::<AnthropicOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    assert_open_content_without_synthesized_end(&events);
    assert_one_timestamp(&events);
    server.requests().await;
}

#[tokio::test]
async fn anthropic_rechecks_cancellation_before_done() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let cancellation = CancellationToken::new();
    let options = AnthropicOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            cancellation: cancellation.clone(),
            ..Default::default()
        },
        ..Default::default()
    };
    let stream = anthropic::stream(
        &model.typed::<AnthropicOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    );

    let events = cancel_after_text_end(stream, &cancellation).await;

    assert_cancelled_terminal(&events);
    server.requests().await;
}

#[tokio::test]
async fn anthropic_reports_pre_cancelled_setup_as_aborted() {
    let server = serve(std::iter::empty::<Reply>()).await;
    let mut model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let options = AnthropicOptions {
        stream: StreamOptions {
            cancellation,
            ..Default::default()
        },
        ..Default::default()
    };

    let events = anthropic::stream(
        &model.typed::<AnthropicOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: ErrorReason::Aborted,
            error,
        }) if error.error_message.as_deref() == Some("No API key for provider: anthropic")
    ));
    assert!(server.requests().await.is_empty());
}

#[tokio::test]
async fn exposes_progressive_codex_sse_tool_arguments() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"weather\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"city\\\":\\\"Lo\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"London\\\"}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = codex_model(&server.base_url);
    let options = codex_options(Transport::Sse);

    let events = codex::stream(
        &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
        &Context::new([Message::user("Weather")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    assert_progressive_arguments(&events);
    server.request_bytes().await;
}

#[tokio::test]
async fn codex_sse_keeps_one_timestamp_without_synthesizing_text_end() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = codex_model(&server.base_url);
    let options = codex_options(Transport::Sse);

    let events = codex::stream(
        &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    assert_open_content_without_synthesized_end(&events);
    assert_one_timestamp(&events);
    server.request_bytes().await;
}

#[tokio::test]
async fn codex_sse_rechecks_cancellation_before_done() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = codex_model(&server.base_url);
    let cancellation = CancellationToken::new();
    let mut options = codex_options(Transport::Sse);
    options.stream.cancellation = cancellation.clone();
    let stream = codex::stream(
        &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    );

    let events = cancel_after_text_end(stream, &cancellation).await;

    assert_cancelled_terminal(&events);
    server.request_bytes().await;
}

#[tokio::test]
async fn exposes_progressive_codex_websocket_tool_arguments() {
    let (base_url, sent) = serve_codex_websocket([
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "weather",
                "arguments": ""
            }
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{\"city\":\"Lo"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{\"city\":\"London\"}"
            }
        }),
        json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "usage": {}}
        }),
    ])
    .await;
    let model = codex_model(&base_url);
    let options = codex_options(Transport::WebSocket);

    let events = codex::stream(
        &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
        &Context::new([Message::user("Weather")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    assert_progressive_arguments(&events);
    sent.await.unwrap();
}

#[tokio::test]
async fn preserves_codex_function_argument_prefixes_across_sse_and_websocket() {
    let provider_events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_prefix",
                "call_id": "call_prefix",
                "name": "lookup",
                "arguments": "{\"city\":\""
            }
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "Paris\"}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_prefix",
                "call_id": "call_prefix",
                "name": "lookup"
            }
        }),
        json!({
            "type": "response.completed",
            "response": {"id": "resp_prefix", "status": "completed", "usage": {}}
        }),
    ];

    for transport in [Transport::Sse, Transport::WebSocket] {
        let events = stream_codex_provider_events(transport, &provider_events, "Look up").await;
        let arguments = events
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::ToolCallStart { partial, .. }
                | AssistantMessageEvent::ToolCallDelta { partial, .. } => {
                    let AssistantContent::ToolCall(call) = &partial.content[0] else {
                        panic!("expected tool call partial");
                    };
                    Some(call.arguments.clone())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            arguments,
            [json!({}), json!({"city": "Paris"})],
            "{transport:?}"
        );
        assert_terminal_tool_arguments(&events, json!({"city": "Paris"}));
    }
}

#[tokio::test]
async fn preserves_codex_custom_input_prefixes_across_sse_and_websocket() {
    let provider_events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_prefix",
                "call_id": "call_prefix",
                "name": "query",
                "input": "ab"
            }
        }),
        json!({
            "type": "response.custom_tool_call_input.delta",
            "output_index": 0,
            "delta": "c"
        }),
        json!({
            "type": "response.custom_tool_call_input.done",
            "output_index": 0,
            "input": "abc"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_prefix",
                "call_id": "call_prefix",
                "name": "query",
                "input": "abc"
            }
        }),
        json!({
            "type": "response.completed",
            "response": {"id": "resp_prefix", "status": "completed", "usage": {}}
        }),
    ];

    for transport in [Transport::Sse, Transport::WebSocket] {
        let events = stream_codex_provider_events(transport, &provider_events, "Query").await;
        let deltas = events
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(deltas, ["{\"input\":\"ab", "c", "\"}"], "{transport:?}");
        assert_terminal_tool_arguments(&events, json!({"input": "abc"}));
    }
}

#[tokio::test]
async fn codex_websocket_keeps_one_timestamp_without_synthesizing_text_end() {
    let (base_url, sent) = serve_codex_websocket([
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_1", "type": "message", "content": []}
        }),
        json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "Hello"
        }),
        json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "usage": {}}
        }),
    ])
    .await;
    let model = codex_model(&base_url);
    let options = codex_options(Transport::WebSocket);

    let events = codex::stream(
        &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;

    assert_open_content_without_synthesized_end(&events);
    assert_one_timestamp(&events);
    sent.await.unwrap();
}

#[tokio::test]
async fn codex_websocket_rechecks_cancellation_before_done() {
    let (base_url, sent) = serve_codex_websocket([
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_1", "type": "message", "content": []}
        }),
        json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "Hello"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "msg_1",
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello"}]
            }
        }),
        json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "usage": {}}
        }),
    ])
    .await;
    let model = codex_model(&base_url);
    let cancellation = CancellationToken::new();
    let mut options = codex_options(Transport::WebSocket);
    options.stream.cancellation = cancellation.clone();
    let stream = codex::stream(
        &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    );

    let events = cancel_after_text_end(stream, &cancellation).await;

    assert_cancelled_terminal(&events);
    sent.await.unwrap();
}

fn assert_progressive_arguments(events: &[AssistantMessageEvent]) {
    let partial = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::ToolCallDelta { partial, .. } => Some(partial),
            _ => None,
        })
        .unwrap();
    let [AssistantContent::ToolCall(tool_call)] = partial.content.as_slice() else {
        panic!("expected one tool call");
    };
    assert_eq!(tool_call.arguments, json!({"city": "Lo"}));
}

fn assert_terminal_tool_arguments(events: &[AssistantMessageEvent], expected: Value) {
    let Some(AssistantMessageEvent::Done { message, .. }) = events.last() else {
        panic!("expected done event");
    };
    let [AssistantContent::ToolCall(tool_call)] = message.content.as_slice() else {
        panic!("expected one tool call");
    };
    assert_eq!(tool_call.arguments, expected);
}

fn assert_open_content_without_synthesized_end(events: &[AssistantMessageEvent]) {
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hello"
    )));
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done { .. })
    ));
    assert!(!events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextEnd { .. }
            | AssistantMessageEvent::ThinkingEnd { .. }
            | AssistantMessageEvent::ToolCallEnd { .. }
    )));
}

fn assert_one_timestamp(events: &[AssistantMessageEvent]) {
    let timestamps = events
        .iter()
        .map(|event| match event {
            AssistantMessageEvent::Start { partial }
            | AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolCallStart { partial, .. }
            | AssistantMessageEvent::ToolCallDelta { partial, .. }
            | AssistantMessageEvent::ToolCallEnd { partial, .. } => partial.timestamp,
            AssistantMessageEvent::Done { message, .. } => message.timestamp,
            AssistantMessageEvent::Error { error, .. } => error.timestamp,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(timestamps.len(), 1);
}

async fn cancel_after_text_end(
    mut stream: ds_ai::AssistantMessageEventStream,
    cancellation: &CancellationToken,
) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    loop {
        let event = stream.next().await.unwrap();
        let is_text_end = matches!(event, AssistantMessageEvent::TextEnd { .. });
        events.push(event);
        if is_text_end {
            break;
        }
    }
    cancellation.cancel();
    events.extend(stream.collect::<Vec<_>>().await);
    events
}

fn assert_cancelled_terminal(events: &[AssistantMessageEvent]) {
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: ErrorReason::Aborted,
            ..
        })
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::Done { .. }))
    );
    assert_one_timestamp(events);
}

fn codex_model(base_url: &str) -> ds_ai::Model {
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.base_url = base_url.into();
    model
}

fn codex_options(transport: Transport) -> OpenAiCodexResponsesOptions {
    OpenAiCodexResponsesOptions {
        stream: StreamOptions {
            api_key: Some(codex_token("acc_stream_core")),
            transport: Some(transport),
            ..Default::default()
        },
        ..Default::default()
    }
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

async fn stream_codex_provider_events(
    transport: Transport,
    provider_events: &[Value],
    prompt: &str,
) -> Vec<AssistantMessageEvent> {
    match transport {
        Transport::Sse => {
            let body = provider_events
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect::<Vec<_>>()
                .concat();
            let server = serve([Reply::sse(body)]).await;
            let events = codex::stream(
                &codex_model(&server.base_url)
                    .typed::<OpenAiCodexResponsesOptions>()
                    .unwrap(),
                &Context::new([Message::user(prompt)]),
                &codex_options(transport),
            )
            .collect::<Vec<_>>()
            .await;
            server.request_bytes().await;
            events
        }
        Transport::WebSocket => {
            let (base_url, sent) = serve_codex_websocket(provider_events.iter().cloned()).await;
            let events = codex::stream(
                &codex_model(&base_url)
                    .typed::<OpenAiCodexResponsesOptions>()
                    .unwrap(),
                &Context::new([Message::user(prompt)]),
                &codex_options(transport),
            )
            .collect::<Vec<_>>()
            .await;
            sent.await.unwrap();
            events
        }
        Transport::WebSocketCached | Transport::Auto => unreachable!(),
    }
}

async fn serve_codex_websocket(
    events: impl IntoIterator<Item = Value>,
) -> (String, oneshot::Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let events = events.into_iter().collect::<Vec<_>>();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(socket).await.unwrap();
        assert!(matches!(
            socket.next().await,
            Some(Ok(WebSocketMessage::Text(_)))
        ));
        for event in events {
            socket
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        sender.send(()).unwrap();
    });
    (format!("http://{address}"), receiver)
}
