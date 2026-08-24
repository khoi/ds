use crate::support::{Reply, serve};
use ds_ai::{
    Api, AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageFrame,
    AssistantToolCall, Context, Message, OpenAiResponsesOptions, ProviderId, StopReason,
    StreamOptions, TextContent, ThinkingContent, assistant_message_event_to_frame, builtin_model,
    reduce_assistant_message_frames,
};
use futures_util::StreamExt;

fn seed() -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::Other("test-api".into()),
        provider: ProviderId::new("test-provider"),
        model: "test-model".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::Pending,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

fn frame(event: AssistantMessageEvent) -> AssistantMessageFrame {
    assistant_message_event_to_frame(&event)
        .unwrap()
        .expect("non-terminal event")
}

fn text(value: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: value.into(),
        text_signature: None,
    })
}

fn thinking(value: impl Into<String>) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent {
        thinking: value.into(),
        thinking_signature: None,
        redacted: None,
    })
}

fn tool(id: &str, name: &str, arguments: serde_json::Value) -> AssistantToolCall {
    AssistantToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
        thought_signature: None,
        namespace: None,
    }
}

#[test]
fn uses_authoritative_text_end_content_and_signature() {
    let mut partial = seed();
    let mut frames = vec![frame(AssistantMessageEvent::Start {
        partial: partial.clone(),
    })];
    partial.content.push(text("Hello "));
    frames.push(frame(AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: partial.clone(),
    }));
    partial.content[0] = AssistantContent::Text(TextContent {
        text: "Hello world".into(),
        text_signature: Some("sig-text".into()),
    });
    frames.push(frame(AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "incorrect".into(),
        partial: partial.clone(),
    }));
    frames.push(frame(AssistantMessageEvent::TextEnd {
        content_index: 0,
        content: "Hello world".into(),
        partial,
    }));

    assert_eq!(
        reduce_assistant_message_frames(frames)
            .unwrap()
            .unwrap()
            .content,
        [AssistantContent::Text(TextContent {
            text: "Hello world".into(),
            text_signature: Some("sig-text".into()),
        })]
    );
}

#[test]
fn preserves_authoritative_thinking_metadata() {
    let mut partial = seed();
    let mut frames = vec![frame(AssistantMessageEvent::Start {
        partial: partial.clone(),
    })];
    partial
        .content
        .push(AssistantContent::Thinking(ThinkingContent {
            thinking: "[redacted]".into(),
            thinking_signature: Some("encrypted-start".into()),
            redacted: Some(true),
        }));
    frames.push(frame(AssistantMessageEvent::ThinkingStart {
        content_index: 0,
        partial: partial.clone(),
    }));
    partial.content[0] = AssistantContent::Thinking(ThinkingContent {
        thinking: "[redacted]".into(),
        thinking_signature: Some("encrypted-final".into()),
        redacted: Some(true),
    });
    frames.push(frame(AssistantMessageEvent::ThinkingEnd {
        content_index: 0,
        content: "[redacted]".into(),
        partial,
    }));

    assert_eq!(
        reduce_assistant_message_frames(frames)
            .unwrap()
            .unwrap()
            .content[0],
        AssistantContent::Thinking(ThinkingContent {
            thinking: "[redacted]".into(),
            thinking_signature: Some("encrypted-final".into()),
            redacted: Some(true),
        })
    );
}

#[test]
fn parses_unfinished_tool_json_and_uses_final_arguments() {
    let initial = vec![
        AssistantMessageFrame::Start {
            partial: Box::new(seed()),
        },
        AssistantMessageFrame::ToolCallStart {
            content_index: 0,
            tool_call: tool("initial-id", "write", serde_json::json!({})),
        },
        AssistantMessageFrame::ToolCallDelta {
            content_index: 0,
            delta: r#"{"path":"READ"#.into(),
        },
    ];
    let partial = reduce_assistant_message_frames(initial.clone())
        .unwrap()
        .unwrap();
    assert_eq!(
        partial.content[0],
        AssistantContent::ToolCall(tool(
            "initial-id",
            "write",
            serde_json::json!({"path": "READ"})
        ))
    );

    let complete = initial.into_iter().chain([
        AssistantMessageFrame::ToolCallDelta {
            content_index: 0,
            delta: r#"ME.md","lines":[1,2]}"#.into(),
        },
        AssistantMessageFrame::ToolCallEnd {
            content_index: 0,
            id: "final-id".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "final.md", "lines": [3]}),
            thought_signature: Some("thought".into()),
            namespace: Some("files".into()),
        },
    ]);
    let complete = reduce_assistant_message_frames(complete).unwrap().unwrap();
    assert_eq!(
        complete.content[0],
        AssistantContent::ToolCall(AssistantToolCall {
            id: "final-id".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "final.md", "lines": [3]}),
            thought_signature: Some("thought".into()),
            namespace: Some("files".into()),
        })
    );
}

#[tokio::test]
async fn round_trips_provider_content_from_authoritative_end_events() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"response\"}}\n\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg\",\"content\":[]}}\n\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg\",\"content\":[{\"type\":\"output_text\",\"text\":\"final text\"}]}}\n\ndata: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc\",\"call_id\":\"call\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\ndata: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc\",\"call_id\":\"call\",\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"pi\\\"}\"}}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"response\",\"status\":\"completed\",\"output\":[]}}\n\n",
    )])
    .await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let mut stream = ds_ai::openai::stream(
        &model,
        &Context::new([Message::user("Hello")]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let mut frames = Vec::new();

    while let Some(event) = stream.next().await {
        if let Some(frame) = assistant_message_event_to_frame(&event).unwrap() {
            frames.push(frame);
        }
    }

    let message = stream.result().await.unwrap();
    assert_eq!(
        reduce_assistant_message_frames(frames)
            .unwrap()
            .unwrap()
            .content,
        message.content
    );
    server.requests().await;
}

#[test]
fn treats_absent_end_metadata_as_authoritative() {
    let frames = [
        AssistantMessageFrame::Start {
            partial: Box::new(seed()),
        },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: TextContent {
                text: String::new(),
                text_signature: Some("stale-text".into()),
            },
        },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::new(),
            text_signature: None,
        },
        AssistantMessageFrame::ThinkingStart {
            content_index: 1,
            content: ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some("stale-thinking".into()),
                redacted: Some(true),
            },
        },
        AssistantMessageFrame::ThinkingEnd {
            content_index: 1,
            content: String::new(),
            thinking_signature: Some(String::new()),
            redacted: Some(false),
        },
        AssistantMessageFrame::ToolCallStart {
            content_index: 2,
            tool_call: AssistantToolCall {
                thought_signature: Some("stale-tool".into()),
                namespace: Some("stale-namespace".into()),
                ..tool("call", "read", serde_json::json!({}))
            },
        },
        AssistantMessageFrame::ToolCallEnd {
            content_index: 2,
            id: "call".into(),
            name: "read".into(),
            arguments: serde_json::json!({}),
            thought_signature: None,
            namespace: None,
        },
    ];

    assert_eq!(
        reduce_assistant_message_frames(frames)
            .unwrap()
            .unwrap()
            .content,
        [
            text(""),
            AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(String::new()),
                redacted: Some(false),
            }),
            AssistantContent::ToolCall(tool("call", "read", serde_json::json!({}))),
        ]
    );
}

#[test]
fn supports_interleaved_content_indexes() {
    let frames = [
        AssistantMessageFrame::Start {
            partial: Box::new(seed()),
        },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: TextContent {
                text: String::new(),
                text_signature: None,
            },
        },
        AssistantMessageFrame::ToolCallStart {
            content_index: 1,
            tool_call: tool("call", "lookup", serde_json::json!({})),
        },
        AssistantMessageFrame::ThinkingStart {
            content_index: 2,
            content: ThinkingContent {
                thinking: String::new(),
                thinking_signature: None,
                redacted: None,
            },
        },
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: "answer".into(),
        },
        AssistantMessageFrame::ToolCallDelta {
            content_index: 1,
            delta: r#"{"query":"pi"}"#.into(),
        },
        AssistantMessageFrame::ThinkingDelta {
            content_index: 2,
            delta: "check".into(),
        },
        AssistantMessageFrame::ToolCallEnd {
            content_index: 1,
            id: "call".into(),
            name: "lookup".into(),
            arguments: serde_json::json!({"query": "pi"}),
            thought_signature: None,
            namespace: None,
        },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: "answer".into(),
            text_signature: None,
        },
        AssistantMessageFrame::ThinkingEnd {
            content_index: 2,
            content: "check".into(),
            thinking_signature: None,
            redacted: None,
        },
    ];
    assert_eq!(
        reduce_assistant_message_frames(frames)
            .unwrap()
            .unwrap()
            .content,
        [
            text("answer"),
            AssistantContent::ToolCall(tool("call", "lookup", serde_json::json!({"query": "pi"}))),
            thinking("check"),
        ]
    );
}

#[test]
fn snapshots_events_and_reduces_without_mutating_frames() {
    let mut partial = seed();
    let start = frame(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });
    partial.usage.cost.total = 99.0;
    partial.content.push(AssistantContent::ToolCall(tool(
        "call",
        "run",
        serde_json::json!({"nested": {"value": "original"}}),
    )));
    let tool_start = frame(AssistantMessageEvent::ToolCallStart {
        content_index: 0,
        partial,
    });
    let mut reduced = reduce_assistant_message_frames([start, tool_start.clone()])
        .unwrap()
        .unwrap();
    assert_eq!(reduced.usage.cost.total, 0.0);
    reduced.content[0] = text("changed");

    let AssistantMessageFrame::ToolCallStart { tool_call, .. } = tool_start else {
        panic!("tool start")
    };
    assert_eq!(
        tool_call.arguments,
        serde_json::json!({"nested": {"value": "original"}})
    );
}

#[test]
fn omits_terminal_events() {
    let mut message = seed();
    message.stop_reason = StopReason::Stop;
    assert_eq!(
        assistant_message_event_to_frame(&AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: message.clone(),
        })
        .unwrap(),
        None
    );
    message.stop_reason = StopReason::Error;
    assert_eq!(
        assistant_message_event_to_frame(&AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: message,
        })
        .unwrap(),
        None
    );
}

#[test]
fn returns_none_without_a_start_frame() {
    assert_eq!(reduce_assistant_message_frames([]).unwrap(), None);
    assert_eq!(
        reduce_assistant_message_frames([AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: "x".into(),
        }])
        .unwrap(),
        None
    );
}

#[test]
fn rejects_invalid_frame_sequences() {
    let error = reduce_assistant_message_frames([
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: "x".into(),
        },
        AssistantMessageFrame::Start {
            partial: Box::new(seed()),
        },
    ])
    .unwrap_err();
    assert!(error.to_string().contains("before the start frame"));

    let error = reduce_assistant_message_frames([
        AssistantMessageFrame::Start {
            partial: Box::new(seed()),
        },
        AssistantMessageFrame::ToolCallStart {
            content_index: 0,
            tool_call: tool("call", "run", serde_json::json!({})),
        },
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: "wrong".into(),
        },
    ])
    .unwrap_err();
    assert!(error.to_string().contains("expected text block"));

    let error = reduce_assistant_message_frames([
        AssistantMessageFrame::Start {
            partial: Box::new(seed()),
        },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: TextContent {
                text: String::new(),
                text_signature: None,
            },
        },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::new(),
            text_signature: None,
        },
        AssistantMessageFrame::TextEnd {
            content_index: 0,
            content: String::new(),
            text_signature: None,
        },
    ])
    .unwrap_err();
    assert!(error.to_string().contains("follows the end"));

    let error = reduce_assistant_message_frames([
        AssistantMessageFrame::Start {
            partial: Box::new(seed()),
        },
        AssistantMessageFrame::TextStart {
            content_index: 1,
            content: TextContent {
                text: String::new(),
                text_signature: None,
            },
        },
    ])
    .unwrap_err();
    assert!(error.to_string().contains("would leave a gap"));
}

#[test]
fn rejects_event_indexes_with_the_wrong_block_kind() {
    let mut partial = seed();
    partial.content.push(thinking(""));
    let error = assistant_message_event_to_frame(&AssistantMessageEvent::TextStart {
        content_index: 0,
        partial,
    })
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("text_start event points to thinking block")
    );
}

#[test]
fn serializes_frames_with_pi_field_names_and_public_content() {
    let start = AssistantMessageFrame::Start {
        partial: Box::new(seed()),
    };
    let start = serde_json::to_value(start).unwrap();
    assert_eq!(start["type"], "start");
    assert_eq!(start["partial"]["role"], "assistant");
    assert_eq!(start["partial"]["stopReason"], "pending");
    assert!(start["partial"].get("responseId").is_none());

    let text_start = AssistantMessageFrame::TextStart {
        content_index: 3,
        content: TextContent {
            text: "visible".into(),
            text_signature: Some("signature".into()),
        },
    };
    let value = serde_json::to_value(&text_start).unwrap();
    assert_eq!(value["type"], "text_start");
    assert_eq!(value["contentIndex"], 3);
    assert_eq!(value["content"]["type"], "text");
    assert_eq!(value["content"]["text"], "visible");
    assert_eq!(
        serde_json::from_value::<AssistantMessageFrame>(value).unwrap(),
        text_start
    );
}
