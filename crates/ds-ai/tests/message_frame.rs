use crate::support::{Reply, serve};
use ds_ai::{
    Api, AssistantContent, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    AssistantMessageFrame, AssistantMessageFrameEncoder, AssistantToolCall, Context, DoneReason,
    ErrorReason, Message, OpenAiResponsesOptions, ProviderId, StopReason, StreamOptions,
    TextContent, ThinkingContent, builtin_model, reduce_assistant_message_frames,
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

fn frame(
    encoder: &mut AssistantMessageFrameEncoder,
    event: AssistantMessageEvent,
) -> AssistantMessageFrame {
    encoder.encode(&event).unwrap().expect("non-terminal event")
}

fn reduce(frames: impl IntoIterator<Item = AssistantMessageFrame>) -> AssistantMessage {
    reduce_assistant_message_frames(frames).unwrap().unwrap()
}

fn encoding_error(
    encoder: &mut AssistantMessageFrameEncoder,
    event: AssistantMessageEvent,
) -> String {
    encoder.encode(&event).unwrap_err().to_string()
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
fn normalizes_start_and_reconciles_queued_text() {
    let mut partial = seed();
    partial.stop_reason = StopReason::Stop;
    partial.error_message = Some("stale".into());
    partial.content.push(text("Hello world"));
    let events = [
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "Hell".into(),
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "o".into(),
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: " ".into(),
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "world".into(),
            partial,
        },
    ];

    let mut encoder = AssistantMessageFrameEncoder::new();
    let frames = events
        .iter()
        .filter_map(|event| encoder.encode(event).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(
        frames
            .iter()
            .map(|frame| serde_json::to_value(frame).unwrap()["type"].clone())
            .collect::<Vec<_>>(),
        [serde_json::json!("start"), serde_json::json!("text_start")]
    );
    let AssistantMessageFrame::Start { partial } = &frames[0] else {
        panic!("start frame")
    };
    assert!(partial.content.is_empty());
    assert_eq!(partial.stop_reason, StopReason::Pending);
    assert_eq!(partial.error_message, None);
    assert_eq!(reduce(frames).content, [text("Hello world")]);
}
#[test]
fn trims_only_the_covered_text_prefix() {
    let mut partial = seed();
    let mut encoder = AssistantMessageFrameEncoder::new();
    let mut frames = vec![frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    )];
    partial.content.push(text("Hello"));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    assert_eq!(
        encoder
            .encode(&AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "Hell".into(),
                partial: partial.clone(),
            })
            .unwrap(),
        None
    );
    let remainder = frame(
        &mut encoder,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "o world".into(),
            partial,
        },
    );
    assert_eq!(
        remainder,
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: " world".into(),
        }
    );
    frames.push(remainder);
    assert_eq!(reduce(frames).content, [text("Hello world")]);
}

#[test]
fn uses_authoritative_text_end_content_and_signature() {
    let mut partial = seed();
    let mut encoder = AssistantMessageFrameEncoder::new();
    let mut frames = vec![frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    )];
    partial.content.push(text("Hello "));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    partial.content[0] = AssistantContent::Text(TextContent {
        text: "Hello world".into(),
        text_signature: Some("sig-text".into()),
    });
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "incorrect".into(),
            partial: partial.clone(),
        },
    ));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::TextEnd {
            content_index: 0,
            content: "Hello world".into(),
            partial,
        },
    ));

    assert_eq!(
        reduce(frames).content,
        [AssistantContent::Text(TextContent {
            text: "Hello world".into(),
            text_signature: Some("sig-text".into()),
        })]
    );
}

#[test]
fn preserves_authoritative_thinking_metadata() {
    let mut partial = seed();
    let mut encoder = AssistantMessageFrameEncoder::new();
    let mut frames = vec![frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    )];
    partial
        .content
        .push(AssistantContent::Thinking(ThinkingContent {
            thinking: "[redacted]".into(),
            thinking_signature: Some("encrypted-start".into()),
            redacted: Some(true),
        }));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    partial.content[0] = AssistantContent::Thinking(ThinkingContent {
        thinking: "[redacted]".into(),
        thinking_signature: Some("encrypted-final".into()),
        redacted: Some(true),
    });
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ThinkingEnd {
            content_index: 0,
            content: "[redacted]".into(),
            partial,
        },
    ));

    assert_eq!(
        reduce(frames).content[0],
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
    let partial = reduce(initial.clone());
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
    let complete = reduce(complete);
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

#[test]
fn checkpoints_queued_tool_json_without_replaying_deltas() {
    let mut partial = seed();
    partial.content.push(AssistantContent::ToolCall(tool(
        "call",
        "write",
        serde_json::json!({"path": "README.md"}),
    )));
    let events = [
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
        AssistantMessageEvent::ToolCallStart {
            content_index: 0,
            partial: partial.clone(),
        },
        AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: r#"{"path":"READ"#.into(),
            partial: partial.clone(),
        },
        AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: r#"ME.md"}"#.into(),
            partial,
        },
    ];
    let mut encoder = AssistantMessageFrameEncoder::new();
    let frames = events
        .iter()
        .filter_map(|event| encoder.encode(event).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(frames.len(), 3);
    assert_eq!(
        frames[2],
        AssistantMessageFrame::ToolCallCheckpoint {
            content_index: 0,
            json: r#"{"path":"README.md"}"#.into(),
        }
    );
    assert_eq!(
        reduce(frames).content,
        [AssistantContent::ToolCall(tool(
            "call",
            "write",
            serde_json::json!({"path": "README.md"})
        ))]
    );
}

#[test]
fn streams_complete_tool_json_from_an_empty_start() {
    let mut partial = seed();
    let mut encoder = AssistantMessageFrameEncoder::new();
    let mut frames = vec![frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    )];
    partial.content.push(AssistantContent::ToolCall(tool(
        "call",
        "bash",
        serde_json::json!({}),
    )));
    frames.push(frame(
        &mut encoder,
        AssistantMessageEvent::ToolCallStart {
            content_index: 0,
            partial: partial.clone(),
        },
    ));
    let delta = frame(
        &mut encoder,
        AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: r#"{"command":"ls -la /tmp"}"#.into(),
            partial,
        },
    );
    assert_eq!(
        delta,
        AssistantMessageFrame::ToolCallDelta {
            content_index: 0,
            delta: r#"{"command":"ls -la /tmp"}"#.into(),
        }
    );
    frames.push(delta);
    assert_eq!(
        reduce(frames).content,
        [AssistantContent::ToolCall(tool(
            "call",
            "bash",
            serde_json::json!({"command": "ls -la /tmp"})
        ))]
    );
}

#[test]
fn stores_authoritative_tool_call_end_fields_in_the_frame() {
    let mut partial = seed();
    partial.content.push(AssistantContent::ToolCall(tool(
        "stale-id",
        "stale-name",
        serde_json::json!({"stale": true}),
    )));
    let event = AssistantMessageEvent::ToolCallEnd {
        content_index: 0,
        tool_call: AssistantToolCall {
            id: "final-id".into(),
            name: "final-name".into(),
            arguments: serde_json::json!({"value": 1}),
            thought_signature: Some("thought".into()),
            namespace: Some("files".into()),
        },
        partial,
    };
    let mut encoder = AssistantMessageFrameEncoder::new();
    frame(
        &mut encoder,
        AssistantMessageEvent::Start { partial: seed() },
    );
    let AssistantMessageEvent::ToolCallEnd { partial, .. } = &event else {
        unreachable!()
    };
    frame(
        &mut encoder,
        AssistantMessageEvent::ToolCallStart {
            content_index: 0,
            partial: partial.clone(),
        },
    );

    assert_eq!(
        frame(&mut encoder, event),
        AssistantMessageFrame::ToolCallEnd {
            content_index: 0,
            id: "final-id".into(),
            name: "final-name".into(),
            arguments: serde_json::json!({"value": 1}),
            thought_signature: Some("thought".into()),
            namespace: Some("files".into()),
        }
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
        &model.typed::<ds_ai::OpenAiResponsesOptions>().unwrap(),
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
    let mut encoder = AssistantMessageFrameEncoder::new();

    while let Some(event) = stream.next().await {
        if let Some(frame) = encoder.encode(&event).unwrap() {
            frames.push(frame);
        }
    }

    let message = stream.result().await.unwrap();
    assert_eq!(reduce(frames).content, message.content);
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
        reduce(frames).content,
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
        reduce(frames).content,
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
    partial.diagnostics = Some(vec![AssistantMessageDiagnostic {
        r#type: "test".into(),
        timestamp: 2,
        error: None,
        details: Some(std::collections::BTreeMap::from([(
            "value".into(),
            serde_json::json!("original"),
        )])),
    }]);
    let mut encoder = AssistantMessageFrameEncoder::new();
    let start = frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    );
    partial.diagnostics.as_mut().unwrap()[0]
        .details
        .as_mut()
        .unwrap()
        .insert("value".into(), serde_json::json!("mutated"));
    partial.usage.cost.total = 99.0;
    partial.content.push(AssistantContent::ToolCall(tool(
        "call",
        "run",
        serde_json::json!({"nested": {"value": "original"}}),
    )));
    let tool_start = frame(
        &mut encoder,
        AssistantMessageEvent::ToolCallStart {
            content_index: 0,
            partial,
        },
    );
    let mut reduced = reduce([start, tool_start.clone()]);
    assert_eq!(
        reduced.diagnostics.as_ref().unwrap()[0]
            .details
            .as_ref()
            .unwrap()["value"],
        serde_json::json!("original")
    );
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
    let mut completed = AssistantMessageFrameEncoder::new();
    frame(
        &mut completed,
        AssistantMessageEvent::Start {
            partial: message.clone(),
        },
    );
    message.stop_reason = StopReason::Stop;
    assert_eq!(
        completed
            .encode(&AssistantMessageEvent::Done {
                reason: DoneReason::Stop,
                message: message.clone(),
            })
            .unwrap(),
        None
    );
    message.stop_reason = StopReason::Error;
    assert_eq!(
        AssistantMessageFrameEncoder::new()
            .encode(&AssistantMessageEvent::Error {
                reason: ErrorReason::Error,
                error: message,
            })
            .unwrap(),
        None
    );
}

#[test]
fn enforces_encoder_protocol_order() {
    let mut encoder = AssistantMessageFrameEncoder::new();
    assert!(
        encoding_error(
            &mut encoder,
            AssistantMessageEvent::Done {
                reason: DoneReason::Stop,
                message: seed(),
            },
        )
        .contains("done event appears before start")
    );
    assert!(
        encoding_error(
            &mut encoder,
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "x".into(),
                partial: seed(),
            },
        )
        .contains("text_delta event appears before start")
    );

    let mut partial = seed();
    let mut encoder = AssistantMessageFrameEncoder::new();
    frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    );
    assert!(
        encoding_error(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )
        .contains("more than one start event")
    );
    partial.content.push(text(""));
    frame(
        &mut encoder,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        },
    );
    assert!(
        encoding_error(
            &mut encoder,
            AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: partial.clone(),
            },
        )
        .contains("starts more than once")
    );
    frame(
        &mut encoder,
        AssistantMessageEvent::TextEnd {
            content_index: 0,
            content: String::new(),
            partial: partial.clone(),
        },
    );
    assert!(
        encoding_error(
            &mut encoder,
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "x".into(),
                partial: partial.clone(),
            },
        )
        .contains("has not started")
    );
    assert_eq!(
        encoder
            .encode(&AssistantMessageEvent::Done {
                reason: DoneReason::Stop,
                message: partial.clone(),
            })
            .unwrap(),
        None
    );
    assert!(
        encoding_error(
            &mut encoder,
            AssistantMessageEvent::Error {
                reason: ErrorReason::Error,
                error: partial,
            },
        )
        .contains("follows a terminal event")
    );
}

#[test]
fn repairs_incomplete_streaming_json_values() {
    for (input, expected) in [
        (&"true"[..3], serde_json::json!(true)),
        (&"false"[..3], serde_json::json!(false)),
        (&"null"[..3], serde_json::Value::Null),
        ("1e", serde_json::json!(1)),
        (r#"{"a":1,"b":"#, serde_json::json!({"a": 1})),
    ] {
        let frames = [
            AssistantMessageFrame::Start {
                partial: Box::new(seed()),
            },
            AssistantMessageFrame::ToolCallStart {
                content_index: 0,
                tool_call: tool("call", "parse", serde_json::json!({})),
            },
            AssistantMessageFrame::ToolCallDelta {
                content_index: 0,
                delta: input.into(),
            },
        ];
        let message = reduce(frames);
        let AssistantContent::ToolCall(tool_call) = &message.content[0] else {
            panic!("tool call")
        };
        assert_eq!(tool_call.arguments, expected, "{input}");
    }
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
fn stops_reading_frames_after_a_prestart_protocol_error() {
    let frames = [
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: "x".into(),
        },
        AssistantMessageFrame::Start {
            partial: Box::new(seed()),
        },
    ]
    .into_iter()
    .chain(std::iter::once_with(|| panic!("read past protocol error")));

    assert!(
        reduce_assistant_message_frames(frames)
            .unwrap_err()
            .to_string()
            .contains("before the start frame")
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
    let mut encoder = AssistantMessageFrameEncoder::new();
    frame(
        &mut encoder,
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
    );
    partial.content.push(thinking(""));
    let error = encoder
        .encode(&AssistantMessageEvent::TextStart {
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
fn serializes_frames_with_wire_field_names_and_public_content() {
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

    let checkpoint = AssistantMessageFrame::ToolCallCheckpoint {
        content_index: 4,
        json: r#"{"value":1}"#.into(),
    };
    let value = serde_json::to_value(&checkpoint).unwrap();
    assert_eq!(value["type"], "toolcall_checkpoint");
    assert_eq!(value["contentIndex"], 4);
    assert_eq!(
        serde_json::from_value::<AssistantMessageFrame>(value).unwrap(),
        checkpoint
    );
}

#[test]
fn serializes_diagnostics_with_wire_field_names() {
    let mut message = seed();
    message.diagnostics = Some(vec![AssistantMessageDiagnostic {
        r#type: "transport".into(),
        timestamp: 2,
        error: None,
        details: Some(std::collections::BTreeMap::from([(
            "retry".into(),
            serde_json::json!(1),
        )])),
    }]);

    let value = serde_json::to_value(message).unwrap();
    assert_eq!(value["diagnostics"][0]["type"], "transport");
    assert_eq!(value["diagnostics"][0]["timestamp"], 2);
    assert_eq!(value["diagnostics"][0]["details"]["retry"], 1);
}
