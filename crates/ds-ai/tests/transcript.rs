use ds_ai::{
    Api, AssistantContent, AssistantMessage, AssistantMessageDiagnostic, AssistantToolCall,
    ConstrainedSampling, ConstrainedSamplingStrictness, Context, GrammarVariants, InputContent,
    Message, ProviderId, StopReason, TextContent, ThinkingContent, Tool, ToolResultMessage, Usage,
    UsageCost, UserContent, UserMessage,
};

#[test]
fn serializes_user_messages_with_wire_roles_and_content_shapes() {
    let text = UserMessage::new("Hello", 42);
    assert_eq!(
        serde_json::to_value(&text).unwrap(),
        serde_json::json!({"role": "user", "content": "Hello", "timestamp": 42})
    );

    let blocks = UserMessage::with_blocks(
        [
            InputContent::text("Inspect"),
            InputContent::image("image/png", "aGVsbG8="),
        ],
        43,
    );
    assert_eq!(
        serde_json::to_value(&blocks).unwrap(),
        serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect"},
                {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}
            ],
            "timestamp": 43
        })
    );
    assert!(
        serde_json::from_value::<UserMessage>(serde_json::json!({
            "role": "assistant",
            "content": "Hello",
            "timestamp": 42
        }))
        .is_err()
    );
}

#[test]
fn serializes_tool_results_with_optional_execution_data() {
    let message = ToolResultMessage {
        tool_call_id: "call_1".into(),
        tool_name: "read".into(),
        content: vec![InputContent::text("done")],
        details: Some(serde_json::json!({"path": "README.md"})),
        usage: Some(ds_ai::Usage {
            input: 1,
            output: 2,
            total_tokens: 3,
            ..Default::default()
        }),
        added_tool_names: Some(vec!["write".into()]),
        is_error: false,
        timestamp: 44,
    };
    assert_eq!(
        serde_json::to_value(Message::tool_result(message)).unwrap(),
        serde_json::json!({
            "role": "toolResult",
            "toolCallId": "call_1",
            "toolName": "read",
            "content": [{"type": "text", "text": "done"}],
            "details": {"path": "README.md"},
            "usage": {
                "input": 1,
                "output": 2,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 3,
                "cost": {
                    "input": 0.0,
                    "output": 0.0,
                    "cacheRead": 0.0,
                    "cacheWrite": 0.0,
                    "total": 0.0
                }
            },
            "addedToolNames": ["write"],
            "isError": false,
            "timestamp": 44
        })
    );
}

#[test]
fn serializes_complete_assistant_messages_with_wire_fields() {
    let message = AssistantMessage {
        content: vec![
            AssistantContent::Text(TextContent {
                text: "Answer".into(),
                text_signature: Some("text-signature".into()),
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "Reason".into(),
                thinking_signature: Some("thinking-signature".into()),
                redacted: Some(false),
            }),
            AssistantContent::ToolCall(AssistantToolCall {
                id: "call_1".into(),
                name: "lookup".into(),
                arguments: serde_json::json!({"query": "pi"}),
                thought_signature: Some("tool-signature".into()),
                namespace: Some("tools".into()),
            }),
        ],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: "gpt-requested".into(),
        response_model: Some("gpt-actual".into()),
        response_id: Some("resp_1".into()),
        diagnostics: Some(vec![AssistantMessageDiagnostic {
            r#type: "transport".into(),
            timestamp: 45,
            error: None,
            details: Some(std::collections::BTreeMap::from([(
                "mode".into(),
                serde_json::json!("sse"),
            )])),
        }]),
        usage: Usage {
            input: 1,
            output: 2,
            cache_read: 3,
            cache_write: 4,
            cache_write_1h: Some(5),
            reasoning: Some(6),
            total_tokens: 15,
            cost: UsageCost {
                input: 0.1,
                output: 0.2,
                cache_read: 0.3,
                cache_write: 0.4,
                total: 1.0,
            },
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: Some("tool_use".into()),
        end_turn: Some(false),
        timestamp: 46,
    };
    let value = serde_json::to_value(&message).unwrap();

    assert_eq!(
        value,
        serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Answer", "textSignature": "text-signature"},
                {
                    "type": "thinking",
                    "thinking": "Reason",
                    "thinkingSignature": "thinking-signature",
                    "redacted": false
                },
                {
                    "type": "toolCall",
                    "id": "call_1",
                    "name": "lookup",
                    "arguments": {"query": "pi"},
                    "thoughtSignature": "tool-signature",
                    "namespace": "tools"
                }
            ],
            "api": "openai-responses",
            "provider": "openai",
            "model": "gpt-requested",
            "responseModel": "gpt-actual",
            "responseId": "resp_1",
            "diagnostics": [{"type": "transport", "timestamp": 45, "details": {"mode": "sse"}}],
            "usage": {
                "input": 1,
                "output": 2,
                "cacheRead": 3,
                "cacheWrite": 4,
                "cacheWrite1h": 5,
                "reasoning": 6,
                "totalTokens": 15,
                "cost": {
                    "input": 0.1,
                    "output": 0.2,
                    "cacheRead": 0.3,
                    "cacheWrite": 0.4,
                    "total": 1.0
                }
            },
            "stopReason": "toolUse",
            "rawStopReason": "tool_use",
            "endTurn": false,
            "timestamp": 46
        })
    );
    assert_eq!(
        serde_json::from_value::<AssistantMessage>(value).unwrap(),
        message
    );
}

#[test]
fn serializes_context_and_constrained_sampling() {
    let tools = [
        Tool::new("read", "Read", serde_json::json!({"type": "object"})).with_strict(),
        Tool {
            name: "match".into(),
            description: "Match".into(),
            parameters: serde_json::json!({"type": "object"}),
            constrained_sampling: Some(ConstrainedSampling::Grammar {
                variants: GrammarVariants {
                    openai_lark: Some("start: WORD".into()),
                    openai_regex: None,
                },
            }),
        },
        Tool {
            name: "plain".into(),
            description: "Plain".into(),
            parameters: serde_json::json!({"type": "object"}),
            constrained_sampling: Some(ConstrainedSampling::Disabled),
        },
    ];
    let context = Context {
        system_prompt: Some("Be brief".into()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Hello".into()),
            timestamp: 42,
        })],
        tools: tools.to_vec(),
    };
    let value = serde_json::to_value(&context).unwrap();
    assert_eq!(value["systemPrompt"], "Be brief");
    assert_eq!(
        value["tools"][0]["constrainedSampling"]["type"],
        "json_schema"
    );
    assert_eq!(
        value["tools"][0]["constrainedSampling"]["strict"],
        "require"
    );
    assert_eq!(
        value["tools"][1]["constrainedSampling"]["variants"]["openai_lark"],
        "start: WORD"
    );
    assert_eq!(value["tools"][2]["constrainedSampling"], false);

    let decoded: Context = serde_json::from_value(value).unwrap();
    assert_eq!(decoded, context);
    assert_eq!(
        tools[0].constrained_sampling,
        Some(ConstrainedSampling::JsonSchema {
            strict: ConstrainedSamplingStrictness::Require
        })
    );
}
