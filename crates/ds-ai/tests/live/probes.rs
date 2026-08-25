#[tokio::test]
#[ignore = "requires selected-provider credentials"]
async fn live_abort_probe_matrix() {
    for target in [OPENAI_MINI, ANTHROPIC_OAUTH_SONNET, CODEX_55] {
        let immediate = CancellationToken::new();
        immediate.cancel();
        let message = live_complete(
            target,
            &Context::new([Message::user("Hello")]),
            LiveCall {
                cancellation: immediate,
                ..Default::default()
            },
        )
        .await;
        assert_eq!(message.stop_reason, StopReason::Aborted);

        let cancellation = CancellationToken::new();
        let mut stream = live_stream(
            target,
            &Context::new([Message::user(
                "List one hundred first names, one per line, with no preface.",
            )]),
            LiveCall {
                cancellation: cancellation.clone(),
                reasoning: LiveReasoning::High,
                ..Default::default()
            },
        );
        let mut output = String::new();
        while let Some(event) = stream.next().await {
            match event {
                AssistantMessageEvent::TextDelta { delta, .. }
                | AssistantMessageEvent::ThinkingDelta { delta, .. } => output.push_str(&delta),
                _ => {}
            }
            if output.len() >= 50 {
                cancellation.cancel();
            }
        }
        let message = stream.result().await.unwrap();
        assert!(cancellation.is_cancelled());
        assert_eq!(message.stop_reason, StopReason::Aborted);
        assert!(!message.content.is_empty());
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and ANTHROPIC_OAUTH_TOKEN"]
async fn live_anthropic_feature_probe_matrix() {
    let mut eager_model = ANTHROPIC_HAIKU.model();
    if let Some(ModelCompatibility::Anthropic(compat)) = &mut eager_model.compat {
        compat.supports_eager_tool_input_streaming = Some(true);
    }
    for model in [ANTHROPIC_HAIKU.model(), eager_model] {
        let context = Context::new([Message::user(
            "Call echo_value with value set to eager-input-streaming-compat.",
        )])
        .with_tools([tool("echo_value")]);
        let response = live_stream_model(
            ANTHROPIC_HAIKU,
            model,
            &context,
            LiveCall {
                max_tokens: Some(128),
                force_tool: Some("echo_value".into()),
                ..Default::default()
            },
        )
        .result()
        .await
        .unwrap();
        assert_ne!(
            response.stop_reason,
            StopReason::Error,
            "{:?}",
            response.error_message
        );
    }

    let mut cache_model = ANTHROPIC_HAIKU.model();
    if let Some(ModelCompatibility::Anthropic(compat)) = &mut cache_model.compat {
        compat.supports_long_cache_retention = Some(true);
    }
    let response = live_stream_model(
        ANTHROPIC_HAIKU,
        cache_model,
        &Context::new([Message::user(
            "Reply with exactly: long cache retention accepted",
        )]),
        LiveCall {
            max_tokens: Some(128),
            cache_retention: CacheRetention::Long,
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();
    assert_success(&response);

    let response = live_complete(
        ANTHROPIC_OPUS_48,
        &Context::new([Message::user(
            "Compute 48291 * 7317 and 90844 - 17729, add the results, and state if the sum is divisible by 11.",
        )]),
        LiveCall {
            reasoning: LiveReasoning::High,
            ..Default::default()
        },
    )
    .await;
    assert_success(&response);
    assert!(response.content.iter().any(|content| matches!(
        content,
        AssistantContent::Thinking(thinking)
            if thinking.thinking_signature.as_deref().is_some_and(|value| !value.is_empty())
    )));

    for name in ["todowrite", "read", "find", "my_custom_tool"] {
        let (_, response, call) =
            live_tool_call(ANTHROPIC_OAUTH_SONNET, name, LiveReasoning::None).await;
        assert_eq!(call.name, name);
        assert_ne!(response.stop_reason, StopReason::Error);
    }
}

#[tokio::test]
#[ignore = "requires selected-provider credentials and sends oversized prompts"]
async fn live_context_overflow_probe_matrix() {
    for target in [ANTHROPIC_HAIKU, ANTHROPIC_OAUTH_SONNET, OPENAI_4O, CODEX_55] {
        let model = target.model();
        let paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ";
        let chars = model
            .context_window
            .saturating_add(10_000)
            .saturating_mul(6);
        let repetitions = chars
            .div_ceil(u64::try_from(paragraph.len()).unwrap())
            .try_into()
            .unwrap();
        let message = live_complete(
            target,
            &Context::new([Message::user(paragraph.repeat(repetitions))]),
            LiveCall {
                max_tokens: Some(128),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(message.stop_reason, StopReason::Error);
        assert!(is_context_overflow(&message, Some(model.context_window)));
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY, ANTHROPIC_API_KEY, and DS_AI_CODEX_ACCESS_TOKEN"]
async fn live_cross_provider_handoff_probe_matrix() {
    let targets = [OPENAI_54, ANTHROPIC_HAIKU, CODEX_55];
    let mut fixtures = Vec::new();
    for target in targets {
        let user = Message::user("Reply with exactly: fixture ready");
        let response =
            live_complete(target, &Context::new([user.clone()]), LiveCall::default()).await;
        assert_success(&response);
        fixtures.push((target, user, response));
    }
    assert!(fixtures.len() >= 2);
    for (target, _, _) in &fixtures {
        let messages = fixtures
            .iter()
            .filter(|(source, _, _)| source.provider != target.provider)
            .flat_map(|(_, user, response)| [user.clone(), Message::assistant(response.clone())])
            .chain([Message::user("Reply with exactly: handoff successful")]);
        let response = live_complete(*target, &Context::new(messages), LiveCall::default()).await;
        assert_success(&response);
        assert!(
            message_text(&response)
                .to_lowercase()
                .contains("handoff successful")
        );
    }
}

#[tokio::test]
#[ignore = "requires selected-provider credentials"]
async fn live_empty_message_probe_matrix() {
    for target in [
        OPENAI_MINI,
        ANTHROPIC_HAIKU,
        ANTHROPIC_OAUTH_SONNET,
        CODEX_55,
    ] {
        let model = target.model();
        let contexts = [
            Context::new([Message::user_content(Vec::<InputContent>::new())]),
            Context::new([Message::user("")]),
            Context::new([Message::user("   \n\t  ")]),
            Context::new([
                Message::user("Hello"),
                Message::assistant(empty_assistant(&model)),
                Message::user("Please respond this time."),
            ]),
        ];
        for context in contexts {
            let response = live_complete(target, &context, LiveCall::default()).await;
            if response.stop_reason == StopReason::Error {
                assert!(
                    response
                        .error_message
                        .as_deref()
                        .is_some_and(|error| !error.is_empty())
                );
            } else {
                assert!(!response.content.is_empty());
            }
        }
    }
}

const RED_PIXEL: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2ZQAAAABJRU5ErkJggg==";

async fn image_tool_result(
    target: LiveTarget,
    with_text: bool,
    capture: bool,
) -> Option<serde_json::Value> {
    let (mut context, response, call) =
        live_tool_call(target, "get_image", LiveReasoning::None).await;
    context.messages.push(Message::assistant(response));
    let mut content = Vec::new();
    if with_text {
        content.push(InputContent::text("The image is a one-pixel sample."));
    }
    content.push(InputContent::image("image/png", RED_PIXEL));
    context.messages.push(tool_result(&call, content));
    let captured = capture.then(|| Arc::new(Mutex::new(None)));
    let response = live_complete(
        target,
        &context,
        LiveCall {
            reasoning: if matches!(target.provider, LiveProvider::OpenAi | LiveProvider::Codex) {
                LiveReasoning::Medium
            } else {
                LiveReasoning::None
            },
            capture: captured.clone(),
            ..Default::default()
        },
    )
    .await;
    assert_success(&response);
    captured.and_then(|captured| captured.lock().unwrap().clone())
}

#[tokio::test]
#[ignore = "requires selected-provider credentials"]
async fn live_image_tool_result_probe_matrix() {
    for target in [
        OPENAI_MINI,
        ANTHROPIC_HAIKU,
        ANTHROPIC_OAUTH_SONNET,
        CODEX_55,
    ] {
        image_tool_result(target, false, false).await;
        image_tool_result(target, true, false).await;
    }
    for target in [OPENAI_MINI, CODEX_55] {
        let payload = image_tool_result(target, false, true).await.unwrap();
        let output = payload["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .unwrap();
        let encoded = serde_json::to_string(output).unwrap();
        assert!(encoded.contains("input_image"));
        assert!(encoded.contains("data:image/png;base64,"));
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn live_interleaved_thinking_probe_matrix() {
    for target in [ANTHROPIC_OPUS_45, ANTHROPIC_OPUS_46] {
        let (mut context, first, call) =
            live_tool_call(target, "calculate", LiveReasoning::High).await;
        assert!(
            first
                .content
                .iter()
                .any(|content| matches!(content, AssistantContent::Thinking(_)))
        );
        context.messages.push(Message::assistant(first));
        context
            .messages
            .push(tool_result(&call, [InputContent::text("42")]));
        let second = live_complete(
            target,
            &context,
            LiveCall {
                reasoning: LiveReasoning::High,
                ..Default::default()
            },
        )
        .await;
        assert_success(&second);
        assert!(
            second
                .content
                .iter()
                .any(|content| matches!(content, AssistantContent::Thinking(_)))
        );
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and DS_AI_CODEX_ACCESS_TOKEN"]
async fn live_cache_affinity_probe_matrix() {
    for target in [OPENAI_54, CODEX_55.with_transport(Transport::Sse)] {
        let expected = if target.provider == LiveProvider::OpenAi {
            "openai cache affinity e2e success"
        } else {
            "cache affinity e2e success"
        };
        let response = live_complete(
            target,
            &Context::new([Message::user(format!("Reply with exactly: {expected}"))]),
            LiveCall {
                session_id: Some("0195d6e4-4cf9-7f44-a2d8-f8f7f49ee9d3".into()),
                ..Default::default()
            },
        )
        .await;
        assert_success(&response);
        assert!(message_text(&response).to_lowercase().contains(expected));
    }
}

async fn continue_tool_result(
    target: LiveTarget,
    mut context: Context,
    response: AssistantMessage,
    call: AssistantToolCall,
    reasoning: LiveReasoning,
) -> AssistantMessage {
    context.messages.push(Message::assistant(response));
    context
        .messages
        .push(tool_result(&call, [InputContent::text("42")]));
    context.messages.push(Message::user(
        "What was the result? Reply with just the number.",
    ));
    live_complete(
        target,
        &context,
        LiveCall {
            reasoning,
            ..Default::default()
        },
    )
    .await
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and ANTHROPIC_API_KEY"]
async fn live_reasoning_replay_probe_matrix() {
    let (context, response, _) =
        live_tool_call(OPENAI_MINI, "double_number", LiveReasoning::High).await;
    let mut aborted = response.clone();
    aborted
        .content
        .retain(|content| matches!(content, AssistantContent::Thinking(_)));
    assert!(!aborted.content.is_empty());
    aborted.stop_reason = StopReason::Aborted;
    let response = live_complete(
        OPENAI_MINI,
        &Context::new(
            context
                .messages
                .into_iter()
                .chain([Message::assistant(aborted), Message::user("Say hello")]),
        )
        .with_tools([tool("double_number")]),
        LiveCall {
            reasoning: LiveReasoning::High,
            ..Default::default()
        },
    )
    .await;
    assert_success(&response);

    let (context, response, call) =
        live_tool_call(OPENAI_MINI, "double_number", LiveReasoning::High).await;
    let response =
        continue_tool_result(OPENAI_55, context, response, call, LiveReasoning::High).await;
    assert_success(&response);
    assert!(message_text(&response).contains("42"));

    let (context, response, call) =
        live_tool_call(ANTHROPIC_SONNET, "double_number", LiveReasoning::High).await;
    let response =
        continue_tool_result(OPENAI_55, context, response, call, LiveReasoning::High).await;
    assert_success(&response);
    assert!(message_text(&response).contains("42"));
}

#[tokio::test]
#[ignore = "requires selected-provider credentials"]
async fn live_response_id_probe_matrix() {
    for target in [OPENAI_MINI, ANTHROPIC_SONNET, CODEX_55] {
        let response = live_complete(
            target,
            &Context::new([Message::user("Reply with exactly: response id test")]),
            LiveCall::default(),
        )
        .await;
        assert_success(&response);
        assert!(
            response
                .response_id
                .as_deref()
                .is_some_and(|id| !id.is_empty())
        );
    }
}

#[derive(Clone, Copy)]
enum StreamScenario {
    Basic(LiveReasoning),
    Tool,
    Streaming,
    Thinking(LiveReasoning),
    MultiTurn(LiveReasoning),
    Image,
}

async fn run_stream_scenario(target: LiveTarget, scenario: StreamScenario) {
    match scenario {
        StreamScenario::Basic(reasoning) => {
            let first = live_complete(
                target,
                &Context::new([Message::user("Reply with exactly: Hello test successful")]),
                LiveCall {
                    reasoning,
                    ..Default::default()
                },
            )
            .await;
            assert_success(&first);
            assert!(message_text(&first).contains("Hello test successful"));
            let second = live_complete(
                target,
                &Context::new([
                    Message::user("Reply with exactly: Hello test successful"),
                    Message::assistant(first),
                    Message::user("Reply with exactly: Goodbye test successful"),
                ]),
                LiveCall {
                    reasoning,
                    ..Default::default()
                },
            )
            .await;
            assert_success(&second);
            assert!(message_text(&second).contains("Goodbye test successful"));
        }
        StreamScenario::Tool => {
            let context = Context::new([Message::user(
                "Call math_operation with value set to live-probe",
            )])
            .with_tools([tool("math_operation")]);
            let mut stream = live_stream(
                target,
                &context,
                LiveCall {
                    force_tool: Some("math_operation".into()),
                    ..Default::default()
                },
            );
            let mut started = false;
            let mut delta = String::new();
            let mut ended = false;
            while let Some(event) = stream.next().await {
                match event {
                    AssistantMessageEvent::ToolCallStart { .. } => started = true,
                    AssistantMessageEvent::ToolCallDelta {
                        delta: value,
                        partial,
                        content_index,
                    } => {
                        delta.push_str(&value);
                        assert!(matches!(
                            partial.content.get(content_index),
                            Some(AssistantContent::ToolCall(call)) if call.arguments.is_object()
                        ));
                    }
                    AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                        ended = true;
                        assert_eq!(tool_call.name, "math_operation");
                        assert!(tool_call.arguments.is_object());
                    }
                    _ => {}
                }
            }
            let response = stream.result().await.unwrap();
            assert_eq!(response.stop_reason, StopReason::ToolUse);
            assert!(started && !delta.is_empty() && ended);
        }
        StreamScenario::Streaming => {
            let mut stream = live_stream(
                target,
                &Context::new([Message::user("Count from 1 to 3")]),
                LiveCall::default(),
            );
            let mut started = false;
            let mut delta = String::new();
            let mut ended = false;
            while let Some(event) = stream.next().await {
                match event {
                    AssistantMessageEvent::TextStart { .. } => started = true,
                    AssistantMessageEvent::TextDelta { delta: value, .. } => delta.push_str(&value),
                    AssistantMessageEvent::TextEnd { .. } => ended = true,
                    _ => {}
                }
            }
            let response = stream.result().await.unwrap();
            assert_success(&response);
            assert!(started && !delta.is_empty() && ended);
        }
        StreamScenario::Thinking(reasoning) => {
            let mut stream = live_stream(
                target,
                &Context::new([Message::user(
                    "Think step by step about 48 + 27, then state the answer.",
                )]),
                LiveCall {
                    reasoning,
                    ..Default::default()
                },
            );
            let mut saw_thinking = false;
            while let Some(event) = stream.next().await {
                saw_thinking |= matches!(
                    event,
                    AssistantMessageEvent::ThinkingStart { .. }
                        | AssistantMessageEvent::ThinkingDelta { .. }
                        | AssistantMessageEvent::ThinkingEnd { .. }
                );
            }
            let response = stream.result().await.unwrap();
            assert_success(&response);
            assert!(saw_thinking);
            assert!(
                response
                    .content
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Thinking(_)))
            );
        }
        StreamScenario::MultiTurn(reasoning) => {
            let (context, response, call) =
                live_tool_call(target, "math_operation", reasoning).await;
            let response = continue_tool_result(target, context, response, call, reasoning).await;
            assert_success(&response);
        }
        StreamScenario::Image => {
            assert!(target.model().input.contains(&ModelInput::Image));
            let response = live_complete(
                target,
                &Context::new([Message::user_content([
                    InputContent::text("Describe this image in one sentence."),
                    InputContent::image("image/png", RED_PIXEL),
                ])]),
                LiveCall::default(),
            )
            .await;
            assert_success(&response);
            assert!(!message_text(&response).is_empty());
        }
    }
}

#[tokio::test]
#[ignore = "requires selected-provider credentials and runs the full stream matrix"]
async fn live_stream_probe_matrix() {
    let basic = [
        StreamScenario::Basic(LiveReasoning::None),
        StreamScenario::Tool,
        StreamScenario::Streaming,
    ];
    for target in [
        OPENAI_54,
        ANTHROPIC_OAUTH_SONNET,
        ANTHROPIC_OAUTH_OPUS_46,
        CODEX_54,
        CODEX_55,
        CODEX_55_WS,
    ] {
        for scenario in basic {
            run_stream_scenario(target, scenario).await;
        }
    }
    for scenario in [
        StreamScenario::Basic(LiveReasoning::High),
        StreamScenario::Tool,
        StreamScenario::Streaming,
        StreamScenario::Image,
    ] {
        run_stream_scenario(ANTHROPIC_HAIKU, scenario).await;
    }
    for target in [OPENAI_54, ANTHROPIC_OAUTH_SONNET, CODEX_54] {
        for scenario in [
            StreamScenario::Thinking(LiveReasoning::High),
            StreamScenario::MultiTurn(LiveReasoning::High),
            StreamScenario::Image,
        ] {
            run_stream_scenario(target, scenario).await;
        }
    }
    for target in [CODEX_55, CODEX_55_WS] {
        for scenario in [
            StreamScenario::Thinking(LiveReasoning::XHigh),
            StreamScenario::MultiTurn(LiveReasoning::XHigh),
            StreamScenario::Image,
        ] {
            run_stream_scenario(target, scenario).await;
        }
    }
    for scenario in [
        StreamScenario::Thinking(LiveReasoning::High),
        StreamScenario::Thinking(LiveReasoning::Medium),
        StreamScenario::MultiTurn(LiveReasoning::High),
        StreamScenario::Image,
    ] {
        run_stream_scenario(ANTHROPIC_OAUTH_OPUS_46, scenario).await;
    }
}

async fn abort_with_usage(target: LiveTarget) -> AssistantMessage {
    let cancellation = CancellationToken::new();
    let mut stream = live_stream(
        target,
        &Context::new([Message::user(
            "Write a long poem with twenty stanzas about nature.",
        )]),
        LiveCall {
            cancellation: cancellation.clone(),
            reasoning: LiveReasoning::High,
            ..Default::default()
        },
    );
    let mut output = String::new();
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::TextDelta { delta, .. }
            | AssistantMessageEvent::ThinkingDelta { delta, .. } => output.push_str(&delta),
            _ => {}
        }
        if output.len() >= 1000 {
            cancellation.cancel();
        }
    }
    assert!(cancellation.is_cancelled());
    stream.result().await.unwrap()
}

#[tokio::test]
#[ignore = "requires selected-provider credentials"]
async fn live_aborted_usage_probe_matrix() {
    for target in [
        OPENAI_54,
        ANTHROPIC_SONNET,
        ANTHROPIC_OAUTH_SONNET,
        CODEX_55,
    ] {
        let response = abort_with_usage(target).await;
        assert_eq!(response.stop_reason, StopReason::Aborted);
        match target.provider {
            LiveProvider::AnthropicApiKey | LiveProvider::AnthropicOAuth => {
                assert!(response.usage.input > 0);
                assert!(response.usage.output > 0);
            }
            LiveProvider::OpenAi | LiveProvider::Codex => {
                assert_eq!(response.usage.input, 0);
                assert_eq!(response.usage.output, 0);
            }
        }
    }
}

fn foreign_tool_message(id: String) -> (Message, Message) {
    let call = AssistantToolCall {
        id: id.clone(),
        name: "echo".into(),
        arguments: serde_json::json!({"value": "hello"}),
        thought_signature: None,
        namespace: None,
    };
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(call.clone())],
        api: Api::Other("foreign".into()),
        provider: ProviderId::new("foreign"),
        model: "foreign".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: timestamp(),
    };
    (
        Message::assistant(assistant),
        tool_result(&call, [InputContent::text("hello")]),
    )
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and DS_AI_CODEX_ACCESS_TOKEN"]
async fn live_tool_call_id_probe_matrix() {
    let (context, mut response, mut call) =
        live_tool_call(OPENAI_54, "echo", LiveReasoning::High).await;
    let id = format!("{}|{}", call.id, "A+/=".repeat(120));
    for content in &mut response.content {
        if let AssistantContent::ToolCall(response_call) = content {
            response_call.id.clone_from(&id);
        }
    }
    call.id = id;
    let response =
        continue_tool_result(CODEX_55, context, response, call, LiveReasoning::High).await;
    assert_success(&response);

    let id = format!("call_live|{}", "B+/=".repeat(120));
    let (assistant, result) = foreign_tool_message(id);
    let response = live_complete(
        CODEX_55,
        &Context::new([
            Message::user("Use the supplied echo result."),
            assistant,
            result,
            Message::user("Say hi"),
        ])
        .with_tools([tool("echo")]),
        LiveCall::default(),
    )
    .await;
    assert_success(&response);
}

#[tokio::test]
#[ignore = "requires selected-provider credentials"]
async fn live_missing_tool_result_probe_matrix() {
    for target in [
        OPENAI_MINI,
        ANTHROPIC_HAIKU,
        ANTHROPIC_OAUTH_SONNET,
        CODEX_55,
    ] {
        let (mut context, response, _) =
            live_tool_call(target, "calculate", LiveReasoning::None).await;
        context.messages.push(Message::assistant(response));
        context
            .messages
            .push(Message::user("Never mind. What is 2 + 2?"));
        let response = live_complete(target, &context, LiveCall::default()).await;
        assert_success(&response);
        assert!(matches!(
            response.stop_reason,
            StopReason::Stop | StopReason::ToolUse
        ));
    }
}

fn assert_total_tokens(usage: &Usage) {
    assert_eq!(
        usage.total_tokens,
        usage.input + usage.output + usage.cache_read + usage.cache_write
    );
}

#[tokio::test]
#[ignore = "requires selected-provider credentials"]
async fn live_total_tokens_probe_matrix() {
    let system = "Keep this stable cache context. ".repeat(1000);
    for target in [
        ANTHROPIC_SONNET,
        ANTHROPIC_OAUTH_SONNET,
        OPENAI_4O,
        CODEX_55,
    ] {
        let first_context = Context::new([Message::user("What is 2 + 2?")]).with_system(&system);
        let first = live_complete(target, &first_context, LiveCall::default()).await;
        assert_success(&first);
        let second = live_complete(
            target,
            &Context::new([
                Message::user("What is 2 + 2?"),
                Message::assistant(first.clone()),
                Message::user("What is 3 + 3?"),
            ])
            .with_system(&system),
            LiveCall::default(),
        )
        .await;
        assert_success(&second);
        assert_total_tokens(&first.usage);
        assert_total_tokens(&second.usage);
        if matches!(
            target.provider,
            LiveProvider::AnthropicApiKey | LiveProvider::AnthropicOAuth
        ) {
            assert!(
                first.usage.cache_write > 0
                    || second.usage.cache_read > 0
                    || second.usage.cache_write > 0
            );
        }
    }
}

async fn unicode_tool_result(target: LiveTarget, value: &str) {
    let (mut context, response, call) = live_tool_call(target, "echo", LiveReasoning::None).await;
    context.messages.push(Message::assistant(response));
    context
        .messages
        .push(tool_result(&call, [InputContent::text(value)]));
    let response = live_complete(target, &context, LiveCall::default()).await;
    assert_success(&response);
}

#[tokio::test]
#[ignore = "requires selected-provider credentials"]
async fn live_unicode_probe_matrix() {
    let values = [
        "Hello 😀 🚀 ❤️ 👍🏽 world",
        "Great work! 🎉 The release is ready ✅ Thanks to everyone 👏🏽",
    ];
    for target in [
        OPENAI_MINI,
        ANTHROPIC_HAIKU,
        ANTHROPIC_OAUTH_SONNET,
        CODEX_55,
    ] {
        for value in values {
            unicode_tool_result(target, value).await;
        }
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn live_xhigh_probe_matrix() {
    let context = Context::new([Message::user("What is 48 + 27? Think step by step.")]);
    let response = live_complete(
        OPENAI_55,
        &context,
        LiveCall {
            reasoning: LiveReasoning::XHigh,
            ..Default::default()
        },
    )
    .await;
    assert_success(&response);
    assert!(
        response
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Thinking(_)))
    );

    let valid = live_complete(
        OPENAI_MINI,
        &context,
        LiveCall {
            reasoning: LiveReasoning::High,
            ..Default::default()
        },
    )
    .await;
    assert_success(&valid);
    let response = live_complete(
        OPENAI_MINI,
        &context,
        LiveCall {
            reasoning: LiveReasoning::XHigh,
            ..Default::default()
        },
    )
    .await;
    assert_eq!(response.stop_reason, StopReason::Error);
    assert!(
        response
            .error_message
            .as_deref()
            .is_some_and(|error| error.contains("xhigh"))
    );
}

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("missing {name}"))
}
