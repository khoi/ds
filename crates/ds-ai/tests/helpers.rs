use async_trait::async_trait;
use ds_ai::{
    Api, AssistantContent, AssistantMessage, AssistantToolCall, Context, InputContent, Message,
    Model, ModelCost, ModelInput, ProviderId, RetryCallbacks, RetryPolicy, StopReason, TextContent,
    ThinkingContent, Tool, ToolResultMessage, Usage, calculate_context_tokens,
    clamp_max_tokens_to_context, content_text, content_text_with_separator,
    estimate_context_tokens, estimate_message_tokens, is_context_overflow, is_recoverable_length,
    is_retryable_assistant_error, retry_assistant_call,
};
use serde_json::json;
use std::sync::Mutex;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[test]
fn extracts_text_from_supported_content() {
    let assistant = vec![
        AssistantContent::Thinking(ThinkingContent {
            thinking: "reasoning".into(),
            thinking_signature: None,
            redacted: None,
        }),
        AssistantContent::Text(TextContent {
            text: "first".into(),
            text_signature: None,
        }),
        AssistantContent::ToolCall(AssistantToolCall {
            id: "1".into(),
            name: "read".into(),
            arguments: json!({}),
            thought_signature: None,
            namespace: None,
        }),
        AssistantContent::Text(TextContent {
            text: "second".into(),
            text_signature: None,
        }),
    ];
    let input = vec![
        InputContent::text("first"),
        InputContent::image("image/png", "..."),
        InputContent::text("second"),
    ];
    assert_eq!(content_text(&assistant), "first\nsecond");
    assert_eq!(content_text_with_separator(&assistant, ""), "firstsecond");
    assert_eq!(content_text("hello"), "hello");
    assert_eq!(content_text_with_separator(&input, ""), "firstsecond");
}

#[test]
fn estimates_context_from_the_latest_applicable_usage() {
    let stale = Context::new([
        Message::User(ds_ai::UserMessage::new("summary", 200)),
        Message::assistant(assistant(100, StopReason::Stop, None, 9_500)),
        Message::User(ds_ai::UserMessage::new("x".repeat(4_000), 300)),
    ])
    .with_system("system");
    assert_eq!(
        estimate_context_tokens(&stale),
        ds_ai::ContextUsageEstimate {
            tokens: 1_005,
            usage_tokens: 0,
            trailing_tokens: 1_005,
            last_usage_index: None,
        }
    );
    assert_eq!(clamp_max_tokens_to_context(&model(), &stale, 8_000), 4_899);

    let current = Context::new([
        Message::User(ds_ai::UserMessage::new("summary", 200)),
        Message::assistant(assistant(100, StopReason::Stop, None, 9_500)),
        Message::User(ds_ai::UserMessage::new("new prompt", 300)),
        Message::assistant(assistant(400, StopReason::Stop, None, 2_000)),
        Message::User(ds_ai::UserMessage::new("tail", 500)),
    ]);
    assert_eq!(
        estimate_context_tokens(&current),
        ds_ai::ContextUsageEstimate {
            tokens: 2_001,
            usage_tokens: 2_000,
            trailing_tokens: 1,
            last_usage_index: Some(3),
        }
    );
}

#[test]
fn estimates_message_content_and_deferred_tool_definitions() {
    let tool = Tool::new("late_tool", "x".repeat(4_000), json!({"type": "object"}));
    let checkpoint = Message::assistant(assistant(2, StopReason::Stop, None, 100));
    let marker = Message::tool_result(ToolResultMessage {
        added_tool_names: Some(vec!["late_tool".into()]),
        ..ToolResultMessage::new("call", "base_tool", [InputContent::text("done")])
    });
    let plain = estimate_context_tokens(&Context::new([checkpoint.clone(), marker.clone()]));
    let marked = estimate_context_tokens(&Context::new([checkpoint, marker]).with_tools([tool]));
    assert!(marked.tokens > plain.tokens + 500);
    assert!(marked.trailing_tokens > plain.trailing_tokens + 500);

    let image = Message::User(ds_ai::UserMessage::with_blocks(
        [InputContent::image("image/png", "...")],
        1,
    ));
    assert_eq!(estimate_message_tokens(&image), 1_200);
    assert_eq!(calculate_context_tokens(&usage(10)), 10);
}

#[test]
fn detects_context_overflow_and_recoverable_length() {
    for message in [
        "400 `prompt too long; exceeded max context length by 100918 tokens`",
        "400 The input (516368 tokens) is longer than the model's context length (262144 tokens).",
        "Requested token count exceeds the model's maximum context length of 131072 tokens.",
        "Input length (265330) exceeds model's maximum context length (262144).",
        "Input length 131393 exceeds the maximum allowed input length of 131040 tokens.",
        "Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens",
    ] {
        assert!(is_context_overflow(
            &assistant(1, StopReason::Error, Some(message), 0),
            Some(262_144)
        ));
    }
    for message in [
        "500 model runner crashed unexpectedly",
        "Throttling error: Too many tokens, please wait before trying again.",
        "Service unavailable: The service is temporarily unavailable.",
        "Rate limit exceeded, please retry after 30 seconds.",
        "Too many requests. Please slow down.",
    ] {
        assert!(!is_context_overflow(
            &assistant(1, StopReason::Error, Some(message), 0),
            Some(200_000)
        ));
    }

    let mut filled = assistant(1, StopReason::Length, None, 0);
    filled.usage.input = 58;
    filled.usage.cache_read = 1_048_512;
    assert!(is_context_overflow(&filled, Some(1_048_576)));
    assert!(is_recoverable_length(&filled, 128_000));

    let mut reached = assistant(1, StopReason::Length, None, 0);
    reached.usage.output = 1_024;
    assert!(!is_recoverable_length(&reached, 1_024));
}

#[test]
fn classifies_retryable_assistant_errors() {
    for message in [
        "You can retry your request",
        "Try your request again",
        "ResourceExhausted: Worker request limit reached",
        "The socket connection was closed unexpectedly",
        "exceeded request buffer limit while retrying upstream",
        "getaddrinfo ENOTFOUND example.invalid",
        "EAI_AGAIN example.invalid",
        "stream ended before a terminal response event",
        "overloaded_error",
        "524 status code (no body)",
    ] {
        assert!(is_retryable_assistant_error(&assistant(
            1,
            StopReason::Error,
            Some(message),
            0
        )));
    }
    for message in ["429 quota exceeded", "insufficient_quota", "billing"] {
        assert!(!is_retryable_assistant_error(&assistant(
            1,
            StopReason::Error,
            Some(message),
            0
        )));
    }
    assert!(!is_retryable_assistant_error(&assistant(
        1,
        StopReason::Stop,
        None,
        0
    )));
}

#[tokio::test(start_paused = true)]
async fn retries_transient_assistant_calls_and_emits_callbacks() {
    let calls = Mutex::new(0);
    let callbacks = RecordingCallbacks::default();
    let response = retry_assistant_call(
        || async {
            let mut calls = calls.lock().unwrap();
            *calls += 1;
            if *calls < 3 {
                assistant(1, StopReason::Error, Some("terminated"), 0)
            } else {
                assistant(1, StopReason::Stop, None, 0)
            }
        },
        Some(&RetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay: Duration::from_secs(1),
        }),
        None,
        Some(&callbacks),
    )
    .await;
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(*calls.lock().unwrap(), 3);
    assert_eq!(
        callbacks.events.lock().unwrap().as_slice(),
        [
            "scheduled:1:3:1000:terminated",
            "start",
            "scheduled:2:3:2000:terminated",
            "start",
            "finished:true:2:none",
        ]
    );
}

#[tokio::test]
async fn skips_retry_for_success_abort_disabled_and_terminal_errors() {
    let enabled = RetryPolicy {
        enabled: true,
        max_retries: 3,
        base_delay: Duration::ZERO,
    };
    let disabled = RetryPolicy {
        enabled: false,
        ..enabled
    };
    for (policy, reason, error) in [
        (&enabled, StopReason::Stop, None),
        (&enabled, StopReason::Aborted, None),
        (&enabled, StopReason::Error, Some("insufficient_quota")),
        (&disabled, StopReason::Error, Some("terminated")),
    ] {
        let calls = Mutex::new(0);
        let callbacks = RecordingCallbacks::default();
        let response = retry_assistant_call(
            || async {
                *calls.lock().unwrap() += 1;
                assistant(1, reason, error, 0)
            },
            Some(policy),
            None,
            Some(&callbacks),
        )
        .await;
        assert_eq!(response.stop_reason, reason);
        assert_eq!(*calls.lock().unwrap(), 1);
        assert!(callbacks.events.lock().unwrap().is_empty());
    }
}

#[tokio::test(start_paused = true)]
async fn exhausts_retries_and_aborts_a_retry_wait() {
    let callbacks = RecordingCallbacks::default();
    let exhausted = retry_assistant_call(
        || async { assistant(1, StopReason::Error, Some("terminated"), 0) },
        Some(&RetryPolicy {
            enabled: true,
            max_retries: 2,
            base_delay: Duration::ZERO,
        }),
        None,
        Some(&callbacks),
    )
    .await;
    assert_eq!(exhausted.stop_reason, StopReason::Error);
    assert!(
        callbacks
            .events
            .lock()
            .unwrap()
            .contains(&"finished:false:2:terminated".into())
    );

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let aborted = retry_assistant_call(
        || async { assistant(1, StopReason::Error, Some("terminated"), 0) },
        Some(&RetryPolicy {
            enabled: true,
            max_retries: 1,
            base_delay: Duration::from_secs(10),
        }),
        Some(&cancellation),
        None,
    )
    .await;
    assert_eq!(aborted.stop_reason, StopReason::Aborted);
    assert_eq!(aborted.error_message, None);
}

#[derive(Default)]
struct RecordingCallbacks {
    events: Mutex<Vec<String>>,
}

#[async_trait]
impl RetryCallbacks for RecordingCallbacks {
    async fn on_retry_scheduled(
        &self,
        attempt: usize,
        max_attempts: usize,
        delay: Duration,
        error_message: &str,
    ) {
        self.events.lock().unwrap().push(format!(
            "scheduled:{attempt}:{max_attempts}:{}:{error_message}",
            delay.as_millis()
        ));
    }

    async fn on_retry_attempt_start(&self) {
        self.events.lock().unwrap().push("start".into());
    }

    async fn on_retry_finished(&self, success: bool, attempt: usize, final_error: Option<&str>) {
        self.events.lock().unwrap().push(format!(
            "finished:{success}:{attempt}:{}",
            final_error.unwrap_or("none")
        ));
    }
}

fn model() -> Model {
    Model {
        id: "test-model".into(),
        name: "Test Model".into(),
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        base_url: "https://api.openai.com/v1".into(),
        reasoning: false,
        thinking_level_map: Default::default(),
        input: vec![ModelInput::Text],
        cost: ModelCost::default(),
        context_window: 10_000,
        max_tokens: 8_000,
        sampling_params: Default::default(),
        headers: Default::default(),
        compat: None,
    }
}

fn usage(tokens: u64) -> Usage {
    Usage {
        input: tokens,
        total_tokens: tokens,
        ..Default::default()
    }
}

fn assistant(
    timestamp: u64,
    stop_reason: StopReason,
    error_message: Option<&str>,
    tokens: u64,
) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::Text(TextContent {
            text: "kept".into(),
            text_signature: None,
        })],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: "test-model".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: usage(tokens),
        stop_reason,
        error_message: error_message.map(str::to_owned),
        raw_stop_reason: None,
        end_turn: None,
        timestamp,
    }
}
