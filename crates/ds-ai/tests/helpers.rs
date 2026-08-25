use crate::support::{Reply, serve};
use async_trait::async_trait;
use ds_ai::{
    Api, AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantToolCall, Context,
    InputContent, Message, Model, ModelCost, ModelInput, OpenAiResponsesOptions, ProviderId,
    RetryCallbacks, RetryPolicy, StopReason, StreamOptions, TextContent, ThinkingContent, Tool,
    ToolResultMessage, Usage, calculate_context_tokens, clamp_max_tokens_to_context, content_text,
    content_text_with_separator, estimate_context_tokens, estimate_message_tokens,
    is_context_overflow, is_recoverable_length, is_retryable_assistant_error, openai,
    retry_assistant_call,
};
use futures_util::StreamExt;
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::Notify,
    task::JoinHandle,
};
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
        "HTTP 429: request throttled",
    ] {
        assert!(!is_context_overflow(
            &assistant(1, StopReason::Error, Some(message), 0),
            Some(200_000)
        ));
    }
    assert!(is_context_overflow(
        &assistant(
            1,
            StopReason::Error,
            Some("provider returned HTTP 429: Too many tokens, please slow down."),
            0
        ),
        Some(200_000)
    ));

    let mut filled = assistant(1, StopReason::Length, None, 0);
    filled.usage.input = 58;
    filled.usage.cache_read = 1_048_512;
    assert!(is_context_overflow(&filled, Some(1_048_576)));
    assert!(is_recoverable_length(&filled, 128_000));

    let zero_output = assistant(1, StopReason::Length, None, 100);
    assert!(is_recoverable_length(&zero_output, 128_000));

    let mut normal_length = assistant(1, StopReason::Length, None, 1_000);
    normal_length.usage.output = 4_096;
    assert!(!is_context_overflow(&normal_length, Some(200_000)));

    let far_below_context = assistant(1, StopReason::Length, None, 100);
    assert!(!is_context_overflow(&far_below_context, Some(200_000)));

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
        "provider stream ended before a terminal event",
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

#[tokio::test]
async fn preserves_structured_provider_error_fields() {
    let server = serve([Reply::json(
        403,
        json!({
            "error": {
                "message": "Provider returned error",
                "code": 403,
                "metadata": {"raw": "upstream WAF blocked policy XYZ"}
            }
        }),
    )])
    .await;
    let mut provider_model = model();
    provider_model.base_url = server.base_url.clone();
    let options = OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let events = openai::stream(
        &provider_model
            .typed::<ds_ai::OpenAiResponsesOptions>()
            .unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;
    let Some(AssistantMessageEvent::Error { error, .. }) = events.last() else {
        panic!("stream did not fail");
    };
    let message = error.error_message.as_deref().unwrap();
    assert!(message.contains("\"code\":403"));
    assert!(message.contains("upstream WAF blocked policy XYZ"));
    assert_eq!(
        message.matches("upstream WAF blocked policy XYZ").count(),
        1
    );
    server.requests().await;
}

#[tokio::test]
async fn preserves_nested_provider_codes_and_metadata_once() {
    let server = serve([
        Reply::json(
            400,
            json!({
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "slow down",
                    "metadata": {"raw": "upstream WAF blocked policy XYZ"}
                }
            }),
        ),
        Reply::json(400, json!({"error": {"message": "slow down"}})),
        Reply::json(403, json!({"error": {}})),
        Reply::json(429, json!({"message": "Too many requests"})),
    ])
    .await;
    let mut provider_model = model();
    provider_model.base_url = server.base_url.clone();
    let options = OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let context = Context::new([Message::user("Hello")]);

    let structured = openai::stream(
        &provider_model
            .typed::<ds_ai::OpenAiResponsesOptions>()
            .unwrap(),
        &context,
        &options,
    )
    .collect::<Vec<_>>()
    .await;
    let Some(AssistantMessageEvent::Error { error, .. }) = structured.last() else {
        panic!("stream did not fail");
    };
    let structured_message = error.error_message.as_deref().unwrap();
    assert_eq!(
        structured_message,
        r#"OpenAI API error (400): {"code":"rate_limit_exceeded","message":"slow down","metadata":{"raw":"upstream WAF blocked policy XYZ"}}"#
    );

    let simple = openai::stream(
        &provider_model
            .typed::<ds_ai::OpenAiResponsesOptions>()
            .unwrap(),
        &context,
        &options,
    )
    .collect::<Vec<_>>()
    .await;
    let Some(AssistantMessageEvent::Error { error, .. }) = simple.last() else {
        panic!("stream did not fail");
    };
    assert_eq!(
        error.error_message.as_deref(),
        Some(r#"OpenAI API error (400): {"message":"slow down"}"#)
    );

    let empty_inner = openai::stream(
        &provider_model
            .typed::<ds_ai::OpenAiResponsesOptions>()
            .unwrap(),
        &context,
        &options,
    )
    .collect::<Vec<_>>()
    .await;
    let Some(AssistantMessageEvent::Error { error, .. }) = empty_inner.last() else {
        panic!("stream did not fail");
    };
    assert_eq!(
        error.error_message.as_deref(),
        Some("OpenAI API error (403): 403 status code (no body)")
    );

    let top_level = openai::stream(
        &provider_model
            .typed::<ds_ai::OpenAiResponsesOptions>()
            .unwrap(),
        &context,
        &options,
    )
    .collect::<Vec<_>>()
    .await;
    let Some(AssistantMessageEvent::Error { error, .. }) = top_level.last() else {
        panic!("stream did not fail");
    };
    assert_eq!(
        error.error_message.as_deref(),
        Some("OpenAI API error (429): {\"message\":\"Too many requests\"}")
    );
    server.requests().await;
}

#[tokio::test]
async fn rejects_a_provider_error_body_read_failure() {
    let (base_url, server) = serve_truncated_error().await;
    let mut provider_model = model();
    provider_model.base_url = base_url;
    let options = OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let events = openai::stream(
        &provider_model
            .typed::<ds_ai::OpenAiResponsesOptions>()
            .unwrap(),
        &Context::new([Message::user("Hello")]),
        &options,
    )
    .collect::<Vec<_>>()
    .await;
    let Some(AssistantMessageEvent::Error { error, .. }) = events.last() else {
        panic!("stream did not fail");
    };
    let message = error.error_message.as_deref().unwrap();
    assert!(message.starts_with("HTTP request failed:"));
    assert!(!message.contains("partial"));
    server.await.unwrap();
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
    let calls = Mutex::new(0);
    let exhausted = retry_assistant_call(
        || async {
            *calls.lock().unwrap() += 1;
            assistant(1, StopReason::Error, Some("terminated"), 0)
        },
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
    assert_eq!(*calls.lock().unwrap(), 3);
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

#[tokio::test]
async fn aborts_an_active_retry_backoff_without_a_second_call() {
    let cancellation = CancellationToken::new();
    let scheduled = Arc::new(Notify::new());
    let callbacks = Arc::new(CancelOnSchedule {
        scheduled: scheduled.clone(),
    });
    let calls = Arc::new(Mutex::new(0));
    let task_calls = calls.clone();
    let task_cancellation = cancellation.clone();
    let task_callbacks = callbacks.clone();
    let task = tokio::spawn(async move {
        retry_assistant_call(
            || {
                let calls = task_calls.clone();
                async move {
                    *calls.lock().unwrap() += 1;
                    assistant(1, StopReason::Error, Some("terminated"), 0)
                }
            },
            Some(&RetryPolicy {
                enabled: true,
                max_retries: 5,
                base_delay: Duration::from_secs(60),
            }),
            Some(&task_cancellation),
            Some(task_callbacks.as_ref()),
        )
        .await
    });

    scheduled.notified().await;
    cancellation.cancel();
    let response = task.await.unwrap();

    assert_eq!(response.stop_reason, StopReason::Aborted);
    assert_eq!(response.error_message, None);
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn does_not_retry_with_a_pre_cancelled_zero_delay() {
    for _ in 0..16 {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let calls = Mutex::new(0);
        let response = retry_assistant_call(
            || async {
                *calls.lock().unwrap() += 1;
                assistant(1, StopReason::Error, Some("terminated"), 0)
            },
            Some(&RetryPolicy {
                enabled: true,
                max_retries: 1,
                base_delay: Duration::ZERO,
            }),
            Some(&cancellation),
            None,
        )
        .await;

        assert_eq!(response.stop_reason, StopReason::Aborted);
        assert_eq!(response.error_message, None);
        assert_eq!(*calls.lock().unwrap(), 1);
    }
}

#[tokio::test]
async fn reports_a_transient_then_aborted_retry_as_unsuccessful() {
    let calls = Mutex::new(0);
    let callbacks = RecordingCallbacks::default();
    let response = retry_assistant_call(
        || async {
            let mut calls = calls.lock().unwrap();
            *calls += 1;
            if *calls == 1 {
                assistant(1, StopReason::Error, Some("terminated"), 0)
            } else {
                assistant(1, StopReason::Aborted, None, 0)
            }
        },
        Some(&RetryPolicy {
            enabled: true,
            max_retries: 1,
            base_delay: Duration::ZERO,
        }),
        None,
        Some(&callbacks),
    )
    .await;

    assert_eq!(response.stop_reason, StopReason::Aborted);
    assert_eq!(*calls.lock().unwrap(), 2);
    assert_eq!(
        callbacks.events.lock().unwrap().as_slice(),
        [
            "scheduled:1:1:0:terminated",
            "start",
            "finished:false:1:none"
        ]
    );
}

#[derive(Default)]
struct RecordingCallbacks {
    events: Mutex<Vec<String>>,
}

struct CancelOnSchedule {
    scheduled: Arc<Notify>,
}

#[async_trait]
impl RetryCallbacks for CancelOnSchedule {
    async fn on_retry_scheduled(
        &self,
        _attempt: usize,
        _max_attempts: usize,
        _delay: Duration,
        _error_message: &str,
    ) {
        self.scheduled.notify_one();
    }
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

async fn serve_truncated_error() -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        loop {
            let mut chunk = [0; 1024];
            let count = socket.read(&mut chunk).await.unwrap();
            request.extend_from_slice(&chunk[..count]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        socket
            .write_all(
                b"HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: 100\r\nconnection: close\r\n\r\n{\"message\":\"partial",
            )
            .await
            .unwrap();
    });
    (format!("http://{address}"), server)
}
