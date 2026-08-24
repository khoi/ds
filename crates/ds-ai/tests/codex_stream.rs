use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{
    ApiStreamOptions, AssistantContent, AssistantMessage, AssistantMessageEvent, CacheRetention,
    Context, Message, OpenAiCodexResponsesOptions, Provider as _, SimpleStreamOptions, StopReason,
    StreamOptions, ThinkingLevel, Transport as ProviderTransport, builtin_model, codex,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};
use tokio_tungstenite::{
    accept_hdr_async,
    tungstenite::{
        Message as WebSocketMessage,
        handshake::server::{Callback, ErrorResponse, Request, Response},
        protocol::{CloseFrame, frame::coding::CloseCode},
    },
};

#[tokio::test]
async fn retries_and_compresses_a_codex_sse_request() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_codex\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_codex\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_codex\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_codex\",\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}}\n\n",
    ]
    .concat();
    let server = serve([
        Reply::json(503, json!({"error": {"message": "busy"}})).with_header("retry-after-ms", "0"),
        Reply::sse(sse),
    ])
    .await;
    let token = token("acc_test");
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Say hello")]).with_system("Be brief");
    let options = options(token.clone(), |options| {
        options.stream.max_retries = Some(1);
        options.stream.session_id = Some("session_1".into());
        options.stream.transport = Some(ProviderTransport::Sse);
    });

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
    assert_eq!(response.response_id.as_deref(), Some("resp_codex"));
    assert_text(response, "Hello");
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(response.usage.input, 5);
    assert_eq!(response.usage.output, 3);

    let requests = server.request_bytes().await;
    assert_eq!(requests.len(), 2);
    for bytes in requests {
        let (headers, body) = request(&bytes);
        assert!(headers.starts_with("POST /codex/responses HTTP/1.1\r\n"));
        assert!(headers.contains(&format!("authorization: Bearer {token}\r\n")));
        assert!(headers.contains("chatgpt-account-id: acc_test\r\n"));
        assert!(headers.contains("openai-beta: responses=experimental\r\n"));
        assert!(headers.contains("originator: ds\r\n"));
        assert!(headers.contains("user-agent: ds-ai/0.1.0\r\n"));
        assert!(headers.contains("accept: text/event-stream\r\n"));
        assert!(headers.contains("content-type: application/json\r\n"));
        assert!(headers.contains("content-encoding: zstd\r\n"));
        assert!(headers.contains("session-id: session_1\r\n"));
        assert!(headers.contains("x-client-request-id: session_1\r\n"));
        assert!(!headers.contains("x-api-key"));
        let body: Value = serde_json::from_slice(&zstd::stream::decode_all(body).unwrap()).unwrap();
        assert_eq!(
            body,
            json!({
                "model": "gpt-5.6-codex",
                "store": false,
                "stream": true,
                "instructions": "Be brief",
                "input": [{
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Say hello"}]
                }],
                "text": {"verbosity": "low"},
                "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": "session_1",
                "tool_choice": "auto",
                "parallel_tool_calls": true
            })
        );
    }
}

#[tokio::test]
async fn follows_codex_specific_retry_statuses_and_error_text() {
    for (status, message, should_retry) in [
        (408, "failed", false),
        (409, "failed", false),
        (429, "failed", true),
        (500, "failed", true),
        (501, "failed", false),
        (502, "failed", true),
        (503, "failed", true),
        (504, "failed", true),
        (599, "failed", false),
        (400, "upstream connect failed", true),
    ] {
        let mut replies = vec![
            Reply::json(status, json!({"error": {"message": message}}))
                .with_header("retry-after-ms", "0"),
        ];
        if should_retry {
            replies.push(Reply::sse(sse_text_events(
                &format!("resp_retry_{status}"),
                &format!("msg_retry_{status}"),
                "Done",
            )));
        }
        let server = serve(replies).await;
        let model = model(&server.base_url);
        let options = options(token("acc_retry_matrix"), |options| {
            options.stream.max_retries = Some(1);
            options.stream.transport = Some(ProviderTransport::Sse);
        });
        let events = events(&model, &Context::new([Message::user("Retry")]), &options).await;

        if should_retry {
            assert_text(done(&events), "Done");
            assert_eq!(server.request_count(), 2);
        } else {
            assert!(
                failed(&events).error_message.as_deref().is_some_and(
                    |message| message.starts_with(&format!("provider returned HTTP {status}:"))
                )
            );
            assert_eq!(server.request_count(), 1);
        }
    }
}

#[tokio::test]
async fn does_not_retry_a_terminal_codex_quota_error() {
    let reset_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 600;
    let failure = || {
        Reply::json(
            429,
            json!({
                "error": {
                    "code": "usage_limit_reached",
                    "message": "Monthly usage limit reached",
                    "plan_type": "PLUS",
                    "resets_at": reset_at
                }
            }),
        )
        .with_header("retry-after-ms", "0")
    };
    let server = serve([failure(), failure(), failure()]).await;
    let model = model(&server.base_url);
    let options = options(token("acc_quota"), |options| {
        options.stream.max_retries = Some(2);
        options.stream.transport = Some(ProviderTransport::Sse);
    });
    let events = events(&model, &Context::new([Message::user("Quota")]), &options).await;

    assert_eq!(
        failed(&events).error_message.as_deref(),
        Some(
            "provider returned HTTP 429: You have hit your ChatGPT usage limit (plus plan). Try again in ~10 min."
        )
    );
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn disables_the_codex_retry_delay_cap_with_zero() {
    let server = serve([
        Reply::json(429, json!({"error": {"message": "retry"}})).with_header("retry-after-ms", "1"),
        Reply::sse(sse_text_events("resp_no_cap", "msg_no_cap", "Done")),
    ])
    .await;
    let model = model(&server.base_url);
    let options = options(token("acc_no_cap"), |options| {
        options.stream.max_retries = Some(1);
        options.stream.max_retry_delay = Some(Duration::ZERO);
        options.stream.transport = Some(ProviderTransport::Sse);
    });
    let events = events(&model, &Context::new([Message::user("Retry")]), &options).await;

    assert_text(done(&events), "Done");
    assert_eq!(server.request_count(), 2);
}

#[tokio::test]
async fn rejects_a_codex_retry_delay_above_the_cap() {
    for status in [429, 503] {
        let server = serve([Reply::json(status, json!({"error": {"message": "retry"}}))
            .with_header("retry-after-ms", "1000")])
        .await;
        let model = model(&server.base_url);
        let options = options(token("acc_capped"), |options| {
            options.stream.max_retries = Some(1);
            options.stream.max_retry_delay = Some(Duration::from_millis(999));
            options.stream.transport = Some(ProviderTransport::Sse);
        });
        let events = events(&model, &Context::new([Message::user("Retry")]), &options).await;

        assert_eq!(
            failed(&events).error_message.as_deref(),
            Some("provider retry delay 1s exceeds 999ms")
        );
        assert_eq!(server.request_count(), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn uses_fixed_exponential_backoff_for_codex_retries() {
    let server = serve([
        Reply::json(503, json!({"error": {"message": "busy"}})),
        Reply::json(503, json!({"error": {"message": "busy"}})),
        Reply::json(503, json!({"error": {"message": "busy"}})),
        Reply::sse(sse_text_events("resp_backoff", "msg_backoff", "Done")),
    ])
    .await;
    let model = model(&server.base_url);
    let options = options(token("acc_backoff"), |options| {
        options.stream.max_retries = Some(3);
        options.stream.transport = Some(ProviderTransport::Sse);
    });
    let task = tokio::spawn(async move {
        events(&model, &Context::new([Message::user("Retry")]), &options).await
    });

    server.wait_for_requests(1).await;
    for (before, after, count) in [
        (Duration::from_millis(999), Duration::from_millis(1), 2),
        (Duration::from_millis(1999), Duration::from_millis(1), 3),
        (Duration::from_millis(3999), Duration::from_millis(1), 4),
    ] {
        tokio::time::advance(before).await;
        tokio::task::yield_now().await;
        assert_eq!(server.request_count(), count - 1);
        tokio::time::advance(after).await;
        server.wait_for_requests(count).await;
    }

    assert_text(done(&task.await.unwrap()), "Done");
}

#[tokio::test(start_paused = true)]
async fn honors_numeric_codex_retry_headers() {
    for (header, value, delay) in [
        ("retry-after-ms", "1500", Duration::from_millis(1500)),
        ("retry-after", "60", Duration::from_secs(60)),
    ] {
        let server = serve([
            Reply::json(429, json!({"error": {"message": "retry"}})).with_header(header, value),
            Reply::sse(sse_text_events("resp_delay", "msg_delay", "Done")),
        ])
        .await;
        let model = model(&server.base_url);
        let options = options(token("acc_delay"), |options| {
            options.stream.max_retries = Some(1);
            options.stream.max_retry_delay = None;
            options.stream.transport = Some(ProviderTransport::Sse);
        });
        let task = tokio::spawn(async move {
            events(&model, &Context::new([Message::user("Retry")]), &options).await
        });

        server.wait_for_requests(1).await;
        tokio::time::advance(delay - Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert_eq!(server.request_count(), 1);
        tokio::time::advance(Duration::from_millis(1)).await;
        server.wait_for_requests(2).await;
        assert_text(done(&task.await.unwrap()), "Done");
    }
}

#[tokio::test]
async fn parses_codex_http_date_retry_headers() {
    let retry_at = std::time::SystemTime::now() + Duration::from_secs(60);
    let server = serve([Reply::json(429, json!({"error": {"message": "retry"}}))
        .with_header("retry-after", httpdate::fmt_http_date(retry_at))])
    .await;
    let model = model(&server.base_url);
    let options = options(token("acc_date_delay"), |options| {
        options.stream.max_retries = Some(1);
        options.stream.max_retry_delay = Some(Duration::from_secs(1));
        options.stream.transport = Some(ProviderTransport::Sse);
    });
    let events = events(&model, &Context::new([Message::user("Retry")]), &options).await;

    let message = failed(&events).error_message.as_deref().unwrap();
    assert!(message.starts_with("provider retry delay "));
    assert!(message.ends_with(" exceeds 1s"));
}

#[tokio::test]
async fn does_not_retry_codex_requests_by_default() {
    let server = serve([
        Reply::json(503, json!({"error": {"message": "busy"}})),
        Reply::sse(sse_text_events(
            "resp_unexpected",
            "msg_unexpected",
            "Unexpected",
        )),
    ])
    .await;
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = codex::Provider::new([model.clone()]);

    let message = provider
        .stream(
            &model,
            &Context::new([Message::user("Retry")]),
            &ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
                stream: StreamOptions {
                    api_key: Some(token("acc_default_retry")),
                    transport: Some(ProviderTransport::Sse),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn finishes_codex_sse_at_a_terminal_event_while_the_body_stays_open() {
    for (terminal, expected_reason) in [
        (
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_open_completed",
                    "status": "completed",
                    "end_turn": false,
                    "usage": {}
                }
            }),
            StopReason::Stop,
        ),
        (
            json!({
                "type": "response.incomplete",
                "response": {
                    "id": "resp_open_incomplete",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {}
                }
            }),
            StopReason::Length,
        ),
    ] {
        let sse = [
            json!({"type": "response.created", "response": {"id": "resp_open"}}),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": "msg_open", "type": "message", "content": []}
            }),
            json!({"type": "response.output_text.delta", "output_index": 0, "delta": "Done"}),
            terminal,
        ]
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();
        let server = serve([Reply::open_sse(sse)]).await;
        let model = model(&server.base_url);
        let options = options(token("acc_open_terminal"), |options| {
            options.stream.transport = Some(ProviderTransport::Sse);
        });

        let events = tokio::time::timeout(Duration::from_secs(1), async {
            events(&model, &Context::new([Message::user("Finish")]), &options).await
        })
        .await
        .unwrap();

        let response = done(&events);
        assert_text(response, "Done");
        assert_eq!(response.stop_reason, expected_reason);
        if expected_reason == StopReason::Stop {
            assert_eq!(response.end_turn, Some(false));
        }
        server.request_bytes().await;
    }
}

#[tokio::test]
async fn cancels_an_active_codex_sse_body_with_partial_content() {
    let server = serve([Reply::open_sse(
        [
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_cancel\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_cancel\",\"type\":\"message\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
        ]
        .concat(),
    )])
    .await;
    let cancellation = tokio_util::sync::CancellationToken::new();
    let model = model(&server.base_url);
    let options = options(token("acc_cancel"), |options| {
        options.stream.cancellation = cancellation.clone();
        options.stream.transport = Some(ProviderTransport::Sse);
    });
    let mut stream = codex::stream(&model, &Context::new([Message::user("Cancel")]), &options);

    while !matches!(
        stream.next().await,
        Some(AssistantMessageEvent::TextDelta { .. })
    ) {}
    cancellation.cancel();

    match stream.next().await {
        Some(AssistantMessageEvent::Error { reason, error }) => {
            assert_eq!(reason, StopReason::Aborted);
            assert_eq!(error.response_id.as_deref(), Some("resp_cancel"));
            assert_text(&error, "Visible");
        }
        event => panic!("unexpected cancellation event: {event:?}"),
    }
    drop(stream);
    server.request_bytes().await;
}

#[tokio::test]
async fn applies_codex_sse_cache_affinity_rules() {
    let long_session = "x".repeat(67);
    for (retention, session, expected) in [
        (
            ds_ai::CacheRetention::Short,
            Some(long_session.as_str()),
            Some("x".repeat(64)),
        ),
        (ds_ai::CacheRetention::None, Some("one-shot"), None),
        (ds_ai::CacheRetention::Short, None, None),
    ] {
        let server = serve([Reply::sse(sse_text_events(
            "resp_affinity",
            "msg_affinity",
            "Done",
        ))])
        .await;
        let model = model(&server.base_url);
        let options = options(token("acc_affinity"), |options| {
            options.stream.cache_retention = retention;
            options.stream.transport = Some(ProviderTransport::Sse);
            options.stream.session_id = session.map(str::to_owned);
        });

        let events = events(&model, &Context::new([Message::user("Cache")]), &options).await;
        done(&events);

        let request_bytes = server.request_bytes().await.pop().unwrap();
        let (headers, body) = request(&request_bytes);
        let body: Value = serde_json::from_slice(&zstd::stream::decode_all(body).unwrap()).unwrap();
        match expected {
            Some(expected) => {
                assert!(headers.contains(&format!("session-id: {expected}\r\n")));
                assert!(headers.contains(&format!("x-client-request-id: {expected}\r\n")));
                assert_eq!(body["prompt_cache_key"], expected);
            }
            None => {
                assert!(!headers.contains("session-id:"));
                assert!(!headers.contains("x-client-request-id:"));
                assert!(body.get("prompt_cache_key").is_none());
            }
        }
    }
}

#[tokio::test]
async fn routes_simple_codex_options_to_cached_websocket_transport() {
    codex::reset_websocket_debug_stats(Some("session_simple"));
    let (base_url, capture) =
        serve_websocket([text_events("resp_simple", "msg_simple", "Done")]).await;
    let mut model = builtin_model("openai-codex", "gpt-5.5").unwrap();
    model.base_url = base_url;
    let provider = codex::Provider::new([model.clone()]);

    let message = provider
        .stream_simple(
            &model,
            &Context::new([Message::user("Connect")]),
            &SimpleStreamOptions {
                stream: StreamOptions {
                    api_key: Some(token("acc_simple")),
                    session_id: Some("session_simple".into()),
                    transport: Some(ProviderTransport::Auto),
                    ..Default::default()
                },
                thinking: Some(ThinkingLevel::XHigh),
                ..Default::default()
            },
        )
        .result()
        .await
        .unwrap();

    assert_eq!(message.stop_reason, StopReason::Stop);
    let capture = capture.await.unwrap();
    assert_eq!(capture.headers["session-id"], "session_simple");
    assert_eq!(capture.headers["x-client-request-id"], "session_simple");
    assert_eq!(capture.bodies[0]["reasoning"]["effort"], "xhigh");
    assert_eq!(
        codex::websocket_debug_stats("session_simple"),
        Some(codex::WebSocketDebugStats {
            requests: 1,
            connections_created: 1,
            cached_context_requests: 1,
            full_context_requests: 1,
            last_input_items: 1,
            ..Default::default()
        })
    );
    codex::close_websocket_sessions(Some("session_simple"));
    codex::reset_websocket_debug_stats(Some("session_simple"));
}

#[tokio::test]
async fn streams_a_codex_websocket_request() {
    let (base_url, capture) = serve_websocket([vec![
        json!({"type": "response.created", "response": {"id": "resp_ws"}}),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_ws", "type": "message", "content": []}
        }),
        json!({"type": "response.output_text.delta", "output_index": 0, "delta": "WebSocket"}),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "msg_ws",
                "type": "message",
                "content": [{"type": "output_text", "text": "WebSocket"}]
            }
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_ws",
                "status": "completed",
                "service_tier": "priority",
                "end_turn": true,
                "usage": {}
            }
        }),
    ]])
    .await;
    let token = token("acc_ws");
    let model = model(base_url);
    let context = Context::new([Message::user("Connect")]).with_system("Be brief");
    let options = options(token.clone(), |options| {
        options.stream.session_id = Some("session_ws".into());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });

    let events = events(&model, &context, &options).await;

    let response = done(&events);
    assert_text(response, "WebSocket");
    assert_eq!(response.end_turn, Some(true));
    let capture = capture.await.unwrap();
    assert_eq!(capture.path, "/codex/responses");
    assert_eq!(capture.headers["authorization"], format!("Bearer {token}"));
    assert_eq!(capture.headers["chatgpt-account-id"], "acc_ws");
    assert_eq!(capture.headers["originator"], "ds");
    assert_eq!(capture.headers["user-agent"], "ds-ai/0.1.0");
    assert_eq!(capture.headers["x-client-request-id"], "session_ws");
    assert_eq!(capture.headers["session-id"], "session_ws");
    assert_eq!(
        capture.headers["openai-beta"],
        "responses_websockets=2026-02-06"
    );
    assert!(!capture.headers.contains_key("content-encoding"));
    assert_eq!(capture.bodies[0]["type"], "response.create");
    assert_eq!(capture.bodies[0]["model"], "gpt-5.6-codex");
    assert_eq!(
        capture.bodies[0]["input"][0]["content"][0]["text"],
        "Connect"
    );
}

#[tokio::test]
async fn uses_uuid_v7_for_a_sessionless_codex_websocket_request() {
    let (base_url, capture) = serve_websocket([text_events("resp_uuid", "msg_uuid", "Done")]).await;
    let model = model(base_url);
    let options = options(token("acc_uuid"), |options| {
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let events = events(&model, &Context::new([Message::user("Hello")]), &options).await;
    done(&events);

    let capture = capture.await.unwrap();
    let request_id = &capture.headers["x-client-request-id"];
    assert_eq!(request_id, &capture.headers["session-id"]);
    assert_eq!(request_id.len(), 36);
    assert_eq!(&request_id[14..15], "7");
    assert!(matches!(&request_id[19..20], "8" | "9" | "a" | "b"));
}

#[tokio::test]
async fn encodes_codex_generation_options() {
    let (base_url, capture) =
        serve_websocket([text_events("resp_options", "msg_options", "Configured")]).await;
    let model = model(base_url);
    let options = options(token("acc_options"), |options| {
        options.stream.cache_retention = CacheRetention::None;
        options.stream.temperature = Some(0.25);
        options.stream.transport = Some(ProviderTransport::WebSocket);
        options.reasoning_effort = Some(codex::ReasoningEffort::High);
        options.reasoning_summary = Some(codex::ReasoningSummary::Concise);
        options.service_tier = Some(codex::ServiceTier::Priority);
        options.text_verbosity = Some(codex::TextVerbosity::High);
        options.tool_choice = Some(codex::ToolChoice::Required);
    });
    let events = events(
        &model,
        &Context::new([Message::user("Configure")]),
        &options,
    )
    .await;

    assert_text(done(&events), "Configured");
    let body = &capture.await.unwrap().bodies[0];
    assert_eq!(body["temperature"], 0.25);
    assert_eq!(
        body["reasoning"],
        json!({"effort": "high", "summary": "concise"})
    );
    assert_eq!(body["service_tier"], "priority");
    assert_eq!(body["text"], json!({"verbosity": "high"}));
    assert_eq!(body["tool_choice"], "required");
}

#[tokio::test]
async fn reuses_a_codex_websocket_with_an_input_delta() {
    codex::reset_websocket_debug_stats(Some("session_reuse"));
    let (base_url, capture) = serve_websocket([
        text_events("resp_first", "msg_first", "First"),
        text_events("resp_second", "msg_second", "Second"),
    ])
    .await;
    let model = model(base_url);
    let options = options(token("acc_reuse"), |options| {
        options.stream.session_id = Some("session_reuse".into());
        options.stream.transport = Some(ProviderTransport::WebSocketCached);
    });
    let first_context = Context::new([Message::user("First")]).with_system("Be brief");
    let first_events = events(&model, &first_context, &options).await;
    let first_response = done(&first_events).clone();
    let second_context = Context::new([
        Message::user("First"),
        Message::assistant(first_response),
        Message::user("Continue"),
    ])
    .with_system("Be brief");

    let second_events = events(&model, &second_context, &options).await;

    assert_text(done(&second_events), "Second");
    let capture = capture.await.unwrap();
    assert_eq!(capture.bodies.len(), 2);
    assert!(capture.bodies[0].get("previous_response_id").is_none());
    assert_eq!(capture.bodies[0]["input"].as_array().unwrap().len(), 1);
    assert_eq!(capture.bodies[1]["previous_response_id"], "resp_first");
    assert_eq!(
        capture.bodies[1]["input"],
        json!([{
            "role": "user",
            "content": [{"type": "input_text", "text": "Continue"}]
        }])
    );
    assert_eq!(
        codex::websocket_debug_stats("session_reuse"),
        Some(codex::WebSocketDebugStats {
            requests: 2,
            connections_created: 1,
            connections_reused: 1,
            cached_context_requests: 2,
            full_context_requests: 1,
            delta_requests: 1,
            last_input_items: 1,
            last_delta_input_items: Some(1),
            last_previous_response_id: Some("resp_first".into()),
            ..Default::default()
        })
    );
    codex::close_websocket_sessions(Some("session_reuse"));
    codex::reset_websocket_debug_stats(Some("session_reuse"));
}

#[tokio::test]
async fn scopes_cached_codex_websockets_to_the_account() {
    let (base_url, capture) = serve_kept_websocket_connections([
        text_events("resp_account_one", "msg_account_one", "First"),
        text_events("resp_account_two", "msg_account_two", "Second"),
    ])
    .await;
    let model = model(base_url);
    let context = Context::new([Message::user("Connect")]);
    let first_options = options(token("acc_scope_one"), |options| {
        options.stream.session_id = Some("session_scope".into());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let first = events(&model, &context, &first_options).await;
    let second_options = options(token("acc_scope_two"), |options| {
        options.stream.session_id = Some("session_scope".into());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let second = events(&model, &context, &second_options).await;

    assert_text(done(&first), "First");
    assert_text(done(&second), "Second");
    assert_eq!(capture.await.unwrap().len(), 2);
}

#[tokio::test]
async fn opens_a_one_shot_codex_websocket_while_the_cached_session_is_busy() {
    codex::reset_websocket_debug_stats(Some("session_busy"));
    let (base_url, first_received, second_received) = serve_concurrent_session_websockets().await;
    let model = model(base_url);
    let context = Context::new([Message::user("Connect")]);
    let first_model = model.clone();
    let first_context = context.clone();
    let first = tokio::spawn(async move {
        let options = options(token("acc_busy"), |options| {
            options.stream.session_id = Some("session_busy".into());
            options.stream.transport = Some(ProviderTransport::WebSocketCached);
        });
        events(&first_model, &first_context, &options).await
    });
    first_received.await.unwrap();

    let second = tokio::spawn(async move {
        let options = options(token("acc_busy"), |options| {
            options.stream.session_id = Some("session_busy".into());
            options.stream.transport = Some(ProviderTransport::WebSocketCached);
        });
        events(&model, &context, &options).await
    });
    tokio::time::timeout(Duration::from_secs(1), second_received)
        .await
        .unwrap()
        .unwrap();

    assert_text(done(&first.await.unwrap()), "First");
    assert_text(done(&second.await.unwrap()), "Second");
    assert_eq!(
        codex::websocket_debug_stats("session_busy"),
        Some(codex::WebSocketDebugStats {
            requests: 2,
            connections_created: 2,
            cached_context_requests: 2,
            full_context_requests: 2,
            last_input_items: 1,
            ..Default::default()
        })
    );
    codex::close_websocket_sessions(Some("session_busy"));
    codex::reset_websocket_debug_stats(Some("session_busy"));
}

#[tokio::test]
async fn scopes_cached_codex_websockets_to_the_unclamped_session() {
    let (base_url, reused) = serve_session_isolation_websockets().await;
    let model = model(base_url);
    let context = Context::new([Message::user("Connect")]);
    let prefix = "s".repeat(64);
    let first_session = format!("{prefix}a");
    let second_session = format!("{prefix}b");

    let first_options = options(token("acc_session_scope"), |options| {
        options.stream.session_id = Some(first_session.clone());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let first = events(&model, &context, &first_options).await;
    let second_options = options(token("acc_session_scope"), |options| {
        options.stream.session_id = Some(second_session.clone());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let second = events(&model, &context, &second_options).await;

    assert_text(done(&first), "First");
    assert_text(done(&second), "Second");
    assert!(!reused.await.unwrap());
    codex::close_websocket_sessions(Some(&first_session));
    codex::close_websocket_sessions(Some(&second_session));
}

#[tokio::test]
async fn retries_a_missing_codex_continuation_with_full_context() {
    codex::reset_websocket_debug_stats(Some("session_recovery"));
    let (base_url, capture) = serve_missing_continuation_websockets().await;
    let model = model(base_url);
    let options = options(token("acc_recovery"), |options| {
        options.stream.session_id = Some("session_recovery".into());
        options.stream.transport = Some(ProviderTransport::WebSocketCached);
    });
    let first_events = events(
        &model,
        &Context::new([Message::user("Seed")]).with_system("Be brief"),
        &options,
    )
    .await;
    let first_response = done(&first_events).clone();
    let second_context = Context::new([
        Message::user("Seed"),
        Message::assistant(first_response),
        Message::user("Continue"),
    ])
    .with_system("Be brief");

    let second_events = events(&model, &second_context, &options).await;

    assert_text(done(&second_events), "Recovered");
    let bodies = capture.await.unwrap();
    assert_eq!(bodies.len(), 3);
    assert_eq!(bodies[1]["previous_response_id"], "resp_seed");
    assert!(bodies[2].get("previous_response_id").is_none());
    assert_eq!(bodies[2]["input"].as_array().unwrap().len(), 3);
    assert_eq!(
        codex::websocket_debug_stats("session_recovery"),
        Some(codex::WebSocketDebugStats {
            requests: 3,
            connections_created: 2,
            connections_reused: 1,
            cached_context_requests: 3,
            full_context_requests: 2,
            delta_requests: 1,
            last_input_items: 3,
            ..Default::default()
        })
    );
    codex::close_websocket_sessions(Some("session_recovery"));
    codex::reset_websocket_debug_stats(Some("session_recovery"));
}

#[tokio::test]
async fn falls_back_to_sse_when_missing_continuation_recovery_cannot_start() {
    codex::reset_websocket_debug_stats(Some("session_recovery_sse"));
    let (base_url, capture) = serve_missing_continuation_then_sse().await;
    let model = model(base_url);
    let options = options(token("acc_recovery_sse"), |options| {
        options.stream.session_id = Some("session_recovery_sse".into());
        options.stream.transport = Some(ProviderTransport::WebSocketCached);
    });
    let first = events(&model, &Context::new([Message::user("Seed")]), &options).await;
    let context = Context::new([
        Message::user("Seed"),
        Message::assistant(done(&first).clone()),
        Message::user("Continue"),
    ]);

    let second = events(&model, &context, &options).await;

    assert_text(done(&second), "Fallback");
    let capture = capture.await.unwrap();
    assert_eq!(capture.websocket_bodies.len(), 3);
    assert_eq!(
        capture.websocket_bodies[1]["previous_response_id"],
        "resp_seed_sse"
    );
    assert!(
        capture.websocket_bodies[2]
            .get("previous_response_id")
            .is_none()
    );
    assert_eq!(
        capture.websocket_bodies[2]["input"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(capture.http_requests, 1);
    let stats = codex::websocket_debug_stats("session_recovery_sse").unwrap();
    assert_eq!(stats.requests, 3);
    assert_eq!(stats.connections_created, 2);
    assert_eq!(stats.connections_reused, 1);
    assert_eq!(stats.full_context_requests, 2);
    assert_eq!(stats.delta_requests, 1);
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
    codex::close_websocket_sessions(Some("session_recovery_sse"));
    codex::reset_websocket_debug_stats(Some("session_recovery_sse"));
}

#[tokio::test]
async fn reconnects_once_when_the_codex_websocket_limit_is_reached() {
    codex::reset_websocket_debug_stats(Some("session_limit"));
    let (base_url, capture) = serve_websocket_connections([
        vec![json!({
            "type": "error",
            "error": {"code": "websocket_connection_limit_reached"}
        })],
        text_events("resp_reconnected", "msg_reconnected", "Reconnected"),
    ])
    .await;
    let model = model(base_url);
    let options = options(token("acc_limit"), |options| {
        options.stream.session_id = Some("session_limit".into());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let events = events(&model, &Context::new([Message::user("Connect")]), &options).await;

    assert_text(done(&events), "Reconnected");
    let bodies = capture.await.unwrap();
    assert_eq!(bodies.len(), 2);
    assert!(
        bodies
            .iter()
            .all(|body| body.get("previous_response_id").is_none())
    );
    assert!(
        bodies
            .iter()
            .all(|body| body["input"].as_array().unwrap().len() == 1)
    );
    assert_eq!(
        codex::websocket_debug_stats("session_limit"),
        Some(codex::WebSocketDebugStats {
            requests: 2,
            connections_created: 2,
            full_context_requests: 2,
            last_input_items: 1,
            ..Default::default()
        })
    );
    codex::close_websocket_sessions(Some("session_limit"));
    codex::reset_websocket_debug_stats(Some("session_limit"));
}

#[tokio::test]
async fn falls_back_after_the_codex_websocket_limit_retry_is_exhausted() {
    codex::reset_websocket_debug_stats(Some("session_limit_twice"));
    let (base_url, requests) = serve_repeated_limit_then_sse().await;
    let model = model(base_url);
    let options = options(token("acc_limit_twice"), |options| {
        options.stream.session_id = Some("session_limit_twice".into());
        options.stream.transport = Some(ProviderTransport::Auto);
    });
    let events = events(&model, &Context::new([Message::user("Connect")]), &options).await;

    assert_text(done(&events), "Fallback");
    assert_eq!(requests.await.unwrap(), 3);
    let stats = codex::websocket_debug_stats("session_limit_twice").unwrap();
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
    codex::reset_websocket_debug_stats(Some("session_limit_twice"));
}

#[tokio::test]
async fn replaces_a_stale_cached_codex_websocket() {
    let (base_url, capture) = serve_websocket_connections([
        text_events("resp_stale_seed", "msg_stale_seed", "First"),
        text_events("resp_stale_fresh", "msg_stale_fresh", "Second"),
    ])
    .await;
    let model = model(base_url);
    let options = options(token("acc_stale"), |options| {
        options.stream.session_id = Some("session_stale".into());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let context = Context::new([Message::user("Connect")]);
    let first = events(&model, &context, &options).await;
    assert_text(done(&first), "First");

    let second = events(&model, &context, &options).await;

    assert_text(done(&second), "Second");
    assert_eq!(capture.await.unwrap().len(), 2);
}

#[tokio::test]
async fn expires_an_idle_cached_codex_websocket() {
    let (base_url, closed) = serve_one_shot_websocket(
        text_events("resp_idle", "msg_idle", "Done"),
        Duration::from_secs(6 * 60),
    )
    .await;
    let model = model(base_url);
    let options = options(token("acc_idle"), |options| {
        options.stream.session_id = Some("session_idle".into());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let events = events(&model, &Context::new([Message::user("Connect")]), &options).await;

    assert_text(done(&events), "Done");
    tokio::task::yield_now().await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(5 * 60)).await;
    assert!(closed.await.unwrap());
    tokio::time::resume();
}

#[tokio::test]
async fn replaces_a_codex_websocket_at_the_connection_age_limit() {
    let (base_url, connections) = serve_websocket_age_limit(13).await;
    let model = model(base_url);
    let options = options(token("acc_age"), |options| {
        options.stream.session_id = Some("session_age".into());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let context = Context::new([Message::user("Connect")]);

    let response = events(&model, &context, &options).await;
    assert_text(done(&response), "Done");
    tokio::task::yield_now().await;
    tokio::time::pause();
    let prevent_auto_advance = tokio::spawn(async {
        loop {
            tokio::task::yield_now().await;
        }
    });

    for _ in 1..13 {
        tokio::time::advance(Duration::from_secs(4 * 60 + 30)).await;
        let response = events(&model, &context, &options).await;
        assert_text(done(&response), "Done");
        tokio::task::yield_now().await;
    }

    tokio::time::advance(Duration::from_secs(61)).await;
    let response = events(&model, &context, &options).await;
    assert_text(done(&response), "Fresh");
    assert_eq!(connections.await.unwrap(), 2);
    codex::close_websocket_sessions(Some("session_age"));
    prevent_auto_advance.abort();
    tokio::time::resume();
}

#[tokio::test]
async fn evicts_a_cached_codex_websocket_after_stream_failure() {
    codex::reset_websocket_debug_stats(Some("session_failed_cache"));
    let (base_url, evicted) = serve_failed_then_fresh_websockets().await;
    let model = model(base_url);
    let options = options(token("acc_failed_cache"), |options| {
        options.stream.session_id = Some("session_failed_cache".into());
        options.stream.timeout = Some(Duration::from_millis(10));
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let context = Context::new([Message::user("Connect")]);

    let failed_events = events(&model, &context, &options).await;
    assert_eq!(
        failed(&failed_events).error_message.as_deref(),
        Some("provider timed out during Idle")
    );
    assert_eq!(
        codex::websocket_debug_stats("session_failed_cache")
            .unwrap()
            .websocket_failures,
        1
    );
    codex::reset_websocket_debug_stats(Some("session_failed_cache"));

    let fresh = events(&model, &context, &options).await;

    assert_text(done(&fresh), "Fresh");
    assert!(evicted.await.unwrap());
    codex::close_websocket_sessions(Some("session_failed_cache"));
}

#[tokio::test]
async fn closes_a_one_shot_codex_websocket_after_completion() {
    let (base_url, closed) = serve_one_shot_websocket(
        text_events("resp_one_shot", "msg_one_shot", "Done"),
        Duration::from_secs(1),
    )
    .await;
    let model = model(base_url);
    let options = options(token("acc_one_shot"), |options| {
        options.stream.cache_retention = CacheRetention::None;
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let events = events(&model, &Context::new([Message::user("Connect")]), &options).await;

    assert_text(done(&events), "Done");
    assert!(closed.await.unwrap());
}

#[tokio::test]
async fn falls_back_to_sse_when_codex_websocket_connect_fails() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_fallback\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Fallback\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fallback\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::disconnect(), Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Connect")]),
        &options(token("acc_fallback"), |_| {}),
    )
    .await;

    assert_text(done(&events), "Fallback");
    let requests = server.request_bytes().await;
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with(b"GET /codex/responses HTTP/1.1\r\n"));
    assert!(requests[1].starts_with(b"POST /codex/responses HTTP/1.1\r\n"));
    let (_, body) = request(&requests[1]);
    let body: Value = serde_json::from_slice(&zstd::stream::decode_all(body).unwrap()).unwrap();
    assert_eq!(body["input"][0]["content"][0]["text"], "Connect");
}

#[tokio::test]
async fn falls_back_to_sse_after_the_codex_websocket_connect_timeout() {
    let base_url = serve_stalled_websocket_handshake_then_sse().await;
    let model = model(base_url);
    let options = options(token("acc_connect_timeout"), |options| {
        options.stream.timeout = Some(Duration::from_millis(500));
        options.stream.websocket_connect_timeout = Some(Duration::from_millis(10));
        options.stream.transport = Some(ProviderTransport::Auto);
    });

    let events = tokio::time::timeout(Duration::from_secs(1), async {
        events(&model, &Context::new([Message::user("Connect")]), &options).await
    })
    .await
    .unwrap();

    assert_text(done(&events), "Fallback");
}

#[tokio::test]
async fn reports_a_codex_websocket_fallback_on_the_assistant_message() {
    let server = serve([
        Reply::json(400, json!({"error": "websocket unavailable"})),
        Reply::sse(sse_text_events(
            "resp_diagnostic",
            "msg_diagnostic",
            "Fallback",
        )),
    ])
    .await;
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = codex::Provider::new([model.clone()]);

    let message = provider
        .stream(
            &model,
            &Context::new([Message::user("Connect")]),
            &ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
                stream: StreamOptions {
                    api_key: Some(token("acc_diagnostic")),
                    transport: Some(ProviderTransport::Auto),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let diagnostic = &message.diagnostics.unwrap()[0];
    assert_eq!(diagnostic.r#type, "provider_transport_failure");
    assert!(
        diagnostic
            .error
            .as_ref()
            .is_some_and(|error| !error.message.is_empty())
    );
    let details = diagnostic.details.as_ref().unwrap();
    assert_eq!(details["configuredTransport"], "auto");
    assert_eq!(details["fallbackTransport"], "sse");
    assert_eq!(details["eventsEmitted"], false);
    assert_eq!(details["phase"], "before_message_stream_start");
    assert!(details["requestBytes"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn applies_the_provider_timeout_to_codex_sse_response_headers() {
    let server = serve([Reply::pending()]).await;
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = codex::Provider::new([model.clone()]);

    let message = provider
        .stream(
            &model,
            &Context::new([Message::user("Connect")]),
            &ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
                stream: StreamOptions {
                    api_key: Some(token("acc_sse_timeout")),
                    timeout: Some(Duration::from_millis(10)),
                    max_retries: Some(0),
                    transport: Some(ProviderTransport::Sse),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("provider timed out during Connection")
    );
}

#[tokio::test]
async fn applies_the_provider_timeout_to_codex_websocket_idle_time() {
    let base_url = serve_started_idle_websocket().await;
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.base_url = base_url;
    let provider = codex::Provider::new([model.clone()]);

    let message = provider
        .stream(
            &model,
            &Context::new([Message::user("Connect")]),
            &ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
                stream: StreamOptions {
                    api_key: Some(token("acc_websocket_timeout")),
                    timeout: Some(Duration::from_millis(10)),
                    max_retries: Some(0),
                    transport: Some(ProviderTransport::WebSocket),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("provider timed out during Idle")
    );
}

#[tokio::test]
async fn keeps_a_codex_session_on_sse_after_a_websocket_failure() {
    codex::reset_websocket_debug_stats(Some("session_sticky"));
    let first_server = serve([
        Reply::json(400, json!({"error": "websocket unavailable"})),
        Reply::sse(sse_text_events("resp_first", "msg_first", "First")),
    ])
    .await;
    let first_model = model(&first_server.base_url);
    let options = options(token("acc_sticky"), |options| {
        options.stream.session_id = Some("session_sticky".into());
        options.stream.transport = Some(ProviderTransport::Auto);
    });

    let first = events(
        &first_model,
        &Context::new([Message::user("First")]),
        &options,
    )
    .await;

    assert_text(done(&first), "First");
    let first_requests = first_server.request_bytes().await;
    assert!(
        request(&first_requests[0])
            .0
            .starts_with("GET /codex/responses HTTP/1.1\r\n")
    );
    assert!(
        request(&first_requests[1])
            .0
            .starts_with("POST /codex/responses HTTP/1.1\r\n")
    );

    let second_server = serve([Reply::sse(sse_text_events(
        "resp_second",
        "msg_second",
        "Second",
    ))])
    .await;
    let second_model = model(&second_server.base_url);

    let second = events(
        &second_model,
        &Context::new([Message::user("Second")]),
        &options,
    )
    .await;

    assert_text(done(&second), "Second");
    let second_requests = second_server.request_bytes().await;
    assert_eq!(second_requests.len(), 1);
    assert!(
        request(&second_requests[0])
            .0
            .starts_with("POST /codex/responses HTTP/1.1\r\n")
    );
    let stats = codex::websocket_debug_stats("session_sticky").unwrap();
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 2);
    assert_eq!(stats.websocket_fallback_active, Some(true));
    assert!(stats.last_websocket_error.is_some());
    codex::reset_websocket_debug_stats(Some("session_sticky"));
}

#[tokio::test]
async fn closes_cached_codex_websockets_by_session() {
    let (base_url, closed) = serve_one_shot_websocket(
        text_events("resp_close", "msg_close", "Done"),
        Duration::from_secs(1),
    )
    .await;
    let model = model(base_url);
    let options = options(token("acc_close"), |options| {
        options.stream.session_id = Some("session_close".into());
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });

    let events = events(&model, &Context::new([Message::user("Connect")]), &options).await;

    assert_text(done(&events), "Done");
    codex::close_websocket_sessions(Some("session_close"));
    assert!(closed.await.unwrap());
}

#[tokio::test]
async fn falls_back_to_sse_when_codex_websocket_has_no_first_event() {
    let (base_url, capture) = serve_idle_websocket_then_sse().await;
    let model = model(base_url);
    let options = options(token("acc_idle_fallback"), |options| {
        options.stream.cache_retention = CacheRetention::None;
        options.stream.timeout = Some(Duration::from_millis(10));
    });
    let events = events(&model, &Context::new([Message::user("Wait")]), &options).await;

    assert_text(done(&events), "Fallback");
    let requests = capture.await.unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with(b"GET /codex/responses HTTP/1.1\r\n"));
    assert!(requests[1].starts_with(b"POST /codex/responses HTTP/1.1\r\n"));
}

#[tokio::test]
async fn does_not_fall_back_after_a_codex_websocket_event() {
    codex::reset_websocket_debug_stats(Some("session_started_idle"));
    let base_url = serve_started_idle_websocket().await;
    let model = model(base_url);
    let options = options(token("acc_started_idle"), |options| {
        options.stream.session_id = Some("session_started_idle".into());
        options.stream.timeout = Some(Duration::from_millis(10));
    });
    let events = events(&model, &Context::new([Message::user("Wait")]), &options).await;

    let error = failed(&events);
    assert_eq!(error.response_id.as_deref(), Some("resp_started_idle"));
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider timed out during Idle")
    );
    let stats = codex::websocket_debug_stats("session_started_idle").unwrap();
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 0);
    assert_eq!(stats.websocket_fallback_active, Some(true));
    codex::reset_websocket_debug_stats(Some("session_started_idle"));
}

#[tokio::test]
async fn preserves_partial_content_after_an_oversized_websocket_close() {
    let base_url = serve_websocket_close([
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_partial", "type": "message", "content": []}
        }),
        json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "Partial"
        }),
    ])
    .await;
    let model = model(base_url);
    let options = options(token("acc_large"), |options| {
        options.stream.cache_retention = CacheRetention::None;
        options.stream.transport = Some(ProviderTransport::WebSocket);
    });
    let events = events(
        &model,
        &Context::new([Message::user("Large response")]),
        &options,
    )
    .await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Partial"
    )));
    let error = failed(&events);
    assert!(
        error
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("code 1009"))
    );
    assert_text(error, "Partial");
}

fn token(account_id: &str) -> String {
    let payload = BASE64_URL_SAFE_NO_PAD.encode(
        serde_json::to_vec(&json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id}
        }))
        .unwrap(),
    );
    format!("aaa.{payload}.bbb")
}

fn request(request: &[u8]) -> (&str, &[u8]) {
    let header_end = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap();
    (
        std::str::from_utf8(&request[..header_end + 4]).unwrap(),
        &request[header_end + 4..],
    )
}

fn model(base_url: impl Into<String>) -> ds_ai::Model {
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.id = "gpt-5.6-codex".into();
    model.name = "gpt-5.6-codex".into();
    model.base_url = base_url.into();
    model
}

fn options(
    access_token: impl Into<String>,
    configure: impl FnOnce(&mut OpenAiCodexResponsesOptions),
) -> OpenAiCodexResponsesOptions {
    let mut options = OpenAiCodexResponsesOptions {
        stream: StreamOptions {
            api_key: Some(access_token.into()),
            ..Default::default()
        },
        ..Default::default()
    };
    configure(&mut options);
    options
}

async fn events(
    model: &ds_ai::Model,
    context: &Context,
    options: &OpenAiCodexResponsesOptions,
) -> Vec<AssistantMessageEvent> {
    codex::stream(model, context, options).collect().await
}

fn assert_text(message: &AssistantMessage, expected: &str) {
    assert!(matches!(
        message.content.as_slice(),
        [AssistantContent::Text(content)] if content.text == expected
    ));
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

struct WebSocketCapture {
    path: String,
    headers: BTreeMap<String, String>,
    bodies: Vec<Value>,
}

type CapturedHandshake = (String, BTreeMap<String, String>);

async fn serve_websocket(
    event_batches: impl IntoIterator<Item = Vec<Value>>,
) -> (String, oneshot::Receiver<WebSocketCapture>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let event_batches = event_batches.into_iter().collect::<Vec<_>>();
    let handshake = Arc::new(Mutex::new(None));
    let task_handshake = handshake.clone();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = accept_hdr_async(
            socket,
            CaptureHandshake {
                capture: task_handshake,
            },
        )
        .await
        .unwrap();
        let mut bodies = Vec::new();
        for events in event_batches {
            let body = match socket.next().await {
                Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
                message => panic!("unexpected websocket request: {message:?}"),
            };
            bodies.push(body);
            for event in events {
                socket
                    .send(WebSocketMessage::Text(event.to_string().into()))
                    .await
                    .unwrap();
            }
        }
        let (path, headers) = handshake.lock().unwrap().take().unwrap();
        sender
            .send(WebSocketCapture {
                path,
                headers,
                bodies,
            })
            .ok();
    });
    (format!("http://{address}"), receiver)
}

async fn serve_websocket_close(events: impl IntoIterator<Item = Value>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let events = events.into_iter().collect::<Vec<_>>();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
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
        socket
            .close(Some(CloseFrame {
                code: CloseCode::Size,
                reason: "too large".into(),
            }))
            .await
            .unwrap();
    });
    format!("http://{address}")
}

async fn serve_one_shot_websocket(
    events: impl IntoIterator<Item = Value>,
    close_timeout: Duration,
) -> (String, oneshot::Receiver<bool>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let events = events.into_iter().collect::<Vec<_>>();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
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
        let closed = matches!(
            tokio::time::timeout(close_timeout, socket.next()).await,
            Ok(Some(Ok(WebSocketMessage::Close(_))))
        );
        sender.send(closed).ok();
    });
    (format!("http://{address}"), receiver)
}

async fn serve_websocket_age_limit(
    first_connection_requests: usize,
) -> (String, oneshot::Receiver<usize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        for index in 0..first_connection_requests {
            let request = socket.next().await;
            assert!(
                matches!(request, Some(Ok(WebSocketMessage::Text(_)))),
                "unexpected request {index}: {request:?}"
            );
            for event in text_events(
                &format!("resp_age_{index}"),
                &format!("msg_age_{index}"),
                "Done",
            ) {
                socket
                    .send(WebSocketMessage::Text(event.to_string().into()))
                    .await
                    .unwrap();
            }
        }
        assert!(matches!(
            socket.next().await,
            Some(Ok(WebSocketMessage::Close(_)))
        ));

        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        assert!(matches!(
            socket.next().await,
            Some(Ok(WebSocketMessage::Text(_)))
        ));
        for event in text_events("resp_age_fresh", "msg_age_fresh", "Fresh") {
            socket
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        sender.send(2).ok();
    });
    (format!("http://{address}"), receiver)
}

async fn serve_failed_then_fresh_websockets() -> (String, oneshot::Receiver<bool>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        assert!(matches!(
            socket.next().await,
            Some(Ok(WebSocketMessage::Text(_)))
        ));
        socket
            .send(WebSocketMessage::Text(
                json!({"type": "response.created", "response": {"id": "resp_failed"}})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        let evicted = matches!(
            tokio::time::timeout(Duration::from_secs(1), socket.next()).await,
            Ok(Some(Ok(WebSocketMessage::Close(_))))
        );
        if !evicted {
            sender.send(false).ok();
            return;
        }
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        assert!(matches!(
            socket.next().await,
            Some(Ok(WebSocketMessage::Text(_)))
        ));
        for event in text_events("resp_fresh", "msg_fresh", "Fresh") {
            socket
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        sender.send(true).ok();
    });
    (format!("http://{address}"), receiver)
}

async fn serve_missing_continuation_websockets() -> (String, oneshot::Receiver<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        let mut bodies = Vec::new();
        bodies.push(match socket.next().await {
            Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
            message => panic!("unexpected websocket request: {message:?}"),
        });
        for event in text_events("resp_seed", "msg_seed", "Seed") {
            socket
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        bodies.push(match socket.next().await {
            Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
            message => panic!("unexpected websocket request: {message:?}"),
        });
        socket
            .send(WebSocketMessage::Text(
                json!({
                    "type": "error",
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "Continuation expired"
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        assert!(matches!(
            socket.next().await,
            Some(Ok(WebSocketMessage::Close(_)))
        ));

        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        bodies.push(match socket.next().await {
            Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
            message => panic!("unexpected websocket request: {message:?}"),
        });
        for event in text_events("resp_recovered", "msg_recovered", "Recovered") {
            socket
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        sender.send(bodies).ok();
    });
    (format!("http://{address}"), receiver)
}

struct MissingContinuationSseCapture {
    websocket_bodies: Vec<Value>,
    http_requests: usize,
}

async fn serve_missing_continuation_then_sse()
-> (String, oneshot::Receiver<MissingContinuationSseCapture>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        let mut websocket_bodies = Vec::new();
        websocket_bodies.push(match socket.next().await {
            Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
            message => panic!("unexpected websocket request: {message:?}"),
        });
        for event in text_events("resp_seed_sse", "msg_seed_sse", "Seed") {
            socket
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        websocket_bodies.push(match socket.next().await {
            Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
            message => panic!("unexpected websocket request: {message:?}"),
        });
        socket
            .send(WebSocketMessage::Text(
                json!({
                    "type": "error",
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "Continuation expired"
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        assert!(matches!(
            socket.next().await,
            Some(Ok(WebSocketMessage::Close(_)))
        ));

        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        websocket_bodies.push(match socket.next().await {
            Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
            message => panic!("unexpected websocket request: {message:?}"),
        });
        socket.close(None).await.unwrap();

        let (mut socket, _) = listener.accept().await.unwrap();
        read_http_request(&mut socket).await;
        let sse = sse_text_events("resp_recovery_sse", "msg_recovery_sse", "Fallback");
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                    sse.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        sender
            .send(MissingContinuationSseCapture {
                websocket_bodies,
                http_requests: 1,
            })
            .ok();
    });
    (format!("http://{address}"), receiver)
}

async fn serve_session_isolation_websockets() -> (String, oneshot::Receiver<bool>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut first = tokio_tungstenite::accept_async(socket).await.unwrap();
        assert!(matches!(
            first.next().await,
            Some(Ok(WebSocketMessage::Text(_)))
        ));
        for event in text_events("resp_session_first", "msg_session_first", "First") {
            first
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        tokio::select! {
            message = first.next() => {
                assert!(matches!(message, Some(Ok(WebSocketMessage::Text(_)))));
                for event in text_events("resp_session_second", "msg_session_second", "Second") {
                    first
                        .send(WebSocketMessage::Text(event.to_string().into()))
                        .await
                        .unwrap();
                }
                sender.send(true).ok();
            }
            accepted = listener.accept() => {
                let (socket, _) = accepted.unwrap();
                let mut second = tokio_tungstenite::accept_async(socket).await.unwrap();
                assert!(matches!(
                    second.next().await,
                    Some(Ok(WebSocketMessage::Text(_)))
                ));
                for event in text_events("resp_session_second", "msg_session_second", "Second") {
                    second
                        .send(WebSocketMessage::Text(event.to_string().into()))
                        .await
                        .unwrap();
                }
                sender.send(false).ok();
            }
        }
    });
    (format!("http://{address}"), receiver)
}

async fn serve_concurrent_session_websockets()
-> (String, oneshot::Receiver<()>, oneshot::Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (first_sender, first_receiver) = oneshot::channel();
    let (second_sender, second_receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut first = tokio_tungstenite::accept_async(socket).await.unwrap();
        assert!(matches!(
            first.next().await,
            Some(Ok(WebSocketMessage::Text(_)))
        ));
        first_sender.send(()).ok();

        let (socket, _) = listener.accept().await.unwrap();
        let mut second = tokio_tungstenite::accept_async(socket).await.unwrap();
        assert!(matches!(
            second.next().await,
            Some(Ok(WebSocketMessage::Text(_)))
        ));
        second_sender.send(()).ok();
        for event in text_events("resp_busy_second", "msg_busy_second", "Second") {
            second
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        for event in text_events("resp_busy_first", "msg_busy_first", "First") {
            first
                .send(WebSocketMessage::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        assert!(matches!(
            second.next().await,
            Some(Ok(WebSocketMessage::Close(_)))
        ));
    });
    (format!("http://{address}"), first_receiver, second_receiver)
}

async fn serve_idle_websocket_then_sse() -> (String, oneshot::Receiver<Vec<Vec<u8>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
        let websocket_request = match websocket.next().await {
            Some(Ok(WebSocketMessage::Text(_))) => b"GET /codex/responses HTTP/1.1\r\n".to_vec(),
            message => panic!("unexpected websocket request: {message:?}"),
        };
        let (mut socket, _) = listener.accept().await.unwrap();
        let http_request = read_http_request(&mut socket).await;
        let sse = [
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_idle_fallback\",\"type\":\"message\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Fallback\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_idle_fallback\",\"usage\":{}}}\n\n",
        ]
        .concat();
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                    sse.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        sender.send(vec![websocket_request, http_request]).ok();
        drop(websocket);
    });
    (format!("http://{address}"), receiver)
}

async fn serve_stalled_websocket_handshake_then_sse() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stalled, _) = listener.accept().await.unwrap();
        read_http_request(&mut stalled).await;

        let (mut socket, _) = listener.accept().await.unwrap();
        read_http_request(&mut socket).await;
        let sse = sse_text_events("resp_connect_timeout", "msg_connect_timeout", "Fallback");
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                    sse.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    });
    format!("http://{address}")
}

async fn serve_started_idle_websocket() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        assert!(matches!(
            socket.next().await,
            Some(Ok(WebSocketMessage::Text(_)))
        ));
        socket
            .send(WebSocketMessage::Text(
                json!({"type": "response.created", "response": {"id": "resp_started_idle"}})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        socket.next().await;
    });
    format!("http://{address}")
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let header_end = loop {
        let mut bytes = [0; 1024];
        let count = socket.read(&mut bytes).await.unwrap();
        request.extend_from_slice(&bytes[..count]);
        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or_default();
    while request.len() < header_end + content_length {
        let mut bytes = [0; 1024];
        let count = socket.read(&mut bytes).await.unwrap();
        request.extend_from_slice(&bytes[..count]);
    }
    request
}

async fn serve_websocket_connections(
    event_batches: impl IntoIterator<Item = Vec<Value>>,
) -> (String, oneshot::Receiver<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let event_batches = event_batches.into_iter().collect::<Vec<_>>();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let mut bodies = Vec::new();
        for events in event_batches {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let body = match socket.next().await {
                Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
                message => panic!("unexpected websocket request: {message:?}"),
            };
            bodies.push(body);
            for event in events {
                socket
                    .send(WebSocketMessage::Text(event.to_string().into()))
                    .await
                    .unwrap();
            }
            socket.close(None).await.unwrap();
        }
        sender.send(bodies).ok();
    });
    (format!("http://{address}"), receiver)
}

async fn serve_repeated_limit_then_sse() -> (String, oneshot::Receiver<usize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        for _ in 0..2 {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
            assert!(matches!(
                socket.next().await,
                Some(Ok(WebSocketMessage::Text(_)))
            ));
            socket
                .send(WebSocketMessage::Text(
                    json!({
                        "type": "error",
                        "error": {
                            "code": "websocket_connection_limit_reached",
                            "message": "Still full"
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            assert!(matches!(
                socket.next().await,
                Some(Ok(WebSocketMessage::Close(_)))
            ));
        }
        let (mut socket, _) = listener.accept().await.unwrap();
        read_http_request(&mut socket).await;
        let sse = sse_text_events("resp_limit_fallback", "msg_limit_fallback", "Fallback");
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                    sse.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        sender.send(3).ok();
    });
    (format!("http://{address}"), receiver)
}

async fn serve_kept_websocket_connections(
    event_batches: impl IntoIterator<Item = Vec<Value>>,
) -> (String, oneshot::Receiver<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let event_batches = event_batches.into_iter().collect::<Vec<_>>();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let mut bodies = Vec::new();
        let mut sockets = Vec::new();
        for events in event_batches {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let body = match socket.next().await {
                Some(Ok(WebSocketMessage::Text(body))) => serde_json::from_str(&body).unwrap(),
                message => panic!("unexpected websocket request: {message:?}"),
            };
            bodies.push(body);
            for event in events {
                socket
                    .send(WebSocketMessage::Text(event.to_string().into()))
                    .await
                    .unwrap();
            }
            sockets.push(socket);
        }
        sender.send(bodies).ok();
    });
    (format!("http://{address}"), receiver)
}

fn text_events(response_id: &str, message_id: &str, text: &str) -> Vec<Value> {
    vec![
        json!({"type": "response.created", "response": {"id": response_id}}),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": []
            }
        }),
        json!({"type": "response.output_text.delta", "output_index": 0, "delta": text}),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "phase": null
            }
        }),
        json!({
            "type": "response.done",
            "response": {"id": response_id, "status": "completed", "usage": {}}
        }),
    ]
}

fn sse_text_events(response_id: &str, message_id: &str, text: &str) -> String {
    text_events(response_id, message_id, text)
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

struct CaptureHandshake {
    capture: Arc<Mutex<Option<CapturedHandshake>>>,
}

impl Callback for CaptureHandshake {
    fn on_request(self, request: &Request, response: Response) -> Result<Response, ErrorResponse> {
        let headers = request
            .headers()
            .iter()
            .map(|(name, value)| (name.as_str().to_owned(), value.to_str().unwrap().to_owned()))
            .collect();
        *self.capture.lock().unwrap() = Some((request.uri().path().to_owned(), headers));
        Ok(response)
    }
}
