use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{Context, Event, Message, StopReason, codex};
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
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Say hello")]).with_system("Be brief");
    let options = codex::Options::new(token.clone())
        .with_max_retries(1)
        .with_session_id("session_1")
        .with_transport(codex::Transport::Sse);

    let events = codex::stream(&model, &context, &options)
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
    assert_eq!(response.id.as_deref(), Some("resp_codex"));
    assert_eq!(response.content, [ds_ai::Content::Text("Hello".into())]);
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
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let context = Context::new([Message::user("Connect")]).with_system("Be brief");
    let options = codex::Options::new(token.clone())
        .with_session_id("session_ws")
        .with_transport(codex::Transport::WebSocket);

    let events = codex::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    let response = done(&events);
    assert_eq!(response.content, [ds_ai::Content::Text("WebSocket".into())]);
    assert_eq!(response.service_tier.as_deref(), Some("priority"));
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
async fn encodes_codex_generation_options() {
    let (base_url, capture) =
        serve_websocket([text_events("resp_options", "msg_options", "Configured")]).await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let events = codex::stream(
        &model,
        &Context::new([Message::user("Configure")]),
        &codex::Options::new(token("acc_options"))
            .with_cache_retention(ds_ai::CacheRetention::None)
            .with_temperature(0.25)
            .with_reasoning(
                codex::ReasoningEffort::High,
                codex::ReasoningSummary::Concise,
            )
            .with_service_tier(codex::ServiceTier::Priority)
            .with_text_verbosity(codex::TextVerbosity::High)
            .with_tool_choice(codex::ToolChoice::Required)
            .with_transport(codex::Transport::WebSocket),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(
        done(&events).content,
        [ds_ai::Content::Text("Configured".into())]
    );
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
    let (base_url, capture) = serve_websocket([
        text_events("resp_first", "msg_first", "First"),
        text_events("resp_second", "msg_second", "Second"),
    ])
    .await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let options = codex::Options::new(token("acc_reuse"))
        .with_session_id("session_reuse")
        .with_transport(codex::Transport::WebSocket);
    let first_context = Context::new([Message::user("First")]).with_system("Be brief");
    let first_events = codex::stream(&model, &first_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let first_response = done(&first_events).clone();
    let second_context = Context::new([
        Message::user("First"),
        Message::assistant(first_response),
        Message::user("Continue"),
    ])
    .with_system("Be brief");

    let second_events = codex::stream(&model, &second_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        done(&second_events).content,
        [ds_ai::Content::Text("Second".into())]
    );
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
}

#[tokio::test]
async fn scopes_cached_codex_websockets_to_the_account() {
    let (base_url, capture) = serve_kept_websocket_connections([
        text_events("resp_account_one", "msg_account_one", "First"),
        text_events("resp_account_two", "msg_account_two", "Second"),
    ])
    .await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let context = Context::new([Message::user("Connect")]);
    let first = codex::stream(
        &model,
        &context,
        &codex::Options::new(token("acc_scope_one"))
            .with_session_id("session_scope")
            .with_transport(codex::Transport::WebSocket),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;
    let second = codex::stream(
        &model,
        &context,
        &codex::Options::new(token("acc_scope_two"))
            .with_session_id("session_scope")
            .with_transport(codex::Transport::WebSocket),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(done(&first).content, [ds_ai::Content::Text("First".into())]);
    assert_eq!(
        done(&second).content,
        [ds_ai::Content::Text("Second".into())]
    );
    assert_eq!(capture.await.unwrap().len(), 2);
}

#[tokio::test]
async fn retries_a_missing_codex_continuation_with_full_context() {
    let (base_url, capture) = serve_websocket([
        text_events("resp_seed", "msg_seed", "Seed"),
        vec![json!({
            "type": "error",
            "error": {
                "code": "previous_response_not_found",
                "message": "Continuation expired"
            }
        })],
        text_events("resp_recovered", "msg_recovered", "Recovered"),
    ])
    .await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let options = codex::Options::new(token("acc_recovery"))
        .with_session_id("session_recovery")
        .with_transport(codex::Transport::WebSocket);
    let first_events = codex::stream(
        &model,
        &Context::new([Message::user("Seed")]).with_system("Be brief"),
        &options,
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;
    let first_response = done(&first_events).clone();
    let second_context = Context::new([
        Message::user("Seed"),
        Message::assistant(first_response),
        Message::user("Continue"),
    ])
    .with_system("Be brief");

    let second_events = codex::stream(&model, &second_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        done(&second_events).content,
        [ds_ai::Content::Text("Recovered".into())]
    );
    let capture = capture.await.unwrap();
    assert_eq!(capture.bodies.len(), 3);
    assert_eq!(capture.bodies[1]["previous_response_id"], "resp_seed");
    assert!(capture.bodies[2].get("previous_response_id").is_none());
    assert_eq!(capture.bodies[2]["input"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn reconnects_once_when_the_codex_websocket_limit_is_reached() {
    let (base_url, capture) = serve_websocket_connections([
        vec![json!({
            "type": "error",
            "error": {"code": "websocket_connection_limit_reached"}
        })],
        text_events("resp_reconnected", "msg_reconnected", "Reconnected"),
    ])
    .await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let events = codex::stream(
        &model,
        &Context::new([Message::user("Connect")]),
        &codex::Options::new(token("acc_limit"))
            .with_session_id("session_limit")
            .with_transport(codex::Transport::WebSocket),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(
        done(&events).content,
        [ds_ai::Content::Text("Reconnected".into())]
    );
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
}

#[tokio::test]
async fn surfaces_a_second_codex_websocket_limit_error() {
    let (base_url, capture) = serve_websocket_connections([
        vec![json!({
            "type": "error",
            "error": {"code": "websocket_connection_limit_reached"}
        })],
        vec![json!({
            "type": "error",
            "error": {
                "code": "websocket_connection_limit_reached",
                "message": "Still full"
            }
        })],
    ])
    .await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let events = codex::stream(
        &model,
        &Context::new([Message::user("Connect")]),
        &codex::Options::new(token("acc_limit_twice"))
            .with_cache_retention(ds_ai::CacheRetention::None)
            .with_transport(codex::Transport::WebSocket),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert!(matches!(
        events.last(),
        Some(Err(ds_ai::Error::Response { code, message, partial }))
            if code.as_deref() == Some("websocket_connection_limit_reached")
                && message == "Still full"
                && partial.content.is_empty()
    ));
    assert_eq!(capture.await.unwrap().len(), 2);
}

#[tokio::test]
async fn replaces_a_stale_cached_codex_websocket() {
    let (base_url, capture) = serve_websocket_connections([
        text_events("resp_stale_seed", "msg_stale_seed", "First"),
        text_events("resp_stale_fresh", "msg_stale_fresh", "Second"),
    ])
    .await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let options = codex::Options::new(token("acc_stale"))
        .with_session_id("session_stale")
        .with_transport(codex::Transport::WebSocket);
    let context = Context::new([Message::user("Connect")]);
    let first = codex::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(done(&first).content, [ds_ai::Content::Text("First".into())]);

    let second = codex::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        done(&second).content,
        [ds_ai::Content::Text("Second".into())]
    );
    assert_eq!(capture.await.unwrap().len(), 2);
}

#[tokio::test]
async fn expires_an_idle_cached_codex_websocket() {
    let (base_url, closed) =
        serve_one_shot_websocket(text_events("resp_idle", "msg_idle", "Done")).await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let options = codex::Options::new(token("acc_idle"))
        .with_session_id("session_idle")
        .with_websocket_cache_ttl(Duration::from_millis(10))
        .with_transport(codex::Transport::WebSocket);
    let events = codex::stream(&model, &Context::new([Message::user("Connect")]), &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(done(&events).content, [ds_ai::Content::Text("Done".into())]);
    assert!(closed.await.unwrap());
}

#[tokio::test]
async fn evicts_a_cached_codex_websocket_after_stream_failure() {
    let (base_url, evicted) = serve_failed_then_fresh_websockets().await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let options = codex::Options::new(token("acc_failed_cache"))
        .with_session_id("session_failed_cache")
        .with_idle_timeout(Duration::from_millis(10))
        .with_transport(codex::Transport::WebSocket);
    let context = Context::new([Message::user("Connect")]);

    let failed = codex::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        failed.last(),
        Some(Err(ds_ai::Error::Timeout {
            phase: ds_ai::TimeoutPhase::Idle,
            ..
        }))
    ));

    let fresh = codex::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(done(&fresh).content, [ds_ai::Content::Text("Fresh".into())]);
    assert!(evicted.await.unwrap());
}

#[tokio::test]
async fn replaces_a_codex_websocket_at_the_connection_age_limit() {
    let (base_url, capture) = serve_kept_websocket_connections([
        text_events("resp_age_seed", "msg_age_seed", "First"),
        text_events("resp_age_fresh", "msg_age_fresh", "Second"),
    ])
    .await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let options = codex::Options::new(token("acc_age"))
        .with_session_id("session_age")
        .with_websocket_cache_ttl(Duration::from_secs(56 * 60))
        .with_transport(codex::Transport::WebSocket);
    let context = Context::new([Message::user("Connect")]);
    let first = codex::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(done(&first).content, [ds_ai::Content::Text("First".into())]);
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(55 * 60)).await;
    tokio::time::resume();

    let second = codex::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        done(&second).content,
        [ds_ai::Content::Text("Second".into())]
    );
    assert_eq!(capture.await.unwrap().len(), 2);
}

#[tokio::test]
async fn closes_a_one_shot_codex_websocket_after_completion() {
    let (base_url, closed) =
        serve_one_shot_websocket(text_events("resp_one_shot", "msg_one_shot", "Done")).await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let events = codex::stream(
        &model,
        &Context::new([Message::user("Connect")]),
        &codex::Options::new(token("acc_one_shot"))
            .with_cache_retention(ds_ai::CacheRetention::None)
            .with_transport(codex::Transport::WebSocket),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(done(&events).content, [ds_ai::Content::Text("Done".into())]);
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
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(&server.base_url);
    let events = codex::stream(
        &model,
        &Context::new([Message::user("Connect")]),
        &codex::Options::new(token("acc_fallback")),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(
        done(&events).content,
        [ds_ai::Content::Text("Fallback".into())]
    );
    let requests = server.request_bytes().await;
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with(b"GET /codex/responses HTTP/1.1\r\n"));
    assert!(requests[1].starts_with(b"POST /codex/responses HTTP/1.1\r\n"));
    let (_, body) = request(&requests[1]);
    let body: Value = serde_json::from_slice(&zstd::stream::decode_all(body).unwrap()).unwrap();
    assert_eq!(body["input"][0]["content"][0]["text"], "Connect");
}

#[tokio::test]
async fn falls_back_to_sse_when_codex_websocket_has_no_first_event() {
    let (base_url, capture) = serve_idle_websocket_then_sse().await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let events = codex::stream(
        &model,
        &Context::new([Message::user("Wait")]),
        &codex::Options::new(token("acc_idle_fallback"))
            .with_cache_retention(ds_ai::CacheRetention::None)
            .with_first_event_timeout(Duration::from_millis(10)),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(
        done(&events).content,
        [ds_ai::Content::Text("Fallback".into())]
    );
    let requests = capture.await.unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with(b"GET /codex/responses HTTP/1.1\r\n"));
    assert!(requests[1].starts_with(b"POST /codex/responses HTTP/1.1\r\n"));
}

#[tokio::test]
async fn does_not_fall_back_after_a_codex_websocket_event() {
    let base_url = serve_started_idle_websocket().await;
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let events = codex::stream(
        &model,
        &Context::new([Message::user("Wait")]),
        &codex::Options::new(token("acc_started_idle"))
            .with_cache_retention(ds_ai::CacheRetention::None)
            .with_first_event_timeout(Duration::from_millis(10))
            .with_idle_timeout(Duration::from_millis(10))
            .with_overall_timeout(Duration::from_millis(100)),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert!(matches!(
        events.last(),
        Some(Err(ds_ai::Error::Timeout {
            phase: ds_ai::TimeoutPhase::Idle,
            partial: Some(partial),
        })) if partial.id.as_deref() == Some("resp_started_idle")
    ));
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
    let model = codex::Model::new("gpt-5.6-codex").with_base_url(base_url);
    let events = codex::stream(
        &model,
        &Context::new([Message::user("Large response")]),
        &codex::Options::new(token("acc_large"))
            .with_cache_retention(ds_ai::CacheRetention::None)
            .with_transport(codex::Transport::WebSocket),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(
        events.first(),
        Some(&Ok(Event::TextDelta {
            content_index: 0,
            delta: "Partial".into(),
        }))
    );
    assert!(matches!(
        events.last(),
        Some(Err(ds_ai::Error::Stream { message, partial }))
            if message.contains("code 1009")
                && partial.content == [ds_ai::Content::Text("Partial".into())]
    ));
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

fn done(events: &[Result<Event, ds_ai::Error>]) -> &ds_ai::Response {
    match events.last() {
        Some(Ok(Event::Done(response))) => response,
        _ => panic!("stream did not complete"),
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
            tokio::time::timeout(Duration::from_secs(1), socket.next()).await,
            Ok(Some(Ok(WebSocketMessage::Close(_))))
        );
        sender.send(closed).ok();
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
