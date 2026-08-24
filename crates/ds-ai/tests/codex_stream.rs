use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{Context, Event, Message, StopReason, codex};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tokio::{net::TcpListener, sync::oneshot};
use tokio_tungstenite::{
    accept_hdr_async,
    tungstenite::{
        Message as WebSocketMessage,
        handshake::server::{Callback, ErrorResponse, Request, Response},
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
            "response": {"id": "resp_ws", "status": "completed", "usage": {}}
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

    assert_eq!(
        done(&events).content,
        [ds_ai::Content::Text("WebSocket".into())]
    );
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
