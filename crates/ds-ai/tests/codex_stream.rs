use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{Context, Event, Message, StopReason, codex};
use futures_util::StreamExt;
use serde_json::{Value, json};

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
        .with_session_id("session_1");

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
