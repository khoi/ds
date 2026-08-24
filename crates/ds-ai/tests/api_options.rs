use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{
    AnthropicOptions, ApiStreamOptions, Context, Message, OpenAiCodexResponsesOptions,
    OpenAiResponsesOptions, Provider, StopReason, StreamOptions, Transport, builtin_model,
};
use serde_json::{Value, json};

#[tokio::test]
async fn routes_openai_specific_options() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let options = ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            temperature: Some(0.25),
            max_tokens: Some(4096),
            ..Default::default()
        },
        reasoning_effort: Some(ds_ai::openai::ReasoningEffort::High),
        reasoning_summary: Some(ds_ai::openai::ReasoningSummary::Detailed),
        service_tier: Some(ds_ai::openai::ServiceTier::Flex),
        tool_choice: Some(ds_ai::openai::ToolChoice::Required),
    });

    provider
        .stream(&model, &Context::new([Message::user("Hello")]), &options)
        .result()
        .await
        .unwrap();

    let request = server.requests().await.pop().unwrap();
    let payload = request_json(&request);
    assert_eq!(payload["max_output_tokens"], 4096);
    assert_eq!(payload["temperature"], 0.25);
    assert_eq!(
        payload["reasoning"],
        json!({"effort": "high", "summary": "detailed"})
    );
    assert_eq!(payload["service_tier"], "flex");
    assert_eq!(payload["tool_choice"], "required");
}

#[tokio::test]
async fn routes_anthropic_specific_options() {
    let server = serve([Reply::sse(anthropic_done())]).await;
    let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);
    let options = ApiStreamOptions::AnthropicMessages(AnthropicOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            temperature: Some(0.25),
            max_tokens: Some(4096),
            ..Default::default()
        },
        thinking_enabled: Some(true),
        thinking_budget_tokens: Some(2048),
        effort: None,
        thinking_display: Some(ds_ai::anthropic::ThinkingDisplay::Omitted),
        interleaved_thinking: Some(true),
        tool_choice: Some(ds_ai::anthropic::ToolChoice::Tool("search".into())),
    });

    provider
        .stream(&model, &Context::new([Message::user("Hello")]), &options)
        .result()
        .await
        .unwrap();

    let request = server.requests().await.pop().unwrap();
    assert!(request.contains("anthropic-beta: interleaved-thinking-2025-05-14\r\n"));
    let payload = request_json(&request);
    assert_eq!(payload["max_tokens"], 4096);
    assert_eq!(
        payload["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048, "display": "omitted"})
    );
    assert_eq!(
        payload["tool_choice"],
        json!({"type": "tool", "name": "search"})
    );
    assert!(payload.get("temperature").is_none());
}

#[tokio::test]
async fn routes_codex_specific_options() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::codex::Provider::new([model.clone()]);
    let options = ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
        stream: StreamOptions {
            api_key: Some(token()),
            temperature: Some(0.25),
            transport: Some(Transport::Sse),
            ..Default::default()
        },
        reasoning_effort: Some(ds_ai::codex::ReasoningEffort::High),
        reasoning_summary: Some(ds_ai::codex::ReasoningSummary::Detailed),
        service_tier: Some(ds_ai::codex::ServiceTier::Priority),
        text_verbosity: Some(ds_ai::codex::TextVerbosity::High),
        tool_choice: Some(ds_ai::codex::ToolChoice::Required),
    });

    provider
        .stream(&model, &Context::new([Message::user("Hello")]), &options)
        .result()
        .await
        .unwrap();

    let request = server.request_bytes().await.pop().unwrap();
    let split = request
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap()
        + 4;
    let payload: Value =
        serde_json::from_slice(&zstd::stream::decode_all(&request[split..]).unwrap()).unwrap();
    assert_eq!(payload["temperature"], 0.25);
    assert_eq!(
        payload["reasoning"],
        json!({"effort": "high", "summary": "detailed"})
    );
    assert_eq!(payload["service_tier"], "priority");
    assert_eq!(payload["text"], json!({"verbosity": "high"}));
    assert_eq!(payload["tool_choice"], "required");
}

#[tokio::test]
async fn rejects_options_for_a_different_api() {
    let model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let result = provider
        .stream(
            &model,
            &Context::new([]),
            &ApiStreamOptions::AnthropicMessages(Default::default()),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("invalid request: OpenAI Responses options are required")
    );
}

fn request_json(request: &str) -> Value {
    serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

fn openai_done() -> &'static str {
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n"
}

fn anthropic_done() -> &'static str {
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

fn token() -> String {
    let payload = BASE64_URL_SAFE_NO_PAD.encode(
        json!({"https://api.openai.com/auth": {"chatgpt_account_id": "account"}}).to_string(),
    );
    format!("header.{payload}.signature")
}
