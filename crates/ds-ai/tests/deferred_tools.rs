use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{
    Api, AssistantContent, AssistantMessage, AssistantToolCall, Context, InputContent, Message,
    Model, ModelCompatibility, OpenAiResponsesCompatibility, Provider, ProviderId, StopReason,
    StreamOptions, Tool, ToolResultMessage, Transport, Usage, builtin_model,
};
use serde_json::{Value, json};

#[tokio::test]
async fn loads_an_openai_tool_through_additional_tools() {
    let payload = capture_openai(
        model("openai", "gpt-5.4"),
        context([tool("base_tool"), tool("late_tool")]),
    )
    .await;

    assert_eq!(tool_names(&payload), ["base_tool"]);
    let marker = item(&payload, "additional_tools");
    assert_eq!(marker["role"], "developer");
    assert_eq!(marker["tools"][0]["name"], "late_tool");
    assert!(item_opt(&payload, "tool_search_call").is_none());
}

#[tokio::test]
async fn falls_back_to_client_tool_search() {
    let mut model = model("openai", "gpt-5.4");
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_additional_tools: Some(false),
        supports_tool_search: Some(true),
        ..Default::default()
    }));
    let payload = capture_openai(model, context([tool("base_tool"), tool("late_tool")])).await;

    assert_eq!(tool_names(&payload), ["base_tool"]);
    let call = item(&payload, "tool_search_call");
    let output = item(&payload, "tool_search_output");
    assert_eq!(call["execution"], "client");
    assert_eq!(call["status"], "completed");
    assert_eq!(output["call_id"], call["call_id"]);
    assert_eq!(output["tools"][0]["name"], "late_tool");
    assert_eq!(output["tools"][0]["defer_loading"], true);
    assert!(item_opt(&payload, "additional_tools").is_none());
}

#[tokio::test]
async fn keeps_all_openai_tools_immediate_without_support() {
    let mut model = model("openai", "gpt-5.4");
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_additional_tools: Some(false),
        supports_tool_search: Some(false),
        ..Default::default()
    }));
    let payload = capture_openai(model, context([tool("base_tool"), tool("late_tool")])).await;

    assert_eq!(tool_names(&payload), ["base_tool", "late_tool"]);
    assert!(item_opt(&payload, "additional_tools").is_none());
    assert!(item_opt(&payload, "tool_search_output").is_none());
}

#[tokio::test]
async fn loads_an_anthropic_tool_at_its_result_marker() {
    let payload = capture_anthropic(
        model("anthropic", "claude-opus-4-6"),
        context([tool("base_tool"), tool("late_tool")]),
    )
    .await;

    assert_eq!(payload["tools"][0]["name"], "base_tool");
    assert_eq!(payload["tools"][1]["name"], "late_tool");
    assert_eq!(payload["tools"][1]["defer_loading"], true);
    let blocks = anthropic_tool_result_blocks(&payload);
    assert_eq!(blocks[0]["content"][0]["type"], "tool_reference");
    assert_eq!(blocks[0]["content"][0]["tool_name"], "late_tool");
    assert_eq!(blocks[1], json!({"type": "text", "text": "done"}));
}

#[tokio::test]
async fn keeps_a_previously_used_anthropic_tool_immediate() {
    let mut context = context([tool("base_tool"), tool("late_tool")]);
    let Message::Assistant(message) = &mut context.messages[1] else {
        panic!("expected assistant message");
    };
    let AssistantContent::ToolCall(call) = &mut message.content[0] else {
        panic!("expected tool call");
    };
    call.name = "late_tool".into();

    let payload = capture_anthropic(model("anthropic", "claude-opus-4-6"), context).await;

    assert_eq!(payload["tools"][0]["name"], "base_tool");
    assert_eq!(payload["tools"][1]["name"], "late_tool");
    assert!(
        payload["tools"]
            .as_array()
            .unwrap()
            .iter()
            .all(|tool| tool.get("defer_loading").is_none())
    );
    assert!(
        anthropic_tool_result_blocks(&payload)[0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block["type"] != "tool_reference")
    );
}

#[tokio::test]
async fn keeps_one_anthropic_tool_immediate_when_all_are_marked() {
    let payload = capture_anthropic(
        model("anthropic", "claude-opus-4-6"),
        context([tool("late_tool")]),
    )
    .await;

    assert_eq!(payload["tools"].as_array().unwrap().len(), 1);
    assert_eq!(payload["tools"][0]["name"], "late_tool");
    assert!(payload["tools"][0].get("defer_loading").is_none());
    assert!(
        anthropic_tool_result_blocks(&payload)[0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block["type"] != "tool_reference")
    );
}

#[tokio::test]
async fn selects_each_codex_tool_placement_mode() {
    let context = context([tool("base_tool"), tool("late_tool")]);
    let additional = capture_codex(model("openai-codex", "gpt-5.6-sol"), context.clone()).await;
    let search = capture_codex(model("openai-codex", "gpt-5.4"), context.clone()).await;
    let immediate = capture_codex(model("openai-codex", "gpt-5.3-codex-spark"), context).await;

    assert_eq!(tool_names(&additional), ["base_tool"]);
    assert!(item_opt(&additional, "additional_tools").is_some());
    assert_eq!(tool_names(&search), ["base_tool"]);
    assert!(item_opt(&search, "tool_search_output").is_some());
    assert_eq!(tool_names(&immediate), ["base_tool", "late_tool"]);
    assert!(item_opt(&immediate, "additional_tools").is_none());
    assert!(item_opt(&immediate, "tool_search_output").is_none());
}

fn tool(name: &str) -> Tool {
    Tool::new(
        name,
        format!("The {name} tool"),
        json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        }),
    )
}

fn context(tools: impl IntoIterator<Item = Tool>) -> Context {
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(AssistantToolCall {
            id: "call_1".into(),
            name: "base_tool".into(),
            arguments: json!({}),
            thought_signature: None,
            namespace: None,
        })],
        api: Api::AnthropicMessages,
        provider: ProviderId::new("anthropic"),
        model: "claude-opus-4-6".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    };
    let mut result = ToolResultMessage::new("call_1", "base_tool", [InputContent::text("done")]);
    result.added_tool_names = Some(vec!["late_tool".into()]);
    result.timestamp = 3;
    Context::new([
        Message::user("Hello"),
        Message::assistant(assistant),
        Message::tool_result(result),
        Message::user("Continue"),
    ])
    .with_tools(tools)
}

fn model(provider: &str, id: &str) -> Model {
    builtin_model(provider, id).unwrap()
}

async fn capture_openai(mut model: Model, context: Context) -> Value {
    let server = serve([Reply::sse(openai_done())]).await;
    model.base_url = server.base_url.clone();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    provider
        .stream(
            &model,
            &context,
            &StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
        )
        .result()
        .await
        .unwrap();
    request_json(&server.requests().await[0])
}

async fn capture_anthropic(mut model: Model, context: Context) -> Value {
    let server = serve([Reply::sse(anthropic_done())]).await;
    model.base_url = server.base_url.clone();
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);
    provider
        .stream(
            &model,
            &context,
            &StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
        )
        .result()
        .await
        .unwrap();
    request_json(&server.requests().await[0])
}

async fn capture_codex(mut model: Model, context: Context) -> Value {
    let server = serve([Reply::sse(openai_done())]).await;
    model.base_url = server.base_url.clone();
    let provider = ds_ai::codex::Provider::new([model.clone()]);
    provider
        .stream(
            &model,
            &context,
            &StreamOptions {
                api_key: Some(token()),
                transport: Some(Transport::Sse),
                ..Default::default()
            },
        )
        .result()
        .await
        .unwrap();
    let request = &server.request_bytes().await[0];
    let split = request
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap()
        + 4;
    serde_json::from_slice(&zstd::stream::decode_all(&request[split..]).unwrap()).unwrap()
}

fn request_json(request: &str) -> Value {
    serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

fn tool_names(payload: &Value) -> Vec<&str> {
    payload["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect()
}

fn item_opt<'a>(payload: &'a Value, kind: &str) -> Option<&'a Value> {
    payload["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == kind)
}

fn item<'a>(payload: &'a Value, kind: &str) -> &'a Value {
    item_opt(payload, kind).unwrap()
}

fn anthropic_tool_result_blocks(payload: &Value) -> &[Value] {
    payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|message| {
            message["content"]
                .as_array()
                .filter(|content| content.iter().any(|block| block["type"] == "tool_result"))
        })
        .unwrap()
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
