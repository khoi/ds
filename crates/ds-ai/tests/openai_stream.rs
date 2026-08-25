use crate::support::{Reply, serve};
use ds_ai::{
    Api, AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantToolCall,
    CacheRetention, Context, ErrorReason, InputContent, Message, ModelCompatibility,
    OpenAiResponsesCompatibility, OpenAiResponsesOptions, ProviderId, ResponseHook,
    SimpleStreamOptions, StopReason, StreamOptions, TextContent, ThinkingContent, Tool,
    ToolResultMessage, Usage, builtin_model, openai,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn streams_openai_text_until_the_provider_completes() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\",\"annotations\":[]}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"service_tier\":\"flex\",\"output\":[],\"usage\":{\"input_tokens\":4,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":5}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

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
    assert_eq!(response.response_id.as_deref(), Some("resp_1"));
    assert_eq!(
        response.content,
        [text("Hello", Some(("msg_1", Some("final_answer"))))]
    );
    assert_eq!(response.usage.input, 4);
    assert_eq!(response.usage.output, 1);
    assert_eq!(response.usage.cache_read, 0);
    assert_eq!(response.usage.reasoning, Some(0));
    assert_eq!(response.usage.total_tokens, 5);

    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /responses HTTP/1.1\r\n"));
    assert!(request.contains("authorization: Bearer test-key\r\n"));
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "gpt-5.6",
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            }],
            "stream": true,
            "store": false,
            "reasoning": {"effort": "none"}
        })
    );
}

#[tokio::test]
async fn accepts_openai_header_auth_and_merges_model_headers() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_headers\",\"usage\":{}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let mut model = model(&server.base_url);
    model
        .headers
        .insert("x-model-header".into(), "model".into());
    let mut options = options(|_| {});
    options.stream.api_key = None;
    options.stream.headers = [
        ("Authorization".into(), Some("Bearer gateway-token".into())),
        ("x-request-header".into(), Some("request".into())),
    ]
    .into();

    done(&events(&model, &Context::new([Message::user("Hello")]), &options).await);

    let request = server.requests().await.pop().unwrap().to_ascii_lowercase();
    assert!(request.contains("authorization: bearer gateway-token\r\n"));
    assert!(request.contains("x-model-header: model\r\n"));
    assert!(request.contains("x-request-header: request\r\n"));
}

#[tokio::test]
async fn requires_direct_openai_auth_from_options_or_caller_headers() {
    let server = serve([Reply::sse(Vec::new())]).await;
    let mut model = model(&server.base_url);
    model
        .headers
        .insert("Authorization".into(), "Bearer model-token".into());
    let mut options = options(|_| {});
    options.stream.api_key = None;

    let events = events(&model, &Context::new([Message::user("Hello")]), &options).await;
    let error = failed(&events);

    assert_eq!(
        error.error_message.as_deref(),
        Some("No API key for provider: openai")
    );
    assert_eq!(server.request_count(), 0);
}

#[tokio::test]
async fn preserves_openai_custom_tool_namespaces() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_1\",\"call_id\":\"call_1\",\"name\":\"query\",\"input\":\"\"}}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"input\":\"abc\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_1\",\"call_id\":\"call_1\",\"name\":\"query\",\"input\":\"abc\",\"namespace\":\"custom\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_custom\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let result_events = events(
        &model,
        &Context::new([Message::user("Query")]),
        &options(|_| {}),
    )
    .await;
    let result = done(&result_events);

    let [AssistantContent::ToolCall(call)] = result.content.as_slice() else {
        panic!("expected one custom tool call");
    };
    assert_eq!(call.id, "call_1|ctc_1");
    assert_eq!(call.name, "query");
    assert_eq!(call.arguments, json!({"input": "abc"}));
    assert_eq!(call.namespace.as_deref(), Some("custom"));
    server.request_bytes().await;
}

#[tokio::test]
async fn preserves_function_arguments_from_the_added_item() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_prefix\",\"call_id\":\"call_prefix\",\"name\":\"lookup\",\"arguments\":\"{\\\"city\\\":\\\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"Paris\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_prefix\",\"call_id\":\"call_prefix\",\"name\":\"lookup\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_prefix\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;

    let result_events = events(
        &model(&server.base_url),
        &Context::new([Message::user("Look up")]),
        &options(|_| {}),
    )
    .await;

    let arguments = result_events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolCallStart { partial, .. }
            | AssistantMessageEvent::ToolCallDelta { partial, .. } => {
                let AssistantContent::ToolCall(call) = &partial.content[0] else {
                    panic!("expected tool call partial");
                };
                Some(call.arguments.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(arguments, [json!({}), json!({"city": "Paris"})]);

    let result = done(&result_events);
    let [AssistantContent::ToolCall(call)] = result.content.as_slice() else {
        panic!("expected one tool call");
    };
    assert_eq!(call.arguments, json!({"city": "Paris"}));
    server.requests().await;
}

#[tokio::test]
async fn emits_custom_tool_input_from_the_added_item() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_prefix\",\"call_id\":\"call_prefix\",\"name\":\"query\",\"input\":\"ab\"}}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.delta\",\"output_index\":0,\"delta\":\"c\"}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"input\":\"abc\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_prefix\",\"call_id\":\"call_prefix\",\"name\":\"query\",\"input\":\"abc\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_prefix\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;

    let result_events = events(
        &model(&server.base_url),
        &Context::new([Message::user("Query")]),
        &options(|_| {}),
    )
    .await;

    let deltas = result_events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(deltas, ["{\"input\":\"ab", "c", "\"}"]);
    let [AssistantContent::ToolCall(call)] = done(&result_events).content.as_slice() else {
        panic!("expected one tool call");
    };
    assert_eq!(call.arguments, json!({"input": "abc"}));
    server.requests().await;
}

#[tokio::test]
async fn keeps_streamed_reasoning_when_the_terminal_item_is_empty() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_empty\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Keep this\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_empty\",\"summary\":[],\"content\":[]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_reasoning\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;

    let result_events = events(
        &model(&server.base_url),
        &Context::new([Message::user("Think")]),
        &options(|_| {}),
    )
    .await;
    let result = done(&result_events);

    let [AssistantContent::Thinking(thinking)] = result.content.as_slice() else {
        panic!("expected one reasoning block");
    };
    assert_eq!(thinking.thinking, "Keep this");
    server.requests().await;
}

#[tokio::test]
async fn keeps_streamed_function_arguments_when_the_terminal_item_is_empty() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_empty\",\"call_id\":\"call_empty\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_empty\",\"call_id\":\"call_empty\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_arguments\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;

    let result_events = events(
        &model(&server.base_url),
        &Context::new([Message::user("Look up")]),
        &options(|_| {}),
    )
    .await;
    let result = done(&result_events);

    let [AssistantContent::ToolCall(call)] = result.content.as_slice() else {
        panic!("expected one tool call");
    };
    assert_eq!(call.arguments, json!({"city": "Paris"}));
    server.requests().await;
}

#[tokio::test]
async fn keeps_tool_namespaces_when_the_terminal_items_omit_them() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_namespace\",\"call_id\":\"call_function\",\"name\":\"lookup\",\"arguments\":\"{}\",\"namespace\":\"functions\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_namespace\",\"call_id\":\"call_function\",\"name\":\"lookup\",\"arguments\":\"{}\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_namespace\",\"call_id\":\"call_custom\",\"name\":\"query\",\"input\":\"value\",\"namespace\":\"custom\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_namespace\",\"call_id\":\"call_custom\",\"name\":\"query\",\"input\":\"value\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_namespaces\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;

    let result_events = events(
        &model(&server.base_url),
        &Context::new([Message::user("Use tools")]),
        &options(|_| {}),
    )
    .await;
    let result = done(&result_events);

    let namespaces = result
        .content
        .iter()
        .map(|content| match content {
            AssistantContent::ToolCall(call) => call.namespace.as_deref(),
            _ => panic!("expected tool calls"),
        })
        .collect::<Vec<_>>();
    assert_eq!(namespaces, [Some("functions"), Some("custom")]);
    server.requests().await;
}

#[tokio::test]
async fn keeps_tool_identity_from_the_added_items() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_added\",\"call_id\":\"call_added\",\"name\":\"lookup\",\"arguments\":\"{}\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_terminal\",\"call_id\":\"call_terminal\",\"name\":\"replace\",\"arguments\":\"{}\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_added\",\"call_id\":\"custom_added\",\"name\":\"query\",\"input\":\"value\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_terminal\",\"call_id\":\"custom_terminal\",\"name\":\"replace\",\"input\":\"value\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_identity\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;

    let result_events = events(
        &model(&server.base_url),
        &Context::new([Message::user("Use tools")]),
        &options(|_| {}),
    )
    .await;
    let result = done(&result_events);

    let calls = result
        .content
        .iter()
        .map(|content| match content {
            AssistantContent::ToolCall(call) => (call.id.as_str(), call.name.as_str()),
            _ => panic!("expected tool calls"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        calls,
        [
            ("call_added|fc_added", "lookup"),
            ("custom_added|ctc_added", "query"),
        ]
    );
    server.requests().await;
}

#[tokio::test]
async fn leaves_reasoning_usage_absent_when_terminal_usage_is_missing() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_without_usage\"}}\n\n",
    )])
    .await;

    let result_events = events(
        &model(&server.base_url),
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;
    let result = done(&result_events);

    assert_eq!(result.usage, Usage::default());
    server.requests().await;
}

#[tokio::test]
async fn retains_an_empty_created_response_id() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;

    let result_events = events(
        &model(&server.base_url),
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;
    let result = done(&result_events);

    assert_eq!(result.response_id.as_deref(), Some(""));
    server.requests().await;
}

#[tokio::test]
async fn validates_openai_text_signatures_for_replay() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_signatures\",\"usage\":{}}}\n\n",
    )])
    .await;
    let replay = AssistantMessage {
        content: vec![
            text("Fallback", Some(("", Some("final_answer")))),
            text("No phase", Some(("msg_unknown", Some("unknown")))),
        ],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: "gpt-5.6".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    };

    done(
        &events(
            &model(&server.base_url),
            &Context::new([
                Message::user("Hello"),
                Message::assistant(replay),
                Message::user("Continue"),
            ]),
            &options(|_| {}),
        )
        .await,
    );

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["input"][1],
        json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Fallback", "annotations": []}],
            "status": "completed",
            "id": "msg_pi_1",
            "phase": "final_answer"
        })
    );
    assert_eq!(
        body["input"][2],
        json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "No phase", "annotations": []}],
            "status": "completed",
            "id": "msg_unknown"
        })
    );
}

#[tokio::test]
async fn uses_utf16_units_for_openai_message_signature_ids() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_utf16\",\"usage\":{}}}\n\n",
    )])
    .await;
    let retained_id = format!("{}é", "a".repeat(63));
    let hashed_id = format!("{}😀Z", "a".repeat(62));
    let replay = AssistantMessage {
        content: vec![
            text("Retained", Some((&retained_id, None))),
            text("Hashed", Some((&hashed_id, None))),
        ],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: "gpt-5.6".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    };

    done(
        &events(
            &model(&server.base_url),
            &Context::new([Message::user("Hello"), Message::assistant(replay)]),
            &options(|_| {}),
        )
        .await,
    );

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["input"][1]["id"], retained_id);
    assert_eq!(body["input"][2]["id"], "msg_s9g3b1spa9oz");
}

#[tokio::test]
async fn rejects_malformed_same_model_reasoning_signatures() {
    let replay = AssistantMessage {
        content: vec![AssistantContent::Thinking(ThinkingContent {
            thinking: "Reasoning".into(),
            thinking_signature: Some("{".into()),
            redacted: None,
        })],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: "gpt-5.6".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    };

    let result_events = events(
        &model("http://127.0.0.1:9"),
        &Context::new([Message::assistant(replay)]),
        &options(|_| {}),
    )
    .await;

    assert!(
        failed(&result_events)
            .error_message
            .as_deref()
            .is_some_and(|message| message.starts_with("invalid request: EOF while parsing"))
    );
}

#[tokio::test]
async fn indexes_fallback_message_ids_after_failed_turns_are_removed() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_index\",\"usage\":{}}}\n\n",
    )])
    .await;
    let assistant = |value: &str, stop_reason| AssistantMessage {
        content: vec![text(value, None)],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: "gpt-5.6".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    };
    let context = Context::new([
        Message::user("Hello"),
        Message::assistant(assistant("Discard", StopReason::Error)),
        Message::assistant(assistant("Keep", StopReason::Stop)),
        Message::user("Continue"),
    ]);

    done(&events(&model(&server.base_url), &context, &options(|_| {})).await);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    let kept = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "message" && item["role"] == "assistant")
        .unwrap();
    assert_eq!(kept["id"], "msg_pi_1");
}

#[test]
fn serializes_the_full_openai_tool_choice_union() {
    for (choice, expected) in [
        (openai::ToolChoice::Auto, json!("auto")),
        (openai::ToolChoice::None, json!("none")),
        (openai::ToolChoice::Required, json!("required")),
        (
            openai::ToolChoice::Function("lookup".into()),
            json!({"type": "function", "name": "lookup"}),
        ),
        (
            openai::ToolChoice::Custom("query".into()),
            json!({"type": "custom", "name": "query"}),
        ),
        (
            openai::ToolChoice::AllowedTools {
                mode: openai::AllowedToolsMode::Required,
                tools: vec![
                    json!({"type": "function", "name": "lookup"}),
                    json!({"type": "mcp", "server_label": "docs"}),
                ],
            },
            json!({
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "function", "name": "lookup"},
                    {"type": "mcp", "server_label": "docs"}
                ]
            }),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::FileSearch),
            json!({"type": "file_search"}),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::WebSearchPreview),
            json!({"type": "web_search_preview"}),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::Computer),
            json!({"type": "computer"}),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::ComputerUsePreview),
            json!({"type": "computer_use_preview"}),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::ComputerUse),
            json!({"type": "computer_use"}),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::WebSearchPreview20250311),
            json!({"type": "web_search_preview_2025_03_11"}),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::ImageGeneration),
            json!({"type": "image_generation"}),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::CodeInterpreter),
            json!({"type": "code_interpreter"}),
        ),
        (
            openai::ToolChoice::Hosted(openai::HostedTool::Mcp),
            json!({"type": "mcp"}),
        ),
        (
            openai::ToolChoice::Mcp {
                server_label: "docs".into(),
                name: None,
            },
            json!({"type": "mcp", "server_label": "docs"}),
        ),
        (
            openai::ToolChoice::Mcp {
                server_label: "docs".into(),
                name: Some(Some("search".into())),
            },
            json!({"type": "mcp", "server_label": "docs", "name": "search"}),
        ),
        (
            openai::ToolChoice::Mcp {
                server_label: "docs".into(),
                name: Some(None),
            },
            json!({"type": "mcp", "server_label": "docs", "name": null}),
        ),
        (
            openai::ToolChoice::ApplyPatch,
            json!({"type": "apply_patch"}),
        ),
        (openai::ToolChoice::Shell, json!({"type": "shell"})),
    ] {
        assert_eq!(serde_json::to_value(choice).unwrap(), expected);
    }
}

#[tokio::test]
async fn serializes_scale_and_explicit_null_openai_service_tiers() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tier\",\"usage\":{}}}\n\n";
    let server = serve([Reply::sse(completed), Reply::sse(completed)]).await;

    for tier in [openai::ServiceTier::Scale, openai::ServiceTier::Null] {
        done(
            &events(
                &model(&server.base_url),
                &Context::new([Message::user("Hello")]),
                &options(|options| options.service_tier = Some(tier)),
            )
            .await,
        );
    }

    let requests = server.requests().await;
    let bodies = requests
        .iter()
        .map(|request| {
            serde_json::from_str::<Value>(request.split("\r\n\r\n").nth(1).unwrap()).unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(bodies[0]["service_tier"], "scale");
    assert_eq!(bodies[1]["service_tier"], Value::Null);
}

#[tokio::test]
async fn omits_the_default_simple_openai_tool_choice() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_simple\",\"usage\":{}}}\n\n",
    )])
    .await;
    let stream = openai::provider().stream_simple(
        &model(&server.base_url),
        &Context::new([Message::user("Hello")]),
        &SimpleStreamOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let result_events = stream.collect::<Vec<_>>().await;
    done(&result_events);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert!(body.get("tool_choice").is_none());
}

#[tokio::test]
async fn hashes_deferred_tool_loads_from_the_original_tool_result_id() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_deferred\",\"usage\":{}}}\n\n",
    )])
    .await;
    let mut model = model(&server.base_url);
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_additional_tools: Some(false),
        supports_tool_search: Some(true),
        ..Default::default()
    }));
    let raw_id = "call_source|foreign_item";
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(AssistantToolCall {
            id: raw_id.into(),
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
    let mut result = ToolResultMessage::new(raw_id, "base_tool", [InputContent::text("done")]);
    result.added_tool_names = Some(vec!["late_tool".into()]);
    let context = Context::new([
        Message::user("Hello"),
        Message::assistant(assistant),
        Message::tool_result(result),
        Message::user("Continue"),
    ])
    .with_tools([
        Tool::new("base_tool", "Base tool", json!({"type": "object"})),
        Tool::new("late_tool", "Late tool", json!({"type": "object"})),
    ]);

    done(&events(&model, &context, &options(|_| {})).await);

    let body: Value = serde_json::from_str(
        server
            .requests()
            .await
            .pop()
            .unwrap()
            .split("\r\n\r\n")
            .nth(1)
            .unwrap(),
    )
    .unwrap();
    let call = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "tool_search_call")
        .unwrap();
    assert_eq!(call["call_id"], "pi_tool_load_qq9zvz1smp2zs");
}

#[tokio::test]
async fn uses_only_the_first_two_pipe_fields_for_long_foreign_openai_tool_item_ids() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_foreign\",\"usage\":{}}}\n\n",
    )])
    .await;
    let model = model(&server.base_url);
    let raw_id = "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA==|ignored|also_ignored";
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(AssistantToolCall {
            id: raw_id.into(),
            name: "edit".into(),
            arguments: json!({"path": "src/styles/app.css"}),
            thought_signature: None,
            namespace: None,
        })],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("github-copilot"),
        model: "gpt-5.5".into(),
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
    let context = Context::new([
        Message::user("Use the tool."),
        Message::assistant(assistant),
        Message::tool_result(ToolResultMessage::new(
            raw_id,
            "edit",
            [InputContent::text("ok")],
        )),
        Message::user("Continue"),
    ]);

    done(&events(&model, &context, &options(|_| {})).await);

    let body: Value = serde_json::from_str(
        server
            .requests()
            .await
            .pop()
            .unwrap()
            .split("\r\n\r\n")
            .nth(1)
            .unwrap(),
    )
    .unwrap();
    let function_call = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call")
        .unwrap();
    assert_eq!(function_call["id"], "fc_ifd2c719fz6a9");
    assert_eq!(function_call["id"].as_str().unwrap().len(), 16);
    assert_eq!(function_call["call_id"], "call_4VnzVawQXPB9MgYib7CiQFEY");
}

#[tokio::test]
async fn uses_only_the_first_two_pipe_fields_for_same_provider_openai_tool_item_ids() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_same_model\",\"usage\":{}}}\n\n",
    )])
    .await;
    let model = model(&server.base_url);
    let assistant = |id: &str, model: &str, name: &str| AssistantMessage {
        content: vec![AssistantContent::ToolCall(AssistantToolCall {
            id: id.into(),
            name: name.into(),
            arguments: json!({}),
            thought_signature: None,
            namespace: None,
        })],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: model.into(),
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
    let same_model_id = "call_same|fc_same|ignored|also_ignored";
    let different_model_id = "call_other|item_other|ignored|also_ignored";
    let context = Context::new([
        Message::user("Use the tool."),
        Message::assistant(assistant(same_model_id, "gpt-5.6", "edit")),
        Message::tool_result(ToolResultMessage::new(
            same_model_id,
            "edit",
            [InputContent::text("ok")],
        )),
        Message::assistant(assistant(different_model_id, "gpt-5.5", "lookup")),
        Message::tool_result(ToolResultMessage::new(
            different_model_id,
            "lookup",
            [InputContent::text("found")],
        )),
        Message::user("Continue"),
    ]);

    done(&events(&model, &context, &options(|_| {})).await);

    let body: Value = serde_json::from_str(
        server
            .requests()
            .await
            .pop()
            .unwrap()
            .split("\r\n\r\n")
            .nth(1)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        body["input"],
        json!([
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Use the tool."}]
            },
            {
                "type": "function_call",
                "id": "fc_same",
                "call_id": "call_same",
                "name": "edit",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_same",
                "output": "ok"
            },
            {
                "type": "function_call",
                "call_id": "call_other",
                "name": "lookup",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_other",
                "output": "found"
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn normalizes_foreign_openai_tool_call_ids_without_item_fields() {
    let server = serve([Reply::sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_no_item\",\"usage\":{}}}\n\n",
    )])
    .await;
    let model = model(&server.base_url);
    let raw_id = "call/foreign😀Z";
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(AssistantToolCall {
            id: raw_id.into(),
            name: "lookup".into(),
            arguments: json!({"value": "hello"}),
            thought_signature: None,
            namespace: None,
        })],
        api: Api::AnthropicMessages,
        provider: ProviderId::new("anthropic"),
        model: "claude-sonnet-4-5".into(),
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
    let context = Context::new([
        Message::user("Use the tool."),
        Message::assistant(assistant),
        Message::tool_result(ToolResultMessage::new(
            raw_id,
            "lookup",
            [InputContent::text("ok")],
        )),
        Message::user("Continue"),
    ]);

    done(&events(&model, &context, &options(|_| {})).await);

    let body: Value = serde_json::from_str(
        server
            .requests()
            .await
            .pop()
            .unwrap()
            .split("\r\n\r\n")
            .nth(1)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        body["input"],
        json!([
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Use the tool."}]
            },
            {
                "type": "function_call",
                "call_id": "call_foreign__Z",
                "name": "lookup",
                "arguments": "{\"value\":\"hello\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_foreign__Z",
                "output": "ok"
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn drops_unreplayable_openai_tool_namespaces() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_namespace_drop\",\"usage\":{}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let mut model = model(&server.base_url);
    model.id = "gpt-5.5".into();
    model.name = "gpt-5.5".into();
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::ToolCall(AssistantToolCall {
                id: "call_function|fc_function".into(),
                name: "lookup".into(),
                arguments: json!({"value": "hello"}),
                thought_signature: None,
                namespace: Some("dynamic_tools".into()),
            }),
            AssistantContent::ToolCall(AssistantToolCall {
                id: "call_custom|ctc_custom".into(),
                name: "query".into(),
                arguments: json!({"input": "hello"}),
                thought_signature: None,
                namespace: Some("dynamic_tools".into()),
            }),
        ],
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: "gpt-5.4".into(),
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
    let context = Context::new([Message::assistant(assistant), Message::user("Continue")])
        .with_tools([
            Tool::new(
                "lookup",
                "Lookup",
                json!({"type": "object", "properties": {"value": {"type": "string"}}}),
            ),
            Tool {
                name: "query".into(),
                description: "Query".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"]
                }),
                constrained_sampling: Some(ds_ai::ConstrainedSampling::Grammar {
                    variants: ds_ai::GrammarVariants {
                        openai_lark: Some("start: /[a-z]+/".into()),
                        openai_regex: None,
                    },
                }),
            },
        ]);

    done(&events(&model, &context, &options(|_| {})).await);

    let body: Value = serde_json::from_str(
        server
            .requests()
            .await
            .pop()
            .unwrap()
            .split("\r\n\r\n")
            .nth(1)
            .unwrap(),
    )
    .unwrap();
    let calls = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| {
            matches!(
                item["type"].as_str(),
                Some("function_call" | "custom_tool_call")
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|call| call.get("namespace").is_none()));
}

#[tokio::test]
async fn rejects_an_openai_stream_that_ends_without_a_terminal_event() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_partial\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_partial\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Part\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Part"
    )));
    let partial = failed(&events);
    assert_eq!(partial.response_id.as_deref(), Some("resp_partial"));
    assert_eq!(partial.content, [text("Part", None)]);
    assert_eq!(partial.usage, ds_ai::Usage::default());
    assert_eq!(
        partial.error_message.as_deref(),
        Some("OpenAI Responses stream ended before a terminal response event")
    );
    server.requests().await;
}

#[tokio::test]
async fn treats_openai_response_done_as_non_terminal() {
    let sse = "data: {\"type\":\"response.done\",\"response\":{\"id\":\"resp_done\",\"status\":\"completed\",\"usage\":{}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);

    let events = events(&model, &context, &options(|_| {})).await;
    let error = failed(&events);

    assert_eq!(
        error.error_message.as_deref(),
        Some("OpenAI Responses stream ended before a terminal response event")
    );
    server.requests().await;
}

#[tokio::test]
async fn decodes_openai_sse_across_arbitrary_chunks() {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_chunks\"}}\r\n\r\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_chunks\",\"type\":\"message\",\"content\":[]}}\r\n\r\n",
        "event: response.output_text.delta\r\n",
        "data: {\"type\":\"response.output_text.delta\",\r\n",
        "data: \"output_index\":0,\"content_index\":0,\"delta\":\"Hé\"}\r\n\r\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_chunks\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hé\"}]}}\r\n\r\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_chunks\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":1,\"output_tokens_details\":{}}}}\r\n\r\n",
    );
    let accent = sse.find('é').unwrap();
    let split_points = [1, 7, 79, accent + 1, accent + 2, sse.len() - 1];
    let mut start = 0;
    let mut chunks = Vec::new();
    for end in split_points {
        chunks.push(sse.as_bytes()[start..end].to_vec());
        start = end;
    }
    chunks.push(sse.as_bytes()[start..].to_vec());
    let server = serve([Reply::sse_chunks(chunks)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Hé"
    )));
    let response = done(&events);
    assert_eq!(response.response_id.as_deref(), Some("resp_chunks"));
    assert_eq!(response.content, [text("Hé", Some(("msg_chunks", None)))]);
    assert_eq!(response.usage.input, 3);
    assert_eq!(response.usage.output, 1);
    assert_eq!(response.usage.cache_read, 0);
    assert_eq!(response.usage.reasoning, Some(0));
    assert_eq!(response.usage.total_tokens, 0);
    server.requests().await;
}

#[tokio::test]
async fn retries_openai_before_streaming_starts() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_retry\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        )
        .with_header("retry-after-ms", "0"),
        Reply::sse(completed),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));

    let events = events(&model, &context, &options).await;

    let response = done(&events);
    assert_eq!(response.response_id.as_deref(), Some("resp_retry"));
    assert!(response.content.is_empty());
    assert_eq!(
        response.usage,
        ds_ai::Usage {
            reasoning: Some(0),
            ..Default::default()
        }
    );
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn waits_for_openai_retry_headers() {
    for (header, value, delay_ms) in [("retry-after-ms", "1500", 1500), ("retry-after", "2", 2000)]
    {
        let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_delayed\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
        let server = serve([
            Reply::json(
                429,
                json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
            )
            .with_header(header, value),
            Reply::sse(completed),
        ])
        .await;
        let model = model(&server.base_url);
        let context = Context::new([Message::user("Hello")]);
        let options = options(|options| options.stream.max_retries = Some(1));
        let task = tokio::spawn(async move { events(&model, &context, &options).await });

        server.wait_for_requests_paused(1).await;
        server.wait_for_replies_paused(1).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        tokio::time::advance(std::time::Duration::from_millis(delay_ms - 1)).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(server.request_count(), 1);

        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        server.wait_for_requests_paused(2).await;
        server.wait_for_replies_paused(2).await;
        let events = task.await.unwrap();
        done(&events);
        assert_eq!(server.requests().await.len(), 2);
    }
}

#[tokio::test(start_paused = true)]
async fn waits_for_an_openai_retry_after_http_date() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_date\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let retry_at =
        httpdate::fmt_http_date(std::time::SystemTime::now() + std::time::Duration::from_secs(60));
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        )
        .with_header("retry-after", retry_at),
        Reply::sse(completed),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));
    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests_paused(1).await;
    server.wait_for_replies_paused(1).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    tokio::time::advance(std::time::Duration::from_secs(58)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_secs(3)).await;
    server.wait_for_requests_paused(2).await;
    server.wait_for_replies_paused(2).await;
    let events = task.await.unwrap();
    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn cancels_an_openai_retry_wait() {
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        )
        .with_header("retry-after-ms", "60000"),
        Reply::sse(Vec::new()),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| {
        options.stream.max_retries = Some(1);
        options.stream.cancellation = cancellation.clone();
    });
    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests_paused(1).await;
    server.wait_for_replies_paused(1).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    cancellation.cancel();
    let events = task.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("Request aborted"));
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn follows_openai_retry_status_and_override_headers() {
    let cases = [
        (408, None, true),
        (409, None, true),
        (429, None, true),
        (500, None, true),
        (600, None, true),
        (599, None, true),
        (400, None, false),
        (400, Some(("x-should-retry", "true")), true),
        (500, Some(("x-should-retry", "false")), false),
    ];

    for (status, header, should_retry) in cases {
        let mut failure = Reply::json(status, json!({"error": {"message": "failed"}}));
        if let Some((name, value)) = header {
            failure = failure.with_header(name, value);
        }
        if should_retry {
            failure = failure.with_header("retry-after-ms", "0");
        }
        let mut replies = vec![failure];
        if should_retry {
            replies.push(Reply::sse(format!(
                "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_{status}\",\"usage\":{{\"input_tokens\":0,\"input_tokens_details\":{{}},\"output_tokens\":0,\"output_tokens_details\":{{}}}}}}}}\n\n"
            )));
        }
        let server = serve(replies).await;
        let model = model(&server.base_url);
        let context = Context::new([Message::user("Hello")]);
        let options = options(|options| options.stream.max_retries = Some(1));

        let events = events(&model, &context, &options).await;

        if should_retry {
            done(&events);
            assert_eq!(server.requests().await.len(), 2);
        } else {
            assert_eq!(failed(&events).stop_reason, StopReason::Error);
            assert!(
                failed(&events).error_message.as_deref().is_some_and(
                    |message| message.starts_with(&format!("OpenAI API error ({status}):"))
                )
            );
            assert_eq!(server.request_count(), 1);
        }
    }
}

#[tokio::test(start_paused = true)]
async fn retries_openai_network_failures_before_streaming_starts() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_network\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::disconnect(), Reply::sse(completed)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));

    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests_paused(1).await;
    server.wait_for_replies_paused(1).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    tokio::time::advance(std::time::Duration::from_millis(374)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_millis(126)).await;
    server.wait_for_requests_paused(2).await;
    server.wait_for_replies_paused(2).await;
    let events = task.await.unwrap();
    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test]
async fn rejects_an_openai_retry_delay_above_a_custom_limit() {
    let server = serve([Reply::json(
        429,
        json!({"error": {"type": "rate_limit_error", "message": "retry later"}}),
    )
    .with_header("retry-after", "2")])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.max_retries = Some(1);
        options.stream.max_retry_delay = Some(std::time::Duration::from_secs(1));
    });

    let events = events(&model, &context, &options).await;
    assert_eq!(
        failed(&events).error_message.as_deref(),
        Some("Server requested 2s retry delay (max: 1s). 429 retry later",)
    );
    assert_eq!(server.requests().await.len(), 1);
}

#[tokio::test]
async fn uses_the_default_openai_retry_delay_cap_when_unspecified() {
    let server = serve([
        Reply::json(429, json!({"error": {"message": "retry later"}}))
            .with_header("retry-after", "61"),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.max_retries = Some(1);
        options.stream.max_retry_delay = None;
    });

    let events = events(&model, &context, &options).await;

    assert_eq!(
        failed(&events).error_message.as_deref(),
        Some("Server requested 61s retry delay (max: 60s). 429 retry later",)
    );
    assert_eq!(server.request_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn backs_off_before_an_openai_retry_without_a_header() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_backoff\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        ),
        Reply::sse(completed),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));
    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests_paused(1).await;
    server.wait_for_replies_paused(1).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    tokio::time::advance(std::time::Duration::from_millis(374)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_millis(126)).await;
    server.wait_for_requests_paused(2).await;
    server.wait_for_replies_paused(2).await;
    let events = task.await.unwrap();
    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn accepts_fractional_openai_retry_headers() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fractional\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        )
        .with_header("retry-after", "1.5"),
        Reply::sse(completed),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));
    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests_paused(1).await;
    server.wait_for_replies_paused(1).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    tokio::time::advance(std::time::Duration::from_millis(1499)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    server.wait_for_requests_paused(2).await;
    server.wait_for_replies_paused(2).await;
    let events = task.await.unwrap();
    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test]
async fn streams_openai_reasoning_and_text_in_content_order() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_reasoning\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"Need answer.\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Need answer.\"}],\"encrypted_content\":\"encrypted\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"content_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_reasoning\",\"usage\":{\"input_tokens\":5,\"input_tokens_details\":{},\"output_tokens\":4,\"output_tokens_details\":{\"reasoning_tokens\":3}}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Need answer."
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 1,
            delta,
            ..
        } if delta == "Hello"
    )));
    let response = done(&events);
    assert_eq!(response.response_id.as_deref(), Some("resp_reasoning"));
    assert_eq!(
        response.content,
        [
            thinking("Need answer.", Some("rs_1"), Some("encrypted")),
            text("Hello", Some(("msg_1", Some("final_answer")))),
        ]
    );
    assert_eq!(response.usage.input, 5);
    assert_eq!(response.usage.output, 4);
    assert_eq!(response.usage.cache_read, 0);
    assert_eq!(response.usage.reasoning, Some(3));
    assert_eq!(response.usage.total_tokens, 0);
}

#[tokio::test]
async fn streams_openai_reasoning_text_and_refusal_content() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_text\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"delta\":\"Private\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_text\",\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"Private\"}]}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_refusal\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.refusal.delta\",\"output_index\":1,\"delta\":\"Denied\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_refusal\",\"type\":\"message\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"Denied\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_variants\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);

    let context = Context::new([Message::user("Answer")]);
    let events = events(&model, &context, &options(|_| {})).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Private"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 1,
            delta,
            ..
        } if delta == "Denied"
    )));
    assert_eq!(
        done(&events).content,
        [
            AssistantContent::Thinking(ThinkingContent {
                thinking: "Private".into(),
                thinking_signature: Some(
                    json!({
                        "id": "rs_text",
                        "type": "reasoning",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": "Private"}]
                    })
                    .to_string(),
                ),
                redacted: None,
            }),
            text("Denied", Some(("msg_refusal", None))),
        ]
    );
    server.requests().await;
}

#[tokio::test]
async fn replays_serialized_openai_reasoning_and_message_items() {
    let first_sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_first\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_replay\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Need answer.\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_replay\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Need answer.\"}]}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_replay\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_replay\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_first\",\"output\":[{\"id\":\"rs_replay\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Need answer.\"}],\"encrypted_content\":\"encrypted\"}],\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":2,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(first_sse)]).await;
    let first_model = model(&first_server.base_url);
    let first_context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});
    let first_events = events(&first_model, &first_context, &options).await;
    let response: AssistantMessage =
        serde_json::from_value(serde_json::to_value(done(&first_events)).unwrap()).unwrap();
    first_server.requests().await;

    let second_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_second\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model = model(&second_server.base_url);
    let second_context = Context::new([Message::assistant(response), Message::user("Continue")]);

    let second_events = events(&second_model, &second_context, &options).await;
    done(&second_events);

    let request = second_server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["input"],
        json!([
            {
                "type": "reasoning",
                "id": "rs_replay",
                "summary": [{"type": "summary_text", "text": "Need answer."}],
                "encrypted_content": "encrypted"
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                "status": "completed",
                "id": "msg_replay",
                "phase": "final_answer"
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn preserves_the_complete_openai_reasoning_item_for_replay() {
    let first_sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_content\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"delta\":\"Private\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_content\",\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"Private\"}],\"encrypted_content\":\"opaque\",\"status\":\"completed\",\"metadata\":{\"trace\":\"x\"}}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_content\",\"status\":\"completed\",\"output\":[{\"id\":\"rs_content\",\"type\":\"reasoning\",\"encrypted_content\":\"terminal\"}],\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([
        Reply::sse(first_sse),
        Reply::sse(
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_next\",\"status\":\"completed\",\"usage\":{}}}\n\n",
        ),
    ])
    .await;
    let model = model(&server.base_url);
    let options = options(|_| {});
    let first = events(&model, &Context::new([Message::user("Think")]), &options).await;
    let response = done(&first).clone();
    let expected = json!({
        "id": "rs_content",
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Private"}],
        "encrypted_content": "opaque",
        "status": "completed",
        "metadata": {"trace": "x"}
    });
    let AssistantContent::Thinking(thinking) = &response.content[0] else {
        panic!("missing reasoning content");
    };
    assert_eq!(
        serde_json::from_str::<Value>(thinking.thinking_signature.as_deref().unwrap()).unwrap(),
        expected
    );

    let second = events(
        &model,
        &Context::new([Message::assistant(response), Message::user("Continue")]),
        &options,
    )
    .await;
    done(&second);
    let requests = server.requests().await;
    let replay: Value =
        serde_json::from_str(requests[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(replay["input"][0], expected);
}

#[tokio::test]
async fn generates_distinct_replay_ids_for_text_without_provider_ids() {
    let first_sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"First\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"First\"}]}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Second\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Second\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_without_ids\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let done_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_next\",\"usage\":{}}}\n\n";
    let server = serve([Reply::sse(first_sse), Reply::sse(done_sse)]).await;
    let model = model(&server.base_url);
    let options = options(|_| {});
    let first = events(&model, &Context::new([Message::user("Split")]), &options).await;
    let replay = done(&first).clone();

    let second = events(
        &model,
        &Context::new([
            Message::user("Split"),
            Message::assistant(replay),
            Message::user("Continue"),
        ]),
        &options,
    )
    .await;
    done(&second);

    let requests = server.requests().await;
    let body: Value = serde_json::from_str(requests[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    let messages = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "message" && item["role"] == "assistant")
        .collect::<Vec<_>>();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["id"], "msg_pi_1");
    assert_eq!(messages[1]["id"], "msg_pi_1_1");
}

#[tokio::test]
async fn streams_and_replays_openai_tool_calls() {
    let first_sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_edit\",\"type\":\"function_call\",\"call_id\":\"call_edit\",\"name\":\"edit\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"path\\\":\\\"README.md\\\"\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\",\\\"content\\\":\\\"updated\\\"}\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{\\\"path\\\":\\\"README.md\\\",\\\"content\\\":\\\"updated\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_edit\",\"type\":\"function_call\",\"call_id\":\"call_edit\",\"name\":\"edit\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\",\\\"content\\\":\\\"updated\\\"}\",\"namespace\":\"dynamic_tools\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":2,\"output_tokens_details\":{}}}}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(first_sse)]).await;
    let first_model = model(&first_server.base_url);
    let first_context = Context::new([Message::user("Edit the file")]).with_tools([Tool::new(
        "edit",
        "Edit a file",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
    )
    .with_strict()]);
    let options = options(|_| {});

    let first_events = events(&first_model, &first_context, &options).await;

    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "{\"path\":\"README.md\""
    )));
    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta,
            ..
        } if delta == ",\"content\":\"updated\"}"
    )));
    let response = done(&first_events);
    assert_eq!(
        response.content,
        [AssistantContent::ToolCall(AssistantToolCall {
            id: "call_edit|fc_edit".into(),
            name: "edit".into(),
            arguments: json!({"path": "README.md", "content": "updated"}),
            thought_signature: None,
            namespace: Some("dynamic_tools".into()),
        })]
    );

    let first_request = first_server.requests().await.pop().unwrap();
    let first_body: Value =
        serde_json::from_str(first_request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        first_body["tools"],
        json!([{
            "type": "function",
            "name": "edit",
            "description": "Edit a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            },
            "strict": true
        }])
    );

    let restored: AssistantMessage =
        serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap();
    let second_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_second\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model = model(&second_server.base_url);
    let second_context = Context::new([Message::assistant(restored), Message::user("Continue")]);

    let second_events = events(&second_model, &second_context, &options).await;
    done(&second_events);

    let second_request = second_server.requests().await.pop().unwrap();
    let second_body: Value =
        serde_json::from_str(second_request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        second_body["input"],
        json!([
            {
                "type": "function_call",
                "id": "fc_edit",
                "call_id": "call_edit",
                "name": "edit",
                "arguments": "{\"path\":\"README.md\",\"content\":\"updated\"}",
                "namespace": "dynamic_tools"
            },
            {
                "type": "function_call_output",
                "call_id": "call_edit",
                "output": "No result provided"
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn uses_the_provider_terminal_token_total() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_total\",\"status\":\"completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":2,\"total_tokens\":99}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;

    assert_eq!(done(&events).usage.total_tokens, 99);
    server.requests().await;
}

#[tokio::test]
async fn ignores_end_turn_on_openai_terminal_responses() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_end_turn\",\"status\":\"completed\",\"end_turn\":true,\"usage\":{}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;

    assert_eq!(done(&events).end_turn, None);
    server.requests().await;
}

#[tokio::test]
async fn rejects_an_incomplete_response_without_a_reason() {
    let server = serve([
        Reply::sse("data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_incomplete\",\"status\":\"incomplete\",\"incomplete_details\":null}}\n\n"),
        Reply::sse("data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_empty_reason\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"\"}}}\n\n"),
    ]).await;
    let model = model(&server.base_url);
    for _ in 0..2 {
        let events = events(
            &model,
            &Context::new([Message::user("Hello")]),
            &options(|_| {}),
        )
        .await;

        let error = failed(&events);
        assert_eq!(
            error.error_message.as_deref(),
            Some("Response incomplete without a provider reason")
        );
        assert_eq!(error.raw_stop_reason.as_deref(), Some("incomplete"));
    }
    server.requests().await;
}

#[tokio::test]
async fn maps_an_openai_incomplete_event_using_its_response_status() {
    let sse = "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_incomplete_status\",\"status\":\"cancelled\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;

    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.raw_stop_reason.as_deref(), Some("cancelled"));
    assert_eq!(
        error.error_message.as_deref(),
        Some("An unknown error occurred")
    );
    server.requests().await;
}

#[tokio::test]
async fn rejects_invalid_strict_tool_schemas_before_connecting() {
    let model = model("http://127.0.0.1:9");
    let context = Context::new([Message::user("Look up")]).with_tools([Tool::new(
        "lookup",
        "Look up a value",
        json!({
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "$defs": {"value": {"type": "string"}}
        }),
    )
    .with_strict()]);

    let result_events = events(&model, &context, &options(|_| {})).await;

    assert_eq!(
        failed(&result_events).error_message.as_deref(),
        Some(
            "invalid request: Tool \"lookup\" requires JSON-schema constrained sampling, but $defs schemas are unsupported."
        )
    );

    let context = Context::new([Message::user("Look up")]).with_tools([Tool::new(
        "lookup",
        "Look up a value",
        json!({
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }),
    )
    .with_strict()]);
    let additional_properties_events = events(&model, &context, &options(|_| {})).await;
    assert_eq!(
        failed(&additional_properties_events)
            .error_message
            .as_deref(),
        Some(
            "invalid request: Tool \"lookup\" requires JSON-schema constrained sampling, but schema-valued or true additionalProperties is unsupported."
        )
    );
}

#[tokio::test]
async fn sends_openai_tool_result_text_images_and_empty_output() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_result\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([
        Message::user("Inspect the image"),
        Message::tool_result(ToolResultMessage::new(
            "call_image",
            "inspect",
            [
                InputContent::text("A red circle"),
                InputContent::image("image/png", "iVBORw0KGgo="),
            ],
        )),
        Message::tool_result(ToolResultMessage::new(
            "call_empty",
            "noop",
            [InputContent::text("")],
        )),
        Message::tool_result(ToolResultMessage::new(
            "call_image_only",
            "inspect",
            [InputContent::image("image/jpeg", "/9j/4AAQ")],
        )),
    ]);
    let events = events(&model, &context, &options(|_| {})).await;
    done(&events);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["input"],
        json!([
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect the image"}]
            },
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {"type": "input_text", "text": "A red circle"},
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": "data:image/png;base64,iVBORw0KGgo="
                    }
                ]
            },
            {
                "type": "function_call_output",
                "call_id": "call_empty",
                "output": "(no tool output)"
            },
            {
                "type": "function_call_output",
                "call_id": "call_image_only",
                "output": [
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": "data:image/jpeg;base64,/9j/4AAQ"
                    }
                ]
            }
        ])
    );
}

#[tokio::test]
async fn finalizes_an_incomplete_openai_response_as_a_length_stop() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_incomplete\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_incomplete\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Partial\"}\n\n",
        "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_incomplete\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":30,\"input_tokens_details\":{\"cached_tokens\":5},\"output_tokens\":12,\"output_tokens_details\":{},\"total_tokens\":42}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let events = events(&model, &context, &options(|_| {})).await;

    let response = done(&events);
    assert_eq!(response.stop_reason, StopReason::Length);
    assert_eq!(
        response.raw_stop_reason.as_deref(),
        Some("incomplete.max_output_tokens")
    );
    assert_eq!(response.content, [text("Partial", None)]);
    assert_eq!(response.usage.input, 25);
    assert_eq!(response.usage.cache_read, 5);
    assert_eq!(response.usage.output, 12);
}

#[tokio::test]
async fn rejects_a_content_filtered_openai_response_with_partial_content() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_filtered\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_filtered\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
        "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_filtered\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"},\"usage\":{\"input_tokens\":2,\"input_tokens_details\":{},\"output_tokens\":1,\"output_tokens_details\":{}}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let events = events(&model, &context, &options(|_| {})).await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("Response incomplete: content_filter")
    );
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.raw_stop_reason.as_deref(),
        Some("incomplete.content_filter")
    );
    assert_eq!(error.content, [text("Visible", None)]);
}

#[tokio::test]
async fn preserves_an_unknown_openai_incomplete_reason() {
    let sse = "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_unknown_reason\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_time_limit\"}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Write")]),
        &options(|_| {}),
    )
    .await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("Response incomplete: max_time_limit")
    );
    assert_eq!(
        error.raw_stop_reason.as_deref(),
        Some("incomplete.max_time_limit")
    );
}

#[tokio::test]
async fn rejects_a_failed_openai_response_with_code_and_partial_content() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_failed\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_failed\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Partial thought\"}\n\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Think")]);
    let events = events(&model, &context, &options(|_| {})).await;

    let error = failed(&events);
    assert_eq!(error.error_message.as_deref(), Some("server_error: boom"));
    assert_eq!(error.response_id.as_deref(), Some("resp_failed"));
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.raw_stop_reason.as_deref(), Some("failed"));
    assert_eq!(error.content, [thinking("Partial thought", None, None)]);
}

#[tokio::test]
async fn does_not_copy_metadata_from_a_failed_openai_response() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_created\"}}\n\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed\",\"status\":\"failed\",\"end_turn\":true,\"service_tier\":\"priority\",\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Think")]),
        &options(|_| {}),
    )
    .await;

    let error = failed(&events);
    assert_eq!(error.response_id.as_deref(), Some("resp_created"));
    assert_eq!(error.end_turn, None);
    server.requests().await;
}

#[tokio::test]
async fn reports_failed_openai_response_without_error_details() {
    let sse = "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed_unknown\",\"status\":\"failed\"}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Write")]),
        &options(|_| {}),
    )
    .await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("Unknown error (no error details in response)")
    );
}

#[tokio::test]
async fn reports_failed_openai_response_incomplete_reason() {
    let sse = "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed_incomplete\",\"status\":\"failed\",\"incomplete_details\":{\"reason\":\"max_time_limit\"}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Write")]),
        &options(|_| {}),
    )
    .await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("incomplete: max_time_limit")
    );
}

#[tokio::test]
async fn matches_empty_openai_terminal_and_event_fields() {
    let server = serve([
        Reply::sse(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"\"}}\n\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"\",\"status\":\"\",\"error\":{\"code\":\"\",\"message\":\"\"}}}\n\n",
        ),
        Reply::sse(
            "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"\"}}}\n\n",
        ),
        Reply::sse(
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"\",\"status\":\"\",\"usage\":{}}}\n\n",
        ),
        Reply::sse("data: {\"type\":\"error\",\"code\":\"\",\"message\":\"\"}\n\n"),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);

    let first_events = events(&model, &context, &options(|_| {})).await;
    let first = failed(&first_events);
    assert_eq!(first.response_id.as_deref(), Some(""));
    assert_eq!(first.raw_stop_reason, None);
    assert_eq!(first.error_message.as_deref(), Some("unknown: no message"));

    let second_events = events(&model, &context, &options(|_| {})).await;
    let second = failed(&second_events);
    assert_eq!(
        second.error_message.as_deref(),
        Some("server_error: no message")
    );

    let third_events = events(&model, &context, &options(|_| {})).await;
    let third = done(&third_events);
    assert_eq!(third.response_id, None);
    assert_eq!(third.raw_stop_reason, None);
    assert_eq!(third.stop_reason, StopReason::Stop);

    let fourth_events = events(&model, &context, &options(|_| {})).await;
    let fourth = failed(&fourth_events);
    assert_eq!(fourth.error_message.as_deref(), Some("Error Code : "));
    server.requests().await;
}

#[tokio::test]
async fn rejects_an_openai_error_event_with_code_and_partial_content() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_error\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_error\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
        "data: {\"type\":\"error\",\"code\":\"rate_limit_exceeded\",\"message\":\"slow down\",\"param\":null}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let events = events(&model, &context, &options(|_| {})).await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("Error Code rate_limit_exceeded: slow down")
    );
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.raw_stop_reason, None);
    assert_eq!(error.content, [text("Visible", None)]);
}

#[tokio::test]
async fn does_not_use_nested_error_details_for_openai_error_events() {
    let sse = "data: {\"type\":\"error\",\"error\":{\"code\":\"nested_code\",\"message\":\"nested message\"}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Write")]),
        &options(|_| {}),
    )
    .await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("Error Code undefined: undefined")
    );
    server.requests().await;
}

#[tokio::test]
async fn emits_a_reasoning_summary_part_boundary_delta() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_boundary\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Part\"}\n\n",
        "data: {\"type\":\"response.reasoning_summary_part.done\",\"output_index\":0}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_boundary\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Part\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_boundary\",\"status\":\"completed\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Think")]),
        &options(|_| {}),
    )
    .await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta { delta, .. } if delta == "\n\n"
    )));
    assert_eq!(
        done(&events).content,
        [thinking("Part", Some("rs_boundary"), None)]
    );
}

#[tokio::test]
async fn encodes_openai_multimodal_context_and_generation_options() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_options\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user_content([
        InputContent::text("Describe this"),
        InputContent::image("image/png", "iVBORw0KGgo="),
    ])])
    .with_system("Be brief")
    .with_tools([Tool::new(
        "inspect",
        "Inspect an image",
        json!({"type": "object", "properties": {}, "additionalProperties": false}),
    )]);
    let options = options(|options| {
        options.stream.max_tokens = Some(128);
        options.stream.temperature = Some(0.2);
        options.reasoning_effort = Some(openai::ReasoningEffort::High);
        options.reasoning_summary = Some(openai::ReasoningSummary::Concise);
        options.tool_choice = Some(openai::ToolChoice::Required);
        options.service_tier = Some(openai::ServiceTier::Priority);
    });

    let events = events(&model, &context, &options).await;
    done(&events);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "gpt-5.6",
            "input": [
                {
                    "role": "developer",
                    "content": "Be brief"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this"},
                        {
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": "data:image/png;base64,iVBORw0KGgo="
                        }
                    ]
                }
            ],
            "tools": [{
                "type": "function",
                "name": "inspect",
                "description": "Inspect an image",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                },
                "strict": false
            }],
            "stream": true,
            "store": false,
            "max_output_tokens": 128,
            "temperature": 0.2,
            "reasoning": {"effort": "high", "summary": "concise"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "required",
            "service_tier": "priority"
        })
    );
}

#[tokio::test]
async fn cancels_an_openai_request_before_response_headers() {
    let server = serve([Reply::pending()]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| options.stream.cancellation = cancellation.clone());
    let request = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    cancellation.cancel();

    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("Request aborted"));
    assert!(error.content.is_empty());
}

#[tokio::test]
async fn cancels_while_reading_an_openai_error_body() {
    let server = serve([Reply::open_json(
        500,
        json!({"error": {"message": "unfinished"}}),
    )])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| options.stream.cancellation = cancellation.clone());
    let request = tokio::spawn(async move { events(&model, &context, &options).await });
    server.wait_for_requests(1).await;
    tokio::task::yield_now().await;

    cancellation.cancel();

    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("Request aborted"));
    assert!(error.content.is_empty());
    server.requests().await;
}

#[tokio::test]
async fn cancels_an_active_openai_stream_with_partial_content() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_cancel\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_cancel\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::open_sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| options.stream.cancellation = cancellation.clone());
    let mut response = openai::stream(
        &model.typed::<ds_ai::OpenAiResponsesOptions>().unwrap(),
        &context,
        &options,
    );

    while !matches!(
        response.next().await,
        Some(AssistantMessageEvent::TextDelta { .. })
    ) {}
    cancellation.cancel();

    match response.next().await {
        Some(AssistantMessageEvent::Error { reason, error }) => {
            assert_eq!(reason, ErrorReason::Aborted);
            assert_eq!(error.stop_reason, StopReason::Aborted);
            assert_eq!(error.raw_stop_reason.as_deref(), Some("cancelled"));
            assert_eq!(
                error.error_message.as_deref(),
                Some("OpenAI Responses stream ended before a terminal response event")
            );
            assert_eq!(error.content, [text("Visible", None)]);
        }
        event => panic!("unexpected cancellation event: {event:?}"),
    }
}

#[tokio::test]
async fn times_out_an_openai_request_before_response_headers_per_attempt() {
    let server = serve([Reply::pending()]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::from_millis(50));
    });
    let request = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider timed out during Connection")
    );
    assert!(error.content.is_empty());
}

#[tokio::test]
async fn does_not_apply_an_openai_timeout_to_an_error_body() {
    let server = serve([Reply::open_json(
        500,
        json!({"error": {"message": "unfinished"}}),
    )])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::from_millis(10));
        options.stream.cancellation = cancellation.clone();
    });
    let request = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancellation.cancel();

    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.error_message.as_deref(), Some("Request aborted"));
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert!(error.content.is_empty());
    server.requests().await;
}

#[tokio::test]
async fn accepts_an_openai_stream_body_after_the_header_timeout() {
    let server = serve([Reply::open_sse(Vec::new())]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::from_secs(5));
        options.stream.cancellation = cancellation.clone();
    });
    let mut response = openai::stream(
        &model.typed::<ds_ai::OpenAiResponsesOptions>().unwrap(),
        &context,
        &options,
    );
    assert!(matches!(
        response.next().await,
        Some(AssistantMessageEvent::Start { .. })
    ));

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancellation.cancel();

    match response.next().await {
        Some(AssistantMessageEvent::Error { reason, error }) => {
            assert_eq!(reason, ErrorReason::Aborted);
            assert_eq!(error.stop_reason, StopReason::Aborted);
            assert_eq!(error.raw_stop_reason.as_deref(), Some("cancelled"));
            assert_eq!(
                error.error_message.as_deref(),
                Some("OpenAI Responses stream ended before a terminal response event")
            );
            assert!(error.content.is_empty());
        }
        event => panic!("unexpected timeout event: {event:?}"),
    }
}

#[tokio::test]
async fn retries_an_openai_header_timeout_with_a_fresh_timeout() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_retry_timeout\",\"status\":\"completed\",\"usage\":{}}}\n\n";
    let server = serve([Reply::pending(), Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::from_millis(25));
        options.stream.max_retries = Some(1);
        options.stream.max_retry_delay = Some(std::time::Duration::ZERO);
    });
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        events(&model, &context, &options),
    )
    .await
    .expect("retry should complete");
    done(&events);
    assert_eq!(server.request_count(), 2);
    server.requests().await;
}

#[tokio::test]
async fn uses_the_sdk_zero_timeout_behavior_for_openai() {
    let server = serve([Reply::pending()]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::ZERO);
    });
    let events = events(&model, &context, &options).await;
    assert_eq!(
        failed(&events).error_message.as_deref(),
        Some("provider timed out during Connection")
    );
}

#[tokio::test]
async fn encodes_openai_prompt_cache_retention_and_session_keys() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_cache\",\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::sse(sse), Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let session_id = "🦀".repeat(70);
    let long = options(|options| {
        options.stream.session_id = Some(session_id.clone());
        options.stream.cache_retention = CacheRetention::Long;
    });
    let first = events(&model, &context, &long).await;
    done(&first);

    let disabled = options(|options| {
        options.stream.session_id = Some(session_id.clone());
        options.stream.cache_retention = CacheRetention::None;
    });
    let second = events(&model, &context, &disabled).await;
    done(&second);

    let requests = server.requests().await;
    let long_body: Value =
        serde_json::from_str(requests[0].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(long_body["prompt_cache_key"], "🦀".repeat(64));
    assert_eq!(long_body["prompt_cache_retention"], "24h");
    let disabled_body: Value =
        serde_json::from_str(requests[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert!(disabled_body.get("prompt_cache_key").is_none());
    assert!(disabled_body.get("prompt_cache_retention").is_none());
    assert_eq!(
        disabled_body["prompt_cache_options"],
        json!({"mode": "explicit"})
    );
}

#[tokio::test]
async fn exposes_openai_error_headers_to_the_response_hook() {
    let failure = Reply::json(
        429,
        json!({"error": {"code": "rate_limit_exceeded", "message": "Too many requests"}}),
    )
    .with_header("retry-after-ms", "0");
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_metadata\",\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let success = Reply::sse(sse)
        .with_header("x-request-id", "req_success")
        .with_header("x-repeated", "one")
        .with_header("x-repeated", "two")
        .with_header("x-ratelimit-limit-tokens", "1000")
        .with_header("x-ratelimit-remaining-tokens", "900")
        .with_header("x-ratelimit-reset-tokens", "2s");
    let server = serve([failure, success]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let responses = Arc::new(Mutex::new(Vec::new()));
    let captured = responses.clone();
    let mut options = options(|_| {});
    options.stream.max_retries = Some(1);
    options.stream.on_response = Some(ResponseHook::new(move |response, _| {
        let captured = captured.clone();
        async move {
            captured.lock().unwrap().push(response);
            Ok(())
        }
    }));

    let success_events = events(&model, &context, &options).await;
    done(&success_events);
    let captured = responses.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let success_response = captured[0].clone();
    assert_eq!(success_response.status, 200);
    assert_eq!(
        success_response
            .headers
            .get("x-request-id")
            .map(String::as_str),
        Some("req_success")
    );
    assert_eq!(
        success_response
            .headers
            .get("x-repeated")
            .map(String::as_str),
        Some("one, two")
    );
    assert_eq!(
        success_response
            .headers
            .get("x-ratelimit-limit-tokens")
            .map(String::as_str),
        Some("1000")
    );
    assert_eq!(
        success_response
            .headers
            .get("x-ratelimit-remaining-tokens")
            .map(String::as_str),
        Some("900")
    );
    assert_eq!(
        success_response
            .headers
            .get("x-ratelimit-reset-tokens")
            .map(String::as_str),
        Some("2s")
    );
    assert_eq!(server.request_count(), 2);
}

#[tokio::test]
async fn normalizes_and_caps_openai_provider_error_bodies() {
    let long_message = "x".repeat(5_000);
    let serialized_length = serde_json::to_string(&json!({"message": long_message.clone()}))
        .unwrap()
        .len();
    let server = serve([
        Reply::json(400, json!({"error": {"message": long_message}})),
        Reply::json(403, json!({})),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);

    let first = events(&model, &context, &options(|_| {})).await;
    let error = failed(&first);
    let message = error.error_message.as_deref().unwrap();
    assert!(message.starts_with("OpenAI API error (400): "));
    let suffix = format!("... [truncated {} chars]", serialized_length - 4_000);
    assert!(message.ends_with(&suffix));
    assert_eq!(
        message.chars().count(),
        "OpenAI API error (400): ".chars().count() + 4_000 + suffix.chars().count()
    );

    let second = events(&model, &context, &options(|_| {})).await;
    assert_eq!(
        failed(&second).error_message.as_deref(),
        Some("OpenAI API error (403): 403 status code (no body)")
    );
    server.requests().await;
}

fn model(base_url: &str) -> ds_ai::Model {
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.id = "gpt-5.6".into();
    model.name = "gpt-5.6".into();
    model.base_url = base_url.into();
    model
}

fn options(configure: impl FnOnce(&mut OpenAiResponsesOptions)) -> OpenAiResponsesOptions {
    let mut options = OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
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
    options: &OpenAiResponsesOptions,
) -> Vec<AssistantMessageEvent> {
    openai::stream(
        &model.typed::<ds_ai::OpenAiResponsesOptions>().unwrap(),
        context,
        options,
    )
    .collect()
    .await
}

fn text(value: &str, signature: Option<(&str, Option<&str>)>) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: value.into(),
        text_signature: signature.map(|(id, phase)| {
            let mut signature = json!({"v": 1, "id": id});
            if let Some(phase) = phase {
                signature["phase"] = phase.into();
            }
            signature.to_string()
        }),
    })
}

fn thinking(value: &str, id: Option<&str>, encrypted: Option<&str>) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent {
        thinking: value.into(),
        thinking_signature: id.map(|id| {
            let mut signature = json!({
                "id": id,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": value}],
            });
            if let Some(encrypted) = encrypted {
                signature["encrypted_content"] = encrypted.into();
            }
            signature.to_string()
        }),
        redacted: None,
    })
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
