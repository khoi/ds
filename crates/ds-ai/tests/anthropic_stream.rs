use crate::support::{Reply, serve};
use async_trait::async_trait;
use ds_ai::{
    AnthropicMessagesCompatibility, AnthropicOptions, Api, AssistantContent, AssistantMessage,
    AssistantMessageEvent, AssistantToolCall, AuthContext, CacheRetention, Context, InputContent,
    Message, ModelCompatibility, Models, ResponseHook, SimpleStreamOptions, StopReason,
    StreamOptions, TextContent, ThinkingContent, Tool, ToolResultMessage, anthropic, builtin_model,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[tokio::test]
async fn streams_anthropic_text_until_message_stop() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":12,\"output_tokens\":0,\"cache_read_input_tokens\":2,\"cache_creation_input_tokens\":3}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)
        .with_header("request-id", "req_anthropic")
        .with_header("anthropic-ratelimit-requests-limit", "100")
        .with_header("anthropic-ratelimit-requests-remaining", "90")
        .with_header("anthropic-ratelimit-requests-reset", "2026-08-24T12:00:00Z")
        .with_header("anthropic-ratelimit-tokens-limit", "10000")
        .with_header("anthropic-ratelimit-tokens-remaining", "9000")
        .with_header("anthropic-ratelimit-tokens-reset", "2026-08-24T12:01:00Z")])
    .await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]).with_system("Be brief");
    let options = options(|stream| stream.max_tokens = Some(1024));

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
    assert_eq!(response.response_id.as_deref(), Some("msg_1"));
    assert_eq!(response.content, [text("Hello")]);
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(response.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(response.usage.input, 12);
    assert_eq!(response.usage.output, 5);
    assert_eq!(response.usage.cache_read, 2);
    assert_eq!(response.usage.cache_write, 3);
    assert_eq!(response.usage.cache_write_1h, Some(0));
    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /v1/messages HTTP/1.1\r\n"));
    assert!(request.contains("x-api-key: test-key\r\n"));
    assert!(request.contains("anthropic-version: 2023-06-01\r\n"));
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "claude-sonnet-4-5",
            "system": [{
                "type": "text",
                "text": "Be brief",
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Hello",
                    "cache_control": {"type": "ephemeral"}
                }]
            }],
            "max_tokens": 1024,
            "stream": true
        })
    );
}

#[tokio::test]
async fn accepts_anthropic_eof_flush_and_cr_line_endings() {
    let eof = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_eof\",\"usage\":{}}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}"
    );
    let cr = concat!(
        "event: message_start\rdata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_cr\",\"usage\":{}}}\r\r",
        "event: message_delta\rdata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\r\r",
        "event: message_stop\rdata: {\"type\":\"message_stop\"}\r\r"
    );

    for (body, response_id) in [(eof, "msg_eof"), (cr, "msg_cr")] {
        let server = serve([Reply::sse(body)]).await;
        let model = model("claude-sonnet-4-5", &server.base_url);
        let streamed = events(
            &model,
            &Context::new([Message::user("Hello")]),
            &options(|_| {}),
        )
        .await;
        let response = done(&streamed);

        assert_eq!(response.response_id.as_deref(), Some(response_id));
        assert_eq!(response.stop_reason, StopReason::Stop);
        server.requests().await;
    }
}

#[tokio::test]
async fn merges_model_headers_into_direct_anthropic_streams() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    let mut model = model("claude-sonnet-4-5", &server.base_url);
    model
        .headers
        .insert("x-model-header".into(), "model".into());
    model
        .headers
        .insert("User-Agent".into(), "model-agent".into());
    let options = options(|stream| {
        stream
            .headers
            .insert("X-Request-Header".into(), Some("request".into()));
        stream
            .headers
            .insert("user-agent".into(), Some("request-agent".into()));
    });

    done(&events(&model, &Context::new([Message::user("Hello")]), &options).await);

    let request = server.requests().await.pop().unwrap();
    assert!(request.contains("x-model-header: model\r\n"));
    assert!(request.contains("x-request-header: request\r\n"));
    assert!(request.contains("user-agent: request-agent\r\n"));
    assert!(!request.contains("user-agent: model-agent\r\n"));
    assert_eq!(request.matches("user-agent:").count(), 1);
}

#[tokio::test]
async fn rejects_model_auth_headers_without_caller_auth() {
    let server = serve(std::iter::empty::<Reply>()).await;
    let mut model = model("claude-sonnet-4-5", &server.base_url);
    model
        .headers
        .insert("Authorization".into(), "Bearer model-token".into());
    let options = AnthropicOptions {
        stream: StreamOptions::default(),
        ..Default::default()
    };

    let events = events(&model, &Context::new([Message::user("Hello")]), &options).await;
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some("No API key for provider: anthropic")
    );
    assert!(server.requests().await.is_empty());
}

#[tokio::test]
async fn rejects_case_insensitive_auth_suppression_after_request_headers() {
    let server = serve(std::iter::empty::<Reply>()).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let mut stream = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    stream.headers.insert("X-API-Key".into(), None);
    let options = AnthropicOptions {
        stream,
        ..Default::default()
    };

    let events = events(&model, &Context::new([Message::user("Hello")]), &options).await;
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some("No API key for provider: anthropic")
    );
    assert!(server.requests().await.is_empty());
}

#[tokio::test]
async fn resolves_anthropic_environment_auth_and_request_header_precedence() {
    let auth_token = auth_request(
        [("ANTHROPIC_AUTH_TOKEN", "auth-token")],
        BTreeMap::new(),
        None,
    )
    .await;
    assert!(auth_token.contains("authorization: Bearer auth-token\r\n"));
    assert!(!auth_token.contains("x-api-key:"));
    assert!(!auth_token.contains("oauth-2025-04-20"));

    let combined = auth_request(
        [
            ("ANTHROPIC_AUTH_TOKEN", "auth-token"),
            ("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat-test"),
            ("ANTHROPIC_API_KEY", "api-key"),
        ],
        BTreeMap::new(),
        None,
    )
    .await;
    assert!(combined.contains("authorization: Bearer auth-token\r\n"));
    assert!(!combined.contains("x-api-key:"));
    assert!(!combined.contains("oauth-2025-04-20"));

    let oauth_over_api_key = auth_request(
        [
            ("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat-test"),
            ("ANTHROPIC_API_KEY", "api-key"),
        ],
        BTreeMap::new(),
        None,
    )
    .await;
    assert!(oauth_over_api_key.contains("authorization: Bearer sk-ant-oat-test\r\n"));
    assert!(!oauth_over_api_key.contains("x-api-key:"));

    let oauth_token = auth_request(
        [("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat-test")],
        BTreeMap::new(),
        None,
    )
    .await;
    assert!(oauth_token.contains("authorization: Bearer sk-ant-oat-test\r\n"));
    assert!(oauth_token.contains("anthropic-beta: claude-code-20250219,oauth-2025-04-20"));

    let api_key = auth_request([("ANTHROPIC_API_KEY", "api-key")], BTreeMap::new(), None).await;
    assert!(api_key.contains("x-api-key: api-key\r\n"));
    assert!(!api_key.contains("oauth-2025-04-20"));

    let explicit = auth_request(
        [("ANTHROPIC_AUTH_TOKEN", "context-token")],
        BTreeMap::from([(
            String::from("Authorization"),
            Some(String::from("Bearer explicit")),
        )]),
        None,
    )
    .await;
    assert!(explicit.contains("authorization: Bearer explicit\r\n"));
    assert!(!explicit.contains("authorization: Bearer context-token\r\n"));
}

#[tokio::test]
async fn limits_anthropic_session_affinity_to_nonempty_api_key_sessions() {
    let mut model = model("claude-opus-4-5", "");
    anthropic_compat(&mut model).send_session_affinity_headers = Some(true);

    let (_, api_key_request) = request_for_model(
        model.clone(),
        Context::new([Message::user("Hello")]),
        options(|stream| {
            stream.cache_retention = CacheRetention::None;
            stream.session_id = Some("session-1".into());
        }),
    )
    .await;
    assert!(api_key_request.contains("x-session-affinity: session-1\r\n"));

    let (_, empty_session_request) = request_for_model(
        model.clone(),
        Context::new([Message::user("Hello")]),
        options(|stream| {
            stream.cache_retention = CacheRetention::None;
            stream.session_id = Some(String::new());
        }),
    )
    .await;
    assert!(!empty_session_request.contains("x-session-affinity:"));

    let (_, oauth_request) = request_for_model(
        model,
        Context::new([Message::user("Hello")]),
        options(|stream| {
            stream.api_key = Some("sk-ant-oat-test".into());
            stream.cache_retention = CacheRetention::Long;
            stream.session_id = Some("session-1".into());
        }),
    )
    .await;
    assert!(!oauth_request.contains("x-session-affinity:"));
}

#[tokio::test]
async fn preserves_anthropic_direct_max_tokens_above_model_cap() {
    let model = model("claude-sonnet-4-5", "");
    let requested = model.max_tokens + 123;
    let body = request_body_for(
        "claude-sonnet-4-5",
        Context::new([Message::user("Hello")]),
        options(|stream| stream.max_tokens = Some(requested)),
    )
    .await;

    assert_eq!(body["max_tokens"], requested);
}

#[tokio::test]
async fn omits_empty_anthropic_user_content_but_keeps_images() {
    let empty = request_body(Context::new([Message::user("")])).await;
    assert_eq!(empty["messages"], json!([]));

    let whitespace = request_body(Context::new([Message::user("  \n\t ")])).await;
    assert_eq!(whitespace["messages"], json!([]));

    let mixed = request_body(Context::new([Message::user_content([
        InputContent::text("  "),
        InputContent::image("image/png", "iVBORw0KGgo="),
    ])]))
    .await;
    assert_eq!(
        mixed["messages"],
        json!([{
            "role": "user",
            "content": [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgo="
                },
                "cache_control": {"type": "ephemeral"}
            }]
        }])
    );
}

#[tokio::test]
async fn keeps_anthropic_plain_text_and_same_role_messages_separate() {
    let body = request_body_for(
        "claude-sonnet-4-5",
        Context::new([Message::user("first"), Message::user("second")]),
        options(|stream| stream.cache_retention = CacheRetention::None),
    )
    .await;

    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"}
        ])
    );
}

#[tokio::test]
async fn preserves_anthropic_same_model_signatures_and_tool_ids() {
    let response = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "Think".into(),
                thinking_signature: Some("  padded signature  ".into()),
                redacted: None,
            }),
            AssistantContent::ToolCall(AssistantToolCall {
                id: "call|same-model".into(),
                name: "lookup".into(),
                arguments: json!({}),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: Api::AnthropicMessages,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-5".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: Some("tool_use".into()),
        end_turn: None,
        timestamp: 0,
    };
    let body = request_body_for(
        "claude-sonnet-4-5",
        Context::new([Message::assistant(response)]),
        options(|stream| stream.cache_retention = CacheRetention::None),
    )
    .await;

    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "thinking", "thinking": "Think", "signature": "  padded signature  "},
            {"type": "tool_use", "id": "call|same-model", "name": "lookup", "input": {}}
        ])
    );
}

#[tokio::test]
async fn normalizes_anthropic_cross_model_tool_ids() {
    let mut response = assistant_tool_call();
    response.content = vec![AssistantContent::ToolCall(AssistantToolCall {
        id: "call|cross-model".into(),
        name: "get_image".into(),
        arguments: json!({}),
        thought_signature: None,
        namespace: None,
    })];
    response.model = "claude-sonnet-4-4".into();
    let body = request_body_for(
        "claude-sonnet-4-5",
        Context::new([Message::assistant(response)]),
        options(|stream| stream.cache_retention = CacheRetention::None),
    )
    .await;

    assert_eq!(body["messages"][0]["content"][0]["id"], "call_cross-model");
}

#[tokio::test]
async fn encodes_anthropic_adaptive_and_disabled_thinking_by_model() {
    let mut adaptive = options(|_| {});
    adaptive.thinking_enabled = Some(true);
    adaptive.effort = Some(anthropic::Effort::XHigh);
    let adaptive = request_body_for(
        "claude-opus-4-8",
        Context::new([Message::user("Hello")]),
        adaptive,
    )
    .await;
    assert_eq!(
        adaptive["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(adaptive["output_config"], json!({"effort": "xhigh"}));

    let mut legacy = options(|_| {});
    legacy.thinking_enabled = Some(true);
    let legacy = request_body_for(
        "claude-sonnet-4-5",
        Context::new([Message::user("Hello")]),
        legacy,
    )
    .await;
    assert_eq!(
        legacy["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024, "display": "summarized"})
    );
    assert!(legacy.get("output_config").is_none());

    let mut disabled = options(|_| {});
    disabled.thinking_enabled = Some(false);
    let disabled = request_body_for(
        "claude-opus-4-8",
        Context::new([Message::user("Hello")]),
        disabled,
    )
    .await;
    assert_eq!(disabled["thinking"], json!({"type": "disabled"}));
    assert!(disabled.get("output_config").is_none());

    let mut omitted = options(|_| {});
    omitted.thinking_enabled = Some(false);
    let omitted = request_body_for(
        "claude-fable-5",
        Context::new([Message::user("Hello")]),
        omitted,
    )
    .await;
    assert!(omitted.get("thinking").is_none());
}

#[tokio::test]
async fn applies_anthropic_force_adaptive_overrides() {
    let mut custom = model("claude-sonnet-4-5", "");
    custom.id = "vendor--claude-opus-latest".into();
    anthropic_compat(&mut custom).force_adaptive_thinking = Some(true);
    let mut adaptive = options(|_| {});
    adaptive.thinking_enabled = Some(true);
    adaptive.effort = Some(anthropic::Effort::Medium);
    let (custom_body, _) =
        request_for_model(custom, Context::new([Message::user("Hello")]), adaptive).await;
    assert_eq!(
        custom_body["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(custom_body["output_config"], json!({"effort": "medium"}));

    let mut opt_out = model("claude-opus-4-8", "");
    anthropic_compat(&mut opt_out).force_adaptive_thinking = Some(false);
    let mut legacy = options(|_| {});
    legacy.thinking_enabled = Some(true);
    let (legacy_body, _) =
        request_for_model(opt_out, Context::new([Message::user("Hello")]), legacy).await;
    assert_eq!(
        legacy_body["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024, "display": "summarized"})
    );
    assert!(legacy_body.get("output_config").is_none());
}

#[tokio::test]
async fn applies_anthropic_temperature_compatibility() {
    let opus = request_body_for(
        "claude-opus-4-7",
        Context::new([Message::user("Hello")]),
        options(|stream| stream.temperature = Some(0.0)),
    )
    .await;
    assert!(opus.get("temperature").is_none());

    let sonnet = request_body_for(
        "claude-sonnet-4-6",
        Context::new([Message::user("Hello")]),
        options(|stream| stream.temperature = Some(0.0)),
    )
    .await;
    assert_eq!(sonnet["temperature"], 0.0);

    let opus_48 = request_body_for(
        "claude-opus-4-8",
        Context::new([Message::user("Hello")]),
        options(|stream| stream.temperature = Some(1.0)),
    )
    .await;
    assert!(opus_48.get("temperature").is_none());

    let mut custom = model("claude-sonnet-4-5", "");
    anthropic_compat(&mut custom).supports_temperature = Some(false);
    let (custom_body, _) = request_for_model(
        custom,
        Context::new([Message::user("Hello")]),
        options(|stream| stream.temperature = Some(0.0)),
    )
    .await;
    assert!(custom_body.get("temperature").is_none());
}

#[tokio::test]
async fn covers_anthropic_adaptive_thinking_model_and_override_matrix() {
    let adaptive_models = [
        "claude-fable-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
    ];
    for model_id in adaptive_models {
        let mut options = options(|_| {});
        options.thinking_enabled = Some(true);
        options.effort = Some(anthropic::Effort::High);
        let body =
            request_body_for(model_id, Context::new([Message::user("Hello")]), options).await;
        assert_eq!(
            body["thinking"],
            json!({"type": "adaptive", "display": "summarized"}),
            "model {model_id}"
        );
        assert_eq!(body["output_config"], json!({"effort": "high"}));
    }

    let legacy_models = ["claude-haiku-4-5", "claude-opus-4-5", "claude-sonnet-4-5"];
    for model_id in legacy_models {
        let mut options = options(|_| {});
        options.thinking_enabled = Some(true);
        let body =
            request_body_for(model_id, Context::new([Message::user("Hello")]), options).await;
        assert_eq!(
            body["thinking"],
            json!({
                "type": "enabled",
                "budget_tokens": 1024,
                "display": "summarized"
            }),
            "model {model_id}"
        );
        assert!(body.get("output_config").is_none());
    }

    let mut custom = model("claude-sonnet-4-5", "");
    custom.id = "vendor--claude-opus-latest".into();
    custom.compat = None;
    let mut custom_legacy_options = options(|_| {});
    custom_legacy_options.thinking_enabled = Some(true);
    let (body, _) = request_for_model(
        custom,
        Context::new([Message::user("Hello")]),
        custom_legacy_options,
    )
    .await;
    assert_eq!(
        body["thinking"],
        json!({
            "type": "enabled",
            "budget_tokens": 1024,
            "display": "summarized"
        })
    );
    assert!(body.get("output_config").is_none());

    let mut custom_adaptive = model("claude-sonnet-4-5", "");
    custom_adaptive.id = "vendor--claude-opus-latest".into();
    anthropic_compat(&mut custom_adaptive).force_adaptive_thinking = Some(true);
    let mut custom_options = options(|_| {});
    custom_options.thinking_enabled = Some(true);
    custom_options.effort = Some(anthropic::Effort::Medium);
    let (body, _) = request_for_model(
        custom_adaptive,
        Context::new([Message::user("Hello")]),
        custom_options,
    )
    .await;
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(body["output_config"], json!({"effort": "medium"}));

    for model_id in adaptive_models {
        let mut opt_out = model(model_id, "");
        anthropic_compat(&mut opt_out).force_adaptive_thinking = Some(false);
        let mut options = options(|_| {});
        options.thinking_enabled = Some(true);
        let (body, _) =
            request_for_model(opt_out, Context::new([Message::user("Hello")]), options).await;
        assert_eq!(
            body["thinking"],
            json!({
                "type": "enabled",
                "budget_tokens": 1024,
                "display": "summarized"
            }),
            "model {model_id}"
        );
        assert!(body.get("output_config").is_none());
    }
}

#[tokio::test]
async fn covers_anthropic_temperature_compatibility_matrix() {
    let omitted = [
        ("claude-opus-4-7", 0.0),
        ("claude-opus-4-7", 1.0),
        ("claude-opus-4-8", 0.0),
        ("claude-opus-4-8", 1.0),
    ];
    for (model_id, temperature) in omitted {
        let body = request_body_for(
            model_id,
            Context::new([Message::user("Hello")]),
            options(|stream| stream.temperature = Some(temperature)),
        )
        .await;
        assert!(body.get("temperature").is_none(), "model {model_id}");
    }

    let retained = [
        ("claude-fable-5", 0.0),
        ("claude-haiku-4-5", 1.0),
        ("claude-opus-4-5", 0.0),
        ("claude-opus-4-6", 0.0),
        ("claude-sonnet-4-5", 1.0),
        ("claude-sonnet-4-6", 0.0),
    ];
    for (model_id, temperature) in retained {
        let body = request_body_for(
            model_id,
            Context::new([Message::user("Hello")]),
            options(|stream| stream.temperature = Some(temperature)),
        )
        .await;
        assert_eq!(body["temperature"], json!(temperature), "model {model_id}");
    }

    let mut custom = model("claude-sonnet-4-5", "");
    custom.id = "vendor--claude-opus-latest".into();
    custom.compat = None;
    let (body, _) = request_for_model(
        custom,
        Context::new([Message::user("Hello")]),
        options(|stream| stream.temperature = Some(0.0)),
    )
    .await;
    assert_eq!(body["temperature"], 0.0);
}

#[tokio::test]
async fn covers_anthropic_disabled_thinking_matrix() {
    let disabled_models = [
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
    ];
    for model_id in disabled_models {
        let body = request_body_for(model_id, Context::new([Message::user("Hello")]), {
            let mut options = options(|_| {});
            options.thinking_enabled = Some(false);
            options
        })
        .await;
        assert_eq!(
            body["thinking"],
            json!({"type": "disabled"}),
            "model {model_id}"
        );
        assert!(body.get("output_config").is_none());
    }

    let body = request_body_for("claude-fable-5", Context::new([Message::user("Hello")]), {
        let mut options = options(|_| {});
        options.thinking_enabled = Some(false);
        options
    })
    .await;
    assert!(body.get("thinking").is_none());
    assert!(body.get("output_config").is_none());

    for force_adaptive_thinking in [false, true] {
        let mut custom = model("claude-sonnet-4-5", "");
        custom.id = "vendor--claude-opus-latest".into();
        anthropic_compat(&mut custom).force_adaptive_thinking = Some(force_adaptive_thinking);
        let (body, _) = request_for_model(custom, Context::new([Message::user("Hello")]), {
            let mut options = options(|_| {});
            options.thinking_enabled = Some(false);
            options
        })
        .await;
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
        assert!(body.get("output_config").is_none());
    }
}

#[tokio::test]
async fn normalizes_all_canonical_anthropic_oauth_tool_names() {
    let canonical_names = [
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Grep",
        "Glob",
        "AskUserQuestion",
        "EnterPlanMode",
        "ExitPlanMode",
        "KillShell",
        "NotebookEdit",
        "Skill",
        "Task",
        "TaskOutput",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
    ];
    let tool_response = |name: &str| {
        format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_tool\",\"usage\":{{}}}}}}\n\nevent: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"{name}\",\"input\":{{}}}}}}\n\nevent: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\nevent: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"tool_use\"}},\"usage\":{{}}}}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
        )
    };
    let server = serve(
        canonical_names
            .iter()
            .map(|name| Reply::sse(tool_response(name))),
    )
    .await;

    for canonical_name in canonical_names {
        let original_name = canonical_name.to_ascii_lowercase();
        let model = model("claude-sonnet-4-6", &server.base_url);
        let context = Context::new([Message::user("Use the tool")]).with_tools([Tool::new(
            &original_name,
            "Test tool",
            json!({"type": "object"}),
        )]);
        let mut options = options(|_| {});
        options.stream.api_key = Some("sk-ant-oat-test".into());
        let stream_events = events(&model, &context, &options).await;
        let response = done(&stream_events);
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(
            response.content[0],
            AssistantContent::ToolCall(AssistantToolCall {
                id: "toolu_1".into(),
                name: original_name,
                arguments: json!({}),
                thought_signature: None,
                namespace: None,
            })
        );
    }

    let requests = server.requests().await;
    assert_eq!(requests.len(), canonical_names.len());
    for (request, canonical_name) in requests.iter().zip(canonical_names) {
        let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["tools"][0]["name"], canonical_name);
    }
}

#[tokio::test]
async fn preserves_noncanonical_anthropic_oauth_tool_names() {
    let names = ["find", "my_custom_tool"];
    let tool_response = |name: &str| {
        format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_tool\",\"usage\":{{}}}}}}\n\nevent: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"{name}\",\"input\":{{}}}}}}\n\nevent: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\nevent: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"tool_use\"}},\"usage\":{{}}}}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
        )
    };
    let server = serve(names.iter().map(|name| Reply::sse(tool_response(name)))).await;

    for name in names {
        let model = model("claude-sonnet-4-6", &server.base_url);
        let context = Context::new([Message::user("Use the tool")]).with_tools([Tool::new(
            name,
            "Test tool",
            json!({"type": "object"}),
        )]);
        let mut options = options(|_| {});
        options.stream.api_key = Some("sk-ant-oat-test".into());
        let events = events(&model, &context, &options).await;
        let response = done(&events);

        assert_eq!(response.stop_reason, StopReason::ToolUse);
        let AssistantContent::ToolCall(call) = &response.content[0] else {
            panic!("expected a tool call");
        };
        assert_eq!(call.name, name);
    }

    let requests = server.requests().await;
    assert_eq!(requests.len(), names.len());
    for (request, name) in requests.iter().zip(names) {
        let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["tools"][0]["name"], name);
    }
}

#[tokio::test]
async fn batches_anthropic_deferred_tool_results_before_sibling_content() {
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::ToolCall(AssistantToolCall {
                id: "call_1".into(),
                name: "base_tool".into(),
                arguments: json!({}),
                thought_signature: None,
                namespace: None,
            }),
            AssistantContent::ToolCall(AssistantToolCall {
                id: "call_2".into(),
                name: "base_tool_2".into(),
                arguments: json!({}),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: Api::AnthropicMessages,
        provider: "anthropic".into(),
        model: "claude-opus-4-6".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: Some("tool_use".into()),
        end_turn: None,
        timestamp: 0,
    };
    let mut first =
        ToolResultMessage::new("call_1", "base_tool", [InputContent::text("first sibling")]);
    first.added_tool_names = Some(vec!["late_tool_1".into()]);
    let mut second = ToolResultMessage::new(
        "call_2",
        "base_tool_2",
        [InputContent::text("second sibling")],
    );
    second.added_tool_names = Some(vec!["late_tool_2".into()]);
    let context = Context::new([
        Message::user("Use tools"),
        Message::assistant(assistant),
        Message::tool_result(first),
        Message::tool_result(second),
    ])
    .with_tools([
        Tool::new("base_tool", "Base", json!({"type": "object"})),
        Tool::new("base_tool_2", "Base 2", json!({"type": "object"})),
        Tool::new("late_tool_1", "Late 1", json!({"type": "object"})),
        Tool::new("late_tool_2", "Late 2", json!({"type": "object"})),
    ]);
    let body = request_body_for(
        "claude-opus-4-6",
        context,
        options(|stream| stream.cache_retention = CacheRetention::None),
    )
    .await;
    assert_eq!(
        body["messages"][2]["content"],
        json!([
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [{"type": "tool_reference", "tool_name": "late_tool_1"}],
                "is_error": false
            },
            {
                "type": "tool_result",
                "tool_use_id": "call_2",
                "content": [{"type": "tool_reference", "tool_name": "late_tool_2"}],
                "is_error": false
            },
            {"type": "text", "text": "first sibling"},
            {"type": "text", "text": "second sibling"}
        ])
    );
}

#[tokio::test]
async fn preserves_emoji_in_anthropic_tool_results() {
    let assistant = assistant_tool_call();
    let body = request_body_for(
        "claude-sonnet-4-5",
        Context::new([
            Message::assistant(assistant),
            Message::tool_result(ToolResultMessage::new(
                "toolu_image",
                "get_image",
                [InputContent::text("done 😀")],
            )),
        ]),
        options(|_| {}),
    )
    .await;
    assert_eq!(
        body["messages"][1]["content"][0]["content"],
        json!("done 😀")
    );
}

#[tokio::test]
async fn encodes_anthropic_image_only_and_mixed_tool_results() {
    let assistant = assistant_tool_call();
    let image_only = request_body_for(
        "claude-sonnet-4-5",
        Context::new([
            Message::assistant(assistant.clone()),
            Message::tool_result(ToolResultMessage::new(
                "toolu_image",
                "get_image",
                [InputContent::image("image/png", "iVBORw0KGgo=")],
            )),
        ]),
        options(|_| {}),
    )
    .await;
    assert_eq!(
        image_only["messages"][1]["content"],
        json!([{
            "type": "tool_result",
            "tool_use_id": "toolu_image",
            "content": [{
                "type": "text",
                "text": "(see attached image)"
            }, {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgo="
                }
            }],
            "is_error": false,
            "cache_control": {"type": "ephemeral"}
        }])
    );

    let mixed = request_body_for(
        "claude-sonnet-4-5",
        Context::new([
            Message::assistant(assistant),
            Message::tool_result(ToolResultMessage::new(
                "toolu_image",
                "get_image",
                [
                    InputContent::text("diameter: 100 pixels"),
                    InputContent::image("image/png", "iVBORw0KGgo="),
                ],
            )),
        ]),
        options(|_| {}),
    )
    .await;
    assert_eq!(
        mixed["messages"][1]["content"],
        json!([{
            "type": "tool_result",
            "tool_use_id": "toolu_image",
            "content": [
                {"type": "text", "text": "diameter: 100 pixels"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo="
                    }
                }
            ],
            "is_error": false,
            "cache_control": {"type": "ephemeral"}
        }])
    );
}

#[tokio::test]
async fn encodes_anthropic_eager_legacy_and_strict_tool_modes() {
    let tool = Tool::new(
        "lookup",
        "Look up a value",
        json!({"type": "object", "properties": {"value": {"type": "string"}}}),
    );
    let (eager, eager_raw) = request_for_model(
        model("claude-sonnet-4-5", ""),
        Context::new([Message::user("Use the tool")]).with_tools([tool.clone()]),
        options(|_| {}),
    )
    .await;
    assert_eq!(eager["tools"][0]["eager_input_streaming"], true);
    assert!(!eager_raw.contains("fine-grained-tool-streaming-2025-05-14"));

    let mut legacy_model = model("claude-sonnet-4-5", "");
    anthropic_compat(&mut legacy_model).supports_eager_tool_input_streaming = Some(false);
    let (legacy, legacy_raw) = request_for_model(
        legacy_model.clone(),
        Context::new([Message::user("Use the tool")]).with_tools([tool.clone()]),
        options(|_| {}),
    )
    .await;
    assert!(legacy["tools"][0].get("eager_input_streaming").is_none());
    assert!(legacy_raw.contains("fine-grained-tool-streaming-2025-05-14"));

    let (no_tools, no_tools_raw) = request_for_model(
        legacy_model,
        Context::new([Message::user("No tool")]),
        options(|_| {}),
    )
    .await;
    assert!(no_tools.get("tools").is_none());
    assert!(!no_tools_raw.contains("fine-grained-tool-streaming-2025-05-14"));

    let strict_model = model("claude-opus-4-8", "");
    let (non_strict, _) = request_for_model(
        strict_model.clone(),
        Context::new([Message::user("Use the tool")]).with_tools([tool]),
        options(|_| {}),
    )
    .await;
    assert!(non_strict["tools"][0].get("strict").is_none());

    let strict_tool = Tool::new(
        "lookup",
        "Look up a value",
        json!({"type": "object", "properties": {"value": {"type": "string"}}}),
    )
    .with_strict();
    let (strict, _) = request_for_model(
        strict_model,
        Context::new([Message::user("Use the tool")]).with_tools([strict_tool]),
        options(|_| {}),
    )
    .await;
    assert_eq!(strict["tools"][0]["strict"], true);
}

#[tokio::test]
async fn prices_anthropic_standard_and_fallback_one_hour_cache_writes() {
    let standard = cost_response(
        "claude-opus-4-8",
        Some(json!({
            "ephemeral_5m_input_tokens": 600_000,
            "ephemeral_1h_input_tokens": 400_000
        })),
    )
    .await;
    assert_eq!(standard.usage.cache_write, 1_000_000);
    assert_eq!(standard.usage.cache_write_1h, Some(400_000));
    assert!((standard.usage.cost.cache_write - 7.75).abs() < f64::EPSILON);

    let fallback = cost_response(
        "claude-fable-5",
        Some(json!({
            "ephemeral_5m_input_tokens": 600_000,
            "ephemeral_1h_input_tokens": 400_000
        })),
    )
    .await;
    assert_eq!(fallback.response_model.as_deref(), Some("claude-opus-4-8"));
    assert_eq!(fallback.usage.cache_write_1h, Some(400_000));
    assert!((fallback.usage.cost.cache_write - 7.75).abs() < f64::EPSILON);

    let no_breakdown = cost_response("claude-opus-4-8", None).await;
    assert_eq!(no_breakdown.usage.cache_write, 1_000_000);
    assert_eq!(no_breakdown.usage.cache_write_1h, Some(0));
    assert!((no_breakdown.usage.cost.cache_write - 6.25).abs() < f64::EPSILON);
}

#[tokio::test]
async fn accepts_anthropic_message_delta_without_usage_and_ignores_late_events() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_usage\",\"usage\":{\"input_tokens\":4,\"output_tokens\":2}}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":null}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        "event: unknown\ndata: {\"type\":\"unknown\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);

    let events = events(
        &model,
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;
    let response = done(&events);

    assert_eq!(response.usage.input, 4);
    assert_eq!(response.usage.output, 2);
    assert_eq!(response.usage.total_tokens, 6);
    assert_eq!(response.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn preserves_valid_anthropic_emoji_text() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_emoji\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello 😀\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("😀")]),
        &options(|_| {}),
    )
    .await;
    let response = done(&events);
    assert_eq!(response.content, [text("Hello 😀")]);
}

#[tokio::test]
async fn does_not_expose_failed_anthropic_responses_to_the_response_hook() {
    let server = serve([Reply::json(
        429,
        json!({"error": {"type": "rate_limit_error", "message": "Too many requests"}}),
    )
    .with_header("request-id", "req_anthropic_failure")
    .with_header("anthropic-ratelimit-requests-limit", "100")
    .with_header("anthropic-ratelimit-requests-remaining", "0")
    .with_header("anthropic-ratelimit-tokens-limit", "10000")
    .with_header("anthropic-ratelimit-tokens-remaining", "200")])
    .await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let responses = Arc::new(Mutex::new(Vec::new()));
    let captured = responses.clone();
    let mut options = options(|_| {});
    options.stream.on_response = Some(ResponseHook::new(move |response, _| {
        let captured = captured.clone();
        async move {
            captured.lock().unwrap().push(response);
            Ok(())
        }
    }));
    let context = Context::new([Message::user("Hello")]);

    let events = events(&model, &context, &options).await;
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some(r#"429 {"error":{"type":"rate_limit_error","message":"Too many requests"}}"#)
    );
    assert!(responses.lock().unwrap().is_empty());
    server.requests().await;
}

#[tokio::test]
async fn streams_and_replays_anthropic_thinking_and_tool_calls() {
    let first_sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_thinking\",\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"I\",\"signature\":\"initial signature\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" think\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\" plus delta\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"encrypted\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"edit\",\"input\":{}}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"README.md\\\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"}\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8,\"output_tokens_details\":{\"thinking_tokens\":3}}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(first_sse)]).await;
    let first_model = model("claude-sonnet-4-5", &first_server.base_url);
    let first_context = Context::new([Message::user("Edit")]);
    let options = options(|_| {});

    let first_events = events(&first_model, &first_context, &options).await;

    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial,
        } if matches!(
            partial.content.first(),
            Some(AssistantContent::Thinking(ThinkingContent { thinking, .. })) if thinking == "I"
        )
    )));
    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta,
            ..
        } if delta == " think"
    )));
    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ToolCallDelta {
            content_index: 2,
            delta,
            ..
        } if delta == "{\"path\":\"README.md\""
    )));
    let response = done(&first_events);
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.usage.reasoning, Some(3));
    assert_eq!(response.usage.total_tokens, 12);
    assert_eq!(
        response.content,
        [
            AssistantContent::Thinking(ThinkingContent {
                thinking: "I think".into(),
                thinking_signature: Some("initial signature plus delta".into()),
                redacted: None,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "[Reasoning redacted]".into(),
                thinking_signature: Some("encrypted".into()),
                redacted: Some(true),
            }),
            AssistantContent::ToolCall(AssistantToolCall {
                id: "toolu_1".into(),
                name: "edit".into(),
                arguments: json!({"path": "README.md"}),
                thought_signature: None,
                namespace: None,
            }),
        ]
    );
    let restored: AssistantMessage =
        serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap();
    first_server.requests().await;

    let second_sse = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model = model("claude-sonnet-4-5", &second_server.base_url);
    let second_context = Context::new([
        Message::assistant(restored),
        Message::tool_result(ToolResultMessage::new(
            "toolu_1",
            "edit",
            [InputContent::text("done")],
        )),
    ]);

    let second_events = events(&second_model, &second_context, &options).await;
    done(&second_events);

    let request = second_server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["messages"],
        json!([
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I think", "signature": "initial signature plus delta"},
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "edit",
                        "input": {"path": "README.md"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "done",
                    "is_error": false,
                    "cache_control": {"type": "ephemeral"}
                }]
            }
        ])
    );
}

#[tokio::test]
async fn defaults_anthropic_tool_input_to_an_object() {
    let sse = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_input\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_input\",\"name\":\"lookup\",\"input\":\"not an object\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Use lookup")]),
        &options(|_| {}),
    )
    .await;

    assert_eq!(
        done(&events).content[0],
        AssistantContent::ToolCall(AssistantToolCall {
            id: "toolu_input".into(),
            name: "lookup".into(),
            arguments: json!({}),
            thought_signature: None,
            namespace: None,
        })
    );
}

#[tokio::test]
async fn preserves_anthropic_start_content_and_refusal_details() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_refusal\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Blocked\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" request\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"type\":\"refusal\",\"category\":\"policy\",\"explanation\":\"Request denied\"}},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-fable-5", &server.base_url);
    let context = Context::new([Message::user("Blocked request")]);
    let events = events(&model, &context, &options(|_| {})).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial,
        } if partial.content == [text("Blocked")]
    )));
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.raw_stop_reason.as_deref(), Some("refusal"));
    assert_eq!(error.error_message.as_deref(), Some("Request denied"));
    assert_eq!(error.content, [text("Blocked request")]);
}

#[tokio::test]
async fn rejects_sensitive_and_unknown_anthropic_stop_reasons() {
    for (reason, expected) in [
        ("sensitive", "Provider stopped with: sensitive"),
        ("new_reason", "Unhandled stop reason: new_reason"),
    ] {
        let sse = [
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"",
            reason,
            "\"},\"usage\":{}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]
        .concat();
        let server = serve([Reply::sse(sse)]).await;
        let model = model("claude-haiku-4-5", &server.base_url);
        let context = Context::new([Message::user("Blocked request")]);
        let events = events(&model, &context, &options(|_| {})).await;
        let error = failed(&events);

        assert_eq!(error.stop_reason, StopReason::Error);
        assert_eq!(error.raw_stop_reason.as_deref(), Some(reason));
        assert_eq!(error.error_message.as_deref(), Some(expected));
    }
}

#[tokio::test]
async fn retries_anthropic_before_streaming_starts() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([
        Reply::json(
            529,
            json!({"error": {"type": "overloaded_error", "message": "busy"}}),
        )
        .with_header("retry-after-ms", "0"),
        Reply::sse(completed),
    ])
    .await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|stream| stream.max_retries = Some(1));

    let events = events(&model, &context, &options).await;

    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test]
async fn cancels_an_active_anthropic_stream_with_partial_content() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_cancel\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Visible\"}}\n\n",
    ]
    .concat();
    let server = serve([Reply::open_sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|stream| stream.cancellation = cancellation.clone());
    let mut response = anthropic::stream(
        &model.typed::<ds_ai::AnthropicOptions>().unwrap(),
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
            assert_eq!(reason, StopReason::Aborted);
            assert_eq!(error.stop_reason, StopReason::Aborted);
            assert_eq!(error.error_message.as_deref(), Some("Request was aborted"));
            assert_eq!(error.content, [text("Visible")]);
        }
        event => panic!("unexpected cancellation event: {event:?}"),
    }
}

#[tokio::test]
async fn times_out_an_anthropic_request_before_response_headers_per_attempt() {
    let server = serve([Reply::pending()]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|stream| {
        stream.timeout = Some(std::time::Duration::from_millis(25));
    });
    let events = events(&model, &context, &options).await;
    assert_eq!(
        failed(&events).error_message.as_deref(),
        Some("provider timed out during Connection")
    );
}

#[tokio::test]
async fn does_not_apply_an_anthropic_timeout_to_an_error_body() {
    let server = serve([Reply::open_json(
        500,
        json!({"error": {"message": "unfinished"}}),
    )])
    .await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|stream| {
        stream.timeout = Some(std::time::Duration::from_millis(10));
        stream.cancellation = cancellation.clone();
    });
    let request = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancellation.cancel();

    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.error_message.as_deref(), Some("Request was aborted"));
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert!(error.content.is_empty());
    server.requests().await;
}

#[tokio::test]
async fn accepts_an_anthropic_stream_body_after_the_header_timeout() {
    let server = serve([Reply::open_sse(Vec::new())]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|stream| {
        stream.timeout = Some(std::time::Duration::from_secs(5));
        stream.cancellation = cancellation.clone();
    });
    let mut response = anthropic::stream(
        &model.typed::<ds_ai::AnthropicOptions>().unwrap(),
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
            assert_eq!(reason, StopReason::Aborted);
            assert_eq!(error.stop_reason, StopReason::Aborted);
            assert_eq!(error.raw_stop_reason.as_deref(), Some("cancelled"));
            assert_eq!(error.error_message.as_deref(), Some("Request was aborted"));
            assert!(error.content.is_empty());
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[tokio::test]
async fn retries_an_anthropic_header_timeout_with_a_fresh_timeout() {
    let sse = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let server = serve([Reply::pending(), Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|stream| {
        stream.timeout = Some(std::time::Duration::from_millis(25));
        stream.max_retries = Some(1);
        stream.max_retry_delay = Some(std::time::Duration::ZERO);
    });
    let streamed = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        events(&model, &context, &options),
    )
    .await
    .expect("retry should complete");
    done(&streamed);
    assert_eq!(server.request_count(), 2);
    server.requests().await;
}

#[tokio::test]
async fn uses_the_sdk_zero_timeout_behavior_for_anthropic() {
    let server = serve([Reply::pending()]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|stream| {
        stream.timeout = Some(std::time::Duration::ZERO);
    });
    let events = events(&model, &context, &options).await;
    assert_eq!(
        failed(&events).error_message.as_deref(),
        Some("provider timed out during Connection")
    );
}

#[tokio::test]
async fn maps_anthropic_pause_turn_to_stop() {
    let sse = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"pause_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let response = done(&events);
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(response.raw_stop_reason.as_deref(), Some("pause_turn"));
}

#[tokio::test]
async fn rejects_anthropic_stream_closure_before_message_stop() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_partial\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Partial\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let error = failed(&events);
    assert_eq!(error.response_id.as_deref(), Some("msg_partial"));
    assert_eq!(error.content, [text("Partial")]);
    assert_eq!(error.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(
        error.error_message.as_deref(),
        Some("Anthropic stream ended before message_stop")
    );
}

#[tokio::test]
async fn distinguishes_anthropic_missing_start_stop_and_reason() {
    let no_reason = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_no_reason\",\"usage\":{}}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    let empty_reason = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_empty_reason\",\"usage\":{}}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    for (body, expected) in [
        ("", "Anthropic stream ended without a stop reason"),
        (no_reason, "Anthropic stream ended without a stop reason"),
        (empty_reason, "Anthropic stream ended without a stop reason"),
    ] {
        let server = serve([Reply::sse(body)]).await;
        let model = model("claude-sonnet-4-5", &server.base_url);
        let streamed = events(
            &model,
            &Context::new([Message::user("Hello")]),
            &options(|_| {}),
        )
        .await;

        assert_eq!(failed(&streamed).error_message.as_deref(), Some(expected));
        server.requests().await;
    }
}

#[tokio::test]
async fn rejects_an_anthropic_error_event_with_partial_content() {
    let error_data = r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#;
    let sse = format!(
        concat!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_error\",\"usage\":{{}}}}}}\n\n",
            "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"Visible\"}}}}\n\n",
            "event: error\ndata: {0}\n\n"
        ),
        error_data
    );
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.raw_stop_reason, None);
    assert_eq!(error.error_message.as_deref(), Some(error_data));
    assert_eq!(error.content, [text("Visible")]);
}

#[tokio::test]
async fn ignores_unknown_anthropic_events_before_message_stop() {
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_unknown\",\"usage\":{}}}\n\n",
        "event: proxy.stats\ndata: not json\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Visible\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let response = done(&events);
    assert_eq!(response.stop_reason, StopReason::Stop);
    assert_eq!(response.content, [text("Visible")]);
}

#[tokio::test]
async fn encodes_anthropic_generation_thinking_and_cache_options() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    let model = model("claude-opus-4-8", &server.base_url);
    let context = Context::new([Message::user_content([
        InputContent::text("Inspect"),
        InputContent::image("image/png", "iVBORw0KGgo="),
    ])])
    .with_system("Be brief")
    .with_tools([Tool::new(
        "inspect",
        "Inspect the input",
        json!({"type": "object", "properties": {}}),
    )]);
    let mut options = options(|stream| {
        stream.temperature = Some(0.2);
        stream.max_tokens = Some(4096);
        stream.cache_retention = CacheRetention::Long;
        stream.metadata.insert("user_id".into(), json!("user_1"));
    });
    options.thinking_enabled = Some(true);
    options.effort = Some(anthropic::Effort::High);
    options.thinking_display = Some(anthropic::ThinkingDisplay::Summarized);
    options.tool_choice = Some(anthropic::ToolChoice::Tool("inspect".into()));

    let response = events(&model, &context, &options).await;
    done(&response);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "claude-opus-4-8",
            "system": [{
                "type": "text",
                "text": "Be brief",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    }
                ]
            }],
            "tools": [{
                "name": "inspect",
                "description": "Inspect the input",
                "eager_input_streaming": true,
                "input_schema": {"type": "object", "properties": {}, "required": []},
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }],
            "max_tokens": 4096,
            "stream": true,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"},
            "metadata": {"user_id": "user_1"},
            "tool_choice": {"type": "tool", "name": "inspect"}
        })
    );
    assert!(body.get("temperature").is_none());
}

#[tokio::test]
async fn keeps_anthropic_temperature_with_disabled_thinking_and_cache() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    let mut model = model("claude-sonnet-4-5", &server.base_url);
    anthropic_compat(&mut model).supports_eager_tool_input_streaming = Some(false);
    let context = Context::new([Message::user("Hello")]).with_system("Be brief");
    let mut options = options(|stream| {
        stream.temperature = Some(0.0);
        stream.cache_retention = CacheRetention::None;
    });
    options.thinking_enabled = Some(false);

    let response = events(&model, &context, &options).await;
    done(&response);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("output_config").is_none());
    assert!(!request.contains("cache_control"));
    assert!(request.contains("anthropic-beta: interleaved-thinking-2025-05-14\r\n"));
}

#[tokio::test]
async fn encodes_legacy_tool_streaming_and_strict_schemas() {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    let mut model = model("claude-opus-4-8", &server.base_url);
    let compat = anthropic_compat(&mut model);
    compat.supports_eager_tool_input_streaming = Some(false);
    compat.supports_strict_tools = Some(true);
    let context = Context::new([Message::user("Look up")]).with_tools([Tool::new(
        "lookup",
        "Look up a value",
        json!({
            "type": "object",
            "title": "LookupInput",
            "properties": {
                "value": {"type": "string"},
                "optional": {"type": "number"}
            },
            "required": ["value"]
        }),
    )
    .with_strict()]);

    let response = events(&model, &context, &options(|_| {})).await;
    done(&response);

    let request = server.requests().await.pop().unwrap();
    assert!(request.contains("anthropic-beta: fine-grained-tool-streaming-2025-05-14\r\n"));
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["tools"],
        json!([{
            "name": "lookup",
            "description": "Look up a value",
            "strict": true,
            "input_schema": {
                "type": "object",
                "title": "LookupInput",
                "properties": {
                    "value": {"type": "string"},
                    "optional": {"anyOf": [{"type": "number"}, {"type": "null"}]}
                },
                "required": ["value", "optional"],
                "additionalProperties": false
            },
            "cache_control": {"type": "ephemeral"}
        }])
    );
}

#[tokio::test]
async fn replays_empty_signature_thinking_as_text_unless_enabled() {
    let thinking = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_unsigned\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"unsigned\",\"signature\":\"\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"signed\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(thinking)]).await;
    let first_model = model("claude-sonnet-4-5", &first_server.base_url);
    let context = Context::new([Message::user("Think")]);
    let first_events = events(&first_model, &context, &options(|_| {})).await;
    let response = done(&first_events).clone();

    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let default_server = serve([Reply::sse(completed.clone())]).await;
    let default_model = model("claude-sonnet-4-5", &default_server.base_url);
    let default_events = events(
        &default_model,
        &Context::new([Message::assistant(response.clone())]),
        &options(|_| {}),
    )
    .await;
    done(&default_events);
    let default_request = default_server.requests().await.pop().unwrap();
    let default_body: Value =
        serde_json::from_str(default_request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        default_body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "unsigned"},
            {"type": "thinking", "thinking": "", "signature": "signed"}
        ])
    );

    let enabled_server = serve([Reply::sse(completed)]).await;
    let mut enabled_model = model("claude-sonnet-4-5", &enabled_server.base_url);
    anthropic_compat(&mut enabled_model).allow_empty_signature = Some(true);
    let enabled_events = events(
        &enabled_model,
        &Context::new([Message::assistant(response)]),
        &options(|_| {}),
    )
    .await;
    done(&enabled_events);
    let enabled_request = enabled_server.requests().await.pop().unwrap();
    let enabled_body: Value =
        serde_json::from_str(enabled_request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        enabled_body["messages"][0]["content"],
        json!([
            {"type": "thinking", "thinking": "unsigned", "signature": ""},
            {"type": "thinking", "thinking": "", "signature": "signed"}
        ])
    );
}

#[tokio::test]
async fn reports_anthropic_sse_parse_diagnostics_with_raw_lines() {
    let sse = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_invalid\",\"usage\":{}}}\n\n",
        "event: message_delta\ndata: not json\n\n"
    );
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;

    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some(
            "Could not parse Anthropic SSE event message_delta: expected ident at line 1 column 2; data=not json; raw=event: message_delta\\ndata: not json"
        )
    );
}

#[tokio::test]
async fn repairs_malformed_anthropic_event_and_tool_json() {
    let malformed = r#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1	col2\",\"unicode\":\"\u12xz\"}"}}

"#;
    let sse = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_repair\",\"usage\":{}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_repair\",\"name\":\"edit\",\"input\":{}}}\n\n",
        malformed,
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let context = Context::new([Message::user("Edit")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    let response = done(&events);
    assert_eq!(
        response.content,
        [AssistantContent::ToolCall(AssistantToolCall {
            id: "toolu_repair".into(),
            name: "edit".into(),
            arguments: json!({"path": "A\\H", "text": "col1\tcol2", "unicode": "\\u12xz"}),
            thought_signature: None,
            namespace: None,
        })]
    );
}

fn model(id: &str, base_url: &str) -> ds_ai::Model {
    let mut model = builtin_model("anthropic", id).unwrap();
    model.base_url = base_url.into();
    model
}

fn assistant_tool_call() -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::ToolCall(AssistantToolCall {
            id: "toolu_image".into(),
            name: "get_image".into(),
            arguments: json!({}),
            thought_signature: None,
            namespace: None,
        })],
        api: Api::AnthropicMessages,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-5".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        raw_stop_reason: Some("tool_use".into()),
        end_turn: None,
        timestamp: 0,
    }
}

fn anthropic_compat(model: &mut ds_ai::Model) -> &mut AnthropicMessagesCompatibility {
    if !matches!(model.compat, Some(ModelCompatibility::Anthropic(_))) {
        model.compat = Some(ModelCompatibility::Anthropic(Default::default()));
    }
    let Some(ModelCompatibility::Anthropic(compat)) = &mut model.compat else {
        unreachable!()
    };
    compat
}

fn options(configure: impl FnOnce(&mut StreamOptions)) -> AnthropicOptions {
    let mut stream = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    configure(&mut stream);
    AnthropicOptions {
        stream,
        ..Default::default()
    }
}

async fn auth_request(
    env: impl IntoIterator<Item = (&'static str, &'static str)>,
    headers: BTreeMap<String, Option<String>>,
    api_key: Option<&str>,
) -> String {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    let model = model("claude-sonnet-4-5", &server.base_url);
    let auth_context = TestAuthContext {
        env: env
            .into_iter()
            .map(|(name, value)| (name.into(), value.into()))
            .collect(),
    };
    let mut models = Models::with_auth(
        Arc::new(ds_ai::InMemoryCredentialStore::new()),
        Arc::new(auth_context),
    );
    models.set_provider(Arc::new(anthropic::Provider::new([model.clone()])));
    let options = SimpleStreamOptions {
        stream: StreamOptions {
            api_key: api_key.map(str::to_owned),
            headers,
            ..Default::default()
        },
        ..Default::default()
    };
    models
        .complete_simple(
            &model.typed::<ds_ai::AnthropicOptions>().unwrap(),
            &Context::new([Message::user("Hello")]),
            &options,
        )
        .await
        .unwrap();
    server.requests().await.pop().unwrap()
}

async fn request_body(context: Context) -> Value {
    request_body_for("claude-sonnet-4-5", context, options(|_| {})).await
}

async fn request_body_for(model_id: &str, context: Context, options: AnthropicOptions) -> Value {
    request_for_model(model(model_id, ""), context, options)
        .await
        .0
}

async fn request_for_model(
    mut model: ds_ai::Model,
    context: Context,
    options: AnthropicOptions,
) -> (Value, String) {
    let completed = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(completed)]).await;
    model.base_url = server.base_url.clone();
    done(&events(&model, &context, &options).await);
    let request = server.requests().await.pop().unwrap();
    let body = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    (body, request)
}

async fn cost_response(model_id: &str, cache_creation: Option<Value>) -> AssistantMessage {
    let mut usage = json!({
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 1_000_000
    });
    if let Some(cache_creation) = cache_creation {
        usage["cache_creation"] = cache_creation;
    }
    let sse = format!(
        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_cost\",\"model\":\"claude-opus-4-8\",\"usage\":{usage}}}}}\n\nevent: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":5}}}}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
    );
    let server = serve([Reply::sse(sse)]).await;
    let model = model(model_id, &server.base_url);
    let response = done(
        &events(
            &model,
            &Context::new([Message::user("Hello")]),
            &options(|_| {}),
        )
        .await,
    )
    .clone();
    server.requests().await;
    response
}

struct TestAuthContext {
    env: BTreeMap<String, String>,
}

#[async_trait]
impl AuthContext for TestAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        self.env.get(name).cloned()
    }

    async fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

async fn events(
    model: &ds_ai::Model,
    context: &Context,
    options: &AnthropicOptions,
) -> Vec<AssistantMessageEvent> {
    anthropic::stream(
        &model.typed::<ds_ai::AnthropicOptions>().unwrap(),
        context,
        options,
    )
    .collect()
    .await
}

fn text(value: &str) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: value.into(),
        text_signature: None,
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
