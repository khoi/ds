use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{
    AnthropicMessagesCompatibility, AnthropicOptions, ApiStreamOptions, AssistantContent,
    ConstrainedSampling, ConstrainedSamplingStrictness, Context, GrammarVariants, InputContent,
    Message, ModelCompatibility, OpenAiCodexResponsesOptions, OpenAiResponsesCompatibility,
    OpenAiResponsesOptions, Provider, StopReason, StreamOptions, Tool, ToolResultMessage,
    Transport, builtin_model,
};
use futures_util::StreamExt;
use serde_json::{Value, json};

#[tokio::test]
async fn encodes_streams_and_replays_openai_constrained_tools() {
    let server = serve([Reply::sse(custom_tool_events()), Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_strict_mode: Some(true),
        supports_open_ai_grammar_tools: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let context = Context::new([Message::user("Use tools")]).with_tools(tools());
    let options = ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    });

    let mut stream = provider.stream(&model, &context, &options);
    let mut deltas = Vec::new();
    while let Some(event) = stream.next().await {
        if let ds_ai::AssistantMessageEvent::ToolCallDelta { delta, .. } = event {
            deltas.push(delta);
        }
    }
    let message = stream.result().await.unwrap();
    let delta_json = deltas.iter().map(String::as_str).collect::<String>();
    assert_eq!(
        serde_json::from_str::<Value>(&delta_json).unwrap(),
        json!({"payload": "a\"\nb"})
    );
    assert_eq!(deltas.len(), 2);
    assert_eq!(
        deltas,
        vec![String::from(r#"{"payload":"a\""#), String::from(r#"\nb"}"#),]
    );
    let AssistantContent::ToolCall(call) = &message.content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(call.id, "call_1|ctc_1");
    assert_eq!(call.arguments, json!({"payload": "a\"\nb"}));
    let call_id = call.id.clone();
    let call_name = call.name.clone();

    let replay = Context::new([
        Message::assistant(message),
        Message::tool_result(ToolResultMessage::new(
            call_id,
            call_name,
            [InputContent::text("done")],
        )),
    ])
    .with_tools(tools());
    provider
        .stream(&model, &replay, &options)
        .result()
        .await
        .unwrap();

    let requests = server.requests().await;
    let first = request_json(&requests[0]);
    assert_eq!(first["tools"][0]["strict"], false);
    assert_eq!(first["tools"][1]["strict"], true);
    assert_eq!(first["tools"][2]["type"], "custom");
    assert_eq!(
        first["tools"][2]["format"],
        json!({"type": "grammar", "syntax": "lark", "definition": "start: /[a-z]+/"})
    );
    let second = request_json(&requests[1]);
    assert!(second["input"].as_array().unwrap().contains(&json!({
        "type": "custom_tool_call",
        "call_id": "call_1",
        "id": "ctc_1",
        "name": "grammar",
        "input": "a\"\nb"
    })));
    assert!(second["input"].as_array().unwrap().contains(&json!({
        "type": "custom_tool_call_output",
        "call_id": "call_1",
        "output": "done"
    })));
}

#[tokio::test]
async fn rejects_required_strict_sampling_when_openai_does_not_support_it() {
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_strict_mode: Some(false),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Use a tool")]).with_tools([strict_tool()]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("strict tools are unsupported")
    );
}

#[tokio::test]
async fn falls_back_from_preferred_strict_sampling_without_mutating_the_tool() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_strict_mode: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let tool = Tool {
        name: "lookup".into(),
        description: "Lookup".into(),
        parameters: json!({
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "$defs": {"value": {"type": "string"}}
        }),
        constrained_sampling: Some(ConstrainedSampling::JsonSchema {
            strict: ConstrainedSamplingStrictness::Prefer,
        }),
    };
    let original = tool.clone();

    provider
        .stream(
            &model,
            &Context::new([Message::user("Use a tool")]).with_tools([tool.clone()]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let payload = request_json(&server.requests().await[0]);
    assert_eq!(payload["tools"][0]["strict"], false);
    assert_eq!(payload["tools"][0]["parameters"], tool.parameters);
    assert_eq!(tool, original);
}

#[tokio::test]
async fn normalizes_nested_optional_properties_for_strict_sampling() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_strict_mode: Some(true),
        ..Default::default()
    }));
    let tool = Tool {
        name: "nested".into(),
        description: "Nested".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}}
                },
                "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            },
            "required": ["path", "metadata"]
        }),
        constrained_sampling: Some(ConstrainedSampling::JsonSchema {
            strict: ConstrainedSamplingStrictness::Prefer,
        }),
    };
    let original = tool.parameters.clone();
    let provider = ds_ai::openai::Provider::new([model.clone()]);

    provider
        .stream(
            &model,
            &Context::new([Message::user("Use the nested tool")]).with_tools([tool.clone()]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let payload = request_json(&server.requests().await[0]);
    assert_eq!(
        payload["tools"][0]["parameters"],
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "enabled": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
                    },
                    "required": ["enabled"],
                    "additionalProperties": false
                },
                "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            },
            "required": ["path", "metadata", "nullable"],
            "additionalProperties": false
        })
    );
    assert_eq!(tool.parameters, original);
}

#[tokio::test]
async fn orders_strict_required_names_like_properties() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_strict_mode: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let tool = Tool::new(
        "ordered",
        "Ordered",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "number"},
                "metadata": {"type": "string"},
                "nullable": {"type": "string"}
            },
            "required": ["path", "metadata"]
        }),
    )
    .with_strict();

    provider
        .stream(
            &model,
            &Context::new([Message::user("Use the tool")]).with_tools([tool]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let payload = request_json(&server.requests().await[0]);
    assert_eq!(
        payload["tools"][0]["parameters"]["required"],
        json!(["path", "offset", "metadata", "nullable"])
    );
}

#[tokio::test]
async fn keeps_bare_strict_object_schemas_without_properties() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_strict_mode: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let tool = Tool::new("bare", "Bare", json!({"type": "object"})).with_strict();

    provider
        .stream(
            &model,
            &Context::new([Message::user("Use the tool")]).with_tools([tool]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let payload = request_json(&server.requests().await[0]);
    assert_eq!(
        payload["tools"][0]["parameters"],
        json!({
            "type": "object",
            "required": [],
            "additionalProperties": false
        })
    );
}

#[tokio::test]
async fn falls_back_or_rejects_each_unsupported_strict_schema() {
    let schemas = [
        json!({
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }),
        json!({
            "type": "object",
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "number"}}}
            ]
        }),
        json!({
            "type": "object",
            "properties": {
                "value": {"anyOf": [{"type": "object", "properties": {}}, {"type": "null"}]}
            }
        }),
        json!({
            "type": "object",
            "properties": {"child": {"$ref": "https://example.com/child.json"}},
            "required": ["child"]
        }),
    ];
    let server = serve((0..schemas.len()).map(|_| Reply::sse(openai_done()))).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_strict_mode: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);

    for (index, parameters) in schemas.iter().enumerate() {
        let tool = Tool {
            name: format!("fallback_{index}"),
            description: "Fallback".into(),
            parameters: parameters.clone(),
            constrained_sampling: Some(ConstrainedSampling::JsonSchema {
                strict: ConstrainedSamplingStrictness::Prefer,
            }),
        };
        provider
            .stream(
                &model,
                &Context::new([Message::user("Use the tool")]).with_tools([tool]),
                &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                    stream: StreamOptions {
                        api_key: Some("test-key".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            )
            .result()
            .await
            .unwrap();
    }
    let requests = server.requests().await;
    for (parameters, request) in schemas.iter().zip(requests.iter()) {
        let payload = request_json(request);
        assert_eq!(payload["tools"][0]["strict"], false);
        assert_eq!(payload["tools"][0]["parameters"], *parameters);
    }
    for (index, parameters) in schemas.iter().enumerate() {
        let tool = Tool {
            name: format!("required_{index}"),
            description: "Required".into(),
            parameters: parameters.clone(),
            constrained_sampling: Some(ConstrainedSampling::JsonSchema {
                strict: ConstrainedSamplingStrictness::Require,
            }),
        };
        let result = provider
            .stream(
                &model,
                &Context::new([Message::user("Use the tool")]).with_tools([tool]),
                &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                    stream: StreamOptions {
                        api_key: Some("test-key".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            )
            .result()
            .await
            .unwrap();
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(
            result
                .error_message
                .unwrap()
                .contains("requires JSON-schema constrained sampling")
        );
    }
}

#[tokio::test]
async fn rejects_supported_grammar_without_a_provider_variant() {
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_open_ai_grammar_tools: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let tool = Tool {
        name: "grammar".into(),
        description: "Grammar".into(),
        parameters: schema(),
        constrained_sampling: Some(ConstrainedSampling::Grammar {
            variants: GrammarVariants::default(),
        }),
    };

    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Use a tool")]).with_tools([tool]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("no supported grammar variant was provided")
    );
}

#[tokio::test]
async fn rejects_grammar_input_that_changes_after_close() {
    let server = serve([Reply::sse(grammar_changed_after_close())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_open_ai_grammar_tools: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Use the grammar tool")]).with_tools([Tool {
                name: "grammar".into(),
                description: "Grammar".into(),
                parameters: schema(),
                constrained_sampling: Some(ConstrainedSampling::Grammar {
                    variants: GrammarVariants {
                        openai_lark: Some("start: /[a-z]+/".into()),
                        openai_regex: None,
                    },
                }),
            }]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("changed after it was closed")
    );
}

#[tokio::test]
async fn accepts_repeated_closed_grammar_input() {
    let server = serve([Reply::sse(grammar_repeated_closed())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_open_ai_grammar_tools: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let mut stream = provider.stream(
        &model,
        &Context::new([Message::user("Use the grammar tool")]).with_tools([Tool {
            name: "grammar".into(),
            description: "Grammar".into(),
            parameters: schema(),
            constrained_sampling: Some(ConstrainedSampling::Grammar {
                variants: GrammarVariants {
                    openai_lark: Some("start: /[a-z]+/".into()),
                    openai_regex: None,
                },
            }),
        }]),
        &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            ..Default::default()
        }),
    );
    let mut deltas = Vec::new();
    while let Some(event) = stream.next().await {
        if let ds_ai::AssistantMessageEvent::ToolCallDelta { delta, .. } = event {
            deltas.push(delta);
        }
    }
    let result = stream.result().await.unwrap();

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas, vec![String::from(r#"{"payload":"abc"}"#)]);
    let AssistantContent::ToolCall(call) = &result.content[0] else {
        panic!("expected grammar tool call");
    };
    assert_eq!(call.arguments, json!({"payload": "abc"}));
}

#[tokio::test]
async fn applies_anthropic_strict_sampling_capability() {
    let server = serve([Reply::sse(anthropic_done())]).await;
    let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::Anthropic(
        AnthropicMessagesCompatibility {
            supports_strict_tools: Some(true),
            ..Default::default()
        },
    ));
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);
    provider
        .stream(
            &model,
            &Context::new([Message::user("Use a tool")]).with_tools([strict_tool()]),
            &ApiStreamOptions::AnthropicMessages(AnthropicOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    let payload = request_json(&server.requests().await[0]);
    assert_eq!(payload["tools"][0]["strict"], true);
    assert_eq!(
        payload["tools"][0]["input_schema"]["additionalProperties"],
        false
    );
}

#[tokio::test]
async fn encodes_codex_strict_defaults_and_grammar_tools() {
    let server = serve([Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_strict_mode: Some(true),
        supports_open_ai_grammar_tools: Some(true),
        ..Default::default()
    }));
    let provider = ds_ai::codex::Provider::new([model.clone()]);
    provider
        .stream(
            &model,
            &Context::new([Message::user("Use tools")]).with_tools(tools()),
            &ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
                stream: StreamOptions {
                    api_key: Some(token()),
                    transport: Some(Transport::Sse),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
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
    assert!(payload["tools"][0]["strict"].is_null());
    assert_eq!(payload["tools"][1]["strict"], true);
    assert_eq!(payload["tools"][2]["type"], "custom");
}

#[tokio::test]
async fn applies_openai_role_and_cache_compatibility() {
    let server = serve([Reply::sse(openai_done()), Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        supports_developer_role: Some(false),
        supports_long_cache_retention: Some(false),
        supports_explicit_prompt_cache_mode: Some(false),
        ..Default::default()
    }));
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    for retention in [ds_ai::CacheRetention::Long, ds_ai::CacheRetention::None] {
        provider
            .stream(
                &model,
                &Context::new([Message::user("Hello")]).with_system("System"),
                &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                    stream: StreamOptions {
                        api_key: Some("test-key".into()),
                        cache_retention: retention,
                        session_id: Some("session".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            )
            .result()
            .await
            .unwrap();
    }

    let requests = server.requests().await;
    let long = request_json(&requests[0]);
    assert_eq!(long["input"][0]["role"], "system");
    assert_eq!(long["prompt_cache_key"], "session");
    assert!(long.get("prompt_cache_retention").is_none());
    let none = request_json(&requests[1]);
    assert!(none.get("prompt_cache_key").is_none());
    assert!(none.get("prompt_cache_options").is_none());
}

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "plain".into(),
            description: "Plain".into(),
            parameters: schema(),
            constrained_sampling: Some(ConstrainedSampling::Disabled),
        },
        Tool {
            name: "strict".into(),
            description: "Strict".into(),
            parameters: schema(),
            constrained_sampling: Some(ConstrainedSampling::JsonSchema {
                strict: ConstrainedSamplingStrictness::Prefer,
            }),
        },
        Tool {
            name: "grammar".into(),
            description: "Grammar".into(),
            parameters: schema(),
            constrained_sampling: Some(ConstrainedSampling::Grammar {
                variants: GrammarVariants {
                    openai_lark: Some("start: /[a-z]+/".into()),
                    openai_regex: None,
                },
            }),
        },
    ]
}

fn strict_tool() -> Tool {
    Tool::new("strict", "Strict", schema()).with_strict()
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {"payload": {"type": "string"}},
        "required": ["payload"]
    })
}

fn custom_tool_events() -> &'static str {
    concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"call_id\":\"call_1\",\"id\":\"ctc_1\",\"name\":\"grammar\",\"input\":\"\"}}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.delta\",\"output_index\":0,\"delta\":\"a\\\"\"}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.delta\",\"output_index\":0,\"delta\":\"\"}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"input\":\"a\\\"\\nb\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"call_id\":\"call_1\",\"id\":\"ctc_1\",\"name\":\"grammar\",\"input\":\"a\\\"\\nb\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":1,\"output_tokens_details\":{}}}}\n\n"
    )
}

fn grammar_changed_after_close() -> &'static str {
    concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"call_id\":\"call_1\",\"id\":\"ctc_1\",\"name\":\"grammar\",\"input\":\"\"}}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.delta\",\"output_index\":0,\"delta\":\"abc\"}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"input\":\"abc\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"call_id\":\"call_1\",\"id\":\"ctc_1\",\"name\":\"grammar\",\"input\":\"changed\"}}\n\n"
    )
}

fn grammar_repeated_closed() -> &'static str {
    concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"call_id\":\"call_1\",\"id\":\"ctc_1\",\"name\":\"grammar\",\"input\":\"\"}}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"input\":\"abc\"}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"input\":\"abc\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"call_id\":\"call_1\",\"id\":\"ctc_1\",\"name\":\"grammar\",\"input\":\"abc\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_3\",\"status\":\"completed\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n"
    )
}

fn openai_done() -> &'static str {
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"status\":\"completed\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n"
}

fn anthropic_done() -> &'static str {
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

fn request_json(request: &str) -> Value {
    serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

fn token() -> String {
    let payload = BASE64_URL_SAFE_NO_PAD.encode(
        json!({"https://api.openai.com/auth": {"chatgpt_account_id": "account"}}).to_string(),
    );
    format!("header.{payload}.signature")
}
