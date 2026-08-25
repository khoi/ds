use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::{
    AnthropicOptions, ApiStreamOptions, Context, Message, Models, OpenAiCodexResponsesOptions,
    OpenAiResponsesOptions, PayloadHook, Provider, ProviderResponse, ResponseHook,
    SimpleStreamOptions, StopReason, StreamOptions, Transport, builtin_model,
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn uses_a_custom_http_client_for_each_api() {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        "x-custom-client",
        reqwest::header::HeaderValue::from_static("present"),
    );
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .unwrap();

    let openai_server = serve([Reply::sse(openai_done())]).await;
    let mut openai_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    openai_model.base_url = openai_server.base_url.clone();
    ds_ai::openai::stream(
        &openai_model
            .typed::<ds_ai::OpenAiResponsesOptions>()
            .unwrap(),
        &Context::new([Message::user("Hello")]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                http_client: Some(client.clone()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();

    let anthropic_server = serve([Reply::sse(anthropic_done())]).await;
    let mut anthropic_model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    anthropic_model.base_url = anthropic_server.base_url.clone();
    ds_ai::anthropic::stream(
        &anthropic_model.typed::<ds_ai::AnthropicOptions>().unwrap(),
        &Context::new([Message::user("Hello")]),
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                http_client: Some(client.clone()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();

    let codex_server = serve([Reply::sse(openai_done())]).await;
    let mut codex_model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    codex_model.base_url = codex_server.base_url.clone();
    ds_ai::codex::stream(
        &codex_model
            .typed::<ds_ai::OpenAiCodexResponsesOptions>()
            .unwrap(),
        &Context::new([Message::user("Hello")]),
        &OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(token()),
                http_client: Some(client),
                transport: Some(Transport::Sse),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .result()
    .await
    .unwrap();

    for request in [
        &openai_server.requests().await[0],
        &anthropic_server.requests().await[0],
    ] {
        assert!(request.contains("x-custom-client: present\r\n"));
    }
    let request = codex_server.request_bytes().await.pop().unwrap();
    let header_end = request
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap();
    let headers = String::from_utf8(request[..header_end].to_vec()).unwrap();
    assert!(headers.contains("x-custom-client: present\r\n"));
}

#[tokio::test]
async fn transforms_one_payload_and_observes_the_successful_response() {
    let server = serve([
        Reply::json(429, json!({"error": {"message": "retry"}}))
            .with_header("retry-after-ms", "0")
            .with_header("x-attempt", "first"),
        Reply::sse(openai_done()).with_header("x-attempt", "second"),
    ])
    .await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let transformed_models = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::new(Mutex::new(Vec::new()));
    let options = ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            max_retries: Some(1),
            on_payload: Some(PayloadHook::new({
                let transformed_models = transformed_models.clone();
                move |mut payload, model| {
                    transformed_models.lock().unwrap().push(model.id);
                    async move {
                        payload["hooked"] = true.into();
                        Ok(Some(payload))
                    }
                }
            })),
            on_response: Some(ResponseHook::new({
                let observed = observed.clone();
                move |response: ProviderResponse, model| {
                    observed.lock().unwrap().push((
                        response.status,
                        response.headers.get("x-attempt").cloned(),
                        model.id,
                    ));
                    async { Ok(()) }
                }
            })),
            ..Default::default()
        },
        ..Default::default()
    });

    provider
        .stream(&model, &Context::new([Message::user("Hello")]), &options)
        .result()
        .await
        .unwrap();

    assert_eq!(*transformed_models.lock().unwrap(), ["gpt-5.6-sol"]);
    assert_eq!(
        *observed.lock().unwrap(),
        [(200, Some("second".into()), "gpt-5.6-sol".into())]
    );
    for request in server.requests().await {
        assert_eq!(request_json(&request)["hooked"], true);
    }
}

#[tokio::test]
async fn forces_anthropic_payloads_to_stream_after_hook_replacement() {
    let cases = [
        (
            json!({"stream": false, "hooked": true}),
            json!({"stream": true, "hooked": true}),
        ),
        (
            json!(["first", 2]),
            json!({"0": "first", "1": 2, "stream": true}),
        ),
        (json!("ab"), json!({"0": "a", "1": "b", "stream": true})),
        (json!(42), json!({"stream": true})),
        (json!(true), json!({"stream": true})),
        (Value::Null, json!({"stream": true})),
    ];

    for (payload, expected) in cases {
        let server = serve([Reply::sse(anthropic_done())]).await;
        let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
        model.base_url = server.base_url.clone();
        let hook_payload = payload.clone();
        let options = AnthropicOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                on_payload: Some(PayloadHook::new(move |_, _| {
                    let payload = hook_payload.clone();
                    async move { Ok(Some(payload)) }
                })),
                ..Default::default()
            },
            ..Default::default()
        };

        ds_ai::anthropic::stream(
            &model.typed::<ds_ai::AnthropicOptions>().unwrap(),
            &Context::new([Message::user("Hello")]),
            &options,
        )
        .result()
        .await
        .unwrap();

        let request = server.requests().await.pop().unwrap();
        assert_eq!(request_json(&request), expected);
    }
}

#[tokio::test]
async fn does_not_retry_openai_response_hook_failures() {
    let server = serve([Reply::sse(openai_done()), Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::openai::Provider::new([model.clone()]);

    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Hello")]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    max_retries: Some(1),
                    on_response: Some(ResponseHook::new(|_, _| async {
                        Err("response rejected".into())
                    })),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("response rejected"));
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn sends_and_suppresses_headers_for_each_api() {
    let openai_server = serve([Reply::sse(openai_done())]).await;
    let mut openai_model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    openai_model.base_url = openai_server.base_url.clone();
    openai_model
        .headers
        .insert("X-Model-Remove".into(), "model".into());
    let openai = ds_ai::openai::Provider::new([openai_model.clone()]);
    let openai_options = stream_options("openai").with_transform_headers(transform_headers);
    openai
        .stream(
            &openai_model,
            &Context::new([Message::user("Hello")]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: openai_options,
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();
    let openai_request = openai_server.requests().await[0].to_ascii_lowercase();
    assert!(openai_request.contains("x-ds-test: openai\r\n"));
    assert!(openai_request.contains("x-transformed: yes\r\n"));
    assert!(!openai_request.contains("x-model-remove:"));

    let anthropic_server = serve([Reply::sse(anthropic_done())]).await;
    let mut anthropic_model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    anthropic_model.base_url = anthropic_server.base_url.clone();
    anthropic_model
        .headers
        .insert("X-Model-Remove".into(), "model".into());
    let anthropic = ds_ai::anthropic::Provider::new([anthropic_model.clone()]);
    let mut anthropic_stream = stream_options("anthropic");
    anthropic_stream
        .headers
        .insert("Anthropic-Version".into(), None);
    anthropic_stream = anthropic_stream.with_transform_headers(transform_headers);
    anthropic
        .stream(
            &anthropic_model,
            &Context::new([Message::user("Hello")]),
            &ApiStreamOptions::AnthropicMessages(AnthropicOptions {
                stream: anthropic_stream,
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();
    let anthropic_request = anthropic_server.requests().await[0].to_ascii_lowercase();
    assert!(anthropic_request.contains("x-ds-test: anthropic\r\n"));
    assert!(anthropic_request.contains("x-transformed: yes\r\n"));
    assert!(!anthropic_request.contains("x-model-remove:"));
    assert!(!anthropic_request.contains("anthropic-version:"));

    let codex_server = serve([Reply::sse(openai_done())]).await;
    let mut codex_model = builtin_model("openai-codex", "gpt-5.6-sol").unwrap();
    codex_model.base_url = codex_server.base_url.clone();
    codex_model
        .headers
        .insert("X-Model-Remove".into(), "model".into());
    let codex = ds_ai::codex::Provider::new([codex_model.clone()]);
    let mut codex_stream = stream_options("codex");
    codex_stream.api_key = Some(token());
    codex_stream.transport = Some(Transport::Sse);
    codex_stream
        .headers
        .insert("Originator".into(), Some("overridden".into()));
    codex_stream = codex_stream.with_transform_headers(transform_headers);
    codex
        .stream(
            &codex_model,
            &Context::new([Message::user("Hello")]),
            &ApiStreamOptions::OpenAiCodexResponses(OpenAiCodexResponsesOptions {
                stream: codex_stream,
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();
    let request = codex_server.request_bytes().await.pop().unwrap();
    let split = request
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap()
        + 4;
    let codex_request = String::from_utf8(request[..split].to_vec())
        .unwrap()
        .to_ascii_lowercase();
    assert!(codex_request.contains("x-ds-test: codex\r\n"));
    assert!(codex_request.contains("x-transformed: yes\r\n"));
    assert!(!codex_request.contains("x-model-remove:"));
    assert!(codex_request.contains("originator: ds\r\n"));
    assert!(!codex_request.contains("originator: overridden\r\n"));
}

#[tokio::test]
async fn sends_transformed_models_headers_on_the_provider_request() {
    let server = serve([Reply::sse(openai_done()), Reply::sse(openai_done())]).await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    model
        .headers
        .insert("X-Model-Header".into(), "model".into());
    let mut models = Models::new();
    models.set_provider(Arc::new(ds_ai::openai::Provider::new([model.clone()])));
    let hook_models = Arc::new(Mutex::new(Vec::new()));
    let stream = StreamOptions {
        api_key: Some("test-key".into()),
        headers: [
            ("X-Request-Header".into(), Some("request".into())),
            ("X-Shared".into(), Some("request".into())),
        ]
        .into_iter()
        .collect(),
        on_payload: Some(PayloadHook::new({
            let hook_models = hook_models.clone();
            move |_, model| {
                hook_models.lock().unwrap().push(model);
                async { Ok(None) }
            }
        })),
        on_response: Some(ResponseHook::new({
            let hook_models = hook_models.clone();
            move |_, model| {
                hook_models.lock().unwrap().push(model);
                async { Ok(()) }
            }
        })),
        ..Default::default()
    }
    .with_transform_headers(|mut headers| async move {
        headers.remove("X-Model-Header");
        headers.insert("X-Transformed".into(), Some("yes".into()));
        Ok(headers)
    });

    models
        .complete(
            &model.typed::<ds_ai::OpenAiResponsesOptions>().unwrap(),
            &Context::new([Message::user("Hello")]),
            &OpenAiResponsesOptions {
                stream: stream.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    models
        .complete_simple(
            &model,
            &Context::new([Message::user("Hello")]),
            &SimpleStreamOptions {
                stream,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    for request in server.requests().await {
        let request = request.to_ascii_lowercase();
        assert!(!request.contains("x-model-header:"));
        assert!(request.contains("x-request-header: request\r\n"));
        assert!(request.contains("x-shared: request\r\n"));
        assert!(request.contains("x-transformed: yes\r\n"));
    }
    assert_eq!(
        hook_models
            .lock()
            .unwrap()
            .iter()
            .map(|model| model.headers.get("X-Model-Header").map(String::as_str))
            .collect::<Vec<_>>(),
        [Some("model"), Some("model"), Some("model"), Some("model")]
    );
}

#[tokio::test]
async fn sends_and_overrides_the_anthropic_user_agent() {
    let server = serve([Reply::sse(anthropic_done()), Reply::sse(anthropic_done())]).await;
    let mut model = builtin_model("anthropic", "claude-opus-4-5").unwrap();
    model.base_url = server.base_url.clone();
    let provider = ds_ai::anthropic::Provider::new([model.clone()]);

    for user_agent in [None, Some("caller/1.0")] {
        let mut stream = StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        };
        if let Some(user_agent) = user_agent {
            stream
                .headers
                .insert("User-Agent".into(), Some(user_agent.into()));
        }
        provider
            .stream(
                &model,
                &Context::new([Message::user("Hello")]),
                &ApiStreamOptions::AnthropicMessages(AnthropicOptions {
                    stream,
                    ..Default::default()
                }),
            )
            .result()
            .await
            .unwrap();
    }

    let requests = server.requests().await;
    assert!(requests[0].contains("user-agent: ds-ai/0.1.0\r\n"));
    assert!(requests[1].contains("user-agent: caller/1.0\r\n"));
}

#[tokio::test]
async fn turns_hook_failures_into_terminal_errors() {
    let model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Hello")]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    on_payload: Some(PayloadHook::new(|_, _| async {
                        Err("payload rejected".into())
                    })),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("payload rejected"));
}

#[tokio::test]
async fn classifies_a_hook_failure_after_cancellation_as_aborted() {
    let model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    let provider = ds_ai::openai::Provider::new([model.clone()]);
    let cancellation = CancellationToken::new();
    let hook_cancellation = cancellation.clone();
    let result = provider
        .stream(
            &model,
            &Context::new([Message::user("Hello")]),
            &ApiStreamOptions::OpenAiResponses(OpenAiResponsesOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    cancellation,
                    on_payload: Some(PayloadHook::new(move |_, _| {
                        let hook_cancellation = hook_cancellation.clone();
                        async move {
                            hook_cancellation.cancel();
                            Err("payload rejected".into())
                        }
                    })),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .result()
        .await
        .unwrap();

    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("payload rejected"));
}

fn stream_options(value: &str) -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        headers: [("X-DS-Test".into(), Some(value.into()))]
            .into_iter()
            .collect(),
        ..Default::default()
    }
}

async fn transform_headers(
    mut headers: std::collections::BTreeMap<String, Option<String>>,
) -> Result<std::collections::BTreeMap<String, Option<String>>, String> {
    headers.remove("X-Model-Remove");
    headers.insert("X-Transformed".into(), Some("yes".into()));
    Ok(headers)
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
