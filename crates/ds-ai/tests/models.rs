use async_trait::async_trait;
use ds_ai::{
    Api, ApiKeyAuth, ApiStreamOptions, AssistantMessage, AssistantMessageEvent,
    AssistantMessageEventStream, AuthCheck, AuthContext, AuthError, AuthResolutionOverrides,
    AuthResult, Context, Credential, CredentialStore, CredentialType, DoneReason, HeaderHook,
    Model, ModelAuth, ModelCost, ModelInput, Models, OAuthAuth, OpenAiResponsesOptions, Provider,
    ProviderAuth, ProviderId, SimpleStreamOptions, StopReason, StreamOptions,
};
use futures_util::StreamExt;
use futures_util::stream;
use std::future::pending;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::{collections::BTreeMap, sync::Arc};
use tokio::sync::{Barrier, Notify};
use tokio_util::sync::CancellationToken;

use crate::support::{Reply, serve};

struct TestProvider {
    id: ProviderId,
    name: String,
    models: Vec<Model>,
    marker: String,
    auth: ds_ai::ProviderAuth,
    headers: BTreeMap<String, Option<String>>,
    captured: Option<Arc<Mutex<Option<StreamOptions>>>>,
}

struct ModelCaptureProvider {
    model: Model,
    auth: ProviderAuth,
    captured: Arc<Mutex<Vec<Model>>>,
    headers: BTreeMap<String, Option<String>>,
}

impl Provider for ModelCaptureProvider {
    fn id(&self) -> &ProviderId {
        &self.model.provider
    }

    fn name(&self) -> &str {
        "Model capture"
    }

    fn base_url(&self) -> Option<&str> {
        None
    }

    fn headers(&self) -> &BTreeMap<String, Option<String>> {
        &self.headers
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn models(&self) -> Vec<Model> {
        vec![self.model.clone()]
    }

    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: &ApiStreamOptions,
    ) -> AssistantMessageEventStream {
        self.captured.lock().unwrap().push(model.clone());
        completed(model, "done")
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        _options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.captured.lock().unwrap().push(model.clone());
        completed(model, "done")
    }
}

impl Provider for TestProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> Option<&str> {
        None
    }

    fn headers(&self) -> &BTreeMap<String, Option<String>> {
        &self.headers
    }

    fn auth(&self) -> &ds_ai::ProviderAuth {
        &self.auth
    }

    fn models(&self) -> Vec<Model> {
        self.models.clone()
    }

    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        options: &ApiStreamOptions,
    ) -> AssistantMessageEventStream {
        if let Some(captured) = &self.captured {
            *captured.lock().unwrap() = Some(options.stream().clone());
        }
        completed(model, &self.marker)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        if let Some(captured) = &self.captured {
            *captured.lock().unwrap() = Some(options.stream.clone());
        }
        completed(model, &self.marker)
    }
}

#[tokio::test]
async fn registers_replaces_lists_routes_and_deletes_providers() {
    let model = model("openai", "gpt-test");
    let mut models = collection();
    models.set_provider(provider(&model, "first"));
    assert_eq!(models.providers().len(), 1);
    assert_eq!(models.models(None), std::slice::from_ref(&model));
    assert_eq!(models.model("openai", "gpt-test"), Some(model.clone()));

    models.set_provider(provider(&model, "replacement"));
    let result = models
        .complete(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(result.error_message.as_deref(), Some("replacement"));

    assert!(models.delete_provider("openai").is_some());
    assert!(models.providers().is_empty());
    models.set_provider(provider(&model, "again"));
    models.clear_providers();
    assert!(models.providers().is_empty());
}

#[tokio::test]
async fn streams_and_completes_a_runtime_model_through_its_registered_provider() {
    let mut model = model("runtime", "model-a");
    model.api = Api::Other("runtime-api".into());
    let mut models = collection();
    models.set_provider(provider(&model, "runtime"));
    let model = models.model("runtime", "model-a").unwrap();
    let options = SimpleStreamOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };

    let result = models
        .stream_simple(&model, &Context::new([]), &options)
        .result()
        .await
        .unwrap();

    assert_eq!(result.api, Api::Other("runtime-api".into()));
    assert_eq!(result.error_message.as_deref(), Some("runtime"));

    let result = models
        .complete_simple(&model, &Context::new([]), &options)
        .await
        .unwrap();

    assert_eq!(result.api, Api::Other("runtime-api".into()));
    assert_eq!(result.error_message.as_deref(), Some("runtime"));
}

#[test]
fn leaves_simple_tool_choice_unspecified_by_default() {
    assert_eq!(SimpleStreamOptions::default().tool_choice, None);
}

#[tokio::test]
async fn returns_terminal_stream_errors_for_unknown_providers() {
    let model = model("missing", "gpt-test");
    let result = collection()
        .complete(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(StreamOptions::default()),
        )
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Unknown provider missing")
    );
}

#[tokio::test]
async fn emits_only_error_when_auth_fails_before_provider_start() {
    let model = model("openai", "gpt-test");
    let mut models = collection();
    models.set_provider(provider(&model, "done"));

    let typed = models
        .stream(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(StreamOptions::default()),
        )
        .collect::<Vec<_>>()
        .await;
    let simple = models
        .stream_simple(&model, &Context::new([]), &SimpleStreamOptions::default())
        .collect::<Vec<_>>()
        .await;

    for events in [typed, simple] {
        assert!(matches!(
            events.as_slice(),
            [AssistantMessageEvent::Error { error, .. }]
                if error.error_message.as_deref() == Some("Provider is not configured: openai")
        ));
    }
}

#[tokio::test]
async fn returns_terminal_stream_errors_for_unknown_provider_apis() {
    let mut model = model("openai", "gpt-test");
    model.api = Api::AnthropicMessages;
    let mut models = collection();
    models.set_provider(Arc::new(ds_ai::openai::Provider::new([model.clone()])));
    let result = models
        .complete(
            &model.typed::<ds_ai::AnthropicOptions>().unwrap(),
            &Context::new([]),
            &ds_ai::AnthropicOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(
        result
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("no API implementation"))
    );
}

#[tokio::test]
async fn openai_provider_returns_a_stream_before_setup_and_emits_provider_events() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\",\"annotations\":[]}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":2}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = model("openai", "gpt-test");
    model.base_url = server.base_url;
    let mut models = collection();
    models.set_provider(Arc::new(ds_ai::openai::Provider::new([model.clone()])));
    let options = api_options(StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    });

    let events = models
        .stream(
            &model.typed().unwrap(),
            &Context::new([ds_ai::Message::user("Hello")]),
            &options,
        )
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events[0], AssistantMessageEvent::Start { .. }));
    assert!(matches!(
        events[1],
        AssistantMessageEvent::TextStart {
            content_index: 0,
            ..
        }
    ));
    assert!(matches!(
        &events[2],
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Hello"
    ));
    assert!(matches!(events[3], AssistantMessageEvent::TextEnd { .. }));
    assert!(matches!(events[4], AssistantMessageEvent::Done { .. }));
}

#[tokio::test]
async fn resolves_explicit_and_stored_auth_and_lists_available_models() {
    let model = model("openai", "gpt-test");
    let mut models = collection();
    models.set_provider(provider(&model, "done"));

    let explicit = models
        .auth(
            "openai",
            AuthResolutionOverrides {
                api_key: Some("explicit".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(explicit.auth.api_key.as_deref(), Some("explicit"));

    let cancellation = CancellationToken::new();
    models
        .credentials()
        .modify(
            "openai",
            Box::new(|_| {
                Box::pin(async {
                    Ok(Some(Credential::ApiKey {
                        key: Some("stored".into()),
                        env: BTreeMap::new(),
                    }))
                })
            }),
            &cancellation,
        )
        .await
        .unwrap();
    let stored = models
        .auth("openai", AuthResolutionOverrides::default())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.auth.api_key.as_deref(), Some("stored"));
    assert_eq!(
        models.check_auth("openai", &cancellation).await.unwrap(),
        Some(AuthCheck {
            source: Some("stored credential".into()),
            credential_type: CredentialType::ApiKey,
        })
    );
    assert_eq!(
        models.available_models(None, &cancellation).await.unwrap(),
        [model]
    );
}

#[tokio::test]
async fn wrong_stored_credential_kind_blocks_ambient_auth_fallback() {
    let stored = Arc::new(ds_ai::InMemoryCredentialStore::new());
    let cancellation = CancellationToken::new();
    stored
        .modify(
            "openai",
            Box::new(|_| Box::pin(async { Ok(Some(oauth("stored", u64::MAX))) })),
            &cancellation,
        )
        .await
        .unwrap();
    let model = model("openai", "gpt-test");
    let mut models = Models::with_auth(stored, Arc::new(AmbientAuthContext));
    models.set_provider(provider(&model, "done"));

    assert_eq!(
        models
            .auth("openai", AuthResolutionOverrides::default())
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        models.check_auth("openai", &cancellation).await.unwrap(),
        None
    );
    assert!(
        models
            .available_models(None, &cancellation)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn checks_provider_availability_concurrently_and_keeps_provider_order() {
    let first = model("first", "model-a");
    let second = model("second", "model-b");
    let barrier = Arc::new(Barrier::new(2));
    let mut models = collection();
    for model in [&first, &second] {
        models.set_provider(Arc::new(TestProvider {
            id: model.provider.clone(),
            name: model.name.clone(),
            models: vec![model.clone()],
            marker: "done".into(),
            auth: ProviderAuth::api_key(BarrierApiAuth {
                barrier: barrier.clone(),
            }),
            headers: BTreeMap::new(),
            captured: None,
        }));
    }

    let available = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        models.available_models(None, &CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap();

    assert_eq!(available, [first, second]);
}

#[tokio::test]
async fn serializes_oauth_refresh_and_preserves_failed_credentials() {
    let model = model("oauth", "gpt-test");
    let refreshes = Arc::new(AtomicUsize::new(0));
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "OAuth".into(),
        models: vec![model.clone()],
        marker: "done".into(),
        auth: ProviderAuth::oauth(TestOAuth {
            refreshes: refreshes.clone(),
            fail: false,
        }),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let cancellation = CancellationToken::new();
    let expired = oauth("old", 0);
    models
        .credentials()
        .modify(
            "oauth",
            Box::new({
                let expired = expired.clone();
                move |_| Box::pin(async move { Ok(Some(expired)) })
            }),
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(
        models.check_auth("oauth", &cancellation).await.unwrap(),
        Some(AuthCheck {
            source: Some("OAuth".into()),
            credential_type: CredentialType::OAuth,
        })
    );
    assert_eq!(refreshes.load(Ordering::SeqCst), 0);

    let (first, second) = tokio::join!(
        models.auth("oauth", AuthResolutionOverrides::default()),
        models.auth("oauth", AuthResolutionOverrides::default()),
    );
    assert_eq!(
        first.unwrap().unwrap().auth.api_key.as_deref(),
        Some("refreshed")
    );
    assert_eq!(
        second.unwrap().unwrap().auth.api_key.as_deref(),
        Some("refreshed")
    );
    assert_eq!(refreshes.load(Ordering::SeqCst), 1);

    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "OAuth".into(),
        models: vec![model],
        marker: "done".into(),
        auth: ProviderAuth::oauth(TestOAuth {
            refreshes,
            fail: true,
        }),
        headers: BTreeMap::new(),
        captured: None,
    }));
    models
        .credentials()
        .modify(
            "oauth",
            Box::new({
                let expired = expired.clone();
                move |_| Box::pin(async move { Ok(Some(expired)) })
            }),
            &cancellation,
        )
        .await
        .unwrap();
    assert!(matches!(
        models
            .auth("oauth", AuthResolutionOverrides::default())
            .await,
        Err(AuthError::OAuth(message))
            if message.contains("OAuth refresh failed for provider oauth")
                && message.contains("rejected")
    ));
    assert_eq!(
        models
            .credentials()
            .read("oauth", &cancellation)
            .await
            .unwrap(),
        Some(expired)
    );
}

#[tokio::test]
async fn merges_model_auth_and_request_headers_once() {
    let mut model = model("headers", "gpt-test");
    model.headers.insert("X-Test".into(), "model".into());
    let captured = Arc::new(Mutex::new(None));
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Headers".into(),
        models: vec![model.clone()],
        marker: "done".into(),
        auth: ProviderAuth::api_key(HeaderAuth),
        headers: BTreeMap::from([
            ("x-test".into(), Some("provider".into())),
            ("Provider-Only".into(), Some("provider".into())),
        ]),
        captured: Some(captured.clone()),
    }));
    models
        .complete(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(StreamOptions {
                api_key: Some("key".into()),
                headers: BTreeMap::from([
                    ("X-TEST".into(), Some("request".into())),
                    ("auth-only".into(), None),
                ]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(
        captured.lock().unwrap().as_ref().unwrap().headers,
        BTreeMap::from([
            ("Auth-Added".into(), Some("auth".into())),
            ("X-TEST".into(), Some("request".into())),
            ("auth-only".into(), None),
        ])
    );
}

#[tokio::test]
async fn transforms_final_headers_before_provider_dispatch() {
    let mut model = model("headers-transform", "gpt-test");
    model.headers.insert("Model-Only".into(), "model".into());
    model.headers.insert("x-test".into(), "model".into());
    let captured = Arc::new(Mutex::new(None));
    let transformed = Arc::new(Mutex::new(None));
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Headers".into(),
        models: vec![model.clone()],
        marker: "done".into(),
        auth: ProviderAuth::api_key(HeaderAuth),
        headers: BTreeMap::from([
            ("Provider-Only".into(), Some("provider".into())),
            ("x-test".into(), Some("provider".into())),
        ]),
        captured: Some(captured.clone()),
    }));
    let transformed_for_hook = transformed.clone();
    models
        .complete(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(StreamOptions {
                api_key: Some("key".into()),
                headers: BTreeMap::from([
                    ("Request-Only".into(), Some("request".into())),
                    ("X-TEST".into(), Some("request".into())),
                ]),
                transform_headers: Some(HeaderHook::new(move |headers| {
                    *transformed_for_hook.lock().unwrap() = Some(headers.clone());
                    async move {
                        let mut headers = headers;
                        headers.insert("Transformed".into(), Some("yes".into()));
                        Ok(headers)
                    }
                })),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    assert_eq!(
        *transformed.lock().unwrap(),
        Some(BTreeMap::from([
            ("Auth-Added".into(), Some("auth".into())),
            ("Auth-Only".into(), Some("auth".into())),
            ("Model-Only".into(), Some("model".into())),
            ("Request-Only".into(), Some("request".into())),
            ("X-TEST".into(), Some("request".into())),
        ]))
    );
    let options = captured.lock().unwrap().clone().unwrap();
    assert_eq!(
        options.headers,
        BTreeMap::from([
            ("Auth-Added".into(), Some("auth".into())),
            ("Auth-Only".into(), Some("auth".into())),
            ("Model-Only".into(), Some("model".into())),
            ("Request-Only".into(), Some("request".into())),
            ("Transformed".into(), Some("yes".into())),
            ("X-TEST".into(), Some("request".into())),
        ])
    );
    assert!(options.transform_headers.is_none());
}

#[tokio::test]
async fn keeps_model_headers_visible_to_typed_and_simple_provider_dispatch() {
    let mut model = model("model-visibility", "gpt-test");
    model
        .headers
        .insert("X-Model-Header".into(), "model".into());
    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut models = collection();
    models.set_provider(Arc::new(ModelCaptureProvider {
        model: model.clone(),
        auth: ProviderAuth::api_key(HeaderAuth),
        captured: captured.clone(),
        headers: BTreeMap::new(),
    }));
    let stream = StreamOptions {
        api_key: Some("key".into()),
        ..Default::default()
    }
    .with_transform_headers(|mut headers| async move {
        headers.remove("X-Model-Header");
        Ok(headers)
    });

    models
        .complete(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(stream.clone()),
        )
        .await
        .unwrap();
    models
        .complete_simple(
            &model,
            &Context::new([]),
            &SimpleStreamOptions {
                stream,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(captured.lock().unwrap().as_slice(), [model.clone(), model]);
}

#[tokio::test]
async fn resolves_model_headers_as_part_of_model_auth() {
    let mut model = model("model-auth", "gpt-test");
    model.headers.insert("X-Model".into(), "model".into());
    model.headers.insert("X-Test".into(), "model".into());
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Headers".into(),
        models: vec![model.clone()],
        marker: "done".into(),
        auth: ProviderAuth::api_key(HeaderAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));

    let result = models
        .auth_for_model(
            &model,
            AuthResolutionOverrides {
                api_key: Some("key".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        result.auth.headers.get("X-Model"),
        Some(&Some("model".into()))
    );
    assert_eq!(
        result.auth.headers.get("X-Test"),
        Some(&Some("model".into()))
    );
}

#[tokio::test]
async fn returns_header_transform_errors_before_provider_dispatch() {
    let model = model("headers-transform-error", "gpt-test");
    let captured = Arc::new(Mutex::new(None));
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Headers".into(),
        models: vec![model.clone()],
        marker: "done".into(),
        auth: ProviderAuth::api_key(HeaderAuth),
        headers: BTreeMap::new(),
        captured: Some(captured.clone()),
    }));

    let result = models
        .complete(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(
                StreamOptions {
                    api_key: Some("key".into()),
                    ..Default::default()
                }
                .with_transform_headers(|_| async { Err("rejected".into()) }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("rejected"));
    assert!(captured.lock().unwrap().is_none());
}

#[tokio::test]
async fn cancels_model_auth_before_provider_dispatch() {
    let model = model("auth-cancel", "gpt-test");
    let captured = Arc::new(Mutex::new(None));
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Auth".into(),
        models: vec![model.clone()],
        marker: "done".into(),
        auth: ProviderAuth::api_key(HeaderAuth),
        headers: BTreeMap::new(),
        captured: Some(captured.clone()),
    }));
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let result = models
        .complete(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(StreamOptions {
                api_key: Some("key".into()),
                cancellation,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("authentication operation cancelled")
    );
    assert!(captured.lock().unwrap().is_none());
}

#[tokio::test]
async fn stops_waiting_for_non_cooperative_models_auth_and_check_callbacks() {
    let blocking_model = model("blocking", "gpt-test");
    let started = Arc::new(Notify::new());
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: blocking_model.provider.clone(),
        name: "Blocking".into(),
        models: vec![blocking_model],
        marker: "done".into(),
        auth: ProviderAuth::api_key(BlockingApiAuth {
            started: started.clone(),
        }),
        headers: BTreeMap::new(),
        captured: None,
    }));

    let cancellation = CancellationToken::new();
    let auth_cancellation = cancellation.clone();
    let auth_started = started.clone();
    let auth_task = tokio::spawn(async move {
        let overrides = AuthResolutionOverrides {
            cancellation: auth_cancellation,
            ..Default::default()
        };
        models.auth("blocking", overrides).await
    });
    auth_started.notified().await;
    cancellation.cancel();
    assert!(matches!(
        auth_task.await.unwrap(),
        Err(AuthError::Cancelled)
    ));

    let cancellation = CancellationToken::new();
    let mut models = collection();
    let started = Arc::new(Notify::new());
    models.set_provider(Arc::new(TestProvider {
        id: ProviderId::new("blocking"),
        name: "Blocking".into(),
        models: vec![model("blocking", "gpt-test")],
        marker: "done".into(),
        auth: ProviderAuth::api_key(BlockingApiAuth {
            started: started.clone(),
        }),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let check_cancellation = cancellation.clone();
    let check_task =
        tokio::spawn(async move { models.available_models(None, &check_cancellation).await });
    started.notified().await;
    cancellation.cancel();
    assert!(matches!(
        check_task.await.unwrap(),
        Err(AuthError::Cancelled)
    ));
}

#[tokio::test]
async fn stops_waiting_for_non_cooperative_login_and_logout_store_callbacks() {
    let started = Arc::new(Notify::new());
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let login_started = started.clone();
    let login_task = tokio::spawn(async move {
        let login_model = model("login", "gpt-test");
        let mut models = collection();
        models.set_provider(Arc::new(TestProvider {
            id: login_model.provider.clone(),
            name: "Login".into(),
            models: vec![login_model],
            marker: "done".into(),
            auth: ProviderAuth::api_key(BlockingApiAuth {
                started: login_started,
            }),
            headers: BTreeMap::new(),
            captured: None,
        }));
        let interaction = TestInteraction {
            cancellation: task_cancellation,
        };
        models
            .login("login", CredentialType::ApiKey, &interaction)
            .await
    });
    started.notified().await;
    cancellation.cancel();
    assert!(matches!(
        login_task.await.unwrap(),
        Err(AuthError::Cancelled)
    ));

    let started = Arc::new(Notify::new());
    let models = Models::with_auth(
        Arc::new(BlockingStore {
            read_started: Arc::new(Notify::new()),
            delete_started: started.clone(),
        }),
        Arc::new(EmptyAuthContext),
    );
    let cancellation = CancellationToken::new();
    let logout_cancellation = cancellation.clone();
    let logout = tokio::spawn(async move { models.logout("login", &logout_cancellation).await });
    started.notified().await;
    cancellation.cancel();
    assert!(matches!(logout.await.unwrap(), Err(AuthError::Cancelled)));

    let read_started = Arc::new(Notify::new());
    let mut models = Models::with_auth(
        Arc::new(BlockingStore {
            read_started: read_started.clone(),
            delete_started: Arc::new(Notify::new()),
        }),
        Arc::new(EmptyAuthContext),
    );
    let read_model = model("login", "gpt-test");
    models.set_provider(Arc::new(TestProvider {
        id: read_model.provider.clone(),
        name: "Login".into(),
        models: vec![read_model],
        marker: "done".into(),
        auth: ProviderAuth::api_key(LoginApiAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let cancellation = CancellationToken::new();
    let auth_cancellation = cancellation.clone();
    let auth = tokio::spawn(async move {
        models
            .auth(
                "login",
                AuthResolutionOverrides {
                    cancellation: auth_cancellation,
                    ..Default::default()
                },
            )
            .await
    });
    read_started.notified().await;
    cancellation.cancel();
    assert!(matches!(auth.await.unwrap(), Err(AuthError::Cancelled)));
}

#[tokio::test]
async fn runs_models_login_and_logout_through_the_credential_store() {
    let model = model("login", "gpt-test");
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Login".into(),
        models: vec![model],
        marker: "done".into(),
        auth: ProviderAuth::api_key(LoginApiAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let interaction = TestInteraction::default();
    let credential = models
        .login("login", CredentialType::ApiKey, &interaction)
        .await
        .unwrap();
    assert_eq!(credential, api_key("logged-in"));
    assert_eq!(
        models
            .credentials()
            .read("login", &CancellationToken::new())
            .await
            .unwrap(),
        Some(api_key("logged-in"))
    );

    models
        .logout("login", &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        models
            .credentials()
            .read("login", &CancellationToken::new())
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn returns_a_committed_login_when_cancellation_fires_during_store_mutation() {
    let model = model("login-commit", "gpt-test");
    let cancellation = CancellationToken::new();
    let mut models = Models::with_auth(
        Arc::new(CancelAfterMutationStore {
            cancellation: cancellation.clone(),
            credential: Mutex::new(None),
        }),
        Arc::new(EmptyAuthContext),
    );
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Login".into(),
        models: vec![model],
        marker: "done".into(),
        auth: ProviderAuth::api_key(LoginApiAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let interaction = TestInteraction {
        cancellation: cancellation.clone(),
    };

    assert_eq!(
        models
            .login("login-commit", CredentialType::ApiKey, &interaction)
            .await
            .unwrap(),
        api_key("logged-in")
    );
}

#[tokio::test]
async fn rejects_login_credentials_with_the_wrong_variant() {
    let api_model = model("login-variant", "gpt-test");
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: api_model.provider.clone(),
        name: "Login".into(),
        models: vec![api_model],
        marker: "done".into(),
        auth: ProviderAuth::api_key(WrongVariantApiAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let error = models
        .login(
            "login-variant",
            CredentialType::ApiKey,
            &TestInteraction::default(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, AuthError::Authentication(message) if message.contains("API key login returned an OAuth credential"))
    );

    let oauth_model = model("oauth-variant", "gpt-test");
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: oauth_model.provider.clone(),
        name: "OAuth".into(),
        models: vec![oauth_model],
        marker: "done".into(),
        auth: ProviderAuth::oauth(WrongVariantOAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let error = models
        .login(
            "oauth-variant",
            CredentialType::OAuth,
            &TestInteraction::default(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, AuthError::OAuth(message) if message.contains("OAuth login returned an API key credential"))
    );
}

#[tokio::test]
async fn rejects_empty_api_key_login_credentials_without_storing_them() {
    let model = model("empty-login", "gpt-test");
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Empty login".into(),
        models: vec![model],
        marker: "done".into(),
        auth: ProviderAuth::api_key(EmptyApiAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));

    let error = models
        .login(
            "empty-login",
            CredentialType::ApiKey,
            &TestInteraction::default(),
        )
        .await
        .unwrap_err();

    assert!(
        matches!(error, AuthError::Authentication(message) if message.contains("empty credential"))
    );
    assert_eq!(
        models
            .credentials()
            .read("empty-login", &CancellationToken::new())
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn names_provider_and_type_for_unsupported_login() {
    let model = model("unsupported", "gpt-test");
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Unsupported provider".into(),
        models: vec![model],
        marker: "done".into(),
        auth: ProviderAuth::default(),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let error = models
        .login(
            "unsupported",
            CredentialType::OAuth,
            &TestInteraction::default(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, AuthError::Unsupported(message) if message == "Provider unsupported does not support OAuth login")
    );
}

#[tokio::test]
async fn rejects_a_non_oauth_refresh_without_replacing_the_stored_credential() {
    let model = model("oauth-variant", "gpt-test");
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "OAuth".into(),
        models: vec![model],
        marker: "done".into(),
        auth: ProviderAuth::oauth(WrongVariantOAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    let cancellation = CancellationToken::new();
    let stored = oauth("old", 0);
    models
        .credentials()
        .modify(
            "oauth-variant",
            Box::new({
                let stored = stored.clone();
                move |_| Box::pin(async move { Ok(Some(stored)) })
            }),
            &cancellation,
        )
        .await
        .unwrap();
    let error = models
        .auth("oauth-variant", AuthResolutionOverrides::default())
        .await
        .unwrap_err();
    assert!(
        matches!(error, AuthError::OAuth(message) if message.contains("OAuth refresh returned an API key credential"))
    );
    assert_eq!(
        models
            .credentials()
            .read("oauth-variant", &cancellation)
            .await
            .unwrap(),
        Some(stored)
    );
}

#[tokio::test]
async fn keeps_unknown_login_providers_in_the_provider_error_category() {
    let error = collection()
        .login(
            "missing",
            CredentialType::ApiKey,
            &TestInteraction::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, AuthError::Provider(message) if message == "Unknown provider missing"));
}

#[tokio::test]
async fn keeps_api_and_credential_store_failures_in_their_categories() {
    let api_error_model = model("api-error", "gpt-test");
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: api_error_model.provider.clone(),
        name: "API error".into(),
        models: vec![api_error_model],
        marker: "done".into(),
        auth: ProviderAuth::api_key(FailingApiAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    assert!(matches!(
        models
            .auth(
                "api-error",
                AuthResolutionOverrides {
                    api_key: Some("key".into()),
                    ..Default::default()
                },
            )
            .await,
        Err(AuthError::Authentication(message))
            if message.contains("API key auth failed for provider api-error")
                && message.contains("rejected")
    ));

    let mut models = Models::with_auth(Arc::new(FailingReadStore), Arc::new(EmptyAuthContext));
    models.set_provider(Arc::new(TestProvider {
        id: ProviderId::new("store-error"),
        name: "Store error".into(),
        models: vec![model("store-error", "gpt-test")],
        marker: "done".into(),
        auth: ProviderAuth::api_key(LoginApiAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    assert!(matches!(
        models
            .auth("store-error", AuthResolutionOverrides::default())
            .await,
        Err(AuthError::Store(message)) if message.contains("credential store read")
    ));
}

#[tokio::test]
async fn preserves_auth_failure_reasons_in_stream_setup_errors() {
    let model = model("api-error-stream", "gpt-test");
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "API error".into(),
        models: vec![model.clone()],
        marker: "done".into(),
        auth: ProviderAuth::api_key(FailingApiAuth),
        headers: BTreeMap::new(),
        captured: None,
    }));

    let result = models
        .complete(
            &model.typed().unwrap(),
            &Context::new([]),
            &api_options(StreamOptions::default()),
        )
        .await
        .unwrap();
    assert_eq!(
        result.error_message.as_deref(),
        Some("API key auth failed for provider api-error-stream: rejected")
    );
}

#[tokio::test]
async fn refreshes_oauth_only_when_the_requested_validity_window_requires_it() {
    let model = model("oauth-window", "gpt-test");
    let refreshes = Arc::new(AtomicUsize::new(0));
    let models = oauth_models(
        &model,
        TestOAuth {
            refreshes: refreshes.clone(),
            fail: false,
        },
    );
    let cancellation = CancellationToken::new();
    models
        .credentials()
        .modify(
            "oauth-window",
            Box::new(|_| Box::pin(async { Ok(Some(oauth("valid", now() + 10 * 60_000))) })),
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(
        models
            .auth("oauth-window", AuthResolutionOverrides::default())
            .await
            .unwrap()
            .unwrap()
            .auth
            .api_key
            .as_deref(),
        Some("valid")
    );
    assert_eq!(refreshes.load(Ordering::SeqCst), 0);

    models
        .credentials()
        .modify(
            "oauth-window",
            Box::new(|_| Box::pin(async { Ok(Some(oauth("soon", now() + 60_000))) })),
            &cancellation,
        )
        .await
        .unwrap();
    models
        .auth("oauth-window", AuthResolutionOverrides::default())
        .await
        .unwrap();
    assert_eq!(refreshes.load(Ordering::SeqCst), 1);

    models
        .credentials()
        .modify(
            "oauth-window",
            Box::new(|_| Box::pin(async { Ok(Some(oauth("custom", now() + 10 * 60_000))) })),
            &cancellation,
        )
        .await
        .unwrap();
    models
        .auth(
            "oauth-window",
            AuthResolutionOverrides {
                minimum_oauth_validity: Some(std::time::Duration::from_secs(15 * 60)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(refreshes.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn cancellation_during_oauth_refresh_preserves_the_previous_credential() {
    let model = model("oauth-blocking", "gpt-test");
    let started = Arc::new(Notify::new());
    let models = oauth_models(
        &model,
        BlockingOAuth {
            started: started.clone(),
            block_refresh: true,
            block_to_auth: false,
        },
    );
    let cancellation = CancellationToken::new();
    let old = oauth("old", 0);
    models
        .credentials()
        .modify(
            "oauth-blocking",
            Box::new({
                let old = old.clone();
                move |_| Box::pin(async move { Ok(Some(old)) })
            }),
            &cancellation,
        )
        .await
        .unwrap();
    let credentials = models.credentials();
    let task_cancellation = cancellation.clone();
    let task = tokio::spawn(async move {
        models
            .auth(
                "oauth-blocking",
                AuthResolutionOverrides {
                    cancellation: task_cancellation,
                    ..Default::default()
                },
            )
            .await
    });
    started.notified().await;
    cancellation.cancel();
    assert!(matches!(task.await.unwrap(), Err(AuthError::Cancelled)));
    assert_eq!(
        credentials
            .read("oauth-blocking", &CancellationToken::new())
            .await
            .unwrap(),
        Some(old)
    );
}

#[tokio::test]
async fn cancellation_during_oauth_to_auth_stops_without_changing_credentials() {
    let model = model("oauth-to-auth", "gpt-test");
    let started = Arc::new(Notify::new());
    let models = oauth_models(
        &model,
        BlockingOAuth {
            started: started.clone(),
            block_refresh: false,
            block_to_auth: true,
        },
    );
    let cancellation = CancellationToken::new();
    let valid = oauth("valid", now() + 10 * 60_000);
    models
        .credentials()
        .modify(
            "oauth-to-auth",
            Box::new({
                let valid = valid.clone();
                move |_| Box::pin(async move { Ok(Some(valid)) })
            }),
            &cancellation,
        )
        .await
        .unwrap();
    let credentials = models.credentials();
    let task_cancellation = cancellation.clone();
    let task = tokio::spawn(async move {
        models
            .auth(
                "oauth-to-auth",
                AuthResolutionOverrides {
                    cancellation: task_cancellation,
                    ..Default::default()
                },
            )
            .await
    });
    started.notified().await;
    cancellation.cancel();
    assert!(matches!(task.await.unwrap(), Err(AuthError::Cancelled)));
    assert_eq!(
        credentials
            .read("oauth-to-auth", &CancellationToken::new())
            .await
            .unwrap(),
        Some(valid)
    );
}

#[tokio::test]
async fn returns_no_auth_when_oauth_refresh_finds_removed_or_changed_credentials() {
    for replacement in [None, Some(api_key("changed"))] {
        let model = model("oauth-race", "gpt-test");
        let old = oauth("old", 0);
        let store = Arc::new(PostRefreshStore {
            current: Mutex::new(Some(old.clone())),
            replacement,
        });
        let mut models = Models::with_auth(store, Arc::new(EmptyAuthContext));
        models.set_provider(Arc::new(TestProvider {
            id: model.provider.clone(),
            name: "OAuth".into(),
            models: vec![model],
            marker: "done".into(),
            auth: ProviderAuth::oauth(TestOAuth {
                refreshes: Arc::new(AtomicUsize::new(0)),
                fail: false,
            }),
            headers: BTreeMap::new(),
            captured: None,
        }));
        assert_eq!(
            models
                .auth("oauth-race", AuthResolutionOverrides::default())
                .await
                .unwrap(),
            None
        );
    }
}

#[tokio::test]
async fn resolves_a_stored_anthropic_oauth_credential_through_models() {
    let model = ds_ai::builtin_provider_models("anthropic")
        .first()
        .cloned()
        .unwrap();
    let mut models = Models::with_auth(
        Arc::new(ds_ai::InMemoryCredentialStore::new()),
        Arc::new(EmptyAuthContext),
    );
    models.set_provider(Arc::new(ds_ai::anthropic::Provider::new([model])));
    let cancellation = CancellationToken::new();
    models
        .credentials()
        .modify(
            "anthropic",
            Box::new(|_| {
                Box::pin(async {
                    Ok(Some(Credential::OAuth {
                        refresh: "refresh".into(),
                        access: "access".into(),
                        expires: now() + 10 * 60_000,
                        extra: BTreeMap::new(),
                    }))
                })
            }),
            &cancellation,
        )
        .await
        .unwrap();
    let result = models
        .auth("anthropic", AuthResolutionOverrides::default())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.auth.api_key.as_deref(), Some("access"));
    assert_eq!(result.source.as_deref(), Some("OAuth"));
}

#[tokio::test]
async fn prepares_simple_options_from_the_model_and_context() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse.clone()), Reply::sse(sse)]).await;
    let mut model = model("simple", "gpt-test");
    model.provider = ProviderId::new("openai");
    model.base_url = server.base_url.clone();
    model.context_window = 10_000;
    model.max_tokens = 8_000;
    model
        .sampling_params
        .insert("top_p".into(), serde_json::json!(0.7));
    model
        .sampling_params
        .insert("min_p".into(), serde_json::json!(0.05));
    model
        .sampling_params
        .insert("temperature".into(), serde_json::json!(1));
    model
        .sampling_params
        .insert("seed".into(), serde_json::json!(42));
    let mut models = collection();
    models.set_provider(Arc::new(ds_ai::openai::Provider::new([model.clone()])));
    models
        .complete_simple(
            &model,
            &Context::new([ds_ai::Message::user("x".repeat(4_000))]),
            &SimpleStreamOptions {
                stream: StreamOptions {
                    api_key: Some("key".into()),
                    temperature: Some(0.25),
                    sampling_params: BTreeMap::from([
                        ("temperature".into(), serde_json::json!(0)),
                        ("top_p".into(), serde_json::json!(0.9)),
                        ("top_k".into(), serde_json::json!(0)),
                    ]),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let mut plain_model = model.clone();
    plain_model.sampling_params.clear();
    models
        .complete_simple(
            &plain_model,
            &Context::new([ds_ai::Message::user("Hello")]),
            &SimpleStreamOptions {
                stream: StreamOptions {
                    api_key: Some("key".into()),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let requests = server.requests().await;
    let payload: serde_json::Value =
        serde_json::from_str(requests[0].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(payload["max_output_tokens"], 4_904);
    assert_eq!(payload["seed"], serde_json::json!(42));
    assert_eq!(payload["min_p"], serde_json::json!(0.05));
    assert_eq!(payload["top_p"], serde_json::json!(0.9));
    assert_eq!(payload["top_k"], serde_json::json!(0));
    assert_eq!(payload["temperature"], serde_json::json!(0));
    let plain_payload: serde_json::Value =
        serde_json::from_str(requests[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert!(plain_payload.get("temperature").is_none());
    assert!(plain_payload.get("top_p").is_none());
}

fn provider(model: &Model, marker: &str) -> Arc<dyn Provider> {
    Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Test".into(),
        models: vec![model.clone()],
        marker: marker.into(),
        auth: ds_ai::ProviderAuth::api_key(ds_ai::EnvApiKeyAuth::new("Test key", ["TEST_KEY"])),
        headers: BTreeMap::new(),
        captured: None,
    })
}

fn collection() -> Models {
    Models::with_auth(
        Arc::new(ds_ai::InMemoryCredentialStore::new()),
        Arc::new(EmptyAuthContext),
    )
}

struct EmptyAuthContext;

#[async_trait]
impl ds_ai::AuthContext for EmptyAuthContext {
    async fn env(&self, _name: &str) -> Option<String> {
        None
    }

    async fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

struct AmbientAuthContext;

#[async_trait]
impl ds_ai::AuthContext for AmbientAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        (name == "TEST_KEY").then(|| "ambient".into())
    }

    async fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

struct HeaderAuth;

#[async_trait]
impl ApiKeyAuth for HeaderAuth {
    fn name(&self) -> &str {
        "Header auth"
    }

    async fn resolve(
        &self,
        _context: &dyn AuthContext,
        _credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError> {
        Ok(Some(AuthResult {
            auth: ModelAuth {
                api_key: Some("resolved".into()),
                headers: BTreeMap::from([
                    ("x-test".into(), Some("auth".into())),
                    ("Auth-Only".into(), Some("auth".into())),
                    ("Auth-Added".into(), Some("auth".into())),
                ]),
                base_url: None,
            },
            ..Default::default()
        }))
    }
}

struct TestOAuth {
    refreshes: Arc<AtomicUsize>,
    fail: bool,
}

#[async_trait]
impl OAuthAuth for TestOAuth {
    fn name(&self) -> &str {
        "Test OAuth"
    }

    async fn login(
        &self,
        _interaction: &dyn ds_ai::AuthInteraction,
    ) -> Result<Credential, AuthError> {
        Ok(oauth("login", now().saturating_add(3_600_000)))
    }

    async fn refresh(
        &self,
        _credential: &Credential,
        _cancellation: &CancellationToken,
    ) -> Result<Credential, AuthError> {
        self.refreshes.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
        if self.fail {
            Err(AuthError::OAuth("rejected".into()))
        } else {
            Ok(oauth("refreshed", now().saturating_add(3_600_000)))
        }
    }

    async fn to_auth(&self, credential: &Credential) -> Result<ModelAuth, AuthError> {
        let Credential::OAuth { access, .. } = credential else {
            return Err(AuthError::OAuth("wrong credential".into()));
        };
        Ok(ModelAuth {
            api_key: Some(access.clone()),
            ..Default::default()
        })
    }
}

struct BlockingApiAuth {
    started: Arc<Notify>,
}

struct BarrierApiAuth {
    barrier: Arc<Barrier>,
}

#[async_trait]
impl ApiKeyAuth for BarrierApiAuth {
    fn name(&self) -> &str {
        "Barrier auth"
    }

    async fn check(
        &self,
        _context: &dyn AuthContext,
        _credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthCheck>, AuthError> {
        self.barrier.wait().await;
        Ok(Some(AuthCheck {
            source: Some("barrier".into()),
            credential_type: CredentialType::ApiKey,
        }))
    }

    async fn resolve(
        &self,
        _context: &dyn AuthContext,
        _credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError> {
        Ok(None)
    }
}

#[async_trait]
impl ApiKeyAuth for BlockingApiAuth {
    fn name(&self) -> &str {
        "Blocking auth"
    }

    async fn check(
        &self,
        _context: &dyn AuthContext,
        _credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthCheck>, AuthError> {
        self.started.notify_one();
        pending::<Result<Option<AuthCheck>, AuthError>>().await
    }

    async fn login(
        &self,
        _interaction: &dyn ds_ai::AuthInteraction,
    ) -> Result<Credential, AuthError> {
        self.started.notify_one();
        pending::<Result<Credential, AuthError>>().await
    }

    async fn resolve(
        &self,
        _context: &dyn AuthContext,
        _credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError> {
        self.started.notify_one();
        pending::<Result<Option<AuthResult>, AuthError>>().await
    }
}

struct LoginApiAuth;

#[async_trait]
impl ApiKeyAuth for LoginApiAuth {
    fn name(&self) -> &str {
        "Login auth"
    }

    async fn login(
        &self,
        _interaction: &dyn ds_ai::AuthInteraction,
    ) -> Result<Credential, AuthError> {
        Ok(api_key("logged-in"))
    }

    async fn resolve(
        &self,
        _context: &dyn AuthContext,
        credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError> {
        Ok(credential.map(|_| AuthResult::default()))
    }
}

struct EmptyApiAuth;

#[async_trait]
impl ApiKeyAuth for EmptyApiAuth {
    fn name(&self) -> &str {
        "Empty auth"
    }

    async fn login(
        &self,
        _interaction: &dyn ds_ai::AuthInteraction,
    ) -> Result<Credential, AuthError> {
        Ok(api_key("   "))
    }

    async fn resolve(
        &self,
        _context: &dyn AuthContext,
        _credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError> {
        Ok(None)
    }
}

struct WrongVariantApiAuth;

#[async_trait]
impl ApiKeyAuth for WrongVariantApiAuth {
    fn name(&self) -> &str {
        "Wrong variant auth"
    }

    async fn login(
        &self,
        _interaction: &dyn ds_ai::AuthInteraction,
    ) -> Result<Credential, AuthError> {
        Ok(oauth("wrong", now() + 3_600_000))
    }

    async fn resolve(
        &self,
        _context: &dyn AuthContext,
        _credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError> {
        Ok(None)
    }
}

struct WrongVariantOAuth;

#[async_trait]
impl OAuthAuth for WrongVariantOAuth {
    fn name(&self) -> &str {
        "Wrong variant OAuth"
    }

    async fn login(
        &self,
        _interaction: &dyn ds_ai::AuthInteraction,
    ) -> Result<Credential, AuthError> {
        Ok(api_key("wrong"))
    }

    async fn refresh(
        &self,
        _credential: &Credential,
        _cancellation: &CancellationToken,
    ) -> Result<Credential, AuthError> {
        Ok(api_key("wrong"))
    }

    async fn to_auth(&self, _credential: &Credential) -> Result<ModelAuth, AuthError> {
        Err(AuthError::OAuth("wrong credential".into()))
    }
}

struct FailingApiAuth;

#[async_trait]
impl ApiKeyAuth for FailingApiAuth {
    fn name(&self) -> &str {
        "Failing auth"
    }

    async fn resolve(
        &self,
        _context: &dyn AuthContext,
        _credential: Option<&Credential>,
        _cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError> {
        Err(AuthError::Authentication("rejected".into()))
    }
}

struct BlockingOAuth {
    started: Arc<Notify>,
    block_refresh: bool,
    block_to_auth: bool,
}

#[async_trait]
impl OAuthAuth for BlockingOAuth {
    fn name(&self) -> &str {
        "Blocking OAuth"
    }

    async fn login(
        &self,
        _interaction: &dyn ds_ai::AuthInteraction,
    ) -> Result<Credential, AuthError> {
        Err(AuthError::Unsupported("OAuth login".into()))
    }

    async fn refresh(
        &self,
        _credential: &Credential,
        _cancellation: &CancellationToken,
    ) -> Result<Credential, AuthError> {
        if self.block_refresh {
            self.started.notify_one();
            pending::<Result<Credential, AuthError>>().await
        } else {
            Ok(oauth("refreshed", now() + 3_600_000))
        }
    }

    async fn to_auth(&self, credential: &Credential) -> Result<ModelAuth, AuthError> {
        if self.block_to_auth {
            self.started.notify_one();
            pending::<Result<ModelAuth, AuthError>>().await
        } else {
            let Credential::OAuth { access, .. } = credential else {
                return Err(AuthError::OAuth("wrong credential".into()));
            };
            Ok(ModelAuth {
                api_key: Some(access.clone()),
                ..Default::default()
            })
        }
    }
}

struct TestInteraction {
    cancellation: CancellationToken,
}

impl Default for TestInteraction {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
        }
    }
}

#[async_trait]
impl ds_ai::AuthInteraction for TestInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, _prompt: ds_ai::AuthPrompt) -> Result<String, AuthError> {
        Ok(String::new())
    }

    fn notify(&self, _event: ds_ai::AuthEvent) {}
}

struct BlockingStore {
    read_started: Arc<Notify>,
    delete_started: Arc<Notify>,
}

struct CancelAfterMutationStore {
    cancellation: CancellationToken,
    credential: Mutex<Option<Credential>>,
}

#[async_trait]
impl ds_ai::CredentialStore for CancelAfterMutationStore {
    async fn read(
        &self,
        _provider_id: &str,
        _cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        Ok(self.credential.lock().unwrap().clone())
    }

    async fn list(
        &self,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ds_ai::CredentialInfo>, AuthError> {
        Ok(Vec::new())
    }

    async fn modify(
        &self,
        _provider_id: &str,
        mutation: ds_ai::CredentialMutation,
        _cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        let current = self.credential.lock().unwrap().clone();
        let next = mutation(current).await?;
        self.cancellation.cancel();
        tokio::task::yield_now().await;
        *self.credential.lock().unwrap() = next.clone();
        Ok(next)
    }

    async fn delete(
        &self,
        _provider_id: &str,
        _cancellation: &CancellationToken,
    ) -> Result<(), AuthError> {
        *self.credential.lock().unwrap() = None;
        Ok(())
    }
}

struct PostRefreshStore {
    current: Mutex<Option<Credential>>,
    replacement: Option<Credential>,
}

struct FailingReadStore;

#[async_trait]
impl ds_ai::CredentialStore for FailingReadStore {
    async fn read(
        &self,
        _provider_id: &str,
        _cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        Err(AuthError::Authentication("read rejected".into()))
    }

    async fn list(
        &self,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ds_ai::CredentialInfo>, AuthError> {
        Ok(Vec::new())
    }

    async fn modify(
        &self,
        _provider_id: &str,
        _mutation: ds_ai::CredentialMutation,
        _cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        Ok(None)
    }

    async fn delete(
        &self,
        _provider_id: &str,
        _cancellation: &CancellationToken,
    ) -> Result<(), AuthError> {
        Ok(())
    }
}

#[async_trait]
impl ds_ai::CredentialStore for PostRefreshStore {
    async fn read(
        &self,
        _provider_id: &str,
        _cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        Ok(self.current.lock().unwrap().clone())
    }

    async fn list(
        &self,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ds_ai::CredentialInfo>, AuthError> {
        Ok(Vec::new())
    }

    async fn modify(
        &self,
        _provider_id: &str,
        _mutation: ds_ai::CredentialMutation,
        _cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        *self.current.lock().unwrap() = self.replacement.clone();
        Ok(self.replacement.clone())
    }

    async fn delete(
        &self,
        _provider_id: &str,
        _cancellation: &CancellationToken,
    ) -> Result<(), AuthError> {
        *self.current.lock().unwrap() = None;
        Ok(())
    }
}

#[async_trait]
impl ds_ai::CredentialStore for BlockingStore {
    async fn read(
        &self,
        _provider_id: &str,
        _cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        self.read_started.notify_one();
        pending::<Result<Option<Credential>, AuthError>>().await
    }

    async fn list(
        &self,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ds_ai::CredentialInfo>, AuthError> {
        Ok(Vec::new())
    }

    async fn modify(
        &self,
        _provider_id: &str,
        _mutation: ds_ai::CredentialMutation,
        _cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        Ok(None)
    }

    async fn delete(
        &self,
        _provider_id: &str,
        _cancellation: &CancellationToken,
    ) -> Result<(), AuthError> {
        self.delete_started.notify_one();
        pending::<Result<(), AuthError>>().await
    }
}

fn oauth_models(model: &Model, oauth: impl OAuthAuth + 'static) -> Models {
    let mut models = collection();
    models.set_provider(Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "OAuth".into(),
        models: vec![model.clone()],
        marker: "done".into(),
        auth: ProviderAuth::oauth(oauth),
        headers: BTreeMap::new(),
        captured: None,
    }));
    models
}

fn api_key(key: &str) -> Credential {
    Credential::ApiKey {
        key: Some(key.into()),
        env: BTreeMap::new(),
    }
}

fn oauth(access: &str, expires: u64) -> Credential {
    Credential::OAuth {
        refresh: "refresh".into(),
        access: access.into(),
        expires,
        extra: BTreeMap::new(),
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap()
}

fn model(provider: &str, id: &str) -> Model {
    Model {
        id: id.into(),
        name: id.into(),
        api: Api::OpenAiResponses,
        provider: ProviderId::new(provider),
        base_url: "https://example.com".into(),
        reasoning: false,
        thinking_level_map: BTreeMap::new(),
        input: vec![ModelInput::Text],
        cost: ModelCost::default(),
        context_window: 100,
        max_tokens: 20,
        sampling_params: BTreeMap::new(),
        headers: BTreeMap::new(),
        compat: None,
    }
}

fn api_options(stream: StreamOptions) -> OpenAiResponsesOptions {
    OpenAiResponsesOptions {
        stream,
        ..Default::default()
    }
}

fn completed(model: &Model, marker: &str) -> AssistantMessageEventStream {
    let message = AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::Stop,
        error_message: Some(marker.into()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 42,
    };
    AssistantMessageEventStream::new(stream::iter([AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message,
    }]))
}
