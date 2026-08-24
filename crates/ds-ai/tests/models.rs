use async_trait::async_trait;
use ds_ai::{
    Api, ApiKeyAuth, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    AuthCheck, AuthContext, AuthError, AuthResolutionOverrides, AuthResult, Context, Credential,
    CredentialType, Model, ModelAuth, ModelCost, ModelInput, Models, OAuthAuth, Provider,
    ProviderAuth, ProviderId, SimpleStreamOptions, StopReason, StreamOptions,
};
use futures_util::StreamExt;
use futures_util::stream;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::{collections::BTreeMap, sync::Arc};
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
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        if let Some(captured) = &self.captured {
            *captured.lock().unwrap() = Some(options.clone());
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
            &model,
            &Context::new([]),
            &StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
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
async fn returns_terminal_stream_errors_for_unknown_providers() {
    let model = model("missing", "gpt-test");
    let result = collection()
        .complete(&model, &Context::new([]), &StreamOptions::default())
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Unknown provider missing")
    );
}

#[tokio::test]
async fn returns_terminal_stream_errors_for_unknown_provider_apis() {
    let mut model = model("openai", "gpt-test");
    model.api = Api::AnthropicMessages;
    let mut models = collection();
    models.set_provider(ds_ai::openai::provider([model.clone()]));
    let result = models
        .complete(
            &model,
            &Context::new([]),
            &StreamOptions {
                api_key: Some("test-key".into()),
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
async fn openai_provider_returns_a_stream_before_setup_and_emits_pi_events() {
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
    models.set_provider(ds_ai::openai::provider([model.clone()]));
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let events = models
        .stream(
            &model,
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
    assert!(
        models
            .auth("oauth", AuthResolutionOverrides::default())
            .await
            .is_err()
    );
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
async fn merges_provider_model_auth_and_request_headers_once() {
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
            &model,
            &Context::new([]),
            &StreamOptions {
                api_key: Some("key".into()),
                headers: BTreeMap::from([
                    ("X-TEST".into(), Some("request".into())),
                    ("auth-only".into(), None),
                ]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        captured.lock().unwrap().as_ref().unwrap().headers,
        BTreeMap::from([
            ("Auth-Added".into(), Some("auth".into())),
            ("Provider-Only".into(), Some("provider".into())),
            ("X-TEST".into(), Some("request".into())),
            ("auth-only".into(), None),
        ])
    );
}

#[tokio::test]
async fn prepares_simple_options_from_the_model_and_context() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
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
        .insert("seed".into(), serde_json::json!(42));
    let mut models = collection();
    models.set_provider(ds_ai::openai::provider([model.clone()]));
    models
        .complete_simple(
            &model,
            &Context::new([ds_ai::Message::user("x".repeat(4_000))]),
            &SimpleStreamOptions {
                stream: StreamOptions {
                    api_key: Some("key".into()),
                    sampling_params: BTreeMap::from([("top_p".into(), serde_json::json!(0.9))]),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let request = server.requests().await.pop().unwrap();
    let payload: serde_json::Value =
        serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(payload["max_output_tokens"], 4_904);
    assert_eq!(payload["seed"], serde_json::json!(42));
    assert_eq!(payload["top_p"], serde_json::json!(0.9));
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
        reason: StopReason::Stop,
        message,
    }]))
}
