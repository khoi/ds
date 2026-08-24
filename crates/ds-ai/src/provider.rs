use crate::{
    AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantMessageStreamError, CacheRetention, Context, Model, ProviderId, StopReason,
    ThinkingLevel,
};
use async_stream::stream;
use async_trait::async_trait;
use futures_util::{StreamExt, stream as futures_stream};
use std::{
    collections::BTreeMap,
    fmt,
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;

type PayloadFuture =
    Pin<Box<dyn Future<Output = Result<Option<serde_json::Value>, String>> + Send + 'static>>;
type ResponseFuture = Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'static>>;

#[derive(Clone)]
pub struct PayloadHook {
    hook: Arc<dyn Fn(serde_json::Value, Model) -> PayloadFuture + Send + Sync>,
}

impl PayloadHook {
    pub fn new<F, Fut>(hook: F) -> Self
    where
        F: Fn(serde_json::Value, Model) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<serde_json::Value>, String>> + Send + 'static,
    {
        Self {
            hook: Arc::new(move |payload, model| Box::pin(hook(payload, model))),
        }
    }

    async fn run(
        &self,
        payload: serde_json::Value,
        model: Model,
    ) -> Result<Option<serde_json::Value>, String> {
        (self.hook)(payload, model).await
    }
}

impl fmt::Debug for PayloadHook {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PayloadHook")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
}

#[derive(Clone)]
pub struct ResponseHook {
    hook: Arc<dyn Fn(ProviderResponse, Model) -> ResponseFuture + Send + Sync>,
}

impl ResponseHook {
    pub fn new<F, Fut>(hook: F) -> Self
    where
        F: Fn(ProviderResponse, Model) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), String>> + Send + 'static,
    {
        Self {
            hook: Arc::new(move |response, model| Box::pin(hook(response, model))),
        }
    }

    async fn run(&self, response: ProviderResponse, model: Model) -> Result<(), String> {
        (self.hook)(response, model).await
    }
}

impl fmt::Debug for ResponseHook {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ResponseHook")
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RequestHooks {
    model: Model,
    payload: Option<PayloadHook>,
    response: Option<ResponseHook>,
}

impl RequestHooks {
    pub(crate) async fn payload(
        &self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, crate::Error> {
        let Some(hook) = &self.payload else {
            return Ok(payload);
        };
        Ok(hook
            .run(payload.clone(), self.model.clone())
            .await
            .map_err(crate::Error::Hook)?
            .unwrap_or(payload))
    }

    pub(crate) async fn response(&self, response: ProviderResponse) -> Result<(), crate::Error> {
        let Some(hook) = &self.response else {
            return Ok(());
        };
        hook.run(response, self.model.clone())
            .await
            .map_err(crate::Error::Hook)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Transport {
    Sse,
    WebSocket,
    WebSocketCached,
    Auto,
}

#[derive(Clone, Debug)]
pub struct StreamOptions {
    pub cancellation: CancellationToken,
    pub api_key: Option<String>,
    pub env: BTreeMap<String, String>,
    pub headers: BTreeMap<String, Option<String>>,
    pub on_payload: Option<PayloadHook>,
    pub on_response: Option<ResponseHook>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    pub max_retry_delay: Option<Duration>,
    pub temperature: Option<f64>,
    pub sampling_params: BTreeMap<String, serde_json::Value>,
    pub max_tokens: Option<u64>,
    pub transport: Option<Transport>,
    pub cache_retention: CacheRetention,
    pub session_id: Option<String>,
    pub websocket_connect_timeout: Option<Duration>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            api_key: None,
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
            on_payload: None,
            on_response: None,
            timeout: None,
            max_retries: None,
            max_retry_delay: Some(Duration::from_secs(60)),
            temperature: None,
            sampling_params: BTreeMap::new(),
            max_tokens: None,
            transport: None,
            cache_retention: CacheRetention::default(),
            session_id: None,
            websocket_connect_timeout: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl StreamOptions {
    pub(crate) fn request_hooks(&self, model: &Model) -> RequestHooks {
        RequestHooks {
            model: model.clone(),
            payload: self.on_payload.clone(),
            response: self.on_response.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
}

#[derive(Clone, Debug, Default)]
pub struct ThinkingBudgets {
    pub minimal: Option<u64>,
    pub low: Option<u64>,
    pub medium: Option<u64>,
    pub high: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct SimpleStreamOptions {
    pub stream: StreamOptions,
    pub thinking: Option<ThinkingLevel>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub tool_choice: ToolChoice,
}

#[derive(Clone, Debug, Default)]
pub struct OpenAiResponsesOptions {
    pub stream: StreamOptions,
    pub reasoning_effort: Option<crate::openai::ReasoningEffort>,
    pub reasoning_summary: Option<crate::openai::ReasoningSummary>,
    pub service_tier: Option<crate::openai::ServiceTier>,
    pub tool_choice: Option<crate::openai::ToolChoice>,
}

#[derive(Clone, Debug, Default)]
pub struct AnthropicOptions {
    pub stream: StreamOptions,
    pub thinking_enabled: Option<bool>,
    pub thinking_budget_tokens: Option<u64>,
    pub effort: Option<crate::anthropic::Effort>,
    pub thinking_display: Option<crate::anthropic::ThinkingDisplay>,
    pub interleaved_thinking: Option<bool>,
    pub tool_choice: Option<crate::anthropic::ToolChoice>,
}

#[derive(Clone, Debug, Default)]
pub struct OpenAiCodexResponsesOptions {
    pub stream: StreamOptions,
    pub reasoning_effort: Option<crate::codex::ReasoningEffort>,
    pub reasoning_summary: Option<crate::codex::ReasoningSummary>,
    pub service_tier: Option<crate::codex::ServiceTier>,
    pub text_verbosity: Option<crate::codex::TextVerbosity>,
    pub tool_choice: Option<crate::codex::ToolChoice>,
}

#[derive(Clone, Debug)]
pub enum ApiStreamOptions {
    OpenAiResponses(OpenAiResponsesOptions),
    AnthropicMessages(AnthropicOptions),
    OpenAiCodexResponses(OpenAiCodexResponsesOptions),
    Other(StreamOptions),
}

impl ApiStreamOptions {
    pub fn for_model(model: &Model) -> Self {
        match model.api {
            crate::Api::OpenAiResponses => Self::OpenAiResponses(Default::default()),
            crate::Api::AnthropicMessages => Self::AnthropicMessages(Default::default()),
            crate::Api::OpenAiCodexResponses => Self::OpenAiCodexResponses(Default::default()),
            crate::Api::Other(_) => Self::Other(Default::default()),
        }
    }

    pub fn stream(&self) -> &StreamOptions {
        match self {
            Self::OpenAiResponses(options) => &options.stream,
            Self::AnthropicMessages(options) => &options.stream,
            Self::OpenAiCodexResponses(options) => &options.stream,
            Self::Other(options) => options,
        }
    }

    fn stream_mut(&mut self) -> &mut StreamOptions {
        match self {
            Self::OpenAiResponses(options) => &mut options.stream,
            Self::AnthropicMessages(options) => &mut options.stream,
            Self::OpenAiCodexResponses(options) => &mut options.stream,
            Self::Other(options) => options,
        }
    }
}

pub trait Provider: Send + Sync {
    fn id(&self) -> &ProviderId;
    fn name(&self) -> &str;
    fn base_url(&self) -> Option<&str>;
    fn headers(&self) -> &BTreeMap<String, Option<String>>;
    fn auth(&self) -> &crate::ProviderAuth;
    fn models(&self) -> Vec<Model>;
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &ApiStreamOptions,
    ) -> AssistantMessageEventStream;
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream;
}

pub struct Models {
    providers: Vec<Arc<dyn Provider>>,
    credentials: Arc<dyn crate::CredentialStore>,
    auth_context: Arc<dyn crate::AuthContext>,
}

impl Default for Models {
    fn default() -> Self {
        Self {
            providers: Vec::new(),
            credentials: Arc::new(crate::InMemoryCredentialStore::new()),
            auth_context: Arc::new(crate::SystemAuthContext),
        }
    }
}

impl Models {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_auth(
        credentials: Arc<dyn crate::CredentialStore>,
        auth_context: Arc<dyn crate::AuthContext>,
    ) -> Self {
        Self {
            providers: Vec::new(),
            credentials,
            auth_context,
        }
    }

    pub fn credentials(&self) -> Arc<dyn crate::CredentialStore> {
        self.credentials.clone()
    }

    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        if let Some(current) = self
            .providers
            .iter_mut()
            .find(|current| current.id() == provider.id())
        {
            *current = provider;
        } else {
            self.providers.push(provider);
        }
    }

    pub fn delete_provider(&mut self, id: &str) -> Option<Arc<dyn Provider>> {
        let index = self
            .providers
            .iter()
            .position(|provider| provider.id().as_str() == id)?;
        Some(self.providers.remove(index))
    }

    pub fn clear_providers(&mut self) {
        self.providers.clear();
    }

    pub fn providers(&self) -> Vec<Arc<dyn Provider>> {
        self.providers.clone()
    }

    pub fn provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|provider| provider.id().as_str() == id)
            .cloned()
    }

    pub fn models(&self, provider: Option<&str>) -> Vec<Model> {
        match provider {
            Some(provider) => self
                .providers
                .iter()
                .find(|entry| entry.id().as_str() == provider)
                .map_or_else(Vec::new, |entry| entry.models()),
            None => self
                .providers
                .iter()
                .flat_map(|provider| provider.models())
                .collect(),
        }
    }

    pub fn model(&self, provider: &str, id: &str) -> Option<Model> {
        self.models(Some(provider))
            .into_iter()
            .find(|model| model.id == id)
    }

    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &ApiStreamOptions,
    ) -> AssistantMessageEventStream {
        let Some(provider) = self.provider(model.provider.as_str()) else {
            return error_stream(model, format!("Unknown provider {}", model.provider));
        };
        let model = model.clone();
        let context = context.clone();
        let options = options.clone();
        let credentials = self.credentials.clone();
        let auth_context = self.auth_context.clone();
        lazy_stream(model.clone(), async move {
            let resolution = resolve_auth(
                provider.clone(),
                credentials,
                auth_context,
                crate::AuthResolutionOverrides {
                    api_key: options.stream().api_key.clone(),
                    env: options.stream().env.clone(),
                    cancellation: options.stream().cancellation.clone(),
                    ..Default::default()
                },
            )
            .await?
            .ok_or_else(|| {
                crate::AuthError::Authentication(format!(
                    "Provider is not configured: {}",
                    model.provider
                ))
            })?;
            let (request_model, request_options) =
                apply_auth(&provider, &model, options, resolution);
            Ok(provider.stream(&request_model, &context, &request_options))
        })
    }

    pub async fn complete(
        &self,
        model: &Model,
        context: &Context,
        options: &ApiStreamOptions,
    ) -> Result<AssistantMessage, AssistantMessageStreamError> {
        self.stream(model, context, options).result().await
    }

    pub fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let Some(provider) = self.provider(model.provider.as_str()) else {
            return error_stream(model, format!("Unknown provider {}", model.provider));
        };
        let model = model.clone();
        let context = context.clone();
        let options = options.clone();
        let credentials = self.credentials.clone();
        let auth_context = self.auth_context.clone();
        lazy_stream(model.clone(), async move {
            let resolution = resolve_auth(
                provider.clone(),
                credentials,
                auth_context,
                crate::AuthResolutionOverrides {
                    api_key: options.stream.api_key.clone(),
                    env: options.stream.env.clone(),
                    cancellation: options.stream.cancellation.clone(),
                    ..Default::default()
                },
            )
            .await?
            .ok_or_else(|| {
                crate::AuthError::Authentication(format!(
                    "Provider is not configured: {}",
                    model.provider
                ))
            })?;
            let mut stream_options = options.stream;
            let request_model =
                apply_stream_auth(&provider, &model, &mut stream_options, resolution);
            Ok(provider.stream_simple(
                &request_model,
                &context,
                &SimpleStreamOptions {
                    stream: stream_options,
                    thinking: options.thinking,
                    thinking_budgets: options.thinking_budgets,
                    tool_choice: options.tool_choice,
                },
            ))
        })
    }

    pub async fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> Result<AssistantMessage, AssistantMessageStreamError> {
        self.stream_simple(model, context, options).result().await
    }

    pub async fn auth(
        &self,
        provider_id: &str,
        overrides: crate::AuthResolutionOverrides,
    ) -> Result<Option<crate::AuthResult>, crate::AuthError> {
        let Some(provider) = self.provider(provider_id) else {
            return Ok(None);
        };
        resolve_auth(
            provider,
            self.credentials.clone(),
            self.auth_context.clone(),
            overrides,
        )
        .await
    }

    pub async fn check_auth(
        &self,
        provider_id: &str,
        cancellation: &CancellationToken,
    ) -> Result<Option<crate::AuthCheck>, crate::AuthError> {
        let Some(provider) = self.provider(provider_id) else {
            return Ok(None);
        };
        let credential = self.credentials.read(provider_id, cancellation).await?;
        check_provider_auth(
            &provider,
            credential.as_ref(),
            self.auth_context.as_ref(),
            cancellation,
        )
        .await
    }

    pub async fn available_models(
        &self,
        provider_id: Option<&str>,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Model>, crate::AuthError> {
        let providers = match provider_id {
            Some(provider_id) => self.provider(provider_id).into_iter().collect(),
            None => self.providers(),
        };
        let mut models = Vec::new();
        for provider in providers {
            if self
                .check_auth(provider.id().as_str(), cancellation)
                .await?
                .is_some()
            {
                models.extend(provider.models());
            }
        }
        Ok(models)
    }

    pub async fn login(
        &self,
        provider_id: &str,
        credential_type: crate::CredentialType,
        interaction: &dyn crate::AuthInteraction,
    ) -> Result<crate::Credential, crate::AuthError> {
        let provider = self.provider(provider_id).ok_or_else(|| {
            crate::AuthError::Authentication(format!("Unknown provider {provider_id}"))
        })?;
        let credential = match credential_type {
            crate::CredentialType::ApiKey => {
                provider
                    .auth()
                    .api_key
                    .as_ref()
                    .ok_or_else(|| crate::AuthError::Unsupported("API key login".into()))?
                    .login(interaction)
                    .await?
            }
            crate::CredentialType::OAuth => {
                provider
                    .auth()
                    .oauth
                    .as_ref()
                    .ok_or_else(|| crate::AuthError::Unsupported("OAuth login".into()))?
                    .login(interaction)
                    .await?
            }
        };
        let saved = credential.clone();
        self.credentials
            .modify(
                provider_id,
                Box::new(move |_| Box::pin(async move { Ok(Some(saved)) })),
                interaction.cancellation(),
            )
            .await?;
        Ok(credential)
    }

    pub async fn logout(
        &self,
        provider_id: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), crate::AuthError> {
        self.credentials.delete(provider_id, cancellation).await
    }
}

pub(crate) fn build_simple_stream_options(
    model: &Model,
    context: &Context,
    mut options: StreamOptions,
) -> StreamOptions {
    let mut sampling_params = model.sampling_params.clone();
    sampling_params.extend(options.sampling_params);
    options.sampling_params = sampling_params;
    options.max_tokens = Some(crate::clamp_max_tokens_to_context(
        model,
        context,
        options.max_tokens.unwrap_or(model.max_tokens),
    ));
    options
}

fn error_stream(model: &Model, message: String) -> AssistantMessageEventStream {
    AssistantMessageEventStream::new(futures_stream::iter([error_event(model, message)]))
}

fn error_event(model: &Model, message: String) -> AssistantMessageEvent {
    let error = AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::Error,
        error_message: Some(message),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: timestamp(),
    };
    AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error,
    }
}

fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn lazy_stream(
    model: Model,
    setup: impl Future<Output = Result<AssistantMessageEventStream, crate::AuthError>> + Send + 'static,
) -> AssistantMessageEventStream {
    let output = stream! {
        let mut source = match setup.await {
            Ok(source) => source,
            Err(error) => {
                yield error_event(&model, error.to_string());
                return;
            }
        };
        let mut terminal = false;
        while let Some(event) = source.next().await {
            terminal = matches!(
                event,
                AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
            );
            yield event;
        }
        if !terminal {
            yield error_event(&model, "Provider stream ended without a terminal event".into());
        }
    };
    AssistantMessageEventStream::new(output)
}

fn apply_auth(
    provider: &Arc<dyn Provider>,
    model: &Model,
    mut options: ApiStreamOptions,
    resolution: crate::AuthResult,
) -> (Model, ApiStreamOptions) {
    let stream = options.stream_mut();
    let model = apply_stream_auth(provider, model, stream, resolution);
    (model, options)
}

fn apply_stream_auth(
    provider: &Arc<dyn Provider>,
    model: &Model,
    stream: &mut StreamOptions,
    resolution: crate::AuthResult,
) -> Model {
    let mut model = model.clone();
    if let Some(base_url) = resolution.auth.base_url {
        model.base_url = base_url;
    }
    stream.api_key = stream.api_key.take().or(resolution.auth.api_key);
    stream.env = resolution
        .env
        .into_iter()
        .chain(std::mem::take(&mut stream.env))
        .collect();
    stream.headers = merge_headers(
        provider.headers(),
        &model.headers,
        &resolution.auth.headers,
        &stream.headers,
    );
    model
}

fn merge_headers(
    provider: &BTreeMap<String, Option<String>>,
    model: &BTreeMap<String, String>,
    auth: &BTreeMap<String, Option<String>>,
    request: &BTreeMap<String, Option<String>>,
) -> BTreeMap<String, Option<String>> {
    let mut headers = BTreeMap::new();
    for (name, value) in provider {
        insert_header(&mut headers, name, value.clone());
    }
    for (name, value) in model {
        insert_header(&mut headers, name, Some(value.clone()));
    }
    for (name, value) in auth {
        insert_header(&mut headers, name, value.clone());
    }
    for (name, value) in request {
        insert_header(&mut headers, name, value.clone());
    }
    headers
}

fn insert_header(
    headers: &mut BTreeMap<String, Option<String>>,
    name: &str,
    value: Option<String>,
) {
    if let Some(existing) = headers
        .keys()
        .find(|existing| existing.eq_ignore_ascii_case(name))
        .cloned()
    {
        headers.remove(&existing);
    }
    headers.insert(name.to_owned(), value);
}

async fn resolve_auth(
    provider: Arc<dyn Provider>,
    credentials: Arc<dyn crate::CredentialStore>,
    auth_context: Arc<dyn crate::AuthContext>,
    overrides: crate::AuthResolutionOverrides,
) -> Result<Option<crate::AuthResult>, crate::AuthError> {
    if overrides.cancellation.is_cancelled() {
        return Err(crate::AuthError::Cancelled);
    }
    let context = OverlayAuthContext {
        base: auth_context,
        env: overrides.env.clone(),
    };
    if let (Some(api_key), Some(auth)) = (&overrides.api_key, &provider.auth().api_key) {
        return auth
            .resolve(
                &context,
                Some(&crate::Credential::ApiKey {
                    key: Some(api_key.clone()),
                    env: overrides.env,
                }),
                &overrides.cancellation,
            )
            .await;
    }
    let stored = credentials
        .read(provider.id().as_str(), &overrides.cancellation)
        .await?;
    match stored {
        Some(credential @ crate::Credential::OAuth { .. }) => {
            let Some(oauth) = &provider.auth().oauth else {
                return Ok(None);
            };
            resolve_oauth(
                credentials,
                provider.id().clone(),
                oauth.clone(),
                credential,
                overrides,
            )
            .await
        }
        Some(mut credential @ crate::Credential::ApiKey { .. }) => {
            let Some(auth) = &provider.auth().api_key else {
                return Ok(None);
            };
            if let crate::Credential::ApiKey { env, .. } = &mut credential {
                env.extend(overrides.env);
            }
            auth.resolve(&context, Some(&credential), &overrides.cancellation)
                .await
        }
        None => match &provider.auth().api_key {
            Some(auth) => auth.resolve(&context, None, &overrides.cancellation).await,
            None => Ok(None),
        },
    }
}

async fn resolve_oauth(
    credentials: Arc<dyn crate::CredentialStore>,
    provider_id: ProviderId,
    oauth: Arc<dyn crate::OAuthAuth>,
    stored: crate::Credential,
    overrides: crate::AuthResolutionOverrides,
) -> Result<Option<crate::AuthResult>, crate::AuthError> {
    let minimum = overrides
        .minimum_oauth_validity
        .unwrap_or(Duration::from_secs(5 * 60))
        .max(Duration::from_secs(5 * 60));
    let explicit_minimum = overrides.minimum_oauth_validity.is_some();
    let mut credential = stored;
    if expires_soon(&credential, minimum) {
        let refresh = oauth.clone();
        let cancellation = overrides.cancellation.clone();
        credential = credentials
            .modify(
                provider_id.as_str(),
                Box::new(move |current| {
                    Box::pin(async move {
                        let Some(current @ crate::Credential::OAuth { .. }) = current else {
                            return Ok(None);
                        };
                        if !expires_soon(&current, minimum) {
                            return Ok(None);
                        }
                        match tokio::time::timeout(
                            Duration::from_secs(15),
                            refresh.refresh(&current, &cancellation),
                        )
                        .await
                        {
                            Ok(result) => result.map(Some),
                            Err(_) => Err(crate::AuthError::OAuth("refresh timed out".into())),
                        }
                    })
                }),
                &overrides.cancellation,
            )
            .await?
            .ok_or_else(|| crate::AuthError::OAuth("credential was removed".into()))?;
        if explicit_minimum && expires_soon(&credential, minimum) {
            return Err(crate::AuthError::OAuth(
                "refresh returned a token that expires too soon".into(),
            ));
        }
    }
    let auth = oauth.to_auth(&credential).await?;
    Ok(Some(crate::AuthResult {
        auth,
        source: Some("OAuth".into()),
        ..Default::default()
    }))
}

fn expires_soon(credential: &crate::Credential, minimum: Duration) -> bool {
    let crate::Credential::OAuth { expires, .. } = credential else {
        return false;
    };
    timestamp().saturating_add(duration_millis(minimum)) >= *expires
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

async fn check_provider_auth(
    provider: &Arc<dyn Provider>,
    credential: Option<&crate::Credential>,
    auth_context: &dyn crate::AuthContext,
    cancellation: &CancellationToken,
) -> Result<Option<crate::AuthCheck>, crate::AuthError> {
    match credential {
        Some(crate::Credential::OAuth { .. }) if provider.auth().oauth.is_some() => {
            Ok(Some(crate::AuthCheck {
                source: Some("OAuth".into()),
                credential_type: crate::CredentialType::OAuth,
            }))
        }
        Some(crate::Credential::ApiKey { .. }) | None => match &provider.auth().api_key {
            Some(auth) => auth.check(auth_context, credential, cancellation).await,
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

struct OverlayAuthContext {
    base: Arc<dyn crate::AuthContext>,
    env: BTreeMap<String, String>,
}

#[async_trait]
impl crate::AuthContext for OverlayAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        match self.env.get(name).filter(|value| !value.is_empty()) {
            Some(value) => Some(value.clone()),
            None => self.base.env(name).await,
        }
    }

    async fn file_exists(&self, path: &str) -> bool {
        self.base.file_exists(path).await
    }
}
