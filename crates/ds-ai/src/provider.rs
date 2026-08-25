use crate::auth::resolution::{
    api_login_error, check_provider_auth, oauth_login_error, race_cancellation, resolve_auth,
    store_error,
};
use crate::{
    AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantMessageStreamError, CacheRetention, Context, Model, ProviderId, StopReason,
    ThinkingLevel,
};
use async_stream::stream;
use futures_util::{StreamExt, future::try_join_all, stream as futures_stream};
use std::{
    collections::BTreeMap,
    fmt,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;

type PayloadFuture =
    Pin<Box<dyn Future<Output = Result<Option<serde_json::Value>, String>> + Send + 'static>>;
type ResponseFuture = Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'static>>;
type HeaderMap = BTreeMap<String, Option<String>>;
type HeaderFuture = Pin<Box<dyn Future<Output = Result<HeaderMap, String>> + Send + 'static>>;

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

#[derive(Clone)]
pub struct HeaderHook {
    hook: Arc<dyn Fn(HeaderMap) -> HeaderFuture + Send + Sync>,
}

impl HeaderHook {
    pub fn new<F, Fut>(hook: F) -> Self
    where
        F: Fn(BTreeMap<String, Option<String>>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<BTreeMap<String, Option<String>>, String>> + Send + 'static,
    {
        Self {
            hook: Arc::new(move |headers| Box::pin(hook(headers))),
        }
    }

    async fn run(&self, headers: HeaderMap) -> Result<HeaderMap, String> {
        (self.hook)(headers).await
    }
}

impl fmt::Debug for HeaderHook {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HeaderHook")
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
    cancellation: CancellationToken,
}

impl RequestHooks {
    pub(crate) async fn payload(
        &self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, crate::Error> {
        let Some(hook) = &self.payload else {
            return Ok(payload);
        };
        match hook.run(payload.clone(), self.model.clone()).await {
            Ok(payload_override) => Ok(payload_override.unwrap_or(payload)),
            Err(message) => Err(self.error(message)),
        }
    }

    pub(crate) async fn response(&self, response: ProviderResponse) -> Result<(), crate::Error> {
        let Some(hook) = &self.response else {
            return Ok(());
        };
        hook.run(response, self.model.clone())
            .await
            .map_err(|message| self.error(message))
    }

    fn error(&self, message: String) -> crate::Error {
        crate::Error::Hook {
            message,
            aborted: self.cancellation.is_cancelled(),
        }
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
    pub http_client: Option<reqwest::Client>,
    pub env: BTreeMap<String, String>,
    pub headers: BTreeMap<String, Option<String>>,
    pub transform_headers: Option<HeaderHook>,
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
            http_client: None,
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
            transform_headers: None,
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
    pub fn with_transform_headers<F, Fut>(mut self, hook: F) -> Self
    where
        F: Fn(BTreeMap<String, Option<String>>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<BTreeMap<String, Option<String>>, String>> + Send + 'static,
    {
        self.transform_headers = Some(HeaderHook::new(hook));
        self
    }

    pub(crate) fn request_hooks(&self, model: &Model) -> RequestHooks {
        RequestHooks {
            model: model.clone(),
            payload: self.on_payload.clone(),
            response: self.on_response.clone(),
            cancellation: self.cancellation.clone(),
        }
    }

    pub(crate) async fn request_headers(
        &mut self,
        model: &Model,
    ) -> Result<HeaderMap, crate::Error> {
        let model_headers = model
            .headers
            .iter()
            .map(|(name, value)| (name.clone(), Some(value.clone())))
            .collect();
        let option_headers = std::mem::take(&mut self.headers);
        let headers = merge_headers(&[&model_headers, &option_headers]);
        self.run_header_transform(headers)
            .await
            .map_err(crate::Error::HeaderTransform)
    }

    async fn run_header_transform(&mut self, headers: HeaderMap) -> Result<HeaderMap, String> {
        match self.transform_headers.take() {
            Some(hook) => hook.run(headers).await,
            None => Ok(headers),
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
    pub reasoning: Option<ThinkingLevel>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub tool_choice: Option<ToolChoice>,
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

pub trait ApiOptions: Clone + fmt::Debug + Send + Sync + 'static {
    fn api() -> crate::Api;
    fn erase(self) -> ApiStreamOptions;
}

impl ApiOptions for OpenAiResponsesOptions {
    fn api() -> crate::Api {
        crate::Api::OpenAiResponses
    }

    fn erase(self) -> ApiStreamOptions {
        ApiStreamOptions::OpenAiResponses(self)
    }
}

impl ApiOptions for AnthropicOptions {
    fn api() -> crate::Api {
        crate::Api::AnthropicMessages
    }

    fn erase(self) -> ApiStreamOptions {
        ApiStreamOptions::AnthropicMessages(self)
    }
}

impl ApiOptions for OpenAiCodexResponsesOptions {
    fn api() -> crate::Api {
        crate::Api::OpenAiCodexResponses
    }

    fn erase(self) -> ApiStreamOptions {
        ApiStreamOptions::OpenAiCodexResponses(self)
    }
}

pub type OpenAiResponsesModel = crate::ApiModel<OpenAiResponsesOptions>;
pub type AnthropicModel = crate::ApiModel<AnthropicOptions>;
pub type OpenAiCodexResponsesModel = crate::ApiModel<OpenAiCodexResponsesOptions>;

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

    pub fn stream<O: ApiOptions>(
        &self,
        model: &crate::ApiModel<O>,
        context: &Context,
        options: &O,
    ) -> AssistantMessageEventStream {
        self.stream_erased(model.as_model(), context, &options.clone().erase())
    }

    fn stream_erased(
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
            let resolution = resolve_model_auth(
                provider.clone(),
                credentials,
                auth_context,
                &model,
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
            let (request_model, request_options) = apply_auth(&model, options, resolution).await?;
            Ok(provider.stream(&request_model, &context, &request_options))
        })
    }

    pub async fn complete<O: ApiOptions>(
        &self,
        model: &crate::ApiModel<O>,
        context: &Context,
        options: &O,
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
            let resolution = resolve_model_auth(
                provider.clone(),
                credentials,
                auth_context,
                &model,
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
            let request_model = apply_stream_auth(&model, &mut stream_options, resolution);
            apply_header_transform(&request_model, &mut stream_options).await?;
            Ok(provider.stream_simple(
                &request_model,
                &context,
                &SimpleStreamOptions {
                    stream: stream_options,
                    reasoning: options.reasoning,
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

    pub fn api_model<O: ApiOptions>(
        &self,
        provider: &str,
        id: &str,
    ) -> Result<Option<crate::ApiModel<O>>, crate::ApiModelError> {
        self.model(provider, id)
            .map(|model| crate::ApiModel::new(model))
            .transpose()
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

    pub async fn auth_for_model(
        &self,
        model: &Model,
        overrides: crate::AuthResolutionOverrides,
    ) -> Result<Option<crate::AuthResult>, crate::AuthError> {
        let Some(provider) = self.provider(model.provider.as_str()) else {
            return Ok(None);
        };
        resolve_model_auth(
            provider,
            self.credentials.clone(),
            self.auth_context.clone(),
            model,
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
        let credential = race_cancellation(
            cancellation,
            self.credentials.read(provider_id, cancellation),
        )
        .await
        .map_err(|error| store_error("credential store read failed", provider_id, error))?;
        race_cancellation(
            cancellation,
            check_provider_auth(
                &provider,
                credential.as_ref(),
                self.auth_context.as_ref(),
                cancellation,
            ),
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
        let checks = try_join_all(providers.into_iter().map(|provider| async move {
            let auth = self
                .check_auth(provider.id().as_str(), cancellation)
                .await?;
            Ok::<_, crate::AuthError>((provider, auth))
        }))
        .await?;
        Ok(checks
            .into_iter()
            .flat_map(|(provider, auth)| auth.map_or_else(Vec::new, |_| provider.models()))
            .collect())
    }

    pub async fn login(
        &self,
        provider_id: &str,
        credential_type: crate::CredentialType,
        interaction: &dyn crate::AuthInteraction,
    ) -> Result<crate::Credential, crate::AuthError> {
        let provider = self
            .provider(provider_id)
            .ok_or_else(|| crate::AuthError::Provider(format!("Unknown provider {provider_id}")))?;
        let cancellation = interaction.cancellation();
        let credential = match credential_type {
            crate::CredentialType::ApiKey => {
                let auth = provider.auth().api_key.as_ref().ok_or_else(|| {
                    crate::AuthError::Unsupported(format!(
                        "Provider {provider_id} does not support API key login"
                    ))
                })?;
                race_cancellation(cancellation, auth.login(interaction))
                    .await
                    .map_err(|error| api_login_error(provider_id, error))?
            }
            crate::CredentialType::OAuth => {
                let auth = provider.auth().oauth.as_ref().ok_or_else(|| {
                    crate::AuthError::Unsupported(format!(
                        "Provider {provider_id} does not support OAuth login"
                    ))
                })?;
                race_cancellation(cancellation, auth.login(interaction))
                    .await
                    .map_err(|error| oauth_login_error(provider_id, error))?
            }
        };
        match (&credential_type, &credential) {
            (crate::CredentialType::ApiKey, crate::Credential::OAuth { .. }) => {
                return Err(crate::AuthError::Authentication(format!(
                    "API key login returned an OAuth credential for provider {provider_id}"
                )));
            }
            (crate::CredentialType::OAuth, crate::Credential::ApiKey { .. }) => {
                return Err(crate::AuthError::OAuth(format!(
                    "OAuth login returned an API key credential for provider {provider_id}"
                )));
            }
            _ => {}
        }
        if cancellation.is_cancelled() {
            return Err(crate::AuthError::Cancelled);
        }
        let saved = credential.clone();
        let mutation_started = Arc::new(AtomicBool::new(false));
        let mutation_started_for_callback = mutation_started.clone();
        let mutation = self.credentials.modify(
            provider_id,
            Box::new(move |_| {
                mutation_started_for_callback.store(true, Ordering::Release);
                Box::pin(async move { Ok(Some(saved)) })
            }),
            cancellation,
        );
        tokio::pin!(mutation);
        let result = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                if !mutation_started.load(Ordering::Acquire) {
                    return Err(crate::AuthError::Cancelled);
                }
                mutation.await
            }
            result = &mut mutation => result,
        };
        result
            .map_err(|error| store_error("credential store modify failed", provider_id, error))?;
        Ok(credential)
    }

    pub async fn logout(
        &self,
        provider_id: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), crate::AuthError> {
        race_cancellation(
            cancellation,
            self.credentials.delete(provider_id, cancellation),
        )
        .await
        .map_err(|error| store_error("credential store delete failed", provider_id, error))
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
        reason: crate::ErrorReason::Error,
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
                yield error_event(&model, auth_error_message(&error));
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

fn auth_error_message(error: &crate::AuthError) -> String {
    match error {
        crate::AuthError::Cancelled => error.to_string(),
        crate::AuthError::Store(message)
        | crate::AuthError::Provider(message)
        | crate::AuthError::Authentication(message)
        | crate::AuthError::OAuth(message)
        | crate::AuthError::Unsupported(message) => message.clone(),
    }
}

async fn apply_auth(
    model: &Model,
    mut options: ApiStreamOptions,
    resolution: crate::AuthResult,
) -> Result<(Model, ApiStreamOptions), crate::AuthError> {
    let stream = options.stream_mut();
    let request_model = apply_stream_auth(model, stream, resolution);
    apply_header_transform(&request_model, stream).await?;
    Ok((request_model, options))
}

fn apply_stream_auth(
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
    stream.headers = merge_headers(&[&resolution.auth.headers, &stream.headers]);
    model
}

async fn apply_header_transform(
    model: &Model,
    stream: &mut StreamOptions,
) -> Result<(), crate::AuthError> {
    let headers = std::mem::take(&mut stream.headers);
    stream.headers = stream
        .run_header_transform(headers)
        .await
        .map_err(crate::AuthError::Provider)?;
    for name in model.headers.keys() {
        if !contains_header(&stream.headers, name) {
            insert_header(&mut stream.headers, name, None);
        }
    }
    Ok(())
}

fn merge_headers(layers: &[&BTreeMap<String, Option<String>>]) -> HeaderMap {
    let mut headers = BTreeMap::new();
    for layer in layers {
        for (name, value) in *layer {
            insert_header(&mut headers, name, value.clone());
        }
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

fn contains_header(headers: &HeaderMap, name: &str) -> bool {
    headers
        .keys()
        .any(|existing| existing.eq_ignore_ascii_case(name))
}

async fn resolve_model_auth(
    provider: Arc<dyn Provider>,
    credentials: Arc<dyn crate::CredentialStore>,
    auth_context: Arc<dyn crate::AuthContext>,
    model: &Model,
    overrides: crate::AuthResolutionOverrides,
) -> Result<Option<crate::AuthResult>, crate::AuthError> {
    let result = resolve_auth(provider, credentials, auth_context, overrides).await?;
    Ok(result.map(|mut result| {
        let model_headers = model
            .headers
            .iter()
            .map(|(name, value)| (name.clone(), Some(value.clone())))
            .collect();
        result.auth.headers = merge_headers(&[&result.auth.headers, &model_headers]);
        result
    }))
}
