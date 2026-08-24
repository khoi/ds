use crate::{
    CacheRetention, Context, Error, ResponseMetadata, ResponseStream,
    deferred_tools::{DeferredToolsMode, ToolPlacement},
    http, openai, retry, transport,
};
use async_trait::async_trait;
use base64::prelude::*;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex as StdMutex, OnceLock},
    time::Duration,
};
use tokio::{
    net::TcpStream,
    sync::{Mutex as AsyncMutex, OwnedMutexGuard},
    time::Instant,
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{Message as WebSocketMessage, client::IntoClientRequest, http::HeaderValue},
};
use tokio_util::sync::CancellationToken;

pub use crate::openai::{ReasoningEffort, ReasoningSummary, ServiceTier, ToolChoice};

pub mod auth;

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
const DEFAULT_WEBSOCKET_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const WEBSOCKET_IDLE_TTL: Duration = Duration::from_secs(5 * 60);
const WEBSOCKET_MAX_AGE: Duration = Duration::from_secs(55 * 60);

pub struct Provider {
    id: crate::ProviderId,
    models: Vec<crate::Model>,
    headers: BTreeMap<String, Option<String>>,
    auth: crate::ProviderAuth,
}

impl Provider {
    pub fn new(models: impl IntoIterator<Item = crate::Model>) -> Self {
        Self {
            id: crate::ProviderId::new("openai-codex"),
            models: models.into_iter().collect(),
            headers: BTreeMap::new(),
            auth: crate::ProviderAuth::oauth(CodexOAuthAuth {
                client: auth::Client::new(),
            }),
        }
    }

    fn request(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::StreamOptions,
        thinking: Option<crate::ThinkingLevel>,
        tool_choice: Option<crate::ToolChoice>,
    ) -> crate::AssistantMessageEventStream {
        if model.api != crate::Api::OpenAiCodexResponses {
            let model = model.clone();
            let api = model.api.clone();
            return crate::legacy::adapt(model, async move {
                Err(Error::InvalidRequest(format!(
                    "Codex provider has no API implementation for {api}"
                )))
            });
        }
        let requested_model = model.clone();
        let context = context.clone();
        let options = options.clone();
        crate::legacy::adapt(requested_model.clone(), async move {
            let access_token = options
                .api_key
                .ok_or_else(|| Error::InvalidRequest("Codex access token is required".into()))?;
            let provider_model =
                Model::new(&requested_model.id).with_base_url(requested_model.base_url.clone());
            let mut provider_options = Options::new(access_token)
                .with_cancellation(options.cancellation)
                .with_max_retries(options.max_retries.unwrap_or_default())
                .with_max_retry_delay(options.max_retry_delay)
                .with_cache_retention(options.cache_retention);
            if let Some(crate::ModelCompatibility::OpenAi(compat)) = &requested_model.compat {
                let mode = if compat.supports_additional_tools == Some(true) {
                    Some(DeferredToolsMode::AdditionalTools)
                } else if compat.supports_tool_search == Some(true) {
                    Some(DeferredToolsMode::ToolSearch)
                } else {
                    None
                };
                provider_options = provider_options.with_deferred_tools_mode(mode);
            }
            if let Some(temperature) = options.temperature {
                provider_options = provider_options.with_temperature(temperature);
            }
            if let Some(timeout) = options.timeout {
                provider_options = provider_options.with_overall_timeout(timeout);
            }
            if let Some(session_id) = options.session_id {
                provider_options = provider_options.with_session_id(session_id);
            }
            if let Some(timeout) = options.websocket_connect_timeout {
                provider_options = provider_options.with_websocket_connect_timeout(timeout);
            }
            if let Some(transport) = options.transport {
                provider_options = provider_options.with_transport(match transport {
                    crate::Transport::Sse => Transport::Sse,
                    crate::Transport::WebSocket | crate::Transport::WebSocketCached => {
                        Transport::WebSocket
                    }
                    crate::Transport::Auto => Transport::Auto,
                });
            }
            if let Some(thinking) = thinking.and_then(reasoning_effort) {
                provider_options =
                    provider_options.with_reasoning(thinking, ReasoningSummary::Auto);
            }
            if let Some(tool_choice) = tool_choice {
                provider_options = provider_options.with_tool_choice(match tool_choice {
                    crate::ToolChoice::Auto => ToolChoice::Auto,
                    crate::ToolChoice::None => ToolChoice::None,
                });
            }
            stream(&provider_model, &context, &provider_options).await
        })
    }
}

impl crate::Provider for Provider {
    fn id(&self) -> &crate::ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "OpenAI Codex"
    }

    fn base_url(&self) -> Option<&str> {
        Some(DEFAULT_BASE_URL)
    }

    fn headers(&self) -> &BTreeMap<String, Option<String>> {
        &self.headers
    }

    fn auth(&self) -> &crate::ProviderAuth {
        &self.auth
    }

    fn models(&self) -> Vec<crate::Model> {
        self.models.clone()
    }

    fn stream(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::StreamOptions,
    ) -> crate::AssistantMessageEventStream {
        self.request(model, context, options, None, None)
    }

    fn stream_simple(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::SimpleStreamOptions,
    ) -> crate::AssistantMessageEventStream {
        let stream =
            crate::provider::build_simple_stream_options(model, context, options.stream.clone());
        self.request(
            model,
            context,
            &stream,
            options
                .thinking
                .map(|level| model.clamp_thinking_level(level)),
            Some(options.tool_choice),
        )
    }
}

pub fn provider() -> Arc<dyn crate::Provider> {
    Arc::new(Provider::new(crate::codex_models().iter().cloned()))
}

fn reasoning_effort(level: crate::ThinkingLevel) -> Option<ReasoningEffort> {
    match level {
        crate::ThinkingLevel::Off => None,
        crate::ThinkingLevel::Minimal => Some(ReasoningEffort::Minimal),
        crate::ThinkingLevel::Low => Some(ReasoningEffort::Low),
        crate::ThinkingLevel::Medium => Some(ReasoningEffort::Medium),
        crate::ThinkingLevel::High => Some(ReasoningEffort::High),
        crate::ThinkingLevel::XHigh => Some(ReasoningEffort::XHigh),
        crate::ThinkingLevel::Max => Some(ReasoningEffort::Max),
    }
}

struct CodexOAuthAuth {
    client: auth::Client,
}

#[async_trait]
impl crate::OAuthAuth for CodexOAuthAuth {
    fn name(&self) -> &str {
        "OpenAI Codex"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    async fn login(
        &self,
        interaction: &dyn crate::AuthInteraction,
    ) -> Result<crate::Credential, crate::AuthError> {
        let method = interaction
            .prompt(crate::AuthPrompt::Select {
                message: "Choose a sign-in method".into(),
                options: vec![
                    crate::AuthSelectOption {
                        id: "browser".into(),
                        label: "Browser".into(),
                        description: None,
                    },
                    crate::AuthSelectOption {
                        id: "device".into(),
                        label: "Device code".into(),
                        description: None,
                    },
                ],
            })
            .await?;
        let adapter = CodexInteraction(interaction);
        let credentials = match method.as_str() {
            "browser" => {
                self.client
                    .login_browser(&adapter, interaction.cancellation())
                    .await
            }
            "device" => {
                self.client
                    .login_device(&adapter, interaction.cancellation())
                    .await
            }
            method => {
                return Err(crate::AuthError::Authentication(format!(
                    "unknown sign-in method {method}"
                )));
            }
        }
        .map_err(|error| crate::AuthError::OAuth(error.to_string()))?;
        Ok(oauth_credential(credentials))
    }

    async fn refresh(
        &self,
        credential: &crate::Credential,
        cancellation: &CancellationToken,
    ) -> Result<crate::Credential, crate::AuthError> {
        let crate::Credential::OAuth { refresh, .. } = credential else {
            return Err(crate::AuthError::OAuth("expected OAuth credential".into()));
        };
        self.client
            .refresh(refresh, cancellation)
            .await
            .map(oauth_credential)
            .map_err(|error| crate::AuthError::OAuth(error.to_string()))
    }

    async fn to_auth(
        &self,
        credential: &crate::Credential,
    ) -> Result<crate::ModelAuth, crate::AuthError> {
        let crate::Credential::OAuth { access, .. } = credential else {
            return Err(crate::AuthError::OAuth("expected OAuth credential".into()));
        };
        Ok(crate::ModelAuth {
            api_key: Some(access.clone()),
            ..Default::default()
        })
    }
}

struct CodexInteraction<'a>(&'a dyn crate::AuthInteraction);

#[async_trait]
impl auth::Interaction for CodexInteraction<'_> {
    fn notify(&self, notification: auth::Notification) {
        let event = match notification {
            auth::Notification::AuthorizationUrl { url } => crate::AuthEvent::AuthUrl {
                url,
                instructions: None,
            },
            auth::Notification::DeviceCode {
                user_code,
                verification_uri,
                interval,
                expires_in,
            } => crate::AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                interval_seconds: Some(interval.as_secs()),
                expires_in_seconds: Some(expires_in.as_secs()),
            },
        };
        self.0.notify(event);
    }

    async fn manual_authorization(
        &self,
        cancellation: CancellationToken,
    ) -> Result<String, auth::Error> {
        self.0
            .prompt(crate::AuthPrompt::ManualCode {
                message: "Paste the authorization redirect".into(),
                placeholder: None,
                cancellation,
            })
            .await
            .map_err(|error| auth::Error::InvalidResponse(error.to_string()))
    }
}

fn oauth_credential(credentials: auth::Credentials) -> crate::Credential {
    crate::Credential::OAuth {
        refresh: credentials.refresh_token,
        access: credentials.access_token,
        expires: credentials.expires_at,
        extra: BTreeMap::new(),
    }
}

type WebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone)]
struct CachedWebSocket {
    socket: Arc<AsyncMutex<WebSocket>>,
    created_at: Instant,
    metadata: ResponseMetadata,
    continuation: Arc<StdMutex<Option<Continuation>>>,
    last_used: Arc<StdMutex<Instant>>,
}

struct Continuation {
    request: serde_json::Value,
    response_id: String,
    response_items: Vec<serde_json::Value>,
}

struct WebSocketLease {
    key: Option<String>,
}

impl WebSocketLease {
    fn complete(&mut self, idle_ttl: Duration) {
        if let Some(key) = self.key.take() {
            schedule_websocket_expiry(key, idle_ttl);
        }
    }

    fn evict(&mut self) {
        let Some(key) = self.key.take() else {
            return;
        };
        let connection = websockets()
            .lock()
            .expect("websocket cache lock")
            .remove(&key);
        if let Some(connection) = connection {
            close_websocket(connection);
        }
    }
}

impl Drop for WebSocketLease {
    fn drop(&mut self) {
        self.evict();
    }
}

static WEBSOCKETS: OnceLock<StdMutex<HashMap<String, CachedWebSocket>>> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Model {
    id: String,
    base_url: String,
}

impl Model {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: DEFAULT_BASE_URL.into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

pub struct Options {
    access_token: String,
    max_retries: usize,
    max_retry_delay: Option<Duration>,
    cancellation: CancellationToken,
    connection_timeout: Option<Duration>,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_timeout: Option<Duration>,
    session_id: Option<String>,
    cache_retention: CacheRetention,
    transport: Transport,
    websocket_connect_timeout: Duration,
    websocket_cache_ttl: Duration,
    temperature: Option<f64>,
    reasoning: Option<Reasoning>,
    service_tier: Option<ServiceTier>,
    text_verbosity: TextVerbosity,
    tool_choice: ToolChoice,
    deferred_tools_mode: Option<DeferredToolsMode>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Transport {
    #[default]
    Auto,
    Sse,
    WebSocket,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    #[default]
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug)]
struct Reasoning {
    effort: ReasoningEffort,
    summary: ReasoningSummary,
}

impl Options {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            max_retries: 0,
            max_retry_delay: Some(DEFAULT_MAX_RETRY_DELAY),
            cancellation: CancellationToken::new(),
            connection_timeout: None,
            first_event_timeout: None,
            idle_timeout: None,
            overall_timeout: None,
            session_id: None,
            cache_retention: CacheRetention::Short,
            transport: Transport::Auto,
            websocket_connect_timeout: DEFAULT_WEBSOCKET_CONNECT_TIMEOUT,
            websocket_cache_ttl: WEBSOCKET_IDLE_TTL,
            temperature: None,
            reasoning: None,
            service_tier: None,
            text_verbosity: TextVerbosity::Low,
            tool_choice: ToolChoice::Auto,
            deferred_tools_mode: None,
        }
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_max_retry_delay(mut self, max_retry_delay: Option<Duration>) -> Self {
        self.max_retry_delay = max_retry_delay;
        self
    }

    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    pub fn with_connection_timeout(mut self, timeout: Duration) -> Self {
        self.connection_timeout = Some(timeout);
        self
    }

    pub fn with_first_event_timeout(mut self, timeout: Duration) -> Self {
        self.first_event_timeout = Some(timeout);
        self
    }

    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = Some(timeout);
        self
    }

    pub fn with_overall_timeout(mut self, timeout: Duration) -> Self {
        self.overall_timeout = Some(timeout);
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_cache_retention(mut self, retention: CacheRetention) -> Self {
        self.cache_retention = retention;
        self
    }

    pub fn with_transport(mut self, transport: Transport) -> Self {
        self.transport = transport;
        self
    }

    pub fn with_websocket_connect_timeout(mut self, timeout: Duration) -> Self {
        self.websocket_connect_timeout = timeout;
        self
    }

    pub fn with_websocket_cache_ttl(mut self, ttl: Duration) -> Self {
        self.websocket_cache_ttl = ttl;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_reasoning(mut self, effort: ReasoningEffort, summary: ReasoningSummary) -> Self {
        self.reasoning = Some(Reasoning { effort, summary });
        self
    }

    pub fn with_service_tier(mut self, service_tier: ServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    pub fn with_text_verbosity(mut self, verbosity: TextVerbosity) -> Self {
        self.text_verbosity = verbosity;
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    fn with_deferred_tools_mode(mut self, mode: Option<DeferredToolsMode>) -> Self {
        self.deferred_tools_mode = mode;
        self
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    store: bool,
    stream: bool,
    instructions: &'a str,
    input: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
    text: TextOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<RequestReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ServiceTier>,
    include: [&'static str; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
}

#[derive(Serialize)]
struct TextOptions {
    verbosity: TextVerbosity,
}

#[derive(Serialize)]
struct RequestReasoning {
    effort: ReasoningEffort,
    summary: ReasoningSummary,
}

#[derive(Clone)]
struct SseRequest {
    base_url: String,
    model: String,
    access_token: String,
    account_id: String,
    session_id: Option<String>,
    json: Vec<u8>,
    max_retries: usize,
    max_retry_delay: Option<Duration>,
    cancellation: CancellationToken,
    connection_timeout: Option<Duration>,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
}

pub async fn stream(
    model: &Model,
    context: &Context,
    options: &Options,
) -> Result<ResponseStream, Error> {
    let overall_deadline = options
        .overall_timeout
        .map(|timeout| Instant::now() + timeout);
    let account_id = account_id(&options.access_token).map_err(Error::InvalidRequest)?;
    let session_id = match options.cache_retention {
        CacheRetention::None => None,
        CacheRetention::Short | CacheRetention::Long => {
            options.session_id.as_deref().map(openai::clamp_cache_key)
        }
    };
    let placement = crate::deferred_tools::split(
        context,
        options.deferred_tools_mode.is_some(),
        str::to_owned,
    );
    let request = Request {
        model: &model.id,
        store: false,
        stream: true,
        instructions: context.system().unwrap_or("You are a helpful assistant."),
        input: openai::response_input(
            &model.id,
            context,
            false,
            options.deferred_tools_mode.map(|mode| (&placement, mode)),
        )
        .map_err(Error::InvalidRequest)?,
        tools: tools(&placement).map_err(Error::InvalidRequest)?,
        text: TextOptions {
            verbosity: options.text_verbosity,
        },
        temperature: options.temperature,
        reasoning: options.reasoning.map(|reasoning| RequestReasoning {
            effort: reasoning.effort,
            summary: reasoning.summary,
        }),
        service_tier: options.service_tier,
        include: ["reasoning.encrypted_content"],
        prompt_cache_key: session_id.clone(),
        tool_choice: options.tool_choice,
        parallel_tool_calls: true,
    };
    let value =
        serde_json::to_value(&request).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let json =
        serde_json::to_vec(&value).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let sse_request = SseRequest {
        base_url: model.base_url.clone(),
        model: model.id.clone(),
        access_token: options.access_token.clone(),
        account_id: account_id.clone(),
        session_id: session_id.clone(),
        json,
        max_retries: options.max_retries,
        max_retry_delay: options.max_retry_delay,
        cancellation: options.cancellation.clone(),
        connection_timeout: options.connection_timeout,
        first_event_timeout: options.first_event_timeout,
        idle_timeout: options.idle_timeout,
        overall_deadline,
    };
    if options.transport != Transport::Sse {
        let mut websocket_value = value.clone();
        websocket_value
            .as_object_mut()
            .expect("request serializes as an object")
            .insert("type".into(), "response.create".into());
        match websocket_stream(
            WebSocketRequest {
                base_url: &model.base_url,
                model: &model.id,
                access_token: &options.access_token,
                account_id: &account_id,
                session_id: session_id.as_deref(),
                body: websocket_value,
            },
            options,
            overall_deadline,
        )
        .await
        {
            Ok(mut websocket) => {
                let output = async_stream::stream! {
                    match websocket.next().await {
                        Some(event) if !should_fallback_to_sse(&event) => {
                            yield event;
                            while let Some(event) = websocket.next().await {
                                yield event;
                            }
                        }
                        Some(_) | None => match sse_stream(&sse_request).await {
                            Ok(mut fallback) => {
                                while let Some(event) = fallback.next().await {
                                    yield event;
                                }
                            }
                            Err(error) => yield Err(error),
                        },
                    }
                };
                return Ok(Box::pin(output));
            }
            Err(WebSocketConnectError::Cancelled) => {
                return Err(Error::Cancelled { partial: None });
            }
            Err(WebSocketConnectError::OverallTimeout) => {
                return Err(Error::Timeout {
                    phase: crate::TimeoutPhase::Overall,
                    partial: None,
                });
            }
            Err(WebSocketConnectError::Transport) => {}
        }
    }
    sse_stream(&sse_request).await
}

async fn sse_stream(request: &SseRequest) -> Result<ResponseStream, Error> {
    let body = zstd::stream::encode_all(request.json.as_slice(), 3)
        .map_err(|error| Error::Compression(error.to_string()))?;
    let client = reqwest::Client::new();
    let url = response_url(&request.base_url);
    let response = transport::connect(
        retry::send(
            retry::Policy {
                max_retries: request.max_retries,
                max_delay: request.max_retry_delay,
                cancellation: &request.cancellation,
            },
            || {
                let mut builder = client
                    .post(&url)
                    .bearer_auth(&request.access_token)
                    .header("chatgpt-account-id", &request.account_id)
                    .header("openai-beta", "responses=experimental")
                    .header("originator", "ds")
                    .header("user-agent", concat!("ds-ai/", env!("CARGO_PKG_VERSION")))
                    .header("accept", "text/event-stream")
                    .header("content-type", "application/json")
                    .header("content-encoding", "zstd");
                if let Some(session_id) = &request.session_id {
                    builder = builder
                        .header("session-id", session_id)
                        .header("x-client-request-id", session_id);
                }
                builder.body(body.clone()).send()
            },
        ),
        request.connection_timeout,
        request.overall_deadline,
    )
    .await?;
    if !response.status().is_success() {
        let metadata = http::metadata(response.headers());
        return Err(http::provider_error(
            response,
            metadata,
            &request.cancellation,
            request.overall_deadline,
        )
        .await);
    }
    Ok(openai::decode_stream(
        response,
        request.model.clone(),
        request.cancellation.clone(),
        request.first_event_timeout,
        request.idle_timeout,
        request.overall_deadline,
    ))
}

fn should_fallback_to_sse(event: &Result<crate::Event, Error>) -> bool {
    match event {
        Err(Error::Timeout {
            phase: crate::TimeoutPhase::FirstEvent,
            ..
        }) => true,
        Err(Error::Stream { partial, .. } | Error::IncompleteStream { partial }) => {
            partial.id.is_none() && partial.content.is_empty()
        }
        _ => false,
    }
}

enum WebSocketConnectError {
    Cancelled,
    OverallTimeout,
    Transport,
}

struct WebSocketRequest<'a> {
    base_url: &'a str,
    model: &'a str,
    access_token: &'a str,
    account_id: &'a str,
    session_id: Option<&'a str>,
    body: serde_json::Value,
}

#[derive(Clone)]
struct WebSocketHandshake {
    url: String,
    access_token: String,
    account_id: String,
    request_id: String,
}

async fn websocket_stream(
    request: WebSocketRequest<'_>,
    options: &Options,
    overall_deadline: Option<Instant>,
) -> Result<ResponseStream, WebSocketConnectError> {
    let cache_key = request.session_id.map(|session_id| {
        format!(
            "{}\u{1f}{}\u{1f}{session_id}",
            request.base_url, request.account_id
        )
    });
    let request_id = request
        .session_id
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{:032x}", rand::random::<u128>()));
    let handshake = WebSocketHandshake {
        url: websocket_url(request.base_url),
        access_token: request.access_token.to_owned(),
        account_id: request.account_id.to_owned(),
        request_id,
    };
    let cached = cache_key
        .as_deref()
        .and_then(|key| cached_websocket(key, options.websocket_cache_ttl));
    let reused = cached.is_some();
    let connection = if let Some(cached) = cached {
        cached
    } else {
        let connection = connect_websocket(
            &handshake,
            &options.cancellation,
            options.websocket_connect_timeout,
            overall_deadline,
        )
        .await?;
        if let Some(cache_key) = &cache_key {
            websockets()
                .lock()
                .expect("websocket cache lock")
                .insert(cache_key.clone(), connection.clone());
        }
        connection
    };
    let mut lease = WebSocketLease {
        key: cache_key.clone(),
    };
    let CachedWebSocket {
        socket,
        metadata,
        mut continuation,
        mut last_used,
        ..
    } = connection;
    let mut socket = tokio::select! {
        biased;
        _ = options.cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        socket = socket.lock_owned() => socket,
    };
    *last_used.lock().expect("websocket last-used lock") = Instant::now();
    let full_request = request.body;
    let full_body =
        serde_json::to_string(&full_request).map_err(|_| WebSocketConnectError::Transport)?;
    let body = continuation_request(
        &full_request,
        continuation
            .lock()
            .expect("websocket continuation lock")
            .as_ref(),
    );
    let mut used_continuation = body.get("previous_response_id").is_some();
    let body = serde_json::to_string(&body).map_err(|_| WebSocketConnectError::Transport)?;
    let sent = tokio::select! {
        biased;
        _ = options.cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        result = socket.send(WebSocketMessage::Text(body.into())) => result,
    };
    let mut retried_socket = false;
    if sent.is_err() {
        if !reused {
            return Err(WebSocketConnectError::Transport);
        }
        (socket, continuation, last_used) = replace_websocket(
            socket,
            cache_key.as_deref(),
            &handshake,
            &options.cancellation,
            options.websocket_connect_timeout,
            overall_deadline,
            &full_body,
        )
        .await?;
        used_continuation = false;
        retried_socket = true;
    }
    let cancellation = options.cancellation.clone();
    let websocket_connect_timeout = options.websocket_connect_timeout;
    let first_event_timeout = options.first_event_timeout;
    let idle_timeout = options.idle_timeout;
    let events = async_stream::stream! {
        let mut continuation = continuation;
        let mut retried_socket = retried_socket;
        let mut saw_event = false;
        let mut event_deadline = first_event_timeout.map(|timeout| Instant::now() + timeout);
        let mut response_id = None;
        let mut response_items = Vec::new();
        let mut emitted = false;
        let mut retried_missing_continuation = false;
        loop {
            let message = tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    yield Err(transport::ReadError::Cancelled);
                    return;
                }
                _ = transport::wait_until(overall_deadline) => {
                    yield Err(transport::ReadError::Timeout(crate::TimeoutPhase::Overall));
                    return;
                }
                _ = transport::wait_until(event_deadline) => {
                    yield Err(transport::ReadError::Timeout(if saw_event {
                        crate::TimeoutPhase::Idle
                    } else {
                        crate::TimeoutPhase::FirstEvent
                    }));
                    return;
                }
                message = socket.next() => message,
            };
            if reused
                && !emitted
                && !retried_socket
                && matches!(
                    &message,
                    Some(Ok(WebSocketMessage::Close(_))) | Some(Err(_)) | None
                )
            {
                match replace_websocket(
                    socket,
                    cache_key.as_deref(),
                    &handshake,
                    &cancellation,
                    websocket_connect_timeout,
                    overall_deadline,
                    &full_body,
                )
                .await
                {
                    Ok((fresh_socket, fresh_continuation, fresh_last_used)) => {
                        socket = fresh_socket;
                        continuation = fresh_continuation;
                        last_used = fresh_last_used;
                    }
                    Err(WebSocketConnectError::Cancelled) => {
                        yield Err(transport::ReadError::Cancelled);
                        return;
                    }
                    Err(WebSocketConnectError::OverallTimeout) => {
                        yield Err(transport::ReadError::Timeout(crate::TimeoutPhase::Overall));
                        return;
                    }
                    Err(WebSocketConnectError::Transport) => {
                        yield Err(transport::ReadError::Stream("websocket reconnect failed".into()));
                        return;
                    }
                }
                retried_socket = true;
                used_continuation = false;
                saw_event = false;
                event_deadline = first_event_timeout.map(|timeout| Instant::now() + timeout);
                response_id = None;
                response_items.clear();
                continue;
            }
            let data = match message {
                Some(Ok(WebSocketMessage::Text(text))) => Some(text.to_string()),
                Some(Ok(WebSocketMessage::Binary(bytes))) => {
                    match String::from_utf8(bytes.to_vec()) {
                        Ok(text) => Some(text),
                        Err(error) => {
                            yield Err(transport::ReadError::Stream(error.to_string()));
                            return;
                        }
                    }
                }
                Some(Ok(WebSocketMessage::Close(frame))) => {
                    let message = frame.map_or_else(
                        || "websocket closed before a terminal event".into(),
                        |frame| {
                            format!(
                                "websocket closed with code {}: {}",
                                u16::from(frame.code),
                                frame.reason
                            )
                        },
                    );
                    yield Err(transport::ReadError::Stream(message));
                    return;
                }
                None => {
                    yield Err(transport::ReadError::Stream(
                        "websocket closed before a terminal event".into(),
                    ));
                    return;
                }
                Some(Ok(_)) => None,
                Some(Err(error)) => {
                    yield Err(transport::ReadError::Stream(error.to_string()));
                    return;
                }
            };
            if let Some(mut data) = data {
                let code = codex_error_code(&data);
                if !emitted
                    && !retried_socket
                    && code.as_deref() == Some("websocket_connection_limit_reached")
                {
                    *continuation.lock().expect("websocket continuation lock") = None;
                    match replace_websocket(
                        socket,
                        cache_key.as_deref(),
                        &handshake,
                        &cancellation,
                        websocket_connect_timeout,
                        overall_deadline,
                        &full_body,
                    )
                    .await
                    {
                        Ok((fresh_socket, fresh_continuation, fresh_last_used)) => {
                            socket = fresh_socket;
                            continuation = fresh_continuation;
                            last_used = fresh_last_used;
                        }
                        Err(WebSocketConnectError::Cancelled) => {
                            yield Err(transport::ReadError::Cancelled);
                            return;
                        }
                        Err(WebSocketConnectError::OverallTimeout) => {
                            yield Err(transport::ReadError::Timeout(crate::TimeoutPhase::Overall));
                            return;
                        }
                        Err(WebSocketConnectError::Transport) => {
                            yield Err(transport::ReadError::Stream("websocket reconnect failed".into()));
                            return;
                        }
                    }
                    retried_socket = true;
                    used_continuation = false;
                    saw_event = false;
                    event_deadline = first_event_timeout.map(|timeout| Instant::now() + timeout);
                    response_id = None;
                    response_items.clear();
                    continue;
                }
                if used_continuation
                    && !emitted
                    && !retried_missing_continuation
                    && code.as_deref() == Some("previous_response_not_found")
                {
                    *continuation.lock().expect("websocket continuation lock") = None;
                    let sent = tokio::select! {
                        biased;
                        _ = cancellation.cancelled() => {
                            yield Err(transport::ReadError::Cancelled);
                            return;
                        }
                        _ = transport::wait_until(overall_deadline) => {
                            yield Err(transport::ReadError::Timeout(crate::TimeoutPhase::Overall));
                            return;
                        }
                        sent = socket.send(WebSocketMessage::Text(full_body.clone().into())) => sent,
                    };
                    if let Err(error) = sent {
                        yield Err(transport::ReadError::Stream(error.to_string()));
                        return;
                    }
                    retried_missing_continuation = true;
                    saw_event = false;
                    event_deadline = first_event_timeout.map(|timeout| Instant::now() + timeout);
                    response_id = None;
                    response_items.clear();
                    continue;
                }
                data = normalize_codex_error(data);
                saw_event = true;
                emitted = true;
                event_deadline = idle_timeout.map(|timeout| Instant::now() + timeout);
                let terminal = observe_websocket_event(
                    &data,
                    &mut response_id,
                    &mut response_items,
                );
                if terminal
                    && let Some(response_id) = response_id.take()
                {
                    *continuation.lock().expect("websocket continuation lock") = Some(Continuation {
                        request: full_request.clone(),
                        response_id,
                        response_items: std::mem::take(&mut response_items),
                    });
                    *last_used.lock().expect("websocket last-used lock") = Instant::now();
                }
                if terminal && cache_key.is_none() {
                    let _ = socket.close(None).await;
                }
                yield Ok(data);
                if terminal {
                    return;
                }
            }
        }
    };
    let mut decoded = openai::decode_events(Box::pin(events), request.model.to_owned(), metadata);
    let websocket_cache_ttl = options.websocket_cache_ttl;
    let output = async_stream::stream! {
        while let Some(event) = decoded.next().await {
            match event {
                Ok(crate::Event::Done(response)) => {
                    drop(decoded);
                    lease.complete(websocket_cache_ttl);
                    yield Ok(crate::Event::Done(response));
                    return;
                }
                Err(error) => {
                    drop(decoded);
                    lease.evict();
                    yield Err(error);
                    return;
                }
                Ok(event) => yield Ok(event),
            }
        }
    };
    Ok(Box::pin(output))
}

async fn connect_websocket(
    handshake: &WebSocketHandshake,
    cancellation: &CancellationToken,
    connect_timeout: Duration,
    overall_deadline: Option<Instant>,
) -> Result<CachedWebSocket, WebSocketConnectError> {
    let mut connection_request = handshake
        .url
        .as_str()
        .into_client_request()
        .map_err(|_| WebSocketConnectError::Transport)?;
    for (name, value) in [
        (
            "authorization",
            format!("Bearer {}", handshake.access_token),
        ),
        ("chatgpt-account-id", handshake.account_id.clone()),
        ("originator", "ds".into()),
        (
            "user-agent",
            concat!("ds-ai/", env!("CARGO_PKG_VERSION")).into(),
        ),
        ("x-client-request-id", handshake.request_id.clone()),
        ("openai-beta", "responses_websockets=2026-02-06".into()),
    ] {
        connection_request.headers_mut().insert(
            name,
            HeaderValue::from_str(&value).map_err(|_| WebSocketConnectError::Transport)?,
        );
    }
    let session_header = connection_request.headers()["x-client-request-id"].clone();
    connection_request
        .headers_mut()
        .insert("session-id", session_header);
    let connect_deadline = Instant::now() + connect_timeout;
    let (socket, response) = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        _ = tokio::time::sleep_until(connect_deadline) => {
            return Err(WebSocketConnectError::Transport);
        }
        connection = connect_async(connection_request) => {
            connection.map_err(|_| WebSocketConnectError::Transport)?
        }
    };
    Ok(CachedWebSocket {
        socket: Arc::new(AsyncMutex::new(socket)),
        created_at: Instant::now(),
        metadata: http::metadata(response.headers()),
        continuation: Arc::new(StdMutex::new(None)),
        last_used: Arc::new(StdMutex::new(Instant::now())),
    })
}

async fn replace_websocket(
    mut socket: OwnedMutexGuard<WebSocket>,
    cache_key: Option<&str>,
    handshake: &WebSocketHandshake,
    cancellation: &CancellationToken,
    connect_timeout: Duration,
    overall_deadline: Option<Instant>,
    body: &str,
) -> Result<
    (
        OwnedMutexGuard<WebSocket>,
        Arc<StdMutex<Option<Continuation>>>,
        Arc<StdMutex<Instant>>,
    ),
    WebSocketConnectError,
> {
    if let Some(cache_key) = cache_key {
        websockets()
            .lock()
            .expect("websocket cache lock")
            .remove(cache_key);
    }
    let _ = socket.close(None).await;
    drop(socket);
    let connection =
        connect_websocket(handshake, cancellation, connect_timeout, overall_deadline).await?;
    let continuation = connection.continuation.clone();
    let last_used = connection.last_used.clone();
    let fresh_socket = connection.socket.clone();
    let mut socket = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        socket = fresh_socket.lock_owned() => socket,
    };
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        sent = socket.send(WebSocketMessage::Text(body.to_owned().into())) => {
            sent.map_err(|_| WebSocketConnectError::Transport)?;
        }
    }
    if let Some(cache_key) = cache_key {
        websockets()
            .lock()
            .expect("websocket cache lock")
            .insert(cache_key.to_owned(), connection);
    }
    Ok((socket, continuation, last_used))
}

fn codex_error_code(data: &str) -> Option<String> {
    let event = serde_json::from_str::<serde_json::Value>(data).ok()?;
    event
        .get("code")
        .or_else(|| event.get("error").and_then(|error| error.get("code")))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

fn normalize_codex_error(data: String) -> String {
    let Ok(mut event) = serde_json::from_str::<serde_json::Value>(&data) else {
        return data;
    };
    if event.get("type").and_then(serde_json::Value::as_str) != Some("error") {
        return data;
    }
    let code = codex_error_code(&data);
    let message = event
        .get("message")
        .or_else(|| event.get("error").and_then(|error| error.get("message")))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let Some(event) = event.as_object_mut() else {
        return data;
    };
    if let Some(code) = code {
        event.insert("code".into(), code.into());
    }
    if let Some(message) = message {
        event.insert("message".into(), message.into());
    }
    serde_json::to_string(event).unwrap_or(data)
}

fn websockets() -> &'static StdMutex<HashMap<String, CachedWebSocket>> {
    WEBSOCKETS.get_or_init(|| StdMutex::new(HashMap::new()))
}

fn close_websocket(connection: CachedWebSocket) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    runtime.spawn(async move {
        let mut socket = connection.socket.lock().await;
        let _ = socket.close(None).await;
    });
}

fn schedule_websocket_expiry(key: String, idle_ttl: Duration) {
    let Some(connection) = websockets()
        .lock()
        .expect("websocket cache lock")
        .get(&key)
        .cloned()
    else {
        return;
    };
    let idle_remaining = idle_ttl.saturating_sub(
        connection
            .last_used
            .lock()
            .expect("websocket last-used lock")
            .elapsed(),
    );
    let age_remaining = WEBSOCKET_MAX_AGE.saturating_sub(connection.created_at.elapsed());
    tokio::spawn(async move {
        tokio::time::sleep(idle_remaining.min(age_remaining)).await;
        if !websocket_expired(&connection, idle_ttl) {
            return;
        }
        let connection = {
            let mut cache = websockets().lock().expect("websocket cache lock");
            let same_connection = cache
                .get(&key)
                .is_some_and(|current| Arc::ptr_eq(&current.socket, &connection.socket));
            if !same_connection {
                return;
            }
            cache.remove(&key).expect("cached websocket exists")
        };
        let mut socket = connection.socket.lock().await;
        let _ = socket.close(None).await;
    });
}

fn cached_websocket(key: &str, idle_ttl: Duration) -> Option<CachedWebSocket> {
    let mut cache = websockets().lock().expect("websocket cache lock");
    let expired = cache
        .get(key)
        .is_some_and(|connection| websocket_expired(connection, idle_ttl));
    if expired {
        cache.remove(key);
    }
    cache.get(key).cloned()
}

fn websocket_expired(connection: &CachedWebSocket, idle_ttl: Duration) -> bool {
    if connection.created_at.elapsed() >= WEBSOCKET_MAX_AGE {
        return true;
    }
    if connection.socket.try_lock().is_err() {
        return false;
    }
    connection
        .last_used
        .lock()
        .expect("websocket last-used lock")
        .elapsed()
        >= idle_ttl
}

fn continuation_request(
    request: &serde_json::Value,
    continuation: Option<&Continuation>,
) -> serde_json::Value {
    let Some(continuation) = continuation else {
        return request.clone();
    };
    if request_configuration(request) != request_configuration(&continuation.request) {
        return request.clone();
    }
    let Some(current) = request.get("input").and_then(serde_json::Value::as_array) else {
        return request.clone();
    };
    let mut baseline = continuation
        .request
        .get("input")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    baseline.extend(continuation.response_items.iter().cloned());
    if current.len() < baseline.len()
        || !current[..baseline.len()]
            .iter()
            .zip(&baseline)
            .all(|(current, baseline)| equivalent_json(current, baseline))
    {
        return request.clone();
    }
    let mut request = request.clone();
    let request = request.as_object_mut().expect("request is an object");
    request.insert(
        "previous_response_id".into(),
        continuation.response_id.clone().into(),
    );
    request.insert("input".into(), current[baseline.len()..].to_vec().into());
    serde_json::Value::Object(request.clone())
}

fn equivalent_json(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    match (left, right) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            left.iter()
                .filter(|(_, value)| !value.is_null())
                .all(|(key, value)| {
                    right
                        .get(key)
                        .is_some_and(|right| equivalent_json(value, right))
                })
                && right
                    .iter()
                    .filter(|(_, value)| !value.is_null())
                    .all(|(key, value)| {
                        left.get(key)
                            .is_some_and(|left| equivalent_json(left, value))
                    })
        }
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| equivalent_json(left, right))
        }
        _ => left == right,
    }
}

fn request_configuration(request: &serde_json::Value) -> serde_json::Value {
    let mut request = request.clone();
    if let Some(request) = request.as_object_mut() {
        request.remove("input");
        request.remove("previous_response_id");
    }
    request
}

fn observe_websocket_event(
    data: &str,
    response_id: &mut Option<String>,
    response_items: &mut Vec<serde_json::Value>,
) -> bool {
    let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
        return false;
    };
    let event_type = event.get("type").and_then(serde_json::Value::as_str);
    if event_type == Some("response.output_item.done")
        && let Some(item) = event.get("item")
    {
        response_items.push(item.clone());
    }
    if let Some(id) = event
        .get("response")
        .and_then(|response| response.get("id"))
        .and_then(serde_json::Value::as_str)
    {
        *response_id = Some(id.to_owned());
    }
    matches!(
        event_type,
        Some("response.done" | "response.completed" | "response.incomplete")
    )
}

fn account_id(token: &str) -> Result<String, String> {
    let payload = token
        .split('.')
        .nth(1)
        .ok_or_else(|| "access token is not a JWT".to_string())?;
    let bytes = BASE64_URL_SAFE_NO_PAD
        .decode(payload)
        .or_else(|_| BASE64_URL_SAFE.decode(payload))
        .or_else(|_| BASE64_STANDARD.decode(payload))
        .map_err(|_| "access token has an invalid JWT payload".to_string())?;
    let payload: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| "access token has an invalid JWT payload".to_string())?;
    payload
        .get("https://api.openai.com/auth")
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(serde_json::Value::as_str)
        .filter(|account_id| !account_id.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "access token has no account ID".to_string())
}

fn response_url(base_url: &str) -> String {
    let base_url = base_url.trim().trim_end_matches('/');
    if base_url.ends_with("/codex/responses") {
        base_url.into()
    } else if base_url.ends_with("/codex") {
        format!("{base_url}/responses")
    } else {
        format!("{base_url}/codex/responses")
    }
}

fn websocket_url(base_url: &str) -> String {
    response_url(base_url)
        .replacen("https://", "wss://", 1)
        .replacen("http://", "ws://", 1)
}

fn tools(placement: &ToolPlacement) -> Result<Vec<serde_json::Value>, String> {
    openai::response_tools(&placement.immediate, false)
}
