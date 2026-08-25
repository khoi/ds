use crate::{
    CacheRetention, Context, Error, Response,
    deferred_tools::{DeferredToolsMode, ToolPlacement},
    http, openai, retry, transport,
};
use base64::prelude::*;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{
        Arc, Mutex as StdMutex, Once, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    sync::{Mutex as AsyncMutex, OwnedMutexGuard},
    time::Instant,
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, client_async_tls, connect_async,
    tungstenite::{
        Message as WebSocketMessage,
        client::IntoClientRequest,
        http::{HeaderName, HeaderValue},
    },
};
use tokio_util::sync::CancellationToken;

pub use crate::openai::ServiceTier;

pub mod auth;

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
const DEFAULT_WEBSOCKET_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const WEBSOCKET_IDLE_TTL: Duration = Duration::from_secs(5 * 60);
const WEBSOCKET_MAX_AGE: Duration = Duration::from_secs(55 * 60);
const UNSUPPORTED_PROXY_PROTOCOL_MESSAGE: &str = "Unsupported proxy protocol. HTTPS, SOCKS, and PAC proxy URLs are not supported; use an HTTP proxy URL.";

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
            auth: crate::ProviderAuth::oauth(auth::OAuth::new()),
        }
    }
}

pub fn stream(
    model: &crate::OpenAiCodexResponsesModel,
    context: &Context,
    options: &crate::OpenAiCodexResponsesOptions,
) -> crate::AssistantMessageEventStream {
    stream_model(model.as_model(), context, options)
}

fn stream_model(
    model: &crate::Model,
    context: &Context,
    options: &crate::OpenAiCodexResponsesOptions,
) -> crate::AssistantMessageEventStream {
    let requested_model = model.clone();
    let context = context.for_model(&requested_model);
    let options = options.clone();
    let cancellation = options.stream.cancellation.clone();
    crate::provider_stream::adapt(requested_model.clone(), cancellation, async move {
        let mut stream_options = options.stream;
        let request_hooks = stream_options.request_hooks(&requested_model);
        let access_token = stream_options
            .api_key
            .take()
            .filter(|access_token| !access_token.is_empty())
            .ok_or_else(|| Error::MissingApiKey(requested_model.provider.clone()))?;
        let headers = stream_options.request_headers(&requested_model).await?;
        let provider_model =
            Model::new(&requested_model.id).with_base_url(requested_model.base_url.clone());
        let mut provider_options =
            Options::new(access_token, stream_options.http_client.unwrap_or_default())
                .with_cancellation(stream_options.cancellation)
                .with_max_retries(stream_options.max_retries.unwrap_or_default())
                .with_max_retry_delay(stream_options.max_retry_delay)
                .with_cache_retention(stream_options.cache_retention)
                .with_env(std::mem::take(&mut stream_options.env))
                .with_request_options(headers, request_hooks);
        let compat = match &requested_model.compat {
            Some(crate::ModelCompatibility::OpenAi(compat)) => compat.clone(),
            _ => Default::default(),
        };
        let mode = if compat.supports_additional_tools == Some(true) {
            Some(DeferredToolsMode::AdditionalTools)
        } else if compat.supports_tool_search == Some(true) {
            Some(DeferredToolsMode::ToolSearch)
        } else {
            None
        };
        provider_options = provider_options
            .with_deferred_tools_mode(mode)
            .with_compatibility(&compat);
        if let Some(temperature) = stream_options.temperature {
            provider_options = provider_options.with_temperature(temperature);
        }
        if let Some(timeout) = stream_options.timeout {
            provider_options = provider_options.with_timeout(timeout);
        }
        if let Some(session_id) = stream_options.session_id {
            provider_options = provider_options.with_session_id(session_id);
        }
        if let Some(timeout) = stream_options.websocket_connect_timeout {
            provider_options = provider_options.with_websocket_connect_timeout(timeout);
        }
        if let Some(transport) = stream_options.transport {
            provider_options = provider_options.with_transport(match transport {
                crate::Transport::Sse => Transport::Sse,
                crate::Transport::WebSocket => Transport::WebSocket,
                crate::Transport::WebSocketCached => Transport::WebSocketCached,
                crate::Transport::Auto => Transport::Auto,
            });
        }
        if let Some(effort) = options.reasoning_effort
            && let Some(effort) = mapped_reasoning_effort(&requested_model, effort)
        {
            provider_options = provider_options.with_reasoning_value(
                effort,
                options.reasoning_summary.unwrap_or(ReasoningSummary::Auto),
            );
        }
        if let Some(service_tier) = options.service_tier {
            provider_options = provider_options.with_service_tier(service_tier);
        }
        if let Some(text_verbosity) = options.text_verbosity {
            provider_options = provider_options.with_text_verbosity(text_verbosity);
        }
        if let Some(tool_choice) = options.tool_choice {
            provider_options = provider_options.with_tool_choice(tool_choice);
        }
        response_events(&provider_model, &context, &provider_options).await
    })
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
        options: &crate::ApiStreamOptions,
    ) -> crate::AssistantMessageEventStream {
        if model.api != crate::Api::OpenAiCodexResponses {
            let model = model.clone();
            let api = model.api.clone();
            return crate::provider_stream::failure(
                model,
                Error::InvalidRequest(format!(
                    "Codex provider has no API implementation for {api}"
                )),
            );
        }
        let crate::ApiStreamOptions::OpenAiCodexResponses(options) = options else {
            let model = model.clone();
            return crate::provider_stream::failure(
                model,
                Error::InvalidRequest("OpenAI Codex Responses options are required".into()),
            );
        };
        stream_model(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::SimpleStreamOptions,
    ) -> crate::AssistantMessageEventStream {
        let stream_options =
            crate::provider::build_simple_stream_options(model, context, options.stream.clone());
        stream_model(
            model,
            context,
            &crate::OpenAiCodexResponsesOptions {
                stream: stream_options,
                reasoning_effort: options
                    .reasoning
                    .map(|level| model.clamp_thinking_level(level))
                    .and_then(reasoning_effort),
                tool_choice: Some(match options.tool_choice {
                    None | Some(crate::ToolChoice::Auto) => ToolChoice::Auto,
                    Some(crate::ToolChoice::None) => ToolChoice::None,
                }),
                ..Default::default()
            },
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

fn mapped_reasoning_effort(model: &crate::Model, effort: ReasoningEffort) -> Option<String> {
    model
        .thinking_level_map
        .get(&effort.thinking_level())
        .cloned()
        .unwrap_or_else(|| Some(effort.as_str().into()))
}

type WebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone)]
struct CachedWebSocket {
    socket: Arc<AsyncMutex<WebSocket>>,
    shutdown: CancellationToken,
    busy: Arc<AtomicBool>,
    created_at: Instant,
    session_id: Option<String>,
    continuation: Arc<StdMutex<Option<Continuation>>>,
    last_used: Arc<StdMutex<Instant>>,
}

struct Continuation {
    request: serde_json::Value,
    response_id: String,
    response_items: Vec<serde_json::Value>,
}

type ContinuationHandle = Arc<StdMutex<Option<Continuation>>>;

#[derive(Clone)]
struct ContinuationState {
    continuation: ContinuationHandle,
    last_used: Arc<StdMutex<Instant>>,
}

type SharedContinuationState = Arc<StdMutex<ContinuationState>>;

struct WebSocketLease {
    key: Option<String>,
    busy: Arc<StdMutex<Option<Arc<AtomicBool>>>>,
}

impl WebSocketLease {
    fn complete(&mut self, idle_ttl: Duration) {
        if let Some(key) = self.key.take() {
            self.busy
                .lock()
                .expect("cached websocket busy lock")
                .take()
                .expect("cached websocket busy state")
                .store(false, Ordering::Release);
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WebSocketDebugStats {
    pub requests: usize,
    pub connections_created: usize,
    pub connections_reused: usize,
    pub cached_context_requests: usize,
    pub store_true_requests: usize,
    pub full_context_requests: usize,
    pub delta_requests: usize,
    pub last_input_items: usize,
    pub last_delta_input_items: Option<usize>,
    pub last_previous_response_id: Option<String>,
    pub websocket_failures: usize,
    pub sse_fallbacks: usize,
    pub websocket_fallback_active: Option<bool>,
    pub last_websocket_error: Option<String>,
}

#[derive(Default)]
struct WebSocketDebugState {
    stats: HashMap<String, WebSocketDebugStats>,
    fallback_sessions: HashSet<String>,
}

struct CachedWebSocketLookup {
    connection: Option<CachedWebSocket>,
    cacheable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Model {
    id: String,
    base_url: String,
}

impl Model {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: DEFAULT_BASE_URL.into(),
        }
    }

    fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = normalize_base_url(&base_url.into());
        self
    }
}

struct Options {
    access_token: String,
    http_client: reqwest::Client,
    max_retries: usize,
    max_retry_delay: Option<Duration>,
    cancellation: CancellationToken,
    timeout: Option<Duration>,
    session_id: Option<String>,
    cache_retention: CacheRetention,
    transport: Transport,
    websocket_connect_timeout: Option<Duration>,
    temperature: Option<f64>,
    reasoning: Option<Reasoning>,
    service_tier: Option<ServiceTier>,
    text_verbosity: TextVerbosity,
    tool_choice: ToolChoice,
    deferred_tools_mode: Option<DeferredToolsMode>,
    supports_strict_mode: bool,
    supports_grammar_tools: bool,
    env: BTreeMap<String, String>,
    headers: BTreeMap<String, Option<String>>,
    request_hooks: Option<crate::provider::RequestHooks>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Transport {
    #[default]
    Auto,
    Sse,
    WebSocket,
    WebSocketCached,
}

impl Transport {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Sse => "sse",
            Self::WebSocket => "websocket",
            Self::WebSocketCached => "websocket-cached",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    #[default]
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
    Off,
    On,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }

    fn thinking_level(self) -> crate::ThinkingLevel {
        match self {
            Self::None => crate::ThinkingLevel::Off,
            Self::Minimal => crate::ThinkingLevel::Minimal,
            Self::Low => crate::ThinkingLevel::Low,
            Self::Medium => crate::ThinkingLevel::Medium,
            Self::High => crate::ThinkingLevel::High,
            Self::XHigh => crate::ThinkingLevel::XHigh,
            Self::Max => crate::ThinkingLevel::Max,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

#[derive(Clone, Debug)]
struct Reasoning {
    effort: String,
    summary: ReasoningSummary,
}

impl Options {
    fn new(access_token: impl Into<String>, http_client: reqwest::Client) -> Self {
        Self {
            access_token: access_token.into(),
            http_client,
            max_retries: 0,
            max_retry_delay: Some(DEFAULT_MAX_RETRY_DELAY),
            cancellation: CancellationToken::new(),
            timeout: None,
            session_id: None,
            cache_retention: CacheRetention::Short,
            transport: Transport::Auto,
            websocket_connect_timeout: Some(DEFAULT_WEBSOCKET_CONNECT_TIMEOUT),
            temperature: None,
            reasoning: None,
            service_tier: None,
            text_verbosity: TextVerbosity::Low,
            tool_choice: ToolChoice::Auto,
            deferred_tools_mode: None,
            supports_strict_mode: true,
            supports_grammar_tools: false,
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
            request_hooks: None,
        }
    }

    fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    fn with_max_retry_delay(mut self, max_retry_delay: Option<Duration>) -> Self {
        self.max_retry_delay = max_retry_delay;
        self
    }

    fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = (!timeout.is_zero()).then_some(timeout);
        self
    }

    fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    fn with_cache_retention(mut self, retention: CacheRetention) -> Self {
        self.cache_retention = retention;
        self
    }

    fn with_transport(mut self, transport: Transport) -> Self {
        self.transport = transport;
        self
    }

    fn with_websocket_connect_timeout(mut self, timeout: Duration) -> Self {
        self.websocket_connect_timeout = Some(timeout);
        self
    }

    fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    fn with_reasoning_value(mut self, effort: String, summary: ReasoningSummary) -> Self {
        self.reasoning = Some(Reasoning { effort, summary });
        self
    }

    fn with_service_tier(mut self, service_tier: ServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    fn with_text_verbosity(mut self, verbosity: TextVerbosity) -> Self {
        self.text_verbosity = verbosity;
        self
    }

    fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    fn with_deferred_tools_mode(mut self, mode: Option<DeferredToolsMode>) -> Self {
        self.deferred_tools_mode = mode;
        self
    }

    fn with_compatibility(mut self, compat: &crate::OpenAiResponsesCompatibility) -> Self {
        self.supports_strict_mode = compat.supports_strict_mode.unwrap_or(true);
        self.supports_grammar_tools = compat.supports_open_ai_grammar_tools.unwrap_or(false);
        self
    }

    fn with_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.env = env;
        self
    }

    fn with_request_options(
        mut self,
        headers: BTreeMap<String, Option<String>>,
        request_hooks: crate::provider::RequestHooks,
    ) -> Self {
        self.headers = headers;
        self.request_hooks = Some(request_hooks);
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
    reasoning: Option<RequestReasoning<'a>>,
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
struct RequestReasoning<'a> {
    effort: &'a str,
    summary: ReasoningSummary,
}

#[derive(Clone)]
struct SseRequest {
    http_client: reqwest::Client,
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
    grammar_input_properties: BTreeMap<String, String>,
    service_tier: Option<ServiceTier>,
    headers: BTreeMap<String, Option<String>>,
    request_hooks: Option<crate::provider::RequestHooks>,
}

async fn response_events(
    model: &Model,
    context: &Context,
    options: &Options,
) -> Result<crate::provider_stream::ProviderStreamSetup, Error> {
    let overall_deadline: Option<Instant> = None;
    let account_id = account_id(&options.access_token).map_err(Error::InvalidRequest)?;
    let base_url = normalize_base_url(&model.base_url);
    let requested_session_id = match options.cache_retention {
        CacheRetention::None => None,
        CacheRetention::Short | CacheRetention::Long => options.session_id.clone(),
    };
    let prompt_cache_key = requested_session_id.as_deref().map(openai::clamp_cache_key);
    let cache_session_id = requested_session_id.filter(|session_id| !session_id.is_empty());
    let session_id = prompt_cache_key
        .as_deref()
        .filter(|session_id| !session_id.is_empty())
        .map(str::to_owned);
    let placement = crate::deferred_tools::split(
        context,
        options.deferred_tools_mode.is_some(),
        str::to_owned,
    );
    let tool_options = openai::ResponseToolOptions::codex(
        options.supports_strict_mode,
        options.supports_grammar_tools,
    );
    let grammar_input_properties =
        openai::grammar_input_properties(context.tools(), options.supports_grammar_tools)
            .map_err(Error::InvalidRequest)?;
    let request = Request {
        model: &model.id,
        store: false,
        stream: true,
        instructions: context
            .system()
            .filter(|system| !system.is_empty())
            .unwrap_or("You are a helpful assistant."),
        input: openai::response_input(
            openai::ResponseInputTarget::codex(&model.id),
            context,
            None,
            options.deferred_tools_mode.map(|mode| (&placement, mode)),
            &grammar_input_properties,
            tool_options,
        )
        .map_err(Error::InvalidRequest)?,
        tools: tools(&placement, tool_options).map_err(Error::InvalidRequest)?,
        text: TextOptions {
            verbosity: options.text_verbosity,
        },
        temperature: options.temperature,
        reasoning: options
            .reasoning
            .as_ref()
            .map(|reasoning| RequestReasoning {
                effort: &reasoning.effort,
                summary: reasoning.summary,
            }),
        service_tier: options.service_tier,
        include: ["reasoning.encrypted_content"],
        prompt_cache_key,
        tool_choice: options.tool_choice,
        parallel_tool_calls: true,
    };
    let value =
        serde_json::to_value(&request).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let value = match &options.request_hooks {
        Some(hooks) => hooks.payload(value).await?,
        None => value,
    };
    let json =
        serde_json::to_vec(&value).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let request_bytes = json.len();
    let sse_request = SseRequest {
        http_client: options.http_client.clone(),
        base_url: base_url.clone(),
        model: model.id.clone(),
        access_token: options.access_token.clone(),
        account_id: account_id.clone(),
        session_id: session_id.clone(),
        json,
        max_retries: options.max_retries,
        max_retry_delay: options.max_retry_delay,
        cancellation: options.cancellation.clone(),
        connection_timeout: options.timeout,
        first_event_timeout: None,
        idle_timeout: None,
        overall_deadline,
        grammar_input_properties: grammar_input_properties.clone(),
        service_tier: options.service_tier,
        headers: options.headers.clone(),
        request_hooks: options.request_hooks.clone(),
    };
    let websocket_disabled_for_session = options.transport != Transport::Sse
        && websocket_sse_fallback_active(cache_session_id.as_deref());
    if websocket_disabled_for_session {
        record_websocket_sse_fallback(cache_session_id.as_deref());
    }
    let mut fallback_diagnostic = None;
    if options.transport != Transport::Sse && !websocket_disabled_for_session {
        let websocket_value = match value.clone() {
            serde_json::Value::Object(mut object) => {
                object
                    .entry("type")
                    .or_insert_with(|| serde_json::Value::String("response.create".into()));
                serde_json::Value::Object(object)
            }
            _ => serde_json::json!({"type": "response.create"}),
        };
        match websocket_stream(
            WebSocketRequest {
                base_url: &base_url,
                model: &model.id,
                access_token: &options.access_token,
                account_id: &account_id,
                cache_session_id: cache_session_id.as_deref(),
                session_id: session_id.as_deref(),
                body: websocket_value,
                grammar_input_properties: &grammar_input_properties,
                headers: &options.headers,
            },
            options,
            overall_deadline,
        )
        .await
        {
            Ok(mut websocket) => {
                let fallback_session_id = cache_session_id.clone();
                let configured_transport = options.transport;
                match websocket.next().await {
                    Some(mut event) if !should_fallback_to_sse(&event) => {
                        diagnose_websocket_failure(
                            &mut event,
                            fallback_session_id.as_deref(),
                            configured_transport,
                            request_bytes,
                        );
                        let output = async_stream::stream! {
                            yield event;
                            while let Some(mut event) = websocket.next().await {
                                diagnose_websocket_failure(
                                    &mut event,
                                    fallback_session_id.as_deref(),
                                    configured_transport,
                                    request_bytes,
                                );
                                yield event;
                            }
                        };
                        return Ok(crate::provider_stream::ProviderStreamSetup::new(Box::pin(
                            output,
                        )));
                    }
                    event => {
                        let error = event.as_ref().map_or_else(
                            || "websocket closed before a terminal event".into(),
                            websocket_error_message,
                        );
                        record_websocket_failure(fallback_session_id.as_deref(), &error);
                        record_websocket_sse_fallback(fallback_session_id.as_deref());
                        let diagnostic = websocket_diagnostic(
                            configured_transport,
                            &error,
                            false,
                            request_bytes,
                        );
                        return match sse_stream(&sse_request).await {
                            Ok(fallback) => {
                                Ok(crate::provider_stream::ProviderStreamSetup::new(fallback)
                                    .with_diagnostic(diagnostic))
                            }
                            Err(error) => Err(with_websocket_diagnostic(error, diagnostic)),
                        };
                    }
                }
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
            Err(WebSocketConnectError::Transport(error)) => {
                record_websocket_failure(cache_session_id.as_deref(), &error);
                record_websocket_sse_fallback(cache_session_id.as_deref());
                fallback_diagnostic = Some(websocket_diagnostic(
                    options.transport,
                    &error,
                    false,
                    request_bytes,
                ));
            }
        }
    }
    let stream = sse_stream(&sse_request).await?;
    Ok(match fallback_diagnostic {
        Some(diagnostic) => {
            crate::provider_stream::ProviderStreamSetup::new(stream).with_diagnostic(diagnostic)
        }
        None => crate::provider_stream::ProviderStreamSetup::new(stream),
    })
}

async fn sse_stream(
    request: &SseRequest,
) -> Result<crate::provider_stream::ProviderEventStream, Error> {
    let (body, compressed) = match zstd::stream::encode_all(request.json.as_slice(), 3) {
        Ok(body) => (body, true),
        Err(_) => (request.json.clone(), false),
    };
    let client = &request.http_client;
    let url = response_url(&request.base_url);
    let mut headers =
        http::request_headers(BTreeMap::new(), &request.headers).map_err(Error::InvalidRequest)?;
    let mut fixed = BTreeMap::from([
        (
            "authorization".into(),
            format!("Bearer {}", request.access_token),
        ),
        ("chatgpt-account-id".into(), request.account_id.clone()),
        ("openai-beta".into(), "responses=experimental".into()),
        ("originator".into(), "ds".into()),
        (
            "user-agent".into(),
            concat!("ds-ai/", env!("CARGO_PKG_VERSION")).into(),
        ),
        ("accept".into(), "text/event-stream".into()),
        ("content-type".into(), "application/json".into()),
    ]);
    if compressed {
        fixed.insert("content-encoding".into(), "zstd".into());
    }
    if let Some(session_id) = &request.session_id {
        fixed.insert("session-id".into(), session_id.clone());
        fixed.insert("x-client-request-id".into(), session_id.clone());
    }
    headers.extend(http::request_headers(fixed, &BTreeMap::new()).map_err(Error::InvalidRequest)?);
    let response = transport::connect(
        retry::send(
            retry::Policy {
                max_retries: request.max_retries,
                max_delay: request.max_retry_delay,
                cancellation: &request.cancellation,
                deadline: request.overall_deadline,
                profile: retry::Profile::Codex,
                request_timeout: request
                    .connection_timeout
                    .filter(|timeout| !timeout.is_zero()),
            },
            || {
                client
                    .post(&url)
                    .headers(headers.clone())
                    .body(body.clone())
                    .send()
            },
            |response| async {
                match &request.request_hooks {
                    Some(hooks) => hooks.response(response).await,
                    None => Ok(()),
                }
            },
        ),
        None,
        request.overall_deadline,
    )
    .await?;
    if !response.status().is_success() {
        return Err(http::codex_provider_error(
            response,
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
        openai::ResponseEventOptions {
            grammar_input_properties: request.grammar_input_properties.clone(),
            requested_service_tier: request
                .service_tier
                .and_then(|service_tier| service_tier.as_str().map(str::to_owned)),
            use_requested_for_default: true,
            mode: openai::ResponseMode::CodexSse,
        },
    ))
}

fn should_fallback_to_sse(event: &Result<crate::provider_stream::ProviderEvent, Error>) -> bool {
    match event {
        Err(Error::Timeout {
            phase: crate::TimeoutPhase::FirstEvent,
            ..
        }) => true,
        Err(Error::Stream { partial, .. } | Error::IncompleteStream { partial }) => {
            partial.id.is_none() && partial.content.is_empty()
        }
        Err(Error::Response {
            code: Some(code),
            partial,
            ..
        }) => {
            code == "websocket_connection_limit_reached"
                && partial.id.is_none()
                && partial.content.is_empty()
        }
        _ => false,
    }
}

fn diagnose_websocket_failure(
    event: &mut Result<crate::provider_stream::ProviderEvent, Error>,
    session_id: Option<&str>,
    transport: Transport,
    request_bytes: usize,
) {
    let Some(error) = websocket_transport_failure(event) else {
        return;
    };
    record_websocket_failure(session_id, &error);
    add_websocket_diagnostic(
        event,
        websocket_diagnostic(transport, &error, true, request_bytes),
    );
}

fn websocket_transport_failure(
    event: &Result<crate::provider_stream::ProviderEvent, Error>,
) -> Option<String> {
    match event {
        Err(
            error @ (Error::Http(_)
            | Error::Stream { .. }
            | Error::IncompleteStream { .. }
            | Error::Timeout { .. }),
        ) => Some(error.to_string()),
        _ => None,
    }
}

fn websocket_error_message(event: &Result<crate::provider_stream::ProviderEvent, Error>) -> String {
    event.as_ref().err().map_or_else(
        || "websocket closed before a terminal event".into(),
        ToString::to_string,
    )
}

fn websocket_diagnostic(
    transport: Transport,
    error: &str,
    events_emitted: bool,
    request_bytes: usize,
) -> crate::AssistantMessageDiagnostic {
    let mut details = BTreeMap::from([
        (
            "configuredTransport".into(),
            serde_json::Value::String(transport.as_str().into()),
        ),
        (
            "eventsEmitted".into(),
            serde_json::Value::Bool(events_emitted),
        ),
        (
            "phase".into(),
            serde_json::Value::String(
                if events_emitted {
                    "after_message_stream_start"
                } else {
                    "before_message_stream_start"
                }
                .into(),
            ),
        ),
        ("requestBytes".into(), request_bytes.into()),
    ]);
    if !events_emitted {
        details.insert("fallbackTransport".into(), "sse".into());
    }
    crate::AssistantMessageDiagnostic {
        r#type: "provider_transport_failure".into(),
        timestamp: timestamp(),
        error: Some(crate::DiagnosticError {
            name: Some("Error".into()),
            message: error.into(),
            stack: None,
            code: None,
        }),
        details: Some(details),
    }
}

fn add_websocket_diagnostic(
    event: &mut Result<crate::provider_stream::ProviderEvent, Error>,
    diagnostic: crate::AssistantMessageDiagnostic,
) {
    match event {
        Ok(crate::provider_stream::ProviderEvent::Done(response)) => {
            response.add_diagnostic(diagnostic)
        }
        Err(error) if !matches!(&*error, Error::Protocol { .. }) => {
            if let Some(partial) = error.partial_mut() {
                partial.add_diagnostic(diagnostic);
            }
        }
        _ => {}
    }
}

fn with_websocket_diagnostic(
    mut error: Error,
    diagnostic: crate::AssistantMessageDiagnostic,
) -> Error {
    if matches!(
        &error,
        Error::Protocol { .. } | Error::Cancelled { partial: None }
    ) {
        return error;
    }
    if let Some(partial) = error.partial_mut() {
        partial.add_diagnostic(diagnostic);
        return error;
    }
    let message = match error {
        Error::Provider { message, .. } => message,
        error => error.to_string(),
    };
    let mut partial = Response::default();
    partial.add_diagnostic(diagnostic);
    Error::Response {
        code: None,
        message,
        partial,
    }
}

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

enum WebSocketConnectError {
    Cancelled,
    OverallTimeout,
    Transport(String),
}

struct WebSocketRequest<'a> {
    base_url: &'a str,
    model: &'a str,
    access_token: &'a str,
    account_id: &'a str,
    cache_session_id: Option<&'a str>,
    session_id: Option<&'a str>,
    body: serde_json::Value,
    grammar_input_properties: &'a BTreeMap<String, String>,
    headers: &'a BTreeMap<String, Option<String>>,
}

#[derive(Clone)]
struct WebSocketHandshake {
    url: String,
    proxy: Option<url::Url>,
    access_token: String,
    account_id: String,
    session_id: Option<String>,
    request_id: String,
    headers: BTreeMap<String, Option<String>>,
}

struct ActiveWebSocket {
    socket: OwnedMutexGuard<WebSocket>,
    shutdown: CancellationToken,
    continuation: Arc<StdMutex<Option<Continuation>>>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ReconnectCause {
    StaleSocket,
    ConnectionLimit,
    MissingContinuation,
}

impl ReconnectCause {
    fn clears_continuation(self) -> bool {
        self != Self::StaleSocket
    }
}

struct WebSocketReconnect {
    cache_key: Option<String>,
    handshake: WebSocketHandshake,
    cancellation: CancellationToken,
    connect_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
    continuation_state: SharedContinuationState,
    lease_busy: Arc<StdMutex<Option<Arc<AtomicBool>>>>,
    cached_context: bool,
    request: serde_json::Value,
}

impl WebSocketReconnect {
    async fn run(&self, active: ActiveWebSocket) -> Result<ActiveWebSocket, WebSocketConnectError> {
        let body = serde_json::to_string(&self.request)
            .map_err(|error| WebSocketConnectError::Transport(error.to_string()))?;
        let (socket, shutdown, continuation, last_used, busy) = replace_websocket(
            active.socket,
            self.cache_key.as_deref(),
            &self.handshake,
            &self.cancellation,
            self.connect_timeout,
            self.overall_deadline,
            &body,
        )
        .await?;
        *self
            .continuation_state
            .lock()
            .expect("websocket continuation state lock") = ContinuationState {
            continuation: continuation.clone(),
            last_used,
        };
        *self.lease_busy.lock().expect("cached websocket busy lock") = Some(busy);
        record_websocket_request(
            self.handshake.session_id.as_deref(),
            false,
            self.cached_context,
            &self.request,
        );
        Ok(ActiveWebSocket {
            socket,
            shutdown,
            continuation,
        })
    }
}

fn websocket_reconnect_read_error(error: WebSocketConnectError) -> transport::ReadError {
    match error {
        WebSocketConnectError::Cancelled => transport::ReadError::Cancelled,
        WebSocketConnectError::OverallTimeout => {
            transport::ReadError::Timeout(crate::TimeoutPhase::Overall)
        }
        WebSocketConnectError::Transport(_) => {
            transport::ReadError::Stream("websocket reconnect failed".into())
        }
    }
}

async fn websocket_stream(
    request: WebSocketRequest<'_>,
    options: &Options,
    overall_deadline: Option<Instant>,
) -> Result<crate::provider_stream::ProviderEventStream, WebSocketConnectError> {
    let url = websocket_url(request.base_url);
    let proxy =
        resolve_websocket_proxy(&url, &options.env).map_err(WebSocketConnectError::Transport)?;
    let cache_key = request.cache_session_id.map(|session_id| {
        websocket_cache_key(
            request.base_url,
            request.account_id,
            session_id,
            proxy.as_ref(),
        )
    });
    let request_id = request
        .session_id
        .map(str::to_owned)
        .map_or_else(crate::uuid_v7, Ok)
        .map_err(|error| WebSocketConnectError::Transport(error.to_string()))?;
    let handshake = WebSocketHandshake {
        url,
        proxy,
        access_token: request.access_token.to_owned(),
        account_id: request.account_id.to_owned(),
        session_id: request.cache_session_id.map(str::to_owned),
        request_id,
        headers: request.headers.clone(),
    };
    let lookup = cache_key.as_deref().map_or(
        CachedWebSocketLookup {
            connection: None,
            cacheable: true,
        },
        |key| acquire_cached_websocket(key, WEBSOCKET_IDLE_TTL),
    );
    let (connection, reused, active_cache_key) = if let Some(cached) = lookup.connection {
        (cached, true, cache_key.clone())
    } else {
        let connection = connect_websocket(
            &handshake,
            &options.cancellation,
            options.websocket_connect_timeout,
            overall_deadline,
        )
        .await?;
        let active_cache_key = if lookup.cacheable {
            cache_key.clone()
        } else {
            None
        };
        if let Some(cache_key) = &active_cache_key {
            websockets()
                .lock()
                .expect("websocket cache lock")
                .insert(cache_key.clone(), connection.clone());
        }
        (connection, false, active_cache_key)
    };
    let mut lease = WebSocketLease {
        busy: Arc::new(StdMutex::new(
            active_cache_key.as_ref().map(|_| connection.busy.clone()),
        )),
        key: active_cache_key.clone(),
    };
    let lease_busy = lease.busy.clone();
    let CachedWebSocket {
        socket,
        shutdown,
        continuation,
        last_used,
        ..
    } = connection;
    let socket = tokio::select! {
        biased;
        _ = options.cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = shutdown.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        socket = socket.lock_owned() => socket,
    };
    *last_used.lock().expect("websocket last-used lock") = Instant::now();
    let mut active = ActiveWebSocket {
        socket,
        shutdown,
        continuation,
    };
    let cached_context = matches!(
        options.transport,
        Transport::Auto | Transport::WebSocketCached
    );
    let full_request = request.body;
    let body = if cached_context {
        continuation_request(
            &full_request,
            active
                .continuation
                .lock()
                .expect("websocket continuation lock")
                .as_ref(),
        )
    } else {
        full_request.clone()
    };
    let mut used_continuation = body.get("previous_response_id").is_some();
    record_websocket_request(
        handshake.session_id.as_deref(),
        reused,
        cached_context,
        &body,
    );
    let body = serde_json::to_string(&body)
        .map_err(|error| WebSocketConnectError::Transport(error.to_string()))?;
    let continuation_state = Arc::new(StdMutex::new(ContinuationState {
        continuation: active.continuation.clone(),
        last_used: last_used.clone(),
    }));
    let output_continuation_state = continuation_state.clone();
    let reconnect = WebSocketReconnect {
        cache_key: active_cache_key.clone(),
        handshake,
        cancellation: options.cancellation.clone(),
        connect_timeout: options.websocket_connect_timeout,
        overall_deadline,
        continuation_state,
        lease_busy,
        cached_context,
        request: full_request.clone(),
    };
    let sent = tokio::select! {
        biased;
        _ = options.cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = active.shutdown.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        result = active.socket.send(WebSocketMessage::Text(body.into())) => result,
    };
    let mut retried_socket = false;
    if let Err(error) = sent {
        if !reused {
            return Err(WebSocketConnectError::Transport(error.to_string()));
        }
        active = reconnect.run(active).await?;
        used_continuation = false;
        retried_socket = true;
    }
    let cancellation = options.cancellation.clone();
    let first_event_timeout = options.timeout;
    let idle_timeout = options.timeout;
    let output_request = full_request.clone();
    let output_model = request.model.to_owned();
    let output_grammar_input_properties = request.grammar_input_properties.clone();
    let output_tool_options = openai::ResponseToolOptions::codex(
        options.supports_strict_mode,
        options.supports_grammar_tools,
    );
    let events = async_stream::stream! {
        let mut retried_socket = retried_socket;
        let mut saw_event = false;
        let mut event_deadline = first_event_timeout.map(|timeout| Instant::now() + timeout);
        let mut emitted = false;
        let mut retried_missing_continuation = false;
        loop {
            let message = tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    yield Err(transport::ReadError::Cancelled);
                    return;
                }
                _ = active.shutdown.cancelled() => {
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
                message = active.socket.next() => message,
            };
            let reconnect_cause = (reused
                && !emitted
                && !retried_socket
                && matches!(
                    &message,
                    Some(Ok(WebSocketMessage::Close(_))) | Some(Err(_)) | None
                ))
            .then_some(ReconnectCause::StaleSocket);
            let data = if reconnect_cause.is_some() {
                None
            } else {
                match message {
                    Some(Ok(WebSocketMessage::Text(text))) => Some(text.to_string()),
                    Some(Ok(WebSocketMessage::Binary(bytes))) => {
                        Some(String::from_utf8_lossy(&bytes).into_owned())
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
                }
            };
            let code = data.as_deref().and_then(codex_error_code);
            let reconnect_cause = reconnect_cause.or_else(|| {
                if !emitted
                    && !retried_socket
                    && code.as_deref() == Some("websocket_connection_limit_reached")
                {
                    Some(ReconnectCause::ConnectionLimit)
                } else if used_continuation
                    && !retried_missing_continuation
                    && code.as_deref() == Some("previous_response_not_found")
                {
                    Some(ReconnectCause::MissingContinuation)
                } else {
                    None
                }
            });
            if let Some(cause) = reconnect_cause {
                if cause.clears_continuation() {
                    *active
                        .continuation
                        .lock()
                        .expect("websocket continuation lock") = None;
                }
                active = match reconnect.run(active).await {
                    Ok(active) => active,
                    Err(error) => {
                        yield Err(websocket_reconnect_read_error(error));
                        return;
                    }
                };
                retried_missing_continuation |= cause == ReconnectCause::MissingContinuation;
                retried_socket = true;
                used_continuation = false;
                saw_event = false;
                event_deadline = first_event_timeout.map(|timeout| Instant::now() + timeout);
                continue;
            }
            if let Some(mut data) = data {
                data = normalize_codex_error(data);
                saw_event = true;
                emitted = true;
                event_deadline = idle_timeout.map(|timeout| Instant::now() + timeout);
                let terminal = observe_websocket_event(&data);
                if terminal && active_cache_key.is_none() {
                    let _ = active.socket.close(None).await;
                }
                yield Ok(data);
                if terminal {
                    return;
                }
            }
        }
    };
    let mut decoded = openai::decode_events(
        Box::pin(events),
        request.model.to_owned(),
        openai::ResponseEventOptions {
            grammar_input_properties: request.grammar_input_properties.clone(),
            requested_service_tier: options
                .service_tier
                .and_then(|service_tier| service_tier.as_str().map(str::to_owned)),
            use_requested_for_default: true,
            mode: openai::ResponseMode::CodexWebSocket,
        },
    );
    let output = async_stream::stream! {
        while let Some(event) = decoded.next().await {
            match event {
                Ok(crate::provider_stream::ProviderEvent::Done(response)) => {
                    if cached_context {
                        let state = output_continuation_state
                            .lock()
                            .expect("websocket continuation state lock")
                            .clone();
                        cache_continuation(
                            &state.continuation,
                            &state.last_used,
                            &output_request,
                            &output_model,
                            &response,
                            &output_grammar_input_properties,
                            output_tool_options,
                        );
                    }
                    drop(decoded);
                    lease.complete(WEBSOCKET_IDLE_TTL);
                    yield Ok(crate::provider_stream::ProviderEvent::Done(response));
                    return;
                }
                Err(error) => {
                    drop(decoded);
                    let error = codex_protocol_error(error);
                    let continuation = output_continuation_state
                        .lock()
                        .expect("websocket continuation state lock")
                        .continuation
                        .clone();
                    *continuation
                        .lock()
                        .expect("websocket continuation lock") = None;
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

fn codex_protocol_error(error: Error) -> Error {
    match error {
        Error::Protocol { message, partial } => Error::Response {
            code: None,
            message: format!("Invalid Codex WebSocket JSON: {message}"),
            partial,
        },
        error => error,
    }
}

fn resolve_websocket_proxy(
    target: &str,
    env: &BTreeMap<String, String>,
) -> Result<Option<url::Url>, String> {
    let target = url::Url::parse(target)
        .map_err(|error| format!("Invalid WebSocket URL {target:?}: {error}"))?;
    let host = target
        .host_str()
        .ok_or_else(|| format!("WebSocket URL has no host: {target}"))?;
    let port = target
        .port_or_known_default()
        .ok_or_else(|| format!("WebSocket URL has no port: {target}"))?;
    if no_proxy_matches(host, port, proxy_env(env, "NO_PROXY")) {
        return Ok(None);
    }
    let (proxy_name, default_scheme) = match target.scheme() {
        "ws" => ("HTTP_PROXY", "http"),
        "wss" => ("HTTPS_PROXY", "https"),
        scheme => return Err(format!("Unsupported WebSocket protocol: {scheme}")),
    };
    let Some(proxy) = proxy_env(env, proxy_name).or_else(|| proxy_env(env, "ALL_PROXY")) else {
        return Ok(None);
    };
    let proxy = if proxy.contains("://") {
        proxy.to_owned()
    } else {
        format!("{default_scheme}://{proxy}")
    };
    let proxy =
        url::Url::parse(&proxy).map_err(|error| format!("Invalid proxy URL {proxy:?}: {error}"))?;
    if proxy.scheme() != "http" {
        return Err(format!(
            "{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {}:",
            proxy.scheme()
        ));
    }
    Ok(Some(proxy))
}

fn websocket_cache_key(
    base_url: &str,
    account_id: &str,
    session_id: &str,
    proxy: Option<&url::Url>,
) -> String {
    format!(
        "{base_url}\u{1f}{account_id}\u{1f}{session_id}\u{1f}{}",
        proxy.map(url::Url::as_str).unwrap_or_default()
    )
}

fn proxy_env<'a>(env: &'a BTreeMap<String, String>, name: &str) -> Option<&'a str> {
    env.get(&name.to_ascii_lowercase())
        .or_else(|| env.get(&name.to_ascii_uppercase()))
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

fn no_proxy_matches(host: &str, port: u16, no_proxy: Option<&str>) -> bool {
    let Some(no_proxy) = no_proxy else {
        return false;
    };
    let host = host.to_ascii_lowercase();
    let no_proxy = no_proxy.to_ascii_lowercase();
    no_proxy == "*"
        || no_proxy
            .split(|character: char| character == ',' || character.is_whitespace())
            .filter(|entry| !entry.is_empty())
            .any(|entry| no_proxy_entry_matches(&host, port, entry))
}

fn no_proxy_entry_matches(host: &str, port: u16, entry: &str) -> bool {
    let (pattern, pattern_port) = entry
        .rsplit_once(':')
        .and_then(|(pattern, port)| port.parse::<u16>().ok().map(|port| (pattern, port)))
        .map_or((entry, None), |(pattern, port)| (pattern, Some(port)));
    if pattern_port.is_some_and(|pattern_port| pattern_port != port) {
        return false;
    }
    if pattern.starts_with(['.', '*']) {
        host.ends_with(pattern.trim_start_matches('*'))
    } else {
        host == pattern
    }
}

async fn open_websocket(
    request: tokio_tungstenite::tungstenite::http::Request<()>,
    handshake: &WebSocketHandshake,
) -> Result<WebSocket, String> {
    let Some(proxy) = &handshake.proxy else {
        return connect_async(request)
            .await
            .map(|(socket, _)| socket)
            .map_err(|error| error.to_string());
    };
    let stream = connect_proxy_tunnel(proxy, &handshake.url).await?;
    client_async_tls(request, stream)
        .await
        .map(|(socket, _)| socket)
        .map_err(|error| error.to_string())
}

async fn connect_proxy_tunnel(proxy: &url::Url, target: &str) -> Result<TcpStream, String> {
    let proxy_host = proxy
        .host_str()
        .ok_or_else(|| format!("Proxy URL has no host: {proxy}"))?;
    let proxy_port = proxy
        .port_or_known_default()
        .ok_or_else(|| format!("Proxy URL has no port: {proxy}"))?;
    let target = url::Url::parse(target)
        .map_err(|error| format!("Invalid WebSocket URL {target:?}: {error}"))?;
    let target_host = target
        .host_str()
        .ok_or_else(|| format!("WebSocket URL has no host: {target}"))?;
    let target_port = target
        .port_or_known_default()
        .ok_or_else(|| format!("WebSocket URL has no port: {target}"))?;
    let authority = if target_host.contains(':') {
        format!("[{target_host}]:{target_port}")
    } else {
        format!("{target_host}:{target_port}")
    };
    let mut request = format!(
        "CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\nProxy-Connection: Keep-Alive\r\n"
    );
    if !proxy.username().is_empty() {
        let credentials = format!(
            "{}:{}",
            proxy.username(),
            proxy.password().unwrap_or_default()
        );
        request.push_str(&format!(
            "Proxy-Authorization: Basic {}\r\n",
            BASE64_STANDARD.encode(credentials)
        ));
    }
    request.push_str("\r\n");
    let mut stream = TcpStream::connect((proxy_host, proxy_port))
        .await
        .map_err(|error| format!("Failed to connect to proxy {proxy}: {error}"))?;
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|error| format!("Failed to write proxy CONNECT request: {error}"))?;
    let mut response = Vec::new();
    while !response.ends_with(b"\r\n\r\n") {
        if response.len() == 16 * 1024 {
            return Err("Proxy CONNECT response headers exceed 16384 bytes".into());
        }
        let byte = stream
            .read_u8()
            .await
            .map_err(|error| format!("Failed to read proxy CONNECT response: {error}"))?;
        response.push(byte);
    }
    let status_line = std::str::from_utf8(
        response
            .split(|byte| *byte == b'\n')
            .next()
            .unwrap_or_default(),
    )
    .map_err(|error| format!("Invalid proxy CONNECT response: {error}"))?;
    let status = status_line
        .split_ascii_whitespace()
        .nth(1)
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or_else(|| format!("Invalid proxy CONNECT response: {status_line}"))?;
    if !(200..300).contains(&status) {
        return Err(format!("Proxy CONNECT failed with HTTP {status}"));
    }
    Ok(stream)
}

async fn connect_websocket(
    handshake: &WebSocketHandshake,
    cancellation: &CancellationToken,
    connect_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
) -> Result<CachedWebSocket, WebSocketConnectError> {
    let mut connection_request = handshake
        .url
        .as_str()
        .into_client_request()
        .map_err(|error| WebSocketConnectError::Transport(error.to_string()))?;
    for (name, value) in &handshake.headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| WebSocketConnectError::Transport(error.to_string()))?;
        match value {
            Some(value) => {
                connection_request.headers_mut().insert(
                    name,
                    HeaderValue::from_str(value)
                        .map_err(|error| WebSocketConnectError::Transport(error.to_string()))?,
                );
            }
            None => {
                connection_request.headers_mut().remove(name);
            }
        }
    }
    connection_request.headers_mut().remove("accept");
    connection_request.headers_mut().remove("content-type");
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
            HeaderValue::from_str(&value)
                .map_err(|error| WebSocketConnectError::Transport(error.to_string()))?,
        );
    }
    let session_header = connection_request.headers()["x-client-request-id"].clone();
    connection_request
        .headers_mut()
        .insert("session-id", session_header);
    let connect_timeout = connect_timeout.filter(|timeout| !timeout.is_zero());
    let connect_deadline = connect_timeout.map(|timeout| Instant::now() + timeout);
    let socket = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        _ = transport::wait_until(connect_deadline) => {
            return Err(WebSocketConnectError::Transport(format!(
                "websocket connect timeout after {}ms",
                connect_timeout
                    .unwrap_or(DEFAULT_WEBSOCKET_CONNECT_TIMEOUT)
                    .as_millis()
            )));
        }
        connection = open_websocket(connection_request, handshake) => {
            connection.map_err(WebSocketConnectError::Transport)?
        }
    };
    Ok(CachedWebSocket {
        socket: Arc::new(AsyncMutex::new(socket)),
        shutdown: CancellationToken::new(),
        busy: Arc::new(AtomicBool::new(true)),
        created_at: Instant::now(),
        session_id: handshake.session_id.clone(),
        continuation: Arc::new(StdMutex::new(None)),
        last_used: Arc::new(StdMutex::new(Instant::now())),
    })
}

async fn replace_websocket(
    mut socket: OwnedMutexGuard<WebSocket>,
    cache_key: Option<&str>,
    handshake: &WebSocketHandshake,
    cancellation: &CancellationToken,
    connect_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
    body: &str,
) -> Result<
    (
        OwnedMutexGuard<WebSocket>,
        CancellationToken,
        Arc<StdMutex<Option<Continuation>>>,
        Arc<StdMutex<Instant>>,
        Arc<AtomicBool>,
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
    let shutdown = connection.shutdown.clone();
    let last_used = connection.last_used.clone();
    let fresh_busy = connection.busy.clone();
    let fresh_socket = connection.socket.clone();
    let mut socket = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = shutdown.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        socket = fresh_socket.lock_owned() => socket,
    };
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = shutdown.cancelled() => return Err(WebSocketConnectError::Cancelled),
        _ = transport::wait_until(overall_deadline) => {
            return Err(WebSocketConnectError::OverallTimeout);
        }
        sent = socket.send(WebSocketMessage::Text(body.to_owned().into())) => {
            sent.map_err(|error| WebSocketConnectError::Transport(error.to_string()))?;
        }
    }
    if let Some(cache_key) = cache_key {
        websockets()
            .lock()
            .expect("websocket cache lock")
            .insert(cache_key.to_owned(), connection);
    }
    Ok((socket, shutdown, continuation, last_used, fresh_busy))
}

fn codex_error_code(data: &str) -> Option<String> {
    let event = serde_json::from_str::<serde_json::Value>(data).ok()?;
    event
        .get("code")
        .or_else(|| event.get("error").and_then(|error| error.get("code")))
        .or_else(|| event.pointer("/response/error/code"))
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
    static REGISTERED: Once = Once::new();
    REGISTERED.call_once(|| {
        let _ = crate::register_session_resource_cleanup(|session_id| {
            close_websocket_sessions(session_id);
            Ok(())
        });
    });
    WEBSOCKETS.get_or_init(|| StdMutex::new(HashMap::new()))
}

pub fn websocket_debug_stats(session_id: &str) -> Option<WebSocketDebugStats> {
    websocket_debug_state()
        .lock()
        .expect("websocket debug lock")
        .stats
        .get(session_id)
        .cloned()
}

pub fn reset_websocket_debug_stats(session_id: Option<&str>) {
    let mut state = websocket_debug_state()
        .lock()
        .expect("websocket debug lock");
    if let Some(session_id) = session_id {
        state.stats.remove(session_id);
        state.fallback_sessions.remove(session_id);
    } else {
        state.stats.clear();
        state.fallback_sessions.clear();
    }
}

pub fn close_websocket_sessions(session_id: Option<&str>) {
    let connections = {
        let mut cache = websockets().lock().expect("websocket cache lock");
        if let Some(session_id) = session_id {
            let keys = cache
                .iter()
                .filter(|(_, connection)| connection.session_id.as_deref() == Some(session_id))
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>();
            keys.into_iter()
                .filter_map(|key| cache.remove(&key))
                .collect::<Vec<_>>()
        } else {
            cache.drain().map(|(_, connection)| connection).collect()
        }
    };
    close_websockets(connections);
}

fn websocket_sse_fallback_active(session_id: Option<&str>) -> bool {
    session_id.is_some_and(|session_id| {
        websocket_debug_state()
            .lock()
            .expect("websocket debug lock")
            .fallback_sessions
            .contains(session_id)
    })
}

fn record_websocket_request(
    session_id: Option<&str>,
    reused: bool,
    cached_context: bool,
    body: &serde_json::Value,
) {
    let Some(session_id) = session_id else {
        return;
    };
    let mut state = websocket_debug_state()
        .lock()
        .expect("websocket debug lock");
    let stats = state.stats.entry(session_id.into()).or_default();
    stats.requests += 1;
    if reused {
        stats.connections_reused += 1;
    } else {
        stats.connections_created += 1;
    }
    if cached_context {
        stats.cached_context_requests += 1;
    }
    if body.get("store").and_then(serde_json::Value::as_bool) == Some(true) {
        stats.store_true_requests += 1;
    }
    stats.last_input_items = body
        .get("input")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if let Some(response_id) = body
        .get("previous_response_id")
        .and_then(serde_json::Value::as_str)
    {
        stats.delta_requests += 1;
        stats.last_delta_input_items = Some(stats.last_input_items);
        stats.last_previous_response_id = Some(response_id.into());
    } else {
        stats.full_context_requests += 1;
        stats.last_delta_input_items = None;
        stats.last_previous_response_id = None;
    }
}

fn record_websocket_sse_fallback(session_id: Option<&str>) {
    let Some(session_id) = session_id else {
        return;
    };
    let mut state = websocket_debug_state()
        .lock()
        .expect("websocket debug lock");
    let active = state.fallback_sessions.contains(session_id);
    let stats = state.stats.entry(session_id.into()).or_default();
    stats.sse_fallbacks += 1;
    stats.websocket_fallback_active = Some(active);
}

fn record_websocket_failure(session_id: Option<&str>, error: impl Into<String>) {
    let Some(session_id) = session_id else {
        return;
    };
    let mut state = websocket_debug_state()
        .lock()
        .expect("websocket debug lock");
    state.fallback_sessions.insert(session_id.into());
    let stats = state.stats.entry(session_id.into()).or_default();
    stats.websocket_failures += 1;
    stats.last_websocket_error = Some(error.into());
    stats.websocket_fallback_active = Some(true);
}

fn websocket_debug_state() -> &'static StdMutex<WebSocketDebugState> {
    static STATE: OnceLock<StdMutex<WebSocketDebugState>> = OnceLock::new();
    STATE.get_or_init(|| StdMutex::new(WebSocketDebugState::default()))
}

fn close_websocket(connection: CachedWebSocket) {
    close_websockets([connection]);
}

fn close_websockets(connections: impl IntoIterator<Item = CachedWebSocket>) {
    let connections = connections.into_iter().collect::<Vec<_>>();
    for connection in &connections {
        connection.shutdown.cancel();
    }
    let close = async move {
        for connection in connections {
            let mut socket = connection.socket.lock().await;
            let _ = socket.close(None).await;
        }
    };
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => {
            runtime.spawn(close);
        }
        Err(_) => {
            drop(std::thread::spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("websocket cleanup runtime")
                    .block_on(close);
            }));
        }
    }
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

fn acquire_cached_websocket(key: &str, idle_ttl: Duration) -> CachedWebSocketLookup {
    let mut cache = websockets().lock().expect("websocket cache lock");
    let expired = cache
        .get(key)
        .is_some_and(|connection| websocket_expired(connection, idle_ttl));
    if expired {
        let connection = cache.remove(key).expect("expired websocket exists");
        drop(cache);
        close_websocket(connection);
        return CachedWebSocketLookup {
            connection: None,
            cacheable: true,
        };
    }
    let Some(connection) = cache.get(key) else {
        return CachedWebSocketLookup {
            connection: None,
            cacheable: true,
        };
    };
    if connection.busy.swap(true, Ordering::AcqRel) {
        CachedWebSocketLookup {
            connection: None,
            cacheable: false,
        }
    } else {
        CachedWebSocketLookup {
            connection: Some(connection.clone()),
            cacheable: false,
        }
    }
}

fn websocket_expired(connection: &CachedWebSocket, idle_ttl: Duration) -> bool {
    if connection.busy.load(Ordering::Acquire) {
        return false;
    }
    if connection.created_at.elapsed() >= WEBSOCKET_MAX_AGE {
        return true;
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
    left == right
}

fn request_configuration(request: &serde_json::Value) -> serde_json::Value {
    let mut request = request.clone();
    if let Some(request) = request.as_object_mut() {
        request.remove("input");
        request.remove("previous_response_id");
    }
    request
}

fn observe_websocket_event(data: &str) -> bool {
    let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
        return false;
    };
    let event_type = event.get("type").and_then(serde_json::Value::as_str);
    matches!(
        event_type,
        Some("response.done" | "response.completed" | "response.incomplete")
    )
}

fn cache_continuation(
    continuation: &Arc<StdMutex<Option<Continuation>>>,
    last_used: &Arc<StdMutex<Instant>>,
    request: &serde_json::Value,
    model: &str,
    response: &Response,
    grammar_input_properties: &BTreeMap<String, String>,
    tool_options: openai::ResponseToolOptions,
) {
    *last_used.lock().expect("websocket last-used lock") = Instant::now();
    let Some(response_id) = response.id.as_deref().filter(|id| !id.is_empty()) else {
        *continuation.lock().expect("websocket continuation lock") = None;
        return;
    };
    let message = crate::AssistantMessage {
        content: response.content.clone(),
        api: crate::Api::OpenAiCodexResponses,
        provider: crate::ProviderId::new("openai-codex"),
        model: model.into(),
        response_model: (response.response_model != model).then(|| response.response_model.clone()),
        response_id: response.id.clone(),
        diagnostics: None,
        usage: response.usage.clone(),
        stop_reason: response.stop_reason,
        error_message: None,
        raw_stop_reason: response.raw_stop_reason.clone(),
        end_turn: response.end_turn,
        timestamp: 0,
    };
    let context = Context::new([crate::Message::assistant(message)]);
    let Ok(mut response_items) = openai::response_input(
        openai::ResponseInputTarget::codex(model),
        &context,
        None,
        None,
        grammar_input_properties,
        tool_options,
    ) else {
        *continuation.lock().expect("websocket continuation lock") = None;
        return;
    };
    response_items.retain(|item| {
        !matches!(
            item.get("type").and_then(serde_json::Value::as_str),
            Some("function_call_output" | "custom_tool_call_output")
        )
    });
    *continuation.lock().expect("websocket continuation lock") = Some(Continuation {
        request: request.clone(),
        response_id: response_id.into(),
        response_items,
    });
}

fn account_id(token: &str) -> Result<String, String> {
    let mut parts = token.split('.');
    let (Some(_), Some(payload), Some(_), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err("Failed to extract accountId from token".into());
    };
    let bytes = BASE64_URL_SAFE_NO_PAD
        .decode(payload)
        .or_else(|_| BASE64_URL_SAFE.decode(payload))
        .or_else(|_| BASE64_STANDARD.decode(payload))
        .map_err(|_| "Failed to extract accountId from token".to_string())?;
    let payload: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| "Failed to extract accountId from token".to_string())?;
    payload
        .get("https://api.openai.com/auth")
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(serde_json::Value::as_str)
        .filter(|account_id| !account_id.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "Failed to extract accountId from token".to_string())
}

fn normalize_base_url(base_url: &str) -> String {
    let base_url = base_url.trim();
    if base_url.is_empty() {
        DEFAULT_BASE_URL.into()
    } else {
        base_url.into()
    }
}

fn response_url(base_url: &str) -> String {
    let normalized = normalize_base_url(base_url);
    let base_url = normalized.trim_end_matches('/');
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

fn tools(
    placement: &ToolPlacement,
    options: openai::ResponseToolOptions,
) -> Result<Vec<serde_json::Value>, String> {
    openai::response_tools(&placement.immediate, options)
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_BASE_URL, UNSUPPORTED_PROXY_PROTOCOL_MESSAGE, normalize_base_url,
        resolve_websocket_proxy, response_url, websocket_cache_key, websocket_url,
    };
    use std::collections::BTreeMap;

    #[test]
    fn defaults_an_empty_base_url_to_the_codex_endpoint() {
        assert_eq!(normalize_base_url(" \t\n"), DEFAULT_BASE_URL);
        assert_eq!(
            response_url(" \t\n"),
            format!("{DEFAULT_BASE_URL}/codex/responses")
        );
        assert_eq!(
            websocket_url(" \t\n"),
            "wss://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn resolves_scoped_websocket_proxy_environment() {
        let env = BTreeMap::from([
            ("HTTP_PROXY".into(), "http://http-proxy.example:8080".into()),
            (
                "HTTPS_PROXY".into(),
                "http://https-proxy.example:8080".into(),
            ),
        ]);

        assert_eq!(
            resolve_websocket_proxy("ws://api.example/responses", &env)
                .unwrap()
                .unwrap()
                .as_str(),
            "http://http-proxy.example:8080/"
        );
        assert_eq!(
            resolve_websocket_proxy("wss://api.example/responses", &env)
                .unwrap()
                .unwrap()
                .as_str(),
            "http://https-proxy.example:8080/"
        );
    }

    #[test]
    fn respects_websocket_no_proxy_environment() {
        let env = BTreeMap::from([
            ("HTTPS_PROXY".into(), "http://proxy.example:8080".into()),
            ("NO_PROXY".into(), ".example.com:443".into()),
        ]);

        assert!(
            resolve_websocket_proxy("wss://api.example.com/responses", &env)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn rejects_unsupported_websocket_proxy_protocols() {
        for (proxy, scheme) in [
            ("https://proxy.example:8080", "https"),
            ("socks://proxy.example:1080", "socks"),
            ("socks5://proxy.example:1080", "socks5"),
            ("pac+http://proxy.example/proxy.pac", "pac+http"),
        ] {
            let env = BTreeMap::from([("HTTPS_PROXY".into(), proxy.into())]);

            assert_eq!(
                resolve_websocket_proxy("wss://api.example/responses", &env).unwrap_err(),
                format!("{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {scheme}:")
            );
        }
    }

    #[test]
    fn isolates_cached_websockets_by_proxy() {
        let first = url::Url::parse("http://first-proxy.example:8080").unwrap();
        let second = url::Url::parse("http://second-proxy.example:8080").unwrap();

        assert_ne!(
            websocket_cache_key("https://api.example", "account", "session", Some(&first)),
            websocket_cache_key("https://api.example", "account", "session", Some(&second))
        );
        assert_ne!(
            websocket_cache_key("https://api.example", "account", "session", Some(&first)),
            websocket_cache_key("https://api.example", "account", "session", None)
        );
    }
}
