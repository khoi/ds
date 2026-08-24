use crate::{
    CacheRetention, Content, Context, Error, InputContent, Message, ResponseMetadata,
    ResponseStream, http, openai, retry, schema, transport, types::OpenAiReplay,
};
use base64::prelude::*;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use std::{
    collections::HashMap,
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

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
const DEFAULT_WEBSOCKET_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const WEBSOCKET_IDLE_TTL: Duration = Duration::from_secs(5 * 60);
const WEBSOCKET_MAX_AGE: Duration = Duration::from_secs(55 * 60);

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
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Transport {
    #[default]
    Auto,
    Sse,
    WebSocket,
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
    include: [&'static str; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
}

#[derive(Serialize)]
struct TextOptions {
    verbosity: &'static str,
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
    let request = Request {
        model: &model.id,
        store: false,
        stream: true,
        instructions: context.system().unwrap_or("You are a helpful assistant."),
        input: input(model, context),
        tools: tools(context).map_err(Error::InvalidRequest)?,
        text: TextOptions { verbosity: "low" },
        include: ["reasoning.encrypted_content"],
        prompt_cache_key: session_id.clone(),
        tool_choice: "auto",
        parallel_tool_calls: true,
    };
    let value =
        serde_json::to_value(&request).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let json =
        serde_json::to_vec(&value).map_err(|error| Error::InvalidRequest(error.to_string()))?;
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
            Ok(stream) => return Ok(stream),
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
    let body = zstd::stream::encode_all(json.as_slice(), 3)
        .map_err(|error| Error::Compression(error.to_string()))?;
    let client = reqwest::Client::new();
    let url = response_url(&model.base_url);
    let response = transport::connect(
        retry::send(
            retry::Policy {
                max_retries: options.max_retries,
                max_delay: options.max_retry_delay,
                cancellation: &options.cancellation,
            },
            || {
                let mut request = client
                    .post(&url)
                    .bearer_auth(&options.access_token)
                    .header("chatgpt-account-id", &account_id)
                    .header("openai-beta", "responses=experimental")
                    .header("originator", "ds")
                    .header("user-agent", concat!("ds-ai/", env!("CARGO_PKG_VERSION")))
                    .header("accept", "text/event-stream")
                    .header("content-type", "application/json")
                    .header("content-encoding", "zstd");
                if let Some(session_id) = &session_id {
                    request = request
                        .header("session-id", session_id)
                        .header("x-client-request-id", session_id);
                }
                request.body(body.clone()).send()
            },
        ),
        options.connection_timeout,
        overall_deadline,
    )
    .await?;
    if !response.status().is_success() {
        return Err(http::provider_error(response).await);
    }
    Ok(openai::decode_stream(
        response,
        model.id.clone(),
        options.cancellation.clone(),
        options.first_event_timeout,
        options.idle_timeout,
        overall_deadline,
    ))
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
                yield Ok(data);
                if terminal {
                    return;
                }
            }
        }
    };
    Ok(openai::decode_events(
        Box::pin(events),
        request.model.to_owned(),
        metadata,
    ))
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

fn cached_websocket(key: &str, idle_ttl: Duration) -> Option<CachedWebSocket> {
    let mut cache = websockets().lock().expect("websocket cache lock");
    let expired = cache.get(key).is_some_and(|connection| {
        connection.created_at.elapsed() >= WEBSOCKET_MAX_AGE
            || connection.socket.try_lock().is_ok()
                && connection
                    .last_used
                    .lock()
                    .expect("websocket last-used lock")
                    .elapsed()
                    >= idle_ttl
    });
    if expired {
        cache.remove(key);
    }
    cache.get(key).cloned()
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
    if current.len() < baseline.len() || current[..baseline.len()] != baseline {
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

fn input(model: &Model, context: &Context) -> Vec<serde_json::Value> {
    let mut input = Vec::new();
    for message in context.messages() {
        match message {
            Message::User(content) => input.push(serde_json::json!({
                "role": "user",
                "content": content.iter().map(input_content).collect::<Vec<_>>()
            })),
            Message::Assistant(response) => {
                let Some(items) = response.openai_items(&model.id) else {
                    continue;
                };
                for (content_index, content) in response.content.iter().enumerate() {
                    let item = items.iter().find(|item| match item {
                        OpenAiReplay::Reasoning {
                            content_index: index,
                            ..
                        }
                        | OpenAiReplay::Message {
                            content_index: index,
                            ..
                        }
                        | OpenAiReplay::ToolCall {
                            content_index: index,
                            ..
                        } => *index == content_index,
                    });
                    match (content, item) {
                        (
                            Content::Reasoning(text),
                            Some(OpenAiReplay::Reasoning {
                                id,
                                encrypted_content,
                                ..
                            }),
                        ) => input.push(serde_json::json!({
                            "type": "reasoning",
                            "id": id,
                            "summary": [{"type": "summary_text", "text": text}],
                            "encrypted_content": encrypted_content
                        })),
                        (Content::Text(text), Some(OpenAiReplay::Message { id, phase, .. })) => {
                            input.push(serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text, "annotations": []}],
                                "status": "completed",
                                "id": id,
                                "phase": phase
                            }));
                        }
                        (
                            Content::ToolCall(call),
                            Some(OpenAiReplay::ToolCall {
                                item_id, namespace, ..
                            }),
                        ) => input.push(serde_json::json!({
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": serde_json::to_string(&call.arguments).expect("tool arguments serialize"),
                            "namespace": namespace
                        })),
                        _ => {}
                    }
                }
            }
            Message::ToolResult(result) => input.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": result.id,
                "output": openai::tool_result_output(result)
            })),
        }
    }
    input
}

fn input_content(content: &InputContent) -> serde_json::Value {
    match content {
        InputContent::Text(text) => serde_json::json!({"type": "input_text", "text": text}),
        InputContent::Image { media_type, data } => serde_json::json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": format!("data:{media_type};base64,{data}")
        }),
    }
}

fn tools(context: &Context) -> Result<Vec<serde_json::Value>, String> {
    context
        .tools()
        .iter()
        .map(|tool| {
            let parameters = if tool.strict() {
                schema::strict(&tool.parameters).map_err(|error| {
                    format!("tool {:?} has an invalid strict schema: {error}", tool.name)
                })?
            } else {
                tool.parameters.clone()
            };
            Ok(serde_json::json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "strict": tool.strict()
            }))
        })
        .collect()
}
