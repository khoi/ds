use crate::{AssistantMessage, Error, StopReason};
use async_trait::async_trait;
use bytes::BytesMut;
use futures_util::StreamExt;
use regex::RegexSet;
use reqwest::{Response, header::HeaderMap};
use std::{
    future::{Future, pending},
    sync::LazyLock,
    time::{Duration, SystemTime},
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const ERROR_BODY_LIMIT: usize = 1024 * 1024;
const ERROR_BODY_TIMEOUT: Duration = Duration::from_secs(30);

static NON_RETRYABLE: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)GoUsageLimitError",
        r"(?i)FreeUsageLimitError",
        r"(?i)Monthly usage limit reached",
        r"(?i)available balance",
        r"(?i)insufficient_quota",
        r"(?i)out of budget",
        r"(?i)quota exceeded",
        r"(?i)billing",
    ])
    .expect("valid non-retryable patterns")
});

static RETRYABLE: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)overloaded",
        r"(?i)rate.?limit",
        r"(?i)too many requests",
        r"429",
        r"500",
        r"502",
        r"503",
        r"504",
        r"524",
        r"(?i)service.?unavailable",
        r"(?i)server.?error",
        r"(?i)internal.?error",
        r"(?i)provider.?returned.?error",
        r"(?i)exceeded request buffer limit while retrying upstream",
        r"(?i)network.?error",
        r"(?i)connection.?error",
        r"(?i)connection.?refused",
        r"(?i)connection.?lost",
        r"(?i)other side closed",
        r"(?i)fetch failed",
        r"(?i)getaddrinfo",
        r"(?i)ENOTFOUND",
        r"(?i)EAI_AGAIN",
        r"(?i)upstream.?connect",
        r"(?i)reset before headers",
        r"(?i)socket hang up",
        r"(?i)socket connection was closed",
        r"(?i)timed? out",
        r"(?i)timeout",
        r"(?i)terminated",
        r"(?i)websocket.?closed",
        r"(?i)websocket.?error",
        r"(?i)ended without",
        r"(?i)stream ended before message_stop",
        r"(?i)stream ended before a terminal response event",
        r"(?i)http2 request did not get a response",
        r"(?i)retry delay",
        r"(?i)you can retry your request",
        r"(?i)try your request again",
        r"(?i)please retry your request",
        r"(?i)ResourceExhausted",
    ])
    .expect("valid retryable patterns")
});

static CODEX_RETRYABLE: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)rate.?limit",
        r"(?i)overloaded",
        r"(?i)service.?unavailable",
        r"(?i)upstream.?connect",
        r"(?i)connection.?refused",
    ])
    .expect("valid Codex retryable patterns")
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetryPolicy {
    pub enabled: bool,
    pub max_retries: usize,
    pub base_delay: Duration,
}

#[async_trait]
pub trait RetryCallbacks: Send + Sync {
    async fn on_retry_scheduled(
        &self,
        _attempt: usize,
        _max_attempts: usize,
        _delay: Duration,
        _error_message: &str,
    ) {
    }

    async fn on_retry_attempt_start(&self) {}

    async fn on_retry_finished(&self, _success: bool, _attempt: usize, _final_error: Option<&str>) {
    }
}

pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let Some(error) = &message.error_message else {
        return false;
    };
    !NON_RETRYABLE.is_match(error) && RETRYABLE.is_match(error)
}

pub async fn retry_assistant_call<F, Fut>(
    mut produce: F,
    policy: Option<&RetryPolicy>,
    cancellation: Option<&CancellationToken>,
    callbacks: Option<&dyn RetryCallbacks>,
) -> AssistantMessage
where
    F: FnMut() -> Fut,
    Fut: Future<Output = AssistantMessage>,
{
    let max_attempts = policy
        .filter(|policy| policy.enabled)
        .map_or(0, |policy| policy.max_retries);
    let mut attempt = 0;
    loop {
        let mut response = produce().await;
        if response.stop_reason == StopReason::Aborted {
            if let Some(callbacks) = callbacks.filter(|_| attempt > 0) {
                callbacks.on_retry_finished(false, attempt, None).await;
            }
            return response;
        }
        if response.stop_reason != StopReason::Error {
            if let Some(callbacks) = callbacks.filter(|_| attempt > 0) {
                callbacks.on_retry_finished(true, attempt, None).await;
            }
            return response;
        }
        if attempt >= max_attempts || !is_retryable_assistant_error(&response) {
            if let Some(callbacks) = callbacks.filter(|_| attempt > 0) {
                callbacks
                    .on_retry_finished(false, attempt, response.error_message.as_deref())
                    .await;
            }
            return response;
        }
        attempt += 1;
        let error = response
            .error_message
            .clone()
            .unwrap_or_else(|| "Unknown error".into());
        let delay = policy
            .map(|policy| exponential_delay(policy.base_delay, attempt))
            .unwrap_or_default();
        if let Some(callbacks) = callbacks {
            callbacks
                .on_retry_scheduled(attempt, max_attempts, delay, &error)
                .await;
        }
        if wait_for_retry(delay, cancellation).await.is_err() {
            if let Some(callbacks) = callbacks {
                callbacks
                    .on_retry_finished(false, attempt, Some(&error))
                    .await;
            }
            response.stop_reason = StopReason::Aborted;
            response.error_message = None;
            return response;
        }
        if let Some(callbacks) = callbacks {
            callbacks.on_retry_attempt_start().await;
        }
    }
}

fn exponential_delay(base: Duration, attempt: usize) -> Duration {
    let exponent = attempt.saturating_sub(1).min(31) as u32;
    base.saturating_mul(1_u32 << exponent)
}

async fn wait_for_retry(
    delay: Duration,
    cancellation: Option<&CancellationToken>,
) -> Result<(), ()> {
    match cancellation {
        Some(cancellation) => tokio::select! {
            _ = tokio::time::sleep(delay) => Ok(()),
            _ = cancellation.cancelled() => Err(()),
        },
        None => {
            tokio::time::sleep(delay).await;
            Ok(())
        }
    }
}

pub(crate) struct Policy<'a> {
    pub max_retries: usize,
    pub max_delay: Option<Duration>,
    pub cancellation: &'a CancellationToken,
    pub deadline: Option<Instant>,
    pub profile: Profile,
    pub request_timeout: Option<Duration>,
}

#[derive(Clone, Copy)]
pub(crate) enum Profile {
    Standard,
    Codex,
}

pub(crate) async fn send<F, Fut, O, OFut>(
    policy: Policy<'_>,
    mut request: F,
    mut observe: O,
) -> Result<Response, Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Response, reqwest::Error>>,
    O: FnMut(crate::ProviderResponse) -> OFut,
    OFut: Future<Output = Result<(), Error>>,
{
    let mut retries = 0;
    loop {
        let requested = tokio::select! {
            biased;
            _ = policy.cancellation.cancelled() => {
                return Err(Error::Cancelled { partial: None });
            }
            _ = wait_for(policy.request_timeout) => Err(RequestFailure::Timeout),
            response = request() => response.map_err(RequestFailure::Http),
        };
        let response = match requested {
            Ok(response) => response,
            Err(_) if retries < policy.max_retries => {
                let delay = backoff(policy.profile, retries);
                retries += 1;
                wait(delay, policy.cancellation).await?;
                continue;
            }
            Err(RequestFailure::Http(error)) => return Err(Error::Http(error.to_string())),
            Err(RequestFailure::Timeout) => {
                return Err(Error::Timeout {
                    phase: crate::TimeoutPhase::Connection,
                    partial: None,
                });
            }
        };
        observe(crate::http::provider_response(&response)).await?;
        if response.status().is_success() {
            return Ok(response);
        }
        let (response, retryable) = match policy.profile {
            Profile::Standard => {
                let retryable = is_retryable(&response);
                (response, retryable)
            }
            Profile::Codex => {
                match buffer_error_response(response, policy.cancellation, policy.deadline).await {
                    Ok((response, body)) => {
                        let retryable = is_retryable_codex(response.status().as_u16(), &body);
                        (response, retryable)
                    }
                    Err(Error::Http(_)) if retries < policy.max_retries => {
                        let delay = backoff(policy.profile, retries);
                        retries += 1;
                        wait(delay, policy.cancellation).await?;
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }
        };
        if retries >= policy.max_retries || !retryable {
            return Ok(response);
        }

        let requested = requested_delay(response.headers());
        let delay = requested.unwrap_or_else(|| backoff(policy.profile, retries));
        retries += 1;
        if let Some(maximum) = policy.max_delay
            && !maximum.is_zero()
            && requested.is_some()
            && delay > maximum
        {
            return Err(Error::RetryDelayExceeded {
                requested: delay,
                maximum,
            });
        }
        wait(delay, policy.cancellation).await?;
    }
}

enum RequestFailure {
    Http(reqwest::Error),
    Timeout,
}

fn is_retryable(response: &Response) -> bool {
    match response
        .headers()
        .get("x-should-retry")
        .and_then(|value| value.to_str().ok())
    {
        Some("true") => true,
        Some("false") => false,
        _ => matches!(response.status().as_u16(), 408 | 409 | 429 | 500..=599),
    }
}

fn is_retryable_codex(status: u16, body: &str) -> bool {
    if status == 429 && NON_RETRYABLE.is_match(body) {
        return false;
    }
    matches!(status, 429 | 500 | 502 | 503 | 504) || CODEX_RETRYABLE.is_match(body)
}

pub(crate) fn requested_delay(headers: &HeaderMap) -> Option<Duration> {
    if let Some(delay) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| parse_duration(value, 0.001))
    {
        return Some(delay);
    }
    let value = headers
        .get("retry-after")
        .and_then(|value| value.to_str().ok())?;
    if let Some(delay) = parse_duration(value, 1.0) {
        return Some(delay);
    }
    httpdate::parse_http_date(value)
        .ok()
        .and_then(|time| time.duration_since(SystemTime::now()).ok())
}

fn parse_duration(value: &str, seconds_per_unit: f64) -> Option<Duration> {
    let value = value.parse::<f64>().ok()?;
    Duration::try_from_secs_f64((value * seconds_per_unit).max(0.0)).ok()
}

fn backoff(profile: Profile, retry_index: usize) -> Duration {
    match profile {
        Profile::Standard => {
            let base_seconds = (0.5 * 2_f64.powi(retry_index as i32)).min(8.0);
            Duration::from_secs_f64(base_seconds * (1.0 - rand::random::<f64>() * 0.25))
        }
        Profile::Codex => Duration::from_secs(1_u64 << retry_index.min(31)),
    }
}

async fn buffer_error_response(
    response: Response,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<(Response, String), Error> {
    let status = response.status();
    let version = response.version();
    let headers = response.headers().clone();
    let mut chunks = response.bytes_stream();
    let mut body = BytesMut::new();
    let timeout = Instant::now() + ERROR_BODY_TIMEOUT;
    let deadline = deadline.map_or(timeout, |deadline| deadline.min(timeout));
    while body.len() < ERROR_BODY_LIMIT {
        let chunk = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled { partial: None }),
            _ = tokio::time::sleep_until(deadline) => {
                return Err(Error::Timeout {
                    phase: crate::TimeoutPhase::Overall,
                    partial: None,
                });
            }
            chunk = chunks.next() => chunk,
        };
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|error| Error::Http(error.to_string()))?;
        let remaining = ERROR_BODY_LIMIT - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    let text = String::from_utf8_lossy(&body).into_owned();
    let mut rebuilt = ::http::Response::builder()
        .status(status)
        .version(version)
        .body(body.freeze())
        .expect("valid provider response");
    *rebuilt.headers_mut() = headers;
    Ok((Response::from(rebuilt), text))
}

async fn wait(delay: Duration, cancellation: &CancellationToken) -> Result<(), Error> {
    if delay.is_zero() {
        return Ok(());
    }
    tokio::select! {
        _ = tokio::time::sleep(delay) => Ok(()),
        _ = cancellation.cancelled() => Err(Error::Cancelled { partial: None }),
    }
}

async fn wait_for(timeout: Option<Duration>) {
    match timeout.filter(|timeout| !timeout.is_zero()) {
        Some(timeout) => tokio::time::sleep(timeout).await,
        None => pending().await,
    }
}
