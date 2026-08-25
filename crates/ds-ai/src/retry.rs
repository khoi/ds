use crate::{AssistantMessage, Error, StopReason};
use async_trait::async_trait;
use regex::RegexSet;
use reqwest::{Response, header::HeaderMap};
use std::{
    future::{Future, pending},
    sync::LazyLock,
    time::{Duration, SystemTime},
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
pub(crate) const NO_BODY_TIMEOUT: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);

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
        r"(?i)stream ended before a terminal event",
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
    if cancellation.is_some_and(|cancellation| cancellation.is_cancelled()) {
        return Err(());
    }
    match cancellation {
        Some(cancellation) => tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(()),
            _ = tokio::time::sleep(delay) => Ok(()),
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
    Anthropic,
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
        if matches!(policy.profile, Profile::Codex)
            && let Err(error) = observe(crate::http::provider_response(&response)).await
        {
            if retries < policy.max_retries && should_retry_codex_observation(&error) {
                let delay = backoff(policy.profile, retries);
                retries += 1;
                wait(delay, policy.cancellation).await?;
                continue;
            }
            return Err(error);
        }
        if response.status().is_success() {
            if !matches!(policy.profile, Profile::Codex) {
                observe(crate::http::provider_response(&response)).await?;
            }
            return Ok(response);
        }
        let (response, retryable) = match policy.profile {
            Profile::Standard | Profile::Anthropic => {
                let retryable = is_retryable(&response);
                (response, retryable)
            }
            Profile::Codex => {
                match crate::http::buffer_error_response(
                    response,
                    policy.cancellation,
                    policy.deadline,
                )
                .await
                {
                    Ok(buffered) => {
                        let retryable =
                            is_retryable_codex(buffered.response.status().as_u16(), &buffered.body);
                        (buffered.response, retryable)
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

        let requested = requested_delay(policy.profile, response.headers());
        let delay = requested.unwrap_or_else(|| backoff(policy.profile, retries));
        retries += 1;
        let maximum = policy.max_delay.unwrap_or(DEFAULT_MAX_RETRY_DELAY);
        if !maximum.is_zero() && requested.is_some() && delay > maximum {
            return Err(match policy.profile {
                Profile::Standard => {
                    let provider_error = crate::http::openai_retry_error_message(
                        response,
                        policy.cancellation,
                        policy.deadline,
                    )
                    .await?;
                    Error::RetryDelayExceededWithProvider {
                        requested: crate::RetryDelay(delay),
                        maximum: crate::RetryDelay(maximum),
                        provider_error,
                    }
                }
                Profile::Anthropic => {
                    let provider_error = crate::http::anthropic_retry_error_message(
                        response,
                        policy.cancellation,
                        policy.deadline,
                    )
                    .await?;
                    Error::RetryDelayExceededWithProvider {
                        requested: crate::RetryDelay(delay),
                        maximum: crate::RetryDelay(maximum),
                        provider_error,
                    }
                }
                Profile::Codex => Error::RetryDelayExceeded {
                    requested: crate::RetryDelay(delay),
                    maximum: crate::RetryDelay(maximum),
                },
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
        _ => {
            matches!(response.status().as_u16(), 408 | 409 | 429)
                || response.status().as_u16() >= 500
        }
    }
}

fn should_retry_codex_observation(error: &Error) -> bool {
    !matches!(
        error,
        Error::Hook { message, .. }
            if message == "Request was aborted" || message.contains("usage limit")
    )
}

fn is_retryable_codex(status: u16, body: &str) -> bool {
    if status == 429 && NON_RETRYABLE.is_match(body) {
        return false;
    }
    matches!(status, 429 | 500 | 502 | 503 | 504) || CODEX_RETRYABLE.is_match(body)
}

pub(crate) fn requested_delay(profile: Profile, headers: &HeaderMap) -> Option<Duration> {
    requested_delay_at(profile, headers, SystemTime::now())
}

fn requested_delay_at(profile: Profile, headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    match profile {
        Profile::Standard | Profile::Anthropic => requested_delay_sdk(headers, now),
        Profile::Codex => requested_delay_codex(headers, now),
    }
}

fn requested_delay_sdk(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    if let Some(value) = header_value(headers, "retry-after-ms")
        && !value.is_empty()
        && let Some(millis) = parse_float_prefix(value)
    {
        return Some(nonnegative_duration(millis, 0.001));
    }

    let value = header_value(headers, "retry-after")?;
    if value.is_empty() {
        return None;
    }
    if let Some(seconds) = parse_float_prefix(value) {
        return Some(nonnegative_duration(seconds, 1.0));
    }
    date_delay(value, now)
}

fn requested_delay_codex(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    if let Some(value) = header_value(headers, "retry-after-ms")
        && let Some(millis) = parse_number(value).filter(|value| value.is_finite())
    {
        return Some(nonnegative_duration(millis, 0.001));
    }

    let value = header_value(headers, "retry-after")?;
    if value.is_empty() {
        return None;
    }
    if let Some(seconds) = parse_number(value).filter(|value| value.is_finite()) {
        return Some(nonnegative_duration(seconds, 1.0));
    }
    date_delay(value, now)
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

fn date_delay(value: &str, now: SystemTime) -> Option<Duration> {
    let time = httpdate::parse_http_date(value).ok()?;
    Some(time.duration_since(now).unwrap_or_default())
}

fn parse_float_prefix(value: &str) -> Option<f64> {
    let value = value.trim_start();
    let mut end = 0;
    let bytes = value.as_bytes();
    if matches!(bytes.first(), Some(b'+' | b'-')) {
        end = 1;
    }
    if value[end..].starts_with("Infinity") {
        return value[..end + "Infinity".len()].parse().ok();
    }

    let mut digits_before = 0;
    while bytes
        .get(end + digits_before)
        .is_some_and(u8::is_ascii_digit)
    {
        digits_before += 1;
    }
    let mut digits_after = 0;
    let has_decimal = bytes.get(end + digits_before) == Some(&b'.');
    if has_decimal {
        while bytes
            .get(end + digits_before + 1 + digits_after)
            .is_some_and(u8::is_ascii_digit)
        {
            digits_after += 1;
        }
    }
    if digits_before == 0 && digits_after == 0 {
        return None;
    }

    let mut number_end = end + digits_before + usize::from(has_decimal) + digits_after;
    if bytes
        .get(number_end)
        .is_some_and(|byte| *byte == b'e' || *byte == b'E')
    {
        let exponent_start = number_end;
        let mut exponent_end = exponent_start + 1;
        if matches!(bytes.get(exponent_end), Some(b'+' | b'-')) {
            exponent_end += 1;
        }
        let exponent_digits = bytes[exponent_end..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if exponent_digits > 0 {
            number_end = exponent_end + exponent_digits;
        }
    }
    value[..number_end].parse().ok()
}

fn parse_number(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty() {
        return Some(0.0);
    }
    if value == "Infinity" || value == "+Infinity" {
        return Some(f64::INFINITY);
    }
    if value == "-Infinity" {
        return Some(f64::NEG_INFINITY);
    }
    if value.starts_with(['+', '-'])
        && (value[1..].starts_with("0x")
            || value[1..].starts_with("0X")
            || value[1..].starts_with("0b")
            || value[1..].starts_with("0B")
            || value[1..].starts_with("0o")
            || value[1..].starts_with("0O"))
    {
        return None;
    }
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0b", 2),
        ("0B", 2),
        ("0o", 8),
        ("0O", 8),
    ] {
        if let Some(digits) = value.strip_prefix(prefix) {
            return (!digits.is_empty())
                .then(|| u64::from_str_radix(digits, radix).ok())
                .flatten()
                .map(|value| value as f64);
        }
    }
    value.parse().ok()
}

fn nonnegative_duration(value: f64, seconds_per_unit: f64) -> Duration {
    if value <= 0.0 {
        return Duration::ZERO;
    }
    if value.is_infinite() {
        return Duration::MAX;
    }
    Duration::try_from_secs_f64(value * seconds_per_unit).unwrap_or(Duration::MAX)
}

fn backoff(profile: Profile, retry_index: usize) -> Duration {
    match profile {
        Profile::Standard | Profile::Anthropic => {
            let base_seconds = (0.5 * 2_f64.powi(retry_index as i32)).min(8.0);
            Duration::from_secs_f64(base_seconds * (1.0 - rand::random::<f64>() * 0.25))
        }
        Profile::Codex => Duration::from_secs(1_u64 << retry_index.min(31)),
    }
}

async fn wait(delay: Duration, cancellation: &CancellationToken) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        return Err(Error::Cancelled { partial: None });
    }
    if delay.is_zero() {
        return Ok(());
    }
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(Error::Cancelled { partial: None }),
        _ = tokio::time::sleep(delay) => Ok(()),
    }
}

async fn wait_for(timeout: Option<Duration>) {
    match timeout {
        Some(timeout) => tokio::time::sleep(timeout).await,
        None => pending().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderName, HeaderValue};

    fn headers(values: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in values {
            headers.insert(
                HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        headers
    }

    #[test]
    fn sdk_profiles_parse_numeric_prefixes() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        for profile in [Profile::Standard, Profile::Anthropic] {
            assert_eq!(
                requested_delay_at(profile, &headers(&[("retry-after-ms", " 1.5seconds")]), now,),
                Some(Duration::from_micros(1500)),
            );
            assert_eq!(
                requested_delay_at(profile, &headers(&[("retry-after", "2seconds")]), now),
                Some(Duration::from_secs(2)),
            );
            assert_eq!(
                requested_delay_at(
                    profile,
                    &headers(&[("retry-after-ms", "invalid"), ("retry-after", "3s")]),
                    now,
                ),
                Some(Duration::from_secs(3)),
            );
        }
    }

    #[test]
    fn sdk_profiles_parse_dates_and_clamp_past_dates() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let future = httpdate::fmt_http_date(now + Duration::from_secs(5));
        let past = httpdate::fmt_http_date(now - Duration::from_secs(5));
        for profile in [Profile::Standard, Profile::Anthropic] {
            assert_eq!(
                requested_delay_at(profile, &headers(&[("retry-after", &future)]), now),
                Some(Duration::from_secs(5)),
            );
            assert_eq!(
                requested_delay_at(profile, &headers(&[("retry-after", &past)]), now),
                Some(Duration::ZERO),
            );
        }
    }

    #[test]
    fn codex_uses_strict_number_parsing() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        assert_eq!(
            requested_delay_at(
                Profile::Codex,
                &headers(&[("retry-after-ms", ""), ("retry-after", "2")]),
                now,
            ),
            Some(Duration::ZERO),
        );
        assert_eq!(
            requested_delay_at(Profile::Codex, &headers(&[("retry-after-ms", "0x10")]), now,),
            Some(Duration::from_millis(16)),
        );
        assert_eq!(
            requested_delay_at(Profile::Codex, &headers(&[("retry-after-ms", "16ms")]), now,),
            None,
        );
        assert_eq!(
            requested_delay_at(
                Profile::Codex,
                &headers(&[("retry-after", "2seconds")]),
                now,
            ),
            None,
        );
    }

    #[test]
    fn codex_dates_clamp_past_dates() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let future = httpdate::fmt_http_date(now + Duration::from_secs(5));
        let past = httpdate::fmt_http_date(now - Duration::from_secs(5));
        assert_eq!(
            requested_delay_at(Profile::Codex, &headers(&[("retry-after", &future)]), now),
            Some(Duration::from_secs(5)),
        );
        assert_eq!(
            requested_delay_at(Profile::Codex, &headers(&[("retry-after", &past)]), now),
            Some(Duration::ZERO),
        );
    }
}
