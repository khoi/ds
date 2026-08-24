use crate::Error;
use reqwest::{Response, header::HeaderMap};
use std::{
    future::Future,
    time::{Duration, SystemTime},
};
use tokio_util::sync::CancellationToken;

pub(crate) struct Policy<'a> {
    pub max_retries: usize,
    pub max_delay: Option<Duration>,
    pub cancellation: &'a CancellationToken,
}

pub(crate) async fn send<F, Fut>(policy: Policy<'_>, mut request: F) -> Result<Response, Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Response, reqwest::Error>>,
{
    let mut retries = 0;
    loop {
        let response = match request().await {
            Ok(response) => response,
            Err(_) if retries < policy.max_retries => {
                let delay = backoff(retries);
                retries += 1;
                wait(delay, policy.cancellation).await?;
                continue;
            }
            Err(error) => return Err(Error::Http(error.to_string())),
        };
        if response.status().is_success() {
            return Ok(response);
        }
        if retries >= policy.max_retries || !is_retryable(&response) {
            return Ok(response);
        }

        let delay = delay(response.headers(), retries);
        retries += 1;
        if let Some(maximum) = policy.max_delay
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

fn delay(headers: &HeaderMap, retry_index: usize) -> Duration {
    if let Some(delay) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| parse_duration(value, 0.001))
    {
        return delay;
    }
    let Some(value) = headers
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
    else {
        return backoff(retry_index);
    };
    if let Some(delay) = parse_duration(value, 1.0) {
        return delay;
    }
    httpdate::parse_http_date(value)
        .ok()
        .and_then(|time| time.duration_since(SystemTime::now()).ok())
        .unwrap_or_else(|| backoff(retry_index))
}

fn parse_duration(value: &str, seconds_per_unit: f64) -> Option<Duration> {
    let value = value.parse::<f64>().ok()?;
    Duration::try_from_secs_f64(value * seconds_per_unit).ok()
}

fn backoff(retry_index: usize) -> Duration {
    let base_seconds = (0.5 * 2_f64.powi(retry_index as i32)).min(8.0);
    Duration::from_secs_f64(base_seconds * (1.0 - rand::random::<f64>() * 0.25))
}

async fn wait(delay: Duration, cancellation: &CancellationToken) -> Result<(), Error> {
    if delay.is_zero() {
        return Ok(());
    }
    tokio::select! {
        _ = tokio::time::sleep(delay) => Ok(()),
        _ = cancellation.cancelled() => Err(Error::Cancelled),
    }
}
