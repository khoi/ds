use crate::{
    Error, ProviderResponse, RateLimits, ResponseMetadata, TimeoutPhase, retry, transport,
};
use reqwest::{
    Response,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use std::collections::BTreeMap;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

pub(crate) fn metadata(headers: &HeaderMap) -> ResponseMetadata {
    ResponseMetadata {
        request_id: header(headers, "x-request-id").or_else(|| header(headers, "request-id")),
        rate_limits: RateLimits {
            limit_requests: header_u64(headers, "x-ratelimit-limit-requests"),
            remaining_requests: header_u64(headers, "x-ratelimit-remaining-requests"),
            reset_requests: header(headers, "x-ratelimit-reset-requests"),
            limit_tokens: header_u64(headers, "x-ratelimit-limit-tokens"),
            remaining_tokens: header_u64(headers, "x-ratelimit-remaining-tokens"),
            reset_tokens: header(headers, "x-ratelimit-reset-tokens"),
        },
    }
}

pub(crate) async fn provider_error(
    response: Response,
    response_metadata: ResponseMetadata,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Error {
    let status = response.status().as_u16();
    let retry_after = retry::requested_delay(response.headers());
    let body = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Error::Cancelled { partial: None },
        _ = transport::wait_until(overall_deadline) => {
            return Error::Timeout {
                phase: TimeoutPhase::Overall,
                partial: None,
            };
        }
        body = response.text() => body.unwrap_or_default(),
    };
    let parsed = serde_json::from_str::<serde_json::Value>(&body).ok();
    let code = parsed
        .as_ref()
        .and_then(|body| {
            body.pointer("/error/code")
                .or_else(|| body.pointer("/error/type"))
        })
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let message = parsed
        .as_ref()
        .and_then(|body| body.pointer("/error/message"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| {
            if body.is_empty() {
                format!("provider returned HTTP {status}")
            } else {
                body
            }
        });
    Error::Provider {
        status,
        code,
        message,
        request_id: response_metadata.request_id,
        retry_after,
        rate_limits: response_metadata.rate_limits,
    }
}

pub(crate) fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

pub(crate) fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    header(headers, name).and_then(|value| value.parse().ok())
}

pub(crate) fn request_headers(
    mut defaults: BTreeMap<String, String>,
    overrides: &BTreeMap<String, Option<String>>,
) -> Result<HeaderMap, String> {
    for (name, value) in overrides {
        if let Some(existing) = defaults
            .keys()
            .find(|existing| existing.eq_ignore_ascii_case(name))
            .cloned()
        {
            defaults.remove(&existing);
        }
        if let Some(value) = value {
            defaults.insert(name.clone(), value.clone());
        }
    }
    defaults
        .into_iter()
        .map(|(name, value)| {
            let name =
                HeaderName::from_bytes(name.as_bytes()).map_err(|error| error.to_string())?;
            let value = HeaderValue::from_str(&value).map_err(|error| error.to_string())?;
            Ok((name, value))
        })
        .collect()
}

pub(crate) fn provider_response(response: &Response) -> ProviderResponse {
    ProviderResponse {
        status: response.status().as_u16(),
        headers: response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_owned(), value.to_owned()))
            })
            .collect(),
    }
}
