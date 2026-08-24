use crate::{Error, RateLimits, ResponseMetadata, retry};
use reqwest::{Response, header::HeaderMap};

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

pub(crate) async fn provider_error(response: Response) -> Error {
    let status = response.status().as_u16();
    let response_metadata = metadata(response.headers());
    let retry_after = retry::requested_delay(response.headers());
    let body = response.text().await.unwrap_or_default();
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

fn header(headers: &HeaderMap, name: &'static str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn header_u64(headers: &HeaderMap, name: &'static str) -> Option<u64> {
    header(headers, name).and_then(|value| value.parse().ok())
}
