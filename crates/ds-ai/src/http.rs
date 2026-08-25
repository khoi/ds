use crate::{Error, ProviderResponse, TimeoutPhase};
use bytes::BytesMut;
use futures_util::StreamExt;
use reqwest::{
    Response,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const MAX_PROVIDER_ERROR_BODY_BYTES: usize = 1024 * 1024;
const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4_000;
const PROVIDER_ERROR_BODY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

pub(crate) struct BufferedErrorResponse {
    pub response: Response,
    pub body: String,
}

pub(crate) async fn openai_provider_error(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Error {
    provider_error_with(
        response,
        cancellation,
        overall_deadline,
        ProviderErrorFormat::OpenAi,
    )
    .await
}

pub(crate) async fn openai_retry_error_message(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Result<String, Error> {
    retry_error_message(
        response,
        cancellation,
        overall_deadline,
        RetryErrorFormat::OpenAi,
    )
    .await
}

pub(crate) async fn anthropic_provider_error(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Error {
    provider_error_with(
        response,
        cancellation,
        overall_deadline,
        ProviderErrorFormat::Anthropic,
    )
    .await
}

pub(crate) async fn anthropic_retry_error_message(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Result<String, Error> {
    retry_error_message(
        response,
        cancellation,
        overall_deadline,
        RetryErrorFormat::Anthropic,
    )
    .await
}

pub(crate) async fn codex_provider_error(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Error {
    provider_error_with(
        response,
        cancellation,
        overall_deadline,
        ProviderErrorFormat::Codex,
    )
    .await
}

#[derive(Clone, Copy)]
enum ProviderErrorFormat {
    OpenAi,
    Anthropic,
    Codex,
}

#[derive(Clone, Copy)]
enum RetryErrorFormat {
    OpenAi,
    Anthropic,
}

async fn provider_error_with(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
    format: ProviderErrorFormat,
) -> Error {
    let status = response.status().as_u16();
    let status_text = response
        .status()
        .canonical_reason()
        .unwrap_or("Request failed");
    let body = match read_error_body(response, cancellation, overall_deadline).await {
        Ok(body) => body,
        Err(error) => return error,
    };
    let parsed = serde_json::from_str::<serde_json::Value>(&body).ok();
    let message = match format {
        ProviderErrorFormat::OpenAi => openai_body_message(status, &body, parsed.as_ref()),
        ProviderErrorFormat::Anthropic => {
            Some(anthropic_body_message(status, &body, parsed.as_ref()))
        }
        ProviderErrorFormat::Codex => Some(codex_body_message(
            status,
            status_text,
            &body,
            parsed.as_ref(),
        )),
    };
    match message {
        Some(message) => Error::Provider {
            status,
            message: truncate_provider_error(&message),
        },
        None => Error::EmptyProviderResponse { status },
    }
}

async fn retry_error_message(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
    format: RetryErrorFormat,
) -> Result<String, Error> {
    let status = response.status().as_u16();
    let body = read_error_body(response, cancellation, overall_deadline).await?;
    let parsed = serde_json::from_str::<serde_json::Value>(&body).ok();
    Ok(match format {
        RetryErrorFormat::OpenAi => openai_retry_message(status, &body, parsed.as_ref()),
        RetryErrorFormat::Anthropic => anthropic_retry_message(status, &body, parsed.as_ref()),
    })
}

async fn read_error_body(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Result<String, Error> {
    Ok(
        buffer_error_response(response, cancellation, overall_deadline)
            .await?
            .body,
    )
}

pub(crate) async fn buffer_error_response(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Result<BufferedErrorResponse, Error> {
    let status = response.status();
    let version = response.version();
    let headers = response.headers().clone();
    let mut chunks = response.bytes_stream();
    let mut body = BytesMut::new();
    let safety_deadline = Instant::now() + PROVIDER_ERROR_BODY_TIMEOUT;
    let deadline =
        overall_deadline.map_or(safety_deadline, |deadline| deadline.min(safety_deadline));
    while body.len() < MAX_PROVIDER_ERROR_BODY_BYTES {
        let chunk = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled { partial: None }),
            _ = tokio::time::sleep_until(deadline) => {
                return Err(Error::Timeout {
                    phase: TimeoutPhase::Overall,
                    partial: None,
                });
            }
            chunk = chunks.next() => chunk,
        };
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|error| Error::Http(error.to_string()))?;
        let remaining = MAX_PROVIDER_ERROR_BODY_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    let body = body.freeze();
    let text = String::from_utf8_lossy(&body).into_owned();
    let mut rebuilt = ::http::Response::builder()
        .status(status)
        .version(version)
        .body(body)
        .expect("valid provider response");
    *rebuilt.headers_mut() = headers;
    Ok(BufferedErrorResponse {
        response: Response::from(rebuilt),
        body: text,
    })
}

fn openai_body_message(
    status: u16,
    body: &str,
    parsed: Option<&serde_json::Value>,
) -> Option<String> {
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    let Some(parsed) = parsed else {
        return Some(format!("{status} {body}"));
    };
    let error = parsed.get("error")?;
    if !json_truthy(error) {
        return None;
    }
    let serialized = serde_json::to_string(error).ok()?;
    match error {
        serde_json::Value::Object(object) if !object.is_empty() => Some(serialized),
        _ => Some(format!("{status} {serialized}")),
    }
}

fn openai_retry_message(status: u16, body: &str, parsed: Option<&serde_json::Value>) -> String {
    let message = parsed
        .and_then(|response| response.get("error"))
        .and_then(|error| {
            error
                .get("message")
                .filter(|message| json_truthy(message))
                .map(|message| match message {
                    serde_json::Value::String(message) => message.clone(),
                    message => serde_json::to_string(message).unwrap_or_default(),
                })
                .or_else(|| {
                    json_truthy(error).then(|| serde_json::to_string(error).unwrap_or_default())
                })
        })
        .or_else(|| parsed.is_none().then(|| body.trim().to_owned()))
        .filter(|message| !message.is_empty());
    message.map_or_else(
        || format!("{status} status code (no body)"),
        |message| format!("{status} {message}"),
    )
}

fn anthropic_body_message(status: u16, body: &str, parsed: Option<&serde_json::Value>) -> String {
    let body = body.trim();
    if body.is_empty() {
        return format!("{status} status code (no body)");
    };
    let detail = match parsed {
        Some(parsed) => parsed
            .get("message")
            .filter(|message| json_truthy(message))
            .map(|message| match message {
                serde_json::Value::String(message) => message.clone(),
                message => serde_json::to_string(message).unwrap_or_else(|_| body.to_owned()),
            })
            .unwrap_or_else(|| serde_json::to_string(parsed).unwrap_or_else(|_| body.to_owned())),
        None => body.to_owned(),
    };
    format!("{status} {detail}")
}

fn anthropic_retry_message(status: u16, body: &str, parsed: Option<&serde_json::Value>) -> String {
    let message = parsed
        .and_then(|response| response.get("message"))
        .filter(|message| json_truthy(message))
        .map(|message| match message {
            serde_json::Value::String(message) => message.clone(),
            message => serde_json::to_string(message).unwrap_or_default(),
        })
        .or_else(|| {
            parsed
                .filter(|response| json_truthy(response))
                .map(|response| serde_json::to_string(response).unwrap_or_default())
        })
        .or_else(|| {
            if parsed.is_some() {
                return None;
            }
            let body = body.trim();
            (!body.is_empty()).then(|| body.to_owned())
        })
        .filter(|message| !message.is_empty());
    message.map_or_else(
        || format!("{status} status code (no body)"),
        |message| format!("{status} {message}"),
    )
}

fn codex_body_message(
    status: u16,
    status_text: &str,
    body: &str,
    parsed: Option<&serde_json::Value>,
) -> String {
    if let Some(message) = codex_usage_limit_message(status, parsed) {
        return message;
    }
    if let Some(message) = parsed
        .and_then(|body| body.pointer("/error/message"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|message| !message.is_empty())
    {
        return message.to_owned();
    }
    match body.trim() {
        "" => status_text.to_owned(),
        body => body.to_owned(),
    }
}

pub(crate) fn json_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        serde_json::Value::String(value) => !value.is_empty(),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => true,
    }
}

fn truncate_provider_error(message: &str) -> String {
    let mut characters = message.chars();
    let prefix = characters
        .by_ref()
        .take(MAX_PROVIDER_ERROR_BODY_CHARS)
        .collect::<String>();
    if characters.next().is_none() {
        return prefix;
    }
    let truncated = message
        .chars()
        .count()
        .saturating_sub(MAX_PROVIDER_ERROR_BODY_CHARS);
    format!("{prefix}... [truncated {truncated} chars]")
}

fn codex_usage_limit_message(status: u16, body: Option<&serde_json::Value>) -> Option<String> {
    let error = body?.pointer("/error")?;
    let code = error
        .get("code")
        .or_else(|| error.get("type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let code = code.to_ascii_lowercase();
    if status != 429
        && ![
            "usage_limit_reached",
            "usage_not_included",
            "rate_limit_exceeded",
        ]
        .iter()
        .any(|candidate| code.contains(candidate))
    {
        return None;
    }
    let plan = error
        .get("plan_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_ascii_lowercase)
        .map(|plan| format!(" ({plan} plan)"))
        .unwrap_or_default();
    let when = error
        .get("resets_at")
        .and_then(serde_json::Value::as_f64)
        .map(|reset| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            let minutes = ((reset - now) / 60.0).round().max(0.0) as u64;
            format!(" Try again in ~{minutes} min.")
        })
        .unwrap_or_default();
    Some(format!(
        "You have hit your ChatGPT usage limit{plan}.{when}"
    ))
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
    let mut headers = BTreeMap::new();
    for (name, value) in response.headers() {
        let Ok(value) = value.to_str() else {
            continue;
        };
        let entry = headers
            .entry(name.as_str().to_owned())
            .or_insert_with(String::new);
        if !entry.is_empty() {
            entry.push_str(", ");
        }
        entry.push_str(value);
    }
    ProviderResponse {
        status: response.status().as_u16(),
        headers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn caps_a_long_codex_usage_message_once() {
        let body = json!({
            "error": {
                "code": "usage_limit_reached",
                "plan_type": "x".repeat(MAX_PROVIDER_ERROR_BODY_CHARS * 2),
            }
        });
        let message = codex_usage_limit_message(429, Some(&body)).unwrap();
        let truncated = truncate_provider_error(&message);
        let suffix = format!(
            "... [truncated {} chars]",
            message.chars().count() - MAX_PROVIDER_ERROR_BODY_CHARS
        );
        let prefix = message
            .chars()
            .take(MAX_PROVIDER_ERROR_BODY_CHARS)
            .collect::<String>();
        assert_eq!(truncated, format!("{prefix}{suffix}"));
        assert_eq!(truncated.matches("... [truncated").count(), 1);
    }

    #[test]
    fn serializes_the_inner_structured_provider_error() {
        let body = json!({
            "error": {
                "code": "rate_limit_exceeded",
                "message": "slow down",
                "metadata": {"raw": "upstream WAF blocked policy XYZ"},
            }
        });

        assert_eq!(
            openai_body_message(400, &body.to_string(), Some(&body)),
            Some(
                r#"{"code":"rate_limit_exceeded","message":"slow down","metadata":{"raw":"upstream WAF blocked policy XYZ"}}"#.into()
            )
        );
    }

    #[test]
    fn matches_openai_error_shapes() {
        let nested = json!({"error": {"message": "Too many requests"}});
        let top_level = json!({"message": "Too many requests"});
        let empty = json!({"error": {}});

        assert_eq!(
            openai_body_message(429, &nested.to_string(), Some(&nested)),
            Some(r#"{"message":"Too many requests"}"#.into())
        );
        assert_eq!(
            openai_body_message(429, &top_level.to_string(), Some(&top_level)),
            None
        );
        assert_eq!(
            openai_body_message(429, &empty.to_string(), Some(&empty)),
            Some("429 {}".into())
        );
        assert_eq!(
            openai_body_message(429, "failed", None),
            Some("429 failed".into())
        );
    }

    #[test]
    fn matches_anthropic_error_shapes() {
        let nested = json!({"error": {"message": "Too many requests"}});
        let top_level = json!({"message": "Too many requests"});

        assert_eq!(
            anthropic_body_message(429, &nested.to_string(), Some(&nested)),
            r#"429 {"error":{"message":"Too many requests"}}"#
        );
        assert_eq!(
            anthropic_body_message(429, &top_level.to_string(), Some(&top_level)),
            "429 Too many requests"
        );
        assert_eq!(anthropic_body_message(429, "failed", None), "429 failed");
    }

    #[test]
    fn matches_anthropic_sdk_retry_falsy_bodies() {
        for body in [json!(null), json!(false), json!(0), json!("")] {
            assert_eq!(
                anthropic_retry_message(429, &body.to_string(), Some(&body)),
                "429 status code (no body)"
            );
        }
        let body = json!({"type": "error", "error": {"message": "retry"}});
        assert_eq!(
            anthropic_retry_message(429, &body.to_string(), Some(&body)),
            r#"429 {"type":"error","error":{"message":"retry"}}"#
        );
    }

    #[test]
    fn matches_codex_error_shapes() {
        let nested = json!({"error": {"code": "server_error", "message": "failed"}});
        let top_level = json!({"message": "failed"});

        assert_eq!(
            codex_body_message(
                500,
                "Internal Server Error",
                &nested.to_string(),
                Some(&nested)
            ),
            "failed"
        );
        assert_eq!(
            codex_body_message(
                500,
                "Internal Server Error",
                &top_level.to_string(),
                Some(&top_level)
            ),
            r#"{"message":"failed"}"#
        );
        assert_eq!(
            codex_body_message(500, "Internal Server Error", "", None),
            "Internal Server Error"
        );
    }
}
