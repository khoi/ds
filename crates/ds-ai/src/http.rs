use crate::{Error, ProviderResponse, TimeoutPhase, transport};
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

pub(crate) async fn provider_error(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Error {
    provider_error_with(response, cancellation, overall_deadline, false).await
}

pub(crate) async fn codex_provider_error(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
) -> Error {
    provider_error_with(response, cancellation, overall_deadline, true).await
}

async fn provider_error_with(
    response: Response,
    cancellation: &CancellationToken,
    overall_deadline: Option<Instant>,
    codex: bool,
) -> Error {
    let status = response.status().as_u16();
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
    let friendly_message = codex
        .then(|| codex_usage_limit_message(status, parsed.as_ref()))
        .flatten();
    let message = friendly_message
        .or_else(|| {
            parsed
                .as_ref()
                .and_then(|body| body.pointer("/error/message"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
                .filter(|message| !message.is_empty())
        })
        .unwrap_or_else(|| {
            if body.is_empty() {
                format!("provider returned HTTP {status}")
            } else {
                body
            }
        });
    Error::Provider { status, message }
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
