use crate::{
    AuthCheck, AuthContext, AuthError, AuthResolutionOverrides, AuthResult, Credential,
    CredentialStore, CredentialType, OAuthAuth, Provider, ProviderId,
};
use async_trait::async_trait;
use std::{
    collections::BTreeMap,
    future::Future,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;

pub(crate) async fn resolve_auth(
    provider: Arc<dyn Provider>,
    credentials: Arc<dyn CredentialStore>,
    auth_context: Arc<dyn AuthContext>,
    overrides: AuthResolutionOverrides,
) -> Result<Option<AuthResult>, AuthError> {
    if overrides.cancellation.is_cancelled() {
        return Err(AuthError::Cancelled);
    }
    let context = OverlayAuthContext {
        base: auth_context,
        env: overrides.env.clone(),
    };
    if let (Some(api_key), Some(auth)) = (&overrides.api_key, &provider.auth().api_key) {
        return race_cancellation(
            &overrides.cancellation,
            auth.resolve(
                &context,
                Some(&Credential::ApiKey {
                    key: Some(api_key.clone()),
                    env: overrides.env.clone(),
                }),
                &overrides.cancellation,
            ),
        )
        .await
        .map_err(|error| api_key_error("API key auth failed", provider.id().as_str(), error));
    }
    let stored = credentials.read(provider.id().as_str(), &overrides.cancellation);
    let stored = race_cancellation(&overrides.cancellation, stored)
        .await
        .map_err(|error| {
            store_error(
                "credential store read failed",
                provider.id().as_str(),
                error,
            )
        })?;
    match stored {
        Some(credential @ Credential::OAuth { .. }) => {
            let Some(oauth) = &provider.auth().oauth else {
                return Ok(None);
            };
            resolve_oauth(
                credentials,
                provider.id().clone(),
                oauth.clone(),
                credential,
                overrides,
            )
            .await
        }
        Some(mut credential @ Credential::ApiKey { .. }) => {
            let Some(auth) = &provider.auth().api_key else {
                return Ok(None);
            };
            if let Credential::ApiKey { env, .. } = &mut credential {
                env.extend(overrides.env);
            }
            race_cancellation(
                &overrides.cancellation,
                auth.resolve(&context, Some(&credential), &overrides.cancellation),
            )
            .await
            .map_err(|error| api_key_error("API key auth failed", provider.id().as_str(), error))
        }
        None => match &provider.auth().api_key {
            Some(auth) => race_cancellation(
                &overrides.cancellation,
                auth.resolve(&context, None, &overrides.cancellation),
            )
            .await
            .map_err(|error| api_key_error("API key auth failed", provider.id().as_str(), error)),
            None => Ok(None),
        },
    }
}

async fn resolve_oauth(
    credentials: Arc<dyn CredentialStore>,
    provider_id: ProviderId,
    oauth: Arc<dyn OAuthAuth>,
    stored: Credential,
    overrides: AuthResolutionOverrides,
) -> Result<Option<AuthResult>, AuthError> {
    let minimum = overrides
        .minimum_oauth_validity
        .unwrap_or(Duration::from_secs(5 * 60))
        .max(Duration::from_secs(5 * 60));
    let explicit_minimum = overrides.minimum_oauth_validity.is_some();
    let mut credential = stored;
    if expires_soon(&credential, minimum) {
        let refresh = oauth.clone();
        let cancellation = overrides.cancellation.clone();
        let refresh_provider_id = provider_id.to_string();
        let post = race_cancellation(
            &overrides.cancellation,
            credentials.modify(
                provider_id.as_str(),
                Box::new(move |current| {
                    Box::pin(async move {
                        let Some(current @ Credential::OAuth { .. }) = current else {
                            return Ok(None);
                        };
                        if !expires_soon(&current, minimum) {
                            return Ok(None);
                        }
                        match tokio::time::timeout(
                            Duration::from_secs(15),
                            refresh.refresh(&current, &cancellation),
                        )
                        .await
                        {
                            Ok(result) => {
                                let result = result.map_err(|error| {
                                    oauth_refresh_error(&refresh_provider_id, error)
                                })?;
                                if !matches!(result, Credential::OAuth { .. }) {
                                    return Err(AuthError::OAuth(format!(
                                        "OAuth refresh returned an API key credential for provider {refresh_provider_id}"
                                    )));
                                }
                                Ok(Some(result))
                            }
                            Err(_) => Err(oauth_refresh_error(
                                &refresh_provider_id,
                                AuthError::OAuth("refresh timed out".into()),
                            )),
                        }
                    })
                }),
                &overrides.cancellation,
            ),
        )
        .await
        .map_err(|error| match error {
            AuthError::Cancelled => AuthError::Cancelled,
            AuthError::OAuth(reason) => AuthError::OAuth(reason),
            error => store_error(
                "credential store modify failed",
                provider_id.as_str(),
                error,
            ),
        })?;
        let Some(Credential::OAuth { .. }) = post.as_ref() else {
            return Ok(None);
        };
        credential = post.unwrap();
        if explicit_minimum && expires_soon(&credential, minimum) {
            return Err(AuthError::OAuth(format!(
                "OAuth refresh returned a token that expires too soon for provider {provider_id}"
            )));
        }
    }
    let auth = race_cancellation(&overrides.cancellation, oauth.to_auth(&credential))
        .await
        .map_err(|error| oauth_auth_error(provider_id.as_str(), error))?;
    Ok(Some(AuthResult {
        auth,
        source: Some("OAuth".into()),
        ..Default::default()
    }))
}

fn expires_soon(credential: &Credential, minimum: Duration) -> bool {
    let Credential::OAuth { expires, .. } = credential else {
        return false;
    };
    timestamp().saturating_add(duration_millis(minimum)) >= *expires
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

pub(crate) async fn check_provider_auth(
    provider: &Arc<dyn Provider>,
    credential: Option<&Credential>,
    auth_context: &dyn AuthContext,
    cancellation: &CancellationToken,
) -> Result<Option<AuthCheck>, AuthError> {
    match credential {
        Some(Credential::OAuth { .. }) if provider.auth().oauth.is_some() => Ok(Some(AuthCheck {
            source: Some("OAuth".into()),
            credential_type: CredentialType::OAuth,
        })),
        Some(Credential::ApiKey { .. }) | None => match &provider.auth().api_key {
            Some(auth) => race_cancellation(
                cancellation,
                auth.check(auth_context, credential, cancellation),
            )
            .await
            .map_err(|error| {
                api_key_error("API key auth check failed", provider.id().as_str(), error)
            }),
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

struct OverlayAuthContext {
    base: Arc<dyn AuthContext>,
    env: BTreeMap<String, String>,
}

#[async_trait]
impl AuthContext for OverlayAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        match self.env.get(name).filter(|value| !value.is_empty()) {
            Some(value) => Some(value.clone()),
            None => self.base.env(name).await,
        }
    }

    async fn file_exists(&self, path: &str) -> bool {
        self.base.file_exists(path).await
    }
}

pub(crate) async fn race_cancellation<T, F>(
    cancellation: &CancellationToken,
    future: F,
) -> Result<T, AuthError>
where
    F: Future<Output = Result<T, AuthError>> + Send,
{
    if cancellation.is_cancelled() {
        return Err(AuthError::Cancelled);
    }
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(AuthError::Cancelled),
        result = future => result,
    }
}

fn api_key_error(operation: &str, provider_id: &str, error: AuthError) -> AuthError {
    match error {
        AuthError::Cancelled => AuthError::Cancelled,
        AuthError::Authentication(reason) => {
            AuthError::Authentication(provider_context(operation, provider_id, reason))
        }
        error => {
            AuthError::Authentication(provider_context(operation, provider_id, error.to_string()))
        }
    }
}

pub(crate) fn api_login_error(provider_id: &str, error: AuthError) -> AuthError {
    match error {
        AuthError::Cancelled => AuthError::Cancelled,
        AuthError::Unsupported(reason) => AuthError::Unsupported(reason),
        AuthError::Authentication(reason) => {
            AuthError::Authentication(provider_context("login failed", provider_id, reason))
        }
        error => AuthError::Authentication(provider_context(
            "login failed",
            provider_id,
            error.to_string(),
        )),
    }
}

pub(crate) fn oauth_login_error(provider_id: &str, error: AuthError) -> AuthError {
    match error {
        AuthError::Cancelled => AuthError::Cancelled,
        AuthError::Unsupported(reason) => AuthError::Unsupported(reason),
        AuthError::OAuth(reason) => {
            AuthError::OAuth(provider_context("OAuth login failed", provider_id, reason))
        }
        error => AuthError::OAuth(provider_context(
            "OAuth login failed",
            provider_id,
            error.to_string(),
        )),
    }
}

fn oauth_refresh_error(provider_id: &str, error: AuthError) -> AuthError {
    match error {
        AuthError::Cancelled => AuthError::Cancelled,
        AuthError::OAuth(reason) => AuthError::OAuth(provider_context(
            "OAuth refresh failed",
            provider_id,
            reason,
        )),
        error => AuthError::OAuth(provider_context(
            "OAuth refresh failed",
            provider_id,
            error.to_string(),
        )),
    }
}

fn oauth_auth_error(provider_id: &str, error: AuthError) -> AuthError {
    match error {
        AuthError::Cancelled => AuthError::Cancelled,
        AuthError::OAuth(reason) => AuthError::OAuth(provider_context(
            "OAuth auth derivation failed",
            provider_id,
            reason,
        )),
        error => AuthError::OAuth(provider_context(
            "OAuth auth derivation failed",
            provider_id,
            error.to_string(),
        )),
    }
}

pub(crate) fn store_error(operation: &str, provider_id: &str, error: AuthError) -> AuthError {
    match error {
        AuthError::Cancelled => AuthError::Cancelled,
        AuthError::OAuth(reason) => {
            AuthError::OAuth(provider_context(operation, provider_id, reason))
        }
        AuthError::Store(reason) => {
            AuthError::Store(provider_context(operation, provider_id, reason))
        }
        error => AuthError::Store(provider_context(operation, provider_id, error.to_string())),
    }
}

fn provider_context(operation: &str, provider_id: &str, reason: String) -> String {
    format!("{operation} for provider {provider_id}: {reason}")
}
