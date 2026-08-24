use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_BASE_URL: &str = "https://auth.openai.com";
const MINIMUM_VALIDITY: Duration = Duration::from_secs(5 * 60);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Credentials {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
}

impl Credentials {
    pub fn account_id(&self) -> Result<String, Error> {
        super::account_id(&self.access_token).map_err(Error::InvalidResponse)
    }
}

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base_url: String,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_owned();
        self
    }

    pub async fn refresh(
        &self,
        refresh_token: &str,
        cancellation: &CancellationToken,
    ) -> Result<Credentials, Error> {
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("grant_type", "refresh_token")
            .append_pair("refresh_token", refresh_token)
            .append_pair("client_id", CLIENT_ID)
            .finish();
        let response = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            response = self
                .http
                .post(format!("{}/oauth/token", self.base_url))
                .header("content-type", "application/x-www-form-urlencoded")
                .body(body)
                .send() => response.map_err(|error| Error::Http(error.to_string()))?,
        };
        read_credentials(response, "refresh").await
    }
}

#[derive(Clone)]
pub struct CredentialManager {
    client: Client,
    credentials: Arc<Mutex<Credentials>>,
}

impl CredentialManager {
    pub fn new(client: Client, credentials: Credentials) -> Self {
        Self {
            client,
            credentials: Arc::new(Mutex::new(credentials)),
        }
    }

    pub async fn access_token(&self, cancellation: &CancellationToken) -> Result<String, Error> {
        let mut credentials = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            credentials = self.credentials.lock() => credentials,
        };
        let minimum_expiry = now_millis().saturating_add(duration_millis(MINIMUM_VALIDITY));
        if credentials.expires_at <= minimum_expiry {
            *credentials = self
                .client
                .refresh(&credentials.refresh_token, cancellation)
                .await?;
        }
        Ok(credentials.access_token.clone())
    }

    pub async fn snapshot(&self) -> Credentials {
        self.credentials.lock().await.clone()
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("OAuth request failed: {0}")]
    Http(String),
    #[error("OAuth {operation} failed with HTTP {status}: {body}")]
    Server {
        operation: &'static str,
        status: u16,
        body: String,
    },
    #[error("invalid OAuth response: {0}")]
    InvalidResponse(String),
    #[error("OAuth operation cancelled")]
    Cancelled,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

async fn read_credentials(
    response: reqwest::Response,
    operation: &'static str,
) -> Result<Credentials, Error> {
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| Error::Http(error.to_string()))?;
    if !status.is_success() {
        return Err(Error::Server {
            operation,
            status: status.as_u16(),
            body,
        });
    }
    let response: TokenResponse =
        serde_json::from_str(&body).map_err(|error| Error::InvalidResponse(error.to_string()))?;
    let access_token = response
        .access_token
        .ok_or_else(|| Error::InvalidResponse("missing access_token".into()))?;
    let refresh_token = response
        .refresh_token
        .ok_or_else(|| Error::InvalidResponse("missing refresh_token".into()))?;
    let expires_in = response
        .expires_in
        .ok_or_else(|| Error::InvalidResponse("missing expires_in".into()))?;
    let credentials = Credentials {
        access_token,
        refresh_token,
        expires_at: now_millis().saturating_add(expires_in.saturating_mul(1000)),
    };
    credentials.account_id()?;
    Ok(credentials)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}
