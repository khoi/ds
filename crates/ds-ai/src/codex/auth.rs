use async_trait::async_trait;
use base64::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Mutex,
};
use tokio_util::sync::CancellationToken;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_BASE_URL: &str = "https://auth.openai.com";
const MINIMUM_VALIDITY: Duration = Duration::from_secs(5 * 60);
const SCOPE: &str = "openid profile email offline_access";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Notification {
    AuthorizationUrl { url: String },
}

#[async_trait]
pub trait Interaction: Send + Sync {
    fn notify(&self, notification: Notification);

    async fn manual_authorization(&self, cancellation: CancellationToken) -> Result<String, Error>;
}

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
    callback_address: SocketAddr,
    callback_host: String,
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
            callback_address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 1455),
            callback_host: "localhost".into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_owned();
        self
    }

    pub fn with_callback_address(mut self, address: SocketAddr) -> Self {
        self.callback_address = address;
        self.callback_host = address.ip().to_string();
        self
    }

    pub async fn login_browser(
        &self,
        interaction: &dyn Interaction,
        cancellation: &CancellationToken,
    ) -> Result<Credentials, Error> {
        let listener = TcpListener::bind(self.callback_address).await.ok();
        let callback_address = listener
            .as_ref()
            .and_then(|listener| listener.local_addr().ok())
            .unwrap_or(self.callback_address);
        let redirect_uri = format!(
            "http://{}:{}/auth/callback",
            self.callback_host,
            callback_address.port()
        );
        let (verifier, challenge) = pkce();
        let state = random_url_token::<16>();
        let authorization_url = self.authorization_url(&redirect_uri, &challenge, &state)?;
        interaction.notify(Notification::AuthorizationUrl {
            url: authorization_url,
        });
        let flow_cancellation = CancellationToken::new();
        let callback = wait_for_callback(listener, state.clone(), flow_cancellation.clone());
        let manual = interaction.manual_authorization(flow_cancellation.clone());
        let code = tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(Error::Cancelled),
            callback = callback => callback,
            manual = manual => {
                let input = manual?;
                let authorization = parse_authorization(&input)?;
                if authorization.state.is_some_and(|manual_state| manual_state != state) {
                    Err(Error::StateMismatch)
                } else {
                    authorization.code.ok_or(Error::MissingCode)
                }
            }
        };
        flow_cancellation.cancel();
        self.exchange(&code?, &verifier, &redirect_uri, cancellation)
            .await
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

    async fn exchange(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
        cancellation: &CancellationToken,
    ) -> Result<Credentials, Error> {
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("grant_type", "authorization_code")
            .append_pair("client_id", CLIENT_ID)
            .append_pair("code", code)
            .append_pair("code_verifier", verifier)
            .append_pair("redirect_uri", redirect_uri)
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
        read_credentials(response, "exchange").await
    }

    fn authorization_url(
        &self,
        redirect_uri: &str,
        challenge: &str,
        state: &str,
    ) -> Result<String, Error> {
        let mut url = url::Url::parse(&format!("{}/oauth/authorize", self.base_url))
            .map_err(|error| Error::InvalidResponse(error.to_string()))?;
        url.query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", CLIENT_ID)
            .append_pair("redirect_uri", redirect_uri)
            .append_pair("scope", SCOPE)
            .append_pair("code_challenge", challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", state)
            .append_pair("id_token_add_organizations", "true")
            .append_pair("codex_cli_simplified_flow", "true")
            .append_pair("originator", "ds");
        Ok(url.into())
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
    #[error("OAuth callback state did not match")]
    StateMismatch,
    #[error("OAuth callback carried no authorization code")]
    MissingCode,
    #[error("OAuth callback failed: {0}")]
    Callback(String),
}

struct Authorization {
    code: Option<String>,
    state: Option<String>,
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

fn pkce() -> (String, String) {
    let verifier = random_url_token::<32>();
    let challenge = BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

fn random_url_token<const N: usize>() -> String {
    BASE64_URL_SAFE_NO_PAD.encode(rand::random::<[u8; N]>())
}

fn parse_authorization(input: &str) -> Result<Authorization, Error> {
    let input = input.trim();
    if input.is_empty() {
        return Err(Error::MissingCode);
    }
    if let Ok(url) = url::Url::parse(input) {
        return Ok(Authorization {
            code: url
                .query_pairs()
                .find(|(name, _)| name == "code")
                .map(|(_, value)| value.into_owned()),
            state: url
                .query_pairs()
                .find(|(name, _)| name == "state")
                .map(|(_, value)| value.into_owned()),
        });
    }
    if let Some((code, state)) = input.split_once('#') {
        return Ok(Authorization {
            code: Some(code.to_owned()),
            state: Some(state.to_owned()),
        });
    }
    if input.contains("code=") {
        let values = url::form_urlencoded::parse(input.as_bytes()).collect::<Vec<_>>();
        return Ok(Authorization {
            code: values
                .iter()
                .find(|(name, _)| name == "code")
                .map(|(_, value)| value.to_string()),
            state: values
                .iter()
                .find(|(name, _)| name == "state")
                .map(|(_, value)| value.to_string()),
        });
    }
    Ok(Authorization {
        code: Some(input.to_owned()),
        state: None,
    })
}

async fn wait_for_callback(
    listener: Option<TcpListener>,
    state: String,
    cancellation: CancellationToken,
) -> Result<String, Error> {
    let Some(listener) = listener else {
        cancellation.cancelled().await;
        return Err(Error::Cancelled);
    };
    loop {
        let (mut socket, _) = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            accepted = listener.accept() => accepted.map_err(|error| Error::Callback(error.to_string()))?,
        };
        let authorization = match read_callback(&mut socket).await {
            Ok(authorization) => authorization,
            Err(error) => {
                write_callback(&mut socket, 400, "Invalid OAuth callback").await;
                if cancellation.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                if matches!(error, Error::StateMismatch | Error::MissingCode) {
                    continue;
                }
                return Err(error);
            }
        };
        if authorization.state.as_deref() != Some(&state) {
            write_callback(&mut socket, 400, "OAuth state mismatch").await;
            continue;
        }
        let Some(code) = authorization.code else {
            write_callback(&mut socket, 400, "Missing authorization code").await;
            continue;
        };
        write_callback(&mut socket, 200, "Authentication complete").await;
        return Ok(code);
    }
}

async fn read_callback(socket: &mut TcpStream) -> Result<Authorization, Error> {
    let mut request = Vec::new();
    loop {
        let mut bytes = [0; 1024];
        let count = socket
            .read(&mut bytes)
            .await
            .map_err(|error| Error::Callback(error.to_string()))?;
        if count == 0 || request.len() + count > 8192 {
            return Err(Error::Callback("invalid callback request".into()));
        }
        request.extend_from_slice(&bytes[..count]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    let request =
        std::str::from_utf8(&request).map_err(|error| Error::Callback(error.to_string()))?;
    let target = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or_else(|| Error::Callback("missing callback target".into()))?;
    let url = url::Url::parse(&format!("http://localhost{target}"))
        .map_err(|error| Error::Callback(error.to_string()))?;
    if url.path() != "/auth/callback" {
        return Err(Error::Callback("unknown callback path".into()));
    }
    Ok(Authorization {
        code: url
            .query_pairs()
            .find(|(name, _)| name == "code")
            .map(|(_, value)| value.into_owned()),
        state: url
            .query_pairs()
            .find(|(name, _)| name == "state")
            .map(|(_, value)| value.into_owned()),
    })
}

async fn write_callback(socket: &mut TcpStream, status: u16, message: &str) {
    let reason = if status == 200 { "OK" } else { "Bad Request" };
    let body = format!("<html><body>{message}</body></html>");
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: text/html; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = socket.write_all(response.as_bytes()).await;
}
