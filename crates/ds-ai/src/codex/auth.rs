use async_trait::async_trait;
use base64::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    future::Future,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::Instant,
};
use tokio_util::sync::CancellationToken;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_BASE_URL: &str = "https://auth.openai.com";
const SCOPE: &str = "openid profile email offline_access";
const DEVICE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const MIN_DEVICE_POLL_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Notification {
    AuthorizationUrl {
        url: String,
    },
    DeviceCode {
        user_code: String,
        verification_uri: String,
        interval: Duration,
        expires_in: Duration,
    },
}

#[async_trait]
pub(crate) trait Interaction: Send + Sync {
    fn notify(&self, notification: Notification);

    async fn manual_authorization(&self, cancellation: CancellationToken) -> Result<String, Error>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct Credentials {
    pub(crate) access_token: String,
    pub(crate) refresh_token: String,
    pub(crate) expires_at: u64,
}

impl Credentials {
    fn account_id(&self) -> Result<String, Error> {
        super::account_id(&self.access_token).map_err(Error::InvalidResponse)
    }
}

#[derive(Clone)]
struct Client {
    http: reqwest::Client,
    base_url: String,
    callback_address: SocketAddr,
    callback_host: String,
    device_timeout: Duration,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.into(),
            callback_address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 1455),
            callback_host: "localhost".into(),
            device_timeout: DEVICE_TIMEOUT,
        }
    }

    fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_owned();
        self
    }

    fn with_callback_address(mut self, address: SocketAddr) -> Self {
        self.callback_address = address;
        self.callback_host = address.ip().to_string();
        self
    }

    fn with_device_timeout(mut self, timeout: Duration) -> Self {
        self.device_timeout = timeout;
        self
    }

    async fn login_browser(
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

    async fn login_device(
        &self,
        interaction: &dyn Interaction,
        cancellation: &CancellationToken,
    ) -> Result<Credentials, Error> {
        let response = self
            .device_request(
                "/api/accounts/deviceauth/usercode",
                serde_json::json!({"client_id": CLIENT_ID}),
                cancellation,
                None,
            )
            .await?;
        let status = response.status();
        let body = read_body(response, cancellation, None).await?;
        if status.as_u16() == 404 {
            return Err(Error::DeviceLoginDisabled);
        }
        if !status.is_success() {
            return Err(Error::Server {
                operation: "device start",
                status: status.as_u16(),
                body,
            });
        }
        let device = parse_device(&body)?;
        let deadline = Instant::now() + self.device_timeout;
        interaction.notify(Notification::DeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: format!("{}/codex/device", self.base_url),
            interval: device.interval,
            expires_in: self.device_timeout,
        });
        let mut interval = device.interval;
        let mut slowed_down = false;
        let authorization = loop {
            let response = self
                .device_request(
                    "/api/accounts/deviceauth/token",
                    serde_json::json!({
                        "device_auth_id": device.id,
                        "user_code": device.user_code,
                    }),
                    cancellation,
                    Some(deadline),
                )
                .await
                .map_err(|error| device_poll_error(error, slowed_down))?;
            let status = response.status();
            let body = read_body(response, cancellation, Some(deadline))
                .await
                .map_err(|error| device_poll_error(error, slowed_down))?;
            match parse_device_poll(status.as_u16(), &body)? {
                DevicePoll::Complete(authorization) => break authorization,
                DevicePoll::Pending => {}
                DevicePoll::SlowDown => {
                    slowed_down = true;
                    interval = next_device_interval(interval);
                }
            }
            wait_for_device_poll(interval, cancellation, deadline)
                .await
                .map_err(|error| device_poll_error(error, slowed_down))?;
        };
        self.exchange(
            &authorization.code,
            &authorization.verifier,
            &format!("{}/deviceauth/callback", self.base_url),
            cancellation,
        )
        .await
    }

    async fn refresh(
        &self,
        refresh_token: &str,
        cancellation: &CancellationToken,
    ) -> Result<Credentials, Error> {
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("grant_type", "refresh_token")
            .append_pair("refresh_token", refresh_token)
            .append_pair("client_id", CLIENT_ID)
            .finish();
        self.send_token_request(body, "refresh", cancellation).await
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
        self.send_token_request(body, "exchange", cancellation)
            .await
    }

    async fn send_token_request(
        &self,
        body: String,
        operation: &'static str,
        cancellation: &CancellationToken,
    ) -> Result<Credentials, Error> {
        let response = await_http(
            self.http
                .post(format!("{}/oauth/token", self.base_url))
                .header("content-type", "application/x-www-form-urlencoded")
                .body(body)
                .send(),
            cancellation,
            None,
        )
        .await?;
        read_credentials(response, operation, cancellation).await
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

    async fn device_request(
        &self,
        path: &str,
        body: serde_json::Value,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<reqwest::Response, Error> {
        await_http(
            self.http
                .post(format!("{}{path}", self.base_url))
                .json(&body)
                .send(),
            cancellation,
            deadline,
        )
        .await
    }
}

#[derive(Clone)]
pub struct OAuth {
    client: Client,
}

impl Default for OAuth {
    fn default() -> Self {
        Self::new()
    }
}

impl OAuth {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.client = self.client.with_base_url(base_url);
        self
    }

    pub fn with_callback_address(mut self, address: SocketAddr) -> Self {
        self.client = self.client.with_callback_address(address);
        self
    }

    pub fn with_device_timeout(mut self, timeout: Duration) -> Self {
        self.client = self.client.with_device_timeout(timeout);
        self
    }
}

#[async_trait]
impl crate::OAuthAuth for OAuth {
    fn name(&self) -> &str {
        "OpenAI (ChatGPT Plus/Pro)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    async fn login(
        &self,
        interaction: &dyn crate::AuthInteraction,
    ) -> Result<crate::Credential, crate::AuthError> {
        let method = interaction
            .prompt(crate::AuthPrompt::Select {
                message: "Select OpenAI Codex login method:".into(),
                options: vec![
                    crate::AuthSelectOption {
                        id: "browser".into(),
                        label: "Browser login (default)".into(),
                        description: None,
                    },
                    crate::AuthSelectOption {
                        id: "device_code".into(),
                        label: "Device code login (headless)".into(),
                        description: None,
                    },
                ],
            })
            .await?;
        let interaction_adapter = InteractionAdapter(interaction);
        let credentials = match method.as_str() {
            "browser" => {
                self.client
                    .login_browser(&interaction_adapter, interaction.cancellation())
                    .await
            }
            "device_code" => {
                self.client
                    .login_device(&interaction_adapter, interaction.cancellation())
                    .await
            }
            method => {
                return Err(crate::AuthError::Authentication(format!(
                    "Unknown OpenAI Codex login method: {method}"
                )));
            }
        }
        .map_err(auth_error)?;
        oauth_credential(credentials).map_err(auth_error)
    }

    async fn refresh(
        &self,
        credential: &crate::Credential,
        cancellation: &CancellationToken,
    ) -> Result<crate::Credential, crate::AuthError> {
        let crate::Credential::OAuth { refresh, .. } = credential else {
            return Err(crate::AuthError::OAuth("expected OAuth credential".into()));
        };
        let credentials = self
            .client
            .refresh(refresh, cancellation)
            .await
            .map_err(auth_error)?;
        oauth_credential(credentials).map_err(auth_error)
    }

    async fn to_auth(
        &self,
        credential: &crate::Credential,
    ) -> Result<crate::ModelAuth, crate::AuthError> {
        let crate::Credential::OAuth { access, .. } = credential else {
            return Err(crate::AuthError::OAuth("expected OAuth credential".into()));
        };
        Ok(crate::ModelAuth {
            api_key: Some(access.clone()),
            ..Default::default()
        })
    }
}

struct InteractionAdapter<'a>(&'a dyn crate::AuthInteraction);

#[async_trait]
impl Interaction for InteractionAdapter<'_> {
    fn notify(&self, notification: Notification) {
        let event = match notification {
            Notification::AuthorizationUrl { url } => crate::AuthEvent::AuthUrl {
                url,
                instructions: Some(
                    "A browser window should open. Complete login to finish.".into(),
                ),
            },
            Notification::DeviceCode {
                user_code,
                verification_uri,
                interval,
                expires_in,
            } => crate::AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                interval_seconds: Some(interval.as_secs()),
                expires_in_seconds: Some(expires_in.as_secs()),
            },
        };
        self.0.notify(event);
    }

    async fn manual_authorization(&self, cancellation: CancellationToken) -> Result<String, Error> {
        self.0
            .prompt(crate::AuthPrompt::ManualCode {
                message: "Complete login in your browser, or paste the authorization code / redirect URL here:"
                    .into(),
                placeholder: Some("http://localhost:1455/auth/callback".into()),
                cancellation,
            })
            .await
            .map_err(|error| match error {
                crate::AuthError::Cancelled => Error::Cancelled,
                error => Error::InvalidResponse(error.to_string()),
            })
    }
}

fn oauth_credential(credentials: Credentials) -> Result<crate::Credential, Error> {
    let account_id = credentials.account_id()?;
    Ok(crate::Credential::OAuth {
        refresh: credentials.refresh_token,
        access: credentials.access_token,
        expires: credentials.expires_at,
        extra: BTreeMap::from([(String::from("accountId"), serde_json::json!(account_id))]),
    })
}

fn auth_error(error: Error) -> crate::AuthError {
    match error {
        Error::Cancelled => crate::AuthError::Cancelled,
        error => crate::AuthError::OAuth(error.to_string()),
    }
}

#[derive(Debug, Error)]
pub(crate) enum Error {
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
    #[error("OAuth device login timed out")]
    Timeout,
    #[error(
        "OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL."
    )]
    DeviceLoginDisabled,
    #[error(
        "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again."
    )]
    SlowDownTimeout,
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

struct Device {
    id: String,
    user_code: String,
    interval: Duration,
}

struct DeviceAuthorization {
    code: String,
    verifier: String,
}

enum DevicePoll {
    Complete(DeviceAuthorization),
    Pending,
    SlowDown,
}

async fn read_credentials(
    response: reqwest::Response,
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Credentials, Error> {
    let status = response.status();
    let body = await_http(response.text(), cancellation, None).await?;
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
        .filter(|token| !token.is_empty())
        .ok_or_else(|| Error::InvalidResponse("missing access_token".into()))?;
    let refresh_token = response
        .refresh_token
        .filter(|token| !token.is_empty())
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

async fn read_body(
    response: reqwest::Response,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<String, Error> {
    await_http(response.text(), cancellation, deadline).await
}

async fn await_http<T>(
    request: impl Future<Output = Result<T, reqwest::Error>>,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<T, Error> {
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(Error::Cancelled),
        _ = wait_until(deadline) => Err(Error::Timeout),
        response = request => response.map_err(|error| Error::Http(error.to_string())),
    }
}

async fn wait_until(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn parse_device(body: &str) -> Result<Device, Error> {
    let response: serde_json::Value =
        serde_json::from_str(body).map_err(|error| Error::InvalidResponse(error.to_string()))?;
    let id = response
        .get("device_auth_id")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Error::InvalidResponse("missing device_auth_id".into()))?;
    let user_code = response
        .get("user_code")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Error::InvalidResponse("missing user_code".into()))?;
    let interval = response
        .get("interval")
        .and_then(parse_seconds)
        .map(clamp_device_interval)
        .ok_or_else(|| Error::InvalidResponse("invalid device interval".into()))?;
    Ok(Device {
        id: id.into(),
        user_code: user_code.into(),
        interval,
    })
}

fn parse_device_poll(status: u16, body: &str) -> Result<DevicePoll, Error> {
    let response = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
    if (200..300).contains(&status) {
        let code = response
            .get("authorization_code")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| Error::InvalidResponse("missing authorization_code".into()))?;
        let verifier = response
            .get("code_verifier")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| Error::InvalidResponse("missing code_verifier".into()))?;
        return Ok(DevicePoll::Complete(DeviceAuthorization {
            code: code.into(),
            verifier: verifier.into(),
        }));
    }
    if matches!(status, 403 | 404) {
        return Ok(DevicePoll::Pending);
    }
    let error = response.get("error").and_then(|error| {
        error
            .as_str()
            .or_else(|| error.get("code").and_then(serde_json::Value::as_str))
    });
    match error {
        Some("authorization_pending" | "deviceauth_authorization_pending") => {
            Ok(DevicePoll::Pending)
        }
        Some("slow_down") => Ok(DevicePoll::SlowDown),
        _ => Err(Error::Server {
            operation: "device poll",
            status,
            body: body.into(),
        }),
    }
}

fn parse_seconds(value: &serde_json::Value) -> Option<Duration> {
    let seconds = value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse::<f64>().ok())?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let milliseconds = (seconds * 1_000.0).floor();
    Some(Duration::from_millis(
        milliseconds.min(u64::MAX as f64) as u64
    ))
}

fn next_device_interval(interval: Duration) -> Duration {
    clamp_device_interval(interval).saturating_add(Duration::from_secs(5))
}

fn clamp_device_interval(interval: Duration) -> Duration {
    interval.max(MIN_DEVICE_POLL_INTERVAL)
}

fn device_poll_error(error: Error, slowed_down: bool) -> Error {
    if slowed_down && matches!(error, Error::Timeout) {
        Error::SlowDownTimeout
    } else {
        error
    }
}

async fn wait_for_device_poll(
    interval: Duration,
    cancellation: &CancellationToken,
    deadline: Instant,
) -> Result<(), Error> {
    let next_poll = Instant::now().checked_add(interval).unwrap_or(deadline);
    let wake = next_poll.min(deadline);
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(Error::Cancelled),
        _ = tokio::time::sleep_until(wake) => {}
    }
    if Instant::now() >= deadline {
        Err(Error::Timeout)
    } else {
        Ok(())
    }
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
        let authorization = match tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            authorization = read_callback(&mut socket) => authorization,
        } {
            Ok(authorization) => authorization,
            Err(_) => {
                write_callback(&mut socket, 400, "Invalid OAuth callback").await;
                if cancellation.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                continue;
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

#[cfg(test)]
mod tests {
    use super::{DevicePoll, next_device_interval, parse_device, parse_device_poll};
    use std::time::Duration;

    #[test]
    fn applies_the_five_second_slow_down_fallback() {
        assert!(matches!(
            parse_device_poll(429, r#"{"error":"slow_down","interval":30}"#),
            Ok(DevicePoll::SlowDown)
        ));
        assert_eq!(
            next_device_interval(Duration::from_secs(2)),
            Duration::from_secs(7)
        );
    }

    #[test]
    fn treats_a_zero_slow_down_interval_as_a_five_second_backoff() {
        assert_eq!(next_device_interval(Duration::ZERO), Duration::from_secs(6));
    }

    #[test]
    fn classifies_pending_device_responses() {
        for (status, body) in [
            (403, "{}"),
            (404, "{}"),
            (400, r#"{"error":"authorization_pending"}"#),
            (
                400,
                r#"{"error":{"code":"deviceauth_authorization_pending"}}"#,
            ),
        ] {
            assert!(matches!(
                parse_device_poll(status, body),
                Ok(DevicePoll::Pending)
            ));
        }
    }

    #[test]
    fn floors_device_intervals_to_milliseconds() {
        let device =
            parse_device(r#"{"device_auth_id":"device","user_code":"code","interval":1.2349}"#)
                .unwrap();

        assert_eq!(device.interval, Duration::from_millis(1_234));
    }
}
