use async_trait::async_trait;
use base64::prelude::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::Instant,
};
use tokio_util::sync::CancellationToken;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const REDIRECT_URI: &str = "http://localhost:53692/callback";
const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const EXPIRY_MARGIN: Duration = Duration::from_secs(5 * 60);

#[derive(Clone)]
pub struct OAuth {
    http: reqwest::Client,
    token_url: String,
    callback_address: String,
    redirect_uri: String,
}

impl Default for OAuth {
    fn default() -> Self {
        Self::new()
    }
}

impl OAuth {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            token_url: TOKEN_URL.into(),
            callback_address: format!(
                "{}:53692",
                std::env::var("PI_OAUTH_CALLBACK_HOST")
                    .ok()
                    .filter(|host| !host.is_empty())
                    .unwrap_or_else(|| "127.0.0.1".into())
            ),
            redirect_uri: REDIRECT_URI.into(),
        }
    }

    pub fn with_token_url(mut self, token_url: impl Into<String>) -> Self {
        self.token_url = token_url.into();
        self
    }

    pub fn with_callback_address(mut self, callback_address: SocketAddr) -> Self {
        self.callback_address = callback_address.to_string();
        self
    }

    pub fn with_redirect_uri(mut self, redirect_uri: impl Into<String>) -> Self {
        self.redirect_uri = redirect_uri.into();
        self
    }

    fn authorization_url(&self, challenge: &str, state: &str) -> Result<String, crate::AuthError> {
        let mut url = url::Url::parse(AUTHORIZE_URL)
            .map_err(|error| crate::AuthError::OAuth(error.to_string()))?;
        url.query_pairs_mut()
            .append_pair("code", "true")
            .append_pair("client_id", CLIENT_ID)
            .append_pair("response_type", "code")
            .append_pair("redirect_uri", &self.redirect_uri)
            .append_pair("scope", SCOPES)
            .append_pair("code_challenge", challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", state);
        Ok(url.into())
    }

    async fn token(
        &self,
        body: serde_json::Value,
        cancellation: &CancellationToken,
    ) -> Result<crate::Credential, crate::AuthError> {
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        let response = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(crate::AuthError::Cancelled),
            _ = tokio::time::sleep_until(deadline) => {
                return Err(crate::AuthError::OAuth(format!(
                    "token request to {} timed out",
                    self.token_url
                )));
            }
            response = self
                .http
                .post(&self.token_url)
                .header(reqwest::header::ACCEPT, "application/json")
                .json(&body)
                .send() => {
                response.map_err(|error| crate::AuthError::OAuth(format!(
                    "token request to {} failed: {error}",
                    self.token_url
                )))?
            }
        };
        let status = response.status();
        let body = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(crate::AuthError::Cancelled),
            _ = tokio::time::sleep_until(deadline) => {
                return Err(crate::AuthError::OAuth(format!(
                    "token response from {} timed out",
                    self.token_url
                )));
            }
            body = response.text() => {
                body.map_err(|error| crate::AuthError::OAuth(format!(
                    "token response from {} failed: {error}",
                    self.token_url
                )))?
            }
        };
        if !status.is_success() {
            return Err(crate::AuthError::OAuth(format!(
                "token request to {} failed with HTTP {}: {body}",
                self.token_url,
                status.as_u16(),
            )));
        }
        let token: TokenResponse = serde_json::from_str(&body).map_err(|error| {
            crate::AuthError::OAuth(format!(
                "invalid token response from {}: {error}; body={body}",
                self.token_url
            ))
        })?;
        Ok(crate::Credential::OAuth {
            refresh: token.refresh_token,
            access: token.access_token,
            expires: now_millis()
                .saturating_add(token.expires_in.saturating_mul(1000))
                .saturating_sub(duration_millis(EXPIRY_MARGIN)),
            extra: BTreeMap::new(),
        })
    }
}

#[async_trait]
impl crate::OAuthAuth for OAuth {
    fn name(&self) -> &str {
        "Anthropic (Claude Pro/Max)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    async fn login(
        &self,
        interaction: &dyn crate::AuthInteraction,
    ) -> Result<crate::Credential, crate::AuthError> {
        let listener = tokio::select! {
            biased;
            _ = interaction.cancellation().cancelled() => return Err(crate::AuthError::Cancelled),
            listener = TcpListener::bind(&self.callback_address) => {
                listener.map_err(|error| crate::AuthError::OAuth(error.to_string()))?
            }
        };
        let (verifier, challenge) = pkce();
        let authorization_url = self.authorization_url(&challenge, &verifier)?;
        interaction.notify(crate::AuthEvent::AuthUrl {
            url: authorization_url,
            instructions: Some(
                "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
                    .into(),
            ),
        });
        let flow_cancellation = CancellationToken::new();
        let callback = wait_for_callback(listener, verifier.clone(), flow_cancellation.clone());
        let manual = interaction.prompt(crate::AuthPrompt::ManualCode {
            message: "Complete login in your browser, or paste the authorization code / redirect URL here:"
                .into(),
            placeholder: Some(self.redirect_uri.clone()),
            cancellation: flow_cancellation.clone(),
        });
        let authorization = tokio::select! {
            biased;
            _ = interaction.cancellation().cancelled() => Err(crate::AuthError::Cancelled),
            callback = callback => callback,
            manual = manual => manual.and_then(|input| parse_authorization(&input, &verifier)),
        };
        flow_cancellation.cancel();
        let authorization = authorization?;
        interaction.notify(crate::AuthEvent::Progress {
            message: "Exchanging authorization code for tokens...".into(),
        });
        self.token(
            serde_json::json!({
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": authorization.code,
                "state": authorization.state,
                "redirect_uri": self.redirect_uri,
                "code_verifier": verifier,
            }),
            interaction.cancellation(),
        )
        .await
    }

    async fn refresh(
        &self,
        credential: &crate::Credential,
        cancellation: &CancellationToken,
    ) -> Result<crate::Credential, crate::AuthError> {
        let crate::Credential::OAuth { refresh, .. } = credential else {
            return Err(crate::AuthError::OAuth("expected OAuth credential".into()));
        };
        self.token(
            serde_json::json!({
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh,
            }),
            cancellation,
        )
        .await
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
            headers: BTreeMap::new(),
            base_url: None,
        })
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
}

struct Authorization {
    code: String,
    state: String,
}

fn pkce() -> (String, String) {
    let verifier = BASE64_URL_SAFE_NO_PAD.encode(rand::random::<[u8; 32]>());
    let challenge = BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

fn parse_authorization(
    input: &str,
    expected_state: &str,
) -> Result<Authorization, crate::AuthError> {
    let input = input.trim();
    let (code, state) = if let Ok(url) = url::Url::parse(input) {
        (
            query(&url, "code"),
            query(&url, "state").unwrap_or_else(|| expected_state.into()),
        )
    } else if let Some((code, state)) = input.split_once('#') {
        (Some(code.into()), state.into())
    } else if input.contains("code=") {
        let mut code = None;
        let mut state = None;
        for (name, value) in url::form_urlencoded::parse(input.as_bytes()) {
            match name.as_ref() {
                "code" => code = Some(value.into_owned()),
                "state" => state = Some(value.into_owned()),
                _ => {}
            }
        }
        (code, state.unwrap_or_else(|| expected_state.into()))
    } else {
        (Some(input.into()), expected_state.into())
    };
    let code = code
        .filter(|code| !code.is_empty())
        .ok_or_else(|| crate::AuthError::OAuth("missing authorization code".into()))?;
    if state != expected_state {
        return Err(crate::AuthError::OAuth("OAuth state mismatch".into()));
    }
    Ok(Authorization { code, state })
}

async fn wait_for_callback(
    listener: TcpListener,
    expected_state: String,
    cancellation: CancellationToken,
) -> Result<Authorization, crate::AuthError> {
    loop {
        let (mut socket, _) = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(crate::AuthError::Cancelled),
            accepted = listener.accept() => {
                accepted.map_err(|error| crate::AuthError::OAuth(error.to_string()))?
            }
        };
        let authorization = read_callback(&mut socket)
            .await
            .and_then(|input| parse_authorization(&input, &expected_state));
        match authorization {
            Ok(authorization) => {
                write_callback(&mut socket, 200, "Authentication complete").await;
                return Ok(authorization);
            }
            Err(_) => write_callback(&mut socket, 400, "Invalid OAuth callback").await,
        }
    }
}

async fn read_callback(socket: &mut TcpStream) -> Result<String, crate::AuthError> {
    let mut request = Vec::new();
    loop {
        let mut bytes = [0; 1024];
        let count = socket
            .read(&mut bytes)
            .await
            .map_err(|error| crate::AuthError::OAuth(error.to_string()))?;
        if count == 0 || request.len() + count > 8192 {
            return Err(crate::AuthError::OAuth("invalid OAuth callback".into()));
        }
        request.extend_from_slice(&bytes[..count]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    let request = std::str::from_utf8(&request)
        .map_err(|error| crate::AuthError::OAuth(error.to_string()))?;
    let target = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or_else(|| crate::AuthError::OAuth("missing OAuth callback target".into()))?;
    let url = url::Url::parse(&format!("http://localhost{target}"))
        .map_err(|error| crate::AuthError::OAuth(error.to_string()))?;
    if url.path() != "/callback" {
        return Err(crate::AuthError::OAuth(
            "unknown OAuth callback path".into(),
        ));
    }
    Ok(url.into())
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

fn query(url: &url::Url, name: &str) -> Option<String> {
    url.query_pairs()
        .find_map(|(key, value)| (key == name).then(|| value.into_owned()))
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}
