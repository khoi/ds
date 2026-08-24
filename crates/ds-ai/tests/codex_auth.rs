use crate::support::{Reply, serve};
use async_trait::async_trait;
use base64::prelude::*;
use ds_ai::codex::auth::OAuth;
use ds_ai::{
    AuthError, AuthEvent, AuthInteraction, AuthPrompt, AuthSelectOption, Credential, OAuthAuth,
    Provider as _,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn refreshes_codex_credentials() {
    let server = serve([Reply::json(
        200,
        json!({
            "access_token": token("acc_refreshed"),
            "refresh_token": "refresh_next",
            "expires_in": 3600
        }),
    )])
    .await;
    let oauth = OAuth::new().with_base_url(&server.base_url);
    let before = now_millis();
    let credential = credential("acc_old", "refresh_old", 0);

    let credentials = oauth
        .refresh(&credential, &CancellationToken::new())
        .await
        .unwrap();

    assert!(matches!(
        credentials,
        Credential::OAuth { refresh, access, expires, extra }
            if refresh == "refresh_next"
                && access == token("acc_refreshed")
                && expires >= before + 3_600_000
                && extra.is_empty()
    ));
    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /oauth/token HTTP/1.1\r\n"));
    let body = request.split("\r\n\r\n").nth(1).unwrap();
    assert!(body.contains("grant_type=refresh_token"));
    assert!(body.contains("refresh_token=refresh_old"));
    assert!(body.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
}

#[tokio::test]
async fn preserves_codex_refresh_failure_details() {
    let server = serve([Reply::json(
        500,
        json!({"error": {"code": "invalid_grant", "message": "expired"}}),
    )])
    .await;
    let oauth = OAuth::new().with_base_url(&server.base_url);
    let credential = credential("acc_old", "refresh_bad", 0);

    let error = oauth
        .refresh(&credential, &CancellationToken::new())
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("refresh"));
    assert!(error.contains("HTTP 500"));
    assert!(error.contains("invalid_grant"));
    assert!(error.contains("expired"));
    server.requests().await;
}

#[tokio::test]
async fn cancels_an_active_codex_refresh() {
    let server = serve([Reply::pending()]).await;
    let oauth = OAuth::new().with_base_url(&server.base_url);
    let credential = credential("acc_old", "refresh_wait", 0);
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let refresh = tokio::spawn(async move { oauth.refresh(&credential, &task_cancellation).await });
    server.wait_for_requests(1).await;

    cancellation.cancel();

    assert!(matches!(refresh.await.unwrap(), Err(AuthError::Cancelled)));
    server.requests().await;
}

#[tokio::test]
async fn cancels_a_codex_refresh_while_reading_the_token_body() {
    let server = serve([Reply::open_json(200, json!({"access_token": "unfinished"}))]).await;
    let oauth = OAuth::new().with_base_url(&server.base_url);
    let credential = credential("acc_old", "refresh_wait", 0);
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let refresh = tokio::spawn(async move { oauth.refresh(&credential, &task_cancellation).await });
    server.wait_for_requests(1).await;
    tokio::task::yield_now().await;

    cancellation.cancel();

    assert!(matches!(refresh.await.unwrap(), Err(AuthError::Cancelled)));
    server.requests().await;
}

#[tokio::test]
async fn logs_in_with_a_manual_codex_redirect() {
    let server = serve([Reply::json(
        200,
        json!({
            "access_token": token("acc_manual"),
            "refresh_token": "refresh_manual",
            "expires_in": 3600
        }),
    )])
    .await;
    let interaction = ManualInteraction::default();
    let oauth = OAuth::new()
        .with_base_url(&server.base_url)
        .with_callback_address(local_address());

    let credentials = oauth.login(&interaction).await.unwrap();

    assert!(
        matches!(credentials, Credential::OAuth { access, .. } if access == token("acc_manual"))
    );
    let authorization_url = interaction.authorization_url();
    let authorization_url = url::Url::parse(&authorization_url).unwrap();
    assert_eq!(authorization_url.path(), "/oauth/authorize");
    assert_eq!(query(&authorization_url, "response_type"), "code");
    assert_eq!(query(&authorization_url, "code_challenge_method"), "S256");
    assert_eq!(query(&authorization_url, "originator"), "ds");
    let request = server.requests().await.pop().unwrap();
    let body = request.split("\r\n\r\n").nth(1).unwrap();
    let form = url::form_urlencoded::parse(body.as_bytes()).collect::<Vec<_>>();
    assert_eq!(form_value(&form, "grant_type"), "authorization_code");
    assert_eq!(form_value(&form, "code"), "manual_code");
    assert_eq!(
        query(&authorization_url, "redirect_uri"),
        form_value(&form, "redirect_uri")
    );
    let verifier = form_value(&form, "code_verifier");
    assert_eq!(
        query(&authorization_url, "code_challenge"),
        BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
    );
}

#[tokio::test]
async fn logs_in_through_the_codex_browser_callback() {
    let server = serve([Reply::json(
        200,
        json!({
            "access_token": token("acc_callback"),
            "refresh_token": "refresh_callback",
            "expires_in": 3600
        }),
    )])
    .await;
    let interaction = CallbackInteraction::default();
    let oauth = OAuth::new()
        .with_base_url(&server.base_url)
        .with_callback_address(local_address());

    let credentials = oauth.login(&interaction).await.unwrap();

    assert!(
        matches!(credentials, Credential::OAuth { access, .. } if access == token("acc_callback"))
    );
    let request = server.requests().await.pop().unwrap();
    let body = request.split("\r\n\r\n").nth(1).unwrap();
    assert!(body.contains("code=callback_code"));
}

#[tokio::test]
async fn logs_in_with_the_codex_device_flow() {
    let server = serve([
        Reply::json(
            200,
            json!({
                "device_auth_id": "device_1",
                "user_code": "ABCD-EFGH",
                "interval": "0"
            }),
        ),
        Reply::json(403, json!({"error": "authorization_pending"})),
        Reply::json(404, json!({"error": "not_found"})),
        Reply::json(
            400,
            json!({"error": {"code": "deviceauth_authorization_pending"}}),
        ),
        Reply::json(429, json!({"error": "slow_down", "interval": 0})),
        Reply::json(
            200,
            json!({
                "authorization_code": "device_code",
                "code_verifier": "device_verifier"
            }),
        ),
        Reply::json(
            200,
            json!({
                "access_token": token("acc_device"),
                "refresh_token": "refresh_device",
                "expires_in": 3600
            }),
        ),
    ])
    .await;
    let interaction = DeviceInteraction::default();
    let oauth = OAuth::new().with_base_url(&server.base_url);

    let credentials = oauth.login(&interaction).await.unwrap();

    assert!(
        matches!(credentials, Credential::OAuth { access, .. } if access == token("acc_device"))
    );
    assert_eq!(
        interaction.event.lock().unwrap().clone(),
        Some(AuthEvent::DeviceCode {
            user_code: "ABCD-EFGH".into(),
            verification_uri: format!("{}/codex/device", server.base_url),
            interval_seconds: Some(0),
            expires_in_seconds: Some(15 * 60),
        })
    );
    let device_redirect = format!("{}/deviceauth/callback", server.base_url);
    let requests = server.requests().await;
    assert_eq!(requests.len(), 7);
    assert!(requests[0].starts_with("POST /api/accounts/deviceauth/usercode HTTP/1.1\r\n"));
    assert!(requests[1].starts_with("POST /api/accounts/deviceauth/token HTTP/1.1\r\n"));
    assert!(requests[6].starts_with("POST /oauth/token HTTP/1.1\r\n"));
    let exchange = requests[6].split("\r\n\r\n").nth(1).unwrap();
    assert!(exchange.contains("code=device_code"));
    assert!(exchange.contains("code_verifier=device_verifier"));
    assert!(exchange.contains(
        &url::form_urlencoded::byte_serialize(device_redirect.as_bytes()).collect::<String>()
    ));
}

#[tokio::test]
async fn cancels_the_codex_device_poll_wait() {
    let server = serve([
        Reply::json(
            200,
            json!({
                "device_auth_id": "device_wait",
                "user_code": "WAIT",
                "interval": 60
            }),
        ),
        Reply::json(403, json!({"error": "authorization_pending"})),
    ])
    .await;
    let oauth = OAuth::new().with_base_url(&server.base_url);
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let login = tokio::spawn(async move {
        let interaction = DeviceInteraction::with_cancellation(task_cancellation);
        oauth.login(&interaction).await
    });
    server.wait_for_requests(2).await;

    cancellation.cancel();

    assert!(matches!(login.await.unwrap(), Err(AuthError::Cancelled)));
    server.requests().await;
}

#[tokio::test]
async fn times_out_the_codex_device_flow() {
    let server = serve([
        Reply::json(
            200,
            json!({
                "device_auth_id": "device_timeout",
                "user_code": "TIME",
                "interval": 60
            }),
        ),
        Reply::json(403, json!({"error": "authorization_pending"})),
    ])
    .await;
    let oauth = OAuth::new()
        .with_base_url(&server.base_url)
        .with_device_timeout(Duration::from_millis(10));

    let result = oauth.login(&DeviceInteraction::default()).await;

    assert!(matches!(result, Err(AuthError::OAuth(message)) if message.contains("timed out")));
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test]
async fn preserves_codex_device_poll_failure_details() {
    let server = serve([
        Reply::json(
            200,
            json!({
                "device_auth_id": "device_fail",
                "user_code": "FAIL",
                "interval": 0
            }),
        ),
        Reply::json(
            500,
            json!({"error": {"code": "server_error", "message": "broken"}}),
        ),
    ])
    .await;
    let oauth = OAuth::new().with_base_url(&server.base_url);

    let result = oauth.login(&DeviceInteraction::default()).await;

    let error = result.unwrap_err().to_string();
    assert!(error.contains("device poll"));
    assert!(error.contains("HTTP 500"));
    assert!(error.contains("server_error"));
    assert!(error.contains("broken"));
    server.requests().await;
}

#[tokio::test]
async fn offers_pi_codex_login_methods_and_preserves_selection_cancellation() {
    let provider = ds_ai::codex::Provider::new([]);
    let oauth = provider.auth().oauth.as_ref().unwrap();
    let interaction = SelectionInteraction {
        answer: None,
        ..Default::default()
    };

    let result = oauth.login(&interaction).await;

    assert_eq!(oauth.name(), "OpenAI (ChatGPT Plus/Pro)");
    assert!(matches!(result, Err(AuthError::Cancelled)));
    assert_eq!(
        interaction.prompt.lock().unwrap().clone(),
        Some(AuthPrompt::Select {
            message: "Select OpenAI Codex login method:".into(),
            options: vec![
                AuthSelectOption {
                    id: "browser".into(),
                    label: "Browser login (default)".into(),
                    description: None,
                },
                AuthSelectOption {
                    id: "device_code".into(),
                    label: "Device code login (headless)".into(),
                    description: None,
                },
            ],
        })
    );
}

#[tokio::test]
async fn rejects_an_unknown_codex_login_method() {
    let provider = ds_ai::codex::Provider::new([]);
    let interaction = SelectionInteraction {
        answer: Some("unknown".into()),
        ..Default::default()
    };

    let error = provider
        .auth()
        .oauth
        .as_ref()
        .unwrap()
        .login(&interaction)
        .await;

    assert!(matches!(
        error,
        Err(AuthError::Authentication(message))
            if message == "Unknown OpenAI Codex login method: unknown"
    ));
}

#[tokio::test]
async fn exposes_pi_codex_browser_login_events_and_manual_prompt() {
    let provider = ds_ai::codex::Provider::new([]);
    let interaction = BrowserSelectionInteraction::default();

    let result = provider
        .auth()
        .oauth
        .as_ref()
        .unwrap()
        .login(&interaction)
        .await;

    assert!(matches!(result, Err(AuthError::OAuth(_))));
    let events = interaction.events.lock().unwrap();
    let [AuthEvent::AuthUrl { url, instructions }] = events.as_slice() else {
        panic!("missing authorization URL event");
    };
    assert_eq!(
        instructions.as_deref(),
        Some("A browser window should open. Complete login to finish.")
    );
    assert_eq!(url::Url::parse(url).unwrap().path(), "/oauth/authorize");
    let prompts = interaction.prompts.lock().unwrap();
    assert!(matches!(
        prompts.as_slice(),
        [AuthPrompt::Select { .. }, AuthPrompt::ManualCode { message, placeholder, .. }]
            if message == "Complete login in your browser, or paste the authorization code / redirect URL here:"
                && placeholder.as_deref() == Some("http://localhost:1455/auth/callback")
    ));
}

#[derive(Default)]
struct SelectionInteraction {
    answer: Option<String>,
    prompt: Mutex<Option<AuthPrompt>>,
    cancellation: CancellationToken,
}

#[async_trait]
impl AuthInteraction for SelectionInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError> {
        *self.prompt.lock().unwrap() = Some(prompt);
        self.answer.clone().ok_or(AuthError::Cancelled)
    }

    fn notify(&self, _event: AuthEvent) {}
}

#[derive(Default)]
struct BrowserSelectionInteraction {
    prompts: Mutex<Vec<AuthPrompt>>,
    events: Mutex<Vec<AuthEvent>>,
    cancellation: CancellationToken,
}

#[async_trait]
impl AuthInteraction for BrowserSelectionInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError> {
        let browser = matches!(prompt, AuthPrompt::Select { .. });
        self.prompts.lock().unwrap().push(prompt);
        if browser {
            Ok("browser".into())
        } else {
            Err(AuthError::Cancelled)
        }
    }

    fn notify(&self, event: AuthEvent) {
        self.events.lock().unwrap().push(event);
    }
}

#[derive(Default)]
struct ManualInteraction {
    authorization_url: Arc<Mutex<Option<String>>>,
    cancellation: CancellationToken,
}

impl ManualInteraction {
    fn authorization_url(&self) -> String {
        self.authorization_url.lock().unwrap().clone().unwrap()
    }
}

#[async_trait]
impl AuthInteraction for ManualInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError> {
        match prompt {
            AuthPrompt::Select { .. } => Ok("browser".into()),
            AuthPrompt::ManualCode { .. } => {
                let authorization_url = url::Url::parse(&self.authorization_url()).unwrap();
                Ok(format!(
                    "{}?code=manual_code&state={}",
                    query(&authorization_url, "redirect_uri"),
                    query(&authorization_url, "state")
                ))
            }
            _ => Err(AuthError::Authentication("unexpected prompt".into())),
        }
    }

    fn notify(&self, event: AuthEvent) {
        if let AuthEvent::AuthUrl { url, .. } = event {
            *self.authorization_url.lock().unwrap() = Some(url);
        }
    }
}

#[derive(Default)]
struct CallbackInteraction {
    cancellation: CancellationToken,
}

#[async_trait]
impl AuthInteraction for CallbackInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError> {
        match prompt {
            AuthPrompt::Select { .. } => Ok("browser".into()),
            AuthPrompt::ManualCode { cancellation, .. } => {
                cancellation.cancelled().await;
                Err(AuthError::Cancelled)
            }
            _ => Err(AuthError::Authentication("unexpected prompt".into())),
        }
    }

    fn notify(&self, event: AuthEvent) {
        let AuthEvent::AuthUrl { url, .. } = event else {
            return;
        };
        let authorization_url = url::Url::parse(&url).unwrap();
        let callback = format!(
            "{}?code=callback_code&state={}",
            query(&authorization_url, "redirect_uri"),
            query(&authorization_url, "state")
        );
        tokio::spawn(async move {
            reqwest::get(callback)
                .await
                .unwrap()
                .error_for_status()
                .unwrap();
        });
    }
}

#[derive(Default)]
struct DeviceInteraction {
    event: Arc<Mutex<Option<AuthEvent>>>,
    cancellation: CancellationToken,
}

impl DeviceInteraction {
    fn with_cancellation(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            ..Default::default()
        }
    }
}

#[async_trait]
impl AuthInteraction for DeviceInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError> {
        match prompt {
            AuthPrompt::Select { .. } => Ok("device_code".into()),
            _ => Err(AuthError::Authentication("unexpected prompt".into())),
        }
    }

    fn notify(&self, event: AuthEvent) {
        *self.event.lock().unwrap() = Some(event);
    }
}

fn token(account_id: &str) -> String {
    let payload = BASE64_URL_SAFE_NO_PAD.encode(
        serde_json::to_vec(&json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id}
        }))
        .unwrap(),
    );
    format!("aaa.{payload}.bbb")
}

fn credential(account_id: &str, refresh: &str, expires: u64) -> Credential {
    Credential::OAuth {
        refresh: refresh.into(),
        access: token(account_id),
        expires,
        extra: Default::default(),
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

fn local_address() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)
}

fn query(url: &url::Url, key: &str) -> String {
    url.query_pairs()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.into_owned())
        .unwrap()
}

fn form_value(
    form: &[(std::borrow::Cow<'_, str>, std::borrow::Cow<'_, str>)],
    key: &str,
) -> String {
    form.iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.to_string())
        .unwrap()
}
