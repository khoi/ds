use crate::support::{Reply, serve};
use async_trait::async_trait;
use base64::prelude::*;
use ds_ai::codex::auth::OAuth;
use ds_ai::{
    AuthError, AuthEvent, AuthInteraction, AuthPrompt, AuthSelectOption, Credential, ModelAuth,
    OAuthAuth, Provider as _,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    process::Command,
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
                && extra.get("accountId").and_then(serde_json::Value::as_str)
                    == Some("acc_refreshed")
    ));
    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /oauth/token HTTP/1.1\r\n"));
    let body = request.split("\r\n\r\n").nth(1).unwrap();
    assert!(body.contains("grant_type=refresh_token"));
    assert!(body.contains("refresh_token=refresh_old"));
    assert!(body.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
}

#[tokio::test]
async fn exposes_codex_subscription_oauth_and_auth() {
    let provider = ds_ai::codex::Provider::new([]);
    let oauth = provider.auth().oauth.as_ref().unwrap();
    let credential = credential("account", "refresh", 0);

    assert_eq!(oauth.name(), "OpenAI (ChatGPT Plus/Pro)");
    assert!(oauth.is_subscription());
    assert_eq!(
        oauth.to_auth(&credential).await.unwrap(),
        ModelAuth {
            api_key: Some(credential_access(&credential)),
            ..Default::default()
        }
    );
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

#[test]
fn codex_refresh_failure_writes_nothing_to_stderr() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "codex_auth::preserves_codex_refresh_failure_details",
            "--exact",
            "--nocapture",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stderr, b"");
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

    assert!(matches!(
        credentials,
        Credential::OAuth { access, extra, .. }
            if access == token("acc_manual")
                && extra.get("accountId").and_then(serde_json::Value::as_str)
                    == Some("acc_manual")
    ));
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
async fn keeps_waiting_after_malformed_codex_browser_callbacks() {
    let server = serve([Reply::json(
        200,
        json!({
            "access_token": token("acc_callback_recovery"),
            "refresh_token": "refresh_callback_recovery",
            "expires_in": 3600
        }),
    )])
    .await;
    let interaction = NoisyCallbackInteraction::default();
    let oauth = OAuth::new()
        .with_base_url(&server.base_url)
        .with_callback_address(local_address());

    let credentials = oauth.login(&interaction).await.unwrap();
    let task = {
        let mut task = interaction.task.lock().unwrap();
        task.take().unwrap()
    };
    let responses = task.await.unwrap();

    assert!(matches!(
        credentials,
        Credential::OAuth { access, .. } if access == token("acc_callback_recovery")
    ));
    assert_eq!(responses, [400, 400, 200]);
    server.requests().await;
}

#[tokio::test]
async fn cancels_while_reading_a_codex_browser_callback() {
    let (interaction, connected) = StalledCallbackInteraction::new();
    let cancellation = interaction.cancellation.clone();
    let oauth = OAuth::new().with_callback_address(local_address());
    let login = tokio::spawn(async move { oauth.login(&interaction).await });
    connected.await.unwrap();

    cancellation.cancel();

    assert!(matches!(login.await.unwrap(), Err(AuthError::Cancelled)));
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
        Reply::json(404, json!({"error": "authorization_pending"})),
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
    let event = interaction.event.clone();
    let oauth = OAuth::new().with_base_url(&server.base_url);
    let login = tokio::spawn(async move { oauth.login(&interaction).await });

    let credentials = login.await.unwrap().unwrap();

    assert!(
        matches!(credentials, Credential::OAuth { access, .. } if access == token("acc_device"))
    );
    assert_eq!(
        event.lock().unwrap().clone(),
        Some(AuthEvent::DeviceCode {
            user_code: "ABCD-EFGH".into(),
            verification_uri: format!("{}/codex/device", server.base_url),
            interval_seconds: Some(1),
            expires_in_seconds: Some(15 * 60),
        })
    );
    let device_redirect = format!("{}/deviceauth/callback", server.base_url);
    let requests = server.requests().await;
    assert_eq!(requests.len(), 5);
    assert!(requests[0].starts_with("POST /api/accounts/deviceauth/usercode HTTP/1.1\r\n"));
    assert!(requests[1].starts_with("POST /api/accounts/deviceauth/token HTTP/1.1\r\n"));
    assert!(requests[2].starts_with("POST /api/accounts/deviceauth/token HTTP/1.1\r\n"));
    assert!(requests[4].starts_with("POST /oauth/token HTTP/1.1\r\n"));
    let exchange = requests[4].split("\r\n\r\n").nth(1).unwrap();
    assert!(exchange.contains("code=device_code"));
    assert!(exchange.contains("code_verifier=device_verifier"));
    assert!(exchange.contains(
        &url::form_urlencoded::byte_serialize(device_redirect.as_bytes()).collect::<String>()
    ));
}

#[tokio::test]
async fn guides_device_login_users_to_browser_login_when_device_auth_is_disabled() {
    let server = serve([Reply::json(404, json!({"error": "not_found"}))]).await;
    let oauth = OAuth::new().with_base_url(&server.base_url);

    let error = oauth
        .login(&DeviceInteraction::default())
        .await
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "OAuth failed: OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL."
    );
    server.requests().await;
}

#[tokio::test]
async fn reports_clock_drift_guidance_after_a_slow_down_timeout() {
    let server = serve([
        Reply::json(
            200,
            json!({
                "device_auth_id": "device_slow_timeout",
                "user_code": "SLOW-TIMEOUT",
                "interval": 0
            }),
        ),
        Reply::json(429, json!({"error": "slow_down", "interval": 0.2})),
    ])
    .await;
    let oauth = OAuth::new()
        .with_base_url(&server.base_url)
        .with_device_timeout(Duration::from_millis(20));
    let login = tokio::spawn(async move { oauth.login(&DeviceInteraction::default()).await });

    let error = login.await.unwrap().unwrap_err();

    assert!(matches!(
        error,
        AuthError::OAuth(message)
            if message == "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again."
    ));
}

#[tokio::test]
async fn rejects_empty_codex_oauth_tokens() {
    for (access_token, refresh_token, missing) in [
        (String::new(), String::from("refresh"), "access_token"),
        (token("account"), String::new(), "refresh_token"),
    ] {
        let server = serve([Reply::json(
            200,
            json!({
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_in": 3600
            }),
        )])
        .await;
        let oauth = OAuth::new().with_base_url(&server.base_url);
        let error = oauth
            .refresh(
                &credential("account", "refresh", 0),
                &CancellationToken::new(),
            )
            .await
            .unwrap_err();

        assert!(
            matches!(error, AuthError::OAuth(message) if message == format!("invalid OAuth response: missing {missing}"))
        );
        server.requests().await;
    }
}

#[tokio::test]
async fn rejects_two_and_four_segment_codex_jwt_access_tokens_during_exchange() {
    for access_token in ["header.payload", "header.payload.signature.extra"] {
        let server = serve([Reply::json(
            200,
            json!({
                "access_token": access_token,
                "refresh_token": "refresh",
                "expires_in": 3600
            }),
        )])
        .await;
        let interaction = ManualInteraction::default();
        let oauth = OAuth::new()
            .with_base_url(&server.base_url)
            .with_callback_address(local_address());
        let error = oauth.login(&interaction).await.unwrap_err();

        assert!(
            matches!(error, AuthError::OAuth(message) if message == "invalid OAuth response: Failed to extract accountId from token")
        );
        server.requests().await;
    }
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
async fn starts_the_codex_device_deadline_after_receiving_the_device_code() {
    let (base_url, server) = serve_delayed_device_start(Duration::from_millis(200)).await;
    let oauth = OAuth::new()
        .with_base_url(base_url)
        .with_device_timeout(Duration::from_millis(100));

    let credentials = oauth.login(&DeviceInteraction::default()).await.unwrap();

    assert!(matches!(
        credentials,
        Credential::OAuth { access, .. } if access == token("acc_delayed_device")
    ));
    server.await.unwrap();
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
async fn offers_codex_login_methods_and_preserves_selection_cancellation() {
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
async fn exposes_codex_browser_login_events_and_manual_prompt() {
    let provider = ds_ai::codex::Provider::new([]);
    let interaction = BrowserSelectionInteraction::default();

    let result = provider
        .auth()
        .oauth
        .as_ref()
        .unwrap()
        .login(&interaction)
        .await;

    assert!(matches!(result, Err(AuthError::Cancelled)));
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

#[derive(Default)]
struct NoisyCallbackInteraction {
    cancellation: CancellationToken,
    task: Mutex<Option<tokio::task::JoinHandle<Vec<u16>>>>,
}

struct StalledCallbackInteraction {
    cancellation: CancellationToken,
    connected: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

impl StalledCallbackInteraction {
    fn new() -> (Self, tokio::sync::oneshot::Receiver<()>) {
        let (connected, receiver) = tokio::sync::oneshot::channel();
        (
            Self {
                cancellation: CancellationToken::new(),
                connected: Mutex::new(Some(connected)),
            },
            receiver,
        )
    }
}

#[async_trait]
impl AuthInteraction for StalledCallbackInteraction {
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
        let redirect = url::Url::parse(&query(&authorization_url, "redirect_uri")).unwrap();
        let address = format!(
            "{}:{}",
            redirect.host_str().unwrap(),
            redirect.port().unwrap()
        );
        let connected = self.connected.lock().unwrap().take().unwrap();
        tokio::spawn(async move {
            let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
            connected.send(()).unwrap();
            let mut byte = [0];
            tokio::io::AsyncReadExt::read(&mut socket, &mut byte)
                .await
                .unwrap();
        });
    }
}

#[async_trait]
impl AuthInteraction for NoisyCallbackInteraction {
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
        let redirect = url::Url::parse(&query(&authorization_url, "redirect_uri")).unwrap();
        let state = query(&authorization_url, "state");
        let task = tokio::spawn(async move {
            let address = format!(
                "{}:{}",
                redirect.host_str().unwrap(),
                redirect.port().unwrap()
            );
            let mut socket = tokio::net::TcpStream::connect(&address).await.unwrap();
            tokio::io::AsyncWriteExt::write_all(&mut socket, b"BROKEN\r\n\r\n")
                .await
                .unwrap();
            let mut malformed = Vec::new();
            tokio::io::AsyncReadExt::read_to_end(&mut socket, &mut malformed)
                .await
                .unwrap();
            let malformed = response_status(&malformed);
            let wrong_path = reqwest::get(format!("http://{address}/wrong-path"))
                .await
                .unwrap()
                .status()
                .as_u16();
            let callback = reqwest::get(format!(
                "http://{address}/auth/callback?code=callback_code&state={state}"
            ))
            .await
            .unwrap()
            .status()
            .as_u16();
            vec![malformed, wrong_path, callback]
        });
        *self.task.lock().unwrap() = Some(task);
    }
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

fn credential_access(credential: &Credential) -> String {
    let Credential::OAuth { access, .. } = credential else {
        unreachable!();
    };
    access.clone()
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

fn response_status(response: &[u8]) -> u16 {
    String::from_utf8_lossy(response)
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}

async fn serve_delayed_device_start(delay: Duration) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (mut start, _) = listener.accept().await.unwrap();
        read_http_request(&mut start).await;
        tokio::time::sleep(delay).await;
        write_json_response(
            &mut start,
            json!({
                "device_auth_id": "device_delayed",
                "user_code": "DELAYED",
                "interval": 0
            }),
        )
        .await;

        let (mut poll, _) = listener.accept().await.unwrap();
        read_http_request(&mut poll).await;
        write_json_response(
            &mut poll,
            json!({
                "authorization_code": "device_code",
                "code_verifier": "device_verifier"
            }),
        )
        .await;

        let (mut exchange, _) = listener.accept().await.unwrap();
        read_http_request(&mut exchange).await;
        write_json_response(
            &mut exchange,
            json!({
                "access_token": token("acc_delayed_device"),
                "refresh_token": "refresh_delayed_device",
                "expires_in": 3600
            }),
        )
        .await;
    });
    (format!("http://{address}"), task)
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) {
    let mut request = Vec::new();
    let header_end = loop {
        let mut bytes = [0; 1024];
        let count = tokio::io::AsyncReadExt::read(socket, &mut bytes)
            .await
            .unwrap();
        request.extend_from_slice(&bytes[..count]);
        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or_default();
    while request.len() < header_end + content_length {
        let mut bytes = [0; 1024];
        let count = tokio::io::AsyncReadExt::read(socket, &mut bytes)
            .await
            .unwrap();
        request.extend_from_slice(&bytes[..count]);
    }
}

async fn write_json_response(socket: &mut tokio::net::TcpStream, body: serde_json::Value) {
    let body = serde_json::to_vec(&body).unwrap();
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    tokio::io::AsyncWriteExt::write_all(socket, response.as_bytes())
        .await
        .unwrap();
    tokio::io::AsyncWriteExt::write_all(socket, &body)
        .await
        .unwrap();
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
