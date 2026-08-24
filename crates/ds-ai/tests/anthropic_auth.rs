use crate::support::{Reply, serve};
use async_trait::async_trait;
use base64::prelude::*;
use ds_ai::{AuthError, AuthEvent, AuthInteraction, AuthPrompt, Credential, ModelAuth, OAuthAuth};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{collections::BTreeMap, net::SocketAddr, sync::Mutex};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn exposes_anthropic_subscription_oauth() {
    let provider = ds_ai::anthropic::provider();
    let oauth = provider.auth().oauth.as_ref().unwrap();
    let credential = Credential::OAuth {
        refresh: "refresh".into(),
        access: "access".into(),
        expires: 0,
        extra: BTreeMap::new(),
    };

    assert_eq!(oauth.name(), "Anthropic (Claude Pro/Max)");
    assert!(oauth.is_subscription());
    assert_eq!(
        oauth.to_auth(&credential).await.unwrap(),
        ModelAuth {
            api_key: Some("access".into()),
            ..Default::default()
        }
    );
}

#[tokio::test]
async fn refreshes_anthropic_oauth_without_a_scope() {
    let server = serve([Reply::json(
        200,
        serde_json::json!({
            "access_token": "access_next",
            "refresh_token": "refresh_next",
            "expires_in": 3600,
            "scope": "ignored"
        }),
    )])
    .await;
    let oauth = ds_ai::anthropic::auth::OAuth::new()
        .with_token_url(format!("{}/v1/oauth/token", server.base_url));
    let credential = Credential::OAuth {
        refresh: "refresh_old".into(),
        access: "access_old".into(),
        expires: 0,
        extra: BTreeMap::new(),
    };
    let before = now_millis();

    let refreshed = oauth
        .refresh(&credential, &CancellationToken::new())
        .await
        .unwrap();

    assert!(matches!(
        refreshed,
        Credential::OAuth {
            refresh,
            access,
            expires,
            extra,
        } if refresh == "refresh_next"
            && access == "access_next"
            && expires >= before + 3_300_000
            && extra.is_empty()
    ));
    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /v1/oauth/token HTTP/1.1\r\n"));
    let body: serde_json::Value =
        serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
            "refresh_token": "refresh_old"
        })
    );
}

#[tokio::test]
async fn preserves_anthropic_oauth_refresh_failure_details() {
    let server = serve([Reply::json(
        500,
        serde_json::json!({"error": "invalid_grant", "message": "expired"}),
    )])
    .await;
    let token_url = format!("{}/v1/oauth/token", server.base_url);
    let oauth = ds_ai::anthropic::auth::OAuth::new().with_token_url(&token_url);
    let credential = Credential::OAuth {
        refresh: "refresh_bad".into(),
        access: "access_old".into(),
        expires: 0,
        extra: BTreeMap::new(),
    };

    let error = oauth
        .refresh(&credential, &CancellationToken::new())
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("HTTP 500"));
    assert!(error.contains(&token_url));
    assert!(error.contains("invalid_grant"));
    assert!(error.contains("expired"));
    server.requests().await;
}

#[tokio::test]
async fn cancels_anthropic_oauth_while_reading_the_token_body() {
    let server = serve([Reply::open_json(
        200,
        serde_json::json!({"access_token": "unfinished"}),
    )])
    .await;
    let oauth = ds_ai::anthropic::auth::OAuth::new()
        .with_token_url(format!("{}/v1/oauth/token", server.base_url));
    let credential = Credential::OAuth {
        refresh: "refresh_wait".into(),
        access: "access_old".into(),
        expires: 0,
        extra: BTreeMap::new(),
    };
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let refresh = tokio::spawn(async move { oauth.refresh(&credential, &task_cancellation).await });
    server.wait_for_requests(1).await;
    tokio::task::yield_now().await;

    cancellation.cancel();

    assert!(matches!(refresh.await.unwrap(), Err(AuthError::Cancelled)));
    server.requests().await;
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap()
}

#[tokio::test]
async fn logs_in_to_anthropic_oauth_with_a_manual_redirect() {
    let server = serve([Reply::json(
        200,
        serde_json::json!({
            "access_token": "access_manual",
            "refresh_token": "refresh_manual",
            "expires_in": 3600
        }),
    )])
    .await;
    let interaction = ManualInteraction::default();
    let oauth = ds_ai::anthropic::auth::OAuth::new()
        .with_token_url(format!("{}/v1/oauth/token", server.base_url))
        .with_callback_address(SocketAddr::from(([127, 0, 0, 1], 0)));

    let credential = oauth.login(&interaction).await.unwrap();

    assert!(matches!(
        credential,
        Credential::OAuth {
            refresh,
            access,
            ..
        } if refresh == "refresh_manual" && access == "access_manual"
    ));
    let authorization_url = url::Url::parse(&interaction.authorization_url()).unwrap();
    assert_eq!(
        authorization_url.as_str().split('?').next().unwrap(),
        "https://claude.ai/oauth/authorize"
    );
    assert_eq!(query(&authorization_url, "code"), "true");
    assert_eq!(query(&authorization_url, "response_type"), "code");
    assert_eq!(
        query(&authorization_url, "redirect_uri"),
        "http://localhost:53692/callback"
    );
    assert_eq!(
        query(&authorization_url, "scope"),
        "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
    );
    assert_eq!(query(&authorization_url, "code_challenge_method"), "S256");
    let request = server.requests().await.pop().unwrap();
    let body: serde_json::Value =
        serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    let verifier = body["code_verifier"].as_str().unwrap();
    assert_eq!(body["grant_type"], "authorization_code");
    assert_eq!(body["code"], "manual_code");
    assert_eq!(body["state"], verifier);
    assert_eq!(body["redirect_uri"], "http://localhost:53692/callback");
    assert_eq!(query(&authorization_url, "state"), verifier);
    assert_eq!(
        query(&authorization_url, "code_challenge"),
        BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
    );
    assert!(interaction.has_exchange_progress());
    assert!(interaction.prompt_was_cancelled());
}

#[tokio::test]
async fn logs_in_to_anthropic_oauth_through_the_browser_callback() {
    let server = serve([Reply::json(
        200,
        serde_json::json!({
            "access_token": "access_browser",
            "refresh_token": "refresh_browser",
            "expires_in": 3600
        }),
    )])
    .await;
    let callback_address = local_address();
    let interaction = CallbackInteraction::default();
    let oauth = ds_ai::anthropic::auth::OAuth::new()
        .with_token_url(format!("{}/v1/oauth/token", server.base_url))
        .with_callback_address(callback_address)
        .with_redirect_uri(format!(
            "http://localhost:{}/callback",
            callback_address.port()
        ));

    let credential = oauth.login(&interaction).await.unwrap();

    assert!(matches!(
        credential,
        Credential::OAuth {
            refresh,
            access,
            ..
        } if refresh == "refresh_browser" && access == "access_browser"
    ));
    assert!(interaction.prompt_was_cancelled());
    let request = server.requests().await.pop().unwrap();
    let body: serde_json::Value =
        serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["code"], "callback_code");
    assert_eq!(body["state"], body["code_verifier"]);
}

#[derive(Default)]
struct ManualInteraction {
    cancellation: CancellationToken,
    events: Mutex<Vec<AuthEvent>>,
    prompt_cancellation: Mutex<Option<CancellationToken>>,
}

impl ManualInteraction {
    fn authorization_url(&self) -> String {
        self.events
            .lock()
            .unwrap()
            .iter()
            .find_map(|event| match event {
                AuthEvent::AuthUrl { url, .. } => Some(url.clone()),
                _ => None,
            })
            .unwrap()
    }

    fn has_exchange_progress(&self) -> bool {
        self.events.lock().unwrap().iter().any(|event| {
            matches!(event, AuthEvent::Progress { message } if message == "Exchanging authorization code for tokens...")
        })
    }

    fn prompt_was_cancelled(&self) -> bool {
        self.prompt_cancellation
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }
}

#[async_trait]
impl AuthInteraction for ManualInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError> {
        let AuthPrompt::ManualCode { cancellation, .. } = prompt else {
            return Err(AuthError::Authentication("unexpected prompt".into()));
        };
        *self.prompt_cancellation.lock().unwrap() = Some(cancellation);
        let authorization_url = url::Url::parse(&self.authorization_url()).unwrap();
        Ok(format!(
            "{}?code=manual_code&state={}",
            query(&authorization_url, "redirect_uri"),
            query(&authorization_url, "state")
        ))
    }

    fn notify(&self, event: AuthEvent) {
        self.events.lock().unwrap().push(event);
    }
}

#[derive(Default)]
struct CallbackInteraction {
    cancellation: CancellationToken,
    prompt_cancellation: Mutex<Option<CancellationToken>>,
}

impl CallbackInteraction {
    fn prompt_was_cancelled(&self) -> bool {
        self.prompt_cancellation
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }
}

#[async_trait]
impl AuthInteraction for CallbackInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError> {
        let AuthPrompt::ManualCode { cancellation, .. } = prompt else {
            return Err(AuthError::Authentication("unexpected prompt".into()));
        };
        *self.prompt_cancellation.lock().unwrap() = Some(cancellation.clone());
        cancellation.cancelled().await;
        Err(AuthError::Cancelled)
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

fn query(url: &url::Url, name: &str) -> String {
    url.query_pairs()
        .find_map(|(key, value)| (key == name).then(|| value.into_owned()))
        .unwrap()
}

fn local_address() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}
