use crate::support::{Reply, serve};
use base64::prelude::*;
use ds_ai::codex::auth::{Client, CredentialManager, Credentials};
use serde_json::json;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
    let client = Client::new().with_base_url(&server.base_url);
    let before = now_millis();

    let credentials = client
        .refresh("refresh_old", &CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(credentials.refresh_token, "refresh_next");
    assert_eq!(credentials.account_id().unwrap(), "acc_refreshed");
    assert!(credentials.expires_at >= before + 3_600_000);
    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /oauth/token HTTP/1.1\r\n"));
    let body = request.split("\r\n\r\n").nth(1).unwrap();
    assert!(body.contains("grant_type=refresh_token"));
    assert!(body.contains("refresh_token=refresh_old"));
    assert!(body.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
}

#[tokio::test]
async fn serializes_concurrent_codex_refreshes() {
    let server = serve([Reply::json(
        200,
        json!({
            "access_token": token("acc_concurrent"),
            "refresh_token": "refresh_new",
            "expires_in": 3600
        }),
    )])
    .await;
    let manager = CredentialManager::new(
        Client::new().with_base_url(&server.base_url),
        Credentials {
            access_token: token("acc_expired"),
            refresh_token: "refresh_old".into(),
            expires_at: 0,
        },
    );
    let first = manager.clone();
    let second = manager.clone();
    let first = tokio::spawn(async move { first.access_token(&CancellationToken::new()).await });
    let second = tokio::spawn(async move { second.access_token(&CancellationToken::new()).await });

    let (first, second) = tokio::join!(first, second);

    assert_eq!(first.unwrap().unwrap(), second.unwrap().unwrap());
    assert_eq!(manager.snapshot().await.refresh_token, "refresh_new");
    assert_eq!(server.requests().await.len(), 1);
}

#[tokio::test]
async fn preserves_codex_refresh_failure_details() {
    let server = serve([Reply::json(
        500,
        json!({"error": {"code": "invalid_grant", "message": "expired"}}),
    )])
    .await;
    let client = Client::new().with_base_url(&server.base_url);

    let error = client
        .refresh("refresh_bad", &CancellationToken::new())
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        ds_ai::codex::auth::Error::Server {
            operation: "refresh",
            status: 500,
            body,
        } if body.contains("invalid_grant") && body.contains("expired")
    ));
    server.requests().await;
}

#[tokio::test]
async fn cancels_an_active_codex_refresh() {
    let server = serve([Reply::pending()]).await;
    let client = Client::new().with_base_url(&server.base_url);
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let refresh =
        tokio::spawn(async move { client.refresh("refresh_wait", &task_cancellation).await });
    server.wait_for_requests(1).await;

    cancellation.cancel();

    assert!(matches!(
        refresh.await.unwrap(),
        Err(ds_ai::codex::auth::Error::Cancelled)
    ));
    server.requests().await;
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

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
