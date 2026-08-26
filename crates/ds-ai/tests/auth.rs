use async_trait::async_trait;
use ds_ai::{
    ApiKeyAuth, AuthContext, AuthError, Credential, CredentialStore, CredentialType, EnvApiKeyAuth,
    InMemoryCredentialStore,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio_util::sync::CancellationToken;

#[test]
fn serializes_oauth_with_pi_compatible_tag_and_reads_legacy_tag() {
    let credential = Credential::OAuth {
        refresh: "refresh".into(),
        access: "access".into(),
        expires: 123,
        extra: BTreeMap::new(),
    };

    let serialized = serde_json::to_value(&credential).unwrap();
    assert_eq!(serialized["type"], "oauth");
    assert_eq!(
        serde_json::to_value(CredentialType::OAuth).unwrap(),
        "oauth"
    );

    let mut legacy = serialized;
    legacy["type"] = "o_auth".into();
    assert_eq!(
        serde_json::from_value::<Credential>(legacy).unwrap(),
        credential
    );
}

#[tokio::test]
async fn stores_lists_updates_and_deletes_credentials_without_exposing_secrets() {
    let store = InMemoryCredentialStore::new();
    let cancellation = CancellationToken::new();
    assert_eq!(store.read("openai", &cancellation).await.unwrap(), None);

    store
        .modify(
            "openai",
            Box::new(|current| {
                Box::pin(async move {
                    assert_eq!(current, None);
                    Ok(Some(api_key("secret")))
                })
            }),
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(
        store.list(&cancellation).await.unwrap(),
        [ds_ai::CredentialInfo {
            provider_id: "openai".into(),
            credential_type: CredentialType::ApiKey,
        }]
    );

    let unchanged = store
        .modify(
            "openai",
            Box::new(|_| Box::pin(async { Ok(None) })),
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(unchanged, Some(api_key("secret")));

    store.delete("openai", &cancellation).await.unwrap();
    assert_eq!(store.read("openai", &cancellation).await.unwrap(), None);
}

#[tokio::test]
async fn serializes_mutation_and_preserves_credentials_after_failure() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let cancellation = CancellationToken::new();
    store
        .modify(
            "openai",
            Box::new(|_| Box::pin(async { Ok(Some(api_key("initial"))) })),
            &cancellation,
        )
        .await
        .unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::new();
    for key in ["first", "second"] {
        let store = store.clone();
        let cancellation = cancellation.clone();
        let active = active.clone();
        let maximum = maximum.clone();
        tasks.push(tokio::spawn(async move {
            store
                .modify(
                    "openai",
                    Box::new(move |_| {
                        Box::pin(async move {
                            let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                            maximum.fetch_max(count, Ordering::SeqCst);
                            tokio::task::yield_now().await;
                            active.fetch_sub(1, Ordering::SeqCst);
                            Ok(Some(api_key(key)))
                        })
                    }),
                    &cancellation,
                )
                .await
                .unwrap();
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    assert_eq!(maximum.load(Ordering::SeqCst), 1);

    let before = store.read("openai", &cancellation).await.unwrap();
    let error = store
        .modify(
            "openai",
            Box::new(|_| Box::pin(async { Err(AuthError::OAuth("refresh rejected".into())) })),
            &cancellation,
        )
        .await;
    assert!(matches!(error, Err(AuthError::OAuth(_))));
    assert_eq!(store.read("openai", &cancellation).await.unwrap(), before);
}

#[tokio::test]
async fn cancels_queued_mutation_without_running_it_later() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let (started_sender, started_receiver) = tokio::sync::oneshot::channel();
    let (finish_sender, finish_receiver) = tokio::sync::oneshot::channel();
    let first_store = store.clone();
    let first = tokio::spawn(async move {
        first_store
            .modify(
                "openai",
                Box::new(move |_| {
                    Box::pin(async move {
                        started_sender.send(()).unwrap();
                        finish_receiver.await.unwrap();
                        Ok(Some(api_key("first")))
                    })
                }),
                &CancellationToken::new(),
            )
            .await
    });
    started_receiver.await.unwrap();

    let cancellation = CancellationToken::new();
    let second_ran = Arc::new(AtomicUsize::new(0));
    let second_store = store.clone();
    let second_ran_clone = second_ran.clone();
    let second_cancellation = cancellation.clone();
    let second = tokio::spawn(async move {
        second_store
            .modify(
                "openai",
                Box::new(move |_| {
                    Box::pin(async move {
                        second_ran_clone.fetch_add(1, Ordering::SeqCst);
                        Ok(Some(api_key("second")))
                    })
                }),
                &second_cancellation,
            )
            .await
    });
    cancellation.cancel();

    assert!(matches!(second.await.unwrap(), Err(AuthError::Cancelled)));
    finish_sender.send(()).unwrap();
    assert!(first.await.unwrap().is_ok());
    assert_eq!(second_ran.load(Ordering::SeqCst), 0);
    assert_eq!(
        store
            .read("openai", &CancellationToken::new())
            .await
            .unwrap(),
        Some(api_key("first"))
    );
}

#[tokio::test]
async fn cancels_a_mutation_before_committing_its_result() {
    let store = InMemoryCredentialStore::new();
    let cancellation = CancellationToken::new();
    store
        .modify(
            "openai",
            Box::new(|_| Box::pin(async { Ok(Some(api_key("initial"))) })),
            &cancellation,
        )
        .await
        .unwrap();

    let mutation_cancellation = cancellation.clone();
    let result = store
        .modify(
            "openai",
            Box::new(move |_| {
                Box::pin(async move {
                    mutation_cancellation.cancel();
                    Ok(Some(api_key("next")))
                })
            }),
            &cancellation,
        )
        .await;

    assert!(matches!(result, Err(AuthError::Cancelled)));
    assert_eq!(
        store
            .read("openai", &CancellationToken::new())
            .await
            .unwrap(),
        Some(api_key("initial"))
    );
}

#[tokio::test]
async fn resolves_stored_keys_before_ambient_keys() {
    let auth = EnvApiKeyAuth::new("API key", ["TEST_API_KEY"]);
    let context = TestContext(BTreeMap::from([("TEST_API_KEY".into(), "ambient".into())]));
    let cancellation = CancellationToken::new();

    let stored = auth
        .resolve(&context, Some(&api_key("stored")), &cancellation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.auth.api_key.as_deref(), Some("stored"));
    assert_eq!(stored.source.as_deref(), Some("stored credential"));

    let ambient = auth
        .resolve(&context, None, &cancellation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(ambient.auth.api_key.as_deref(), Some("ambient"));
    assert_eq!(ambient.source.as_deref(), Some("TEST_API_KEY"));
}

#[tokio::test]
async fn rejects_whitespace_ambient_keys_without_rewriting_stored_keys() {
    let auth = EnvApiKeyAuth::new("API key", ["TEST_API_KEY"]);
    let empty = TestContext(BTreeMap::from([("TEST_API_KEY".into(), "".into())]));
    let cancellation = CancellationToken::new();

    assert_eq!(
        auth.resolve(&empty, Some(&api_key("")), &cancellation)
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        auth.resolve(&empty, None, &cancellation).await.unwrap(),
        None
    );

    let stored = auth
        .resolve(&empty, Some(&api_key("   ")), &cancellation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.auth.api_key.as_deref(), Some("   "));

    let whitespace = TestContext(BTreeMap::from([("TEST_API_KEY".into(), "   ".into())]));
    assert_eq!(
        auth.resolve(&whitespace, None, &cancellation)
            .await
            .unwrap(),
        None
    );
}

fn api_key(key: &str) -> Credential {
    Credential::ApiKey {
        key: Some(key.into()),
        env: BTreeMap::new(),
    }
}

struct TestContext(BTreeMap<String, String>);

#[async_trait]
impl AuthContext for TestContext {
    async fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    async fn file_exists(&self, _path: &str) -> bool {
        false
    }
}
