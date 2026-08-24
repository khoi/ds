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
