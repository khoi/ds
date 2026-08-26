use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex as StdMutex},
};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

pub(crate) mod resolution;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    ApiKey {
        #[serde(skip_serializing_if = "Option::is_none")]
        key: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        env: BTreeMap<String, String>,
    },
    #[serde(rename = "oauth", alias = "o_auth")]
    OAuth {
        refresh: String,
        access: String,
        expires: u64,
        #[serde(default, flatten)]
        extra: BTreeMap<String, serde_json::Value>,
    },
}

impl Credential {
    pub fn credential_type(&self) -> CredentialType {
        match self {
            Self::ApiKey { .. } => CredentialType::ApiKey,
            Self::OAuth { .. } => CredentialType::OAuth,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialType {
    ApiKey,
    #[serde(rename = "oauth", alias = "o_auth")]
    OAuth,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredentialInfo {
    pub provider_id: String,
    pub credential_type: CredentialType,
}

pub type CredentialMutation = Box<
    dyn FnOnce(
            Option<Credential>,
        )
            -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AuthError>> + Send>>
        + Send,
>;

#[async_trait]
pub trait CredentialStore: Send + Sync {
    async fn read(
        &self,
        provider_id: &str,
        cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError>;
    async fn list(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<Vec<CredentialInfo>, AuthError>;
    async fn modify(
        &self,
        provider_id: &str,
        mutation: CredentialMutation,
        cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError>;
    async fn delete(
        &self,
        provider_id: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), AuthError>;
}

#[derive(Default)]
pub struct InMemoryCredentialStore {
    credentials: RwLock<BTreeMap<String, Credential>>,
    locks: StdMutex<BTreeMap<String, Arc<Mutex<()>>>>,
}

impl InMemoryCredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn provider_lock(&self, provider_id: &str) -> Arc<Mutex<()>> {
        self.locks
            .lock()
            .expect("credential lock map")
            .entry(provider_id.to_owned())
            .or_default()
            .clone()
    }
}

#[async_trait]
impl CredentialStore for InMemoryCredentialStore {
    async fn read(
        &self,
        provider_id: &str,
        cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        cancelled(cancellation)?;
        Ok(self.credentials.read().await.get(provider_id).cloned())
    }

    async fn list(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<Vec<CredentialInfo>, AuthError> {
        cancelled(cancellation)?;
        Ok(self
            .credentials
            .read()
            .await
            .iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id: provider_id.clone(),
                credential_type: credential.credential_type(),
            })
            .collect())
    }

    async fn modify(
        &self,
        provider_id: &str,
        mutation: CredentialMutation,
        cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        let provider_lock = self.provider_lock(provider_id);
        let _guard = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(AuthError::Cancelled),
            guard = provider_lock.lock() => guard,
        };
        let current = self.credentials.read().await.get(provider_id).cloned();
        let next = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(AuthError::Cancelled),
            next = mutation(current.clone()) => next?,
        };
        cancelled(cancellation)?;
        if let Some(next) = next {
            let mut credentials = self.credentials.write().await;
            cancelled(cancellation)?;
            credentials.insert(provider_id.to_owned(), next.clone());
            Ok(Some(next))
        } else {
            Ok(current)
        }
    }

    async fn delete(
        &self,
        provider_id: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), AuthError> {
        let provider_lock = self.provider_lock(provider_id);
        let _guard = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(AuthError::Cancelled),
            guard = provider_lock.lock() => guard,
        };
        let mut credentials = self.credentials.write().await;
        cancelled(cancellation)?;
        credentials.remove(provider_id);
        Ok(())
    }
}

#[async_trait]
pub trait AuthContext: Send + Sync {
    async fn env(&self, name: &str) -> Option<String>;
    async fn file_exists(&self, path: &str) -> bool;
}

pub struct SystemAuthContext;

#[async_trait]
impl AuthContext for SystemAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    }

    async fn file_exists(&self, path: &str) -> bool {
        expand_home(path).is_some_and(|path| path.exists())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelAuth {
    pub api_key: Option<String>,
    pub headers: BTreeMap<String, Option<String>>,
    pub base_url: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AuthResult {
    pub auth: ModelAuth,
    pub env: BTreeMap<String, String>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct AuthResolutionOverrides {
    pub api_key: Option<String>,
    pub env: BTreeMap<String, String>,
    pub minimum_oauth_validity: Option<std::time::Duration>,
    pub cancellation: CancellationToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthCheck {
    pub source: Option<String>,
    pub credential_type: CredentialType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthPrompt {
    Text {
        message: String,
        placeholder: Option<String>,
    },
    Secret {
        message: String,
        placeholder: Option<String>,
    },
    Select {
        message: String,
        options: Vec<AuthSelectOption>,
    },
    ManualCode {
        message: String,
        placeholder: Option<String>,
        cancellation: CancellationToken,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthSelectOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthEvent {
    Info {
        message: String,
        links: Vec<AuthInfoLink>,
    },
    AuthUrl {
        url: String,
        instructions: Option<String>,
    },
    DeviceCode {
        user_code: String,
        verification_uri: String,
        interval_seconds: Option<u64>,
        expires_in_seconds: Option<u64>,
    },
    Progress {
        message: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthInfoLink {
    pub url: String,
    pub label: Option<String>,
}

#[async_trait]
pub trait AuthInteraction: Send + Sync {
    fn cancellation(&self) -> &CancellationToken;
    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError>;
    fn notify(&self, event: AuthEvent);
}

#[async_trait]
pub trait ApiKeyAuth: Send + Sync {
    fn name(&self) -> &str;
    async fn login(&self, _interaction: &dyn AuthInteraction) -> Result<Credential, AuthError> {
        Err(AuthError::Unsupported("API key login".into()))
    }
    async fn check(
        &self,
        context: &dyn AuthContext,
        credential: Option<&Credential>,
        cancellation: &CancellationToken,
    ) -> Result<Option<AuthCheck>, AuthError> {
        Ok(self
            .resolve(context, credential, cancellation)
            .await?
            .map(|result| AuthCheck {
                source: result.source,
                credential_type: CredentialType::ApiKey,
            }))
    }
    async fn resolve(
        &self,
        context: &dyn AuthContext,
        credential: Option<&Credential>,
        cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError>;
}

#[async_trait]
pub trait OAuthAuth: Send + Sync {
    fn name(&self) -> &str;
    fn is_subscription(&self) -> bool {
        false
    }
    async fn login(&self, interaction: &dyn AuthInteraction) -> Result<Credential, AuthError>;
    async fn refresh(
        &self,
        credential: &Credential,
        cancellation: &CancellationToken,
    ) -> Result<Credential, AuthError>;
    async fn to_auth(&self, credential: &Credential) -> Result<ModelAuth, AuthError>;
}

#[derive(Clone, Default)]
pub struct ProviderAuth {
    pub api_key: Option<Arc<dyn ApiKeyAuth>>,
    pub oauth: Option<Arc<dyn OAuthAuth>>,
}

impl ProviderAuth {
    pub fn api_key(auth: impl ApiKeyAuth + 'static) -> Self {
        Self {
            api_key: Some(Arc::new(auth)),
            oauth: None,
        }
    }

    pub fn oauth(auth: impl OAuthAuth + 'static) -> Self {
        Self {
            api_key: None,
            oauth: Some(Arc::new(auth)),
        }
    }
}

pub struct EnvApiKeyAuth {
    name: String,
    env_names: Vec<String>,
}

impl EnvApiKeyAuth {
    pub fn new(
        name: impl Into<String>,
        env_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            env_names: env_names.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl ApiKeyAuth for EnvApiKeyAuth {
    fn name(&self) -> &str {
        &self.name
    }

    async fn login(&self, interaction: &dyn AuthInteraction) -> Result<Credential, AuthError> {
        cancelled(interaction.cancellation())?;
        let key = interaction
            .prompt(AuthPrompt::Secret {
                message: format!("Enter {}", self.name),
                placeholder: None,
            })
            .await?;
        cancelled(interaction.cancellation())?;
        Ok(Credential::ApiKey {
            key: Some(key),
            env: BTreeMap::new(),
        })
    }

    async fn resolve(
        &self,
        context: &dyn AuthContext,
        credential: Option<&Credential>,
        cancellation: &CancellationToken,
    ) -> Result<Option<AuthResult>, AuthError> {
        cancelled(cancellation)?;
        if let Some(Credential::ApiKey {
            key: Some(key),
            env,
        }) = credential
            && !key.is_empty()
        {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key.clone()),
                    ..Default::default()
                },
                env: env.clone(),
                source: Some("stored credential".into()),
            }));
        }
        for name in &self.env_names {
            if let Some(key) = context.env(name).await.filter(|key| !key.trim().is_empty()) {
                cancelled(cancellation)?;
                return Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(key),
                        ..Default::default()
                    },
                    source: Some(name.clone()),
                    ..Default::default()
                }));
            }
        }
        Ok(None)
    }
}

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("authentication operation cancelled")]
    Cancelled,
    #[error("credential store failed: {0}")]
    Store(String),
    #[error("{0}")]
    Provider(String),
    #[error("authentication failed: {0}")]
    Authentication(String),
    #[error("OAuth failed: {0}")]
    OAuth(String),
    #[error("unsupported authentication operation: {0}")]
    Unsupported(String),
}

fn cancelled(cancellation: &CancellationToken) -> Result<(), AuthError> {
    if cancellation.is_cancelled() {
        Err(AuthError::Cancelled)
    } else {
        Ok(())
    }
}

fn expand_home(path: &str) -> Option<PathBuf> {
    match path.strip_prefix("~/") {
        Some(suffix) => {
            let mut home = PathBuf::from(std::env::var_os("HOME")?);
            home.push(suffix);
            Some(home)
        }
        None => Some(PathBuf::from(path)),
    }
}
