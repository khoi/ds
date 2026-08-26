use async_trait::async_trait;
use ds_ai::{AuthError, Credential, CredentialInfo, CredentialMutation, CredentialStore};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex as StdMutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

const MAX_AUTH_BYTES: usize = 1024 * 1024;
const LOCK_RETRY_DELAY: Duration = Duration::from_millis(20);

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
static PROCESS_LOCKS: OnceLock<StdMutex<BTreeMap<PathBuf, Weak<Mutex<()>>>>> = OnceLock::new();

/// A file-backed credential store with Pi-style provider-neutral credentials.
///
/// The auth file is a JSON object whose keys are provider IDs and whose values
/// are ds_ai::Credential values. The path is deliberately supplied by the
/// caller so that path discovery and profile selection remain outside this
/// storage implementation.
pub struct PersistentCredentialStore {
    auth_path: PathBuf,
    lock_path: PathBuf,
    process_lock: Arc<Mutex<()>>,
}

impl PersistentCredentialStore {
    /// Creates a store using explicit auth and lock paths.
    ///
    /// The parent directories are created with private permissions on Unix,
    /// and the lock file is created with private permissions. An existing auth
    /// file is checked before the store is returned so an insecure file cannot
    /// be accidentally used.
    pub fn new(
        auth_path: impl Into<PathBuf>,
        lock_path: impl Into<PathBuf>,
    ) -> Result<Self, AuthError> {
        let auth_path = auth_path.into();
        let lock_path = lock_path.into();

        if auth_path == lock_path {
            return Err(store_error(
                "initializing credential store",
                &auth_path,
                "auth and lock paths must differ",
            ));
        }

        ensure_parent_directory(&auth_path, "auth")?;
        ensure_parent_directory(&lock_path, "lock")?;
        validate_auth_path(&auth_path)?;

        let lock_file = open_lock_file(&lock_path)?;
        drop(lock_file);
        let lock_path = fs::canonicalize(&lock_path)
            .map_err(|error| store_error("resolving credential lock path", &lock_path, error))?;

        Ok(Self {
            process_lock: process_lock_for(&lock_path),
            auth_path,
            lock_path,
        })
    }

    /// Returns the explicit path used for the credential JSON file.
    pub fn auth_path(&self) -> &Path {
        &self.auth_path
    }

    /// Returns the explicit path used for the cross-process lock file.
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    async fn lock_process<'a>(
        &'a self,
        cancellation: &CancellationToken,
    ) -> Result<tokio::sync::MutexGuard<'a, ()>, AuthError> {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(AuthError::Cancelled),
            guard = self.process_lock.lock() => Ok(guard),
        }
    }

    async fn lock_file(&self, cancellation: &CancellationToken) -> Result<File, AuthError> {
        let lock_path = self.lock_path.clone();
        let lock_file = blocking(cancellation, "opening credential lock", move || {
            open_lock_file(&lock_path)
        })
        .await?;

        loop {
            if cancellation.is_cancelled() {
                return Err(AuthError::Cancelled);
            }

            match fs4::FileExt::try_lock(&lock_file) {
                Ok(()) => return Ok(lock_file),
                Err(fs4::TryLockError::WouldBlock) => {
                    tokio::select! {
                        biased;
                        _ = cancellation.cancelled() => return Err(AuthError::Cancelled),
                        _ = tokio::time::sleep(LOCK_RETRY_DELAY) => {}
                    }
                }
                Err(fs4::TryLockError::Error(error)) => {
                    return Err(store_error("locking credentials", &self.lock_path, error));
                }
            }
        }
    }

    async fn load(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<BTreeMap<String, Credential>, AuthError> {
        let auth_path = self.auth_path.clone();
        blocking(cancellation, "reading credentials", move || {
            read_credentials(&auth_path)
        })
        .await
    }

    fn persist(&self, credentials: &BTreeMap<String, Credential>) -> Result<(), AuthError> {
        write_credentials(&self.auth_path, credentials)
    }
}

#[async_trait]
impl CredentialStore for PersistentCredentialStore {
    async fn read(
        &self,
        provider_id: &str,
        cancellation: &CancellationToken,
    ) -> Result<Option<Credential>, AuthError> {
        let _process_guard = self.lock_process(cancellation).await?;
        let credentials = self.load(cancellation).await?;
        Ok(credentials.get(provider_id).cloned())
    }

    async fn list(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<Vec<CredentialInfo>, AuthError> {
        let _process_guard = self.lock_process(cancellation).await?;
        let credentials = self.load(cancellation).await?;
        Ok(credentials
            .into_iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id,
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
        let _process_guard = self.lock_process(cancellation).await?;
        let _file_guard = self.lock_file(cancellation).await?;
        let mut credentials = self.load(cancellation).await?;
        let current = credentials.get(provider_id).cloned();

        let next = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(AuthError::Cancelled),
            next = mutation(current.clone()) => next?,
        };

        if cancellation.is_cancelled() {
            return Err(AuthError::Cancelled);
        }
        if let Some(next) = next {
            credentials.insert(provider_id.to_owned(), next.clone());
            self.persist(&credentials)?;
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
        let _process_guard = self.lock_process(cancellation).await?;
        let _file_guard = self.lock_file(cancellation).await?;
        let mut credentials = self.load(cancellation).await?;
        if cancellation.is_cancelled() {
            return Err(AuthError::Cancelled);
        }
        if credentials.remove(provider_id).is_none() {
            return Ok(());
        }
        self.persist(&credentials)
    }
}

async fn blocking<T, F>(
    cancellation: &CancellationToken,
    operation: &'static str,
    function: F,
) -> Result<T, AuthError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, AuthError> + Send + 'static,
{
    if cancellation.is_cancelled() {
        return Err(AuthError::Cancelled);
    }
    let task = tokio::task::spawn_blocking(function);
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(AuthError::Cancelled),
        result = task => result.map_err(|error| AuthError::Store(format!("{operation} task failed: {error}")))?,
    }
}

fn process_lock_for(path: &Path) -> Arc<Mutex<()>> {
    let registry = PROCESS_LOCKS.get_or_init(|| StdMutex::new(BTreeMap::new()));
    let mut registry = match registry.lock() {
        Ok(registry) => registry,
        Err(poisoned) => poisoned.into_inner(),
    };
    registry.retain(|_, lock| lock.strong_count() != 0);
    if let Some(lock) = registry.get(path).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    registry.insert(path.to_path_buf(), Arc::downgrade(&lock));
    lock
}

fn read_credentials(path: &Path) -> Result<BTreeMap<String, Credential>, AuthError> {
    if !validate_auth_path(path)? {
        return Ok(BTreeMap::new());
    }

    let file = File::open(path).map_err(|error| store_error("reading credentials", path, error))?;
    let size = file
        .metadata()
        .map_err(|error| store_error("reading credential metadata", path, error))?
        .len();
    if size > MAX_AUTH_BYTES as u64 {
        return Err(store_error(
            "reading credentials",
            path,
            format!("credential file exceeds {MAX_AUTH_BYTES} bytes"),
        ));
    }

    let mut contents = Vec::with_capacity(size as usize);
    file.take((MAX_AUTH_BYTES + 1) as u64)
        .read_to_end(&mut contents)
        .map_err(|error| store_error("reading credentials", path, error))?;
    if contents.len() > MAX_AUTH_BYTES {
        return Err(store_error(
            "reading credentials",
            path,
            format!("credential file exceeds {MAX_AUTH_BYTES} bytes"),
        ));
    }

    serde_json::from_slice(&contents)
        .map_err(|error| store_error("parsing credentials", path, error))
}

fn write_credentials(
    path: &Path,
    credentials: &BTreeMap<String, Credential>,
) -> Result<(), AuthError> {
    validate_auth_path(path)?;
    let parent = parent_directory(path);
    let serialized = serde_json::to_vec_pretty(credentials)
        .map_err(|error| store_error("serializing credentials", path, error))?;
    if serialized.len() > MAX_AUTH_BYTES {
        return Err(store_error(
            "serializing credentials",
            path,
            format!("credential file exceeds {MAX_AUTH_BYTES} bytes"),
        ));
    }

    let (temporary_path, mut temporary_file) = create_temporary_file(parent, path)?;
    let result = (|| {
        temporary_file
            .write_all(&serialized)
            .map_err(|error| store_error("writing credentials", path, error))?;
        temporary_file
            .flush()
            .map_err(|error| store_error("flushing credentials", path, error))?;
        temporary_file
            .sync_all()
            .map_err(|error| store_error("syncing credentials", path, error))?;
        set_private_file(&temporary_file, path, "credential temp file")?;
        drop(temporary_file);

        validate_auth_path(path)?;
        fs::rename(&temporary_path, path)
            .map_err(|error| store_error("replacing credentials", path, error))?;
        sync_directory(parent, path)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

fn create_temporary_file(parent: &Path, auth_path: &Path) -> Result<(PathBuf, File), AuthError> {
    let basename = auth_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("auth");
    for _ in 0..128 {
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let filename = format!(".{basename}.{}.{}.tmp", std::process::id(), counter);
        let temporary_path = parent.join(filename);
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&temporary_path) {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(store_error(
                    "creating credential temp file",
                    &temporary_path,
                    error,
                ));
            }
        }
    }
    Err(store_error(
        "creating credential temp file",
        auth_path,
        "could not choose a unique temporary path",
    ))
}

fn open_lock_file(path: &Path) -> Result<File, AuthError> {
    validate_lock_path(path)?;
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(path)
        .map_err(|error| store_error("opening credential lock", path, error))?;
    if !file
        .metadata()
        .map_err(|error| store_error("reading credential lock metadata", path, error))?
        .is_file()
    {
        return Err(store_error(
            "opening credential lock",
            path,
            "lock path is not a regular file",
        ));
    }
    set_private_file(&file, path, "credential lock")?;
    Ok(file)
}

fn ensure_parent_directory(path: &Path, label: &str) -> Result<(), AuthError> {
    let parent = parent_directory(path);
    fs::create_dir_all(parent)
        .map_err(|error| store_error("creating credential directory", parent, error))?;
    let metadata = fs::symlink_metadata(parent)
        .map_err(|error| store_error("reading credential directory", parent, error))?;
    if !metadata.is_dir() {
        return Err(store_error(
            "creating credential directory",
            parent,
            format!("{label} parent is not a directory"),
        ));
    }
    #[cfg(unix)]
    if parent != Path::new(".") {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
            .map_err(|error| store_error("securing credential directory", parent, error))?;
    }
    Ok(())
}

fn validate_auth_path(path: &Path) -> Result<bool, AuthError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(store_error("checking credential file", path, error)),
    };
    if metadata.file_type().is_symlink() {
        return Err(store_error(
            "checking credential file",
            path,
            "credential file must not be a symlink",
        ));
    }
    if !metadata.is_file() {
        return Err(store_error(
            "checking credential file",
            path,
            "credential path is not a regular file",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(store_error(
                "checking credential file",
                path,
                format!("credential file permissions {mode:03o} are too broad"),
            ));
        }
    }
    Ok(true)
}

fn validate_lock_path(path: &Path) -> Result<(), AuthError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(store_error("checking credential lock", path, error)),
    };
    if metadata.file_type().is_symlink() {
        return Err(store_error(
            "checking credential lock",
            path,
            "credential lock must not be a symlink",
        ));
    }
    if !metadata.is_file() {
        return Err(store_error(
            "checking credential lock",
            path,
            "credential lock path is not a regular file",
        ));
    }
    Ok(())
}

fn set_private_file(file: &File, path: &Path, label: &str) -> Result<(), AuthError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| store_error(label, path, error))?;
    }
    #[cfg(not(unix))]
    {
        let _ = (file, path, label);
    }
    Ok(())
}

fn sync_directory(parent: &Path, path: &Path) -> Result<(), AuthError> {
    #[cfg(unix)]
    {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| store_error("syncing credential directory", path, error))?;
    }
    #[cfg(not(unix))]
    {
        let _ = (parent, path);
    }
    Ok(())
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn store_error(operation: &str, path: &Path, error: impl std::fmt::Display) -> AuthError {
    AuthError::Store(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds_ai::{CredentialStore, CredentialType};
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn credential(value: &str) -> Credential {
        Credential::ApiKey {
            key: Some(value.to_owned()),
            env: BTreeMap::new(),
        }
    }

    fn paths() -> (TempDir, PathBuf, PathBuf) {
        let directory = tempfile::tempdir().expect("temp directory");
        let auth_path = directory.path().join("auth.json");
        let lock_path = directory.path().join("auth.lock");
        (directory, auth_path, lock_path)
    }

    fn set_auth_file_mode(path: &Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).expect("secure auth file");
        }
        #[cfg(not(unix))]
        let _ = path;
    }

    fn set_credential(value: Credential) -> CredentialMutation {
        Box::new(move |_| Box::pin(async move { Ok(Some(value)) }))
    }

    #[tokio::test]
    async fn roundtrip_list_and_delete() {
        let (_directory, auth_path, lock_path) = paths();
        let store = PersistentCredentialStore::new(auth_path.clone(), lock_path).expect("store");
        let cancellation = CancellationToken::new();

        assert_eq!(store.read("openai", &cancellation).await.unwrap(), None);
        assert_eq!(
            store
                .modify(
                    "openai",
                    set_credential(credential("secret")),
                    &cancellation,
                )
                .await
                .unwrap(),
            Some(credential("secret"))
        );
        assert_eq!(
            store.read("openai", &cancellation).await.unwrap(),
            Some(credential("secret"))
        );
        assert_eq!(
            store.list(&cancellation).await.unwrap(),
            vec![CredentialInfo {
                provider_id: "openai".into(),
                credential_type: CredentialType::ApiKey,
            }]
        );

        store.delete("openai", &cancellation).await.unwrap();
        assert_eq!(store.read("openai", &cancellation).await.unwrap(), None);
        assert_eq!(fs::read_to_string(auth_path).unwrap().trim(), "{}");
    }

    #[tokio::test]
    async fn malformed_auth_is_reported_and_never_overwritten() {
        let (_directory, auth_path, lock_path) = paths();
        let store = PersistentCredentialStore::new(auth_path.clone(), lock_path).expect("store");
        fs::write(&auth_path, b"not json").expect("malformed auth");
        set_auth_file_mode(&auth_path);
        let before = fs::read(&auth_path).expect("read malformed auth");
        let cancellation = CancellationToken::new();

        assert!(matches!(
            store.read("openai", &cancellation).await,
            Err(AuthError::Store(_))
        ));
        assert!(matches!(
            store
                .modify(
                    "openai",
                    set_credential(credential("secret")),
                    &cancellation,
                )
                .await,
            Err(AuthError::Store(_))
        ));
        assert_eq!(fs::read(&auth_path).unwrap(), before);
    }

    #[tokio::test]
    async fn oversized_credential_is_rejected_without_replacing_auth() {
        let (_directory, auth_path, lock_path) = paths();
        let store = PersistentCredentialStore::new(auth_path.clone(), lock_path).expect("store");
        let cancellation = CancellationToken::new();
        store
            .modify("openai", set_credential(credential("small")), &cancellation)
            .await
            .unwrap();
        let before = fs::read(&auth_path).unwrap();

        let result = store
            .modify(
                "openai",
                set_credential(credential(&"x".repeat(MAX_AUTH_BYTES))),
                &cancellation,
            )
            .await;

        assert!(matches!(result, Err(AuthError::Store(_))));
        assert_eq!(fs::read(auth_path).unwrap(), before);
    }

    #[tokio::test]
    async fn creates_private_auth_storage_on_unix() {
        let (_directory, auth_path, lock_path) = paths();
        let store =
            PersistentCredentialStore::new(auth_path.clone(), lock_path.clone()).expect("store");
        let cancellation = CancellationToken::new();
        store
            .modify(
                "openai",
                set_credential(credential("secret")),
                &cancellation,
            )
            .await
            .unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(auth_path.parent().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(&auth_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(lock_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[tokio::test]
    async fn independent_store_instances_preserve_provider_updates() {
        let (_directory, auth_path, lock_path) = paths();
        let first = Arc::new(
            PersistentCredentialStore::new(auth_path.clone(), lock_path.clone())
                .expect("first store"),
        );
        let second =
            Arc::new(PersistentCredentialStore::new(auth_path, lock_path).expect("second store"));
        let first_cancellation = CancellationToken::new();
        let second_cancellation = CancellationToken::new();

        let (first_result, second_result) = tokio::join!(
            first.modify(
                "openai",
                set_credential(credential("openai-secret")),
                &first_cancellation,
            ),
            second.modify(
                "anthropic",
                set_credential(credential("anthropic-secret")),
                &second_cancellation,
            )
        );
        assert_eq!(first_result.unwrap(), Some(credential("openai-secret")));
        assert_eq!(second_result.unwrap(), Some(credential("anthropic-secret")));

        let cancellation = CancellationToken::new();
        assert_eq!(
            first.read("openai", &cancellation).await.unwrap(),
            Some(credential("openai-secret"))
        );
        assert_eq!(
            first.read("anthropic", &cancellation).await.unwrap(),
            Some(credential("anthropic-secret"))
        );
    }
}
