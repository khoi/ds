use std::{
    error::Error,
    fmt,
    sync::{Arc, Mutex, OnceLock},
};

type Cleanup = Arc<dyn Fn(Option<&str>) -> Result<(), String> + Send + Sync + 'static>;

#[derive(Default)]
struct SessionResourceRegistry {
    cleanups: Mutex<Vec<Cleanup>>,
}

impl SessionResourceRegistry {
    fn register<F>(&self, cleanup: F) -> Cleanup
    where
        F: Fn(Option<&str>) -> Result<(), String> + Send + Sync + 'static,
    {
        let cleanup: Cleanup = Arc::new(cleanup);
        self.cleanups
            .lock()
            .expect("session resource registry lock")
            .push(Arc::clone(&cleanup));
        cleanup
    }

    fn unregister(&self, cleanup: &Cleanup) -> bool {
        let mut cleanups = self
            .cleanups
            .lock()
            .expect("session resource registry lock");
        let Some(index) = cleanups
            .iter()
            .position(|registered| Arc::ptr_eq(registered, cleanup))
        else {
            return false;
        };
        cleanups.remove(index);
        true
    }

    fn cleanup(&self, session_id: Option<&str>) -> Result<(), SessionResourceCleanupError> {
        let cleanups = self
            .cleanups
            .lock()
            .expect("session resource registry lock")
            .clone();
        let failures = cleanups
            .into_iter()
            .filter_map(|cleanup| cleanup(session_id).err())
            .collect::<Vec<_>>();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(SessionResourceCleanupError { failures })
        }
    }
}

pub struct SessionResourceCleanupRegistration {
    cleanup: Cleanup,
}

impl SessionResourceCleanupRegistration {
    pub fn unregister(self) -> bool {
        registry().unregister(&self.cleanup)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionResourceCleanupError {
    failures: Vec<String>,
}

impl SessionResourceCleanupError {
    pub fn failures(&self) -> &[String] {
        &self.failures
    }
}

impl fmt::Display for SessionResourceCleanupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "failed to clean up session resources: {}",
            self.failures.join("; ")
        )
    }
}

impl Error for SessionResourceCleanupError {}

pub fn register_session_resource_cleanup<F>(cleanup: F) -> SessionResourceCleanupRegistration
where
    F: Fn(Option<&str>) -> Result<(), String> + Send + Sync + 'static,
{
    SessionResourceCleanupRegistration {
        cleanup: registry().register(cleanup),
    }
}

pub fn cleanup_session_resources(
    session_id: Option<&str>,
) -> Result<(), SessionResourceCleanupError> {
    registry().cleanup(session_id)
}

fn registry() -> &'static SessionResourceRegistry {
    static REGISTRY: OnceLock<SessionResourceRegistry> = OnceLock::new();
    REGISTRY.get_or_init(SessionResourceRegistry::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwards_one_session_and_all_sessions() {
        let registry = SessionResourceRegistry::default();
        let sessions = Arc::new(Mutex::new(Vec::new()));
        registry.register({
            let sessions = Arc::clone(&sessions);
            move |session_id| {
                sessions
                    .lock()
                    .expect("sessions lock")
                    .push(session_id.map(str::to_owned));
                Ok(())
            }
        });

        registry.cleanup(Some("one")).unwrap();
        registry.cleanup(None).unwrap();

        assert_eq!(
            *sessions.lock().expect("sessions lock"),
            vec![Some("one".into()), None]
        );
    }

    #[test]
    fn unregisters_one_cleanup() {
        let registry = SessionResourceRegistry::default();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let first = registry.register(recording_cleanup("first", Arc::clone(&calls)));
        registry.register(recording_cleanup("second", Arc::clone(&calls)));

        assert!(registry.unregister(&first));
        assert!(!registry.unregister(&first));
        registry.cleanup(None).unwrap();

        assert_eq!(*calls.lock().expect("calls lock"), vec!["second"]);
    }

    #[test]
    fn invokes_every_cleanup_and_aggregates_failures() {
        let registry = SessionResourceRegistry::default();
        let calls = Arc::new(Mutex::new(Vec::new()));
        registry.register(failing_cleanup(
            "first",
            "first failure",
            Arc::clone(&calls),
        ));
        registry.register(recording_cleanup("second", Arc::clone(&calls)));
        registry.register(failing_cleanup(
            "third",
            "third failure",
            Arc::clone(&calls),
        ));

        let error = registry.cleanup(None).unwrap_err();

        assert_eq!(
            *calls.lock().expect("calls lock"),
            vec!["first", "second", "third"]
        );
        assert_eq!(error.failures(), ["first failure", "third failure"]);
        assert_eq!(
            error.to_string(),
            "failed to clean up session resources: first failure; third failure"
        );
    }

    #[test]
    fn snapshots_before_invoking_cleanups() {
        let registry = Arc::new(SessionResourceRegistry::default());
        let calls = Arc::new(Mutex::new(Vec::new()));
        registry.register({
            let registry = Arc::downgrade(&registry);
            let calls = Arc::clone(&calls);
            move |_| {
                calls.lock().expect("calls lock").push("first");
                registry
                    .upgrade()
                    .expect("registry")
                    .register(recording_cleanup("second", Arc::clone(&calls)));
                Ok(())
            }
        });

        registry.cleanup(None).unwrap();
        assert_eq!(*calls.lock().expect("calls lock"), vec!["first"]);

        registry.cleanup(None).unwrap();
        assert_eq!(
            *calls.lock().expect("calls lock"),
            vec!["first", "first", "second"]
        );
    }

    #[test]
    fn public_registration_persists_until_explicitly_unregistered() {
        let target = "session-resources-public-registration";
        let calls = Arc::new(Mutex::new(0));
        let registration = register_session_resource_cleanup({
            let calls = Arc::clone(&calls);
            move |session_id| {
                if session_id == Some(target) {
                    *calls.lock().expect("calls lock") += 1;
                }
                Ok(())
            }
        });

        cleanup_session_resources(Some(target)).unwrap();
        assert_eq!(*calls.lock().expect("calls lock"), 1);

        assert!(registration.unregister());
        cleanup_session_resources(Some(target)).unwrap();
        assert_eq!(*calls.lock().expect("calls lock"), 1);
    }

    #[test]
    fn ignored_registration_remains_registered() {
        let target = "session-resources-ignored-registration";
        let calls = Arc::new(Mutex::new(0));
        register_session_resource_cleanup({
            let calls = Arc::clone(&calls);
            move |session_id| {
                if session_id == Some(target) {
                    *calls.lock().expect("calls lock") += 1;
                }
                Ok(())
            }
        });

        cleanup_session_resources(Some(target)).unwrap();

        assert_eq!(*calls.lock().expect("calls lock"), 1);
    }

    fn recording_cleanup(
        name: &'static str,
        calls: Arc<Mutex<Vec<&'static str>>>,
    ) -> impl Fn(Option<&str>) -> Result<(), String> + Send + Sync + 'static {
        move |_| {
            calls.lock().expect("calls lock").push(name);
            Ok(())
        }
    }

    fn failing_cleanup(
        name: &'static str,
        failure: &'static str,
        calls: Arc<Mutex<Vec<&'static str>>>,
    ) -> impl Fn(Option<&str>) -> Result<(), String> + Send + Sync + 'static {
        move |_| {
            calls.lock().expect("calls lock").push(name);
            Err(failure.into())
        }
    }
}
