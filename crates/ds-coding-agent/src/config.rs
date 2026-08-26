use ds_ai::ThinkingLevel;
use serde::{Deserialize, Serialize};
use std::{
    env,
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
};

pub const CONFIG_VERSION: u32 = 1;
pub const DEFAULT_MODEL: &str = "openai-codex/gpt-5.6-luna";
pub const DEFAULT_MAX_TURNS: u32 = 24;

/// Files owned by a ds installation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigPaths {
    root: PathBuf,
    config: PathBuf,
    auth: PathBuf,
    auth_lock: PathBuf,
}

impl ConfigPaths {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_environment(env::var_os("DS_HOME"), env::var_os("HOME"))
    }

    pub fn from_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            config: root.join("config.toml"),
            auth: root.join("auth.json"),
            auth_lock: root.join("auth.lock"),
            root,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &Path {
        &self.config
    }

    pub fn config_path(&self) -> &Path {
        self.config()
    }

    pub fn auth(&self) -> &Path {
        &self.auth
    }

    pub fn auth_path(&self) -> &Path {
        self.auth()
    }

    pub fn auth_lock(&self) -> &Path {
        &self.auth_lock
    }

    pub fn auth_lock_path(&self) -> &Path {
        self.auth_lock()
    }

    fn from_environment(
        ds_home: Option<std::ffi::OsString>,
        home: Option<std::ffi::OsString>,
    ) -> Result<Self, ConfigError> {
        let root = ds_home
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                home.filter(|value| !value.is_empty())
                    .map(|value| PathBuf::from(value).join(".ds"))
            })
            .ok_or(ConfigError::HomeDirectoryUnavailable)?;
        Ok(Self::from_root(root))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub version: u32,
    pub model: String,
    pub max_turns: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ThinkingLevel>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            model: DEFAULT_MODEL.to_owned(),
            max_turns: DEFAULT_MAX_TURNS,
            reasoning: None,
        }
    }
}

impl Config {
    pub fn load(paths: &ConfigPaths) -> Result<Self, ConfigError> {
        let path = paths.config();
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(source) => {
                return Err(ConfigError::Read {
                    path: path.to_owned(),
                    source,
                });
            }
        };

        Self::parse_toml(&contents, path)
    }

    pub fn parse_toml(contents: &str, path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref().to_owned();
        let config = toml::from_str::<Self>(contents).map_err(|source| ConfigError::Parse {
            path: path.clone(),
            source,
        })?;
        config.validate(&path)?;
        Ok(config)
    }

    pub fn validate(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref().to_owned();
        if self.version != CONFIG_VERSION {
            return Err(ConfigError::Invalid {
                path,
                field: "version",
                message: format!(
                    "unsupported version {}; expected {CONFIG_VERSION}",
                    self.version
                ),
            });
        }
        let model = self.model.trim();
        let valid_model = model
            .split_once('/')
            .is_some_and(|(provider, model)| !provider.is_empty() && !model.is_empty());
        if model != self.model || !valid_model {
            return Err(ConfigError::Invalid {
                path,
                field: "model",
                message: "must use provider/model form".into(),
            });
        }
        if self.max_turns == 0 {
            return Err(ConfigError::Invalid {
                path,
                field: "max_turns",
                message: "must be positive".into(),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ConfigError {
    HomeDirectoryUnavailable,
    Read {
        path: PathBuf,
        source: io::Error,
    },
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    Invalid {
        path: PathBuf,
        field: &'static str,
        message: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HomeDirectoryUnavailable => {
                formatter.write_str("could not determine ds home (set DS_HOME or HOME)")
            }
            Self::Read { path, source } => {
                write!(
                    formatter,
                    "could not read config {}: {source}",
                    path.display()
                )
            }
            Self::Parse { path, source } => {
                write!(formatter, "invalid config {}: {source}", path.display())
            }
            Self::Invalid {
                path,
                field,
                message,
            } => write!(
                formatter,
                "invalid config {} ({field}): {message}",
                path.display()
            ),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::HomeDirectoryUnavailable | Self::Invalid { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PATH: &str = "/tmp/ds-test/config.toml";

    #[test]
    fn default_config_has_stable_values() {
        let config = Config::default();

        assert_eq!(config.version, CONFIG_VERSION);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.max_turns, DEFAULT_MAX_TURNS);
        assert_eq!(config.reasoning, None);
        assert!(config.validate(TEST_PATH).is_ok());
    }

    #[test]
    fn missing_config_uses_defaults() {
        let directory = tempfile::tempdir().unwrap();
        let config = Config::load(&ConfigPaths::from_root(directory.path())).unwrap();

        assert_eq!(config, Config::default());
    }

    #[test]
    fn parses_valid_config_and_serializes_as_toml() {
        let config = Config::parse_toml(
            r#"
version = 1
model = "anthropic/claude-sonnet"
max_turns = 12
reasoning = "high"
"#,
            TEST_PATH,
        )
        .unwrap();

        assert_eq!(config.version, 1);
        assert_eq!(config.model, "anthropic/claude-sonnet");
        assert_eq!(config.max_turns, 12);
        assert_eq!(config.reasoning, Some(ThinkingLevel::High));

        let rendered = toml::to_string(&config).unwrap();
        assert!(rendered.contains("version = 1"));
        assert!(rendered.contains("model = \"anthropic/claude-sonnet\""));
        assert!(rendered.contains("max_turns = 12"));
        assert!(rendered.contains("reasoning = \"high\""));
    }

    #[test]
    fn rejects_malformed_config_with_path() {
        let error = Config::parse_toml("version = [", TEST_PATH).unwrap_err();
        let message = error.to_string();

        assert!(message.contains(TEST_PATH));
        assert!(matches!(error, ConfigError::Parse { .. }));
    }

    #[test]
    fn rejects_unsupported_version_with_path() {
        let error = Config::parse_toml("version = 2", TEST_PATH).unwrap_err();

        assert!(error.to_string().contains(TEST_PATH));
        assert!(error.to_string().contains("version"));
        assert!(matches!(
            error,
            ConfigError::Invalid {
                field: "version",
                ..
            }
        ));
    }

    #[test]
    fn rejects_zero_max_turns_with_path() {
        let error = Config::parse_toml("max_turns = 0", TEST_PATH).unwrap_err();

        assert!(error.to_string().contains(TEST_PATH));
        assert!(error.to_string().contains("max_turns"));
        assert!(matches!(
            error,
            ConfigError::Invalid {
                field: "max_turns",
                ..
            }
        ));
    }

    #[test]
    fn rejects_model_without_provider() {
        let error = Config::parse_toml("model = \"gpt-5.6-luna\"", TEST_PATH).unwrap_err();

        assert!(error.to_string().contains(TEST_PATH));
        assert!(error.to_string().contains("provider/model"));
        assert!(matches!(error, ConfigError::Invalid { field: "model", .. }));
    }

    #[test]
    fn ds_home_overrides_home_for_all_paths() {
        let paths =
            ConfigPaths::from_environment(Some("/custom/ds".into()), Some("/users/khoi".into()))
                .unwrap();

        assert_eq!(paths.root(), Path::new("/custom/ds"));
        assert_eq!(paths.config(), Path::new("/custom/ds/config.toml"));
        assert_eq!(paths.auth(), Path::new("/custom/ds/auth.json"));
        assert_eq!(paths.auth_lock(), Path::new("/custom/ds/auth.lock"));
    }

    #[test]
    fn home_is_used_when_ds_home_is_missing() {
        let paths = ConfigPaths::from_environment(None, Some("/users/khoi".into())).unwrap();

        assert_eq!(paths.root(), Path::new("/users/khoi/.ds"));
        assert_eq!(paths.config(), Path::new("/users/khoi/.ds/config.toml"));
        assert_eq!(paths.auth(), Path::new("/users/khoi/.ds/auth.json"));
        assert_eq!(paths.auth_lock(), Path::new("/users/khoi/.ds/auth.lock"));
    }
}
