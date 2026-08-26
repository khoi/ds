use async_trait::async_trait;
use clap::{Parser, Subcommand, ValueEnum};
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use ds_agent_core::{Agent, AgentModelStream};
use ds_ai::{
    AuthError, AuthEvent, AuthInteraction, AuthPrompt, AuthSelectOption, CredentialStore,
    CredentialType, Model, Models, SimpleStreamOptions, SystemAuthContext, ThinkingLevel,
    builtin_providers,
};
use ds_coding_agent::{
    auth::PersistentCredentialStore,
    coding_tools,
    commands::LoginMethod,
    config::{Config, ConfigPaths, DEFAULT_MODEL},
    ui,
};
use futures_util::{Stream, StreamExt};
use std::{
    error::Error,
    io::{self, IsTerminal, Write},
    sync::Arc,
};
use tokio_util::sync::CancellationToken;

const SYSTEM_PROMPT: &str = "You are a concise coding agent. Use read, bash, edit, and write to inspect and change the current project. Explain completed work and verification plainly.";

#[derive(Parser)]
#[command(name = "ds", version, about = "A minimal Pi-style coding agent")]
struct Cli {
    #[arg(long, value_name = "PROVIDER/MODEL")]
    model: Option<String>,
    #[arg(long, value_enum)]
    reasoning: Option<ReasoningArg>,
    #[arg(long, value_parser = parse_positive_u32)]
    max_turns: Option<u32>,
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(value_name = "PROMPT", num_args = 1.., trailing_var_arg = true)]
    prompt: Vec<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Sign in to a provider and save the credential.
    Login {
        provider: String,
        #[arg(long = "type", value_enum)]
        credential_type: Option<CredentialTypeArg>,
    },
    /// Remove a saved provider credential.
    Logout { provider: String },
    /// Inspect authentication without refreshing credentials.
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    /// Inspect the global ds configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
}

#[derive(Subcommand)]
enum AuthCommand {
    /// Show configured providers without printing secrets.
    Status { provider: Option<String> },
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Print the global configuration path.
    Path,
    /// Print the effective global configuration.
    Show,
    /// Validate the global configuration.
    Check,
}

#[derive(Clone, Copy, ValueEnum)]
enum CredentialTypeArg {
    #[value(name = "api-key")]
    ApiKey,
    #[value(name = "oauth")]
    OAuth,
}

impl From<CredentialTypeArg> for CredentialType {
    fn from(value: CredentialTypeArg) -> Self {
        match value {
            CredentialTypeArg::ApiKey => Self::ApiKey,
            CredentialTypeArg::OAuth => Self::OAuth,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum ReasoningArg {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    #[value(name = "xhigh")]
    XHigh,
    Max,
}

impl From<ReasoningArg> for ThinkingLevel {
    fn from(value: ReasoningArg) -> Self {
        match value {
            ReasoningArg::Off => Self::Off,
            ReasoningArg::Minimal => Self::Minimal,
            ReasoningArg::Low => Self::Low,
            ReasoningArg::Medium => Self::Medium,
            ReasoningArg::High => Self::High,
            ReasoningArg::XHigh => Self::XHigh,
            ReasoningArg::Max => Self::Max,
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("ds: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let paths = ConfigPaths::from_env()?;

    if let Some(Command::Config { command }) = &cli.command {
        let config = Config::load(&paths)?;
        return run_config_command(command, &paths, &config);
    }

    let credentials: Arc<dyn CredentialStore> = Arc::new(PersistentCredentialStore::new(
        paths.auth().to_owned(),
        paths.auth_lock().to_owned(),
    )?);
    let models = Arc::new(models_with_credentials(credentials));

    match cli.command {
        Some(Command::Login {
            provider,
            credential_type,
        }) => run_login(&models, &provider, credential_type).await,
        Some(Command::Logout { provider }) => run_logout(&models, &provider).await,
        Some(Command::Auth { command }) => run_auth_command(&models, command).await,
        Some(Command::Config { .. }) => unreachable!("config command returned above"),
        None => {
            let config = Config::load(&paths)?;
            run_agent(models, cli, config, paths).await
        }
    }
}

fn models_with_credentials(credentials: Arc<dyn CredentialStore>) -> Models {
    let mut models = Models::with_auth(credentials, Arc::new(SystemAuthContext));
    for provider in builtin_providers() {
        models.set_provider(provider);
    }
    models
}

async fn run_agent(
    models: Arc<Models>,
    cli: Cli,
    config: Config,
    paths: ConfigPaths,
) -> Result<(), Box<dyn Error>> {
    let allow_model_fallback = cli.prompt.is_empty() && cli.model.is_none();
    let model_name = cli.model.clone().unwrap_or_else(|| config.model.clone());
    let (model, startup_notice) = select_model(&models, &model_name, allow_model_fallback)?;
    let provider = model.provider.as_str().to_owned();
    let cancellation = CancellationToken::new();
    if !cli.prompt.is_empty() && models.check_auth(&provider, &cancellation).await?.is_none() {
        return Err(
            format!("provider {provider} is not configured; run `ds login {provider}`").into(),
        );
    }

    let reasoning = cli.reasoning.map(Into::into).or(config.reasoning);
    let max_turns = cli.max_turns.unwrap_or(config.max_turns) as usize;
    let model_stream: Arc<dyn AgentModelStream> = models.clone();
    let working_directory = std::env::current_dir()?;
    let mut agent = Agent::new(model, model_stream, coding_tools()?, working_directory)
        .with_system_prompt(SYSTEM_PROMPT)
        .with_options(SimpleStreamOptions {
            reasoning,
            ..Default::default()
        })
        .with_max_turns(max_turns);

    if cli.prompt.is_empty() {
        let mut backend = RuntimeBackend {
            models,
            config,
            paths,
            reasoning,
            startup_notice,
        };
        ui::run_interactive(&mut agent, &mut backend).await?;
    } else {
        ui::run_one_shot(&mut agent, cli.prompt.join(" ")).await?;
    }
    Ok(())
}

fn select_model(
    models: &Models,
    requested: &str,
    allow_fallback: bool,
) -> Result<(Model, Option<String>), Box<dyn Error>> {
    let (provider, model_id) = parse_model(requested)?;
    if let Some(model) = models.model(provider, model_id) {
        return Ok((model, None));
    }
    if !allow_fallback {
        return Err(format!("unknown model: {requested}").into());
    }

    let (default_provider, default_model) = parse_model(DEFAULT_MODEL)?;
    let fallback = models
        .model(default_provider, default_model)
        .or_else(|| models.models(None).into_iter().next())
        .ok_or("no models are registered")?;
    let fallback_name = format!("{}/{}", fallback.provider, fallback.id);
    Ok((
        fallback,
        Some(format!(
            "configured model {requested} is unavailable; using {fallback_name} · use /models to save another default"
        )),
    ))
}

async fn run_login(
    models: &Models,
    provider_id: &str,
    requested_type: Option<CredentialTypeArg>,
) -> Result<(), Box<dyn Error>> {
    perform_login(
        models,
        provider_id,
        requested_type.map(CredentialType::from),
    )
    .await?;
    println!("logged in to {provider_id}");
    Ok(())
}

async fn perform_login(
    models: &Models,
    provider_id: &str,
    requested_type: Option<CredentialType>,
) -> Result<(), Box<dyn Error>> {
    let provider = models
        .provider(provider_id)
        .ok_or_else(|| format!("unknown provider: {provider_id}"))?;
    let interaction = TerminalAuthInteraction::new();
    let credential_type = match requested_type {
        Some(credential_type) => credential_type,
        None => match (&provider.auth().api_key, &provider.auth().oauth) {
            (Some(api_key), Some(oauth)) => {
                let selected = interaction
                    .prompt(AuthPrompt::Select {
                        message: format!("Select {provider_id} login method:"),
                        options: vec![
                            AuthSelectOption {
                                id: "api_key".into(),
                                label: api_key.name().into(),
                                description: None,
                            },
                            AuthSelectOption {
                                id: "oauth".into(),
                                label: oauth.name().into(),
                                description: None,
                            },
                        ],
                    })
                    .await?;
                match selected.as_str() {
                    "api_key" => CredentialType::ApiKey,
                    "oauth" => CredentialType::OAuth,
                    _ => return Err(format!("unknown login method: {selected}").into()),
                }
            }
            (Some(_), None) => CredentialType::ApiKey,
            (None, Some(_)) => CredentialType::OAuth,
            (None, None) => {
                return Err(format!("provider {provider_id} does not support login").into());
            }
        },
    };
    models
        .login(provider_id, credential_type, &interaction)
        .await?;
    Ok(())
}

async fn run_logout(models: &Models, provider_id: &str) -> Result<(), Box<dyn Error>> {
    if models.provider(provider_id).is_none() {
        return Err(format!("unknown provider: {provider_id}").into());
    }
    models
        .logout(provider_id, &CancellationToken::new())
        .await?;
    println!("logged out of {provider_id}");
    Ok(())
}

async fn run_auth_command(models: &Models, command: AuthCommand) -> Result<(), Box<dyn Error>> {
    match command {
        AuthCommand::Status { provider } => {
            println!("{}", auth_status_text(models, provider.as_deref()).await?);
            Ok(())
        }
    }
}

async fn auth_status_text(
    models: &Models,
    provider_id: Option<&str>,
) -> Result<String, Box<dyn Error>> {
    let providers = match provider_id {
        Some(provider_id) => vec![
            models
                .provider(provider_id)
                .ok_or_else(|| format!("unknown provider: {provider_id}"))?,
        ],
        None => models.providers(),
    };
    let cancellation = CancellationToken::new();
    let mut lines = Vec::with_capacity(providers.len());
    for provider in providers {
        let provider_id = provider.id().as_str();
        match models.check_auth(provider_id, &cancellation).await? {
            Some(check) => {
                let credential_type = match check.credential_type {
                    CredentialType::ApiKey => "api-key",
                    CredentialType::OAuth => "oauth",
                };
                lines.push(format!(
                    "{provider_id}\t{credential_type}\t{}",
                    check.source.as_deref().unwrap_or("configured")
                ));
            }
            None => lines.push(format!("{provider_id}\tnot configured")),
        }
    }
    Ok(lines.join("\n"))
}

struct RuntimeBackend {
    models: Arc<Models>,
    config: Config,
    paths: ConfigPaths,
    reasoning: Option<ThinkingLevel>,
    startup_notice: Option<String>,
}

#[async_trait]
impl ui::InteractiveBackend for RuntimeBackend {
    fn models(&self) -> Vec<Model> {
        self.models.models(None)
    }

    fn providers(&self) -> Vec<ui::ProviderChoice> {
        self.models
            .providers()
            .into_iter()
            .map(|provider| ui::ProviderChoice {
                id: provider.id().as_str().to_owned(),
                name: provider.name().to_owned(),
                supports_api_key: provider.auth().api_key.is_some(),
                supports_oauth: provider.auth().oauth.is_some(),
            })
            .collect()
    }

    fn reasoning_label(&self) -> String {
        self.reasoning.map_or("auto", thinking_level_name).into()
    }

    fn take_startup_notice(&mut self) -> Option<String> {
        self.startup_notice.take()
    }

    fn persist_model(&mut self, model: &Model) -> Result<(), String> {
        let mut updated = self.config.clone();
        updated.model = format!("{}/{}", model.provider, model.id);
        updated
            .save(&self.paths)
            .map_err(|error| error.to_string())?;
        self.config = updated;
        Ok(())
    }

    async fn provider_configured(&self, provider: &str) -> Result<bool, String> {
        self.models
            .check_auth(provider, &CancellationToken::new())
            .await
            .map(|check| check.is_some())
            .map_err(|error| error.to_string())
    }

    async fn login(&mut self, provider: &str, method: Option<LoginMethod>) -> Result<(), String> {
        let method = method.map(|method| match method {
            LoginMethod::ApiKey => CredentialType::ApiKey,
            LoginMethod::OAuth => CredentialType::OAuth,
        });
        perform_login(&self.models, provider, method)
            .await
            .map_err(|error| error.to_string())
    }

    async fn logout(&mut self, provider: &str) -> Result<(), String> {
        if self.models.provider(provider).is_none() {
            return Err(format!("unknown provider: {provider}"));
        }
        self.models
            .logout(provider, &CancellationToken::new())
            .await
            .map_err(|error| error.to_string())
    }

    async fn auth_status(&self, provider: Option<&str>) -> Result<String, String> {
        auth_status_text(&self.models, provider)
            .await
            .map_err(|error| error.to_string())
    }
}

fn thinking_level_name(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

fn run_config_command(
    command: &ConfigCommand,
    paths: &ConfigPaths,
    config: &Config,
) -> Result<(), Box<dyn Error>> {
    match command {
        ConfigCommand::Path => println!("{}", paths.config().display()),
        ConfigCommand::Show => print!("{}", toml::to_string_pretty(config)?),
        ConfigCommand::Check => println!("{}: ok", paths.config().display()),
    }
    Ok(())
}

struct TerminalAuthInteraction {
    cancellation: CancellationToken,
}

impl TerminalAuthInteraction {
    fn new() -> Self {
        Self {
            cancellation: CancellationToken::new(),
        }
    }

    fn read_line(message: &str) -> Result<String, AuthError> {
        eprint!("{message}: ");
        io::stderr()
            .flush()
            .map_err(|error| terminal_auth_error("write prompt", error))?;
        let mut value = String::new();
        io::stdin()
            .read_line(&mut value)
            .map_err(|error| terminal_auth_error("read prompt", error))?;
        Ok(value.trim().to_owned())
    }

    async fn prompt_blocking(message: String, secret: bool) -> Result<String, AuthError> {
        tokio::task::spawn_blocking(move || {
            if secret {
                rpassword::prompt_password(format!("{message}: "))
                    .map(|value| value.trim().to_owned())
                    .map_err(|error| terminal_auth_error("read secret", error))
            } else {
                Self::read_line(&message)
            }
        })
        .await
        .map_err(|error| {
            AuthError::Authentication(format!("terminal prompt task failed: {error}"))
        })?
    }

    async fn prompt_value(
        &self,
        message: String,
        cancellation: CancellationToken,
        secret: bool,
    ) -> Result<String, AuthError> {
        if !io::stdin().is_terminal() {
            return Self::prompt_blocking(message, secret).await;
        }
        eprint!("{message}: ");
        io::stderr()
            .flush()
            .map_err(|error| terminal_auth_error("write prompt", error))?;

        let raw_mode = ManualPromptMode::enter()?;
        let mut events = EventStream::new();
        let result = self
            .read_manual_line(&mut events, cancellation, secret)
            .await;
        drop(raw_mode);
        eprintln!();
        result
    }

    async fn read_manual_line<S>(
        &self,
        events: &mut S,
        cancellation: CancellationToken,
        secret: bool,
    ) -> Result<String, AuthError>
    where
        S: Stream<Item = io::Result<Event>> + Unpin,
    {
        let mut value = String::new();
        loop {
            let event = tokio::select! {
                biased;
                _ = self.cancellation.cancelled() => return Err(AuthError::Cancelled),
                _ = cancellation.cancelled() => return Err(AuthError::Cancelled),
                event = events.next() => event
                    .ok_or_else(|| AuthError::Authentication("terminal input ended".into()))?
                    .map_err(|error| terminal_auth_error("read prompt", error))?,
            };
            match event {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
                        && key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    return Err(AuthError::Cancelled);
                }
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
                        && key.code == KeyCode::Enter =>
                {
                    return Ok(value.trim().to_owned());
                }
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
                        && key.code == KeyCode::Backspace =>
                {
                    if value.pop().is_some() {
                        eprint!("\u{8} \u{8}");
                        io::stderr()
                            .flush()
                            .map_err(|error| terminal_auth_error("write prompt", error))?;
                    }
                }
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
                        && !key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    if let KeyCode::Char(character) = key.code {
                        value.push(character);
                        if secret {
                            eprint!("*");
                        } else {
                            eprint!("{character}");
                        }
                        io::stderr()
                            .flush()
                            .map_err(|error| terminal_auth_error("write prompt", error))?;
                    }
                }
                Event::Paste(text) => {
                    for character in text.chars() {
                        if matches!(character, '\r' | '\n') {
                            return Ok(value.trim().to_owned());
                        }
                        if !character.is_control() {
                            value.push(character);
                            if secret {
                                eprint!("*");
                            } else {
                                eprint!("{character}");
                            }
                        }
                    }
                    io::stderr()
                        .flush()
                        .map_err(|error| terminal_auth_error("write prompt", error))?;
                }
                Event::FocusGained
                | Event::FocusLost
                | Event::Key(_)
                | Event::Mouse(_)
                | Event::Resize(_, _) => {}
            }
        }
    }
}

struct ManualPromptMode;

impl ManualPromptMode {
    fn enter() -> Result<Self, AuthError> {
        enable_raw_mode().map_err(|error| terminal_auth_error("enter raw mode", error))?;
        Ok(Self)
    }
}

impl Drop for ManualPromptMode {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

#[async_trait]
impl AuthInteraction for TerminalAuthInteraction {
    fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthError> {
        if self.cancellation.is_cancelled() {
            return Err(AuthError::Cancelled);
        }
        match prompt {
            AuthPrompt::Text { message, .. } => {
                self.prompt_value(message, self.cancellation.clone(), false)
                    .await
            }
            AuthPrompt::Secret { message, .. } => {
                self.prompt_value(message, self.cancellation.clone(), true)
                    .await
            }
            AuthPrompt::ManualCode {
                message,
                cancellation,
                ..
            } => self.prompt_value(message, cancellation, false).await,
            AuthPrompt::Select { message, options } => {
                eprintln!("{message}");
                for (index, option) in options.iter().enumerate() {
                    match &option.description {
                        Some(description) => {
                            eprintln!("  {}. {} - {description}", index + 1, option.label);
                        }
                        None => eprintln!("  {}. {}", index + 1, option.label),
                    }
                }
                let selection = self
                    .prompt_value("Select".into(), self.cancellation.clone(), false)
                    .await?;
                if selection.is_empty() {
                    return options
                        .first()
                        .map(|option| option.id.clone())
                        .ok_or_else(|| AuthError::Authentication("login has no choices".into()));
                }
                let index = selection.parse::<usize>().map_err(|_| {
                    AuthError::Authentication(format!("invalid selection: {selection}"))
                })?;
                let index = index.checked_sub(1).ok_or_else(|| {
                    AuthError::Authentication(format!("invalid selection: {selection}"))
                })?;
                options
                    .get(index)
                    .map(|option| option.id.clone())
                    .ok_or_else(|| {
                        AuthError::Authentication(format!("invalid selection: {selection}"))
                    })
            }
        }
    }

    fn notify(&self, event: AuthEvent) {
        match event {
            AuthEvent::Info { message, links } => {
                eprintln!("{message}");
                for link in links {
                    match link.label {
                        Some(label) => eprintln!("{label}: {}", link.url),
                        None => eprintln!("{}", link.url),
                    }
                }
            }
            AuthEvent::AuthUrl { url, instructions } => {
                if let Some(instructions) = instructions {
                    eprintln!("{instructions}");
                }
                eprintln!("Open: {url}");
            }
            AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                ..
            } => {
                eprintln!("Open: {verification_uri}");
                eprintln!("Code: {user_code}");
            }
            AuthEvent::Progress { message } => eprintln!("{message}"),
        }
    }
}

fn terminal_auth_error(operation: &str, error: io::Error) -> AuthError {
    AuthError::Authentication(format!("terminal {operation} failed: {error}"))
}

fn parse_positive_u32(value: &str) -> Result<u32, String> {
    match value.parse::<u32>() {
        Ok(value) if value > 0 => Ok(value),
        _ => Err("must be a positive integer".into()),
    }
}

fn parse_model(value: &str) -> Result<(&str, &str), Box<dyn Error>> {
    let Some((provider, model)) = value.split_once('/') else {
        return Err(format!("model must use provider/model form: {value}").into());
    };
    if provider.is_empty() || model.is_empty() {
        return Err(format!("model must use provider/model form: {value}").into());
    }
    Ok((provider, model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_configured_model_falls_back_only_for_interactive_startup() {
        let models = models_with_credentials(Arc::new(ds_ai::InMemoryCredentialStore::new()));

        let (fallback, notice) = select_model(&models, "missing/retired-model", true).unwrap();

        assert_eq!(
            format!("{}/{}", fallback.provider, fallback.id),
            DEFAULT_MODEL
        );
        assert!(notice.unwrap().contains("use /models"));
        assert!(select_model(&models, "missing/retired-model", false).is_err());
    }

    #[test]
    fn failed_model_persistence_does_not_mutate_runtime_config() {
        let directory = tempfile::tempdir().unwrap();
        let blocked_root = directory.path().join("file-not-directory");
        std::fs::write(&blocked_root, "blocked").unwrap();
        let models = Arc::new(models_with_credentials(Arc::new(
            ds_ai::InMemoryCredentialStore::new(),
        )));
        let model = models.model("openai", "gpt-5.6-luna").unwrap();
        let mut backend = RuntimeBackend {
            models,
            config: Config::default(),
            paths: ConfigPaths::from_root(blocked_root),
            reasoning: None,
            startup_notice: None,
        };

        let result = ui::InteractiveBackend::persist_model(&mut backend, &model);

        assert!(result.is_err());
        assert_eq!(backend.config.model, DEFAULT_MODEL);
    }

    #[tokio::test]
    async fn manual_code_prompt_stops_when_browser_flow_finishes() {
        let interaction = TerminalAuthInteraction::new();
        let cancellation = CancellationToken::new();
        let cancel = cancellation.clone();
        let mut events = futures_util::stream::pending();

        let (result, ()) = tokio::join!(
            interaction.read_manual_line(&mut events, cancellation, false),
            async move {
                tokio::task::yield_now().await;
                cancel.cancel();
            }
        );

        assert!(matches!(result, Err(AuthError::Cancelled)));
    }

    #[tokio::test]
    async fn terminal_auth_prompt_treats_control_c_as_cancellation() {
        let interaction = TerminalAuthInteraction::new();
        let mut events = futures_util::stream::iter([Ok(Event::Key(
            crossterm::event::KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        ))]);

        let result = interaction
            .read_manual_line(&mut events, CancellationToken::new(), true)
            .await;

        assert!(matches!(result, Err(AuthError::Cancelled)));
    }
}
