use ds_ai::{Credential, CredentialStore};
use ds_coding_agent::auth::PersistentCredentialStore;
use std::{collections::BTreeMap, fs, process::Command};
use tokio_util::sync::CancellationToken;

#[test]
fn help_finishes_without_terminal_initialization() {
    let output = Command::new(env!("CARGO_BIN_EXE_ds"))
        .arg("--help")
        .output()
        .unwrap();

    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(!output.stdout.windows(2).any(|bytes| bytes == b"\x1b["));
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("--model <PROVIDER/MODEL>"));
    assert!(help.contains("login"));
    assert!(help.contains("logout"));
    assert!(help.contains("auth"));
    assert!(help.contains("config"));
}

#[test]
fn config_commands_use_ds_home() {
    let directory = tempfile::tempdir().unwrap();
    let expected_path = directory.path().join("config.toml");

    let path = ds_command(directory.path(), ["config", "path"]);
    assert!(path.status.success());
    assert_eq!(
        String::from_utf8(path.stdout).unwrap().trim(),
        expected_path.to_string_lossy()
    );

    let show = ds_command(directory.path(), ["config", "show"]);
    assert!(show.status.success());
    let output = String::from_utf8(show.stdout).unwrap();
    assert!(output.contains("version = 1"));
    assert!(output.contains("model = \"openai-codex/gpt-5.6-luna\""));
    assert!(output.contains("max_turns = 24"));
}

#[test]
fn cli_model_overrides_config_model() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("config.toml"),
        "version = 1\nmodel = \"missing/model\"\nmax_turns = 24\n",
    )
    .unwrap();

    let output = ds_command(
        directory.path(),
        ["--model", "openai/gpt-5.6-luna", "hello"],
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("provider openai is not configured"));
    assert!(!stderr.contains("unknown model"));
}

#[test]
fn auth_status_does_not_depend_on_valid_config() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("config.toml"), "version = [").unwrap();

    let output = ds_command(directory.path(), ["auth", "status", "openai"]);

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        "openai\tnot configured"
    );
}

#[test]
fn double_dash_allows_prompt_starting_with_command_name() {
    let directory = tempfile::tempdir().unwrap();

    let output = ds_command(directory.path(), ["--", "login", "flow"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("provider openai-codex is not configured"));
    assert!(!stderr.contains("required arguments were not provided"));
}

#[tokio::test]
async fn auth_status_reports_persistent_source_without_printing_secret() {
    let directory = tempfile::tempdir().unwrap();
    let store = PersistentCredentialStore::new(
        directory.path().join("auth.json"),
        directory.path().join("auth.lock"),
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let secret = "test-secret-that-must-stay-redacted";
    store
        .modify(
            "openai",
            Box::new(move |_| {
                Box::pin(async move {
                    Ok(Some(Credential::ApiKey {
                        key: Some(secret.into()),
                        env: BTreeMap::new(),
                    }))
                })
            }),
            &cancellation,
        )
        .await
        .unwrap();

    let output = ds_command(directory.path(), ["auth", "status", "openai"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stdout.contains("openai\tapi-key\tstored credential"));
    assert!(!stdout.contains(secret));
    assert!(!stderr.contains(secret));
}

fn ds_command<const N: usize>(home: &std::path::Path, args: [&str; N]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ds"))
        .env("DS_HOME", home)
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("ANTHROPIC_OAUTH_TOKEN")
        .env_remove("ANTHROPIC_API_KEY")
        .args(args)
        .output()
        .unwrap()
}
