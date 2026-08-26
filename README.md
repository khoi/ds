# ds - the minimal coding agent

A minimal coding agent inspired by Pi.

- `ds-ai` translates OpenAI, OpenAI Codex, and Anthropic streams into provider-neutral events.
- `ds-agent-core` owns the conversation loop, tool registry, and agent events.
- `ds-coding-agent` provides the `ds` command, local coding tools, and inline terminal UI.

## Setup

Install [mise](https://mise.jdx.dev), then run:

```sh
mise trust
mise install
```

The install step sets up the pinned Rust toolchain, project tools, and Git hooks.

## Run the coding agent

Sign in once, then start `ds`:

```sh
cargo run -p ds-coding-agent -- login openai-codex
cargo run -p ds-coding-agent
```

`ds` stores provider-neutral credentials in `~/.ds/auth.json`. On Unix, the directory is mode `0700` and the credential and lock files are mode `0600`. Credentials are stored in this file on every platform; `ds` does not use Keychain.

API-key providers use the same persistent login flow:

```sh
cargo run -p ds-coding-agent -- login openai
cargo run -p ds-coding-agent -- login anthropic
```

The key prompt does not echo. Existing provider environment variables still work when no saved credential exists.

Inspect or remove saved authentication without printing secrets:

```sh
cargo run -p ds-coding-agent -- auth status
cargo run -p ds-coding-agent -- logout openai-codex
```

The default model is `openai-codex/gpt-5.6-luna`. Override it for one run with `--model`, or create `~/.ds/config.toml`:

```toml
version = 1
model = "anthropic/claude-sonnet-4-5"
max_turns = 24
reasoning = "high"
```

CLI flags override the global config. `DS_HOME` overrides the `~/.ds` directory. These commands show the effective configuration without creating a file:

```sh
cargo run -p ds-coding-agent -- config path
cargo run -p ds-coding-agent -- config show
cargo run -p ds-coding-agent -- config check
```

The interactive UI stays in the normal terminal screen. Completed messages move into native scrollback. Press Enter to send, Ctrl-J to insert a newline, and Ctrl-C to cancel an active request. When the prompt is empty, Ctrl-C exits.

Pass a prompt to run one request without the interactive UI:

```sh
cargo run -p ds-coding-agent -- \
  --model anthropic/claude-sonnet-4-5 \
  "summarize this repository"
```

`login`, `logout`, `auth`, and `config` are command names. Use `--` when a one-shot prompt begins with one of those words:

```sh
cargo run -p ds-coding-agent -- -- "login flows in this repository"
```

The agent has `read`, `bash`, `edit`, and `write`. These tools run with the `ds` process privileges. `ds` does not add permissions, project trust checks, a sandbox, or path confinement.

## Use `ds-ai` directly

Set an OpenAI API key, then run the checked example:

```sh
export OPENAI_API_KEY=...
cargo run -p ds-ai --example openai
```

The full request path is:

```rust
use ds_ai::{
    Context, Message, OpenAiResponsesOptions, StopReason, builtin_models,
    builtin_openai_model, content_text,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = builtin_models();
    let model = builtin_openai_model("gpt-5.6-luna").expect("model in built-in catalog");
    let context = Context::new([Message::user("Explain this repository in one sentence")]);
    let response = models
        .complete(&model, &context, &OpenAiResponsesOptions::default())
        .await?;

    if matches!(response.stop_reason, StopReason::Error | StopReason::Aborted) {
        return Err(std::io::Error::other(
            response.error_message.unwrap_or_else(|| "request failed".into()),
        )
        .into());
    }

    println!("{}", content_text(&response.content));
    Ok(())
}
```

`model` carries its API type, so `complete` only accepts `OpenAiResponsesOptions` here.
