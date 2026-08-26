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

Set the environment variable for your provider. Then pass a model as `provider/model`:

```sh
export OPENAI_API_KEY=...
cargo run -p ds-coding-agent -- --model openai/gpt-5.6-luna
```

The interactive UI stays in the normal terminal screen. Completed messages move into native scrollback. Press Enter to send, Ctrl-J to insert a newline, and Ctrl-C to cancel an active request. When the prompt is empty, Ctrl-C exits.

Pass a prompt to run one request without the interactive UI:

```sh
cargo run -p ds-coding-agent -- \
  --model anthropic/claude-sonnet-4-5 \
  "summarize this repository"
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
