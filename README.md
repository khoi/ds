# ds - the minimal coding agent

A minimal coding agent heavily inspired by Pi.

- ds-ai - LLM providers API, translate OpenAI, OpenAI Codex, Anthropic into ds provider agnostic events.
- ds-agent-core - The agent runtime (loop, messages, tools, events, sessions), this has no UI or TUI logic.
- ds-coding-agent - The coding agent cli TUI

## Setup

Install [mise](https://mise.jdx.dev), then run:

```sh
mise trust
mise install
```

The install step sets up the pinned Rust toolchain, project tools, and Git hooks.

## Send one request

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
