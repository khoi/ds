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
