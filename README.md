# ds - the minimal coding agent

A minimal coding agent heavily inspired by Pi.

## Setup

Install [mise](https://mise.jdx.dev), then run:

```sh
mise trust
mise install
```

The install step sets up the pinned Rust toolchain, project tools, and Git hooks.

## Checks

```sh
mise run fix
mise run ci
```

The pre-commit hook fixes file hygiene and formatting. The pre-push hook runs the full CI gate.
