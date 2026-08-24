# Repository rules

- Use `mise install` to set up tools and hooks.
- Use `mise run fix` before staging changes.
- Run the narrowest useful test while working.
- Run `mise run ci` before push.
- Add a regression test for each fixed bug.
- Keep provider-specific code inside its provider module.
- Keep the public API provider-neutral.
- Do not add unsafe code.
- Do not suppress lint rules.
- Keep dependency and lockfile changes scoped.
