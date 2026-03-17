# Copilot Instructions for Frigate Identity Service

## Commit Message Convention

When generating commit messages in VS Code (including the **Generate Commit Message** button), use this format:

- `<type>(<scope>): <summary>`
- If scope is not clear, use `<type>: <summary>`
- Keep summary in imperative mood and user-facing where possible
- Do not include trailing period in the summary line

Allowed `type` values:

- `feat` for new user-facing behavior
- `fix` for bug fixes
- `refactor` for internal structural changes without behavior change
- `perf` for performance improvements
- `update` for non-feature updates that still change behavior/config/docs
- `remove` for removals/deprecations

Scope guidance:

- Use concrete module scopes when possible: `identity_service`, `matcher`, `embedding_store`, `mqtt`, `docker`, `config`, `release`, `tests`, `docs`, `ci`

Examples:

- `feat(identity_service): correlate mqtt snapshots with recent detections`
- `fix(mqtt): handle broker reconnect without dropping subscriptions`
- `refactor(embedding_store): simplify retention pruning logic`
- `update(docs): clarify add-on configuration defaults`
- `remove(config): drop deprecated mqtt_host alias`
