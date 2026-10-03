# Fork Divergence

This repository (`antmikinka/lemonade`) is a personal fork of `lemonade-sdk/lemonade`.
This document records every intentional divergence from upstream so syncs stay predictable.

## Fork-only files

These exist only on the fork. They are pure additions, so an upstream sync can never
conflict on them.

| Path | Purpose |
|------|---------|
| `.github/workflows/gui3_beta_build.yml` | Beta build of the GUI3 desktop app |
| `.github/workflows/gui3-renderer-tests.yml` | Renderer test suites on PRs to `GUI3_squashed` |
| `docs/dev/fork-divergence.md` | This document |
| branch `GUI3_squashed` | Long-lived GUI3 integration branch (fork default target) |

## Disabled workflows

Both workflows below depend on secrets owned by the upstream repository and can never
succeed on the fork. They are disabled via the Actions API (zero file diff, survives
syncs). Root causes: `auto_label.py` exits without `ANTHROPIC_API_KEY`; the triage
dashboard push to `lemonade-sdk/lemonade-triage` fails without `DEPLOY_TOKEN` (exit 128).

| ID | Workflow | Missing secret |
|----|----------|----------------|
| 373939427 | Auto-label issues and PRs | `ANTHROPIC_API_KEY` |
| 373939439 | Triage dashboard | `DEPLOY_TOKEN` |

Re-enable (only if the fork ever gains the secrets):

```bash
gh api -X PUT repos/antmikinka/lemonade/actions/workflows/373939427/enable
gh api -X PUT repos/antmikinka/lemonade/actions/workflows/373939439/enable
```

## Upstream sync conflict policy

**GUI3 architecture wins.** When syncing upstream into `GUI3_squashed`, resolve every
conflict in favor of the GUI3 implementation.

Once upstream merges `lemonade-sdk/lemonade#1984` (chat file attachments), expect
delete-vs-modify conflicts on:

- `src/app/src/renderer/features/chat/LLMChatPanel.tsx`
- `src/app/src/renderer/styles.css`
- `src/app/src-tauri/tauri.conf.json`
- `test/app/app-regression/fileAttachments.test.cjs`

GUI3 already implements attachments (fork PR #4), so the GUI3 side of each conflict is
the one to keep.

**Ordering rule:** land the upstream `#1984` PR first, sync afterward. Do not sync
while `#1984` is still open — it would turn a clean merge into the delete-vs-modify
mess described above.

## Stale renderer suites (follow-up)

These 10 suites fail at pre-PR base `ab2f9cd7b` with byte-identical failures — stale
source-contract regexes, unrelated to fork work. They are excluded from
`gui3-renderer-tests.yml` and tracked here for refresh:

- `test:model-providers`
- `test:router-editor`
- `test:global-model-settings`
- `test:configuration-consistency`
- `test:directory-settings`
- `test:model-nav-filters`
- `test:download-polling`
- `test:router-load`
- `test:default-chat-model` (regex refresh)
- `test:model-list-backends` (model-list-backend-layout regex refresh)
