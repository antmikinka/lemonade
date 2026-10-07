# Pack: PR #15 — queue follow-up messages mid-stream

| | |
|---|---|
| PR | https://github.com/antmikinka/lemonade/pull/15 (merged) |
| Title used | `feat(gui3): queue follow-up messages while a chat stream runs` |
| Branch | `feat/gui3-queue-chat` → `GUI3_squashed` |
| Issue | #14 (closed) |
| Merge commit | `ace611cb6` |
| CI | https://github.com/antmikinka/lemonade/actions/runs/37541920063 (success, first pass) |
| Apply at | edit the PR body in the browser: https://github.com/antmikinka/lemonade/pull/15 |

## Template skeleton (tick as shown)

```markdown
## Spec Driven Development

This PR:
- [x] fixes something (closes #14) and does not need a WG/RFC.
- [ ] is within the scope of WG:
- [ ] has approved RFC #

## Summary

<< hand-write from the fact bullets below >>

## Scope

- [x] This PR addresses one clear issue or change.
- [x] I reviewed the full diff myself before submitting.
- [x] I removed unrelated local changes.
- [x] I kept refactoring separate unless it is required for this change.

## Testing

- [x] The code change has been locally tested.

_Testing details:_

<< paste the two code blocks below >>

## Documentation

- [x] Documentation is not affected by this change.
- [ ] Documentation is affected and has been updated.

## Breaking Changes

- [ ] This PR introduces breaking changes.
- [x] This PR does not introduce breaking changes.
```

## Diffstat (paste as-is)

```
 .github/workflows/gui3-renderer-tests.yml       |   1 +
 src/app/package.json                            |   1 +
 src/app/src/components/ChatView.tsx             | 256 ++++++++++++++++++++----
 src/app/src/features/chatQueue.ts               |  38 ++++
 src/app/src/styles/styles.css                   |  88 ++++++++
 src/app/tests/chat-file-attachments.runtime.cjs |   2 +-
 src/app/tests/chat-queue.runtime.cjs            | 130 ++++++++++++
 src/app/tests/chat-queue.unit.mjs               |  70 +++++++
 test/app/app-regression/gui3.test.cjs           |  41 +++-
 9 files changed, 581 insertions(+), 46 deletions(-)
```

## Test commands + observed output (paste as-is)

Re-captured 2026-10-06 on `GUI3_squashed` (tree identical to merged state):

```
$ npm run test:chat-queue                   # in src/app
Chat queue contract checks passed.
Chat queue unit checks passed.

$ node test/app/run-app-regression-tests.cjs  # repo root
PASS gui3.test.cjs :: chat queue helpers cap at ten, mint unique ids, and summarize drafts
App regression tests: 8 passed, 0 skipped, 0 failed.

$ npm run test:chat-file-attachments        # pin touched by this PR
Chat file/PDF attachment contract checks passed.
Chat file attachment unit checks passed.

$ npx tsc --noEmit                          # clean
$ npm run build:renderer:prod               # webpack compiled successfully
$ npx playwright test tests/a11y.spec.ts    # 148 tests green
```

Full CI parity gate before push: all 16 renderer runtime suites
(chat-audio, conversation-rail, conversation-export, conversation-title,
chat-file-attachments, chat-queue, model-kinds, mcp-mock, mcp-runtime,
router-store, icons, storage, effective-settings, sampler-args,
ui-regressions, model-state) — all pass.

CI:

```
https://github.com/antmikinka/lemonade/actions/runs/37541920063  → success
```

Live check (localhost:9123, lemond :13305, Huihui-Qwen3.8-27B):

- While a stream was running, typed follow-ups and hit "Queue message" → QUEUED n/10
  strip appeared above the composer with per-item remove buttons.
- Dropped `alpha.txt` (content `ALPHA=zoom`) on the composer mid-stream → attached to
  a queued message (attachment ingestion has no isBusy gate by design).
- Queued an 11th message → red cap notice, draft text preserved ("One message too many").
- Pressed Stop → queue drained FIFO; the model's reply quoted
  "DRAINED-TWO (the attached file alpha.txt contains: ALPHA=zoom)" → attachments ride
  queued messages and drain replays them correctly.
- Queued items vanish on conversation switch/export (transient, never persisted).

## Screenshots (paste markdown as-is)

```markdown
![Queue strip mid-stream with Stop + Queue message buttons](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-queue-strip.png)
![Cap notice at 10/10 with the rejected draft preserved](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-queue-cap-notice.png)
![Drained messages in the transcript; model quotes the queued attachment](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-queue-drained-transcript.png)
```

## Fact bullets — REWRITE IN YOUR OWN WORDS before posting (ai-content-policy Rule 1)

- New pure module `src/app/src/features/chatQueue.ts`: `MAX_QUEUED_MESSAGES = 10`, `createQueuedMessageId()` (unique even within one millisecond), `withQueueCap()` (11th message rejected, original queue returned untouched), `summarizeQueuedItem()` (plain text / 80-char ellipsis / `File: <name>` / `Empty message`).
- Per-conversation transient queue in ChatView; sending while a stream runs enqueues instead of blocking or interrupting.
- Effect-based FIFO drain when the conversation goes idle (complete, error, stop, reconnect, or switch back); `drainLockRef` prevents double-drain races.
- Attachments ride queued messages and re-fold at drain time; queue is never persisted to localStorage and never exported.
- Composer UI: `composer__queue*` strip inside `.composer` (QUEUED n/10, per-item summaries + remove, cap notice, "Queue message" send-button swap mid-stream).
- One pre-existing pin updated: the send-guard in `chat-file-attachments.runtime.cjs` (isBusy no longer gates send because queueing replaces the block).
- Tests: runtime + unit suites, `test:chat-queue` script, workflow line, new `gui3.test.cjs` case exercising the real module via the transpile-on-require loader.
- Closes fork issue #14.
