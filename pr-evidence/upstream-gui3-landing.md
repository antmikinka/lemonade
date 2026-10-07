# Pack: UPSTREAM landing PR — all 8 GUI3 fork features → lemonade-sdk GUI3_squashed

| | |
|---|---|
| Target | `lemonade-sdk/lemonade`, base **`GUI3_squashed`** (the GUI3 beta/RC line — confirm with Primal that this is the "RC" he meant) |
| Branch | `feat/gui3-upstream-landing` @ origin (14 commits: the 13 fork commits cherry-picked + 1 pin adjustment), built on upstream `efdf99413` |
| Mandate | Primal on Discord: *"Let's just get it merged to RC so we can start using it and then adding everything we want vs having parity."* |
| Open PR at | https://github.com/lemonade-sdk/lemonade/compare/GUI3_squashed...antmikinka:lemonade:feat/gui3-upstream-landing?expand=1 |
| Gate status | **FULL GREEN** on 2026-10-06 (details below) |
| Status | READY — waiting on: (1) Primal confirms RC target, (2) you hand-write the body and open it |

## Title to use

```
feat(gui3): land the fork feature set — rail rename/search, file+PDF attachments, export, auto-titles, resizable rail, queue chat
```

(Shorter alternative: `feat(gui3): fork feature pack for the RC line (8 features, tests included)`)

## Template skeleton (tick as shown)

```markdown
## Spec Driven Development

This PR:
- [ ] fixes something <!-- (closes #issue-number) --> and does not need a WG/RFC.
- [ ] is within the scope of WG: <!-- working group name -->
- [ ] has approved RFC #<!--discussion number -->
```

**Stop — decide this with Primal before opening.** These are UX-surface features, so the
template normally wants a WG or RFC. Primal's Discord message is de-facto approval, but
the form wants a box ticked. Ask him (30-second question): *"For the GUI3 feature-pack
PR into GUI3_squashed — do you want an RFC discussion opened, or does your Discord
go-ahead count and I note it in the Summary?"* Then either tick the RFC box with the
discussion number, or tick the first box and quote his message in the Summary. The rest:

```markdown
## Scope

- [x] This PR addresses one clear issue or change. <!-- one landing; per-feature decomposition table below -->
- [x] I reviewed the full diff myself before submitting.
- [x] I removed unrelated local changes.
- [x] I kept refactoring separate unless it is required for this change.

## Testing

- [x] The code change has been locally tested.

_Testing details:_

<< paste the diffstat + gate-output code blocks below >>

## Documentation

- [x] Documentation is not affected by this change.
- [ ] Documentation is affected and has been updated.

## Breaking Changes

- [ ] This PR introduces breaking changes.
- [x] This PR does not introduce breaking changes.
```

## Diffstat vs upstream/GUI3_squashed @efdf99413 (paste as-is)

```
 .github/actions/prepare-debian-build/action.yml    |   8 +
 src/app/package-lock.json                          | 263 ++++++
 src/app/package.json                               |   6 +
 src/app/src/components/ChatView.tsx                | 890 +++++++++++++++++++--
 src/app/src/components/WorkspacePanels.tsx         |  56 +-
 .../features/chatAttachments/fileAttachments.ts    | 273 +++++++
 src/app/src/features/chatAttachments/pdfText.ts    |  66 ++
 .../features/chatHistory/conversationExport.ts     |  82 ++
 src/app/src/features/chatHistory/conversationTitle.ts |  52 ++
 src/app/src/features/chatQueue.ts                  |  38 +
 src/app/src/styles/styles.css                      | 243 +++++-
 src/app/tests/a11y.spec.ts                         |  65 +-
 src/app/tests/chat-file-attachments.runtime.cjs    | 164 ++++
 src/app/tests/chat-file-attachments.unit.mjs       |  79 ++
 src/app/tests/chat-queue.runtime.cjs               | 126 +++
 src/app/tests/chat-queue.unit.mjs                  |  70 ++
 src/app/tests/conversation-export.runtime.cjs      | 136 ++++
 src/app/tests/conversation-export.unit.mjs         | 137 ++++
 src/app/tests/conversation-rail.runtime.cjs        | 157 ++++
 src/app/tests/conversation-title.runtime.cjs       |  62 ++
 src/app/tests/conversation-title.unit.mjs          |  68 ++
 src/web-app/.gitignore                             |   8 +
 src/web-app/package-lock.json                      | 276 +++++++
 src/web-app/package.json                           |   1 +
 src/web-app/system-stubs/pdfjs-dist.ts             |  16 +
 src/web-app/system-stubs/pdfjs-worker-stub.mjs     |   2 +
 src/web-app/webpack.config.js                      |   4 +
 test/app/app-regression/gui3.test.cjs              |  96 +++
 28 files changed, 3339 insertions(+), 105 deletions(-)
```

## Gate output (paste as-is) — all captured 2026-10-06 on the landing branch

```
$ npx tsc --noEmit                                   # clean
$ node test/app/run-app-regression-tests.cjs         # repo root
App regression tests: 8 passed, 0 skipped, 0 failed.
$ npm run test:<suite> for all 16 renderer suites    # chat-audio, conversation-rail,
  conversation-export, conversation-title, chat-file-attachments, chat-queue,
  model-kinds, mcp-mock, mcp-runtime, router-store, icons, storage,
  effective-settings, sampler-args, ui-regressions, model-state
PASS × 16
$ npx playwright test tests/a11y.spec.ts
148 passed
$ npm run build:renderer:prod
webpack 5.107.2 compiled successfully (pdfjs ships as its own lazy chunk)
```

## Per-feature decomposition (paste the table as-is)

Every feature was separately reviewed, CI-gated, and merged on the fork first:

| Feature | Fork PR | Fork issue | Fork CI |
|---|---|---|---|
| Rail rename + search | antmikinka/lemonade#2 | #1 | pre-workflow, local gates |
| Text/code + PDF attachments | antmikinka/lemonade#4 | #3 | pre-workflow, local gates |
| Markdown export | antmikinka/lemonade#6 | #5 | pre-workflow, local gates |
| Renderer-suite CI (fork only, NOT included) | antmikinka/lemonade#7 | — | [run](https://github.com/antmikinka/lemonade/actions/runs/37163250298) |
| Auto-titles | antmikinka/lemonade#9 | #8 | [run](https://github.com/antmikinka/lemonade/actions/runs/37360033108) |
| Resizable rail + export filename polish | antmikinka/lemonade#11 | #10 | [run](https://github.com/antmikinka/lemonade/actions/runs/37378642247) |
| Attachment hardening (stable ids, named rejections) | antmikinka/lemonade#13 | #12 | [run](https://github.com/antmikinka/lemonade/actions/runs/37442460034) |
| Queue chat mid-stream | antmikinka/lemonade#15 | #14 | [run](https://github.com/antmikinka/lemonade/actions/runs/37541920063) |

## Screenshots (paste markdown as-is)

```markdown
|Rename + search|Auto-titles|
|---|---|
|![rename+search](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-rename-search-rail.png)|![auto-title](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/autotitle-after.png)|

|Attachments (PDF read end-to-end)|Queue strip mid-stream|
|---|---|
|![pdf](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-pdf-transcript.png)|![queue](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-queue-strip.png)|

|Resizable rail|Export toast|
|---|---|
|![rail](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/rail-polish-2-resized-full-title.png)|![export](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/export-toast.png)|
```

## Deliberate exclusions / adjustments (facts for the body)

- `.github/workflows/gui3-renderer-tests.yml` and `docs/dev/fork-divergence.md` are NOT included — fork CI/governance. The fork runs the full renderer-suite gate on every PR; upstream can adopt the workflow separately if wanted (it's ~60 lines and green on 5 consecutive fork PRs).
- Two test pins that asserted fork-workflow wiring were dropped (commit `test(gui3): drop fork-CI workflow pins…`); everything else in the suites is unchanged.
- `prepare-debian-build` deb-version fix IS included — `git describe --always` can return a hash starting a–f, which dpkg rejects; the fix prefixes `0.0.0-g`. Applies to any clone without reachable tags.
- Built directly on upstream `efdf99413` (today's tip, incl. the model-select focus change) — all ChatView pins verified against it, no upstream behavior removed.

## Upstream bug found while gating (report separately — do NOT fix in this PR)

A01 (axe scan of default chat view) intermittently fails **only when the test browser
discovers a real running lemond with a model loaded**: the `loaded-overview` card
(upstream code, untouched by this branch — 0 diff hits) has two WCAG AA contrast
violations: `--cap-chat` capability chip `#a77c00` on `#232015` = **4.28:1**, and the
"Selected" pill `#7a776e` on `#25221b` = **3.54:1** (need 4.5:1). Repro: load any chat
model, run `npx playwright test tests/a11y.spec.ts -g A01` against a lemond-connected
renderer. Suggest filing as its own issue/PR — it predates this branch.

## Fact bullets — REWRITE IN YOUR OWN WORDS before posting (ai-content-policy Rule 1)

- Lands all 8 GUI3 features the fork built during the beta window, as one reviewed-on-fork pack, per Primal's "merge to RC" go-ahead (quote his line).
- Features: rail rename + search; text/code/PDF attachments (1 MB, 4-file cap, lazy pdfjs chunk, Debian web-app stubs); Markdown export (copy + download, safe filenames); auto-titles after first exchange (bounded 24-token request, manual renames always win, silent failure); resizable rail (200–480px, persisted client-side only per Critical Invariant #1, keyboard-accessible separator); attachment hardening (stable ids, id-based removal, named drop rejections); queue chat (10-message transient per-conversation queue, FIFO drain on idle/stop/error/reconnect, attachments ride queued messages, never persisted).
- Every feature carries regression tests: 6 new renderer suites + gui3.test.cjs cases in the repo-root runner; 3,339 insertions include ~1,300 lines of tests.
- All storage stays client-side (localStorage); nothing touches lemond's config or shared state; no server/C++ changes.
- pdfjs-dist added to src/app deps; web-app stubs keep `USE_SYSTEM_NODEJS_MODULES` builds working (Critical Invariant #8 — package.json files stay split).
- Full gate green on the exact merge-base tip efdf99413 (typecheck, 8/8 app-regression, 16/16 renderer suites, 148/148 a11y, prod build).

## Opening checklist

1. Ask Primal: RC = `GUI3_squashed`? WG/RFC box handling? (see above)
2. If `GUI3_squashed` moved: `git fetch upstream && git rebase upstream/GUI3_squashed feat/gui3-upstream-landing`, re-run gates, force-push with lease.
3. Open via the compare URL above; paste skeleton + code blocks + table + screenshots; hand-write Summary/Testing from the fact bullets.
4. After merge: fork `GUI3_squashed` can fast-forward onto upstream's (parity achieved in reverse), and the fork's separate `gui3-renderer-tests` workflow keeps gating future fork PRs.
