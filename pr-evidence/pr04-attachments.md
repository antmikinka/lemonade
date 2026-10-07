# Pack: PR #4 — text/code + PDF chat attachments

| | |
|---|---|
| PR | https://github.com/antmikinka/lemonade/pull/4 (merged) |
| Title used | `feat(gui): attach text, code, and PDF files to GUI3 chat messages` |
| Branch | `feat/gui3-chat-file-attachments` → `GUI3_squashed` |
| Issue | #3 (closed) |
| Merge commit | `7b4fe0bd9` |
| CI | predates `gui3-renderer-tests` (added in #7) — local gates only |
| Apply at | edit the PR body in the browser: https://github.com/antmikinka/lemonade/pull/4 |

## Template skeleton (tick as shown)

```markdown
## Spec Driven Development

This PR:
- [x] fixes something (closes #3) and does not need a WG/RFC.
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
 .github/actions/prepare-debian-build/action.yml    |   8 +
 src/app/package-lock.json                          | 263 ++++++++++++++++++++
 src/app/package.json                               |   2 +
 src/app/src/components/ChatView.tsx                | 192 ++++++++++++--
 .../features/chatAttachments/fileAttachments.ts    | 262 +++++++++++++++++++
 src/app/src/features/chatAttachments/pdfText.ts    |  66 +++++
 src/app/src/styles/styles.css                      |   7 +
 src/app/tests/a11y.spec.ts                         |  12 +-
 src/app/tests/chat-file-attachments.runtime.cjs    | 164 ++++++++++++
 src/app/tests/chat-file-attachments.unit.mjs       |  79 ++++++
 src/web-app/.gitignore                             |   8 +
 src/web-app/package-lock.json                      | 276 +++++++++++++++++++++
 src/web-app/package.json                           |   1 +
 src/web-app/system-stubs/pdfjs-dist.ts             |  16 ++
 src/web-app/system-stubs/pdfjs-worker-stub.mjs     |   2 +
 src/web-app/webpack.config.js                      |   4 +
 16 files changed, 1337 insertions(+), 25 deletions(-)
```

## Test commands + observed output (paste as-is)

Re-captured 2026-10-06 on `GUI3_squashed` (tree identical to merged state):

```
$ npm run test:chat-file-attachments      # in src/app
Chat file/PDF attachment contract checks passed.
Chat file attachment unit checks passed.

$ npx tsc --noEmit                        # clean, no output
$ npm run build:renderer:prod             # webpack compiled successfully
$ npx playwright test tests/a11y.spec.ts  # green
```

Live verification (2026-10-06, localhost:9123 + lemond :13305):

- PDF path: reportlab-generated test PDF whose secret values exist ONLY inside the PDF;
  the model (Huihui-Qwen3.8-27B) answered with the PDF-only values → extraction works end-to-end.
- Text path: `alpha.txt` dropped on the composer attaches, folds into the prompt via
  `composePromptWithFiles`, and renders as a chip in composer + transcript.

## Screenshots (paste markdown as-is)

```markdown
![Composer chips after attaching text/code files](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-attachment-composer-chips.png)
![Attached files rendered in the sent transcript](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-attachment-transcript.png)
![PDF attached as a composer chip](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-pdf-composer-chip.png)
![Model answering from PDF-only content](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-pdf-transcript.png)
```

## Fact bullets — REWRITE IN YOUR OWN WORDS before posting (ai-content-policy Rule 1)

- New feature module `src/app/src/features/chatAttachments/`: `fileAttachments.ts` (classification image/audio/pdf/text/unsupported, 1 MB + 4-file caps, BOM/UTF-16 decoding, binary sniffing, dynamic backtick fencing, `composePromptWithFiles` folds files into plain prompt text) and `pdfText.ts` (lazy `pdfjs-dist` chunk, bundled worker asset URL, 50-page cap, `doc.destroy()` cleanup).
- ChatView wiring: attach button/menu, drag-drop, clipboard paste, pending-file chips with accessible remove buttons, error banner (`role="status"`), attachments cleared on send/capability-switch, stripped from localStorage persistence, re-folded on history replay.
- Capability gating: documents only in chat-completions mode (`acceptsFileAttachments`).
- web-app: `pdfjs-dist` stubbed for Debian `USE_SYSTEM_NODEJS_MODULES` builds (system ships no v4 worker); lockfiles updated on both apps (Critical Invariant #8 respected — no package.json consolidation).
- Tests: runtime contract suite + unit suite + a11y updates; `test:chat-file-attachments` script.
- Closes fork issue #3.
