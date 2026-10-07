# Pack: UPSTREAM PR (not yet opened) — chat file support for lemonade-sdk#1984

| | |
|---|---|
| Target | https://github.com/lemonade-sdk/lemonade — base `main`, fixes issue #1984 |
| Status | **HELD** — do not open without explicit go-ahead |
| Branch | `feat/1984-chat-file-support` @ `2c7defe39` (rebased onto upstream/main, full gate pass done) |
| Track | upstream-legacy app (`src/app/src/renderer/…`), NOT the GUI3 track — separate implementation by design (no PDF, per scope decision on #1984) |
| Compare URL | https://github.com/lemonade-sdk/lemonade/compare/main...antmikinka:lemonade:feat/1984-chat-file-support?expand=1 |
| Rebase first | upstream/main moves (was `e4bcbd643` on 2026-10-06) — rebase and re-run gates before opening |

## Title to use

```
feat(app): attach plaintext and code files in chat (fixes #1984)
```

## Template skeleton (upstream `.github/pull_request_template.md` — tick as shown)

```markdown
## Spec Driven Development

This PR:
- [x] fixes something (closes #1984) and does not need a WG/RFC.
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

## Diffstat vs upstream/main (paste as-is; re-generate after the rebase)

```
 src/app/src-tauri/tauri.conf.json                  |   3 +-
 .../src/renderer/components/FilePreviewList.tsx    |  46 ++++
 src/app/src/renderer/components/Icons.tsx          |  31 +++
 .../renderer/components/panels/LLMChatPanel.tsx    | 247 ++++++++++++++++--
 src/app/src/renderer/utils/chatTypes.ts            |  17 +-
 src/app/src/renderer/utils/fileAttachments.ts      | 275 +++++++++++++++++++++
 src/app/styles/styles.css                          | 134 ++++++++++
 test/app/app-regression/fileAttachments.test.cjs   | 238 ++++++++++++++++++
 8 files changed, 974 insertions(+), 17 deletions(-)
```

## Test commands + observed output (paste as-is)

Re-captured 2026-10-06 on the branch (`2c7defe39`), from the repo root:

```
$ node test/app/run-app-regression-tests.cjs
PASS fileAttachments.test.cjs :: classifyFile routes by MIME first
... (18 fileAttachments cases total)
PASS fileAttachments.test.cjs :: preview chips key on stable attachment ids
App regression tests: 154 passed, 0 skipped, 0 failed.
```

Also run on the branch at gate time (task #28):

```
$ npx tsc --noEmit                                  # in src/app — clean
$ npm run build:renderer:prod                       # webpack compiled successfully
```

Native verification: `tauri dev` desktop build exercised the full attach → send →
backend round-trip against a local lemond (task #30); attachments survive the
`window.api`/tauriShim contract unchanged.

## Screenshots

None on the screenshots branch for this track (legacy UI). If you want visuals,
capture fresh ones from `tauri dev` on the branch before opening — legacy-UI
screenshots do not exist yet.

## Fact bullets — REWRITE IN YOUR OWN WORDS before posting (ai-content-policy Rule 1)

- Implements lemonade-sdk/lemonade#1984: attach plaintext/code files to chat messages in the desktop (legacy renderer) app.
- New `src/app/src/renderer/utils/fileAttachments.ts`: `UploadedFile` with stable `createFileAttachmentId()`, text decoding, size/count caps, content fencing; `convertContentForRequest` folds attachments into the outgoing request.
- `LLMChatPanel.tsx`: attach button, drag-drop, preview list (`FilePreviewList.tsx`), removal by id, error surfacing.
- Scope decision: text/code only — no PDF (pdfjs dependency deliberately avoided on this track; GUI3 fork track has PDF separately).
- Regression suite `test/app/app-regression/fileAttachments.test.cjs` (238 lines) pins classification, fencing, caps, and panel wiring.
- Two commits on the branch: `ebc840fd5` (feature) + `2c7defe39` (hardening: fencing + preview keys).

## Opening checklist (when go-ahead is given)

1. `git fetch upstream && git rebase upstream/main feat/1984-chat-file-support`
2. Re-run the gate block above; re-generate the diffstat (`git diff --stat upstream/main...`).
3. Push (`--force-with-lease` after rebase), open PR via the compare URL, base `lemonade-sdk/lemonade:main`.
4. Body: skeleton + pasted code blocks + hand-written Summary/Testing sentences.
