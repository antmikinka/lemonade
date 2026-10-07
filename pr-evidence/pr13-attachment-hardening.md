# Pack: PR #13 — attachment hardening (stable ids + named rejections)

| | |
|---|---|
| PR | https://github.com/antmikinka/lemonade/pull/13 (merged) |
| Title used | `fix(gui3): stable attachment ids, id-based removal, visible drop rejections` |
| Branch | `feat/gui3-attachment-hardening` → `GUI3_squashed` |
| Issue | #12 (closed) |
| Merge commit | `364a31d34` |
| CI | https://github.com/antmikinka/lemonade/actions/runs/37442460034 (success) |
| Apply at | edit the PR body in the browser: https://github.com/antmikinka/lemonade/pull/13 |

## Template skeleton (tick as shown)

```markdown
## Spec Driven Development

This PR:
- [x] fixes something (closes #12) and does not need a WG/RFC.
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
 src/app/src/components/ChatView.tsx                | 30 +++++++----
 .../features/chatAttachments/fileAttachments.ts    | 11 ++++
 src/app/tests/chat-file-attachments.runtime.cjs    |  4 +-
 test/app/app-regression/gui3.test.cjs              | 61 ++++++++++++++++++++++
 4 files changed, 95 insertions(+), 11 deletions(-)
```

## Test commands + observed output (paste as-is)

Re-captured 2026-10-06 on `GUI3_squashed` (tree identical to merged state):

```
$ node test/app/run-app-regression-tests.cjs        # repo root
PASS gui3.test.cjs :: chat attachments mint stable ids and stay readable for pre-id history
PASS gui3.test.cjs :: ChatView keys attachment chips on stable ids and explains rejected drops
App regression tests: 8 passed, 0 skipped, 0 failed.

$ npm run test:chat-file-attachments                # in src/app
Chat file/PDF attachment contract checks passed.
Chat file attachment unit checks passed.

$ npx tsc --noEmit                                  # clean
$ npm run build:renderer:prod                       # webpack compiled successfully
```

CI:

```
https://github.com/antmikinka/lemonade/actions/runs/37438964785  → failure (mid-development)
https://github.com/antmikinka/lemonade/actions/runs/37442460034  → success (merged state)
```

Live check: switched to Transcription mode (Whisper-Base loaded) and dropped a
`.txt` file → visible rejection banner naming the file and explaining documents are
"not attachable in this mode"; chat mode unchanged.

## Screenshots (paste markdown as-is)

```markdown
![Document drop rejected by name in transcription mode](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/gui3-rejection-transcription-mode.png)
```

## Fact bullets — REWRITE IN YOUR OWN WORDS before posting (ai-content-policy Rule 1)

- Adds `createAttachmentId()` and an optional `id` on `AttachedFile`; both ingest paths in ChatView mint ids, composer chips key on `file.id ?? file.filename`, history chips fall back for storage saved before ids existed.
- Removal now filters by id (`prev.filter(f => f.id !== id)`) instead of a positional index — removing a middle chip no longer shifts and drops the wrong file.
- Drops in modes without a document sink now report a named rejection ("…not attachable in this mode") instead of vanishing silently.
- New `gui3.test.cjs` regression cases pin all of the above (transpile-on-require loader exercises the real feature module).
- Pre-id history stays readable: `wrapFileForPrompt`/`composePromptWithFiles` accept id-less legacy attachments.
- Closes fork issue #12.
