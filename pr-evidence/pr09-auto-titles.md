# Pack: PR #9 — auto-name conversations

| | |
|---|---|
| PR | https://github.com/antmikinka/lemonade/pull/9 (merged) |
| Title used | `feat(gui3): auto-name conversations after the first exchange` |
| Branch | `feat/gui3-auto-titles` → `GUI3_squashed` |
| Issue | #8 (closed) |
| Merge commit | `6c8bd2c13` |
| CI | https://github.com/antmikinka/lemonade/actions/runs/37360033108 (success) |
| Apply at | edit the PR body in the browser: https://github.com/antmikinka/lemonade/pull/9 |

## Template skeleton (tick as shown)

```markdown
## Spec Driven Development

This PR:
- [x] fixes something (closes #8) and does not need a WG/RFC.
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
 .github/workflows/gui3-renderer-tests.yml          |  1 +
 src/app/package.json                               |  1 +
 src/app/src/components/ChatView.tsx                | 44 +++++++++++++-
 .../src/features/chatHistory/conversationTitle.ts  | 52 +++++++++++++++++
 src/app/tests/conversation-title.runtime.cjs       | 66 +++++++++++++++++++++
 src/app/tests/conversation-title.unit.mjs          | 68 ++++++++++++++++++++++
 6 files changed, 230 insertions(+), 2 deletions(-)
```

## Test commands + observed output (paste as-is)

Re-captured 2026-10-06 on `GUI3_squashed` (tree identical to merged state):

```
$ npm run test:conversation-title         # in src/app
Conversation title contract checks passed.
Conversation title unit checks passed.

$ npx tsc --noEmit                        # clean, no output
$ npm run build:renderer:prod             # webpack compiled successfully
$ npx playwright test tests/a11y.spec.ts  # green
```

CI (workflow added by #7):

```
https://github.com/antmikinka/lemonade/actions/runs/37360033108  → success
```

Live check: asked a local model to "plan a sweet weekend trip to Kyoto"; after the
first exchange the rail row renamed itself from the input snippet to the
model-generated title (e.g. "Kyoto Weekend Temple Food Plan"); export then
downloads as `<Title>.md`.

## Screenshots (paste markdown as-is)

```markdown
![Rail before the first exchange — snippet title](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/autotitle-before.png)
![Rail after auto-title applied](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/autotitle-after.png)
![Full-title row after auto-title](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/autotitle-after-full.png)
![Manually renamed conversations are never auto-retitled](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/autotitle-rename-guard.png)
```

## Fact bullets — REWRITE IN YOUR OWN WORDS before posting (ai-content-policy Rule 1)

- New pure module `src/app/src/features/chatHistory/conversationTitle.ts`: builds the auto-title request prompt, parses/validates the model's title, decides eligibility (only untouched snippet titles; manual renames are guarded).
- ChatView consumes it after the first assistant exchange completes (`handleStreamDone` path); failures fall back silently to the snippet title — titling never blocks chat.
- Runs against whatever chat model is loaded; no new endpoint, no server change.
- Tests: runtime contract + unit suites, `test:conversation-title` script, wired into `gui3-renderer-tests.yml`.
- Closes fork issue #8.
