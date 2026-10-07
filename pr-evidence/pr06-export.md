# Pack: PR #6 — export conversations as Markdown

| | |
|---|---|
| PR | https://github.com/antmikinka/lemonade/pull/6 (merged) |
| Title used | `feat(gui): export conversations as Markdown from the chat rail` |
| Branch | `feat/gui3-conversation-export` → `GUI3_squashed` |
| Issue | #5 (closed) |
| Merge commit | `87931a6cb` |
| CI | predates `gui3-renderer-tests` (added in #7) — local gates only |
| Apply at | edit the PR body in the browser: https://github.com/antmikinka/lemonade/pull/6 |

## Template skeleton (tick as shown)

```markdown
## Spec Driven Development

This PR:
- [x] fixes something (closes #5) and does not need a WG/RFC.
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
 src/app/package.json                               |   1 +
 src/app/src/components/ChatView.tsx                |  79 ++++++++++++
 src/app/src/components/WorkspacePanels.tsx         |  56 +++++++--
 .../src/features/chatHistory/conversationExport.ts |  80 +++++++++++++
 src/app/src/styles/styles.css                      |  22 +++-
 src/app/tests/a11y.spec.ts                         |  53 +++++++--
 src/app/tests/conversation-export.runtime.cjs      | 132 +++++++++++++++++++++
 src/app/tests/conversation-export.unit.mjs         | 130 ++++++++++++++++++++
 8 files changed, 530 insertions(+), 23 deletions(-)
```

## Test commands + observed output (paste as-is)

Re-captured 2026-10-06 on `GUI3_squashed` (tree identical to merged state):

```
$ npm run test:conversation-export        # in src/app
Conversation export contract checks passed.
Conversation export unit checks passed.

$ npx tsc --noEmit                        # clean, no output
$ npm run build:renderer:prod             # webpack compiled successfully
$ npx playwright test tests/a11y.spec.ts  # green
```

## Screenshots (paste markdown as-is)

```markdown
![Export actions in the thread header](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/export-action-stack.png)
![Export action on the rail row menu](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/export-action-stack-rail.png)
![Exported Markdown thread](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/export-thread.png)
![Download toast with the exported filename](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/export-toast.png)
```

## Fact bullets — REWRITE IN YOUR OWN WORDS before posting (ai-content-policy Rule 1)

- New feature module `src/app/src/features/chatHistory/conversationExport.ts`: renders a conversation to Markdown (roles, code fences, attachments) and derives a safe filename (`conversationExportFilename` — sanitizes illegal chars, always appends `.md`, falls back to `conversation.md`).
- Two entry points: copy-Markdown and download-`.md` from the thread header action stack and from the rail row menu (WorkspacePanels).
- Download toast names the file; verified with Playwright `page.waitForEvent('download')` → `suggestedFilename()`.
- Tests: runtime contract suite + unit suite + a11y updates; `test:conversation-export` script.
- Closes fork issue #5.
