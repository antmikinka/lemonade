# PR evidence packs — GUI3 series (+ held upstream #1984 branch)

One pack per PR. Each pack contains everything needed to fill the
`.github/pull_request_template.md` sections by hand:

- the template skeleton with the checkbox states that match the facts,
- the exact diffstat of the merged commit (code block),
- the test commands that were run and their observed output (code block),
- screenshot links from the `screenshots` branch (graphics),
- bare fact bullets to be **rewritten in your own words** before posting.

## Why there is no finished prose in here

`docs/dev/ai-content-policy.md` Rule 1: AI prose is not permitted in any
Issue, Discussion, PR body, or PR comment; posted prose must be hand-written
by the human posting it. Code, tables, and graphics are explicitly not prose,
so those travel in these packs verbatim. The Summary/Testing sentences must be
composed by hand in the browser form from the fact bullets.

## Workflow per PR

1. Open the pack below, open the PR page (merged PRs: edit the body in the
   browser) or the compare URL (unopened PRs).
2. Copy the template skeleton, tick the boxes as pre-filled.
3. Paste the diffstat / test-output code blocks and screenshot links as-is.
4. Write Summary and Testing sentences yourself from the fact bullets.
5. Submit.

## Index

| Pack | PR | Status | Issue | Merge |
|------|----|--------|-------|-------|
| [pr02-rename-search.md](pr02-rename-search.md) | #2 | merged | #1 | 211356aee |
| [pr04-attachments.md](pr04-attachments.md) | #4 | merged | #3 | 7b4fe0bd9 |
| [pr06-export.md](pr06-export.md) | #6 | merged | #5 | 87931a6cb |
| [pr07-ci.md](pr07-ci.md) | #7 | merged | — | 5596c6560 |
| [pr09-auto-titles.md](pr09-auto-titles.md) | #9 | merged | #8 | 6c8bd2c13 |
| [pr11-rail-polish.md](pr11-rail-polish.md) | #11 | merged | #10 | bcd77cc14 |
| [pr13-attachment-hardening.md](pr13-attachment-hardening.md) | #13 | merged | #12 | 364a31d34 |
| [pr15-queue-chat.md](pr15-queue-chat.md) | #15 | merged | #14 | ace611cb6 |
| [upstream-1984-file-support.md](upstream-1984-file-support.md) | not opened | HELD | lemonade-sdk#1984 | branch feat/1984-chat-file-support @2c7defe39 |

Screenshot base URL: `https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/<file>.png`
