# Pack: PR #7 — renderer-tests CI workflow

| | |
|---|---|
| PR | https://github.com/antmikinka/lemonade/pull/7 (merged) |
| Title used | `ci(gui3): run renderer suites on GUI3 PRs` |
| Branch | `ci/gui3-renderer-tests` → `GUI3_squashed` |
| Issue | none (CI infrastructure change) |
| Merge commit | `5596c6560` |
| CI | self-proving: https://github.com/antmikinka/lemonade/actions/runs/37163250298 (success) |
| Apply at | edit the PR body in the browser: https://github.com/antmikinka/lemonade/pull/7 |

## Template skeleton (tick as shown)

```markdown
## Spec Driven Development

This PR:
- [x] fixes something and does not need a WG/RFC. <!-- CI infra, no issue -->
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

- [ ] Documentation is not affected by this change.
- [x] Documentation is affected and has been updated. <!-- adds docs/dev/fork-divergence.md -->

## Breaking Changes

- [ ] This PR introduces breaking changes.
- [x] This PR does not introduce breaking changes.
```

## Diffstat (paste as-is)

```
 .github/workflows/gui3-renderer-tests.yml | 61 ++++++++++++++++++++++++++
 docs/dev/fork-divergence.md               | 72 +++++++++++++++++++++++++++++++
 2 files changed, 133 insertions(+)
```

## Test commands + observed output (paste as-is)

The workflow ran green on its own branch (it triggers on PRs targeting `GUI3_squashed`):

```
https://github.com/antmikinka/lemonade/actions/runs/37163250298  → success
```

Local dry-run of everything the workflow executes, from the branch:

```
$ npx tsc --noEmit                                   # in src/app — clean
$ node test/app/run-app-regression-tests.cjs         # repo root — all pass
$ npm run test:<each renderer suite>                 # in src/app — all pass
$ npx playwright test tests/a11y.spec.ts             # green
$ npm run build:renderer:prod                        # webpack compiled successfully
```

## Screenshots (paste markdown as-is)

```markdown
![First green gui3-renderer-tests run](https://raw.githubusercontent.com/antmikinka/lemonade/screenshots/ci-run-green.png)
```

## Fact bullets — REWRITE IN YOUR OWN WORDS before posting (ai-content-policy Rule 1)

- Adds `.github/workflows/gui3-renderer-tests.yml`: on PRs to `GUI3_squashed`, runs typecheck, the repo-root app-regression runner, every `test:*` renderer runtime suite in `src/app`, the a11y Playwright spec, and a production webpack build.
- Adds `docs/dev/fork-divergence.md`: records where the fork's GUI3 tree intentionally diverges from upstream (branch layout, test wiring, workflows) so upstream syncs don't clobber fork-specific CI.
- The workflow has guarded every later GUI3 PR (#9, #11, #13, #15) — all merged green.
- Known GitHub limitation (worth a line in the body): `gui3_beta_build.yml` lives only on `GUI3_squashed`, and `workflow_dispatch` 404s for workflows absent from the default branch — it cannot be triggered from the CLI.
