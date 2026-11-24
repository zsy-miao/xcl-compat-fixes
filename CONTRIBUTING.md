# Contributing to XCL

Thanks for helping. XCL gets far more pull requests than one maintainer can read carefully, so
these rules exist to make every PR **cheap to review and safe to merge**. A PR that ignores them
may be closed without a review, however good the code is.

AI tools are welcome (see [AI-assisted contributions](#ai-assisted-contributions)). The rules are
the same for code written by hand and by an agent.

## What a PR needs to be merged

XCL gets more pull requests in an hour than anyone can test by hand, most of them written with AI
tools. So a PR is merged only when it shows that it was tested and that it breaks nothing. These are
checked before anything else; a PR that does not meet them waits (or is closed after a week).

1. **Both required checks are green:** `CI` (build and tests on Windows, no GPU) and `PR checks`
   (the description: the sections below, and the regression runs when they are needed).
2. **The whole suite ran on a GPU**, with the counts in **Tested**
   (`cargo test --workspace -- --test-threads=4`: N passed, N failed, N ignored, and the SKIPPED tests).
3. **A PR that changes the runtime** (`crates/xcl/src` outside tests) **has a regression run on a real
   title**: `scripts/smoke.ps1 -Game <your game folder>` built from `main` and built from the PR's head
   commit, both `xcl-smoke` blocks pasted under **Regression run**, of the same title for the same
   time. The PR's block must name its head commit with no uncommitted changes (rerun it after every
   push); the other must be made on a commit of `main`. The PR's run must show no device removal and
   no crash (the `main` run may: a PR that fixes a crash shows it there), at least 80 % of `main`'s
   frames and at most 125 % of its peak commit. A run on a title that does not reach the code the PR
   changes still counts as a regression check; say what it does not cover under **Not tested**.
   `scripts/pr_check.py` checks all of this (and the rest of this list it can see) on every PR.
   If a test-only diff touches a runtime file, a maintainer can apply the `test-only` label after
   reviewing the diff. The label event waives only the smoke check for that revision. Adding or
   removing other labels preserves the description check result. After a push or another event
   that reruns the description check, review the updated PR, then remove and reapply the label
   if the diff is still test-only.
4. **No regression on the maintainer's run.** The maintainer merges in batches: the whole suite and a
   retail title run again on the merged tree, on an AMD RDNA4 GPU. A PR that breaks either is taken
   out of the batch and gets a comment with what failed.
5. **No game data.** No file of a game package (archives, executables, video, audio banks, textures,
   `MicrosoftGame.config`, dumps) and no binary outside `tests/data/` or over 1 MB; `.gitignore` keeps
   the usual ones out and the check refuses the rest.
6. **Up to date and reviewable:** it merges into `main` without conflicts, and keeps to the size and
   scope rules below. A PR that duplicates one already merged is closed.

A test result that was not produced by a real run on the hardware named gets the PR closed.

## Before you start

1. **Search open PRs and issues first.** Several PRs have solved the same problem at once and all
   but one were closed. If someone is already on it, review or test their PR instead.
2. **Say what you are doing.** For anything bigger than a small fix, comment on the issue (or open
   one) with the plan in two or three lines before writing it. That avoids duplicate work and wasted
   effort on a direction the maintainer would not take.
3. **One problem per PR.** If you find a second bug while working, it gets its own PR.

## Pull requests: small and single-purpose

The smaller the PR, the faster it gets reviewed and merged.

- **One logical change per PR.** A fix, a feature, a refactor and a test-only change are four PRs,
  not one. "While I was here" changes go into a separate PR.
- **Size target: under ~400 changed lines**, not counting generated files and test data. Over ~800
  lines, the description must say why it cannot be split. Most large PRs can be split.
- **How to split** a large piece of work, in roughly this order:
  1. preparation that changes no behaviour (moving code, a new helper, a renamed type),
  2. new code behind a check or not yet called, with its tests,
  3. turning it on,
  4. diagnostics, logging and env-var switches, each on its own.
- **Stacked PRs** are fine when one change depends on another. Open each PR against `main`, say
  `Stacked on #N` at the top of the description, and make it clear which commits belong to this PR.
  Each PR in the stack must make sense when reviewed on its own.
- **No drive-by reformatting.** The tree is not `rustfmt`-clean. Do not run `cargo fmt` on whole
  files or the whole crate: format only the lines you change. Reformatting buries the real diff.
- **No unrelated churn:** no renames, reordering, comment rewrites or dependency bumps the change
  does not need.
- Keep the PR up to date by rebasing onto `main`, not by merging `main` into it.
- Open the PR as a **draft** until it builds, its tests pass and you have checked it yourself.

### PR description

Use this outline:

```markdown
<What changes and why, in a few sentences. Link the issue: "Fixes #N" / "Part of #N".>
<If it is stacked: "Stacked on #N; only the top commit belongs here.">

**What is in it**
- one bullet per commit or part, with the file it lives in

**Tested**
- hardware (GPU + driver, CPU), Windows build, Rust version, debug/release
- commands run and their results, e.g. `cargo test --workspace -- --test-threads=4`: 541 passed, 0 failed, 21 ignored
- `SKIPPED` GPU tests on your adapter, if any
- game checked (if any), and what got better, with numbers if you have them

**Regression run**
- for a runtime change: the two `xcl-smoke` blocks (main, this PR's head) – see above

**Not tested**
- what you could not check, e.g. AMD / NVIDIA, the retail title, wave32-only GPUs

**AI use**
- none / which tool and model (e.g. Claude Code with Claude Opus 5.5), and for what (see below)
```

"Not tested" is required. An honest "not tested on AMD" is fine. A test you did not run listed as
passing is not, and gets the PR closed.

## Commits

History is merged with merge commits, so every commit on a PR ends up in `main`.

- **Scope prefix + outcome**, as in the existing history:
  ```
  Recompiler: round f16 conversions and products to nearest-even by default
  VFS: preserve the destination when an overlay rename fails
  D3D12.X: draws with untranslated pipelines get the title's root signature
  ```
  The prefix is the subsystem: `Recompiler`, `D3D12.X`, `DirectStorage`, `XGameRuntime`,
  `kernelx`, `XAudio2`, `VFS`, `Memory`, `TLS`, `PE loader`, `Shaders`, `Frame clock`, `xcl run`,
  `Tests`, `e2e`, `Scripts`, `README`, `Docs`, `Issue templates`. Use one that exists already when it fits.
  No `fix:` / `feat(...)` style prefixes; this repo does not use Conventional Commits.
- Subject line of about 72 characters or less, saying what the change does, not "update file.rs" or
  "fix bug".
- The body (optional for small changes) says **why**, and anything a reviewer could not see from the
  diff: the root cause, what was measured, what was ruled out.
- **One logical change per commit.** Each commit builds and passes the tests on its own, so `git bisect`
  works.
- **No fixup commits in the final PR**: squash "fix review", "wip", "fmt", "oops" commits into the
  commit they fix before asking for review (`git rebase -i`, then force-push).
- Reference issues as `(#123)` in the subject or `Fixes #123` in the body.

## Tests

Add or change a test when the change touches logic: a bug fix gets a test that fails before the fix
and passes after it, and new behaviour gets a test that shows it working. Docs, comments, logging
and pure renames need no test.

- **Unit tests** go in a `#[cfg(test)] mod tests` in the same file, the way the rest of the crate does it.
- **Integration / end-to-end tests** go in `crates/xcl/tests/`. The e2e test drives our own console
  title in `tests/testtitle` / `tests/testgamedll`: extend it when a change concerns the path a title
  takes.
- **GPU tests** must run without a GPU too: when the adapter cannot run them, call
  `gpu::recomp::runner::skip_test("no GPU" | "no DXC" | "no wave64" | "no mesh shaders")` and return.
  Never mark a test `#[ignore]` only so that it passes.
- **Tests on game data** read files the contributor owns through the `XCL_SAMPLE_*` variables
  (see README / `scripts/test.ps1`), and pass without doing anything when the variable is not set.
  **Never commit game data**, console SDK headers or binaries, or output dumped from a game.
- **Recompiler instructions** get a per-instruction GPU test compared against an expected result,
  like the existing ones.
- Run the whole suite before opening the PR:
  ```powershell
  .\scripts\test.ps1        # cargo build --workspace + cargo test --workspace
  ```
  XCL builds and runs on Windows only. If you could not run the tests (e.g. you work on macOS or
  Linux), say so in **Not tested**; a PR with no Windows test run waits until someone runs one.

## Licence and provenance

XCL is **GPL-2.0-only**.

- Dependencies must be GPL-2.0-compatible (no Apache-2.0-only, no GPL-3.0). New dependencies need a
  reason in the PR description; avoid them for a few lines of code.
- Code ported or adapted from another project needs a provenance note in the file (project, URL,
  licence) and an entry in `THIRD_PARTY.md` when it is new.
- WinDurango (LGPL-3.0) is a reference only: no code is ported from it.
- Nothing from the console SDK: no headers, no binaries, no copied documentation.
- Generated files (`*_spec.rs`, tables) come with the script in `scripts/` that generates them, and
  the PR says how they were generated. Do not edit them by hand.
- Nothing for decrypting or getting around DRM.

## Docs

- `README.md` tells users what works. Update it when your change makes something work (or stop
  working) for them.
- `HANDOFF.md` and `docs/specs/` are the maintainer's design and status notes. Do not rewrite them in
  a feature PR; if your change makes a statement in them wrong, point that out in the PR description.
- Everything in the repository (code, comments, commit messages, docs) is in English.

## Reviews and test reports

- **Test results on your GPU** go to [#42](https://github.com/ivall/xcl/issues/42), not a new issue.
- Testing someone else's PR helps the most when it follows the **Tested / Not tested** outline above
  and names the exact commit tested. Post one comment and edit it when you rerun the tests; do not
  post a new comment for every run.
- **AI-written test reports are welcome** when they come from a real run: the exact commit, the
  hardware, the commands and their results. Say which tool and model wrote the report.
- **AI reviews:** check each point yourself, post only what you confirmed, and say that a tool helped.
- Reply to review comments in the thread, push the fix (squashed into the right commit), and mark the
  thread resolved only once it is addressed.

## AI-assisted contributions

You can use AI tools. In return:

- **Use a frontier model.** XCL is low-level code (a recompiler, a console runtime, D3D12 memory
  and synchronisation): small and local models get it subtly wrong, and every wrong PR costs a
  review. PRs written with small or open-weight models – Qwen (27B / 32B class and the like),
  DeepSeek, Llama, Mistral, Gemma, Phi and their distills – are closed without a review. Name the
  exact model in **AI use**; the `PR checks` workflow refuses these names there.

- **You are the author and you are responsible.** You have read every line of the diff, you
  understand it, and you can answer questions about it without asking the tool again. "The AI wrote it"
  is not an answer in a review.
- **Disclose it** in the PR's **AI use** section: the tool, the model, and what it did (wrote the
  code, wrote the tests, wrote the description, only reviewed). A trailer in the commit naming the
  model, such as `Co-Authored-By: <model>` or `Assisted-by: <tool> (<model>)`, is welcome too.
- **The tests were run, not just written.** Results in the PR come from a run you did, on hardware
  you name.
- **A person opens the PR.** No PRs or issues opened by an unattended agent.
- **Keep the description short and true.** Leave out generated filler, marketing tone and claims
  nobody measured. A reviewer should be able to read the description in one minute.
- The same size and scope rules apply. An agent can write 2000 lines in an hour, and nobody can
  review that in an hour. Split it.

Agents working in this repository read [`AGENTS.md`](AGENTS.md) (and Claude Code also reads
[`CLAUDE.md`](CLAUDE.md)), which put these rules into instructions for them.
