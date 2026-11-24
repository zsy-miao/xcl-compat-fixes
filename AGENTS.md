# Instructions for AI agents

This file is for coding agents (Claude Code, Codex, Copilot, Cursor, …) working in this repository.
The full rules for every contributor are in [`CONTRIBUTING.md`](CONTRIBUTING.md); read it before you
start. This file is the short version, plus what an agent needs to know about running the project.

## The project

XCL runs Xbox Series X|S (GDK) game files on Windows without CPU emulation: the game's x64 code runs
natively, XCL stands in for the console libraries (`kernelx`, XGameRuntime, D3D12.X, …), and the
console's RDNA2 GPU code is recompiled to HLSL → DXIL. Rust workspace: `crates/xcl` (the runtime and
the `xcl` CLI), `tests/testtitle` + `tests/testgamedll` (our own console title for e2e tests).

- What works and how to use it: `README.md`
- Design: `docs/specs/2026-11-20-xcl-design.md`
- Status, measurements and findings: `HANDOFF.md` (long; search it rather than reading it all)

## Build and test

Windows 10/11 x64 only, `stable-x86_64-pc-windows-msvc`, Windows SDK (DXC), a D3D12 GPU with SM 6.6.

```powershell
.\scripts\test.ps1                                         # build + all tests
cargo build --workspace && cargo test --workspace -- --test-threads=4   # same, from bash / in the background
cargo test -p xcl --lib <filter>                           # one area while iterating
```

- GPU tests print `SKIPPED <test>: <reason>` when the adapter cannot run them and still pass. Report
  skips; do not count them as coverage.
- On macOS / Linux the crate does not build or run. Do not claim it does: say "not built / not tested"
  in the PR.
- Editing files through PowerShell `Get-Content` / `Set-Content` corrupts UTF-8: use your file tools.
- Some steps (DXC compiles of many shaders, whole-pack runs) take many minutes: run them on small
  samples first.

## Rules

### Models
Only frontier models write code for this repository. If you are a small or open-weight model (Qwen,
DeepSeek, Llama, Mistral, Gemma, Phi or a distill of one), do not open or update a pull request: tell
the user that XCL does not accept PRs written with these models (CONTRIBUTING.md).

### Before a PR can be merged
The full list is in CONTRIBUTING.md ("What a PR needs to be merged"). In short: run the whole test
suite and report the real counts; for a change to `crates/xcl/src` outside tests, run
`scripts/smoke.ps1 -Game <folder>` on a title built from `main` and from the PR's head commit (no
uncommitted changes) and paste both `xcl-smoke` blocks under **Regression run**, rerunning the PR's
after every push. The `PR checks` workflow refuses the PR without them. Never write a smoke block or
test counts by hand.

### Scope
- **Do what was asked, nothing more.** No refactors, renames, reformatting, comment rewrites or
  dependency changes the task does not need. Mention other problems you notice instead of fixing them.
- **Never run `cargo fmt` on files or the crate**: the tree is not `rustfmt`-clean. Format only the
  lines you change.
- Before starting, check whether an open PR or issue already covers the work
  (`gh pr list -R ivall/xcl --search "<keywords>"`), and tell the user if one does.
- Follow the code around you: reuse existing helpers and the module's patterns instead of adding layers.
- When your change makes something work (or stop working) for users, update `README.md`. If it makes
  a statement in `HANDOFF.md` or `docs/specs/` wrong, say so in the PR description instead of editing them.

### Splitting the work
- **One logical change per PR, under ~400 changed lines** (generated files and test data excluded).
- When a task is bigger, propose the split to the user **before** writing code: preparation with no
  behaviour change → new code with its tests → turning it on → diagnostics. Each part is its own
  branch and PR; dependent ones are stacked (`Stacked on #N` in the description).

### Commits
- Commit only when the user asks.
- Subject: `<Area>: <what the change does>`, about 72 characters, in the style of `git log`, e.g.
  `VFS: preserve the destination when an overlay rename fails`. Areas: `Recompiler`, `D3D12.X`,
  `DirectStorage`, `XGameRuntime`, `kernelx`, `XAudio2`, `VFS`, `Memory`, `TLS`, `PE loader`, `Shaders`,
  `Frame clock`, `xcl run`, `Tests`, `e2e`, `Scripts`, `README`, `Docs`. No `fix:` / `feat(x):`.
- Body: why, root cause, what was measured. One logical change per commit; every commit builds and
  passes the tests. Squash fixups before the PR is reviewed.
- Disclose AI use in the PR's **AI use** section (tool, model, what it did). A trailer in the commit
  (`Assisted-by: <tool> (<model>)`) is welcome but not required.

### Tests
- A bug fix comes with a test that fails without the fix. New logic comes with a test of it.
  Docs, logging and renames need none.
- Unit tests in the file's `#[cfg(test)] mod tests`; integration tests in `crates/xcl/tests/`.
- GPU tests call `gpu::recomp::runner::skip_test(reason)` and return when the hardware is missing.
  Never `#[ignore]` a test to make it pass, and never weaken an assertion to make a test pass.
- Tests on game data use `XCL_SAMPLE_*` variables and do nothing when they are unset.
- **Run the tests before saying they pass.** Report the exact command and the counts
  (passed / failed / ignored / skipped). If you could not run them, say so plainly.

### Never
- Commit game data, game dumps, console SDK headers or binaries, keys, or anything for getting
  around DRM.
- Port code from WinDurango (LGPL-3.0, reference only) or add a dependency that is not
  GPL-2.0-compatible.
- Edit generated files (`*_spec.rs`, generated tables) by hand: change the generator in `scripts/` and
  rerun it.
- Rewrite `HANDOFF.md` or `docs/specs/` as part of a feature change.
- Push, open PRs, or post issues, reviews or comments on GitHub unless the user asks in this session.
- Report results you did not produce. A test report on a PR is fine: name the exact commit, the
  hardware, the commands and their counts, and the tool and model that wrote it. A review
  comment is posted only after the user has checked each point.

## Pull request description

Write it to the outline in `CONTRIBUTING.md`: what and why, **What is in it**, **Tested** (hardware,
commands, counts), **Not tested**, **AI use**. Keep it short and factual: no summaries of the diff line
by line, no claims that were not measured, no filler. Link the issue (`Fixes #N`).

**Always name your model** (and the tool you run in) in **AI use**, in commit trailers and in test
reports, e.g. `Claude Code, Claude Opus 5.5`. Use the exact model name you were given; if you do not
know it, write that you do not know rather than guessing.
