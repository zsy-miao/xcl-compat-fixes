<!-- What changes and why, in a few sentences. "Fixes #N" / "Part of #N". Stacked: "Stacked on #N". -->

**What is in it**
-

**Tested**
- hardware (GPU + driver, CPU), Rust version, debug / release:
- `cargo test --workspace -- --test-threads=4`: N passed, N failed, N ignored (SKIPPED GPU tests: …)

**Regression run**
<!-- Required when the PR changes crates/xcl/src outside tests: `scripts/smoke.ps1` on a real title,
     built from main and from this PR's head commit (no uncommitted changes); paste both blocks.
     Rerun the PR's after every push. Not needed for docs / tests / scripts only. -->

main:

PR:

**Not tested**
-

**AI use**
- none / tool and model, and what it did
