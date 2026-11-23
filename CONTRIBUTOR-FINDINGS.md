# Verified findings and work ownership

This repository records the fixes we tested, their original authors, and what
we observed on our own hardware. The game-free regression tests are reproducible
from source. Retail captures stay on the test machine.

## BC7 controls and decoder scope

The native decoder has an independent, game-free regression corpus of **2,049
records and 2,066 BC7 blocks**. It covers all 285 configurations, field basis
controls, the two published MIT mode-6 controls, and same-mode prediction history.
The [corpus provenance](tests/data/bcpack-synthetic-controls.md) records the exact
format, checksum and source pins. Reproduce with:

```powershell
cargo test --locked --release -p xcl --test bcpack_controls independent_synthetic_corpus -- --nocapture
```

The entropy work overlaps [PR #208](https://github.com/ivall/xcl/pull/208).
Contributions should focus on a demonstrated fragment-trimming regression or the
independently scoped block reconstruction and controls. Another wholesale
entropy-stage implementation would duplicate ongoing work. [PR #273](https://github.com/ivall/xcl/pull/273)
handles BC1 and is not a replacement for BC7 reconstruction.

Initial BC7 preset history is not included in native XCL. The optional
reference helper, maintained in a separate private repository,
uses public fitted history and reports provisional output. A successful reference
comparison is not a console/hardware comparison.

## Memory mirrors: retain the community fix

The implementation comes from [PR #252](https://github.com/ivall/xcl/pull/252)
and [PR #283](https://github.com/ivall/xcl/pull/283), including the later lazy-page
mapping correction. Our useful additional evidence is a forced Windows fallback
test and actual RTX 3060 Ti texture readback across CPU writes and later
map/unmap changes. Keep that validation separate from credit for the feature.

The rebased PR #283 head `9bd7635` has identical production mirror files to
`a2bfbed`. That head change does not require another implementation or runtime
build. Compare exact files rather than the larger merge-base diff.

## Uncovered copy path: investigation, not a completed fix

The measured game trial logged a refused 16 KiB `CopyMemoryToMemoryX` operation
from a CPU-mirrored graphics view to sparse UMA. The mirrored CPU source is also
outside UMA. An audit of 265 published upstream PR copy-function heads, the
reference fork and relevant discussions found no implementation for this path.

Resolving the address alone does not make the memory GPU-readable. A candidate
implementation needs staging plus a native GPU copy at the recorded command
position, with correct predication, barriers, sparse write/readback tracking and
staging lifetime. Immediate CPU copying could change command ordering. The
request's source ownership and its relation to the remaining stall need direct
measurement before proposing a patch.

The existing late-page-array work in [PR #289](https://github.com/ivall/xcl/pull/289)
addresses a different problem and should be reused if its measured conditions
apply. New work should be checked against the private overlap ledger, current
PR diffs and review discussions before implementation.
