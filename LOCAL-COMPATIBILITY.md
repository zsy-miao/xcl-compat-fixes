# Local compatibility work

This repository preserves the upstream XCL history and GPL-2.0-only license.
Community changes retain their source commit references in `Update-XCL.ps1`,
the source headers, and the packaged `BUILD-INFO.json`. Local additions include
bounded CPU instruction handling, memory validation and regression checks, and
experimental BCPack BC7 support. The alternate community reference is
[theaustinayers/xcl](https://github.com/theaustinayers/xcl).

## BCPack BC7

The native decoder uses MIT entropy tables and supports all eight ordinary BC7
block modes and 285 configurations. It reconstructs verified zero and same-mode
history predictions. Unknown initial presets and unverified prediction changes
return an error before destination writes. Both small published compressed
controls and 2,049 synthetic block controls were checked independently.

The game also needs initial history presets. `XCL_BCPACK_EXTERNAL_DECODER` can
select a separately built research executable that consumes validated symbol
lanes and returns BC7 bytes. It uses fitted history, so successful output is
provisional and has not been compared with a console decoder. XCL rejects
missing entries, invalid blocks, normalization, wrong output sizes, excessive
requests and decoder timeouts. Its process job includes descendant processes,
and the timeout covers all pipe I/O.

`Play-XCL-BCPack-Experimental.cmd` starts the locally built game with this helper.
The ordinary portable archive contains the native decoder; the separate GPL-3.0
helper and its history are not linked into or bundled with XCL. Local capture
files stay under ignored `target` directories and are never published.

## Video-memory corrections

PR #252 watches GPU-visible allocations that fall outside the GPU windows.
PR #283 selects the CPU mirror of an empty GPU view and rechecks that selection
when physical pages are mapped or unmapped. The local regression forces the
Windows fallback, checks real write-watch results, and verifies GPU texture
readback across CPU rewrites and later map/unmap operations.

## Publishing and checks

Ongoing source commits are pushed to `work-in-progress`. `main` advances after
the required checks pass. Game captures, dumps, game assets, binaries and
credentials are excluded. The separate helper retains its own license and
source provenance. A passing synthetic check is not a claim of working gameplay.

## Measured game trial, 2026-11-23

The offline trial of program revision `a767d51c8b5daddc661e63062b0a3d76d9c71598`
ran for 332.81 seconds, ended at the requested timer with exit code 0, and recorded
no crash. It showed 479 frames with no reported pipeline/compiler failures. The
frontend remains partial; navigation and gameplay are not confirmed.

Twenty unique BC7 packets were captured locally: eleven requested 65,536 output
bytes, and nine requested 262,144. All twenty passed independent native/reference
entropy comparison, complete input consumption, and strict external reconstruction
with zero missing, invalid or normalized blocks. Fitted output remains provisional.
The capture data and decoded game bytes are excluded from the repositories.

A later `CopyMemoryToMemoryX` request was refused because its CPU-mirrored source
lies outside UMA. Its role in the remaining stall is unproven. Supporting that
path requires correctly ordered GPU staging; it should not be replaced by an
immediate CPU copy. See [contributor findings](CONTRIBUTOR-FINDINGS.md).

## Account selection and intro timing

Two offline runs on an i5-10400 and RTX 3060 Ti used the same five-minute limit,
frame capture interval, and Y and A inputs at 155 and 180 seconds. Both exited
at the timer with no reported crash or pipeline/compiler failures.

| Build | Frames in 300 seconds | Frames from 200 to 280 seconds |
| --- | ---: | ---: |
| `4734b40` | 455 | 4 |
| `cb8e15f` | 834 | 173 |

The later run displayed the six-star transition and the full-color intro after
account selection. It combined ivall's struct-return fix, gw1337x's texture
detiling optimization and our bounded CLWB loop handling. The first accelerated
flush covered 129,600 iterations. The runtime comparison measures the combined
build. A BC5 storage refusal and a 16 KiB graphics-memory copy refusal still
appeared during the intro.

## Optional BC5 decoding

The experimental launcher can use the separate reference process for single-stage
BC5 texture requests. XCL checks the helper's reported input consumption, output
length, lane counts, and reconstruction status before copying the result to the
requested destination. This mode uses fitted history and has no console decoder
comparison. A run without a helper keeps the normal decompression error.

Thirteen distinct 64 KiB requests captured during the intro matched their source
file ranges and consumed their exact input in the independent entropy check. The
separate reference helper produced matching outputs in two local runs. The suite
includes a game-free BC5 control, a virtual memory destination and a file-backed
request spanning two physical pages. Set `XCL_BCPACK_EXTERNAL_DECODER` to the
absolute helper path to include those two process tests in the packaged build.

The extended run of `cb8e15f` reached readable UI Text Size and Legal Notices
screens on this PC. It displayed 1,132 frames and exited at its 430-second timer
without a reported crash. The subsequent build `b696652` passed 125 release checks,
including both optional BC5 helper tests. Its first 285-second game run exited
normally after 548 frames. The log for that run did not record a BC5 request.
