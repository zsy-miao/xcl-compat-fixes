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
