# Synthetic BCPack BC7 reconstruction controls

`bcpack-synthetic-controls.bin` is **synthetic test data**, not a game packet,
game asset, capture, memory dump, console SDK file, or executable. It contains
2,049 independent input/expected-output records for the native BC7 reconstruction
layer. Tests require neither a game nor an external reference program.

Size: **115,564 bytes**. SHA-256:
`a3ee6df0e2d9062934a72448ee8730befcbef5f1b2d94f042c6a278dd136190e`.

Run the single corpus regression:

```text
cargo test --locked --release -p xcl --test bcpack_controls independent_synthetic_corpus -- --nocapture
```

The test includes the production reconstruction module by relative path. It
compares every output byte with the fixed expected blocks below; it does not
derive expectations from the native decoder being tested. The module's six
other unit tests are filtered out by this command.

## Binary format, version 1

All integers are unsigned 32-bit little-endian. There is no alignment padding.

1. Record count, exactly `2049`.
2. For each record:
   * `N`: number of symbols in each lane, either `2` or `4`.
   * Sixteen consecutive lanes of exactly `N` raw unsigned 8-bit symbols each.
   * `M`: expected output byte count, exactly `8*N`.
   * Exactly `M` raw bytes of standard BC7 blocks, in original block order.
3. Exact end of file, with no trailing bytes.

There are 2,032 one-block records (`N=2`, `M=16`) and 17 two-block records
(`N=4`, `M=32`), totaling 2,066 BC7 blocks. The reader checks the fixed corpus
size and count, validates each small shape before allocating, checks slice
bounds and offset addition, and requires exact EOF.

## Record groups and provenance

Indices below are zero-based, inclusive. The controls were assembled on
2026-11-23 from independently constructed input lanes and fixed reference
observations. Every record was checked against its original synthetic source
before adding this file to the repository.

| Records | Count | Construction and coverage |
|---|---:|---|
| 0–284 | 285 | One seeded random block for each standard BC7 configuration: all modes, partitions, rotations, and mode-4 index selections. Python random seeds `7319 + record_index`; input endpoint symbols randomized within their lane alphabets and selectors limited to 0/1. |
| 285–2030 | 1,746 | One input bit at a time in each endpoint/selector lane 2–14, each of its two symbol slots, and each legal alphabet bit. Nine representative configurations: modes 0–7 at partition/rotation/index-selection zero, plus mode 4 with index selection one. |
| 2031–2032 | 2 | The public MIT source-built one-block and two-block mode-6 controls identified below. Their expected bytes are the original fixed controls. |
| 2033–2048 | 16 | Two authored two-block controls for each mode 0–7, with and without anchor normalization. The second block uses endpoint distance one to test preservation of the first block's raw predictor state. |

The random and basis controls use the zero predictor. Every record's first
block has zero endpoint and index distances. The 16 history controls refer
only to their own preceding block, at endpoint distance one; their index
distance remains zero. **No control requests a fitted initial history preset.**

Expected ordinary BC7 block bytes for the authored controls were observed
through the public `sixng_core::bc7pack::decode_lanes_states` API in a separate
GPL-3.0-only [6to5m](https://github.com/DesignWithTommy/6to5m) process, pinned to
`8183ea3ca9b9fd26e9379b030efce1684c263c4f`. Only the numerical input/output
observations are included. No GPL decoder implementation, source fragment,
fitted preset, or history table is copied, linked, or distributed here. The
native decoder was implemented independently using standard BC7 field math
and the separately attributed MIT packing/partition facts in its source.
These observations are a provisional secondary reference, not a comparison
with Xbox hardware or an official SDK implementation.

The input lanes for the two MIT controls were independently entropy-decoded
from the following public synthetic streams using the MIT alternative of
bcpack-rs and Xbox2Windows's bounded entropy handling:

* [One-block `tests/bcpack_mode6_vector.py`](https://github.com/R0Y00/Xbox2Windows/blob/64ecc8738ea447b3d9a1cb7f9fc891bbb945ee3b/tests/bcpack_mode6_vector.py),
  revision `64ecc8738ea447b3d9a1cb7f9fc891bbb945ee3b`. The source credits
  timelery in Xbox2Windows issue #442.
* [Two-block `tests/bcpack_mode6_multiblock_vector.py`](https://github.com/stan495/Xbox2Windows/blob/5e1af1695674866a784a80720426a0a5384263b1/tests/bcpack_mode6_multiblock_vector.py),
  revision `5e1af1695674866a784a80720426a0a5384263b1`.

Newly authored controls and the reader follow the repository's GPL-2.0-only
license. The two adapted public controls retain the following MIT notice.

## MIT notice for the public synthetic controls

MIT License

Copyright (c) 2026 Xbox2Windows contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
