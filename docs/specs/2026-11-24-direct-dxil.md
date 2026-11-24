# Direct DXIL backend — design and roadmap (2026-11-24)

Branch `shaders/direct-dxil`, code in `crates/xcl/src/gpu/dxil/`.

## Goal

Translate console RDNA2 programs straight to signed DXIL containers, with no
HLSL text and no DXC at run time, while keeping the HLSL → DXC path as a
per-shader fallback until the direct path matches its 100 % corpus coverage.

Why: compile latency on cache misses (DXC takes from hundreds of ms to
seconds, and gigabytes, on large translated programs), no dependency on the
Windows SDK's `dxcompiler.dll`/`dxil.dll`, exact control over float
semantics (no implicit fast-math), and much smaller intermediate output.

## Provenance and licensing

XCL is GPL-2.0-only. References used:

- LLVM 3.7 bitcode format and DXC (`DxilOperations`, `DXIL.rst`, the
  container hash) — University of Illinois/NCSA, GPL-compatible.
- DXC's *output* for small HLSL test programs (`tests/data/dxil/`), dumped with a
  local bitstream reader, as the layout oracle.
- Mesa `src/microsoft/compiler` (MIT) may be ported with a provenance note.

**Not used:** the VIctory project (GPL-3.0-or-later, incompatible). Its
public architecture description informed the approach below; no code was
read for porting or copied.

## Layers (done)

| Module | Role |
|---|---|
| `bitstream` | LLVM bitstream writer: fixed/VBR fields, blocks with back-patched lengths, unabbreviated records |
| `module` | LLVM 3.7 module: interned types, module-level constants, function declarations, one defined entry, instructions (binop, cast, cmp, select, extractvalue, call, phi, load/store/GEP, atomicrmw/cmpxchg, br/switch/ret), metadata; bitcode writer with relative operand numbering |
| `shader` | DXIL layer: `dx.op` calls, `createHandleFromBinding`/`FromHeap` + `annotateHandle`, raw buffer load/store, wave ops, system values, entry metadata, shader flags ↔ `SFI0` bits |
| `container` | DXBC container: `SFI0`, `ISG1`/`OSG1`, `PSV0` (RuntimeInfo3), `DXIL` program header, retail digest (verified against DXC byte for byte) |
| `validate` | DXC 1.9's internal validator and disassembler as test oracles |

Verified: the emitted sample kernel (branch + phi, raw UAV load/store, wave
read-lane, WaveSize) passes the validator, disassembles as expected, and runs
on the GPU with the recompiler's kernel ABI, matching a CPU reference.

## Translator architecture (to build)

The HLSL backend emulates the wave: a uniform `switch (pc)` dispatcher over
register arrays, EXEC as an explicit mask. That form is exact but leans on
DXC's optimiser. Emitting it unoptimised as DXIL would leave register arrays
in scratch memory. The direct backend therefore uses a per-lane model:

1. **SSA construction** (`dxil/ssa.rs`). Build a CFG from the decoded program
   (reuse `rdna2` decode and `recomp` analysis). Each register dword is a
   variable, with SSA built by the Braun et al. on-the-fly algorithm (no
   dominance frontiers needed). SCC, VCC, M0 and EXEC are explicit operands.
   A VGPR write also reads the old value, because disabled lanes keep it.
2. **Lane-mask lowering** (`dxil/lanes.rs`). Lane masks become per-lane
   booleans (this lane's bit of EXEC/VCC/compare results), represented as a
   small hash-consed boolean DAG. Branches on wave-uniform tests of masks
   (`s_cbranch_execz`, `vccz`, SCC of `s_and_saveexec`) become per-lane
   branches. A path-sensitive dataflow proves where this lane's EXEC bit is
   set; vector instructions not proven keep a per-lane guard. Values that
   really are wave-wide (`s_bcnt1` of a mask, `v_readlane`, ballots) are
   rebuilt with `WaveActiveBallot`/`WaveReadLaneAt`/`WaveActiveCountBits`.
3. **Whole-wave regions.** Code between `s_or_saveexec -1` and the restore
   (DPP/permlane reductions and prefix sums the console compiler expands) is
   evaluated symbolically over the console's lanes. The result is
   classified as a reduction or prefix of a known associative operation and
   emitted as the DXIL wave op. This makes wave64 reduction code correct on
   wave32 hosts. Anything unrecognised falls back to the HLSL path.
4. **Control flow.** DXIL needs a reducible CFG. Console code is reducible in
   practice; irreducible input falls back. Blocks are emitted in reverse
   post-order so non-phi operands never forward-reference.
5. **Lowering** (`dxil/lower.rs`). One machine instruction → DXIL
   instructions, sharing the semantic tables the HLSL emitter already
   encodes (float compare classes, DPP/SDWA selects, buffer formats,
   denorm/rounding modes). Exactness rules: no fast-math flags; explicit
   NaN/denorm handling where RDNA and DXIL differ (`V_FRACT_F32(+inf)`,
   min/max NaN rules, `v_rcp` precision).
6. **Resources.** Descriptor addresses are traced to their root/user-data
   sources (reuse the recompiler's T#/V#/S# analysis) and mapped to SM 6.6
   heap indexing or root descriptors, matching `pso.rs`'s ABI.
7. **Light optimisation passes** on the SSA (constant folding, dead code,
   copy propagation, guard merging) before emission. Drivers do the heavy
   optimisation.

## Testing strategy

- Validator + disassembly on every emitted shader in tests.
- **Differential GPU testing:** the existing `recomp/tests.rs` instruction
  tests (≈6k lines, `runner::run`) take DXIL bytes. Every test runs the same
  RDNA program through both backends, which must give identical final
  registers and memory.
- Corpus sweep: translate the bounded validation corpus (9,282 programs)
  and record per-program direct/fallback status and validator results, on
  small samples per the project's time constraints.
- Live: route compute first, behind `XCL_DIRECT_DXIL=1`, with the HLSL path as
  fallback on any unsupported construct.

## Roadmap

| Phase | Content | Exit criterion |
|---|---|---|
| 0 ✅ | Bitstream, module, DXIL layer, container + hash, validator oracle, GPU smoke test | Emitted kernel validates and runs |
| 1 ✅ | SSA + per-lane model for straight-line and branchy **compute kernels in the test ABI**; SALU/VALU integer + float ALU | Differential tests pass for ALU instruction tests |
| 2 ✅ | Memory: `buffer_*`/`tbuffer_*`/`s_load`/`s_buffer_load`/flat/global, LDS (`ds_*` → groupshared), atomics | Memory instruction tests pass |
| 3 ✅ | Cross-lane: DPP16/DPP8, `v_readlane`/`v_writelane`, permlane, whole-wave regions | Cross-lane tests pass, wave64 on wave32 |
| 4 (partial) | Real compute shaders: resource tracing → heap/root bindings; `XCL_DIRECT_DXIL=1` live routing with fallback | Corpus CS coverage measured, live title runs |
| 5 (partial) | Graphics: VS/PS signatures (`ISG1`/`OSG1`/PSV elements), interpolation, exports, images/samplers | VS/PS tests and live draws |
| 6 (partial) | Mesh/two-pass pipelines, mesh output metadata | Mesh tests |
| 7 | Default-on for covered shaders, cache keying on backend version | Parity with the HLSL path |

Status (main-based port, 2026-11-24): phases 1–3 done; 4–6 partial — live routing
behind `XCL_DIRECT_DXIL=1` (compute + mesh compute pass) / `=all` (VS/PS) with a
logged HLSL fallback is in, corpus coverage and live-title runs are not measured
yet. Model: a faithful EXEC/ballot model (EXEC and VCC as wave masks, lanes
predicated per instruction over a structured block CFG); per-lane
structurization (real divergent control flow instead of predication) is future
work. Coverage: `xcl shaders <pack> --direct`.
