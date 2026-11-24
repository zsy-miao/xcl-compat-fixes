# Wave64 console programs on 32-lane GPUs

Status: implemented for compute / kernel mode, the mesh (NGG) first pass and ES+GS
(`gpu/recomp/emit.rs`, `PRELUDE_X64`); wave64 VS / PS, which cannot be paired, get the values
of the lanes ≥ 32 they read instead (`upper_lanes.rs`, "Vertex and pixel shaders" below).
The native path is untouched: the form below is only
generated when the adapter cannot run 64-lane waves (`D3D12_FEATURE_DATA_D3D12_OPTIONS1`
`WaveLaneCountMax < 64`, e.g. NVIDIA, whose D3D12 range is 32..32). On such an adapter a
`[WaveSize(64)]` compute shader is rejected at PSO creation, so until now wave64 compute
programs were refused there, and mesh / ES+GS programs ran as two independent halves only
when they had no cross-half operation (`geometry_wave32_hazards`) and were refused otherwise.

## Execution model it relies on

The recompiler runs every console lane through every instruction; EXEC only masks writes.
Branches are scalar (SCC, VCC/EXEC == 0, SGPR values), so control flow is uniform per
console wave as long as every scalar value is identical in all of the wave's threads.
Scalar values come from scalar code (identical when its inputs are), from scalar memory, and
from a handful of vector → scalar instructions: masks (`v_cmp*`, carries – all through
`ballot()`), `v_readlane` / `v_readfirstlane` (through `rla()`), and nothing else. Cross-lane
vector reads (DPP16, DPP8, `ds_swizzle`, `v_permlane(x)16`, `ds_bpermute`) also go through
`rla()`.

## The form

One console wave64 = two hardware wave32s of the same thread group ("halves"): the console
group keeps its `numthreads` and the shader gets `[WaveSize(32)]`. Which hardware threads
play which console thread is decided at the start (`xcl_x_thread`): the threads of a group
are interchangeable, but the console forms its waves from consecutive thread indices and
the PC GPU need not – on the RTX 5070 Ti, 1-D groups get consecutive `SV_GroupIndex` values
per wave, an 8 × 8 × 4 group does not. Each hardware wave takes the next 32 console threads
from a groupshared counter as it arrives (a partial wave the last ones), and the program
gets that console thread's ids (v0..v2 in compute, the vertex slot in mesh mode). Console
lane = console thread % 64, half = bit 5, console wave = console thread / 64.

Per-lane code is unchanged. The two halves meet ("exchange") exactly where a value has to be
whole-wave:

| operation | exchange |
|---|---|
| `ballot()` (every mask: `v_cmp*`, `v_cmpx*`, carries, EXEC at entry) | one word: each half gives its 32-bit ballot, both get the 64-bit mask – unless the mask is only read lane by lane (below) |
| `v_readlane`, `v_readfirstlane` (lane index uniform) | one word: each half gives the value of lane `l & 31`, the half that owns `l` wins |
| `ds_bpermute` (any lane) | 32 words: each half gives its lanes |
| DPP16, DPP8, `ds_swizzle`, `v_permlane(x)16` | none: the ISA keeps them inside a 32-lane half (`WaveReadLaneAt(v, l & 31)`) |
| `s_load*` / `s_buffer_load*` with GLC (or DLC) | up to 16 words: half 0's values for both (memory another wave writes may change between the two halves' loads; without GLC the console's scalar cache is not coherent either, so plain loads are read by both halves) |

So both halves hold the full 64-bit EXEC / VCC / masks and identical SGPRs; every branch is
taken the same way by both, and the existing scalar code (`s_bcnt1_i32_b64`, `s_ff1`,
`s_cbranch_execz`, `v_mbcnt_hi`, …) needs no change. The kernel-mode runner stores the
SGPRs of both halves and the tests fail if they ever differ.

### Synchronisation

* **One console wave per group** (≤ 64 threads: kernel mode, most compute, ES+GS):
  `GroupMemoryBarrierWithGroupSync` between writing and reading the slot. All hardware
  waves of the group are the two halves of the same console wave, which execute the same
  sequence of exchanges, so every barrier is reached by both. Portable (no forward-progress
  assumption) and cheap.
* **Several console waves per group** (mesh: 128 threads, compute > 64 threads): a group
  barrier would also wait for the *other* console waves, whose control flow may differ (wave
  index, per-wave data). Each pair instead meets through groupshared arrival counters:
  every lane stores its slot word(s) and fences (`GroupMemoryBarrier`); the wave's lane
  count (`WaveActiveCountBits` – a wave operation, so all its lanes have fenced) goes into
  its half's counter in one atomic; one lane waits until the partner's counter reaches
  `lanes × (k + 1)` (k = exchange number, identical in both halves) and the others take its
  verdict (`WaveReadLaneFirst`); a fence, then the read. Slots are double-buffered by the
  parity of k: the partner can be at most one exchange ahead, so it never overwrites the
  slot being read. The wait is bounded (2^20 polls): if it ever runs out (a desynchronised
  pair – should not happen, see hazards), the wave stops exchanging (`xcl_x_dead`) and
  finishes with half-local values instead of hanging the GPU until a TDR. This relies on
  co-resident waves of one group making progress, which holds on GPUs (all waves of a group
  are resident on one SM / CU) but not necessarily on WARP. `XCL_WAVE64_SPIN=1` forces this
  variant everywhere (the test suite passes with it too).

A console wave whose upper half does not exist (group of 32 or 96 threads) exchanges with
nobody: the missing lanes count as inactive (ballot bits 0), exactly as on the console. The
GS phase of ES+GS runs in the lower half only (≤ 21 primitives), likewise without a partner.

### Memory

The exchange needs groupshared space: per console wave 2 counters + 2 halves × 2 parities ×
slot (1 word, 16 with wide GLC scalar loads, 32 with `ds_bpermute`). Without console LDS it
has arrays of its own; with LDS (`xcl_lds`, already the full 32 KB D3D12 limit) it lives in
the top of the LDS block (as PR #33 did), and the program's LDS accesses are clamped below
it. A program that really uses the whole 32 KB of LDS therefore loses its top dwords on a
32-lane GPU (the console allows 64 KB; XCL already wraps at 32 KB).

## Hazards (and why they are handled)

1. **Divergent halves** – anything that lets the two halves compute different scalar values
   desynchronises the exchanges. Covered: masks and lane reads (exchanged), GLC scalar loads
   (half 0's value), `s_memtime` (constant 0). Not exchanged: non-GLC scalar loads of memory
   that changes during the dispatch (incoherent on the console as well).
2. **Barriers in divergent control flow** – the barrier variant is only used when the group
   is exactly one console wave, whose control flow is identical in both halves; console
   `s_barrier` stays a group barrier (all console waves reach it, as on the console). The GS
   phase (lower half only) has no exchanges and its `s_barrier` is dropped (one wave).
3. **Lane numbering** – handled by `xcl_x_thread` (see above); NVIDIA really does form waves
   of non-consecutive thread indices for 3-D groups (`gpu_split_wave64_waves_diverge` fails
   without the remap).
4. **`WaveReadLaneAt` with a per-lane index** – used inside a half as before.
5. **Forward progress** – only the multi-wave variant spins; see above.

### Masks read lane by lane

A mask whose only readers take the reading lane's own bit – `v_cndmask_b32`'s selector, the
carry-in of `v_add_co_ci` / `v_sub(rev)_co_ci` – needs just that lane's half, so it is not
exchanged: `xcl_x_ballot_local` gives the half's own word and 0 for the other half
(`x64_masks.rs`). Which masks those are is a backward liveness analysis over the SGPRs: a
register is *whole-live* after an instruction when some path may read its whole value (any
SALU / SMEM operand, a VALU scalar operand, `s_cbranch_vcc*`, `src_vccz`, …) before it is
written again; a mask written to registers that are not whole-live stays in its half.

The halves then hold different values in those registers, so the rules follow what the
emitter does, not only the ISA's operand tables:
- a write is an overwrite only when the emitter replaces the whole register whatever it held:
  not `s_cmov*` / `s_cmovk` (conditional), `s_addk` / `s_mulk` / `s_bitset*` (they read the
  old value), `s_movreld` / `s_movrelsd_2` (they write `s[dst + m0]`), nor an SDWA compare
  with SD = 1 (it writes its SDST, VCC keeps its value);
- `s_movrels*` read `s[src + m0]`: every register from `src` up to s123, VCC included;
- SDWA scalar sources (the S0 / S1 bits), SMEM register offsets, `src_vccz` / `src_execz` and
  M0 are reads;
- EXEC is always whole-live; an indirect jump (`s_setpc` / `s_swappc` other than a far jump,
  which is a branch) makes every SGPR whole-live, and so does the end of a kernel-mode program
  (the runner reads the SGPRs back – the halves still end with the same SGPRs);
- every emitted loop can be left where its iteration budget (`LOOP_GUARD`) is checked – at
  its top, and in the `pc` dispatcher form before each of its blocks, to the code after it or
  (dispatcher) the end of the program: what is live there is live before each instruction of
  the loop.

A mask with no destination (VOP3 compare into NULL) or one that is 0 in every lane (the
`false` mask of `v_div_scale`) is not exchanged either. Both halves make the same static
choice, so their exchange counts stay in step. Geometry mode (ES → GS through `s_setpc`)
has no liveness pass: its masks are exchanged as before, except those two kinds (no
destination, 0 in every lane), which are the same in both halves either way.
`XCL_X64_ALL_MASKS=1` exchanges every mask (for comparison).

## Cost

A wave64 program pays one exchange per mask write that is read whole (`v_cmp*` feeding
`s_and_saveexec`, `s_cbranch_vccz`, `s_bcnt1`, …) and per readlane / readfirstlane /
ds_bpermute / coherent scalar load, nothing per ALU, memory or DPP instruction, nor for
masks read lane by lane (selects, 64-bit carry chains). Wave32 programs and everything on GPUs with 64-lane waves are
unchanged (the HLSL of every native translation the test suite makes is byte-identical). A
wave64 VS / PS pays nothing for a lane ≥ 32 read as a constant (the lane read is skipped) and
one scalar per spill slot read from its shadow.

Measured on an RTX 5070 Ti (`bench_split_wave64`, an ignored test): a loop of 8
instructions with one exchange, 2^18 threads × 32 768 iterations, wave32 program (no
exchange) vs the same as wave64 split:

| | 1 exchange / 8 instructions | 2 exchanges / 10 instructions |
|---|---|---|
| one console wave per group (barrier) | 1.3× | 1.6–2.2× |
| four console waves per group (counters) | 4.6–5× | 8.4–9.7× |

This is the worst case – real shaders have far more ALU and memory work per mask. The
counter-paired variant is the expensive one: a waiting wave keeps issuing (no way to park
it in HLSL), taking issue slots from the waves it waits for. The next step would be to skip
the exchange for masks only used lane by lane (`v_cmp` + `v_cndmask` selects; each half's
own word is enough there), which needs a liveness pass over the mask registers.

That pass is now in (see "Masks read lane by lane"). The benchmark now counts the
exchanges a loop iteration makes and times each dispatch with GPU timestamps (median of 9
dispatches in one command list; `runner::dispatch_timed`). Before, it took the wall time of
a whole `runner::dispatch` – pipeline creation, buffers, uploads, submission and the CPU's
wait included – which shrank every ratio towards 1 (the 5070 Ti figures above were taken
that way). On an RTX 3090, three runs with all masks exchanged (`XCL_X64_ALL_MASKS=1`) and
five with the default:

| | mask only | mask + `v_readfirstlane` |
|---|---|---|
| exchanges per iteration (all masks / default) | 1 / 0 | 2 / 1 |
| one console wave per group (barrier) | 1.64–1.68× → 1.04–1.11× | 1.99–2.02× → 1.74–1.81× |
| four console waves per group (counters) | 3.90–4.14× → 1.07–1.16× | 6.45–6.64× → 4.09–4.34× |

(wave32 baseline 11.4–14.1 ms.) The 4–16 % left with no exchange at all is the rest of
the split form (64-bit masks, the threads' start), not measured further. The benchmark's
loop mask is exactly the kind the pass keeps local – it shows the exchange gone and what
one costs, not how often real shaders gain.

What is left is readlane / readfirstlane, `ds_bpermute`, coherent scalar loads and masks
read whole. Not measured on real compute shaders (none on that machine); Minecraft's 273
wave64 programs, translated as compute, have 38 mask writes, all only read lane by lane.

## Vertex and pixel shaders

The driver picks the wave size of a VS or PS (HLSL has no `[WaveSize]` for them), and on a
GPU with 32-lane waves only a console wave64 VS / PS runs in waves of 32. Nothing pairs two of
them (no groupshared memory), so such a wave stands for a console wave whose lanes 32..63 are
not populated, as in the last, partial wave of a draw. Per-lane code is exact as it is, and so
are masks (their upper word is 0: those lanes are inactive), DPP / `ds_swizzle` /
`v_permlane(x)16` (inside a 32-lane half) and `v_readfirstlane`. What is not: a read of a lane
≥ 32 – `v_readlane` with a constant lane ≥ 32 or a lane from an SGPR, `ds_bpermute` – and
`v_writelane` to one. On the console those lanes exist, inactive, and hold what `v_writelane`
(which ignores EXEC) or whole-wave code (EXEC's upper word set, `s_or_saveexec_b64 s, -1`)
put there; here they do not exist, and `WaveReadLaneAt(v, l)` with l ≥ 32 is undefined (NVIDIA
gives lane l & 31).

### What the retail pack does

`retail_wave64_graphics_lane_census` (an ignored test) lists every such read in the pack's
wave64 VS / PS (13 210 programs; 749 – 748 PS, 1 VS – were flagged by the old `wave_hazards`):

| pattern | reads | programs (distinct code) | the console lane ≥ 32 holds | before |
|---|---|---|---|---|
| whole-wave reduction: the identity into the lanes outside the mask (`v_cndmask` with mask & saved EXEC), DPP8 xor 1 / 2 / 4, row_mirror, `v_permlanex16`, `v_readlane` 32 for the upper half, `v_readlane` 0 for the total; `v_min_u32` (identity -1), `v_max_u32` (0), `v_or_b32` (0) | 1 082 | 436 (434) | the identity | right by chance: NVIDIA reads lane 0 for lane 32, and min / max / or are idempotent (a sum would double) |
| waterfall / uniformity check: `v_readlane v, s` with s = `s_ff1_i32_b64(mask & EXEC)` | 355 | 353 (351) | nothing (the lane is < 32 unless the mask is empty; then 63, never written) | right |
| SGPR spill slots: `v_writelane` / `v_readlane` of lanes 0..47 of one VGPR | 200 | 11 | the spilled SGPR | wrong: the write is lost, the read returns another slot (lane & 31) |
| `ds_bpermute` | 0 | | | |

### The form

For a wave64 VS / PS with `split_wave64`, `upper_lanes.rs` follows what the console's lanes
32..63 hold through the program, a forward data-flow pass over its control flow:
- EXEC's upper word, with the SGPRs as words known to be 0, all ones, or anything: 0 at the
  start, 0 in every mask a VALU instruction writes while it is 0, combined by the bitwise SALU
  and saveexec forms. While it is 0 no VALU instruction writes lanes 32..63.
- per VGPR lane ≥ 32: unwritten, a constant, a spill slot, or unknown. A VALU write while
  EXEC's upper word is all ones is evaluated (moves, `v_cndmask`, integer min / max / and / or /
  xor / add, their DPP16 / DPP8 forms, `v_permlane(x)16` of lanes that all hold the same);
  `v_writelane` makes the lane a slot. Paths join lane by lane (an unwritten lane takes the
  other path's value: any value is right for it).

Each read then gets:
- unwritten: the read as it is – the console reads what an earlier wave left in the register,
  which no program can depend on;
- a constant: that constant (`uint t = c; if (!xcl_hw32) t = rla(…);`);
- a slot: its shadow `xcl_sh<vgpr>_<lane>`, a scalar every `v_writelane` of that slot also
  writes;
- unknown: stays a hazard – `wave_hazards` reports only these.

Spill slots < 32 are read from their shadows as well, whatever the wave size, as long as no
other instruction has written the VGPR since the slot's `v_writelane`: in a partial wave (a
primitive's edge) a lane that was not launched loses what `v_writelane` put there. 287 pixel
shaders of the pack (3 540 reads) spill to lanes < 32.

The wave size is checked at run time (`xcl_hw32 = WaveGetLaneCount() < 64`), so a driver that
picks 64 lanes runs the program as before.

Which GPUs get the form: every GPU whose wave range includes 32 (`pso::graphics_wave32`), not
only those without 64-lane waves. D3D12 has no `WaveSize` for graphics shaders, so the driver
picks, and an RDNA2 driver (RX 6600 XT, issue #42) runs a wave64 PS in waves of 32 although it
reports 32–64: without the form, `v_readlane` 32 wrapped to lane 0 there
(`gpu_wave64_ps_whole_wave_reductions` summed twice). Compute, mesh and geometry programs set
`[WaveSize(64)]` and still get the split form only on GPUs without 64-lane waves.

On a GPU with 64-lane waves only, nothing changes: the HLSL of every program of the pack is
identical to the one before
(`retail_wave64_graphics_lane_forms` prints digests of the native form, and of the other modes'
32-lane form, to compare builds).

In the pack every read resolves: 1 082 constants, 355 unwritten, 3 740 slot reads; none is
left unsafe. Only a program with a constant or a slot to read gets the form: the HLSL changes
for 613 programs (the 436 reductions and the 287 with spill slots, which overlap), all of
which DXC compiles. Tests on the GPU: `gpu_wave64_ps_whole_wave_reductions`
(the pack's reduction as min and as a sum, and the waterfall read, checked per hardware wave
against a console wave of the same lanes), `gpu_wave64_ps_spill_slots` (whole target and a
6-pixel triangle), `gpu_wave64_vs_spill_slots_and_identity` (a wave of 3 vertices).

### What is still not exact

**Lanes the PC did not launch, inside a wave** – on every GPU and wave size, and before this
change too. The console runs whole-wave code in all 64 lanes, inactive ones included; the
translation runs it only in the lanes the PC GPU launched. In a partial wave, the reduction's
DPP / `v_permlanex16` steps read lanes that were never launched (undefined) instead of the
identity the console put there – not only lanes 32..63. The lanes ≥ 32 of a 32-lane wave are
handled above because they are all missing and all hold the identity; a lane missing among
launched ones holds a partial result that the translation does not have. Making that exact
needs the reduction recognised as a whole (`WaveActiveMin` / `Max` / `BitOr` over the
launched lanes, with the identity for the lanes outside the mask) – not done. Measured on an
RTX 5070 Ti (the reduction of `gpu_wave64_ps_whole_wave_reductions` drawn as a 3-quad
triangle, lanes 12..31 not launched): those lanes read 0, so a min reduction (identity -1)
gives 0, while max / or / sum (identity 0) and the waterfall read are right. 325 of the 436
reduction programs have a min reduction (322 a max, 433 an or).

Spill slots of lanes that were not launched are also still lost in the native form (GPUs with
64-lane waves, which keep the HLSL as it was), and in the 32-lane form for slots ≥ 32 when the
driver picks 64 lanes.

## Relation to earlier work

* **PR #33** (ElHombreAlex, draft): first wave64 → wave32 "rendezvous" through groupshared
  for compute, and the finding that NVIDIA rejects `[WaveSize(64)]`. Its exchange keeps one
  slot per half (the partner can overwrite it with the next exchange before it is read),
  publishes from lane 0 only and times out into stale data; the FSR4 pass it targeted hung
  the GPU. This design keeps its idea of placing the exchange in the LDS tail and fixes the
  protocol (double-buffered slots, counted arrivals, barriers where they are legal).
* **PR #35** (bmjask): a separate per-slot lowering of mesh programs, validation only and
  not connected to translation; its merged parts (`mesh_wave.rs` requirement report,
  `mesh_dispatch.rs` scratch / rendezvous model) are not used by this form.
* **PRs #14 / #39** (split ES+GS / mesh into independent halves and reject cross-half
  operations): replaced by the exchange form, which needs no rejection list.
