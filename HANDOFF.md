# XCL – work handoff (status as of 2026-11-20, session 5)

Xbox Series X|S → Windows compatibility layer (no CPU emulation), Rust (this repository).
Design: `docs/specs/2026-11-20-xcl-design.md` (sections 5.x, including 5.7a – console shader format, 5.7b – recompiler status). Manual: `README.md`.
Conversation with the user in Polish; everything in the repo (code, comments, messages, docs) in English. Git repo: `main`, pushed to origin; commits only at the user's request.

## Environment
Windows 11, AMD Radeon RX 9070 (RDNA4) + iGPU, Rust 1.96 `x86_64-pc-windows-msvc`, VS 2022, Python 3.12.
Build/test: `.\scripts\test.ps1` (= `cargo build --workspace` + `cargo test --workspace`; e2e needs the test title to be built).
NOTE:
- editing files through PowerShell `Get-Content/Set-Content` corrupts UTF-8 (non-ASCII characters) – use the Write/Edit tools
  or `[IO.File]::WriteAllText(..., UTF8Encoding($false))`;
- in Git Bash a heredoc `<<'EOF'` passed to Python eats backslashes (`\\n` in a Rust string → a real newline character,
  paths `C:\U…` → syntax error) – edit text containing backslashes with the Edit tool;
- `scripts\test.ps1` run in the background with `*>` redirection in PowerShell 5.1 ends in an error (cargo's stderr = an exception under
  `ErrorActionPreference=Stop`) – in the background run it through bash: `cargo build --workspace && cargo test --workspace -- --test-threads=4`.

## Rules from the user / constraints
- Work on the sample files and test: `<samples>\shaders.rpf` (shaders),
  `<samples>\audio.rpf` (audio, 28.5 GB – read it in pieces!). Next to them `rpf-viewer.html` –
  a pack viewer (format of the shader records in RPF9, ISG1/OSG1 signature parser).
- `<samples>\recompiler` – a fragment of AnyPS5 (RDNA → SPIR-V recompiler for PS5) from the user, license
  **GPL-2.0-only** (github.com/boykopovar/AnyPS5). **The user agreed to relicense XCL to GPL-2.0-only (2026-11-20)**
  – done (`LICENSE`, `Cargo.toml`, `THIRD_PARTY.md`), so AnyPS5 logic may be ported with a provenance note in the file.
  Consequences: WinDurango code (LGPL-3.0) only as a reference; dependencies must be GPL-2-compatible (no pure Apache-2.0 / GPL-3).
  Technically AnyPS5 is Vulkan/SPIR-V + the PS5 SRT model – useful parts: T#/S#/V# descriptors, images, interpolation, VS inputs.
- **The user has no time for long compiles/runs (2026-11-20)**: avoid steps that take many minutes
  (e.g. `--compile` on thousands of samples, huge DXC measurements); check on small samples, work fast.

## What works (verified by tests)
- **PE loader** (`loader.rs`, `pe.rs`): mapping, relocations, imports + delay-load, DllMain/TLS callbacks, game DLLs.
- **Static TLS** (`tls.rs`): a generated "carrier DLL"; Win11 findings: the loader allocates a slot only for DLLs with imports
  and ignores `SizeOfZeroFill` (the whole size goes into "raw data").
- **C++ exceptions**: hook on `RtlPcToFileHeader` (retour) + `RtlAddFunctionTable` (`eh.rs`).
- **Resolver** (`resolver.rs`, `hle/mod.rs`): HLE → game DLL → Windows → auto-stub with a log (`stubs.rs`, `data/stubdb.toml`).
- **kernelx** (`hle/kernelx.rs`), **XGameRuntime offline** (`hle/xgameruntime/`: XTaskQueue, XAsync, XUser, XSystem, XPackage, PLS, save folder).
- **VFS** (`vfs.rs`, `vfs_hooks.rs`): `G:\`, `GAME:`, `T:\`, copy-on-write, whiteouts, KERNELBASE hooks (they also catch the CRT).
- **Launcher/host/report** (`launcher.rs`, `host.rs`, `report.rs`, `inspect.rs`): child process with the game's command line;
  does not pick `xcl.exe` itself; warns when the exe does not import console DLLs; `report.json` (called_stubs, shaders, d3d12, vfs…).
- **RDNA2 decoder** (`gpu/rdna2.rs` + generated `gpu/rdna2_spec.rs` from the AMD Machine-Readable ISA XML, MIT; generator
  `scripts/gen_rdna2.py`; the XML can be downloaded from gpuopen.com/machine-readable-isa): 23 812 instructions from the LLVM tests
  (`tests/data/llvm-amdgpu`) – 100% name/length agreement; `xcl disasm`. `rdna2::branch_target`, `rdna2::all_instructions`.
- **Console ISA extensions** (`gpu/rdna2_xbox.rs`, NEW): `v_mac_f32` (VOP2 0x1F / VOP3 0x11F), `v_madmk_f32` (0x20),
  `v_madak_f32` (0x21), `v_mad_f32` (VOP3 0x141) – kept from gfx10.1, removed in gfx10.3; the decoder takes them only on
  opcodes that are free in the spec. The VALU literal is fetched when ANY SRC0/1/2 field = 255, even one unused by the opcode
  (the compiler hides a hash in `v_nop` this way) – displayed as `v_nop ; lit 0x…`. SDWA/DPP: src0 from the extension dword (was `?249`).
- **Shader capture** (`gpu/shader.rs`): classification DXIL/DXBC/**xbox-native**/unknown container/ELF/raw RDNA2,
  dump to `<data>\shaders\` (+ `*.fssmN.bin`, `*.rdna2.txt`; since 2026-11-23 only with `XCL_DUMP_SHADERS=1`, as the
  `.hlsl` and `recomp_*` below); `report.json → shaders`: `xbox_hw_stages`, `xbox_errors`,
  `rdna2_instructions`, `rdna2_undecoded_dwords`, `rdna2_mnemonics` (histogram sorted in descending order).
- **Console shader format** (`gpu/xbox.rs`, NEW) – worked out on `shaders.rpf` (23 869 shaders, all stages),
  full description in the design §5.7a: DXIL (bitcode `NOBC`) → XDAT (+0x18; offsets from the program header) → XBCH v3 (zlib from +0x18)
  → FSSM v0x101 (code: u64 +0xC0/+0xC8, offset counted from +0x10; stage descriptors `0xC0_0002_AAHH` from +0x100 every 0x88,
  HH = PAL hardware stage: 2 ES, 3 GS, 4 VS, 5 PS, 6 CS, 7 MS/NGG). Code split: segments up to `s_endpgm`, appended to the
  program when connected by a jump (shared "discard" exit in PS, trampolines); constant blocks `s_endpgm | N | N dwords |
  s_endpgm` separately (`data_blocks`). Every step has a fallback (synthetic tests). Result on the whole pack: 24 064 FSSM =
  24 064 programs, 473 data blocks, 0 parse errors, 0 undecoded dwords in 28.97 million instructions, 395 mnemonics.
- **`xcl shaders <file|folder|RPF9 pack> [--top N] [--json]`** (`shaders.rs` + `gpu/pack.rs`, NEW – RPF9 record scanner
  as in the viewer): formats, DXIL and hardware stages, mnemonic histogram, "in how many shaders", implementation order
  (`--json → implementation_order`) and coverage, list of unusual containers (with the record offset); multithreaded, whole pack
  approx. 1.7 s (release). `xcl disasm` for xbox-native: XDAT table, XBCH, stages, FSSM header qwords, `--hex`, `--fssm FILE`.
- **RDNA2 → HLSL recompiler** (`gpu/recomp/`, session 3; description in the design §5.7b): `translate()` → a `switch(pc)` dispatcher over
  blocks (AMD jumps are uniform across the wave → exact without structurization), SGPRs uniform, VGPRs per thread with the EXEC mask,
  masks via `WaveActiveBallot`, cross-lane via `WaveReadLaneAt`. `emit.rs` – semantics: SALU (exact SCC), VALU
  f32/u32/i32 + VOP3 modifiers, SDWA, DPP16/DPP8, `ds_swizzle`, `v_permlane(x)16`, `v_fma_mix*`, cube, VOPC/VOPCX,
  saveexec/wrexec, `s_load`/`s_buffer_load` (V#), MUBUF/MTBUF (raw + 78 GFX10 formats from LLVM `UfmtSymbolicGFX10`,
  DST_SEL), `buffer_store_dword*`, exports; the rest → a comment + the `unsupported` counter. `s_swappc_b64 s[n:n+1], s[n:n+1]`
  = fetch shader call (ignored in kernel mode). **Kernel** mode (compute `[WaveSize(64)]`, registers from/to a buffer,
  flat memory `mem`; program code in `mem` from 0 for constants from `s_getpc`). `dxc.rs` – in-process DXC (dxcompiler.dll from
  the Windows SDK / next to the exe / `XCL_DXC_DIR`). `runner.rs` – D3D12 compute on the high-performance GPU. `rdna2::encode` – an assembler
  for tests (also SDWA/DPP layouts); `rdna2::register_scale` – FIXED disassembler bug: SBASE counts SGPR pairs, SRSRC/SSAMP
  quads (was `s[8:9]` instead of `s[16:17]`). Result: **`VS_BlitGenAmbCube` from the pack runs on the RX 9070 with a correct
  SV_Position**; corpus: 24 064 programs, 0 translation errors, **7 442 (30.9%) in full**, DXC compiles 500/500 samples.
  `xcl shaders … --recomp [--compile N]` (missing instructions: in how many programs + "the only missing one").
  Note: DXC divides floats approximately → the ends of the UNORM/SNORM range are computed exactly, separately.
- **XAudio2** (`hle/xaudio2.rs`, NEW): HLE module `xaudio2_9.dll` (+ `xaudio2_9redist.dll`) → system XAudio2 2.9
  (loaded dynamically). Create: flags outside 0x2000|0x8000|0x10000 are dropped (NOTE: the old note "0x18002, processor=0"
  was wrong – according to xaudio2.h 0x2 is a voice flag, processor = XAUDIO2_DEFAULT_PROCESSOR = 1). Vtable patching IN PLACE
  (originals per (vtable, slot)): IXAudio2 #5 CreateSourceVoice (XMA/XMA2 → PCM16; another format rejected by Windows,
  e.g. Opus 0x704F → also PCM16), #7 CreateMasteringVoice (console device id → default; `CO_E_NOTINITIALIZED` →
  CoInitializeEx(MTA) + retry – discovered in e2e; then 0/0 + the GameEffects category), SourceVoice #21 SubmitSourceBuffer
  (silence of length PlayBegin+PlayLength or an estimated one; context/flags/loops preserved), #18 DestroyVoice.
  Refactor: `HleModule.system_fallback: bool` → `fallback: &[&str]` (`hle::SELF` = "@self") + `aliases_are_system()`;
  the remaining XAudio2 exports (X3DAudio*, CreateFX…) → system xaudio2_9.dll. `report.json → audio`.
- **Real D3D12 backend** (`hle/d3d12x/real.rs`): real driver objects with an extended vtable; layout from `d3d12.h`
  (`scripts/gen_d3d12_layout.py` → `hle/d3d12_layout.rs`). Finding: driver vtables have internal virtuals behind the public ones
  (Device 125 vs 104) → **guards** (`hle/d3d12x/guard.rs`, machine code: a call from the game image = console method → log+0,
  otherwise → the original). Overrides: QI (unknown IIDs), all Create* (adoption), PSO/root signature with a fallback
  (replacement shaders when the driver rejects the console ones). Fallback `null.rs` when there is no GPU / `graphics="null"`.
- **Presentation** (`hle/d3d12x/present.rs`): XCL window + swap chain; back buffer detected via RTV/OMSetRenderTargets/
  ExecuteCommandLists; a frame per call of the console queue method (probably PresentX). UNTESTED on a game.
- Test title `tests/testtitle` (+`tests/testgamedll`): 26 checks, including a triangle rendered on the RX 9070 through our
  `d3d12_xs` and pixel readback; session 2 added: a shader in the console format (XDAT/XBCH/FSSM with `v_mac_f32`) at
  PSO creation → `report.json` (xbox-native, histogram) and XAudio2 (console flags/device, an XMA2 voice plays silence – a 0.1 s buffer
  ends after ~95 ms, the PCM voice unchanged, callbacks with contexts; without an audio device the voice part is skipped).
  Test status (end of session 3): 75 unit tests (including 11 recompiler tests executed on the GPU) + e2e (28 checks
  in the title) – all PASSED. Tests on real data (`shaders::tests::retail_pack_sample`,
  `gpu::recomp::tests::gpu_retail_vs_blit_gen_amb_cube`) work only with
  `XCL_SAMPLE_RPF=<samples>\shaders.rpf` (skipped without the variable) – PASSED.

## Recompiler – session 4 (2026-11-20)
- **Corpus coverage: 99.1% of programs translate in full** (23 846 / 24 064; was 75%): MS 9 282 / 9 282 (100%),
  PS 99.9%, VS 92.0% (only the 195 ES merged with GS are missing), CS/kernel 96.4%. DXC: 300/300 samples OK.
- New in `emit.rs`: MIMG A16/D16 (address layout as in AnyPS5 `componentShape`), `image_get_lod`, `_cl` (LOD clamp);
  `buffer_store_format_*` (packing of 78 formats, `st_bytes` with atomics for partial dwords), `buffer_atomic_*`;
  global/flat load/store; LDS (`ds_read/write*`, `read2/write2(st64)`, atomics → `groupshared`, 32 KB = the D3D12 limit);
  `ds_bpermute`; 16-bit VALU (op_sel, preserving the other half as in gfx10 / AnyPS5 `write16Bits`), f16/u16/i16 comparisons,
  VOP3P `v_pk_*`, `v_dot*`; 64-bit (`v_cmp_*_u64/i64/f64`, b64 shifts, `v_mad_u64_u32`, f64 add/mul/fma/cvt, 64-bit inline
  constants as double); `v_movrel*` (switch on M0).
- **Mesh mode** (`Mode::Mesh`, `MsInterface` from PSV0: 128 threads, ≤128 vertices/triangles): DXC does not allow
  `WaveSize` in an MS, and console code assumes wave64 → two passes: the program as a CS `[WaveSize(64)]` writes a group record
  (`s_sendmsg 9` GS_ALLOC_REQ M0 = prim<<12 | vert, `exp pos0/param/prim`, vertex = thread in the group), then
  `mesh_passthrough()` rasterizes the records. Inputs measured on the corpus (`xcl shaders --iface`): s3[27:24] = wave in the group,
  v5 = group·128 + thread (group = v5>>7, like the gfx10.3 MS emulation in RADV), s16–s24 user data (from `io`).
  Test `gpu_mesh_mode_two_passes`: 2 groups × 2 waves → correct records and pixels on the RX 9070 (`runner::mesh_draw`).
- Control flow: instead of a single `switch(pc)` loop, a chain of `if (pc == block)` in address order; `while` only for
  ranges with backward jumps (merged into nesting). Smaller DXIL (470 → 310 KB on the largest program).
- **DXC: the optimizer is superlinear on large programs** (median 0.6 s, but an MS with a DS/VS wrapper, 0.7–1.2 MB of HLSL:
  50–160 s at -O1..-O3, 5 s at -O0) → `dxc::compile` adds `-O0` by itself for HLSL > 256 KB (the driver optimizes anyway).
  Do NOT run `--compile 3000` (an hour); for a quick check use `--compile 64`.
- Tools: `xcl shaders … --recomp --example MNEMONIC|name=FRAGMENT` (listing + DXC result; `XCL_EXAMPLE_HLSL=file` saves
  the HLSL), `--iface` → inputs/exports/`s_sendmsg`/PSV0 per stage, DXC times with `--compile N`.
  `dxbc::psv()` (PSV0), `rdna2::sgpr_uses()`. Disassembler: VGPRs in a 9-bit field (`v_readfirstlane … v261` → `v5`).
- Tests: 86 unit tests (including 17 recompiler tests on the GPU) – PASSED (`cargo test -p xcl --lib`, ~0.5 s).

- Later in session 4: `src_pops_exiting_wave_id` (POPS – the wait is passed through, no ROV ordering: a known limitation),
  flat apertures = 0, `image_*_o` (offset from a VGPR → coordinates shifted by offset/mip size; GPU test) → PS 12 097 /
  12 101. **Runtime:** a captured console shader is translated right away (mode from the interface), `.fssmN.pM.hlsl` next to the dumps,
  `report.json → shaders.recomp_*` (programs, complete ones, modes, missing mnemonics); `XCL_RECOMP_DXC=1` compiles through
  `dxc::compile_cached` (DXIL cache in `%LOCALAPPDATA%\xcl\dxil` or `XCL_DXIL_CACHE`). e2e checks this.
- Gallery: `retail_ps_gallery` (ignored test, `XCL_GALLERY_DIR`) – 24 PS shaders from the pack drawn on the GPU (PNG + page
  `index.html` in that folder, kept locally – not in the repository: renders of game shaders), approx. 10 s.
- FSSM header: `--example name=…` prints it (the dwords up to the code). Stage descriptor +0x110 looks like RSRC2 (USER_SGPR in
  bits 5:1, e.g. 0x18 → 12); from approx. +0x25C in PS an export patch table (`exp` variants). The root signature → user
  SGPR mapping is done by the D3D12.X runtime (the SGPRs carry pointers to tables) – to be determined only from the game exe.
- GS (195 + 195 ES) postponed: on-chip GS – the ES writes to LDS and does `s_setpc` to the GS, the GS writes vertices to LDS
  (64 B/vertex, base = s2 + ([s8+0xB8] << 8)), `s_sendmsg` emit/cut/done; no copy shader in the pack.
- Tests: 90 unit + e2e (29 checks in the title) – PASSED (`XCL_SAMPLE_RPF=… cargo test --workspace -- --test-threads=4`, a few seconds).

- **UMA – console memory accessible to the GPU** (`gpu/uma.rs`, NEW): a single region (2048 MB by default, `XCL_UMA_MB`,
  max. 4095 MB) at an address aligned to 4 GiB, fully committed, opened through
  `ID3D12Device3::OpenExistingHeapFromAddress` (measured: a partial commit is rejected, 4 GiB is rejected, 4095 MB OK in 0.5 s).
  The low 32 bits of a console address = offset in the buffer → the recompiler's flat `mem` model works on game memory without copies.
  First-fit allocator (64 KB, coalescing, zeroing). What goes there: `VirtualAlloc` with `MEM_GRAPHICS` (+ new HLE `VirtualFree(Ex)`),
  `XMemAlloc` with a memory type ≠ 0. The `real` backend opens the region as a heap + a UAV buffer at device creation
  (`real::uma_gpu_va()`, `report.json → d3d12.uma_*`). Tests: `gpu_unified_memory_arena` (s_load/buffer_load/store directly on
  process memory), `uma::tests`, a check in the test title, e2e checks `uma_gpu_va`.
  Next: bind the shaders from the game's PSOs to this buffer as `mem` (u2), and fill the SGPRs from root arguments (requires the game exe).

- **Without the exe (session 4, end):**
  - `gpu/desc.rs` (NEW): GFX10 T#/S#/V# → D3D12 (DXGI format from the GFX10 IMG_FORMAT table – 0–77 like buffers, 128+ sRGB/
    packed, 169–182 BC1–BC7; SRV with channel mapping from DST_SEL, dimensions/mips/arrays/cube/3D/MSAA; sampler: addressing,
    filters, aniso, comparison, min/max, LOD, bias, border). Fields confirmed against AnyPS5. Unit + GPU tests
    (`gpu_tsharp_to_d3d12_resources`: 6 kinds of textures created from decoded T#s). Not used in the runtime yet.
  - GameInput: the `gameinput.dll` import goes to the system `GameInput.dll` (export `GameInputCreate`) – works without
    emulation, checked in the title. If the console version of the API turns out to be different – an alias in the config / HLE.
  - XGameSave (`hle/xgameruntime/gamesave.rs`, NEW): provider/containers/blobs/updates (+ Async/Result), files in
    `<data>\gamesave\<configId>\<container>\b\<blob>`, names %XX-encoded, writes through a temporary file, 256 MB limit.
  - `xcl rpf <pack> [--find|--extract … --out …]` (NEW): list/extract records; `textures.rpf` – 21 nested
    RPF9s with an encrypted TOC (fixed key from the game, data without zstd/LZ4/zlib signatures – probably Oodle) → unreadable
    without the exe; the game unpacks the archives by itself anyway, XCL will get the textures in memory (GFX10 tiling).
  - Deliberately NOT done (waiting for the exe, so as not to guess): physical pages (`MapTitlePhysicalPages` at the game's address),
    GFX10 tiling decoder (no real data to verify against), Opus through the console API, GS.
  - Tests: 96 unit + e2e (29 checks in the title) – PASSED.

- **Translator hooked into the graphics PSO** (`gpu/recomp/pso.rs` NEW, hook in `hle/d3d12x/real.rs`):
  `CreateGraphicsPipelineState` with a native console VS (and PS) → `pso::console_to_dxil` (hardware VS/PS variant from FSSM,
  mode from the interface, only complete translations, DXC with cache) → a PSO with translated shaders and the **XCL ABI root signature**
  (`pso::abi_root_signature`: root UAV u0 io / u1 res / u2 mem + directly indexed heaps; `real::abi_root_signature()`).
  Failure → the old path (original → replacement shaders). `report.json → d3d12.translated_pipelines / translation_errors`.
  GS/HS/DS not yet. **Missing:** binding at draw time – the game's root arguments → SGPRs in `io`, `mem` → the UMA buffer
  (requires the user data layout of the console runtime, i.e. the exe).
- E2E test without the exe: `crates/xcl/tests/gpu_pso_e2e.rs` – `VS_Rage_TransformPosCpv` + `PS_Rage_Untextured` from `shaders.rpf`
  through the backend's device hook → `runner::draw_on` (new: drawing with a ready PSO on any device) → 64×64 RGBA32F;
  explicit inputs (s8 → table of 2 V# → 4×3 world matrix and 4×4 view-projection, POSITION+COLOR vertices); 435 pixels
  inside compared with interpolation computed on the CPU (±0.01), 3501 outside = background color; the `translated_pipelines` counter +1,
  `fallback_pipelines` unchanged. Checked by sabotage: with translation disabled the test fails (both on the counter and on the pixels).
  `real::init()` is now idempotent (previously a direct call + `backend()` created two devices).
- Tests: 96 unit + title e2e + `gpu_pso_e2e` – PASSED.

- Second test in `gpu_pso_e2e.rs` – textures through the decoder: `VS_Rage_TransformPosCpvTex` + `PS_Rage_Textured` (T# from the table
  s[8:9]+0x20, S# built into the PS code = static sampler: aniso 4×, wrap). Two linear 64×64 RGBA8 textures with known
  patterns in memory at the addresses from the console T#s → `desc::image` / `image_d3d12` (resource + SRV, data read from the address
  in the T#), S# → `desc::sampler_d3d12`; tables in the XCL ABI form (`desc::xcl_image_words`: address → SRV index in the heap).
  One PSO, one command list, two draw calls that differ only in the table pointer in user data (left half: A,
  right: B with DST_SEL zyxw) – `runner::draw_on_multi` + `runner::ViewTexture`/`Draw` (NEW). 4096 pixels = texel (±0.005),
  2048/2048 on the right differ from A. Sabotage checked: DST_SEL ignored → fails; the same table in both draws → fails.

- **GS emulation** (`Mode::Geometry`, `GsInterface`, `recomp::geometry_program(es, gs)`): legacy on-chip GS from the pack
  (195 `GS_VoxelizeGeneric` + 195 ES variants `VS_*`, paired by hash suffix). ES and GS of one pipeline = one CS
  `[WaveSize(64)]`, group = up to 21 triangles (63 ES vertices); `s_setpc_b64` ES → GS code; LDS = `groupshared`;
  ESGS element = 16 B per ES OSG1 register (measured: variant 6be01b34 writes every 16 B, 912dfba0 every 32 B – the console driver
  derives this from the ES); GS inputs: v0/v1 = vertex offsets in dwords (16-bit packed), s3 = vertices | primitives<<8,
  s2 = GS output area after the 64 ES elements; `s_sendmsg` emit/cut → triangles from the strip (odd ones swapped), the thread's output
  base = the lowest address it wrote in the GS phase (does not depend on the value at [s8+0xB8]); result → mesh mode record →
  `mesh_passthrough`. ES attributes fetched beforehand: float4 per ISG1 register in `mem` at io[111], primitive count io[110].
  **Assumptions to check on the game:** the value of [s8+0xB8] (GS output base, 0 in the test), the meaning of s19/s21/s16/s17 (user
  data), triangles only → triangle strip, no GS PSO/drawing in the backend (command list capture needed).
- **Fixed translator bug:** `ds_read*` with a destination register overlapping the address (`ds_read2_b32 v[0:1], v0`) computed the second
  address from the overwritten register – now the address goes into a variable first. Only a real GS exposed it; regression test
  `gpu_ds_read_destination_overlaps_address`.
- Corpus: **24 056 / 24 064 (99.97%)** – GS 390/390, MS, PS, VS 100%; 8 CS with `image_bvh*_intersect_ray` remained.
- GS tests: `gpu_geometry_mode_es_gs_strip` (synthetic ES+GS: 25 triangles = 2 groups, a 4-vertex strip, every vertex
  and index exact, drawing) and `retail_es_gs_voxelize_geometry_mode` (a real pair from the pack: projection onto the plane of the dominant
  axis, exact – z → (x, y), x → (z, y), depth 0.5·d+0.5, axis 4/0, TEXCOORD1 = (-1,-1,1,1), TEXCOORD3 = unit
  normal, 496 pixels). All tests: 98 unit + e2e + 3 in `gpu_pso_e2e` – PASSED.

- **Ray tracing: `image_bvh(64)_intersect_ray`** (8 programs: `CS_RTShadowTrace*` wave32, `S_ScreenProbes_TraceRays_BVH`):
  HLSL function `bvh_intersect` in PRELUDE_MEM following the public RDNA2/RADV format – node = id (type in bits 2:0) at
  base + (id & ~7)·8, base from the descriptor in 256 B units (word 0), sorting = word 1 bit 31; box32 (5) / box16 (4,
  f16): slab test of 4 children, hits sorted by distance (a 5-comparison network), misses 0xFFFFFFFF; triangle (0..3):
  (t, denominator, i, j) as numerators (Möller–Trumbore), miss t = inf; instance/AABB → 0xFFFFFFFF (the shader handles it).
  A16 (dir/inv_dir as halves) supported. The game code confirms the layout (node address = (id >> 3) << 6, triangle_id at +48).
  Test `gpu_bvh_intersect_ray` (hand-built BVH, 3 kinds of rays, sorted/unsorted, box16, hit t=5 i=j=0.5, misses,
  64-bit variant). Approximations: NaN/equality edge cases and the hardware's watertight variant, return modes other than IJ.
- **Corpus: 24 064 / 24 064 (100%)** of programs translate in full; DXC compiles both RT programs.
  Tests: 99 unit + e2e + 3 in `gpu_pso_e2e` – PASSED.

## Session 5 (2026-11-20) – everything that can be done without the exe
- **Compute mode** (`Mode::Compute`, `CsInterface::from_descriptor`): COMPUTE_PGM_RSRC2 from stage descriptor +0x38
  (USER_SGPR bits 5:1, TGID_X/Y/Z bits 7/8/9, TG_SIZE bit 10, TIDIG_COMP_CNT bits 12:11), numthreads from +0x3C/+0x40/+0x44
  (= PSV0 in all 36 checked CS); group ids after the user SGPRs, v0..v2 = thread id. `runner::dispatch(dxil, groups, sgpr, mem)`.
- **VS/PS wave width** (chosen by the PC driver): a wave32 program in a 64-wide wave runs as two halves (`xcl_w32`,
  `xcl_half`, half-aware `rla()`, `ballot()`); Kernel with `hw_wave == 64 && !wave64` likewise. A wave64 program in a 32-wide wave
  breaks only on readlane of a lane ≥ 32, a lane from an SGPR, and ds_bpermute – `recomp::wave_hazards` reports this, does not fix it.
  Corpus: 1127 wave32 VS/PS handled, 749 wave64 at risk (447 lane ≥ 32, 353 lane from an SGPR).
- `xcl shaders … --precompile [--limit N]`: DXIL into the cache (`dxc::compile_cached`, `%LOCALAPPDATA%\xcl\dxil` or
  `XCL_DXIL_CACHE`); 60 records in 4.6 s, repeat run 0.2 s (from the cache).
- **VS inputs:** SV_VertexID / SV_InstanceID with a register in ISG1 land like attributes in v[4 + 4r] (`VInstanced*` read
  the instance id in v8) – FIXED, test `vs_system_values_in_isg1_registers`. The `--iface` analysis counts from the MUBUF/MTBUF VADDR
  only the registers actually used (IDXEN + OFFEN) – most "outlier" VS were false alarms (the upper half of a 64-bit VADDR).
  Layout v4 + 4·register: 779 / 788 VS. Remaining: 9 `VS_PassthroughComposite_*_Wrapped`: `image_load v[8:11]` with only v8
  set – the analysis counts 4 addresses without looking at DIM, almost certainly a false alarm too.
- **Physical pages** (`hle/kernelx.rs`): Allocate/Free/MapTitlePhysicalPages in UMA – 64 KiB pages, allocation = a contiguous
  UMA block, page number = offset / 64 KiB; mapping consecutive pages returns their address in UMA (visible to the GPU, no copy);
  mapping at an address chosen by the game, or of scattered pages → refused + log (aliasing would require a memory section).
  Test `physical_pages_in_uma`.
- **GFX10 tiling** (`gpu/tiling.rs`; `gpu/tiling_tables.rs` from `scripts/gen_tiling.py` – AMD addrlib patterns from Mesa, MIT, sources
  downloaded from gitlab.freedesktop.org into the scratchpad): 14 2D modes (256 B / 4 KB / 64 KB: S, D, S_T, D_T, S_X, D_X, Z_X, R_X)
  × 5 element sizes × pipe/packer configuration (RB+ tables, gfx10.3). Linear offset over GF(2): `ox[x] ^ oy[y]`;
  `Layout::{new, offset, detile, tile, size}`, `tiling::element(IMG_FORMAT)`, `tiling::image_layout(&desc::Image, cfg)`
  (`base_xor` from the low bits of the T# address). **Verified on real data:** the patterns that the TTPF viewer
  (`tools/ttpf-viewer/src/tile.rs`) found by hand on game textures are bit for bit SW_64KB_D (64 KB blocks) and SW_4KB_S
  (4 KB blocks) for 1/2/4/8/16 B elements. All modes × configurations yield a permutation of the block (test). Unknowns:
  the console's pipe/packer configuration (GB_ADDR_CONFIG) for the _X modes – the `PipeConfig` parameter, to be determined on a texture
  from the game (e.g. pick the configuration that gives the smoothest image); mip chain and mip tail – no (each level separately).
  `StandardSwizzle64KBSupported` on the RX 9070 = 0, so untiling stays on our side anyway.
- **BVH builder** (`gpu/bvh.rs`): BLAS in the RDNA2 format (box32 128 B + triangle 64 B, primitive id @48, geometry @52),
  split at the centroid median on the longest axis, 4 children per node, 1 triangle per leaf. `bvh::intersect_node` is the CPU
  mirror of the HLSL emulation, `Bvh::trace` is a stack-based traversal to the closest hit. Tests: traversal = brute force
  (300 triangles, 2000 rays) and `gpu_built_bvh_nodes_match_the_cpu_node_test` (64 lanes, nodes from the builder through
  the translated `image_bvh_intersect_ray` = CPU result). Missing: instance nodes (TLAS – the console driver's layout, traversed
  by the shader code, to be read from `CS_RTShadowTrace*`), box16 in the builder, hooking into `BuildRaytracingAccelerationStructure`.
- **Opus: decoded** (session 5, PR #3) – see "Opus decoding" in the Audio section; what the exe still has to show is how
  the title hands Opus to the system (XAudio2 0x704F voices with AWC-framed buffers, or a console API).
- Tests: 111 unit (7 ignored helpers) + title e2e + 3 in `gpu_pso_e2e` – PASSED (full `cargo test --workspace`).

### Console DLLs from the game pack (session 5, without the exe)
The user provided 16 DLLs from chunk 1002 (`<samples>\*.dll`, ordinary PEs): the console CRT (vcruntime140(_1),
msvcp140(_1/_2/_atomic_wait/_codecvt_ids), concrt140, vcomp140, vccorlib140), Bink2Scarlett, XCurl, libHttpClient.GDK,
amd_fidelityfx_loader_xbox / amd_fidelityfx_upscaler_xbox, XboxBvhBuilder. The game exe (`game_scarlett_master.exe`) is still missing.
- `xcl inspect`: after fixes **0 missing imports in all 16**. Added:
  - `XMemVirtualQuery(process, address, info) → BOOL` (signature from the call sites in XboxBvhBuilder, objdump from msys64):
    info = MEMORY_BASIC_INFORMATION + a field @48, bits 0–1 = memory readable by the CPU (message on error:
    "pDesc->SourceAccelerationStructureData must be non-null and point to CPU-readable memory");
  - `XMemGetWorkingSetStatistics(allocator, stats) → FALSE` (the FSR upscaler reads +8 only on success → reports 0);
  - `xaudio2_9.dll` ordinals in `data/stubdb.toml` (from the system DLL; Bink imports #1 = XAudio2Create).
- `crates/xcl/tests/console_dlls_e2e.rs` (`XCL_SAMPLE_DLLS` = folder with the DLLs, the test is skipped without it):
  - the console CRT loads with the XCL loader (HLE/Windows/game DLL imports, TLS, DllMain) and **works**: memcpy, `std::mutex`
    (`_Mtx_*`) with 2 threads (40 000 increments), `_Thrd_hardware_concurrency` / `omp_get_num_procs` /
    `Concurrency::GetProcessorCount` = the number of cores;
  - the other DLLs (Bink, XCurl, libHttpClient.GDK, FFX, XboxBvhBuilder, msvcp140_*, vccorlib140): mapping + TLS + DllMain OK;
  - **the real console BVH builder** (`GetRaytracingAccelerationStructurePrebuildInfo` computes on the CPU; a zeroed object
    = default settings) – the result size matches exactly (17/17 cases)
    `align_1K(256 + N·leaf + max(N−1,1)·128 + 4 B per 64 B of nodes)`: BLAS leaf (triangle) 64 B, TLAS leaf
    (**instance node**) 128 B, box32 128 B, header 256 B, parent pointer 4 B per 64 B (layout as in AMD GPURT / RADV).
- XboxBvhBuilder.dll = the C++ class `XboxBvhBuilder(ID3D12Device11*, bool)`: Build, BuildBatched2X, Copy, GetPrebuildInfo,
  `Rehydrate*` (loading prebuilt BVHs – probably the data from `*_bvh4.rpf`), Set64BitMortonCodes, CpuValidateResultX. Builds on the GPU
  (D3D12SerializeVersionedRootSignature → compute shaders in the DLL).
- **AS format read from the game's RT code** (`CS_RTShadowTrace32TemporalHQ_…_0c84ea88`, listing via `xcl shaders … --example`):
  - AS address from the descriptor (`s_load_dwordx4 [s80:81+0x40]`, words 0–1 >> 8 = base for `image_bvh`), **root = dword @0x30
    of the header** (`s_load_dword s55, [base+48]`); node ids counted from the start of the AS (header included); a root of type 6
    (instance) is handled too;
  - **instance node (type 6, 128 B)**, read from +48: @48 id (24 b) | mask << 24 (test `& 0xE5`), @52 contribution
    (24 b) | flags << 24, @56 BLAS root (30 b; 0x3FFFFFFF = none) | bit 31 = translation only (the shader adds t, does not
    rotate), @60 BLAS address >> 8, @64 world → object as [t.xyz, row 0, row 1, row 2] (o' = M·o + t,
    d' = M·d); BLAS triangle: primitive id @48 (consistent). The shader also reads the BLAS header @52 and @56
    (`[52] >> 6 − [56] >> 2`) – meaning unknown, we write 0. Assumed (the shader does not read them): @0 object → world 3×4
    as in D3D12_RAYTRACING_INSTANCE_DESC, @112..127 zero.
  - `gpu::bvh`: 256 B header with the root @0x30, the root always box32, `build_top(&[Instance])` (TLAS, instance boxes =
    transformed BLAS bounds, inverse matrix in f64), `Bvh::trace_scene(BLAS address lookup, …)` = traversal as
    in the shader, `Bvh::bounds`. Tests: `scene_trace_matches_brute_force` (3 BLAS, 24 instances rotated/scaled/
    translated, some translated only, 3000 rays = brute force over the triangles in world space), `instance_node_layout`
    (every field), sizes ≤ the console model without parent pointers. We do not write parent pointers (location unknown,
    no shader seen so far reads them).
- **FSR – decision (session 5): do not emulate the console FSR4.** The upscaler from the pack is an ML model in fp8
  (`fsr4_model_v07_fp8_…`) with pipelines compiled for the console (parts `ACCC`, `DeserializeComputePipelineStateX`,
  "ACO-cooked", with a fallback "NewBE" path). The game does not need it to run – with the exe: HLE of the FFX API
  (ffxCreateContext/ffxDispatch → simple upscaling with our shader, or hand-off to the PC FSR on RDNA4).
  Descriptor tables and textures – to be determined on the game.
- FSR: the loader has the standard FFX API (ffxCreateContext/Configure/Query/Dispatch/DestroyContext) like the PC
  `amd_fidelityfx_loader_dx12.dll`; the upscaler is **FSR4** (`…fsr4_full_720p1080p_input_sanitize_scarlett_opt2.dll`). Plan:
  alias the loader → FFX DX12 from the PC (FSR4 on RDNA4 = the user's RX 9070); check only with the exe.
- **Bink works:** the real `Bink2Scarlett.dll` under XCL decodes a movie from `movies.rpf` (`bink2_console_dll_decodes_a_game_movie`,
  `XCL_SAMPLE_BK2` = an extracted `.bk2`: BinkOpen → BinkDoFrame/BinkCopyToBuffer(BINKSURFACE32 = 3)/BinkNextFrame; 1280×720,
  60 frames, correct image – checked by eye). Movies in the pack: Bink 2 revision `n`, 720p/1080p, 30 fps, uncompressed in the RPF.
- **`core_bvh4.rpf` = compressed BLASes in the console format** (session 5): RAGE resource = 208 B virtual header + physical
  block (separate zlib streams); physical block +0x100 = the source for `XboxBvhBuilder::Rehydrate…X` (a pure CPU function,
  ignores the device): dword 0 = 0x81020006 (bit 24 = compressed, rehydration clears it), @0x04 1503 (meaning?),
  @0x08 CRC32 (standard table, start 0, no final XOR) over NumDescs + per geometry Flags, IndexFormat, IndexCount,
  VertexCount (only Validate… checks it), @0x0C number of triangle nodes, @0x10 AABB, @0x28 size, @0x30 root,
  @0x34 offset of the first triangle node, @0x38 ?, @0x3C per-axis scale exponents (positions in the BVH = position · 2^(e−127)).
  The blob has no vertices – rehydration reads the game's vertex / index buffers from the geometry descriptions (standard
  D3D12_RAYTRACING_GEOMETRY_DESC; without indices 3p+k). Test `xbox_bvh_builder_rehydrates_game_geometry`
  (`XCL_SAMPLE_BVH4` = the pack, zlib resource @0xa4e26070, synthetic vertex buffer): 4 geometries (644/252/744/4
  triangles), 287 box32, 54 box16, 832 pair nodes – every triangle checked (vertices, I/J rotation, unique primitive,
  primitive ranges without gaps).
- **Node format = AMD GPURT gfx10** (MIT sources downloaded into the scratchpad: `TriangleNode1_0.hlsli`, `IntersectCommon.hlsl`,
  `Common.hlsl`), confirmed on the output of the console builder and on the game's RT code:
  - box16 (type 4) in use alongside box32; triangle = **pair**: v0..v3, type 0 = (v0, v1, v2), type 1 = (v1, v3, v2); @52/@56
    primitive 0/1 (index within the geometry), @60 triangle id (a byte per triangle: bits 1:0 I SRC, 3:2 J SRC, 7 opaque) –
    the hardware rearranges the barycentrics (`SwizzleBarycentrics`);
  - **@48 in the Xbox version: geometry << 8 | second triangle << 7 | opaque** (differs from GPURT): boxes point only to type 0,
    the game's shader, after the type 0 test, checks bit 7 (`v76 = [node+48] >> 7`) and tests `node | 1`; bit 0 goes into the
    opaque/any-hit decision.
  - Fixed: the HLSL emulation `bvh_intersect` (type 1, barycentric rearrangement according to @60), the CPU mirror (`bvh::intersect_node`:
    box16, pairs), `Bvh::trace` (bit 7 → type 1), the builder writes @48 = geometry << 8 | 1, @52 primitive, @56 0xFFFFFFFF,
    @60 = 0x89. Tests: `gpu_triangle_pair_and_box16_nodes` (GPU = CPU, the barycentrics of a rotated triangle 0 and of triangle 1 give
    the exact hit point), the old BVH tests with triangle id 9.
- **XboxBvhBuilder on the XCL device (probe, `XCL_PROBE_BVH_DEVICE=1`, test `xbox_bvh_builder_on_xcl_device_probe`)**:
  constructor on the real backend (RX 9070) → `D3D12SerializeVersionedRootSignature` with **console-only flags
  0x3000000** (Windows: "Unsupported bit-flag set") – HLE strips the bits outside 0xFFF and logs → root signature created →
  then the process exits with 0xC0000409 (fail-fast, no SEH exception – suspicion: a device method writes a larger
  structure than the console one / /GS cookie). `QueryInterface(0742a90b-c387-483f-b946-30a7e4e61458)` in the log does not come
  from the builder (more likely from the PC runtime). **Next step:** trace the device vtable calls from the builder (which slot, arguments).
- `OutputDebugStringA/W` (kernelx HLE) → XCL log `[debug] …` (console DLLs and games write a lot there; that is how the XDXR error surfaced).
- **The XboxBvhBuilder constructor goes through completely on the XCL device** (probe `XCL_PROBE_BVH_DEVICE=1`):
  - `XCL_D3D12_TRACE=1` (NEW, `guard::make_tracer`): every public vtable method called from console code → log
    `D3D12 trace: Iface::Method (#slot) ← module+offset`. That is how the sequence CreateRootSignature → SetName → slot 62 came out.
  - **The console device vtable differs from the PC one:** slot 62 on the console = `CreateComputePipelineState(desc, NULL, NULL,
    riid, ppv)` (two extra arguments), on PC = CreateStateObject. `guard::make_split` (NEW): a call from the game image →
    console version, from the runtime → PC; table `real::console_slots` (for now only device #62).
  - Compute PSO with a console CS → `pso::console_to_dxil(…, Stage::Compute)` (cs_6_6) + XCL ABI root signature
    (`real::translated_compute_pso`). Console bytecode must not be passed to the driver: the AMD compiler (amdxc64.dll)
    crashes on DXIL with NOBC bitcode instead of rejecting it – a failed translation → a replacement shader right away.
  - New instructions in the translator (from the builder's shaders, they were not in `shaders.rpf`): `global/flat_atomic_*` (add, sub, and,
    or, xor, swap, s/umin, s/umax, cmpswap, **fmin/fmax via a CAS loop**), `buffer_load_ubyte/sbyte/ushort/sshort`,
    `buffer_store_byte/short`, `ds_write_b8/b16`, `ds_read_u8/i8/u16/i16` (atomic AND/OR on groupshared). Result: 53/53
    builder shaders translated, 0 replacements. **No unit tests for these instructions** – still to be written.
  - **INCIDENT:** the builder's largest shader (1589 instructions, HLSL < 256 KB → full DXC optimization) compiled for
    ~6 min and exhausted 38 GB of committed memory (32 GB RAM + 7 GB page file) – Windows killed the user's
    session (event log: "The system is low on virtual memory", 23:27). The `-O0` threshold in `dxc::compile` lowered from 256 KB
    to 48 KB of HLSL. The DXIL of this shader (314 KB) is already in the DXC cache. Before further probes with large shaders: a
    memory limit for the test process (Job Object), or DXC compilation in a separate process.
  - Tests of the new instructions: `gpu_global_atomics_and_sub_dword_access` (64 lanes at the same addresses: add with GLC,
    sub, fmax/fmin from ±inf, smax; short/byte in a buffer and in LDS with neighbors in the same dword, neighbor read after a barrier).
  - Tests with the console DLLs run under an **8 GB memory limit** (Job Object in `setup`, test `memory_cap_is_active`).

### Root argument binding on the console (determined on 52 shaders of XboxBvhBuilder.dll)
- **s[0:1] = pointer (40 bit, s1 & 0xFF) to the root argument array**: parameters packed in root signature order,
  root constant = 4 B, root descriptor (CBV/SRV/UAV) = 8 B GPU address, no alignment (a UAV after 19 constants lies at 76).
  The shader builds the V# from the address by itself (num_records −1). Checked automatically: 123 reads from the array in 50 shaders – all
  on parameter boundaries (script `check_layout.py` in the scratchpad; shaders extracted from the DLL by their DXBC signature).
  Descriptor tables – not seen yet (the builder does not use them). Group ids after the user SGPRs (s4 = TGID.x with 4 user SGPRs).
- For XCL: on a dispatch with a translated PSO – assemble this array in UMA from the remembered root arguments, s0/s1 = its address
  in `io`, the rest as in Compute mode.
- **Corrected with the console's own compiler** (`tests/console_shaders.rs`, `print_root_argument_placement`): s0…s3 are
  the driver's; the parameters go into user SGPRs **s4…s15 in order while each fits whole** (descriptor 2, constant 1 each,
  descriptor table 1 = 32-bit address of its first descriptor, high half 0); from the first one that does not fit, it and
  all after it go to the table at s[0:1], packed from offset 0 (19 / 20 constants first → everything in the table, as
  in the builder). E.g. 11 constants + UAV: constants s4…s14, UAV at table +0; UAV + 11 constants: UAV s[4:5],
  constants at table +0. Group ids follow the user SGPRs. `console_list::Layout::inline` / `user_data` do this for
  console-list dispatches.

### Console shader compiler as a test generator (`tests/console_shaders.rs`, `XCL_SAMPLE_DXC`)
- The console's `dxcompiler_xs.dll` loads on the PC with plain LoadLibrary (only API-set imports); its back end
  `xbsc_xs.dll` (and `newbe_xs.dll`) is loaded by name at Compile, so the test loads them first from the same directory.
  Its output is a full console container (DXIL + XDAT → XBCH → FSSM with the RDNA2 code, RTS0). Nothing of it goes into
  the repo; the test runs only with `XCL_SAMPLE_DXC` set.
- `console_compiled_kernels_match_the_pc_compile`: HLSL kernels (integers, conversions, float math, per-lane loops and
  branches, groupshared + barriers, atomics, 64-bit integers) through the console compiler → XCL's translation → PC GPU,
  next to the PC compile of the same source: the same memory. Found and fixed: the root argument placement above, and
  `v_mul_hi_u32` / `s_mul_hi_u32` as `(uint64)a * b >> 32` – DXC turns it (compared with 0 in the console's division
  code) into umul.with.overflow and DXIL validation fails; now from 16-bit parts (`mul_hi_u32` in the prelude).
- The console compiler multiplies by constants with `v_mul_legacy_f32` where the HLSL allows it (DXIL fast math, no
  signed zeros): -0 × c = +0 on the console, -0 on the PC – the conversions kernel compares up to the sign of zero.
- Not yet: three kernels (local arrays indexed per lane, switch in nested loops, 16-bit types) made the run hang –
  removed from the list until it is known which step (console compile, translation, DXC or the GPU) hangs.

### D3D12.X: the command list is partly "compiled into" the title's code (probe `XCL_PROBE_BVH_RECORD=1`)
- Recording `BuildRaytracingAccelerationStructure` on a list from the XCL device: the builder calls **console list method #148**
  = `AllocateUploadMemory(size, alignment) → CPU pointer` (for us the guard returns 0 → "D3D12 XDXR Error: Failed to
  allocate upload buffer" → C++ exception), and then **writes directly into the internals of the list object**: +0x18 write pointer,
  +0x30 buffer end (appends a 28 B record), +0x140 flag, +0x544 root signature state; slot 33 here is likely
  SetComputeRootSignature (on PC #29). There are no PM4 headers in the code – the records are in a D3D12.X software format,
  turned into GPU commands by the console driver at Close/Execute.
- **Architectural conclusion (also applies to the game):** replacing the vtable on a real PC object is not enough – console code
  would write all over the internals of the PC driver's object. Console code must get **its own XCL list objects** with the console memory
  layout (a vtable forwarding to the PC list + a "shadow" area for the fields and the record buffer), and XCL must **interpret the records**
  at Close/ExecuteCommandLists. The field layout and record format are to be determined from the inline code (first the builder: compute only,
  then the game). This is probably the largest remaining piece of rendering work.

### D3D12.X command list format (compute) – determined on XboxBvhBuilder.dll
Method: a fake list (`xbox_bvh_builder_records_into_a_fake_console_list`, `XCL_PROBE_BVH_FAKELIST=1`: a vtable of 600 thunks
logging the slot and arguments, #148 allocates from UMA, +0x18/+0x30 = record buffer) – the builder recorded the whole BLAS build without
crashing; plus a static review of all 141 write sites (`record_writers.py` in the scratchpad).
- **List object fields:** +0x00 vtable, +0x18 write pointer, +0x28 limit for dispatch records, +0x30 buffer end,
  +0x140 "dirty" flag (set when the header has bits 31:30 = 0), +0x141 a bit from the pipeline (bit 20 of word +0x20 of the record),
  +0x143 flag, **+0x344 + 4·i = header of root parameter i** (written by a method – for us #33, most likely
  SetComputeRootSignature – the inline code only copies it), +0x544 current pipeline (16 B).
- **Records** (no common frame; the length follows from the header / kind):
  - pipeline 28 B (×34): a copy of +0x10..+0x2B of the builder's "info" object = [PSO pointer (our adopted object)][pointer to
    the D3D12Core object][8 B of flags][4 B]; stands in for slot 25;
  - root parameter i: header = [+0x344 + 4i] (+ (n−1) << 16 for multiple constants), then: descriptor = 64-bit address
    (12 B, stands in for slot 43/45), one constant (8 B, slot 37), n constants (4 + 4n B, slot 39: (i, n, data, offset));
  - Dispatch: `0x4002C400` (or `(0x1000BA00 | flag) << 2`) + x, y, z = 16 B, stands in for slot 14 (= PC Dispatch);
  - `0x4003CC00`, 20 B, stands in for slot 80 (probably indirect dispatch);
  - methods: #148 AllocateUploadMemory(size, alignment) → CPU pointer (= GPU address in UMA), #146 barrier
    (1, pointer to a description), #33 (object pointer – root signature / PSO), #14 Dispatch.
- Type 3 PM4 headers do not appear as constants – the parameter headers come from the list, so **XCL can put its own
  tags there** (e.g. 0xC0000000 | i << 8 | tag) and read the stream unambiguously; the pipeline record can be recognized by the known
  PSO pointer.

### DONE: console command list (compute) – the console BVH builder works on the PC GPU
`hle/d3d12x/console_list.rs` (NEW, enabled with `XCL_CONSOLE_LISTS=1`): test `xbox_bvh_builder_builds_on_the_gpu`
(`XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_GPU=1`, ~1 min with an empty DXC cache, then seconds) – **the real XboxBvhBuilder.dll
builds a BLAS from 64 random triangles on the RX 9070 through XCL**: the D3D12.X inline code records into a list in the console layout,
`console_list::close` turns the records into PC commands (10 dispatches of translated console shaders, 9 barriers, 0
unknown records), the result read with `bvh::Bvh::trace` = brute force on 500 rays.
- List object: a vtable of 600 slots (known: 0–2, 14 Dispatch, 25 SetPipelineState, 33 SetComputeRootSignature,
  37/39 constants, 41/43/45 descriptors, 146 barrier, 148 AllocateUploadMemory from UMA; the rest – logging stubs), fields as
  in the format above, records in 1 MB chunks (4 KB of slack past the limit, because the inline code checks `write < end` and only then
  writes).
- Root parameter headers: `0xC0000000 | i << 10`; the inline code adds **constant offset · 4** (bits 9:2) and
  **(count − 1) << 16**; our own tags `0x8C0FE0xx` (pipeline, root signature, barrier); dispatch `0x4002C400` /
  `0x4002E800|4`.
- PSO for console code: an object in the console shape (`wrap_pso`, at +0x10 a tag + a pointer to itself, methods 3–8 →
  the PC PSO); created on the console path (device slot 62) when console lists are enabled. The root argument layout
  is remembered at CreateRootSignature (`note_root_signature`, RTS0 parser).
- Dispatch: root argument array (layout from RTS0) in UMA, `io[0..1]` = its address, XCL ABI root signature, u0 io / u1 res /
  u2 `mem` = the whole UMA buffer, translated PSO.
- **Important: `mem` in Compute mode is `globallycoherent`** – without it the bottom-up tree build (atomics + reading what
  other groups wrote) produced wrong child pointers (on the console the code uses GLC / cache invalidation). Graphics gets an ordinary view.
- **Hooked up automatically** (when `XCL_CONSOLE_LISTS=1`): device #12 CreateCommandList – split thunk: console code
  gets a console list (the initial PSO unwrapped to the PC one), XCL code gets a PC list; list: #9 Close (interpretation),
  #10 Reset, #3–8 → PC list; the queue's ExecuteCommandLists replaces console lists with PC ones; the recording's UMA memory is freed
  on Reset of its allocator (`alloc_reset` → `console_list::allocator_reset`). The test does 2 rounds through the vtable (64 and 200
  triangles, allocator and list Reset between them) – both agree with brute force.
- Not yet: record `0x4003CC00` (indirect dispatch, slot 80), the #146 barrier description (currently always a global
  UAV barrier), descriptor tables, graphics (draw records – to be determined on the game), freeing UMA memory after execution
  (the list holds the blocks until Release).

### TLAS on the GPU – WORKS (session 5): instance node format confirmed on the console's output
- `xbox_bvh_builder_builds_a_tlas_on_the_gpu` (`XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_TLAS=1`): the console builder builds on the GPU
  2 BLAS + a TLAS of 16 instances (rotated/scaled and translated only), `bvh::Bvh::trace_scene` (instance layout from the game's RT
  code: @48 id|mask, @56 BLAS root | "translation only" bit, @60 BLAS address >> 8, @64 world→object [t, rows])
  = brute force in world space on 800 rays. What unblocked it was **removing branches from the generated HLSL** (below).
- **Branch-free code (the most important performance change in the translator):** in HLSL 2021 `?:`, `&&`, `||` on scalars short-circuit,
  so each of them was a real branch – `lanebit` (lane activity, at every instruction), `scc = (…) ? 1u : 0u`,
  `v_cndmask`, DPP/permlane, buffer range conditions. Now `select()`/`and()`/`or()`/`(uint)(bool)` (memory reads/writes
  still behind an `if`), VGPR write `v = select(active, value, v)`, the VALU path without a named temporary variable.
  Effect on the builder's shaders: 15 720 instructions – DXC `-O3` 5.8 s, 585 KB DXIL (before: > 150 s / driver:
  out of memory); the shader from yesterday's crash (1 589 instr.) – 0.3 s, 48 KB (before: ~6 min and > 30 GB).
  **DXC optimizes everything again** (`-O0` only for > 8 MB of HLSL). A sample of 64 game shaders: 0 errors; median 0.29 s, p90 ~11 s.

### FSR4 for learning about textures (session 5, in progress)
- Probe `fsr4_console_upscaler_creates_a_context_probe` (`XCL_PROBE_FSR=1`): `ffxCreateContext` with
  `ffxCreateContextDescUpscale` (type 0x10000) + a backend description (type **0x1000002**, device at +0x10) – **the context
  is created on the XCL device**. The only configuration in the pack (FSR says so itself via OutputDebugString): render 1280×720
  (or 1920×1080), output 1920×1080, flags HDR | DEPTH_INVERTED | AUTO_EXPOSURE (1 | 8 | 32).
- More console differences (table `real::console_slots`):
  - device **#76 = CreateCommittedResource** (like PC #27); **console resource description**: Layout (+44) = **0x101** (2D texture
    1920×1080 R8G8B8A8, RT|UAV), +52 = 3, description ≥ 64 B – console layouts → UNKNOWN on PC (log in
    `dev_create_committed_resource_x`);
  - descriptor heap **#9 / #10 (GetCPU/GPUDescriptorHandleForHeapStart) return the handle in rax** (PC: a hidden pointer).
- Context creation: 14 resources, 16 UAVs, 10 SRVs, Map/Unmap, GetCopyableFootprints – all through methods (no inline code).
- **`ffxDispatch` on a console list (`XCL_PROBE_FSR_DISPATCH=1`, 1280×720 inputs: RGBA16F color, R32F depth, RG16F
  vectors; 1920×1080 RGBA16F UAV output) – 768 words, all recognized**: 20 dispatches, 24 barriers, heaps, tables.
- **Descriptor tables – format known:**
  - record: parameter header + **1 word = the 32-bit GPU address of the first descriptor** (root CBV: header + 2 words);
    in the packed argument array a table takes **4 B** (`Param::Table`);
  - shader (RDNA2 code from ACCC, see below): `s_mov_b32 s10, s5; s_mov_b32 s11, 0; s_load_dwordx4 s[16:19], s[10:11], 0x240` –
    **upper half of the table address = 0**, descriptor = **32 B per slot** (T# 8 dw, V# 4 dw in a 32 B slot; SRV×21 = 0x2A0);
  - root CBV: the shader builds the V# by itself (`s_mov_b32 s10, -1; s_mov_b32 s11, 0x31016fac` = DST_SEL xyzw + format 32);
  - user SGPRs (14, RSRC2): s4 = table 0 (UAV), s5 = table 1 (SRV), s[6:7], s[8:9]… = root CBVs (the same packed layout
    from s4 on; s[0:3] – unknown), s14/s15 = TGID x/y.
- **Console heaps in XCL (`hle/d3d12x/console_heap.rs`, NEW)** – a CBV/SRV/UAV heap created by console code
  (#14) gets 32 B slots in UMA (CPU handle = GPU handle = address in UMA → the low 32 bits = offset in `mem`) + a PC heap
  with 2 descriptors per slot (slot k → SRV 2k, UAV 2k+1). #15 returns 32, #18/#19 create a PC descriptor and write into the slot
  a T# in the XCL ABI (`desc::xcl_image_for_view`: word 0 = 2k, format/size/type/mips as in the T#; a 2D UAV also gets an SRV at 2k,
  because RDNA does not distinguish them). List: #32 SetDescriptorHeaps, #26 ResourceBarrier (transitions forwarded to PC), #15
  CopyBufferRegion. The probe test checks that the tables from the stream point to the slots with the input textures (formats 71/22/29).
  - **Buffers in UMA (done):** a buffer created by console code (#76, dimension BUFFER) = a placed resource
    (CreatePlacedResource) on the UMA heap (`UMA_HEAP`, flag ALLOW_CROSS_ADAPTER, 64 KB alignment); resource #11
    GetGPUVirtualAddress (console) returns the address in UMA (= the CPU address, low 32 bits = offset in `mem`); **Map on PC returns
    the same address** (checked once, log); Release (all callers) frees the block. Buffer views (SRV/UAV) get a V#
    (`desc::buffer_words`: raw → stride 0, num_records in bytes; structured/typed → stride, num_records in elements,
    GFX10 format). FSR: 3 buffers in UMA, all V#s in the tables point into UMA.
    - Note for the future: all console buffers (DEFAULT ones too) live in system memory (UMA) – the GPU reads them over
      PCIe; for large game buffers this may be slow.
  - Missing: CBV via #17 (not seen), sampler heaps, freeing slots on heap Release, CopyDescriptors.
- **FSR4 PSO = ACCC ("FSPX" container)** – CreateComputePipelineState (NewBE) receives an SFI0/ISG1/OSG1/PSV0/RTS0/HASH/ACCC blob
  without DXIL. FSPX: header 'FSPX', version 1, n entries, size; 16 B entries (type, offset, size, 0): **0** (0x1C B:
  numthreads u16×3 @+0x0C (256,1,1), RSRC1 @+0x14, RSRC2 @+0x18), **1** (0x14, hash), **6 = RDNA2 code** (the first 0x24 B =
  a header with the hash), **7** (resource declarations: 32 B records of type 0x41/0x42/0x43, offset in dwords = slot·8).
  The translator does not read them yet → the dispatches are skipped. Helper: `raw_rdna2_listing` (`XCL_RAW=<code>`).
- List slots recognized from the FSR code: **#53 / #54 = ClearUnorderedAccessViewUint / Float** (GPU handle, CPU handle, resource,
  values[4], 0, NULL; FSR picks #54 for a target in format 0x12) and **#85 = CopyTextureRegion** (destination: COPY_LOCATION
  SUBRESOURCE_INDEX, source: PLACED_FOOTPRINT, x/y/z = 0, box NULL; PC #16 – the console has this method elsewhere).
  Records → PC at Close (console handles → the slot's PC descriptors: UAV 2k+1). FSR: 2 clears, 4 copies, 0 skipped.
- **Test title** calls heap #9 the way console code does (handle in rax) – previously through windows-rs (PC ABI), and it had been failing since 7919cdc.
  XAudio2 test: XMA2 lower bound 80 → 70 ms (10 ms quantum jitter, once it came out at 79 ms).

### (archive) TLAS on the GPU – blocked by the size of a single shader
- Test `xbox_bvh_builder_builds_a_tlas_on_the_gpu` (`XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_TLAS=1`): 2 BLAS + 16 instances
  (some translated only), `trace_scene` vs brute force in world space. The BLASes build, the TLAS comes out empty: 1 dispatch
  skipped, because **the AMD driver rejects the PSO with the translated CS of 15 720 instructions (0x8007000E, out of memory)** – even
  with a 20 GB limit; the second large one (6 302 instr.) goes through at 20 GB, not at 8 GB. DXC `-O1`/`-O3` on it > 150 s.
- Translator improvements made along the way (all tests green, BLAS build on the GPU still 500/500):
  - **structured control flow**: a fully structured program (forward jumps nested, if/else via
    `s_branch`, loops with a single back jump at the end, `s_endpgm` at the end) → plain `if`/`for(;;)` without `pc`;
    otherwise a **hybrid**: the longest single-entry/single-exit stretches structured under a single `pc` guard
    (#19: 628 → 93 guards). `XCL_RECOMP_NO_STRUCTURED=1` disables it, `XCL_RECOMP_WHY=1` tells why it is not structured;
  - `global_atomic_fmin/fmax` as a single integer atomic (positive → int, negative → uint reversed) instead of a CAS loop;
  - VGPR write `v = select(active, value, v)` instead of `if` (linear code);
  - `pso::console_to_hlsl` + helper tests `blob_hlsl_shape` / `blob_dxc_cost` (`XCL_BLOB=…`).
- **To do:** leaner HLSL (fewer temporaries: at `-O0` each one is a separate memory cell, tens of
  thousands in this shader) or a different way of splitting compilation; the same will speed up compiling large game shaders.

### Plan: console command list in XCL (next sessions)
1. `hle/d3d12x/console_list.rs`: an object with the console layout (fields as above, record buffer, parameter headers), wrapping
   a PC list; console vtable: 14 Dispatch, 25 SetPipelineState, 33 SetComputeRootSignature (writes the tags, remembers the
   RS layout), 37/39/43/45 substitute parameter methods (append the same records), 80, 146 barrier, 148 AllocateUploadMemory
   (UMA), Close/Reset; the rest – logging stubs. CreateCommandList from console code → this object.
2. PSO: at creation remember PSO → (translated?, RS layout from RTS0).
3. ExecuteCommandLists: interpretation of the records → PC list: on every dispatch a root argument array (layout from the section
   "Root argument binding") in UMA, `io` = [address lo, hi, …], ABI root signature + io/res/mem, translated PSO, Dispatch;
   barriers → UAV barrier on `mem`.
4. Test: the real XboxBvhBuilder builds a BLAS on the GPU through this list → the result read with `bvh::Bvh::trace` = brute force.

## In progress (unfinished)
Nothing open.

## PresentX, frame pacing, the console command list layout (session 5, 2026-11-21; Fortnite is the next test title)
Slots and semantics as observed on Fortnite (UE RHI call sites); XCL's own evidence agrees where it has any.
- **PresentX = command queue #30** `(planeCount, planes, parameters)`, plane 0 `{ u64 frame token; u32 resource count @8;
  ID3D12Resource** @0x10 }` → `present::present_x` presents that resource at once (S_OK always) and marks the token
  flipped; the back-buffer guessing only runs until a title's first PresentX.
- **Frame clock = device #78–#81** (`d3d12x/frame.rs`): SetFrameIntervalX(output, µs, periodicity, flags) → S_FALSE
  (16 667 / 33 333 / 50 000 / 66 667 µs; 8 333 / 25 000 only on a ≥ 119 Hz display, else E_INVALIDARG);
  ScheduleFrameEventX(type 0 origin / 1 / 2, offset µs, ancillary `{ u32 count; HANDLE* @8; i32* signaled @0x10 }`,
  flags 1 = unschedule) → S_FALSE, 0x10000000 when it starts the clock; WaitFrameEventX(type, timeout ms, ancillary,
  flags 1 wait all / 2 token on timeout, token in/out) → S_OK / S_FALSE (timeout), a new 48-bit token at each origin;
  GetFrameStatisticsX(token, type, count, stats `{ u32 type; i64 origin @0x10; i64 flip @0x18 }`). A clock thread
  (high-resolution waitable timer + spin) releases the origin semaphore / sets the events and the ancillary handles;
  not locked to the host vblank.
- **ID3D12GraphicsCommandList console layout** (`real::console_remap`): 0–27 and 56–59 as on PC; 28–31 = PC 52–55
  (BeginQuery, EndQuery, ResolveQueryData, SetPredication); 32–55 = PC 28–51 (SetDescriptorHeaps … DiscardResource, so
  SetGraphicsRootSignature #34, OMSetRenderTargets #50, ClearRenderTargetView #52, ClearUnorderedAccessViewUint #53 –
  as FSR4 calls it); List1 at 104–109 (PC 60–65); shading rate 124 / 125 (PC 77 / 78). Title calls only (make_split);
  the console-only methods in between (depth bounds 68, KickoffX 93, depth bias 94, …) stay guarded and are reported.
- Console slots past a class's public PC methods now work (inside the runtime's internal entries: title → console
  method, runtime → its own). The test title calls the console layout (#34 / #50 / #52) and runs frame clock + PresentX.
- Next with the exe: the console-only command-list methods Fortnite calls (report.json lists them), title heaps /
  placed resources at title addresses, occlusion queries (the console allows nesting them, PC does not).

## The console device's own methods (session 5, 2026-11-21, `d3d12x/device_x.rs`)
Slots as observed on Fortnite. From #44 on the console device's vtable holds its extensions, so a title call there
must not run the PC method at that index (#44 is PC CreatePipelineLibrary, #48 OpenExistingHeapFromAddress…):
- **#44 CreatePlacedResourceX**(address, desc, state, clear, riid, out): a buffer at a 64 KiB boundary in unified
  memory is placed in that window's heap – it *is* the title's memory (GPU VA = the address; Release leaves the title's
  allocation alone). Textures and other buffers get memory of their own (logged): the console tiling of texels the title
  writes there is not translated yet.
- **#48 CreateCommandQueueX** / #8, #9, **#73 CreateCommandListX**(params `{ type @0; node mask @8 }`, …) / #12: the
  console DMA type 0x10 is the PC copy type, the DMA list IID is created as a graphics list, console priorities → PC.
- **#60 / #62** graphics / compute PSO X (PC desc + extension count / pointer, extensions not modelled), **#86**
  CreatePipelineState (streams); **#61 CreateSamplerX** (0x6C-byte desc → anisotropic PC sampler); **#38**
  GetCopyableFootprints retried with the PC desc when the console desc is rejected; **#41** command signatures (DispatchX
  105 → Dispatch, root signature only for root arguments); descriptor heap type 4 = CBV / SRV / UAV.
- **#52** SetDebugFlagsX → S_OK, **#56 / #66** SerializePipelineStateX → E_ACCESSDENIED (as the release runtime),
  **#57 / #67** DeserializePipelineStateX → E_NOTIMPL (a console pipeline cache is the console driver's format with
  console shaders: `state = { root signature @0; parts { pointer; length }[] @8 }`; nothing decodes it yet – the format
  needs blobs from the exe), **#74 / #75** page pools (index registry). Every other device slot from #44: a logged stub
  answering S_OK.
- **#55 CreateDerivedPipelineStateX**(base, count, patches, riid, out): patches `{ u32 part; payload }` at a 0x14C
  stride replace parts of the base's desc – 0 depth-stencil (0x34 bytes), 1 rasterizer (0x2C), 2 blend (0x148),
  3 sample mask; part 5 (0x30 bytes, meaning unknown) is ignored, others fail. Title graphics pipelines (#10 / #60 /
  derived ones) keep their desc with the input layout copied; the shaders stay the title's and are checked by hash
  before deriving (none of that for pipelines with stream output).
- The test title copies a buffer placed at its own `XMemAlloc` graphics memory through a DMA queue / allocator / list
  created through #48 / #9 / #73, checks the remaining slots, and draws with a pipeline from #60 and one derived from it
  (#55, blend patch: green only → cyan over the blue clear).
- Not yet: deserializing console pipeline caches, placed textures at title addresses (detiling).

## The console command list's own methods (session 5, 2026-11-21, `d3d12x/list_x.rs`)
From #60 on the console list holds its extensions (besides the moved PC methods, `real::console_remap`); a title
call there no longer runs the PC method at that index (#60 was AtomicCopyBufferUINT, #66 WriteBufferImmediate…).
Console GPU addresses in unified memory are the title's own: memory operations go to that window's buffer
(`real::uma_window_buffer`), wrapped in COMMON → COPY_DEST → COMMON transitions (the window buffers stay in COMMON).
- **#60 CopyMemoryToMemoryX**(dst, src, size, flags) → CopyBufferRegion (one window: the buffer is source and
  destination, overlapping ranges are refused); **#61–#63 FillMemoryWith{32,64,128}BitValueX**(va, size, value) → a
  copy from an upload buffer repeating the value (cached per value, up to 256 MiB).
- **#69 / #70 Write{32,64}BitValueBottomOfPipeX**(va, value, flush, flags), **#88 / #89 …TopOfPipeX**(va, value,
  flags), **#90 Write32BitValueBottomOfShaderX** → WriteBufferImmediate (MARKER_OUT / MARKER_IN); with a timestamp
  flag (bottom 1 / 2, top 4 / 8) the GPU timestamp is resolved there from a timestamp query (PC ticks).
- **#78 DispatchX** → Dispatch; **#81 DrawIndexedX**(index address, format, count, flags) → that index buffer for one
  DrawIndexedInstanced, then the bound one again; **#47 IASetIndexBuffer** gives the PC the window buffer's address
  for a title address; **#84 CopyResourceX** → CopyResource (in place: nothing); **#85 SetPredicationBufferX**
  `{ u64 va; u32 op }`; **#68** depth bounds.
- Not modelled (logged, no-op): Wait32 / 64BitValueX #71 / #72, GDS #76 / #77, KickoffX #93 (TomBox splits the list
  and submits the first part), depth bias X #94 (needs pipelines with dynamic depth bias), hints (HiStencil, MSAA,
  compression, limits, prefetch).
- **Nested occlusion queries** (#28 / #29 / #30; the console allows them, the PC one active per list): when one begins
  while another is active, the outer one ends there on the PC and resolves as visible (0x10000 samples, binary 1;
  written over the resolved data, MARKER_OUT); the inner one is a real query – Fortnite nests its per-object queries in
  a dummy outer one. List Reset (#10) forgets the active queries.
- The test title writes / fills / copies its own graphics memory through #60–#89, takes a GPU timestamp, draws
  through DrawIndexedX and #47 with indices in its memory, and nests occlusion queries (inner = a single draw's).

## Fortnite's console imports (session 5, 2026-11-21; from TomBox's library map, Fortnite XSX 2.0.1461.7919)
The EXE (+ EOSSDK) imports 31 functions from console libraries: MFPlat `MFResetDXGIDeviceManagerX`; xmem `XMemAlloc`,
`XMemFree`, `XMemAllocDefault`, `XMemFreeDefault`, `XMemVirtualAlloc`, `XMemGetWorkingSetStatistics`,
`XMemTransferMemory`, `XMemAllocatePhysicalPages`, `XMemMapPhysicalPages`, `XMemFreePhysicalPages`; xg_xs
`XGCreateTextureComputer`; d3d12_xs `D3D12XboxCreateDevice`, `D3D12CreateDevice`, `IID_ID3D12Device` (data),
`D3D12SerializeRootSignature`, `D3D12SerializeVersionedRootSignature`, `D3D12XboxSetProcessDebugFlags`; XCRDAPI
`DStorageGetFactory`; dxcompiler_xs `DxcCreateInstance`; AcpHal `AcpHalCreate`, `ApuAlloc`, `ApuFree`, `ApuCreateHeap`,
`ApuHeapGetState`, `AcpHalAllocateShapeContexts`.
- Now in XCL: `XMemGetWorkingSetStatistics` (0x78 bytes, +0 title limit, +8 used – Fortnite sizes its pools from them;
  it returned FALSE), `XMemTransferMemory` (TRUE), `XMem{Allocate,Map,Free}PhysicalPages` on the title physical pages
  (64 KiB, contiguous; a chosen address / scattered pages are still refused – Fortnite's texture streaming maps pages
  into reserved address space, so that is next), `IID_ID3D12Device` as data, `dxcompiler_xs.dll` = the PC DXC.
- Missing (each a project): `XGCreateTextureComputer` (the console texture layout computer – TomBox reimplements
  `xg_xs`), `DStorageGetFactory` (console DirectStorage – Fortnite's IO), AcpHal / SHAPE / xapu / XMA (the hardware audio
  path Fortnite plays through).
- Nested occlusion queries: Fortnite nests its per-object queries in a dummy outer one, so the inner ones are now the
  real queries and the outer one ends early and resolves as visible (it was the other way round).
- Console DLLs every analysed title (Fortnite, XDefiant, Terraria, Trackmania – TomBox's library map) imports from, which
  XCL did not route (their imports became stubs returning 0): `minuserx.dll` (the console user32: CreateWindowExW,
  RegisterClassW, PeekMessageW, DefWindowProcW…) → the PC user32; `kernel32legacy.dll` (Fortnite: GetProcessAffinityMask,
  SetThreadAffinityMask – the title sees the PC's processors), `advapi32legacy.dll` (RegOpenKeyW) → kernelx;
  `minappmodelx.dll` → kernelx's Register/UnregisterApp{State,Constrained}ChangeNotification (a registration, the
  callbacks never come: titles are not suspended or constrained here).
- **DirectStorage (`xcrdapi.dll`, `hle/dstorage.rs`)** – Fortnite (Series) reads all its data through it.
  `DStorageGetFactory` → IDStorageFactoryX (console vtable order: CreateQueue, OpenFile, CreateStatusArray,
  SetCpuAffinity, SetDebugFlags, SetStagingBufferSize, …), queue desc 0x18 (source type @0, capacity u16 @8, priority i8
  @10, name @0x10), request 0x48 (options @0 – compression bits 0–7: 0 none / 1 zlib, bit 22 memory source –,
  destination @8, size @0x18, source @0x20, file offset @0x28, source size @0x30, intermediate size @0x34, tag @0x38,
  name @0x40), RetrieveErrorRecord 0x290, Query 0x1C. A worker thread per queue runs the submitted requests in order:
  file reads through the VFS (as the title's own CreateFileW), memory copies, zlib (miniz_oxide); a status entry gets
  the first failure since the previous one, a signal sets the D3D12 fence from the CPU; a half-full queue submits by
  itself. BCPack (and other console formats) are refused; the unlisted interface slots are logged E_NOTIMPL. The test
  title reads the start of its own exe through it and waits on the fence. Not yet: cancelling submitted requests,
  multi-stage (intermediate-size) decompression, destinations that are not plain memory.
- **Texture layout library (`xg_xs.dll`, `hle/xg.rs`)** – Fortnite imports `XGCreateTextureComputer`. XCL's own GFX10
  model behind the console's interface, checked against the console's own library (`tests/xg_oracle.rs`,
  `XCL_SAMPLE_XG` = the user's console `xg_xs.dll`, 10.0.22000; nothing of it is in the repo):
  - 29 700 descs (15 formats, 15 sizes incl. odd ones, mip counts, arrays, every explicit mode, the library's own
    choice for textures and UAVs): every XG_RESOURCE_LAYOUT byte the same or both refuse; texel offsets of every mode ×
    element size match (`texel_offsets_and_the_console_pipe_configuration`); 162 subresources tile to the same bytes.
  - What it took (the console's rules, now in `gpu::tiling` – the TTPF retail check still fits 909 / 912): levels are
    padded from mip 0 halved rounding up (also linear, and for the mip tail decision); Z modes of 1 / 2-byte elements
    have a 64 × 128 tail; the XOR modes take pipe bits from coordinates above the block and from the array slice (z of
    the 2D patterns) – `Layout` / `Chain::detile_level` / `tile_level` now have them.
  - **The console's pipe configuration** (`texture::console_pipes`): 32 pipes / 8 packers by default (Series X),
    8 / 4 with XGSetHardwareVersion(4) (taken to be Series S – then mip fields +0x50 / +0x58 are 0xE000 / 0x2000 instead
    of 0x8000 / 0x8000 for the 64KB Z/S/D/R_X modes); it follows the reported device type.
  - Modes the library takes for 2D: 1 2 5 6 9 10 17 18 21 22 25 26 27, 24 (and VAR 28) for ≤ 8-byte non-BC elements,
    VAR 31; `Layout` 0 / 2 = its choice (64KB_D for 2, 4KB_S, UAVs 4KB_S_X up to 64 KB else 64KB_R_X). Not modelled
    (E_NOTIMPL): MSAA, 1D / 3D, VAR modes.
  - **Planes** (`xg::Texture::planes`, checked by `render_target_and_depth_layouts_match`, 1 872 descs): usage 1 color,
    2 CMASK, 4 HTILE, 5 / 6 depth / stencil of a depth-stencil format, 7 / 8 the same in a depth buffer. A render target
    of one level gets a CMASK after its texels (align 0x2000, size = pixels padded to 1024 × 1024 / 128 per slice; not
    for layout 2 or the _T modes); a depth buffer (DS flag, one level) is HTILE first (align 64 KB, same padding / 16),
    then depth, then stencil (1-byte elements), all in mode 24; RT + DS of a color or D32 format is a render target in
    mode 24 (CMASK, no HTILE). Left empty (size 0, offset 0): the texels of a one-level BCn RT / DS and the depth of a
    DS in a color format – with DS the whole resource then reports size 0. D32_S8X24 picks its optimal mode by 64 bits a
    pixel. Title textures that have metadata planes are not mirrored (`title_textures`: GPU-owned).
- **Textures in title memory** (`d3d12x/title_textures.rs`): CreatePlacedResourceX of a texture gets a PC texture kept in
  step with the title's memory – every mip and slice detiled with the XG layout (`hle::xg`) and uploaded at creation;
  unified memory is reserved with MEM_WRITE_WATCH (OpenExistingHeapFromAddress takes it), and before each
  ExecuteCommandLists one GetWriteWatch per window finds the textures whose pages the CPU wrote (streamed mips,
  DirectStorage reads) and uploads them again (on an XCL queue, waiting for it). Unreadable memory (reserved, not mapped)
  reads as zero. The test title writes a 64KB_S_X array through XG, places it, reads every subresource back through
  the GPU, rewrites one and sees it after the next ExecuteCommandLists. Not yet: GPU writes into the title's memory
  (copies to the address), the upload without a CPU wait, the state if the title moved the texture out of its initial
  one.
- **Physical pages mapped into reserved address space** (Fortnite's texture streaming, `kernelx`): a graphics
  reservation (VirtualAlloc / XMemVirtualAlloc, MEM_RESERVE without commit) is Windows address space reserved with
  MEM_WRITE_WATCH and remembered; a page (number = UMA page index + 1, 0 = unmap) lives in its allocation's UMA block;
  mapping pages at another address, in any order, commits it and copies their data there, unmapping (null / 0 pages,
  or mapping them elsewhere) copies it back – a page is in one place at a time and keeps its contents (no aliasing:
  two addresses of one page at once are not possible this way). Title textures placed there read only the committed
  pages and are uploaded again after the mapping / the writes. Tests: reverse-order mapping, moving a page, unmapping
  and mapping it again (xmem); a texture placed in an empty pool gets its texels once a page is mapped in (GPU).
- XGameRuntime task queues: handles are looked up in a set of live ones before being read – an XAsync completing on a
  worker after its queue was closed read freed memory (a rare lib-test panic in `queue_from`).

## Device as IDXGIDevice, XMemVirtualAlloc (session 5, 2026-11-21)
- The console D3D12 device is also an IDXGIDevice / 1 / 2: Microsoft's public GDK samples (ATG DeviceResources) take the
  output for SetFrameIntervalX through `device.As(&dxgiDevice) → GetAdapter → EnumOutputs(0)` and ThrowIfFailed –
  a PC D3D12 device answers E_NOINTERFACE. `d3d12x/dxgi_device.rs`: a small object with the device's real adapter
  (by LUID); every other interface is the device's. EnqueueSetEvent is not supported (needs the title's queues).
  Console code only (device #0 split): DXGI asks a D3D12 device for IDXGIDevice / IDXGIObject while creating a swap
  chain, and answering it made that crash now and then.
- `XMemVirtualAlloc(address, size, type, XALLOC_ATTRIBUTES, protect)` (kernelx; also found by name for `xmem.dll`
  imports): a graphics memory type allocates in unified memory, as MEM_GRAPHICS. It was a stub returning null.
- Public Series S research (SystemOS side, CS-Unit "XboxSeries-Archive"): the texture layout library takes the hardware
  version (Series S vs X) and lays textures out differently for each – so `texture::CONSOLE_PIPES` (still unknown, only
  the _X modes need it) must be found per console and follow what `XSystemGetDeviceType` reports (Anaconda by default,
  Lockhart with the Series S setting). The rest of that write-up is the system partition, not the title's.
- Looked at (independent projects, only facts taken): GDKScarlett (Xbox One `d3d12_x` titles, GCN shaders; its
  PIX / AcpHal / mfplat stubs answer what XCL's automatic stubs already answer) and SlimEra (XDK-era kernelx:
  Era* / Etx* functions – no GDK title seen importing them).

## Contributions merged (session 5, 2026-11-21, PRs #5, #6, #8, part of #7)
- **#5** more RDNA2 instructions (scalar movrel / bitcmp / bfe64, d16 typed buffers, f64 ceil / fract / rndne / frexp /
  sqrt / rsq / class, v_swaprel, pknorm f16…) with tests. Fixed on the GPU (RX 9070), where its GPU tests had not run:
  `ceil_d` without f64 negation, `sqrt_d` (HLSL's sqrt is f32: f32 root + two f64 Newton steps), the swaprel test's VGPR
  range. Relative moves keep the existing rule: their targets lie within the program's VGPRs.
- **#8** DXIL cache: single flight per shader (threads and processes – named mutex `XclDxilCache-…`), container
  validation of cached blobs, atomic writes; RPF scanning / extraction hardened. The cached-only check comes before the
  locks (non-blocking pipeline creation must not wait for a compile in flight). Its XAudio2 part was dropped (PR #3
  decodes Opus; its frame table was wrong for SILK / hybrid).
- **#6** T# validation (mip range, cube faces, array depth), 1D height ignored, UMA span and subresource-count checks,
  physical-page mapping inside one allocation. Its own linear layout was not taken (not addrlib's): linear stays
  `tiling::Layout` mode 0.
- **Second round (#6 rebuilt, #9–#11, #13):** D3D12 texture limits and UTF-16 `MicrosoftGame.config`; DXC toolchain id
  (dxcompiler / dxil size + mtime) in the DXIL cache key – the cache rebuilds once – and fresh DXIL validated; GameSave
  read results survive a too-small-buffer retry; a null process task queue means disabled (`E_NO_TASK_QUEUE`); `xcl rpf`
  probes archives over 1 GB in 4 MB chunks (`--probe` / `--full`). **#12** every documented runtime feature
  available except XStore and XAppCapture; **#14** geometry (ES + GS) as two wave32 halves on adapters without wave64,
  rejecting cross-half operations (no change on wave64 adapters: the sampled shaders translate identically).
- **#7** only the persistent-local-storage accounting (used bytes measured). Its task-queue / XAsync handle registries
  (2.8 k lines against forged handles) were not taken.

## Contributions merged (2026-11-22, PRs #53–#60, all gefydev, tested by him under Wine)
XCL's own XAsync / XTaskQueue matter only without the PC runtime (with Gaming Services they go to the PC's).
- **#53** XAsyncCancel only calls the provider (libHttpClient): XCL's `begin_with` and XAsyncRun ignore Cancel and
  run to their end – completing with E_ABORT there ran Cleanup (freed the payload) under running DoWork.
- **#56** a failing provider Begin fails the call through the block, XAsyncBegin / XAsyncRun return S_OK;
  XAsyncComplete(E_PENDING) does not complete.
- **#54** XTaskQueueTerminate runs the queue's delayed callbacks still in the timer heap (canceled) before its own
  callback; **#57** a terminated queue refuses new work with E_ABORT (submit, delayed submit, waiters) – and,
  added on merge, XAsyncSchedule too.
- **#55** game save names Windows changes (".", "..", device names with any extension, trailing dot / space)
  get one more character %XX-escaped (`..` deleted every provider's saves); on merge also "nul .txt".
- **#58–#60** VFS hooks MoveFileWithProgressW, CopyFileW, ReplaceFileW (ReplaceFile into the game folder =
  CopyFileW backup + MoveFileExW through their hooks). **Measured on Windows 11** (counting hooks): MoveFileW
  reaches only MoveFileWithProgressW (so it was not redirected before #58); kernel32's MoveFileA / MoveFileExA /
  MoveFileWithProgressA reach no kernelbase move entry – hooked in kernel32 on merge; CopyFileW and the ANSI
  copies reach CopyFileExW, ReplaceFileA reaches ReplaceFileW. Probe with the real hooks: all move / copy /
  replace entries on `T:` work, game folder untouched (the probe is a local test, not in the repo: process-wide
  hooks + the global VFS).

## Contributions merged (2026-11-23, PRs #84, #68, #67, #69; #70 superseded by #67)
Checked on the RX 9070 (AMD): workspace tests with the retail samples (`XCL_SAMPLE_RPF` / `_DLLS` / `_TTPF`) natively
and with `XCL_SPLIT_WAVE64=1` (NEW: the 32-lane GPUs' split form on RDNA, which runs both wave sizes); XDefiant
(title screen right, 6 299 translated pipelines, 0 errors), Minecraft and Unicorn Overlord run.
- **#84** (Foniass) console view heaps take the shared heap a page of 256 slots at a time, when a slot is first
  used: NVIDIA's shared heap holds 1 000 000 descriptors, Minecraft's eight heaps of 73 727 slots wanted 1.18 M –
  the last two heaps' views were lost (blank frames, "flicker" on NVIDIA).
- **#68** (Foniass) a UAV barrier for all resources per batch with a UAV barrier / UNORDERED_ACCESS transition
  (translated shaders reach title memory through the window buffers – other resources); `v_fract_f32` of ±inf /
  NaN = NaN as on RDNA.
- **#67** (Foniass) wave64 kernel / compute / mesh / ES+GS programs on 32-lane GPUs: two hardware wave32s per
  console wave exchanging masks and lane values through groupshared (`docs/wave64-on-wave32.md`); replaces the
  refused-on-hazards halves of #14 / #39. Still open on NVIDIA: wave64 VS / PS reading lanes 32..63.
- **Found on merge (AMD):** the PC driver lays 2-D / 3-D thread groups out in 2 x 2 quads (RX 9070: an 8 x 8 group's
  lanes 0, 1, 4, 5, 8, 9 … hold SV_GroupIndex 0, 1, 2, 3 …; 1-D groups in order), the console in order – every
  native 2-D / 3-D compute translation read the wrong lanes in DPP / readlane / bpermute / mbcnt. The native form now
  hands each hardware wave the next console threads as it arrives (as #67's split form does).
- **#69** (Tems-py) PresentX queues XCL's copy before reporting the flip and keeps its source until the copy's fence
  (no allocator reset after a timed-out wait); aliasing barriers keep placed (`vram`) textures; title-texture
  refreshes serialized, no ResetWriteWatch at register (it dropped other users' pending CPU writes). Its heap
  binding was narrowed on merge: a title's own view heap is bound only when shader visible.
- **#70** (gefydev, compute as two independent wave32 halves, hazards refused) closed: #67 covers it without the
  refusals.
- Pre-existing, not from these PRs (fails on the previous `main` the same way): the opt-in GPU probes
  `xbox_bvh_builder_builds_on_the_gpu` / `_a_tlas_on_the_gpu` (`XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_GPU=1`) – the
  CPU-side trace finds no hit where brute force does (likely the builder's GPU output is not read back through the
  CPU view since the sparse windows of PR #40). To look at.

## PRs #46 (merged) and #43 (held), 2026-11-23
- **#46** (grooving6) float compares test NaN explicitly (the AMD driver gave a later `a < b` the unordered result
  of an earlier `!(a >= b)`); f64 floor / ceil / trunc / fract / ldexp / sqrt / rsq / cvt past the f32 and int64
  ranges. Held on 2026-11-22 because Unicorn Overlord's title screen stayed black with it; on current `main`
  (2026-11-23) Unicorn renders its intro and scenes with it (frame dumps), so merged. The prelude changed: every
  title's shaders compile again on their first run (XDefiant black for minutes – not a bug).
- **#43** (Lum7nt, MSAA layouts in `hle::xg`) held again. A local comparison with the console's `xg_xs.dll` (XDefiant's
  copy; 1 536 MSAA descs: 8 formats, 4 sizes, 2x / 4x / 8x, arrays 1 / 6, 8 layout / flag combinations) – 744 the
  same, 480 UAV descs the console accepts and the PR refuses, 72 RT + DS descs the other way round, 216 with another
  metadata plane (plane 1 mip0 +0x48), and 3 729 of 9 874 texel / sample offsets different (depth: almost all;
  colour: 1 in 6). `verify_console_msaa_offsets` only prints the console's layout. `main` keeps E_NOTIMPL for MSAA
  (the PC's allocation info) – safe.
- `XCL_SAMPLE_XG` versions in annn: XDefiant's `xg_xs.dll` passes all of `xg_oracle`; Unicorn Overlord's (a newer game
  OS) differs in 184 render-target cases at plane 1 +0x10 (console 0x2000, ours 0) – a newer library, to look at.
- `dxc::cache_tests::concurrent_cache_misses_compile_once` can fail under load (a 200 ms timing bound), passes alone.

## PRs #61–#66, #72 (merged), #71 (closed), 2026-11-23
- **#61** (Mavixot) finished console shaders kept in memory (`pso::ShaderCache`, 64 MB / 128 entries, keyed by blob
  hash + bytes, stage, sampler base, windows, `split_wave64` – added on merge – and the DXIL cache folder): a
  pipeline sharing a shader with an earlier one skips translation and the disk cache (`gpu_pso_e2e` 28 s → 9 s).
- **#62 / #63** (gefydev) MicrosoftGame.config: `<!-- -->` comments removed after decoding (a commented-out
  Executable / Identity / PLS size / title memory was taken as the game's); `manifest::load` uses the launcher's
  decoder (UTF-16 BE too).
- **#64** VFS lists `FindExSearchLimitToDirectories` like a name match (the flag is advisory) – checked on Windows
  with the real hooks: `G:\data\*` gives `. .. f.txt sub` for both ops.
- **#65** XUserGetGamertag as documented: no modern suffix, so UniqueModern = the gamertag (`Player1`, games showed
  `Player1#0001` before); components cut to their documented buffer sizes.
- **#66** console id = `00000000.00000000.00000000.00000000.00` (38 characters, what the docs give for PC / RETAIL;
  the old 41-character default did not fit `XSystemConsoleIdBytes`); the old default in existing `xcl.toml` files is
  replaced on load. `XSystemGetAppSpecificDeviceId` changes once with it.
- **#72** f64 abs / neg on the sign bit (DXIL's `-x` is `fsub -0.0, x`: a NaN's sign is up to the driver; NVIDIA kept it)
  – cherry-picked without **#71** (GPU tests as independent wave32 halves), which #67 superseded.

## PRs #74–#79 (merged), 2026-11-23
- **#74** (certified-retart) `gpu_dispatch_matches_native_wave64_reference` skips without an SM 6.6 GPU.
- **#75** (certified-retart) optional D3D12 Agility SDK at run time (`gpu::agility`, `ID3D12SDKConfiguration1::
  CreateDeviceFactory`): Windows 10's inbox D3D12Core stops at SM 6.5. Searched: `%XCL_D3D12_DIR%` (only it, when set),
  `<xcl.exe dir>\D3D12`, `<xcl.exe dir>`, `%LOCALAPPDATA%\XCL\D3D12`; none → inbox D3D12 as before. All device
  creation goes through `agility::create_device`. Agility 1.620+ swaps a list's vtable on Close / Reset: `real::
  preserve_vtable` re-adopts the new native table (command list Close / Reset – Reset is now overridden on PC lists
  too –, KickoffX, console Reset, DMA Close). `pe.rs`: checked RVA arithmetic, reads bounded to their section.
  Checked on Windows 11 without an Agility SDK (the fallback): workspace tests, Unicorn Overlord, XDefiant.
- **#76 / #77** XNetworking: no struct padding copied into the default security info; certificate verification
  fails closed (no URL / host, GetLastError 0, malformed pins, unknown thumbprint types, pinned plain-HTTP loopback).
- **#78** (gefydev) `tests/gpu_compute_e2e.rs`: a console compute shader in the retail container layout through
  `CreateComputePipelineState` on the backend device, dispatched (`runner::dispatch_on`) and read back – a 16 x 8
  group, so it also covers the native lane reorder of 2-D groups; passes natively and with `XCL_SPLIT_WAVE64=1`.
- **#79** `ds_write_addtid_b32` / `ds_read_addtid_b32` (M0[15:0] + offset + 4 · lane); on merge also kept below
  the split wave64 exchange (`XCL_X_LDS_TOP`).

## PRs #80–#82, #85, #86 (merged), 2026-11-23
- **#86** (Foniass) a texture placed in a title heap whose PC size is larger than its XG size gets a committed
  resource of the heap's kind (the console's aliasing of that placement is lost): Minecraft's logo (0x29F000 on the
  console, 0x2E0000 on the PC) covered the 16x16 textures packed after it – the magenta / black squares on its
  loading screen (the open item above). On merge: COMMON retried when the PC refuses the state. Minecraft logs ~10
  such placements at start.
- **#80** exact round-to-nearest-even and NaN → 0 for `v_cvt_(pk)norm_{u16,i16}_f16` (integer math on the f16 bits).
- **#81** DirectStorage queues: RetrieveErrorRecord clears the error and resets the event, GetErrorEvent duplicates
  the handle, Release / Close / Enqueue races, status / fence pointers validated. Fortnite (all its IO through it)
  still loads to its splash screen.
- **#82** `XCL_ADAPTER=5070` / `7900`: a number past the adapter count matches a model number in the name.
- **#85** XAccessibility (closed captions, high contrast) as an XCL fallback object and by-name exports when the
  system xgameruntime.dll is missing.

## PRs #87–#91 (merged), issue #73, 2026-11-23
- **#91** (fixes #73) `s_round_mode` / `s_setreg*` of MODE[3:0] are no longer blanket NOOPs: a pre-scan
  (`Em::configure_round_modes`); default writes stay counted NOOPs; a non-default f16 / f64 mode (MODE[3:2], `0xC`
  = f16 / f64 round toward zero, 95 % of the pack's `s_round_mode`) in a program with f16 ops is tracked in
  `xcl_fp16_rm` and the f16 results are rounded with `f32_to_f16_rm`; non-default f32 / f64 modes are reported
  unsupported (none in the pack: still 26 698 / 26 698 translated in full). `v_mad_f16` rounds per step
  (`mad_nf_f16`). DXC on the programs that get the dynamic mode (122 of a 1-in-40 sample of 670): 0 errors, median
  0.84 s. Changed HLSL: those programs compile again on the first run (XDefiant's UI text came back after it).
- **#87** kernelx: GetProcAddress / LoadLibraryEx with null names, GetThreadName sizing, VirtualFree(Ex) bookkeeping
  only after success (sparse / graphics reservations), XMemVirtualQuery guard pages, checked allocation arithmetic.
- **#88** VFS: `TRUNCATE_EXISTING` on a game file without an overlay copy opens the overlay with `CREATE_ALWAYS`
  (was ERROR_FILE_NOT_FOUND), with Win32's checks (GENERIC_WRITE, directories).
- **#89** XAudio2 play / loop regions always valid for SubmitSourceBuffer (`clamp_regions`, `silent_buffer`).
- **#90** AcpHal: APU address ranges without wrap / overlap, SHAPE allocation rolls back, XApuConnect reports OOM.
- Checked: workspace tests with the retail samples (also `XCL_SPLIT_WAVE64=1`), Unicorn Overlord, XDefiant.

## PRs #95–#102 (merged, all raylitebull), 2026-11-23
Checked: workspace tests with the retail samples (also `XCL_SPLIT_WAVE64=1`), XDefiant (title screen; a scripted A
opens its "Xbox network unavailable" dialog – input arrives), Unicorn Overlord, Minecraft starts.
- **#95** XGameSaveDeleteContainer("") deleted the provider's whole folder – now E_GS_INVALID_CONTAINER_NAME.
- **#96** VFS move / copy: case-only renames of game files (were deleted on NTFS), MOVEFILE_REPLACE_EXISTING /
  fail-if-exists across the merged view, a failed copy / move no longer lifts a whiteout (`stage_write` /
  `commit_write`), ReplaceFileW checks both files; whited-out directories list as missing, delete_file /
  remove_dir type checks, a recreated directory does not bring back the original's children.
- **#97** XPackage slot 27 = 3-argument XPackageGetIdentifier (was the 2-argument one: crash); game saves written
  through unique temp files + MoveFileEx (write-through), blob / container names matched case-insensitively,
  XPackageGetUserLocale picks from MicrosoftGame.config's `<Resources>` languages, the controller id from
  XUserFindControllerForUserWithUi is the gamepad's.
- **#98** XTaskQueue / XAsync: termination and submission under one lock (no work after Terminate, no deadlock on
  a manual completion port), per-port in-flight counts, XAsync phases (cleanup exactly once), XErrorReport's
  callback called outside its lock.
- **#99** sparse windows: tiles of placed buffers know their uncommitted pages, sub-tile decommits give the pool
  tile back; ResolveQueryData adds `buffer_delta`; console heap type 4 → CBV/SRV/UAV in copies; DMA list Release.
- **#100** keyboard / mouse emulation by physical scancodes (AZERTY etc.), mouse state reset on focus loss,
  `keyboard_mouse = false` turns off the keyboard as gamepad too, GameInput QueryInterface answers only its own
  IIDs (refused ones logged – added on merge), UnregisterCallback waits for callbacks in flight.
- **#101** ANSI outputs encoded in CP_ACP, stubs of HLE alias DLLs return their canonical module's default,
  GetModuleHandle for console-only aliases, ordinal imports fall back only to the same system DLL, forwarders
  with dotted names, no duplicate stub thunks.
- **#102** async compute PSOs: the real pipeline (not the stand-in) is registered as translated (ExecuteIndirect
  bound the ABI on the stand-in, a failed build bound it anyway); the initial PSO of CreateCommandList / Reset /
  ClearState is tracked; DrawIndexedX goes through `bind`; KickoffX keeps multiple render targets.

## PRs #107–#109 (merged), README controls, 2026-11-23
- **#107** (raylitebull) recompiler: integer PS inputs not converted through float in `v_interp_mov / p2`,
  `ds_swizzle` / `ds_bpermute` give 0 from inactive source lanes, exact `ldexp_f`, 32-bit literals in the high
  word of f64 operands, NaN in SNORM / pknorm → 0 and negative → 0 in the small unsigned floats, signed BFE with an
  arithmetic shift, `v_cmp_class_f16` without the f16 round trip, BVH triangle test without `det²`. Retail pack:
  still 26 698 / 26 698; DXC on every 150th record (179 variants): 0 errors. Prelude changed: titles recompile.
- **#108** (certified-retart) XGameSave async container delete runs in the queued work (serialized with blob
  updates by `SAVE_MUTATION`); conflict with #95's identical check resolved for #108's form.
- **#109** device removal checked before GPU readbacks are mapped (GetCompletedValue = UINT64_MAX after a removal).
- README: "What is missing" removed, a "Controls" section (gamepad recommended; keyboard / mouse as a gamepad).

## XDefiant's UI text: the #107 regression (2026-11-23)
- Bisected (warm cache): `668f550` text, `c813f08` (#107 alone) none. Cause: `v_interp_mov_f32` / `v_interp_p2_f32`
  of int / uint PS inputs as a bit copy (`asuint(xi.aN[c])`) instead of the value converted to float. Reverted
  (with a note at `v_interp_mov_f32`); #107's test now checks the conversion.
- What it means: XDefiant's UI PS (`000f5c62725d684f_ps`, inputs UserBig / UserSmall / RenderType (int4) / Color)
  reads RenderType with `v_interp_mov` and compares it with 1 and 2 (`v_cmpx_eq_u32`): RenderType 1 samples
  the font with `image_sample_d`, 2 takes four `image_sample_lz` taps at offsets from ds_swizzle derivatives.
  With the bits (what the ISA does) the text takes those branches and draws nothing; with the converted value
  (never 1 or 2) it takes the default branch and draws the text in red. If the console's text is not red
  (yellow?), the real bug is in those branches: tried and not it – a helper-lane-aware `ballot` in pixel
  shaders (quad reads + WaveActiveBitOr, entry EXEC live lanes only) and not zeroing ds_swizzle sources.
- Tools on the user's desktop: `xdtest.sh` (runs XDefiant, counts red pixels in the frame dumps),
  `xd_cold_dxil\` (a DXIL cache with the current HLSL), `bisect\` (a git worktree) + `trrr\target-bisect\`.
- DXC compile time (measured on the way, for the optimisation work): -O3 / -O1 / -O0 on 258 random programs
  349 / 304 / 235 s – the optimiser is not the cost; programs with `pc` blocks (13–17 %) take 53–75 % of it
  (DXC's loop passes: unroll ~30 %, jump threading ~18 %, indvars ~12 % on the slowest); the largest program
  (29 000 instructions, 6.5 MB HLSL) peaks at ~0.6 GB in DXC. `XCL_DXC_ARGS` passes DXC arguments in the survey.

## FidelityFX (FSR4) runs on the GPU (2026-11-23)
- The console upscaler (`amd_fidelityfx_upscaler_xbox.dll`) creates its passes with CreateComputePipelineState on a
  blob with an `ACCC` part holding `FSPX` (the console driver's container, "NewBE"; it tries
  DeserializeComputePipelineStateX first, which stays E_NOTIMPL). `pso::fspx` reads it: entry 0 = numthreads (u16 x3
  @+0x0C), RSRC1 @+0x14, RSRC2 @+0x18; entry 6 = the RDNA2 code behind a 0x24-byte header (cut at the last
  s_endpgm). Compute interface from RSRC2 as for FSSM CS. All 29 FSR4 pipelines translate and compile
  (`fspx_containers_translate_and_compile`, `XCL_SAMPLE_FSPX=<folder of dumped *_rootsig.bin>`).
- `v_dot4(c)_i32_i8` / `v_dot4_u32_u8` as SM 6.4 `dot4add_i8packed` / `dot4add_u8packed` (the network passes are
  thousands of them): the biggest pass went 1.55 MB → 1.23 MB DXIL, DXC 604 s → 74 s, and the AMD driver now accepts
  it (it returned E_OUTOFMEMORY before).
- `fsr4_console_upscaler_creates_a_context_probe` with `XCL_CONSOLE_LISTS=1 XCL_PROBE_FSR=1 XCL_PROBE_FSR_DISPATCH=1
  XCL_PROBE_FSR_EXECUTE=1` (`XCL_SAMPLE_DLLS` = the game's DLLs): inputs uploaded (a 0..1 grey gradient, depth 0.5,
  no motion), ffxDispatch records 20 dispatches / 24 barriers / 2 clears / 4 copies on a console list, all
  recognised, the PC list runs on the RX 9070 (~11 ms with submission), the 1920x1080 RGBA16F output is finite and
  keeps the gradient (red mean 0.123 left quarter, 0.872 right). The probe needs `XCL_CONSOLE_LISTS=1` (without it
  the pipelines are not console-shaped and the stream has unknown records).
- Based on PR #33 (ElHombreAlex), rebuilt on current `main`: #33 went through DeserializePipelineStateX with a guessed
  thread geometry; the FSPX path carries the real one. Its wave64 rendezvous is #67; R9G9B9E5 UAVs were not needed here.
- Not yet: a title using FSR4 (none of the four run here ships it; the dump with it has no executable).

## PRs #110–#130 (2026-11-23)
Merged after workspace tests with the retail samples (also `XCL_SPLIT_WAVE64=1`), the retail survey (26 698 / 26 698),
the FSR4 GPU probe, Unicorn Overlord and XDefiant:
- #110 Ogg Opus end trimming and pre-skip; #115 `xcl run` checks the PC first (SM 6.6, mesh shaders, wave width,
  DXC, xgameruntime.dll) and keeps it in `report.json → environment`; #116 a game-report issue form;
  #117 cube T#s as cube arrays (slice from the face coordinate, resinfo layers); #118 command-list Reset and indexed
  draws transactional; #119 OMOD on f64 / conversions, integer clamp on unsigned / packed adds and dots (its clamped
  8-bit dots keep `dot4_u8` / `dot4_i8`: DXC had an internal error on `(int64_t)dot4add_i8packed(a, b, 0)`);
  #121 queue #18 before the layout is known writes 8 bytes; #122 video-memory arenas never overlap (straddling
  placements merge arenas; XDefiant logs one at ~78 s); #124 MS / GS translated with the title's memory windows;
  #127 CPU check at start (AVX2, FMA, F16C, BMI1/2, LZCNT, POPCNT, MOVBE, SSE4.2; `XCL_SKIP_CPU_CHECK=1`), bounded reads
  in the exception logger, thread dump outside the suspended thread, PE directory bounds; #129 frame statistics with
  each frame's own origin, ancillary handles kept alive while the clock signals them.
- Held: **#123** (memory window from the full 64-bit address) – with it the AMD shader compiler (`amdxc64.dll+0xe8f875`,
  access violation) crashes XDefiant ~10 s in, creating the pipeline of CS `8e4af296b9d7fd9c` (2 552 instructions);
  reverting only #123 runs. **#128** (f16 add / fma rounded once under a directed mode, #111) – its own GPU test fails
  on the RX 9070 (`v_sub_f16` under mode 0x4: 0x30804001, expected 0x3C01).
- Closed: #120 (#73 was done by #91; would make 0xC programs incomplete), #126 (= #119), #130 (= #121 + #122), #125
  (a process report, not project documentation).

## Opus decoding (session 5, 2026-11-21, PR #3)
- `hle/opus.rs`: Windows' own decoder, the "Microsoft Opus Audio Decoder MFT" (`MSOpusDecoder.dll`, Media Foundation),
  loaded with `mfplat.dll` at run time through `DllGetClassObject` (no COM registration, no libopus). RFC 6716 packet
  parser (all 32 TOC configurations, codes 0–3), RFC 7845 `OpusHead` (pre-skip, output gain; multistream 3–8 channels
  split over several MFTs), framing: Ogg, `[u16 LE length][packet]…` in blocks (nBlockAlign, or the AWC 2048 B when the
  format has none), a raw packet.
- `hle/xaudio2.rs`: `WAVE_FORMAT_OPUS` (0x704F) / EXTENSIBLE with the Opus subtype → a PCM16 voice; SubmitSourceBuffer
  decodes the buffer, the PCM lives until `OnBufferEnd` (the voice's callback is wrapped; contexts mapped back in the
  callbacks and `GetState`), `FlushSourceBuffers` / end of stream reset the decoder, regions clamped to the decoded
  length; without the decoder: silence of the same length. `report.json → audio`: `opus_voices`,
  `opus_buffers_decoded`, `opus_decode_errors`.
- **Checked on the game's audio** (`XCL_SAMPLE_AWC` = an AWC with one Opus stream, test `decodes_a_game_sound`,
  `XCL_OPUS_WAV` writes a WAV): a 1.6 s effect and a 56.6 s sound from a bank decode completely (whole 20 ms frames,
  < 3 frames past the AWC's sample count – its encoder delay / end), 56 s in 0.2 s. The test packets in the repository
  are libopus-encoded tones, no game data.
- Open (needs the exe): whether the title uses 0x704F voices with AWC-framed buffers at all, the voice format's
  nBlockAlign / extra bytes; streaming AWCs (512 KB blocks with a 2048 B header) are not unpacked by XCL – the title does that.

## Audio – conclusions from `audio.rpf` (the real game)
- AWC containers ("ADAT", version 1): u64 chunk table (type = byte 7, size = bits 28–55, offset = bits 0–27, offset from the
  start of the AWC), types: 0x55 data, 0x76 format (32 B: samples u32, loop i32, frequency u16 @8, …, **codec @28**, channels @29),
  0x48 stream format (512 KB blocks), 0xA3 seek table (start sample of every 2048 B block), 0x36 peak, 0x81, 0x5C, 0xBD.
- **Codec 0x0B = Opus** in 100% of the samples (92 formats + 17 streams from 160 places in the pack): 48 kHz, 2048 B blocks
  `[u16 LE length][Opus packet]…` + zeros, 20 ms frames (960 samples; TOC 0x68/0x78 Hybrid, 0xF8 CELT, mono). Number of
  packets in a block = the difference in the seek table / 960 – verified. **No XMA2.**
- Conclusion: an XMA2 decoder has low priority for this title. Open: whether the game decodes Opus by itself (then PCM → XAudio2 already
  works), or uses the console's hardware Opus decoding (then the API is needed – check with `xcl inspect` on the exe).
  Windows has `C:\Windows\System32\MSOpusDecoder.dll` (a Media Foundation MFT) – a candidate decoder without bundling libopus.
  Registered as "Microsoft Opus Audio Decoder MFT" {63e17c10-2d43-4c42-8fe3-8d8b63e46a6a} (session 5).
- Streaming AWC (session 5, from fragments of the pack): flags 0xFF25, with bit 0 set u16 chunk indices per stream, then 12 B
  of unknown meaning; the u32 stream infos (number of chunks = bits 31:29) lie right before the chunk table, and the table ends
  at `data_offset` from the header. Chunk 0x48 = {number of blocks u32, block size u32 (512 KB), channels u32, per channel 16 B: id,
  samples, u16, frequency u16, codec u8 …}; 0xA3 = start sample of every 512 KB block. 512 KB block = 2048 B header
  (−1, 254 subblocks, 0, samples in the block, …, from +24 the start sample of every subblock) + 2048 B subblocks `[u16 length][packet]…`.
  Unexplained: 6 packets per subblock with a table difference of 12·960 – TOC 0xFC is CELT FB 20 ms **stereo**, so the table
  probably counts samples × channels.

## XGameRuntime: objects from the GDK table, persistent local storage, device type (session 5, 2026-11-21)
- **Persistent local storage gap:** class `f4faf4d4` / `41a4e10c` went to the PC runtime, which fails it (E_FAIL) for an
  unpackaged process – a title would not get its PLS path. Now XCL's (GetPathSize / GetPath / GetSpaceInfo; the space a
  title asks for is granted). Slots agree with the game's libHttpClient thunks (3: 1 ptr, 4: 3 args, 5: 1 ptr, 6, 7, 8).
- **Device type:** `XSystemGetDeviceType` is computed in the thunk library, not through QueryApiImpl: `RegOpenKeyW(HKLM,
  Software\Microsoft\Windows NT\CurrentVersion\OEM)` + `RegQueryValueExW(ConsoleMode)` – mapping read from the thunk in the
  game's XCurl.dll: 0x60 → Lockhart (6), 0x64 → Anaconda (7), 0x68 → devkit (8), 0x40 / 0x50 / 0x58 → Xbox One S / X /
  devkit, no key → PC (1). kernelx now answers `RegOpenKeyW` / `RegOpenKeyExW` / `RegQueryValueExW` / `RegCloseKey` for
  that key only (0x64 or 0x60 by `device_type`); `advapi32.dll` imports go through kernelx (other functions still resolve
  to advapi32 itself).
- **GDK table** (`hle/xgameruntime/gdk_table.rs`, `scripts/gen_gdk_table.py`): 396 functions → (class, interface, slot) as
  the public PC GDK's `xgameruntime.lib` calls them. XCL's objects are built from it **by function name** (`api::layout`):
  a class's interface ids whose slots do not collide share one vtable – they are GDK versions appending functions (XUser:
  base 3..36, `eb9bf948` 37–38, `079415e3` 42, `26f3c674` 43–46, `5131d685` 47–49); `cef4fac0` (XUserGetGamertag) has its
  own. An interface id the table does not know (another GDK version: the game's libHttpClient asks XPackage `e2a4734b`,
  the table has `0b9d556e`) gets the main vtable. Unimplemented functions are logged and reported by name.
  The table agrees with the hand mapping from the game's thunks (test `vtables_follow_the_gdk_table`) and corrects two
  guessed XPackage slots (EnumerateFeatures = 33, 27 = GetIdentifier).
- **New in XCL:** XUser device association (registration tokens, FindForDevice → not found); XUser privilege / issue UI
  (completed), sign-out deferral (not available), AddByIdWithUi, Platform* hooks, IsSignOutPresent / IsStoreUser (false),
  SignOut; XPackage locale (Windows), chunks (all ready, install = done, estimate 0), write stats, kind (Game),
  MountWithUi, EnumeratePackages (the game itself with its details); XSystem app-specific device id, handle tracking;
  analytics (family "Windows.Xbox", OS 10.0.25398, Series X / S); runtime features (true); XDisplay (XLaunchUri refused,
  timeout deferral); XGameEvent (offline: accepted, dropped). Refused: XStore, XAppBroadcast, XAppCapture.
- Tests: `vtables_follow_the_gdk_table`, `console_mode_registry_value`, PLS in `thunk_protocol_with_xcl_objects_on_pc_queues`.
- **Who answers the 396 functions** (counted on the GDK table): XCL 123 (+7 dialogs and 7 text-entry / notification
  calls forwarded to the PC runtime), the PC runtime's whole classes 111 (XAsync / XTaskQueue / XThread, XGameStreaming,
  XNetworking, speech, invites / activation / protocols, XError, closed captions), refused 109 (XStore, XAppBroadcast /
  XAppCapture), not implemented in XCL classes 39 – Xbox Live and account UI (token and signature, player picker,
  profile card, achievements, invites, web authentication, state share, UI response setters), XUserFindControllerFor-
  UserWithUi, and three without public signatures (XGameUiShowManageSpace*, XPackageGetIdentifier) plus
  XPackageUninstallUWPInstance / XPersistentLocalStorageMountForPackage.
- Added: XUserGetGamerPicture* (a plain PNG of the requested size: 64 / 208 / 424 / 1080 px), XUserGetDefaultAudio-
  EndpointUtf16 (Windows' default communications endpoint, looked up on a thread of its own), XGameUiTextEntry* and
  XGameUiSetNotificationPositionHint to the PC runtime (no XCL handles; the PC accepted the title's GDK version, so its
  vtable has these slots). Tests: `gamer_picture_is_a_png`, `default_audio_endpoint_is_windows_default`, the picture and
  the forwarded slot in `thunk_protocol_with_xcl_objects_on_pc_queues`.

## Non-blocking pipeline creation (session 5, 2026-11-21)
- **Gap:** DXC ran in the title's `Create*PipelineState` call: 0.2 s median, seconds for big programs – thousands of
  pipelines while a title loads meant minutes of stall on the first run (the DXIL disk cache helps only later runs).
- **Now** (`hle/d3d12x/pso_async.rs` + `real.rs`): every translated path is first tried with `dxc::cached_only` (no
  compiling; a miss fails at once – `attempt`, translation errors of a missed attempt are dropped). All DXIL cached →
  built in the call as before. Otherwise the title gets a **stand-in** at once (dummy shaders, its own description:
  `standin_graphics` / `standin_compute` / `standin_stream`; mesh stand-ins are DXIL: `dummy_mesh`, one degenerate
  triangle) and the same `translated_*` path runs on a worker (`pso_async::submit`, one thread per core, 64 MB stacks)
  on an **owned copy** of the description (`OwnedDesc`: shader blobs, input layout + semantic names, stream subobjects;
  stream output / input layout in a stream / view instancing → built in the call).
- **Binding:** `SetPipelineState` on PC command lists (new override), console list replay, `two_pass_of`,
  `pso_root_signature` go through `pso_async::resolve` – the real pipeline once built. Graphics draws nothing until then;
  compute is waited for at its first bind / replay (`resolve_wait`; a skipped dispatch would corrupt results).
  `report.json → d3d12.background_pipelines` = (built, failed, still building). `XCL_ASYNC_PSO=0` turns it off.
- Tests: `retail_pipelines_built_in_the_background_behind_a_stand_in` (empty DXIL cache: the game's VS/PS pipeline built
  in the background renders the same triangle pixel for pixel; a mesh pipeline returns after ~0.1 s, real after ~1 s,
  two-pass parts found through the stand-in, title's description freed before the worker ran); the game's BVH builder
  with an empty cache (52 programs compiled in the background, compute waited for at replay) builds BLAS + TLAS.
- Releasing a stand-in for good (PipelineState `Release` override) releases its real pipeline and forgets the address
  (a later object at the same address must not resolve to it); a stand-in released while building: the worker's result is
  released when it arrives (outside the `done` lock: the Release override takes PENDING, `counts` takes PENDING then
  `done`). Any pipeline released for good also leaves `PSO_RS` / `TWO_PASS` (`forget_pso`): a later pipeline at the same
  address must not get its root signature or compute pass; the compute pass is released with the last `TwoPassPso`.
  Tests: `pso_through_hook` returns the real pipeline (works with a cold DXIL cache), the background test restores
  `XCL_DXIL_CACHE` through a guard and scribbles over the mesh stream description after the call.

## Title memory above 4 GB: UMA windows (session 5, 2026-11-21)
- **Gap:** shaders addressed one `mem` buffer with the low 32 bits of the address, and D3D12 opens a heap from an
  address only below 4 GiB (4095 MiB) – so at most 4 GB of title graphics memory, while a Series X title has ~13.5 GB.
- **Now** (`gpu/uma.rs`): up to 4 *windows* of ≤ 4095 MiB, window k at `base + k · 4 GiB`, `base` 16 GiB-aligned ⇒ the
  window of an address is `(addr >> 32) & 3`, the low 32 bits the offset in its heap. `Uma::new(size)` splits into as few
  equal windows as fit; `with_windows(n, size)`, `window(k)`, `window_of(p)`, `alloc_in(k, …)`; allocations never straddle
  windows. Console descriptor heaps and descriptor-table copies stay in window 0 (32-bit console addresses).
- Root signature: u2..u5 = windows 0..3 (`pso::abi_root_signature_with`, 6 root UAVs). Runtime: one heap + placed buffer
  per window (`real::UMA_HEAPS`, `uma_window_vas`, `uma_va_of`); console buffers placed in their window's heap; console
  lists bind all windows. Runner: arenas with several windows (`run_in_arena`).
- Translator: `Options::mem_windows` (from `pso::mem_windows()` = UMA windows). 1 = the single-window HLSL unchanged
  (DXIL cache stays valid). >1: every access site sets `xcl_w` from the high address bits – V# word 1 (MUBUF / MTBUF,
  buffer atomics, d16, s_buffer_load), the SGPR pair (s_load), the 64-bit VGPR address or SADDR pair (FLAT / global,
  atomics), BVH descriptor `dw0 >> 24`; `finish()` declares `mem1`.. and turns every `mem.Method(` (program and
  preludes) into an accessor `xcl_mMethod(` switching on `xcl_w` (`windowed`, `mem_accessors`).
- Config: `uma_mb` up to 16380 (default unchanged: 4095 "advanced", 2048 "standard"; `report.json` shows `uma_used_kb`).
- Tests: `windows_are_4_gib_apart_and_allocations_stay_inside`, GPU `gpu_memory_windows_above_4_gib` (s_load, buffer
  load / store, global load / store / atomic in window 1 next to window 0); DXC with 2 windows on a random 2 % of the pack
  (480 programs, seed 1795270997254744600 region): 0 errors, median 0.29 s (vs ~0.2 s with one window). Survey:
  `XCL_SURVEY_WINDOWS=N`.

## XGameRuntime through the GDK thunks (session 5, 2026-11-21)
- **Gap (blocked the start of every GDK game):** titles do not import XTaskQueueCreate / XUserAddAsync… by name – they have
  compiled-in thunks that load `xgameruntime.dll` and take `InitializeApiImpl(Ex/Ex2)`, `QueryApiImpl`,
  `UninitializeApiImpl`, `XErrorReport`; every API is a vtable slot of an object from `QueryApiImpl(clsid, iid)` (0..2 = IUnknown).
  XCL did not have this → `HCInitialize` from libHttpClient.GDK returned 0x89240101 (the game: XGameRuntimeInitialize likewise).
- **Now** (`hle/xgameruntime/api.rs`): 6 entry points. `QueryApiImpl`: XCL answers by itself for XUser (`01acd177`,
  iid `26f3c674` + `cef4fac0` = XUserGetGamertag), XGameSave (`704c3f58`), XPackage (`af406016`), XGame (`973a344e`:
  XGameGetXboxTitleId from `<TitleId>` in MicrosoftGame.Config / `title_id` in xcl.toml), XSystem (`e349bd1a`: console/sandbox
  id, the rest goes to PC), XGameUi (`dfcd4649`: only dialogs without a user handle – message, error, text – go to PC);
  it refuses XStore (`0dd112ac`) and XUser-device (`7d824997`) – the thunk then returns its own "API unavailable" error;
  it forwards the rest to the system `System32\xgameruntime.dll` (the same classes and GUIDs: 27 of 29 pairs from the console DLLs) –
  XAsync/XTaskQueue/XThread (`073b7dcb`), XNetworking, XGameInvite… After XAsync initialization XCL delegates to the PC one
  (`xasync::use_pc`; slots 3 GetStatus, 4 GetResultSize, 5 Cancel, 6 Run, 7 Begin, 9 Schedule, 10 Complete, 11 GetResult),
  because the queues and blocks then belong to PC – all XCL asyncs (`begin_with`) work unchanged. PC init: Ex2 → Ex → Impl
  (an older runtime does not know the new flags: libHttpClient passes 0x3).
- **Where the slot maps come from:** thunks in XCurl.dll / libHttpClient.GDK.dll (GUID pair, slot, argument and result widths), names
  in the slot code of the PC runtime (XAsync identifiers, `GameSaveImpl::…`), the order of the GDK headers; confirmed
  by matching argument counts and write widths (e.g. XUser 13 writes a byte = GetIsGuest, 14 a dword = GetState).
  XCL functions get the arguments without `this` (per-slot shims). Unhandled slots → E_NOTIMPL (bool → false) + log.
- **Safety:** the PC runtime does not validate foreign handles (XGameUiShowPlayerProfileCardAsync with an XCL handle →
  access violation), which is why nothing that takes an XCL XUser/XGameSave handle goes to PC.
- **Report:** `xgameruntime.apis` (pair → who answered) and `xgameruntime.unimplemented` ("XUser#23" → number of calls).
- **Not determined (needs the exe):** the PLS class (candidate `7f0fe8b8`, requires an unknown IID) and the XSystemGetDeviceType slot (PC will
  report "Pc") – to be read from the exe's thunks; `0a2192aa` is a console-only API (on PC: `WindowsGRTSStub` stubs).
- Tests: `thunk_protocol_with_xcl_objects_on_pc_queues` (XUserAddAsync/Result and XGameSave async through the vtable on
  PC queues, XStore rejected, an unhandled slot in the report), `libhttpclient_initialises_through_the_xgameruntime_thunks`
  (real DLL: HCInitialize → S_OK). Without the system xgameruntime.dll: the XCL classes work, the rest is "missing".

## 3D and linear textures (session 5, 2026-11-21)
- **3D thick modes** (4KB_S, 64KB_S / S_T, 4KB_S_X, 64KB_S_X, 64KB_D_X on a 3D resource; addrlib Gfx10Lib::
  GetSwizzlePatternInfo, `PATTERNS_3D` from `gfx10SwizzlePattern.h` S3 / S3_X / S3_T / D3_X): a block tiles x, y and z
  (`Layout::new_3d`, `block_d`, `oz`); block = addrlib Block1K_3d grown by log2(block) − 10 split over w, h, d
  (ComputeThickBlockDimension: 4 KB with 2 B texels = 8×16×16). The thin 3D modes (64KB_Z_X / R_X) and linear are 2D
  per depth slice. Mip chain (`Chain::new_3d`, ComputeSurfaceInfoMacroTiled / AddrFromCoordMacroTiled): per layer of
  `block_d` z-slices the whole chain (tail block, then the levels from the smallest), layers `slice_bytes` apart; the tail
  holds levels ≤ the block with h (4 KB) / w (64 KB) halved, at most GetMaxNumMipsInTail (thick: log2 − (log2 − 8) / 3)
  levels, origins in Block256_3d units. `detile_volume` / `tile_volume`; `texture::subresources` gives one subresource
  per level with its depth slices, `upload` copies them (slices NumRows × RowPitch apart).
- **Linear** (SW_MODE 0): rows padded to 256 bytes, levels from the smallest (HwlComputeSurfaceInfoLinear) – a "block" of
  256 B × 1 row in the same `Layout` / `Chain`.
- **Retail check (TTPF):** the texture header's dword 1 bits 21–23 are the dimension (1 2D, 2 cube, 3 volume, 5 array);
  the 3 volumes (4 KB tiles, R16F 256×32×8 and RGBA16F 32×64×32 / 32³ LUTs) are thick: the 256×32×8 one takes 256 KB
  (2D slices: 128 KB) – the model now fits 909 of 912 textures – and x / z neighbours differ 5–17× less than read as 2D
  slices. GPU test: thick 4 KB / 64 KB, BC1 3D, linear 3D and 2D upload bit for bit.
- Not covered: MSAA (only render targets, needs the console's pipe configuration and the exe).

## Tiled textures from console memory (session 5, 2026-11-21)
- **GFX10 mip chain** (`tiling::Chain`, `tiling::image_chain(&desc::Image, cfg)`; addrlib
  Gfx10Lib::ComputeSurfaceInfoMacroTiled/MicroTiled): per slice, first the mip tail block (levels ≤ half the block width,
  at most log2(block) − 4 levels; 64 KB / 4 KB blocks), then the levels from the smallest one to mip 0, slices every
  `slice_bytes`. A level in the tail: origin within the block from its offset (odd bits → x, even → y, ×256 B block).
  `detile_level` / `tile_level`. Level sizes = pixels >> i rounded up to whole elements (as in D3D12).
- **`gpu/texture.rs`**: `placement(T#)` (block-aligned address, bytes, chain, slices), `subresources(T#, memory)`
  (D3D12 order: mip + slice × mips), `uma_bytes(T#)`, `create(dev, queue, T#, memory)` → PC texture (SRV state),
  `upload(...)` – re-upload when the title rewrites the memory. `CONSOLE_PIPES` – the console's pipe/packer configuration for
  the _X modes, still unknown (the archives use modes without XOR).
- **Verified on TTPF** (`XCL_SAMPLE_TTPF`, test `retail_ttpf_textures_detile_with_their_mip_chains`): the model gives
  exactly the space occupied by 908 of 912 textures (outliers: 2 cubes in mode 2, 1 volume, the last one truncated). Number of
  mips = `dword0 >> 28` (0 → 1), slices = `(dword1 >> 10) & 0x7FF` + 1. Mips of R8/RGBA8/RGBA16 textures ≈ the previous
  level downscaled (6/6); 15 levels in the mip tail: error 0.66/255, at the wrong place in the block 150/255.
- GPU test `gpu_console_textures_upload_every_mip_and_slice`: tiled memory → PC texture → readback = the data, bit for
  bit, every mip and slice (RGBA8 4 KB array, BC7 64 KB cube).
- **To do with the exe:** binding to the API through which the title creates textures in its memory (CreatePlacedResourceX / heap
  at an address), and the moment of re-upload (CPU write / GPU copy); MSAA.

## Static samplers (session 5, 2026-11-21)
- **Gap:** the translator always took the PC sampler index from S# word 3 [11:0], but an S# built from constants in the code (the console's
  static root signature samplers) has something else there → wrong sampler / read outside the heap. In PS_PedTextured_Eight_Fire 66
  of 89 sampling operations; the e2e test only worked because those bits were 0 in its S#.
- **Now:** the `constant_samplers` pass (mod.rs) – an S# from `s_mov_b32/s_movk_i32/s_mov_b64` constants in the same block
  before the MIMG → static sampler `xcl_ss{k} : register(s{k}, space1)` (`Translation::static_samplers`,
  `Options::static_sampler_base` – PS after VS); `pso::abi_root_signature_with` + `static_sampler_desc` (from
  `desc::sampler_d3d12`; only 3 border colors); root signature cached per set (`abi_rs_for`), **PSO → root signature
  registry** (`real::pso_root_signature`) – drawing must bind this one instead of the title's; ConsolePso / TwoPassPso
  (`compute_rs`, `mesh_rs`) too. S# from descriptor tables (console heaps) – unchanged (index from word 3).
- GPU test `gpu_static_samplers_from_constants_in_code` (point 2.0 vs linear 1.5 at a texel boundary); PS from the game:
  6 static samplers, 61 uses; the PSO tests pass (PS_Rage_Textured – aniso 4× from the code); random 1% DXC: 0 errors.

## MicrosoftGame.Config (session 5, 2026-11-21)
- From the game's manifest (from the user; the rest of the file is irrelevant for XCL): `PersistentLocalStorage/SizeMB` = 2048,
  `VirtualMachine/SMTEnabled` = true, `XboxAnacondaTitleMemory` / `XboxLockhartTitleMemory` = Advanced.
- `manifest.rs` (NEW) reads these fields from the game folder; `Config::resolve` (launcher) fills in the "auto" settings:
  - `pls_mb` → XPersistentLocalStorageGetSpaceInfo (was hard-coded to 4 GiB);
  - SMT is irrelevant: **the game sees the PC's processors** (user's decision – "we're running on PC"; the version with 14 console
    processors was inconsistent: the CRT saw 14, OpenMP 16);
  - `title_memory` → `uma_mb` 4095 for "advanced" (otherwise 2048; `XCL_UMA_MB` still wins). The
    TitleMemoryStatus budget stays standard (13 824 / 8 192 MB) – the "Advanced" mode numbers are not public
    (docs.xbox.com is behind a login), `title_memory_mb` in xcl.toml overrides it.
- Test: the test manifest (`tests/data/testgame/MicrosoftGame.config`) has these values, e2e checks the launcher log and
  boot.json; `manifest::tests`.

## Instructions outside the pack, console heaps, structure (session 5, 2026-11-21)
- **ISA coverage** (`isa_coverage`, every decoder instruction separately; MIMG with DMASK=1, VOP3_SDST with VCC – previously
  false gaps): **975 / 1157 (84.3%)**, was 766. Added (GPU test `gpu_more_atomics_sub_dword_scratch_valu` +
  `image_atomic_inc` in `gpu_images_sample_load_store_atomic`):
  - a shared 32-bit atomic generator (`emit::rmw32`, `At::Mem` / `At::Ref`) for buffer / flat / global / DS / images:
    inc, dec, rsub, csub, mskor, cmpst / fcmpswap (bitwise comparison), add_f32, signed min/max on LDS and images –
    a bounded CAS loop (4096); fmin/fmax in `mem` still without a loop (the sign trick); **64-bit `_x2`**
    (`rmw64`: InterlockedXxx64 SM 6.6) – add/sub/and/or/xor/swap/min/max/cmpswap; `ds_wrxchg2(st64)_rtn_b32`;
  - flat/global/scratch byte/halfword + **d16 / d16_hi** (also buffer_*_d16, ds_read_*_d16, ds_write_*_d16_hi);
  - **scratch_*** – thread-private memory `static uint xcl_scratch[1024]` (4 KB/thread);
  - `v_sad_u8/hi_u8/u16`, `v_msad_u8`, `v_lerp_u8`, `v_div_scale/fmas/fixup_f32` (no scaling: VCC=0, fixup =
    IEEE special cases), `v_add/sub_nc_i32` with clamp, `v_mad_u32_u16/i32_i16`, `v_cvt_pk_u16_u32/i16_i32`,
    `v_cvt_pknorm_*_f32`, `v_mul_hi_i32_i24`, `v_sat_pk_u8_i16`, `v_swap_b32`; `s_memtime/s_memrealtime` (= 0);
    `buffer_gl0/gl1_inv`, `ds_nop` (no effect); `image_sample_cd*` like `_d`.
  - Still missing: f64 (sqrt/rsq/frexp/…), 64-bit LDS, GWS / append / consume, debug/trap, `s_setpc/s_call`, images
    `pck` / `by2` / `msaa_load` / `store_mip`, d16 formats (`buffer/tbuffer_*_format_d16_*`), `v_cvt_*_f16` norm.
- **Console descriptor heaps – finished** (`console_heap.rs`; test `slots_copy_and_release`):
  - **sampler heaps**: 16 B S# slots in UMA, slot k → PC sampler k, S# word 3 [11:0] = k (`#22 CreateSampler`);
  - **`#17 CreateConstantBufferView`** → a raw V# with the console address (UMA; otherwise an empty V#);
  - **`#24 CopyDescriptorsSimple` / `#23 CopyDescriptors`** – copies the PC descriptors (2 per view slot) + the slots, with
    index renumbering (T# word 0, S# word 3; V# unchanged); mixed console/PC ranges → log, PC only;
  - **heap Release** frees its slots (`heap_release`). Console increment: views 32, samplers 16.
  - #17 / #22–#24 assumed to follow the PC layout (like #14–#19 seen from console code) – to be confirmed with the exe.
- **Structure**: loops with an unconditional back jump (exit via flags + break), **`continue`** (a back jump to the
  head of the nearest loop), **entry into the middle of a loop** by a jump from earlier code (the target moves to the body level
  as pending: the code before it runs from the 2nd iteration on), a range equal to a loop (e.g. an if body) = that loop. Whole pack: programs
  with `pc` blocks **2274 → 2046 (8.5%)**. Remaining: a back jump without a cycle (the compiler shares code between paths –
  would require duplication), `continue` to an outer loop, loops with multiple back edges.

## MS (NGG) and GS pipelines in PSO creation (session 5, 2026-11-21)
- 9 282 MS + 195 GS from `shaders.rpf` (~40% of the pack) were getting replacement shaders until now (`real.rs` handled only VS+PS
  and CS). Now:
  - **`CreatePipelineState` with a stream containing a console MS** (`translated_mesh_stream`) and **`CreateGraphicsPipelineState` with a console
    VS + GS** (`translated_gs_pso`) → `pso::console_mesh` / `pso::console_geometry` → **`pso::TwoPass`**: a compute
    pass (the console program, one record per group in `res`) + a pass-through mesh shader with the translated PS. The title gets
    a mesh PSO (state from the stream / description: blend, rasterizer, depth, formats; RS = XCL ABI); the compute PSO and the record
    layout live in `real::two_pass_of(pso)` (`TwoPassPso`: compute, record, mesh_groups_per_record, kind).
    Console code never reaches the driver (failed translation → E_INVALIDARG). AS – not yet (none in the pack).
  - **Pass-through MS for a specific PS** (`emit::mesh_passthrough_for`): outputs = exactly the inputs of the translated PS
    (the same elements, order and types `{type}4` → identical signature packing, D3D12 links them); values from the record slot
    with the same semantic, missing ones = 0 – **in MS+PS pairs from one effect the signatures do NOT match** (the PS
    reads e.g. TEXCOORD20, which the MS does not write, and vice versa; on the console the driver links attributes by index). Clip
    distances are kept. No `SV_CullPrimitive` (bad triangles are degenerate).
  - **D3D12 limit: 32 KB of output per mesh group** – large PSs (17 attributes × 16 B × 128 vertices) exceed it →
    **split mode** (`emit::passthrough_split`): a record goes into N mesh groups, each with a stretch of the triangles and its own
    copies of 3 vertices; at draw time `DispatchMesh(groups · mesh_groups_per_record)`.
  - Tests: `retail_mesh_pipelines_through_the_hook` (MS+PS pairs with the same pass hash in the name, sample
    `XCL_MS_EVERY`, every 1500th by default; every 700th = 14/14), `retail_gs_pipeline_through_the_hook`, `gpu_mesh_mode_two_passes`
    (split mode = the same pixels). The `gpu_pso_e2e` tests are serialized through `setup` (shared PSO counters).
  - **Note:** `gpu_pso_e2e` regression runs without `XCL_SAMPLE_RPF` finish in 0.00 s (skipped) – always run them with the sample.
- **To do with the exe: drawing.** On DispatchMesh (MS) / Draw* (GS) with such a PSO: first `compute` with the ABI RS (io =
  SGPRs from the root arguments, `res` = a record buffer of groups · `record.record_dwords()` dwords, `mem` = UMA), a UAV barrier,
  then the mesh PSO with `DispatchMesh(groups · mesh_groups_per_record)`; for GS additionally fetching the ES vertices into `mem`
  (IO_GS_PRIMS / IO_GS_VERTICES as in `retail_es_gs_voxelize_geometry_mode`). D3D12.X draw records – from the exe.

## Shader pack: every record, every program (2026-11-22, issues #34 / #36)
- **`pack::scan` missed 2 634 of the 26 503 records of `shaders.rpf`**: 2 689 compute shaders have effect-qualified
  names (`alphaResolve:CS_AlphaTransmissionDenoise_2871f2e9_Wrapped`) and the name filter allowed no `:`. 55 of them
  were found anyway under a truncated name: once the real record was rejected, the scan went on byte by byte and where
  the `:` byte (0x3A = 58) equalled the length of the rest of the name it parsed the tail as a record with the same
  header (those were the "CS 55"). A name byte equal to its distance to the terminator always makes such a suffix
  alias; skipping past an accepted record's payload is what discards them (216 in this pack). Test
  `qualified_names_are_whole_records`. The other 21 archives (~150 GB) hold no shader records (`_Wrapped\0`, the end
  of every record name, occurs 26 503 times in `shaders.rpf` and nowhere else).
- **Translator on the new compute shaders** – three gaps, all general:
  - `ds_swizzle_b32`: the modes were told apart wrongly – 0x4000..0x7FFF (the bit mode with xor ≥ 16: 0x401F swaps
    the 16-lane halves, 56 programs) was rejected as "FFT/rotate", while real FFT (≥ 0xE000) and rotate (≥ 0xC000)
    were silently treated as quad permute. Now as in the ISA (`emit::swizzle_lane`; FFT / rotate as a constant
    table); tests `ds_swizzle_modes_follow_the_isa` (the ISA's FFT examples) and `gpu_ds_swizzle_modes`.
  - **Far jumps**: a branch beyond the 16-bit `s_branch` range (±128 KB) is `s_getpc_b64; s_add_u32; s_addc_u32;
    s_setpc_b64` with a constant offset – a direct jump, not an indirect one (12 programs; `ActivateRays` has 1 377).
    `rdna2::resolve_far_jumps` gives such an `s_setpc_b64` its target (`Decoded::far_target`), so the CFG,
    structuring, emission and the FSSM program extent treat it as `s_branch`. Test `far_jumps_are_direct_branches`.
  - **Translation time**: `term()` and the block emission found a block's instructions by scanning all instructions
    (`insts.iter().rev().find(…)`), called for every block of every candidate span – `ActivateRays` (76 699
    instructions, 15 995 blocks, pc blocks) did not finish in 10 minutes; a binary search (`insts_in`): 4.7 s (debug).
- **Survey of the whole pack: 26 698 / 26 698 FSSM variants in full** (PS 12 101, MS/NGG 9 282, VS 2 236, CS 2 689,
  GS 195, ES 195), 44.4 million instructions, 0 errors; 90 s (debug, all cores – was 110 s for 24 064).
  `XCL_SURVEY_FIND=text`, `XCL_SURVEY_SKIP=a|b`, `XCL_SURVEY_TRACE=1` (time and result per program); test
  `retail_swizzle_and_setpc_census` lists the `ds_swizzle` offsets of the pack and the code around `s_setpc_b64`
  (`XCL_NAMES`, `XCL_RAW_OUT` dumps the program).

## Translator survey over the whole pack (session 5, 2026-11-21)
- Test `retail_translation_survey` (ignored; `XCL_SAMPLE_RPF=…\shaders.rpf`, `XCL_SURVEY_EVERY=N`, `XCL_SURVEY_DXC=1`):
  - **translation: 24 064 / 24 064 FSSM variants in full** (PS 12 101, MS/NGG 9 282, VS 2 236, GS 195, ES 195,
    CS 55), 28.97 million instructions, 0 missing instructions, 0 errors, 0 panics; 110 s (debug, all cores);
  - **DXC (every 100th record, 241 variants): 0 errors**; time median 0.26 s, p90 3.0 s, max 9.1 s
    (PS_PedTextured_Eight_Fire), total 224 s → **whole pack ≈ 6 h of CPU** (on the order of 30 min on all cores, if
    there is enough memory). The first game start without a ready DXIL cache would be very long → background compilation +
    a cache are needed (`dxc::compile_cached`, key = HLSL), or filling the cache from `shaders.rpf` beforehand.
  - **Random 1% samples** (`XCL_SURVEY_SEED=<n>|random`, the seed is printed): seed 1795262526659555600 found a DXC
    error – `image_sample_d(_cl)` on a **cube texture**: RDNA supplies 2D gradients in the face's coordinate system, HLSL
    `TextureCube.SampleGrad` wants 3D and has no version with clamp → now LOD = log2(max |grad| · face size) (with clamp)
    and `SampleLevel` (an approximation; the face gradient scale is to be confirmed in the game). After the fix this seed and a new one
    (1795262814426998400): 0 errors. The every-100th-record sample did not hit this – also check randomly.
  - Compile MS (NGG) and GS/ES as **cs_6_6** (they run as compute + a pass-through mesh shader; the ms/gs profile →
    "WaveSize only allowed on compute").
- **DXC time optimization (session 5, second round)** – comparison on the same sample (every 100th record, 241 variants), old
  / new code one right after the other: total **228 → 152 s (−33%)**, median 0.29 → 0.22 s, p90 2.76 → 2.29 s, max 9.3 → 5.9 s;
  slowest PS (PS_PedTextured_Eight_Fire, 8 600 instructions) ~6.5 → 3.4 s, MS (MSNC_VS_PedTransformSkinAlpha) 3.8 → 1.7 s.
  Time measurements taken at different moments jump by ±20% (machine state) – compare only side by side.
  - `dxc.exe -ftime-report` (SDK) showed: at -O0 DXC still takes ~4 s (so it is not the optimizer), at -O3 the most expensive
    was **Jump Threading** (25%) on the `pc` loop, and in MS shaders **loop unrolling** from helper functions.
  - **Structure:** forward jumps leaving a block (break out of a loop, a jump past an exec join, s_endpgm in the middle – discard
    in PS) as **`xcl_skip_<target>` flags**; loops entered in the middle (`s_branch` right before the head, also via `s_nop`) in
    rotated form; if/else only when no pending target falls into the middle (otherwise a plain if / flag); analysis and emission
    go through the same function `item_at`, analysis with memoization (`memo`). If/loop depth ≤ 24 (`MAX_NEST`; discard chains
    in PS produced hundreds of nesting levels). Shaders with `pc` blocks: 102 → 23 / 241 (remaining pattern: a loop exit
    placed in the middle of the loop and a back jump to it).
  - **Loop-free and branch-free HLSL helpers** (HLSL 2021 turns scalar `?:`, `&&`, `||`, `if` into jumps):
    `fmt_comp`/`fmt_bytes`/`bits_at`/`uf_small` (buffer format decoding – 140 times each in an MS), `mul_legacy`
    (1 200 times in a PS), `ballot`, `first_lane`, `cvt_u32/i32`, `bfe_*`, `sbfe_*`, `ffbh_i`, `alignbit`, `img_cls`,
    `dst_sel`, `f16_rtz`. Note: `unorm`/`snorm` are HLSL keywords (do not name variables that way).
  - **DXC on its own threads** (`dxc.rs`: a pool, one thread per core, 64 MB stack, its own compiler): DXC's LLVM passes
    recursively overflowed the 2 MB stack on large PSs (PS_TexturedLit_8); game threads may have a small stack.
  - Unsuccessful (no gain, within noise): computing `lanebit(exec)` once per exec change, removing comments from the HLSL.
  - Helpers: `retail_shader_cost` (`XCL_NAMES=a,b`, `XCL_DXC_ARGS`, `XCL_HLSL_OUT`, 8 GB limit) and, in the survey,
    `XCL_SURVEY_TRACE`, time split structured / with `pc`, the 10 slowest, the smallest unstructured ones.
- The game **does not use FSPX/ACCC**: all 23 869 shaders in `shaders.rpf` found then (26 503 since the scan fix of 2026-11-22) are XDAT/XBCH/FSSM – FSPX is only FSR4's own
  pipelines (irrelevant for the game, possibly a test).

## Console shaders – open questions (from the analysis of `shaders.rpf`)
- Unrecognized FSSM metadata: +0x10 (always 6), +0x78/+0x80, +0xD0, +0xF8, +0x110 (resembles SPI_SHADER_PGM_RSRC1,
  but the bits do not match), after the code a table of 5×u32 entries (3, PC every 8 B, 1, value, 8 – looks like a PC → source
  line mapping), then the shader name; in PS from approx. +0x238 an export patch table (`exp` variants for RT formats). For user SGPRs
  and inputs a cross-reference with ISG1/OSG1/PSV0 is needed (those parts are readable – the signature parser is in `rpf-viewer.html`).
- GS: GS + VS descriptors (copy shader), but FSSM contains only one program – is the copy shader created outside FSSM?
- VS: at the start `s_swappc_b64 s[4:5], s[4:5]` = fetch shader call (vertex attributes → v4…, VS_BlitGenAmbCube
  confirms POSITION.xy in v4/v5); in VS mode the recompiler has to replace it with reading the inputs according to ISG1 (the VGPR layout
  for multiple attributes to be determined on the corpus: which v≥4 are read before being written vs the ISG1 element masks).
- VOP2 0x06 / VOP3 0x140: in gfx10.1 MAC/MAD_LEGACY (unfused), in gfx10.3 FMAC/FMA_LEGACY – unknown which one the console has
  (the decoder uses the names from the gfx10.3 spec).
- Some names in the pack are truncated (names > 255 characters do not fit in the record's length byte) – RPF9 scanner cosmetics.

## XDefiant (Ubisoft.XDefiant-Game, `rogue_scarlett_rtx.exe`) – first runs (2026-11-21)
The user's dump (Mount folder, game data in `hunter\`, read through `d:\hunter\`). Each step below was a stop
of the title (it writes to address 0 after a failed D3D12 call: `call 0x14068c760` then `mov [0], …`):
- **D: mount**: the package drive of the XDK days; default mounts now have `D:` = `{game}`, and default mounts
  added later are merged into existing `xcl.toml` files.
- **Crash reports**: the title sets its own unhandled-exception filter; kernelx `SetUnhandledExceptionFilter` keeps
  it behind XCL's (report first, then the title's). The record has all registers, the module of each register's
  vtable (which object it points at) and return addresses into loaded modules on the stack.
- **CheckFeatureSupport(D3D12_OPTIONS)** with the console's 0x40 bytes (PC 0x3C → E_INVALIDARG): retried with the PC
  size, the extra bytes zero (`dev_check_feature_support`, a size table from the PC structures).
- **Device #25 GetResourceAllocationInfo called the C way**: (return buffer, device, mask, count, descs) – the PC
  runtime took the device as the return buffer and wrote SizeInBytes = −1 over its vtable. `get_resource_allocation_info_x`
  takes whichever argument is an adopted object as `this`; textures get XG's size.
- **The DMA list's vtable** (ID3D12XboxDmaCommandList, created through device #73 with type 0x10): Close #9, Reset #10,
  then CopyBufferRegion #11, CopyTextureRegion #12 (XDefiant: 8×8 BGRA mips from an upload buffer), CopyResource #13,
  CopyTiles #14; later slots unknown (logged stubs). Its own `Kind::DmaList` class.
- **D3D12Core writes its vtable into a copy list again on Reset**: XCL's adoption was lost after the first Reset;
  `reset_x` puts XCL's back.
- **Device #27 CreateCommittedResource from console code**: buffers go to unified memory (as #76); textures get UMA
  memory of the console size and become title-memory textures (`place_console_texture`), their GetGPUVirtualAddress
  is that address – XDefiant clears a 32 MB texture with memset through it.
- **Command lists of console code** (`title_list`): the SDK's inline writers put records into the list object when
  its buffer has room (+0x18 write pointer, +0x30 end, +0x28 dispatch limit) and otherwise call the method the
  record stands for (0xC000B600 + topology → #20 IASetPrimitiveTopology, 0xC000B800 → the method at vtable +0xC0).
  Console code gets a console-shaped object whose buffer never has room: every inline write becomes the method call
  on the adopted PC list behind it (a thunk per slot). `XCL_LIST_TRACE=1` keeps the last 64 slots called and logs
  them when Close fails.
- **Compute root arguments of console code** (`bind`): SetComputeRootSignature / tables / constants / root
  descriptors (#33–#45) are packed by the root signature's layout; a Dispatch (#14) with a translated pipeline binds
  XCL's ABI root signature, writes the user SGPRs and the table to unified memory (`user_data`) and dispatches; that
  memory is freed when the list's allocator is reset. A dispatch whose pipeline has no translation is skipped.
- **ResourceBarrier** (#26): aliasing barriers become ones for any resources (both pointers null) – the PC allows
  them only between placed / reserved resources, and the console's texture memory is mostly XCL's committed
  resources (the list failed to Close with E_INVALIDARG); console state bits the PC does not know are dropped.
- **Result**: the title runs – in one minute 957 presents, 3 443 pipelines from translated console shaders, 966 of
  966 programs translated, no failed pipeline, no crash. Whether the picture is right is not checked yet.
- **Graphics root arguments** (`bind`, measured with the console compiler – `print_graphics_root_argument_placement`
  / `_capacity`): a VS / PS gets root arguments in user SGPRs **s8…s31** (s0…s7 the driver's, s[4:5] of a VS the
  fetch shader), 24 SGPRs, the rest in the table at s[0:1] (`Layout::inline_gfx`, `user_data_gfx`). Set*Graphics*
  (#34–#46) are kept like the compute ones; DrawInstanced / DrawIndexedInstanced (#12 / #13) with a translated pipeline
  bind the ABI root signature and io first (a pipeline still building is waited for).
- **Resources in title memory**: Map of a resource in unified memory returns its console address (also for textures
  the PC cannot map); a buffer at an address that is not 64 KiB-aligned is placed from the 64 KiB below it and XCL adds
  the difference in CopyBufferRegion / CopyTextureRegion / CopyResource (`buffer_delta`); graphics reservations come
  from unified memory when it has room (`Uma::reserve`, not zeroed).
- **Why XDefiant's picture is black (open)**: its main pool is one **8 GB reservation** (VirtualAlloc flags
  0x20402000 = MEM_RESERVE | 0x400000 | MEM_LARGE_PAGES, no MEM_GRAPHICS – `XCL_TRACE_MEM=1`), committed in 64 KiB
  steps, and it places D3D12 resources in it: on the console all of it is GPU-visible. XCL's unified memory is at most
  4 windows of 4095 MiB, each fully committed (D3D12's rule for OpenExistingHeapFromAddress), and translated shaders
  reach memory only through those windows – the pool does not fit (and 12+ GB committed is near this PC's commit
  limit). Options: a much larger UMA (commit cost), or a GPU page table in translated shaders (title memory stays where
  it is, per-chunk heaps) – the general one, larger work. Until then Map of such a buffer fails and the title stops.
- **Aliasing of placed textures** (`vram`): the title places resources at addresses that already hold one – 16 in
  its first seconds (render targets 3840x2160 / 1920x1080, UAV textures, other formats at the same address; logged by
  `note_alias`). GPU-owned textures (render target / depth / UAV flags) now go to *arenas*: default heaps standing for
  ranges of title addresses (from the 32 MiB below, at least 128 MiB), placed at their address's offset, so resources
  at one address share memory. ~16 arenas in XDefiant's first 25 s. Other textures stay committed and mirrored.
- **XDefiant now (2026-11-22, later)**: its title screen is on screen – logo, "Welcome, Player1#0001", "press A to
  start / change account" (in the system's language), ~40–50 frames per second there (release build). After a
  change to the translated pixel shaders' HLSL the DXIL cache misses: ~1 500 pixel shaders compile in the
  background over two or three runs, and until a pipeline is built it draws nothing (black or half-drawn frames –
  check "new dxil" before blaming a change).
- **Seams along triangle edges (fixed)**: dotted lines on the diagonal of the logo's quads. The console takes
  derivatives across a quad's lanes by hand (DPP quad_perm, ds_swizzle) and reads the helper pixels of quads a
  primitive's edge splits; `rla` read lanes with WaveReadLaneAt, which leaves helper lanes out before SM 6.7. Pixel
  shaders now read lanes of their own quad with QuadReadLaneAt (quad operations include helper lanes) – `XCL_PIXEL`
  in the prelude; `console_compiled_derivatives_hold_on_triangle_edges` (two triangles, ddx_fine / ddy_fine: 48 of
  1 024 pixels wrong before, 0 after). Do NOT use `[WaveOpsIncludeHelperLanes]` (ps_6_7): tried, the device hung
  ~9 s in – with it the entry `ballot(true)` (the live mask the console code returns to after `s_wqm`) includes the
  helper lanes, so they never stop. Ballots (v_cmp) still leave helper lanes out (matters for branches inside
  whole-quad sections). The console sets IASetPrimitiveTopology 17 (rectangle lists) for these quads; the AMD driver
  draws them as two triangles here – rectangle lists are not modelled otherwise.
- **Helper lanes of other quads (fixed, 2026-11-22)**: Minecraft's Ore UI (create-world screen) had white dashed
  bands along the shared diagonal and the edges of its image and text quads, in runs as wide as a wave. Its UI pixel
  shader (Renoir) reads the draw type with `v_readfirstlane` after `s_wqm`; the first lane of EXEC is often a helper
  lane on a primitive's edge, and a wave read of another quad's helper lane is undefined before SM 6.7 (0 in
  practice): the wave's other quads took the type-0 branch (a solid quad in the vertex colour). `rla` now takes
  such a lane's value with a quad read inside its own quad and a wave read from a live lane of that quad. Also from
  this work: an ISG1 input the VS never reads gets no fetch-shader VGPRs (`fetched_registers` – the create-world
  screen drew nothing); a placed texture larger on the PC than its console size gets memory of its own (Minecraft
  ended with 0xC0000409); PLS is a drive root (`S:\`) and the disk-space APIs answer for console paths ("very little
  free space"). Diagnostics: `XCL_TRACE_SHADERS`, `XCL_TRACE_MANY`, `XCL_FRAME_DUMP_FULL`, `XCL_DEBUG_PS_SUB`,
  `XCL_DEBUG_PS_EXPR`, `XCL_DEBUG_MAGENTA`, `XCL_DEBUG_LOD0`, `XCL_DEBUG_NO_STENCIL` / `NO_CULL`, `XCL_PRESS_AT`.
- **Minecraft is playable (2026-11-22)**: a world is created, loads and runs; the Ore UI screens take the gamepad.
  - Ore UI ignored the gamepad: Minecraft marks a controller connected only from
    `XUserRegisterForDeviceAssociationChanged` (device arrival through GameInput just attaches it to the user's
    player), and Ore UI's `GameControllerInputHandler` drops events of a controller it never saw connect. XCL now
    calls that callback (gamepad: no user → the local user) when the user reaches the title, on the title's task
    queue – through the PC XAsync object when that owns the queues (`xasync::submit_callback`).
  - The world flickered: `bind_abi` put each draw's io / argument table / res in three `Uma::alloc_in` blocks, and
    every block is a 64 KiB grain – thousands of draws a frame filled the 2 GB in seconds, then most draws were
    skipped ("unified memory is full"). Per-draw data now comes from an `uma::Arena` (256 KiB blocks cut into
    256-byte pieces, freed with the command allocator as before); 16.7 M draws, 0 skipped.
  - Keyboard and mouse play as the gamepad (`hle::kbm`, `keyboard_mouse` / `mouse_sensitivity` in `xcl.toml`):
    WASD left stick, mouse (Raw Input on the XCL window, cursor held there; Alt+Tab frees it) right stick by speed,
    mouse buttons triggers, wheel bumpers, Shift / Ctrl stick clicks, E / Q Y / X.
  - Diagnostic: `XCL_PROBE=rva,…` (up to three) – hardware breakpoints on the title's code, each call with new
    arguments logged; XCL's own short-lived threads that call into the title arm themselves
    (`eh::arm_current_thread`). This is how the controller path above was found without changing the game.
  - Open: small magenta / black "missing texture" quads on the loading screen's logo (a texture the game did not
    load – trace file opens next).
- **Shader compilation in the XCL window's title; DXC with little commit left (2026-11-22)**: while DXC compiles
  programs the disk cache does not have, the title reads "XCL – <game> | compiling shaders: N running, M done"
  (`present::show_compiling`, `dxc::compile_counts`: every cache miss, built in the background or in the title's
  call; it stays 2 s after the last one). XDefiant from an empty cache: ~1750 shaders in ~2 min 15 s. Before,
  `dxc::wait_for_memory` held each compile up to a minute while less than 6 GB of commit was left, and with
  XDefiant's own ~10 GB committed only ~4 GB were: its last ~160 shaders came a few a minute (it looked hung at
  ~1600). Now a compile goes at once with 6 GB left, else alone (one at a time). After a cold-cache run
  XDefiant's logo / UI text was missing at times. Cold-cache runs (empty `XCL_DXIL_CACHE`, 1 756 shaders) show
  the logo at ~70 s and the text ~15 s later, by themselves; draws wait for pipelines still being built
  (`resolve_wait` in `bind::graphics_bound`, 0 draws with stand-ins). The text missing for good (also warm) on
  2026-11-23 was the #107 regression below, not the cache.
- **The title's graphics memory in video memory (PR #40, 2026-11-22)**: XMem graphics allocations, physical pages,
  D3D12 buffer heaps and the console descriptor heaps live in the sparse windows 0..2 (`Uma::for_runtime`); window 3
  is 512 MB of coherent system RAM (XCL's arena pool, readback / write-back heaps). GPU writes reach the CPU's view
  only for the lists' memory operations (Write32/64, Fill, CopyMemoryToMemory, timestamps) and for a Map with a read
  range. Fixed on top: window 0's descriptor range (512 MB after its first 64 KiB, `Uma::desc_range`) is never the
  title's. XDefiant reserves 8 GiB + 4 MiB first thing, more than windows 1..2 hold; with the range at the top of
  window 0 that reservation took all of it, no console descriptor heap could be made ("no room in UMA") and every
  frame was black. Tested: Minecraft, Unicorn Overlord, Fortnite, XDefiant.
  - Committed console textures (`real::place_console_texture`) take the coherent window only while it keeps
    `COHERENT_KEPT` (128 MB) free; beyond that they are plain PC textures. Unicorn Overlord makes 2.2 GB of them
    and had the 512 MB window full in its first second (no room for readback heaps); it now stops at 384 MB, the
    picture unchanged. Putting them in the sparse windows instead made XDefiant render black (cache warm, no errors
    in the log) – not understood yet; the GPU-owned ones go to `vram` arenas either way.
  - A committed console resource XCL does not place is created with the PC's description
    (`device_x::pc_resource_desc`: the driver's alignment, its layout, console flag bits off). With the smaller
    coherent window Fortnite's menu reached that path for its 3264x1836 scene targets; the console desc gave
    E_INVALIDARG and the game crashed reading the missing resource.
  - XDefiant's menu runs at ~53 fps (~50 before PR #40). Its CPU cost of the sparse windows is the per-submit
    write-watch scan over the mapped tiles (~2 GB): ~0.8 ms per submit at ~800 submits a second, then ~0.3 ms of
    uploads (64-byte chunk runs of ~0.5 KB on average). Merging the runs or skipping scans would be faster, but a
    merged run can overwrite what the GPU wrote next to a CPU-written chunk, and a skipped scan leaves the GPU stale
    data. Read-only protection of cold tiles would make the scan cost follow the written pages, but the kernel's own
    writes (ReadFile into title memory) then fail instead of faulting.
  - Diagnostics: `XCL_TRACE_MEM=1` also logs each coherent-window allocation of 1 MB or more and each failed one,
    with the XCL code that asked (`Uma::alloc` is `#[track_caller]`); the log has "N frames presented, X fps over
    the last 300" for titles that call PresentX.
- **Unicorn Overlord (SEGAofAmericaInc.Pa253b0c1Y 1.0.0.6) is playable (2026-11-22)**: logos, movies, menus and play
  with the right picture. It runs on a **newer console D3D12 runtime** (its own game OS) with the same SDK version
  (0x20004) as XDefiant / Minecraft but other vtable layouts:
  - command queue: two more methods before Signal – Signal #16, Wait #17, GetTimestampFrequency #18, SignalX #25,
    PresentX #32 (older: #14, #15, #16, #23, #30). Its Signal landed on GetTimestampFrequency, which wrote 100 MHz
    over the fence's vtable (crash). `real::queue_dispatch` handles the title's queue slots from #14; a fence passed
    to #16 / #17 (never to the older layout's GetTimestampFrequency / GetClockCalibration) marks the newer layout.
  - command list: console methods from CopyTextureRegionX on two slots later (#85, older #83; `NEWER_LIST_SLOTS`,
    picked at each call by `guard::make_layout_split`, xmm arguments intact). Its texture uploads hit
    SetPredicationBufferX ("not 8-byte aligned … not done") and every texture stayed zero (shapes, gradients).
    Where the shift starts below #83 is not known.
  - the device layout is unchanged (#73 CreateCommandListX, #78 SetFrameIntervalX …).
  - Also: console-only resource state bits (0x100000 on its 4K display target) are masked off at creation (the PC
    refused the resource, the title threw); the title commits 2046 of 2048 MB of unified memory at start, so XCL's
    per-draw arena has its own 64 MB pool at the end of window 0 (`Uma::reserve_pool`), draws are no longer skipped;
    `FENCES` is a read-write lock (`end_title` reads it with the title's threads suspended).
  - Diagnostics: `XCL_WATCH=rva` (write watchpoint in the exe, the calling thread armed at once), `XCL_WATCH_FENCE=n`,
    the device's SDK version and console-only interface requests logged, the title's file opens with their flags,
    DirectStorage requests, unified-memory use, and (trace) whether title textures hold data in memory.
- **Why every frame was black: descriptors are values.** XDefiant makes its views in a CPU-only heap (131 072
  slots) and copies them into its shader-visible heap (442 368 slots) as memory – the console header's inline
  CopyDescriptorsSimple; only sampler copies reach the device. T# word 0 was an index into each heap's own PC heap,
  so every copied T# named a descriptor of the wrong heap (empty): shaders read zeros, UAV writes went nowhere – the
  presented buffers stayed 0 including alpha. Now all console view heaps share one shader-visible PC heap
  (`console_heap`, `Global`: up to 2^21 descriptors, OPTIONS19 says this GPU takes 32 M): each heap gets a range,
  T# word 0 is an index into the shared heap (a copied T# still names its PC descriptor), lists bind the shared
  heap whatever the title names (`list_x::pc_heaps`: console #32 and the console-list replay), the title's own PC
  heap is a CPU-only mirror (views are made there and copied over – `publish`; copies come from it; UAV clears take
  its CPU handle). A heap that finds no room keeps its own PC heap (the old way). Copies follow the source PC
  descriptor: a source slot rewritten later changes earlier copies too (not seen).
- **Speed**: the per-submit write-watch scan covered every watched region (4 GiB of window 0, 12 GiB of sparse
  windows, graphics reservations): ~10 ms a submit, more than half of all time. It now covers only where a CPU write
  matters – the textures kept in step and the sparse tiles with video memory (`title_textures::watch_ranges`,
  `sparse::mapped_ranges`). Use the release build for game runs: the debug build hashes the dirty chunks ~10×
  slower (and the queues then wait on each other for the sparse lock). The log has
  "sparse windows timing: write-watch / lock wait / chunk scan / upload" lines.
- **No Xbox Live tokens by default (2026-11-22, latest)**: `XUserGetTokenAndSignature(Utf16)Async` fails with
  0x80072EFD, as on a console that cannot reach Xbox Live; `xbox_live_tokens = true` in `xcl.toml` brings back
  XCL's local placeholder token (only useful for a backend of your own – Xbox Live and the titles' services
  reject it). Minecraft asked for one for PlayFab (`*.minecraft.playfabapi.com`), got the placeholder, was refused
  and retried its "Microsoft account – connecting" dialog for ever; now it shows its offline message ("we can't
  sign you in … access to Realms, profiles and Marketplace items will be limited", code "jasnoglaz") and goes on
  in the menu. Fortnite asks for RTA (`wss://rta.xboxlive.com/connect`) and still reaches its title screen.
  The requests are logged ("XUser: token #n for …"). XCL does not fake sign-in, backends or licences (no
  TomBox-style local Epic / Xbox Live servers). Minecraft stays a trial (XStore refused).
- **Minecraft (Microsoft.MinecraftBeta 1.18.30.64) reaches its main menu (2026-11-22, latest)**: panorama, logo,
  buttons, the character, music and Enter work; it offers the trial (XStore is refused – no fake licences) and
  stops at the Microsoft-account sign-in (Xbox Live). Run `Minecraft.Windows.exe.original`: the `Minecraft.Windows.exe`
  next to it is a 29-byte patch that makes licence checks return true – not run by XCL work. Fixed (all general):
  - the title's top-level window is the console's screen (`hle/user.rs`): frameless, 1920x1080, client = window
    (WM_NCCALCSIZE), cloaked – a PC overlapped window gave a 1904x1041 client area, Minecraft sized its back buffer
    from it and resolved 1920x1080 into it: the list failed to close, the device was removed;
  - ResourceBarrier: VERTEX_AND_CONSTANT_BUFFER / INDEX_BUFFER bits are dropped for non-buffers (the console takes a
    generic-read combination for textures; the PC list fails to close);
  - kernelx `FreeLibraryWhenCallbackReturns`: a no-op for XCL-mapped modules (msvcp140's thread-pool callbacks:
    0xC000070E in 20 workers, then a fail-fast);
  - `Uma::is_title_address` / `list_x::pc_va`: a GPU address the title gives is translated only when it is the
    title's memory – a PC resource's GPU address can fall in a window's range by number (Minecraft's 4 MB dynamic
    vertex buffer was read from an empty sparse window);
  - buffers in the title's heaps live in unified memory (`real::note_heap_uma` / `heap_block` /
    `dev_create_placed_resource_x`): the heap gets a block when its first buffer is placed, the buffer is placed
    at block + offset. Console code reads root CBVs and V#s from unified memory only – Minecraft's constant buffers
    were PC-only and every draw read a zero transform (nothing drew). Eager blocks for every heap filled the 2 GiB and
    draws were skipped for want of argument space (the image flickered).
  Diagnostics: `XCL_TRACE_FRAME` logs PSO states (depth / stencil / blend), resolves, ClearDepthStencilView,
  vertex-buffer bindings (address, window, memory state) and the fixed-function state per draw kind;
  `XCL_TRACE_UI_ANY=1` traces draws into any target; `XCL_TRACE_MEM=1` also logs graphics reservations.
- **PS attributes / VS params are registers (2026-11-22, latest)**: Fortnite's loading art ("ŁĄCZENIE"), title
  screen (FORTNITE logo, "press to start", version, Epic/Unreal logos) and UI text now draw. Cause: `exp param N`
  (VS, MS) and `v_interp attrN` (PS) name a signature *register*, not an element, and the console's compiler
  numbers only the registers its code still uses. XCL counted ISG1/OSG1 elements in register order on both sides –
  right only while VS and PS signatures were alike and unpacked (the two errors cancelled). Fortnite's Slate PS has
  COLOR0, ORIGINAL_POSITION0, TEXCOORD0, TEXCOORD1 and samples attr1 = TEXCOORD1; XCL sampled at the pixel
  position (white 1x1 quads still looked right, every real texture came out as its empty corner). XDefiant packs
  elements into registers (UI: UserBig/UserSmall/RenderType in one register) – its UI read ClipPosition as UV and UV0
  as colour before. The truth is in the FSSM (`recomp::ps_input_table`, `vs_output_table`; checked on every cached
  shader of three titles): PS 32 × 8-byte records at +0x130 (kind, register, mask, attr, 0, 0, register, 1 flat /
  2 interpolated), count at +0x100 after them, SPI_PS_INPUT_ENA at +0x110; VS 32 × 6-byte records at +0x1D4 (kind,
  register, param), count at +0xC0. FSSM sub-version (+0x10) 5 = Fortnite's GDK, 6 = XDefiant's / Minecraft's,
  8 bytes later – XCL read SPI_PS_INPUT_ENA at a fixed +0x248, wrong for sub-version 5 (Fortnite's PS got
  "sample" interpolation; the title screen's gradient was flat). The VS now writes channel c of param N into
  component c of every element at that register, the PS reads channel c from an element there; the MS export
  likewise; struct declarations and the PC linkage (by semantic) are unchanged. A PS input struct of only the read
  inputs is refused by the driver (it links by register) – keep declaring every input.
- **XDefiant "black after a translator change" is a cold DXC cache, not a bug**: the title draws nothing until its
  pipeline set (1 000+ PSOs) is built; after a change to the translated HLSL every one is compiled again (minutes).
  Test with `XCL_DXIL_CACHE=<empty dir>` to see it. The crash it exposed (exit 3, AV in a memcpy from
  `sparse::before_execute`) was real: `kernelx::VirtualFree` decommitted before telling `sparse`, and pages
  decommitted inside a still-mapped tile were read. Now `sparse::decommitted` runs first (under its lock) and
  `sparse` keeps holes (uncommitted pages of mapped tiles: not read, their video memory zeroed).
- Diagnostics added: `ntdll!NtTerminateProcess` hook logs every process end with the calls that led there (and a
  thread dump when the status is not 0; abort() left no trace before); `XCL_TRACE_FRAME` logs one line per distinct
  (VS, PS, render target) drawn with the shaders' dump hashes, GPU read-backs of sampled title textures, copies into
  title textures. Also: title textures a GPU copy wrote are no longer uploaded again from their stale CPU view, and
  empty sampler slots hold the console's zero S# (point, wrap, mip 0). Note: `cargo test --release` uses
  `target/release/xcl_testtitle.exe` – rebuild it (`cargo build --release -p xcl-testtitle -p testgame-mw`) or e2e
  checks a stale title.
- **Fortnite (XSX 2.0.1461.7919, an older GDK than XDefiant's) now (2026-11-22, later)**: loads (~60 s, its own
  UE log in `[debug]` lines), hides its loading screen and shows the splash screen ("press to start") at ~50 fps –
  2 850 draws and 4 700 dispatches every 2 s, all translated – but the image is one solid blue (0xed242424): the UI's
  background without its text and images. A (GameInput) closes the splash and starts the login: it stops at once at
  the console-privileges step (Xbox Live), shows its error box – a white rectangle, again without text. The login
  is an Xbox Live sign-in (XCL does none), so that is as far as the title goes. Next for the image: why UI text and
  images draw nothing (the title textures are not refused – only 1x1 / 3D defaults lack an XG layout). Note: the
  title talks to Epic's services over WinHTTP from the start (MCP "Connected") – the network is the PC's.
- Before that: the garbage collector ended the game at ~41 s – 3 639 textures waited for one render command fence
  (all `IsReadyForFinishDestroy` false, `ReleaseFence` not complete) because the render thread spent its time in
  CreatePlacedResourceX: every title texture was read, detiled and uploaded with a GPU wait as it was made. Now the
  first upload is deferred to the title's next submit and all uploads go in batches (one list, one wait each,
  256 MB of staging at most: `title_textures::upload_all`, `texture::upload_batch`); XCL's own submits do not
  refresh (they read no title memory), and write-watch ranges closer than 1 MiB are scanned as one.
  Fixed on the way (all general):
  - **Run from `<Game>/Binaries/XSX/*.exe`**: the package root is the folder above whose MicrosoftGame.config
    names the exe (`launcher::config_root`) – D: and the title's relative paths start there.
  - **XMemGetWorkingSetStatistics wrote 0x78 bytes**: the structure differs between GDK versions; EOSSDK (older)
    has ≤ 0x50 bytes before its stack cookie – its DllMain static initializer failed the cookie check
    (0xC0000409). Only +0 limit / +8 used are written now.
  - **TLS slot 0**: the process's executable has TLS slot 0 on Windows and code built with /GA reads
    `ThreadLocalStoragePointer[0]` without `_tls_index` (Fortnite: the guard of its thread-safe statics – its
    allocator GMalloc was never created). The title's executable now takes slot 0 and xcl.exe's own TLS moves to
    the carrier's slot (all 78 TLS accesses in xcl.exe go through `_tls_index`): `tls::take_slot_zero` (running
    threads get xcl's block in both slots first, then the title's in slot 0; new threads exchange in xcl's TLS
    callback before anything reads TLS; TEBs come back, so the list is kept until thread exit).
  - **Command queue #23 SignalX(fence, value, flags)** was a stub: the render fences never completed.
  - **CreateHeap / CreatePlacedResource (#29)** with console values (heap flag 0x2, buffer flags 0x20004 and layout
    1): refused by the PC – retried the PC's way (known heap flags 0x3FED, PC alignments, `pc_resource_desc`,
    console state bits dropped); UE took the refusal for a lost device.
  - **VS + GS of an older kit**: the VS container's ES variant ends with `s_setpc_b64` into the GS (merged ES+GS),
    not `s_endpgm` – the FSSM parser found no program. Both the header check and the segment split take that end
    now (`xbox::is_setpc`); the translator already continued ES into GS at the jump.
  - **Ending a run hung the GPU** (AMD driver dialog, Fortnite under load): `end_title` set every title fence to its
    maximum while the title ran on – UE then reset command allocators the GPU still ran. The title's threads are
    suspended first now (a watchdog ends the process after 15 s if anything blocks).
  Diagnostics added: stack-cookie failures are reported with the function (`eh::hook_gs_failure` – both the PC's
  and the console's `bnd`-prefixed `__security_check_cookie`), the UCRT invalid-parameter handler logs its caller,
  first-chance C++ exceptions thrown by title code (type, what(), stack), THREADNAME_INFO (0x406D1388) names
  threads, `XCL_THREAD_DUMP_AT=s1,…` (every thread unwound with the unwind data), `XCL_ACTIVITY=1` (every 2 s:
  presents, submits, draws, dispatches, frame-event waits, frame statistics), GPU value writes into sparse windows
  logged (the CPU's view does not see them – not what stops Fortnite).
- **Controllers** (`hle::gameinput`): GameInput.dll was Windows' own (other interfaces, reads only for its own
  foreground app) – the title never saw a button. Now XCL's: the GDK's v0 interfaces (IGameInput, IGameInputDevice,
  IGameInputReading in the public header's order), one gamepad always connected, read from the PC's first XInput
  controller plus the keyboard while a window of the process has the focus (Enter / Space A, Escape / Backspace B,
  arrows D-pad, Tab View, P Menu); device callbacks (blocking / asynchronous enumeration), reading callbacks
  (polled at 250 Hz), UI navigation state, rumble through XInputSetState. XDefiant imports GameInputCreate by
  ordinal #1 (`stubdb.toml`). `XCL_PRESS_A_AT=s1,s2…` presses A for 200 ms at those seconds (tests). With it the
  title screen takes A and goes to a loading screen (spinner) – it then waits, most likely for Ubisoft's services
  (the network gate, next).
- **Audio processor** (`hle::acphal`, routes acphal.dll and xapu.dll; chad.dll / XDsp stays E_NOTIMPL): APU memory
  (32-bit APU addresses for ApuAlloc / ApuMapVirtualAddress), AcpHalCreate's channel (commands acknowledged at once,
  an audio frame message every 2.667 ms), SHAPE context arrays, the xapu decoder answering with silence. Before,
  `ApuCreateHeap` got E_NOTIMPL and XDefiant dropped its whole audio setup (+0x7639ef). Now it creates a 350 MB APU
  heap, connects the decoder (320 streams), SHAPE contexts [512, 0, 0, 1024, 512] and the channel (commands 2 and
  8). Its audio is still silent: it mixes through SHAPE (not XAudio2 – only bink2scarlett.dll imports XAudio2),
  and SHAPE contexts are not run; that is the next step for sound (context layouts unknown).
- **What made the GPU hang** (four PC reboots – the AMD driver did not always recover from the TDR): vertex buffers.
  Console #48 IASetVertexBuffers gave the title's addresses to the PC as GPU addresses – the index buffer was
  translated, the vertex buffers were not (NaN in every render target, then device hangs). `ia_set_vertex_buffers_x`.
- **Console-only list interfaces**: the title queries its lists for the console headers' ID3D12GraphicsCommandList5 /
  6 – 55050859-4024-474c-98cf-645b51dc4874 / c3827890-4024-474c-98cf-645b51dc4874 (PC GCL5 / 6 end 87f5-6472eaee44ea /
  e548-4cfa-96cf-5689a9370f80) – and copies the answer into its command context unchecked (null → crash after ~25 s).
  The console list object answers them with itself (`title_list::CONSOLE_LIST_IIDS`).
- **Console list #83 = CopyTextureRegionX**(dst, x, y, z, src, box, flags) – the title calls it where the DMA list
  gets #12; it was a stub (texture uploads through the graphics list were lost). **#59 ExecuteIndirect** with a
  translated pipeline binds XCL's ABI first (`bind::execute_indirect`; signatures that change root arguments skipped).
- **Sparse windows** (`d3d12x::sparse`, `gpu::uma`): the title's own reservations of 64 KiB pages (VirtualAlloc
  MEM_RESERVE | MEM_64K_PAGES, XDefiant's 8 GB pool) get addresses in UMA windows 1–3 (reserved only); on the GPU each
  window is a 4 GiB reserved buffer whose 64 KiB tiles are mapped onto video-memory pool heaps as pages are committed.
  The CPU view stays the title's memory; before each ExecuteCommandLists the CPU's writes (write-watch) go to the tiles
  – only the 64-byte chunks that changed (`changed_chunks`: hashes of what was sent), so what the GPU wrote on the same
  page is not overwritten – and the title's queue waits for that copy. Buffers placed there get reserved resources over
  the same tiles. GPU writes are not copied back to the CPU view. Measured first (`tests/heap_probe.rs`): tiles of a
  reserved buffer on default heaps work for a whole 4 GiB window; tiles mapped onto heaps opened from file mappings
  removed the device (do not try again).
- **Graphics root arguments** measured with the console compiler (`print_graphics_root_argument_placement` /
  `_capacity`): VS / PS get them in user SGPRs s8…s31 (24), the rest in the table at s[0:1]; `bind` keeps the graphics
  ones and binds them on draws; checked end to end by `console_compiled_graphics_reads_its_root_arguments`.
- **Ending a run without a stuck process**: a host killed from outside with GPU work in flight stayed in the driver
  holding 12–16 GB. `XCL_RUN_SECONDS=n` ends the title itself: every fence the title created is signalled to the
  maximum (GPU waits on future values complete), the queues drain, then TerminateProcess (`real::end_title`). Shader
  compiles: at most 6 at once and they wait while the system has < 6 GB of commit left (`dxc::wait_for_memory`).
- **Window**: titles create their window with CreateWindowExA / RegisterClassExA but DefWindowProcW (its title showed
  as CJK characters); `hle::user` makes them Unicode calls. The window stays white (console titles never paint it).
- Diagnostics: `XCL_TRACE_FRAME=1` (what console code does with each presented resource; render-target binds, UAV
  views of large render targets, clear colours; with `XCL_FRAME_DUMP` also the busiest render target and raw texel
  histograms; the console descriptor heaps made, the device's descriptor copies, a census of what each view heap's
  slots hold – T# with their own index / another (copied as memory) / buffers – and, for full-screen dispatches, the
  root arguments and constant-buffer words with the resources their small values index in the dispatch's tables), `XCL_TRACE_ADDRESSES=1` (root descriptor addresses in / outside unified memory), `XCL_DRED=1` (DRED
  page fault + breadcrumbs on a device removal – opt-in: twice the title stalled with it), `XCL_LIST_TRACE=1`.
- **Notes from another implementer** (user-provided): XDefiant also calls XGCreateTextureComputerFromDescriptor, then
  libHttpClient / XCurl (XUserGetTokenAndSignature*, XNetworking) and retries wss://rta.xboxlive.com/connect every
  10–25 s; a 3D 120x68x64 RGBA16F resource / view returning E_INVALIDARG is not null-checked by the title.
- **Decided (user): a GPU page table in translated shaders.** Measured with `tests/heap_probe.rs` (GPU copies only):
  - a heap over a file mapping (`OpenExistingHeapFromFileMapping`) works for **SEC_COMMIT** sections – the GPU reads
    what the CPU wrote and the CPU sees GPU writes; **SEC_RESERVE** sections are refused (E_INVALIDARG) even when
    every page is committed – a heap always costs its full commit, like the address one (fully committed VirtualAlloc
    allocation, < 4 GiB);
  - a reservation made as a **placeholder** (`VirtualAlloc2` MEM_RESERVE_PLACEHOLDER), a chunk split off it
    (`VirtualFree` MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) and replaced by a SEC_COMMIT section mapped at that address
    (`MapViewOfFile3` MEM_REPLACE_PLACEHOLDER): each chunk opens as its own heap, GPU and CPU coherent at the title's
    address, neighbouring chunks independent (`heaps_over_sections_in_a_placeholder`, 2 MiB chunks).
  - Plan: (1) kernelx – title reservations become placeholders, MEM_COMMIT inside turns the touched chunks into
    sections (commit charge per used chunk), each a heap + a raw buffer over it; decommit / release restore the
    placeholder; (2) each chunk's buffer gets a UAV in a part of every shader-visible view heap XCL reserves, and a
    page table (chunk of the address → descriptor index) goes to the shaders; (3) the recompiler's `mem` accesses go
    through the table (`ResourceDescriptorHeap[i]`), the UMA windows stay the fast path; (4) PC-side users of buffers
    in chunked memory (IA vertex / index buffers, copies, clears) cannot span chunks: vertex fetch in the translated VS
    from memory, copies and clears as XCL compute passes over `mem`.
- Diagnostics: `XCL_FRAME_DUMP=1` writes every 120th presented frame (a quarter of each side) as `frame_<n>.bmp` in
  the logs folder; `XCL_TRACE_MEM=1` logs allocations ≥ 64 KiB outside unified memory.
- **Safety nets** (after a GPU hang that needed a PC restart – it came from a diagnostic run that skipped barriers):
  every loop of a translated shader has a per-lane budget (`LOOP_GUARD`, 4 M iterations); when the D3D12 device is
  removed (after ExecuteCommandLists, at PresentX) the title is stopped with the reason in the log and the report.
- Diagnostics added: `XCL_WATCH_DEVICE=1` / `XCL_WATCH_DMA=1` put a hardware write watchpoint (debug registers, all
  threads) on the title's device / first DMA list and log each writer with its callers (`eh::watch_writes`);
  `XCL_D3D12_DEBUG=1` turns on the debug layer and copies its messages into the log (needs the Windows "Graphics
  Tools" feature – not installed on the user's PC); `XCL_D3D12_TRACE=1` (existing) logs console calls by slot.
- Also seen: 1D and 3D 1×1 default textures (XG: E_NOTIMPL for 1D / 3D – not mirrored), console-only list IIDs
  55050859-…-98cf-645b51dc4874 / c3827890-… (the title takes a null), D3D12ConfigureVirtualMemoryX (stub).

## Wave64 on 32-lane GPUs (NVIDIA), 2026-11-23
- `Options::split_wave64` (set from `pso::wave32_only_for_device`, in tests from `runner::options`): each console
  wave of a wave64 kernel / compute / mesh / ES+GS program = two hardware wave32s exchanging through groupshared
  (`PRELUDE_X64`: masks, readlane / readfirstlane, ds_bpermute, GLC scalar loads; DPP / swizzle / permlane stay
  in a half). Group barriers with one console wave per group, pairwise counters with several. Replaces the
  half-local split of #14 / #39 and their refusals. Design + hazards + cost: `docs/wave64-on-wave32.md`.
- Measured on an RTX 5070 Ti: all recompiler GPU tests run (only `gpu_wave32_program_in_wave64_hardware`, which
  needs 64-lane hardware, skips); the kernel runner checks both halves end with the same SGPRs; `XCL_WAVE64_SPIN=1`
  runs every split program in the counter variant (passes too). Native HLSL byte-identical for every program the
  suite translates (dumped and diffed against main). Not re-run on AMD in that session (no AMD GPU enabled).
- NVIDIA forms 32-lane waves of non-consecutive `SV_GroupIndex` for 3-D groups (8×8×4): `xcl_x_thread` gives
  each hardware wave the next 32 console threads as it arrives.
- Cost (`bench_split_wave64`, worst case: one mask per 8 instructions): 1.3× with one console wave per group, ~5×
  with four (waiting waves keep issuing).
- Masks only read lane by lane (`v_cndmask` selectors, `v_add_co_ci` / `v_sub(rev)_co_ci` carry-ins) are no longer
  exchanged: `x64_masks.rs`, a backward liveness pass over the SGPRs ("whole-live": some path reads the whole value
  before it is overwritten; EXEC, indirect jumps, loop exits through the iteration budget and the kernel runner's
  end count as whole reads; geometry mode unchanged). It follows the emitter, not only the operand tables: SDWA
  compares into their SDST, `s_cmov*` / `s_addk` / `s_mulk` / `s_bitset*` and `s_movreld` are not overwrites,
  `s_movrels` reads every register from its base, SDWA scalar sources and SMEM register offsets are reads (each was a
  way for the halves to diverge: tests `gpu_split_wave64_masks_read_whole_after_partial_writes`, `x64_masks::tests`).
  `bench_split_wave64` now times GPU work only (timestamps, `runner::dispatch_timed`; before: wall time with
  pipeline creation and copies). RTX 3090, `XCL_X64_ALL_MASKS=1` vs default: mask-only loop 1.64–1.68× → 1.04–1.11×
  (one console wave per group), 3.90–4.14× → 1.07–1.16× (four); with a readfirstlane too 1.99–2.02× → 1.74–1.81×,
  6.45–6.64× → 4.09–4.34×. Test `gpu_split_wave64_lane_local_masks` (fails if the local word lands in the wrong half).
  Merged 2026-11-23 (#94) after: workspace tests on the RX 9070 natively and with `XCL_SPLIT_WAVE64=1`, and
  XDefiant run in the split form on AMD (`XCL_SPLIT_WAVE64=1`, every wave64 compute / mesh program split): its
  title screen right with a warm cache. With a cold cache the title screen's text comes ~15 s after the logo. Also merged: #92 (the VFS hook guard keeps
  GetLastError – GNU builds' TLS access reset it), #93 (under Wine PEB.ImageBaseAddress stays the real image).
- Still open on NVIDIA: wave64 VS / PS with readlane of lanes 32..63 (`wave_hazards`; the driver picks 32-lane
  waves, lanes ≥ 32 do not exist – `WaveReadLaneAt` wraps to lane & 31 there). Done next (below).

## Wave64 VS / PS in 32-lane waves (NVIDIA), 2026-11-23
- Census of the 749 flagged programs (748 PS, 1 VS; `retail_wave64_graphics_lane_census`, ignored): 1 082 reads in
  436 programs are a whole-wave reduction (identity into the lanes outside the mask, DPP8 xor
  1/2/4, row_mirror, `v_permlanex16`, `v_readlane` 32 for the upper half; min_u32 / max_u32 / or_b32) – right on
  NVIDIA only because lane 32 reads lane 0 and those ops are idempotent; 355 reads in 353 programs are waterfall
  reads (`v_readlane v, s_ff1(mask & EXEC)`), lane < 32 – right; 200 reads in 11 programs are SGPR spill slots in
  lanes 32..47 – wrong (the write is lost, the read gets slot lane & 31). No `ds_bpermute`.
- `upper_lanes.rs`: a forward pass over the program follows what the console's lanes 32..63 hold (EXEC's upper word
  through the SGPRs as 0 / all ones / anything; per VGPR lane ≥ 32 unwritten / a constant / a spill slot / unknown,
  evaluating whole-wave moves, selects, integer min / max / logic / add and their DPP forms). With `split_wave64`
  a wave64 VS / PS reads a constant lane as the constant and a slot from a scalar shadow its `v_writelane`s also
  write, when `WaveGetLaneCount() < 64` (`xcl_hw32`); spill slots < 32 (287 PS, 3 540 reads) come from their shadows
  at any wave size – a lane a partial wave did not launch loses its slot. `wave_hazards` now reports only reads the
  pass cannot follow: 0 in the pack (was 749 programs). Native HLSL (no `split_wave64`) identical for all 26 698
  programs, and so is the 32-lane form of every other mode (digests, `retail_wave64_graphics_lane_forms`).
- Tests on the RTX 5070 Ti: `gpu_wave64_ps_whole_wave_reductions` (the pack's reduction as min and as a sum, and the
  waterfall read, per hardware wave against a console wave of the same lanes), `gpu_wave64_ps_spill_slots` (full
  target and a 6-pixel triangle), `gpu_wave64_vs_spill_slots_and_identity` (3 vertices). With the old form all three
  fail there: slots 32..47 sum to slots 0..15, the sum reduction doubles, the VS keeps only its 3 lanes' slots.
- Not exact yet, on every GPU: whole-wave code in a partial wave reads lanes the PC never launched (the console's
  inactive lanes run it). Measured on the 5070 Ti: such lanes read 0 – a min reduction (identity -1) gives 0 in a
  3-quad wave, max / or / sum are right. 325 of the 436 reduction programs have a min reduction. Needs the reduction
  recognised as a whole (`docs/wave64-on-wave32.md`, "Vertex and pixel shaders").
- Merged as PR #133. On the RX 9070 (64-lane waves) the native-form digest of the 26 698 programs equals `main`'s (the
  forms test run on both); the three GPU tests pass, also with `XCL_SPLIT_WAVE64=1`; XDefiant and Minecraft run with
  and without `XCL_SPLIT_WAVE64=1` (the forced split form starts with a cold shader cache: XDefiant black while compiling).

## Retail game bring-up, 2026-11-23

Runs 45 s without a crash (later update below); what it shows was not looked at yet. **Always run it with `XCL_NETWORK=0`**: its crash reporter sends a minidump
to an outside server at every crash (`hle/netoff.rs`: the title's WinHTTP / Winsock calls fail, loopback passes).
- Done: placeholder `VirtualAlloc` (`kernelx/placeholders.rs`); `XMemMapPhysicalPagesScatter`; physical page numbers are
  4 KiB frames (step 16); the title's `__XMemAllocationHooks__` (exported by it and Fortnite) give descriptor heaps and
  display targets their memory; device #26, #70, #71, #108, #127, #133; DirectStorage's newer factory / queue; pipeline
  streams with console shaders are translated as plain descs (raw, they removed the device); fence #12
  (`fence_x.rs`); 1D / 3D textures in `xg.rs`; the newer list layout's shift by two covers #72 - #84 too and
  is also noticed from device #108 / #127 / #133.
- `xg_oracle` passes with XDefiant's library. Fortnite's is newer (UAVs 4KB_S) and fails two tests, as before.
  `XGSuggestSwizzleMode` is inferred: no library at hand exports it.
- Memory (later the same day): XCL took 28.7 GB of commit and the title died in the driver's pipeline compiler. Measured
  with a VirtualQueryEx walk of the game process: ~5.7 GB of physical pages the title maps into its own 6 GB
  reservation were committed twice (the copy and the home), and every home also had video-memory tiles – 126 pool
  heaps of 64 MB, which this AMD driver shows as private write-combined process memory. Now a home in a sparse window
  is only reserved until the page is used there, gives its memory back while the page is mapped elsewhere (in runs,
  256 MB at a time – page by page the start took 20 s): 16 - 17 GB, first frame at 3.6 s. `XCL_TRACE_MEM=n` logs
  the first n allocations.
- Then: `chad.dll` (XDsp) connects and takes commands (it stopped at a breakpoint on E_NOTIMPL); list #139 of the
  newer layout is Reset (as a stub every list stayed closed: the debug layer refused each call, Close failed); the
  console's rect lists (IASetPrimitiveTopology 17, pipeline topology type 19) are triangles – each rect is drawn as its
  first half until a geometry shader expands it; a stream without a topology type is TRIANGLE.
- Open: memory the GPU does not see (~4.8 GB of the title's own `VirtualAlloc`, the pages in its 6 GB reservation:
  shaders get empty V#s); rect lists; Write64BitValueBottomOfPipeX gets addresses like 0xa (the newer list layout's
  shift probably starts below #72); the debug layer's Map of a DEFAULT-heap resource and a CopyResource source in
  UAV state; queue #35, list #131 / #132 and allocator #9 are stubs (#9 registers the title's command memory callback);
  MSAA layouts. With ~21 GB taken by other programs, 1 - 2 GB of commit stays free during a run.
- Loading (later): it shows 2 frames and then loads for minutes – ~3 500 shaders, 2 076 of them compute (13 M RDNA2
  instructions, 611 programs over 10 000). Shader dumps (and a second translation of each) are now only made with
  `XCL_DUMP_SHADERS=1` (they took 4.8 GB and its loading threads' time); translations are kept on disk
  (`<dxil cache>	ranslated`, by the console bytes and the XCL binary's hash); `XCL_FRAME_DUMP=n`. During a run DXC
  compiles one program at a time (`wait_for_memory`: under 6 GB of commit free; ~11 GB of the system's commit is the
  kernel's and drivers'), so the caches are filled without the title: `xcl shaders <shaders.rpf> --precompile --only cs`
  (compute programs through `pso::console_to_dxil_in`, the title's path: both caches).
- Then at ~183 s: 5 mesh pipelines with an "AS" subobject fail and the title writes to address 0. The AS slot (24)
  holds an MS container (`MSNC_VS_TransformUnlit_*`, NGG) – the console's subobject numbering around AS / MS is to be
  checked (the message now lists the stream's subobject types and both hashes); translation failures are each logged
  once (the first five hid this one).
- Then: 30 fps (1 449 frames in 60 s), image black. Why it showed 2 frames: its present thread waits on frame event
  type 2, scheduled at offset 0 like the origin; the clock picked one event per deadline (`min_by_key`: the
  origin) – now every type due at a deadline fires (`frame.rs`). On the way: queue #35 of the newer layout is
  ExecuteCommandListsX(count, lists, value, flags) – all its command lists went through the stub; list #61 is
  ExecuteIndirect (the shift by two from #59 on; #61 went to FillMemoryWith32BitValueX); bottom-of-pipe writes into
  memory outside unified memory are done by the CPU after the queue's work (`cpu_writes.rs`), waits there read it on
  the CPU; compute pipelines of 256 KB of DXIL or more are built in the background (the driver takes ~30 s for
  its 1 MB ray-tracing resolve programs; 2 at a time, within `wait_for_memory`), and a bind of one not started
  builds it itself (`pso_async`); DXC -O1 from 1 MB of HLSL; `xcl shaders --precompile --pipelines` makes the
  compute pipelines on the GPU too (the driver's cache: 41.6 s then 0.2 s; 4 at a time, `XCL_PIPELINE_THREADS`);
  `XCL_TRACE_FRAME_EVENTS`; the end of a timed run logs the PresentX calls.
- Measured: the memory-window `switch` of every memory access doubles DXC + driver time (one 21 000-instruction CS:
  41.6 s with 4 windows, 22.5 s with 1); heap-indexed windows would remove it but a raw buffer view cannot cover a
  4 GiB window – a redesign of how shaders reach title memory.
- XDefiant, Fortnite, Unicorn Overlord and Minecraft were **not** run after these changes (only the test suites).

## Console CPU instructions the PC lacks (2026-11-23)
A retail game on an Intel CPU before Ice Lake / Alder Lake dies with 0xC000001D (illegal instruction) in its own code
right after its first `CreateShaderResourceView` (the first descriptor written into title memory). `insn.rs` decodes
the Zen 2 instructions such a CPU lacks – `clwb` (Intel: Ice Lake+), `clzero`, `rdpru`, `monitorx` / `mwaitx`
(AMD only), `rdpid` – and the vectored handler (`eh::emulate_console_only`) emulates them and steps over
(`clwb` / `monitorx` / `mwaitx`: nothing, the PC's memory is coherent; `clzero`: 64 zero bytes at rAX; `rdpru`:
QueryPerformanceCounter; `rdpid`: the processor number). Logged once per instruction. Any other illegal instruction
(and `ud2`, a title's own trap) is logged with its 16 code bytes, in the first-chance handler and in the crash line.
SHA-NI is not emulated (also missing on those CPUs): add it if a title's code bytes show `0F 38 C8..CD`.

## Next steps (priority)
Little is left to do without the exe – the rest needs an actual run of the game.
1. Once the user has a legal game exe: `xcl inspect`, `xcl run`, analysis of `report.json`
   (`called_stubs` with `(not on PC)` → mapping of console slots, e.g. PresentX; `d3d12.unknown_iids`; `xgameruntime.apis` / `.unimplemented` (thunk classes and slots
   to add: PLS, XSystemGetDeviceType);
   `shaders.rdna2_mnemonics`; `audio` – voice formats, `unsupported_formats`; imports related to Opus/XMA/AcpHal).
2. Draw-time binding: the game's root arguments → SGPRs in `io`, `mem` → UMA buffer; command list capture for GS
   and Compute dispatches; `XboxBvhBuilder` (or `BuildRaytracingAccelerationStructure`) → `gpu::bvh::build/build_top`
   (geometry from GPU buffers in UMA; Rehydrate = prebuilt BVHs from `*_bvh4.rpf` – format to be checked).
3. Textures: binding `gpu::texture` to the title's API (textures in its memory, re-upload after a write), the console's
   pipe/packer configuration for the _X modes (`texture::CONSOLE_PIPES`), MSAA.
4. Audio: Opus is decoded (0x704F voices); with the exe check how the title feeds it (voice format, buffer framing, or a
   console API); XMA2 low priority (the game has none).
5. Smaller: whole-wave reductions in partial waves (325 PS with a min reduction, every GPU), `image_sample_o` with an offset from a VGPR,
   smaller HLSL, further XGameRuntime APIs (XGameUI, XNetworking…), GameInput → system GameInput.dll,
   mapping physical pages at the game's address.
