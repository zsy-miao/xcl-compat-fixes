//! Instruction semantics checked on the real GPU: small RDNA2 programs are assembled,
//! recompiled to HLSL, compiled with DXC and run as kernels. Skipped (with a note) when
//! there is no D3D12 GPU with SM 6.6 + wave64 or no DXC. Tests that use `gpu_waves` also run
//! as wave32 programs, and only need a GPU and DXC.

use super::runner::{self, KernelState};
use super::{translate, Mode, Options};
use crate::gpu::rdna2::encode;

/// A GPU and DXC for the instruction tests. Wave64 programs run natively where the adapter
/// has 64-lane waves and in the split form (`runner::options`) where it has only 32 (an
/// adapter with neither, such as WARP, skips).
fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported(), super::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            println!("GPU/DXC unavailable – skipping ({:?} / {:?} / {:?})", a.as_ref().err(), w.as_ref().ok(), d.as_ref().err());
            runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            false
        }
    }
}

/// 64-lane hardware waves (a test of how a program runs in them).
fn wave64_hardware() -> bool {
    let w = runner::wave64_supported() == Ok(true);
    if !w {
        println!("no 64-lane waves on this GPU – skipping");
        runner::skip_test("no wave64");
    }
    w
}

/// `o` for a program run on this adapter (the split wave64 form where it is needed).
fn on_gpu(o: Options) -> Options {
    Options { split_wave64: runner::split_for(&o.mode), ..o }
}

/// `runner::mesh_draw` needs mesh shaders (not every adapter with wave64 has them).
fn mesh_ready() -> bool {
    match runner::mesh_shaders_supported() {
        Ok(true) => true,
        r => {
            println!("GPU without mesh shaders – skipping ({r:?})");
            runner::skip_test(if r.is_err() { "no GPU" } else { "no mesh shaders" });
            false
        }
    }
}

/// The wave sizes a per-lane test runs in, as the `wave64` option: true where the GPU has
/// wave64, and false on every GPU, so the test also runs on NVIDIA. Empty without a GPU or DXC.
fn gpu_waves() -> Vec<bool> {
    if !gpu_ready() {
        return Vec::new();
    }
    if runner::wave64_supported() == Ok(true) { vec![true, false] } else { vec![false] }
}

/// A kernel compiled as a wave64 or wave32 program, and the code as memory from address 0,
/// where `runner::run_program` puts it.
fn kernel_wave(code: &[u8], wave64: bool) -> (Vec<u8>, Vec<u32>) {
    let mem = code.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    if runner::direct_dxil() {
        return (runner::direct_kernel(code, wave64).unwrap(), mem);
    }
    let t = translate(code, &Options { wave64, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    (dxil, code.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// `XCL_DIRECT_DXIL=1`: the direct DXIL translator's kernel for a kernel-mode program with
/// one memory window (else `hlsl`, the HLSL path's container).
fn direct_or(code: &[u8], opt: &Options, hlsl: Vec<u8>) -> Vec<u8> {
    use crate::gpu::dxil::kernel::{translate, translate_compute, KernelOptions};
    let windows = opt.mem_windows <= 4 && (opt.hw_wave == 0 || (opt.hw_wave == 32 && !opt.wave64));
    if !runner::direct_dxil() || !windows {
        return hlsl;
    }
    let split = opt.wave64 && (opt.split_wave64 || runner::split_wave64());
    // heap windows: `HEAP_WINDOWS` in the count, as `pso::mem_windows` passes it
    let mem_windows = if opt.heap_windows && opt.mem_windows > 1 { opt.mem_windows | super::HEAP_WINDOWS } else { opt.mem_windows };
    let o = KernelOptions { wave64: opt.wave64, split, mem_windows, sampler_base: opt.static_sampler_base };
    match &opt.mode {
        Mode::Kernel => translate(code, o).unwrap_or_else(|e| panic!("direct DXIL: {e}")),
        Mode::Compute(cs) => translate_compute(code, o, cs).unwrap_or_else(|e| panic!("direct DXIL: {e}")).dxil,
        Mode::Mesh(ms) => crate::gpu::dxil::kernel::translate_mesh(code, o, ms).unwrap_or_else(|e| panic!("direct DXIL: {e}")).dxil,
        Mode::Vertex(vs) => crate::gpu::dxil::kernel::translate_vertex(code, o, vs).unwrap_or_else(|e| panic!("direct DXIL: {e}")).dxil,
        Mode::Pixel(ps) => crate::gpu::dxil::kernel::translate_pixel(code, o, ps).unwrap_or_else(|e| panic!("direct DXIL: {e}")).dxil,
        _ => hlsl,
    }
}

/// `XCL_DIRECT_DXIL=1`: a kernel-mode program the HLSL path only compiles must also go
/// through the direct DXIL translator (and its validator check).
fn direct_check(code: &[u8], opt: &Options) {
    let _ = direct_or(code, opt, Vec::new());
}

/// `runner::run_program` as a wave64 or wave32 program, with the number of lanes that ran.
fn run_wave(code: &[u8], st: &KernelState, wave64: bool) -> (runner::KernelResult, usize) {
    let (dxil, mem) = kernel_wave(code, wave64);
    (runner::run(&dxil, st, &mem).unwrap(), if wave64 { 64 } else { 32 })
}

/// Assemble a program from (name, encoding, fields, literal).
fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

const V: u64 = 256; // VALU source code of v0

fn f(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn geometry_wave32_rejects_cross_half_operations() {
    let one = |name: &str, enc: &str, fields: &[(&str, u64)]| encode(name, enc, fields, None).unwrap();

    let lane = one("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V)]);
    assert!(super::geometry_wave32_hazards(&lane).contains("cross-half lane exchange"));

    let branch = one("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 0)]);
    assert!(super::geometry_wave32_hazards(&branch).contains("branch on a whole-wave mask"));

    let reduce = one("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]);
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));

    let compare = one("S_CMP_EQ_U64", "ENC_SOPC", &[("SSRC0", 126), ("SSRC1", 106)]);
    assert!(super::geometry_wave32_hazards(&compare).contains("scalar condition from a whole-wave mask"));

    let mut scalar = one("S_AND_B64", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 126), ("SSRC1", 106)]);
    scalar.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    assert!(super::geometry_wave32_hazards(&scalar).contains("branch on a half-local scalar condition"));

    // a prefix count over EXEC, VCC or a v_cmp result in an SGPR pair; the lane index is fine
    for mask in [126, 107, 10] {
        let count = one("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", mask), ("SRC1", 128)]);
        assert!(super::geometry_wave32_hazards(&count).contains("prefix count over a whole-wave mask"), "SRC0 {mask}");
    }
    let mut lane = one("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 193), ("SRC1", 128)]);
    lane.extend(one("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 193), ("SRC1", V + 1)]));
    assert!(super::geometry_wave32_hazards(&lane).is_empty());

    // a v_cmp into s[0:1]: a mask, its 32-bit words too
    let cmp = one("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 168), ("SRC1", V)]);
    let mut reduce = cmp.clone();
    reduce.extend(one("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)]));
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));
    let mut reduce = one("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)]);
    reduce.extend(one("S_FLBIT_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 127)]));
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));
    let mut branch = cmp.clone();
    branch.extend(one("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]));
    branch.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    assert!(super::geometry_wave32_hazards(&branch).contains("branch on a half-local scalar condition"));
    // the same scalar work on a non-mask SGPR stays accepted
    let mut plain = one("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 4)]);
    plain.extend(one("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)]));
    plain.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    plain.extend(one("S_ENDPGM", "ENC_SOPP", &[]));
    assert!(super::geometry_wave32_hazards(&plain).is_empty());
}

#[test]
fn geometry_wave_path_selection_preserves_native_wave64() {
    assert!(super::pso::wave32_only(32, 32));
    assert!(!super::pso::wave32_only(32, 64));
    assert!(!super::pso::wave32_only(64, 64));
    // a VS / PS: the driver picks, so any range with 32 lanes gets the 32-lane form (#42)
    assert!(super::pso::graphics_wave32(32, 32));
    assert!(super::pso::graphics_wave32(32, 64));
    assert!(!super::pso::graphics_wave32(64, 64));
}

#[test]
fn hlsl_is_generated_for_straight_line_and_branches() {
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 133)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 133)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    assert_eq!(t.blocks, 3, "{}", t.hlsl);
    // forward branches only: a plain `if`, no loop and no block dispatch on pc
    assert!(t.hlsl.contains("if (!(") && !t.hlsl.contains("pc ==") && !t.hlsl.contains("for (;;)") && t.hlsl.contains("[WaveSize(64)]"), "{}", t.hlsl);
    // the guarded block chain (fallback form) for the same program
    // (an option, not the environment variable: other tests translate at the same time)
    let g = translate(&code, &Options { structured: false, ..Options::default() }).unwrap();
    assert!(g.hlsl.contains("if (pc == 0u)") && !g.hlsl.contains("while"), "{}", g.hlsl);
    // a backward branch: only its range becomes a loop
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 133)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None), // back to s_sub
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    // only the range of the backward branch loops; its condition is the loop's exit test
    assert!(t.hlsl.contains("[loop] for (;;)") && t.hlsl.contains("break;") && !t.hlsl.contains("pc =="), "{}", t.hlsl);
    let t = translate(&prog(&[("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 2), ("VSRC1", 3)], None)]), &Options::default()).unwrap();
    assert!(!t.hlsl.contains("pc =="), "{}", t.hlsl);
    // unknown to the recompiler: reported, HLSL still produced
    let t = translate(&prog(&[("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 0), ("VSRC", 0), ("ATTR", 0)], None)]), &Options::default()).unwrap();
    assert_eq!(t.unsupported.get("v_interp_p1_f32"), Some(&1));
    let _ = Mode::Kernel;
}

#[test]
fn scalar_bfe_64_handles_unsigned_signed_and_zero_width() {
    let code = prog(&[
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 4), ("SSRC1", 8)], None),
        ("S_BFE_I64", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 4), ("SSRC1", 9)], None),
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 4), ("SSRC1", 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("& (0xFFFFFFFFFFFFFFFFull >> (64u -") && t.hlsl.contains("(int64_t)(t") && t.hlsl.contains("== 0u) ? 0ull"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn scalar_bitcmp_64_selects_the_indexed_half() {
    let code = prog(&[
        ("S_BITCMP1_B64", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 8)], None),
        ("S_BITCMP0_B64", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 9)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("select((s8 & 32u) == 0u, s4, s5)") && t.hlsl.contains("select((s9 & 32u) == 0u, s4, s5)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("scc = (uint)(") && t.hlsl.contains("== 0u);"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn scalar_64_bit_count_scan_and_set_cluster() {
    let code = prog(&[
        ("S_BCNT0_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 4)], None),
        ("S_FF0_I32_B64", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 4)], None),
        ("S_FLBIT_I32_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 4)], None),
        ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 10)], None),
        ("S_BITSET0_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 11)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("64u - countbits(s4) - countbits(s5)") && t.hlsl.contains("firstbitlow(~s4)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("31u - firstbithigh(s5)") && t.hlsl.contains("s7 | select((s10 & 32u) != 0u") && t.hlsl.contains("s6 & ~select((s11 & 32u) == 0u"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn scalar_signed_leading_bit_scans() {
    let code = prog(&[
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 4)], None),
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 6)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("ffbh_i(s4)"), "{}", t.hlsl);
    // i64: normalize on the 64-bit sign, then scan the high half first and the low half second
    assert!(t.hlsl.contains("select((int)s7 < 0, ~s6, s6)") && t.hlsl.contains("select((int)s7 < 0, ~s7, s7)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("31u - firstbithigh(") && t.hlsl.contains("63u - firstbithigh(") && t.hlsl.contains("0xFFFFFFFFu"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn scalar_relative_move_cluster() {
    let code = prog(&[
        ("S_MOVRELS_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 10)], None),
        ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 10)], None),
        ("S_MOVRELD_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 10)], None),
        ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 10)], None),
        ("S_MOVRELSD_2_B32", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("switch (m0)") && t.hlsl.contains("switch ((m0 & 0x3FFu))") && t.hlsl.contains("switch (((m0 >> 16) & 0x3FFu))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn gpu_scalar_relative_moves() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 130)], None),
        ("S_MOVRELS_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 10)], None),
        ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 10)], None),
        ("S_MOVRELD_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 10)], None),
        ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 10)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0x0003_0001)),
        ("S_MOVRELSD_2_B32", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 10)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[10..14].copy_from_slice(&[100, 101, 102, 103]);
    let r = runner::run_code(&code, &st, &[]).unwrap();
    assert_eq!((r.sgpr[0], r.sgpr[2], r.sgpr[3]), (102, 102, 103), "relative scalar reads");
    assert_eq!((r.sgpr[22], r.sgpr[26], r.sgpr[27]), (100, 100, 101), "relative scalar writes");
    assert_eq!(r.sgpr[33], 101, "split source/destination offsets");
}

/// v_swaprel_b32: both sides are read-write (fixed VDST + the VGPR at SRC0 + M0); emitted
/// as the same bounded switch over M0 as v_movrel, both old values captured before writes.
#[test]
fn v_swaprel_b32_uses_the_m0_relative_switch() {
    let code = prog(&[("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 10)], None)]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // old VDST captured; the M0-relative register read into another temporary
    assert!(t.hlsl.contains("uint t1 = v5;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint t2 = 0u; switch (m0) { case 0u: t2 = v10; break; default: break; }"), "{}", t.hlsl);
    // EXEC-gated exchange of the two captured values
    assert!(t.hlsl.contains("if (lanebit(exec_lo, exec_hi)) { v5 = t2; switch (m0) { case 0u: v10 = t1; break; default: break; } }"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn gpu_v_swaprel_b32() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 131)], None), // m0 = 3
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 10)], None), // v5 <-> v13
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None), // exec = s[30:31]: even lanes
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 11)], None), // v6 <-> v14 (active only)
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 32)], None), // exec = s[32:33]: all lanes
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 129)], None), // m0 = 1
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 6)], None), // v7 <-> v7: no-op
        // relative targets lie within the program's VGPRs (as in gpu_v_movrel): v15 used directly
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 15)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[30] = 0x5555_5555;
    st.sgpr[31] = 0x5555_5555;
    st.sgpr[32] = 0xFFFF_FFFF;
    st.sgpr[33] = 0xFFFF_FFFF;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[5] = 1000 + l as u32;
        r[13] = 2000 + l as u32;
        r[6] = 3000 + l as u32;
        r[14] = 4000 + l as u32;
        r[7] = 700 + l as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!((v[5], v[13]), (2000 + l as u32, 1000 + l as u32), "lane {l} ordinary swap");
        let expected = if l % 2 == 0 { (4000 + l as u32, 3000 + l as u32) } else { (3000 + l as u32, 4000 + l as u32) };
        assert_eq!((v[6], v[14]), expected, "lane {l} EXEC-gated swap");
        assert_eq!(v[7], 700 + l as u32, "lane {l} overlapping swap is a no-op");
    }
    assert_eq!((r.sgpr[126], r.sgpr[127]), (0xFFFF_FFFF, 0xFFFF_FFFF), "exec restored");
}

#[test]
fn atomic_inc_dec_x2_encoding_families() {
    let code = prog(&[
        ("BUFFER_ATOMIC_INC_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 0), ("SOFFSET", 128)], None),
        ("BUFFER_ATOMIC_DEC_X2", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 8), ("SRSRC", 0), ("SOFFSET", 128)], None),
        ("FLAT_ATOMIC_INC_X2", "ENC_FLAT", &[("VDST", 10), ("ADDR", 12), ("DATA", 14)], None),
        ("FLAT_ATOMIC_DEC_X2", "ENC_FLAT", &[("VDST", 16), ("ADDR", 12), ("DATA", 18)], None),
        ("GLOBAL_ATOMIC_INC_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2)], None),
        ("GLOBAL_ATOMIC_DEC_X2", "ENC_FLAT_GLBL", &[("VDST", 26), ("ADDR", 22), ("DATA", 28), ("SADDR", 2)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("InterlockedCompareExchange64").count(), 6, "{}", t.hlsl);
    assert!(t.hlsl.contains(") ? (uint64_t)0 :") && t.hlsl.contains("== (uint64_t)0 ||"), "{}", t.hlsl);
    // memory above 4 GiB: every access of the CAS loop (its first read too) goes through the
    // window accessors
    let w = translate(&code, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert!(!w.hlsl.contains("mem.Load<") && w.hlsl.matches("= xcl_mLoad2(").count() >= 6, "{}", w.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&w.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}
{}", w.hlsl));
    }
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

/// *_atomic_fcmpswap_x2 (buffer / flat / global): 64-bit IEEE-754 float compare-swap in a
/// bounded u64 CAS loop (`-0.0 == +0.0`, `NaN != NaN`, no `DoubleExtensions` required).
/// Deterministic emitted text on all three encoding families, plus multi-window & direct DXIL.
#[test]
fn atomic_fcmpswap_x2_uses_ieee_64_bit_cas() {
    let code = prog(&[
        ("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)], None),
        ("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 12), ("ADDR", 14), ("DATA", 16), ("GLC", 1)], None),
        ("GLOBAL_ATOMIC_FCMPSWAP_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // bounded IEEE-754 float CAS loop using atomic Or-0 initial load, pure u64 bitwise checks,
    // and InterlockedCompareExchange64 (no tearing 64-bit Load2 on mismatch early-exit)
    assert!(!t.hlsl.contains("mem.Load2("), "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("mem.InterlockedOr64(").count(), 3, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("mem.InterlockedCompareExchange64(").count(), 3, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("0x7FF0000000000000ull").count(), 3, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("0x7FFFFFFFFFFFFFFFull").count(), 6, "{}", t.hlsl);
    // [DATA:DATA+1] is the swap value, [DATA+2:DATA+3] the compare value (128-bit operand)
    assert!(t.hlsl.contains("_v = (uint64_t)v4 | ((uint64_t)v5 << 32); uint64_t") && t.hlsl.contains("_cmp = (uint64_t)v6 | ((uint64_t)v7 << 32);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("_cmp = (uint64_t)v18 | ((uint64_t)v19 << 32);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("_cmp = (uint64_t)v26 | ((uint64_t)v27 << 32);"), "{}", t.hlsl);
    // 8-byte alignment of all three cells; the buffer form's bounds check reads records in dwords
    assert_eq!(t.hlsl.matches("& ~7u;").count(), 3, "{}", t.hlsl);
    assert!(t.hlsl.contains("_off + 8u <= s10"), "{}", t.hlsl);
    // GLC returns the old u64 as a low / high dword pair – in place in VDATA for the buffer
    // form, into VDST for flat / global
    assert_eq!(t.hlsl.matches("_old >> 32").count(), 3, "{}", t.hlsl);
    assert!(t.hlsl.contains("v4 = (uint)t") && t.hlsl.contains("v5 = (uint)(t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v12 = (uint)t") && t.hlsl.contains("v13 = (uint)(t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v20 = (uint)t") && t.hlsl.contains("v21 = (uint)(t"), "{}", t.hlsl);
    let w = translate(&code, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert!(!w.hlsl.contains("xcl_mLoad2(t"), "{}", w.hlsl);
    assert_eq!(w.hlsl.matches("xcl_mInterlockedOr64(t").count(), 3, "{}", w.hlsl);
    assert_eq!(w.hlsl.matches("xcl_mInterlockedCompareExchange64(t").count(), 3, "{}", w.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        super::dxc::compile(&w.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", w.hlsl));
        for mem_windows in [1, 2, 2 | super::HEAP_WINDOWS] {
            let o = crate::gpu::dxil::kernel::KernelOptions { wave64: true, split: true, mem_windows, sampler_base: 0 };
            crate::gpu::dxil::kernel::translate_full(&code, o).unwrap_or_else(|e| panic!("dxil windows={mem_windows:#x}: {e}"));
        }
    }
}

/// *_atomic_fmin_x2 / fmax_x2 (buffer / flat / global): f64 min / max as one native 64-bit
/// integer atomic picked by the operand's sign, like the f32 forms – no CAS loop.
#[test]
fn atomic_fmin_fmax_x2_are_native_64_bit_atomics() {
    let code = prog(&[
        ("BUFFER_ATOMIC_FMIN_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_FMAX_X2", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("FLAT_ATOMIC_FMIN_X2", "ENC_FLAT", &[("VDST", 10), ("ADDR", 12), ("DATA", 14), ("GLC", 1)], None),
        ("FLAT_ATOMIC_FMAX_X2", "ENC_FLAT", &[("VDST", 16), ("ADDR", 12), ("DATA", 18)], None),
        ("GLOBAL_ATOMIC_FMIN_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)], None),
        ("GLOBAL_ATOMIC_FMAX_X2", "ENC_FLAT_GLBL", &[("VDST", 26), ("ADDR", 22), ("DATA", 28), ("SADDR", 2)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(!t.hlsl.contains("InterlockedCompareExchange64"), "{}", t.hlsl);
    // one signed and one unsigned branch per instruction, chosen at +inf's bits
    assert_eq!(t.hlsl.matches("<= 0x7FF0000000000000ull)").count(), 6, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("mem.InterlockedMin64(").count(), 6, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("mem.InterlockedMax64(").count(), 6, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("& ~7u;").count(), 6, "{}", t.hlsl);
    // memory above 4 GiB: the program's atomics go through the window accessors (whose own
    // bodies call mem / mem1)
    let w = translate(&code, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert_eq!(w.hlsl.matches("xcl_mInterlockedMin64(t").count(), 6, "{}", w.hlsl);
    assert_eq!(w.hlsl.matches("xcl_mInterlockedMax64(t").count(), 6, "{}", w.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        super::dxc::compile(&w.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", w.hlsl));
        direct_check(&code, &Options { mem_windows: 2, ..Default::default() });
        direct_check(&code, &Options { mem_windows: 2, heap_windows: true, ..Default::default() });
    }
}

#[test]
fn unary_f64_math_vop1_and_vop3() {
    let code = prog(&[
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 10)], None),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 10)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 10)], None),
        ("V_CEIL_F64", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 10), ("NEG", 1)], None),
        ("V_FRACT_F64", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 10), ("ABS", 1)], None),
        ("V_SQRT_F64", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 10)], None),
        ("V_RSQ_F64", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("ceil_d(").count(), 3, "{}", t.hlsl); // helper plus two calls
    assert_eq!(t.hlsl.matches("fract_d(").count(), 3, "{}", t.hlsl); // helper plus two calls
    // f64 roots through sqrt_d (HLSL's sqrt is f32): V_SQRT / V_RSQ, twice each
    assert_eq!(t.hlsl.matches("sqrt_d(").count(), 1 + 4, "{}", t.hlsl); // helper plus four calls
    assert_eq!(t.hlsl.matches("rcp_d(sqrt_d(").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn f64_class_compare_vopc_and_vop3() {
    let code = prog(&[
        ("V_CMP_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 10), ("VSRC1", 12)], None),
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 10), ("SRC1", V + 12), ("ABS", 1)], None),
        ("V_CMPX_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 10), ("VSRC1", 13)], None),
        ("V_CMPX_CLASS_F64", "ENC_VOP3", &[("VDST", 126), ("SRC0", V + 10), ("SRC1", V + 13), ("NEG", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("fclass64(").count(), 5, "{}", t.hlsl); // helper plus four calls
    // SRC0 is the 64-bit f64 operand (VGPR/SGPR pair), SRC1 stays a plain 32-bit mask;
    // abs/neg act on the double operand only
    assert!(t.hlsl.contains("(v12) >> fclass64(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v12) >> fclass64(asdouble(v10, (v11 & 0x7FFFFFFFu)))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v13) >> fclass64(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    // mask destinations go through the shared compare/write-mask path: vcc, an SGPR pair, exec
    assert!(t.hlsl.contains("vcc_lo = t") && t.hlsl.contains("vcc_hi = t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("s20 = t") && t.hlsl.contains("s21 = t"), "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("exec_lo = t").count(), 2, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("exec_hi = t").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_f64_class_compare() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_CMP_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 50), ("VSRC1", 52)], None),                        // vcc
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 50), ("SRC1", V + 52)], None),      // s[20:21]
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 50), ("SRC1", 8), ("ABS", 1)], None), // |x|, mask in s8
        ("V_CMPX_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 50), ("VSRC1", 53)], None),                       // exec
    ]);
    let classes = [
        0x7FF0_0000_0000_0001u64,
        0x7FF8_0000_0000_0000,
        f64::NEG_INFINITY.to_bits(),
        (-1.0f64).to_bits(),
        0x8000_0000_0000_0001,
        (-0.0f64).to_bits(),
        0.0f64.to_bits(),
        1,
        1.0f64.to_bits(),
        f64::INFINITY.to_bits(),
    ];
    let mut st = KernelState::default();
    st.sgpr[8] = 1 << 6 | 1 << 7 | 1 << 8 | 1 << 9; // positive non-NaN classes
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let class = l % classes.len();
        let bits = classes[class];
        (r[50], r[51]) = (bits as u32, (bits >> 32) as u32);
        r[52] = 1 << class;
        r[53] = 1 << 6;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    assert_eq!((r.sgpr[106], r.sgpr[107]), (u32::MAX, u32::MAX), "each lane matches its f64 class (vcc)");
    assert_eq!((r.sgpr[20], r.sgpr[21]), (u32::MAX, u32::MAX), "each lane matches its f64 class (sdst)");
    // |x| ∈ {+0, +denormal, +normal, +inf} exactly when x is not a NaN: negative lanes fold
    let pos: u64 = (0..64).filter(|l| l % 10 >= 2).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[22] as u64 | (r.sgpr[23] as u64) << 32, pos, "abs: every non-NaN becomes positive");
    let zero: u64 = (0..64).filter(|l| l % 10 == 6).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[126] as u64 | (r.sgpr[127] as u64) << 32, zero, "cmpx keeps positive-zero lanes");
}

/// Normalized f16 conversions – v_cvt_norm_u16_f16 / _i16_f16 (VOP1 single half-word
/// result, VOP3 with op_sel picking source/destination halves) and v_cvt_pknorm_u16_f16 /
/// _i16_f16 (VOP3, two f16 halves → a packed dword). Emitted text is deterministic.
#[test]
fn cvt_norm_f16_instructions() {
    let code = prog(&[
        ("V_CVT_NORM_U16_F16", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 9)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 9), ("SRC1", V + 10), ("SRC2", V + 11), ("OP_SEL", 0b1001)], None),
        ("V_CVT_PKNORM_U16_F16", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 9), ("SRC1", V + 10)], None),
        ("V_CVT_PKNORM_I16_F16", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 9), ("SRC1", V + 10), ("OP_SEL", 0b0011)], None),
        ("V_CVT_PKNORM_U16_F16", "ENC_VOP3", &[("VDST", 5), ("SRC0", 242), ("SRC1", 243)], None), // f16 +1.0, −1.0
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    // single f16 source (low half) → unsigned / signed normalized short, 16-bit result
    assert!(t.hlsl.contains("cvt_norm_u16_f16(((v9) & 0xFFFFu))"), "{}", t.hlsl);
    // op_sel[0] reads the source's high half, op_sel[3] writes the destination's high
    // half preserving the low one; abs/neg apply directly on the binary16 sign bit
    assert!(t.hlsl.contains("cvt_norm_i16_f16(((v9) >> 16))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v2 & 0xFFFFu) | (t") && t.hlsl.contains("<< 16), v2)"), "{}", t.hlsl);
    // packed: two whole dwords written; each source half picked by op_sel
    assert!(t.hlsl.contains("(cvt_norm_u16_f16((v9 & 0xFFFFu)) | (cvt_norm_u16_f16((v10 & 0xFFFFu)) << 16))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(cvt_norm_i16_f16((v9 >> 16)) | (cvt_norm_i16_f16((v10 >> 16)) << 16))"), "{}", t.hlsl);
    // inline float constants decode as f16 bit patterns (1.0 = 0x3C00, −1.0 = 0xBC00)
    assert!(t.hlsl.contains("cvt_norm_u16_f16((0x00003C00u & 0xFFFFu))") && t.hlsl.contains("cvt_norm_u16_f16((0x0000BC00u & 0xFFFFu))"), "{}", t.hlsl);
}

/// `v_div_fixup_f16` (VOP3, three f16 sources: quotient, denominator, numerator) is translated
/// like `v_div_fixup_f32`, on half-precision operands, instead of being left as unsupported.
#[test]
fn div_fixup_f16_instruction() {
    let code = prog(&[("V_DIV_FIXUP_F16", "ENC_VOP3", &[("VDST", 1), ("SRC0", V + 9), ("SRC1", V + 10), ("SRC2", V + 11)], None)]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    assert!(t.hlsl.contains("div_fixup(f16tof32(((v9) & 0xFFFFu)), f16tof32(((v10) & 0xFFFFu)), f16tof32(((v11) & 0xFFFFu)))"), "{}", t.hlsl);
}

/// GPU verification of exact RNE rounding and edge cases for `V_CVT_NORM_*_F16` and
/// `V_CVT_PKNORM_*_F16`: catches binary32 double-rounding (`0x3801`/`0xB801`), DXIL
/// `clamp(NaN, -1, 1)` returning ±32767 instead of 0, ±Inf, ±0, subnormals, and VOP3 abs/neg.
#[test]
fn gpu_cvt_norm_f16_exact_rounding_and_edge_cases() {
    if !gpu_ready() {
        return;
    }
    // 16 test vectors (packed into lane % 16): (f16_bits, expected_unorm16, expected_snorm16)
    const CASES: &[(u16, u16, u16)] = &[
        (0x3801, 32799, 16399),         // 1025/2048: fp32 double-rounds to 32800 / 16400
        (0xB801, 0, (-16399i16) as u16), // -1025/2048: fp32 double-rounds to -16400
        (0x3800, 32768, 16384),         // +0.5: exact half tie-to-even
        (0xB800, 0, (-16384i16) as u16), // -0.5: exact half tie-to-even
        (0x7E00, 0, 0),                 // +qNaN -> 0 (not ±32767)
        (0xFC01, 0, 0),                 // -sNaN -> 0
        (0x7C00, 65535, 32767),         // +Inf
        (0xFC00, 0, (-32767i16) as u16), // -Inf
        (0x0000, 0, 0),                 // +0.0
        (0x8000, 0, 0),                 // -0.0
        (0x0080, 0, 0),                 // subnormal 128 * 2^-24: *65535 = 0.499992 -> 0
        (0x0081, 1, 0),                 // subnormal 129 * 2^-24: *65535 = 0.503899 -> 1
        (0x0101, 1, 1),                 // subnormal 257 * 2^-24: *32767 = 0.501938 -> 1
        (0x3BFF, 65503, 32751),         // max f16 < 1.0 (2047/2048)
        (0x3C00, 65535, 32767),         // +1.0
        (0xBC00, 0, (-32767i16) as u16), // -1.0
    ];
    let code = prog(&[
        // v1 = V_CVT_NORM_U16_F16(v0.lo), v2 = V_CVT_NORM_I16_F16(v0.lo)
        ("V_CVT_NORM_U16_F16", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        // Packed: v3 = pknorm_u16(v0.lo, v0.hi), v4 = pknorm_i16(v0.lo, v0.hi)
        ("V_CVT_PKNORM_U16_F16", "ENC_VOP3", &[("VDST", 3), ("SRC0", V), ("SRC1", V), ("OP_SEL", 0b0010)], None),
        ("V_CVT_PKNORM_I16_F16", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V), ("OP_SEL", 0b0010)], None),
        // VOP3 modifiers:
        // - abs(-1025/2048) -> +1025/2048 -> u16 32799 / i16 16399
        // - neg(+1025/2048) -> -1025/2048 -> i16 -16399 (0xBFF1)
        // - neg(abs(-1025/2048)) -> -1025/2048 -> i16 -16399 (verifies abs before neg)
        // - neg(qNaN) -> 0
        ("V_CVT_NORM_U16_F16", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 5), ("ABS", 1)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 5), ("ABS", 1)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 6), ("NEG", 1)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 12), ("NEG", 1)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 5), ("ABS", 1), ("NEG", 1)], None),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let lo = CASES[l % CASES.len()].0 as u32;
        let hi = CASES[(l + 1) % CASES.len()].0 as u32;
        r[0] = lo | (hi << 16);
        r[5] = 0xB801;  // -1025/2048
        r[6] = 0x7E00;  // +qNaN
        r[12] = 0x3801; // +1025/2048
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let (h_lo, exp_u_lo, exp_i_lo) = CASES[l % CASES.len()];
        let (h_hi, exp_u_hi, exp_i_hi) = CASES[(l + 1) % CASES.len()];
        assert_eq!(r.vgpr[l][1] & 0xFFFF, exp_u_lo as u32, "lane {l} u16(0x{h_lo:04X})");
        assert_eq!(r.vgpr[l][2] & 0xFFFF, exp_i_lo as u32, "lane {l} i16(0x{h_lo:04X})");
        assert_eq!(
            r.vgpr[l][3],
            (exp_u_lo as u32) | ((exp_u_hi as u32) << 16),
            "lane {l} pknorm_u16(0x{h_lo:04X}, 0x{h_hi:04X})"
        );
        assert_eq!(
            r.vgpr[l][4],
            (exp_i_lo as u32) | ((exp_i_hi as u32) << 16),
            "lane {l} pknorm_i16(0x{h_lo:04X}, 0x{h_hi:04X})"
        );
        assert_eq!(r.vgpr[l][7] & 0xFFFF, 32799, "lane {l} abs(0xB801) u16");
        assert_eq!(r.vgpr[l][8] & 0xFFFF, 16399, "lane {l} abs(0xB801) i16");
        assert_eq!(r.vgpr[l][9] & 0xFFFF, 0, "lane {l} neg(qNaN) i16");
        assert_eq!(r.vgpr[l][10] & 0xFFFF, (-16399i16) as u16 as u32, "lane {l} neg(0x3801) i16");
        assert_eq!(r.vgpr[l][11] & 0xFFFF, (-16399i16) as u16 as u32, "lane {l} -abs(0xB801) i16");
    }
}

/// The f64 frexp pair: v_frexp_mant_f64 (m in [0.5,1) keeping the sign, input back for
/// 0/inf/NaN) → an f64 pair, v_frexp_exp_i32_f64 (input = m·2^exp, 0 for 0/inf/NaN) → one
/// dword. VOP1 and VOP3 (source abs/neg come from the f64 modifiers). Emitted text is
/// deterministic; helpers normalise denormals bit-exactly.
#[test]
fn frexp_f64_instructions() {
    let code = prog(&[
        ("V_FREXP_MANT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 10)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 10), ("NEG", 1)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 10), ("ABS", 1)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 8), ("SRC0", 243)], None), // inline constant −1.0
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // the source pair becomes a double; the mantissa stays f64, the exponent is one dword
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(uint)frexp_exp_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("frexp_exp_d(asdouble(v10, (v11 & 0x7FFFFFFFu)))"), "{}", t.hlsl);
    // the f64 pair of inline −1.0 = 0xBFF0000000000000
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(0x00000000u, 0xBFF00000u))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn rndne_f64_instructions() {
    let code = prog(&[
        ("V_RNDNE_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_RNDNE_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 10), ("NEG", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("round_ne_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("round_ne_d(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn gpu_rndne_f64() {
    if !gpu_ready() { return; }
    let p = prog(&[("V_RNDNE_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None)]);
    const INPUTS: [u64; 16] = [
        0.0f64.to_bits(), (-0.0f64).to_bits(), 0.5f64.to_bits(), (-0.5f64).to_bits(),
        1.5f64.to_bits(), 2.5f64.to_bits(), (-1.5f64).to_bits(), (-2.5f64).to_bits(),
        0.49999999999999994f64.to_bits(), 0.5000000000000001f64.to_bits(),
        4503599627370495.5f64.to_bits(), 4503599627370496.0f64.to_bits(),
        f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits(), 0x7FF8_0000_0000_1234, 1,
    ];
    let mut st = KernelState::default();
    for (lane, regs) in st.vgpr.iter_mut().enumerate() {
        let x = INPUTS[lane % INPUTS.len()];
        (regs[0], regs[1]) = (x as u32, (x >> 32) as u32);
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for (lane, regs) in r.vgpr.iter().enumerate() {
        let x = f64::from_bits(INPUTS[lane % INPUTS.len()]);
        let got = regs[2] as u64 | ((regs[3] as u64) << 32);
        let want = if x.is_nan() { x.to_bits() } else { x.round_ties_even().to_bits() };
        assert_eq!(got, want, "lane {lane}: {x:?}");
    }
}

#[test]
fn gpu_frexp_exp_f32_denorm_modes() {
    const CASES: [(u32, i32, i32); 13] = [
        (0x0000_0001, -148, 0), (0x0000_0003, -147, 0), (0x007F_FFFF, -126, 0),
        (0x0080_0000, -125, -125), (0x8000_0001, -148, 0), (0x0000_0000, 0, 0),
        (0x8000_0000, 0, 0), (0x4060_0000, 2, 2), (0xC060_0000, 2, 2),
        (0x7F7F_FFFF, 128, 128), (0x7F80_0000, 0, 0), (0xFF80_0000, 0, 0),
        (0x7FC0_0000, 0, 0),
    ];
    let p = prog(&[
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 12)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 13)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V)], None),
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 14)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V)], None),
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 15)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V)], None),
    ]);
    let mut st = KernelState::default();
    for (lane, v) in st.vgpr.iter_mut().enumerate() {
        v[0] = CASES[lane % CASES.len()].0;
        v[4..8].fill(u32::MAX);
    }
    for wave64 in gpu_waves() {
        let (r, lanes) = run_wave(&p, &st, wave64);
        for (lane, v) in r.vgpr.iter().take(lanes).enumerate() {
            let (bits, preserve, flush) = CASES[lane % CASES.len()];
            let got = [v[4] as i32, v[5] as i32, v[6] as i32, v[7] as i32];
            assert_eq!(got, [flush, preserve, flush, preserve], "wave{lanes} lane {lane}: {bits:#010x}");
        }
    }
}

#[test]
fn gpu_frexp_exp_f32_denorm_setreg() {
    let p = prog(&[
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 12)], None),
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 3))], Some(4)),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 5, 1))], Some(0)),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V)], None),
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 4, 2)), ("SDST", 0)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V)], None),
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 4, 2)), ("SDST", 1)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[0] = 2;
    st.sgpr[1] = 1;
    for v in &mut st.vgpr {
        v[0] = 1;
        v[4..8].fill(u32::MAX);
    }
    for wave64 in gpu_waves() {
        let (r, lanes) = run_wave(&p, &st, wave64);
        for (lane, v) in r.vgpr.iter().take(lanes).enumerate() {
            let got = [v[4] as i32, v[5] as i32, v[6] as i32, v[7] as i32];
            assert_eq!(got, [-148, -148, 0, -148], "wave{lanes} lane {lane}");
        }
    }
}

#[test]
fn gpu_frexp_exp_f32_denorm_control_flow() {
    let p = prog(&[
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 13)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 12)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
    ]);
    let mut st = KernelState::default();
    for v in &mut st.vgpr {
        v[0] = 1;
        v[4] = u32::MAX;
    }
    for wave64 in gpu_waves() {
        for (s0, expected) in [(1, 0), (0, -148)] {
            st.sgpr[0] = s0;
            let (r, lanes) = run_wave(&p, &st, wave64);
            for (lane, v) in r.vgpr.iter().take(lanes).enumerate() {
                assert_eq!(v[4] as i32, expected, "wave{lanes} lane {lane}: s0={s0}");
            }
        }
    }
}

#[test]
fn gpu_frexp_exp_f32_denorm_initial_mode() {
    let legacy = prog(&[("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None)]);
    let conditional = prog(&[
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 13)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
    ]);
    let mut st = KernelState::default();
    let inputs = [0x0000_0001, 0x8000_0003, 0x007F_FFFF, 0x0080_0000, 0x8000_0000, 0x7FC0_0000];
    for (lane, v) in st.vgpr.iter_mut().enumerate() {
        v[0] = inputs[lane % inputs.len()];
        v[4] = u32::MAX;
    }
    for wave64 in gpu_waves() {
        let (expected, lanes) = run_wave(&legacy, &st, wave64);
        let (actual, _) = run_wave(&conditional, &st, wave64);
        for lane in 0..lanes {
            assert_ne!(expected.vgpr[lane][4], u32::MAX, "wave{lanes} lane {lane}: unwritten reference");
            assert_eq!(actual.vgpr[lane][4], expected.vgpr[lane][4], "wave{lanes} lane {lane}: initial MODE");
        }
    }
}

#[test]
fn gpu_frexp_f64_exp_and_mant() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_FREXP_MANT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 0)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 0)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 0), ("NEG", 1)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 0), ("ABS", 1)], None),
    ]);
    const INPUTS: [u64; 16] = [
        12.5f64.to_bits(),            // 0.78125 · 2^4
        (-4.0f64).to_bits(),          // −0.5 · 2^3
        0.5f64.to_bits(),             // 0.5 · 2^0
        1.0f64.to_bits(),             // 0.5 · 2^1
        0.0f64.to_bits(),             // special → 0/0
        (-0.0f64).to_bits(),          // special → input/0
        f64::INFINITY.to_bits(),      // special → input/0
        f64::NEG_INFINITY.to_bits(),  // special → input/0
        0x7FF8_0000_0000_0001,        // NaN → input payload, 0
        0x0000_0000_0000_0001,        // smallest denormal → 0.5 · 2^−1073
        0x0000_0000_000C_0000,        // denormal 3·2^−1056 → 0.75 · 2^−1054
        0x000F_FFFF_FFFF_FFFF,        // largest denormal → (1−2^−52) · 2^−1022
        0x0010_0000_0000_0000,        // smallest normal → 0.5 · 2^−1021
        0x7FEF_FFFF_FFFF_FFFF,        // max normal → (1−2^−53) · 2^1024
        0x4059_0000_0000_0000,        // 100.0 → 0.78125 · 2^7
        0x800F_FFFF_FFFF_FFFF,        // negative denormal → −(1−2^−52) · 2^−1022
    ];
    // CPU mirror of the ISA semantics: e = topmost set bit of the 52-bit fraction − 1073
    // for denormals, biased exponent − 1022 for normals; mant = normalised [0.5,1) value.
    fn frexp_ref(bits: u64) -> (u64, i32) {
        let (h, l) = ((bits >> 32) as u32, bits as u32);
        let (e, m, s) = ((h >> 20) & 0x7FF, h & 0xF_FFFF, h & 0x8000_0000);
        if e == 0x7FF || (e == 0 && (m | l) == 0) {
            return (bits, 0);
        }
        if e != 0 {
            return ((((s | m | 0x3FE0_0000) as u64) << 32) | l as u64, e as i32 - 1022);
        }
        let top = if m != 0 { 63 - m.leading_zeros() } else { 31 - l.leading_zeros() };
        let f = ((m as u64) << 32) | l as u64;
        let ff = (f << (52 - top)) & 0x000F_FFFF_FFFF_FFFF;
        ((((s | 0x3FE0_0000 | (ff >> 32) as u32) as u64) << 32) | (ff & 0xFFFF_FFFF), top as i32 - 1073)
    }
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let x = INPUTS[l % INPUTS.len()];
        r[0] = x as u32;
        r[1] = (x >> 32) as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let (mant, exp) = frexp_ref(INPUTS[l % INPUTS.len()]);
        assert_eq!(v[2] as u64 | (v[3] as u64) << 32, mant, "lane {l} v_frexp_mant_f64");
        assert_eq!(v[4] as i32, exp, "lane {l} v_frexp_exp_i32_f64");
        // negated source: mantissa flips its sign, exponent is unchanged
        assert_eq!(v[6] as u64 | (v[7] as u64) << 32, mant ^ 0x8000_0000_0000_0000, "lane {l} v_frexp_mant_f64 neg");
        assert_eq!(v[8] as i32, exp, "lane {l} v_frexp_exp_i32_f64 abs");
    }
}

/// v_frexp_mant_f32 / _f16 keep the input's sign, as the f64 one does (#580): HLSL frexp()
/// returns an unsigned mantissa, so −10.125 came out as +0.6328125 and −0 as +0.
#[test]
fn gpu_frexp_mant_f32_f16_keep_the_sign() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_FREXP_MANT_F32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 0)], None),
        ("V_FREXP_MANT_F32", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 0), ("NEG", 1)], None),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 0)], None),
        ("V_FREXP_MANT_F16", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 1)], None),
    ]);
    // (input, mantissa, exponent): mant · 2^exp = input with |mant| in [0.5, 1); 0 / inf / NaN pass through
    const F32: [(u32, u32, i32); 8] = [
        (0xC122_0000, 0xBF22_0000, 4),  // −10.125 → −0.6328125 · 2^4
        (0x4122_0000, 0x3F22_0000, 4),  // 10.125 → 0.6328125 · 2^4
        (0xBF80_0000, 0xBF00_0000, 1),  // −1 → −0.5 · 2^1
        (0xBE80_0000, 0xBF00_0000, -1), // −0.25 → −0.5 · 2^−1
        (0x8000_0000, 0x8000_0000, 0),  // −0 → −0
        (0x0000_0000, 0x0000_0000, 0),  // +0 → +0
        (0xFF80_0000, 0xFF80_0000, 0),  // −inf → −inf
        (0xC2C8_0000, 0xBF48_0000, 7),  // −100 → −0.78125 · 2^7
    ];
    // f16: −10.125 → −0.6328125, −1 → −0.5, −0 → −0, 3 → 0.75
    const F16: [(u32, u32); 4] = [(0xC910, 0xB910), (0xBC00, 0xB800), (0x8000, 0x8000), (0x4200, 0x3A00)];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = F32[l % F32.len()].0;
        r[1] = F16[l % F16.len()].0;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let (x, mant, exp) = F32[l % F32.len()];
        assert_eq!(v[2], mant, "lane {l} v_frexp_mant_f32 {:#x}", x);
        assert_eq!(v[3], mant ^ 0x8000_0000, "lane {l} v_frexp_mant_f32 neg {:#x}", x);
        assert_eq!(v[4] as i32, exp, "lane {l} v_frexp_exp_i32_f32 {:#x}", x);
        let (h, hmant) = F16[l % F16.len()];
        assert_eq!(v[5] & 0xFFFF, hmant, "lane {l} v_frexp_mant_f16 {:#x}", h);
    }
}

/// Edge inputs of the f64 operations: |x| ≥ 2^63, values outside the f32 range, denormals,
/// signed zeros, ±inf and NaN.
const F64_EDGES: [u64; 22] = [
    0x0000_0000_0000_0000,       // +0
    0x8000_0000_0000_0000,       // −0
    0.5f64.to_bits(),
    (-0.5f64).to_bits(),
    (-1.5f64).to_bits(),
    2.5f64.to_bits(),
    4.0f64.to_bits(),
    123456789.75f64.to_bits(),
    (-4503599627370495.5f64).to_bits(),
    9223372036854775808.0f64.to_bits(),     // 2^63
    (-18446744073709551616.0f64).to_bits(), // −2^64
    1e300f64.to_bits(),
    (-1e300f64).to_bits(),
    1e-300f64.to_bits(),
    (-1e-300f64).to_bits(),
    0x0000_0000_0000_0001,                  // smallest denormal
    0x000F_FFFF_FFFF_FFFF,                  // largest denormal
    f64::MAX.to_bits(),
    f64::INFINITY.to_bits(),
    f64::NEG_INFINITY.to_bits(),
    0x7FF8_0000_0000_1234,                  // NaN
    4294967295.0f64.to_bits(),
];

fn f64_lanes(st: &mut KernelState, inputs: &[u64]) {
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let x = inputs[l % inputs.len()];
        (r[0], r[1]) = (x as u32, (x >> 32) as u32);
    }
}

fn pair(v: &[u32], i: usize) -> f64 {
    f64::from_bits(v[i] as u64 | (v[i + 1] as u64) << 32)
}

/// v_floor, v_ceil, v_trunc and v_fract_f64 and the f64 to i32 and u32 conversions against
/// the CPU's IEEE results, bit for bit, including the sign of zero, |x| ≥ 2^63, ±inf and NaN.
#[test]
fn gpu_f64_rounding_and_conversions() {
    for wave64 in gpu_waves() {
        f64_rounding_and_conversions(wave64);
    }
}

fn f64_rounding_and_conversions(wave64: bool) {
    let p = prog(&[
        ("V_FLOOR_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
        ("V_TRUNC_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 8), ("SRC0", V)], None),
        ("V_CVT_I32_F64", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("V_CVT_U32_F64", "ENC_VOP1", &[("VDST", 11), ("SRC0", V)], None),
        // the inline constant 0 compiles as a constant, where 0.0 − floor(0.0) gave −0
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 12), ("SRC0", 128)], None),
    ]);
    let mut st = KernelState::default();
    f64_lanes(&mut st, &F64_EDGES);
    let (r, lanes) = run_wave(&p, &st, wave64);
    let same = |got: f64, want: f64| got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
    let mut bad = Vec::new();
    for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
        let x = f64::from_bits(F64_EDGES[l % F64_EDGES.len()]);
        // V_FRACT_F64 is x − floor(x) kept below 1.0, +0 for ±0, and NaN for ±inf and NaN
        let fract = if x.is_finite() { (x - x.floor()).min(f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)) } else { f64::NAN };
        for (name, got, want) in [
            ("floor", pair(v, 2), x.floor()),
            ("ceil", pair(v, 4), x.ceil()),
            ("trunc", pair(v, 6), x.trunc()),
            ("fract", pair(v, 8), fract),
        ] {
            if !same(got, want) {
                bad.push(format!("v_{name}_f64({x:e}) = {got:e}, want {want:e}"));
            }
        }
        if pair(v, 12).to_bits() != 0 {
            bad.push(format!("v_fract_f64(inline 0) = {:e}, want +0", pair(v, 12)));
        }
        // saturating, and NaN gives 0, which is what Rust's `as` does
        if v[10] as i32 != x as i32 {
            bad.push(format!("v_cvt_i32_f64({x:e}) = {}, want {}", v[10] as i32, x as i32));
        }
        if v[11] != x as u32 {
            bad.push(format!("v_cvt_u32_f64({x:e}) = {}, want {}", v[11], x as u32));
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

/// v_sqrt_f64 and v_rsq_f64 over the whole f64 range. Denormals and values outside the f32
/// range get real roots within an ulp of the CPU's, a negative x gives NaN and −0 gives −0.
#[test]
fn gpu_f64_roots() {
    for wave64 in gpu_waves() {
        f64_roots(wave64);
    }
}

fn f64_roots(wave64: bool) {
    let p = prog(&[
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
        ("V_RCP_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V)], None),
    ]);
    let mut st = KernelState::default();
    f64_lanes(&mut st, &F64_EDGES);
    let (r, lanes) = run_wave(&p, &st, wave64);
    let ulps =|got: f64, want: f64| -> u64 {
        if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
            0
        } else if got.is_nan() || want.is_nan() || got.is_sign_negative() != want.is_sign_negative() {
            u64::MAX
        } else {
            got.to_bits().abs_diff(want.to_bits())
        }
    };
    let mut bad = Vec::new();
    for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
        let x = f64::from_bits(F64_EDGES[l % F64_EDGES.len()]);
        let checks = [("sqrt", pair(v, 2), x.sqrt(), 1), ("rsq", pair(v, 4), 1.0 / x.sqrt(), 2), ("rcp", pair(v, 6), 1.0 / x, 1)];
        for (name, got, want, tol) in checks {
            if ulps(got, want) > tol {
                bad.push(format!("v_{name}_f64({x:e}) = {got:e}, want {want:e}"));
            }
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

/// v_ldexp_f64 = x · 2^n, rounded once, for every i32 n: results that over- or underflow,
/// denormal results, exponents beyond the f32 range of 2^n.
#[test]
fn gpu_ldexp_f64() {
    for wave64 in gpu_waves() {
        ldexp_f64(wave64);
    }
}

fn ldexp_f64(wave64: bool) {
    let p = prog(&[("V_LDEXP_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 2)], None)]);
    // 2^e as f64 bits for a normal e
    const fn p2(e: i32) -> u64 {
        ((e + 1023) as u64) << 52
    }
    const ONE: u64 = p2(0);
    const CASES: [(u64, i32, u64); 21] = [
        (ONE, 0, ONE),
        (ONE, 1023, p2(1023)),
        (ONE, 1024, 0x7FF0_0000_0000_0000),                 // overflows to inf
        (ONE, -1022, p2(-1022)),                            // smallest normal
        (ONE, -1074, 1),                                    // smallest denormal
        (ONE, -1075, 0),                                    // half of it, a tie that rounds to even 0
        (1.5f64.to_bits(), -1074, 2),                       // 1.5 · 2^−1074, a tie that rounds to even 2
        (1.5f64.to_bits(), -1075, 1),                       // 0.75 · 2^−1074 rounds to 1
        (1, 2000, p2(926)),                                 // 2^−1074 · 2^2000
        (1, 1074, ONE),                                     // back to 1.0
        (p2(1023), -2046, 1 << 51),                         // 2^−1023, a denormal
        (1e300f64.to_bits(), -2000, 0x0137_E43C_8800_759C),  // only the exponent field moves
        (1e300f64.to_bits(), -3000, 0),                     // underflows to 0
        ((-2.0f64).to_bits(), 200, p2(201) | 1 << 63),      // −2^201
        (2.0f64.to_bits(), -200, p2(-199)),
        (ONE, i32::MAX, 0x7FF0_0000_0000_0000),             // inf
        ((-1.0f64).to_bits(), i32::MIN, 1 << 63),           // −0
        (1 << 63, 5, 1 << 63),                              // −0 stays −0
        (0x7FF0_0000_0000_0000, -5000, 0x7FF0_0000_0000_0000), // inf stays inf
        (0x7FF8_0000_0000_1234, 7, 0x7FF8_0000_0000_1234),  // NaN stays NaN
        (0, 3000, 0),                                       // 0 stays 0
    ];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, n, _) = CASES[l % CASES.len()];
        (r[0], r[1], r[2]) = (x as u32, (x >> 32) as u32, n as u32);
    }
    let (r, lanes) = run_wave(&p, &st, wave64);
    let mut bad = Vec::new();
    for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
        let (x, n, want) = CASES[l % CASES.len()];
        let (got, want, x) = (pair(v, 4), f64::from_bits(want), f64::from_bits(x));
        if !(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())) {
            bad.push(format!("v_ldexp_f64({x:e}, {n}) = {got:e} ({:#018x}), want {want:e}", got.to_bits()));
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

#[test]
fn gpu_valu_float_and_int() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 128), ("VSRC1", 3)], None), // 0 * inf = 0
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", 244), ("SRC2", 243)], None), // v1*2-1
        ("V_MAC_F32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 1), ("VSRC1", 2)], None), // v13 += v1*v2
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 14), ("SRC0", V + 4), ("VSRC1", 5)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 15), ("SRC0", 132), ("VSRC1", 4)], None), // v4 << 4
        ("V_BFE_U32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 6), ("SRC1", 136), ("SRC2", 132)], None), // bits 8..11
        ("V_CVT_F32_UBYTE1", "ENC_VOP1", &[("VDST", 17), ("SRC0", V + 6)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 18), ("SRC0", 193), ("SRC1", 128)], None), // lane index (low half)
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 18), ("SRC0", 193), ("SRC1", V + 18)], None), // + high half
        ("V_MED3_F32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 1), ("SRC1", 128), ("SRC2", 242)], None), // clamp to [0,1]
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", V + 2), ("NEG", 1), ("OMOD", 1)], None), // (-v1 + v2) * 2
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = f(l as f32 * 0.25);
        r[2] = f(1.5);
        r[3] = f(f32::INFINITY);
        r[4] = l as u32;
        r[5] = 1000;
        r[6] = 0x0000_AB00 | l as u32;
        r[13] = f(10.0);
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let x = l as f32 * 0.25;
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!(fl(10), x + 1.5, "lane {l} v_add_f32");
        assert_eq!(v[11], 0, "lane {l} v_mul_legacy_f32 0*inf");
        assert_eq!(fl(12), x * 2.0 - 1.0, "lane {l} v_fma_f32");
        assert_eq!(fl(13), 10.0 + x * 1.5, "lane {l} v_mac_f32");
        assert_eq!(v[14], l as u32 + 1000, "lane {l} v_add_nc_u32");
        assert_eq!(v[15], (l as u32) << 4, "lane {l} v_lshlrev_b32");
        assert_eq!(v[16], 0xB, "lane {l} v_bfe_u32");
        assert_eq!(fl(17), 0xAB as f32, "lane {l} v_cvt_f32_ubyte1");
        assert_eq!(v[18], l as u32, "lane {l} v_mbcnt");
        assert_eq!(fl(19), x.clamp(0.0, 1.0), "lane {l} v_med3_f32");
        assert_eq!(fl(20), (1.5 - x) * 2.0, "lane {l} VOP3 neg+omod");
    }
}

/// Output modifiers each instruction kind applies: omod on f64 (clamp stays in double
/// precision) and on the f64 → f32 / f32 → f16 conversions, and the integer clamp (saturation) of the
/// unsigned adds, the packed 16-bit adds and the integer dot products.
#[test]
fn gpu_clamp_and_omod_per_instruction() {
    if !gpu_ready() {
        return;
    }
    let p = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "ENC_VOP3P", f, None);
    let v = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "ENC_VOP3", f, None);
    let code = prog(&[
        // f64: v[1:2] = 0.1, v[3:4] = 0.2, v[5:6] = 1.25
        v("V_ADD_F64", &[("VDST", 10), ("SRC0", V + 1), ("SRC1", V + 3), ("CLAMP", 1)]),
        v("V_ADD_F64", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 3), ("OMOD", 1)]),
        v("V_ADD_F64", &[("VDST", 14), ("SRC0", V + 5), ("SRC1", V + 1), ("CLAMP", 1)]),
        v("V_CVT_F32_F64", &[("VDST", 16), ("SRC0", V + 5), ("CLAMP", 1)]),
        v("V_CVT_F32_F64", &[("VDST", 17), ("SRC0", V + 5), ("OMOD", 3)]),
        // f32 → f16: v7 = 3.0
        v("V_CVT_F16_F32", &[("VDST", 18), ("SRC0", V + 7), ("CLAMP", 1)]),
        v("V_CVT_F16_F32", &[("VDST", 19), ("SRC0", V + 7), ("OMOD", 1)]),
        // unsigned: v8 = 0xFFFFFFF0, v9 = 0x20, v20 = 0x10
        v("V_ADD_NC_U32", &[("VDST", 21), ("SRC0", V + 8), ("SRC1", V + 9), ("CLAMP", 1)]),
        v("V_SUB_NC_U32", &[("VDST", 22), ("SRC0", V + 20), ("SRC1", V + 9), ("CLAMP", 1)]),
        v("V_SUBREV_NC_U32", &[("VDST", 23), ("SRC0", V + 9), ("SRC1", V + 20), ("CLAMP", 1)]),
        v("V_ADD3_U32", &[("VDST", 24), ("SRC0", V + 8), ("SRC1", V + 9), ("SRC2", V + 9), ("CLAMP", 1)]),
        v("V_ADD_NC_U32", &[("VDST", 25), ("SRC0", V + 8), ("SRC1", V + 9)]), // no clamp: wraps
        // packed 16-bit: v26 = (0xFFF0, 0x0005), v27 = (0x0020, 0x0002), v28 = (0x7FF0, 0x8010), v29 = (0x0020, 0xFFE0)
        p("V_PK_ADD_U16", &[("VDST", 30), ("SRC0", V + 26), ("SRC1", V + 27), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        p("V_PK_SUB_U16", &[("VDST", 31), ("SRC0", V + 27), ("SRC1", V + 26), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        p("V_PK_ADD_I16", &[("VDST", 32), ("SRC0", V + 28), ("SRC1", V + 29), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        // halves of src1 swapped: (0x7FF0 − (−32), −32752 − 32)
        p("V_PK_SUB_I16", &[("VDST", 33), ("SRC0", V + 28), ("SRC1", V + 29), ("OP_SEL", 2), ("OP_SEL_HI", 1), ("CLAMP", 1)]),
        p("V_PK_MAD_U16", &[("VDST", 34), ("SRC0", V + 26), ("SRC1", V + 27), ("SRC2", V + 27), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("CLAMP", 1)]),
        p("V_PK_MAD_I16", &[("VDST", 35), ("SRC0", V + 28), ("SRC1", V + 27), ("SRC2", V + 29), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("CLAMP", 1)]),
        p("V_PK_ADD_U16", &[("VDST", 36), ("SRC0", V + 26), ("SRC1", V + 27), ("OP_SEL_HI", 3)]), // no clamp: wraps
        // dot products: v37 = 0x7F7F7F7F, v38 = 0xFFFFFFFF, v39 = 0x7FFFFFF0, v40 = 0x80000010
        p("V_DOT4_I32_I8", &[("VDST", 41), ("SRC0", V + 37), ("SRC1", V + 37), ("SRC2", V + 39), ("CLAMP", 1)]),
        p("V_DOT4_I32_I8", &[("VDST", 42), ("SRC0", V + 37), ("SRC1", V + 38), ("SRC2", V + 40), ("CLAMP", 1)]),
        p("V_DOT4_U32_U8", &[("VDST", 43), ("SRC0", V + 38), ("SRC1", V + 38), ("SRC2", V + 8), ("CLAMP", 1)]),
        p("V_DOT8_I32_I4", &[("VDST", 44), ("SRC0", V + 37), ("SRC1", V + 37), ("SRC2", V + 39), ("CLAMP", 1)]),
        p("V_DOT8_U32_U4", &[("VDST", 45), ("SRC0", V + 38), ("SRC1", V + 38), ("SRC2", V + 8), ("CLAMP", 1)]),
        p("V_DOT2_I32_I16", &[("VDST", 46), ("SRC0", V + 28), ("SRC1", V + 28), ("SRC2", V + 39), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        p("V_DOT2_U32_U16", &[("VDST", 47), ("SRC0", V + 26), ("SRC1", V + 26), ("SRC2", V + 8), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        p("V_DOT4_I32_I8", &[("VDST", 48), ("SRC0", V + 37), ("SRC1", V + 37), ("SRC2", V + 39), ("OP_SEL_HI", 3)]), // no clamp: wraps
    ]);
    let d = |x: f64, r: &mut [u32], i: usize| {
        r[i] = x.to_bits() as u32;
        r[i + 1] = (x.to_bits() >> 32) as u32;
    };
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        d(0.1, r, 1);
        d(0.2, r, 3);
        d(1.25, r, 5);
        r[7] = f(3.0);
        r[8] = 0xFFFF_FFF0;
        r[9] = 0x20;
        r[20] = 0x10;
        r[26] = 0x0005_FFF0;
        r[27] = 0x0002_0020;
        r[28] = 0x8010_7FF0;
        r[29] = 0xFFE0_0020;
        r[37] = 0x7F7F_7F7F;
        r[38] = 0xFFFF_FFFF;
        r[39] = 0x7FFF_FFF0;
        r[40] = 0x8000_0010;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    let d64 = |v: &[u32], i: usize| f64::from_bits(v[i] as u64 | (v[i + 1] as u64) << 32);
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert_eq!(d64(v, 10), 0.1 + 0.2, "lane {l} v_add_f64 clamp keeps double precision");
        assert_eq!(d64(v, 12), (0.1 + 0.2) * 2.0, "lane {l} v_add_f64 omod *2");
        assert_eq!(d64(v, 14), 1.0, "lane {l} v_add_f64 clamp");
        assert_eq!(f32::from_bits(v[16]), 1.0, "lane {l} v_cvt_f32_f64 clamp");
        assert_eq!(f32::from_bits(v[17]), 0.625, "lane {l} v_cvt_f32_f64 omod /2");
        assert_eq!(v[18] & 0xFFFF, 0x3C00, "lane {l} v_cvt_f16_f32 clamp");
        assert_eq!(v[19] & 0xFFFF, 0x4600, "lane {l} v_cvt_f16_f32 omod *2");
        assert_eq!(v[21..26], [0xFFFF_FFFF, 0, 0, 0xFFFF_FFFF, 0x10], "lane {l} add / sub / subrev / add3 u32 clamp, add without");
        assert_eq!(v[30], 0x0007_FFFF, "lane {l} v_pk_add_u16 clamp");
        assert_eq!(v[31], 0x0000_0000, "lane {l} v_pk_sub_u16 clamp");
        assert_eq!(v[32], 0x8000_7FFF, "lane {l} v_pk_add_i16 clamp");
        assert_eq!(v[33], 0x8000_7FFF, "lane {l} v_pk_sub_i16 clamp");
        assert_eq!(v[34], 0x000C_FFFF, "lane {l} v_pk_mad_u16 clamp");
        assert_eq!(v[35], 0x8000_7FFF, "lane {l} v_pk_mad_i16 clamp");
        assert_eq!(v[36], 0x0007_0010, "lane {l} v_pk_add_u16 without clamp");
        assert_eq!(v[41], 0x7FFF_FFFF, "lane {l} v_dot4_i32_i8 clamp high");
        assert_eq!(v[42], 0x8000_0000, "lane {l} v_dot4_i32_i8 clamp low");
        assert_eq!(v[43], 0xFFFF_FFFF, "lane {l} v_dot4_u32_u8 clamp");
        assert_eq!(v[44], 0x7FFF_FFFF, "lane {l} v_dot8_i32_i4 clamp");
        assert_eq!(v[45], 0xFFFF_FFFF, "lane {l} v_dot8_u32_u4 clamp");
        assert_eq!(v[46], 0x7FFF_FFFF, "lane {l} v_dot2_i32_i16 clamp");
        assert_eq!(v[47], 0xFFFF_FFFF, "lane {l} v_dot2_u32_u16 clamp");
        assert_eq!(v[48], 0x7FFF_FFF0u32.wrapping_add(4 * 127 * 127), "lane {l} v_dot4_i32_i8 without clamp");
    }
}

#[test]
fn gpu_bench_really_runs() {
    if !gpu_ready() {
        return;
    }
    println!("GPU: {} / DXC: {}", runner::adapter().unwrap(), super::dxc::location().unwrap().display());
    // v1 = lane * 3 must differ per lane – proves the kernel ran on the GPU, per lane
    let code = prog(&[("V_MUL_U32_U24", "ENC_VOP2", &[("VDST", 1), ("SRC0", 131), ("VSRC1", 0)], None)]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[1] = 0xDEAD;
    }
    // EXEC = only even lanes
    st.sgpr[126] = 0x5555_5555;
    st.sgpr[127] = 0x5555_5555;
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let want = if l % 2 == 0 { 3 * l as u32 } else { 0xDEAD };
        assert_eq!(r.vgpr[l][1], want, "lane {l}");
    }
    assert_eq!((r.sgpr[126], r.sgpr[127]), (0x5555_5555, 0x5555_5555));
}

#[test]
fn gpu_scalar_loop_and_scc() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),             // 0: s0 = 0
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)], None),             // 4: s1 = 10
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)], None),  // 8: loop: s0 += s1
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None), // 12: s1 -= 1
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),         // 16
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),                 // 20: → 8
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),                // v1 = s0
        // SCC rules
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 255), ("SSRC1", 130)], Some(0xFFFF_FFFF)), // carry
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 128), ("SSRC1", 128)], None),             // 0+0+carry
        ("S_ADD_I32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 255), ("SSRC1", 129)], Some(0x7FFF_FFFF)), // overflow
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 135), ("SSRC1", 136)], None),           // scc ? 7 : 8
        ("S_BFE_U32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 255), ("SSRC1", 20)], Some(0x00AB_CD00)), // s20 = (8 << 16) | 8
        ("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 255)], Some(0x0000_0100)),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 126)], None), // popcount(exec)
        ("S_LSHL_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 255), ("SSRC1", 160)], Some(0x0000_0003)), // 3 << 32
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 193), ("SSRC1", 255)], Some(1234)),  // -1 * 1234
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 24), ("SSRC1", 22)], None),
        ("S_BFE_I64", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 24), ("SSRC1", 23)], None),
        ("S_BITCMP1_B64", "ENC_SOPC", &[("SSRC0", 24), ("SSRC1", 26)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 27), ("SSRC0", 135), ("SSRC1", 136)], None),
        ("S_BITCMP0_B64", "ENC_SOPC", &[("SSRC0", 24), ("SSRC1", 28)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 29), ("SSRC0", 135), ("SSRC1", 136)], None),
        ("S_BCNT0_I32_B64", "ENC_SOP1", &[("SDST", 34), ("SSRC0", 24)], None),
        ("S_FF0_I32_B64", "ENC_SOP1", &[("SDST", 35), ("SSRC0", 24)], None),
        ("S_FLBIT_I32_B64", "ENC_SOP1", &[("SDST", 36), ("SSRC0", 24)], None),
        ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 32)], None),
        ("S_BITSET0_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 33)], None),
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 37), ("SSRC0", 24)], None),
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 38), ("SSRC0", 24)], None),
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 39), ("SSRC0", 128)], None),       // 0: nothing differs from the sign
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 40), ("SSRC0", 193)], None),       // -1: all bits are the sign
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 41), ("SSRC0", 44)], None),    // high half 0 → scan the low half
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 42), ("SSRC0", 46)], None),    // negative, ~high = 0 → scan the low half
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 43), ("SSRC0", 128)], None),   // 0 → all bits equal the sign
    ]);
    let mut st = KernelState::default();
    st.sgpr[20] = (8 << 16) | 8;
    st.sgpr[22] = (20 << 16) | 12;
    st.sgpr[23] = (4 << 16) | 60;
    st.sgpr[24] = 0x7654_3210;
    st.sgpr[25] = 0xFEDC_BA98;
    st.sgpr[26] = 63;
    st.sgpr[28] = 0;
    st.sgpr[30] = 1 << 4;
    st.sgpr[31] = 0;
    st.sgpr[32] = 36;
    st.sgpr[33] = 4;
    st.sgpr[44] = 0x8000;                                                        // s[44:45] = 0x0000_0000_0000_8000
    st.sgpr[45] = 0;
    st.sgpr[46] = 0x8000_0000;                                                   // s[46:47] = 0xFFFF_FFFF_8000_0000
    st.sgpr[47] = 0xFFFF_FFFF;
    let r = runner::run_code(&code, &st, &[]).unwrap();
    assert_eq!(r.sgpr[0], 55, "loop");
    assert!(r.vgpr.iter().all(|v| v[1] == 55));
    assert_eq!(r.sgpr[2], 1, "s_add_u32 wraps");
    assert_eq!(r.sgpr[3], 1, "s_addc_u32 uses the carry");
    assert_eq!(r.sgpr[4], 0x8000_0000, "s_add_i32");
    assert_eq!(r.sgpr[5], 7, "s_cselect after signed overflow → scc=1");
    assert_eq!(r.sgpr[6], 0xCD, "s_bfe_u32");
    assert_eq!(r.sgpr[7], 8, "s_ff1");
    assert_eq!(r.sgpr[8], 64, "s_bcnt1 exec");
    assert_eq!((r.sgpr[10], r.sgpr[11]), (0, 3), "s_lshl_b64");
    assert_eq!(r.sgpr[12] as i32, -1234, "s_mul_i32");
    assert_eq!((r.sgpr[14], r.sgpr[15]), (0x0007_6543, 0), "s_bfe_u64");
    assert_eq!((r.sgpr[16], r.sgpr[17]), (u32::MAX, u32::MAX), "s_bfe_i64 sign extension");
    assert_eq!((r.sgpr[27], r.sgpr[29]), (7, 7), "s_bitcmp1/0_b64");
    assert_eq!((r.sgpr[34], r.sgpr[35], r.sgpr[36]), (32, 0, 0), "s_bcnt0/ff0/flbit 64");
    assert_eq!((r.sgpr[30], r.sgpr[31]), (0, 1 << 4), "s_bitset1/0_b64");
    assert_eq!((r.sgpr[37], r.sgpr[38]), (1, 7), "signed scalar leading-bit scans");
    assert_eq!((r.sgpr[39], r.sgpr[40]), (u32::MAX, u32::MAX), "flbit: nothing differs from the sign");
    assert_eq!((r.sgpr[41], r.sgpr[42], r.sgpr[43]), (48, 33, u32::MAX), "flbit_i64 high-then-low scan");
}

#[test]
fn gpu_divergence_through_exec() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None), // vcc = 32 > lane
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 129)], None), // lanes 0..31: 1
        ("S_XOR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 130)], None), // lanes 32..63: 2
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 2)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 128), ("VSRC1", 3)], None), // vcc ? v3 : 0
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 3)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V), ("SRC1", 255)], Some(45)),
        // nobody matches: exec = 0 → the block is skipped
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(1000)), // 36: (literal: 8 bytes)
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None),                    // 44 → 56
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 192)], None), // 48 (skipped)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 129)], None),             // 52 (skipped)
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),           // 56: exec = -1
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 8), ("SDST", 20), ("SRC0", V + 9), ("SRC1", 255)], Some(0x8000_0000)),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[3] = 7 + l as u32;
        r[6] = 0x66;
        r[9] = if l % 4 == 0 { 0x8000_0000 } else { 1 };
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!(r.vgpr[l][1], if l < 32 { 1 } else { 2 }, "lane {l} masked writes");
        assert_eq!(r.vgpr[l][2], if l < 32 { 7 + l as u32 } else { 0 }, "lane {l} cndmask");
        assert_eq!(r.vgpr[l][6], 0x66, "lane {l}: skipped block must not run");
    }
    assert_eq!((r.sgpr[2], r.sgpr[3]), (u32::MAX, u32::MAX), "saved exec");
    assert_eq!((r.sgpr[106], r.sgpr[107]), (u32::MAX, 0), "vcc from v_cmp");
    assert_eq!(r.sgpr[4], 7, "readfirstlane");
    assert_eq!(r.sgpr[5], 45, "readlane 45");
    assert_eq!(r.sgpr[7], 0, "scalar code in the skipped block");
    assert_eq!((r.sgpr[126], r.sgpr[127]), (u32::MAX, u32::MAX));
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0x1111_1111, 0x1111_1111), "carry mask of v_add_co_u32");
}

#[test]
fn gpu_memory_and_exports() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0x10), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SBASE", 6), ("OFFSET", 4), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 11), ("SBASE", 6), ("OFFSET", 0x400), ("SOFFSET", 125)], None), // out of range
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 8), ("OFFEN", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 8)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 3), ("VSRC1", 4), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1)], None),
        ("V_CVT_PKRTZ_F16_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 242), ("VSRC1", 6)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0x3), ("VSRC0", 5), ("VSRC1", 5), ("COMPR", 1), ("DONE", 1), ("VM", 1)], None),
    ]);
    let mut mem = vec![0u32; 256];
    for (i, m) in mem.iter_mut().enumerate() {
        *m = 0x1000 + i as u32;
    }
    let mut st = KernelState::default();
    st.sgpr[0] = 0; // flat pointer
    st.sgpr[12] = 0x40; // V#: base 0x40, stride 0, 0x100 bytes
    st.sgpr[13] = 0;
    st.sgpr[14] = 0x100;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 4 * l as u32;
        r[6] = f(0.5);
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!((r.sgpr[8], r.sgpr[9]), (0x1004, 0x1005), "s_load_dwordx2");
    assert_eq!(r.sgpr[10], 0x1000 + 0x44 / 4, "s_buffer_load_dword");
    assert_eq!(r.sgpr[11], 0, "s_buffer_load out of range → 0");
    for l in 0..64 {
        let want = if 8 + 4 * l < 0x100 { 0x1000 + (0x40 + 8 + 4 * l as u32) / 4 } else { 0 };
        assert_eq!(r.vgpr[l][3], want, "lane {l} buffer_load_dword offen");
        assert_eq!(r.export(12, l, 0), want, "lane {l} pos0.x");
        assert_eq!(r.export(12, l, 1), 0x1004, "lane {l} pos0.y");
        assert_eq!(r.export(12, l, 2), 4 * l as u32, "lane {l} pos0.z");
        assert_eq!((r.export(0, l, 0), r.export(0, l, 1)), (0x3C00, 0x3800), "lane {l} mrt0 compr (1.0, 0.5)");
    }
}

/// Multi-dword loads as the emitter groups them (four dwords per memory access, bounds checks
/// without branches): an unaligned `s_load`, `s_buffer_load`s that straddle num_records, and
/// vector buffer loads with lanes past the end and inactive lanes (which keep their registers).
#[test]
fn gpu_grouped_loads_bounds_and_inactive_lanes() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 0), ("OFFSET", 0x22), ("SOFFSET", 125)], None), // low bits ignored
        ("S_BUFFER_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)], None), // 5 of 8 in range
        ("S_BUFFER_LOAD_DWORDX16", "ENC_SMEM", &[("SDATA", 32), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 48), ("SBASE", 2), ("OFFSET", 16), ("SOFFSET", 125)], None), // 1 of 2
        ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 48), ("VSRC1", 1)], None), // lanes 0..47
        ("BUFFER_LOAD_DWORDX3", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 1), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_DWORDX2", "ENC_MUBUF", &[("VDATA", 13), ("VADDR", 2), ("SRSRC", 13), ("SOFFSET", 128), ("OFFEN", 1)], None),
    ]);
    let mut mem = vec![0u32; 1024];
    for (i, m) in mem.iter_mut().enumerate() {
        *m = 0x1000 + i as u32;
    }
    let mut st = KernelState::default();
    st.sgpr[0] = 0; // flat pointer 0
    st.sgpr[4..8].copy_from_slice(&[0x80, 0, 20, 0]); // V#: base 0x80, 20 bytes
    st.sgpr[8..12].copy_from_slice(&[0x100, 0, 0x100, 0]); // V#: base 0x100, 0x100 bytes
    st.sgpr[12..16].copy_from_slice(&[0x100, 12 << 16, 40, 0]); // V#: base 0x100, stride 12, 40 records
    st.sgpr[52..56].copy_from_slice(&[0x400, 0, 0x104, 0]); // V#: base 0x400, 0x104 bytes
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = l as u32;
        r[2] = 8 * l as u32;
        for k in 10..15 {
            r[k] = 0xDEAD;
        }
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for i in 0..8 {
        assert_eq!(r.sgpr[16 + i], 0x1008 + i as u32, "s_load_dwordx8 dword {i}");
        assert_eq!(r.sgpr[24 + i], if i < 5 { 0x1020 + i as u32 } else { 0 }, "s_buffer_load_dwordx8 dword {i}");
    }
    for i in 0..16 {
        assert_eq!(r.sgpr[32 + i], 0x1040 + i as u32, "s_buffer_load_dwordx16 dword {i}");
    }
    assert_eq!((r.sgpr[48], r.sgpr[49]), (0x1024, 0), "s_buffer_load_dwordx2 across num_records");
    for l in 0..64usize {
        for k in 0..3usize {
            let want = if l >= 48 { 0xDEAD } else if l < 40 { 0x1040 + 3 * l as u32 + k as u32 } else { 0 };
            assert_eq!(r.vgpr[l][10 + k], want, "lane {l} buffer_load_dwordx3 (idxen) dword {k}");
        }
        for k in 0..2usize {
            let off = 8 * l + 4 * k;
            let want = if l >= 48 { 0xDEAD } else if off + 4 <= 0x104 { 0x1100 + (off / 4) as u32 } else { 0 };
            assert_eq!(r.vgpr[l][13 + k], want, "lane {l} buffer_load_dwordx2 (offen) dword {k}");
        }
    }
}

/// A real console vertex shader from the retail pack, run on the PC GPU
/// (needs XCL_SAMPLE_RPF=…\shaders.rpf).
#[test]
fn gpu_retail_vs_blit_gen_amb_cube() {
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else {
        println!("XCL_SAMPLE_RPF not set – skipping");
        return;
    };
    if !gpu_ready() {
        return;
    }
    let data = std::fs::read(&path).unwrap();
    let rec = crate::gpu::pack::scan(&data, 0x40).into_iter().find(|r| r.name == "VS_BlitGenAmbCube_0cfdb358_Wrapped").expect("record");
    let blob = crate::gpu::pack::inflate(&data, &rec).unwrap();
    let sh = crate::gpu::xbox::parse(&blob).unwrap();
    let fs = &sh.fssm[0];
    let p = fs.programs[0];
    let code = &fs.data[p.offset..p.offset + p.len];
    let t = translate(code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported, t.hlsl);
    println!("{}", t.hlsl);
    // the fetch shader would load POSITION into v4..; give every lane its own vertex
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[4] = f((l % 8) as f32 / 7.0);
        r[5] = f((l / 8) as f32 / 7.0);
    }
    let (r, _) = runner::run_program(code, &st, &[]).unwrap();
    for l in 0..64 {
        let x = (l % 8) as f32 / 7.0;
        let y = (l / 8) as f32 / 7.0;
        let pos: Vec<f32> = (0..4).map(|c| f32::from_bits(r.export(12, l, c))).collect();
        assert_eq!(pos, vec![x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0], "lane {l}: SV_Position");
        let p0: Vec<f32> = (0..4).map(|c| f32::from_bits(r.export(32, l, c))).collect();
        assert_eq!(p0, vec![0.0, 0.0, 0.0, 1.0], "lane {l}: param0");
    }
}

#[test]
fn gpu_sdwa_dpp_and_cross_lane() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // v10 = (float)(v1 byte 2)      SRC0=0xF9 (SDWA), VSRC0=1, SRC0_SEL=BYTE_2, DST_SEL=DWORD
        ("V_CVT_F32_U32", "VOP1_VOP_SDWA", &[("VDST", 10), ("SRC0", 0xF9), ("VSRC0", 1), ("SRC0_SEL", 2), ("DST_SEL", 6)], None),
        // v11 = sext(v1 word1) + v2 written into word 0 of v11, upper word preserved
        ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 2), ("SRC0_SEL", 5), ("SRC0_SEXT", 1), ("SRC1_SEL", 6), ("DST_SEL", 4), ("DST_UNUSED", 2)], None),
        // v12 = v0 of lane (lane ^ 1 within quad): quad_perm [1,0,3,2] = 0b10_11_00_01 = 0xB1
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 12), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0xB1), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
        // v13 = v0 of lane-1 within row; lane 0 of each row has no source → 0 (bound_ctrl)
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 13), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF), ("BOUND_CTRL", 1)], None),
        // v14 = row_mirror, rows 1 and 3 only (row_mask 0b1010)
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 14), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xA), ("BANK_MASK", 0xF)], None),
        // v15 = v0 of lane 7 - (lane & 7) in each group of 8 (DPP8)
        ("V_MOV_B32", "VOP1_VOP_DPP8", &[("VDST", 15), ("SRC0", 0xE9), ("VSRC0", 0), ("LANE_SEL_0", 7), ("LANE_SEL_1", 6), ("LANE_SEL_2", 5), ("LANE_SEL_3", 4), ("LANE_SEL_4", 3), ("LANE_SEL_5", 2), ("LANE_SEL_6", 1), ("LANE_SEL_7", 0)], None),
        // v16 = swizzle: lane ^ 4 (BitMode: and 0x1F, or 0, xor 4)
        ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 16), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 4 << 2)], None),
        // v17 = permlanex16 with identity selects → lane ^ 16
        ("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V), ("SRC1", 255), ("SRC2", 3)], Some(0x7654_3210)),
        // v18 = fma_mix(f16 lo of v3, f16 hi of v3, f32 v4) = 1.5 * 2.0 + 0.25
        ("V_FMA_MIX_F32", "ENC_VOP3P", &[("VDST", 18), ("SRC0", V + 3), ("SRC1", V + 3), ("SRC2", V + 4), ("OP_SEL", 0b010), ("OP_SEL_HI", 0b11), ("OP_SEL_HI_2", 0)], None),
        // v19 = cube face of (v5, v6, v7)
        ("V_CUBEID_F32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 7)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[3] = 0xFEDC_BA98; // permlanex16 selects for lanes 8..15 of a row
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 100 + l as u32;
        r[1] = 0x8001_0000 | ((l as u32) << 16 & 0x00F0_0000) ;
        r[2] = 10;
        r[11] = 0xAAAA_5555;
        r[14] = 0xDEAD;
        r[3] = 0x4000_3E00; // hi = 2.0h, lo = 1.5h
        r[4] = f(0.25);
        let axis = l % 3;
        r[5] = f(if axis == 0 { -3.0 } else { 0.5 });
        r[6] = f(if axis == 1 { 3.0 } else { 0.5 });
        r[7] = f(if axis == 2 { -3.0 } else { 0.5 });
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let v1 = 0x8001_0000u32 | ((l as u32) << 16 & 0x00F0_0000);
        assert_eq!(f32::from_bits(v[10]), ((v1 >> 16) & 0xFF) as f32, "lane {l} sdwa byte2");
        let w1 = ((v1 >> 16) as u16 as i16 as i32) as u32;
        assert_eq!(v[11], 0xAAAA_0000 | (w1.wrapping_add(10) & 0xFFFF), "lane {l} sdwa word1 sext + preserve");
        assert_eq!(v[12], 100 + (l ^ 1) as u32, "lane {l} quad_perm");
        assert_eq!(v[13], if l % 16 == 0 { 0 } else { 100 + l as u32 - 1 }, "lane {l} row_shr:1 bound_ctrl");
        let row = l / 16;
        assert_eq!(v[14], if row % 2 == 1 { 100 + (row * 16 + 15 - l % 16) as u32 } else { 0xDEAD }, "lane {l} row_mirror row_mask");
        assert_eq!(v[15], 100 + ((l & !7) | (7 - (l & 7))) as u32, "lane {l} dpp8");
        assert_eq!(v[16], 100 + (l ^ 4) as u32, "lane {l} ds_swizzle");
        let n = if l % 16 < 8 { (0x7654_3210u32 >> (4 * (l % 8))) & 15 } else { (0xFEDC_BA98u32 >> (4 * (l % 8))) & 15 };
        assert_eq!(v[17], 100 + ((l & !31) | ((l & 16) ^ 16) | n as usize) as u32, "lane {l} permlanex16");
        assert_eq!(f32::from_bits(v[18]), 1.5 * 2.0 + 0.25, "lane {l} fma_mix");
        let face = match l % 3 {
            0 => 1.0,
            1 => 2.0,
            _ => 5.0,
        };
        assert_eq!(f32::from_bits(v[19]), face, "lane {l} cubeid");
    }
}

#[test]
fn buffer_format_table_matches_llvm() {
    // every BUF_FMT_* in LLVM's gfx10 MTBUF tests must decode to the same name
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/llvm-amdgpu");
    let mut checked = 0;
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        for line in std::fs::read_to_string(e.path()).unwrap_or_default().lines() {
            let Some(i) = line.find("format:[BUF_FMT_") else { continue };
            if !line.starts_with("# GFX10") {
                continue;
            }
            let name = &line[i + 16..line[i..].find(']').unwrap() + i];
            let Some(enc) = line.split("encoding: [").nth(1) else { continue };
            let b: Vec<u8> = enc.trim_end_matches(']').split(',').filter_map(|x| u8::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok()).collect();
            let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            assert_eq!(super::emit::BUF_FORMATS[((w >> 19) & 0x7F) as usize], name, "{line}");
            checked += 1;
        }
    }
    assert!(checked >= 10, "{checked}");
}

#[test]
fn buffer_format_d16_packs_and_unpacks_halves() {
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_D16_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_D16_HI_X", "ENC_MUBUF", &[("VDATA", 12), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("TBUFFER_STORE_FORMAT_D16_XYZ", "ENC_MTBUF", &[("VDATA", 14), ("SRSRC", 4), ("SOFFSET", 128), ("FORMAT", 71)], None),
        ("BUFFER_STORE_FORMAT_D16_HI_X", "ENC_MUBUF", &[("VDATA", 16), ("SRSRC", 3), ("SOFFSET", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("v10 = t1_ok ? (select(t1_int") && t.hlsl.contains("v11 = t1_ok ? (select(t1_int"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v12 = (v12 & 0xFFFFu) | (t2_ok ? (select(t2_int") && t.hlsl.contains("<< 16) : 0u)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint t3_ty = (t3_info >> 3) & 7u") && t.hlsl.contains("f16tof32(((v14) & 0xFFFFu))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("f16tof32(((v16 >> 16) & 0xFFFFu))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

#[test]
fn gpu_typed_buffer_loads_and_stores() {
    if !gpu_ready() {
        return;
    }
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let bgra = 6 | 5 << 3 | 4 << 6 | 7 << 9;
    let vs = |base: u32, fmt: u32, sel: u32| [base, 0, 0x100, sel | fmt << 12];
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),             // A
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 14), ("SRSRC", 4), ("SOFFSET", 128)], None),             // B
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 18), ("SRSRC", 5), ("SOFFSET", 128)], None),             // C
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 22), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 2)], None), // D (unaligned)
        ("BUFFER_LOAD_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 23), ("SRSRC", 7), ("SOFFSET", 128)], None),              // E
        ("TBUFFER_LOAD_FORMAT_XY", "ENC_MTBUF", &[("VDATA", 26), ("SRSRC", 8), ("SOFFSET", 128), ("FORMAT", 64)], None), // F
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 28), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 0x200)], None), // out of range
        ("BUFFER_LOAD_FORMAT_D16_XY", "ENC_MUBUF", &[("VDATA", 31), ("SRSRC", 4), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", &[("VDATA", 32), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 2)], None),
        ("TBUFFER_LOAD_FORMAT_D16_XY", "ENC_MTBUF", &[("VDATA", 33), ("SRSRC", 8), ("SOFFSET", 128), ("FORMAT", 64)], None),
        ("BUFFER_STORE_FORMAT_D16_XY", "ENC_MUBUF", &[("VDATA", 34), ("SRSRC", 10), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 35), ("SRSRC", 11), ("SOFFSET", 128)], None),
        ("BUFFER_STORE_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 35), ("SRSRC", 13), ("SOFFSET", 128)], None),
        ("V_CVT_PK_U8_F32", "ENC_VOP3", &[("VDST", 29), ("SRC0", V + 30), ("SRC1", 130), ("SRC2", 255)], Some(0x1122_3344)),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 1), ("SRSRC", 9), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("S_BITSET1_B32", "ENC_SOP1", &[("SDST", 50), ("SSRC0", 133)], None),
    ]);
    let mut mem = vec![0u32; 512];
    mem[0x100 / 4] = u32::from_le_bytes([0x00, 0x80, 0xFF, 0x40]); // A: 8_8_8_8_UNORM
    mem[0x110 / 4] = 0x3C00 | 0xC000 << 16; // B: 16_16_FLOAT (1.0, -2.0)
    mem[0x114 / 4] = 1023 | 512 << 20 | 3 << 30; // C: 2_10_10_10_UNORM
    mem[0x118 / 4] = 0x8001 << 16; // D: 16_SNORM at byte 0x11A
    mem[0x11C / 4] = 0x3C0 | 0x400 << 11 | 0x1C0 << 22; // E: 10_11_11_FLOAT (1.0, 2.0, 0.5)
    mem[0x120 / 4] = f(1.5);
    mem[0x124 / 4] = f(-3.25); // F: 32_32_FLOAT
    mem[0x140 / 4..0x150 / 4].copy_from_slice(&[f(1.25), f(-2.5), f(3.75), f(-4.5)]);
    let mut st = KernelState::default();
    for (i, v) in [
        vs(0x100, 56, bgra),
        vs(0x110, 29, ident),
        vs(0x114, 50, ident),
        vs(0x118, 8, ident),
        vs(0x11C, 36, ident),
        vs(0x120, 0, ident),
        [0x400, 0, 0x100, 0],
    ]
    .iter()
    .enumerate()
    {
        st.sgpr[12 + 4 * i..16 + 4 * i].copy_from_slice(v);
    }
    st.sgpr[40..44].copy_from_slice(&vs(0x130, 29, ident));
    st.sgpr[44..48].copy_from_slice(&vs(0x140, 77, 0 | 1 << 3 | 4 << 6 | 7 << 9));
    st.sgpr[52..56].copy_from_slice(&vs(0x150, 77, ident));
    st.sgpr[50] = 1;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 0xC0DE_0000 | l as u32;
        r[1] = 4 * l as u32;
        r[30] = f(200.6);
        r[34] = 0xB800_4000; // packed f16: +2.0, -0.5
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let fl = |x: u32| f32::from_bits(x);
    // normalised mid values: within 2 ULP (DXC divides approximately); endpoints exact
    let near = |got: u32, want: f32| (got as i64 - want.to_bits() as i64).abs() <= 2;
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert!(
            v[10] == f(1.0) && near(v[11], 128.0 / 255.0) && v[12] == 0 && near(v[13], 64.0 / 255.0),
            "lane {l} A unorm8 bgra {:?}",
            [fl(v[10]), fl(v[11]), fl(v[12]), fl(v[13])]
        );
        assert_eq!([fl(v[14]), fl(v[15]), fl(v[16]), fl(v[17])], [1.0, -2.0, 0.0, 1.0], "lane {l} B 16_16_float");
        assert!(
            v[18] == f(1.0) && v[19] == 0 && near(v[20], 512.0 / 1023.0) && v[21] == f(1.0),
            "lane {l} C 2_10_10_10 {:?}",
            [fl(v[18]), fl(v[19]), fl(v[20]), fl(v[21])]
        );
        assert_eq!(fl(v[22]), -1.0, "lane {l} D 16_snorm unaligned");
        assert_eq!([fl(v[23]), fl(v[24]), fl(v[25])], [1.0, 2.0, 0.5], "lane {l} E 10_11_11_float");
        assert_eq!([fl(v[26]), fl(v[27])], [1.5, -3.25], "lane {l} F tbuffer 32_32_float");
        assert_eq!(v[28], 0, "lane {l} out of range");
        assert_eq!(v[31], 0xC000_3C00, "lane {l} d16 16_16_float");
        assert_eq!(v[32], 0x0000_BC00, "lane {l} d16 16_snorm");
        assert_eq!(v[33], 0xC280_3E00, "lane {l} d16 tbuffer 32_32_float");
        assert_eq!([fl(v[35]), fl(v[36]), fl(v[37]), fl(v[38])], [0.0, 1.0, 1.25, -4.5], "lane {l} 32_32_32_32 selectors");
        assert_eq!(v[29], 0x11C9_3344, "lane {l} v_cvt_pk_u8_f32 into byte 2");
        assert_eq!(r.mem[0x400 / 4 + l], 0xC0DE_0000 | l as u32, "lane {l} buffer_store_dword");
    }
    assert_eq!(r.mem[0x130 / 4], 0xB800_4000, "d16 store 16_16_float");
    assert_eq!(&r.mem[0x150 / 4..0x160 / 4], &[f(0.0), f(1.0), f(1.25), f(-4.5)], "32_32_32_32 store");
    assert_eq!(r.sgpr[50], 1 | 1 << 5, "s_bitset1");
}

fn el(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> crate::gpu::dxbc::SigElement {
    crate::gpu::dxbc::SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

/// VS + PS pair recompiled to graphics stages and drawn on the GPU: VS inputs from the
/// fetch-shader VGPRs, exp pos0/param0, PS interpolation through v_interp, discard via v_cmpx.
#[test]
fn gpu_graphics_vs_ps_draw() {
    if !gpu_ready() {
        return;
    }
    use super::{PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_CMPX_LT_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(8.0f32.to_bits())), // keep pos.x > 8
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 2)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 240), ("VSRC1", 4)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let (vo, po) = (Options { mode: Mode::Vertex(vsi), wave64: true, ..Default::default() }, Options { mode: Mode::Pixel(psi), wave64: true, ..Default::default() });
    let vt = translate(&vs_code, &vo).unwrap();
    let pt = translate(&ps_code, &po).unwrap();
    assert!(vt.complete() && pt.complete(), "{:?} {:?}", vt.unsupported, pt.unsupported);
    let vs = direct_or(&vs_code, &vo, super::dxc::compile(&vt.hlsl, "main", "vs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", vt.hlsl)));
    let ps = direct_or(&ps_code, &po, super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl)));
    let c = [0.8f32, 0.6, 0.4, 1.0];
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], c[0], c[1], c[2], c[3]]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    let clear = [0.0, 0.0, 1.0, 1.0];
    let px = runner::draw(&vs, &ps, &layout, &verts, 24, &[], &[], 16, clear).unwrap();
    for y in 0..16 {
        for x in 0..16 {
            let want = if x >= 8 { [0.4, 0.6, 0.4, 1.0] } else { clear };
            let got = px[y * 16 + x];
            assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "pixel ({x},{y}) = {got:?}, expected {want:?}");
        }
    }
}

/// Build the four pixel-shader `EXP` `VM` test cases for XCL-BT-012 (#1488):
/// (name, code, uses_vm_semantics). Each shades `x >= 8` as `(0.4, 0.6, 0.4, 1.0)` and
/// discards `x < 8`.
fn ps_exp_vm_cases() -> [(&'static str, Vec<u8>, bool); 4] {
    let setup = [
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)][..], None), // save entry EXEC
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 255)], Some(0.4f32.to_bits())),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 255)], Some(0.6f32.to_bits())),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 255)], Some(0.4f32.to_bits())),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
    ];
    let narrow = ("V_CMPX_LT_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)][..], Some(8.0f32.to_bits()));
    let restore = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)][..], None);
    let build = |tail: &[(&str, &str, &[(&str, u64)], Option<u32>)]| {
        let mut all = setup.to_vec();
        all.extend_from_slice(tail);
        prog(&all)
    };
    [
        // A: VM=1 DONE=0 under narrowed EXEC, then restored EXEC + VM=0 DONE=1 must not revive x < 8.
        ("vm1_done0_then_vm0_done1", build(&[
            narrow,
            ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("VM", 1)], None),
            restore,
            ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("DONE", 1)], None),
        ]), true),
        // B: Two VM=1 exports, second narrower — must overwrite (not OR) the earlier VM=1 mask.
        ("two_vm1_second_narrower_overwrites", build(&[
            ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("VM", 1)], None),
            narrow,
            ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("VM", 1)], None),
            restore,
            ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("DONE", 1)], None),
        ]), true),
        // C: Undeclared MRT1 export with VM=1 DONE=0 followed by restored EXEC + MRT0 VM=0 DONE=1.
        ("undeclared_mrt_vm1_done0_then_mrt0_vm0_done1", build(&[
            narrow,
            ("EXP", "ENC_EXP", &[("TGT", 1), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("VM", 1)], None),
            restore,
            ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1)], None),
        ]), true),
        // D: Legacy synthetic shader with zero VM=1 exports falls back to DONE=1.
        ("no_vm_falls_back_to_done1", build(&[
            narrow,
            ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1)], None),
        ]), false),
    ]
}

/// CPU check for XCL-BT-012 (#1488): pixel-shader `EXP` `VM` codegen in HLSL (`emit.rs`)
/// and container validation in Direct DXIL (`dxil/lower_gfx.rs`).
#[test]
fn ps_exp_vm_hlsl_and_direct_dxil_codegen() {
    use super::PsInterface;
    use crate::gpu::dxil::kernel::{translate_pixel, KernelOptions};
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let po = Options { mode: Mode::Pixel(psi.clone()), wave64: true, ..Default::default() };
    for (name, ps_code, has_vm) in ps_exp_vm_cases() {
        let pt = translate(&ps_code, &po).unwrap();
        assert!(pt.complete(), "{name}: {:?}", pt.unsupported);
        assert!(pt.hlsl.contains("if (!xcl_live) discard;"), "{name}: missing discard:\n{}", pt.hlsl);
        if has_vm {
            assert!(pt.hlsl.contains("xcl_live = ("), "{name}: VM=1 must overwrite xcl_live:\n{}", pt.hlsl);
            assert!(!pt.hlsl.contains("xcl_live = true;"), "{name}: VM=0 DONE=1 must not OR-set xcl_live:\n{}", pt.hlsl);
        } else {
            assert!(pt.hlsl.contains("xcl_live = true;"), "{name}: no-VM fallback must keep DONE=1:\n{}", pt.hlsl);
        }
        if super::dxc::location().is_ok() {
            super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{name} DXC: {e}\n{}", pt.hlsl));
            let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
            let dxil = translate_pixel(&ps_code, ko, &psi).unwrap_or_else(|e| panic!("{name} direct DXIL: {e}")).dxil;
            crate::gpu::dxil::validate::validate(&dxil).unwrap_or_else(|e| panic!("{name} DXIL validate: {e}"));
        }
    }
}

/// GPU draw regression for XCL-BT-012 (#1488): verifies all four `EXP` `VM` pixel-survival
/// cases on the GPU through both the HLSL and Direct DXIL pixel-shader backends.
#[test]
fn gpu_ps_exp_vm_controls_pixel_survival() {
    if !gpu_ready() {
        return;
    }
    use super::{PsInterface, VsInterface};
    use crate::gpu::dxil::kernel::{translate_pixel, translate_vertex, KernelOptions};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)],
        attr_base: 4,
        params: None,
    };
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let vo = Options { mode: Mode::Vertex(vsi.clone()), wave64: true, ..Default::default() };
    let po = Options { mode: Mode::Pixel(psi.clone()), wave64: true, ..Default::default() };
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    let vs_hlsl = super::dxc::compile(&translate(&vs_code, &vo).unwrap().hlsl, "main", "vs_6_6", &[]).unwrap();
    let vs_dxil = translate_vertex(&vs_code, ko, &vsi).unwrap().dxil;
    let verts = [-1.0f32, -1.0, -1.0, 3.0, 3.0, -1.0];
    let layout = [runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 }];
    let clear = [0.0, 0.0, 1.0, 1.0];
    let shaded = [0.4, 0.6, 0.4, 1.0];

    for (name, ps_code, _) in ps_exp_vm_cases() {
        let ps_hlsl = super::dxc::compile(&translate(&ps_code, &po).unwrap().hlsl, "main", "ps_6_6", &[]).unwrap();
        let ps_dxil = translate_pixel(&ps_code, ko, &psi).unwrap().dxil;
        for (backend, vs, ps) in [("hlsl", &vs_hlsl, &ps_hlsl), ("direct_dxil", &vs_dxil, &ps_dxil)] {
            let px = runner::draw(vs, ps, &layout, &verts, 8, &[], &[], 16, clear).unwrap();
            for y in 0..16 {
                for x in 0..16 {
                    let want = if x >= 8 { shaded } else { clear };
                    let got = px[y * 16 + x];
                    assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "{name} ({backend}) pixel ({x},{y}) = {got:?}, expected {want:?}");
                }
            }
        }
    }
}

/// `emit::rect_list_gs` declares the VS's output struct; the fourth vertex is v[0] with its
/// float outputs continued from the other three, its integer (flat) ones left as v[0]'s.
#[test]
fn rect_list_gs_outputs() {
    let outputs = [el("SV_Position", 0, 1, 3, 0, 0xF), el("TEXCOORD", 0, 0, 3, 1, 0xF), el("TEXCOORD", 1, 0, 1, 2, 0xF), el("SV_ClipDistance", 0, 2, 3, 3, 1)];
    let h = super::emit::rect_list_gs(&outputs);
    assert!(h.contains("triangle XclOut v[3]") && h.contains("XclOut d = v[0];"), "{h}");
    assert!(h.contains("d.p0 = a.p0 + b.p0 - k.p0;") && h.contains("d.p2 = a.p2 + b.p2 - k.p2;") && !h.contains("d.p1 ="), "{h}");
    if super::dxc::location().is_ok() {
        super::dxc::compile(&h, "main", "gs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{h}"));
    }
}

/// `emit::rect_list_gs` on the full-screen pass a downstream build (theaustinayers/xcl, build 5,
/// via #189) reports from a retail title: positions (−1,1) (−1,−1) (1,−1), texture coordinates
/// (0,0) (0,1) (1,1), the shared corner v1 and the implied one (1,1) with (1,0). Plain HLSL
/// VS/PS around the GS. Every pixel must be drawn with its texture coordinate, for w 1 and 2;
/// without the GS about half are. The other half of that rect as a triangle (v0, v2 and the
/// implied corner – what a draw expanded to two triangles a rect on the CPU sends next) is three
/// corners of the same rect, so the GS completes it to the whole rect as well: such a draw must
/// not also get this GS, or each rect is drawn twice.
#[test]
fn gpu_rect_list_fullscreen_order() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let outputs = [el("SV_Position", 0, 1, 3, 0, 0xF), el("TEXCOORD", 0, 0, 3, 1, 0xF)];
    let vs_hlsl = "struct V { float4 pos : SV_Position; float4 uv : TEXCOORD0; };\nV main(float4 pos : POSITION, float4 uv : TEXCOORD0) { V o; o.pos = pos; o.uv = uv; return o; }\n";
    let ps_hlsl = "float4 main(float4 pos : SV_Position, float4 uv : TEXCOORD0) : SV_Target { return uv; }\n";
    let vs = super::dxc::compile(vs_hlsl, "main", "vs_6_6", &[]).unwrap();
    let ps = super::dxc::compile(ps_hlsl, "main", "ps_6_6", &[]).unwrap();
    let gs_hlsl = super::emit::rect_list_gs(&outputs);
    let gs = super::dxc::compile(&gs_hlsl, "main", "gs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{gs_hlsl}"));
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "TEXCOORD", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 16 },
    ];
    let (n, clear) = (16usize, [0.0f32, 0.0, 1.0, 0.5]);
    let first = [([-1.0f32, 1.0], [0.0f32, 0.0]), ([-1.0, -1.0], [0.0, 1.0]), ([1.0, -1.0], [1.0, 1.0])];
    let second = [([-1.0f32, 1.0], [0.0f32, 0.0]), ([1.0, -1.0], [1.0, 1.0]), ([1.0, 1.0], [1.0, 0.0])];
    for (name, corners, w) in [("v0 v1 v2", first, 1.0f32), ("v0 v1 v2", first, 2.0), ("v0 v2 v3", second, 1.0)] {
        let v: Vec<f32> = corners.iter().flat_map(|&([x, y], [s, t])| [x * w, y * w, 0.0, w, s, t, 0.0, 1.0]).collect();
        let run = |with_gs: bool| runner::draw_gs(&vs, with_gs.then_some(gs.as_slice()), &ps, &layout, &v, 32, &[], &[], n as u32, clear).unwrap();
        let px = run(true);
        for y in 0..n {
            for x in 0..n {
                let want = [(x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32, 0.0, 1.0];
                let got = px[y * n + x];
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "{name}, w {w}: pixel ({x},{y}) = {got:?}, expected {want:?}");
            }
        }
        // the negative control: the three vertices alone draw half the rect
        let drawn = run(false).iter().filter(|p| **p != clear).count();
        assert!(drawn > n * n / 3 && drawn < n * n * 2 / 3, "{name}, w {w}: without the GS {drawn} of {} pixels drawn", n * n);
    }
}

/// A console rect list with flat outputs – an integer one, and a float one the PS reads
/// `nointerpolation` (what `v_interp_mov_f32` is translated to) – drawn indexed through
/// `emit::rect_list_gs`, the corner at v0, v1 or v2: the whole rect is drawn, the smooth output
/// continues into the implied corner's half, and both flat ones are the rect's first vertex's
/// in both halves. On the console that is the provoking vertex under D3D's setting
/// (PA_SU_SC_MODE_CNTL.PROVOKING_VTX_LAST = 0, see `emit::rect_list_gs`). Synthetic: not a
/// measurement of the hardware. Without the GS part of every rect stays clear.
#[test]
fn gpu_rect_list_flat_outputs() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_UINT};
    let outputs = [el("SV_Position", 0, 1, 3, 0, 0xF), el("TEXCOORD", 0, 0, 3, 1, 0xF), el("TEXCOORD", 1, 0, 1, 2, 0xF), el("TEXCOORD", 2, 0, 3, 3, 0xF)];
    let vs_hlsl = "struct V { float4 pos : SV_Position; float4 uv : TEXCOORD0; uint4 id : TEXCOORD1; float4 tint : TEXCOORD2; };\nV main(float4 pos : POSITION, uint id : ID) { V o; o.pos = pos; o.uv = float4(pos.xy, 0.0, 1.0); o.id = uint4(id, 0u, 0u, 0u); o.tint = float4(id * 0.125, 0.0, 0.0, 0.0); return o; }\n";
    let ps_hlsl = "float4 main(float4 pos : SV_Position, float4 uv : TEXCOORD0, nointerpolation uint4 id : TEXCOORD1, nointerpolation float4 tint : TEXCOORD2) : SV_Target { return float4(uv.xy, id.x, tint.x); }\n";
    let vs = super::dxc::compile(vs_hlsl, "main", "vs_6_6", &[]).unwrap();
    let ps = super::dxc::compile(ps_hlsl, "main", "ps_6_6", &[]).unwrap();
    let gs_hlsl = super::emit::rect_list_gs(&outputs);
    let gs = super::dxc::compile(&gs_hlsl, "main", "gs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{gs_hlsl}"));
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "ID", index: 0, format: DXGI_FORMAT_R32_UINT, offset: 16 },
    ];
    // a 3 × 2 grid: columns x = -0.75, 0, 0.75, rows y = -0.5, 0.5; vertex 2·column + row has id = its number + 1
    let grid: Vec<f32> = (0..6u32).flat_map(|v| [[-0.75f32, 0.0, 0.75][v as usize / 2], [-0.5f32, 0.5][v as usize % 2], 0.0, 1.0, f32::from_bits(v + 1)]).collect();
    let (left, right) = ([-0.75f32, 0.0, -0.5, 0.5], [0.0f32, 0.75, -0.5, 0.5]);
    let (n, clear) = (16usize, [0.0f32, 0.0, 1.0, 0.5]);
    // (indices, the rects (x1, x2, y1, y2) and their first vertex's id)
    for (name, indices, rects) in [
        ("corner v0", vec![0u32, 1, 2], vec![(left, 1.0f32)]),
        ("corner v1", vec![0, 1, 3], vec![(left, 1.0)]),
        ("corner v1, the other way round", vec![1, 0, 2], vec![(left, 2.0)]),
        ("corner v2", vec![1, 2, 0], vec![(left, 2.0)]),
        ("two rects sharing vertices", vec![0, 1, 2, 5, 3, 2], vec![(left, 1.0), (right, 6.0)]),
    ] {
        let run = |with_gs: bool| runner::draw_gs_indexed(&vs, with_gs.then_some(gs.as_slice()), &ps, &layout, &grid, 20, &indices, n as u32, clear).unwrap();
        let wrong = |px: &[[f32; 4]], report: bool| {
            let mut wrong = 0;
            for y in 0..n {
                for x in 0..n {
                    let (fx, fy) = ((x as f32 + 0.5) / 8.0 - 1.0, 1.0 - (y as f32 + 0.5) / 8.0);
                    let rect = rects.iter().find(|&&([x1, x2, y1, y2], _)| fx > x1 && fx < x2 && fy > y1 && fy < y2);
                    let want = rect.map_or(clear, |&(_, id)| [fx, fy, id, id * 0.125]);
                    let got = px[y * n + x];
                    if !got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5) {
                        assert!(!report, "{name}: pixel ({x},{y}) = {got:?}, expected {want:?}");
                        wrong += 1;
                    }
                }
            }
            wrong
        };
        wrong(&run(true), true);
        // the negative control: as a triangle list part of every rect is not drawn
        let missing = wrong(&run(false), false);
        assert!(missing >= 12 * rects.len(), "{name}: without the GS {missing} pixels differ");
    }
}

/// Console rect lists (DI_PT_RECTLIST) drawn through `emit::rect_list_gs`: three corners of a
/// screen-aligned rectangle, the corner between the other two at v0 (Mesa radeonsi's blits),
/// v1 (xf86-video-ati's r600 EXA) or v2. Each covers the whole rectangle and its attribute – the
/// vertex's own position – is the pixel's position everywhere, the implied corner's quarter
/// included (both drivers' texture coordinates depend on it). Without the GS (as before) part
/// of the rectangle stays clear; three vertices that are not such corners, or differ in w, draw
/// the same with and without it. Synthetic: the expected images are the rectangle and the
/// triangle, not a measurement of the hardware.
#[test]
fn gpu_rect_list_corners() {
    if !gpu_ready() {
        return;
    }
    use super::{PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let vs_code = prog(&[
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        attr_base: 4,
        params: None,
    };
    let mut ps_ops: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = Vec::new();
    for c in 0..2u64 {
        ps_ops.push(("V_INTERP_P1_F32", "ENC_VINTRP", vec![("VDST", 4 + c), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", c)], None));
        ps_ops.push(("V_INTERP_P2_F32", "ENC_VINTRP", vec![("VDST", 4 + c), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", c)], None));
    }
    ps_ops.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 6), ("SRC0", 128)], None));
    ps_ops.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 242)], None));
    ps_ops.push(("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None));
    let ps_list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = ps_ops.iter().map(|(n, e, f, l)| (*n, *e, f.as_slice(), *l)).collect();
    let ps_code = prog(&ps_list);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let outputs = vsi.outputs.clone();
    let vt = translate(&vs_code, &Options { mode: Mode::Vertex(vsi), wave64: true, ..Default::default() }).unwrap();
    let pt = translate(&ps_code, &Options { mode: Mode::Pixel(psi), wave64: true, ..Default::default() }).unwrap();
    assert!(vt.complete() && pt.complete(), "{:?} {:?}", vt.unsupported, pt.unsupported);
    let vs = super::dxc::compile(&vt.hlsl, "main", "vs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", vt.hlsl));
    let ps = super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl));
    let gs_hlsl = super::emit::rect_list_gs(&outputs);
    let gs = super::dxc::compile(&gs_hlsl, "main", "gs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{gs_hlsl}"));
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 16 },
    ];
    // a vertex at NDC (x, y) with clip w: position (x·w, y·w, 0, w), COLOR (x, y, 0, 1)
    let verts = |ps: &[[f32; 3]]| -> Vec<f32> { ps.iter().flat_map(|&[x, y, w]| [x * w, y * w, 0.0, w, x, y, 0.0, 1.0]).collect() };
    let (n, clear) = (16usize, [0.0f32, 0.0, 1.0, 0.5]);
    let ndc = |x: usize, y: usize| ((x as f32 + 0.5) / 8.0 - 1.0, 1.0 - (y as f32 + 0.5) / 8.0);
    let run = |v: &[f32], with_gs: bool| runner::draw_gs(&vs, with_gs.then_some(gs.as_slice()), &ps, &layout, v, 32, &[], &[], n as u32, clear).unwrap();
    let (x1, x2, y1, y2) = (-0.5f32, 0.5, -0.5, 0.5);
    for (name, corners) in [
        ("corner v0 (radeonsi)", [[x1, y1], [x1, y2], [x2, y1]]),
        ("corner v1 (r600 EXA)", [[x1, y1], [x1, y2], [x2, y2]]),
        ("corner v2", [[x1, y2], [x2, y1], [x1, y1]]),
    ] {
        let v = verts(&corners.map(|[x, y]| [x, y, 1.0]));
        let px = run(&v, true);
        for y in 0..n {
            for x in 0..n {
                let (fx, fy) = ndc(x, y);
                let want = if fx > x1 && fx < x2 && fy > y1 && fy < y2 { [fx, fy, 0.0, 1.0] } else { clear };
                let got = px[y * n + x];
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "{name}: pixel ({x},{y}) = {got:?}, expected {want:?}");
            }
        }
        // the negative control: as a triangle list only part of the rectangle is drawn
        let tri = run(&v, false);
        let missing = (0..n * n).filter(|&i| px[i] != clear && tri[i] == clear).count();
        assert!(missing >= 16, "{name}: without the GS {missing} of the rectangle's pixels stay clear");
    }
    // left as the triangle they make: not three rectangle corners, and corners with unequal w
    for (name, v) in [
        ("not a rectangle", verts(&[[-0.5, -0.5, 1.0], [0.25, 0.5, 1.0], [0.5, -0.25, 1.0]])),
        ("unequal w", verts(&[[x1, y1, 1.0], [x1, y2, 2.0], [x2, y1, 1.0]])),
    ] {
        let (with, without) = (run(&v, true), run(&v, false));
        assert!(without.iter().any(|p| *p != clear), "{name}: the triangle draws");
        assert!(with.iter().zip(&without).all(|(a, b)| a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-5)), "{name}: the GS changed the image");
    }
}

/// One instruction (programs built in loops).
fn ins(name: &str, enc: &str, fields: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    encode(name, enc, fields, lit).unwrap_or_else(|x| panic!("{name}: {x}"))
}

/// A program from instructions, ending in `s_endpgm`.
fn program_of(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut v: Vec<u8> = parts.concat();
    v.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}

/// The pixel interface of `gpu_graphics_vs_ps_draw`: v0 / v1 = i / j, v2 / v3 = pixel centre
/// x / y, attribute 0 = COLOR.
fn draw_psi() -> super::PsInterface {
    super::PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    }
}

/// The vertex interface of `gpu_graphics_vs_ps_draw`: POSITION in v4 / v5, COLOR in v8..v11;
/// exports pos0 and param0 (COLOR).
fn draw_vsi() -> super::VsInterface {
    super::VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        attr_base: 4,
        params: None,
    }
}

fn draw_passthrough_vs() -> Vec<u8> {
    prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ])
}

#[test]
fn gpu_ps_p0_reads_flat_integer_selectors() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R32G32B32A32_SINT, DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32_FLOAT,
    };

    let vs_code = draw_passthrough_vs();
    // Match the parameter word against 1, 2, and all ones, one output channel per match.
    let ps_code = prog(&[
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 2), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 242)], None), // alpha and matching channel: 1.0
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 129), ("VSRC1", 4)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 128), ("VSRC1", 8)], None),
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 130), ("VSRC1", 4)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 8)], None),
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 193), ("VSRC1", 4)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 128), ("VSRC1", 8)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 5), ("VSRC1", 6), ("VSRC2", 7), ("VSRC3", 8), ("DONE", 1), ("VM", 1)], None),
    ]);
    let compile = |code: &[u8], mode: Mode, profile: &str| {
        let t = translate(code, &runner::options(mode)).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        super::dxc::compile(&t.hlsl, "main", profile, &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl))
    };
    let cases = [
        (0, [0.0, 0.0, 0.0, 1.0]),
        (1, [1.0, 0.0, 0.0, 1.0]),
        (2, [0.0, 1.0, 0.0, 1.0]),
        (u32::MAX, [0.0, 0.0, 1.0, 1.0]),
    ];
    for (component_type, format) in [(1, DXGI_FORMAT_R32G32B32A32_UINT), (2, DXGI_FORMAT_R32G32B32A32_SINT)] {
        let mut vsi = draw_vsi();
        vsi.inputs[1].component_type = component_type;
        vsi.outputs[1].component_type = component_type;
        let mut psi = draw_psi();
        psi.inputs[1].component_type = component_type;
        let vs = compile(&vs_code, Mode::Vertex(vsi), "vs_6_6");
        let ps = compile(&ps_code, Mode::Pixel(psi), "ps_6_6");
        let layout = [
            runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
            runner::VertexElement { semantic: "COLOR", index: 0, format, offset: 8 },
        ];
        for (selector, expected) in cases {
            // The runner uploads these vertex bytes directly. The input format reads the word as an integer.
            let vertices: Vec<f32> = [[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]]
                .into_iter()
                .flat_map(|[x, y]| [x, y, f32::from_bits(selector), 0.0, 0.0, 0.0])
                .collect();
            let pixels = runner::draw(&vs, &ps, &layout, &vertices, 24, &[], &[], 16, [-1.0; 4]).unwrap();
            for (i, pixel) in pixels.iter().enumerate() {
                assert_eq!(*pixel, expected, "{format:?} selector {selector:#010x}, pixel {i}");
            }
        }
    }
}

/// Draw console wave64 VS + PS (`vs` None: position and COLOR passed through; `ps` None:
/// COLOR interpolated) with triangles given in pixels of a 16 × 16 target, cleared to -1.
/// Translated for this adapter (`runner::options`: on a GPU with 32-lane waves only, the
/// form of `upper_lanes`).
fn draw_console(vs: Option<&[u8]>, ps: Option<&[u8]>, tris_px: &[[f32; 2]]) -> Vec<[f32; 4]> {
    draw_console_views(vs, ps, tris_px, &[], &[], &[])
}

/// `draw_console` with initial SGPRs and textures / samplers (see `runner::ViewTexture`).
fn draw_console_views(
    vs: Option<&[u8]>,
    ps: Option<&[u8]>,
    tris_px: &[[f32; 2]],
    sgprs: &[u32],
    textures: &[runner::ViewTexture],
    samplers: &[windows::Win32::Graphics::Direct3D12::D3D12_SAMPLER_DESC],
) -> Vec<[f32; 4]> {
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    let pass_vs = draw_passthrough_vs();
    let pass_ps = prog(&[
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 2)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 2)], None),
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 7), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 3)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 7), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 3)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let compile = |code: &[u8], mode: Mode, profile: &str| {
        let t = translate(code, &runner::options(mode)).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        super::dxc::compile(&t.hlsl, "main", profile, &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl))
    };
    let vsb = compile(vs.unwrap_or(&pass_vs), Mode::Vertex(draw_vsi()), "vs_6_6");
    let psb = compile(ps.unwrap_or(&pass_ps), Mode::Pixel(draw_psi()), "ps_6_6");
    // pixels → NDC (y down), COLOR 0
    let verts: Vec<f32> = tris_px.iter().flat_map(|p| [p[0] / 8.0 - 1.0, 1.0 - p[1] / 8.0, 0.0, 0.0, 0.0, 0.0]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    runner::draw_views(&vsb, &psb, &layout, &verts, 24, sgprs, &[], textures, samplers, 16, [-1.0; 4]).unwrap()
}

/// `image_sample_o` in a pixel shader, implicit LOD between two mips (trilinear): the console
/// moves each mip's texel coordinate by the offset in that mip's texels. A 4-level texture
/// (8², 4², 2², 1²; level k texel x holds 100·k + x), point min / mag, linear mip, clamp;
/// u = v = pixel centre · 2^0.25 / 8 gives LOD 0.25; offset (+1, 0).
#[test]
fn gpu_ps_image_sample_offset_implicit_lod_between_mips() {
    // S#: clamp u / v / w, point min / mag, MIP_FILTER linear (2), LOD 0..15.99, sampler 0
    implicit_lod_between_mips([2 | 2 << 3 | 2 << 6, 0xFFF << 12, 2 << 26, 0]);
}

/// The same with an aniso point (2) min / mag filter and a 4× anisotropy ratio: a nearest filter
/// with anisotropy on (what a driver writes for one), so the texels are the point filter's.
#[test]
fn gpu_ps_image_sample_with_an_aniso_point_filter_samples_points() {
    implicit_lod_between_mips([2 | 2 << 3 | 2 << 6 | 2 << 9, 0xFFF << 12, 2 << 20 | 2 << 22 | 2 << 26, 0]);
}

fn implicit_lod_between_mips(s_sharp: [u32; 4]) {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_SAMPLE_DESC};
    let scale = 2f32.powf(0.25) / 8.0;
    let ps = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 129)], None), // offset x +1, y 0
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 255), ("VSRC1", 2)], Some(scale.to_bits())),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 3)], Some(scale.to_bits())),
        ("IMAGE_SAMPLE_O", "ENC_MIMG", &[("VDATA", 8), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 8), ("VSRC2", 8), ("VSRC3", 8), ("DONE", 1), ("VM", 1)], None),
    ]);
    let mut data = Vec::new();
    for k in 0..4u32 {
        let n = 8 >> k;
        for _y in 0..n {
            for x in 0..n {
                data.extend_from_slice(&[(100 * k + x) as f32; 4].map(f32::to_le_bytes).concat());
            }
        }
    }
    let res = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 8,
        Height: 8,
        DepthOrArraySize: 1,
        MipLevels: 4,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    srv.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 4, ..Default::default() };
    let tex = runner::ViewTexture { res, srv, data, row_bytes: 8 * 16 };
    let mut sgprs = vec![0u32; 32];
    sgprs[16..24].copy_from_slice(&tsharp(0, 77, 9));
    sgprs[24..28].copy_from_slice(&s_sharp);
    let px = draw_console_views(None, Some(&ps), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]], &sgprs, &[tex], &[crate::gpu::desc::sampler_d3d12(&s_sharp, false)]);
    for y in 0..16 {
        for x in 0..16 {
            let u = (x as f32 + 0.5) * scale;
            let texel = |size: f32| ((u * size).floor() + 1.0).clamp(0.0, size - 1.0);
            let want = 0.75 * texel(8.0) + 0.25 * (100.0 + texel(4.0));
            let got = px[y * 16 + x][0];
            assert!((got - want).abs() < 0.02, "pixel ({x},{y}) = {got}, expected {want}");
        }
    }
}

/// 48 SGPR spill slots in lanes 0..47 of v30 (`v_writelane` / `v_readlane`, as 287 retail
/// pixel shaders do – 11 up to lane 47): slot k = `slot(k)`, then s44 = the sum of slots
/// 0..31, s45 = the sum of slots 32..47.
fn spill_slots(slot: fn(u32) -> u32) -> Vec<u8> {
    let mut code = Vec::new();
    for k in 0..48u32 {
        code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 255)], Some(slot(k))));
        code.extend(ins("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 30), ("SRC0", 20), ("SRC1", 128 + k as u64)], None));
    }
    code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 44), ("SSRC0", 128)], None));
    code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 45), ("SSRC0", 128)], None));
    for k in 0..48u64 {
        code.extend(ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 30), ("SRC1", 128 + k)], None));
        let sum = if k < 32 { 44 } else { 45 };
        code.extend(ins("S_ADD_U32", "ENC_SOP2", &[("SDST", sum), ("SSRC0", sum), ("SSRC1", 20)], None));
    }
    code
}

/// The whole-wave reduction of the retail pixel shaders (`upper_lanes`), run as a wave64 PS:
/// identity into the lanes outside the mask, DPP8 xor 1 / 2 / 4, row_mirror, the other row
/// through `v_permlanex16`, `v_readlane` 32 for the upper half's result, `v_readlane` 0 for
/// the total – as min (identity -1, as in the pack) and as a sum (identity 0: wrong if lane
/// 32 is read as anything but the identity, e.g. as lane 0, where a 32-lane GPU's wave read
/// wraps). Over the odd pixels of each wave; also the waterfall read of the pack
/// (`v_readlane` of the lane `s_ff1` picks from mask & EXEC). The waves are found from the
/// pixels: each exports its wave's `v_readfirstlane` of the pixel number and its lane
/// (`v_mbcnt`), and the CPU computes what a console wave of those lanes gives.
#[test]
fn gpu_wave64_ps_whole_wave_reductions() {
    use std::collections::{BTreeMap, BTreeSet};
    if !gpu_ready() {
        return;
    }
    let reduce = |op: &str, identity: u64, out: u64| -> Vec<u8> {
        let mut c = vec![
            ins("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)], None),
            ins("S_AND_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 18), ("SSRC1", 30)], None),
            ins("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", identity), ("SRC1", V + 13), ("SRC2", 10)], None),
        ];
        let names = ["LANE_SEL_0", "LANE_SEL_1", "LANE_SEL_2", "LANE_SEL_3", "LANE_SEL_4", "LANE_SEL_5", "LANE_SEL_6", "LANE_SEL_7"];
        for sel in [[1u64, 0, 3, 2, 5, 4, 7, 6], [2, 3, 0, 1, 6, 7, 4, 5], [4, 5, 6, 7, 0, 1, 2, 3]] {
            let mut f = vec![("VDST", 6u64), ("SRC0", 0xE9), ("VSRC0", 6), ("VSRC1", 6)];
            f.extend(names.iter().copied().zip(sel));
            c.push(ins(op, "VOP2_VOP_DPP8", &f, None));
        }
        c.extend([
            ins(op, "VOP2_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 6), ("VSRC1", 6), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
            ins("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 6), ("SRC1", 128), ("SRC2", 128)], None),
            ins(op, "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 7), ("VSRC1", 6)], None),
            ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
            ins(op, "ENC_VOP2", &[("VDST", 6), ("SRC0", 10), ("VSRC1", 6)], None),
            ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", out), ("SRC0", V + 6), ("SRC1", 128)], None),
            ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None),
        ]);
        c.concat()
    };
    let code = program_of(&[
        // v12 = pixel number y·16 + x, v13 = v12 + 1, s[18:19] = the odd pixels
        ins("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 2)], None),
        ins("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 3)], None),
        ins("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 11), ("SRC1", 128 + 4), ("SRC2", V + 10)], None),
        ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", 129), ("VSRC1", 12)], None),
        ins("V_AND_B32", "ENC_VOP2", &[("VDST", 14), ("SRC0", 129), ("VSRC1", 12)], None),
        ins("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 14)], None),
        ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 18), ("SSRC0", 106)], None),
        // the wave: its first lane's pixel number; the lane
        ins("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 62), ("SRC0", V + 12)], None),
        ins("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", 193), ("SRC1", 128)], None),
        ins("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", 193), ("SRC1", V + 15)], None),
        reduce("V_MIN_U32", 193, 40),
        reduce("V_ADD_NC_U32", 128, 41),
        ins("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 18), ("SSRC1", 126)], None),
        ins("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 0)], None),
        ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 43), ("SRC0", V + 13), ("SRC1", 0)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 40)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 41)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 22), ("SRC0", 43)], None),
        ins("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 62), ("SRC1", 128 + 6), ("SRC2", V + 15)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 23), ("SRC0", V + 23)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 21), ("VSRC2", 22), ("VSRC3", 23), ("DONE", 1), ("VM", 1)], None),
    ]);
    let px = draw_console(None, Some(&code), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]]);
    // by wave (its first lane's pixel number): (pixel number, lane, results)
    let mut waves: BTreeMap<u32, Vec<(u32, u32, [f32; 4])>> = BTreeMap::new();
    for (id, p) in px.iter().enumerate() {
        let kl = p[3] as u32;
        waves.entry(kl >> 6).or_default().push((id as u32, kl & 63, *p));
    }
    println!("waves of {:?} pixels", waves.values().map(|w| w.len()).collect::<Vec<_>>());
    for (first, w) in &waves {
        let lanes: BTreeSet<u32> = w.iter().map(|x| x.1).collect();
        assert_eq!(lanes.len(), w.len(), "wave of pixel {first}: a lane twice");
        let lowest = w.iter().min_by_key(|x| x.1).unwrap();
        assert_eq!(lowest.0, *first, "v_readfirstlane: the wave's lowest lane");
        let odd: Vec<&(u32, u32, [f32; 4])> = w.iter().filter(|x| x.0 % 2 == 1).collect();
        let Some(first_odd) = odd.iter().min_by_key(|x| x.1) else { continue };
        let want = [odd.iter().map(|x| x.0 + 1).min().unwrap(), odd.iter().map(|x| x.0 + 1).sum(), first_odd.0 + 1];
        for (id, lane, p) in w {
            let got = [p[0] as u32, p[1] as u32, p[2] as u32];
            assert_eq!(got, want, "pixel {id} (lane {lane}, wave of pixel {first}): min / sum over the odd pixels, waterfall read");
        }
    }
}

/// SGPR spill slots in lanes 0..47 of a VGPR (`spill_slots`) in a wave64 PS: slots ≥ 32 do
/// not exist in a 32-lane wave, and a slot in a lane that was not launched is lost in a
/// partial wave (a primitive's edge). Drawn over the whole target and as a 6-pixel triangle.
#[test]
fn gpu_wave64_ps_spill_slots() {
    if !gpu_ready() {
        return;
    }
    let slot: fn(u32) -> u32 = |k| k * k + 7 * k + 1;
    let code = program_of(&[
        spill_slots(slot),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 44)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 45)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 22), ("SRC0", 242)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 23), ("SRC0", 242)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 20), ("VSRC1", 21), ("VSRC2", 22), ("VSRC3", 23), ("DONE", 1), ("VM", 1)], None),
    ]);
    let want = [(0..32).map(slot).sum::<u32>() as f32, (32..48).map(slot).sum::<u32>() as f32];
    let full = draw_console(None, Some(&code), &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]]);
    for (i, p) in full.iter().enumerate() {
        assert_eq!([p[0], p[1]], want, "pixel {i}, full waves: slots 0..31, 32..47");
    }
    // pixels (0..2, 0), (0..1, 1), (0, 2): three quads
    let small = draw_console(None, Some(&code), &[[0.0, 0.0], [3.2, 0.0], [0.0, 3.2]]);
    let covered: Vec<(usize, &[f32; 4])> = small.iter().enumerate().filter(|(_, p)| p[3] == 1.0).collect();
    assert_eq!(covered.len(), 6);
    // the 32-lane form keeps slots < 32 whatever the wave size, slots ≥ 32 in 32-lane waves;
    // the native form (GPUs with 64-lane waves) still loses the slots of lanes not launched
    let (lo, hi) = (runner::graphics_wave32(), runner::wave64_supported() == Ok(false));
    for (i, p) in covered {
        println!("pixel {i}, partial wave: {:?} (want {want:?})", &p[..2]);
        if lo {
            assert_eq!(p[0], want[0], "pixel {i}, partial wave: slots 0..31");
        }
        if hi {
            assert_eq!(p[1], want[1], "pixel {i}, partial wave: slots 32..47");
        }
    }
}

/// A wave64 VS with SGPR spill slots in lanes 0..47 (`spill_slots`) and a whole-wave identity
/// read from lane 32 (`v_cndmask` with the saved EXEC, `v_readlane` 32): three vertices, so a
/// wave of three lanes. The VS exports the slots' sums and whether lane 32 held the identity
/// as param0, which the PS interpolates (the same at every vertex: exact).
#[test]
fn gpu_wave64_vs_spill_slots_and_identity() {
    if !gpu_ready() {
        return;
    }
    let slot: fn(u32) -> u32 = |k| 3 * k * k + k + 5;
    let code = program_of(&[
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        spill_slots(slot),
        ins("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)], None),
        ins("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", 193), ("SRC1", V), ("SRC2", 30)], None),
        ins("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
        ins("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None),
        ins("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 193)], None),
        ins("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 242), ("SSRC1", 128)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 44)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 45)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 11)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 242)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ]);
    let px = draw_console(Some(&code), None, &[[0.0, 0.0], [0.0, 32.0], [32.0, 0.0]]);
    let want = [(0..32).map(slot).sum::<u32>() as f32, (32..48).map(slot).sum::<u32>() as f32, 1.0, 1.0];
    // as in `gpu_wave64_ps_spill_slots`: slots < 32 in the 32-lane form; slots ≥ 32 and the
    // identity in lane 32 in 32-lane waves (in a 64-lane wave of three vertices, lane 32 was
    // not launched either)
    let (lo, hi) = (runner::graphics_wave32(), runner::wave64_supported() == Ok(false));
    println!("{:?} (want {want:?})", px[0]);
    for (i, p) in px.iter().enumerate() {
        if lo {
            assert_eq!(p[0], want[0], "pixel {i}: slots 0..31");
        }
        if hi {
            assert_eq!(&p[1..], &want[1..], "pixel {i}: slots 32..47, lane 32");
        }
    }
}

/// Masks in whole-quad mode hold the helper lanes (#105): after `s_wqm` the helper pixels of a
/// quad a primitive's edge splits are in EXEC, and the console's `v_cmp` sets their bits like
/// any other lane's. Wave ops leave helper lanes out before SM 6.7, so a mask taken with
/// `WaveActiveBallot` alone had them 0: `v_cndmask` gave them the false value, `s_and_saveexec`
/// dropped them, and a derivative taken across the quad by hand (DPP) was wrong along the
/// diagonal of two triangles. A waterfall loop over `v_readfirstlane` never removed a helper
/// lane from EXEC and ran into the loop budget. The target is two triangles sharing the
/// diagonal; per pixel: |ddx| of a value picked by `v_cndmask`, |ddy| of a value written
/// under `s_and_saveexec` (both 1 everywhere), and the waterfall loop's iteration count.
#[test]
fn gpu_ps_masks_hold_helper_lanes_on_triangle_edges() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
    let vs = super::dxc::compile("float4 main(float2 p : POSITION) : SV_Position { return float4(p, 0.5, 1.0); }\n", "main", "vs_6_6", &[]).unwrap();
    for wave64 in [true, false] {
        let pt = helper_lane_masks_ps(wave64);
        assert!(pt.complete(), "{:?}\n{}", pt.unsupported_why, pt.hlsl);
        let ps = super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl));
        // the whole target as two triangles sharing the diagonal x = y
        let quad = [-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        let layout = [runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 }];
        let size = 32usize;
        let px = runner::draw(&vs, &ps, &layout, &quad, 8, &[], &[], size as u32, [9.0; 4]).unwrap();
        let lanes = if wave64 { 64.0 } else { 32.0 };
        let bad: Vec<(usize, usize, [f32; 4])> = (0..size * size)
            .filter(|&i| {
                let p = px[i];
                p[0].abs() != 1.0 || p[1].abs() != 1.0 || !(1.0..=lanes).contains(&p[2]) || p[3] != 1.0
            })
            .map(|i| (i % size, i / size, px[i]))
            .collect();
        let count = |k: usize, ok: &dyn Fn(f32) -> bool| px.iter().filter(|p| !ok(p[k])).count();
        let (dx, dy, it) = (count(0, &|v| v.abs() == 1.0), count(1, &|v| v.abs() == 1.0), count(2, &|v| (1.0..=lanes).contains(&v)));
        println!("wave64 {wave64}: {} of {} pixels wrong (ddx {dx}, ddy {dy}, loop {it}), first: {:?}", bad.len(), size * size, bad.iter().take(6).collect::<Vec<_>>());
        assert!(bad.is_empty(), "wave64 {wave64}: {} pixels wrong (ddx {dx}, ddy {dy}, loop {it}; |ddx|, |ddy|, loop iterations, 1), first {:?}", bad.len(), &bad[..bad.len().min(6)]);
    }
}

/// The masks of a waterfall loop in a pixel shader take the helper lanes' bits from quad reads
/// of the compare's vector operand, not of the compare (#134): AMD's RDNA2 compiler crashed in
/// `CreateGraphicsPipelineState` on quad reads of a value computed from a `WaveReadLaneAt` in a
/// loop (WarmMilkCodes, RX 6600 XT), which `ballot_px` gave the loop's `v_cmp`. Outside loops too:
/// AMD's RDNA3 compiler crashed on a `WaveReadLaneAt` whose lane came from such a mask, with no
/// loop (`s_ff1` of it → `v_readlane`, RX 7800 XT), so every mask `quad_bits` can express takes
/// this form. CPU only: what the GPU sees is `gpu_ps_masks_hold_helper_lanes_on_triangle_edges`.
#[test]
fn ps_masks_in_loops_quad_read_the_operands_not_the_compare() {
    for wave64 in [true, false] {
        let pt = helper_lane_masks_ps(wave64);
        assert!(pt.complete(), "{:?}\n{}", pt.unsupported_why, pt.hlsl);
        let (before, rest) = pt.hlsl.split_once("[loop] for (;;) {").unwrap_or_else(|| panic!("no loop\n{}", pt.hlsl));
        let body = &rest[..rest.find("\n    }\n").unwrap_or_else(|| panic!("no loop end\n{}", pt.hlsl))];
        assert!(!body.contains("ballot_px(") && !body.contains("QuadReadAcross"), "wave64 {wave64}: a quad read of a compare in the loop\n{body}");
        assert!(body.contains("ballot_px_quad(") && body.contains("QuadReadLaneAt(v2, 3u)"), "wave64 {wave64}: the loop's v_cmp without its operand's quad reads\n{body}");
        // one read per register and member, though the float compare names v2 twice (NaN test, ==)
        assert_eq!(body.matches("QuadReadLaneAt(v2, ").count(), 4, "wave64 {wave64}\n{body}");
        assert!(!before.contains("= ballot_px(and(") && before.contains("ballot_px_quad("), "wave64 {wave64}: a mask before the loop quad-reads its compare\n{before}");
    }
}

/// The pixel shader of the two tests above, translated: after `s_wqm`, a mask read by
/// `v_cndmask` (A), one for `s_and_saveexec` (B) and a waterfall loop over `v_readfirstlane` of
/// pos.x (C); exports |ddx| of A, |ddy| of B, C's iteration count and 1.
fn helper_lane_masks_ps(wave64: bool) -> super::Translation {
    use super::PsInterface;
    {
        // the wave-size forms of the scalar mask instructions; EXEC / VCC = their low words in wave32
        let (mov, wqm, and_save, xor) = if wave64 {
            ("S_MOV_B64", "S_WQM_B64", "S_AND_SAVEEXEC_B64", "S_XOR_B64")
        } else {
            ("S_MOV_B32", "S_WQM_B32", "S_AND_SAVEEXEC_B32", "S_XOR_B32")
        };
        let sop1 = |n: &'static str, d: u64, s: u64| (n, "ENC_SOP1", vec![("SDST", d), ("SSRC0", s)], None);
        type I = (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>);
        let head: Vec<I> = vec![
            sop1(mov, 20, 126), // s[20:21] = the live lanes
            sop1(wqm, 126, 126),
            // A: v4 = v_cndmask(0, pos.x, 0 < pos.x) – pos.x in every lane; v6 = v4 of the quad's
            // other column (quad_perm [1,0,3,2]) - v4
            ("V_CMP_LT_F32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 2)], None),
            ("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", 128), ("VSRC1", 2)], None),
            ("V_MOV_B32", "VOP1_VOP_DPP16", vec![("VDST", 5), ("SRC0", 0xFA), ("VSRC0", 4), ("DPP_CTRL", 0xB1), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
            ("V_SUB_F32", "ENC_VOP2", vec![("VDST", 6), ("SRC0", V + 5), ("VSRC1", 4)], None),
            // B: v7 = pos.y under s_and_saveexec(0 < pos.y); v9 = v7 of the quad's other row
            // (quad_perm [2,3,0,1]) - v7
            ("V_CMP_LT_F32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 3)], None),
            sop1(and_save, 22, 106),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", V + 3)], None),
            sop1(mov, 126, 22),
            ("V_MOV_B32", "VOP1_VOP_DPP16", vec![("VDST", 8), ("SRC0", 0xFA), ("VSRC0", 7), ("DPP_CTRL", 0x4E), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
            ("V_SUB_F32", "ENC_VOP2", vec![("VDST", 9), ("SRC0", V + 8), ("VSRC1", 7)], None),
            // C: a waterfall loop over pos.x, s8 = its iterations
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 8), ("SSRC0", 128)], None),
            sop1(mov, 24, 126),
        ];
        let body: Vec<I> = vec![
            ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 4), ("SRC0", V + 2)], None),
            ("V_CMP_EQ_F32", "ENC_VOPC", vec![("SRC0", 4), ("VSRC1", 2)], None),
            sop1(and_save, 26, 106),
            ("S_ADD_U32", "ENC_SOP2", vec![("SDST", 8), ("SSRC0", 8), ("SSRC1", 129)], None),
            (xor, "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 26), ("SSRC1", 126)], None),
        ];
        let tail: Vec<I> = vec![
            sop1(mov, 126, 24),
            ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 10), ("SRC0", 8)], None),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 11), ("SRC0", 242)], None),
            sop1(mov, 126, 20), // back to the live lanes for the export
            ("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 6), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11), ("DONE", 1), ("VM", 1)], None),
        ];
        let enc = |l: &[I]| -> Vec<u8> { l.iter().flat_map(|(n, e, f, x)| encode(n, e, f, *x).unwrap_or_else(|err| panic!("{n}: {err}"))).collect() };
        let body_code = enc(&body);
        let back = (-((body_code.len() as i64 + 4) / 4)) as u16 as u64;
        let mut ps_code = enc(&head);
        ps_code.extend(&body_code);
        ps_code.extend(encode("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", back)], None).unwrap());
        ps_code.extend(enc(&tail));
        ps_code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let psi = PsInterface { inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x302, attrs: None };
        translate(&ps_code, &Options { mode: Mode::Pixel(psi), wave64, ..Default::default() }).unwrap()
    }
}

/// What the pixel shaders' masks (`ballot_px`) and lane reads (`rla`) rely on, checked on the
/// adapter with native HLSL on the same two triangles: a quad is hardware lanes 4k..4k+3 (X = ^1,
/// Y = ^2, diagonal = ^3), and a helper lane gets the wave's result of a `WaveActiveBallot` –
/// each lane holds its own copy of EXEC, a helper lane's included. Not every wave op does that:
/// `WaveActiveBitOr` gave helper lanes another value on an RTX 5070 Ti (printed, not checked).
#[test]
fn gpu_ps_helper_lanes_see_ballots_and_quads_are_lanes_4k() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
    let vs = super::dxc::compile("float4 main(float2 p : POSITION) : SV_Position { return float4(p, 0.5, 1.0); }\n", "main", "vs_6_6", &[]).unwrap();
    // per pixel: is the quad's other column a helper lane; does it see this lane's ballot and
    // bit-or; are this lane and that one at lanes 4k..4k+3
    let src = "float4 main(float4 p : SV_Position) : SV_Target {\n\
        uint l = WaveGetLaneIndex();\n\
        uint hl = IsHelperLane() ? 1u : 0u;\n\
        uint b = WaveActiveBallot(((uint)p.x & 3u) == 1u).x;\n\
        uint o = WaveActiveBitOr(1u << (l & 31u));\n\
        bool lay = QuadReadAcrossX(l) == (l ^ 1u) && QuadReadAcrossY(l) == (l ^ 2u) && QuadReadAcrossDiagonal(l) == (l ^ 3u) && QuadReadLaneAt(l, 0u) == (l & ~3u);\n\
        bool both = lay && QuadReadAcrossX((uint)lay) != 0u;\n\
        return float4(QuadReadAcrossX(hl), QuadReadAcrossX(b) == b ? 1.0 : 0.0, QuadReadAcrossX(o) == o ? 1.0 : 0.0, both ? 1.0 : 0.0);\n}\n";
    let ps = super::dxc::compile(src, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let quad = [-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
    let layout = [runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 }];
    let px = runner::draw(&vs, &ps, &layout, &quad, 8, &[], &[], 32, [9.0; 4]).unwrap();
    let helpers: Vec<[f32; 4]> = px.iter().copied().filter(|p| p[0] == 1.0).collect();
    let ballot = helpers.iter().filter(|p| p[1] == 1.0).count();
    let bit_or = helpers.iter().filter(|p| p[2] == 1.0).count();
    let quads = px.iter().filter(|p| p[3] == 1.0).count();
    println!("{:?}: {} pixels next to a helper lane; its ballot right in {ballot}, its bit-or in {bit_or}; quads at lanes 4k in {quads} of {}", runner::adapter(), helpers.len(), px.len());
    assert!(!helpers.is_empty(), "no helper lanes along the diagonal");
    assert_eq!(ballot, helpers.len(), "helper lanes without the wave's ballot");
    assert_eq!(quads, px.len(), "quads not at hardware lanes 4k..4k+3");
}

/// XCL image descriptor (T#, GFX10 layout): BASE_ADDRESS = SRV heap index, FORMAT, TYPE.
fn tsharp(index: u32, format: u32, ty: u32) -> [u32; 8] {
    [index, format << 20, 0, ty << 28, 0, 0, 0, 0]
}

/// A cube T# of two cubes: the face coordinate (face + 8 · cube, as v_cubeid and the console's
/// compiler give it) picks the cube, and image_get_resinfo gives the layers (6 per cube).
#[test]
fn gpu_cube_array_slices() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // v[0:2] = (s, t, face + 8 · cube), T# in s[16:23], S# point in s[24:27]
    code.extend(mimg("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 3)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 20), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 3)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 11));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (face, cube) = (l % 6, l / 6 % 2);
        r[0] = f(1.5); // the middle of the face
        r[1] = f(1.5);
        r[2] = f((face + 8 * cube) as f32);
        r[3] = 0; // mip
    }
    // 2 x 2 faces, every texel of layer (cube, face) = (face, cube, 6 · cube + face, 1)
    let mut data = Vec::new();
    for cube in 0..2 {
        for face in 0..6 {
            for _ in 0..4 {
                for v in [face as f32, cube as f32, (6 * cube + face) as f32, 1.0] {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    let tex = [TestTexture { dim: TexDim::Cube, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 2, height: 2, depth: 12, texel_bytes: 16, data, uav: false }];
    let samp = [TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (face, cube) = ((l % 6) as f32, (l / 6 % 2) as f32);
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [face, cube, 6.0 * cube + face, 1.0], "lane {l}: cube {cube} face {face}");
        assert_eq!(&v[20..24], &[2, 2, 12, 1], "lane {l} resinfo (w, h, layers, mips)");
    }
}

/// image_get_resinfo on a 1D array (DIM 4) gives the layers in y, as RDNA does (GFX9, which
/// binds 1D arrays as 2D arrays, gives them in z and Mesa moves them to y there only).
#[test]
fn image_get_resinfo_1d_array_layers_in_y() {
    let code = prog(&[("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 4)], None)]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.hlsl.contains("Texture1DArray<float4>"), "{}", t.hlsl);
    let call = t.hlsl.split("_T.GetDimensions(").nth(1).and_then(|s| s.split(')').next()).unwrap_or_else(|| panic!("{}", t.hlsl));
    let args: Vec<&str> = call.split(", ").collect();
    assert!(args.len() == 4 && args[2].ends_with("_h"), "GetDimensions({call}) must write the elements into _h\n{}", t.hlsl);
}

/// A 4-wide 1D array of 3 layers: image_get_resinfo gives (width, layers, _, mips).
#[test]
fn gpu_image_get_resinfo_1d_array() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mut code = encode("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 4)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 12));
    for r in st.vgpr.iter_mut() {
        r[3] = 0; // mip
    }
    let tex = [TestTexture { dim: TexDim::D1Array, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 1, depth: 3, texel_bytes: 16, data: vec![0; 4 * 3 * 16], uav: false }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert_eq!([v[20], v[21], v[23]], [4, 3, 1], "lane {l} resinfo (w, layers, mips)");
    }
}

#[test]
fn gpu_images_sample_load_store_atomic() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_UINT};
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // T# tex0 in s[16:23] (SRSRC = 4), S# point in s[24:27] (SSAMP = 6), S# linear in s[28:31] (7)
    code.extend(mimg("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)]));
    code.extend(mimg("IMAGE_SAMPLE_L", &[("VDATA", 20), ("VADDR", 30), ("SRSRC", 4), ("SSAMP", 7), ("DMASK", 0xF), ("DIM", 1)]));
    code.extend(mimg("IMAGE_GATHER4_LZ", &[("VDATA", 40), ("VADDR", 33), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)]));
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 14), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0x5), ("DIM", 1), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 50), ("VADDR", 5), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1)]));
    // tex1: R32_UINT UAV, T# in s[32:39] (SRSRC = 8); tex2: 1x1 R32_UINT UAV in s[40:47] (10)
    code.extend(mimg("IMAGE_STORE", &[("VDATA", 6), ("VADDR", 37), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_ATOMIC_UMAX", &[("VDATA", 7), ("VADDR", 35), ("SRSRC", 10), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    // tex3: 1x1 R32_UINT UAV in s[48:55] (12): image_atomic_inc (no HLSL intrinsic: CAS loop)
    code.extend(mimg("IMAGE_ATOMIC_INC", &[("VDATA", 39), ("VADDR", 35), ("SRSRC", 12), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    st.sgpr[24..28].copy_from_slice(&[0, 0, 0, 0]);
    st.sgpr[28..32].copy_from_slice(&[0, 0, 0, 1]);
    st.sgpr[32..40].copy_from_slice(&tsharp(2, 20, 9));
    st.sgpr[40..48].copy_from_slice(&tsharp(4, 20, 9));
    st.sgpr[48..56].copy_from_slice(&tsharp(6, 20, 9));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as u32, (l / 4 % 4) as u32);
        r[0] = f((x as f32 + 0.5) / 4.0);
        r[1] = f((y as f32 + 0.5) / 4.0);
        r[3] = x;
        r[4] = y;
        r[5] = 0;
        r[6] = if l < 16 { 10 * l as u32 } else { 0 };
        r[7] = l as u32;
        r[30] = f(0.5); // between texel columns 1 and 2
        r[31] = f(0.125); // row 0
        r[32] = f(0.0); // lod
        r[33] = f(0.5);
        r[34] = f(0.5);
        r[35] = 0;
        r[36] = 0;
        r[37] = x; // lanes 16.. land outside the 4x4 texture: their stores are dropped
        r[38] = l as u32 / 4;
        r[39] = 100;
    }
    // tex0: 4x4 RGBA32F, texel (x, y) = (x, y, x + 4y, 1)
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 4, height: 4, depth: 1, texel_bytes: 4, data: vec![0xEE; 64], uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 1, height: 1, depth: 1, texel_bytes: 4, data: vec![0; 4], uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 1, height: 1, depth: 1, texel_bytes: 4, data: vec![0; 4], uav: true },
    ];
    let samp = vec![
        TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None },
        TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None },
    ];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [x, y, x + 4.0 * y, 1.0], "lane {l} sample_lz point");
        assert_eq!([fl(20), fl(21), fl(22), fl(23)], [1.5, 0.0, 1.5, 1.0], "lane {l} sample_l linear");
        assert_eq!([fl(40), fl(41), fl(42), fl(43)], [1.0, 2.0, 2.0, 1.0], "lane {l} gather4 red");
        assert_eq!([fl(14), fl(15)], [x, x + 4.0 * y], "lane {l} image_load dmask x,z");
        assert_eq!(&v[50..54], &[4, 4, 0, 1], "lane {l} get_resinfo (depth 0: a DIM without layers, as RDNA2)");
    }
    let t1: Vec<u32> = r.textures[0].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for i in 0..16 {
        assert_eq!(t1[i], 10 * i as u32, "image_store texel {i}");
    }
    assert_eq!(u32::from_le_bytes(r.textures[1][..4].try_into().unwrap()), 63, "image_atomic_umax");
    assert_eq!(u32::from_le_bytes(r.textures[2][..4].try_into().unwrap()), 64, "image_atomic_inc (64 lanes, limit 100)");
}

/// An image_load of a T# the program also stores through reads through the UAV when the
/// invocation stored through that T# before (`xcl_iw16`); a program without image stores
/// keeps the SRV-only load.
#[test]
fn image_load_of_a_stored_image_reads_through_the_uav() {
    let store = ("IMAGE_STORE", "ENC_MIMG", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)][..], None);
    let load = ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)][..], None);
    let t = translate(&prog(&[store, load]), &Options::default()).unwrap();
    assert!(t.hlsl.contains("static uint xcl_iw16 = 0u;") && t.hlsl.contains("xcl_iw16 = s16 + 1u;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("_rw = xcl_iw16 == s16 + 1u;") && t.hlsl.matches("RWTexture2D<uint4>").count() == 2, "{}", t.hlsl);
    let t = translate(&prog(&[load]), &Options::default()).unwrap();
    assert!(!t.hlsl.contains("xcl_iw") && !t.hlsl.contains("RWTexture2D"), "{}", t.hlsl);
}

/// An image the program stores to and then loads in the same dispatch (in-place accumulation,
/// in-place blur, TAA history): RDNA2 reads its own write – the same lane right after the
/// store, a neighbour after the barrier. Each lane stores 1000 + lane to its texel of an 8x8
/// R32_UINT image, loads that texel back, and after `s_barrier` loads the texel of lane ^ 1.
#[test]
fn gpu_image_load_reads_a_store_of_the_same_dispatch() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT;
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // T# in s[16:23] (SRSRC = 4): own texel (v0, v1), neighbour's texel (v3, v4)
    code.extend(mimg("IMAGE_STORE", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 11), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 20, 9));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let n = l ^ 1;
        r[0] = (l % 8) as u32;
        r[1] = (l / 8) as u32;
        r[2] = 1000 + l as u32;
        r[3] = (n % 8) as u32;
        r[4] = (n / 8) as u32;
    }
    let tex = [TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 8, height: 8, depth: 1, texel_bytes: 4, data: vec![0xEE; 8 * 8 * 4], uav: true }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &[]).unwrap();
    let t: Vec<u32> = r.textures[0].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for l in 0..64 {
        assert_eq!(t[l], 1000 + l as u32, "image_store texel {l}");
    }
    let bad: Vec<String> = (0..64)
        .filter(|&l| r.vgpr[l][10] != 1000 + l as u32 || r.vgpr[l][11] != 1000 + (l ^ 1) as u32)
        .map(|l| format!("lane {l}: own {:#x}, neighbour {:#x}", r.vgpr[l][10], r.vgpr[l][11]))
        .collect();
    assert!(bad.is_empty(), "image_load after image_store of the same dispatch (want 1000 + lane, 1000 + (lane ^ 1)):\n{}", bad.join("\n"));
}

/// Static samplers: the program builds two S# from constants (point / linear, clamp) – as
/// the console compiler places a root signature's static samplers – and samples the same
/// texel boundary with each; they become static samplers of the root signature.
#[test]
fn gpu_static_samplers_from_constants_in_code() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let clamp = 2 | 2 << 3 | 2 << 6;
    let point = [clamp, 0x00FF_F000u32, 0, 0];
    let linear = [clamp, 0x00FF_F000, 1 << 20 | 1 << 22, 0];
    let mov = |r: u64, v: u32| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        if v == 0 {
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", r), ("SSRC0", 128)], None)
        } else {
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", r), ("SSRC0", 255)], Some(v))
        }
    };
    let mut list = Vec::new();
    for k in 0..4 {
        list.push(mov(24 + k as u64, point[k]));
        list.push(mov(28 + k as u64, linear[k]));
    }
    // T# tex0 in s[16:23] (SRSRC = 4); S# point s[24:27] (SSAMP = 6), linear s[28:31] (7)
    list.push(("IMAGE_SAMPLE_LZ", "ENC_MIMG", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)], None));
    list.push(("IMAGE_SAMPLE_LZ", "ENC_MIMG", vec![("VDATA", 20), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 7), ("DMASK", 0xF), ("DIM", 1)], None));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let t = translate(&code, &Options::default()).unwrap();
    assert_eq!(t.static_samplers, vec![(point, false), (linear, false)], "two static samplers from the code");
    assert!(!t.hlsl.contains("SamplerDescriptorHeap"), "no sampler heap lookups left");

    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    for r in st.vgpr.iter_mut() {
        r[0] = f(0.5); // the boundary between texel columns 1 and 2
        r[1] = f(0.125); // row 0
    }
    // 4x4 RGBA32F, texel (x, y) = (x, y, x + 4y, 1)
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [2.0, 0.0, 2.0, 1.0], "lane {l}: static point sampler");
        assert_eq!([fl(20), fl(21), fl(22), fl(23)], [1.5, 0.0, 1.5, 1.0], "lane {l}: static linear sampler");
    }
}

/// A static sampler's S# word that the compiler writes bit-reversed (`s_brev_b32 s26, 40` is
/// 0x14000000: a constant with only high bits set) is a constant like one from s_mov: the
/// S# is a static sampler, not the sampler-heap entry its word 3 (0 here) would name.
fn brev_sampler_program() -> (Vec<u8>, [u32; 4]) {
    let clamp = 2 | 2 << 3 | 2 << 6;
    let point = [clamp, 0x00FF_F000u32, 40u32.reverse_bits(), 0];
    // T# tex0 in s[16:23] (SRSRC = 4); S# s[24:27] (SSAMP = 6)
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 255)], Some(point[0])),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 25), ("SSRC0", 255)], Some(point[1])),
        ("S_BREV_B32", "ENC_SOP1", &[("SDST", 26), ("SSRC0", 128 + 40)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 27), ("SSRC0", 128)], None),
        ("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)], None),
    ]);
    (code, point)
}

#[test]
fn static_sampler_word_from_s_brev_is_a_constant() {
    let (code, point) = brev_sampler_program();
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert_eq!(t.static_samplers, vec![(point, false)], "the S# with a bit-reversed word is a static sampler");
    assert!(!t.hlsl.contains("SamplerDescriptorHeap"), "no sampler heap lookup left");
}

/// The same program on the GPU: the texel a point sampler picks at a texel boundary.
#[test]
fn gpu_static_sampler_word_from_s_brev() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let (code, _) = brev_sampler_program();
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    for r in st.vgpr.iter_mut() {
        r[0] = f(0.5); // the boundary between texel columns 1 and 2
        r[1] = f(0.125); // row 0
    }
    // 4x4 RGBA32F, texel (x, y) = (x, y, x + 4y, 1)
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [2.0, 0.0, 2.0, 1.0], "lane {l}: static point sampler");
    }
}

#[test]
fn gpu_images_a16_d16() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // coordinates (u, v) as f16 in v0; result as 4 halves in v[10:11]
    code.extend(mimg("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1), ("A16", 1), ("D16", 1)]));
    // integer coordinates (x, y) as 16-bit in v1; x and z in v[12:13] as 32-bit
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 12), ("VADDR", 1), ("SRSRC", 4), ("DMASK", 0x5), ("DIM", 1), ("UNORM", 1), ("A16", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    let h = |x: f32| half_bits(x);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as u32, (l / 4 % 4) as u32);
        r[0] = h((x as f32 + 0.5) / 4.0) | h((y as f32 + 0.5) / 4.0) << 16;
        r[1] = x | y << 16;
    }
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let samp = vec![TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        assert_eq!((v[10], v[11]), (h(x) | h(y) << 16, h(x + 4.0 * y) | h(1.0) << 16), "lane {l} a16+d16 sample");
        assert_eq!([f32::from_bits(v[12]), f32::from_bits(v[13])], [x, x + 4.0 * y], "lane {l} a16 load");
    }
}

/// f32 → IEEE half bits (exact for the small values used in tests).
fn half_bits(x: f32) -> u32 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    (b >> 16 & 0x8000) | ((e as u32) << 10) | ((b >> 13) & 0x3FF)
}

#[test]
fn gpu_format_stores_atomics_global_lds() {
    if !gpu_ready() {
        return;
    }
    let mubuf = |n: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(n, "ENC_MUBUF", f, None).unwrap() };
    let mut code = Vec::new();
    // V# A (s[12:15]): 8_8_8_8_UNORM at 0x100, stride 4 (idxen with v0 = lane)
    code.extend(mubuf("BUFFER_STORE_FORMAT_XYZW", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)]));
    // V# B (s[16:19]): 16_FLOAT at 0x300, stride 2 → neighbouring lanes share dwords
    code.extend(mubuf("BUFFER_STORE_FORMAT_X", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)]));
    // V# C (s[20:23]): raw at 0x400: every lane adds 1, umax returns the old value
    code.extend(mubuf("BUFFER_ATOMIC_ADD", &[("VDATA", 15), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128)]));
    code.extend(mubuf("BUFFER_ATOMIC_UMAX", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128), ("OFFSET", 4), ("GLC", 1)]));
    // global: store lane*3 to s[2:3] + v1 (lane*4), load it back through the 64-bit form
    code.extend(encode("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 17), ("SADDR", 2), ("OFFSET", 0)], None).unwrap());
    code.extend(encode("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 18), ("ADDR", 20), ("SADDR", 0x7D), ("OFFSET", 0xFFC)], None).unwrap()); // offset -4
    // LDS: write lane at lane*4, barrier, read reversed, read2 of two neighbours, counter, bpermute
    code.extend(encode("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None).unwrap());
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(encode("DS_READ_B32", "ENC_DS", &[("VDST", 22), ("ADDR", 23)], None).unwrap());
    code.extend(encode("DS_READ2_B32", "ENC_DS", &[("VDST", 24), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 26), ("ADDR", 27), ("DATA0", 28), ("OFFSET0", 0), ("OFFSET1", 2)], None).unwrap()); // at 512
    code.extend(encode("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 29), ("ADDR", 23), ("DATA0", 0)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x500; // global base
    st.sgpr[12..16].copy_from_slice(&[0x100, 4 << 16, 64, (4 | 5 << 3 | 6 << 6 | 7 << 9) | 56 << 12]);
    st.sgpr[16..20].copy_from_slice(&[0x300, 2 << 16, 64, (4 | 5 << 3 | 6 << 6 | 7 << 9) | 13 << 12]);
    st.sgpr[20..24].copy_from_slice(&[0x400, 0, 16, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = l32;
        r[1] = 4 * l32;
        r[10] = f(1.0);
        r[11] = f(0.0);
        r[12] = f(l as f32 / 63.0);
        r[13] = f(0.5);
        r[14] = f(l as f32);
        r[15] = 1;
        r[16] = 100 + l32;
        r[17] = 3 * l32;
        r[20] = 0x500 + 4 * l32 + 4; // 64-bit address (hi in v21 = 0), offset -4
        r[23] = 4 * (63 - l32);
        r[27] = 0;
        r[28] = 1;
    }
    let r = runner::run_code(&code, &st, &vec![0u32; 1024]).unwrap();
    let byte = |a: usize| (r.mem[a / 4] >> (8 * (a % 4))) & 0xFF;
    for l in 0..64usize {
        let px = [byte(0x100 + 4 * l), byte(0x100 + 4 * l + 1), byte(0x100 + 4 * l + 2), byte(0x100 + 4 * l + 3)];
        assert_eq!(px, [255, 0, ((l as f32 / 63.0) * 255.0).round() as u32, 128], "lane {l} store 8_8_8_8_UNORM");
        let h = (r.mem[(0x300 + 2 * l) / 4] >> (16 * (l % 2))) & 0xFFFF;
        assert_eq!(h, half_bits(l as f32), "lane {l} store 16_FLOAT (shared dwords)");
        let v = &r.vgpr[l];
        assert_eq!(r.mem[0x500 / 4 + l], 3 * l as u32, "lane {l} global_store saddr");
        assert_eq!(v[18], 3 * l as u32, "lane {l} global_load 64-bit address, offset -4");
        assert_eq!(v[22], 63 - l as u32, "lane {l} LDS reversed");
        assert_eq!(v[24], l as u32, "lane {l} ds_read2 (first)");
        if l < 63 {
            assert_eq!(v[25], l as u32 + 1, "lane {l} ds_read2 (second; LDS is not zeroed past the written area)");
        }
        assert_eq!(v[29], 63 - l as u32, "lane {l} ds_bpermute");
        assert!(v[16] < 164, "lane {l} umax old value");
    }
    assert_eq!(r.mem[0x400 / 4], 64, "buffer_atomic_add");
    assert_eq!(r.mem[0x404 / 4], 163, "buffer_atomic_umax");
    let olds: Vec<u32> = r.vgpr.iter().map(|v| v[26]).collect();
    // LDS is not zeroed between dispatches (nor on the console): consecutive values from any
    // start, which may wrap (0xFFFFFFFF, 0, 1, …)
    let base = *olds.iter().find(|&&v| !olds.contains(&v.wrapping_sub(1))).unwrap_or(&olds[0]);
    let mut got: Vec<u32> = olds.iter().map(|v| v.wrapping_sub(base)).collect();
    got.sort();
    assert_eq!(got, (0..64).collect::<Vec<u32>>(), "ds_add_rtn_u32 returns every old value once (from {base:#x}): {olds:?}");
}

/// global_*_dword_addtid (ENC_FLAT_GLBL): the spec gives them no ADDR operand – the
/// address is the scalar SADDR pair ("off"/null encodes a zero base) plus 4 times the
/// lane index plus the signed 12-bit OFFSET. Emitted HLSL is deterministic.
#[test]
fn global_dword_addtid_address_is_saddr_plus_4_per_lane() {
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mut code = Vec::new();
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 5), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 7), ("SADDR", 0x7D), ("OFFSET", 0xFFC)])); // null base, offset -4
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // s[2:3] base, signed +8 offset; the ADDR field bits are never read as a VGPR
    assert!(t.hlsl.contains("uint t1_a = s2 + 4u * lane_id + 0x00000008u;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("mem.Store((t1_a + 0u) & ~3u, v5);"), "{}", t.hlsl);
    // "off"/null SADDR → base 0; the 12-bit offset sign-extends (0xFFC → -4)
    assert!(t.hlsl.contains("uint t2_a = 4u * lane_id + 0xFFFFFFFCu;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v7 = mem.Load((t2_a + 0u) & ~3u);"), "{}", t.hlsl);
    // exactly the two address expressions above – no ADDR VGPR read anywhere
    assert_eq!(t.hlsl.matches("_a = ").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        direct_check(&code, &Options::default());
    }
}

/// On hardware: every lane touches its own dword (EA = SADDR + 4·lane + OFFSET), the
/// "off" SADDR sentinel gives a zero base and the signed offset can point below the base.
#[test]
fn gpu_dword_addtid_per_lane_addresses() {
    if !gpu_ready() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mut code = Vec::new();
    // each lane writes its signature at s[2:3] + 4·lane + 8, then reads it back (+8) and
    // the previous lane's dword (+4)
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 5), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 7), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 8), ("SADDR", 2), ("OFFSET", 4)]));
    // s[4:5] + 4·lane − 4 → preset values just below 0x200; offset −4
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 10), ("SADDR", 4), ("OFFSET", 0xFFC)]));
    // null ("off") SADDR: base 0 → per-lane dwords at 0x40 + 4·lane
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 9), ("SADDR", 0x7D), ("OFFSET", 0x40)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 11), ("SADDR", 0x7D), ("OFFSET", 0x40)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[4] = 0x200;
    let sig = |l: usize| 0xA500 + l as u32;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[5] = sig(l);
        r[9] = 0xBEEF + l as u32;
    }
    let mut mem = vec![0u32; 512];
    mem[0x604 / 4] = 0xDEAD_CAFE; // in the gap below the stores: lane 0's +4 load lands here
    for l in 0..64usize {
        mem[0x1FC / 4 + l] = 0xF00 + l as u32; // under s[4:5] for the offset −4 loads
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for l in 0..64usize {
        assert_eq!(r.mem[0x608 / 4 + l], sig(l), "lane {l}: store at saddr + 4·lane + 8");
        assert_eq!(r.vgpr[l][7], sig(l), "lane {l}: load its own dword");
        let prev = if l == 0 { 0xDEAD_CAFE } else { sig(l - 1) };
        assert_eq!(r.vgpr[l][8], prev, "lane {l}: offset +4 reads the previous lane's dword");
        assert_eq!(r.vgpr[l][10], 0xF00 + l as u32, "lane {l}: signed offset −4 below the base");
        assert_eq!(r.mem[0x40 / 4 + l], 0xBEEF + l as u32, "lane {l}: store with a null SADDR base");
        assert_eq!(r.vgpr[l][11], 0xBEEF + l as u32, "lane {l}: load with a null SADDR base");
    }
}

#[test]
fn gpu_valu_16_and_64_bit() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_ADD_F16", "ENC_VOP2", &[("VDST", 2), ("SRC0", V), ("VSRC1", 1)], None), // low halves, high kept
        ("V_MUL_F16", "ENC_VOP3", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL", 0b1001)], None), // src0.hi * src1.lo → dst.hi
        ("V_LSHRREV_B16", "ENC_VOP3", &[("VDST", 4), ("SRC0", 132), ("SRC1", V + 5)], None),
        ("V_CMP_LT_F16", "ENC_VOPC", &[("SRC0", V + 6), ("VSRC1", 7)], None),
        ("V_LSHLREV_B64", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 10), ("SRC1", V + 12)], None),
        ("V_CMP_EQ_U64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 8), ("SRC1", V + 14)], None),
        ("V_MAD_U64_U32", "VOP3_SDST_ENC", &[("VDST", 16), ("SDST", 22), ("SRC0", V + 18), ("SRC1", V + 19), ("SRC2", V + 24)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 26), ("SRC1", 242)], None), // + 1.0 (double inline constant)
        ("V_CVT_F32_F64", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 26)], None),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 40)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 32), ("SRC0", V + 40)], None),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 34), ("SRC0", V + 40)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 36), ("SRC0", V + 40)], None),
    ]);
    let h = half_bits;
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = h(1.5) | h(-2.0) << 16;
        r[1] = h(2.25) | h(0.5) << 16;
        r[2] = 0xBEEF_1234;
        r[3] = 0x0000_1111;
        r[4] = 0xCAFE_0000;
        r[5] = 0xFFFF_8421;
        r[6] = h(l as f32);
        r[7] = h(31.5);
        r[10] = l32;
        r[12] = 1;
        let e = if l % 2 == 0 { 1u64 << l } else { 0 };
        r[14] = e as u32;
        r[15] = (e >> 32) as u32;
        r[18] = 0xFFFF_FFFF;
        r[19] = l32 + 1;
        let c = if l % 2 == 1 { u64::MAX } else { l as u64 };
        r[24] = c as u32;
        r[25] = (c >> 32) as u32;
        let dv = (l as f64 + 0.25).to_bits();
        r[26] = dv as u32;
        r[27] = (dv >> 32) as u32;
        let x = (l as f64 + 1.25).powi(2).to_bits();
        r[40] = x as u32;
        r[41] = (x >> 32) as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!(v[2], 0xBEEF_0000 | h(3.75), "lane {l} v_add_f16 keeps the high half");
        assert_eq!(v[3], 0x1111 | h(-4.5) << 16, "lane {l} v_mul_f16 op_sel");
        assert_eq!(v[4], 0xCAFE_0842, "lane {l} v_lshrrev_b16");
        assert_eq!(v[8] as u64 | (v[9] as u64) << 32, 1u64 << l, "lane {l} v_lshlrev_b64");
        let c = if l % 2 == 1 { u64::MAX } else { l as u64 };
        let (w, carry) = (0xFFFF_FFFFu64 * (l as u64 + 1)).overflowing_add(c);
        assert_eq!(v[16] as u64 | (v[17] as u64) << 32, w, "lane {l} v_mad_u64_u32");
        let _ = carry;
        assert_eq!(f64::from_bits(v[26] as u64 | (v[27] as u64) << 32), l as f64 + 1.25, "lane {l} v_add_f64");
        assert_eq!(f32::from_bits(v[28]), l as f32 + 1.25, "lane {l} v_cvt_f32_f64");
        let d = |n: usize| f64::from_bits(v[n] as u64 | (v[n + 1] as u64) << 32);
        let root = l as f64 + 1.25;
        let input = root * root;
        assert_eq!((d(30), d(32), d(34)), (input.ceil(), input.fract(), root), "lane {l} unary f64");
        assert!((d(36) - root.recip()).abs() < 1e-15, "lane {l} v_rsq_f64");
    }
    assert_eq!((r.sgpr[106], r.sgpr[107]), (0xFFFF_FFFF, 0), "v_cmp_lt_f16 → vcc");
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0x5555_5555, 0x5555_5555), "v_cmp_eq_u64 → sdst");
    let carries: u64 = (0..64).filter(|&l| (0xFFFF_FFFFu64 * (l as u64 + 1)).overflowing_add(if l % 2 == 1 { u64::MAX } else { l as u64 }).1).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[22] as u64 | (r.sgpr[23] as u64) << 32, carries, "v_mad_u64_u32 carry mask");
}

/// NEG_HI[2] of v_dot2_f32_f16 is the ABS of its f32 accumulator (SRC2), applied before NEG[2]
/// (LLVM `SIDefines.h` `NEG_HI = ABS`; `llvm.amdgcn.fdot2.ll` `v_fdot2_abs_c`: `neg_hi:[0,0,1]`).
#[test]
fn dot2_f32_f16_neg_hi_on_src2_is_abs() {
    let hlsl = |neg: u64, neg_hi: u64| {
        let code = prog(&[(
            "V_DOT2_F32_F16",
            "ENC_VOP3P",
            &[("VDST", 4), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("NEG", neg), ("NEG_HI", neg_hi)],
            None,
        )]);
        let t = translate(&code, &Options::default()).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        t.hlsl
    };
    let plain = hlsl(0, 0);
    assert!(plain.contains("asfloat(v3)") && !plain.contains("abs(asfloat(v3))"), "{plain}");
    let abs = hlsl(0, 0b100);
    assert!(abs.contains("abs(asfloat(v3))") && !abs.contains("(-abs(asfloat(v3)))"), "{abs}");
    let neg_abs = hlsl(0b100, 0b100);
    assert!(neg_abs.contains("(-abs(asfloat(v3)))"), "{neg_abs}");
}

/// v_dot2_f32_f16 with ABS (NEG_HI[2]) and NEG (NEG[2]) on the f32 accumulator: 1·1 + 1·1 + c.
#[test]
fn gpu_dot2_f32_f16_abs_neg_on_src2() {
    for wave64 in gpu_waves() {
        let p = prog(&[
            // neg_hi:[0,0,1]: 2 + |c|
            ("V_DOT2_F32_F16", "ENC_VOP3P", &[("VDST", 4), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("NEG_HI", 0b100)], None),
            // neg_lo:[0,0,1] neg_hi:[0,0,1]: 2 - |c|
            ("V_DOT2_F32_F16", "ENC_VOP3P", &[("VDST", 5), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("NEG", 0b100), ("NEG_HI", 0b100)], None),
            // neg_lo:[0,0,1]: 2 - c
            ("V_DOT2_F32_F16", "ENC_VOP3P", &[("VDST", 6), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("NEG", 0b100)], None),
        ]);
        let mut st = KernelState::default();
        for r in st.vgpr.iter_mut() {
            r[1] = 0x3C00_3C00; // 1.0h, 1.0h
            r[2] = 0x3C00_3C00;
            r[3] = f(-5.0);
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(v[4], f(7.0), "wave{lanes} lane {l} v_dot2_f32_f16 neg_hi:[0,0,1]");
            assert_eq!(v[5], f(-3.0), "wave{lanes} lane {l} v_dot2_f32_f16 neg_lo:[0,0,1] neg_hi:[0,0,1]");
            assert_eq!(v[6], f(7.0), "wave{lanes} lane {l} v_dot2_f32_f16 neg_lo:[0,0,1]");
        }
    }
}

#[test]
fn gpu_packed_math_dots_movrel() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_PK_ADD_F16", "ENC_VOP3P", &[("VDST", 2), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        // lo = src0.hi * src1.lo, hi = -(src0.lo * src1.hi)
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL", 1), ("OP_SEL_HI", 2), ("NEG_HI", 1)], None),
        ("V_DOT2_F32_F16", "ENC_VOP3P", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 5), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_DOT4_U32_U8", "ENC_VOP3P", &[("VDST", 6), ("SRC0", V + 7), ("SRC1", V + 8), ("SRC2", V + 9), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_DOT4_I32_I8", "ENC_VOP3P", &[("VDST", 11), ("SRC0", V + 7), ("SRC1", V + 12), ("SRC2", V + 9), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_PK_FMAC_F16", "ENC_VOP2", &[("VDST", 13), ("SRC0", V), ("VSRC1", 1)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 131)], None), // m0 = 3
        ("V_MOVRELS_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 20)], None), // v10 = v23
        ("V_MOVRELD_B32", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 9)], None), // v33 = v9
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 40), ("SRC0", V + 33)], None),
    ]);
    let h = half_bits;
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = h(1.5) | h(-2.0) << 16;
        r[1] = h(2.25) | h(0.5) << 16;
        r[5] = f(10.0);
        r[7] = 0x0102_0304;
        r[8] = 0x0506_0708;
        r[9] = l as u32;
        r[12] = 0xFF06_0708; // -1, 6, 7, 8
        r[13] = h(1.0) | h(2.0) << 16;
        for k in 0..8 {
            r[20 + k] = 100 + k as u32;
        }
        r[33] = 0xDEAD;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!(v[2], h(3.75) | h(-1.5) << 16, "lane {l} v_pk_add_f16");
        assert_eq!(v[3], h(-4.5) | h(-0.75) << 16, "lane {l} v_pk_mul_f16 op_sel / neg_hi");
        assert_eq!(f32::from_bits(v[4]), 12.375, "lane {l} v_dot2_f32_f16");
        assert_eq!(v[6], 70 + l as u32, "lane {l} v_dot4_u32_u8");
        assert_eq!(v[11] as i32, 4 * 8 + 3 * 7 + 2 * 6 - 1 + l as i32, "lane {l} v_dot4_i32_i8 (bytes from the low end)");
        assert_eq!(v[13], h(1.0 + 1.5 * 2.25) | h(2.0 - 2.0 * 0.5) << 16, "lane {l} v_pk_fmac_f16");
        assert_eq!(v[10], 103, "lane {l} v_movrels_b32");
        assert_eq!(v[40], l as u32, "lane {l} v_movreld_b32");
    }
}

#[test]
fn dxc_mesh_shader_probe() {
    use super::dxc;
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    let src = r#"
struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };
struct P { bool cull : SV_CullPrimitive; };
groupshared uint xcl_nv;
[WaveSize(64)]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(uint gi : SV_GroupIndex, out vertices V verts[128], out indices uint3 tris[128], out primitives P prims[128]) {
    if (gi == 0) xcl_nv = 3;
    GroupMemoryBarrierWithGroupSync();
    uint n = xcl_nv;
    SetMeshOutputCounts(n, 1);
    if (gi < n) { verts[gi].pos = float4(gi, 0, 0, 1); verts[gi].p0 = 0; }
    if (gi < 1) { tris[gi] = uint3(0, 1, 2); prims[gi].cull = false; }
}
"#;
    for prof in ["ms_6_6", "ms_6_8"] {
        let r = dxc::compile(src, "main", prof, &[]);
        println!("{prof}: {:?}", r.as_ref().map(|b| b.len()).map_err(|e| e.lines().take(3).collect::<Vec<_>>().join(" | ")));
    }
}

/// Mesh mode end to end: an NGG program in the console's style (wave index from s3, group
/// from v5 >> 7, GS_ALLOC_REQ, exp pos0/param0/prim, two waves per group) runs as a compute
/// shader, the pass-through mesh shader rasterises its records.
#[test]
fn gpu_mesh_mode_two_passes() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    use super::{dxc, mesh_passthrough, MsInterface, MESH_HEADER};
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let ms = MsInterface { outputs: vec![sig("SV_Position", 1, 0), sig("TEXCOORD", 0, 1)], threads: 128, max_vertices: 128, max_primitives: 128 };
    let exp = |tgt: u64, en: u64, v: [u64; 4], done: u64| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        ("EXP", "ENC_EXP", vec![("TGT", tgt), ("EN", en), ("VSRC0", v[0]), ("VSRC1", v[1]), ("VSRC2", v[2]), ("VSRC3", v[3]), ("DONE", done)], None)
    };
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("S_BFE_U32", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)), // wave index
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None), // thread in group
        ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 1), ("SRC0", V + 5)], None),
        ("S_LSHR_B32", "ENC_SOP2", vec![("SDST", 1), ("SSRC0", 1), ("SSRC1", 135)], None), // group = v5 >> 7
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)), // 1 primitive, 3 vertices
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", vec![], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None), // threads 0..2
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 20), ("SRC0", 1)], None),
        ("V_ADD_F32", "ENC_VOP2", vec![("VDST", 10), ("SRC0", V + 20), ("VSRC1", 10)], None), // x += group
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", V + 20)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 15), ("SRC0", V)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 16), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 17), ("SRC0", 242)], None),
    ];
    list.push(exp(12, 0xF, [10, 11, 12, 13], 1));
    list.push(exp(32, 0xF, [14, 15, 16, 17], 0));
    list.push(("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None));
    list.push(("V_CMPX_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 0)], None)); // thread 0
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)));
    list.push(exp(20, 1, [21, 0, 0, 0], 1));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let opt = on_gpu(Options { mode: Mode::Mesh(ms.clone()), wave64: true, ..Default::default() });
    let t = translate(&code, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let cs = direct_or(&code, &opt, dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap());
    let pass = mesh_passthrough(&ms);
    let msb = dxc::compile(&pass, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass}"));
    let ps_src = "struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return v.p0; }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();
    // vertex table at byte 0x100, V# in s[16:19] (stride 16, 3 records)
    let mut mem = vec![0u32; 256];
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[f(*x), f(*y), f(0.0), f(1.0)]);
    }
    let mut sg = vec![0u32; 128];
    sg[16..20].copy_from_slice(&[0x100, 16 << 16, 3, 0]);
    let (px, rec) = runner::mesh_draw(&cs, &msb, &ps, 2, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
    let (rd, vd) = (ms.record_dwords(), ms.vert_dwords());
    for g in 0..2usize {
        let r = &rec[g * rd..(g + 1) * rd];
        assert_eq!((r[0], r[1]), (3, 1), "group {g}: GS_ALLOC_REQ counts");
        for v in 0..3usize {
            let b = MESH_HEADER + v * vd;
            let x = [-1.0f32, -1.0, 0.0][v] + g as f32;
            assert_eq!(f32::from_bits(r[b]), x, "group {g} vertex {v} pos.x");
            assert_eq!([f32::from_bits(r[b + 4]), f32::from_bits(r[b + 5]), f32::from_bits(r[b + 7])], [g as f32, v as f32, 1.0], "group {g} vertex {v} param0");
        }
        assert_eq!(r[MESH_HEADER + 128 * vd], 1 << 10 | 2 << 20, "group {g} primitive");
    }
    let at = |x: usize, y: usize| px[y * 32 + x];
    assert_eq!((at(3, 29)[0], at(3, 29)[3]), (0.0, 1.0), "group 0 triangle (left)");
    assert_eq!((at(19, 29)[0], at(19, 29)[3]), (1.0, 1.0), "group 1 triangle (right)");
    assert_eq!(at(29, 3), [-1.0; 4], "outside both triangles");

    // the same records through a pass-through for a pixel shader with 16 inputs: over the
    // mesh output limit (17 · 16 B · 128 vertices), so 4 mesh groups per record, each a run
    // of triangles with their own vertex copies; inputs the program does not write read 0
    let ins: Vec<crate::gpu::dxbc::SigElement> = (0..16).map(|k| crate::gpu::dxbc::SigElement { index: k, register: k + 1, ..sig("TEXCOORD", 0, 0) }).collect();
    let split = super::emit::passthrough_split(&ms, super::emit::passthrough_elements(&ms, Some(&ins)));
    assert_eq!(split, 4);
    let pass16 = super::emit::mesh_passthrough_for(&ms, Some(&ins));
    let msb16 = dxc::compile(&pass16, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass16}"));
    let fields: String = (0..16).map(|k| format!(" float4 t{k} : TEXCOORD{k};")).collect();
    let ps16 = dxc::compile(&format!("struct V {{ float4 pos : SV_Position;{fields} }};\nfloat4 main(V v) : SV_Target0 {{ return v.t0 + v.t15; }}\n"), "main", "ps_6_6", &[]).unwrap();
    let (px16, _) = runner::mesh_draw_split(&cs, &msb16, &ps16, 2, split, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
    assert!(px16 == px, "split pass-through draws the same pixels");

    // #304's wide-output boundary, including one compact group: run the same actual NGG
    // compute program, compare every pixel and every record word to the narrow control.
    let wide_inputs: Vec<crate::gpu::dxbc::SigElement> = (0..21).map(|k| crate::gpu::dxbc::SigElement { index: k, register: k + 1, ..sig("TEXCOORD", 0, 0) }).collect();
    let fields: String = (0..21).map(|k| format!(" float4 t{k} : TEXCOORD{k};")).collect();
    let wide_ps = dxc::compile(&format!("struct V {{ float4 pos : SV_Position;{fields} }};\nfloat4 main(V v) : SV_Target0 {{ return v.t0 + v.t20; }}\n"), "main", "ps_6_6", &[]).unwrap();
    for (shape, expected_groups, declaration) in [
        (ms.clone(), 7, "verts[57]"),
        (MsInterface { max_vertices: 80, max_primitives: 1, ..ms.clone() }, 1, "verts[3]"),
    ] {
        let groups = super::emit::passthrough_split(&shape, super::emit::passthrough_elements(&shape, Some(&wide_inputs)));
        assert_eq!(groups, expected_groups);
        let pass = super::emit::mesh_passthrough_for(&shape, Some(&wide_inputs));
        assert!(pass.contains(declaration), "wide compact declaration: {pass}");
        let mesh = dxc::compile(&pass, "main", "ms_6_6", &[]).unwrap();
        let tr = translate(&code, &on_gpu(Options { mode: Mode::Mesh(shape.clone()), wave64: true, ..Default::default() })).unwrap();
        assert!(tr.complete(), "{:?}", tr.unsupported_why);
        let compute = dxc::compile(&tr.hlsl, "main", "cs_6_6", &[]).unwrap();
        let (pixels, records) = runner::mesh_draw_split(&compute, &mesh, &wide_ps, 2, groups, shape.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
        assert_eq!(pixels, px, "all 1024 pixels match, including both triangles and background");
        let mut expected = Vec::new();
        for g in 0..2 {
            let old = &rec[g * rd..(g + 1) * rd];
            expected.extend_from_slice(&old[..MESH_HEADER + shape.max_vertices as usize * vd]);
            let prim = MESH_HEADER + ms.max_vertices as usize * vd;
            expected.extend_from_slice(&old[prim..prim + shape.max_primitives as usize]);
        }
        assert_eq!(records, expected, "all compute record words survive the wide mesh pass");
    }
}

#[test]
fn mesh_passthrough_strips_rdna2_edge_flag_bits() {
    use super::{mesh_passthrough, MsInterface};
    let sig = |name: &str, sv: u32, reg: u32, idx: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: idx,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let ms = MsInterface {
        outputs: vec![sig("SV_Position", 1, 0, 0), sig("TEXCOORD", 0, 1, 0)],
        threads: 128,
        max_vertices: 128,
        max_primitives: 128,
    };
    let unsplit = mesh_passthrough(&ms);
    assert!(unsplit.contains("uint3(p & 0x1FFu, (p >> 10) & 0x1FFu, (p >> 20) & 0x1FFu)"), "{unsplit}");
    assert!(!unsplit.contains("0x3FFu"), "{unsplit}");

    let ins: Vec<_> = (0..16).map(|k| sig("TEXCOORD", 0, k + 1, k)).collect();
    assert!(super::emit::passthrough_split(&ms, super::emit::passthrough_elements(&ms, Some(&ins))) > 1);
    let split = super::emit::mesh_passthrough_for(&ms, Some(&ins));
    assert!(split.contains("uint3(p & 0x1FFu, (p >> 10) & 0x1FFu, (p >> 20) & 0x1FFu)"), "{split}");
    assert!(!split.contains("0x3FFu"), "{split}");
}

/// The mesh output budget with wide vertices: DXC accounts whole chunks of 32 vertices even
/// when fewer are declared, so 78 vertices with 22 float4 rows are accounted as 96 (33792 B >
/// 32768) and the record must go out in split groups. A retail title's NGG transform programs
/// have this shape (128 vertices / 128 primitives, 21 params + SV_ClipDistance0); the previous
/// budget (32768 / stride) emitted verts[78] and DXC refused every one of them.
#[test]
fn mesh_split_accounts_vertex_chunks() {
    use super::emit::{mesh_passthrough, passthrough_elements, passthrough_split};
    use super::MsInterface;
    if super::dxc::location().is_err() {
        println!("no DXC – skipping");
        runner::skip_test("no DXC");
        return;
    }
    let sig = |name: &str, sv: u32, reg: u32, index: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let mut outputs = vec![sig("SV_Position", 1, 0, 0)];
    for k in 0..20u32 {
        outputs.push(sig("TEXCOORD", 0, k + 1, k));
    }
    outputs.push(sig("SV_ClipDistance", 2, 21, 0));
    let ms = MsInterface { outputs, threads: 128, max_vertices: 128, max_primitives: 128 };
    let elems = passthrough_elements(&ms, None);
    assert_eq!(elems, 21);
    // 22 rows * 16 B = 352 B/vertex: 64 accounted vertices fit (96 would be 33792 B), so at
    // most 21 triangles' worth of copies per group, ceil(128/21) = 7 groups of 19
    let split = passthrough_split(&ms, elems);
    assert_eq!(split, 7, "chunks of 32: 78 verts are accounted as 96 and refused");
    let pass = mesh_passthrough(&ms);
    assert!(pass.contains("verts[57]"), "19 triangles per group:\n{pass}");
    super::dxc::compile(&pass, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass}"));
    // 20 rows * 16 B = 320 B/vertex: 96 accounted vertices fit (30720 B), but a whole record
    // is 128 vertices (40960 B) – 4 groups of 32 triangles, 96 vertices each
    let mut ms20 = ms.clone();
    ms20.outputs.pop();
    assert_eq!(passthrough_split(&ms20, passthrough_elements(&ms20, None)), 4);
    let pass20 = mesh_passthrough(&ms20);
    assert!(pass20.contains("verts[96]"), "96 vertices per group:\n{pass20}");
    super::dxc::compile(&pass20, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass20}"));
    // 8 rows * 16 B = 128 B/vertex: 128 vertices are 16384 B – the record stays whole
    let mut ms8 = ms.clone();
    ms8.outputs.truncate(8);
    assert_eq!(passthrough_split(&ms8, passthrough_elements(&ms8, None)), 1);
    let pass8 = mesh_passthrough(&ms8);
    assert!(pass8.contains("verts[128]"), "whole record:\n{pass8}");
    super::dxc::compile(&pass8, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass8}"));
}

/// The layout and the dispatch count are chosen separately: a record of 80 vertices with 22
/// float4 rows does not fit the mesh output budget as a whole (DXC accounts 96 vertices:
/// 33792 B > 32768 B), but its single triangle's copies fit one compact group. The pass-
/// through must then declare the compact layout while `passthrough_split` returns 1, and DXC
/// accepts it (the whole-record declaration is refused).
#[test]
fn mesh_layout_independent_of_group_count() {
    use super::emit::{mesh_passthrough, passthrough_elements, passthrough_split};
    use super::MsInterface;
    if super::dxc::location().is_err() {
        println!("no DXC – skipping");
        runner::skip_test("no DXC");
        return;
    }
    let sig = |name: &str, sv: u32, reg: u32, index: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let mut outputs = vec![sig("SV_Position", 1, 0, 0)];
    for k in 0..20u32 {
        outputs.push(sig("TEXCOORD", 0, k + 1, k));
    }
    outputs.push(sig("SV_ClipDistance", 2, 21, 0));
    // 22 rows * 16 B = 352 B/vertex: 64 accounted vertices fit; 80 declared do not
    let ms = MsInterface { outputs, threads: 128, max_vertices: 80, max_primitives: 1 };
    let elems = passthrough_elements(&ms, None);
    assert_eq!(elems, 21);
    // one triangle: its 3 copies fit one compact group, so the dispatch count is 1
    assert_eq!(passthrough_split(&ms, elems), 1, "one triangle fits one compact group");
    let pass = mesh_passthrough(&ms);
    assert!(pass.contains("verts[3]"), "the compact layout is chosen, not verts[80]:\n{pass}");
    super::dxc::compile(&pass, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass}"));
    // the whole-record form is what DXC refuses at this width: the same record declared whole
    // (verts[80], accounted as 96: 33792 B) does not validate
    let whole = pass.replace("verts[3]", "verts[80]");
    assert!(whole.contains("verts[80]"), "the whole-record declaration for the refusal check:\n{whole}");
    assert!(super::dxc::compile(&whole, "main", "ms_6_6", &[]).is_err(), "DXC refuses 96 accounted vertices of 352 B");
}

/// DXC optimisation levels on a saved HLSL file (`XCL_TIME_HLSL=path`, profile in `XCL_TIME_PROFILE`).
#[test]
#[ignore]
fn dxc_time_levels() {
    use super::dxc;
    let Ok(path) = std::env::var("XCL_TIME_HLSL") else { return };
    let src = std::fs::read_to_string(path).unwrap();
    let prof = std::env::var("XCL_TIME_PROFILE").unwrap_or("cs_6_6".into());
    for args in [vec!["-O0"], vec!["-O1"], vec!["-O2"], vec!["-O3"]] {
        let t0 = std::time::Instant::now();
        let r = dxc::compile(&src, "main", &prof, &args);
        println!("{args:?}: {:.1} s, {:?}", t0.elapsed().as_secs_f64(), r.map(|b| b.len()).map_err(|e| e.chars().take(200).collect::<String>()));
    }
}

/// Minimal RGB8 PNG writer (zlib via miniz_oxide).
fn write_png(path: &std::path::Path, w: u32, h: u32, rgb: &[u8]) {
    fn crc(data: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }
    let chunk = |out: &mut Vec<u8>, ty: &[u8], d: &[u8]| {
        out.extend_from_slice(&(d.len() as u32).to_be_bytes());
        let mut td = ty.to_vec();
        td.extend_from_slice(d);
        out.extend_from_slice(&td);
        out.extend_from_slice(&crc(&td).to_be_bytes());
    };
    let mut raw = Vec::with_capacity((w * 3 + 1) as usize * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * w as usize * 3..(y + 1) * w as usize * 3]);
    }
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = w.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6));
    chunk(&mut out, b"IEND", &[]);
    std::fs::write(path, out).unwrap();
}

/// Retail pixel shaders (no textures) translated and drawn full screen on the GPU:
/// attributes = screen uv, constants from a pattern. `XCL_SAMPLE_RPF`, `XCL_GALLERY_DIR`.
#[test]
#[ignore]
fn retail_ps_gallery() {
    use super::{dxc, ranked};
    let (Ok(path), Ok(dir)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_GALLERY_DIR")) else { return };
    if !gpu_ready() {
        return;
    }
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let data = std::fs::read(&path).unwrap();
    // descriptor table at 0x100: every V# points at the constants at 0x4000
    let mut mem = vec![0u32; 0x3000];
    for i in 0..256 {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[0x4000, 0, 0x8000, 0x27FAC]);
    }
    for i in 0..0x2000 {
        mem[0x1000 + i] = f(0.5 + 0.45 * ((i as f32) * 0.37).sin());
    }
    let mut io = vec![0u32; 128];
    for k in (0..106).step_by(2) {
        io[k] = 0x100;
    }
    let size = 160u32;
    let t0 = std::time::Instant::now();
    let (mut tried, mut kept) = (0, Vec::new());
    for rec in crate::gpu::pack::scan(&data, 0x40) {
        if kept.len() >= 24 || t0.elapsed().as_secs() > 150 {
            break;
        }
        if !rec.name.starts_with("PS_") {
            continue;
        }
        let Some(blob) = crate::gpu::pack::inflate(&data, &rec) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        let Some(fs) = sh.fssm.first() else { continue };
        let Some(p) = fs.programs.first() else { continue };
        let Some(mode) = super::interface(&blob, &sh, fs) else { continue };
        let Mode::Pixel(ps) = mode.clone() else { continue };
        if !ps.outputs.iter().any(|e| e.system_value == 64 && e.index == 0) {
            continue;
        }
        let code = &fs.data[p.offset..p.offset + p.len];
        let Ok(t) = translate(code, &Options { mode, wave64: super::detect_wave64(code), ..Default::default() }) else { continue };
        if !t.complete() || t.hlsl.contains("ResourceDescriptorHeap") || t.hlsl.len() > 150_000 {
            continue;
        }
        tried += 1;
        let mut vs = String::from("struct O {\n    float4 pos : SV_Position;\n");
        let ins = ranked(&ps.inputs, false);
        for (k, e) in ins.iter().enumerate() {
            let ty = match e.component_type {
                1 => "uint",
                2 => "int",
                _ => "float",
            };
            vs.push_str(&format!("    {ty}4 a{k} : {};\n", e.semantic()));
        }
        vs.push_str("};\nO main(uint id : SV_VertexID) {\n    O o;\n    float2 uv = float2((id << 1) & 2, id & 2);\n");
        vs.push_str("    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0.5, 1);\n");
        for (k, e) in ins.iter().enumerate() {
            let v = match e.component_type {
                1 => "uint4(uv * 255.0, 0, 1)",
                2 => "int4(uv * 255.0, 0, 1)",
                _ => "float4(uv, 1.0 - uv.x, 1.0)",
            };
            vs.push_str(&format!("    o.a{k} = {v};\n"));
        }
        vs.push_str("    return o;\n}\n");
        let (Ok(vsb), Ok(psb)) = (dxc::compile(&vs, "main", "vs_6_6", &[]), dxc::compile(&t.hlsl, "main", "ps_6_6", &[])) else { continue };
        let px = match runner::draw(&vsb, &psb, &[], &[0.0f32; 3], 4, &io, &mem, size, [0.0, 0.0, 0.0, 1.0]) {
            Ok(px) => px,
            Err(e) => {
                println!("{}: {e}", rec.name);
                break; // device trouble: stop
            }
        };
        let rgb: Vec<f32> = px.iter().flat_map(|c| [c[0], c[1], c[2]]).map(|x| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }).collect();
        let mean = rgb.iter().sum::<f32>() / rgb.len() as f32;
        let var = rgb.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / rgb.len() as f32;
        if var < 0.002 {
            continue; // flat colour: not worth a picture
        }
        let bytes: Vec<u8> = rgb.iter().map(|x| (x.powf(1.0 / 2.2) * 255.0).round() as u8).collect();
        let file = format!("{:02}_{}.png", kept.len(), rec.name.split('_').take(4).collect::<Vec<_>>().join("_"));
        write_png(&dir.join(&file), size, size, &bytes);
        println!("{file}: {} instructions, variance {var:.3}", t.instructions);
        kept.push(format!("{file}\t{}\t{}", rec.name, t.instructions));
    }
    std::fs::write(dir.join("index.txt"), kept.join("\n")).unwrap();
    println!("tried {tried}, saved {} in {:.0} s", kept.len(), t0.elapsed().as_secs_f64());
}

#[test]
fn gpu_image_sample_offset() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mut code = encode("IMAGE_SAMPLE_LZ_O", "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x3), ("DIM", 1)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        r[0] = 1 | 0x3F << 8; // offset (+1, -1)
        r[1] = f((x + 0.5) / 4.0);
        r[2] = f((y + 0.5) / 4.0);
    }
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, 0.0, 1.0] {
                t0.extend_from_slice(&f32::to_le_bytes(v));
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let samp = vec![TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        let got = [f32::from_bits(r.vgpr[l][10]), f32::from_bits(r.vgpr[l][11])];
        assert_eq!(got, [(x + 1.0).min(3.0), (y - 1.0).max(0.0)], "lane {l} sample_lz_o");
    }
}

/// Console unified memory: the GPU reads constants and buffers straight from process memory
/// (4 GiB-aligned arena opened as a D3D12 heap) and its stores land there, no copies.
#[test]
fn gpu_unified_memory_arena() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::new(1 << 20).unwrap();
    let va = arena.base() as u64;
    assert_eq!(va & 0xFFFF_FFFF, 0, "4 GiB alignment");
    let words = || unsafe { std::slice::from_raw_parts_mut(arena.base() as *mut u32, arena.size() / 4) };
    let w = words();
    w[0x1000 / 4] = 0xC0FFEE;
    w[0x1000 / 4 + 1] = 0x1234;
    for l in 0..64 {
        w[0x2000 / 4 + l] = 1000 + l as u32;
    }
    let p = prog(&[
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 1), ("OFFSET", 0)], None), // s[2:3] = console address
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
    ]);
    let t = translate(&p, &on_gpu(Options::default())).unwrap();
    let dxil = direct_or(&p, &on_gpu(Options::default()), super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap());
    let hi = (va >> 32) as u32;
    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[(va + 0x1000) as u32, hi]);
    st.sgpr[8..12].copy_from_slice(&[(va + 0x2000) as u32, hi & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[12..16].copy_from_slice(&[(va + 0x3000) as u32, hi & 0xFFFF | 4 << 16, 64, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[2] = 7 * l as u32;
    }
    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0xC0FFEE, 0x1234), "s_load from process memory");
    for l in 0..64 {
        assert_eq!(r.vgpr[l][1], 1000 + l as u32, "lane {l} buffer_load from process memory");
        assert_eq!(words()[0x3000 / 4 + l], 7 * l as u32, "lane {l} buffer_store visible to the CPU");
    }
}

/// An NGG program (mesh mode) reading title memory above the first window: its vertex table
/// is in window 1, decoys sit at the same offset in window 0. Translated with the title's
/// windows (`translate_mesh`'s `mem_windows`, as `pso::console_mesh` now does) it reads window
/// 1; with one window – what MS / GS programs got before – it read the decoys.
#[test]
fn gpu_mesh_mode_reads_memory_windows() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, MsInterface, MESH_HEADER};
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let ms = MsInterface { outputs: vec![sig("SV_Position", 1, 0)], threads: 64, max_vertices: 64, max_primitives: 64 };
    let list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None), // thread
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)), // 1 primitive, 3 vertices
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None), // threads 0..2
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("EXP", "ENC_EXP", vec![("TGT", 12), ("EN", 0xF), ("VSRC0", 10), ("VSRC1", 11), ("VSRC2", 12), ("VSRC3", 13), ("DONE", 1)], None),
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        for (k, v) in [*x, *y, 0.0, 1.0].iter().enumerate() {
            *word(w1 + 0x100 + 16 * i as u64 + 4 * k as u64) = f(*v);
            *word(w0 + 0x100 + 16 * i as u64 + 4 * k as u64) = f(9.0); // must not be read
        }
    }
    let mut st = KernelState::default();
    // V#: the vertex table in window 1, stride 16, 3 records
    st.sgpr[16..20].copy_from_slice(&[(w1 + 0x100) as u32, ((w1 >> 32) as u32) & 0xFFFF | 16 << 16, 3, 0]);
    let vd = ms.vert_dwords();
    let positions = |windows: u32| -> Vec<[f32; 4]> {
        let t = super::translate_mesh(&code, ms.clone(), true, runner::split_wave64(), windows).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let r = runner::run_in_arena(&cs, &st, &arena).unwrap();
        assert_eq!((r.raw[0], r.raw[1]), (3, 1), "GS_ALLOC_REQ counts ({windows} windows)");
        (0..3).map(|v| std::array::from_fn(|c| f32::from_bits(r.raw[MESH_HEADER + v * vd + c]))).collect()
    };
    assert_eq!(positions(2), vec![[-1.0, -1.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0]], "the vertex table from window 1");
    assert_eq!(positions(1)[0], [9.0; 4], "one window: window 0 at the same offset (the old MS / GS translation)");
}

/// Memory above 4 GiB: two UMA windows 4 GiB apart; every access kind picks its window from
/// the high address bits (V#, SGPR pair, 64-bit VGPR address, SADDR pair) and reaches the
/// right one – s_load, buffer load / store, global load / store / atomic in window 1 next to
/// accesses in window 0.
#[test]
fn gpu_memory_windows_above_4_gib() {
    memory_windows_above_4_gib(false);
}

/// The same program with the windows taken from the descriptor heap (`Options::heap_windows`):
/// same results, and no `switch` over windows in the HLSL.
#[test]
fn gpu_memory_windows_from_the_heap() {
    memory_windows_above_4_gib(true);
}

fn memory_windows_above_4_gib(heap: bool) {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };
    *word(w1 + 0x1000) = 0xC0FFEE;
    *word(w1 + 0x1004) = 0x1234;
    for l in 0..64u64 {
        *word(w1 + 0x2000 + 4 * l) = 1000 + l as u32;
        *word(w1 + 0x4000 + 4 * l) = 5000 + l as u32;
        *word(w0 + 0x4000 + 4 * l) = 9; // same low address in window 0: must not be read
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        (n, "ENC_FLAT_GLBL", f.iter().map(|&(a, b)| (unsafe { std::mem::transmute::<&str, &'static str>(a) }, b)).collect(), None)
    };
    let list = vec![
        ("S_LOAD_DWORDX2", "ENC_SMEM", vec![("SDATA", 20), ("SBASE", 1), ("OFFSET", 0)], None), // s[2:3] → window 1
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None), // V# in window 1
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", vec![("SDATA", 24), ("SBASE", 4), ("OFFSET", 4)], None), // that V#: one switch, 4 dwords
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None), // V# in window 0
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 6), ("ADDR", 4), ("SADDR", 0x7D), ("OFFSET", 0)]), // v[4:5] → window 1
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 8), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", 0)]), // v[8:9] → window 1
        glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 10), ("ADDR", 3), ("DATA", 11), ("SADDR", 16), ("OFFSET", 0), ("GLC", 1)]), // s[16:17] + v3 → window 1
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let p = prog(&refs);
    let one = translate(&p, &Options::default()).unwrap();
    assert!(!one.hlsl.contains("mem1") && !one.hlsl.contains("xcl_w =") && !one.hlsl.contains("xcl_m"), "one window: the single-window HLSL");
    let t = translate(&p, &on_gpu(Options { mem_windows: 2, heap_windows: heap, ..Default::default() })).unwrap();
    assert!(t.hlsl.contains("RWByteAddressBuffer mem1 : register(u3)"));
    assert_eq!(t.hlsl.contains("switch (xcl_w)"), !heap, "heap windows: no switch over windows");
    assert_eq!(t.hlsl.contains("ResourceDescriptorHeap[NonUniformResourceIndex(4u + xcl_w)]"), heap);
    if heap {
        let accessors: Vec<_> = t.hlsl.lines().filter(|line| line.contains(" XCL_MW;")).collect();
        assert!(!accessors.is_empty(), "heap accesses use the direct descriptor lookup");
        for accessor in accessors {
            let bounds = accessor.find("if (xcl_w >= 2u)").expect("reject an invalid window");
            let lookup = accessor.find(" XCL_MW;").unwrap();
            assert!(bounds < lookup, "reject before accessing the descriptor heap: {accessor}");
        }
    }
    let hlsl_dxil = || super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();
    let dxil = direct_or(&p, &on_gpu(Options { mem_windows: 2, heap_windows: heap, ..Default::default() }), hlsl_dxil());
    let hi = |a: u64| (a >> 32) as u32;
    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[(w1 + 0x1000) as u32, hi(w1)]);
    st.sgpr[8..12].copy_from_slice(&[(w1 + 0x2000) as u32, hi(w1) & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[12..16].copy_from_slice(&[(w0 + 0x3000) as u32, hi(w0) & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[16..18].copy_from_slice(&[(w1 + 0x6000) as u32, hi(w1)]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l = l as u64;
        r[0] = l as u32;
        r[2] = 7 * l as u32;
        r[3] = 0; // atomic offset: all lanes on one counter
        r[4] = (w1 + 0x4000 + 4 * l) as u32;
        r[5] = hi(w1);
        r[8] = (w1 + 0x5000 + 4 * l) as u32;
        r[9] = hi(w1);
        r[11] = 1;
    }
    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0xC0FFEE, 0x1234), "s_load from window 1");
    assert_eq!(r.sgpr[24..28], [1001, 1002, 1003, 1004], "s_buffer_load_dwordx4 from window 1");
    for l in 0..64u64 {
        assert_eq!(r.vgpr[l as usize][1], 1000 + l as u32, "lane {l}: buffer_load from window 1");
        assert_eq!(r.vgpr[l as usize][6], 5000 + l as u32, "lane {l}: global_load from window 1");
        assert_eq!(*word(w0 + 0x3000 + 4 * l), 7 * l as u32, "lane {l}: buffer_store into window 0");
        assert_eq!(*word(w1 + 0x5000 + 4 * l), 7 * l as u32, "lane {l}: global_store into window 1");
    }
    assert_eq!(*word(w1 + 0x6000), 64, "global_atomic_add in window 1");
    assert_eq!(*word(w0 + 0x6000), 0, "nothing in window 0 at the same low address");
}

/// Out-of-UMA addresses must NEVER alias onto live UMA windows via `& 3`:
/// - high dwords outside `[base_hi, base_hi + mem_windows)` (such as `base_hi + 4` / `+ 5` /
///   `base_hi - 1` / `0xFFFFFFFF` that share low bits with live windows) drop stores & atomics
///   (including CAS-loop atomics like `global_atomic_inc`) and read 0;
/// - `hi == 0` reads still reach window 0 (32-bit console descriptor tables), while `hi == 0`
///   writes are dropped so null pointers cannot clobber window 0;
/// - V# stride bits `d1[29:16]` are masked off before window selection, and 64-bit carry/borrow
///   on `global_load`/`global_store` updates the target window accurately.
#[test]
fn gpu_memory_windows_reject_out_of_uma_aliasing() {
    memory_windows_reject_out_of_uma_aliasing(false);
}

/// The same with the windows taken from the descriptor heap (`Options::heap_windows`): an
/// address outside them reads 0 and drops its stores and atomics there too.
#[test]
fn gpu_heap_windows_reject_out_of_uma_aliasing() {
    memory_windows_reject_out_of_uma_aliasing(true);
}

fn memory_windows_reject_out_of_uma_aliasing(heap_windows: bool) {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let base_hi = (w0 >> 32) as u32;
    assert!(base_hi >= 4);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };

    // Canaries in window 0 and window 1 that out-of-UMA stores/atomics must NOT touch
    const CANARY0: u32 = 0xCAFE_BABE;
    const CANARY1: u32 = 0x1337_BEEF;
    *word(w0 + 0x1000) = CANARY0;
    *word(w1 + 0x1000) = CANARY1;
    *word(w0 + 0x2000) = CANARY0;
    *word(w1 + 0x2000) = CANARY1;
    *word(w0 + 0x3000) = CANARY0;
    *word(w1 + 0x3000) = CANARY1;
    *word(w0 + 0x4000) = 0xABCD_0123; // 32-bit hi=0 descriptor-table style read target
    *word(w1 + 0x5000) = 0x5566_7788; // valid strided V# buffer_load target
    *word(w1 + 0x0010) = 0x99AA_BBCC; // 64-bit carry target: (w0 + 0xFFFF_FFF0) + 0x20 == w1 + 0x10

    let glbl = |n: &'static str, f: &[(&str, u64)]| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        (n, "ENC_FLAT_GLBL", f.iter().map(|&(a, b)| (unsafe { std::mem::transmute::<&str, &'static str>(a) }, b)).collect(), None)
    };
    let list = vec![
        // 1. Out-of-UMA s_load from hi = base_hi + 4 (would alias w0 under `& 3`) -> must read 0
        ("S_LOAD_DWORD", "ENC_SMEM", vec![("SDATA", 30), ("SBASE", 1), ("OFFSET", 0)], None), // s[2:3]
        // 2. 32-bit hi == 0 s_load at offset 0x4000 -> MUST read w0 + 0x4000
        ("S_LOAD_DWORD", "ENC_SMEM", vec![("SDATA", 31), ("SBASE", 2), ("OFFSET", 0)], None), // s[4:5]
        // 3. Out-of-UMA buffer_store with V# hi = base_hi + 5 and stride = 4 -> must NOT touch w1 + 0x1000
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None), // s[8:11]
        // 4. Out-of-UMA buffer_load with V# hi = base_hi + 5 and stride = 4 -> must read 0 (v2)
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 2), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None),
        // 5. Valid strided V# buffer_load from w1 + 0x5000 (d1 has non-zero stride 4 in bits 29:16) -> must read 0x5566_7788 (v3)
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 3), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None), // s[12:15]
        // 6. Out-of-UMA global_store with hi = base_hi + 4 (would alias w0 + 0x1000) -> dropped
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 4), ("DATA", 1), ("SADDR", 0x7D), ("OFFSET", 0)]),
        // 7. Out-of-UMA global_store with hi = 0 (null/32-bit pointer at 0x2000) -> dropped, must NOT touch w0 + 0x2000
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 1), ("SADDR", 0x7D), ("OFFSET", 0)]),
        // 8. Out-of-UMA global_atomic_add (hi = base_hi + 5) & global_atomic_inc CAS loop (hi = 0xFFFF_FFFF) -> dropped & terminates cleanly
        glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 10), ("ADDR", 8), ("DATA", 1), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]),
        glbl("GLOBAL_ATOMIC_INC", &[("VDST", 11), ("ADDR", 12), ("DATA", 1), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]),
        // 9. Out-of-UMA buffer_atomic_add with V# hi = base_hi + 4 (would alias w0 + 0x3000) -> dropped
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", vec![("VDATA", 14), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1), ("GLC", 1)], None), // s[16:19]
        // 10. 64-bit carry on global_load: s[20:21] = (w0 + 0xFFFF_FFF0) + v16 (0x20) -> w1 + 0x10 -> must read 0x99AA_BBCC (v17)
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 17), ("ADDR", 16), ("SADDR", 20), ("OFFSET", 0)]),
        // 11. 64-bit signed borrow out of UMA: s[22:23] = w0 + 0x0 + (-16) -> (base_hi - 1, 0xFFFF_FFF0) -> dropped, reads 0 (v18)
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 18), ("ADDR", 0), ("SADDR", 22), ("OFFSET", 0xFF0)]),
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let p = prog(&refs);
    let t = translate(&p, &on_gpu(Options { mem_windows: 2, heap_windows, ..Default::default() })).unwrap();
    assert_eq!(t.hlsl.contains("switch (xcl_w)"), !heap_windows);
    let hlsl_dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();
    let dxil = direct_or(&p, &on_gpu(Options { mem_windows: 2, heap_windows, ..Default::default() }), hlsl_dxil);

    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[0x1000, base_hi + 4]); // s[2:3]: out-of-UMA hi = base_hi + 4
    st.sgpr[4..6].copy_from_slice(&[0x4000, 0]); // s[4:5]: hi == 0 32-bit descriptor read
    st.sgpr[8..12].copy_from_slice(&[0x1000, ((base_hi + 5) & 0xFFFF) | (4 << 16), 64, 0]); // out-of-UMA V# (aliases w1 under & 3)
    st.sgpr[12..16].copy_from_slice(&[0x5000, ((base_hi + 1) & 0xFFFF) | (4 << 16), 64, 0]); // valid strided V# in w1
    st.sgpr[16..20].copy_from_slice(&[0x3000, ((base_hi + 4) & 0xFFFF) | (4 << 16), 64, 0]); // out-of-UMA atomic V# (aliases w0 under & 3)
    st.sgpr[20..22].copy_from_slice(&[0xFFFF_FFF0, base_hi]); // carry base
    st.sgpr[22..24].copy_from_slice(&[0, base_hi]); // borrow base (0 - 16 borrows into base_hi - 1)
    for r in st.vgpr.iter_mut() {
        r[0] = 0; // index 0
        r[1] = 0xBAD0_DEAD; // poison value to store
        r[2] = 0x7777_7777; // sentinel overwritten to 0 by out-of-UMA buffer_load
        r[10] = 0x7777_7777;
        r[11] = 0x7777_7777;
        r[14] = 0xBAD0_DEAD;
        r[4] = 0x1000;
        r[5] = base_hi + 4; // out-of-UMA global_store (aliases w0 under & 3)
        r[6] = 0x2000;
        r[7] = 0; // hi == 0 global_store (null pointer -> must be dropped)
        r[8] = 0x2000;
        r[9] = base_hi + 5; // out-of-UMA global_atomic_add (aliases w1 under & 3)
        r[12] = 0x3000;
        r[13] = 0xFFFF_FFFF; // out-of-UMA global_atomic_inc CAS loop
        r[16] = 0x20; // carry offset: 0xFFFF_FFF0 + 0x20 = 0x1_0000_0010 -> w1 + 0x10
        r[18] = 0x7777_7777;
    }

    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!(r.sgpr[30], 0, "out-of-UMA s_load must return 0");
    assert_eq!(r.sgpr[31], 0xABCD_0123, "hi == 0 s_load must still read window 0 (32-bit descriptor tables)");
    assert_eq!(r.vgpr[0][2], 0, "out-of-UMA buffer_load must return 0");
    assert_eq!(r.vgpr[0][3], 0x5566_7788, "valid strided V# buffer_load must mask stride out of d1");
    assert_eq!(r.vgpr[0][10], 0, "out-of-UMA global_atomic_add GLC must return 0");
    assert_eq!(r.vgpr[0][11], 0, "out-of-UMA global_atomic_inc CAS loop must return 0 and terminate");
    assert_eq!(r.vgpr[0][14], 0, "out-of-UMA buffer_atomic_add GLC must return 0");
    assert_eq!(r.vgpr[0][17], 0x99AA_BBCC, "64-bit carry global_load must cross from w0 into w1");
    assert_eq!(r.vgpr[0][18], 0, "64-bit signed borrow below window 0 must be rejected and read 0");

    assert_eq!(*word(w0 + 0x1000), CANARY0, "out-of-UMA global_store must not clobber w0");
    assert_eq!(*word(w1 + 0x1000), CANARY1, "out-of-UMA buffer_store must not clobber w1");
    assert_eq!(*word(w0 + 0x2000), CANARY0, "hi == 0 global_store must not clobber w0");
    assert_eq!(*word(w1 + 0x2000), CANARY1, "out-of-UMA global_atomic_add must not clobber w1");
    assert_eq!(*word(w0 + 0x3000), CANARY0, "out-of-UMA buffer_atomic_add must not clobber w0");
    assert_eq!(*word(w1 + 0x3000), CANARY1, "out-of-UMA global_atomic_inc must not clobber w1");
}

#[test]
fn gpu_window_at_32bit_carry_edge_cases() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let base_hi = (w0 >> 32) as u32;
    assert!(base_hi >= 4);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };

    *word(w0 + 0x0000) = 0x2222_3333;
    *word(w0 + 0x0010) = 0x9999_AAAA;
    *word(w0 + 0x0020) = 0xDEAD_DEAD;
    *word(w1 + 0x0000) = 0x1111_2222;
    *word(w1 + 0x0004) = 0x7777_8888;
    *word(w1 + 0x0008) = 0x3333_4444;
    *word(w1 + 0x0010) = 0x5555_6666;
    *word(w1 + 0x0020) = 0xCAFE_CAFE;

    let glbl = |n: &'static str, f: &[(&str, u64)]| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        (n, "ENC_FLAT_GLBL", f.iter().map(|&(a, b)| (unsafe { std::mem::transmute::<&str, &'static str>(a) }, b)).collect(), None)
    };
    let list = vec![
        // 1. Ea::U carry: (w0 + 0xFFFF_FFFC) + v0(4) = w1 + 0x0000 -> 0x1111_2222
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 10), ("ADDR", 0), ("SADDR", 2), ("OFFSET", 0)]),
        // 2. Ea::S positive constant carry: (w0 + 0xFFFF_FFF8) + OFFSET(+16) = w1 + 0x0008 -> 0x3333_4444
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 11), ("ADDR", 1), ("SADDR", 4), ("OFFSET", 16)]),
        // 3. Ea::Mul pure high carry: w0 + (0x1000 * 0x10_0000) + v3(0x10) = w1 + 0x10 -> 0x5555_6666
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 12), ("VADDR", 2), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1), ("OFFEN", 1), ("OFFSET", 0)], None),
        // 4. Ea::Mul simultaneous lo-carry + phi: ((base_hi - 1):0xFFFF_F004) + (0x1000 * 0x10_0001) = w1 + 4 -> 0x7777_8888
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 13), ("VADDR", 4), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1), ("OFFEN", 0), ("OFFSET", 0)], None),
        // 5. S_LOAD_DWORD negative Ea::S(OFFSET = -16) borrows base_hi -> base_hi-1, then Ea::U(SOFFSET = s16 = 0x20) carries back to w0 + 0x10 -> 0x9999_AAAA
        ("S_LOAD_DWORD", "ENC_SMEM", vec![("SDATA", 20), ("SBASE", 3), ("OFFSET", 0x1F_FFF0), ("SOFFSET", 16)], None),
        // 6. Carry past end of UMA: (w1 + 0xFFFF_FFF0) + v6(0x20) = base_hi+2 -> read 0, drop store
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 14), ("ADDR", 6), ("SADDR", 24), ("OFFSET", 0)]),
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 7), ("SADDR", 24), ("OFFSET", 0)]),
        // 7. Ea::S negative OFFSET(-16) with uaddc carry=1 cancels -1+1=0: (w0 + 0x0010) + (-16) = w0 + 0 -> 0x2222_3333
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 15), ("ADDR", 1), ("SADDR", 18), ("OFFSET", 0xFF0)]),
        // 8. Ea::S negative OFFSET(-16) without carry borrows below w0: (w0 + 0x0030) - 0x30 + (-16) => store at (w0 + 0x0000) - 16 is dropped, w0 + 0x0020 intact
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 1), ("DATA", 8), ("SADDR", 6), ("OFFSET", 0xFF0)]),
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let p = prog(&refs);
    let t = translate(&p, &on_gpu(Options { mem_windows: 2, ..Default::default() })).unwrap();
    let hlsl_dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();
    let dxil = direct_or(&p, &on_gpu(Options { mem_windows: 2, ..Default::default() }), hlsl_dxil);

    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[0xFFFF_FFFC, base_hi]); // s[2:3] = w0 + 0xFFFF_FFFC
    st.sgpr[4..6].copy_from_slice(&[0xFFFF_FFF8, base_hi]); // s[4:5] = w0 + 0xFFFF_FFF8
    st.sgpr[6..8].copy_from_slice(&[0, base_hi]); // s[6:7] = w0 + 0 (SBASE 3)
    st.sgpr[8..12].copy_from_slice(&[0, (base_hi & 0xFFFF) | (0x1000 << 16), 0x20_0000, 0]); // SRSRC 2 (s[8:11])
    st.sgpr[12..16].copy_from_slice(&[0xFFFF_F004, ((base_hi - 1) & 0xFFFF) | (0x1000 << 16), 0x20_0000, 0]); // SRSRC 3 (s[12:15])
    st.sgpr[16] = 0x20; // SOFFSET for S_LOAD_DWORD
    st.sgpr[18..20].copy_from_slice(&[0x0010, base_hi]); // s[18:19] = w0 + 0x10
    st.sgpr[24..26].copy_from_slice(&[0xFFFF_FFF0, base_hi + 1]); // s[24:25] = w1 + 0xFFFF_FFF0
    for r in st.vgpr.iter_mut() {
        r[0] = 4;
        r[1] = 0;
        r[2] = 0x10_0000;
        r[3] = 0x10;
        r[4] = 0x10_0001;
        r[6] = 0x20;
        r[7] = 0xDEAD_C0DE;
        r[8] = 0x1234_5678;
        r[14] = 0x9999_9999;
    }

    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!(r.vgpr[0][10], 0x1111_2222, "Ea::U carry w0 -> w1");
    assert_eq!(r.vgpr[0][11], 0x3333_4444, "Ea::S positive carry w0 -> w1");
    assert_eq!(r.vgpr[0][12], 0x5555_6666, "Ea::Mul high dword carry w0 -> w1");
    assert_eq!(r.vgpr[0][13], 0x7777_8888, "Ea::Mul lo-carry + phi w0-1 -> w1");
    assert_eq!(r.sgpr[20], 0x9999_AAAA, "S_LOAD_DWORD borrow then SOFFSET carry-back to w0");
    assert_eq!(r.vgpr[0][14], 0, "carry past UMA limit returns 0");
    assert_eq!(r.vgpr[0][15], 0x2222_3333, "Ea::S negative OFFSET with carry=1 stays in w0");
    assert_eq!(*word(w0 + 0x0020), 0xDEAD_DEAD, "store borrowing below w0 is dropped");
    assert_eq!(*word(w1 + 0x0020), 0xCAFE_CAFE, "store past UMA limit dropped");
}

/// The start-up check of the coherent UMA windows (`real::check_uma_import`): the GPU sees what
/// the CPU writes there. Fails, as it should, where the import fell back to memory of its own
/// (vkd3d-proton under a Wine without `userfaultfd` write watches, #175).
#[test]
fn gpu_uma_import_check_at_init() {
    if !gpu_ready() {
        return;
    }
    assert_eq!(crate::hle::d3d12x::backend(), crate::hle::d3d12x::Backend::Real, "real D3D12.X backend");
    let s = crate::hle::d3d12x::real::stats().uma_import_check;
    assert!(s.starts_with("ok") || s.starts_with("off"), "{s}");
}

/// End-to-end GPU test of the sparse-window VRAM runtime layout:
/// - Window 0..2 = sparse VRAM (tiled reserved buffers on DEFAULT heaps)
/// - Window 3 = committed coherent RAM (`OpenExistingHeapFromAddress`)
/// Exercises a DXC-compiled SM 6.6 translated RDNA2 compute shader (`mem_windows = 4`)
/// reading a 64-bit spilled root-argument constant from Window 3 (`s[0:1]`), loading two
/// V# buffer descriptors through a 32-bit `hi = 0` descriptor table pointer in Window 0
/// sparse VRAM (`console_heap`), reading input from `XMemAlloc` graphics memory in sparse
/// VRAM, writing to Window 3, and then mutating both the buffer and the descriptor slot in
/// Window 0 sparse VRAM between submits.
#[test]
fn gpu_sparse_vram_descriptors_buffers_and_xmem() {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::*;

    if !gpu_ready() {
        return;
    }
    assert_eq!(crate::hle::d3d12x::backend(), crate::hle::d3d12x::Backend::Real, "real D3D12.X backend");
    let u = crate::gpu::uma::get().expect("UMA");
    assert_eq!(u.shader_windows(), 4, "runtime layout has 4 shader windows");
    assert!(u.is_sparse_window(0) && u.is_sparse_window(1) && u.is_sparse_window(2), "windows 0..2 sparse VRAM");
    assert!(u.is_committed_window(3), "window 3 coherent committed RAM");
    let windows = crate::hle::d3d12x::real::uma_window_vas();
    assert_eq!(windows.len(), 4);
    assert!(windows.iter().all(|&va| va != 0), "all 4 window GPU VAs must be valid: {windows:x?}");

    // 1. Console descriptor heap in Window 0 sparse VRAM (32-bit hi = 0 table addressing)
    let heap_obj = 0xDEAD_5EA7usize;
    assert!(crate::hle::d3d12x::console_heap::register(heap_obj, 4, 0x1000, 0x5000, 32, false, false, true));
    let heap_va = crate::hle::d3d12x::console_heap::start(heap_obj).unwrap() as usize;
    assert_eq!(u.window_of(heap_va), Some(0), "descriptor heap must live in window 0");
    assert!(u.is_sparse(heap_va), "descriptor heap must live in sparse VRAM");

    // 2. XMem graphics allocations in sparse VRAM
    let gfx_attrs = 1u64 << 29;
    let in_a = unsafe { crate::hle::kernelx::XMemAllocDefault(0x1_0000, gfx_attrs) } as usize;
    let in_b = unsafe { crate::hle::kernelx::XMemAllocDefault(0x1_0000, gfx_attrs) } as usize;
    assert!(in_a != 0 && in_b != 0 && u.is_sparse(in_a) && u.is_sparse(in_b), "XMem graphics buffers must live in sparse VRAM");

    // 3. Coherent Window 3 allocations: spilled root argument table + output buffer + io/res
    let arg_table = u.alloc(0x1_0000, 256).unwrap() as usize;
    let out_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    let io_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    let res_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    assert_eq!(u.window_of(arg_table), Some(3));
    assert_eq!(u.window_of(out_buf), Some(3));

    // Populate initial data & descriptors (slot 0 = V# in_a, slot 1 = V# out_buf)
    let vsharp = |addr: usize| -> [u32; 4] { [addr as u32, (((addr >> 32) as u32) & 0xFFFF) | (4 << 16), 64, 0] };
    unsafe {
        *(arg_table as *mut u32) = 77;
        for l in 0..64usize {
            *((in_a + 4 * l) as *mut u32) = 1000 + l as u32;
            *((in_b + 4 * l) as *mut u32) = 5000 + 2 * l as u32;
        }
    }
    let slot0 = crate::hle::d3d12x::console_heap::slot(heap_va as u64).unwrap();
    let slot1 = crate::hle::d3d12x::console_heap::slot(heap_va as u64 + 32).unwrap();
    crate::hle::d3d12x::console_heap::write(&slot0, &vsharp(in_a));
    crate::hle::d3d12x::console_heap::write(&slot1, &vsharp(out_buf));

    // Translate + DXC-compile RDNA2 compute shader (4 windows, 32-bit descriptor table with hi = 0)
    let p = prog(&[
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125)], None), // s[0:1] -> Window 3 root arg table
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)], None), // s5 = 0 (32-bit descriptor table pointer)
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)], None), // s[4:5] + 0 -> Window 0 sparse desc slot 0
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 2), ("OFFSET", 32), ("SOFFSET", 125)], None), // s[4:5] + 32 -> Window 0 sparse desc slot 1
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 20), ("VSRC1", 1)], None),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
    ]);
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 6, ..Default::default() };
    let opt = on_gpu(Options { mode: Mode::Compute(cs), mem_windows: 4, ..Default::default() });
    let t = translate(&p, &opt).unwrap();
    let dxil = direct_or(&p, &opt, super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap());

    // Execute on the real D3D12.X device + queue (goes through q_execute_command_lists -> sparse::before_execute)
    unsafe {
        let dev_raw = crate::hle::d3d12x::real::device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&dev_raw).expect("device").clone();
        let rs = crate::gpu::recomp::pso::abi_root_signature(&dev, false).expect("ABI root signature");
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
                ..Default::default()
            })
            .expect("CreateComputePipelineState");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso)).unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        // The ABI root signature indexes both heaps directly, so they must be bound before it.
        let heap = |ty| -> ID3D12DescriptorHeap {
            dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 }).unwrap()
        };
        let heaps = [Some(heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)), Some(heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER))];

        // User SGPRs: s[0:1] = 64-bit arg_table (Window 3), s4 = 32-bit descriptor table offset in Window 0
        let mut io = vec![0u32; crate::gpu::recomp::KERNEL_SGPRS];
        io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
        io[0] = arg_table as u32;
        io[1] = (arg_table >> 32) as u32;
        io[4] = (heap_va - u.base()) as u32;
        io[5] = 0xDEAD_BEEF; // shader overwrites s5 = 0 to prove hi == 0 32-bit addressing hits Window 0 sparse VRAM
        std::ptr::copy_nonoverlapping(io.as_ptr() as *const u8, io_buf as *mut u8, io.len() * 4);

        let record_and_run = |fence_val: u64, first: bool| {
            if !first {
                alloc.Reset().unwrap();
                list.Reset(&alloc, Some(&pso)).unwrap();
            }
            list.SetDescriptorHeaps(&heaps);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(0, crate::hle::d3d12x::real::uma_va_of(io_buf).unwrap());
            list.SetComputeRootUnorderedAccessView(1, crate::hle::d3d12x::real::uma_va_of(res_buf).unwrap());
            for (k, &va) in windows.iter().enumerate() {
                list.SetComputeRootUnorderedAccessView(2 + k as u32, va);
            }
            list.Dispatch(1, 1, 1);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, fence_val).unwrap();
            if fence.GetCompletedValue() < fence_val {
                let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap();
                fence.SetEventOnCompletion(fence_val, ev).unwrap();
                windows::Win32::System::Threading::WaitForSingleObject(ev, 5000);
                let _ = windows::Win32::Foundation::CloseHandle(ev);
            }
        };

        // Submit 1: reads slot 0 -> in_a (sparse VRAM) + arg_table constant 77 (Window 3)
        record_and_run(1, true);
        for l in 0..64usize {
            let got = *((out_buf + 4 * l) as *const u32);
            assert_eq!(got, 1000 + l as u32 + 77, "submit 1 lane {l} mismatch");
        }

        // Submit 2: mutate descriptor slot 0 in Window 0 sparse VRAM -> in_b, and mutate in_b[3] + arg_table
        *(arg_table as *mut u32) = 13;
        *((in_b + 4 * 3) as *mut u32) = 9999;
        crate::hle::d3d12x::console_heap::write(&slot0, &vsharp(in_b));
        record_and_run(2, false);
        for l in 0..64usize {
            let expected = if l == 3 { 9999 + 13 } else { 5000 + 2 * l as u32 + 13 };
            let got = *((out_buf + 4 * l) as *const u32);
            assert_eq!(got, expected, "submit 2 (mutated sparse descriptor + XMem buffer) lane {l} mismatch");
        }

        // Clean up
        crate::hle::d3d12x::console_heap::unregister(heap_obj);
        crate::hle::kernelx::XMemFreeDefault(in_a as *mut _, gfx_attrs);
        crate::hle::kernelx::XMemFreeDefault(in_b as *mut _, gfx_attrs);
        assert!(u.free(arg_table) && u.free(out_buf) && u.free(io_buf) && u.free(res_buf));
    }
}

/// Experiment: does OpenExistingHeapFromAddress accept a reservation with only part committed?
#[test]
#[ignore]
fn gpu_arena_partial_commit_probe() {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows_sys::Win32::System::Memory::*;
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).unwrap();
        let dev3: ID3D12Device3 = dev.unwrap().cast().unwrap();
        let size = 256usize << 20;
        let p = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE);
        let c = VirtualAlloc(p, 1 << 20, MEM_COMMIT, PAGE_READWRITE);
        println!("reserve {p:?} commit {c:?}");
        let r: windows::core::Result<ID3D12Heap> = dev3.OpenExistingHeapFromAddress(p);
        println!("partial commit: {:?}", r.as_ref().map(|h| h.GetDesc().SizeInBytes));
        let q = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        let r2: windows::core::Result<ID3D12Heap> = dev3.OpenExistingHeapFromAddress(q);
        println!("full commit 256 MB: {:?}", r2.as_ref().map(|h| h.GetDesc().SizeInBytes));
        for mb in [1024usize, 2047, 2048, 3072, 4095] {
            let big = mb << 20;
            let b = VirtualAlloc(std::ptr::null(), big, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            let t0 = std::time::Instant::now();
            let r3: windows::core::Result<ID3D12Heap> = if b.is_null() { Err(windows::core::Error::from_thread()) } else { dev3.OpenExistingHeapFromAddress(b) };
            println!("full commit {mb} MB: {:?} in {:.2} s", r3.as_ref().map(|h| h.GetDesc().SizeInBytes), t0.elapsed().as_secs_f64());
            drop(r3);
            VirtualFree(b, 0, MEM_RELEASE);
        }
    }
}

/// Decoded GFX10 T#s give resource descriptions the driver accepts.
#[test]
fn gpu_tsharp_to_d3d12_resources() {
    if !gpu_ready() {
        return;
    }
    use crate::gpu::desc;
    let t = |fmt: u32, w: u32, h: u32, depth: u32, ty: u32, mips: u32| -> [u32; 8] {
        let (w1, h1) = (w - 1, h - 1);
        [0, fmt << 20 | (w1 & 3) << 30, (w1 >> 2) | h1 << 14, 4 | 5 << 3 | 6 << 6 | 7 << 9 | (mips - 1) << 16 | ty << 28, depth - 1, (mips - 1) << 4, 0, 0]
    };
    for (name, words) in [
        ("2D RGBA8 9 mips", t(56, 256, 256, 1, 9, 9)),
        ("2D BC7 sRGB", t(182, 128, 64, 1, 9, 1)),
        ("cube RGBA16F", t(71, 64, 64, 6, 11, 7)),
        ("3D R11G11B10", t(36, 32, 32, 16, 10, 1)),
        ("2D array BC1 ×8", t(169, 64, 64, 8, 13, 4)),
        ("1D R32F", t(22, 1024, 1, 1, 8, 1)),
    ] {
        let img = desc::image(&words).unwrap();
        let (res, _srv) = desc::image_d3d12(&img).unwrap();
        runner::try_create_texture(&res).unwrap_or_else(|e| panic!("{name}: {e} ({res:?})"));
    }
}

/// Prints signatures / PSV0 of named retail shaders (`XCL_SAMPLE_RPF`, `XCL_DUMP=a,b`).
#[test]
#[ignore]
fn dump_retail_interfaces() {
    let (Ok(path), Ok(names)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_DUMP")) else { return };
    let data = std::fs::read(path).unwrap();
    for r in crate::gpu::pack::scan(&data, 0x40) {
        if !names.split(',').any(|n| r.name.contains(n)) {
            continue;
        }
        let blob = crate::gpu::pack::inflate(&data, &r).unwrap();
        let (i, o) = crate::gpu::dxbc::signatures(&blob);
        let sh = crate::gpu::xbox::parse(&blob).unwrap();
        println!("{} hw {:?}", r.name, sh.fssm.iter().map(|f| f.stage_label()).collect::<Vec<_>>());
        println!("  ISG1: {:?}", i.iter().map(|e| format!("{}@r{}:{:x} sv{} t{}", e.semantic(), e.register, e.mask, e.system_value, e.component_type)).collect::<Vec<_>>());
        println!("  OSG1: {:?}", o.iter().map(|e| format!("{}@r{}:{:x} sv{} t{}", e.semantic(), e.register, e.mask, e.system_value, e.component_type)).collect::<Vec<_>>());
        if let Some(parts) = crate::gpu::shader::container_parts(&blob) {
            if let Some(p) = parts.iter().find(|p| p.fourcc == "PSV0") {
                let d = &blob[p.offset..p.offset + p.size];
                println!("  PSV0 {:02x?}", &d[..d.len().min(60)]);
            }
        }
    }
}

/// Geometry mode end to end: an ES program (vertex → LDS, s_setpc to the GS) and a GS
/// program (triangle from LDS through the v0/v1 offsets, a 4-vertex strip = 2 triangles,
/// emit / cut) run as one compute shader; 25 triangles = 2 groups (21 + 4); records checked
/// vertex by vertex (strip winding included), then drawn by the mesh pass-through.
#[test]
fn gpu_geometry_mode_es_gs_strip() {
    geometry_es_gs_strip(false);
}

/// The same program with branches in the ES (forward) and in the GS (EXEC-zero skip and a
/// scalar loop): the guarded block chain must keep the phases apart in the split form – the
/// ES stops at its s_setpc, the GS phase enters at `gs_start` (a structured stretch across the
/// s_setpc ran the GS in the ES phase and left the GS phase without an entry: empty records).
#[test]
fn gpu_geometry_mode_es_gs_strip_with_branches() {
    geometry_es_gs_strip(true);
}

/// The geometry program of `gpu_geometry_mode_es_gs_strip`, optionally with branches.
fn geometry_es_gs_program(branches: bool) -> (Vec<u8>, super::GsInterface) {
    use super::GsInterface;
    let e = |n: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, enc, f, lit).unwrap();
    // ES: vertex (v9..v12) → LDS at lane·32, then into the GS
    let mut es = Vec::new();
    if branches {
        // if (s3 == 0) skip a s_mov (always taken or not: no effect on the result)
        es.extend(e("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 3), ("SSRC1", 128)], None));
        es.extend(e("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None));
        es.extend(e("S_MOV_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 129)], None));
    }
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None));
    es.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 133), ("VSRC1", 20)], None));
    es.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 9), ("DATA1", 10), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    es.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 11), ("DATA1", 12), ("OFFSET0", 2), ("OFFSET1", 3)], None));
    es.extend(e("S_BARRIER", "ENC_SOPP", &[], None));
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None));
    // GS: EXEC = primitives of this wave (s3[15:8]), vertices from LDS, parallelogram strip
    let mut gs = Vec::new();
    gs.extend(e("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)));
    gs.extend(e("S_BFM_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)], None));
    let mut skip_at = None;
    if branches {
        // no primitives: straight to GS done (patched below)
        skip_at = Some(gs.len());
        gs.extend(e("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 0)], None));
        // s8 = 2; do { s8 -= 1 } while (s8 != 0)
        gs.extend(e("S_MOV_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 130)], None));
        gs.extend(e("S_SUB_U32", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 8), ("SSRC1", 129)], None));
        gs.extend(e("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 8), ("SSRC1", 128)], None));
        gs.extend(e("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None));
    }
    gs.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    gs.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 130), ("VSRC1", 0)], None));
    gs.extend(e("V_AND_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 255), ("VSRC1", 30)], Some(0x3FFFC)));
    gs.extend(e("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 144), ("VSRC1", 0)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 130), ("VSRC1", 31)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 32), ("SRC0", 130), ("VSRC1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 40), ("ADDR", 30), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 42), ("ADDR", 31), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 44), ("ADDR", 32), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    for (d, a, b) in [(46u64, 42u64, 44u64), (47, 43, 45)] {
        gs.extend(e("V_ADD_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)], None));
        gs.extend(e("V_SUB_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + d), ("VSRC1", d - 6)], None)); // − x0 / − y0
    }
    // output base: s2 + lane · (4 vertices · 32 bytes)
    gs.extend(e("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 50), ("SRC0", V + 20), ("SRC1", 255), ("SRC2", 2)], Some(128)));
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 51), ("SRC0", 240)], None)); // 0.5
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 52), ("SRC0", 242)], None)); // 1.0
    for (k, (x, y)) in [(40u64, 41u64), (42, 43), (44, 45), (46, 47)].into_iter().enumerate() {
        let o = 8 * k as u64;
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", x), ("DATA1", y), ("OFFSET0", o), ("OFFSET1", o + 1)], None));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 51), ("DATA1", 52), ("OFFSET0", o + 2), ("OFFSET1", o + 3)], None));
        gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 53), ("SRC0", 128 + k as u64)], None));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 2), ("DATA1", 53), ("OFFSET0", o + 4), ("OFFSET1", o + 5)], None));
        gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None)); // emit
    }
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x12)], None)); // cut
    if let Some(at) = skip_at {
        let off = ((gs.len() - at - 4) / 4) as u64;
        gs[at..at + 4].copy_from_slice(&e("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", off)], None));
    }
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None)); // GS done
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));

    let sig = |name: &str, sv: u32, reg: u32, ty: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: ty,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let g = GsInterface {
        es_inputs: vec![sig("POSITION", 0, 0, 3)],
        // two ES output registers: 32-byte ESGS items (the ES above writes at lane · 32)
        es_outputs: vec![sig("SV_Position", 1, 0, 3), sig("TEXCOORD", 0, 1, 3)],
        outputs: vec![sig("SV_Position", 1, 0, 3), sig("TEXCOORD", 0, 1, 1)],
        max_vertex_count: 4,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es;
    code.extend_from_slice(&gs);
    (code, g)
}

/// Split form with branches: the ES phase ends at the s_setpc and the GS phase has its entry
/// (`if (pc == gs_start)`), also where no GPU runs the program.
#[test]
fn geometry_split_phases_with_branches_have_entry_and_exit() {
    let (code, g) = geometry_es_gs_program(true);
    let s = g.gs_start.unwrap();
    let t = translate(&code, &Options { mode: Mode::Geometry(g.clone()), wave64: true, split_wave64: true, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains(&format!("if (pc == {s}u)")), "{}", t.hlsl);
    assert!(t.hlsl.contains(&format!("pc = select(xcl_es_phase, 0xFFFFFFFFu, {s}u);")), "{}", t.hlsl);
    let n = translate(&code, &Options { mode: Mode::Geometry(g), wave64: true, split_wave64: false, ..Default::default() }).unwrap();
    assert!(n.complete() && !n.hlsl.contains("xcl_es_phase"), "{}", n.hlsl);
}

fn geometry_es_gs_strip(branches: bool) {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    use super::{dxc, mesh_passthrough, IO_GS_PRIMS, IO_GS_VERTICES, MESH_HEADER};
    let (code, g) = geometry_es_gs_program(branches);
    let native = translate(&code, &Options { mode: Mode::Geometry(g.clone()), wave64: true, split_wave64: false, ..Default::default() }).unwrap();
    assert!(native.hlsl.contains("[WaveSize(64)]"));
    assert!(!native.hlsl.contains("xcl_es_phase") && !native.hlsl.contains("GroupMemoryBarrierWithGroupSync();\n    if (gi < 32u)"));
    let t = translate(&code, &Options { mode: Mode::Geometry(g.clone()), wave64: true, split_wave64: true, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("[WaveSize(32)]"));
    assert!(t.hlsl.contains("xcl_es_phase") && t.hlsl.contains("GroupMemoryBarrierWithGroupSync();\n    if (gi < 32u)"));
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let rec = g.record();
    let msb = dxc::compile(&mesh_passthrough(&rec), "main", "ms_6_6", &[]).unwrap();
    let ps_src = "struct V { float4 pos : SV_Position; nointerpolation uint4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return float4(1.0, (float)v.p0.y, 0.0, 1.0); }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();

    // 25 triangles, vertex attributes pre-fetched at 0x1000 (one float4 per vertex)
    let total = 25usize;
    let vtx = |p: usize, i: usize| [p as f32 * 0.03 - 0.6 + i as f32 * 0.2, p as f32 * 0.02 - 0.5 + (i == 2) as u32 as f32 * 0.25, 0.5, 1.0];
    let mut mem = vec![0u32; 0x1000 / 4 + total * 3 * 4];
    for p in 0..total {
        for i in 0..3 {
            let at = 0x1000 / 4 + (p * 3 + i) * 4;
            mem[at..at + 4].copy_from_slice(&vtx(p, i).map(f32::to_bits));
        }
    }
    let mut io = vec![0u32; 128];
    io[IO_GS_PRIMS] = total as u32;
    io[IO_GS_VERTICES] = 0x1000;
    let (px, recs) = runner::mesh_draw(&cs, &msb, &ps, 2, rec.record_dwords(), &io, &mem, 32, [0.0; 4]).unwrap();
    let (rd, vd) = (rec.record_dwords(), rec.vert_dwords());
    for grp in 0..2usize {
        let r = &recs[grp * rd..(grp + 1) * rd];
        let np = if grp == 0 { 21 } else { 4 };
        assert_eq!((r[0], r[1]), (4 * np as u32, 2 * np as u32), "group {grp}: vertices / triangles");
        for lane in 0..np {
            let p = grp * 21 + lane;
            let (a, b, c) = (vtx(p, 0), vtx(p, 1), vtx(p, 2));
            let xy = [[a[0], a[1]], [b[0], b[1]], [c[0], c[1]], [b[0] + c[0] - a[0], b[1] + c[1] - a[1]]];
            for (k, want) in xy.iter().enumerate() {
                let v = MESH_HEADER + (4 * lane + k) * vd;
                let pos = [f32::from_bits(r[v]), f32::from_bits(r[v + 1]), f32::from_bits(r[v + 2]), f32::from_bits(r[v + 3])];
                assert_eq!(pos, [want[0], want[1], 0.5, 1.0], "group {grp} lane {lane} vertex {k}");
                assert_eq!((r[v + 4], r[v + 5]), (p as u32, k as u32), "group {grp} lane {lane} vertex {k}: TEXCOORD0");
            }
            let pk = |i: usize, j: usize, l: usize| (i | j << 10 | l << 20) as u32;
            let tris = MESH_HEADER + rec.max_vertices as usize * vd;
            let b0 = 4 * lane;
            assert_eq!(r[tris + 2 * lane], pk(b0, b0 + 1, b0 + 2), "group {grp} lane {lane}: strip triangle 0");
            assert_eq!(r[tris + 2 * lane + 1], pk(b0 + 2, b0 + 1, b0 + 3), "group {grp} lane {lane}: strip triangle 1 (odd: swapped)");
        }
    }
    let covered = px.iter().filter(|p| p[0] == 1.0).count();
    assert!(covered > 20, "pass-through drew the GS output: {covered} pixels");
}

/// v_mbcnt over a v_cmp mask in a geometry program (the compaction index of an ES / GS):
/// lanes 32..63 count the active lanes of 0..31 too. The split form (two wave32 halves) must
/// give the logical wave64 counts – it exchanges the halves' mask words – and so must the
/// native form where the GPU has 64-lane waves (the compute pass alone: no mesh shaders needed).
#[test]
fn gpu_geometry_wave32_prefix_count_over_a_mask() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, translate_geometry, GsInterface, IO_GS_PRIMS};
    let e = |n: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, enc, f, lit).unwrap();
    let mut es = Vec::new();
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None)); // lane
    es.extend(e("V_AND_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 131), ("VSRC1", 20)], None));
    es.extend(e("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 21)], None)); // vcc = lane & 3 != 0
    es.extend(e("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 10), ("SRC0", 168), ("SRC1", V + 20)], None)); // s[10:11] = lane < 40
    // v22: below this lane in VCC, v23: in s[10:11], v24: in EXEC (= VCC; the others keep 0xFFFF)
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", 106), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", 107), ("SRC1", V + 22)], None));
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 10), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 11), ("SRC1", V + 23)], None));
    es.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", 255)], Some(0xFFFF)));
    es.extend(e("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 106)], None));
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 24), ("SRC0", 126), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 24), ("SRC0", 127), ("SRC1", V + 24)], None));
    es.extend(e("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 12)], None));
    // s[8:9] + lane · 4: v22 at +0, v23 at +256, v24 at +512
    es.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 25), ("SRC0", 130), ("VSRC1", 20)], None));
    for (k, d) in [22u64, 23, 24].into_iter().enumerate() {
        es.extend(e("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 25), ("DATA", d), ("SADDR", 8), ("OFFSET", 256 * k as u64)], None));
    }
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None));
    let mut gs = Vec::new();
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None)); // GS done, nothing emitted
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let g = GsInterface {
        es_inputs: vec![],
        es_outputs: vec![sig("SV_Position", 1, 0)],
        outputs: vec![sig("SV_Position", 1, 0)],
        max_vertex_count: 3,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es;
    code.extend_from_slice(&gs);

    // what one wave64 computes
    let below = |m: u64, l: usize| (m & ((1u64 << l) - 1)).count_ones();
    let (vcc, s10) = ((0..64).filter(|l| l & 3 != 0).fold(0u64, |m, l| m | 1 << l), (1u64 << 40) - 1);
    let want: Vec<[u32; 3]> = (0..64).map(|l| [below(vcc, l), below(s10, l), if vcc >> l & 1 == 1 { below(vcc, l) } else { 0xFFFF }]).collect();
    let mut io = vec![0u32; 128];
    io[8] = 0x400;
    io[IO_GS_PRIMS] = 21;
    let run = |t: &super::Translation| {
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let mem = runner::dispatch(&cs, [1, 1, 1], &io, &vec![0u32; 0x400 / 4 + 3 * 64]).unwrap();
        let got: Vec<[u32; 3]> = (0..64).map(|l| [0, 1, 2].map(|k| mem[0x100 + 64 * k + l])).collect();
        let wrong: Vec<_> = (0..64).filter(|&l| got[l] != want[l]).map(|l| (l, got[l], want[l])).collect();
        assert!(wrong.is_empty(), "lanes (got [vcc, s[10:11], exec], want): {wrong:?}");
    };
    run(&translate_geometry(&code, g.clone(), true, 1).unwrap());
    // with a title's four memory windows (`pso::console_geometry`): all of them declared and
    // the program still compiles
    let windowed = translate_geometry(&code, g.clone(), true, 4).unwrap();
    assert!(windowed.hlsl.contains("RWByteAddressBuffer mem3 : register(u5)"));
    super::dxc::compile(&windowed.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    // the native wave64 form keeps the program, and computes the reference where it can run
    let native = translate_geometry(&code, g.clone(), false, 1).unwrap();
    if runner::wave64_supported() == Ok(true) {
        run(&native);
    } else {
        runner::skip_test("no wave64");
    }
    // the direct DXIL translator's compute pass computes the same, split and native
    let direct = |split: bool| {
        let o = crate::gpu::dxil::kernel::KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 };
        let cs = crate::gpu::dxil::kernel::translate_geometry(&code, o, &g).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(2000)])).dxil;
        let mem = runner::dispatch(&cs, [1, 1, 1], &io, &vec![0u32; 0x400 / 4 + 3 * 64]).unwrap();
        let got: Vec<[u32; 3]> = (0..64).map(|l| [0, 1, 2].map(|k| mem[0x100 + 64 * k + l])).collect();
        let wrong: Vec<_> = (0..64).filter(|&l| got[l] != want[l]).map(|l| (l, got[l], want[l])).collect();
        assert!(wrong.is_empty(), "direct split {split} (got [vcc, s[10:11], exec], want): {wrong:?}");
    };
    direct(true);
    if runner::wave64_supported() == Ok(true) {
        direct(false);
    }
}

/// A DS read whose destination overlaps its address register reads the address first
/// (retail GS code: `ds_read2_b32 v[0:1], v0 offset1:1`).
#[test]
fn gpu_ds_read_destination_overlaps_address() {
    if !gpu_ready() {
        return;
    }
    let mut code = Vec::new();
    code.extend(encode("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 2), ("DATA1", 3), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("DS_READ2_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = 8 * l as u32; // address
        r[2] = 1000 + l as u32;
        r[3] = 2000 + l as u32;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!((r.vgpr[l][1], r.vgpr[l][2]), (1000 + l as u32, 2000 + l as u32), "lane {l}");
    }
}

/// `DS_WRITE_ADDTID_B32` / `DS_READ_ADDTID_B32` compute `M0[15:0] + offset16 + 4 * lane_id`
/// without touching unused `ADDR`/`DATA1` VGPR fields, and reject `GDS=1`.
#[test]
fn ds_addtid_translates_with_m0_offset_and_lane_id() {
    let mut code = Vec::new();
    code.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0xDEAD_0100)).unwrap());
    // Populate unused ADDR/DATA1 (and DATA0 on read) with nonzero indices to verify they are ignored.
    code.extend(encode("DS_WRITE_ADDTID_B32", "ENC_DS", &[("ADDR", 9), ("DATA0", 2), ("DATA1", 11), ("OFFSET0", 0x40), ("OFFSET1", 0x01)], None).unwrap());
    code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("ADDR", 10), ("DATA0", 12), ("DATA1", 13), ("VDST", 3), ("OFFSET0", 0x40), ("OFFSET1", 0x01)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.unsupported.is_empty(), "{:?}\n{}", t.unsupported, t.hlsl);
    assert!(t.hlsl.contains("groupshared uint xcl_lds"), "{}", t.hlsl);
    assert!(t.hlsl.contains("((m0 & 0xFFFFu) + 320u + 4u * lane_id)"), "{}", t.hlsl);
    for unused in ["uint v9 =", "uint v10 =", "uint v11 =", "uint v12 =", "uint v13 ="] {
        assert!(!t.hlsl.contains(unused), "unexpected {unused} in:\n{}", t.hlsl);
    }

    // GDS=1 must be reported as unsupported and not allocate LDS.
    let mut gds_code = Vec::new();
    gds_code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 3), ("GDS", 1)], None).unwrap());
    gds_code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let tg = translate(&gds_code, &Options::default()).unwrap();
    assert_eq!(tg.unsupported.get("ds_read_addtid_b32"), Some(&1), "{:?}", tg.unsupported);
    assert!(tg.hlsl.contains("GDS"), "{}", tg.hlsl);
    assert!(!tg.hlsl.contains("groupshared uint xcl_lds"), "{}", tg.hlsl);
}

/// `GDS=1` on any LDS-memory DS instruction (not only the ADDTID forms) addresses the global
/// data share, not the group's LDS: it is reported as unsupported and allocates no LDS.
#[test]
fn ds_gds_plain_forms_are_unsupported() {
    for (op, fields) in [
        ("DS_ADD_U32", &[("ADDR", 1), ("DATA0", 2), ("GDS", 1)][..]),
        ("DS_WRITE_B32", &[("ADDR", 1), ("DATA0", 2), ("GDS", 1)][..]),
        ("DS_READ_B32", &[("ADDR", 1), ("VDST", 3), ("GDS", 1)][..]),
    ] {
        let mut code = Vec::new();
        code.extend(encode(op, "ENC_DS", fields, None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let t = translate(&code, &Options::default()).unwrap();
        assert_eq!(t.unsupported.get(op.to_lowercase().as_str()), Some(&1), "{op}: {:?}\n{}", t.unsupported, t.hlsl);
        assert!(t.hlsl.contains("GDS"), "{op}: {}", t.hlsl);
        assert!(!t.hlsl.contains("groupshared uint xcl_lds"), "{op}: {}", t.hlsl);
    }
}

/// GPU verification for `DS_WRITE_ADDTID_B32` / `DS_READ_ADDTID_B32`: verifies exact LDS byte
/// address (`M0[15:0] + ((OFFSET1 << 8) | OFFSET0) + 4 * lane_id`, ignoring `M0[31:16]`) in both
/// directions against explicit `DS_READ_B32` / `DS_WRITE_B32`, plus EXEC masking on writes & reads.
#[test]
fn gpu_ds_addtid_roundtrip_and_exec_masking() {
    if !gpu_ready() {
        return;
    }
    let mut code = Vec::new();
    // m0 = 0xDEAD_0100 -> M0[15:0] = 256, offset = 0x4040 = 16448 -> base byte offset = 16704 + 4 * lane
    code.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0xDEAD_0100)).unwrap());
    // Direction 1: seed LDS via explicit DS_WRITE_B32 at v1 (= 16704 + 4 * lane) and read via DS_READ_ADDTID_B32 -> v7
    code.extend(encode("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 8)], None).unwrap());
    code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 7), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    // Direction 2: all lanes write v2 via DS_WRITE_ADDTID_B32 and verify via explicit DS_READ_B32 -> v4
    code.extend(encode("DS_WRITE_ADDTID_B32", "ENC_DS", &[("DATA0", 2), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    code.extend(encode("DS_READ_B32", "ENC_DS", &[("VDST", 4), ("ADDR", 1)], None).unwrap());
    // Mask EXEC to even lanes (s[0:1] = 0x5555_5555_5555_5555): overwrite LDS with v6 and read into v3
    code.extend(encode("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)], None).unwrap());
    code.extend(encode("DS_WRITE_ADDTID_B32", "ENC_DS", &[("DATA0", 6), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 3), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    // Restore all 64 lanes (EXEC = -1) and read final LDS contents into v5
    code.extend(encode("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None).unwrap());
    code.extend(encode("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 5), ("OFFSET0", 0x40), ("OFFSET1", 0x40)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[0] = 0x5555_5555;
    st.sgpr[1] = 0x5555_5555;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l = l as u32;
        r[1] = 16704 + 4 * l; // explicit byte address (256 + 0x4040 + 4 * lane)
        r[2] = 0x1000 + l;    // ADDTID write value
        r[3] = 0xCAFE_0000 + l; // sentinel preserved on inactive (odd) lanes
        r[6] = 0x2000 + l;    // masked overwrite value (even lanes only)
        r[8] = 0xBEEF_0000 + l; // seeded via DS_WRITE_B32 for ADDTID read check
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let lu = l as u32;
        assert_eq!(r.vgpr[l][7], 0xBEEF_0000 + lu, "lane {l} DS_WRITE_B32 -> DS_READ_ADDTID_B32");
        assert_eq!(r.vgpr[l][4], 0x1000 + lu, "lane {l} DS_WRITE_ADDTID_B32 -> DS_READ_B32");
        if l % 2 == 0 {
            assert_eq!(r.vgpr[l][3], 0x2000 + lu, "even lane {l} masked DS_READ_ADDTID_B32");
            assert_eq!(r.vgpr[l][5], 0x2000 + lu, "even lane {l} final DS_READ_ADDTID_B32");
        } else {
            assert_eq!(r.vgpr[l][3], 0xCAFE_0000 + lu, "odd lane {l} sentinel must be preserved");
            assert_eq!(r.vgpr[l][5], 0x1000 + lu, "odd lane {l} LDS slot must be untouched");
        }
    }
}

/// Ray tracing: image_bvh(64)_intersect_ray on a hand-built BVH (box32 root with 4 children,
/// a box16 node, triangles), several rays; sorted and unsorted box results, triangle hit as
/// (t, denominator, i, j) numerators, misses, and the 64-bit node pointer form.
#[test]
fn gpu_bvh_intersect_ray() {
    if !gpu_ready() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let w = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let (root, box16, tri_a, tri_b) = (5u32, (0x80u32 >> 3) | 4, 0xC0u32 >> 3, 0x100u32 >> 3);
    // root box32 at 0: children + boxes (min xyz, max xyz)
    w(&mut mem, 0, &[tri_a, box16, 0xFFFF_FFFF, tri_b]);
    w(&mut mem, 16, &ff(&[-1.0, -1.0, 5.0, 1.0, 1.0, 6.0])); // triangle A's box: entry t = 5
    w(&mut mem, 40, &ff(&[-1.0, -1.0, 2.0, 1.0, 1.0, 3.0])); // box16 node's box: t = 2
    w(&mut mem, 64, &ff(&[-1.0, -1.0, 0.5, 1.0, 1.0, 0.7])); // invalid child: never reported
    w(&mut mem, 88, &ff(&[10.0, -1.0, 8.0, 11.0, 1.0, 9.0])); // off the ray
    // box16 node at 0x80: one child (triangle B) in z 6..7, half floats
    let h = |x: f32| half_bits(x);
    w(&mut mem, 0x80, &[tri_b, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF]);
    w(&mut mem, 0x90, &[h(-1.0) | h(-1.0) << 16, h(6.0) | h(1.0) << 16, h(1.0) | h(7.0) << 16]);
    // triangles: A in z = 5 (hit at u = v = 0.5 from the origin), B in z = 6.5
    w(&mut mem, 0xC0, &ff(&[-1.0, -1.0, 5.0, 1.0, -1.0, 5.0, -1.0, 1.0, 5.0]));
    w(&mut mem, 0x100, &ff(&[-1.0, -1.0, 6.5, 1.0, -1.0, 6.5, -1.0, 1.0, 6.5]));
    // triangle id 9: I = vertex 1, J = vertex 2 (no rotation)
    w(&mut mem, 0xC0 + 60, &[9]);
    w(&mut mem, 0x100 + 60, &[9]);
    let mimg = |n: &'static str, vdata: u64, vaddr: u64, srsrc: u64| encode(n, "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", srsrc), ("DMASK", 15)], None).unwrap();
    let mut code = Vec::new();
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 100, 0, 2)); // root, sorting
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 104, 0, 3)); // root, no sorting
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 20, 2)); // box16 node
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 112, 40, 2)); // triangle A
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 116, 60, 2)); // triangle A, 64-bit node id
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    st.sgpr[12..16].copy_from_slice(&[(base >> 8) as u32, 0, u32::MAX, 0]);
    // lane 0: ray along +z from the origin; lane 1: the same with t max 3.5; lane 2: from (5, 5)
    let ray = |l: usize| -> ([f32; 3], f32) {
        match l % 3 {
            0 => ([0.0, 0.0, 0.0], 100.0),
            1 => ([0.0, 0.0, 0.0], 3.5),
            _ => ([5.0, 5.0, 0.0], 100.0),
        }
    };
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (o, tmax) = ray(l);
        let rest = |r: &mut Vec<u32>, at: usize| {
            let v = [tmax, o[0], o[1], o[2], 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
            for (k, x) in v.iter().enumerate() {
                r[at + k] = x.to_bits();
            }
        };
        r[0] = root;
        rest(r, 1);
        r[20] = box16;
        rest(r, 21);
        r[40] = tri_a;
        rest(r, 41);
        r[60] = tri_a;
        r[61] = 0;
        rest(r, 62);
    }
    let res = runner::run_code(&code, &st, &mem).unwrap();
    const F: u32 = 0xFFFF_FFFF;
    for l in 0..64usize {
        let v = &res.vgpr[l];
        let q = |at: usize| [v[at], v[at + 1], v[at + 2], v[at + 3]];
        match l % 3 {
            0 => {
                assert_eq!(q(100), [box16, tri_a, F, F], "lane {l}: box32, closest first");
                assert_eq!(q(104), [tri_a, box16, F, F], "lane {l}: box32 without sorting keeps child order");
                assert_eq!(q(108), [tri_b, F, F, F], "lane {l}: box16");
            }
            1 => {
                assert_eq!(q(100), [box16, F, F, F], "lane {l}: t max 3.5 excludes the box at t = 5");
                assert_eq!(q(108), [F, F, F, F], "lane {l}: box16 at t = 6 beyond t max");
            }
            _ => {
                assert_eq!(q(100), [F, F, F, F], "lane {l}: ray from (5, 5) misses every box");
                assert_eq!(q(112)[0], f32::INFINITY.to_bits(), "lane {l}: triangle missed → t = inf");
            }
        }
        if l % 3 != 2 {
            let tri = q(112).map(f32::from_bits);
            let (t, i, j) = (tri[0] / tri[1], tri[2] / tri[1], tri[3] / tri[1]);
            assert_eq!((t, i, j), (5.0, 0.5, 0.5), "lane {l}: triangle hit (t, i, j)");
            assert_eq!(q(116), q(112), "lane {l}: 64-bit node pointer gives the same");
        }
    }
}

/// Checks the compute ABI read from the FSSM stage descriptor (+0x38 COMPUTE_PGM_RSRC2,
/// +0x3C.. numthreads) on every CS of the pack against PSV0 and the SGPRs/VGPRs the code reads.
#[test]
#[ignore]
fn compute_abi_survey() {
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let data = std::fs::read(path).unwrap();
    let (mut n, mut threads_ok, mut sgpr_ok, mut vgpr_ok) = (0, 0, 0, 0);
    for r in crate::gpu::pack::scan(&data, 0x40) {
        // compute records are recognised by their hardware stage below, not by name
        let Some(blob) = crate::gpu::pack::inflate(&data, &r) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        let Some(f) = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6)) else { continue };
        let d = f.stages[0].offset;
        let rd = |o: usize| u32::from_le_bytes(f.data[d + o..d + o + 4].try_into().unwrap());
        let rsrc2 = rd(0x38);
        let nt = [rd(0x3C), rd(0x40), rd(0x44)];
        let user = (rsrc2 >> 1) & 31;
        let tg = [(rsrc2 >> 7) & 1, (rsrc2 >> 8) & 1, (rsrc2 >> 9) & 1, (rsrc2 >> 10) & 1];
        let tid = (rsrc2 >> 11) & 3;
        let sys = tg.iter().sum::<u32>();
        n += 1;
        let psv = crate::gpu::dxbc::psv(&blob).and_then(|p| p.numthreads);
        threads_ok += (psv == Some(nt)) as usize;
        // registers read before written
        let p = f.programs[0];
        let code = &f.data[p.offset..p.offset + p.len];
        let (mut sr, mut sw, mut vr, mut vw) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new(), std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
        let mut off = 0;
        while off + 4 <= code.len() {
            let Ok(dd) = crate::gpu::rdna2::decode(&code[off..], off as u64) else { break };
            off += dd.len;
            let (r, w) = crate::gpu::rdna2::sgpr_uses(&dd);
            sr.extend(r.into_iter().filter(|x| !sw.contains(x) && *x < 106));
            sw.extend(w);
            let (r, w) = crate::gpu::rdna2::vgpr_uses(&dd);
            vr.extend(r.into_iter().filter(|x| !vw.contains(x)));
            vw.extend(w);
        }
        let max_s = sr.iter().max().copied().unwrap_or(0);
        let s_in = max_s < (user + sys) as u64;
        sgpr_ok += s_in as usize;
        let v_in = vr.iter().all(|&v| v <= tid as u64);
        vgpr_ok += v_in as usize;
        if !(s_in && v_in && psv == Some(nt)) {
            println!("{}: rsrc2 {rsrc2:#x} user {user} tgid {tg:?} tid {tid} threads {nt:?} psv {psv:?} SGPRs read {sr:?} VGPR {vr:?}", r.name);
        }
    }
    println!("CS: {n}, numthreads = PSV0: {threads_ok}, SGPRs within user+system: {sgpr_ok}, VGPRs within the thread ID: {vgpr_ok}");
}

/// Compute mode with the console dispatch ABI: numthreads 16×8 (two wave64 waves per group),
/// 2×3 groups; group ids from the SGPRs after the user data, thread ids in v0/v1; LDS shared
/// across the two waves through a barrier.
#[test]
fn gpu_compute_mode_dispatch_abi() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    let cs = CsInterface { threads: [16, 8, 1], user_sgprs: 2, tgid: [true, true, false], tg_size: false, tid_components: 2 };
    let p = prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 132), ("VSRC1", 1)], None), // y · 16
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 2), ("VSRC1", 0)], None), // index in group
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 2), ("SSRC1", 132)], None), // group x · 16
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 4), ("VSRC1", 0)], None), // global x
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 3), ("SSRC1", 131)], None), // group y · 8
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 5), ("VSRC1", 1)], None), // global y
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 144), ("VSRC1", 4)], None),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 5), ("VSRC1", 3)], None), // x | y << 16
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 2)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 6), ("DATA0", 5)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 2)], Some(127)), // partner in the other wave
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 130), ("VSRC1", 7)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 7)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 5)], None),
        ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 4), ("SRC1", 1)], None), // y · width
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", V + 9), ("VSRC1", 3)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 131), ("VSRC1", 9)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 0), ("VSRC1", 9)], None), // + output base
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 9), ("DATA", 11), ("SADDR", 0x7D)], None),
    ]);
    let opt = on_gpu(Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() });
    let t = translate(&p, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = direct_or(&p, &opt, dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}")));
    let (w, h, base) = (32usize, 24usize, 0x100usize);
    let mut io = vec![0u32; 128];
    io[0] = base as u32;
    io[1] = w as u32;
    let out = runner::dispatch(&dxil, [2, 3, 1], &io, &vec![0u32; (base + w * h * 8) / 4]).unwrap();
    for y in 0..h {
        for x in 0..w {
            let at = (base + (y * w + x) * 8) / 4;
            assert_eq!(out[at], (x | y << 16) as u32, "({x}, {y}): own value");
            let (gx, gy, i) = (x / 16, y / 8, (y % 8) * 16 + x % 16);
            let (px, py) = (gx * 16 + (127 - i) % 16, gy * 8 + (127 - i) / 16);
            assert_eq!(out[at + 1], (px | py << 16) as u32, "({x}, {y}): value of thread {} through LDS", 127 - i);
        }
    }
}

/// A wave32 program in a 64-lane hardware wave (what VS / PS may get from the driver): the
/// two halves behave as separate waves – ballots, lane counts, readlane and mbcnt per half.
#[test]
fn gpu_wave32_program_in_wave64_hardware() {
    if !gpu_ready() || !wave64_hardware() {
        return;
    }
    let p = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 138), ("VSRC1", 0)], None), // vcc_lo = lanes with v0 < 10
        ("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 131)], None), // s1 = v0 of lane 3
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 1)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 193), ("SRC1", 128)], None),
    ]);
    let t = translate(&p, &Options { mode: Mode::Kernel, wave64: false, hw_wave: 64, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("[WaveSize(64)]") && t.hlsl.contains("xcl_half"), "{}", t.hlsl);
    let dxil = direct_or(&p, &Options { mode: Mode::Kernel, wave64: false, hw_wave: 64, ..Default::default() }, super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap());
    let mut st = KernelState::default();
    st.sgpr[126] = u32::MAX; // exec of each half
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = (l % 32 + if l >= 32 { 5 } else { 0 }) as u32; // second half shifted by 5
    }
    let r = runner::run(&dxil, &st, &[]).unwrap();
    for l in 0..64 {
        let half = l / 32;
        let v = &r.vgpr[l];
        assert_eq!(v[1], if half == 0 { 10 } else { 5 }, "lane {l}: active lanes of its own half");
        assert_eq!(v[2], 3 + 5 * half as u32, "lane {l}: readlane 3 of its own half");
        assert_eq!(v[3], (l % 32) as u32, "lane {l}: lane index within the half");
    }
}

/// SV_VertexID / SV_InstanceID listed in ISG1 with a register arrive like attributes,
/// v[4 + 4r] (retail VInstanced* shaders read the instance id in v8 = register 1).
#[test]
fn vs_system_values_in_isg1_registers() {
    use super::VsInterface;
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: if sv == 0 { 3 } else { 1 },
        register: reg,
        mask: if sv == 0 { 7 } else { 1 },
        rw_mask: 1,
        stream: 0,
    };
    let vs = VsInterface {
        inputs: vec![sig("POSITION", 0, 0), sig("SV_InstanceID", 8, 1), sig("SV_VertexID", 6, 2)],
        outputs: vec![sig("SV_Position", 1, 0)],
        attr_base: 4,
        params: None,
    };
    let code = prog(&[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 20), ("SRC0", V + 8), ("VSRC1", 12)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 21), ("SRC0", V + 4), ("VSRC1", 5)], None),
    ]);
    let t = translate(&code, &Options { mode: Mode::Vertex(vs), wave64: true, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("uint v8 = xi.xcl_iid;") && t.hlsl.contains("uint v12 = xi.xcl_vid;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint v4 = asuint(xi.a0[0]);"), "{}", t.hlsl);
}

#[test]
#[ignore]
fn probe_standard_swizzle_support() {
    use windows::Win32::Graphics::Direct3D12::*;
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0, &mut dev).unwrap();
        let dev = dev.unwrap();
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut _, std::mem::size_of_val(&o) as u32).unwrap();
        println!("StandardSwizzle64KBSupported = {:?}", o.StandardSwizzle64KBSupported);
    }
}

/// A BVH from the builder (gpu::bvh) through the translated image_bvh_intersect_ray: every
/// lane tests one built node (box32 or triangle) with its own ray; the GPU must agree with
/// the CPU node test the builder's traversal uses.
#[test]
fn gpu_built_bvh_nodes_match_the_cpu_node_test() {
    use crate::gpu::bvh;
    if !gpu_ready() {
        return;
    }
    let tris = bvh::tests::random_tris(40, 5);
    let b = bvh::build(&tris, 0);
    // every node id, root first
    let mut nodes = vec![];
    let mut todo = vec![b.root];
    while let Some(n) = todo.pop() {
        nodes.push(n);
        if n & 7 == bvh::BOX32 {
            let at = ((n & !7) as usize) << 3;
            for k in 0..4 {
                let c = u32::from_le_bytes(b.data[at + 4 * k..at + 4 * k + 4].try_into().unwrap());
                if c != bvh::INVALID {
                    todo.push(c);
                }
            }
        }
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; (base + b.data.len()) / 4 + 64];
    for (k, c) in b.data.chunks(4).enumerate() {
        mem[base / 4 + k] = u32::from_le_bytes(c.try_into().unwrap());
    }
    let mut code = encode("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", 100), ("VADDR", 0), ("SRSRC", 2), ("DMASK", 15)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    let mut r = bvh::tests::rng(17);
    let mut rays = vec![];
    for (l, v) in st.vgpr.iter_mut().enumerate() {
        let node = nodes[l % nodes.len()];
        let o = [r(-12.0, 12.0), r(-12.0, 12.0), r(-12.0, 12.0)];
        let to = [r(-6.0, 6.0), r(-6.0, 6.0), r(-6.0, 6.0)];
        let d = [to[0] - o[0], to[1] - o[1], to[2] - o[2]];
        let inv = d.map(|x| 1.0 / x);
        v[0] = node;
        for (k, x) in [100.0f32, o[0], o[1], o[2], d[0], d[1], d[2], inv[0], inv[1], inv[2]].iter().enumerate() {
            v[1 + k] = x.to_bits();
        }
        rays.push((node, o, d, inv));
    }
    let res = runner::run_code(&code, &st, &mem).unwrap();
    let (mut boxes, mut tri_hits) = (0, 0);
    for (l, &(node, o, d, inv)) in rays.iter().enumerate() {
        let gpu: [u32; 4] = res.vgpr[l][100..104].try_into().unwrap();
        let cpu = bvh::intersect_node(&b.data, node, 100.0, o, d, inv, true);
        if node & 7 == bvh::BOX32 {
            assert_eq!(gpu, cpu, "lane {l}: box node {node:#x}");
            boxes += 1;
        } else {
            let (g, c) = (gpu.map(f32::from_bits), cpu.map(f32::from_bits));
            assert_eq!(g[0].is_infinite(), c[0].is_infinite(), "lane {l}: triangle {node:#x} hit / miss");
            if !c[0].is_infinite() {
                assert!((g[0] / g[1] - c[0] / c[1]).abs() < 1e-4, "lane {l}: t {g:?} vs {c:?}");
                tri_hits += 1;
            }
        }
    }
    assert!(boxes > 10 && tri_hits > 0, "boxes {boxes}, triangle hits {tri_hits}");
}

/// The console's node forms (seen in the console builder's output): a box16 node and a GPURT
/// triangle pair – type 0 = (v0, v1, v2), type 1 = (v1, v3, v2), barycentrics put back in the
/// original vertex order by the triangle id. GPU (HLSL emulation) = CPU mirror, and the
/// barycentrics of a known point come out right.
#[test]
fn gpu_triangle_pair_and_box16_nodes() {
    use crate::gpu::bvh;
    if !gpu_ready() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let w = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    // corners of a quad in z = 5; triangle 0 = (p0, p1, p2) stored rotated: v0 = p1, v1 = p2,
    // v2 = p0, so its I barycentric (weight of p1) is vertex 0 and J (p2) vertex 1
    let (p0, p1, p2, p3) = ([-1.0f32, -1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, 1.0, 5.0], [1.0, 1.0, 5.0]);
    let pair = 0x40usize;
    w(&mut mem, pair, &ff(&[p1, p2, p0, p3].concat()));
    // triangle 1 = (v1, v3, v2) = (p2, p3, p0), original order (p0, p3, p2): I (p3) is vertex 1
    // of the triple, J (p2) vertex 0. Triangle id: byte 0 = 0x04, byte 1 = 0x01
    w(&mut mem, pair + 48, &[0, 11, 12, 0x0104]);
    // box16 root with the two triangles
    let h = |x: f32| half_bits(x);
    let (t0, t1) = ((pair as u32) >> 3, (pair as u32) >> 3 | 1);
    w(&mut mem, 0, &[t0, t1, 0xFFFF_FFFF, 0xFFFF_FFFF]);
    for k in 0..2 {
        w(&mut mem, 16 + 12 * k, &[h(-1.0) | h(-1.0) << 16, h(4.5) | h(1.0) << 16, h(1.0) | h(5.5) << 16]);
    }
    let root = 4u32; // box16 at 0
    let mut code = encode("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", 100), ("VADDR", 0), ("SRSRC", 2), ("DMASK", 15)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    let mut r = bvh::tests::rng(5);
    let nodes = [root, t0, t1];
    let mut rays = vec![];
    for (l, v) in st.vgpr.iter_mut().enumerate() {
        let node = nodes[l % 3];
        let o = [r(-1.5, 1.5), r(-1.5, 1.5), 0.0];
        let d = [r(-0.1, 0.1), r(-0.1, 0.1), 1.0];
        let inv = d.map(|x| 1.0 / x);
        v[0] = node;
        for (k, x) in [100.0f32, o[0], o[1], o[2], d[0], d[1], d[2], inv[0], inv[1], inv[2]].iter().enumerate() {
            v[1 + k] = x.to_bits();
        }
        rays.push((node, o, d, inv));
    }
    let res = runner::run_code(&code, &st, &mem).unwrap();
    let data: Vec<u8> = mem[base / 4..].iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut checked = 0;
    for (l, &(node, o, d, inv)) in rays.iter().enumerate() {
        let gpu: [u32; 4] = res.vgpr[l][100..104].try_into().unwrap();
        let cpu = bvh::intersect_node(&data, node, 100.0, o, d, inv, true);
        if node == root {
            assert_eq!(gpu, cpu, "lane {l}: box16");
            continue;
        }
        let (g, c) = (gpu.map(f32::from_bits), cpu.map(f32::from_bits));
        assert_eq!(g[0].is_infinite(), c[0].is_infinite(), "lane {l}: hit / miss");
        if c[0].is_infinite() {
            continue;
        }
        for k in 0..4 {
            assert!((g[k] / g[1] - c[k] / c[1]).abs() < 1e-5, "lane {l}: {g:?} vs {c:?}");
        }
        // hit point from the barycentrics in the ORIGINAL order of the stored triangle
        let (t, bi, bj) = (c[0] / c[1], c[2] / c[1], c[3] / c[1]);
        let hit = [0, 1, 2].map(|a| o[a] + t * d[a]);
        if node == t0 {
            let q = [0, 1, 2].map(|a| (1.0 - bi - bj) * p0[a] + bi * p1[a] + bj * p2[a]);
            assert!((0..3).all(|a| (q[a] - hit[a]).abs() < 1e-4), "lane {l}: triangle 0 barycentrics {q:?} vs {hit:?}");
        } else {
            let q = [0, 1, 2].map(|a| (1.0 - bi - bj) * p0[a] + bi * p3[a] + bj * p2[a]);
            assert!((0..3).all(|a| (q[a] - hit[a]).abs() < 1e-4), "lane {l}: triangle 1 barycentrics {q:?} vs {hit:?}");
        }
        checked += 1;
    }
    assert!(checked > 5, "{checked} triangle hits");
}

/// Multi-window `image_bvh64_intersect_ray` (full 64-bit node pointer into UMA window 1 with
/// base = 0, plus 48-bit base in window 1) and axis-aligned grazing ray / NaN-disabled box.
#[test]
fn gpu_bvh64_intersect_ray_across_memory_windows_and_grazing() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let write_words = |addr: u64, vals: &[u32]| unsafe {
        let p = addr as *mut u32;
        for (k, &v) in vals.iter().enumerate() {
            *p.add(k) = v;
        }
    };
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let base_off = 0x1000u64;
    let bvh_addr = w1 + base_off;
    let tri_rel = 0x80u32 >> 3;
    let nan_child = 0xC0u32 >> 3;
    let inv_child = 0x100u32 >> 3;
    // Window 1 root box32 at bvh_addr: child 0 = normal box [-1..1, -1..1, 5..6],
    // child 1 = NaN-disabled box, child 2 = inverted box (mn > mx), child 3 = invalid
    write_words(bvh_addr, &[tri_rel, nan_child, inv_child, u32::MAX]);
    write_words(bvh_addr + 16, &ff(&[-1.0, -1.0, 5.0, 1.0, 1.0, 6.0]));
    write_words(bvh_addr + 40, &ff(&[f32::NAN, -1.0, 2.0, 1.0, 1.0, 3.0]));
    write_words(bvh_addr + 64, &ff(&[2.0, -1.0, 2.0, -2.0, 1.0, 3.0]));
    // Triangle at bvh_addr + 0x80 in z = 5
    write_words(bvh_addr + 0x80, &ff(&[-1.0, -1.0, 5.0, 1.0, -1.0, 5.0, -1.0, 1.0, 5.0]));
    write_words(bvh_addr + 0x80 + 60, &[9]);
    // Poison window 0 at the same low offset: reading window 0 would return bogus child 0xDEAD_BEEF
    write_words(w0 + base_off, &[0xDEAD_BEEF, u32::MAX, u32::MAX, u32::MAX]);
    write_words(w0 + base_off + 16, &ff(&[-100.0, -100.0, 1.0, 100.0, 100.0, 2.0]));

    let mimg = |n: &'static str, vdata: u64, vaddr: u64, srsrc: u64| encode(n, "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", srsrc), ("DMASK", 15)], None).unwrap();
    let mut code = Vec::new();
    // s[8:11] = base 0 (box sort enabled), v[0:1] = full 64-bit node pointer into window 1
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 100, 0, 2));
    // s[8:11] = base 0, v[20:21] = full 64-bit triangle node pointer in window 1
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 104, 20, 2));
    // s[12:15] = 48-bit base pointing at window 1 (`bvh_words(bvh_addr, true)`), v40 = 32-bit root
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 40, 3));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let t = translate(&code, &on_gpu(Options { mem_windows: 2, ..Default::default() })).unwrap();
    let dxil = direct_or(&code, &on_gpu(Options { mem_windows: 2, ..Default::default() }), super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap());

    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&crate::gpu::desc::bvh_words(0, true));
    st.sgpr[12..16].copy_from_slice(&crate::gpu::desc::bvh_words(bvh_addr, true));
    let root64 = (bvh_addr >> 3) | 5;
    let tri64 = (bvh_addr + 0x80) >> 3;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        // Even lanes graze the box edge (o.x = -1.0 == mn.x, dx = 0, inv.x = +inf);
        // odd lanes fire through the center (0, 0, 0) along +z.
        let ox = if l % 2 == 0 { -1.0f32 } else { 0.0f32 };
        let ray = [100.0f32, ox, 0.0, 0.0, 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
        let put_ray = |r: &mut Vec<u32>, at: usize| {
            for (k, x) in ray.iter().enumerate() {
                r[at + k] = x.to_bits();
            }
        };
        r[0] = root64 as u32;
        r[1] = (root64 >> 32) as u32;
        put_ray(r, 2);
        r[20] = tri64 as u32;
        r[21] = (tri64 >> 32) as u32;
        put_ray(r, 22);
        r[40] = 5; // relative root box32 at bvh_addr
        put_ray(r, 41);
    }
    let res = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    const F: u32 = u32::MAX;
    for l in 0..64usize {
        let v = &res.vgpr[l];
        let q = |at: usize| [v[at], v[at + 1], v[at + 2], v[at + 3]];
        assert_eq!(q(100), [tri_rel, F, F, F], "lane {l}: 64-bit box32 in window 1 (grazing/NaN/inverted)");
        assert_eq!(q(108), [tri_rel, F, F, F], "lane {l}: 48-bit base box32 in window 1");
        let tri = q(104).map(f32::from_bits);
        assert!(!tri[0].is_infinite() && ((tri[0] / tri[1]) - 5.0).abs() < 1e-4, "lane {l}: 64-bit triangle in window 1 → {tri:?}");
    }
}

/// Instructions first met in XboxBvhBuilder.dll's shaders: global atomics (float min / max by
/// a CAS loop), 8 / 16-bit buffer loads / stores and 8 / 16-bit LDS accesses – neighbouring
/// lanes share dwords, so a non-atomic read-modify-write would lose writes.
#[test]
fn gpu_global_atomics_and_sub_dword_access() {
    if !gpu_ready() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mubuf = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_MUBUF", f, None).unwrap();
    let ds = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_DS", f, None).unwrap();
    let mut code = Vec::new();
    // global atomics on s[2:3] + v30 (= 0 for every lane) + offset
    code.extend(glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 40), ("ADDR", 30), ("DATA", 31), ("SADDR", 2), ("OFFSET", 0), ("GLC", 1)]));
    code.extend(glbl("GLOBAL_ATOMIC_SUB", &[("VDST", 41), ("ADDR", 30), ("DATA", 35), ("SADDR", 2), ("OFFSET", 4)]));
    code.extend(glbl("GLOBAL_ATOMIC_FMAX", &[("VDST", 42), ("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_ATOMIC_FMIN", &[("VDST", 43), ("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", 12)]));
    code.extend(glbl("GLOBAL_ATOMIC_SMAX", &[("VDST", 44), ("ADDR", 30), ("DATA", 33), ("SADDR", 2), ("OFFSET", 16)]));
    // V# s[24:27]: raw at 0x700; short at lane*2, byte at 0x80 + lane (offen)
    code.extend(mubuf("BUFFER_STORE_SHORT", &[("VDATA", 34), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_STORE_BYTE", &[("VDATA", 36), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_USHORT", &[("VDATA", 45), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_SSHORT", &[("VDATA", 46), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_UBYTE", &[("VDATA", 47), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_SBYTE", &[("VDATA", 48), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    // LDS: 16-bit at lane*2, 8-bit at 256 + lane; after the barrier read the neighbour's (lane ^ 1)
    code.extend(ds("DS_WRITE_B16", &[("ADDR", 2), ("DATA0", 34)]));
    code.extend(ds("DS_WRITE_B8", &[("ADDR", 4), ("DATA0", 36)]));
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(ds("DS_READ_U16", &[("VDST", 49), ("ADDR", 5)]));
    code.extend(ds("DS_READ_I16", &[("VDST", 50), ("ADDR", 5)]));
    code.extend(ds("DS_READ_U8", &[("VDST", 51), ("ADDR", 6)]));
    code.extend(ds("DS_READ_I8", &[("VDST", 52), ("ADDR", 6)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[24..28].copy_from_slice(&[0x700, 0, 0x200, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[2] = 2 * l32;
        r[3] = 0x80 + l32;
        r[4] = 256 + l32;
        r[5] = 2 * (l32 ^ 1);
        r[6] = 256 + (l32 ^ 1);
        r[30] = 0;
        r[31] = l32 + 1;
        r[32] = f(l as f32 - 20.5);
        r[33] = (l as i32 - 30) as u32;
        r[34] = 0xAB00 + l32;
        r[35] = 1;
        r[36] = 0x90 + l32;
    }
    let mut mem = vec![0u32; 1024];
    mem[0x604 / 4] = 1000;
    mem[0x608 / 4] = f32::NEG_INFINITY.to_bits();
    mem[0x60C / 4] = f32::INFINITY.to_bits();
    mem[0x610 / 4] = (-100i32) as u32;
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!(r.mem[0x600 / 4], 64 * 65 / 2, "global_atomic_add");
    let mut olds: Vec<u32> = r.vgpr.iter().map(|v| v[40]).collect();
    olds.sort_unstable();
    olds.dedup();
    assert_eq!(olds.len(), 64, "global_atomic_add with GLC returns 64 distinct old values");
    assert_eq!(r.mem[0x604 / 4], 1000 - 64, "global_atomic_sub");
    assert_eq!(f32::from_bits(r.mem[0x608 / 4]), 42.5, "global_atomic_fmax");
    assert_eq!(f32::from_bits(r.mem[0x60C / 4]), -20.5, "global_atomic_fmin");
    assert_eq!(r.mem[0x610 / 4] as i32, 33, "global_atomic_smax");
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let (s, b) = (0xAB00 + l as u32, 0x90 + l as u32);
        let n = l ^ 1;
        assert_eq!((r.mem[(0x700 + 2 * l) / 4] >> (16 * (l % 2))) & 0xFFFF, s, "lane {l}: buffer_store_short");
        assert_eq!((r.mem[(0x780 + l) / 4] >> (8 * (l % 4))) & 0xFF, b, "lane {l}: buffer_store_byte");
        assert_eq!((v[45], v[46]), (s, s | 0xFFFF_0000), "lane {l}: buffer_load_ushort / sshort");
        assert_eq!((v[47], v[48]), (b, b | 0xFFFF_FF00), "lane {l}: buffer_load_ubyte / sbyte");
        let (sn, bn) = (0xAB00 + n as u32, 0x90 + n as u32);
        assert_eq!((v[49], v[50]), (sn, sn | 0xFFFF_0000), "lane {l}: ds_read_u16 / i16 of the neighbour");
        assert_eq!((v[51], v[52]), (bn, bn | 0xFFFF_FF00), "lane {l}: ds_read_u8 / i8 of the neighbour");
    }
}

/// The instructions added for code outside the retail pack: global inc / dec / csub /
/// 64-bit add, LDS signed min / max, cmpst, mskor, float add (CAS loops), global sub-dword
/// and d16 accesses, scratch, SAD / lerp, the division helpers, saturating adds, packs, swap.
#[test]
fn gpu_more_atomics_sub_dword_scratch_valu() {
    if !gpu_ready() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let glbl = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_FLAT_GLBL", f);
    let ds = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_DS", f);
    let vop3 = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_VOP3", f);
    let mut code = Vec::new();
    // global atomics at 0x600 + offset (s[2:3] + v30 = 0)
    code.extend(glbl("GLOBAL_ATOMIC_INC", &[("ADDR", 30), ("DATA", 21), ("SADDR", 2), ("OFFSET", 0)]));
    code.extend(glbl("GLOBAL_ATOMIC_DEC", &[("ADDR", 30), ("DATA", 21), ("SADDR", 2), ("OFFSET", 4)]));
    code.extend(glbl("GLOBAL_ATOMIC_CSUB", &[("ADDR", 30), ("DATA", 35), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_ATOMIC_ADD_X2", &[("ADDR", 30), ("DATA", 38), ("SADDR", 2), ("OFFSET", 16)]));
    code.extend(glbl("GLOBAL_ATOMIC_INC_X2", &[("ADDR", 30), ("DATA", 90), ("SADDR", 2), ("OFFSET", 24)]));
    code.extend(glbl("GLOBAL_ATOMIC_DEC_X2", &[("ADDR", 30), ("DATA", 90), ("SADDR", 2), ("OFFSET", 32)]));
    // LDS: initial values, then the atomics, then read back
    for (a, v) in [(7, 8), (17, 9), (18, 10), (19, 12), (20, 15)] {
        code.extend(ds("DS_WRITE_B32", &[("ADDR", a), ("DATA0", v)]));
    }
    code.extend(enc("S_BARRIER", "ENC_SOPP", &[]));
    code.extend(ds("DS_MAX_I32", &[("ADDR", 7), ("DATA0", 33)]));
    code.extend(ds("DS_MIN_I32", &[("ADDR", 17), ("DATA0", 33)]));
    code.extend(ds("DS_CMPST_RTN_B32", &[("VDST", 53), ("ADDR", 18), ("DATA0", 10), ("DATA1", 11)]));
    code.extend(ds("DS_MSKOR_B32", &[("ADDR", 19), ("DATA0", 13), ("DATA1", 14)]));
    code.extend(ds("DS_ADD_F32", &[("ADDR", 20), ("DATA0", 16)]));
    code.extend(enc("S_BARRIER", "ENC_SOPP", &[]));
    for (k, a) in [7u64, 17, 18, 19, 20].iter().enumerate() {
        code.extend(ds("DS_READ_B32", &[("VDST", 54 + k as u64), ("ADDR", *a)]));
    }
    // global bytes at 0x780 + lane, shorts at 0x700 + 2 · lane (s[2:3] + v + 0x100 / 0x180)
    code.extend(glbl("GLOBAL_STORE_BYTE", &[("ADDR", 3), ("DATA", 36), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_UBYTE", &[("VDST", 59), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_SBYTE", &[("VDST", 60), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_UBYTE_D16_HI", &[("VDST", 61), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_STORE_SHORT_D16_HI", &[("ADDR", 2), ("DATA", 62), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_USHORT", &[("VDST", 63), ("ADDR", 2), ("SADDR", 2), ("OFFSET", 0x100)]));
    // scratch (per lane): a dword at 2 · lane + 16, a byte at 2 · lane + 40
    code.extend(enc("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 2), ("DATA", 11), ("SADDR", 0x7F), ("OFFSET", 16)]));
    code.extend(enc("SCRATCH_STORE_BYTE", "ENC_FLAT_SCRATCH", &[("ADDR", 2), ("DATA", 36), ("SADDR", 0x7F), ("OFFSET", 40)]));
    code.extend(enc("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", &[("VDST", 64), ("ADDR", 2), ("SADDR", 0x7F), ("OFFSET", 16)]));
    code.extend(enc("SCRATCH_LOAD_UBYTE", "ENC_FLAT_SCRATCH", &[("VDST", 65), ("ADDR", 2), ("SADDR", 0x7F), ("OFFSET", 40)]));
    // VALU
    let vv = |n: u64| V + n;
    code.extend(vop3("V_SAD_U8", &[("VDST", 66), ("SRC0", vv(22)), ("SRC1", vv(23)), ("SRC2", vv(24))]));
    code.extend(vop3("V_MSAD_U8", &[("VDST", 67), ("SRC0", vv(22)), ("SRC1", vv(25)), ("SRC2", vv(24))]));
    code.extend(vop3("V_LERP_U8", &[("VDST", 68), ("SRC0", vv(22)), ("SRC1", vv(23)), ("SRC2", vv(26))]));
    code.extend(vop3("V_DIV_FIXUP_F32", &[("VDST", 69), ("SRC0", vv(37)), ("SRC1", vv(28)), ("SRC2", vv(27))]));
    code.extend(vop3("V_DIV_FIXUP_F32", &[("VDST", 70), ("SRC0", vv(37)), ("SRC1", 128), ("SRC2", vv(27))]));
    code.extend(vop3("V_DIV_FMAS_F32", &[("VDST", 71), ("SRC0", vv(28)), ("SRC1", vv(27)), ("SRC2", vv(28))]));
    code.extend(enc("V_DIV_SCALE_F32", "VOP3_SDST_ENC", &[("VDST", 72), ("SDST", 106), ("SRC0", vv(27)), ("SRC1", vv(28)), ("SRC2", vv(27))]));
    code.extend(vop3("V_ADD_NC_I32", &[("VDST", 73), ("SRC0", vv(29)), ("SRC1", vv(35)), ("CLAMP", 1)]));
    code.extend(vop3("V_CVT_PK_U16_U32", &[("VDST", 74), ("SRC0", vv(76)), ("SRC1", vv(24))]));
    code.extend(vop3("V_MAD_U32_U16", &[("VDST", 75), ("SRC0", vv(77)), ("SRC1", vv(78)), ("SRC2", vv(35))]));
    code.extend(vop3("V_CVT_PKNORM_U16_F32", &[("VDST", 79), ("SRC0", 242), ("SRC1", 240)]));
    // normalized f16 conversions: v82 = halves (0.5, +1.0), v83 = (−1.0, −0.5)
    code.extend(vop3("V_CVT_PKNORM_U16_F16", &[("VDST", 84), ("SRC0", vv(82)), ("SRC1", vv(83))]));
    code.extend(vop3("V_CVT_PKNORM_I16_F16", &[("VDST", 85), ("SRC0", vv(82)), ("SRC1", vv(83)), ("OP_SEL", 3)]));
    code.extend(vop3("V_CVT_NORM_U16_F16", &[("VDST", 86), ("SRC0", vv(82))]));
    code.extend(vop3("V_CVT_NORM_I16_F16", &[("VDST", 86), ("SRC0", vv(83)), ("OP_SEL", 9)]));
    code.extend(enc("V_CVT_NORM_U16_F16", "ENC_VOP1", &[("VDST", 87), ("SRC0", 242)]));
    code.extend(enc("V_SWAP_B32", "ENC_VOP1", &[("VDST", 80), ("SRC0", vv(81))]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[2] = 2 * l32;
        r[3] = 0x80 + l32;
        for (k, a) in [(7, 512u32), (17, 516), (18, 520), (19, 524), (20, 528)] {
            r[k] = a;
        }
        r[8] = (-100i32) as u32;
        r[9] = 100;
        r[10] = 7;
        r[11] = l32 + 1;
        r[12] = 0xFF00;
        r[13] = 0xF0F0;
        r[14] = 0x0101;
        r[15] = 0;
        r[16] = f(0.5);
        r[21] = 10;
        r[22] = 0x1020_3040;
        r[23] = 0x4030_2010;
        r[24] = 5;
        r[25] = 0x0030_2010;
        r[26] = 0x0101_0101;
        r[27] = f(7.0);
        r[28] = f(2.0);
        r[29] = 0x7FFF_FFFF;
        r[30] = 0;
        r[33] = (l as i32 - 30) as u32;
        r[35] = 1;
        r[36] = 0x90 + l32;
        r[37] = f(3.5);
        r[38] = 1;
        r[39] = 0;
        r[61] = 0x1234;
        r[62] = 0xBEEF_1111;
        r[76] = 70000;
        r[77] = 0x1_0003;
        r[78] = 0x2_0004;
        r[80] = 1;
        r[81] = 2;
        r[82] = 0x3C00_3800; // f16 halves: lo +0.5, hi +1.0
        r[83] = 0xB800_BC00; // lo −1.0, hi −0.5
        r[90] = 10;
        r[91] = 0;
    }
    let mut mem = vec![0u32; 1024];
    mem[0x604 / 4] = 5;
    mem[0x608 / 4] = 50;
    mem[0x610 / 4] = 0xFFFF_FFF0;
    mem[0x620 / 4] = 5;
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!(r.mem[0x600 / 4], 64 % 11, "global_atomic_inc wraps at 10");
    assert_eq!(r.mem[0x604 / 4], 7, "global_atomic_dec wraps to 10 below 0");
    assert_eq!(r.mem[0x608 / 4], 0, "global_atomic_csub clamps at 0");
    assert_eq!((r.mem[0x610 / 4], r.mem[0x614 / 4]), (0x30, 1), "global_atomic_add_x2 carries into the high dword");
    assert_eq!((r.mem[0x618 / 4], r.mem[0x61C / 4]), (64 % 11, 0), "global_atomic_inc_x2 wraps at 10");
    assert_eq!((r.mem[0x620 / 4], r.mem[0x624 / 4]), (7, 0), "global_atomic_dec_x2 wraps to 10 below 0");
    assert_eq!(r.vgpr.iter().filter(|v| v[53] == 7).count(), 1, "ds_cmpst: one lane finds the compare value");
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let b = 0x90 + l as u32;
        assert_eq!((v[54] as i32, v[55] as i32), (33, -30), "lane {l}: ds_max_i32 / ds_min_i32");
        assert!((1..=64).contains(&v[56]), "lane {l}: ds_cmpst stored a lane's value");
        assert_eq!((v[57], f32::from_bits(v[58])), (0x0F01, 32.0), "lane {l}: ds_mskor_b32 / ds_add_f32");
        assert_eq!((v[59], v[60], v[61]), (b, b | 0xFFFF_FF00, 0x1234 | b << 16), "lane {l}: global ubyte / sbyte / ubyte_d16_hi");
        assert_eq!(v[63], 0xBEEF, "lane {l}: global_store_short_d16_hi + load_ushort");
        assert_eq!((v[64], v[65]), (l as u32 + 1, b), "lane {l}: scratch dword / byte");
        assert_eq!((v[66], v[67], v[68]), (133, 85, 0x2828_2828), "lane {l}: v_sad_u8 / v_msad_u8 / v_lerp_u8");
        assert_eq!((f32::from_bits(v[69]), f32::from_bits(v[70]), f32::from_bits(v[71])), (3.5, f32::INFINITY, 16.0), "lane {l}: div_fixup / fixup(x/0) / div_fmas");
        assert_eq!((f32::from_bits(v[72]), v[73]), (7.0, 0x7FFF_FFFF), "lane {l}: v_div_scale_f32 / v_add_nc_i32 clamp");
        assert_eq!((v[74], v[75], v[79]), (0x0005_FFFF, 13, 0x8000_FFFF), "lane {l}: cvt_pk_u16_u32 / mad_u32_u16 / cvt_pknorm_u16_f32");
        assert_eq!((v[84], v[85], v[86], v[87]), (0x0000_8000, 0xC000_7FFF, 0xC000_8000, 0xFFFF), "lane {l}: cvt norm f16 (unorm/snorm, packed, hi halves)");
        assert_eq!((v[80], v[81]), (2, 1), "lane {l}: v_swap_b32");
    }
    assert_eq!(r.sgpr[106] | r.sgpr[107], 0, "v_div_scale_f32: VCC 0 (no post-scale)");
}

/// *_atomic_fcmpswap_x2 on hardware: IEEE-754 64-bit float compare-swap (XCL-BT-042).
/// `-0.0 == +0.0` swaps (returning the pre-swap zero's sign bits), `NaN == NaN` keeps, and GLC
/// returns the old qword. Per-lane cells are deterministic; the three shared cells end in the
/// unique winner's value.
#[test]
fn gpu_atomic_fcmpswap_x2() {
    if !gpu_ready() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let glbl = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_FLAT_GLBL", f);
    let mut code = Vec::new();
    // buffer: V# s[8:11] = raw base 0x600; per-lane cell at base + voff (OFFEN), voff = 8·lane
    code.extend(enc("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 2), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)]));
    // flat: cell at v14 (0x800 + 8·lane)
    code.extend(enc("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 12), ("ADDR", 14), ("DATA", 16), ("GLC", 1)]));
    // global: cell at s2 + v22 (0xA00 + 8·lane)
    code.extend(glbl("GLOBAL_ATOMIC_FCMPSWAP_X2", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)]));
    // one cell shared by all 64 lanes per family: buffer 0xC00 (V# s[12:15]), flat 0xE00,
    // global s3 + v9 (0x1200)
    code.extend(enc("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 30), ("VADDR", 8), ("SRSRC", 3), ("SOFFSET", 128), ("GLC", 1)]));
    code.extend(enc("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 40), ("ADDR", 10), ("DATA", 34), ("GLC", 1)]));
    code.extend(glbl("GLOBAL_ATOMIC_FCMPSWAP_X2", &[("VDST", 44), ("ADDR", 9), ("DATA", 46), ("SADDR", 3), ("GLC", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0xA00;
    st.sgpr[3] = 0x1200;
    st.sgpr[8..12].copy_from_slice(&[0x600, 0, 0x200, 0]); // V#: base, stride/flags, records, –
    st.sgpr[12..16].copy_from_slice(&[0xC00, 0, 0x200, 0]);
    const INIT: u64 = 0x3FF8_0000_0000_0000; // f64 bits of 1.5
    const NEG_ZERO: u64 = 0x8000_0000_0000_0000; // -0.0
    const QNAN: u64 = 0x7FF8_0000_0000_0000; // qNaN
    // (initial cell, cmp, swaps)
    let lane_case = |l: usize| match l % 4 {
        0 => (INIT, INIT, true),
        1 => (0u64, NEG_ZERO, true),
        2 => (NEG_ZERO, 0u64, true),
        _ => (QNAN, QNAN, false),
    };
    let newv = |tag: u64, l: u64| tag << 48 | l;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l64 = l as u64;
        r[2] = 8 * l as u32; // buffer offset -> cell 0x600 + 8·lane
        r[14] = 0x800 + 8 * l as u32;
        r[22] = 8 * l as u32; // + s2 -> 0xA00 + 8·lane
        r[10] = 0xE00; // flat shared cell (all lanes)
        let (_, cmp, _) = lane_case(l);
        let put = |r: &mut Vec<u32>, at: usize, v: u64| {
            r[at] = v as u32;
            r[at + 1] = (v >> 32) as u32;
        };
        put(r, 4, newv(0xB000, l64)); // buffer: new at [4:5], cmp at [6:7]
        put(r, 6, cmp);
        put(r, 16, newv(0xF000, l64));
        put(r, 18, cmp);
        put(r, 24, newv(0xA000, l64));
        put(r, 26, cmp);
        put(r, 30, newv(0x5B00, l64)); // shared cells: every lane compares against INIT
        put(r, 32, INIT);
        put(r, 34, newv(0x5F00, l64));
        put(r, 36, INIT);
        put(r, 46, newv(0x5A00, l64));
        put(r, 48, INIT);
    }
    let mut mem = vec![0u32; 0x1400 / 4];
    for l in 0..64usize {
        let (b, _, _) = lane_case(l);
        for a in [0x600 + 8 * l, 0x800 + 8 * l, 0xA00 + 8 * l] {
            mem[a / 4] = b as u32;
            mem[a / 4 + 1] = (b >> 32) as u32;
        }
    }
    for a in [0xC00, 0xE00, 0x1200] {
        mem[a / 4] = INIT as u32;
        mem[a / 4 + 1] = (INIT >> 32) as u32;
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let old = |v: &[u32], at: usize| v[at] as u64 | (v[at + 1] as u64) << 32;
    let cell = |a: usize| r.mem[a / 4] as u64 | (r.mem[a / 4 + 1] as u64) << 32;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let l64 = l as u64;
        let (init_cell, cmp, swaps) = lane_case(l);
        let (bs, fs, gs) = if swaps { (newv(0xB000, l64), newv(0xF000, l64), newv(0xA000, l64)) } else { (init_cell, init_cell, init_cell) };
        assert_eq!(cell(0x600 + 8 * l), bs, "lane {l}: buffer cell");
        assert_eq!(cell(0x800 + 8 * l), fs, "lane {l}: flat cell");
        assert_eq!(cell(0xA00 + 8 * l), gs, "lane {l}: global cell");
        // GLC old value: VDATA in place for buffer, VDST for flat / global
        assert_eq!(old(v, 4), init_cell, "lane {l}: buffer GLC old");
        assert_eq!(old(v, 12), init_cell, "lane {l}: flat GLC old");
        assert_eq!(old(v, 20), init_cell, "lane {l}: global GLC old");
        // the compare register half is untouched
        assert_eq!(old(v, 6), cmp, "lane {l}: buffer cmp regs");
    }
    // shared cells: exactly one lane sees INIT (the winner); the cell holds its value
    for (i, (dst, a, tag)) in [(30, 0xC00usize, 0x5B00u64), (40, 0xE00, 0x5F00), (44, 0x1200, 0x5A00)].iter().enumerate() {
        let winners: Vec<usize> = r.vgpr.iter().enumerate().filter(|(_, v)| old(v, *dst) == INIT).map(|(l, _)| l).collect();
        assert_eq!(winners.len(), 1, "shared cell {i}: exactly one lane wins the swap");
        assert_eq!(cell(*a), newv(*tag, winners[0] as u64), "shared cell {i}: the winner's value landed");
    }
}

/// Float compare-and-swap atomics across LDS (`ds_cmpst_rtn_f32`), buffer/global/flat 32-bit and
/// 64-bit (`*_atomic_fcmpswap`, `*_atomic_fcmpswap_x2`), and image UAV (`image_atomic_fcmpswap`)
/// use IEEE-754 float equality (`-0.0 == +0.0` swaps, `qNaN == qNaN` and `sNaN == sNaN` keep,
/// denormal != `+0.0`), while integer `cmpswap` / `ds_cmpst_rtn_b32` controls remain bitwise.
/// Verified on the GPU through both HLSL and Direct DXIL (`XCL-BT-042`).
#[test]
fn gpu_float_cas_uses_ieee_equality() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT;
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut code = Vec::new();
    // Initialize LDS float and int-control cells with init32 (v10)
    code.extend(enc("DS_WRITE_B32", "ENC_DS", &[("ADDR", 0), ("DATA0", 10)]));
    code.extend(enc("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 10)]));
    // 1) LDS: DS_CMPST_RTN_F32 (DATA0=cmp v11, DATA1=new v12) + bitwise DS_CMPST_RTN_B32 control
    code.extend(enc("DS_CMPST_RTN_F32", "ENC_DS", &[("VDST", 22), ("ADDR", 0), ("DATA0", 11), ("DATA1", 12)]));
    code.extend(enc("DS_READ_B32", "ENC_DS", &[("VDST", 23), ("ADDR", 0)]));
    code.extend(enc("DS_CMPST_RTN_B32", "ENC_DS", &[("VDST", 24), ("ADDR", 1), ("DATA0", 11), ("DATA1", 12)]));
    code.extend(enc("DS_READ_B32", "ENC_DS", &[("VDST", 25), ("ADDR", 1)]));
    // 2) 32-bit buffer / global / flat float CAS + global bitwise control
    code.extend(enc("BUFFER_ATOMIC_FCMPSWAP", "ENC_MUBUF", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)]));
    code.extend(enc("GLOBAL_ATOMIC_FCMPSWAP", "ENC_FLAT_GLBL", &[("VDST", 26), ("ADDR", 0), ("DATA", 16), ("SADDR", 4), ("GLC", 1)]));
    code.extend(enc("FLAT_ATOMIC_FCMPSWAP", "ENC_FLAT", &[("VDST", 27), ("ADDR", 2), ("DATA", 16), ("GLC", 1)]));
    code.extend(enc("GLOBAL_ATOMIC_CMPSWAP", "ENC_FLAT_GLBL", &[("VDST", 28), ("ADDR", 0), ("DATA", 16), ("SADDR", 6), ("GLC", 1)]));
    // 3) 32-bit image UAV float CAS + bitwise control (64x1 R32_UINT UAVs)
    code.extend(enc("IMAGE_ATOMIC_FCMPSWAP", "ENC_MIMG", &[("VDATA", 18), ("VADDR", 8), ("SRSRC", 6), ("DMASK", 0x3), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]));
    code.extend(enc("IMAGE_ATOMIC_CMPSWAP", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 8), ("SRSRC", 8), ("DMASK", 0x3), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]));
    // 4) 64-bit buffer / global / flat float CAS + global bitwise control
    code.extend(enc("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 30), ("VADDR", 4), ("SRSRC", 3), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)]));
    code.extend(enc("GLOBAL_ATOMIC_FCMPSWAP_X2", "ENC_FLAT_GLBL", &[("VDST", 40), ("ADDR", 4), ("DATA", 34), ("SADDR", 16), ("GLC", 1)]));
    code.extend(enc("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 42), ("ADDR", 6), ("DATA", 34), ("GLC", 1)]));
    code.extend(enc("GLOBAL_ATOMIC_CMPSWAP_X2", "ENC_FLAT_GLBL", &[("VDST", 44), ("ADDR", 4), ("DATA", 34), ("SADDR", 18), ("GLC", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));

    // 8 IEEE-754 edge cases: (init32, cmp32, init64, cmp64, float_swaps, int_swaps)
    const CASES: [(u32, u32, u64, u64, bool, bool); 8] = [
        (0x8000_0000, 0x0000_0000, 0x8000_0000_0000_0000, 0x0000_0000_0000_0000, true, false),  // -0.0 vs +0.0
        (0x0000_0000, 0x8000_0000, 0x0000_0000_0000_0000, 0x8000_0000_0000_0000, true, false),  // +0.0 vs -0.0
        (0x7FC0_0000, 0x7FC0_0000, 0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000, false, true),  // qNaN vs qNaN
        (0x7F80_0001, 0x7F80_0001, 0x7FF0_0000_0000_0001, 0x7FF0_0000_0000_0001, false, true),  // sNaN vs sNaN
        (0x0000_0001, 0x0000_0000, 0x0000_0000_0000_0001, 0x0000_0000_0000_0000, false, false), // denorm vs +0.0
        (0x3FC0_0000, 0x3FC0_0000, 0x3FF8_0000_0000_0000, 0x3FF8_0000_0000_0000, true, true),   // 1.5 vs 1.5
        (0x3FC0_0000, 0x4020_0000, 0x3FF8_0000_0000_0000, 0x4004_0000_0000_0000, false, false), // 1.5 vs 2.5
        (0xFF80_0000, 0xFF80_0000, 0xFFF0_0000_0000_0000, 0xFFF0_0000_0000_0000, true, true),   // -inf vs -inf
    ];
    const NEW32: u32 = 0x3F80_0000; // 1.0f
    const NEW64: u64 = 0x3FF0_0000_0000_0000; // 1.0

    let mut st = KernelState::default();
    st.sgpr[4] = 0x0400;
    st.sgpr[6] = 0x0600;
    st.sgpr[8..12].copy_from_slice(&[0x0200, 0, 0x100, 0]);
    st.sgpr[12..16].copy_from_slice(&[0x0A00, 0, 0x200, 0]);
    st.sgpr[16] = 0x0C00;
    st.sgpr[18] = 0x1000;
    st.sgpr[24..32].copy_from_slice(&tsharp(0, 20, 9));
    st.sgpr[32..40].copy_from_slice(&tsharp(2, 20, 9));

    let mut mem = vec![0u32; 0x1400 / 4];
    let mut tex_init = Vec::with_capacity(64 * 4);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (init32, cmp32, init64, cmp64, _, _) = CASES[l % CASES.len()];
        let off32 = 4 * l as u32;
        let off64 = 8 * l as u32;
        r[0] = off32;
        r[1] = 0x100 + off32;
        r[2] = 0x0500 + off32;
        r[4] = off64;
        r[6] = 0x0E00 + off64;
        r[8] = l as u32;
        r[9] = 0;
        r[10] = init32;
        r[11] = cmp32;
        r[12] = NEW32;
        for base_v in [14, 16, 18, 20] {
            r[base_v] = NEW32;
            r[base_v + 1] = cmp32;
        }
        for base_v in [30, 34] {
            r[base_v] = NEW64 as u32;
            r[base_v + 1] = (NEW64 >> 32) as u32;
            r[base_v + 2] = cmp64 as u32;
            r[base_v + 3] = (cmp64 >> 32) as u32;
        }
        for a in [0x0200 + 4 * l, 0x0400 + 4 * l, 0x0500 + 4 * l, 0x0600 + 4 * l] {
            mem[a / 4] = init32;
        }
        for a in [0x0A00 + 8 * l, 0x0C00 + 8 * l, 0x0E00 + 8 * l, 0x1000 + 8 * l] {
            mem[a / 4] = init64 as u32;
            mem[a / 4 + 1] = (init64 >> 32) as u32;
        }
        tex_init.extend_from_slice(&init32.to_le_bytes());
    }
    let tex = vec![
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 64, height: 1, depth: 1, texel_bytes: 4, data: tex_init.clone(), uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 64, height: 1, depth: 1, texel_bytes: 4, data: tex_init, uav: true },
    ];

    let t = translate(&code, &runner::options(Mode::Kernel)).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    let hlsl_dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|x| format!("{x}\n--- HLSL ---\n{}", t.hlsl)).unwrap();
    let direct_dxil = runner::direct_kernel_full(&code, true).unwrap().dxil;

    for (backend, dxil) in [("hlsl", hlsl_dxil), ("direct_dxil", direct_dxil)] {
        let r = runner::run_with_statics(&dxil, &st, &mem, &tex, &[], &[]).unwrap_or_else(|e| panic!("{backend}: {e}"));
        let old64 = |v: &[u32], at: usize| v[at] as u64 | (v[at + 1] as u64) << 32;
        let cell64 = |a: usize| r.mem[a / 4] as u64 | (r.mem[a / 4 + 1] as u64) << 32;
        let tex32 = |ti: usize, l: usize| u32::from_le_bytes(r.textures[ti][4 * l..4 * l + 4].try_into().unwrap());
        for l in 0..64usize {
            let (init32, _, init64, _, fswap, iswap) = CASES[l % CASES.len()];
            let want_f32 = if fswap { NEW32 } else { init32 };
            let want_i32 = if iswap { NEW32 } else { init32 };
            let want_f64 = if fswap { NEW64 } else { init64 };
            let want_i64 = if iswap { NEW64 } else { init64 };
            let v = &r.vgpr[l];
            // 1) LDS
            assert_eq!((v[22], v[23]), (init32, want_f32), "{backend} lane {l} case {}: ds_cmpst_rtn_f32 (old, cell)", l % 8);
            assert_eq!((v[24], v[25]), (init32, want_i32), "{backend} lane {l} case {}: ds_cmpst_rtn_b32 control (old, cell)", l % 8);
            // 2) 32-bit buffer / global / flat + int control
            assert_eq!((v[14], r.mem[(0x0200 + 4 * l) / 4]), (init32, want_f32), "{backend} lane {l} case {}: buffer_atomic_fcmpswap", l % 8);
            assert_eq!((v[26], r.mem[(0x0400 + 4 * l) / 4]), (init32, want_f32), "{backend} lane {l} case {}: global_atomic_fcmpswap", l % 8);
            assert_eq!((v[27], r.mem[(0x0500 + 4 * l) / 4]), (init32, want_f32), "{backend} lane {l} case {}: flat_atomic_fcmpswap", l % 8);
            assert_eq!((v[28], r.mem[(0x0600 + 4 * l) / 4]), (init32, want_i32), "{backend} lane {l} case {}: global_atomic_cmpswap control", l % 8);
            // 3) 32-bit image UAV + int control
            assert_eq!((v[18], tex32(0, l)), (init32, want_f32), "{backend} lane {l} case {}: image_atomic_fcmpswap", l % 8);
            assert_eq!((v[20], tex32(1, l)), (init32, want_i32), "{backend} lane {l} case {}: image_atomic_cmpswap control", l % 8);
            // 4) 64-bit buffer / global / flat + int control
            assert_eq!((old64(v, 30), cell64(0x0A00 + 8 * l)), (init64, want_f64), "{backend} lane {l} case {}: buffer_atomic_fcmpswap_x2", l % 8);
            assert_eq!((old64(v, 40), cell64(0x0C00 + 8 * l)), (init64, want_f64), "{backend} lane {l} case {}: global_atomic_fcmpswap_x2", l % 8);
            assert_eq!((old64(v, 42), cell64(0x0E00 + 8 * l)), (init64, want_f64), "{backend} lane {l} case {}: flat_atomic_fcmpswap_x2", l % 8);
            assert_eq!((old64(v, 44), cell64(0x1000 + 8 * l)), (init64, want_i64), "{backend} lane {l} case {}: global_atomic_cmpswap_x2 control", l % 8);
        }
    }
}

/// 32-bit float atomic min / max in memory with a -0.0 operand. The translation turns them
/// into integer atomics chosen by the operand's sign; -0.0 has its sign bit set, so it has to
/// take the negative path: fmin(-5, -0) = -5, fmax(-5, -0) = -0, fmin(3, -0) = -0,
/// fmax(3, -0) = 3. Both code paths: global (flat / global) and buffer.
#[test]
fn gpu_atomic_fmin_fmax_negative_zero() {
    if !gpu_ready() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mubuf = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_MUBUF", f, None).unwrap();
    let mut code = Vec::new();
    // global: s2 + v30 (= 0) + offset; buffer: V# s[8:11], raw base 0x700, + offset
    for (op, off) in [("FMIN", 0u64), ("FMAX", 4), ("FMIN", 8), ("FMAX", 12)] {
        let (g, b) = match op {
            "FMIN" => ("GLOBAL_ATOMIC_FMIN", "BUFFER_ATOMIC_FMIN"),
            _ => ("GLOBAL_ATOMIC_FMAX", "BUFFER_ATOMIC_FMAX"),
        };
        code.extend(glbl(g, &[("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", off)]));
        code.extend(mubuf(b, &[("VDATA", 32), ("VADDR", 30), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", off)]));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[8..12].copy_from_slice(&[0x700, 0, 0x200, 0]);
    for r in st.vgpr.iter_mut() {
        r[30] = 0;
        r[32] = (-0.0f32).to_bits();
    }
    let mut mem = vec![0u32; 0x800 / 4];
    for base in [0x600usize, 0x700] {
        mem[base / 4] = (-5.0f32).to_bits();
        mem[base / 4 + 1] = (-5.0f32).to_bits();
        mem[base / 4 + 2] = 3.0f32.to_bits();
        mem[base / 4 + 3] = 3.0f32.to_bits();
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for (family, base) in [("global", 0x600usize), ("buffer", 0x700)] {
        let cell = |k: usize| r.mem[base / 4 + k];
        assert_eq!(cell(0), (-5.0f32).to_bits(), "{family}: fmin(-5, -0) = -5");
        assert_eq!(cell(1), (-0.0f32).to_bits(), "{family}: fmax(-5, -0) = -0");
        assert_eq!(cell(2), (-0.0f32).to_bits(), "{family}: fmin(3, -0) = -0");
        assert_eq!(cell(3), 3.0f32.to_bits(), "{family}: fmax(3, -0) = 3");
    }
}

/// *_atomic_fmin_x2 / fmax_x2 on hardware, all three families. Six cells each: 64 lanes
/// race on a max (from -inf) and a min (from +inf) of lane - 20.5, then the -0.0 cases of
/// the f32 test. A seventh, per-lane cell checks the GLC return (global).
#[test]
fn gpu_atomic_fmin_fmax_x2() {
    if !gpu_ready() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut code = Vec::new();
    // (op, data VGPR, cell offset): the lane values at v[32:33], -0.0 at v[34:35]
    let ops = [("FMAX", 32u64, 0u64), ("FMIN", 32, 8), ("FMIN", 34, 16), ("FMAX", 34, 24), ("FMIN", 34, 32), ("FMAX", 34, 40)];
    for (op, data, off) in ops {
        let x2 = |family: &str| -> &'static str {
            match (family, op) {
                ("BUFFER", "FMIN") => "BUFFER_ATOMIC_FMIN_X2",
                ("BUFFER", _) => "BUFFER_ATOMIC_FMAX_X2",
                ("FLAT", "FMIN") => "FLAT_ATOMIC_FMIN_X2",
                ("FLAT", _) => "FLAT_ATOMIC_FMAX_X2",
                (_, "FMIN") => "GLOBAL_ATOMIC_FMIN_X2",
                _ => "GLOBAL_ATOMIC_FMAX_X2",
            }
        };
        // buffer: V# s[8:11], raw base 0x800; flat: address v[36:37] = 0x900; global: s2 = 0xA00
        code.extend(enc(x2("BUFFER"), "ENC_MUBUF", &[("VDATA", data), ("VADDR", 30), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", off)]));
        code.extend(enc(x2("FLAT"), "ENC_FLAT", &[("ADDR", 36), ("DATA", data), ("OFFSET", off)]));
        code.extend(enc(x2("GLOBAL"), "ENC_FLAT_GLBL", &[("ADDR", 30), ("DATA", data), ("SADDR", 2), ("OFFSET", off)]));
    }
    // per-lane cell at s3 + v38 (0xB00 + 8·lane): max(1.0, 2.0), the old 1.0 back in v[40:41]
    code.extend(enc("GLOBAL_ATOMIC_FMAX_X2", "ENC_FLAT_GLBL", &[("VDST", 40), ("ADDR", 38), ("DATA", 42), ("SADDR", 3), ("GLC", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0xA00;
    st.sgpr[3] = 0xB00;
    st.sgpr[8..12].copy_from_slice(&[0x800, 0, 0x100, 0]);
    let put = |r: &mut Vec<u32>, at: usize, v: u64| {
        r[at] = v as u32;
        r[at + 1] = (v >> 32) as u32;
    };
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[30] = 0;
        put(r, 32, (l as f64 - 20.5).to_bits());
        put(r, 34, (-0.0f64).to_bits());
        put(r, 36, 0x900);
        r[38] = 8 * l as u32;
        put(r, 42, 2.0f64.to_bits());
    }
    let init = [f64::NEG_INFINITY, f64::INFINITY, -5.0, -5.0, 3.0, 3.0];
    let mut mem = vec![0u32; 0xE00 / 4];
    let set = |mem: &mut Vec<u32>, a: usize, v: f64| {
        mem[a / 4] = v.to_bits() as u32;
        mem[a / 4 + 1] = (v.to_bits() >> 32) as u32;
    };
    for base in [0x800usize, 0x900, 0xA00] {
        for (k, v) in init.iter().enumerate() {
            set(&mut mem, base + 8 * k, *v);
        }
    }
    for l in 0..64usize {
        set(&mut mem, 0xB00 + 8 * l, 1.0);
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let cell = |a: usize| r.mem[a / 4] as u64 | (r.mem[a / 4 + 1] as u64) << 32;
    let want = [42.5f64.to_bits(), (-20.5f64).to_bits(), (-5.0f64).to_bits(), (-0.0f64).to_bits(), (-0.0f64).to_bits(), 3.0f64.to_bits()];
    let what = ["max of the lanes", "min of the lanes", "fmin(-5, -0) = -5", "fmax(-5, -0) = -0", "fmin(3, -0) = -0", "fmax(3, -0) = 3"];
    for (family, base) in [("buffer", 0x800usize), ("flat", 0x900), ("global", 0xA00)] {
        for k in 0..6 {
            assert_eq!(f64::from_bits(cell(base + 8 * k)), f64::from_bits(want[k]), "{family}: {}", what[k]);
            assert_eq!(cell(base + 8 * k), want[k], "{family}: {} (bits)", what[k]);
        }
    }
    for (l, v) in r.vgpr.iter().enumerate() {
        assert_eq!(cell(0xB00 + 8 * l), 2.0f64.to_bits(), "lane {l}: per-lane fmax_x2");
        let old = v[40] as u64 | (v[41] as u64) << 32;
        assert_eq!(old, 1.0f64.to_bits(), "lane {l}: GLC returns the old f64");
    }
}

/// `XCL_BLOB=<console CS container>`: HLSL size and shape of its translation.
#[test]
#[ignore]
fn blob_hlsl_shape() {
    let Some(p) = std::env::var_os("XCL_BLOB") else { return };
    let blob = std::fs::read(p).unwrap();
    let (h, _) = super::pso::console_to_hlsl(&blob, super::pso::Stage::Compute).unwrap();
    println!("HLSL {} KB, pc-dispatch: {}, if: {}, for(;;): {}", h.len() / 1024, h.matches("pc ==").count(), h.matches("if (!(").count(), h.matches("for (;;)").count());
    if let Some(out) = std::env::var_os("XCL_HLSL_OUT") {
        std::fs::write(out, &h).unwrap();
    }
}

/// `XCL_BLOB=<console CS>` `XCL_DXC_ARGS="-O3"`: DXC time and DXIL size for its HLSL (this
/// process is capped at 8 GB of commit).
#[test]
#[ignore]
fn blob_dxc_cost() {
    let Some(p) = std::env::var_os("XCL_BLOB") else { return };
    unsafe {
        use windows_sys::Win32::System::JobObjects::*;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = 8 << 30;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
        AssignProcessToJobObject(job, GetCurrentProcess());
    }
    let blob = std::fs::read(p).unwrap();
    let (h, profile) = super::pso::console_to_hlsl(&blob, super::pso::Stage::Compute).unwrap();
    let args = std::env::var("XCL_DXC_ARGS").unwrap_or_default();
    let extra: Vec<&str> = args.split_whitespace().collect();
    let t0 = std::time::Instant::now();
    let r = super::dxc::compile(&h, "main", profile, &extra);
    println!("HLSL {} KB, DXC {args}: {:.1} s → {}", h.len() / 1024, t0.elapsed().as_secs_f32(), match r {
        Ok(b) => format!("DXIL {} KB", b.len() / 1024),
        Err(e) => format!("error: {}", e.chars().take(200).collect::<String>()),
    });
}

/// `XCL_RAW=<file of raw RDNA2 code>`: its disassembly (reverse-engineering helper).
#[test]
#[ignore]
fn raw_rdna2_listing() {
    let Some(p) = std::env::var_os("XCL_RAW") else { return };
    let code = std::fs::read(p).unwrap();
    for l in crate::gpu::rdna2::disassemble(&code, 0, false) {
        println!("  {:06x}: {}", l.pc, l.text);
    }
}

/// Every FSSM variant of the pack through the translator (no GPU): how many translate
/// completely, per stage, and which instructions are missing (weighted by shaders).
/// `XCL_SAMPLE_RPF=…\shaders.rpf`, `XCL_SURVEY_EVERY=N` (every N-th record, default 1),
/// `XCL_SURVEY_FIND=text` (only records whose name contains it), `XCL_SURVEY_SKIP=a|b` (leave out
/// records whose name contains one of these), `XCL_SURVEY_TRACE=1` (start / time / result per program);
/// `XCL_SURVEY_DXC=1` also compiles each one with DXC (4 threads, process capped at 8 GB):
/// failures and compile times.
#[test]
#[ignore]
fn retail_translation_survey() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let every: usize = std::env::var("XCL_SURVEY_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let data = std::fs::read(path).unwrap();
    // XCL_SURVEY_SEED=<n> or "random": a random 1/every of the records (splitmix64), else every N-th
    let seed = std::env::var("XCL_SURVEY_SEED").ok().map(|v| {
        v.parse::<u64>().unwrap_or_else(|_| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64))
    });
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut all = crate::gpu::pack::scan(&data, 0x40);
    if let Ok(find) = std::env::var("XCL_SURVEY_FIND") {
        all.retain(|r| r.name.contains(&find));
        println!("{} records whose name contains {find:?}", all.len());
    }
    if let Ok(skip) = std::env::var("XCL_SURVEY_SKIP") {
        all.retain(|r| !skip.split('|').any(|s| !s.is_empty() && r.name.contains(s)));
        println!("{} records after skipping {skip:?}", all.len());
    }
    let records: Vec<_> = match seed {
        Some(sd) => {
            println!("random sample: seed {sd}");
            all.into_iter().enumerate().filter(|(i, _)| mix(sd ^ (*i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) % every as u64 == 0).map(|(_, r)| r).collect()
        }
        None => all.into_iter().step_by(every).collect(),
    };
    #[derive(Default)]
    struct Acc {
        // stage → (variants, complete)
        stages: BTreeMap<String, (usize, usize)>,
        // "mnemonic: reason" → shaders
        missing: BTreeMap<String, usize>,
        // error / no interface → shaders (with one example name)
        errors: BTreeMap<String, (usize, String)>,
        instructions: usize,
        // DXC: seconds per variant, failures "stage: first line" → (count, example)
        dxc_times: Vec<(f64, String)>,
        // DXC seconds / variants: fully structured, with `pc` guards
        dxc_structured: (f64, usize),
        dxc_hybrid: (f64, usize),
        hybrid: usize,
        hybrid_names: Vec<(usize, String)>,
        dxc_errors: BTreeMap<String, (usize, String)>,
    }
    let with_dxc = std::env::var_os("XCL_SURVEY_DXC").is_some() && super::dxc::location().is_ok();
    if with_dxc {
        unsafe {
            use windows_sys::Win32::System::JobObjects::*;
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            info.ProcessMemoryLimit = 8 << 30;
            SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
            AssignProcessToJobObject(job, GetCurrentProcess());
        }
    }
    let workers = if with_dxc { 4 } else { std::thread::available_parallelism().map_or(4, |n| n.get()) };
    let acc = Mutex::new(Acc::default());
    let next = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    std::panic::set_hook(Box::new(|_| {}));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                for f in &sh.fssm {
                    let hw = f.stages.first().map_or(0, |s| s.hw);
                    let stage = format!("{} ({})", sh.stage, crate::gpu::xbox::hw_stage_name(hw));
                    let res = std::panic::catch_unwind(|| -> Result<(super::Translation, &'static str), String> {
                        let mode = super::interface(&blob, &sh, f).ok_or("no interface")?;
                        let profile = match &mode {
                            Mode::Vertex(_) => "vs_6_6",
                            Mode::Pixel(_) => "ps_6_6",
                            // MS (NGG) and GS run as compute shaders feeding a pass-through mesh shader
                            Mode::Compute(_) | Mode::Mesh(_) | Mode::Geometry(_) | Mode::Kernel => "cs_6_6",
                        };
                        let p = f.programs.first().ok_or("FSSM without a program")?;
                        let code = &f.data[p.offset..p.offset + p.len];
                        // XCL_SURVEY_WINDOWS=N: translate for N memory windows (above 4 GiB)
                        let mem_windows = std::env::var("XCL_SURVEY_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
                        let trace = std::env::var_os("XCL_SURVEY_TRACE").is_some();
                        if trace {
                            eprintln!("translate start {} ({} B)", r.name, code.len());
                        }
                        let t1 = std::time::Instant::now();
                        let t = super::translate(code, &Options { mode, wave64: super::detect_wave64(code), mem_windows, ..Default::default() });
                        if trace {
                            let state = match &t {
                                Ok(t) if t.complete() => "complete".to_string(),
                                Ok(t) => format!("partial {:?}", t.unsupported_why.keys().collect::<Vec<_>>()),
                                Err(e) => format!("error {e}"),
                            };
                            eprintln!("translate done {} {:.2} s {state}", r.name, t1.elapsed().as_secs_f64());
                        }
                        Ok((t?, profile))
                    });
                    // DXC outside the lock
                    let guarded = matches!(&res, Ok(Ok((t, _))) if t.hlsl.contains("pc =="));
                    // XCL_SURVEY_ONLY=a|b: DXC only for programs whose HLSL contains one of these
                    let only = std::env::var("XCL_SURVEY_ONLY").ok();
                    let wanted = |h: &str| only.as_ref().is_none_or(|o| o.split('|').any(|x| h.contains(x)));
                    let dxc = match (&res, with_dxc) {
                        (Ok(Ok((t, profile))), true) if wanted(&t.hlsl) => {
                            if std::env::var_os("XCL_SURVEY_TRACE").is_some() {
                                eprintln!("dxc start {} ({} KB)", r.name, t.hlsl.len() / 1024);
                            }
                            let t1 = std::time::Instant::now();
                            // XCL_DXC_ARGS="-O1 …": extra DXC arguments (comparing optimisation levels)
                            let args = std::env::var("XCL_DXC_ARGS").unwrap_or_default();
                            let args: Vec<&str> = args.split_whitespace().collect();
                            let r = super::dxc::compile(&t.hlsl, "main", profile, &args);
                            // the specific message (validation failures list it after "validation errors")
                            let first = |e: &str| {
                                e.lines().find(|l| l.contains("error") && !l.contains("validation errors")).or_else(|| e.lines().find(|l| l.contains("error"))).unwrap_or(e.lines().next().unwrap_or("")).to_string()
                            };
                            Some((t1.elapsed().as_secs_f64(), r.err().map(|e| first(&e))))
                        }
                        _ => None,
                    };
                    let mut a = acc.lock().unwrap();
                    let e = a.stages.entry(stage.clone()).or_default();
                    e.0 += 1;
                    let error = |a: &mut Acc, msg: String| {
                        let x = a.errors.entry(format!("{stage}: {msg}")).or_insert((0, r.name.clone()));
                        x.0 += 1;
                    };
                    a.hybrid += guarded as usize;
                    if let (true, Ok(Ok((t, _)))) = (guarded, &res) {
                        a.hybrid_names.push((t.instructions, r.name.clone()));
                    }
                    if let Some((secs, err)) = dxc {
                        a.dxc_times.push((secs, r.name.clone()));
                        let b = if guarded { &mut a.dxc_hybrid } else { &mut a.dxc_structured };
                        b.0 += secs;
                        b.1 += 1;
                        if let Some(e) = err {
                            // drop the file:line:col prefix so equal errors group
                            let e = e.split("error:").nth(1).unwrap_or(&e).trim().to_string();
                            let x = a.dxc_errors.entry(format!("{stage}: {e}")).or_insert((0, r.name.clone()));
                            x.0 += 1;
                        }
                    }
                    match res {
                        Ok(Ok((t, _))) => {
                            a.instructions += t.instructions;
                            if t.complete() {
                                a.stages.get_mut(&stage).unwrap().1 += 1;
                            }
                            for k in t.unsupported_why.keys() {
                                *a.missing.entry(k.clone()).or_default() += 1;
                            }
                        }
                        Ok(Err(m)) => error(&mut a, m),
                        Err(_) => error(&mut a, "translator panic".into()),
                    }
                }
            });
        }
    });
    let _ = std::panic::take_hook();
    let a = acc.into_inner().unwrap();
    let (all, done): (usize, usize) = a.stages.values().fold((0, 0), |s, v| (s.0 + v.0, s.1 + v.1));
    println!("records: {} (1 in {every}), FSSM variants: {all}, fully translated: {done} ({:.1}%), instructions: {}, time {:.1} s", records.len(), 100.0 * done as f64 / all.max(1) as f64, a.instructions, t0.elapsed().as_secs_f64());
    println!("with pc blocks (not fully structured): {} / {all}", a.hybrid);
    let mut hn = a.hybrid_names;
    hn.sort();
    println!("  smallest: {:?}", hn.iter().take(6).collect::<Vec<_>>());
    for (st, (n, ok)) in &a.stages {
        println!("  {st:<14} {ok:>6} / {n:<6} ({:.1}%)", 100.0 * *ok as f64 / (*n).max(1) as f64);
    }
    let mut missing: Vec<_> = a.missing.into_iter().collect();
    missing.sort_by(|x, y| y.1.cmp(&x.1));
    println!("missing instructions (in how many variants):");
    for (k, n) in missing.iter().take(40) {
        println!("  {n:>6}  {k}");
    }
    let mut errors: Vec<_> = a.errors.into_iter().collect();
    errors.sort_by(|x, y| y.1 .0.cmp(&x.1 .0));
    println!("errors:");
    for (k, (n, ex)) in errors.iter().take(20) {
        println!("  {n:>6}  {k}   (e.g. {ex})");
    }
    if with_dxc {
        let mut t = a.dxc_times;
        t.sort_by(|x, y| x.0.total_cmp(&y.0));
        let q = |f: f64| t.get(((t.len() as f64 * f) as usize).min(t.len().saturating_sub(1))).map_or(0.0, |x| x.0);
        let failed: usize = a.dxc_errors.values().map(|v| v.0).sum();
        println!(
            "DXC structured: {} variants, {:.0} s (mean {:.2} s); with pc blocks: {} variants, {:.0} s (mean {:.2} s)",
            a.dxc_structured.1,
            a.dxc_structured.0,
            a.dxc_structured.0 / a.dxc_structured.1.max(1) as f64,
            a.dxc_hybrid.1,
            a.dxc_hybrid.0,
            a.dxc_hybrid.0 / a.dxc_hybrid.1.max(1) as f64
        );
        println!("DXC: {} compiled, {failed} errors; time median {:.2} s, p90 {:.2} s, max {:.1} s ({}), total {:.0} s", t.len(), q(0.5), q(0.9), t.last().map_or(0.0, |x| x.0), t.last().map_or("", |x| x.1.as_str()), t.iter().map(|x| x.0).sum::<f64>());
        for (secs, name) in t.iter().rev().take(10) {
            println!("  {secs:>6.2} s  {name}");
        }
        let mut errs: Vec<_> = a.dxc_errors.into_iter().collect();
        errs.sort_by(|x, y| y.1 .0.cmp(&x.1 .0));
        for (k, (n, ex)) in errs.iter().take(20) {
            println!("  {n:>6}  {k}   (e.g. {ex})");
        }
    }
}

/// Shape of the HLSL and DXC cost of named retail shaders (reverse engineering of compile
/// times): `XCL_SAMPLE_RPF`, `XCL_NAMES=a,b` (substring of the record name), `XCL_DXC_ARGS`,
/// `XCL_HLSL_OUT=<folder>` (writes the HLSL). The process is capped at 8 GB.
#[test]
#[ignore]
fn retail_shader_cost() {
    let (Ok(path), Ok(names)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_NAMES")) else { return };
    unsafe {
        use windows_sys::Win32::System::JobObjects::*;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = 8 << 30;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
        AssignProcessToJobObject(job, GetCurrentProcess());
    }
    let args = std::env::var("XCL_DXC_ARGS").unwrap_or_default();
    let extra: Vec<&str> = args.split_whitespace().collect();
    let data = std::fs::read(path).unwrap();
    for r in crate::gpu::pack::scan(&data, 0x40) {
        if !names.split(',').any(|n| !n.is_empty() && r.name.contains(n)) {
            continue;
        }
        let Some(blob) = crate::gpu::pack::inflate(&data, &r) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        // effect-qualified names (`effect:CS_…`) are not file names (`:` opens an NTFS stream)
        let stem = r.name.replace(['/', '\\', ':'], "_");
        for f in &sh.fssm {
            let Some(mode) = super::interface(&blob, &sh, f) else { continue };
            let profile = match &mode {
                Mode::Vertex(_) => "vs_6_6",
                Mode::Pixel(_) => "ps_6_6",
                _ => "cs_6_6",
            };
            let p = f.programs[0];
            let code = &f.data[p.offset..p.offset + p.len];
            let t = super::translate(code, &Options { mode, wave64: super::detect_wave64(code), ..Default::default() }).unwrap();
            let owned = t.hlsl.clone();
            let h = &owned;
            if let Ok(dir) = std::env::var("XCL_HLSL_OUT") {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.hlsl")), h);
                eprintln!("HLSL written: {} ({} KB)", r.name, h.len() / 1024);
            }
            let t0 = std::time::Instant::now();
            let r2 = super::dxc::compile(h, "main", profile, &extra);
            let secs = t0.elapsed().as_secs_f64();
            if let (Err(_), Ok(dir)) = (&r2, std::env::var("XCL_HLSL_OUT")) {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.failed.hlsl")), h);
            }
            println!(
                "{} [{profile}] instructions {}, HLSL {} KB, lines {}, pc-dispatch {}, if {}, for(;;) {}, select {} → DXC {:.2} s, {}",
                r.name,
                t.instructions,
                h.len() / 1024,
                h.lines().count(),
                h.matches("pc ==").count(),
                h.matches("if (").count(),
                h.matches("for (;;)").count(),
                h.matches("select(").count(),
                secs,
                r2.as_ref().map(|b| format!("DXIL {} KB", b.len() / 1024)).unwrap_or_else(|e| e.lines().next().unwrap_or("").to_string())
            );
            if let Ok(dir) = std::env::var("XCL_HLSL_OUT") {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.hlsl")), h);
            }
        }
    }
}

/// ISA coverage: every instruction the decoder knows (RDNA2 spec + console extensions), one
/// at a time with zero operands, through the translator (compute kernel mode, then pixel
/// mode for graphics-only ones). Per encoding: translated / all, and the missing ones.
#[test]
#[ignore]
fn isa_coverage() {
    use crate::gpu::rdna2::{all_instructions, encode};
    use crate::gpu::rdna2_spec::ENCODINGS;
    use std::collections::BTreeMap;
    let mut per_enc: BTreeMap<&str, (usize, usize, Vec<&str>)> = BTreeMap::new();
    let try_mode = |code: &[u8], mode: Mode| match translate(code, &Options { mode, ..Default::default() }) {
        Ok(t) => t.complete(),
        Err(_) => false,
    };
    std::panic::set_hook(Box::new(|_| {}));
    for inst in all_instructions() {
        // the first plain encoding (not an SDWA / DPP / literal variant)
        let Some(ie) = inst.encs.iter().find(|e| ENCODINGS[e.encoding as usize].primary) else { continue };
        let enc = ENCODINGS[ie.encoding as usize].name;
        let lit = inst.always_literal.then_some(0);
        // operands that must not be 0: MIMG channel mask, VOP3 SDST (VCC), VGPR-only sources
        let fields: &[(&str, u64)] = match enc {
            "ENC_MIMG" => &[("DMASK", 1)],
            "VOP3_SDST_ENC" => &[("SDST", 106)],
            _ if inst.name == "V_SWAPREL_B32" => &[("SRC0", V)],
            _ if inst.name == "V_MQSAD_U32_U8" => &[("SRC2", V)],
            _ => &[],
        };
        let Ok(mut code) = encode(inst.name, enc, fields, lit) else { continue };
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let ok = std::panic::catch_unwind(|| {
            try_mode(&code, Mode::Kernel) || try_mode(&code, Mode::Pixel(super::PsInterface { inputs: vec![], outputs: vec![], input_ena: 0x7F, attrs: None }))
        })
        .unwrap_or(false);
        let e = per_enc.entry(enc.trim_start_matches("ENC_")).or_default();
        e.1 += 1;
        if ok {
            e.0 += 1;
        } else {
            e.2.push(inst.name);
        }
    }
    let _ = std::panic::take_hook();
    let (ok, all) = per_enc.values().fold((0, 0), |s, v| (s.0 + v.0, s.1 + v.1));
    println!("RDNA2 instructions (spec + console): {all}, translated: {ok} ({:.1}%)", 100.0 * ok as f64 / all.max(1) as f64);
    for (enc, (ok, all, missing)) in &per_enc {
        let m: Vec<String> = missing.iter().map(|n| n.to_lowercase()).collect();
        let full = std::env::var_os("XCL_ISA_FULL").is_some();
        println!("  {enc:<12} {ok:>4} / {all:<4} missing: {}", if m.len() > 14 && !full { format!("{} … (+{})", m[..14].join(", "), m.len() - 14) } else { m.join(", ") });
    }
}

/// Census of the cross-lane / control-flow forms the translator special-cases: every
/// `ds_swizzle_b32` offset in the pack, and the instructions around each `s_setpc_b64` /
/// `s_swappc_b64` of the records whose name contains one of `XCL_NAMES` (`a,b`).
#[test]
#[ignore]
fn retail_swizzle_and_setpc_census() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let names = std::env::var("XCL_NAMES").unwrap_or_default();
    let data = std::fs::read(path).unwrap();
    let records = crate::gpu::pack::scan(&data, 0x40);
    let swz: Mutex<BTreeMap<u64, (usize, String)>> = Mutex::new(BTreeMap::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                let show = names.split(',').any(|n| !n.is_empty() && r.name.contains(n));
                for f in &sh.fssm {
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        if let (true, Ok(dir)) = (show, std::env::var("XCL_RAW_OUT")) {
                            let stem = r.name.replace(['/', '\\', ':'], "_");
                            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.{:x}.bin", p.offset)), code);
                        }
                        let mut off = 0usize;
                        let mut recent: Vec<usize> = Vec::new();
                        while off + 4 <= code.len() {
                            let Ok(d) = crate::gpu::rdna2::decode(&code[off..], off as u64) else { off += 4; continue };
                            let name = d.inst.name;
                            if name.eq_ignore_ascii_case("ds_swizzle_b32") {
                                let v = |n: &str| crate::gpu::rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
                                let mut m = swz.lock().unwrap();
                                m.entry(v("OFFSET0") | v("OFFSET1") << 8).or_insert((0, r.name.clone())).0 += 1;
                            }
                            if show && (name.eq_ignore_ascii_case("s_setpc_b64") || name.eq_ignore_ascii_case("s_swappc_b64")) {
                                let from = recent.get(recent.len().saturating_sub(12)).copied().unwrap_or(off);
                                let to = (off + 24).min(code.len());
                                let text: Vec<String> = crate::gpu::rdna2::disassemble(&code[from..to], from as u64, false)
                                    .iter()
                                    .map(|l| format!("  {:06x}: {}", l.pc, l.text))
                                    .collect();
                                println!("{} @{off:#x}:
{}", r.name, text.join("
"));
                            }
                            recent.push(off);
                            off += d.len;
                        }
                    }
                }
            });
        }
    });
    println!("ds_swizzle_b32 offsets (uses, example):");
    for (o, (n, ex)) in swz.into_inner().unwrap() {
        println!("  {o:#06x} {n:>6}  {ex}");
    }
}

/// Every cross-lane read `upper_lanes` resolves (or not) in the pack's wave64 VS / PS
/// programs, with the instructions around it: `XCL_SAMPLE_RPF`, `XCL_CENSUS_OUT=<file>` (the
/// listing; marks `>>` unresolved, `C ` constant, `S ` shadow, `N ` native), `XCL_CENSUS_FULL=1`
/// (whole programs), `XCL_CENSUS_UNKNOWN=1` (only programs with an unresolved read). Prints
/// sites and programs (all, and distinct code) per resolution.
#[test]
#[ignore]
fn retail_wave64_graphics_lane_census() {
    use super::upper_lanes::Read;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let (Ok(path), Ok(out)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_CENSUS_OUT")) else { return };
    let full = std::env::var_os("XCL_CENSUS_FULL").is_some();
    let only_unknown = std::env::var_os("XCL_CENSUS_UNKNOWN").is_some();
    let data = std::fs::read(path).unwrap();
    let records = crate::gpu::pack::scan(&data, 0x40);
    let text: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
    // resolution → (sites, programs, distinct programs by code hash)
    let stats: Mutex<BTreeMap<&'static str, (usize, usize, BTreeSet<u64>)>> = Mutex::new(BTreeMap::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                for (fi, f) in sh.fssm.iter().enumerate() {
                    let mode = match super::interface(&blob, &sh, f) {
                        Some(Mode::Vertex(_)) => "VS",
                        Some(Mode::Pixel(_)) => "PS",
                        _ => continue,
                    };
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        if !super::detect_wave64(code) {
                            continue;
                        }
                        let mut insts = Vec::new();
                        let mut off = 0;
                        while off + 4 <= code.len() {
                            let Ok(d) = crate::gpu::rdna2::decode(&code[off..], off as u64) else { break };
                            off += d.len;
                            insts.push(d);
                        }
                        crate::gpu::rdna2::resolve_far_jumps(&mut insts);
                        let a = super::upper_lanes::analyze(&insts);
                        if a.reads.is_empty() || (only_unknown && !a.reads.values().any(|r| *r == Read::Unknown)) {
                            continue;
                        }
                        let hash = crate::gpu::shader::fnv64(code);
                        {
                            let mut st = stats.lock().unwrap();
                            let mut kinds = BTreeSet::new();
                            for r in a.reads.values() {
                                let k = match r {
                                    Read::Native => "native (lanes >= 32 unwritten / inactive)",
                                    Read::Const(_) => "constant (whole-wave identity)",
                                    Read::Shadow => "spill slot shadow",
                                    Read::Unknown => "unresolved",
                                };
                                st.entry(k).or_default().0 += 1;
                                kinds.insert(k);
                            }
                            for k in kinds {
                                let e = st.entry(k).or_default();
                                e.1 += 1;
                                e.2.insert(hash);
                            }
                        }
                        let lines = crate::gpu::rdna2::disassemble(code, 0, false);
                        let mut s = format!("=== {} FSSM#{fi} {mode} {} B {hash:016x}
", r.name, code.len());
                        if a.reads.values().any(|r| *r == Read::Unknown) {
                            // where EXEC's upper word and the VGPRs' lanes ≥ 32 stopped being known
                            super::upper_lanes::TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
                            super::upper_lanes::analyze(&insts);
                            for l in super::upper_lanes::TRACE.with(|t| t.borrow_mut().take()).unwrap_or_default() {
                                s.push_str(&format!("   trace {l}
"));
                            }
                        }
                        let mark = |pc: u64| match a.reads.get(&pc) {
                            Some(Read::Unknown) => ">>",
                            Some(Read::Const(_)) => "C ",
                            Some(Read::Shadow) => "S ",
                            Some(Read::Native) => "N ",
                            None => "  ",
                        };
                        let sites: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| a.reads.contains_key(&l.pc)).map(|(k, _)| k).collect();
                        let mut shown = 0usize;
                        for &k in &sites {
                            let (from, to) = if full { (0, lines.len()) } else { (k.saturating_sub(40), (k + 8).min(lines.len())) };
                            let from = from.max(shown);
                            if from > shown || shown == 0 {
                                s.push_str("  ...
");
                            }
                            for l in &lines[from..to.max(from)] {
                                s.push_str(&format!("{} {:05x}: {}
", mark(l.pc), l.pc, l.text));
                            }
                            shown = shown.max(to);
                        }
                        text.lock().unwrap().push((r.name.clone(), s));
                    }
                }
            });
        }
    });
    let mut v = text.into_inner().unwrap();
    v.sort();
    println!("{} programs listed", v.len());
    for (k, (sites, progs, distinct)) in stats.into_inner().unwrap() {
        println!("  {k:<45} {sites:>6} sites {progs:>6} programs {:>6} distinct", distinct.len());
    }
    std::fs::write(out, v.into_iter().map(|(_, s)| s).collect::<String>()).unwrap();
}

/// The 32-lane form of the pack's wave64 VS / PS (`upper_lanes`): every program whose
/// translation it changes is compiled with DXC. Digests of every program's translation in the
/// native form (`split_wave64` off – the form a GPU with 64-lane waves gets) and of the other
/// modes' 32-lane form are printed, to compare with another build: `XCL_SAMPLE_RPF`;
/// `XCL_FORMS_NO_DXC=1` skips the compiles.
#[test]
#[ignore]
fn retail_wave64_graphics_lane_forms() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let dxc = std::env::var_os("XCL_FORMS_NO_DXC").is_none();
    let data = std::fs::read(path).unwrap();
    let records = crate::gpu::pack::scan(&data, 0x40);
    let (digest, programs, changed, compiled) = (AtomicU64::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let split_digest = AtomicU64::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                for (fi, f) in sh.fssm.iter().enumerate() {
                    let mode = super::interface(&blob, &sh, f).unwrap_or(Mode::Kernel);
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        let wave64 = super::detect_wave64(code);
                        let opt = Options { mode: mode.clone(), wave64, ..Default::default() };
                        let Ok(native) = translate(code, &opt) else { continue };
                        programs.fetch_add(1, Ordering::Relaxed);
                        let key = format!("{} FSSM#{fi} {:x}", r.name, p.offset);
                        let h = crate::gpu::shader::fnv64(format!("{key}
{}", native.hlsl).as_bytes());
                        digest.fetch_add(h, Ordering::Relaxed);
                        let profile = match &mode {
                            Mode::Vertex(_) => "vs_6_6",
                            Mode::Pixel(_) => "ps_6_6",
                            _ => {
                                // the other modes' 32-lane form: unchanged by `upper_lanes`
                                if let Ok(t) = translate(code, &Options { split_wave64: true, ..opt }) {
                                    split_digest.fetch_add(crate::gpu::shader::fnv64(format!("{key}\\n{}", t.hlsl).as_bytes()), Ordering::Relaxed);
                                }
                                continue;
                            }
                        };
                        if !wave64 {
                            continue;
                        }
                        let t = translate(code, &Options { split_wave64: true, ..opt }).unwrap();
                        if t.hlsl == native.hlsl {
                            continue;
                        }
                        changed.fetch_add(1, Ordering::Relaxed);
                        if dxc {
                            match super::dxc::compile(&t.hlsl, "main", profile, &[]) {
                                Ok(_) => {
                                    compiled.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(e) => errors.lock().unwrap().push(format!("{key}: {}", e.lines().take(4).collect::<Vec<_>>().join(" | "))),
                            }
                        }
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    println!(
        "{} programs, native-form digest {:016x}, 32-lane-form digest of the other modes {:016x}; 32-lane form differs for {} wave64 VS / PS, {} compiled, {} DXC errors",
        programs.into_inner(),
        digest.into_inner(),
        split_digest.into_inner(),
        changed.into_inner(),
        compiled.into_inner(),
        errors.len()
    );
    for e in errors.iter().take(10) {
        println!("  {e}");
    }
    assert!(errors.is_empty());
}

/// `ds_swizzle_b32`'s four modes against the RDNA ISA (its FFT examples, rotate, quad
/// permute, and the bit mode with xor ≥ 16, which retail compute shaders use to swap the
/// 16-lane halves: 0x401F).
#[test]
fn ds_swizzle_modes_follow_the_isa() {
    use super::emit::swizzle_lane;
    let lanes = |off: u64| (0..32).map(|i| swizzle_lane(off, i)).collect::<Vec<u32>>();
    // ISA examples (inputs 1..0x20, so each value − 1 is the source lane)
    let fft0 = [0x01, 0x11, 0x09, 0x19, 0x05, 0x15, 0x0d, 0x1d, 0x03, 0x13, 0x0b, 0x1b, 0x07, 0x17, 0x0f, 0x1f, 0x02, 0x12, 0x0a, 0x1a, 0x06, 0x16, 0x0e, 0x1e, 0x04, 0x14, 0x0c, 0x1c, 0x08, 0x18, 0x10, 0x20];
    let fft10 = [0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f, 0x02, 0x0a, 0x06, 0x0e, 0x04, 0x0c, 0x08, 0x10, 0x11, 0x19, 0x15, 0x1d, 0x13, 0x1b, 0x17, 0x1f, 0x12, 0x1a, 0x16, 0x1e, 0x14, 0x1c, 0x18, 0x20];
    assert_eq!(lanes(0xE000), fft0.iter().map(|v| v - 1).collect::<Vec<u32>>());
    assert_eq!(lanes(0xE010), fft10.iter().map(|v| v - 1).collect::<Vec<u32>>());
    assert_eq!(lanes(0xE01F), (0..32).collect::<Vec<u32>>(), "FFT mask 0x1F: no swizzle");
    // rotate left by 3 / right by 5 (offset[10]), whole 32 lanes
    assert_eq!(lanes(0xC000 | 3 << 5), (0..32).map(|i| (i + 3) % 32).collect::<Vec<u32>>());
    assert_eq!(lanes(0xC400 | 5 << 5), (0..32).map(|i| (i + 27) % 32).collect::<Vec<u32>>());
    // quad permute [3, 2, 1, 0]
    assert_eq!(lanes(0x8000 | 0x1B), (0..32).map(|i| (i & !3) | (3 - (i & 3))).collect::<Vec<u32>>());
    // bit mode: and 0x1F, xor 16 (swap halves) / xor 18
    assert_eq!(lanes(0x401F), (0..32).map(|i| i ^ 16).collect::<Vec<u32>>());
    assert_eq!(lanes(0x481F), (0..32).map(|i| i ^ 18).collect::<Vec<u32>>());
    assert_eq!(lanes(0x041F), (0..32).map(|i| i ^ 1).collect::<Vec<u32>>());
}

#[test]
fn gpu_ds_swizzle_modes() {
    if !gpu_ready() {
        return;
    }
    let offsets = [0x401Fu64, 0x481F, 0xE000, 0xE010, 0xC000 | 3 << 5, 0xC400 | 5 << 5 | 0x3, 0x8000 | 0x1B];
    let list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = offsets
        .iter()
        .enumerate()
        .map(|(k, &o)| ("DS_SWIZZLE_B32", "ENC_DS", vec![("VDST", 16 + k as u64), ("ADDR", 0), ("OFFSET0", o & 0xFF), ("OFFSET1", o >> 8)], None))
        .collect();
    let list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(n, e, f, l)| (*n, *e, f.as_slice(), *l)).collect();
    let code = prog(&list);
    let t = super::translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 100 + l as u32;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64u32 {
        for (k, &o) in offsets.iter().enumerate() {
            let want = 100 + ((l & 0x20) | super::emit::swizzle_lane(o, l));
            assert_eq!(r.vgpr[l as usize][16 + k], want, "lane {l} offset {o:#06x}");
        }
    }
}

/// The compiler's far jump (`s_getpc_b64`, add of a 64-bit constant, `s_setpc_b64`) out of
/// `s_branch` range is a direct branch: it gets its target, the program stays structured and
/// complete, and the code it jumps over does not run.
#[test]
fn far_jumps_are_direct_branches() {
    let far = |sub: bool, offset: u32| -> Vec<u8> {
        let (lo, hi) = if sub { ("S_SUB_U32", "S_SUBB_U32") } else { ("S_ADD_U32", "S_ADDC_U32") };
        prog(&[
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 4)], None),
            (lo, "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 255)], Some(offset)),
            (hi, "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 128)], None),
            ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)], None), // 0x14: v1 = 7 (jumped over)
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 137)], None), // 0x18: v2 = 9
        ])
    };
    let decoded = |code: &[u8]| {
        let mut v = Vec::new();
        let mut off = 0;
        while off + 4 <= code.len() {
            let d = crate::gpu::rdna2::decode(&code[off..], off as u64).unwrap();
            off += d.len;
            v.push(d);
        }
        crate::gpu::rdna2::resolve_far_jumps(&mut v);
        v
    };
    // s_getpc_b64 returns 0x4 (the next instruction): 0x4 + 0x14 = 0x18
    let code = far(false, 0x14);
    let insts = decoded(&code);
    assert_eq!(insts[3].far_target, Some(0x18));
    assert_eq!(crate::gpu::rdna2::branch_target(&insts[3]), Some(0x18));
    // backward form: 0x4 − 4 = 0x0
    assert_eq!(decoded(&far(true, 4))[3].far_target, Some(0));
    // an s_setpc_b64 of another register pair stays indirect
    let mut other = code.clone();
    other[0x10..0x14].copy_from_slice(&encode("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None).unwrap());
    assert_eq!(decoded(&other)[3].far_target, None);
    let t = super::translate(&other, &Options::default()).unwrap();
    assert!(!t.complete());

    let t = super::translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    assert!(!t.hlsl.contains("pc =="), "far jump left the structured form:\n{}", t.hlsl);
    if !gpu_ready() {
        return;
    }
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        r[1] = 1;
        r[2] = 2;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!((r.vgpr[l][1], r.vgpr[l][2]), (1, 9), "lane {l}");
    }
}

/// Mesh mode for a GPU with 32-lane waves only: the console's two wave64s per group as four
/// hardware wave32s exchanging masks and lane reads (counter-paired: two console waves),
/// programs with cross-half operations included; native wave64 and wave32 programs unchanged.
#[test]
fn mesh_split_wave64_form() {
    let ms = super::MsInterface { threads: 128, max_vertices: 64, max_primitives: 64, outputs: vec![] };
    let lane = prog(&[("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V)], None)]);
    let branch = prog(&[("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 0)], None)]);
    for code in [&lane, &branch] {
        let native = super::translate_mesh(code, ms.clone(), true, false, 1).unwrap();
        assert!(native.hlsl.contains("[WaveSize(64)]") && native.hlsl.contains("[numthreads(128, 1, 1)]") && !native.hlsl.contains("xcl_x_"));
        let split = super::translate_mesh(code, ms.clone(), true, true, 1).unwrap();
        assert!(split.complete(), "{:?}", split.unsupported_why);
        assert!(split.hlsl.contains("[WaveSize(32)]") && split.hlsl.contains("[numthreads(128, 1, 1)]"));
        assert!(split.hlsl.contains("uint xcl_tid = xcl_x_thread(gi);") && split.hlsl.contains("#define XCL_X_SPIN 1"));
        assert!(split.hlsl.contains("xcl_x_ballot(true)"));
        // a wave32 program is not split
        let w32 = super::translate_mesh(code, ms.clone(), false, true, 1).unwrap();
        assert!(w32.hlsl.contains("[WaveSize(32)]") && !w32.hlsl.contains("xcl_x_"));
        if super::dxc::location().is_ok() {
            super::dxc::compile(&split.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", split.hlsl));
        }
    }
    assert!(super::translate_mesh(&lane, ms, true, true, 1).unwrap().hlsl.contains("xcl_x_rla_uniform("));
}

/// Operations whose result spans both 32-lane halves of a wave64 – what the split form
/// exchanges: masks with bits in either half (counted, scanned, branched on), readlane /
/// readfirstlane of a lane in the other half, ds_bpermute across the halves, a GLC scalar
/// load. The same results natively (64-lane waves) and split (32-lane waves only); split,
/// the runner also checks both halves end with the same SGPRs.
#[test]
fn gpu_split_wave64_cross_half_operations() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40)), // vcc = lanes 0..39
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V + 1), ("SRC1", 2)], None), // lane s2 = 37
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 106), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 107), ("SRC1", V + 5)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 128)], None), // exec = lanes 48..63
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0xFFFF_0000)),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V + 1)], None),
        ("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 126)], None),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None), // not taken: only the upper half is active
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 3)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 4), ("DATA0", 1)], None), // lane 63 - l
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(50)), // vcc = lane 50 only
        ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), // not taken
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 129)], None),
        ("S_AND_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 126), ("SSRC1", 106)], None), // scc = vcc != 0
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 7), ("SSRC0", 129), ("SSRC1", 130)], None),
        // s[8:9] = 0: the test memory
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125), ("GLC", 1)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[2] = 37;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[1] = 1000 + l as u32;
        r[2] = 0x77;
        r[4] = (63 - l as u32) * 4;
    }
    let mem = [0xABCDu32, 0x1234, 0x5678, 0x9ABC];
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!(r.sgpr[0], 40, "s_bcnt1 of a mask over both halves");
    assert_eq!(r.sgpr[1], 1037, "readlane of lane 37 (upper half)");
    assert_eq!(r.sgpr[3], 1048, "readfirstlane with only the upper half active");
    assert_eq!(r.sgpr[4], 48, "s_ff1 of exec");
    assert_eq!(r.sgpr[5], 1, "branch on a vcc with a bit in the upper half only");
    assert_eq!(r.sgpr[7], 1, "scc from a 64-bit mask operation");
    assert_eq!((r.sgpr[10], r.sgpr[11]), (0, 1 << 18), "s_and_b64 exec, vcc");
    assert_eq!(&r.sgpr[12..16], &mem, "s_load_dwordx4 glc");
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert_eq!(v[5], (l as u32).min(40), "lane {l}: v_mbcnt over vcc");
        assert_eq!(v[2], if l >= 48 { 1048 } else { 0x77 }, "lane {l}: write under exec of the upper half");
        assert_eq!(v[6], 1000 + 63 - l as u32, "lane {l}: ds_bpermute from lane {}", 63 - l);
    }
}

/// Masks only read lane by lane (`x64_masks`): a v_cndmask selector and a 64-bit carry chain,
/// both with bits in the two halves, need no exchange in the split form – and give the same
/// lanes as a whole-wave mask.
#[test]
fn gpu_split_wave64_lane_local_masks() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40)), // vcc = lanes 0..39
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 2), ("VSRC1", 3)], None),
        // s[20:21] = lanes 0..49, as a VOP3 selector
        ("V_CMP_LT_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", 255)], Some(50)),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", 20)], None),
        // v[4:5] + v[6:7] as 64-bit numbers: the carry of the low words feeds the high words
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 4), ("SDST", 106), ("SRC0", V + 4), ("SRC1", V + 6)], None),
        ("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 5), ("VSRC1", 7)], None),
        // the masks end dead (the kernel runner reads every SGPR back)
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)], None),
    ]);
    if runner::split_wave64() {
        let t = translate(&code, &runner::options(Mode::Kernel)).unwrap();
        assert_eq!(t.hlsl.matches("= xcl_x_ballot_local(and(").count(), 4, "every mask stays in its half");
        assert_eq!(t.hlsl.matches("= xcl_x_ballot(and(").count(), 0);
    }
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l = l as u32;
        r[0] = l;
        r[2] = 100 + l;
        r[3] = 200 + l;
        // a carry out of the low word in every third lane
        r[4] = if l.is_multiple_of(3) { 0xFFFF_FFF0 } else { l };
        r[5] = 1000 + l;
        r[6] = 0x20;
        r[7] = 7;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64u32 {
        let v = &r.vgpr[l as usize];
        assert_eq!(v[1], if l < 40 { 200 + l } else { 100 + l }, "lane {l}: v_cndmask on vcc");
        assert_eq!(v[8], if l < 50 { 200 + l } else { 100 + l }, "lane {l}: v_cndmask on s[20:21]");
        let a = ((1000 + l as u64) << 32) | if l % 3 == 0 { 0xFFFF_FFF0 } else { l as u64 };
        let sum = a + ((7u64 << 32) | 0x20);
        assert_eq!((v[4], v[5]), (sum as u32, (sum >> 32) as u32), "lane {l}: 64-bit add with carry");
    }
}

/// Split-wave64 reduction-tail folding rules: 32-bit reduction tails, 64-bit `v_add_f64` tails,
/// and direct `(L, L^32)` pairs fold into one `xcl_x_meet()`, while modifiers, mismatched lanes,
/// `EXEC` writes, jump targets, and native wave64 stay untouched.
#[test]
fn split_wave64_reduction_tail_rules() {
    let split = Options { mode: Mode::Kernel, wave64: true, split_wave64: true, ..Default::default() };

    // 32-bit reduction tail: v_readlane(s8, v2, 32) ; v_min_u32 v2, s8, v2 ; v_readlane(s12, v2, 0)
    let tail32 = prog(&[
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_MIN_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 8), ("VSRC1", 2)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", 128)], None),
    ]);
    let t = translate(&tail32, &split).unwrap();
    assert_eq!(t.hlsl.matches("xcl_x_meet() ?").count(), 1 + 3, "1 folded meet + 3 in prelude:\n{}", t.hlsl);
    assert_eq!(t.hlsl.matches("= xcl_x_rla_uniform(").count(), 0);

    // Native wave64 is unchanged
    let t_native = translate(&tail32, &Options { mode: Mode::Kernel, wave64: true, split_wave64: false, ..Default::default() }).unwrap();
    assert!(!t_native.hlsl.contains("xcl_x_meet"));

    // 64-bit v_add_f64 reduction tail: 4 readlanes -> 1 2-word meet
    let tail64 = prog(&[
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 3), ("SRC1", 128 + 32)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 2), ("SRC1", V + 2)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 2), ("SRC1", 128)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 3), ("SRC1", 128)], None),
    ]);
    let t = translate(&tail64, &split).unwrap();
    assert_eq!(t.hlsl.matches("= xcl_x_meet();").count(), 1, "64-bit tail shares 1 meet");
    assert_eq!(t.hlsl.matches("= xcl_x_rla_uniform(").count(), 0);
    assert!(t.hlsl.contains("#define XCL_X_SLOT 2u"));

    // Direct (L, L^32) pair: 2 readlanes -> 1 meet
    let direct = prog(&[
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", 128 + 5)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 2), ("SRC1", 128 + 37)], None),
    ]);
    let t = translate(&direct, &split).unwrap();
    assert_eq!(t.hlsl.matches("= xcl_x_rla_uniform(").count(), 0);

    // Negative guards: CLAMP modifier, mismatched lanes, jump target in middle
    let clamped = prog(&[
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 8), ("SRC1", V + 2), ("CLAMP", 1)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", 128)], None),
    ]);
    assert_eq!(translate(&clamped, &split).unwrap().hlsl.matches("= xcl_x_rla_uniform(").count(), 2);

    let mismatched = prog(&[
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_MIN_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 8), ("VSRC1", 2)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", 128 + 1)], None),
    ]);
    assert_eq!(translate(&mismatched, &split).unwrap().hlsl.matches("= xcl_x_rla_uniform(").count(), 2);

    let branched = prog(&[
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)], None), // targets v_min_u32 (past 8-byte VOP3 readlane)
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_MIN_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 8), ("VSRC1", 2)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", 128)], None),
    ]);
    assert_eq!(translate(&branched, &split).unwrap().hlsl.matches("= xcl_x_rla_uniform(").count(), 2);
}

/// GPU verification of folded split-wave64 reduction tails: 32-bit `v_min_u32` / `v_add_f32` /
/// `v_add_f16` (preserving high 16 bits), 64-bit `v_add_f64`, inactive lane 0 under `EXEC`, and
/// direct `(L, L^32)` pair.
#[test]
fn gpu_split_wave64_reduction_tails() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // 1. v_min_u32 reduction tail: v0[32] & v0[0] -> s10 = v0[32], s11 = min(v0[32], v0[0])
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 0), ("SRC1", 128 + 32)], None),
        ("V_MIN_U32", "ENC_VOP2", &[("VDST", 0), ("SRC0", 10), ("VSRC1", 0)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", 128)], None),
        // 2. v_add_f32 reduction tail (swapped operand order VOP3: v1, s12)
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", 128 + 32)], None),
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V + 1), ("SRC1", 12)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 1), ("SRC1", 128)], None),
        // 3. v_add_f16 reduction tail: high 16 bits of v2[0] must stay intact
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_ADD_F16", "ENC_VOP2", &[("VDST", 2), ("SRC0", 14), ("VSRC1", 2)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", V + 2), ("SRC1", 128)], None),
        // 4. 64-bit v_add_f64 reduction tail on v[4:5] -> s[16:17]
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 4), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V + 5), ("SRC1", 128 + 32)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", 16), ("SRC1", V + 4)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 4), ("SRC1", 128)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V + 5), ("SRC1", 128)], None),
        // 5. Direct (lane 5, lane 37) pair on v6 -> s18, s19
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 6), ("SRC1", 128 + 5)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 6), ("SRC1", 128 + 37)], None),
        // 6. Lane 0 masked off in EXEC: v_add_nc_u32 must NOT alter lane 0, so s21 stays old v7[0]
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 128)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 7), ("SRC1", 128 + 32)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 20), ("VSRC1", 7)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 21), ("SRC0", V + 7), ("SRC1", 128)], None),
    ]);
    if runner::split_wave64() {
        let t = translate(&code, &runner::options(Mode::Kernel)).unwrap();
        assert_eq!(t.hlsl.matches("= xcl_x_rla_uniform(").count(), 0, "all 6 patterns folded:\n{}", t.hlsl);
    }
    let mut st = KernelState::default();
    let d0 = 123456.75f64.to_bits();
    let d32 = 876543.125f64.to_bits();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = match l { 0 => 500, 32 => 123, _ => 999 };
        r[1] = match l { 0 => 1.5f32.to_bits(), 32 => 2.25f32.to_bits(), _ => 0 };
        // f16: 1.5 (0x3E00) + 2.5 (0x4100) = 4.0 (0x4400); high 16 bits = 0xCAFE on lane 0
        r[2] = match l { 0 => 0xCAFE_3E00, 32 => 0xBEEF_4100, _ => 0 };
        let db = match l { 0 => d0, 32 => d32, _ => 0 };
        r[4] = db as u32;
        r[5] = (db >> 32) as u32;
        r[6] = 0x6000 + l as u32;
        r[7] = match l { 0 => 0x1111, 32 => 0x2222, _ => 0 };
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    assert_eq!(r.sgpr[10], 123, "v_min_u32 s_tmp = v0[32]");
    assert_eq!(r.sgpr[11], 123, "v_min_u32 s_out = min(500, 123)");
    assert_eq!(r.vgpr[0][0], 123, "v0[0] updated in VGPR too");
    assert_eq!(r.sgpr[13], 3.75f32.to_bits(), "v_add_f32 reduction");
    assert_eq!(r.sgpr[15], 0xCAFE_4400, "v_add_f16 reduction preserves high 16 bits");
    let got_d = (r.sgpr[16] as u64) | ((r.sgpr[17] as u64) << 32);
    assert_eq!(got_d, (123456.75f64 + 876543.125f64).to_bits(), "v_add_f64 64-bit reduction");
    assert_eq!(r.sgpr[18], 0x6005, "direct pair lane 5");
    assert_eq!(r.sgpr[19], 0x6025, "direct pair lane 37");
    assert_eq!(r.sgpr[20], 0x2222, "inactive-lane-0 s_tmp still reads lane 32");
    assert_eq!(r.sgpr[21], 0x1111, "inactive-lane-0 s_out keeps unmodified old v7[0]");
}

/// Masks that are read whole after a write `x64_masks` must not take for an overwrite (an
/// SDWA compare into its SDST, `s_bitset1`, `s_movreld`) or through a read the operand
/// tables do not show (`s_movrels`, an SMEM register offset, an SDWA scalar src1): each is
/// still exchanged, and both halves end with the same SGPRs (the runner checks) and the
/// whole-wave values. A mask that goes nowhere is not exchanged.
#[test]
fn gpu_split_wave64_masks_read_whole_after_partial_writes() {
    if !gpu_ready() {
        return;
    }
    type Inst = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);
    // vcc / s[20:21] = lanes 0..39, or lane 2 only (low word 4)
    let cmp_vcc: Inst = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40));
    let cmp_s20: Inst = ("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 255), ("SRC1", V)], Some(40));
    let lane2_s20: Inst = ("V_CMP_EQ_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", 130)], None);
    let s20_dead: Inst = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)], None);
    let cases: Vec<(&str, Vec<Inst>)> = vec![
        (
            "SDWA compare into s[22:23]",
            vec![
                cmp_vcc,
                ("V_CMP_EQ_U32", "VOPC_VOP_SDWA_SDST_ENC", &[("SRC0", 0xF9), ("VSRC0", 0), ("VSRC1", 0), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("SD", 1), ("SDST", 22)], None),
                ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
            ],
        ),
        ("s_bitset1_b64", vec![cmp_s20, ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 191)], None), ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 20)], None)]),
        (
            "s_movrels_b64 from VCC",
            vec![
                cmp_vcc,
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 134)], None), // m0 = 6
                ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 100)], None), // s[0:1] = s[106:107]
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)], None),
                ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 0)], None),
            ],
        ),
        (
            "s_movreld_b64 past the mask",
            vec![
                cmp_s20,
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 132)], None), // m0 = 4
                ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 2)], None), // s[24:25] = s[2:3]
                ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 20)], None),
            ],
        ),
        ("SMEM register offset", vec![lane2_s20, ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 0), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 20)], None), s20_dead]),
        (
            "SDWA scalar src1",
            vec![
                lane2_s20,
                ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 9), ("SRC0", 0xF9), ("VSRC0", 0), ("VSRC1", 20), ("S1", 1), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6)], None),
                s20_dead,
            ],
        ),
    ];
    let mem = [11u32, 22, 33, 44];
    for (what, list) in cases {
        let code = prog(&list);
        if runner::split_wave64() {
            let t = translate(&code, &runner::options(Mode::Kernel)).unwrap();
            assert!(t.hlsl.contains("= xcl_x_ballot(and("), "{what}: the mask is exchanged");
        }
        let mut st = KernelState::default();
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            r[0] = l as u32;
        }
        st.sgpr[2] = 0x1234;
        let r = runner::run_code(&code, &st, &mem).unwrap_or_else(|e| panic!("{what}: {e}"));
        match what {
            "SDWA compare into s[22:23]" => {
                assert_eq!(r.sgpr[0], 40, "{what}");
                assert_eq!((r.sgpr[22], r.sgpr[23]), (!0, !0), "{what}");
            }
            "s_bitset1_b64" => assert_eq!(r.sgpr[0], 41, "{what}"),
            "s_movrels_b64 from VCC" => assert_eq!(r.sgpr[0], 40, "{what}"),
            "s_movreld_b64 past the mask" => {
                assert_eq!(r.sgpr[0], 40, "{what}");
                assert_eq!(r.sgpr[24], 0x1234, "{what}");
            }
            "SMEM register offset" => assert_eq!(r.sgpr[0], mem[1], "{what}"),
            _ => {
                for l in 0..64u32 {
                    assert_eq!(r.vgpr[l as usize][9], l + 4, "{what}: lane {l}");
                }
            }
        }
    }
    if runner::split_wave64() {
        // v_cmp into NULL: nothing to exchange
        let code = prog(&[("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 125), ("SRC0", V), ("SRC1", V + 1)], None)]);
        let t = translate(&code, &runner::options(Mode::Kernel)).unwrap();
        assert_eq!(t.hlsl.matches("= xcl_x_ballot(and(").count(), 0);
        assert_eq!(t.hlsl.matches("= xcl_x_ballot_local(and(").count(), 1);
    }
}

/// Four wave64s per group whose waves loop a different number of times (the counter-paired
/// form on 32-lane GPUs): each iteration a mask read lane by lane (`v_cndmask`) that only the
/// odd waves also read whole (`s_bcnt1`, on their own path), and a 64-bit carry chain (not
/// exchanged). The exchanged and the local masks interleave without the halves losing step.
#[test]
fn gpu_split_wave64_local_and_exchanged_masks_in_diverging_waves() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    let p = prog(&[
        // v0 = thread index in the group, v1 = wave, v2 = lane (as gpu_split_wave64_waves_diverge)
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", 1), ("SRC2", V)], None),
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 2), ("SRC1", 2), ("SRC2", V + 20)], None),
        ("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 134), ("VSRC1", 0)], None),
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 0)], Some(63)),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 1)], None),
        ("S_AND_B32", "ENC_SOP2", &[("SDST", 17), ("SSRC0", 10), ("SSRC1", 129)], None), // odd wave
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None), // n = 2 (w + 1)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None), // sum (odd waves)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None), // i
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 128)], None), // per-lane count
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 2)], Some(0x9E37_79B9)),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)], None),
        // loop: k = 5 i + 3
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 12), ("SSRC1", 133)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 13), ("SSRC1", 131)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 13), ("VSRC1", 2)], None), // vcc = lanes < k
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 129), ("VSRC1", 4)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 5)], None), // count += lane < k
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 17), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)], None), // even waves: skip
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 106)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 14)], None),
        // v[8:9] += 0x9E3779B9 + lane, through s[20:21]
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 8), ("SDST", 20), ("SRC0", V + 8), ("SRC1", V + 6)], None),
        ("V_ADD_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 9), ("SDST", 20), ("SRC0", V + 9), ("SRC1", 128), ("SRC2", 20)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 10)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF0)], None), // → loop
        // out[group · 256 + thread] = (count, sum, v8, v9)
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 25), ("SRC0", 11)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", V + 8)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 27), ("SRC0", V + 9)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 4), ("SSRC1", 136)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 16), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 132), ("VSRC1", 30)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 0), ("VSRC1", 30)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 31), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", &[("ADDR", 30), ("DATA", 24), ("SADDR", 0x7D)], None),
    ]);
    for dims in [[256u32, 1, 1], [8, 8, 4]] {
        let threads = dims.iter().product::<u32>();
        let cs = CsInterface { threads: dims, user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 3 };
        let opt = on_gpu(Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() });
        let t = translate(&p, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        if runner::split_wave64() {
            assert_eq!(t.hlsl.matches("= xcl_x_ballot(and(").count(), 1, "the v_cmp: read whole by the odd waves");
            assert_eq!(t.hlsl.matches("= xcl_x_ballot_local(and(").count(), 2, "the carries");
        }
        let dxil = direct_or(&p, &opt, dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}")));
        let base = 0x100usize;
        let mut io = vec![0u32; 128];
        io[0] = base as u32;
        io[1] = dims[0];
        io[2] = dims[0] * dims[1];
        let groups = 3usize;
        let out = runner::dispatch(&dxil, [groups as u32, 1, 1], &io, &vec![0u32; (base + groups * 256 * 16) / 4]).unwrap();
        for g in 0..groups {
            for tid in 0..threads as usize {
                let (w, l) = (tid / 64, (tid % 64) as u32);
                let n = 2 * (w as u32 + 1);
                let count = (0..n).filter(|i| 5 * i + 3 > l).count() as u32;
                let sum = if w % 2 == 1 { (0..n).map(|i| (5 * i + 3).min(64)).sum() } else { 0 };
                let acc = n as u64 * (0x9E37_79B9u64 + l as u64);
                let at = (base + (g * 256 + tid) * 16) / 4;
                let what = format!("{dims:?}, group {g} thread {tid} (wave {w}, lane {l})");
                assert_eq!(out[at], count, "{what}: per-lane count");
                assert_eq!(out[at + 1], sum, "{what}: s_bcnt1 sum");
                assert_eq!((out[at + 2], out[at + 3]), (acc as u32, (acc >> 32) as u32), "{what}: 64-bit carry chain");
            }
        }
    }
}

/// A group of four wave64s whose waves take different paths: each loops 2 · (wave + 1)
/// times through a ballot, s_bcnt1 and a readlane, then exchanges lanes across its halves
/// with ds_bpermute. Split (32-lane GPUs) this is the counter-paired form – a group barrier
/// per exchange would deadlock or mix the waves up. 3-D groups too: NVIDIA does not form its
/// waves from consecutive thread indices there (8 × 8 × 4), the console does.
#[test]
fn gpu_split_wave64_waves_diverge() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    let p = prog(&[
        // v0 = thread index in the group: v0 + v1 · s1 + v2 · s2 (s1 = x, s2 = x · y)
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", 1), ("SRC2", V)], None),
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 2), ("SRC1", 2), ("SRC2", V + 20)], None),
        ("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 134), ("VSRC1", 0)], None), // wave
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 0)], Some(63)), // lane
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 1)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None), // n = 2 (w + 1)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None), // sum
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None), // i
        // loop: k = 5 i + 3; sum += popcount(lanes < k) + v0 of lane k
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 12), ("SSRC1", 133)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 13), ("SSRC1", 131)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 13), ("VSRC1", 2)], None),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 106)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 14)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", V), ("SRC1", 13)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 15)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 10)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF5)], None), // → loop
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 2)], Some(33)),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 130), ("VSRC1", 3)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 3), ("DATA0", 0)], None), // v0 of lane ^ 33
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 11)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 4), ("SSRC1", 136)], None), // group · 256
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 16), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 131), ("VSRC1", 7)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 0), ("VSRC1", 7)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 7), ("DATA", 5), ("SADDR", 0x7D)], None),
    ]);
    for dims in [[256u32, 1, 1], [64, 1, 1], [8, 8, 4], [16, 4, 1]] {
        let threads = dims.iter().product::<u32>();
        let cs = CsInterface { threads: dims, user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 3 };
        let opt = on_gpu(Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() });
        let t = translate(&p, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        if runner::split_wave64() && std::env::var("XCL_WAVE64_SPIN").map_or(true, |v| v.trim() != "0") {
            assert!(t.hlsl.contains("[WaveSize(32)]") && t.hlsl.contains("#define XCL_X_SPIN 1"), "{dims:?}");
        }
        let dxil = direct_or(&p, &opt, dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}")));
        let base = 0x100usize;
        let mut io = vec![0u32; 128];
        io[0] = base as u32;
        io[1] = dims[0];
        io[2] = dims[0] * dims[1];
        let groups = 3usize;
        let out = runner::dispatch(&dxil, [groups as u32, 1, 1], &io, &vec![0u32; (base + groups * 256 * 8) / 4]).unwrap();
        for g in 0..groups {
            for tid in 0..threads as usize {
                let (w, l) = (tid / 64, tid % 64);
                let want: u32 = (0..2 * (w + 1)).map(|i| (5 * i + 3).min(64) as u32 + (w * 64 + 5 * i + 3) as u32).sum();
                let at = (base + (g * 256 + tid) * 8) / 4;
                assert_eq!(out[at], want, "{dims:?}, group {g} thread {tid}: loop of wave {w}");
                assert_eq!(out[at + 1], (w * 64 + (l ^ 33)) as u32, "{dims:?}, group {g} thread {tid}: ds_bpermute");
            }
        }
    }
}

/// What the split wave64 form costs (`cargo test --release -- --ignored bench_split_wave64
/// --nocapture`): a loop of a few VALU instructions and one mask (`v_cmp`, an exchange in the
/// split form), plus a readfirstlane (another exchange) in the second variant, as a wave32
/// program (no exchanges) and as a wave64 program – natively on a GPU with 64-lane waves,
/// split on one with 32-lane waves only. GPU time of the dispatch (timestamps), median of 9.
#[test]
#[ignore]
fn bench_split_wave64() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};
    let body = |readfirstlane: bool| {
        let mut l: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 10), ("SSRC0", 128)], None),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 2), ("SRC0", V)], None),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 4), ("SRC0", 128)], None),
            // loop: an LCG step, a mask from it, a conditional count
            ("V_MUL_LO_U32", "ENC_VOP3", vec![("VDST", 2), ("SRC0", V + 2), ("SRC1", 255)], Some(1_103_515_245)),
            ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 255), ("VSRC1", 2)], Some(12345)),
            ("V_LSHRREV_B32", "ENC_VOP2", vec![("VDST", 3), ("SRC0", 144), ("VSRC1", 2)], None),
            ("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 255), ("VSRC1", 3)], Some(0x4000)),
            ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 5), ("SRC0", 129), ("VSRC1", 4)], None),
            ("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", V + 4), ("VSRC1", 5)], None),
        ];
        let mut back = 0xFFF3u64; // → the loop start: 48 bytes of loop body before the branch
        if readfirstlane {
            l.push(("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 11), ("SRC0", V + 3)], None));
            l.push(("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", 11), ("VSRC1", 4)], None));
            back -= 2;
        }
        l.extend([
            ("S_ADD_U32", "ENC_SOP2", vec![("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
            ("S_CMP_LG_U32", "ENC_SOPC", vec![("SSRC0", 10), ("SSRC1", 1)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", vec![("SIMM16", back)], None),
            ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
            ("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 6), ("SRC0", 0), ("VSRC1", 6)], None),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 128)], None),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", vec![("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
        ]);
        let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = l.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
        prog(&refs)
    };
    let iterations = 32768u32;
    println!("GPU: {} ({} lanes per wave at most)", runner::adapter().unwrap(), if runner::wave64_supported() == Ok(true) { 64 } else { 32 });
    for readfirstlane in [false, true] {
        let p = body(readfirstlane);
        for threads in [64u32, 256] {
            let mut ms = Vec::new();
            let mut exchanges = 0;
            for wave64 in [false, true] {
                let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
                let opt = on_gpu(Options { mode: Mode::Compute(cs), wave64, ..Default::default() });
                let t = translate(&p, &opt).unwrap();
                // the loop's exchanges: masks read whole (`x64_masks`: not the v_cndmask
                // selector, unless `XCL_X64_ALL_MASKS`) and the readfirstlane
                exchanges = t.hlsl.matches("= xcl_x_ballot(and(").count() + t.hlsl.matches("xcl_x_rla_uniform(").count().saturating_sub(1);
                assert!(t.complete(), "{:?}", t.unsupported_why);
                let dxil = direct_or(&p, &opt, dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}")));
                let groups = (1u32 << 18) / threads;
                let mut io = vec![0u32; 128];
                io[0] = 0;
                io[1] = iterations;
                let mem = vec![0u32; threads as usize];
                // GPU time of each dispatch (timestamps: no pipeline creation, uploads or CPU
                // wait in it), the median of 9
                let mut t = runner::dispatch_timed(&dxil, [groups, 1, 1], &io, &mem, 9).unwrap();
                t.sort_by(f64::total_cmp);
                ms.push(t[t.len() / 2]);
            }
            // the form `x64_spin` picks: counters for groups of more than one console wave,
            // and by default for one as well (`XCL_WAVE64_SPIN=0`: a group barrier)
            let spin = threads > 64 || std::env::var("XCL_WAVE64_SPIN").map_or(true, |v| v.trim() != "0");
            let form = if runner::split_wave64() { if spin { "split, counter-paired" } else { "split, barrier" } } else { "native" };
            println!(
                "v_cmp + v_cndmask{}: {exchanges} exchange(s) / iteration, {threads} threads per group, 2^18 threads x {iterations} iterations: wave32 {:.1} ms, wave64 ({form}) {:.1} ms = {:.2}x",
                if readfirstlane { " + v_readfirstlane" } else { "" },
                ms[0],
                ms[1],
                ms[1] / ms[0]
            );
        }
    }
}

/// TG_SIZE (the SGPR after the group ids): the group's wave count | the thread's wave index
/// << 6. A group that is not a multiple of the wave size has a last, partial wave: 96 threads
/// are 2 console waves of 64 (3 of 32), not 96 / 64 = 1.
#[test]
fn gpu_compute_tg_size_counts_a_partial_last_wave() {
    if !gpu_ready() {
        return;
    }
    // global_store_dword [v0 * 4], s2 (TG_SIZE: user SGPRs s0, s1, no group ids)
    let p = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 2)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
    ]);
    for (threads, wave64) in [(96u32, true), (96, false), (64, true), (40, false), (128, true)] {
        let lanes = if wave64 { 64 } else { 32 };
        let cs = super::CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: true, tid_components: 1 };
        let opt = on_gpu(Options { mode: Mode::Compute(cs), wave64, ..Default::default() });
        let t = translate(&p, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        // `XCL_DIRECT_DXIL=1`: the direct translator's kernel, which sets up TG_SIZE on its own
        let dxil = direct_or(&p, &opt, super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl)));
        let mem = runner::dispatch(&dxil, [1, 1, 1], &[0u32; 128], &vec![0xFFFF_FFFFu32; threads as usize]).unwrap();
        let waves = threads.div_ceil(lanes);
        for (k, &got) in mem.iter().take(threads as usize).enumerate() {
            assert_eq!(got, waves | (k as u32 / lanes) << 6, "{threads} threads, wave{lanes}: thread {k}");
        }
    }
}

/// A loop that only ends when its counter wraps to 0, then global_store_dword [v0 * 4], the
/// counter: the passes a lane ran before the loop was ended.
fn endless_loop_program() -> Vec<u8> {
    prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 10)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
    ])
}

/// The same loop inside one of three passes: global_store_dword [v0 * 4], the sum of the
/// inner loop's passes over its three entries.
fn endless_loop_entered_three_times() -> Vec<u8> {
    prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None),
        // the outer loop: three passes
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 10)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 11), ("SSRC1", 131)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF8)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 12)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
    ])
}

/// `IO_LOOP_BUDGET`: one thread group's share of `DISPATCH_LOOP_PASSES` – all of it for a
/// dispatch of one group, less the more groups there are, and never 0 ("not known").
#[test]
fn dispatch_loop_budget_shrinks_with_the_groups() {
    use super::{dispatch_loop_budget as budget, DISPATCH_LOOP_PASSES};
    assert_eq!(budget([1, 1, 1]) as u64, DISPATCH_LOOP_PASSES);
    assert_eq!(budget([16, 16, 16]) as u64, DISPATCH_LOOP_PASSES / 4096);
    // 64 threads a group: 1024 passes a lane
    assert_eq!(budget([16, 16, 16]) / 64, 1024);
    // a count of 0 dispatches nothing: as 1
    assert_eq!(budget([0, 8, 0]), budget([1, 8, 1]));
    assert_eq!(budget([65_535, 65_535, 65_535]), 1);
}

/// Only the loops of a compute program count their own passes against the dispatch's share:
/// the HLSL of every other program (and with it the DXIL cache's key) is as it was.
#[test]
fn only_compute_programs_with_a_loop_read_the_dispatch_loop_budget() {
    let cs = super::CsInterface { threads: [8, 4, 2], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
    let compute = Options { mode: Mode::Compute(cs), wave64: false, ..Default::default() };
    let looping = translate(&endless_loop_program(), &compute).unwrap();
    assert!(looping.complete(), "{:?}", looping.unsupported_why);
    let read = format!("uint xcl_gb = io.Load({}u); if (xcl_gb != 0u) xcl_cap = clamp(xcl_gb / 64u, 1024u, 1u << 16);", super::IO_LOOP_BUDGET * 4);
    assert_eq!(looping.hlsl.matches(&read).count(), 1, "{}", looping.hlsl);
    assert_eq!(looping.hlsl.matches("static uint xcl_cap = 1u << 16;").count(), 1, "{}", looping.hlsl);
    assert!(looping.hlsl.contains("uint xcl_n1 = xcl_cap;"), "{}", looping.hlsl);
    assert!(looping.hlsl.contains("if (xcl_n1 == 0u) break; xcl_n1--; if (xcl_budget == 0u) break; xcl_budget--;"), "{}", looping.hlsl);
    // each loop has a count of its own
    let nested = translate(&endless_loop_entered_three_times(), &compute).unwrap();
    assert!(nested.hlsl.contains("uint xcl_n1 = xcl_cap;") && nested.hlsl.contains("uint xcl_n2 = xcl_cap;"), "{}", nested.hlsl);
    // the same program as a kernel, and a compute program without a loop
    let kernel = translate(&endless_loop_program(), &Options { wave64: false, ..Default::default() }).unwrap();
    assert!(kernel.hlsl.contains("xcl_budget--") && !kernel.hlsl.contains("xcl_cap") && !kernel.hlsl.contains("xcl_gb"), "{}", kernel.hlsl);
    let straight = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 2)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
    ]);
    let straight = translate(&straight, &compute).unwrap();
    assert!(!straight.hlsl.contains("xcl_cap") && !straight.hlsl.contains("xcl_gb") && !straight.hlsl.contains("xcl_budget--"), "{}", straight.hlsl);
}

/// A compute program's loop that never meets its exit runs the passes its dispatch gives one
/// entry of a loop: `IO_LOOP_BUDGET` among the group's threads, between 1024 and 65 536 a
/// lane, the same in every thread (both halves of a split wave64). A dispatch that does not
/// say (0) leaves it the lane's 65 536. Entered three times the loop runs them three times:
/// loops inside one another are not cut to one loop's share. The HLSL path: the direct
/// translator has its own fixed budget.
#[test]
fn gpu_compute_loop_passes_follow_the_dispatch() {
    if !gpu_ready() {
        return;
    }
    let threads = 64u32;
    let run = |p: &[u8], wave64: bool, slot: u32| -> u32 {
        let cs = super::CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
        let opt = on_gpu(Options { mode: Mode::Compute(cs), wave64, ..Default::default() });
        let t = translate(p, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let mut io = [0u32; 128];
        io[super::IO_LOOP_BUDGET] = slot;
        let mem = runner::dispatch(&dxil, [1, 1, 1], &io, &vec![0xFFFF_FFFFu32; threads as usize]).unwrap();
        for (k, &got) in mem.iter().take(threads as usize).enumerate() {
            assert_eq!(got, mem[0], "wave{}, slot {slot}: thread {k}", if wave64 { 64 } else { 32 });
        }
        mem[0]
    };
    for wave64 in [false, true] {
        // (the slot, passes): not known; 5000 a lane; shares below and above the bounds
        for (slot, passes) in [(0u32, 65_536u32), (threads * 5000, 5000), (1, 1024), (u32::MAX, 65_536)] {
            assert_eq!(run(&endless_loop_program(), wave64, slot), passes, "wave64 {wave64}, slot {slot}");
        }
        // three entries of 1024 passes; without a share the first entry spends the lane's
        // budget (but for the outer loop's own pass) and ends the program
        assert_eq!(run(&endless_loop_entered_three_times(), wave64, 1), 3 * 1024, "wave64 {wave64}");
        assert_eq!(run(&endless_loop_entered_three_times(), wave64, 0), 65_535, "wave64 {wave64}");
    }
}

/// v_fract_f32 at the edges, as RDNA computes it (x - floor(x) kept below 1.0): ±inf gives
/// NaN and a NaN stays NaN – a `min()` clamp returned 0.99999994 there on NVIDIA (HLSL min
/// returns its non-NaN operand); a tiny negative value rounds to the largest float below 1.
#[test]
fn gpu_fract_f32_edges() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[("V_FRACT_F32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 1)], None)]);
    let inputs = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -0.0, 1.5, -0.25, -(2f32.powi(-25)), 7.0];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = inputs[l % inputs.len()].to_bits();
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let (x, got) = (inputs[l % inputs.len()], f32::from_bits(r.vgpr[l][2]));
        if x.is_infinite() || x.is_nan() {
            assert!(got.is_nan(), "lane {l}: fract({x}) = {got}, want NaN");
        } else if x == -(2f32.powi(-25)) {
            assert_eq!(r.vgpr[l][2], 0x3F7F_FFFF, "lane {l}: fract({x}) below 1.0");
        } else {
            assert_eq!(got.to_bits(), (x - x.floor()).to_bits(), "lane {l}: fract({x})");
        }
    }
}

/// f64 compares with NaN operands, both the ordered ones and the negated ones that are true
/// when unordered, bit for bit against IEEE.
#[test]
fn gpu_f64_compares_with_nan() {
    for wave64 in gpu_waves() {
        f64_compares_with_nan(wave64);
    }
}

fn f64_compares_with_nan(wave64: bool) {
    const OPS: [&str; 8] = ["O", "U", "NEQ", "NLT", "NGE", "EQ", "LT", "LG"];
    let insts: Vec<(String, u64)> = OPS.iter().enumerate().map(|(i, op)| (format!("V_CMP_{op}_F64"), 20 + 2 * i as u64)).collect();
    let fields: Vec<[(&str, u64); 3]> = insts.iter().map(|(_, d)| [("VDST", *d), ("SRC0", V), ("SRC1", V + 2)]).collect();
    let list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> =
        insts.iter().zip(&fields).map(|((n, _), f)| (n.as_str(), "ENC_VOP3", &f[..], None)).collect();
    let p = prog(&list);
    let nan = f64::from_bits(0x7FF8_0000_0000_1234);
    let pairs = [(1.0, 1.0), (nan, 1.0), (2.0, nan), (-0.0, 0.0), (f64::INFINITY, f64::INFINITY), (nan, nan), (1.0, 2.0)];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (a, b): (f64, f64) = pairs[l % pairs.len()];
        (r[0], r[1], r[2], r[3]) = (a.to_bits() as u32, (a.to_bits() >> 32) as u32, b.to_bits() as u32, (b.to_bits() >> 32) as u32);
    }
    let (r, lanes) = run_wave(&p, &st, wave64);
    let mut bad = Vec::new();
    for (i, op) in OPS.iter().enumerate() {
        let mask = r.sgpr[20 + 2 * i] as u64 | (r.sgpr[21 + 2 * i] as u64) << 32;
        for l in 0..lanes {
            let (a, b) = pairs[l % pairs.len()];
            let want = match *op {
                "O" => !a.is_nan() && !b.is_nan(),
                "U" => a.is_nan() || b.is_nan(),
                "NEQ" => !(a == b),
                "NLT" => !(a < b),
                "NGE" => !(a >= b),
                "EQ" => a == b,
                "LT" => a < b,
                _ => a < b || a > b,
            };
            if (mask >> l & 1 == 1) != want {
                bad.push(format!("v_cmp_{}_f64({a:e}, {b:e}) = {}, want {want}", op.to_lowercase(), !want));
            }
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

/// Every f32 compare with NaN operands against IEEE. Before the explicit NaN tests, lt, ge
/// and lg came back true and nlg false for a NaN operand on an AMD RX 7900 XTX. The driver
/// gave lt the result of the nge before it, as `float_compare` describes.
#[test]
fn gpu_f32_compares_with_nan() {
    for wave64 in gpu_waves() {
        f32_compares_with_nan(wave64);
    }
}

fn f32_compares_with_nan(wave64: bool) {
    const OPS: [&str; 12] = ["O", "U", "NEQ", "NLT", "NGE", "EQ", "LT", "LG", "LE", "GT", "GE", "NLG"];
    let insts: Vec<(String, u64)> = OPS.iter().enumerate().map(|(i, op)| (format!("V_CMP_{op}_F32"), 20 + 2 * i as u64)).collect();
    let fields: Vec<[(&str, u64); 3]> = insts.iter().map(|(_, d)| [("VDST", *d), ("SRC0", V), ("SRC1", V + 1)]).collect();
    let list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> =
        insts.iter().zip(&fields).map(|((n, _), f)| (n.as_str(), "ENC_VOP3", &f[..], None)).collect();
    let p = prog(&list);
    let nan = f32::from_bits(0x7FC0_1234);
    let pairs = [(1.0f32, 1.0f32), (nan, 1.0), (2.0, nan), (-0.0, 0.0), (f32::INFINITY, f32::INFINITY), (nan, nan), (1.0, 2.0)];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (a, b) = pairs[l % pairs.len()];
        (r[0], r[1]) = (a.to_bits(), b.to_bits());
    }
    let (r, lanes) = run_wave(&p, &st, wave64);
    let mut bad = Vec::new();
    for (i, op) in OPS.iter().enumerate() {
        let mask = r.sgpr[20 + 2 * i] as u64 | (r.sgpr[21 + 2 * i] as u64) << 32;
        for l in 0..lanes {
            let (a, b) = pairs[l % pairs.len()];
            let want = match *op {
                "O" => !a.is_nan() && !b.is_nan(),
                "U" => a.is_nan() || b.is_nan(),
                "NEQ" => !(a == b),
                "NLT" => !(a < b),
                "NGE" => !(a >= b),
                "EQ" => a == b,
                "LT" => a < b,
                "LG" => a < b || a > b,
                "LE" => a <= b,
                "GT" => a > b,
                "GE" => a >= b,
                _ => !(a < b || a > b),
            };
            if (mask >> l & 1 == 1) != want {
                bad.push(format!("v_cmp_{}_f32({a:e}, {b:e}) = {}, want {want}", op.to_lowercase(), !want));
            }
        }
    }
    bad.sort();
    bad.dedup();
    assert!(bad.is_empty(), "wave{lanes}:\n{}", bad.join("\n"));
}

/// xorshift64, the fixed-seed generator of the randomized GPU tests
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

/// A random f64 weighted towards the edges: specials, any bit pattern, values next to
/// integers up to 2^64, denormals, magnitudes around 2^±65, small values on 1/8 steps.
fn rand_f64(r: &mut Rng) -> f64 {
    const SPECIAL: [u64; 12] = [0, 1 << 63, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000, 0x7FF8_0000_0000_0000,
        0x7FF0_0000_0000_0001, 0xFFF8_0000_0000_1234, 1, 0x000F_FFFF_FFFF_FFFF, 0x7FEF_FFFF_FFFF_FFFF, 0x0010_0000_0000_0000, 0x8000_0000_0000_0001];
    match r.below(6) {
        0 => f64::from_bits(SPECIAL[r.below(12) as usize]),
        1 => f64::from_bits(r.next()),                                        // any bits
        2 => { // near an integer of any size up to 2^64
            let k = (r.next() >> r.below(64)) as f64;
            let d = [0.0, 0.5, -0.5, 0.25, 1e-9, -1e-9, 0.9999999999][r.below(7) as usize];
            let v = k + d;
            if r.below(2) == 0 { -v } else { v }
        }
        3 => f64::from_bits(r.next() & 0x800F_FFFF_FFFF_FFFF),                 // denormal
        4 => { let e = 1023 + r.below(130) - 65; f64::from_bits((e << 52) | (r.next() & 0x800F_FFFF_FFFF_FFFF)) } // |x| around 2^±65
        _ => (r.below(2000) as f64 - 1000.0) / 8.0,                           // small values on 1/8 steps
    }
}

/// 2^e for a normal e
fn p2f(e: i32) -> f64 { f64::from_bits(((e + 1023) as u64) << 52) }

/// x · 2^n on the CPU with a single rounding, since each step but the last is exact. None when
/// that is not possible.
fn ldexp_ref(x: f64, n: i32) -> Option<f64> {
    if x.is_nan() || x.is_infinite() || x == 0.0 { return Some(x); }
    if n > 4000 { return Some(f64::INFINITY.copysign(x)); }
    if n < -4000 { return Some(x * 0.0); }
    if (-1022..=1023).contains(&n) { return Some(x * p2f(n)); }
    if n > 1023 && n <= 2046 { return Some((x * p2f(1023)) * p2f(n - 1023)); }
    if (-2044..-1022).contains(&n) && (1.0..2.0).contains(&x.abs()) { return Some((x * p2f(-1022)) * p2f(n + 1022)); }
    None
}

/// The f64 math against the CPU on random inputs, one wave per dispatch in 50 dispatches, or
/// n dispatches with XCL_FUZZ_ITERS=n, as wave64 and as wave32. floor, ceil, trunc, fract and the i32 and u32
/// conversions must match bit for bit, and so must ldexp wherever the CPU can compute it with
/// one rounding. sqrt must be within 1 ulp and rsq within 2.
#[test]
fn gpu_f64_math_random() {
    for wave64 in gpu_waves() {
        f64_math_random(wave64);
    }
}

fn f64_math_random(wave64: bool) {
    let iters: usize = std::env::var("XCL_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let p = prog(&[
        ("V_FLOOR_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None),
        ("V_TRUNC_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 8), ("SRC0", V)], None),
        ("V_CVT_I32_F64", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("V_CVT_U32_F64", "ENC_VOP1", &[("VDST", 11), ("SRC0", V)], None),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 12), ("SRC0", V)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 14), ("SRC0", V)], None),
        ("V_LDEXP_F64", "ENC_VOP3", &[("VDST", 16), ("SRC0", V), ("SRC1", V + 20)], None),
        ("V_TRUNC_F64", "ENC_VOP3", &[("VDST", 22), ("SRC0", V), ("NEG", 1)], None),
        ("V_FLOOR_F64", "ENC_VOP3", &[("VDST", 24), ("SRC0", V), ("ABS", 1)], None),
    ]);
    let (dxil, mem) = kernel_wave(&p, wave64);
    let lanes = if wave64 { 64 } else { 32 };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let same = |g: f64, w: f64| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
    let ulps = |g: f64, w: f64| -> u64 {
        if same(g, w) { 0 } else if g.is_nan() || w.is_nan() || g.is_sign_negative() != w.is_sign_negative() { u64::MAX } else { g.to_bits().abs_diff(w.to_bits()) }
    };
    let (mut checked, mut ldexp_checked, mut sqrt_exact, mut sqrt_1ulp, mut rsq_hist) = (0usize, 0usize, 0usize, 0usize, [0usize; 4]);
    let mut bad: Vec<String> = Vec::new();
    for _ in 0..iters {
        let mut st = KernelState::default();
        let mut xs = Vec::new();
        for r in st.vgpr.iter_mut() {
            let x = rand_f64(&mut rng);
            let n: i32 = match rng.below(5) {
                0 => rng.below(2047) as i32 - 1023,
                1 => rng.below(1025) as i32 + 1023,
                2 => -(rng.below(1023) as i32) - 1022,
                3 => rng.next() as i32,
                _ => rng.below(120) as i32 - 60,
            };
            (r[0], r[1], r[20]) = (x.to_bits() as u32, (x.to_bits() >> 32) as u32, n as u32);
            xs.push((x, n));
        }
        let r = runner::run(&dxil, &st, &mem).unwrap();
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            let (x, n) = xs[l];
            checked += 1;
            let fract = if x.is_finite() { (x - x.floor()).min(f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)) } else { f64::NAN };
            for (name, g, w) in [("floor", pair(v, 2), x.floor()), ("ceil", pair(v, 4), x.ceil()), ("trunc", pair(v, 6), x.trunc()),
                                 ("fract", pair(v, 8), fract), ("trunc(-x)", pair(v, 22), (-x).trunc()), ("floor(|x|)", pair(v, 24), x.abs().floor())] {
                if !same(g, w) {
                    bad.push(format!("{name}({x:e} / {:#018x}) = {g:e}, want {w:e}", x.to_bits()));
                }
            }
            if v[10] as i32 != x as i32 { bad.push(format!("cvt_i32({x:e}) = {}, want {}", v[10] as i32, x as i32)); }
            if v[11] != x as u32 { bad.push(format!("cvt_u32({x:e}) = {}, want {}", v[11], x as u32)); }
            let (gs, ws) = (pair(v, 12), x.sqrt());
            match ulps(gs, ws) { 0 => sqrt_exact += 1, 1 => sqrt_1ulp += 1, _ => bad.push(format!("sqrt({x:e} / {:#018x}) = {gs:e}, want {ws:e}", x.to_bits())) }
            let u = ulps(pair(v, 14), 1.0 / x.sqrt());
            if u <= 2 { rsq_hist[u as usize] += 1 } else { rsq_hist[3] += 1; bad.push(format!("rsq({x:e}) = {:e}, want {:e}", pair(v, 14), 1.0 / x.sqrt())) }
            if let Some(w) = ldexp_ref(x, n) {
                ldexp_checked += 1;
                if !same(pair(v, 16), w) { bad.push(format!("ldexp({x:e} / {:#018x}, {n}) = {:e}, want {w:e}", x.to_bits(), pair(v, 16))); }
            }
        }
    }
    println!("wave{lanes}: lanes {checked}, ldexp checked {ldexp_checked}, sqrt exact {sqrt_exact} / 1 ulp {sqrt_1ulp}, rsq ulps 0/1/2/>2 {rsq_hist:?}");
    bad.sort(); bad.dedup();
    println!("{} distinct failures", bad.len());
    for b in bad.iter().take(40) { println!("  {b}"); }
    assert!(bad.is_empty());
}

fn f16_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (h >> 10) & 0x1F;
    let m = (h & 0x3FF) as f64;
    match e {
        0 => s * m * (2.0f64).powi(-24),
        31 => if m == 0.0 { s * f64::INFINITY } else { f64::NAN },
        _ => s * (1.0 + m / 1024.0) * (2.0f64).powi(e as i32 - 15),
    }
}

/// All 16 float compares of f32, f64 and f16 and v_cmpx_lt against the CPU on random operand
/// pairs, a quarter of them equal. The VOP3 forms write SGPR pairs and the VOPC forms VCC.
/// One wave of pairs per dispatch in 50 dispatches, or n dispatches with XCL_FUZZ_ITERS=n,
/// as wave64 and as wave32. An f32
/// result that only matches with f32 denormals flushed to zero is counted and does not fail
/// the test. That comes from the shader's denormal mode, not from the compare.
#[test]
fn gpu_float_compares_random() {
    for wave64 in gpu_waves() {
        float_compares_random(wave64);
    }
}

fn float_compares_random(wave64: bool) {
    let iters: usize = std::env::var("XCL_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    const OPS: [&str; 16] = ["F", "LT", "EQ", "LE", "GT", "LG", "GE", "O", "U", "NGE", "NLG", "NGT", "NLE", "NEQ", "NLT", "TRU"];
    let want_of = |op: &str, a: f64, b: f64| match op {
        "F" => false, "LT" => a < b, "EQ" => a == b, "LE" => a <= b, "GT" => a > b, "LG" => a < b || a > b, "GE" => a >= b,
        "O" => !a.is_nan() && !b.is_nan(), "U" => a.is_nan() || b.is_nan(), "NGE" => !(a >= b), "NLG" => !(a < b || a > b),
        "NGT" => !(a > b), "NLE" => !(a <= b), "NEQ" => !(a == b), "NLT" => !(a < b), _ => true,
    };
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut total_bad = 0usize;
    for ty in ["F32", "F64", "F16"] {
        let (s0, s1) = if ty == "F64" { (V + 2, 4u64) } else { (V, 1u64) };
        let names: Vec<String> = OPS.iter().map(|o| format!("V_CMP_{o}_{ty}")).collect();
        let f3: Vec<[(&str, u64); 3]> = (0..16).map(|i| [("VDST", 20 + 2 * i as u64), ("SRC0", s0), ("SRC1", V + s1)]).collect();
        let fc: Vec<[(&str, u64); 2]> = (0..16).map(|_| [("SRC0", s0), ("VSRC1", s1)]).collect();
        let fm: Vec<[(&str, u64); 2]> = (0..16).map(|i| [("SDST", 60 + 2 * i as u64), ("SSRC0", 106)]).collect();
        let cmpx = format!("V_CMPX_LT_{ty}");
        let fx = [("SRC0", s0), ("VSRC1", s1)];
        let mut list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = Vec::new();
        for i in 0..16 { list.push((names[i].as_str(), "ENC_VOP3", &f3[i][..], None)); }
        for i in 0..16 { list.push((names[i].as_str(), "ENC_VOPC", &fc[i][..], None)); list.push(("S_MOV_B64", "ENC_SOP1", &fm[i][..], None)); }
        list.push((cmpx.as_str(), "ENC_VOPC", &fx[..], None));
        let p = prog(&list);
        let (dxil, mem) = kernel_wave(&p, wave64);
        let n = if wave64 { 64 } else { 32 };
        let mut bad: Vec<String> = Vec::new();
        let mut lanes = 0usize;
        let mut denorm = 0usize;
        for _ in 0..iters {
            let mut st = KernelState::default();
            let mut ab = Vec::new();
            for r in st.vgpr.iter_mut() {
                let (a, b) = match ty {
                    "F64" => {
                        let a = rand_f64(&mut rng);
                        let b = if rng.below(4) == 0 { a } else { rand_f64(&mut rng) };
                        (r[2], r[3], r[4], r[5]) = (a.to_bits() as u32, (a.to_bits() >> 32) as u32, b.to_bits() as u32, (b.to_bits() >> 32) as u32);
                        (a, b)
                    }
                    "F32" => {
                        let g = |rng: &mut Rng| -> f32 { match rng.below(3) { 0 => rand_f64(rng) as f32, 1 => f32::from_bits(rng.next() as u32),
                            _ => f32::from_bits([0u32, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0x7F80_0001, 0xFFC1_2345, 1][rng.below(8) as usize]) } };
                        let a = g(&mut rng);
                        let b = if rng.below(4) == 0 { a } else { g(&mut rng) };
                        (r[0], r[1]) = (a.to_bits(), b.to_bits());
                        (a as f64, b as f64)
                    }
                    _ => {
                        let g = |rng: &mut Rng| -> u16 { if rng.below(3) == 0 { [0u16, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE12, 1, 0x3C00][rng.below(9) as usize] } else { rng.next() as u16 } };
                        let a = g(&mut rng);
                        let b = if rng.below(4) == 0 { a } else { g(&mut rng) };
                        // the high halves hold junk: compares read the low half
                        (r[0], r[1]) = (a as u32 | (rng.next() as u32) << 16, b as u32 | (rng.next() as u32) << 16);
                        (f16_to_f64(a), f16_to_f64(b))
                    }
                };
                ab.push((a, b));
            }
            let r = runner::run(&dxil, &st, &mem).unwrap();
            ab.truncate(n);
            lanes += n;
            for (i, op) in OPS.iter().enumerate() {
                let m3 = r.sgpr[20 + 2 * i] as u64 | (r.sgpr[21 + 2 * i] as u64) << 32;
                let mc = r.sgpr[60 + 2 * i] as u64 | (r.sgpr[61 + 2 * i] as u64) << 32;
                for (l, &(a, b)) in ab.iter().enumerate() {
                    let w = want_of(op, a, b);
                    let ftz = |x: f64| if ty == "F32" && x != 0.0 && x.abs() < f32::MIN_POSITIVE as f64 { 0.0f64.copysign(x) } else { x };
                    let wz = want_of(op, ftz(a), ftz(b));
                    for (enc, m) in [("VOP3", m3), ("VOPC", mc)] {
                        let g = m >> l & 1 == 1;
                        if g != w && g != wz { bad.push(format!("{ty} {enc} {op}({a:e}, {b:e}) = {g}")); } else if g != w { denorm += 1; }
                    }
                }
            }
            let ex = r.sgpr[126] as u64 | (r.sgpr[127] as u64) << 32;
            for (l, &(a, b)) in ab.iter().enumerate() {
                let ftz = |x: f64| if ty == "F32" && x != 0.0 && x.abs() < f32::MIN_POSITIVE as f64 { 0.0f64.copysign(x) } else { x };
                let g = ex >> l & 1 == 1;
                if g != (a < b) && g != (ftz(a) < ftz(b)) { bad.push(format!("{ty} CMPX LT({a:e}, {b:e}) = {g}")); } else if g != (a < b) { denorm += 1; }
            }
        }
        bad.sort(); bad.dedup();
        println!("wave{n} {ty}: {lanes} lanes x 33 compares, {} distinct failures, {denorm} results that match only a flush of f32 denormals", bad.len());
        for b in bad.iter().take(15) { println!("  {b}"); }
        total_bad += bad.len();
    }
    assert_eq!(total_bad, 0);
}

/// image_gather4h(_pck) / image_gather8h_pck read a row of texels: not translated as a 2x2 Gather.
#[test]
fn image_gather4h_is_unsupported() {
    for name in ["IMAGE_GATHER4H", "IMAGE_GATHER4H_PCK", "IMAGE_GATHER8H_PCK"] {
        let code = prog(&[(name, "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)], None)]);
        let t = translate(&code, &Options::default()).unwrap();
        assert!(!t.complete(), "{name}\n{}", t.hlsl);
        let key = format!("{}: image_gather4h", name.to_ascii_lowercase());
        assert!(t.unsupported_why.contains_key(&key), "{name}: {:?}", t.unsupported_why);
    }
}

fn hwreg_imm(id: u64, offset: u64, width: u64) -> u64 {
    id | (offset << 6) | ((width - 1) << 11)
}

#[test]
fn round_mode_classification_and_lowering() {
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };

    // Default RNE (0x0): exact NOOP even when f16/f64 ops are present.
    let def = prog(&[
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x0)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None),
    ]);
    let t = translate(&def, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("xcl_fp16_rm"));
    assert_eq!(t.ignored.get("s_round_mode"), Some(&1));

    // Retail graphics pattern (`s_round_mode 0xC`: f32=RNE, f16/f64=RTZ) with no f16/f64 ops:
    // exact NOOP, zero HLSL overhead.
    let f32_only = prog(&[
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V), ("VSRC1", 2)], None),
    ]);
    let t = translate(&f32_only, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("xcl_fp16_rm"));
    assert_eq!(t.ignored.get("s_round_mode"), Some(&1));

    // `s_round_mode 0xC` with f16 ops: emits `xcl_fp16_rm` and routes f16 rounding through `f32_to_f16_rm`.
    let f16_rtz = prog(&[
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let t = translate(&f16_rtz, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("static uint xcl_fp16_rm = 0u;"));
    assert!(t.hlsl.contains("xcl_fp16_rm = 3u;"));
    assert!(t.hlsl.contains("f32_to_f16_rm("));

    // Non-default f64 round mode: rejected instead of silently running RNE.
    let f64_rtz = prog(&[
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None),
    ]);
    let t = translate(&f64_rtz, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_round_mode: non-default f64 round mode"), "{:?}", t.unsupported_why);

    // Non-default f32 round mode: rejected.
    for sp in [1u64, 2, 3] {
        let f32_rm = prog(&[
            ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", sp)], None),
            ("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V), ("VSRC1", 2)], None),
        ]);
        let t = translate(&f32_rm, &opt).unwrap();
        assert!(!t.complete(), "sp={sp}");
        assert!(t.unsupported_why.contains_key("s_round_mode: non-default f32 round mode"), "{:?}", t.unsupported_why);
    }

    // `s_setreg_imm32_b32`: full MODE[3:0] write, partial bit writes, non-MODE & denorm-bit NOOPs, rejections.
    let setreg_partial = prog(&[
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 1))], Some(1)),
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 3, 1))], Some(1)),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let t = translate(&setreg_partial, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("xcl_fp16_rm = (xcl_fp16_rm & 2u) | 1u;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("xcl_fp16_rm = (xcl_fp16_rm & 1u) | 2u;"), "{}", t.hlsl);

    let setreg_f32_bad = prog(&[
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 0, 4))], Some(0x1)),
    ]);
    let t = translate(&setreg_f32_bad, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_setreg_imm32_b32: non-default f32 round mode"), "{:?}", t.unsupported_why);

    let setreg_f64_bad = prog(&[
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2))], Some(0x3)),
        ("V_MUL_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None),
    ]);
    let t = translate(&setreg_f64_bad, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_setreg_imm32_b32: non-default f64 round mode"), "{:?}", t.unsupported_why);

    let setreg_noop = prog(&[
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(20, 0, 32))], Some(0x1234_5678)), // flat scratch
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 4, 4))], Some(0xF)),           // MODE denorm bits
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let t = translate(&setreg_noop, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(!t.hlsl.contains("xcl_fp16_rm"));
    assert_eq!(t.ignored.get("s_setreg_imm32_b32"), Some(&2));

    // `s_setreg_b32`: dynamic SGPR write to MODE[3:2] with f16 ops succeeds; f32/f64 rejected.
    let setreg_dyn = prog(&[
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2)), ("SDST", 0)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None),
    ]);
    let t = translate(&setreg_dyn, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("xcl_fp16_rm = (s0 & 3u);"), "{}", t.hlsl);

    let setreg_dyn_f32 = prog(&[
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 0, 2)), ("SDST", 0)], None),
    ]);
    let t = translate(&setreg_dyn_f32, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_setreg_b32: dynamic f32 round mode"), "{:?}", t.unsupported_why);

    let setreg_dyn_f64 = prog(&[
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2)), ("SDST", 0)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None),
    ]);
    let t = translate(&setreg_dyn_f64, &opt).unwrap();
    assert!(!t.complete());
    assert!(t.unsupported_why.contains_key("s_setreg_b32: dynamic f64 round mode"), "{:?}", t.unsupported_why);

    // Straight-line pre-scan stops at `s_endpgm`: unreachable trailing f64 op does not break `s_round_mode 0xC`.
    let mut trailing = prog(&[("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None)]);
    trailing.extend(encode("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 6)], None).unwrap());
    let t = translate(&trailing, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
}

#[test]
fn gpu_nonfused_mad_and_mac_prevent_fma_contraction() {
    if !gpu_ready() {
        return;
    }
    // Two golden vectors where fused FMA and non-fused multiply-then-add disagree:
    //   lane 0 (overflow):     a = f32::MAX, b = 1 + 2^-23, c = -f32::MAX
    //     non-fused -> +inf (0x7F800000), fused -> 0x73FFFFFF
    //   lane 1 (cancellation): a = 1 + 2^-23, b = 1 + 2^-23, c = -(1 + 2^-22)
    //     non-fused -> +0.0 (0x00000000), fused -> 2^-46 (0x28800000)
    let code = prog(&[
        ("V_MAC_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 0), ("VSRC1", 1)], None),
        ("V_MAC_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", V + 1)], None),
        ("V_MAD_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2)], None),
        ("V_MADMK_F32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 0), ("VSRC1", 2)], Some(0x3F80_0001)),
        ("V_MADAK_F32", "ENC_VOP2", &[("VDST", 14), ("SRC0", V + 0), ("VSRC1", 1)], Some(0xFF7F_FFFF)),
        ("V_MADAK_F32", "ENC_VOP2", &[("VDST", 15), ("SRC0", V + 0), ("VSRC1", 1)], Some(0xBF80_0002)),
        // Fused control on the exact same inputs proves the backend has hardware FMA and `precise` mattered.
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 2)], None),
    ]);
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };
    let t = translate(&code, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = direct_or(&code, &opt, super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap());

    let mut st = KernelState::default();
    st.sgpr[127] = 0;
    let vectors = [
        (0x7F7F_FFFFu32, 0x3F80_0001u32, 0xFF7F_FFFFu32), // overflow
        (0x3F80_0001u32, 0x3F80_0001u32, 0xBF80_0002u32), // cancellation
    ];
    for (l, &(a, b, c)) in vectors.iter().enumerate() {
        st.vgpr[l][0] = a;
        st.vgpr[l][1] = b;
        st.vgpr[l][2] = c;
        st.vgpr[l][10] = c;
        st.vgpr[l][11] = c;
    }
    let r = runner::run(&dxil, &st, &[]).unwrap();

    let v0 = &r.vgpr[0];
    assert_eq!(v0[10], 0x7F80_0000, "v_mac_f32 (VOP2) overflow must not fuse");
    assert_eq!(v0[11], 0x7F80_0000, "v_mac_f32 (VOP3) overflow must not fuse");
    assert_eq!(v0[12], 0x7F80_0000, "v_mad_f32 overflow must not fuse");
    assert_eq!(v0[13], 0x7F80_0000, "v_madmk_f32 overflow must not fuse");
    assert_eq!(v0[14], 0x7F80_0000, "v_madak_f32 overflow must not fuse");
    assert_eq!(v0[16], 0x73FF_FFFF, "v_fma_f32 overflow control should stay fused");

    let v1 = &r.vgpr[1];
    assert_eq!(v1[10], 0x0000_0000, "v_mac_f32 (VOP2) cancellation must not fuse");
    assert_eq!(v1[11], 0x0000_0000, "v_mac_f32 (VOP3) cancellation must not fuse");
    assert_eq!(v1[12], 0x0000_0000, "v_mad_f32 cancellation must not fuse");
    assert_eq!(v1[13], 0x0000_0000, "v_madmk_f32 cancellation must not fuse");
    assert_eq!(v1[15], 0x0000_0000, "v_madak_f32 cancellation must not fuse");
    assert_eq!(v1[16], 0x2880_0000, "v_fma_f32 cancellation control should stay fused");
}

#[test]
fn gpu_f16_defaults_to_round_to_nearest_even() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
    ]);
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };
    let t = translate(&code, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();

    let cases: [(f32, u16); 7] = [
        (f32::from_bits(3075.0f32.to_bits() - 1), 0x6A01),
        (3075.0, 0x6A02),
        (f32::from_bits(3075.0f32.to_bits() + 1), 0x6A02),
        (3081.0, 0x6A04),
        (-3075.0, 0xEA02),
        (65520.0, 0x7C00),
        (f32::from_bits(0x3300_0001), 0x0001),
    ];
    let mut st = KernelState::default();
    st.sgpr[127] = 0;
    for (lane, &(value, _)) in cases.iter().enumerate() {
        st.vgpr[lane][0] = value.to_bits();
    }
    // 3072 * (1 + 2^-10) is the exact binary16 midpoint 3075.
    st.vgpr[0][1] = 0xEA00_6A00;
    st.vgpr[0][2] = 0x3C01_3C01;

    let r = runner::run(&dxil, &st, &[]).unwrap();
    let actual: [u16; 7] = std::array::from_fn(|lane| r.vgpr[lane][10] as u16);
    let expected = cases.map(|(_, bits)| bits);
    let product = r.vgpr[0][11] as u16;
    let packed = r.vgpr[0][12];
    assert_eq!(
        (actual, product, packed),
        (expected, 0x6A02, 0xEA02_6A02),
        "conversion {actual:04x?}, multiplication {product:04x}, packed multiplication {packed:08x}",
    );
}

#[test]
fn gpu_s_round_mode_controls_f16_rounding_across_modes() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // Default RNE (0) -> v10 (v_cvt_f16_f32), v20 (v_mul_f16), v21 (v_pk_mul_f16), v22 (v_fma_mixlo_f16)
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 0)], None),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 20), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 21), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
        ("V_FMA_MIXLO_F16", "ENC_VOP3P", &[("VDST", 22), ("SRC0", V + 0), ("SRC1", 242), ("SRC2", 128)], None),
        // Switch to RTZ (3) via S_ROUND_MODE 0xC -> v11, v23, v24, v25
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 0)], None),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 23), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 24), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
        ("V_FMA_MIXLO_F16", "ENC_VOP3P", &[("VDST", 25), ("SRC0", V + 0), ("SRC1", 242), ("SRC2", 128)], None),
        // Switch to +inf (1) via S_ROUND_MODE 0x4 -> v12
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x4)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 0)], None),
        // Switch to -inf (2) via S_ROUND_MODE 0x8 -> v13
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x8)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 0)], None),
        // Switch back to RNE (0) via S_ROUND_MODE 0x0 -> v14
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x0)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 0)], None),
        // Switch to RTZ (3) via S_SETREG_IMM32_B32, then clear bit 0 -> -inf (2) -> v15
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2))], Some(3)),
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 1))], Some(0)),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 0)], None),
        // Switch dynamically via S_SETREG_B32 from s0 (= 1 -> +inf) -> v16
        ("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hwreg_imm(1, 2, 2)), ("SDST", 0)], None),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 16), ("SRC0", V + 0)], None),
    ]);
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };
    let t = translate(&code, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = direct_or(&code, &opt, super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap());

    // (f32 bits, RNE, RTZ, +inf, -inf)
    let cases: &[(u32, u16, u16, u16, u16)] = &[
        (0x3F80_0800, 0x3C00, 0x3C00, 0x3C01, 0x3C00), // +(1 + 2^-12)
        (0xBF80_0800, 0xBC00, 0xBC00, 0xBC00, 0xBC01), // -(1 + 2^-12)
        (0x3F80_1000, 0x3C00, 0x3C00, 0x3C01, 0x3C00), // +(1 + 2^-11): tie to even (0x3C00)
        (0x3F80_3000, 0x3C02, 0x3C01, 0x3C02, 0x3C01), // +(1 + 3*2^-11): tie to odd -> rounds up to 0x3C02
        (0xBF80_3000, 0xBC02, 0xBC01, 0xBC01, 0xBC02), // -(1 + 3*2^-11)
        (0x477F_E040, 0x7BFF, 0x7BFF, 0x7C00, 0x7BFF), // +65504.5
        (0xC77F_E040, 0xFBFF, 0xFBFF, 0xFBFF, 0xFC00), // -65504.5
        (0x477F_F000, 0x7C00, 0x7BFF, 0x7C00, 0x7BFF), // +65520.0 (tie to +inf)
        (0xC77F_F000, 0xFC00, 0xFBFF, 0xFBFF, 0xFC00), // -65520.0 (tie to -inf)
        (0x4788_B800, 0x7C00, 0x7BFF, 0x7C00, 0x7BFF), // +70000.0 (> 65536 overflow)
        (0xC788_B800, 0xFC00, 0xFBFF, 0xFBFF, 0xFC00), // -70000.0
        (0x33C0_0000, 0x0002, 0x0001, 0x0002, 0x0001), // +1.5 * 2^-24 (subnormal tie to even)
        (0x387F_FFFF, 0x0400, 0x03FF, 0x0400, 0x03FF), // subnormal-to-normal carry
        (0x3300_0000, 0x0000, 0x0000, 0x0001, 0x0000), // +2^-25 (exact tie to 0)
        (0x3300_0001, 0x0001, 0x0000, 0x0001, 0x0000), // +(2^-25 + eps)
        (0x8D80_0000, 0x8000, 0x8000, 0x8000, 0x8001), // tiny negative underflow
        (0x0000_0000, 0x0000, 0x0000, 0x0000, 0x0000), // +0.0
        (0x8000_0000, 0x8000, 0x8000, 0x8000, 0x8000), // -0.0
        (0x7F80_0000, 0x7C00, 0x7C00, 0x7C00, 0x7C00), // +inf
        (0xFF80_0000, 0xFC00, 0xFC00, 0xFC00, 0xFC00), // -inf
    ];

    let mut st = KernelState::default();
    st.sgpr[0] = 1; // +inf for S_SETREG_B32
    st.sgpr[127] = 0;
    for (l, &(bits, ..)) in cases.iter().enumerate() {
        st.vgpr[l][0] = bits;
        // Pack two f16 overflow pairs: 65504 * (1 + 2^-10) = 65567.96875 -> +inf (RNE) vs 65504 (RTZ)
        // High half: -65504 * (1 + 2^-10) -> -inf (RNE) vs -65504 (RTZ)
        st.vgpr[l][1] = 0xFBFF_7BFF;
        st.vgpr[l][2] = 0x3C01_3C01;
    }
    // Lane 20: quiet NaN preservation check
    st.vgpr[20][0] = 0x7FC0_2000;

    let r = runner::run(&dxil, &st, &[]).unwrap();
    for (l, &(bits, rne, rtz, rtp, rtn)) in cases.iter().enumerate() {
        let v = &r.vgpr[l];
        assert_eq!(v[10] as u16, rne, "lane {l} ({bits:#010x}) RNE");
        assert_eq!(v[11] as u16, rtz, "lane {l} ({bits:#010x}) RTZ");
        assert_eq!(v[12] as u16, rtp, "lane {l} ({bits:#010x}) +inf");
        assert_eq!(v[13] as u16, rtn, "lane {l} ({bits:#010x}) -inf");
        assert_eq!(v[14] as u16, rne, "lane {l} ({bits:#010x}) reset to RNE");
        assert_eq!(v[15] as u16, rtn, "lane {l} ({bits:#010x}) setreg_imm32 partial -> -inf");
        assert_eq!(v[16] as u16, rtp, "lane {l} ({bits:#010x}) setreg_b32 dynamic -> +inf");
        assert_eq!(v[22] as u16, rne, "lane {l} ({bits:#010x}) v_fma_mixlo_f16 RNE");
        assert_eq!(v[25] as u16, rtz, "lane {l} ({bits:#010x}) v_fma_mixlo_f16 RTZ");
    }
    // Verify V_MUL_F16 & V_PK_MUL_F16 honor RNE vs RTZ
    assert_eq!(r.vgpr[0][20] as u16, 0x7C00, "v_mul_f16 overflow RNE");
    assert_eq!(r.vgpr[0][23] as u16, 0x7BFF, "v_mul_f16 overflow RTZ");
    assert_eq!(r.vgpr[0][21], 0xFC00_7C00, "v_pk_mul_f16 overflow RNE");
    assert_eq!(r.vgpr[0][24], 0xFBFF_7BFF, "v_pk_mul_f16 overflow RTZ");
    // Verify NaN stays NaN across all 4 rounding modes
    for reg in [10, 11, 12, 13] {
        let h = r.vgpr[20][reg] as u16;
        assert!((h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0, "reg v{reg} NaN corrupted: {h:#06x}");
    }
}

#[test]
fn gpu_mad_nf_f16_rounds_intermediate_in_binary16() {
    if !gpu_ready() {
        return;
    }
    // Compile a wave32 kernel containing the recompiler PRELUDE and exercise `mad_nf_f16` directly.
    let base = prog(&[("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 0)], None)]);
    let opt = Options { wave64: false, hw_wave: 32, ..Options::default() };
    let mut hlsl = translate(&base, &opt).unwrap().hlsl;
    let probe = "\
    if (lane_id == 0u) v0 = f32_to_f16_rm(mad_nf_f16(f16tof32(0x7BFFu), f16tof32(0x3C01u), f16tof32(0xFBFFu), 0u), 0u);\n\
    if (lane_id == 1u) v0 = f32_to_f16_rm(mad_nf_f16(f16tof32(0x7BFFu), f16tof32(0x3C01u), f16tof32(0xFBFFu), 3u), 3u);\n\
    if (lane_id == 2u) v0 = f32_to_f16_rm(mad_nf_f16(f16tof32(0x3C01u), f16tof32(0x3C01u), f16tof32(0xBC02u), 0u), 0u);\n";
    let marker = "    v0 = select((((exec_lo >> lane_id) & 1u) != 0u), (v0), v0);\n";
    assert!(hlsl.contains(marker), "unexpected kernel body:\n{hlsl}");
    hlsl = hlsl.replace(marker, probe);

    let dxil = super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap();
    let mut st = KernelState::default();
    st.sgpr[127] = 0;
    let r = runner::run(&dxil, &st, &[]).unwrap();

    assert_eq!(r.vgpr[0][0], 0x7C00, "mad_nf_f16 RNE overflow: 65504*(1+2^-10)-65504 -> +inf");
    assert_eq!(r.vgpr[1][0], 0x0000, "mad_nf_f16 RTZ clamp: 65504*(1+2^-10)->65504, -65504 -> 0");
    assert_eq!(r.vgpr[2][0], 0x0000, "mad_nf_f16 cancellation: (1+2^-10)^2-(1+2^-9) -> 0 (not 2^-20)");
}

/// `exp mrtz` carries the sample mask in z (32-bit formats; Mesa, ac_nir_lower_ps_late.c
/// `emit_ps_mrtz_export`), or in the low half of VSRC1 when compressed (UINT16_ABGR): it goes
/// to SV_Coverage when the console's output signature has one. A depth-only PS is as before.
#[test]
fn ps_mrtz_sample_mask_writes_sv_coverage() {
    use super::PsInterface;
    let ps = |outputs: Vec<crate::gpu::dxbc::SigElement>, mrtz: &[(&str, u64)]| {
        let code = prog(&[
            ("EXP", "ENC_EXP", mrtz, None),
            ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1), ("VM", 1)], None),
        ]);
        let psi = PsInterface { inputs: vec![], outputs, input_ena: 0x2, attrs: None };
        let pt = translate(&code, &Options { mode: Mode::Pixel(psi), wave64: true, ..Default::default() }).unwrap();
        assert!(pt.complete(), "{:?}\n{}", pt.unsupported_why, pt.hlsl);
        if super::dxc::location().is_ok() {
            super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl));
        }
        pt.hlsl
    };
    let target = el("SV_Target", 0, 64, 3, 0, 0xF);
    let h = ps(vec![target.clone(), el("SV_Coverage", 0, 66, 1, 0xFFFF_FFFF, 1)], &[("TGT", 8), ("EN", 0x4), ("VSRC2", 5)]);
    assert!(h.contains("    uint xcl_mask : SV_Coverage;\n") && h.contains("    xo.xcl_mask = 0xFFFFFFFFu;\n") && h.contains("xo.xcl_mask = v5;"), "{h}");
    // compressed: the mask in VSRC1 [15:0] (EN 0xC)
    let h = ps(vec![target.clone(), el("SV_Coverage", 0, 66, 1, 0xFFFF_FFFF, 1)], &[("TGT", 8), ("EN", 0xC), ("COMPR", 1), ("VSRC0", 6), ("VSRC1", 5)]);
    assert!(h.contains("    xo.xcl_mask = 0xFFFFFFFFu;\n") && h.contains("xo.xcl_mask = (v5 & 0xFFFFu);"), "{h}");
    // SV_Coverage in the signature, but mrtz does not enable the mask channel: do not declare it (would zero every sample)
    let h = ps(vec![target.clone(), el("SV_Depth", 0, 65, 3, 0xFFFF_FFFF, 1), el("SV_Coverage", 0, 66, 1, 0xFFFF_FFFF, 1)], &[("TGT", 8), ("EN", 0x1), ("VSRC0", 4)]);
    assert!(!h.contains("SV_Coverage") && !h.contains("xcl_mask"), "{h}");
    // coverage-only PS: XclOut has SV_Coverage and does not need the dummy target
    let h = ps(vec![el("SV_Coverage", 0, 66, 1, 0xFFFF_FFFF, 1)], &[("TGT", 8), ("EN", 0x4), ("VSRC2", 5)]);
    assert!(h.contains("    uint xcl_mask : SV_Coverage;\n") && !h.contains("xcl_unused"), "{h}");
    // no SV_Coverage in the signature: nothing written
    let h = ps(vec![target.clone()], &[("TGT", 8), ("EN", 0x4), ("VSRC2", 5)]);
    assert!(!h.contains("SV_Coverage") && !h.contains("xcl_mask"), "{h}");
    // depth only: the interface and the depth write of before
    let h = ps(vec![target, el("SV_Depth", 0, 65, 3, 0xFFFF_FFFF, 1)], &[("TGT", 8), ("EN", 0x1), ("VSRC0", 4)]);
    assert!(h.contains("struct XclOut {\n    float4 t0 : SV_Target0;\n    float xcl_depth : SV_Depth;\n};\n"), "{h}");
    assert!(h.contains("{ xo.xcl_depth = asfloat(v4); }") && !h.contains("xcl_mask"), "{h}");
}

#[test]
fn ps_interp_p0_preserves_bits_and_p2_converts_to_float() {
    use super::PsInterface;
    let ps_code = prog(&[
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 2), ("ATTR", 1), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 1)], None),
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 7), ("VSRC", 2), ("ATTR", 2), ("ATTRCHAN", 0)], None),
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 8), ("VSRC", 2), ("ATTR", 3), ("ATTRCHAN", 0)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![
            el("SV_Position", 0, 1, 3, 0, 0xF),
            el("TEXCOORD", 0, 0, 3, 1, 0xF), // float4
            el("TEXCOORD", 1, 0, 1, 2, 0xF), // uint4
            el("TEXCOORD", 2, 0, 2, 3, 0xF), // int4
            el("TEXCOORD", 3, 0, 3, 4, 0xF), // flat float4
        ],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let pt = translate(&ps_code, &Options { mode: Mode::Pixel(psi), wave64: true, ..Default::default() }).unwrap();
    assert!(pt.complete(), "{:?}\n{}", pt.unsupported_why, pt.hlsl);
    for read in ["asuint((float)xi.a0[0])", "asuint(xi.a1[0])", "asuint((float)xi.a1[1])", "asuint(xi.a2[0])", "asuint(xi.a3[0])"] {
        assert!(pt.hlsl.contains(read), "{read}:\n{}", pt.hlsl);
    }
    if super::dxc::location().is_ok() {
        super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl));
    }
}

#[test]
fn gpu_ds_swizzle_and_bpermute_zero_inactive_lanes() {
    let run_case = |wave64: bool, split_wave64: bool| {
        let mut insns: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = vec![
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 255)], Some(0x5555_5555)),
        ];
        if wave64 {
            insns.push(("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0x5555_5555)));
        }
        insns.extend_from_slice(&[
            // swizzle XOR 1: even lanes read inactive odd neighbor -> must be 0
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 10), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 1 << 2)], None),
            // bpermute targeting odd neighbor -> must be 0
            ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 11), ("ADDR", 1), ("DATA0", 0)], None),
            // bpermute targeting cross-half odd lane (lane ^ 33) -> must be 0
            ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 2), ("DATA0", 0)], None),
            // swizzle XOR 2: even lanes read active even lane -> must return its value
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 13), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 2 << 2)], None),
        ]);
        let code = prog(&insns);
        let t = translate(&code, &Options { wave64, split_wave64, ..Default::default() }).unwrap();
        assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
        let dxil = direct_or(&code, &Options { wave64, split_wave64, ..Default::default() }, super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl)));
        let mut st = KernelState::default();
        let lanes = if wave64 { 64 } else { 32 };
        for (l, r) in st.vgpr.iter_mut().take(lanes).enumerate() {
            r[0] = if l & 1 == 1 { 0xDEAD } else { 0x1000 + l as u32 };
            r[1] = 4 * (l ^ 1) as u32;
            r[2] = 4 * ((l ^ 33) & (lanes - 1)) as u32;
            for dst in 10..=13 {
                r[dst] = 0xCAFE_BABE;
            }
        }
        let r = runner::run(&dxil, &st, &[]).unwrap();
        for l in 0..lanes {
            let v = &r.vgpr[l];
            if l & 1 == 0 {
                assert_eq!(v[10], 0, "wave{lanes} split={split_wave64} lane {l}: ds_swizzle inactive src");
                assert_eq!(v[11], 0, "wave{lanes} split={split_wave64} lane {l}: ds_bpermute inactive src");
                assert_eq!(v[12], 0, "wave{lanes} split={split_wave64} lane {l}: ds_bpermute cross-half inactive src");
                assert_eq!(v[13], 0x1000 + (l ^ 2) as u32, "wave{lanes} split={split_wave64} lane {l}: ds_swizzle active src");
            } else {
                for dst in 10..=13 {
                    assert_eq!(v[dst], 0xCAFE_BABE, "wave{lanes} split={split_wave64} lane {l}: inactive dst v{dst}");
                }
            }
        }
    };
    for wave64 in gpu_waves() {
        run_case(wave64, false);
    }
    if gpu_ready() {
        run_case(true, true);
    }
}

#[test]
fn gpu_ldexp_f32() {
    for wave64 in gpu_waves() {
        let p = prog(&[("V_LDEXP_F32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 1)], None)]);
        const CASES: [(u32, i32, u32); 15] = [
            (1e30f32.to_bits(), -150, 1e30f32.to_bits() - (150 << 23)),
            (0.5f32.to_bits(), 128, 0x7F00_0000),            // 2^127
            (2.0f32.to_bits(), -127, 0x0080_0000),           // 2^-126 (min normal)
            (0.0f32.to_bits(), 200, 0x0000_0000),            // +0 stays +0
            (0x8000_0000, 200, 0x8000_0000),                 // -0 stays -0
            (f32::INFINITY.to_bits(), -200, 0x7F80_0000),    // +inf stays +inf
            (f32::NEG_INFINITY.to_bits(), -200, 0xFF80_0000),// -inf stays -inf
            (0x7FC0_1234, 10, 0x7FC0_1234),                  // NaN stays NaN
            (1.0f32.to_bits(), -149, 0x0000_0001),           // min subnormal
            (1.0f32.to_bits(), -150, 0x0000_0000),           // tie to even -> 0
            (1.5f32.to_bits(), -149, 0x0000_0002),           // tie to even -> 2
            (0x0000_0001, 149, 1.0f32.to_bits()),            // subnormal input -> 1.0
            (0x0000_0001, 276, 0x7F00_0000),                 // subnormal input -> 2^127
            (1.0f32.to_bits(), i32::MAX, 0x7F80_0000),       // overflow to +inf
            ((-1.0f32).to_bits(), i32::MIN, 0x8000_0000),    // underflow to -0
        ];
        let mut st = KernelState::default();
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            let (x, n, _) = CASES[l % CASES.len()];
            (r[0], r[1]) = (x, n as u32);
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            let (x, n, want) = CASES[l % CASES.len()];
            let got = v[4];
            let both_nan = f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan();
            assert!(
                got == want || both_nan,
                "wave{lanes} lane {l} v_ldexp_f32({x:#010x}, {n}) = {got:#010x}, want {want:#010x}"
            );
        }
    }
}

#[test]
fn gpu_f64_literal_high_word() {
    for wave64 in gpu_waves() {
        let p = prog(&[
            // 1.0 + literal(1.5 = 0x3FF8_0000_0000_0000) == 2.5
            ("V_ADD_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", 255)], Some(0x3FF8_0000)),
            // compare result against literal(2.5 = 0x4004_0000_0000_0000)
            ("V_CMP_EQ_F64", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 4), ("SRC1", 255)], Some(0x4004_0000)),
            // verify integer 64-bit literal still goes in low 32 bits
            ("V_CMP_EQ_U64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 6), ("SRC1", 255)], Some(0x1234_5678)),
        ]);
        let mut st = KernelState::default();
        let one = 1.0f64.to_bits();
        for r in st.vgpr.iter_mut() {
            (r[0], r[1]) = (one as u32, (one >> 32) as u32);
            (r[6], r[7]) = (0x1234_5678, 0);
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(pair(v, 4), 2.5f64, "wave{lanes} lane {l} v_add_f64 literal");
        }
        let expected_hi = if wave64 { u32::MAX } else { 0 };
        assert_eq!((r.sgpr[0], r.sgpr[1]), (u32::MAX, expected_hi), "wave{lanes} v_cmp_eq_f64 literal");
        assert_eq!((r.sgpr[2], r.sgpr[3]), (u32::MAX, expected_hi), "wave{lanes} v_cmp_eq_u64 literal");
    }
}

#[test]
fn gpu_pknorm_i16_f32_nan_and_fmt_encode_snorm_ufloat() {
    if !gpu_ready() {
        return;
    }
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let vs = |base: u32, fmt: u32| [base, 0, 0x100, ident | fmt << 12];
    // Formats: 43 = 11_11_10_FLOAT (X:10b, Y:11b, Z:11b), 36 = 10_11_11_FLOAT (X:11b, Y:11b, Z:10b), 24 = 16_16_SNORM
    let p = prog(&[
        ("V_CVT_PKNORM_I16_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 0), ("SRC1", V + 0)], None), // (NaN, NaN)
        ("V_CVT_PKNORM_I16_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", V + 1)], None), // (NaN, 1.0)
        ("V_CVT_PKNORM_I16_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", V + 0)], None), // (-1.0, NaN)
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 20), ("SRSRC", 4), ("SOFFSET", 128)], None), // 11_11_10_FLOAT (-1, -inf, 1)
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 24), ("SRSRC", 5), ("SOFFSET", 128)], None), // 11_11_10_FLOAT (NaN, +inf, -0.5)
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 20), ("SRSRC", 6), ("SOFFSET", 128)], None), // 10_11_11_FLOAT (-1, -inf, 1)
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 28), ("SRSRC", 7), ("SOFFSET", 128)], None),  // 16_16_SNORM (NaN, -1.0)
    ]);
    let mut st = KernelState::default();
    st.sgpr[16..20].copy_from_slice(&vs(0x200, 43));
    st.sgpr[20..24].copy_from_slice(&vs(0x204, 43));
    st.sgpr[24..28].copy_from_slice(&vs(0x208, 36));
    st.sgpr[28..32].copy_from_slice(&vs(0x20C, 24));
    for r in st.vgpr.iter_mut() {
        r[0] = f32::NAN.to_bits();
        r[1] = 1.0f32.to_bits();
        r[2] = (-1.0f32).to_bits();
        // (-1.0, -Inf, 1.0)
        r[20] = (-1.0f32).to_bits();
        r[21] = f32::NEG_INFINITY.to_bits();
        r[22] = 1.0f32.to_bits();
        // (NaN, +Inf, -0.5)
        r[24] = f32::NAN.to_bits();
        r[25] = f32::INFINITY.to_bits();
        r[26] = (-0.5f32).to_bits();
        // (NaN, -1.0)
        r[28] = f32::NAN.to_bits();
        r[29] = (-1.0f32).to_bits();
    }
    let mem = vec![0u32; 0x100];
    let r = runner::run_code(&p, &st, &mem).unwrap();
    assert_eq!(r.vgpr[0][10], 0x0000_0000, "v_cvt_pknorm_i16_f32(NaN, NaN)");
    assert_eq!(r.vgpr[0][11], 0x7FFF_0000, "v_cvt_pknorm_i16_f32(NaN, 1.0)");
    assert_eq!(r.vgpr[0][12], 0x0000_8001, "v_cvt_pknorm_i16_f32(-1.0, NaN)");
    // 11_11_10_FLOAT (-1, -inf, 1): X(10b)=0, Y(11b)=0, Z(11b)=0x3C0 -> 0x3C0 << 21 = 0x7800_0000
    assert_eq!(r.mem[0x200 / 4], 0x3C0 << 21, "11_11_10_FLOAT (-1, -inf, 1)");
    // 11_11_10_FLOAT (NaN, +inf, -0.5): X(10b)=0x3F0 (NaN), Y(11b)=0x7C0 (+inf), Z(11b)=0
    assert_eq!(r.mem[0x204 / 4], 0x3F0 | (0x7C0 << 10), "11_11_10_FLOAT (NaN, +inf, -0.5)");
    // 10_11_11_FLOAT (-1, -inf, 1): X(11b)=0, Y(11b)=0, Z(10b)=0x1E0 -> 0x1E0 << 22 = 0x7800_0000
    assert_eq!(r.mem[0x208 / 4], 0x1E0 << 22, "10_11_11_FLOAT (-1, -inf, 1)");
    // 16_16_SNORM (NaN, -1.0): lo=0, hi=-32767 (0x8001)
    assert_eq!(r.mem[0x20C / 4], 0x8001_0000, "16_16_SNORM (NaN, -1.0)");
}

#[test]
fn gpu_fmt_encode_int_sat_and_f16_rtz() {
    if !gpu_ready() {
        return;
    }
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let vs = |base: u32, fmt: u32| [base, 0, 0x100, ident | fmt << 12];
    let p = prog(&[
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 0), ("SRSRC", 4), ("SOFFSET", 128)], None),  // 16_16_UINT
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 2), ("SRSRC", 5), ("SOFFSET", 128)], None),  // 16_16_SINT
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 4), ("SRSRC", 6), ("SOFFSET", 128)], None),  // 16_16_FLOAT
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 6), ("SRSRC", 7), ("SOFFSET", 128)], None), // 11_11_10_FLOAT
    ]);
    let mut st = KernelState::default();
    st.sgpr[16..20].copy_from_slice(&vs(0x200, 27));
    st.sgpr[20..24].copy_from_slice(&vs(0x204, 28));
    st.sgpr[24..28].copy_from_slice(&vs(0x208, 29));
    st.sgpr[28..32].copy_from_slice(&vs(0x20C, 43));
    for r in st.vgpr.iter_mut() {
        r[0] = 70_000u32;
        r[1] = 65_535u32;
        r[2] = 40_000i32 as u32;
        r[3] = (-40_000i32) as u32;
        r[4] = 65520.0f32.to_bits();
        r[5] = (-65520.0f32).to_bits();
        r[6] = 65520.0f32.to_bits();
        r[7] = f32::INFINITY.to_bits();
        r[8] = 65520.0f32.to_bits();
    }
    let mem = vec![0u32; 0x100];
    let r = runner::run_code(&p, &st, &mem).unwrap();
    assert_eq!(r.mem[0x200 / 4], 0xFFFF_FFFF, "16_16_UINT saturates 70000 -> 65535");
    assert_eq!(r.mem[0x204 / 4], 0x8000_7FFF, "16_16_SINT saturates (40000, -40000)");
    assert_eq!(r.mem[0x208 / 4], 0xFBFF_7BFF, "16_16_FLOAT saturates finite overflow to max finite");
    assert_eq!(r.mem[0x20C / 4], 0x3DF | (0x7C0 << 10) | (0x7BF << 21), "11_11_10_FLOAT saturates finite overflow and keeps +inf");
}

/// Direct DXIL and HLSL parity for `fmt_encode`: UINT/SINT saturation (`w < 32`), `w == 32`
/// passthrough, and RTZ binary16 / 11_11_10_FLOAT rounding (`XCL-BT-026`, #1488, #1269).
#[test]
fn gpu_fmt_encode_direct_dxil_int_sat_and_f16_rtz() {
    if !gpu_ready() {
        return;
    }
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let vs = |base: u32, fmt: u32| [base, 0, 0x100, ident | fmt << 12];
    let p = prog(&[
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 0), ("SRSRC", 4), ("SOFFSET", 128)], None),    // 16_16_UINT
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 2), ("SRSRC", 5), ("SOFFSET", 128)], None),    // 16_16_SINT
        ("BUFFER_STORE_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 4), ("SRSRC", 6), ("SOFFSET", 128)], None),  // 8_8_8_8_UINT
        ("BUFFER_STORE_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 8), ("SRSRC", 7), ("SOFFSET", 128)], None),  // 8_8_8_8_SINT
        ("BUFFER_STORE_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 12), ("SRSRC", 8), ("SOFFSET", 128)], None), // 2_10_10_10_SINT
        ("BUFFER_STORE_FORMAT_X", "ENC_MUBUF", &[("VDATA", 16), ("SRSRC", 9), ("SOFFSET", 128)], None),    // 32_UINT
        ("BUFFER_STORE_FORMAT_X", "ENC_MUBUF", &[("VDATA", 17), ("SRSRC", 10), ("SOFFSET", 128)], None),   // 32_SINT
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 18), ("SRSRC", 11), ("SOFFSET", 128)], None),  // 16_16_FLOAT overflow
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 20), ("SRSRC", 12), ("SOFFSET", 128)], None),  // 16_16_FLOAT normal RTZ
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 22), ("SRSRC", 13), ("SOFFSET", 128)], None),  // 16_16_FLOAT ±inf
        ("BUFFER_STORE_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 24), ("SRSRC", 14), ("SOFFSET", 128)], None),  // 16_16_FLOAT subnormal RTZ / -0
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 26), ("SRSRC", 15), ("SOFFSET", 128)], None), // 11_11_10_FLOAT overflow + inf
        ("BUFFER_STORE_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 29), ("SRSRC", 16), ("SOFFSET", 128)], None), // 11_11_10_FLOAT NaN / neg / RTZ
    ]);
    let mut st = KernelState::default();
    let fmts = [
        (4, 0x200, 27), (5, 0x204, 28), (6, 0x208, 60), (7, 0x20C, 61), (8, 0x210, 55),
        (9, 0x214, 20), (10, 0x218, 21), (11, 0x21C, 29), (12, 0x220, 29), (13, 0x224, 29),
        (14, 0x228, 29), (15, 0x22C, 43), (16, 0x230, 43),
    ];
    for (sr, base, fmt) in fmts {
        st.sgpr[sr * 4..sr * 4 + 4].copy_from_slice(&vs(base, fmt));
    }
    for r in st.vgpr.iter_mut() {
        r[0] = 70_000u32;
        r[1] = u32::MAX;
        r[2] = 40_000i32 as u32;
        r[3] = (-40_000i32) as u32;
        r[4] = 300u32;
        r[5] = 255u32;
        r[6] = 0u32;
        r[7] = u32::MAX;
        r[8] = 200i32 as u32;
        r[9] = (-200i32) as u32;
        r[10] = 127i32 as u32;
        r[11] = (-128i32) as u32;
        r[12] = 600i32 as u32;
        r[13] = (-600i32) as u32;
        r[14] = 0u32;
        r[15] = (-3i32) as u32;
        r[16] = 0xDEAD_BEEFu32;
        r[17] = i32::MIN as u32;
        r[18] = 65520.0f32.to_bits();
        r[19] = (-65520.0f32).to_bits();
        r[20] = 0x3F80_1800; // 1.0 + 0.75 * 2^-10: RNE -> 0x3C01, RTZ -> 0x3C00
        r[21] = 0xBF80_1800; // -(1.0 + 0.75 * 2^-10): RNE -> 0xBC01, RTZ -> 0xBC00
        r[22] = f32::INFINITY.to_bits();
        r[23] = f32::NEG_INFINITY.to_bits();
        r[24] = 0x3340_0000; // 0.75 * 2^-24: RNE -> 0x0001, RTZ -> 0x0000
        r[25] = (-0.0f32).to_bits();
        r[26] = 65520.0f32.to_bits();
        r[27] = f32::INFINITY.to_bits();
        r[28] = 65520.0f32.to_bits();
        r[29] = f32::NAN.to_bits();
        r[30] = (-1.0f32).to_bits();
        r[31] = 0x3F81_F800; // 1.0 + 15.75 * 2^-10: RNE -> 0x3C10 (>>4 = 0x3C1), RTZ -> 0x3C0F (>>4 = 0x3C0)
    }
    let mem = vec![0u32; 0x100];
    let t = translate(&p, &runner::options(Mode::Kernel)).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    let hlsl_dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();
    let direct_dxil = runner::direct_kernel_full(&p, true).unwrap().dxil;
    for (backend, dxil) in [("hlsl", hlsl_dxil), ("direct_dxil", direct_dxil)] {
        let r = runner::run_with_statics(&dxil, &st, &mem, &[], &[], &[]).unwrap_or_else(|e| panic!("{backend}: {e}"));
        assert_eq!(r.mem[0x200 / 4], 0xFFFF_FFFF, "{backend}: 16_16_UINT");
        assert_eq!(r.mem[0x204 / 4], 0x8000_7FFF, "{backend}: 16_16_SINT");
        assert_eq!(r.mem[0x208 / 4], 0xFF00_FFFF, "{backend}: 8_8_8_8_UINT");
        assert_eq!(r.mem[0x20C / 4], 0x807F_807F, "{backend}: 8_8_8_8_SINT");
        assert_eq!(r.mem[0x210 / 4], 0x1FF | (0x200 << 10) | (2 << 30), "{backend}: 2_10_10_10_SINT");
        assert_eq!(r.mem[0x214 / 4], 0xDEAD_BEEF, "{backend}: 32_UINT passthrough");
        assert_eq!(r.mem[0x218 / 4], i32::MIN as u32, "{backend}: 32_SINT passthrough");
        assert_eq!(r.mem[0x21C / 4], 0xFBFF_7BFF, "{backend}: 16_16_FLOAT finite overflow RTZ");
        assert_eq!(r.mem[0x220 / 4], 0xBC00_3C00, "{backend}: 16_16_FLOAT normal RTZ");
        assert_eq!(r.mem[0x224 / 4], 0xFC00_7C00, "{backend}: 16_16_FLOAT ±inf");
        assert_eq!(r.mem[0x228 / 4], 0x8000_0000, "{backend}: 16_16_FLOAT subnormal RTZ / -0.0");
        assert_eq!(r.mem[0x22C / 4], 0x3DF | (0x7C0 << 10) | (0x7BF << 21), "{backend}: 11_11_10_FLOAT overflow + inf");
        assert_eq!(r.mem[0x230 / 4], 0x3F0 | (0x3C0 << 21), "{backend}: 11_11_10_FLOAT NaN / neg / RTZ");
    }
}

#[test]
fn gpu_signed_bfe_arithmetic_shift() {
    for wave64 in gpu_waves() {
        let p = prog(&[
            // s_bfe_i64(-1, offset=8, width=64) -> -1
            ("S_BFE_I64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 4), ("SSRC1", 6)], None),
            // s_bfe_i64(0x8000_0000_0000_0000, offset=32, width=40) -> 0xFFFF_FFFF_8000_0000
            ("S_BFE_I64", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 8), ("SSRC1", 10)], None),
            // s_bfe_i32(0x8000_0000, offset=16, width=20) -> 0xFFFF_8000
            ("S_BFE_I32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 11), ("SSRC1", 13)], None),
            // v_bfe_i32(0x8000_0000, offset=16, width=20) -> 0xFFFF_8000
            ("V_BFE_I32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 0), ("SRC1", 144), ("SRC2", 148)], None),
            // v_bfe_i32(0x0000_8000, offset=0, width=16) -> 0xFFFF_8000 (o + w < 32)
            ("V_BFE_I32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 1), ("SRC1", 128), ("SRC2", 144)], None),
            // v_bfe_i32(0xFFFF_FFFF, offset=4, width=0) -> 0
            ("V_BFE_I32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 2), ("SRC1", 132), ("SRC2", 128)], None),
        ]);
        let mut st = KernelState::default();
        (st.sgpr[4], st.sgpr[5]) = (u32::MAX, u32::MAX);
        st.sgpr[6] = (64 << 16) | 8;
        (st.sgpr[8], st.sgpr[9]) = (0, 0x8000_0000);
        st.sgpr[10] = (40 << 16) | 32;
        st.sgpr[11] = 0x8000_0000;
        st.sgpr[13] = (20 << 16) | 16;
        for r in st.vgpr.iter_mut() {
            r[0] = 0x8000_0000;
            r[1] = 0x0000_8000;
            r[2] = u32::MAX;
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        assert_eq!((r.sgpr[0], r.sgpr[1]), (u32::MAX, u32::MAX), "wave{lanes} s_bfe_i64(-1, 8, 64)");
        assert_eq!((r.sgpr[2], r.sgpr[3]), (0x8000_0000, u32::MAX), "wave{lanes} s_bfe_i64(i64::MIN, 32, 40)");
        assert_eq!(r.sgpr[12], 0xFFFF_8000, "wave{lanes} s_bfe_i32(i32::MIN, 16, 20)");
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(v[4], 0xFFFF_8000, "wave{lanes} lane {l} v_bfe_i32(i32::MIN, 16, 20)");
            assert_eq!(v[5], 0xFFFF_8000, "wave{lanes} lane {l} v_bfe_i32(0x8000, 0, 16)");
            assert_eq!(v[6], 0, "wave{lanes} lane {l} v_bfe_i32(width=0)");
        }
    }
}

#[test]
fn gpu_cmp_class_f16_preserves_snan() {
    for wave64 in gpu_waves() {
        let p = prog(&[
            // 0x7C01 (sNaN) tested against mask 1 << 0 (sNaN) -> true
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 0), ("SRC1", 129)], None),
            // 0x7C01 (sNaN) tested against mask 1 << 1 (qNaN) -> false
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 0), ("SRC1", 130)], None),
            // 0x7E01 (qNaN) tested against mask 1 << 0 (sNaN) -> false
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 1), ("SRC1", 129)], None),
            // 0x7E01 (qNaN) tested against mask 1 << 1 (qNaN) -> true
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 1), ("SRC1", 130)], None),
            // 0xBC00 (-1.0) with ABS tested against mask 1 << 8 (+normal) -> true
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", 255), ("ABS", 1)], Some(1 << 8)),
            // 0xBC00 (-1.0) with NEG tested against mask 1 << 8 (+normal) -> true
            ("V_CMP_CLASS_F16", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", 255), ("NEG", 1)], Some(1 << 8)),
        ]);
        let mut st = KernelState::default();
        for r in st.vgpr.iter_mut() {
            r[0] = 0x7C01; // f16 sNaN
            r[1] = 0x7E01; // f16 qNaN
            r[2] = 0xBC00; // f16 -1.0
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        let all = (u32::MAX, if wave64 { u32::MAX } else { 0 });
        assert_eq!((r.sgpr[0], r.sgpr[1]), all, "wave{lanes} f16 sNaN matches sNaN bit");
        assert_eq!((r.sgpr[2], r.sgpr[3]), (0, 0), "wave{lanes} f16 sNaN does not match qNaN bit");
        assert_eq!((r.sgpr[4], r.sgpr[5]), (0, 0), "wave{lanes} f16 qNaN does not match sNaN bit");
        assert_eq!((r.sgpr[6], r.sgpr[7]), all, "wave{lanes} f16 qNaN matches qNaN bit");
        assert_eq!((r.sgpr[8], r.sgpr[9]), all, "wave{lanes} f16 abs(-1.0) matches +normal");
        assert_eq!((r.sgpr[10], r.sgpr[11]), all, "wave{lanes} f16 neg(-1.0) matches +normal");
    }
}

#[test]
fn wave32_src_vccz_execz_and_saveexec_b32_hlsl() {
    let w32_code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 251)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 251)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 252)], None),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 124)], None),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 251)], None),
    ]);
    let t32 = translate(&w32_code, &Options { wave64: false, ..Default::default() }).unwrap();
    assert!(t32.complete(), "{:?}", t32.unsupported_why);
    assert!(t32.hlsl.contains("(uint)(vcc_lo == 0u)"), "{}", t32.hlsl);
    assert!(t32.hlsl.contains("(uint)(exec_lo == 0u)"), "{}", t32.hlsl);
    assert!(!t32.hlsl.contains("vcc_lo | vcc_hi"), "{}", t32.hlsl);
    assert!(!t32.hlsl.contains("exec_lo | exec_hi"), "{}", t32.hlsl);
    assert!(!t32.hlsl.contains("& exec_hi"), "{}", t32.hlsl);

    let w64_code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 251)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 252)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 251)], None),
    ]);
    let t64 = translate(&w64_code, &Options { wave64: true, ..Default::default() }).unwrap();
    assert!(t64.complete(), "{:?}", t64.unsupported_why);
    assert!(t64.hlsl.contains("(uint)((vcc_lo | vcc_hi) == 0u)"), "{}", t64.hlsl);
    assert!(t64.hlsl.contains("(uint)((exec_lo | exec_hi) == 0u)"), "{}", t64.hlsl);
}

#[test]
fn sdwa_f16_inline_constants_hlsl() {
    let code = prog(&[
        ("V_ADD_F16", "VOP2_VOP_SDWA", &[("VDST", 10), ("SRC0", 0xF9), ("VSRC0", 242), ("S0", 1), ("VSRC1", 1), ("SRC0_SEL", 6), ("SRC1_SEL", 5), ("DST_SEL", 6)], None),
        ("V_MUL_F16", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 244), ("S1", 1), ("SRC0_SEL", 5), ("SRC1_SEL", 6), ("DST_SEL", 6)], None),
        ("V_CVT_F32_F16", "VOP1_VOP_SDWA", &[("VDST", 12), ("SRC0", 0xF9), ("VSRC0", 243), ("S0", 1), ("SRC0_SEL", 6), ("DST_SEL", 6)], None),
        ("V_CMP_LT_F16", "VOPC_VOP_SDWA_SDST_ENC", &[("SRC0", 0xF9), ("VSRC0", 242), ("S0", 1), ("VSRC1", 2), ("SRC0_SEL", 6), ("SRC1_SEL", 5), ("SD", 1), ("SDST", 20)], None),
    ]);
    let t = translate(&code, &Options { wave64: false, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert!(t.hlsl.contains("sdwa_sel(0x00003C00u"), "{}", t.hlsl);
    assert!(t.hlsl.contains("sdwa_sel(0x00004000u"), "{}", t.hlsl);
    assert!(t.hlsl.contains("sdwa_sel(0x0000BC00u"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("sdwa_sel(0x3F800000u"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("sdwa_sel(0x40000000u"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("sdwa_sel(0xBF800000u"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("uint v244"), "{}", t.hlsl);

    let int_src = prog(&[("V_CVT_F16_U16", "ENC_VOP1", &[("VDST", 13), ("SRC0", 242)], None)]);
    let t_int = translate(&int_src, &Options { wave64: false, ..Default::default() }).unwrap();
    assert!(t_int.complete(), "{:?}", t_int.unsupported_why);
    assert!(t_int.hlsl.contains("((0x3F800000u) & 0xFFFFu)"), "{}", t_int.hlsl);
    assert!(!t_int.hlsl.contains("0x00003C00u"), "{}", t_int.hlsl);
}

#[test]
fn gpu_wave32_and_wave64_src_vccz_execz() {
    if !gpu_ready() {
        return;
    }
    let code32 = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 251)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 251)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 129)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 251)], None),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 252)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 2)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 252)], None),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 124)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 126)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 5)], None),
    ]);
    let mut st32 = KernelState::default();
    st32.sgpr[106] = 0;
    st32.sgpr[107] = 0x1234_5678;
    st32.sgpr[127] = 1;
    st32.sgpr[124] = 0x0000_000F;
    let (r32, lanes32) = run_wave(&code32, &st32, false);
    assert_eq!(lanes32, 32);
    assert_eq!(r32.sgpr[0], 1, "wave32 src_vccz ignores vcc_hi");
    assert_eq!(r32.sgpr[1], 0, "wave32 src_vccz clears once vcc_lo != 0");
    assert_eq!(r32.sgpr[3], 1, "wave32 src_execz ignores exec_hi");
    assert_eq!(r32.sgpr[4], 0, "wave32 src_execz clears once exec_lo != 0");
    assert_eq!(r32.sgpr[6], 0xF, "wave32 s_and_saveexec_b32 with m0");
    for (l, v) in r32.vgpr.iter().take(32).enumerate() {
        assert_eq!(v[0], 1, "wave32 lane {l} v_mov_b32 src_vccz");
    }

    let code64 = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 251)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 107), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 251)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 251)], None),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 2)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 252)], None),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 252)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 4)], None),
    ]);
    let mut st64 = KernelState::default();
    st64.sgpr[106] = 0;
    st64.sgpr[107] = 1;
    st64.sgpr[2] = 0;
    st64.sgpr[3] = 1;
    let r64 = runner::run_code(&code64, &st64, &[]).unwrap();
    assert_eq!(r64.sgpr[0], 0, "wave64 src_vccz checks vcc_hi");
    assert_eq!(r64.sgpr[1], 1, "wave64 src_vccz once vcc_hi == 0");
    assert_eq!((r64.sgpr[8], r64.sgpr[9]), (1, 0), "wave64 s_mov_b64 src_vccz");
    assert_eq!(r64.sgpr[6], 0, "wave64 src_execz checks exec_hi");
    assert_eq!(r64.sgpr[7], 1, "wave64 src_execz once exec_hi == 0");
}

#[test]
fn gpu_sdwa_f16_inline_constants() {
    let code = prog(&[
        ("V_ADD_F16", "VOP2_VOP_SDWA", &[("VDST", 10), ("SRC0", 0xF9), ("VSRC0", 242), ("S0", 1), ("VSRC1", 1), ("SRC0_SEL", 6), ("SRC1_SEL", 5), ("DST_SEL", 6)], None),
        ("V_MUL_F16", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 244), ("S1", 1), ("SRC0_SEL", 5), ("SRC1_SEL", 6), ("DST_SEL", 6)], None),
        ("V_CVT_F32_F16", "VOP1_VOP_SDWA", &[("VDST", 12), ("SRC0", 0xF9), ("VSRC0", 243), ("S0", 1), ("SRC0_SEL", 6), ("DST_SEL", 6)], None),
        ("V_CMP_LT_F16", "VOPC_VOP_SDWA_SDST_ENC", &[("SRC0", 0xF9), ("VSRC0", 242), ("S0", 1), ("VSRC1", 2), ("SRC0_SEL", 6), ("SRC1_SEL", 5), ("SD", 1), ("SDST", 20)], None),
        ("V_CMP_LT_F16", "VOPC_VOP_SDWA_SDST_ENC", &[("SRC0", 0xF9), ("VSRC0", 2), ("VSRC1", 242), ("S1", 1), ("SRC0_SEL", 5), ("SRC1_SEL", 6), ("SD", 1), ("SDST", 22)], None),
    ]);
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        r[1] = 0x4000_3800; // WORD_1 = 2.0h, WORD_0 = 0.5h
        r[2] = 0x3800_0000; // WORD_1 = 0.5h
    }
    for wave64 in gpu_waves() {
        let (r, lanes) = run_wave(&code, &st, wave64);
        let all = (u32::MAX, if wave64 { u32::MAX } else { 0 });
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            assert_eq!(v[10], 0x4200, "wave{lanes} lane {l} v_add_f16_sdwa 1.0 + 2.0");
            assert_eq!(v[11], 0x4400, "wave{lanes} lane {l} v_mul_f16_sdwa 2.0 * 2.0");
            assert_eq!(v[12], (-1.0f32).to_bits(), "wave{lanes} lane {l} v_cvt_f32_f16_sdwa -1.0");
        }
        assert_eq!((r.sgpr[20], r.sgpr[21]), (0, 0), "wave{lanes} v_cmp_lt_f16_sdwa 1.0 < 0.5");
        assert_eq!((r.sgpr[22], r.sgpr[23]), all, "wave{lanes} v_cmp_lt_f16_sdwa 0.5 < 1.0");
    }
}

#[test]
fn fspx_compute_user_sgprs_from_rsrc2() {
    use crate::gpu::xbox::tests::container;

    let counts = [4u32, 8].map(|count| {
        // Two directory entries: type 0 holds the compute registers, type 6 holds the code.
        let mut part = Vec::from(*b"FSPX");
        for word in [1u32, 2, 0x74, 0, 0x30, 0x1c, 0, 6, 0x4c, 0x28, 0] {
            part.extend_from_slice(&word.to_le_bytes());
        }
        part.resize(0x74, 0);
        let header = &mut part[0x30..0x4c];
        header[0x0c..0x12].copy_from_slice(&[1, 0, 1, 0, 1, 0]); // numthreads(1, 1, 1)
        let rsrc2 = (count << 1) | 0x81; // set unrelated bits on both sides of USER_SGPR
        header[0x18..0x1c].copy_from_slice(&rsrc2.to_le_bytes());
        part[0x70..].copy_from_slice(&0xbf81_0000u32.to_le_bytes()); // s_endpgm
        let blob = container(&[(b"ACCC", &part)]);
        assert_eq!(super::pso::fspx(&blob).unwrap().rsrc2, rsrc2);
        super::pso::compute_user_sgprs(&blob)
    });
    assert_eq!(counts, [Some(4), Some(8)]);
}

/// Verifies early-exit-safe, bounded `xcl_s_barrier()` on real multi-warp compute shaders:
/// 1. All 4 hardware wave32s in `numthreads(128, 1, 1)` exchange LDS values across two back-to-back
///    `s_barrier` generations (both split-wave64 and native wave32).
/// 2. Upper console wave (`tid >= 64`) takes an early `s_endpgm` while the lower console wave
///    (`tid 0..63`, 2 hardware wave32s) executes two `s_barrier`s and exchanges LDS across its
///    surviving warps — legal on RDNA2, would permanently deadlock `bar.sync 0` on NVIDIA.
#[test]
fn gpu_safe_s_barrier_multi_warp_and_early_endpgm() {
    if !gpu_ready() {
        return;
    }
    use super::{dxc, CsInterface};

    // Shader 1: 4 hardware wave32s, 2 consecutive s_barrier generations exchanging across waves
    // using disjoint LDS banks (bank 0: 0..512, bank 1: 512..1024) so there is no WAR data race.
    let p_all = prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None),           // v1 = tid * 4 (bank 0)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 255), ("VSRC1", 1)], Some(512)),       // v9 = 512 + tid * 4 (bank 1)
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 192), ("VSRC1", 0)], None),               // v2 = tid ^ 64
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 2)], None),           // v2 = (tid ^ 64) * 4 (bank 0)
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 160), ("VSRC1", 0)], None),               // v3 = tid ^ 32
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 130), ("VSRC1", 3)], None),           // v3 = (tid ^ 32) * 4
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 3)], Some(512)),       // v3 = 512 + (tid ^ 32) * 4 (bank 1)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 0)], Some(0x100)),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 6)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 4), ("ADDR", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 129), ("VSRC1", 4)], None),            // +1
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 9), ("DATA0", 4)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 5), ("ADDR", 3)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 0), ("VSRC1", 1)], None),              // s0 + tid*4
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 7), ("DATA", 5), ("SADDR", 0x7D)], None),
    ]);

    let cs = CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [false, false, false], tg_size: false, tid_components: 1 };
    let base = 0x100usize;
    let mut io = vec![0u32; 128];
    io[0] = base as u32;
    for wave64 in [true, false] {
        let t = translate(&p_all, &on_gpu(Options { mode: Mode::Compute(cs.clone()), wave64, ..Default::default() })).unwrap();
        assert!(t.complete(), "wave64={wave64}: {:?}", t.unsupported_why);
        if std::env::var("XCL_SAFE_BARRIER").map_or(true, |v| v.trim() != "0") {
            assert!(t.hlsl.contains("xcl_s_barrier();"), "wave64={wave64}: expected xcl_s_barrier()");
        }
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let out = runner::dispatch(&dxil, [1, 1, 1], &io, &vec![0u32; (base + 128 * 4) / 4]).unwrap();
        for tid in 0..128usize {
            let want = ((tid ^ 96) + 0x101) as u32;
            assert_eq!(out[(base + tid * 4) / 4], want, "wave64={wave64} tid={tid}: 2-gen xcl_s_barrier LDS exchange");
        }
    }

    // Shader 2: upper console wave (tid >= 64) exits early via `s_endpgm` before 2 `s_barrier`s;
    // lower console wave (tid 0..63, 2 hardware wave32s) synchronizes and exchanges LDS across halves.
    let p_early = prog(&[
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)], None),                    // s10 = first tid in wave
        ("S_CMP_GE_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 192)], None),                       // scc = (s10 >= 64)
        ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None),                                     // skip s_endpgm if < 64
        ("S_ENDPGM", "ENC_SOPP", &[], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None),           // v1 = tid * 4 (bank 0)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 255), ("VSRC1", 1)], Some(512)),       // v9 = 512 + tid * 4 (bank 1)
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 160), ("VSRC1", 0)], None),               // v3 = tid ^ 32
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 130), ("VSRC1", 3)], None),           // v3 = (tid ^ 32) * 4 (bank 0)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 3)], Some(512)),       // v2 = 512 + (tid ^ 32) * 4 (bank 1)
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 0)], Some(0x200)),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 6)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 4), ("ADDR", 3)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 135), ("VSRC1", 4)], None),            // +7
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 9), ("DATA0", 4)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 5), ("ADDR", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 0), ("VSRC1", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 7), ("DATA", 5), ("SADDR", 0x7D)], None),
    ]);
    let t = translate(&p_early, &on_gpu(Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() })).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    let out = runner::dispatch(&dxil, [2, 1, 1], &io, &vec![0u32; (base + 128 * 4) / 4]).unwrap();
    for tid in 0..64usize {
        assert_eq!(out[(base + tid * 4) / 4], (tid + 0x207) as u32, "tid={tid}: surviving warps must pass 2 s_barriers after upper wave early s_endpgm");
    }
    for tid in 64..128usize {
        assert_eq!(out[(base + tid * 4) / 4], 0, "tid={tid}: exited wave must not store");
    }
}

#[test]
fn qsad_and_mqsad_operand_forms_and_validation() {
    let vop3 = |name: &str, f: &[(&str, u64)]| encode(name, "ENC_VOP3", f, None).unwrap();
    let end = || encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap();
    // Valid LLVM GFX10 operand combinations: VGPR pairs/quads (up to v255), SGPR pairs, VCC, EXEC, 0, -1, CLAMP.
    let mut valid = Vec::new();
    for name in ["V_QSAD_PK_U16_U8", "V_MQSAD_PK_U16_U8"] {
        valid.extend(vop3(name, &[("VDST", 254), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]));
        valid.extend(vop3(name, &[("VDST", 4), ("SRC0", 2), ("SRC1", 103), ("SRC2", 126), ("CLAMP", 1)]));
        valid.extend(vop3(name, &[("VDST", 6), ("SRC0", 193), ("SRC1", 128), ("SRC2", 106)]));
        valid.extend(vop3(name, &[("VDST", 8), ("SRC0", 128), ("SRC1", 193), ("SRC2", 193)]));
    }
    valid.extend(vop3("V_MQSAD_U32_U8", &[("VDST", 252), ("SRC0", 193), ("SRC1", V + 2), ("SRC2", V + 3)]));
    valid.extend(vop3("V_MQSAD_U32_U8", &[("VDST", 12), ("SRC0", 126), ("SRC1", 106), ("SRC2", V + 20), ("CLAMP", 1)]));
    valid.extend(end());
    let t = translate(&valid, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }

    // Invalid VGPR bounds / non-VGPR SRC2 on v_mqsad_u32_u8 must be rejected cleanly.
    for bad in [
        &[("VDST", 255), ("SRC0", V), ("SRC1", V + 2), ("SRC2", V + 3)][..],
        &[("VDST", 4), ("SRC0", V + 255), ("SRC1", V + 2), ("SRC2", V + 3)],
        &[("VDST", 4), ("SRC0", V), ("SRC1", V + 2), ("SRC2", V + 255)],
    ] {
        let mut p = vop3("V_QSAD_PK_U16_U8", bad);
        p.extend(end());
        assert!(!translate(&p, &Options::default()).unwrap().complete(), "{bad:?}");
    }
    for bad in [
        &[("VDST", 253), ("SRC0", V), ("SRC1", V + 2), ("SRC2", V + 3)][..],
        &[("VDST", 4), ("SRC0", V), ("SRC1", V + 2), ("SRC2", 0)],
        &[("VDST", 4), ("SRC0", V), ("SRC1", V + 2), ("SRC2", 128)],
        &[("VDST", 4), ("SRC0", V), ("SRC1", V + 2), ("SRC2", V + 253)],
    ] {
        let mut p = vop3("V_MQSAD_U32_U8", bad);
        p.extend(end());
        assert!(!translate(&p, &Options::default()).unwrap().complete(), "{bad:?}");
    }
}

/// GPU verification of `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`, and `v_mqsad_u32_u8`: sliding
/// windows across `S0[63:0]`, zero-byte masking on `S1`, wrapping vs `CLAMP` saturation,
/// SGPR/inline sources, overlapping `VDST`/`SRC` ranges, and `EXEC` masking.
#[test]
fn gpu_qsad_and_mqsad() {
    if !gpu_ready() {
        return;
    }
    let vop3 = |name: &str, f: &[(&str, u64)]| encode(name, "ENC_VOP3", f, None).unwrap();
    // both backends: HLSL + DXC, and direct DXIL
    let run_on = |code: &[u8], st: &KernelState, wave64: bool, direct: bool| {
        let dxil = if direct {
            let o = crate::gpu::dxil::kernel::KernelOptions { wave64, split: wave64 && !gpu_waves().contains(&true), mem_windows: 1, sampler_base: 0 };
            crate::gpu::dxil::kernel::translate(code, o).unwrap_or_else(|e| panic!("direct DXIL: {e}"))
        } else {
            let t = translate(code, &on_gpu(Options { wave64, ..Default::default() })).unwrap();
            assert!(t.complete(), "wave64={wave64}: {:?}", t.unsupported_why);
            super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}
{}", t.hlsl))
        };
        runner::run(&dxil, st, &[]).unwrap()
    };
    let mut code = Vec::new();
    // Unmasked vs masked sliding windows (v4 = 0x4000_2015 has zero byte 2; v3 top byte 0xFF unused)
    code.extend(vop3("V_QSAD_PK_U16_U8", &[("VDST", 8), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 6)]));
    code.extend(vop3("V_QSAD_PK_U16_U8", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 10)]));
    code.extend(vop3("V_QSAD_PK_U16_U8", &[("VDST", 14), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 10), ("CLAMP", 1)]));
    code.extend(vop3("V_MQSAD_PK_U16_U8", &[("VDST", 16), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 6)]));
    code.extend(vop3("V_MQSAD_PK_U16_U8", &[("VDST", 18), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 10), ("CLAMP", 1)]));
    // Inline -1 SRC0, SGPR SRC1 (s2) and SGPR pair SRC2 (s[4:5])
    code.extend(vop3("V_MQSAD_PK_U16_U8", &[("VDST", 20), ("SRC0", 193), ("SRC1", 2), ("SRC2", 4)]));
    // 128-bit accumulator wrap vs CLAMP saturation
    code.extend(vop3("V_MQSAD_U32_U8", &[("VDST", 26), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 22)]));
    code.extend(vop3("V_MQSAD_U32_U8", &[("VDST", 30), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 22), ("CLAMP", 1)]));
    // Overlapping VDST/SRC0/SRC2 (in-place pk and staggered v[36:39] <- v[37:40])
    code.extend(vop3("V_MQSAD_PK_U16_U8", &[("VDST", 34), ("SRC0", V + 34), ("SRC1", V + 4), ("SRC2", V + 34)]));
    code.extend(vop3("V_MQSAD_U32_U8", &[("VDST", 36), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", V + 37)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[2] = 0x00FF_0001;
    st.sgpr[4] = 0x0002_0001;
    st.sgpr[5] = 0x0004_0003;
    for r in &mut st.vgpr {
        r[2] = 0x4030_2010;
        r[3] = 0xFF60_5000;
        r[4] = 0x4000_2015;
        r[6] = 0x0002_0001;
        r[7] = 0x0004_0003;
        r[10] = 0x0001_FFFE;
        r[11] = 0x0100_FFE0;
        r[22] = 0xFFFF_FFFC;
        r[23] = 0xFFFF_FFA0;
        r[24] = 0xFFFF_FFFF;
        r[25] = 5;
        r[34] = 0x4030_2010;
        r[35] = 0xFF60_5000;
        r[37] = 10;
        r[38] = 20;
        r[39] = 30;
        r[40] = 40;
    }

    for (wave64, lanes, direct) in [(false, 32usize, false), (true, 64usize, false), (false, 32, true), (true, 64, true)] {
        let r = run_on(&code, &st, wave64, direct);
        for (l, v) in r.vgpr.iter().enumerate().take(lanes) {
            assert_eq!((v[8], v[9]), (54 | (157 << 16), 78 | (191 << 16)), "direct={direct} wave64={wave64} lane {l}: v_qsad_pk_u16_u8");
            assert_eq!((v[12], v[13]), (51 | (156 << 16), 43 | (443 << 16)), "direct={direct} wave64={wave64} lane {l}: v_qsad_pk_u16_u8 wrap");
            assert_eq!((v[14], v[15]), (0xFFFF | (156 << 16), 0xFFFF | (443 << 16)), "direct={direct} wave64={wave64} lane {l}: v_qsad_pk_u16_u8 clamp");
            assert_eq!((v[16], v[17]), (6 | (93 << 16), 78 | (111 << 16)), "direct={direct} wave64={wave64} lane {l}: v_mqsad_pk_u16_u8");
            assert_eq!((v[18], v[19]), (0xFFFF | (92 << 16), 0xFFFF | (363 << 16)), "direct={direct} wave64={wave64} lane {l}: v_mqsad_pk_u16_u8 clamp");
            assert_eq!((v[20], v[21]), (255 | (256 << 16), 257 | (258 << 16)), "direct={direct} wave64={wave64} lane {l}: v_mqsad_pk_u16_u8 -1/SGPR");
            assert_eq!((v[26], v[27], v[28], v[29]), (1, 0xFFFF_FFFB, 74, 112), "direct={direct} wave64={wave64} lane {l}: v_mqsad_u32_u8 wrap");
            assert_eq!((v[30], v[31], v[32], v[33]), (u32::MAX, 0xFFFF_FFFB, u32::MAX, 112), "direct={direct} wave64={wave64} lane {l}: v_mqsad_u32_u8 clamp");
            assert_eq!((v[34], v[35]), (0x408B_2015, 0xFFCB_504B), "direct={direct} wave64={wave64} lane {l}: overlapping pk");
            assert_eq!((v[36], v[37], v[38], v[39], v[40]), (15, 111, 105, 147, 40), "direct={direct} wave64={wave64} lane {l}: staggered u32");
        }
    }

    // EXEC masking (32-lane and split/native 64-lane) + VCC pair as SRC2: inactive lanes keep old VDST.
    let mut masked = Vec::new();
    masked.extend(vop3("V_MQSAD_PK_U16_U8", &[("VDST", 0), ("SRC0", 193), ("SRC1", 129), ("SRC2", 128)]));
    masked.extend(vop3("V_MQSAD_U32_U8", &[("VDST", 2), ("SRC0", 193), ("SRC1", 129), ("SRC2", V + 2)]));
    masked.extend(vop3("V_QSAD_PK_U16_U8", &[("VDST", 6), ("SRC0", 193), ("SRC1", 129), ("SRC2", 106)]));
    masked.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut ms = KernelState::default();
    ms.sgpr[106] = 0x0002_0001;
    ms.sgpr[107] = 0x0004_0003;
    ms.sgpr[126] = 0x00FF_00FF;
    ms.sgpr[127] = 0x0F0F_0F0F;
    for r in &mut ms.vgpr {
        r[0] = 0xAAAA_AAAA;
        r[1] = 0xBBBB_BBBB;
        r[2] = 11;
        r[3] = 22;
        r[4] = 33;
        r[5] = 44;
        r[6] = 0xCCCC_CCCC;
        r[7] = 0xDDDD_DDDD;
    }
    for (wave64, lanes, direct) in [(false, 32usize, false), (true, 64usize, false), (false, 32, true), (true, 64, true)] {
        let r = run_on(&masked, &ms, wave64, direct);
        for (l, v) in r.vgpr.iter().enumerate().take(lanes) {
            let active = ((if l < 32 { ms.sgpr[126] } else { ms.sgpr[127] }) >> (l & 31)) & 1 != 0;
            if active {
                assert_eq!((v[0], v[1]), (0x00FE_00FE, 0x00FE_00FE), "direct={direct} wave64={wave64} active lane {l} pk");
                assert_eq!((v[2], v[3], v[4], v[5]), (265, 276, 287, 298), "direct={direct} wave64={wave64} active lane {l} u32");
                assert_eq!((v[6], v[7]), (1020 | (1021 << 16), 1022 | (1023 << 16)), "direct={direct} wave64={wave64} active lane {l} vcc");
            } else {
                assert_eq!((v[0], v[1]), (0xAAAA_AAAA, 0xBBBB_BBBB), "direct={direct} wave64={wave64} inactive lane {l} pk");
                assert_eq!((v[2], v[3], v[4], v[5]), (11, 22, 33, 44), "direct={direct} wave64={wave64} inactive lane {l} u32");
                assert_eq!((v[6], v[7]), (0xCCCC_CCCC, 0xDDDD_DDDD), "direct={direct} wave64={wave64} inactive lane {l} vcc");
            }
        }
    }
}

/// `XCL_SAMPLE_FSPX=<folder>`: every `*.bin` there that is an FSPX (driver) compute container –
/// e.g. the FSR4 pipelines XCL dumps to `<data>\shaders` – translates and compiles.
#[test]
#[ignore]
fn fspx_containers_translate_and_compile() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let Some(dir) = std::env::var_os("XCL_SAMPLE_FSPX") else { return };
    let files: Vec<std::path::PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
    let next = AtomicUsize::new(0);
    let (n, ok) = (AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|s| {
        for _ in 0..super::dxc::compile_threads() {
            s.spawn(|| loop {
                let Some(path) = files.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                let blob = std::fs::read(path).unwrap();
                let Some(f) = super::pso::fspx(&blob) else { continue };
                n.fetch_add(1, Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                let name = path.file_name().unwrap().to_string_lossy();
                match super::pso::console_to_dxil(&blob, super::pso::Stage::Compute, 0) {
                    Ok(c) => {
                        ok.fetch_add(1, Ordering::Relaxed);
                        println!("FSPX {name}: threads {:?} rsrc2 {:#x}, {} B code -> {} B DXIL, {:.2} s", f.threads, f.rsrc2, f.code.len(), c.dxil.len(), t0.elapsed().as_secs_f64());
                    }
                    Err(err) => println!("FSPX {name}: FAILED {err}"),
                }
            });
        }
    });
    let (n, ok) = (n.into_inner(), ok.into_inner());
    println!("FSPX: {ok} / {n} translated and compiled");
    assert_eq!(ok, n);
}

/// Structured code past `STRUCTURED_BYTES_PER_CODE_BYTE` is translated in the `pc` form (#1307).
/// n conditional forward jumps whose targets all come after them, in the same order, keep up to
/// n skip flags pending, and every guard lists them: 200 cross the bound, 50 do not. This checks
/// the switch only. Here the `pc` form keeps the same stretch under one `pc` guard and is no
/// smaller; the gain is on programs whose `pc` form is split into blocks, such as the retail
/// one of #1307 (148 MB -> 14 MB).
#[test]
fn oversized_structured_code_falls_back_to_pc_blocks() {
    let crossing = |n: usize| {
        let i = |name: &str, e: &str, f: &[(&str, u64)]| encode(name, e, f, None).unwrap();
        let mut code: Vec<u8> = Vec::new();
        let mut at = Vec::new();
        for k in 0..n as u64 {
            code.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)]));
            code.extend(i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128 + k % 16)]));
            at.push(code.len());
            code.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
        }
        for a in at {
            let d = ((code.len() - a) / 4 - 1) as u16;
            code[a..a + 2].copy_from_slice(&d.to_le_bytes());
            code.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 0), ("VSRC1", 1)]));
        }
        code.extend(i("S_ENDPGM", "ENC_SOPP", &[]));
        let t = translate(&code, &Options::default()).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        t.hlsl.contains("pc ==")
    };
    assert!(!crossing(50), "50 crossing jumps: structured");
    assert!(crossing(200), "200 crossing jumps: past the bound, pc blocks");
}

/// Random nests of loops with `continue` / `break` of any loop around them – backward jumps out of
/// inner loops, loops whose end is an inner loop's – give the same state as structured code and as
/// the block dispatcher (`structured: false`).
#[test]
fn jumps_to_outer_loop_heads_are_structured() {
    let i = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) % n
    };
    /// (loop head, jumps to its end and into its middle to be filled in)
    struct Open {
        head: usize,
        ends: Vec<usize>,
        entries: Vec<usize>,
    }
    let (mut structured, mut ran, mut entered) = (0, 0, 0);
    for round in 0..120 {
        let mut code: Vec<u8> = Vec::new();
        let mut open: Vec<Open> = Vec::new();
        // s20: steps left – every backward jump is taken only while there are some; s0 - s3: state
        let work = |code: &mut Vec<u8>, k: u64| {
            code.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", k % 4), ("SSRC0", k % 4), ("SSRC1", 129 + k % 7)]));
            code.extend(i("S_XOR_B32", "ENC_SOP2", &[("SDST", (k + 1) % 4), ("SSRC0", (k + 1) % 4), ("SSRC1", k % 4)]));
            code.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1 + k % 3), ("SRC0", k % 4), ("VSRC1", 1 + k % 3)]));
        };
        // a step used up; SCC = some left and a bit of the state
        let test = |code: &mut Vec<u8>, bit: u64| {
            code.extend(i("S_SUB_U32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 20), ("SSRC1", 129)]));
            code.extend(i("S_LSHR_B32", "ENC_SOP2", &[("SDST", 21), ("SSRC0", bit % 4), ("SSRC1", 128 + bit % 3)]));
            code.extend(i("S_AND_B32", "ENC_SOP2", &[("SDST", 21), ("SSRC0", 21), ("SSRC1", 129)]));
            code.extend(i("S_CMP_GT_I32", "ENC_SOPC", &[("SSRC0", 20), ("SSRC1", 128)]));
            code.extend(i("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 21), ("SSRC0", 21), ("SSRC1", 128)]));
            code.extend(i("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 21), ("SSRC1", 128)]));
        };
        let branch = |code: &mut Vec<u8>, name: &str, to: Option<usize>| -> usize {
            let at = code.len();
            let d = to.map_or(0, |t| (t as i64 - at as i64) / 4 - 1);
            code.extend(i(name, "ENC_SOPP", &[("SIMM16", d as u64 & 0xFFFF)]));
            at
        };
        let patch = |code: &mut Vec<u8>, at: usize| {
            let d = ((code.len() - at) / 4 - 1) as u16;
            code[at..at + 2].copy_from_slice(&d.to_le_bytes());
        };
        for _ in 0..(8 + rnd(14)) {
            let k = rnd(1000);
            work(&mut code, k);
            match rnd(10) {
                0..=2 if open.len() < 4 => {
                    // half of them entered in their middle by a jump from before
                    let mut entries = vec![];
                    if rnd(2) == 0 {
                        test(&mut code, k + 1);
                        entries.push(branch(&mut code, "S_CBRANCH_SCC1", None));
                        work(&mut code, k + 2);
                    }
                    open.push(Open { head: code.len(), ends: vec![], entries });
                }
                8 | 9 if open.last().is_some_and(|o| !o.entries.is_empty()) => {
                    for at in std::mem::take(&mut open.last_mut().unwrap().entries) {
                        patch(&mut code, at);
                        entered += 1;
                    }
                    work(&mut code, k + 3);
                }
                3 | 4 if !open.is_empty() => {
                    // conditional `continue` of any loop around
                    let l = rnd(open.len() as u64) as usize;
                    test(&mut code, k);
                    let head = open[l].head;
                    branch(&mut code, "S_CBRANCH_SCC1", Some(head));
                }
                5 if !open.is_empty() => {
                    // conditional `break` of any loop around
                    let l = rnd(open.len() as u64) as usize;
                    test(&mut code, k);
                    let at = branch(&mut code, "S_CBRANCH_SCC1", None);
                    open[l].ends.push(at);
                }
                6 | 7 if !open.is_empty() => {
                    // the innermost loop ends, with a back edge of its own or without
                    let o = open.pop().unwrap();
                    for at in o.entries {
                        patch(&mut code, at);
                        work(&mut code, k + 4);
                    }
                    if rnd(3) > 0 {
                        test(&mut code, k);
                        branch(&mut code, "S_CBRANCH_SCC1", Some(o.head));
                    }
                    for at in o.ends {
                        patch(&mut code, at);
                    }
                }
                _ => {}
            }
        }
        while let Some(o) = open.pop() {
            for at in o.entries {
                patch(&mut code, at);
                work(&mut code, 4);
            }
            test(&mut code, 3);
            branch(&mut code, "S_CBRANCH_SCC1", Some(o.head));
            for at in o.ends {
                patch(&mut code, at);
            }
        }
        work(&mut code, 5);
        code.extend(i("S_ENDPGM", "ENC_SOPP", &[]));
        let t = |s: bool| translate(&code, &Options { structured: s, ..Options::default() }).unwrap();
        let (a, b) = (t(true), t(false));
        assert!(a.complete() && b.complete(), "{:?}", a.unsupported_why);
        structured += !a.hlsl.contains("if (pc ==") as usize;
        if !gpu_waves().contains(&true) {
            continue;
        }
        ran += 1;
        let mut st = KernelState::default();
        for k in 0..4 {
            st.sgpr[k] = rnd(1 << 16) as u32;
        }
        st.sgpr[20] = 60;
        let mem: Vec<u32> = code.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let run = |h: &str| runner::run(&super::dxc::compile(h, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{h}")), &st, &mem).unwrap();
        let (u, w) = (run(&a.hlsl), run(&b.hlsl));
        assert_eq!((&u.sgpr[..24], u.scc), (&w.sgpr[..24], w.scc), "round {round}\n{}", a.hlsl);
        assert_eq!(u.vgpr, w.vgpr, "round {round}");
    }
    println!("{structured} of 120 programs structured ({entered} jumps into a loop), {ran} run both ways");
    assert!(structured >= 100 && entered > 20, "{structured} structured, {entered} jumps into loops");
}

#[test]
fn gpu_bvh_triangle_types_2_and_3() {
    if !gpu_ready() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let w = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    // 5 vertices at byte 0x100: v0=(0,0,1), v1=(2,0,1), v2=(1,2,1), v3=(3,2,2), v4=(1,4,3)
    w(&mut mem, 0x100, &ff(&[0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 1.0, 4.0, 3.0]));
    w(&mut mem, 0x100 + 60, &[0x0909_0909]);
    let node_base = 0x100u32 >> 3;
    let (node2, node3) = (node_base | 2, node_base | 3);

    let mimg = |n: &'static str, vdata: u64, vaddr: u64| encode(n, "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", 2), ("DMASK", 15)], None).unwrap();
    let mut code = Vec::new();
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 100, 0));  // type 2 hit
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 104, 20)); // type 3 hit
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 40)); // type 2 miss at type 0's center
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 0, u32::MAX, 0]);
    let set_ray = |r: &mut Vec<u32>, at: usize, node: u32, ox: f32, oy: f32| {
        r[at] = node;
        let v = [100.0f32, ox, oy, 0.0, 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
        for (k, x) in v.iter().enumerate() {
            r[at + 1 + k] = x.to_bits();
        }
    };
    for r in st.vgpr.iter_mut() {
        set_ray(r, 0, node2, 5.0 / 3.0, 8.0 / 3.0);
        set_ray(r, 20, node3, 2.0 / 3.0, 2.0);
        set_ray(r, 40, node2, 1.0, 0.5);
    }
    let res = runner::run_code(&code, &st, &mem).unwrap();
    let v = &res.vgpr[0];
    let hit2 = [v[100], v[101], v[102], v[103]].map(f32::from_bits);
    let hit3 = [v[104], v[105], v[106], v[107]].map(f32::from_bits);
    let miss2 = [v[108], v[109], v[110], v[111]].map(f32::from_bits);
    assert!(hit2[0].is_finite() && (hit2[0] / hit2[1] - 2.0).abs() < 1e-4, "gpu type 2 hit: {hit2:?}");
    assert!(hit3[0].is_finite() && (hit3[0] / hit3[1] - 5.0 / 3.0).abs() < 1e-4, "gpu type 3 hit: {hit3:?}");
    assert!(miss2[0].is_infinite(), "gpu type 2 must miss type 0's interior: {miss2:?}");
}

#[test]
fn gpu_permlane16_wqm_and_lds_stride64() {
    if !gpu_ready() {
        return;
    }
    // Check S_WQM_B32 / S_WQM_B64 (including SCC = 0 on empty input), V_PERMLANE16_B32 /
    // V_PERMLANEX16_B32 across FI / BOUND_CTRL with masked EXEC, and DS_*2ST64_B32 / _B64 in LDS.
    let code = prog(&[
        // Scalar WQM: non-zero and zero cases (s6..s9 read back before returning)
        ("S_WQM_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 0)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 7), ("SSRC0", 129), ("SSRC1", 128)], None),
        ("S_WQM_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 128)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 9), ("SSRC0", 129), ("SSRC1", 128)], None),
        ("S_WQM_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 2)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 129), ("SSRC1", 128)], None),
        // LDS stride-64 dual writes and reads (base offset v3 = lane * 4 for b32, + 4096 for b64)
        ("DS_WRITE2ST64_B32", "ENC_DS", &[("ADDR", 3), ("DATA0", 1), ("DATA1", 2), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_WRITE2ST64_B64", "ENC_DS", &[("ADDR", 4), ("DATA0", 6), ("DATA1", 8), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ2ST64_B32", "ENC_DS", &[("VDST", 10), ("ADDR", 3), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_READ2ST64_B64", "ENC_DS", &[("VDST", 12), ("ADDR", 4), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        // Drop odd lanes from EXEC before permlane16/permlanex16 so inactive-source modes are exercised
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 126)], None),
        ("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)], None),
        // FI=0, BOUND_CTRL=0: reading an odd (inactive) source keeps old v20
        ("V_PERMLANE16_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", 255), ("SRC2", 255), ("OP_SEL", 0)], Some(0x1111_1111)),
        // FI=0, BOUND_CTRL=1: reading an odd (inactive) source returns 0
        ("V_PERMLANE16_B32", "ENC_VOP3", &[("VDST", 21), ("SRC0", V), ("SRC1", 255), ("SRC2", 255), ("OP_SEL", 2)], Some(0x1111_1111)),
        // FI=1, BOUND_CTRL=0: fetch-inactive reads v0 even from inactive odd lane 1
        ("V_PERMLANE16_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V), ("SRC1", 255), ("SRC2", 255), ("OP_SEL", 1)], Some(0x1111_1111)),
        // Cross-row (other 16-lane row of each 32-lane half) reading even (active) lane 2
        ("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 23), ("SRC0", V), ("SRC1", 255), ("SRC2", 255), ("OP_SEL", 0)], Some(0x2222_2222)),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 14)], None),
    ]);

    let mut st = KernelState::default();
    st.sgpr[0] = 0x0002_0011; // quads 0, 1, 4 set -> 0x000F_00FF
    st.sgpr[2] = 0x0000_0001; // quad 0 in lo
    st.sgpr[3] = 0x8000_0000; // quad 7 in hi
    st.sgpr[20] = 0x5555_5555; // even lanes active
    st.sgpr[21] = 0x5555_5555;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 1000 + l as u32;
        r[1] = 200 + l as u32;
        r[2] = 500 + l as u32;
        r[3] = (l as u32) * 4;
        r[4] = 4096 + (l as u32) * 8;
        r[6] = 0x1000 + l as u32;
        r[7] = 0x2000 + l as u32;
        r[8] = 0x3000 + l as u32;
        r[9] = 0x4000 + l as u32;
        r[20] = 0xDEAD_0000 | l as u32;
        r[21] = 0xBEEF_0000 | l as u32;
        r[22] = 0xCAFE_0000 | l as u32;
        r[23] = 0xFEED_0000 | l as u32;
    }

    let r = runner::run_program(&code, &st, &[]).unwrap().0;
    assert_eq!((r.sgpr[6], r.sgpr[7]), (0x000F_00FF, 1), "s_wqm_b32 nonzero");
    assert_eq!((r.sgpr[8], r.sgpr[9]), (0, 0), "s_wqm_b32 zero");
    assert_eq!((r.sgpr[10], r.sgpr[11], r.sgpr[12]), (0x0000_000F, 0xF000_0000, 1), "s_wqm_b64");

    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!(v[10], st.vgpr[l][1], "lane {l} ds_read2st64_b32 off0");
        assert_eq!(v[11], st.vgpr[l][2], "lane {l} ds_read2st64_b32 off1");
        assert_eq!(v[12], st.vgpr[l][6], "lane {l} ds_read2st64_b64 off0 lo");
        assert_eq!(v[13], st.vgpr[l][7], "lane {l} ds_read2st64_b64 off0 hi");
        assert_eq!(v[14], st.vgpr[l][8], "lane {l} ds_read2st64_b64 off1 lo");
        assert_eq!(v[15], st.vgpr[l][9], "lane {l} ds_read2st64_b64 off1 hi");

        let half_base = l & !31;
        let row_base = l & 16;
        let other_row = row_base ^ 16;
        if l % 2 == 0 {
            // Even lanes were active; lane 1 in row is inactive, lane 2 in other row is active.
            assert_eq!(v[20], st.vgpr[l][20], "lane {l} permlane16 bound=0 keeps vdst");
            assert_eq!(v[21], 0, "lane {l} permlane16 bound=1 zeroes");
            assert_eq!(v[22], st.vgpr[half_base | row_base | 1][0], "lane {l} permlane16 fi=1 fetches inactive");
            assert_eq!(v[23], st.vgpr[half_base | other_row | 2][0], "lane {l} permlanex16 reads other row");
        } else {
            // Odd lanes were inactive under EXEC mask and retain original values.
            assert_eq!(v[20], st.vgpr[l][20], "inactive lane {l} permlane16");
            assert_eq!(v[21], st.vgpr[l][21], "inactive lane {l} permlane16 bound=1");
            assert_eq!(v[22], st.vgpr[l][22], "inactive lane {l} permlane16 fi=1");
            assert_eq!(v[23], st.vgpr[l][23], "inactive lane {l} permlanex16");
        }
    }
}

#[test]
fn gpu_scalar_sext_mulhi_and_vector_i24_sub_co_ci_u64_shifts() {
    if !gpu_ready() {
        return;
    }
    let v = |name: &'static str, f: &'static [(&'static str, u64)]| (name, "ENC_VOP3", f, None);
    let code = prog(&[
        // Scalar sign extension, abs (with SCC), and 32x32->hi32 multiplies
        ("S_SEXT_I32_I8", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 0)], None),
        ("S_SEXT_I32_I16", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 0)], None),
        ("S_ABS_I32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 1)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 129), ("SSRC1", 128)], None),
        ("S_MUL_HI_U32", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 2), ("SSRC1", 3)], None),
        ("S_MUL_HI_I32", "ENC_SOP2", &[("SDST", 15), ("SSRC0", 2), ("SSRC1", 3)], None),
        // Vector 32x32->hi32 and signed 24-bit multiply/mad
        v("V_MUL_HI_U32", &[("VDST", 10), ("SRC0", V), ("SRC1", V + 1)]),
        v("V_MUL_HI_I32", &[("VDST", 11), ("SRC0", V), ("SRC1", V + 1)]),
        v("V_MUL_I32_I24", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", V + 3)]),
        v("V_MUL_HI_I32_I24", &[("VDST", 13), ("SRC0", V + 2), ("SRC1", V + 3)]),
        v("V_MAD_I32_I24", &[("VDST", 14), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", V + 4)]),
        // 64-bit vector logical and arithmetic right shifts (including >= 32 shift amounts)
        v("V_LSHRREV_B64", &[("VDST", 16), ("SRC0", V + 5), ("SRC1", V + 6)]),
        v("V_ASHRREV_I64", &[("VDST", 18), ("SRC0", V + 5), ("SRC1", V + 6)]),
        // Chained 64-bit subtraction with borrow across VOP3B (V_SUB_CO_CI_U32) and VOP2 (V_SUBREV_CO_CI_U32)
        ("V_SUB_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 20), ("SDST", 20), ("SRC0", V + 6), ("SRC1", V + 8), ("SRC2", 22)], None),
        ("V_SUB_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 21), ("SDST", 24), ("SRC0", V + 7), ("SRC1", V + 9), ("SRC2", 20)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 22), ("SRC0", V + 6)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 22)], None), // vcc = borrow-in
        ("V_SUBREV_CO_CI_U32", "ENC_VOP2", &[("VDST", 22), ("SRC0", V + 8), ("VSRC1", 22)], None),
    ]);

    let mut st = KernelState::default();
    st.sgpr[0] = 0x0000_80FE; // low byte -2, low 16 bits -32514
    st.sgpr[1] = -42i32 as u32;
    st.sgpr[2] = 0xFFFF_FFFE; // u32::MAX - 1, or -2i32
    st.sgpr[3] = 0x8000_0003;
    st.sgpr[22] = 0x5555_5555; // borrow-in = 1 on even lanes, 0 on odd lanes
    st.sgpr[23] = 0x5555_5555;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 0x8000_0000u32.wrapping_add(l as u32 * 0x0101_0001);
        r[1] = 0xFFFF_FFFEu32.wrapping_sub(l as u32);
        // i24 inputs with garbage upper 8 bits that must be ignored
        r[2] = (0xAAu32 << 24) | if l % 2 == 0 { 0x00FF_FFF5 } else { 0x0000_0025 }; // -11 or +37
        r[3] = (0x55u32 << 24) | if l % 3 == 0 { 0x0080_0000 } else { 0x0001_0003 }; // -8388608 or +65539
        r[4] = 1000u32.wrapping_sub(l as u32 * 37);
        r[5] = (l as u32) % 68; // tests shifts <32, ==32, >32 and modulo 64 masking
        r[6] = 0x8765_4321u32.wrapping_sub(l as u32);
        r[7] = if l % 2 == 0 { 0xF123_4567 } else { 0x7123_4567 };
        r[8] = 0x8765_4321; // matches r[6] at lane 0 to force borrow propagation on borrow-in = 1
        r[9] = 0x1000_0000;
    }

    let r = runner::run_program(&code, &st, &[]).unwrap().0;
    assert_eq!(r.sgpr[10], -2i32 as u32, "s_sext_i32_i8");
    assert_eq!(r.sgpr[11], -32514i32 as u32, "s_sext_i32_i16");
    assert_eq!((r.sgpr[12], r.sgpr[13]), (42, 1), "s_abs_i32");
    assert_eq!(
        r.sgpr[14],
        (((st.sgpr[2] as u64) * (st.sgpr[3] as u64)) >> 32) as u32,
        "s_mul_hi_u32"
    );
    assert_eq!(
        r.sgpr[15],
        ((((st.sgpr[2] as i32) as i64) * ((st.sgpr[3] as i32) as i64)) >> 32) as u32,
        "s_mul_hi_i32"
    );

    let sext24 = |x: u32| -> i64 { (((x << 8) as i32) >> 8) as i64 };
    for l in 0..64usize {
        let inp = &st.vgpr[l];
        let v = &r.vgpr[l];
        assert_eq!(v[10], (((inp[0] as u64) * (inp[1] as u64)) >> 32) as u32, "lane {l} v_mul_hi_u32");
        assert_eq!(
            v[11],
            ((((inp[0] as i32) as i64) * ((inp[1] as i32) as i64)) >> 32) as u32,
            "lane {l} v_mul_hi_i32"
        );
        let prod24 = sext24(inp[2]) * sext24(inp[3]);
        assert_eq!(v[12], prod24 as u32, "lane {l} v_mul_i32_i24");
        assert_eq!(v[13], (prod24 >> 32) as u32, "lane {l} v_mul_hi_i32_i24");
        assert_eq!(v[14], (prod24 as i32).wrapping_add(inp[4] as i32) as u32, "lane {l} v_mad_i32_i24");

        let val64 = ((inp[7] as u64) << 32) | (inp[6] as u64);
        let sh = inp[5] & 63;
        let lshr = val64 >> sh;
        let ashr = ((val64 as i64) >> sh) as u64;
        assert_eq!((v[16], v[17]), (lshr as u32, (lshr >> 32) as u32), "lane {l} v_lshrrev_b64");
        assert_eq!((v[18], v[19]), (ashr as u32, (ashr >> 32) as u32), "lane {l} v_ashrrev_i64");

        let borrow_in = if l % 2 == 0 { 1u128 } else { 0u128 };
        let a64 = val64 as u128;
        let b64 = (((inp[9] as u64) << 32) | (inp[8] as u64)) as u128;
        let diff64 = a64.wrapping_sub(b64).wrapping_sub(borrow_in) as u64;
        assert_eq!((v[20], v[21]), (diff64 as u32, (diff64 >> 32) as u32), "lane {l} v_sub_co_ci_u32");
        let subrev = (inp[6] as u64).wrapping_sub(inp[8] as u64).wrapping_sub(borrow_in as u64) as u32;
        assert_eq!(v[22], subrev, "lane {l} v_subrev_co_ci_u32");
    }
}

/// image_get_resinfo with DIM 1D or 2D on a 2D MSAA T# (TYPE 14) or a 2D MSAA-array T# (TYPE 15)
/// gives (width, height, 0, 1) on RDNA2 (AnyPS5#25723, gfx1036). A 4x2 texture of 4 samples, and
/// a 4x2 array of 3 slices of 4 samples. The level count is 1 on every host: queried through a
/// Texture2D handle, the multisampled view reported log2(samples) + 1 = 3 on RADV.
#[test]
fn gpu_image_get_resinfo_dim_2d_of_an_msaa_tsharp() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mimg = |vdata: u64, srsrc: u64, dim: u64| encode("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", vdata), ("VADDR", 3), ("SRSRC", srsrc), ("DMASK", 0xF), ("DIM", dim)], None).unwrap();
    // T# 14 in s[16:23], T# 15 in s[24:31]; DIM 2D, then DIM 1D, on each
    let mut code = Vec::new();
    code.extend(mimg(20, 4, 1));
    code.extend(mimg(24, 4, 0));
    code.extend(mimg(28, 6, 1));
    code.extend(mimg(32, 6, 0));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 14));
    st.sgpr[24..32].copy_from_slice(&tsharp(2, 77, 15));
    for r in st.vgpr.iter_mut() {
        r[3] = 0;
    }
    let tex = [
        TestTexture { dim: TexDim::D2Ms(4), format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 2, depth: 1, texel_bytes: 16, data: vec![0; 4 * 2 * 4 * 16], uav: false },
        TestTexture { dim: TexDim::D2MsArray(4), format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 2, depth: 3, texel_bytes: 16, data: vec![0; 3 * 4 * 2 * 4 * 16], uav: false },
    ];
    let r = runner::run_code_with(&code, &st, &[], &tex, &[]).unwrap();
    assert_eq!(&r.vgpr[0][20..24], &[4, 2, 0, 1], "resinfo (w, h, d, mips) of DIM 2D on a 4-sample T#");
    assert_eq!(&r.vgpr[0][24..28], &[4, 2, 0, 1], "resinfo (w, h, d, mips) of DIM 1D on a 4-sample T#");
    assert_eq!(&r.vgpr[0][28..32], &[4, 2, 0, 1], "resinfo (w, h, d, mips) of DIM 2D on a 4-sample array T#");
    assert_eq!(&r.vgpr[0][32..36], &[4, 2, 0, 1], "resinfo (w, h, d, mips) of DIM 1D on a 4-sample array T#");
}

/// image_get_resinfo at a mip past the T#'s last level: RDNA2 gives 0 for width, height and
/// depth or layers and the level count as usual (AnyPS5#22724, gfx1036; D3D12 GetDimensions
/// zeroes them past the view's levels, so only channels the translators fill in can differ).
/// Single-level 2D, 2D array, 3D and cube T#s queried at mips 0, 1, 2, 15 and 0xffffffff.
#[test]
fn gpu_image_get_resinfo_past_last_level() {
    if !gpu_ready() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mimg = |vdata: u64, srsrc: u64, dim: u64| encode("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", vdata), ("VADDR", 3), ("SRSRC", srsrc), ("DMASK", 0xF), ("DIM", dim)], None).unwrap();
    // (name, VDATA, SRSRC, DIM, T# TYPE)
    let cases: [(&str, u64, u64, u64, u32); 4] = [("2D", 20, 4, 1, 9), ("2D array", 24, 6, 5, 13), ("3D", 28, 8, 2, 10), ("cube", 32, 10, 3, 11)];
    let mut code = Vec::new();
    for &(_, vdata, srsrc, dim, _) in &cases {
        code.extend(mimg(vdata, srsrc, dim));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    for (i, &(_, _, srsrc, _, ty)) in cases.iter().enumerate() {
        let s = 4 * srsrc as usize;
        st.sgpr[s..s + 8].copy_from_slice(&tsharp(2 * i as u32, 77, ty));
    }
    let mips = [0u32, 1, 2, 15, 0xFFFF_FFFF];
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[3] = mips[l % mips.len()];
    }
    let t = |dim: TexDim, width: u32, height: u32, depth: u32| TestTexture { dim, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width, height, depth, texel_bytes: 16, data: vec![0; (width * height * depth * 16) as usize], uav: false };
    let tex = [t(TexDim::D2, 8, 4, 1), t(TexDim::D2Array, 8, 4, 3), t(TexDim::D3, 8, 4, 2), t(TexDim::Cube, 4, 4, 12)];
    let r = runner::run_code_with(&code, &st, &[], &tex, &[]).unwrap();
    let mut bad = Vec::new();
    for (l, &mip) in mips.iter().enumerate() {
        for &(name, vdata, ..) in &cases {
            let v = &r.vgpr[l][vdata as usize..vdata as usize + 4];
            eprintln!("resinfo {name:9} mip {mip:#010x}: {v:?}");
            if mip >= 1 && v != [0, 0, 0, 1] {
                bad.push(format!("{name} mip {mip:#x}: {v:?}, RDNA2 gives [0, 0, 0, 1]"));
            }
        }
    }
    assert!(bad.is_empty(), "resinfo past the last level:\n{}", bad.join("\n"));
}

/// The list loop of a retail volumetric light PS – a wave32 program under `s_wqm`: per-lane
/// mask words OR-reduced over the wave (`s_or_saveexec -1`, dpp8 x3, dpp,
/// permlanex16, `v_readlane 0`), then a bit loop whose state is written back per lane (`v_mov`)
/// and read with `v_readfirstlane`: `s_ff1`, `1 << (n & 31)`, xor, `s_cmp_eq` / `s_cbranch_scc0`.
/// Exports (iterations, the wave's OR, the pixel's own word, 1).
fn list_loop_ps() -> super::Translation {
    use super::PsInterface;
    let raw = |b: &[u8]| b.to_vec();
    let head = [
        ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 126)], None),
        ins("S_WQM_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)], None),
        ins("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 2)], None),
        ins("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 3)], None),
        ins("V_AND_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 135), ("VSRC1", 5)], None),
        ins("V_AND_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 135), ("VSRC1", 6)], None),
        ins("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 136), ("VSRC1", 6)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 129)], None),
        ins("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 5), ("VSRC1", 11)], None),
        ins("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 6), ("VSRC1", 11)], None),
        ins("V_OR_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 7), ("VSRC1", 4)], None),
        // the console's wave OR of v4 over the live lanes, into s3
        ins("S_OR_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 193)], None),
        ins("S_AND_B32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 20), ("SSRC1", 1)], None),
        ins("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 128), ("SRC1", V + 4), ("SRC2", 3)], None),
        raw(&[0xe9, 0x00, 0x00, 0x38, 0x00, 0xc1, 0x54, 0xde]), // v_or_b32_dpp8 v0, v0, v0
        raw(&[0xe9, 0x00, 0x00, 0x38, 0x00, 0x12, 0x60, 0x93]),
        raw(&[0xe9, 0x00, 0x00, 0x38, 0x00, 0x24, 0x09, 0x00]),
        raw(&[0xfa, 0x00, 0x00, 0x38, 0x00, 0x40, 0x09, 0xff]), // v_or_b32_dpp v0, v0, v0
        raw(&[0x01, 0x00, 0x78, 0xd7, 0x00, 0x01, 0x01, 0x02]), // v_permlanex16_b32 v1, v0, 0, 0
        raw(&[0x01, 0x01, 0x00, 0x38]),                         // v_or_b32 v0, v1, v0
        raw(&[0x03, 0x00, 0x60, 0xd7, 0x00, 0x01, 0x01, 0x00]), // v_readlane_b32 s3, v0, 0
        raw(&[0x01, 0x03, 0xfe, 0xbe]),                         // s_mov_b32 exec_lo, s1
        ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 128)], None),
        ins("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 3), ("SSRC1", 128)], None),
    ]
    .concat();
    let body = [
        ins("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 8)], None),
        ins("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 4)], None),
        ins("S_AND_B32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 5), ("SSRC1", 159)], None),
        ins("S_LSHL_B32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 129), ("SSRC1", 12)], None),
        ins("S_XOR_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 5), ("SSRC1", 4)], None),
        ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 8), ("SSRC1", 129)], None),
        ins("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 5)], None),
    ]
    .concat();
    let tail = [
        ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 20)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 8)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", 3)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 12)], None),
        ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 4)], None),
        ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 242)], None),
        ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 10), ("VSRC1", 12), ("VSRC2", 13), ("VSRC3", 11), ("DONE", 1), ("VM", 1)], None),
    ]
    .concat();
    // head; scc1 → tail; v8 = s3; branch over the write-back; L: v8 = s1; body; scc0 → L; tail
    let mut code = head;
    let skip_all = (4 + 4 + 4 + body.len() + 4) / 4; // v_mov, s_branch, v_mov, body, s_cbranch
    code.extend(ins("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", skip_all as u64)], None));
    code.extend(ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 3)], None));
    code.extend(ins("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None));
    code.extend(ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 1)], None));
    code.extend(&body);
    let back = (-((4 + body.len() as i64 + 4) / 4)) as u16 as u64;
    code.extend(ins("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", back)], None));
    code.extend(tail);
    code.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
    let psi = PsInterface { inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x302, attrs: None };
    translate(&code, &Options { mode: Mode::Pixel(psi), wave64: false, ..Default::default() }).unwrap()
}

/// `list_loop_ps` drawn on two triangles sharing a diagonal: every pixel's bit loop runs
/// popcount(wave OR) times (at most 16), never to the loop budget. The waves on the diagonal start
/// with helper lanes; `v_readfirstlane` read the first of them, whose registers an RDNA2 driver
/// gives as 0 once a wave op ran in the loop, and the loop never saw its state change.
#[test]
fn gpu_ps_readfirstlane_loop_on_a_primitive_edge_ends() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
    let vs = super::dxc::compile("float4 main(float2 p : POSITION) : SV_Position { return float4(p, 0.5, 1.0); }\n", "main", "vs_6_6", &[]).unwrap();
    let pt = list_loop_ps();
    assert!(pt.complete(), "{:?}\n{}", pt.unsupported_why, pt.hlsl);
    // the fourth channel: the hardware wave's lane count
    let hlsl = pt.hlsl.replacen("    if (!xcl_live) discard;", "    xo.t0[3] = (float)WaveGetLaneCount();\n    if (!xcl_live) discard;", 1);
    let ps = super::dxc::compile(&hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    let quad = [-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
    let layout = [runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 }];
    let size = 32usize;
    let px = runner::draw(&vs, &ps, &layout, &quad, 8, &[], &[], size as u32, [9.0; 4]).unwrap();
    let lanes: std::collections::BTreeSet<u32> = px.iter().map(|p| p[3] as u32).collect();
    let bad: Vec<(usize, usize, [f32; 4])> = (0..size * size)
        .filter(|&i| {
            let p = px[i];
            let (it, or, own) = (p[0] as u32, p[1] as u32, p[2] as u32);
            own.count_ones() != 2 || or & own != own || it != or.count_ones() || it > 16
        })
        .map(|i| (i % size, i / size, px[i]))
        .collect();
    let max_it = px.iter().map(|p| p[0] as u32).max().unwrap_or(0);
    println!("{:?}: lane counts {lanes:?}; most iterations {max_it}; {} of {} pixels wrong, first {:?}", runner::adapter(), bad.len(), size * size, bad.iter().take(8).collect::<Vec<_>>());
    assert!(bad.is_empty(), "{} pixels wrong (iterations, wave OR, own word, lane count), first {:?}", bad.len(), &bad[..bad.len().min(8)]);
}

/// A wave32 pixel program under `s_wqm` with the march loop of a retail volumetric PS after
/// another loop: every lane counts steps in a vector register, and the loop goes on as long as
/// a lane of the mask taken from EXEC has not reached 6 (`v_cmp_le_f32`, `s_andn2 mask, mask,
/// vcc`, `s_cbranch_scc1`). Exports (passes of the march, the lane's steps, 1, 1).
fn mask_loop_ps() -> super::Translation {
    use super::PsInterface;
    let before = [
        ins("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 2)], None),
        ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 9), ("SSRC0", 9), ("SSRC1", 129)], None),
        ins("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 9), ("SSRC1", 130)], None),
    ]
    .concat();
    let march = [
        ins("V_ADD_F32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 242), ("VSRC1", 9)], None),
        ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 8), ("SSRC1", 129)], None),
        ins("V_CMP_LE_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 9)], Some(0x40C0_0000)),
        ins("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 106)], None),
    ]
    .concat();
    let back = |body: &[u8]| (-((body.len() as i64 + 4) / 4)) as u16 as u64;
    let mut code = [
        ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 126)], None),
        ins("S_WQM_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)], None),
        ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 9), ("SSRC0", 128)], None),
    ]
    .concat();
    code.extend(&before);
    code.extend(ins("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", back(&before))], None));
    code.extend(ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)], None));
    code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 128)], None));
    code.extend(ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 126)], None));
    code.extend(&march);
    code.extend(ins("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", back(&march))], None));
    code.extend(
        [
            ins("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 20)], None),
            ins("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 8)], None),
            ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 242)], None),
            ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 10), ("VSRC1", 9), ("VSRC2", 11), ("VSRC3", 11), ("DONE", 1), ("VM", 1)], None),
            ins("S_ENDPGM", "ENC_SOPP", &[], None),
        ]
        .concat(),
    );
    let psi = PsInterface { inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x302, attrs: None };
    translate(&code, &Options { mode: Mode::Pixel(psi), wave64: false, ..Default::default() }).unwrap()
}

/// Draws `mask_loop_ps` on two triangles sharing a diagonal (helper lanes along it): the march
/// ends after its 6 passes in every pixel. A helper lane's steps cannot be read from the second
/// pass on with an RDNA2 driver, its bit stayed in the mask and the loop ran to the loop budget
/// (`mask_live` in the prelude).
#[test]
fn gpu_ps_mask_loop_on_a_primitive_edge_ends() {
    if !gpu_ready() {
        return;
    }
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
    let vs = super::dxc::compile("float4 main(float2 p : POSITION) : SV_Position { return float4(p, 0.5, 1.0); }\n", "main", "vs_6_6", &[]).unwrap();
    let pt = mask_loop_ps();
    assert!(pt.complete(), "{:?}\n{}", pt.unsupported_why, pt.hlsl);
    assert!(pt.hlsl.contains("scc = (uint)(mask_live("), "{}", pt.hlsl);
    // the fourth channel: the hardware wave's lane count
    let hlsl = pt.hlsl.replacen("    if (!xcl_live) discard;", "    xo.t0[3] = (float)WaveGetLaneCount();\n    if (!xcl_live) discard;", 1);
    let ps = super::dxc::compile(&hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    let quad = [-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
    let layout = [runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 }];
    let size = 32usize;
    let px = runner::draw(&vs, &ps, &layout, &quad, 8, &[], &[], size as u32, [9.0; 4]).unwrap();
    let lanes: std::collections::BTreeSet<u32> = px.iter().map(|p| p[3] as u32).collect();
    let bad: Vec<(usize, usize, [f32; 4])> = (0..size * size).filter(|&i| px[i][0] != 6.0 || px[i][1] != 6.0).map(|i| (i % size, i / size, px[i])).collect();
    println!("{:?}: lane counts {lanes:?}; {} of {} pixels wrong, first {:?}", runner::adapter(), bad.len(), size * size, bad.iter().take(8).collect::<Vec<_>>());
    assert!(bad.is_empty(), "{} pixels wrong (passes, steps, 1, lane count), first {:?}", bad.len(), &bad[..bad.len().min(8)]);
}

/// Compute shaders only declare `globallycoherent` on their flat memory windows (`mem`, `mem1`…
/// and `XCL_HEAP_WINDOWS` heap views) when the program uses cross-workgroup coherence: memory
/// atomics, `GLC`/`DLC`/`SLC` memory ops, or cache invalidations. Both the HLSL and direct DXIL
/// backends follow the same predicate.
#[test]
fn compute_globallycoherent_only_when_needed() {
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 1 };
    let enc = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let prog = |insts: &[(&str, &str, &[(&str, u64)])]| {
        let mut c = Vec::new();
        for &(n, e, f) in insts {
            c.extend(enc(n, e, f));
        }
        c.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
        c
    };

    // Plain compute: SMEM + MUBUF load/store + LDS + s_barrier, no GLC/DLC/SLC/atomics/inv.
    let plain = prog(&[
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125)]),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 1), ("ADDR", 0), ("SADDR", 0)]),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 0), ("DATA0", 1)]),
        ("S_BARRIER", "ENC_SOPP", &[]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 0), ("DATA", 1), ("SADDR", 0)]),
    ]);

    let coherent_cases: Vec<(&str, Vec<u8>)> = vec![
        ("smem glc", prog(&[("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125), ("GLC", 1)])])),
        ("smem dlc", prog(&[("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125), ("DLC", 1)])])),
        ("mubuf glc", prog(&[("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128), ("GLC", 1)])])),
        ("global dlc", prog(&[("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 1), ("ADDR", 0), ("SADDR", 0), ("DLC", 1)])])),
        ("mubuf slc", prog(&[("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128), ("SLC", 1)])])),
        ("buffer_atomic", prog(&[("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)])])),
        ("global_atomic", prog(&[("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 1), ("ADDR", 0), ("DATA", 2), ("SADDR", 0)])])),
        ("flat_atomic", prog(&[("FLAT_ATOMIC_ADD", "ENC_FLAT", &[("VDST", 1), ("ADDR", 2), ("DATA", 4)])])),
        ("s_dcache_inv", prog(&[("S_DCACHE_INV", "ENC_SMEM", &[])])),
        ("s_gl1_inv", prog(&[("S_GL1_INV", "ENC_SMEM", &[])])),
        ("buffer_gl0_inv", prog(&[("BUFFER_GL0_INV", "ENC_MUBUF", &[])])),
        ("buffer_gl1_inv", prog(&[("BUFFER_GL1_INV", "ENC_MUBUF", &[])])),
    ];

    for (windows, heap) in [(1u32, false), (2, false), (2, true)] {
        let opt = Options { mode: Mode::Compute(cs.clone()), wave64: false, mem_windows: windows, heap_windows: heap, ..Default::default() };
        let t = translate(&plain, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        assert!(!t.hlsl.contains("globallycoherent"), "plain CS (windows={windows}, heap={heap}) must not emit globallycoherent:\n{}", t.hlsl);

        for (label, code) in &coherent_cases {
            let t = translate(code, &opt).unwrap();
            assert!(t.complete(), "{label}: {:?}", t.unsupported_why);
            assert!(t.hlsl.contains("globallycoherent"), "{label} (windows={windows}, heap={heap}) must emit globallycoherent");
            if heap {
                assert!(t.hlsl.contains("#define XCL_MW globallycoherent RWByteAddressBuffer"), "{label}: heap window accessor must be globallycoherent");
            }
        }
    }

    // Kernel mode stays non-coherent even with atomics.
    let k = translate(&coherent_cases[6].1, &Options { mode: Mode::Kernel, wave64: false, ..Default::default() }).unwrap();
    assert!(!k.hlsl.contains("globallycoherent"), "kernel mode must not emit globallycoherent");

    // Direct DXIL backend: check the IsGloballyCoherent bit (0x4000) in annotateHandle properties
    // (RAW_BUFFER=11 | IsUAV=0x1000 = 4107 vs RAW_BUFFER=11 | IsUAV=0x1000 | IsGloballyCoherent=0x4000 = 20491).
    for mw in [1u32, super::HEAP_WINDOWS | 2] {
        let o = crate::gpu::dxil::kernel::KernelOptions { wave64: false, split: false, mem_windows: mw, sampler_base: 0 };
        let plain_k = crate::gpu::dxil::kernel::translate_compute(&plain, o, &cs).unwrap();
        let coh_k = crate::gpu::dxil::kernel::translate_compute(&coherent_cases[6].1, o, &cs).unwrap();
        if super::dxc::location().is_ok() {
            let plain_dis = crate::gpu::dxil::validate::disassemble(&plain_k.dxil).unwrap();
            let coh_dis = crate::gpu::dxil::validate::disassemble(&coh_k.dxil).unwrap();
            assert!(!plain_dis.contains("i32 20491"), "direct DXIL plain CS (mw={mw:#x}) must not set IsGloballyCoherent:\n{plain_dis}");
            assert!(coh_dis.contains("i32 20491"), "direct DXIL atomic CS (mw={mw:#x}) must set IsGloballyCoherent:\n{coh_dis}");
        }
    }
}

/// Real GPU check of both non-coherent (cached) and coherent (cross-workgroup atomic + GLC)
/// compute shaders across HLSL and direct DXIL.
#[test]
fn gpu_compute_coherent_and_cached_loads() {
    if !gpu_ready() {
        return;
    }
    let cs = super::CsInterface { threads: [32, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 };
    let enc = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));

    // 1. Plain cached compute: each thread reads a uniform dword via SMEM (s[0:1] = 0x100) and
    //    writes (val + group_id * 32 + tid) to mem[0x200 + (group_id * 32 + tid) * 4].
    let mut plain = Vec::new();
    plain.extend(enc("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125)]));
    plain.extend(enc("S_LSHL_B32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 2), ("SSRC1", 133)])); // s3 = tgid * 32
    plain.extend(enc("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 3), ("VSRC1", 0)])); // v1 = tgid * 32 + v0
    plain.extend(enc("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 4), ("VSRC1", 1)])); // v2 = mem[0x100] + idx
    plain.extend(enc("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 130), ("VSRC1", 1)])); // v3 = idx * 4
    plain.extend(enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 128)]));
    plain.extend(enc("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 3), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", 0x200)]));
    plain.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));

    let t_plain = translate(&plain, &Options { mode: Mode::Compute(cs.clone()), wave64: false, ..Default::default() }).unwrap();
    assert!(t_plain.complete(), "{:?}", t_plain.unsupported_why);
    assert!(!t_plain.hlsl.contains("globallycoherent"));
    let dxil_plain_hlsl = super::dxc::compile(&t_plain.hlsl, "main", "cs_6_6", &[]).unwrap();
    let dxil_plain_direct = crate::gpu::dxil::kernel::translate_compute(&plain, crate::gpu::dxil::kernel::KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }, &cs).unwrap().dxil;

    let mut io = vec![0u32; super::KERNEL_SGPRS];
    io[0] = 0x100;
    let mut mem = vec![0u32; 0x400 / 4];
    mem[0x100 / 4] = 0x1234_0000;
    for (tag, dxil) in [("hlsl", &dxil_plain_hlsl), ("direct", &dxil_plain_direct)] {
        let out = runner::dispatch(dxil, [4, 1, 1], &io, &mem).unwrap();
        for idx in 0..128usize {
            assert_eq!(out[(0x200 + idx * 4) / 4], 0x1234_0000 + idx as u32, "{tag} plain cached idx={idx}");
        }
    }

    // 2. Cross-workgroup coherent compute: 16 groups of 32 threads atomically increment a shared
    //    counter at mem[0x100] and record their returned ticket at mem[0x200 + ticket * 4].
    let mut coh = Vec::new();
    coh.extend(enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)])); // v1 = 0
    coh.extend(enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 129)])); // v2 = 1
    coh.extend(enc("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 3), ("ADDR", 1), ("DATA", 2), ("SADDR", 0), ("GLC", 1)]));
    coh.extend(enc("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 130), ("VSRC1", 3)])); // v4 = ticket * 4
    coh.extend(enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 128)]));
    coh.extend(enc("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 4), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", 0x200), ("GLC", 1)]));
    coh.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));

    let t_coh = translate(&coh, &Options { mode: Mode::Compute(cs.clone()), wave64: false, ..Default::default() }).unwrap();
    assert!(t_coh.complete(), "{:?}", t_coh.unsupported_why);
    assert!(t_coh.hlsl.contains("globallycoherent"));
    let dxil_coh_hlsl = super::dxc::compile(&t_coh.hlsl, "main", "cs_6_6", &[]).unwrap();
    let dxil_coh_direct = crate::gpu::dxil::kernel::translate_compute(&coh, crate::gpu::dxil::kernel::KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }, &cs).unwrap().dxil;

    let mem2 = vec![0u32; (0x200 + 512 * 4) / 4];
    for (tag, dxil) in [("hlsl", &dxil_coh_hlsl), ("direct", &dxil_coh_direct)] {
        let out = runner::dispatch(dxil, [16, 1, 1], &io, &mem2).unwrap();
        assert_eq!(out[0x100 / 4], 512, "{tag} coherent cross-group counter");
        for ticket in 0..512usize {
            assert_eq!(out[(0x200 + ticket * 4) / 4], ticket as u32, "{tag} coherent ticket={ticket}");
        }
    }
}

#[test]
fn gpu_v_dot8_u32_u4_edge_cases() {
    const CASES: [(u32, u32, u32); 16] = [
        (0, 0, 0),
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0),
        (0x0F0F_0F0F, 0x0F0F_0F0F, 123),
        (0xF0F0_F0F0, 0xF0F0_F0F0, 456),
        (0x0F0F_0F0F, 0xF0F0_F0F0, 0x1234_5678),
        (0x0123_4567, 0x89AB_CDEF, 0x1111_2222),
        (0xFEDC_BA98, 0x7654_3210, 0),
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_F8F7), // exact u32::MAX with sum 8*15*15 = 1800 (0x708): 0xFFFF_F8F7 + 0x708 = 0xFFFF_FFFF
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_F8F8), // wraps to 0 unclamped, saturates to u32::MAX clamped
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFF0),
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF),
        (0x8888_8888, 0x9999_9999, 0xFFFF_FE00),
        (0x1010_1010, 0x0204_0608, 0),
        (0x0102_0304, 0x0506_0708, 0x8000_0000),
        (0xA5A5_5A5A, 0x5A5A_A5A5, 0x7FFF_FFFF),
        (0x0000_000F, 0x0000_000F, 0xFFFF_FF1F),
    ];
    let p = prog(&[
        ("V_DOT8_U32_U4", "ENC_VOP3P", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2)], None),
        ("V_DOT8_U32_U4", "ENC_VOP3P", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2), ("CLAMP", 1)], None),
    ]);
    let dot_u4 = |a: u32, b: u32, c: u32, clamp: bool| -> u32 {
        let mut sum = c as u64;
        for i in 0..8 {
            sum += (((a >> (4 * i)) & 0xF) * ((b >> (4 * i)) & 0xF)) as u64;
        }
        if clamp { sum.min(u32::MAX as u64) as u32 } else { sum as u32 }
    };
    for wave64 in gpu_waves() {
        let mut st = KernelState::default();
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            let (a, b, c) = CASES[l % CASES.len()];
            (r[0], r[1], r[2]) = (a, b, c);
        }
        let (r, lanes) = run_wave(&p, &st, wave64);
        for (l, v) in r.vgpr.iter().take(lanes).enumerate() {
            let (a, b, c) = CASES[l % CASES.len()];
            assert_eq!(v[3], dot_u4(a, b, c, false), "wave{lanes} lane {l} unclamped v_dot8_u32_u4({a:#x}, {b:#x}, {c:#x})");
            assert_eq!(v[4], dot_u4(a, b, c, true), "wave{lanes} lane {l} clamped v_dot8_u32_u4({a:#x}, {b:#x}, {c:#x})");
        }
    }
}
