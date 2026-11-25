//! The compact form (`compact`) against the plain one.

use super::kernel::{compact_mode, translate_full, translate_full_as, KernelOptions};
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::runner::{self, KernelState};
use crate::gpu::recomp::KERNEL_LANES;

fn repeated(copies: usize) -> Vec<u8> {
    let i = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut v = Vec::new();
    for _ in 0..copies {
        v.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)]));
        v.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 256), ("VSRC1", 1)]));
        v.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 130)]));
        v.extend(i("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", 5), ("VSRC1", 0)]));
        v.extend(i("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 106)]));
        v.extend(i("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]));
        v.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 256), ("VSRC1", 3)]));
        v.extend(i("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 6)]));
        v.extend(i("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 2)]));
        v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]));
        v.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 3), ("SSRC1", 129)]));
    }
    v.extend(i("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 133)]));
    v.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 256), ("VSRC1", 2)]));
    v.extend(i("S_SUB_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)]));
    v.extend(i("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)]));
    v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)]));
    v.extend(i("S_ENDPGM", "ENC_SOPP", &[]));
    v
}

fn size(copies: usize, o: KernelOptions, compact: bool) -> usize {
    translate_full_as(&repeated(copies), o, compact).unwrap_or_else(|e| panic!("{copies} copies, compact {compact}: {e}")).dxil.len()
}

#[test]
fn the_compact_form_lowers_a_repeated_block_once() {
    for (wave64, split) in [(false, false), (true, true), (true, false)] {
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let (compact, plain) = (size(128, o, true) - size(64, o, true), size(128, o, false) - size(64, o, false));
        assert!(compact * 20 < plain, "wave64 {wave64} split {split}: 64 more copies cost {compact} B compact, {plain} B plain");
    }
}

#[test]
fn a_large_program_that_repeats_itself_takes_the_compact_form() {
    if compact_mode().is_some() {
        return;
    }
    let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    for (copies, compact) in [(2500, true), (2000, false)] {
        assert_eq!(translate_full(&repeated(copies), o).unwrap().dxil.len(), size(copies, o, compact), "{copies} copies");
    }
}

#[test]
fn gpu_compact_form_runs_as_the_plain_form() {
    match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {}
        (a, _, d) => return runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" }),
    }
    let mut st = KernelState::default();
    (st.sgpr[1], st.sgpr[2]) = (3, 12);
    for lane in 0..KERNEL_LANES {
        st.vgpr[lane][0] = lane as u32;
    }
    let code = repeated(40);
    let o = KernelOptions { wave64: true, split: runner::split_wave64(), mem_windows: 1, sampler_base: 0 };
    let run = |compact: bool| {
        let k = translate_full_as(&code, o, compact).unwrap_or_else(|e| panic!("compact {compact}: {e}"));
        runner::run_with_statics(&k.dxil, &st, &[], &[], &[], &k.static_samplers).expect("GPU run")
    };
    let (c, p) = (run(true), run(false));
    assert_eq!(&c.sgpr[..6], &[120, 3, 12, 1, 0, 80]);
    for lane in 0..KERNEL_LANES as u32 {
        let masked = lane * (lane.saturating_sub(1) / 2);
        assert_eq!(&c.vgpr[lane as usize][..4], &[lane, 40 * lane, 5 * lane, masked], "lane {lane}");
    }
    assert_eq!((c.sgpr, c.vgpr, c.scc), (p.sgpr, p.vgpr, p.scc));
}

#[test]
fn gpu_compact_form_spends_the_loop_budget_as_the_plain_form() {
    match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {}
        (a, _, d) => return runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" }),
    }
    let mut v = Vec::new();
    v.extend(encode("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None).unwrap());
    v.extend(encode("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 255)], Some(super::lower::LOOP_BUDGET + 1)).unwrap());
    v.extend(encode("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None).unwrap());
    v.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 129)], None).unwrap());
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let o = KernelOptions { wave64: true, split: runner::split_wave64(), mem_windows: 1, sampler_base: 0 };
    let run = |compact: bool| {
        let k = translate_full_as(&v, o, compact).unwrap_or_else(|e| panic!("compact {compact}: {e}"));
        runner::run_with_statics(&k.dxil, &KernelState::default(), &[], &[], &[], &k.static_samplers).expect("GPU run")
    };
    let (c, p) = (run(true), run(false));
    assert_eq!(&c.sgpr[..2], &[super::lower::LOOP_BUDGET, 0]);
    assert_eq!(&c.sgpr[..2], &p.sgpr[..2]);
}

#[test]
fn a_mask_read_whole_after_one_copy_only_splits_the_class() {
    let i = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let cmp = |v: &mut Vec<u8>| {
        v.extend(i("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)]));
        v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    };
    let mut v = Vec::new();
    cmp(&mut v);
    v.extend(i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 128), ("VSRC1", 3)]));
    v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    cmp(&mut v);
    v.extend(i("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]));
    v.extend(i("S_ENDPGM", "ENC_SOPP", &[]));
    let cfg = super::cfg::Cfg::build(&v).unwrap();
    let leaders = cfg.blocks.iter().filter_map(|b| b.insts.first().map(|&i| cfg.insts[i].pc)).collect();
    let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    let lw = super::lower::Lw::new(&cfg.insts, &leaders, o, &super::kernel::Entry::Kernel);
    let p = super::compact::plan(&cfg, &lw).unwrap();
    assert_eq!(p.order.len(), 4);
    assert_ne!(p.class[0], p.class[2]);
}
