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

/// Builds a straight-line + loop program with `extra_classes` distinct blocks (each adding a
/// distinct immediate to `s0` / `v1`, separated by a trivial conditional branch so every block
/// becomes its own compact class), plus a bounded loop at the end.
fn multi_class_program(extra_classes: usize, loop_iters: u32) -> Vec<u8> {
    let i = |n: &str, e: &str, f: &[(&str, u64)], lit: Option<u32>| {
        encode(n, e, f, lit).unwrap_or_else(|x| panic!("{n}: {x}"))
    };
    let mut v = Vec::new();
    for k in 0..extra_classes {
        let lit = (k as u32 + 1) * 7 + 3;
        v.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 255)], Some(lit)));
        v.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 0), ("VSRC1", 1)], None));
        v.extend(i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None));
        v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)], None));
    }
    // Loop header block (distinct class) that counts `s4` up to `loop_iters`
    v.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)], None));
    v.extend(i("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 255)], Some(loop_iters)));
    v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None));
    v.extend(i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 129)], None));
    v.extend(i("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}

#[test]
fn parse_nest_and_group_size_policy() {
    use super::compact::{parse_nest, Nest, AUTO_NEST_GROUP_SIZE, AUTO_NEST_MIN_CLASSES};

    assert_eq!(parse_nest(None), Nest::Auto);
    assert_eq!(parse_nest(Some("")), Nest::Auto);
    assert_eq!(parse_nest(Some("auto")), Nest::Auto);
    assert_eq!(parse_nest(Some("AUTO")), Nest::Auto);
    assert_eq!(parse_nest(Some("0")), Nest::Flat);
    assert_eq!(parse_nest(Some("off")), Nest::Flat);
    assert_eq!(parse_nest(Some("OFF")), Nest::Flat);
    assert_eq!(parse_nest(Some("2")), Nest::Fixed(2));
    assert_eq!(parse_nest(Some("16")), Nest::Fixed(16));
    assert_eq!(parse_nest(Some("32")), Nest::Fixed(32));
    assert_eq!(parse_nest(Some("65536")), Nest::Fixed(65536));
    // Invalid values fall back to Auto
    for bad in ["1", "3", "30", "131072", "-4", "bogus"] {
        assert_eq!(parse_nest(Some(bad)), Nest::Auto, "bad={bad}");
    }

    assert_eq!(Nest::Flat.group_size(1024), None);
    assert_eq!(Nest::Auto.group_size(AUTO_NEST_MIN_CLASSES - 1), None);
    assert_eq!(Nest::Auto.group_size(AUTO_NEST_MIN_CLASSES), Some(AUTO_NEST_GROUP_SIZE));
    assert_eq!(Nest::Fixed(4).group_size(4), None);
    assert_eq!(Nest::Fixed(4).group_size(5), Some(4));

    assert_eq!(Nest::Auto.cache_tag(), None);
    assert_eq!(Nest::Flat.cache_tag().as_deref(), Some("xcl-direct-compact-nest-0"));
    assert_eq!(Nest::Fixed(16).cache_tag().as_deref(), Some("xcl-direct-compact-nest-16"));
}

#[test]
fn nested_compact_ir_structure_bounds_switch_fanout_and_merge_fanin() {
    use super::compact::{emit_with_nest, plan, Nest};
    use super::module::{Block, InstInfo};

    // 11 extra classes + loop header + exit block = 13 classes -> with group size 4:
    // 4 groups of sizes (4, 4, 4, 1), exercising full inner switches and a singleton tail group.
    let code = multi_class_program(11, 3);
    let cfg = super::cfg::Cfg::build(&code).unwrap();
    let leaders = cfg.blocks.iter().filter_map(|b| b.insts.first().map(|&i| cfg.insts[i].pc)).collect();
    let o = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };

    let inspect = |nest: Nest| -> (usize, usize, usize, usize) {
        let mut lw = super::lower::Lw::new(&cfg.insts, &leaders, o, &super::kernel::Entry::Kernel);
        lw.setup_lanes();
        let lane = lw.lane;
        let k = lw.c((crate::gpu::recomp::KERNEL_VGPRS * 4) as u32);
        let b = lw.mul(lane, k);
        lw.init.vgpr_base = Some(lw.addc(b, (crate::gpu::recomp::KERNEL_SGPRS * 4) as u32));
        let p = plan(&cfg, &lw).unwrap();
        assert_eq!(p.rep.len(), 13);
        let h0 = lw.block(None);
        let exit = lw.block(None);
        lw.edge(Block(0), h0);
        emit_with_nest(&mut lw, &cfg, &p, h0, exit, nest).unwrap();
        let m = &lw.s.m;
        let mut switches = 0usize;
        let mut max_switch_targets = 0usize;
        let mut preds = vec![0usize; m.block_count()];
        for b in 0..m.block_count() {
            if let Some(&last) = m.block_insts(Block(b as u32)).last() {
                if let InstInfo::Term(cond, succs) = m.inst_info(last) {
                    if cond.is_some() && succs.len() > 2 {
                        switches += 1;
                        max_switch_targets = max_switch_targets.max(succs.len());
                    }
                    for s in succs {
                        preds[s.0 as usize] += 1;
                    }
                }
            }
        }
        let max_preds = preds.into_iter().max().unwrap_or(0);
        (p.rep.len(), switches, max_switch_targets, max_preds)
    };

    let (classes, flat_sw, flat_max_sw, flat_max_preds) = inspect(Nest::Flat);
    assert_eq!(flat_sw, 1);
    assert_eq!(flat_max_sw, classes);
    assert!(flat_max_preds >= classes, "flat latch should merge all {classes} classes, got {flat_max_preds}");

    // Group size 4 on 13 classes: 1 outer switch (4 groups) + 3 inner switches (4th group is singleton Br) = 4 switches
    let (_, n4_sw, n4_max_sw, n4_max_preds) = inspect(Nest::Fixed(4));
    assert_eq!(n4_sw, 4);
    assert_eq!(n4_max_sw, 4);
    assert!(n4_max_preds <= 5, "nested group-4 max block fan-in should be <= 5, got {n4_max_preds}");

    // Group size 8 on 13 classes: 1 outer switch (2 groups: 8 and 5) -> outer switch has 2 targets (still Op::Switch),
    // inner switches have 8 and 5 targets; max block fan-in bounded by group size + 1 (loop header early exit).
    let (_, _, n8_max_sw, n8_max_preds) = inspect(Nest::Fixed(8));
    assert_eq!(n8_max_sw, 8);
    assert!(n8_max_preds <= 9, "nested group-8 max block fan-in should be <= 9, got {n8_max_preds}");
}

#[test]
fn gpu_nested_compact_matches_plain_and_flat() {
    use super::compact::Nest;
    use super::kernel::translate_full_nested;

    match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {}
        (a, _, d) => return runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" }),
    }
    let o = KernelOptions { wave64: true, split: runner::split_wave64(), mem_windows: 1, sampler_base: 0 };

    // 1. Repeated-block program (11 classes): exercise Flat, Fixed(2) (singleton tail), Fixed(4), Fixed(8)
    let mut st = KernelState::default();
    (st.sgpr[1], st.sgpr[2]) = (3, 12);
    for lane in 0..KERNEL_LANES {
        st.vgpr[lane][0] = lane as u32;
    }
    let code = repeated(40);
    let plain = {
        let k = translate_full_as(&code, o, false).expect("plain");
        runner::run_with_statics(&k.dxil, &st, &[], &[], &[], &k.static_samplers).expect("plain GPU run")
    };
    for nest in [Nest::Flat, Nest::Fixed(2), Nest::Fixed(4), Nest::Fixed(8)] {
        let k = translate_full_nested(&code, o, nest).unwrap_or_else(|e| panic!("nest {nest:?}: {e}"));
        let got = runner::run_with_statics(&k.dxil, &st, &[], &[], &[], &k.static_samplers)
            .unwrap_or_else(|e| panic!("GPU run {nest:?}: {e}"));
        assert_eq!((&got.sgpr, &got.vgpr, got.scc), (&plain.sgpr, &plain.vgpr, plain.scc), "repeated {nest:?}");
    }

    // 2. Multi-class program (17 classes: 4*4 + 1 singleton tail) with normal loop termination
    let mut st2 = KernelState::default();
    for lane in 0..KERNEL_LANES {
        st2.vgpr[lane][1] = lane as u32;
    }
    let mc = multi_class_program(15, 5);
    let plain2 = {
        let k = translate_full_as(&mc, o, false).expect("plain multi-class");
        runner::run_with_statics(&k.dxil, &st2, &[], &[], &[], &k.static_samplers).expect("plain multi-class GPU run")
    };
    assert_eq!(plain2.sgpr[4], 5);
    assert_eq!(plain2.sgpr[5], 1);
    for nest in [Nest::Flat, Nest::Fixed(4), Nest::Fixed(8)] {
        let k = translate_full_nested(&mc, o, nest).unwrap_or_else(|e| panic!("mc nest {nest:?}: {e}"));
        let got = runner::run_with_statics(&k.dxil, &st2, &[], &[], &[], &k.static_samplers)
            .unwrap_or_else(|e| panic!("mc GPU run {nest:?}: {e}"));
        assert_eq!((&got.sgpr, &got.vgpr, got.scc), (&plain2.sgpr, &plain2.vgpr, plain2.scc), "multi-class {nest:?}");
    }

    // 3. Nested loop-budget exhaustion: budget hits 0 inside a nested group and exits through group_latch -> exit
    let mc_budget = multi_class_program(9, super::lower::LOOP_BUDGET + 1);
    let plain3 = {
        let k = translate_full_as(&mc_budget, o, false).expect("plain budget");
        runner::run_with_statics(&k.dxil, &KernelState::default(), &[], &[], &[], &k.static_samplers)
            .expect("plain budget GPU run")
    };
    assert_eq!((plain3.sgpr[4], plain3.sgpr[5]), (super::lower::LOOP_BUDGET, 0));
    let k_n4 = translate_full_nested(&mc_budget, o, Nest::Fixed(4)).expect("nested budget");
    let got3 = runner::run_with_statics(&k_n4.dxil, &KernelState::default(), &[], &[], &[], &k_n4.static_samplers)
        .expect("nested budget GPU run");
    assert_eq!((got3.sgpr, got3.vgpr, got3.scc), (plain3.sgpr, plain3.vgpr, plain3.scc), "nested loop budget");
}
