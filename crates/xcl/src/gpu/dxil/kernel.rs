//! Direct translation of an RDNA2 program in the recompiler's kernel ABI
//! (`recomp::runner`): u0 `io` = 128 SGPRs (SCC in slot 125) then 64 lanes ×
//! 256 VGPRs; u1 `res` = the same layout after the program (+ exports);
//! u2 `mem` = flat memory. As the HLSL kernel does, the registers the program
//! touches are loaded at entry and stored at the end (SGPRs + SCC by lane 0,
//! VGPRs by every lane); the rest of `res` stays 0.
//!
//! [`translate_compute`] is the recompiler's compute mode (the real CS ABI,
//! `recomp::CsInterface`): user SGPRs from `io`, then the enabled group ids
//! and TG_SIZE; v0..v2 = thread id in the group; results only through memory.

use std::collections::{BTreeSet, HashMap};

use super::cfg::{Cfg, Term};
use super::container::Stage;
use super::lower::{ComputeInit, Init, Lw, Partner, VAR_BUDGET, VAR_SCC, VAR_VGPR};
use super::module::{Block, Pred, Value};
use super::shader::Shader;
use super::ssa::Ssa;
use crate::gpu::recomp::{CsInterface, MsInterface, PsInterface, VsInterface, KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_VGPRS, MESH_HEADER};

#[derive(Debug, Clone, Copy)]
pub struct KernelOptions {
    /// wave64 program (64-bit EXEC / VCC) or wave32
    pub wave64: bool,
    /// the GPU runs only 32-lane waves: a wave64 program runs as two halves
    pub split: bool,
    /// 4 GiB memory windows the program may address (`recomp::Options::mem_windows`):
    /// `mem` at u2, then `mem1`.. at u3..
    pub mem_windows: u32,
    /// first register (space 1) of the static samplers (another stage may come first)
    pub sampler_base: u32,
}

/// Version of the direct translator's output, part of the on-disk translation cache's
/// identity (`recomp::pso`): bump it on any change that alters the emitted containers
/// (the `gpu/dxil` sources are not in build.rs's translator source hash).
pub const TRANSLATOR_VERSION: u32 = 4;

/// `XCL_DIRECT_DXIL=1`: console compute shaders (and the kernel tests) go through this
/// translator, with the HLSL path as the fallback for what it cannot translate yet.
pub fn enabled() -> bool {
    std::env::var("XCL_DIRECT_DXIL").is_ok_and(|v| v == "1" || v == "all")
}

/// `XCL_DIRECT_DXIL=all`: console vertex and pixel shaders too (each stage falls back to
/// the HLSL path on its own; the signatures are declared as the HLSL entries declare them,
/// so a direct stage links with an HLSL one).
pub fn enabled_graphics() -> bool {
    std::env::var("XCL_DIRECT_DXIL").is_ok_and(|v| v == "all")
}

/// Translates `code` into a signed DXIL compute container, or names the first
/// instruction it cannot translate.
pub fn translate(code: &[u8], o: KernelOptions) -> Result<Vec<u8>, String> {
    translate_full(code, o).map(|k| k.dxil)
}

/// A translated kernel: the container and the static samplers it samples with
/// (S# built from constants in the code, register `s<i>, space1` — the HLSL
/// kernel's `Translation::static_samplers`).
pub struct Kernel {
    pub dxil: Vec<u8>,
    pub static_samplers: Vec<([u32; 4], bool)>,
}

pub fn translate_full(code: &[u8], o: KernelOptions) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Kernel)
}

/// A mesh-mode (NGG) program: the first pass that writes one record per group into
/// `res` (the recompiler's `Mode::Mesh`; the pass-through mesh shader draws them).
pub fn translate_mesh(code: &[u8], o: KernelOptions, ms: &MsInterface) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Mesh(ms.clone()))
}

/// A vertex shader (the recompiler's `Mode::Vertex`; wave-size independent).
pub fn translate_vertex(code: &[u8], o: KernelOptions, vs: &VsInterface) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Vertex(vs.clone()))
}

/// A pixel shader (the recompiler's `Mode::Pixel`; wave-size independent).
pub fn translate_pixel(code: &[u8], o: KernelOptions, ps: &PsInterface) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Pixel(ps.clone()))
}

/// The entry ABI.
#[derive(Clone)]
enum Entry {
    Kernel,
    Compute(CsInterface),
    Mesh(MsInterface),
    Vertex(VsInterface),
    Pixel(PsInterface),
}

/// A compute-mode program (the CS ABI of `cs`).
pub fn translate_compute(code: &[u8], o: KernelOptions, cs: &CsInterface) -> Result<Kernel, String> {
    translate_entry(code, o, Entry::Compute(cs.clone()))
}

fn translate_entry(code: &[u8], o: KernelOptions, entry: Entry) -> Result<Kernel, String> {
    // memory windows through the descriptor heap (`recomp::HEAP_WINDOWS` in the count,
    // `XCL_HEAP_WINDOWS=1`) are not translated yet: the caller falls back to HLSL
    if o.mem_windows & crate::gpu::recomp::HEAP_WINDOWS != 0 {
        return Err("unsupported: heap memory windows (XCL_HEAP_WINDOWS)".into());
    }
    // SSA construction recurses along the CFG (one level per guarded memory access between
    // a register's definition and its use): a thread with a large stack for big programs
    let code = code.to_vec();
    std::thread::Builder::new()
        .name("xcl-dxil-translate".into())
        .stack_size(256 << 20)
        .spawn(move || translate_on_this_thread(&code, o, entry))
        .map_err(|e| format!("translator thread: {e}"))?
        .join()
        .map_err(|e| {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            format!("the direct DXIL translator panicked: {msg}")
        })?
}

fn translate_on_this_thread(code: &[u8], o: KernelOptions, entry: Entry) -> Result<Kernel, String> {
    let cfg = Cfg::build(code)?;
    let split = o.wave64 && o.split;
    let kernel = matches!(entry, Entry::Kernel);
    let cs = match &entry {
        Entry::Compute(cs) => Some(cs.clone()),
        _ => None,
    };
    let ms = match &entry {
        Entry::Mesh(ms) => Some(ms.clone()),
        _ => None,
    };
    let gfx = matches!(entry, Entry::Vertex(_) | Entry::Pixel(_));
    let split = split && !gfx;
    let stage = match &entry {
        Entry::Vertex(_) => Stage::Vertex,
        Entry::Pixel(_) => Stage::Pixel,
        _ => Stage::Compute,
    };
    let mut s = Shader::new(stage, (6, 6));
    let io = s.raw_uav(0, 0);
    let res = if kernel || ms.is_some() { Some(s.raw_uav(1, 0)) } else { None };
    // compute work across groups relies on coherent memory (as the HLSL compute mode's
    // `globallycoherent` windows)
    let mems: Vec<_> = (0..o.mem_windows.clamp(1, 4)).map(|k| if cs.is_some() { s.raw_uav_coherent(2 + k, 0) } else { s.raw_uav(2 + k, 0) }).collect();
    let i32 = s.m.i32();
    let z = s.m.i32c(0);
    let mut lw = Lw {
        s,
        ssa: Ssa::new(i32),
        init: Init { io: Some(io), compute: None, vgpr_base: None, entry: Block(0), touched: BTreeSet::new() },
        wave64: o.wave64,
        split,
        lane: z,
        lane31: z,
        upper: z,
        half: z,
        mems,
        res,
        fmt_table: None,
        ntmp: 0,
        scratch: None,
        static_samplers: Default::default(),
        static_used: Vec::new(),
        sampler_base: o.sampler_base,
        dyn_rm: false,
        has_f64_round_op: false,
        exec_cache: HashMap::new(),
        xs: None,
        xw: None,
        xc: None,
        waves: 1,
        cw: z,
        xn: z,
        xslot: 1,
        lds: None,
        lds_top: 8191,
        dpp_enable: None,
        partner: Partner::Present,
        mesh: None,
        gfx: None,
        gfx_half: None,
        whole_live: Default::default(),
        guard_override: None,
        mask_override: None,
        exchange_error: None,
        pc: 0,
    };
    {
        let Lw { s, ssa, init, .. } = &mut lw;
        ssa.expect_preds(s, Block(0), 0, &mut |s, v| init.load(s, v));
    }
    // LDS and the split form's exchange slot size (16 words for a coherent s_load_dwordx16,
    // 32 for ds_bpermute), as the HLSL kernel sizes them
    let mut uses_lds = false;
    let mut slot = 1u32;
    for d in &cfg.insts {
        let n = d.inst.name;
        if d.enc.name == "ENC_DS" && !matches!(n, "DS_SWIZZLE_B32" | "DS_BPERMUTE_B32" | "DS_PERMUTE_B32" | "DS_NOP") {
            uses_lds = true;
        }
        if n == "DS_BPERMUTE_B32" {
            slot = slot.max(32);
        }
        if d.enc.name == "ENC_SMEM" && ["GLC", "DLC"].iter().any(|f| crate::gpu::rdna2::field_value(d.enc, &d.words, f) == Some(1)) {
            let w = match n.rsplit('_').next().unwrap_or("") {
                "DWORDX2" => 2,
                "DWORDX4" => 4,
                "DWORDX8" => 8,
                "DWORDX16" => 16,
                _ => 1,
            };
            slot = slot.max(w);
        }
    }
    let leaders: BTreeSet<u64> = cfg.blocks.iter().filter_map(|b| b.insts.first().map(|&i| cfg.insts[i].pc)).collect();
    lw.static_samplers = crate::gpu::recomp::constant_samplers(&cfg.insts, &leaders);
    let group_threads: Option<u32> = match &entry {
        Entry::Vertex(_) | Entry::Pixel(_) => None,
        Entry::Kernel => None,
        Entry::Compute(cs) => Some(cs.threads.iter().product()),
        Entry::Mesh(ms) => Some(ms.threads),
    };
    if let (Some(t), true) = (group_threads, split) {
        lw.waves = t.div_ceil(64).max(1);
    }
    if split && std::env::var_os("XCL_X64_ALL_MASKS").is_none() {
        lw.whole_live = crate::gpu::recomp::x64_masks::by_pc(&cfg.insts, kernel);
    }
    lw.setup_storage(uses_lds, if split { slot } else { 1 });
    if let (Some(t), true) = (group_threads, split) {
        lw.partner = match t {
            0..=32 => Partner::Missing,
            33..=64 => Partner::Present,
            _ => Partner::Spin,
        };
    }
    let numthreads = match &entry {
        Entry::Kernel => {
            lw.setup_lanes();
            let lane = lw.lane;
            let k = lw.c((KERNEL_VGPRS * 4) as u32);
            let b = lw.mul(lane, k);
            let base = lw.addc(b, (KERNEL_SGPRS * 4) as u32);
            lw.init.vgpr_base = Some(base);
            [if o.wave64 { 64 } else { 32 }, 1, 1]
        }
        Entry::Compute(cs) => {
            let (tid, gtid) = lw.setup_lanes_compute(cs.threads, uses_lds);
            compute_init(&mut lw, cs, tid, gtid);
            cs.threads
        }
        Entry::Mesh(ms) => {
            let (tid, _) = lw.setup_lanes_compute([ms.threads, 1, 1], uses_lds);
            mesh_init(&mut lw, ms, tid, io);
            [ms.threads, 1, 1]
        }
        Entry::Vertex(vs) => {
            lw.setup_lanes_gfx();
            let used: BTreeSet<u64> = cfg.insts.iter().flat_map(|d| {
                let (r, w) = crate::gpu::rdna2::vgpr_uses(d);
                r.into_iter().chain(w)
            }).collect();
            lw.setup_vertex(vs, &used);
            [1, 1, 1]
        }
        Entry::Pixel(ps) => {
            lw.setup_lanes_gfx();
            let modes = crate::gpu::recomp::emit::ps_attr_modes(&cfg.insts, ps.input_ena);
            lw.setup_pixel(ps, &modes);
            [1, 1, 1]
        }
    };
    // the HLSL kernel loads and stores every register its emitted instructions name: all of
    // them in a program with branches, those up to the first s_endpgm in straight-line code
    let branchy = cfg.insts.iter().any(|d| crate::gpu::rdna2::branch_target(d).is_some());
    (lw.dyn_rm, lw.has_f64_round_op) = super::lower::round_modes(&cfg.insts, !branchy);
    for d in cfg.insts.iter().filter(|_| kernel) {
        if !branchy && d.inst.name.starts_with("S_ENDPGM") {
            break;
        }
        let (sr, sw) = crate::gpu::rdna2::sgpr_uses(d);
        let (vr, vw) = crate::gpu::rdna2::vgpr_uses(d);
        for r in sr.into_iter().chain(sw) {
            lw.init.touched.insert(r as u32);
        }
        for r in vr.into_iter().chain(vw).filter(|&r| r < 256) {
            lw.init.touched.insert(VAR_VGPR + r as u32);
        }
    }

    let n = cfg.blocks.len();
    let mut heads: Vec<Option<Block>> = vec![None; n];
    let entry = *cfg.rpo.first().ok_or("empty program")?;
    let mut head = |lw: &mut Lw, b: usize| -> Block {
        if let Some(h) = heads[b] {
            return h;
        }
        let preds = cfg.blocks[b].preds.len() + (b == entry) as usize;
        let h = lw.block(Some(preds));
        heads[b] = Some(h);
        h
    };
    let h0 = head(&mut lw, entry);
    lw.edge(Block(0), h0);
    let exit = lw.block(None);
    for &b in &cfg.rpo {
        let h = head(&mut lw, b);
        lw.s.m.set_block(h);
        if cfg.headers.contains(&b) {
            // a lane's loop iterations are capped (the recompiler's LOOP_GUARD)
            let left = lw.rd(VAR_BUDGET);
            let none = lw.cmpc(Pred::Eq, left, 0);
            let cont = lw.block(Some(1));
            lw.cond_br(none, exit, cont);
            lw.s.m.set_block(cont);
            let one = lw.c(1);
            let l = lw.sub(left, one);
            lw.wr(VAR_BUDGET, l);
        }
        for &i in &cfg.blocks[b].insts {
            lw.inst(&cfg.insts[i])?;
        }
        match cfg.blocks[b].term {
            Term::Jump(t) => {
                let t = head(&mut lw, t);
                lw.br(t);
            }
            Term::Cond(i, t, f) => {
                let c = lw.branch_cond(&cfg.insts[i])?;
                let (t, f) = (head(&mut lw, t), head(&mut lw, f));
                lw.cond_br(c, t, f);
            }
            Term::End => lw.br(exit),
        }
    }
    lw.s.m.set_block(exit);
    lw.seal(exit);
    // the epilogue uses values of the program's blocks, which were created after `exit`: it goes
    // into a block of its own after them (only a phi may refer to a later block's value)
    let tail = lw.block(Some(1));
    lw.br(tail);
    lw.s.m.set_block(tail);
    if let (Some(res), true) = (res, kernel) {
        store_registers(&mut lw, res);
    }
    if gfx {
        lw.finish_gfx();
    }
    lw.s.m.ret();
    lw.s.m.set_block(Block(0));
    lw.s.m.br(h0);
    if let Some(e) = lw.exchange_error.take() {
        return Err(e);
    }
    let hw = if split || !o.wave64 { 32 } else { 64 };
    let c = if gfx {
        let g = lw.gfx.take().ok_or("graphics entry without graphics I/O")?;
        let info = if stage == Stage::Pixel { super::graphics::StageInfo::Pixel { early_depth_stencil: false } } else { super::graphics::StageInfo::Vertex };
        lw.s.try_finish_graphics(g.io, info)
    } else {
        lw.s.try_finish(numthreads, Some(hw))
    }
    .map_err(|e| format!("direct DXIL builder error: {e}"))?;
    if cfg!(any(test, debug_assertions)) {
        if let Err(e) = super::validate::validate(&c) {
            let dis = super::validate::disassemble(&c).unwrap_or_else(|d| d);
            return Err(format!("direct DXIL failed validation: {e}\n--- disassembly ---\n{dis}"));
        }
    }
    Ok(Kernel { dxil: c, static_samplers: lw.static_used.iter().map(|&(w, c, _)| (w, c)).collect() })
}

/// The kernel's epilogue: SGPRs and SCC from lane 0, every touched VGPR per lane.
fn store_registers(lw: &mut Lw, res: super::shader::RawUav) {
    let touched: Vec<u32> = lw.init.touched.iter().copied().collect();
    let mut sg: BTreeSet<u32> = touched.iter().copied().filter(|&v| v < VAR_VGPR && v != VAR_SCC).collect();
    sg.extend([106, 107, 124, 126, 127]);
    let vg: Vec<u32> = touched.iter().copied().filter(|&v| (VAR_VGPR..VAR_BUDGET).contains(&v)).collect();
    let svals: Vec<(u32, Value)> = sg.iter().map(|&r| (r, lw.rd(r))).collect();
    let scc = lw.rd(VAR_SCC);
    let vvals: Vec<(u32, Value)> = vg.iter().map(|&r| (r - VAR_VGPR, lw.rd(r))).collect();
    let lane = lw.lane;
    let first = lw.cmpc(Pred::Eq, lane, 0);
    let _ = lw.guarded(first, &mut |lw| {
        for &(r, v) in &svals {
            let off = lw.c(r * 4);
            lw.s.raw_store(res, off, &[v]);
        }
        let off = lw.c((KERNEL_SCC_SLOT * 4) as u32);
        lw.s.raw_store(res, off, &[scc]);
        Ok(())
    });
    let base = lw.init.vgpr_base.expect("vgpr base");
    for (r, v) in vvals {
        let k = lw.c(r * 4);
        let off = lw.add(base, k);
        lw.s.raw_store(res, off, &[v]);
    }
}

/// The compute entry's fixed registers (the HLSL compute mode's prologue): group ids
/// after the user SGPRs, TG_SIZE, SCC = 0, EXEC = the wave's threads, v0..v2 = thread id.
fn compute_init(lw: &mut Lw, cs: &CsInterface, tid: Value, gtid: [Value; 3]) {
    let mut sgpr = HashMap::new();
    let mut next = cs.user_sgprs;
    for k in 0..3 {
        if cs.tgid[k] {
            let g = lw.s.group_id(k as u32);
            sgpr.insert(next, g);
            next += 1;
        }
    }
    let lanes: u32 = if lw.wave64 { 64 } else { 32 };
    let total: u32 = cs.threads.iter().product();
    if cs.tg_size {
        // (threads / lanes) | (gi / lanes) << 6
        let w = lw.c(total / lanes);
        let l = lw.c(lanes);
        let i = lw.bin(super::module::BinOp::UDiv, tid, l);
        let i = lw.shlc(i, 6);
        let v = lw.or(w, i);
        sgpr.insert(next, v);
    }
    let zero = lw.c(0);
    sgpr.insert(VAR_SCC, zero);
    let (lo, hi) = entry_exec(lw, total, tid);
    sgpr.insert(126, lo);
    sgpr.insert(127, hi);
    let mut vgpr = HashMap::new();
    for (k, &g) in gtid.iter().enumerate() {
        if k == 0 || cs.tid_components as usize > k {
            vgpr.insert(k as u32, g);
        }
    }
    lw.init.compute = Some(ComputeInit { sgpr, vgpr, zero });
}

/// The mesh entry's registers (the HLSL mesh mode's prologue): SGPRs from `io` with the
/// wave index in s3[27:24], SCC 0, EXEC = the wave's threads, v5 = group · threads + slot,
/// other VGPRs 0; and the record slots.
fn mesh_init(lw: &mut Lw, ms: &MsInterface, tid: Value, io: super::shader::RawUav) {
    let lanes: u32 = if lw.wave64 { 64 } else { 32 };
    let l = lw.c(lanes);
    let wave = lw.bin(super::module::BinOp::UDiv, tid, l);
    let lane = lw.lane;
    let base = lw.mul(wave, l);
    let slot = lw.add(base, lane);
    let gid = lw.s.group_id(0);
    let kr = lw.c(ms.record_dwords() as u32);
    let rec = lw.mul(gid, kr);
    let kv = lw.c(ms.vert_dwords() as u32);
    let sv = lw.mul(slot, kv);
    let vtx = lw.addc(rec, MESH_HEADER as u32);
    let vtx = lw.add(vtx, sv);
    let prm = lw.addc(rec, (MESH_HEADER + ms.max_vertices as usize * ms.vert_dwords()) as u32);
    let prm = lw.add(prm, slot);
    let zero = lw.c(0);
    let mut sgpr = HashMap::new();
    let o3 = lw.c(12);
    let s3 = lw.s.raw_load(io, o3, 1)[0];
    let s3 = lw.andc(s3, 0xF0FF_FFFF);
    let w24 = lw.shlc(wave, 24);
    let s3 = lw.or(s3, w24);
    sgpr.insert(3, s3);
    sgpr.insert(VAR_SCC, zero);
    let (lo, hi) = entry_exec(lw, ms.threads, tid);
    sgpr.insert(126, lo);
    sgpr.insert(127, hi);
    let kt = lw.c(ms.threads);
    let g = lw.mul(gid, kt);
    let v5 = lw.add(g, slot);
    let mut vgpr = HashMap::new();
    vgpr.insert(5, v5);
    lw.init.compute = Some(ComputeInit { sgpr, vgpr, zero });
    lw.mesh = Some(super::lower::MeshCtx { rec, vtx, prm, outputs: ms.outputs.clone() });
}

/// EXEC at entry: the console wave's threads (static per wave in the split form: no
/// exchange); a constant when every wave of the group is full, so vector writes need no
/// lane guard.
fn entry_exec(lw: &mut Lw, total: u32, tid: Value) -> (Value, Value) {
    let zero = lw.c(0);
    let wave_threads: u32 = if lw.wave64 { 64 } else { 32 };
    let (lo, hi) = if total % wave_threads == 0 {
        let all = lw.c(u32::MAX);
        (all, if lw.wave64 { all } else { zero })
    } else if lw.split {
        let base = lw.andc(tid, !63);
        let t = lw.c(total);
        let n = lw.sub(t, base);
        let k64 = lw.c(64);
        let n = lw.umin(n, k64);
        let all = lw.c(u32::MAX);
        let one = lw.c(1);
        let n31 = lw.andc(n, 31);
        let part = lw.bin(super::module::BinOp::Shl, one, n31);
        let part = lw.sub(part, one);
        let ge32 = lw.cmpc(Pred::UGe, n, 32);
        let lo = lw.sel(ge32, all, part);
        let k32 = lw.c(32);
        let nh = lw.sub(n, k32);
        let nh31 = lw.andc(nh, 31);
        let parth = lw.bin(super::module::BinOp::Shl, one, nh31);
        let parth = lw.sub(parth, one);
        let ge64 = lw.cmpc(Pred::UGe, n, 64);
        let gt32 = lw.cmpc(Pred::UGt, n, 32);
        let hi = lw.sel(gt32, parth, zero);
        let hi = lw.sel(ge64, all, hi);
        (lo, hi)
    } else {
        let t = lw.s.m.i1c(true);
        let w = lw.s.wave_ballot(t);
        // HLSL ballot(): (m.x, m.y) — the second word is 0 in a 32-lane wave
        (w[0], if lw.wave64 { w[1] } else { zero })
    };
    (lo, hi)
}
