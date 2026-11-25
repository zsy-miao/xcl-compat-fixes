//! Geometry mode (ES+GS) for the direct translator: the recompiler's
//! `Mode::Geometry` (`recomp/emit.rs`), a compute pass that runs the ES for the
//! three vertices of each of 21 triangles per group (pre-fetched by XCL), then
//! the GS, and writes the emitted vertices and strip triangles as one mesh
//! record per group into `res` (drawn by the pass-through mesh shader).
//!
//! Per lane: `emitted` / `strip` / `ntri` counters, the lowest LDS address the
//! GS phase wrote (where its output vertices are), and the strip triangles
//! (`[maxt x i32]` in a local array). The split wave64 form runs the ES in both
//! hardware waves, then the GS in the lower one alone with fresh registers.

use super::lower::{Lw, Partner};
use super::module::{Pred, Value};
use super::ops::WaveOp;
use crate::gpu::rdna2::{self, Decoded};
use crate::gpu::recomp::{ranked, GsInterface, MESH_HEADER};

pub struct GeomCtx {
    pub g: GsInterface,
    pub emitted: u32,
    pub strip: u32,
    pub ntri: u32,
    pub base: u32,
    /// local `[maxt x i32]` of strip triangles (k0 | k1 << 10 | k2 << 20)
    pub tri: Value,
    pub maxt: u32,
    pub rec: Value,
}

impl GeomCtx {
    /// Whether `pc` is in the GS code (after the ES's `s_setpc_b64` target).
    pub fn gs_phase(&self, pc: u64) -> bool {
        self.g.gs_start.is_some_and(|s| pc >= s)
    }
}

impl Lw {
    /// Geometry-mode instructions: `s_sendmsg` GS emit / cut, `s_barrier` (none in the
    /// split form's GS phase, which runs in one hardware wave). `None`: the usual lowering.
    pub fn geom_inst(&mut self, d: &Decoded, name: &str) -> Option<()> {
        let gs_phase = self.geom.as_ref()?.gs_phase(d.pc);
        match name {
            "s_barrier" if self.split && gs_phase => Some(()),
            "s_sendmsg" => {
                // MSG_GS (2): op 1 cut, 2 emit, 3 emit + cut (stream 0); the rest: nothing
                let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0);
                let (msg, op, stream) = (simm & 0xF, (simm >> 4) & 3, (simm >> 8) & 3);
                if msg == 2 && stream == 0 {
                    self.gs_emit_cut(op);
                }
                Some(())
            }
            _ => None,
        }
    }

    fn gs_emit_cut(&mut self, op: u64) {
        let (ve, vs, vn, tri, maxt) = {
            let g = self.geom.as_ref().expect("geometry");
            (g.emitted, g.strip, g.ntri, g.tri, g.maxt)
        };
        let act = self.act();
        if op & 2 != 0 {
            // strip triangle j = (j, j+1, j+2), odd j with the first two swapped
            let k = self.rd(ve);
            let strip = self.rd(vs);
            let ntri = self.rd(vn);
            let two = self.cmpc(Pred::UGe, strip, 2);
            let room = self.cmpc(Pred::ULt, ntri, maxt);
            let c = self.land(act, two);
            let c = self.land(c, room);
            let k1 = self.c(1);
            let k2 = self.c(2);
            let km1 = self.sub(k, k1);
            let km2 = self.sub(k, k2);
            let k20 = self.shlc(k, 20);
            let a = self.shlc(km1, 10);
            let even = self.or(km2, a);
            let even = self.or(even, k20);
            let b = self.shlc(km2, 10);
            let odd = self.or(km1, b);
            let odd = self.or(odd, k20);
            let par = self.andc(strip, 1);
            let ev = self.cmpc(Pred::Eq, par, 0);
            let val = self.sel(ev, even, odd);
            let last = self.c(maxt - 1);
            let idx = self.umin(ntri, last);
            let zero = self.c(0);
            let p = self.s.m.gep(tri, &[zero, idx]);
            let old = self.s.m.load(p, 4);
            let nv = self.sel(c, val, old);
            self.s.m.store(p, nv, 4);
            let n1 = self.addc(ntri, 1);
            let n = self.sel(c, n1, ntri);
            self.wr(vn, n);
            let e1 = self.addc(k, 1);
            let e = self.sel(act, e1, k);
            self.wr(ve, e);
            let s1 = self.addc(strip, 1);
            let st = self.sel(act, s1, strip);
            self.wr(vs, st);
        }
        if op & 1 != 0 {
            let strip = self.rd(vs);
            let z = self.c(0);
            let st = self.sel(act, z, strip);
            self.wr(vs, st);
        }
    }

    /// A GS-phase LDS store at byte address `at` (active lanes only): the lowest one is
    /// where the lane's output vertices start.
    pub fn geom_note_store(&mut self, at: Value) {
        let Some(g) = self.geom.as_ref() else { return };
        if !g.gs_phase(self.pc) {
            return;
        }
        let v = g.base;
        let cur = self.rd(v);
        let m = self.umin(cur, at);
        self.wr(v, m);
    }

    /// The split form's GS phase starts like a new program in the lower hardware wave:
    /// no partner half, registers as at entry, EXEC of the wave's lanes.
    pub fn geom_gs_phase_registers(&mut self, vars: &[u32]) {
        self.partner = Partner::Missing;
        self.xn = self.c(0);
        self.exec_cache.clear();
        for &v in vars {
            let x = {
                let Lw { s, init, .. } = self;
                init.load(s, v)
            };
            self.wr(v, x);
        }
        let t = self.b(true);
        let (lo, hi) = self.ballot(t);
        self.wr(126, lo);
        self.wr(127, hi);
        let z = self.c(0);
        self.wr(super::lower::VAR_SCC, z);
        let g = self.geom.as_ref().expect("geometry");
        let (ve, vs, vn, vb) = (g.emitted, g.strip, g.ntri, g.base);
        for v in [ve, vs, vn] {
            self.wr(v, z);
        }
        let m = self.c(u32::MAX);
        self.wr(vb, m);
    }

    /// The record: vertex / triangle counts (lane 0), the emitted vertices from LDS and
    /// the strip triangles, each lane after the lanes before it (wave prefix sums).
    pub fn geom_epilogue(&mut self) {
        let (g, ve, vn, vbase, tri, rec) = {
            let x = self.geom.as_ref().expect("geometry");
            (x.g.clone(), x.emitted, x.ntri, x.base, x.tri, x.rec)
        };
        let res = self.res.expect("res");
        let r = g.record();
        let (vd, vbytes) = (r.vert_dwords() as u32, g.vertex_bytes());
        let (maxv, maxp) = (r.max_vertices, r.max_primitives);
        let emitted = self.rd(ve);
        let ntri = self.rd(vn);
        let gs_base = self.rd(vbase);
        let vb = self.s.wave_prefix_op(emitted, WaveOp::Sum, true);
        let tb = self.s.wave_prefix_op(ntri, WaveOp::Sum, true);
        let tv = self.s.wave_active_op(emitted, WaveOp::Sum, true);
        let tt = self.s.wave_active_op(ntri, WaveOp::Sum, true);
        let lane = self.lane;
        let first = self.cmpc(Pred::Eq, lane, 0);
        let _ = self.guarded(first, &mut |lw| {
            let a = lw.c(maxv);
            let nv = lw.umin(tv, a);
            let b = lw.c(maxp);
            let np = lw.umin(tt, b);
            let k4 = lw.c(4);
            let o0 = lw.mul(rec, k4);
            let o1 = lw.addc(o0, 4);
            lw.s.raw_store(res, o0, &[nv]);
            lw.s.raw_store(res, o1, &[np]);
            Ok(())
        });
        // vertices: k < emitted (and a GS-phase LDS store happened), vb + k < maxv
        let pos_reg = g.outputs.iter().find(|e| e.system_value == 1).map(|e| e.register).unwrap_or(0);
        let regs: Vec<u32> = std::iter::once(pos_reg).chain(ranked(&g.outputs, true).iter().map(|e| e.register)).collect();
        let has = self.cmpc(Pred::Ne, gs_base, u32::MAX);
        let kmax = self.c(maxv);
        let left = self.sub(kmax, vb);
        let room = self.cmp(Pred::UGt, kmax, vb);
        let z = self.c(0);
        let left = self.sel(room, left, z);
        let n = self.umin(emitted, left);
        let n = self.sel(has, n, z);
        let lds = self.lds.expect("lds");
        self.counted_loop(n, &mut |lw, k| {
            let vi = lw.add(vb, k);
            let kb = lw.c(vbytes);
            let so = lw.mul(k, kb);
            let src = lw.add(gs_base, so);
            let kd = lw.c(vd);
            let dv = lw.mul(vi, kd);
            let dst = lw.addc(rec, MESH_HEADER as u32);
            let dst = lw.add(dst, dv);
            for (slot, &r) in regs.iter().enumerate() {
                for c in 0..4u32 {
                    let a = lw.addc(src, 16 * r + 4 * c);
                    let i = lw.lshrc(a, 2);
                    let i = lw.andc(i, 8191);
                    let v = lw.s.gs_load(lds, i);
                    let o = lw.addc(dst, 4 * slot as u32 + c);
                    let o = lw.shlc(o, 2);
                    lw.s.raw_store(res, o, &[v]);
                }
            }
        });
        // triangles: j < ntri, tb + j < maxp
        let pmax = self.c(maxp);
        let left = self.sub(pmax, tb);
        let room = self.cmp(Pred::UGt, pmax, tb);
        let left = self.sel(room, left, z);
        let n = self.umin(ntri, left);
        let prim_base = (MESH_HEADER + maxv as usize * vd as usize) as u32;
        self.counted_loop(n, &mut |lw, j| {
            let zero = lw.c(0);
            let p = lw.s.m.gep(tri, &[zero, j]);
            let t = lw.s.m.load(p, 4);
            let a = lw.andc(t, 0x3FF);
            let a = lw.add(a, vb);
            let b = lw.lshrc(t, 10);
            let b = lw.andc(b, 0x3FF);
            let b = lw.add(b, vb);
            let b = lw.shlc(b, 10);
            let c = lw.lshrc(t, 20);
            let c = lw.andc(c, 0x3FF);
            let c = lw.add(c, vb);
            let c = lw.shlc(c, 20);
            let v = lw.or(a, b);
            let v = lw.or(v, c);
            let o = lw.addc(rec, prim_base);
            let o = lw.add(o, tb);
            let o = lw.add(o, j);
            let o = lw.shlc(o, 2);
            lw.s.raw_store(res, o, &[v]);
        });
    }

    /// `for (k = 0; k < n; k++) body(k)` with a real loop (the counter in an SSA variable).
    fn counted_loop(&mut self, n: Value, body: &mut dyn FnMut(&mut Lw, Value)) {
        let kv = self.tmp_var();
        let z = self.c(0);
        self.wr(kv, z);
        let head = self.block(None);
        let lp = self.block(Some(1));
        let out = self.block(Some(1));
        self.br(head);
        self.s.m.set_block(head);
        let k = self.rd(kv);
        let more = self.cmp(Pred::ULt, k, n);
        self.cond_br(more, lp, out);
        self.s.m.set_block(lp);
        body(self, k);
        let k1 = self.addc(k, 1);
        self.wr(kv, k1);
        self.br(head);
        self.seal(head);
        self.s.m.set_block(out);
    }
}
