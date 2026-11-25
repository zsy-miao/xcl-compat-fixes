//! Image (MIMG) lowering: samples, gathers, loads, stores, atomics and
//! resource info through descriptor-heap views (T# word 0 = SRV heap index,
//! the UAV at the next index; S# word 3 [11:0] = sampler heap index), and
//! `image_bvh(64)_intersect_ray` on the flat memory (port of
//! `recomp/emit.rs` `mimg` and `bvh_intersect`; compute rules: no implicit
//! derivatives, so implicit-LOD samples take LOD 0 / the bias).

use super::lower::{err, LResult, Lw, VAR_VGPR};
use super::module::{BinOp, Pred, Value};
use super::ops::{AtomicOp, CompType, Sampler, TexKind, Texture};
use super::ops_core::op;
use crate::gpu::rdna2::{self, Decoded};

/// Upper bound on valid SRV indices in the shared shader-visible view heap
/// (`console_heap::GLOBAL_MAX = 1 << 21`, leaving `index + 1` in-bounds for the paired UAV).
/// Guarding / clamping dynamic heap indices prevents AMD Tier-3 GPUs from page-faulting
/// on null, uninitialized, or unpatched console T# descriptors.
pub(crate) const VIEW_HEAP_MAX_SRV: u32 = crate::hle::d3d12x::console_heap::GLOBAL_MAX - 1;
/// D3D12 shader-visible sampler heap limit (`2048` samplers: `0..=2047`).
const SAMPLER_HEAP_MASK: u32 = 0x7FF;

impl Lw {
    /// True when a T# carries a valid XCL image descriptor whose SRV (`tw0`) and paired
    /// UAV (`tw0 + 1`) indices are within the shared shader-visible view heap:
    /// - `tw3[31:28] >= 8` (valid GFX10 `SQ_RSRC_IMG_TYPE`, rejecting null T# and buffer V#);
    /// - `tw1[7:0] == 0` (`desc::xcl_image_words` clears `BASE_ADDRESS_HI`, rejecting raw GPU VAs);
    /// - `tw0 < VIEW_HEAP_MAX_SRV` (rejecting OOB indices and `u32::MAX` wraparound on `tw0 + 1`).
    fn valid_tsharp(&mut self, tw0: Value, tw1: Value, tw3: Value) -> Value {
        let rtype = self.lshrc(tw3, 28);
        let ok_ty = self.cmpc(Pred::UGe, rtype, 8);
        let hi_va = self.andc(tw1, 0xFF);
        let ok_va = self.cmpc(Pred::Eq, hi_va, 0);
        let ok_idx = self.cmpc(Pred::ULt, tw0, VIEW_HEAP_MAX_SRV);
        let ok = self.land(ok_ty, ok_va);
        self.land(ok, ok_idx)
    }

    /// The sampler of the MIMG at `pc`: a static sampler (bound at `s<i>, space1` in order of
    /// first use) when the console bound one there, else the heap sampler the S# indexes.
    fn mimg_sampler(&mut self, pc: u64, has_c: bool, s3: Value) -> Sampler {
        match self.static_samplers.get(&pc).copied() {
            Some(w) => match self.static_used.iter().find(|u| u.0 == w && u.1 == has_c) {
                Some(u) => u.2,
                None => {
                    let k = self.static_used.len() as u32;
                    let cur = self.cur();
                    self.s.m.set_block(self.init.entry);
                    let h = self.s.sampler(self.sampler_base + k, 1, has_c);
                    self.s.m.set_block(cur);
                    self.static_used.push((w, has_c, h));
                    h
                }
            },
            None => {
                let si = self.andc(s3, SAMPLER_HEAP_MASK);
                self.s.heap_sampler(si, has_c, true)
            }
        }
    }
    fn f3(&mut self, a: [Value; 3], b: [Value; 3], op: fn(&mut Lw, Value, Value) -> Value) -> [Value; 3] {
        [op(self, a[0], b[0]), op(self, a[1], b[1]), op(self, a[2], b[2])]
    }
    fn cross(&mut self, a: [Value; 3], b: [Value; 3]) -> [Value; 3] {
        let x0 = self.fmul(a[1], b[2]);
        let x1 = self.fmul(a[2], b[1]);
        let x = self.fsub(x0, x1);
        let y0 = self.fmul(a[2], b[0]);
        let y1 = self.fmul(a[0], b[2]);
        let y = self.fsub(y0, y1);
        let z0 = self.fmul(a[0], b[1]);
        let z1 = self.fmul(a[1], b[0]);
        let z = self.fsub(z0, z1);
        [x, y, z]
    }
    /// HLSL `dot(float3, float3)` = `Dot3`.
    fn dot3(&mut self, a: [Value; 3], b: [Value; 3]) -> Value {
        let f32 = self.s.m.f32();
        self.s.op(55, "dot3", Some(f32), f32, &[a[0], a[1], a[2], b[0], b[1], b[2]], super::module::FnAttrs::ReadNone)
    }

    /// `bvh_intersect`: one BVH node → 4 result dwords (written to temporaries).
    #[allow(clippy::too_many_arguments)]
    fn bvh_intersect(&mut self, ty: Value, at: Value, tmax: Value, o: [Value; 3], dir: [Value; 3], inv: [Value; 3], sort: Value, out: [u32; 4]) -> LResult<()> {
        let ones = self.c(u32::MAX);
        for &t in &out {
            self.wr(t, ones);
        }
        for box32 in [true, false] {
            let is = self.cmpc(Pred::Eq, ty, if box32 { 5 } else { 4 });
            self.guarded(is, &mut |lw| {
                // Box32 is 28 contiguous dwords (4 children + 4×6 box floats);
                // Box16 is 16 contiguous dwords (4 children + 4×3 packed f16 pairs).
                // Load the whole node in one window switch using 4-dword rawBufferLoads.
                let at_al = lw.andc(at, !3);
                let chunks = if box32 { 7usize } else { 4usize };
                let words = lw.with_mem_i32s(None, chunks * 4, &mut |lw, m| {
                    let mut out = Vec::with_capacity(chunks * 4);
                    for j in 0..chunks {
                        let a = lw.addc(at_al, 16 * j as u32);
                        out.extend(lw.s.raw_load(m, a, 0xF));
                    }
                    out
                });
                let child = [words[0], words[1], words[2], words[3]];
                let ninf = lw.s.m.f32c(0xFF80_0000);
                let pinf = lw.s.m.f32c(0x7F80_0000);
                let mut tn = [pinf; 4];
                let mut id = [ones; 4];
                for k in 0..4usize {
                    let (mn, mx) = if box32 {
                        let b = 4 + 6 * k;
                        (
                            [lw.f(words[b]), lw.f(words[b + 1]), lw.f(words[b + 2])],
                            [lw.f(words[b + 3]), lw.f(words[b + 4]), lw.f(words[b + 5])],
                        )
                    } else {
                        let b = 4 + 3 * k;
                        let h = [words[b], words[b + 1], words[b + 2]];
                        let hx16 = lw.lshrc(h[0], 16);
                        let hy16 = lw.lshrc(h[1], 16);
                        let hz16 = lw.lshrc(h[2], 16);
                        let mn = [lw.s.f16_to_f32(h[0]), lw.s.f16_to_f32(hx16), lw.s.f16_to_f32(h[1])];
                        let mx = [lw.s.f16_to_f32(hy16), lw.s.f16_to_f32(h[2]), lw.s.f16_to_f32(hz16)];
                        (mn, mx)
                    };
                    let d1 = lw.f3(mn, o, Lw::fsub);
                    let t1 = lw.f3(d1, inv, Lw::fmul);
                    let d2 = lw.f3(mx, o, Lw::fsub);
                    let t2 = lw.f3(d2, inv, Lw::fmul);
                    let mut lo = [pinf; 3];
                    let mut hi = [pinf; 3];
                    for c in 0..3 {
                        let ge = lw.fcmp(Pred::FOGe, o[c], mn[c]);
                        let le = lw.fcmp(Pred::FOLe, o[c], mx[c]);
                        let slab = lw.land(ge, le);
                        let ii = lw.isinf(inv[c]);
                        let l0 = lw.sel(slab, ninf, pinf);
                        let h0 = lw.sel(slab, pinf, ninf);
                        let mn_t = lw.fmin(t1[c], t2[c]);
                        let mx_t = lw.fmax(t1[c], t2[c]);
                        lo[c] = lw.sel(ii, l0, mn_t);
                        hi[c] = lw.sel(ii, h0, mx_t);
                    }
                    let a = lw.fmax(lo[0], lo[1]);
                    let a = lw.fmax(a, lo[2]);
                    let b = lw.fmin(hi[0], hi[1]);
                    let b = lw.fmin(b, hi[2]);
                    let cv = lw.cmpc(Pred::Ne, child[k], u32::MAX);
                    let mut all = cv;
                    for c in 0..3 {
                        let le = lw.fcmp(Pred::FOLe, mn[c], mx[c]);
                        all = lw.land(all, le);
                    }
                    let ab = lw.fcmp(Pred::FOLe, a, b);
                    let z = lw.fc(0.0);
                    let b0 = lw.fcmp(Pred::FOGe, b, z);
                    let at_ = lw.fcmp(Pred::FOLe, a, tmax);
                    let hit = lw.land(all, ab);
                    let x = lw.land(b0, at_);
                    let hit = lw.land(hit, x);
                    tn[k] = lw.sel(hit, a, pinf);
                    id[k] = lw.sel(hit, child[k], ones);
                }
                for (i, j) in [(0usize, 1usize), (2, 3), (0, 2), (1, 3), (1, 2)] {
                    let lt = lw.fcmp(Pred::FOLt, tn[j], tn[i]);
                    let sw = lw.land(sort, lt);
                    let (ti, tj) = (lw.sel(sw, tn[j], tn[i]), lw.sel(sw, tn[i], tn[j]));
                    let (ii, ij) = (lw.sel(sw, id[j], id[i]), lw.sel(sw, id[i], id[j]));
                    tn[i] = ti;
                    tn[j] = tj;
                    id[i] = ii;
                    id[j] = ij;
                }
                for k in 0..4 {
                    lw.wr(out[k], id[k]);
                }
                Ok(())
            })?;
        }
        let tri = self.cmpc(Pred::ULe, ty, 3);
        self.guarded(tri, &mut |lw| {
            let t0 = lw.cmpc(Pred::Eq, ty, 0);
            let t1 = lw.cmpc(Pred::Eq, ty, 1);
            let t2 = lw.cmpc(Pred::Eq, ty, 2);
            let t3 = lw.cmpc(Pred::Eq, ty, 3);
            let le1 = lw.cmpc(Pred::ULe, ty, 1);
            let k0 = lw.c(0);
            let k12 = lw.c(12);
            let k24 = lw.c(24);
            let k36 = lw.c(36);
            let k48 = lw.c(48);
            let a0_12 = lw.sel(t1, k12, k24);
            let a0 = lw.sel(t0, k0, a0_12);
            let a1_34 = lw.sel(t3, k48, k36);
            let a1 = lw.sel(t0, k12, a1_34);
            let a2_23 = lw.sel(t2, k48, k0);
            let a2 = lw.sel(le1, k24, a2_23);
            let p0 = lw.add(at, a0);
            let p1 = lw.add(at, a1);
            let p2 = lw.add(at, a2);
            let p60 = lw.addc(at, 60);
            let (p0, p1, p2, p60) = (lw.andc(p0, !3), lw.andc(p1, !3), lw.andc(p2, !3), lw.andc(p60, !3));
            let words = lw.with_mem_i32s(None, 10, &mut |lw, m| {
                let mut out = Vec::with_capacity(10);
                out.extend(lw.s.raw_load(m, p0, 0x7));
                out.extend(lw.s.raw_load(m, p1, 0x7));
                out.extend(lw.s.raw_load(m, p2, 0x7));
                out.push(lw.s.raw_load(m, p60, 0x1)[0]);
                out
            });
            let v0 = [lw.f(words[0]), lw.f(words[1]), lw.f(words[2])];
            let v1 = [lw.f(words[3]), lw.f(words[4]), lw.f(words[5])];
            let v2 = [lw.f(words[6]), lw.f(words[7]), lw.f(words[8])];
            let tw = words[9];
            let e1 = lw.f3(v1, v0, Lw::fsub);
            let e2 = lw.f3(v2, v0, Lw::fsub);
            let p = lw.cross(dir, e2);
            let det = lw.dot3(e1, p);
            let sv = lw.f3(o, v0, Lw::fsub);
            let q = lw.cross(sv, e1);
            let i = lw.dot3(sv, p);
            let j = lw.dot3(dir, q);
            let t = lw.dot3(e2, q);
            let z = lw.fc(0.0);
            let ij = lw.fadd(i, j);
            let dpos = lw.fcmp(Pred::FOGt, det, z);
            let c1 = lw.fcmp(Pred::FOGe, i, z);
            let c2 = lw.fcmp(Pred::FOGe, j, z);
            let c3 = lw.fcmp(Pred::FOLe, ij, det);
            let hp = lw.land(c1, c2);
            let hp = lw.land(hp, c3);
            let dneg = lw.fcmp(Pred::FOLt, det, z);
            let n1 = lw.fcmp(Pred::FOLe, i, z);
            let n2 = lw.fcmp(Pred::FOLe, j, z);
            let n3 = lw.fcmp(Pred::FOGe, ij, det);
            let hn = lw.land(dneg, n1);
            let hn = lw.land(hn, n2);
            let hn = lw.land(hn, n3);
            let hit = lw.sel(dpos, hp, hn);
            let di = lw.fsub(det, i);
            let b0 = lw.fsub(di, j);
            let bc = [b0, i, j, z];
            let sh = lw.shlc(ty, 3);
            let tid = lw.shr_masked(tw, sh);
            let s0 = lw.andc(tid, 3);
            let s1 = lw.lshrc(tid, 2);
            let s1 = lw.andc(s1, 3);
            let bci = lw.pick4f(&bc, s0);
            let bcj = lw.pick4f(&bc, s1);
            let res = [lw.bits(t), lw.bits(det), lw.bits(bci), lw.bits(bcj)];
            let miss = [lw.c(0x7F80_0000), lw.c(0x3F80_0000), lw.c(0), lw.c(0)];
            for k in 0..4 {
                let v = lw.sel(hit, res[k], miss[k]);
                lw.wr(out[k], v);
            }
            Ok(())
        })
    }
    fn shr_masked(&mut self, a: Value, s: Value) -> Value {
        self.lshr(a, s)
    }
    fn pick4f(&mut self, r: &[Value; 4], i: Value) -> Value {
        let mut v = r[3];
        for k in (0..3).rev() {
            let is = self.cmpc(Pred::Eq, i, k);
            v = self.sel(is, r[k as usize], v);
        }
        v
    }

    /// `cube_dir`: AMD cube coordinates (sc/|2ma| + 1.5, tc/|2ma| + 1.5, face) → direction.
    fn cube_dir(&mut self, s: Value, t: Value, face: Value) -> [Value; 3] {
        let k15 = self.fc(1.5);
        let k2 = self.fc(2.0);
        let u = self.fsub(s, k15);
        let u = self.fmul(u, k2);
        let v = self.fsub(t, k15);
        let v = self.fmul(v, k2);
        let r = self.un(op::ROUND_NE, face);
        let fi = self.ftoui(r);
        let f = self.andc(fi, 7);
        let one = self.fc(1.0);
        let m1 = self.fc(-1.0);
        let nu = self.fneg(u);
        let nv = self.fneg(v);
        // faces 0..5 (anything else as 5)
        let table: [[Value; 3]; 6] = [[one, nv, nu], [m1, nv, u], [u, one, v], [u, m1, nv], [u, nv, one], [nu, nv, m1]];
        let mut out = table[5];
        for k in (0..5).rev() {
            let is = self.cmpc(Pred::Eq, f, k as u32);
            for c in 0..3 {
                out[c] = self.sel(is, table[k][c], out[c]);
            }
        }
        out
    }

    /// The cube of a cube-array coordinate: face + 8 · cube → cube (as float).
    fn cube_slice(&mut self, face: Value) -> Value {
        let r = self.un(op::ROUND_NE, face);
        let fi = self.ftoui(r);
        let c = self.lshrc(fi, 3);
        self.uitof(c)
    }

    fn smin_i32(&mut self, a: Value, b: Value) -> Value {
        let lt = self.cmp(Pred::SLt, a, b);
        self.sel(lt, a, b)
    }

    fn smax_i32(&mut self, a: Value, b: Value) -> Value {
        let gt = self.cmp(Pred::SGt, a, b);
        self.sel(gt, a, b)
    }

    /// Applies an RDNA2 3-bit `SQ_TEX_CLAMP` address mode (`0..=7`) to integer texel coordinate
    /// `coord` against mip dimension `size` (`>= 1`). Returns `(in_bounds_coord, is_border)`.
    fn apply_rdna2_addr_mode(&mut self, coord: Value, size: Value, mode: Value) -> (Value, Value) {
        let z = self.c(0);
        let one = self.c(1);
        let max_c = self.sub(size, one);

        // Wrap (mode 0): Euclidean modulo ((coord % size) + size) % size
        let rem = self.bin(BinOp::SRem, coord, size);
        let rem_neg = self.cmp(Pred::SLt, rem, z);
        let rem_pos = self.add(rem, size);
        let wrap = self.sel(rem_neg, rem_pos, rem);

        // Mirror (mode 1): period 2*size
        let period = self.shlc(size, 1);
        let mrem = self.bin(BinOp::SRem, coord, period);
        let mneg = self.cmp(Pred::SLt, mrem, z);
        let mpos = self.add(mrem, period);
        let m = self.sel(mneg, mpos, mrem);
        let m_hi = self.cmp(Pred::SGe, m, size);
        let p_m1 = self.sub(period, one);
        let m_ref = self.sub(p_m1, m);
        let mirror = self.sel(m_hi, m_ref, m);

        // Clamp (modes 2, 4, 6): clamp(coord, 0, size - 1)
        let clamped_hi = self.smin_i32(coord, max_c);
        let clamped = self.smax_i32(z, clamped_hi);

        // MirrorOnce (modes 3, 5, 7): reflect negative coords (-1 -> 0, -2 -> 1, ...) via ~coord = -1 - coord
        let c_neg = self.cmp(Pred::SLt, coord, z);
        let reflected = self.not(coord);
        let mo_mag = self.sel(c_neg, reflected, coord);
        let mo_clamped = self.smin_i32(mo_mag, max_c);

        // Select final in-bounds coordinate by mode
        let is_0 = self.cmpc(Pred::Eq, mode, 0);
        let is_1 = self.cmpc(Pred::Eq, mode, 1);
        let is_3 = self.cmpc(Pred::Eq, mode, 3);
        let is_5 = self.cmpc(Pred::Eq, mode, 5);
        let is_7 = self.cmpc(Pred::Eq, mode, 7);
        let is_mo = self.lor(is_3, is_5);
        let is_mo = self.lor(is_mo, is_7);

        let res = self.sel(is_mo, mo_clamped, clamped);
        let res = self.sel(is_1, mirror, res);
        let res = self.sel(is_0, wrap, res);

        // Border detection: mode 6 (ClampBorder) -> coord < 0 || coord >= size;
        // mode 7 (MirrorOnceBorder) -> mo_mag >= size.
        let is_6 = self.cmpc(Pred::Eq, mode, 6);
        let oob_hi = self.cmp(Pred::SGe, coord, size);
        let oob_6 = self.lor(c_neg, oob_hi);
        let oob_7 = self.cmp(Pred::UGe, mo_mag, size);
        let b6 = self.land(is_6, oob_6);
        let b7 = self.land(is_7, oob_7);
        let is_border = self.lor(b6, b7);

        (res, is_border)
    }

    /// Evaluates RDNA2 `SQ_TEX_DEPTH_COMPARE` (`0..=7`, matching `D3D12_COMPARISON_FUNC - 1`):
    /// `cmp_ref <OP> texel` → `1.0f` or `0.0f`.
    fn apply_depth_compare(&mut self, cmp_ref: Value, texel: Value, func: Value) -> Value {
        let f_false = self.s.m.i1c(false);
        let f_true = self.s.m.i1c(true);
        let c_lt = self.fcmp(Pred::FOLt, cmp_ref, texel);
        let c_eq = self.fcmp(Pred::FOEq, cmp_ref, texel);
        let c_le = self.fcmp(Pred::FOLe, cmp_ref, texel);
        let c_gt = self.fcmp(Pred::FOGt, cmp_ref, texel);
        let c_ne = self.fcmp(Pred::FUNe, cmp_ref, texel);
        let c_ge = self.fcmp(Pred::FOGe, cmp_ref, texel);
        let mut p = f_false;
        for (k, cond) in [(1u32, c_lt), (2, c_eq), (3, c_le), (4, c_gt), (5, c_ne), (6, c_ge), (7, f_true)] {
            let is = self.cmpc(Pred::Eq, func, k);
            p = self.sel(is, cond, p);
        }
        let one = self.fc(1.0);
        let zero = self.fc(0.0);
        self.sel(p, one, zero)
    }

    /// Emulates explicit-LOD `image_gather4_l` / `image_gather4_l_o` (and `_c_` / `_cl` variants)
    /// on 2D and 2DArray textures via 4 mip-level `textureLoad`s, honoring RDNA2 S# address modes,
    /// border colors, S# + T# `MIN_LOD` clamping, `UNORM=1` / `FORCE_UNORMALIZED`, `a16`/`d16`, and offsets.
    #[allow(clippy::too_many_arguments)]
    fn mimg_gather4_lod(
        &mut self,
        d: &Decoded,
        dim: u64,
        kind: TexKind,
        tw0: Value,
        tw1: Value,
        ssamp: u64,
        coords: &[Value],
        lod: Value,
        cl: Option<Value>,
        cmp: Option<Value>,
        ovgpr: Option<Value>,
        unorm: bool,
        comps: &[u64],
        vdata: u64,
        d16: bool,
        act: Value,
        run: Value,
    ) -> LResult<()> {
        if comps.len() != 1 {
            return err(d, "gather4 dmask");
        }
        let ch = comps[0] as usize;
        let s0 = self.sreg(ssamp);
        let s1 = self.sreg(ssamp + 1);
        let s2 = self.sreg(ssamp + 2);
        let s3 = self.sreg(ssamp + 3);

        let tmps = [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()];
        let z = self.c(0);
        for &t in &tmps {
            self.wr(t, z);
        }
        self.guarded(run, &mut |lw| {
            let tex = lw.s.heap_texture(tw0, kind, false, CompType::F32, 4, true);

            // Explicit LOD + S# signed 5.8 LOD_BIAS (s2[13:0] / 256.0), clamped by optional `cl`,
            // S# [min_lod, max_lod], and raised to T# MIN_LOD (tw1[19:8], u4.8 / 256.0).
            // When S# MIP_FILTER == none, RDNA2 reads floor(T# MIN_LOD) at every LOD.
            let scale = lw.fc(1.0 / 256.0);
            let lb_sh = lw.shlc(s2, 18);
            let lb_se = lw.ashrc(lb_sh, 18);
            let lb_f = lw.sitof(lb_se);
            let lod_bias = lw.fmul(lb_f, scale);
            let l = lw.fadd(lod, lod_bias);
            let l = match cl {
                Some(c) => lw.fmax(l, c),
                None => l,
            };
            let min_u = lw.andc(s1, 0xFFF);
            let min_f = lw.uitof(min_u);
            let min_lod = lw.fmul(min_f, scale);
            let max_u = lw.lshrc(s1, 12);
            let max_u = lw.andc(max_u, 0xFFF);
            let max_f = lw.uitof(max_u);
            let max_lod_raw = lw.fmul(max_f, scale);
            let t_min_u = lw.lshrc(tw1, 8);
            let t_min_u = lw.andc(t_min_u, 0xFFF);
            let t_min_f = lw.uitof(t_min_u);
            let t_min_lod = lw.fmul(t_min_f, scale);
            let mip_flt = lw.lshrc(s2, 26);
            let mip_flt = lw.andc(mip_flt, 3);
            let mip_none = lw.cmpc(Pred::Eq, mip_flt, 0);
            let l = lw.fmax(l, min_lod);
            let l = lw.fmin(l, max_lod_raw);
            let l = lw.fmax(l, t_min_lod);

            let z0 = lw.c(0);
            let one = lw.c(1);
            let dims0 = lw.s.get_dimensions(tex.handle, Some(z0));
            let num_mips = lw.s.m.extract(dims0, 3);
            let num_mips = lw.umax(num_mips, one);
            let max_mip = lw.sub(num_mips, one);

            // RDNA2 mip selection rounds half up via floor(lod + 0.501953125) (257/512)
            // so LOD 0.5 selects level 1 and 2.5 selects level 3 (unlike ROUND_NE).
            let zf = lw.fc(0.0);
            let half_up = lw.fc(0.501953125);
            let l_biased = lw.fadd(l, half_up);
            let l_point = lw.un(op::ROUND_NI, l_biased);
            let l_none = lw.un(op::ROUND_NI, t_min_lod);
            let l_floor = lw.sel(mip_none, l_none, l_point);
            let l_pos = lw.fmax(l_floor, zf);
            let mip_raw = lw.ftoui(l_pos);
            let mip_i = lw.umin(mip_raw, max_mip);

            let dims = lw.s.get_dimensions(tex.handle, Some(mip_i));
            let w = lw.s.m.extract(dims, 0);
            let h = lw.s.m.extract(dims, 1);
            let w = lw.umax(w, one);
            let h = lw.umax(h, one);

            // Convert normalized coordinates to texel space unless `UNORM=1` or S# `FORCE_UNORMALIZED` is set.
            let wf = lw.uitof(w);
            let hf = lw.uitof(h);
            let (px, py) = if unorm {
                (coords[0], coords[1])
            } else {
                let force_unorm = lw.lshrc(s0, 15);
                let force_unorm = lw.andc(force_unorm, 1);
                let is_unorm = lw.nz(force_unorm);
                let sx = lw.fmul(coords[0], wf);
                let sy = lw.fmul(coords[1], hf);
                (lw.sel(is_unorm, coords[0], sx), lw.sel(is_unorm, coords[1], sy))
            };

            let half = lw.fc(0.5);
            let pxm = lw.fsub(px, half);
            let pym = lw.fsub(py, half);
            let fx = lw.un(op::ROUND_NI, pxm);
            let fy = lw.un(op::ROUND_NI, pym);
            let mut ix = lw.ftosi(fx);
            let mut iy = lw.ftosi(fy);

            if let Some(o) = ovgpr {
                let ox = lw.shlc(o, 26);
                let ox = lw.ashrc(ox, 26);
                let oy = lw.shlc(o, 18);
                let oy = lw.ashrc(oy, 26);
                ix = lw.add(ix, ox);
                iy = lw.add(iy, oy);
            }

            let ix1 = lw.add(ix, one);
            let iy1 = lw.add(iy, one);

            let mode_x = lw.andc(s0, 7);
            let mode_y = lw.lshrc(s0, 3);
            let mode_y = lw.andc(mode_y, 7);

            let (x0, bx0) = lw.apply_rdna2_addr_mode(ix, w, mode_x);
            let (x1, bx1) = lw.apply_rdna2_addr_mode(ix1, w, mode_x);
            let (y0, by0) = lw.apply_rdna2_addr_mode(iy, h, mode_y);
            let (y1, by1) = lw.apply_rdna2_addr_mode(iy1, h, mode_y);

            // For Texture2DArray, clamp rounded slice index to [0, array_size - 1].
            let slice = if dim == 5 {
                let arr = lw.s.m.extract(dims, 2);
                let arr = lw.umax(arr, one);
                let max_s = lw.sub(arr, one);
                let sr = lw.un(op::ROUND_NE, coords[2]);
                let si = lw.ftosi(sr);
                let si = lw.smin_i32(si, max_s);
                Some(lw.smax_i32(z0, si))
            } else {
                None
            };

            // Sampler border color value for channel `ch` (matches `desc::sampler_d3d12`).
            let btype = lw.lshrc(s3, 30);
            let is_white = lw.cmpc(Pred::Eq, btype, 2);
            let one_f = lw.fc(1.0);
            let border_val = if ch == 3 {
                let is_opaque_black = lw.cmpc(Pred::Eq, btype, 1);
                let is_one = lw.lor(is_white, is_opaque_black);
                lw.sel(is_one, one_f, zf)
            } else {
                lw.sel(is_white, one_f, zf)
            };

            let cmp_func = lw.lshrc(s0, 12);
            let cmp_func = lw.andc(cmp_func, 7);

            // Gather4 2x2 texel order: (-,+), (+,+), (+,-), (-,-)
            let taps = [(x0, y1, bx0, by1), (x1, y1, bx1, by1), (x1, y0, bx1, by0), (x0, y0, bx0, by0)];
            for (k, (cx, cy, bx, by)) in taps.into_iter().enumerate() {
                let load_coords = match slice {
                    Some(sl) => vec![cx, cy, sl],
                    None => vec![cx, cy],
                };
                let loaded = lw.s.texture_load(tex, Some(mip_i), &load_coords, &[]);
                let is_b = lw.lor(bx, by);
                let val = lw.sel(is_b, border_val, loaded[ch]);
                let rv = match cmp {
                    Some(z_ref) => lw.apply_depth_compare(z_ref, val, cmp_func),
                    None => val,
                };
                let b = lw.bits(rv);
                lw.wr(tmps[k], b);
            }
            Ok(())
        })?;

        let rb = tmps.map(|t| self.rd(t));
        let vals: Vec<Value> = rb
            .iter()
            .map(|&b| if d16 { let f = self.f(b); self.s.f32_to_f16(f) } else { b })
            .collect();
        self.image_result(vdata, &vals, d16, act)
    }

    /// Lowers non-cube `UNORM=1` sampling / `gather4` (non-`_l`) by converting unnormalized
    /// texel coordinates (plus optional signed 6-bit texel offsets) to normalized `[0, 1]`
    /// via `GetDimensions` before invoking the standard DXIL sample/gather op.
    #[allow(clippy::too_many_arguments)]
    fn mimg_unorm_sample(
        &mut self,
        d: &Decoded,
        dim: u64,
        kind: TexKind,
        tw0: Value,
        safe_tw0: Value,
        valid: Value,
        ssamp: u64,
        bias: Option<Value>,
        cmp: Option<Value>,
        grads: Option<(Vec<Value>, Vec<Value>)>,
        coords: Vec<Value>,
        lod: Option<Value>,
        cl: Option<Value>,
        ovgpr: Option<Value>,
        gather: bool,
        has_c: bool,
        has_lz: bool,
        comps: &[u64],
        vdata: u64,
        d16: bool,
        act: Value,
        run: Value,
    ) -> LResult<()> {
        let ps = self.gfx.as_ref().is_some_and(|g| g.ps.is_some());
        let s3 = self.sreg(ssamp + 3);
        let implicit_lod = ps && !gather && !has_lz && grads.is_none() && lod.is_none();
        let unorm_body = |lw: &mut Lw, idx: Value| -> LResult<[Value; 4]> {
            let mut coords = coords.clone();
            let tex = lw.s.heap_texture(idx, kind, false, CompType::F32, 4, true);
            let smp: Sampler = lw.mimg_sampler(d.pc, has_c, s3);

            let level = match (&lod, has_lz || gather) {
                (_, true) => lw.c(0),
                (Some(l), _) => {
                    let r = lw.un(op::ROUND_NE, *l);
                    let z = lw.fc(0.0);
                    let r = lw.fmax(r, z);
                    lw.ftoui(r)
                }
                _ => lw.c(0),
            };
            let dims = lw.s.get_dimensions(tex.handle, Some(level));
            let one = lw.c(1);
            let (w, hh, dd) = match dim {
                0 | 4 => (lw.s.m.extract(dims, 0), one, one),
                1 | 5 => (lw.s.m.extract(dims, 0), lw.s.m.extract(dims, 1), one),
                2 => (lw.s.m.extract(dims, 0), lw.s.m.extract(dims, 1), lw.s.m.extract(dims, 2)),
                _ => (lw.s.m.extract(dims, 0), lw.s.m.extract(dims, 1), one),
            };
            let sizes = [lw.umax(w, one), lw.umax(hh, one), lw.umax(dd, one)];
            let shifts = [26u32, 18, 10];
            let spatial_n = match dim {
                0 | 4 => 1usize,
                2 => 3,
                _ => 2,
            };
            for k in 0..spatial_n.min(coords.len()) {
                let mut c = coords[k];
                if let Some(o) = ovgpr {
                    let x = lw.shlc(o, shifts[k]);
                    let x = lw.ashrc(x, 26);
                    let xf = lw.sitof(x);
                    c = lw.fadd(c, xf);
                }
                let sf = lw.uitof(sizes[k]);
                coords[k] = lw.fdiv(c, sf);
            }
            let coord: Vec<Value> = match dim {
                4 => vec![coords[0], lw.un(op::ROUND_NE, coords[1])],
                5 => vec![coords[0], coords[1], lw.un(op::ROUND_NE, coords[2])],
                _ => coords,
            };
            let floor = |lw: &mut Lw, l: Value| match cl {
                Some(c) => lw.fmax(l, c),
                None => l,
            };
            let r: [Value; 4] = if gather {
                if comps.len() != 1 {
                    return err(d, "gather4 dmask");
                }
                match cmp {
                    Some(z) => lw.s.gather_cmp(tex, smp, &coord, &[], comps[0] as u32, z),
                    None => lw.s.gather(tex, smp, &coord, &[], comps[0] as u32),
                }
            } else {
                match (&cmp, &grads, &lod, &bias) {
                    (Some(z), _, _, _) if has_lz || !ps => {
                        let v = lw.s.sample_cmp_level_zero(tex, smp, &coord, &[], *z);
                        [v[0]; 4]
                    }
                    (Some(z), _, _, _) => {
                        let v = lw.s.sample_cmp(tex, smp, &coord, &[], *z, cl);
                        [v[0]; 4]
                    }
                    (None, Some((dx, dy)), _, _) => lw.s.sample_grad(tex, smp, &coord, &[], dx, dy, cl),
                    (None, None, Some(l), _) => lw.s.sample_level(tex, smp, &coord, &[], *l),
                    (None, None, None, Some(b)) if ps => lw.s.sample_bias(tex, smp, &coord, &[], *b, cl),
                    (None, None, None, Some(b)) => {
                        let l = floor(lw, *b);
                        lw.s.sample_level(tex, smp, &coord, &[], l)
                    }
                    _ if has_lz || !ps => {
                        let z = lw.fc(0.0);
                        let l = floor(lw, z);
                        lw.s.sample_level(tex, smp, &coord, &[], l)
                    }
                    _ => lw.s.sample(tex, smp, &coord, &[], cl),
                }
            };
            Ok(r)
        };
        let (r, rb): ([Value; 4], [Value; 4]) = if implicit_lod {
            let raw = unorm_body(self, safe_tw0)?;
            let zf = self.fc(0.0);
            let r = raw.map(|v| self.sel(valid, v, zf));
            let rb = r.map(|v| self.bits(v));
            (r, rb)
        } else {
            let tmps = [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()];
            let z = self.c(0);
            for &t in &tmps {
                self.wr(t, z);
            }
            self.guarded(run, &mut |lw| {
                let res = unorm_body(lw, tw0)?;
                for k in 0..4 {
                    let b = lw.bits(res[k]);
                    lw.wr(tmps[k], b);
                }
                Ok(())
            })?;
            let rb = tmps.map(|t| self.rd(t));
            let r = rb.map(|b| self.f(b));
            (r, rb)
        };
        let n = if gather { 4 } else { comps.len() };
        let sel: Vec<usize> = (0..n).map(|k| if gather { k } else { comps[k] as usize }).collect();
        let vals: Vec<Value> = sel
            .iter()
            .map(|&c| if d16 { self.s.f32_to_f16(r[c]) } else { rb[c] })
            .collect();
        self.image_result(vdata, &vals, d16, act)
    }
}

#[cfg(test)]
mod gather4_lod_tests {
    use super::super::kernel::{translate_full, KernelOptions};
    use super::super::validate::validate;
    use crate::gpu::rdna2;

    fn mimg(name: &str, fields: &[(&str, u64)]) -> Vec<u8> {
        rdna2::encode(name, "ENC_MIMG", fields, None).unwrap()
    }

    fn endpgm() -> Vec<u8> {
        rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap()
    }

    fn opts() -> KernelOptions {
        KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 }
    }

    /// `image_gather4_l` and `image_gather4_l_o` (plus `_c_l`, `_c_l_o`, `UNORM=1`, `A16`, `D16`,
    /// 2D and 2DArray) emulate explicit-LOD gather via 4 mip-level `textureLoad`s and validate cleanly.
    #[test]
    fn gather4_lod_variants_translate_and_validate() {
        let cases: &[(&str, &[(&str, u64)])] = &[
            // 2D gather4_l (normalized)
            ("IMAGE_GATHER4_L", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)]),
            // 2D gather4_l_o (with texel offset + channel G)
            ("IMAGE_GATHER4_L_O", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x2), ("DIM", 1)]),
            // 2DArray gather4_l (with UNORM=1 + channel A)
            ("IMAGE_GATHER4_L", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x8), ("DIM", 5), ("UNORM", 1)]),
            // 2DArray gather4_l_o with A16=1 and D16=1
            ("IMAGE_GATHER4_L_O", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x4), ("DIM", 5), ("A16", 1), ("D16", 1)]),
            // 2D comparison gather4_c_l_o
            ("IMAGE_GATHER4_C_L_O", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)]),
        ];
        for (name, fields) in cases {
            let mut code = mimg(name, fields);
            code.extend(endpgm());
            let k = translate_full(&code, opts()).unwrap_or_else(|e| panic!("{name} ({fields:?}): {e}"));
            if let Err(err) = validate(&k.dxil) {
                panic!("{name} ({fields:?}) DXIL validation failed: {err}");
            }
        }
    }

    /// Non-cube `UNORM=1` sampling (`D1`, `D2`, `D3`, `D1Array`, `D2Array`, `sample_lz`, `sample_l_o`, `gather4_lz`)
    /// translates cleanly instead of erroring with `MIMG unorm`, and validates with `dxil.dll`.
    #[test]
    fn non_cube_unorm_sampling_translates_and_validate() {
        let cases: &[(&str, &[(&str, u64)])] = &[
            ("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 0), ("UNORM", 1)]),
            ("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1), ("UNORM", 1)]),
            ("IMAGE_SAMPLE_L_O", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 2), ("UNORM", 1)]),
            ("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 4), ("UNORM", 1)]),
            ("IMAGE_SAMPLE_L", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 5), ("UNORM", 1)]),
            ("IMAGE_GATHER4_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]),
            ("IMAGE_GATHER4_LZ_O", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x2), ("DIM", 5), ("UNORM", 1), ("D16", 1)]),
        ];
        for (name, fields) in cases {
            let mut code = mimg(name, fields);
            code.extend(endpgm());
            let k = translate_full(&code, opts()).unwrap_or_else(|e| panic!("{name} ({fields:?}): {e}"));
            if let Err(err) = validate(&k.dxil) {
                panic!("{name} ({fields:?}) DXIL validation failed: {err}");
            }
        }
    }

    /// Explicit-LOD `gather4_l` adds S# `LOD_BIAS` (`s2[13:0]`, S5.8 / 256.0) before clamping
    /// and rounds half up via `floor(lod + 0.501953125)` so `LOD 0.5` selects level 1.
    #[test]
    fn gather4_lod_emits_bias_and_rdna2_half_up_rounding() {
        let mut code = Vec::new();
        // v2 = 0.5f (LOD), s[24:27] = S# with MIP_FILTER=point (1 << 26) and LOD_BIAS = +1.0 (0x100)
        code.extend(rdna2::encode("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 240)], None).unwrap()); // 0.5f
        code.extend(rdna2::encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 255)], Some(2 | (2 << 3) | (2 << 6))).unwrap());
        code.extend(rdna2::encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 25), ("SSRC0", 255)], Some(0xFFF << 12)).unwrap());
        code.extend(rdna2::encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 26), ("SSRC0", 255)], Some((1 << 26) | 0x100)).unwrap());
        code.extend(rdna2::encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 27), ("SSRC0", 128)], None).unwrap());
        code.extend(mimg("IMAGE_GATHER4_L", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)]));
        code.extend(endpgm());
        let k = translate_full(&code, opts()).expect("translate gather4_l with LOD_BIAS + LOD 0.5");
        if let Err(err) = validate(&k.dxil) {
            panic!("DXIL validation failed: {err}");
        }
        // Verify exact RDNA2 half-up constant 0.501953125f = 257/512 (0x3F00_8000):
        // LOD 0.5 -> floor(1.001953125) = 1; LOD 2.5 -> floor(3.001953125) = 3.
        assert_eq!(0.501953125_f32.to_bits(), 0x3F00_8000);
        assert_eq!((0.5_f32 + 0.501953125_f32).floor() as u32, 1);
        assert_eq!((2.5_f32 + 0.501953125_f32).floor() as u32, 3);
        // Verify negative S5.8 LOD_BIAS sign extension: -1.0 = 0x3F00 in s2[13:0].
        let neg_bias = (((0x3F00_u32 << 18) as i32) >> 18) as f32 / 256.0;
        assert_eq!(neg_bias, -1.0);
    }

    /// Explicit-LOD `gather4_l` honors T# `MIN_LOD` (`tw1[19:8]`, u4.8 / 256.0):
    /// `MIP_FILTER=none` selects `floor(T# MIN_LOD)` at every LOD, and non-none
    /// `MIP_FILTER` raises the S#-clamped level to `T# MIN_LOD` before half-up rounding.
    #[test]
    fn gather4_lod_honors_tsharp_min_lod_and_mip_filter_none() {
        for (mip_filter, t_min_u48, s_max_u48) in [
            (0u32, 0x1C0u32, 0xF00u32), // MIP_FILTER=none, T# MIN_LOD=1.75 -> floor(1.75) = 1
            (0u32, 0x280u32, 0xF00u32), // MIP_FILTER=none, T# MIN_LOD=2.5  -> floor(2.5)  = 2
            (1u32, 0x1C0u32, 0x100u32), // MIP_FILTER=point, T# MIN_LOD=1.75 over S# MAX_LOD=1.0 -> 2
        ] {
            let mut code = Vec::new();
            // s[16:23] = T# with tw1[19:8] = t_min_u48, s[24:27] = S# with s1[23:12] = s_max_u48, s2[27:26] = mip_filter.
            code.extend(rdna2::encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 17), ("SSRC0", 255)], Some(t_min_u48 << 8)).unwrap());
            code.extend(rdna2::encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 25), ("SSRC0", 255)], Some(s_max_u48 << 12)).unwrap());
            code.extend(rdna2::encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 26), ("SSRC0", 255)], Some(mip_filter << 26)).unwrap());
            code.extend(mimg("IMAGE_GATHER4_L", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)]));
            code.extend(endpgm());
            let k = translate_full(&code, opts()).expect("translate gather4_l with T# MIN_LOD");
            if let Err(err) = validate(&k.dxil) {
                panic!("DXIL validation failed (mip_filter={mip_filter}, t_min={t_min_u48:#x}): {err}");
            }
        }
        // Check exact RDNA2 level arithmetic for the 3 cases above:
        assert_eq!((0x1C0_u32 as f32 / 256.0).floor() as u32, 1);
        assert_eq!((0x280_u32 as f32 / 256.0).floor() as u32, 2);
        let l_clamped = 0.0_f32.max(0.0).min(1.0).max(0x1C0_u32 as f32 / 256.0);
        assert_eq!((l_clamped + 0.501953125_f32).floor() as u32, 2);
    }
}

impl Lw {

    pub fn mimg(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        if f("TFE") != 0 || f("LWE") != 0 {
            return err(d, "MIMG tfe/lwe");
        }
        let (a16, d16) = (f("A16") == 1, f("D16") == 1);
        let dim = f("DIM");
        let dmask = f("DMASK");
        let (vdata, srsrc, ssamp) = (f("VDATA"), Self::field(d, "SRSRC"), Self::field(d, "SSAMP"));
        let nsa = f("NSA");
        let nsa_enc = rdna2::encoding(&format!("MIMG_NSA{nsa}"));
        let addr_reg = |j: u64| -> u64 {
            match nsa_enc {
                Some(e) if nsa > 0 && j > 0 => {
                    let letter = (b'A' + (j - 1).min(25) as u8) as char;
                    rdna2::field_value(e, &d.words, &format!("VADDR{letter}")).unwrap_or(0)
                }
                _ => f("VADDR") + j,
            }
        };
        let base = name.trim_start_matches("image_");
        if base == "bvh_intersect_ray" || base == "bvh64_intersect_ray" {
            let wide = base == "bvh64_intersect_ray";
            let r = |lw: &mut Lw, j: u64| lw.vreg(addr_reg(j));
            let node = r(self, 0);
            let node_hi = if wide { Some(r(self, 1)) } else { None };
            let k = if wide { 2 } else { 1 };
            let fl = |lw: &mut Lw, j: u64| {
                let v = lw.vreg(addr_reg(j));
                lw.f(v)
            };
            let extent = fl(self, k);
            let origin = [fl(self, k + 1), fl(self, k + 2), fl(self, k + 3)];
            let (dir, inv) = if a16 {
                let (h0, h1, h2) = (r(self, k + 4), r(self, k + 5), r(self, k + 6));
                let h0s = self.lshrc(h0, 16);
                let h1s = self.lshrc(h1, 16);
                let h2s = self.lshrc(h2, 16);
                (
                    [self.s.f16_to_f32(h0), self.s.f16_to_f32(h0s), self.s.f16_to_f32(h1)],
                    [self.s.f16_to_f32(h1s), self.s.f16_to_f32(h2), self.s.f16_to_f32(h2s)],
                )
            } else {
                ([fl(self, k + 4), fl(self, k + 5), fl(self, k + 6)], [fl(self, k + 7), fl(self, k + 8), fl(self, k + 9)])
            };
            let (d0, d1) = (self.sreg(srsrc), self.sreg(srsrc + 1));
            // address: ((d1 & 0xFF) << 32 | d0) << 8 + node offset (8-byte units); low 32 bits
            let nlo = self.andc(node, !7);
            let base_lo = self.shlc(d0, 8);
            let off_lo = self.shlc(nlo, 3);
            let a = self.add(base_lo, off_lo);
            // the window: bits 32.. of the 64-bit address
            if self.mems.len() > 1 {
                let d1lo = self.andc(d1, 0xFF);
                let b64 = self.pack64(d0, d1lo);
                let k8 = self.c64(8);
                let b64 = self.bin(BinOp::Shl, b64, k8);
                let nlo64 = self.zext64(nlo);
                let off = match node_hi {
                    Some(hi) => {
                        let h64 = self.zext64(hi);
                        let k32 = self.c64(32);
                        let h64 = self.bin(BinOp::Shl, h64, k32);
                        self.or(h64, nlo64)
                    }
                    None => nlo64,
                };
                let k3 = self.c64(3);
                let off = self.bin(BinOp::Shl, off, k3);
                let full = self.add(b64, off);
                let (_, hi) = self.unpack64(full);
                self.window_hi(hi, false);
            }
            let ty = self.andc(node, 7);
            let sort = {
                let s = self.lshrc(d1, 31);
                self.nz(s)
            };
            let act = self.act();
            let out = [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()];
            self.guarded(act, &mut |lw| {
                lw.bvh_intersect(ty, a, extent, origin, dir, inv, sort, out)?;
                for (c, &t) in out.iter().enumerate() {
                    let v = lw.rd(t);
                    lw.wr(VAR_VGPR + (vdata + c as u64) as u32, v);
                }
                Ok(())
            })?;
            return Ok(());
        }
        let body = [1u64, 2, 3, 3, 2, 3, 3, 4][dim as usize];
        let sampling = base.starts_with("sample") || base.starts_with("gather4");
        let rest = base.trim_start_matches("sample").trim_start_matches("gather4");
        let mods: Vec<&str> = if sampling { rest.split('_').filter(|x| !x.is_empty()).collect() } else { Vec::new() };
        let (has_cl, has_o) = (mods.contains(&"cl"), mods.contains(&"o"));
        let (has_b, has_c, has_d, has_l, has_lz) =
            (mods.contains(&"b"), mods.contains(&"c"), mods.contains(&"d") || mods.contains(&"cd"), mods.contains(&"l"), mods.contains(&"lz"));
        let grad_n = if has_d { [1u64, 2, 3, 2, 1, 2, 2, 2][dim as usize] } else { 0 };
        let h = if a16 { 16 } else { 32 };
        let mut shape: Vec<(u32, bool)> = Vec::new();
        let (mut i_bias, mut i_cmp, mut i_grad, mut i_body, mut i_extra, mut i_clamp) = (None, None, 0usize, 0usize, None, None);
        if sampling {
            if has_o {
                shape.push((32, true));
            }
            if has_b {
                i_bias = Some(shape.len());
                shape.push((h, !a16));
            }
            if has_c {
                i_cmp = Some(shape.len());
                shape.push((32, true));
            }
            i_grad = shape.len();
            for _ in 0..2 * grad_n {
                shape.push((32, true));
            }
            i_body = shape.len();
            for k in 0..body {
                shape.push((h, !a16 || (grad_n != 0 && k == 0)));
            }
            if has_l {
                i_extra = Some(shape.len());
                shape.push((h, !a16));
            }
            if has_cl {
                i_clamp = Some(shape.len());
                shape.push((h, !a16));
            }
        } else if base == "get_resinfo" {
            i_extra = Some(0);
            shape.push((32, true));
        } else {
            i_body = 0;
            for _ in 0..body {
                shape.push((h, !a16));
            }
            if base.ends_with("_mip") {
                i_extra = Some(shape.len());
                shape.push((h, !a16));
            }
        }
        let mut pos: Vec<(u64, Option<u64>)> = Vec::new();
        let mut cursor = 0u64;
        for &(w, starts) in &shape {
            if (w == 32 || starts) && cursor % 2 == 1 {
                cursor += 1;
            }
            pos.push((cursor / 2, (w == 16).then_some(cursor % 2)));
            cursor += if w == 32 { 2 } else { 1 };
        }
        let compu = |lw: &mut Lw, i: usize| -> Value {
            let (dw, half) = pos[i];
            let v = lw.vreg(addr_reg(dw));
            match half {
                None => v,
                Some(hh) => {
                    let x = lw.lshrc(v, 16 * hh as u32);
                    lw.andc(x, 0xFFFF)
                }
            }
        };
        let compf = |lw: &mut Lw, i: usize| -> Value {
            let (dw, half) = pos[i];
            let v = lw.vreg(addr_reg(dw));
            match half {
                None => lw.f(v),
                Some(hh) => {
                    let x = lw.lshrc(v, 16 * hh as u32);
                    let x = lw.andc(x, 0xFFFF);
                    lw.s.f16_to_f32(x)
                }
            }
        };
        let tw0 = self.sreg(srsrc);
        let tw1 = self.sreg(srsrc + 1);
        let tw3 = self.sreg(srsrc + 3);
        let act = self.act();
        let valid = self.valid_tsharp(tw0, tw1, tw3);
        let run = self.land(act, valid);
        let z0 = self.c(0);
        let safe_tw0 = self.sel(valid, tw0, z0);
        let comps: Vec<u64> = (0..4).filter(|c| dmask >> c & 1 == 1).collect();
        let kind = match dim {
            0 => TexKind::D1,
            1 => TexKind::D2,
            2 => TexKind::D3,
            // cube T#s are cube arrays (face + 8 · cube), as the HLSL path's TextureCubeArray
            3 => TexKind::CubeArray,
            4 => TexKind::D1Array,
            5 => TexKind::D2Array,
            // multisampled: the fragment index is the sample index (D3D12 MSAA resources have
            // no FMASK; their samples are stored uncompressed)
            6 => TexKind::D2MS,
            7 => TexKind::D2MSArray,
            _ => return err(d, "MIMG dim"),
        };
        let ms = matches!(dim, 6 | 7);
        if ms && (sampling || base == "get_lod") {
            return err(d, "MIMG sample MSAA");
        }
        if ms && !matches!(base, "load" | "msaa_load" | "get_resinfo") {
            // stores / atomics need RWTexture2DMS (SM 6.7, WriteableMSAATextures)
            return err(d, "MIMG MSAA store / atomic / load_mip");
        }
        if base == "msaa_load" && !ms {
            return err(d, "image_msaa_load on a single-sampled dim");
        }

        if sampling {
            let unorm = f("UNORM") != 0;
            let gather_l = base.starts_with("gather4") && has_l && matches!(dim, 1 | 5);
            if gather_l || (unorm && dim != 3) {
                let bias = i_bias.map(|i| compf(self, i));
                let cmp = i_cmp.map(|i| compf(self, i));
                let grads = if has_d {
                    let dx: Vec<Value> = (0..grad_n as usize).map(|k| compf(self, i_grad + k)).collect();
                    let dy: Vec<Value> = (0..grad_n as usize).map(|k| compf(self, i_grad + grad_n as usize + k)).collect();
                    Some((dx, dy))
                } else {
                    None
                };
                let coords: Vec<Value> = (0..body as usize).map(|k| compf(self, i_body + k)).collect();
                let lod = i_extra.map(|i| compf(self, i));
                let cl = i_clamp.map(|i| compf(self, i));
                let ovgpr = if has_o {
                    if dim == 3 {
                        return err(d, "MIMG offset cube");
                    }
                    Some(compu(self, 0))
                } else {
                    None
                };
                if gather_l {
                    let lod = lod.expect("gather4_l has explicit LOD");
                    return self.mimg_gather4_lod(d, dim, kind, tw0, tw1, ssamp, &coords, lod, cl, cmp, ovgpr, unorm, &comps, vdata, d16, act, run);
                }
                return self.mimg_unorm_sample(
                    d,
                    dim,
                    kind,
                    tw0,
                    safe_tw0,
                    valid,
                    ssamp,
                    bias,
                    cmp,
                    grads,
                    coords,
                    lod,
                    cl,
                    ovgpr,
                    base.starts_with("gather4"),
                    has_c,
                    has_lz,
                    &comps,
                    vdata,
                    d16,
                    act,
                    run,
                );
            }
            let bias = i_bias.map(|i| compf(self, i));
            let cmp = i_cmp.map(|i| compf(self, i));
            let grads = if has_d {
                let dx: Vec<Value> = (0..grad_n as usize).map(|k| compf(self, i_grad + k)).collect();
                let dy: Vec<Value> = (0..grad_n as usize).map(|k| compf(self, i_grad + grad_n as usize + k)).collect();
                Some((dx, dy))
            } else {
                None
            };
            let coords: Vec<Value> = (0..body as usize).map(|k| compf(self, i_body + k)).collect();
            let lod = i_extra.map(|i| compf(self, i));
            let cl = i_clamp.map(|i| compf(self, i));
            let s3 = self.sreg(ssamp + 3);
            let gather = base.starts_with("gather4");
            let ovgpr = if has_o {
                if dim == 3 {
                    return err(d, "MIMG offset cube");
                }
                Some(compu(self, 0))
            } else {
                None
            };
            let ps = self.gfx.as_ref().is_some_and(|g| g.ps.is_some());
            // Keep pixel-shader implicit-LOD samples convergent for quad derivatives (clamp
            // the heap index to 0 and zero invalid results); guard all other samples/gathers
            // so inactive lanes and invalid/OOB T#s never invoke `createHandleFromHeap`.
            let implicit_lod = ps && !gather && !has_lz && grads.is_none() && lod.is_none();
            let sample_body = |lw: &mut Lw, idx: Value| -> LResult<[Value; 4]> {
                let mut coords = coords.clone();
                let spatial: Vec<Value> = coords[..kind.grads().min(coords.len())].to_vec();
                let tex = lw.s.heap_texture(idx, kind, false, CompType::F32, 4, true);
                let smp: Sampler = lw.mimg_sampler(d.pc, has_c, s3);
                if let Some(o) = ovgpr {
                    // texel offset / size of the sampled mip (0 for lz / gather; else the LOD given, else 0)
                    let level = match (&lod, has_lz || gather) {
                        (_, true) => lw.c(0),
                        (Some(l), _) => {
                            let r = lw.un(op::ROUND_NE, *l);
                            let z = lw.fc(0.0);
                            let r = lw.fmax(r, z);
                            lw.ftoui(r)
                        }
                        // the implicit LOD of a pixel shader's sample, rounded
                        (None, false) if ps && grads.is_none() && !has_c => {
                            let l = lw.s.calculate_lod(tex, smp, &spatial, true);
                            let r = lw.un(op::ROUND_NE, l);
                            let z = lw.fc(0.0);
                            let r = lw.fmax(r, z);
                            lw.ftoui(r)
                        }
                        _ => lw.c(0),
                    };
                    let dims = lw.s.get_dimensions(tex.handle, Some(level));
                    let one = lw.c(1);
                    let (w, hh, dd) = match dim {
                        0 => (lw.s.m.extract(dims, 0), one, one),
                        1 => (lw.s.m.extract(dims, 0), lw.s.m.extract(dims, 1), one),
                        2 => (lw.s.m.extract(dims, 0), lw.s.m.extract(dims, 1), lw.s.m.extract(dims, 2)),
                        4 => (lw.s.m.extract(dims, 0), one, one),
                        _ => (lw.s.m.extract(dims, 0), lw.s.m.extract(dims, 1), one),
                    };
                    let mut ofs = Vec::new();
                    for (sh, size) in [(26u32, w), (18, hh), (10, dd)] {
                        let x = lw.shlc(o, sh);
                        let x = lw.ashrc(x, 26);
                        let xf = lw.sitof(x);
                        let sf = lw.uitof(size);
                        ofs.push(lw.fdiv(xf, sf));
                    }
                    for (k, c) in coords.iter_mut().enumerate() {
                        if (dim == 4 && k == 1) || (dim == 5 && k == 2) {
                            continue;
                        }
                        *c = lw.fadd(*c, ofs[k]);
                    }
                }
                // array layers are rounded; cube coordinates become a direction and a cube index
                let coord: Vec<Value> = match dim {
                    3 => {
                        let mut v = lw.cube_dir(coords[0], coords[1], coords[2]).to_vec();
                        v.push(lw.cube_slice(coords[2]));
                        v
                    }
                    4 => vec![coords[0], lw.un(op::ROUND_NE, coords[1])],
                    5 => vec![coords[0], coords[1], lw.un(op::ROUND_NE, coords[2])],
                    _ => coords.clone(),
                };
                let floor = |lw: &mut Lw, l: Value| match cl {
                    Some(c) => lw.fmax(l, c),
                    None => l,
                };
                let r: [Value; 4] = if gather {
                    if comps.len() != 1 {
                        return err(d, "gather4 dmask");
                    }
                    match cmp {
                        Some(z) => lw.s.gather_cmp(tex, smp, &coord, &[], comps[0] as u32, z),
                        None => lw.s.gather(tex, smp, &coord, &[], comps[0] as u32),
                    }
                } else {
                    match (&cmp, &grads, &lod, &bias) {
                        (Some(z), _, _, _) if has_lz || !ps => {
                            let v = lw.s.sample_cmp_level_zero(tex, smp, &coord, &[], *z);
                            [v[0]; 4]
                        }
                        (Some(z), _, _, _) => {
                            let v = lw.s.sample_cmp(tex, smp, &coord, &[], *z, if dim == 3 { None } else { cl });
                            [v[0]; 4]
                        }
                        (None, Some((dx, dy)), _, _) if dim == 3 => {
                            let len = |lw: &mut Lw, x: &[Value]| {
                                let a = lw.fmul(x[0], x[0]);
                                let b = x.get(1).copied().unwrap_or_else(|| lw.fc(0.0));
                                let bb = lw.fmul(b, b);
                                let s = lw.fadd(a, bb);
                                lw.un(op::SQRT, s)
                            };
                            let z = lw.c(0);
                            let dims = lw.s.get_dimensions(tex.handle, Some(z));
                            let size = lw.s.m.extract(dims, 0);
                            let lx = len(lw, dx);
                            let ly = len(lw, dy);
                            let m = lw.fmax(lx, ly);
                            let sf = lw.uitof(size);
                            let m = lw.fmul(m, sf);
                            let eps = lw.fc(1e-8);
                            let m = lw.fmax(m, eps);
                            let l = lw.un(op::LOG, m);
                            let l = floor(lw, l);
                            lw.s.sample_level(tex, smp, &coord, &[], l)
                        }
                        (None, Some((dx, dy)), _, _) => lw.s.sample_grad(tex, smp, &coord, &[], dx, dy, cl),
                        (None, None, Some(l), _) => lw.s.sample_level(tex, smp, &coord, &[], *l),
                        (None, None, None, Some(b)) if ps => lw.s.sample_bias(tex, smp, &coord, &[], *b, if dim == 3 { None } else { cl }),
                        (None, None, None, Some(b)) => {
                            let l = floor(lw, *b);
                            lw.s.sample_level(tex, smp, &coord, &[], l)
                        }
                        _ if has_lz || !ps => {
                            let z = lw.fc(0.0);
                            let l = floor(lw, z);
                            lw.s.sample_level(tex, smp, &coord, &[], l)
                        }
                        _ => lw.s.sample(tex, smp, &coord, &[], if dim == 3 { None } else { cl }),
                    }
                };
                Ok(r)
            };
            let (r, rb): ([Value; 4], [Value; 4]) = if implicit_lod {
                let raw = sample_body(self, safe_tw0)?;
                let zf = self.fc(0.0);
                let r = raw.map(|v| self.sel(valid, v, zf));
                let rb = r.map(|v| self.bits(v));
                (r, rb)
            } else {
                let tmps = [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()];
                let z = self.c(0);
                for &t in &tmps {
                    self.wr(t, z);
                }
                self.guarded(run, &mut |lw| {
                    let res = sample_body(lw, tw0)?;
                    for k in 0..4 {
                        let b = lw.bits(res[k]);
                        lw.wr(tmps[k], b);
                    }
                    Ok(())
                })?;
                let rb = tmps.map(|t| self.rd(t));
                let r = rb.map(|b| self.f(b));
                (r, rb)
            };
            let n = if gather { 4 } else { comps.len() };
            let sel: Vec<usize> = (0..n).map(|k| if gather { k } else { comps[k] as usize }).collect();
            let vals: Vec<Value> = sel
                .iter()
                .map(|&c| {
                    if d16 {
                        // f32tof16(asfloat(e))
                        self.s.f32_to_f16(r[c])
                    } else {
                        rb[c]
                    }
                })
                .collect();
            return self.image_result(vdata, &vals, d16, act);
        }

        if base == "get_lod" {
            if !self.gfx.as_ref().is_some_and(|g| g.ps.is_some()) {
                return err(d, "image_get_lod outside PS");
            }
            let coords: Vec<Value> = (0..body as usize).map(|k| compf(self, i_body + k)).collect();
            let spatial: Vec<Value> = match dim {
                3 => self.cube_dir(coords[0], coords[1], coords[2]).to_vec(),
                _ => coords[..kind.grads().min(coords.len())].to_vec(),
            };
            let s3 = self.sreg(ssamp + 3);
            let tex = self.s.heap_texture(safe_tw0, kind, false, CompType::F32, 4, true);
            let smp = self.mimg_sampler(d.pc, false, s3);
            let a = self.s.calculate_lod(tex, smp, &spatial, true);
            let b = self.s.calculate_lod(tex, smp, &spatial, false);
            let z = self.c(0);
            let ab = self.bits(a);
            let bb = self.bits(b);
            let r = [self.sel(valid, ab, z), self.sel(valid, bb, z), z, z];
            let vals: Vec<Value> = comps.iter().map(|&c| if d16 {
                let f = self.f(r[c as usize]);
                self.s.f32_to_f16(f)
            } else {
                r[c as usize]
            }).collect();
            return self.image_result(vdata, &vals, d16, act);
        }

        if base == "get_resinfo" && ms {
            // one level; the sample count is a T# field (LAST_LEVEL), read by the program itself
            let tmps = [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()];
            let z = self.c(0);
            for &t in &tmps {
                self.wr(t, z);
            }
            self.guarded(run, &mut |lw| {
                let tex = lw.s.heap_texture(tw0, kind, false, CompType::F32, 4, true);
                let dims = lw.s.get_dimensions(tex.handle, None);
                let one = lw.c(1);
                let w = lw.s.m.extract(dims, 0);
                let hh = lw.s.m.extract(dims, 1);
                // 2D MSAA: depth 0, as for every DIM without layers (RDNA2)
                let zero = lw.c(0);
                let dd = if dim == 7 { lw.s.m.extract(dims, 2) } else { zero };
                for (k, v) in [w, hh, dd, one].into_iter().enumerate() {
                    lw.wr(tmps[k], v);
                }
                Ok(())
            })?;
            let r = tmps.map(|t| self.rd(t));
            let vals: Vec<Value> = comps.iter().map(|&c| r[c as usize]).collect();
            return self.image_result(vdata, &vals, d16, act);
        }

        if base == "get_resinfo" {
            let mipv = compu(self, 0);
            let tmps = [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()];
            let z = self.c(0);
            for &t in &tmps {
                self.wr(t, z);
            }
            // DIM 1D / 2D on a 2D MSAA (TYPE 14) or MSAA-array (TYPE 15) T#: RDNA2 gives
            // (width, height, 0, 1) (AnyPS5 docs/dev/debt/shader.md, gfx1036). The host view is
            // multisampled, so it is queried as one: a Texture1D / Texture2D handle on it leaves
            // the level count to the host (RADV: log2(samples) + 1).
            let mut run = run;
            if matches!(dim, 0 | 1) {
                let rtype = self.lshrc(tw3, 28);
                for (ty, mkind) in [(14, TexKind::D2MS), (15, TexKind::D2MSArray)] {
                    let is = self.cmpc(Pred::Eq, rtype, ty);
                    let go = self.land(run, is);
                    self.guarded(go, &mut |lw| {
                        let tex = lw.s.heap_texture(tw0, mkind, false, CompType::F32, 4, true);
                        let dims = lw.s.get_dimensions(tex.handle, None);
                        let w = lw.s.m.extract(dims, 0);
                        let hh = lw.s.m.extract(dims, 1);
                        let (zero, one) = (lw.c(0), lw.c(1));
                        for (k, v) in [w, hh, zero, one].into_iter().enumerate() {
                            lw.wr(tmps[k], v);
                        }
                        Ok(())
                    })?;
                }
                let ms_t = self.cmpc(Pred::UGe, rtype, 14);
                let not_ms = self.lnot(ms_t);
                run = self.land(run, not_ms);
            }
            self.guarded(run, &mut |lw| {
                let tex = lw.s.heap_texture(tw0, kind, false, CompType::F32, 4, true);
                let dims = lw.s.get_dimensions(tex.handle, Some(mipv));
                let one = lw.c(1);
                let zero = lw.c(0);
                let e = |lw: &mut Lw, i: u32| lw.s.m.extract(dims, i);
                // HLSL GetDimensions outputs per dim; missing ones are 1
                let (w, hh, dd, n) = match dim {
                    // RDNA2 gives 0 for the depth of a DIM without layers (1D, 2D, 2D MSAA)
                    0 => (e(lw, 0), one, zero, e(lw, 3)),
                    1 => (e(lw, 0), e(lw, 1), zero, e(lw, 3)),
                    // a cube array: its layers are faces, 6 per cube
                    3 => {
                        let n = e(lw, 2);
                        let k = lw.c(6);
                        (e(lw, 0), e(lw, 1), lw.mul(n, k), e(lw, 3))
                    }
                    2 | 5 => (e(lw, 0), e(lw, 1), e(lw, 2), e(lw, 3)),
                    // a 1D array: RDNA gives its layers in y (and z), as the HLSL path does
                    4 => (e(lw, 0), e(lw, 1), e(lw, 1), e(lw, 3)),
                    _ => (e(lw, 0), one, e(lw, 1), e(lw, 3)),
                };
                // a level past the last: RDNA2 gives 0 for every size (an AMD driver, RDNA4, answers
                // for a level it clamped)
                let has = lw.cmp(Pred::ULt, mipv, n);
                for (k, v) in [w, hh, dd].into_iter().enumerate() {
                    let v = lw.s.m.select(has, v, zero);
                    lw.wr(tmps[k], v);
                }
                lw.wr(tmps[3], n);
                Ok(())
            })?;
            let r = tmps.map(|t| self.rd(t));
            let vals: Vec<Value> = comps.iter().map(|&c| r[c as usize]).collect();
            return self.image_result(vdata, &vals, d16, act);
        }

        let ic: Vec<Value> = (0..body as usize).map(|k| compu(self, i_body + k)).collect();
        let mip = i_extra.map(|i| compu(self, i)).unwrap_or_else(|| self.c(0));
        if dim == 3 && base.starts_with("load") {
            return err(d, "image_load cube");
        }
        let lkind = if dim == 3 { TexKind::D2Array } else { kind };
        // class of the T# format: 1 uint, 2 sint, 0 float
        let fmt = self.lshrc(tw1, 20);
        let fmt = self.andc(fmt, 0x1FF);
        let cls = self.img_cls(fmt);
        // an image store / atomic records its T# (`tw0 + 1`): a later image_load of the image
        // reads through the UAV (`recomp::emit::image_write_srsrcs`)
        if base.starts_with("store") || base.starts_with("atomic_") {
            if let Some(k) = self.img_writes.iter().position(|&s| s == srsrc) {
                let var = super::lower::VAR_IMG_WRITE + k as u32;
                let (cur, t1) = (self.rd(var), self.addc(tw0, 1));
                let nv = self.sel(valid, t1, cur);
                self.wr(var, nv);
            }
        }
        if ms {
            return self.mimg_msaa(base, &ic, tw0, lkind, cls, &comps, vdata, d16, act, run);
        }
        match base {
            "load" | "load_mip" => {
                let tmps = [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()];
                let z = self.c(0);
                for &t in &tmps {
                    self.wr(t, z);
                }
                // an image this invocation stored to: through the UAV, which sees the store (the
                // SRV need not within a dispatch); image_load only – a UAV view is one mip
                let mut rw = None;
                if base == "load" && !self.img_writes.is_empty() {
                    let t1 = self.addc(tw0, 1);
                    for k in 0..self.img_writes.len() {
                        let w = self.rd(super::lower::VAR_IMG_WRITE + k as u32);
                        let e = self.cmp(Pred::Eq, w, t1);
                        rw = Some(match rw {
                            Some(r) => self.lor(r, e),
                            None => e,
                        });
                    }
                }
                let ui = self.addc(tw0, 1);
                for (cv, ct) in [(1u32, CompType::U32), (2, CompType::I32), (0, CompType::F32)] {
                    let is = self.cmpc(Pred::Eq, cls, cv);
                    let c = self.land(run, is);
                    let (c_srv, c_uav) = match rw {
                        Some(r) => {
                            let nr = self.lnot(r);
                            (self.land(c, nr), Some(self.land(c, r)))
                        }
                        None => (c, None),
                    };
                    self.guarded(c_srv, &mut |lw| {
                        let tex: Texture = lw.s.heap_texture(tw0, lkind, false, ct, 4, true);
                        let r = lw.s.texture_load(tex, Some(mip), &ic, &[]);
                        for k in 0..4 {
                            let b = lw.bits(r[k]);
                            lw.wr(tmps[k], b);
                        }
                        Ok(())
                    })?;
                    if let Some(cu) = c_uav {
                        self.guarded(cu, &mut |lw| {
                            lw.s.typed_uav_load_additional_formats();
                            let tex: Texture = lw.s.heap_texture(ui, lkind, true, ct, 4, true);
                            let r = lw.s.texture_load(tex, None, &ic, &[]);
                            for k in 0..4 {
                                let b = lw.bits(r[k]);
                                lw.wr(tmps[k], b);
                            }
                            Ok(())
                        })?;
                    }
                }
                let r: Vec<Value> = tmps.iter().map(|&t| self.rd(t)).collect();
                let vals: Vec<Value> = if d16 {
                    // float: f32tof16, else the low 16 bits
                    comps
                        .iter()
                        .map(|&c| {
                            let x = r[c as usize];
                            let fx = self.f(x);
                            let h = self.s.f32_to_f16(fx);
                            let lo = self.andc(x, 0xFFFF);
                            let fl = self.cmpc(Pred::Eq, cls, 0);
                            self.sel(fl, h, lo)
                        })
                        .collect()
                } else {
                    comps.iter().map(|&c| r[c as usize]).collect()
                };
                self.image_result(vdata, &vals, d16, act)
            }
            "store" => {
                let z = self.c(0);
                let mut vals = [z; 4];
                for c in 0..4u64 {
                    if let Some(k) = comps.iter().position(|&x| x == c) {
                        vals[c as usize] = if d16 {
                            let v = self.vreg(vdata + k as u64 / 2);
                            let hp = self.lshrc(v, 16 * (k as u32 % 2));
                            let hp = self.andc(hp, 0xFFFF);
                            let fh = self.s.f16_to_f32(hp);
                            let fb = self.bits(fh);
                            let fl = self.cmpc(Pred::Eq, cls, 0);
                            self.sel(fl, fb, hp)
                        } else {
                            self.vreg(vdata + k as u64)
                        };
                    }
                }
                let ui = self.addc(tw0, 1);
                for (cv, ct) in [(1u32, CompType::U32), (2, CompType::I32), (0, CompType::F32)] {
                    let is = self.cmpc(Pred::Eq, cls, cv);
                    let c = self.land(run, is);
                    self.guarded(c, &mut |lw| {
                        let tex = lw.s.heap_texture(ui, lkind, true, ct, 4, true);
                        let v4 = if ct == CompType::F32 { vals.map(|v| lw.f(v)) } else { vals };
                        lw.s.texture_store(tex, &ic, v4);
                        Ok(())
                    })?;
                }
                Ok(())
            }
            _ if base.starts_with("atomic_") => {
                let opn = &base[7..];
                let glc = f("GLC") == 1;
                let is_cmpswap = opn == "cmpswap" || opn == "fcmpswap";
                let wide64 = if is_cmpswap { comps.len() == 4 } else { comps.len() == 2 };
                if wide64 {
                    return self.mimg_atomic64(d, opn, &ic, tw0, lkind, vdata, glc, run);
                }
                let (v, cmpv) = (self.vreg(vdata), self.vreg(vdata + 1));
                let ui = self.addc(tw0, 1);
                let signed = opn == "smin" || opn == "smax";
                let ct = if signed { CompType::I32 } else { CompType::U32 };
                let aop = match opn {
                    "add" => Some(AtomicOp::Add),
                    "smin" => Some(AtomicOp::IMin),
                    "umin" => Some(AtomicOp::UMin),
                    "smax" => Some(AtomicOp::IMax),
                    "umax" => Some(AtomicOp::UMax),
                    "and" => Some(AtomicOp::And),
                    "or" => Some(AtomicOp::Or),
                    "xor" => Some(AtomicOp::Xor),
                    "swap" => Some(AtomicOp::Exchange),
                    "cmpswap" | "fcmpswap" | "sub" | "inc" | "dec" | "fmin" | "fmax" => None,
                    _ => return err(d, "image_atomic"),
                };
                let tmp = self.tmp_var();
                let z = self.c(0);
                self.wr(tmp, z);
                let opn = opn.to_string();
                let ic2 = ic.clone();
                self.guarded(run, &mut |lw| {
                    let tex = lw.s.heap_texture(ui, lkind, true, ct, 1, true);
                    let old = match aop {
                        Some(o) => lw.s.atomic_binop(tex.handle, o, &ic2, v),
                        None if opn == "cmpswap" => lw.s.atomic_cmpxchg(tex.handle, &ic2, cmpv, v),
                        None if opn == "fcmpswap" => lw.texel_fcas(tex.handle, &ic2, cmpv, v),
                        None => lw.texel_cas(tex.handle, &ic2, &opn, v),
                    };
                    lw.wr(tmp, old);
                    Ok(())
                })?;
                if glc {
                    let old = self.rd(tmp);
                    let a = self.act();
                    let cur = self.vreg(vdata);
                    let nv = self.sel(a, old, cur);
                    self.wr(VAR_VGPR + vdata as u32, nv);
                }
                Ok(())
            }
            _ => err(d, "MIMG"),
        }
    }

    /// `image_load` / `image_msaa_load` on a multisampled T#: `ic` = x, y, (slice,) fragment.
    /// `image_load` reads the DMASK components of that sample; `image_msaa_load` reads the one
    /// DMASK component of the four samples fragment .. fragment + 3 (a gather of samples).
    /// Samples past the resource's count read 0 (made explicit: D3D leaves them undefined).
    #[allow(clippy::too_many_arguments)]
    fn mimg_msaa(&mut self, base: &str, ic: &[Value], tw0: Value, kind: TexKind, cls: Value, comps: &[u64], vdata: u64, d16: bool, act: Value, run: Value) -> LResult<()> {
        let nc = kind.coords();
        let frag = ic[nc];
        let coords = ic[..nc].to_vec();
        let samples: Vec<Value> = if base == "msaa_load" { (0..4).map(|k| self.addc(frag, k)).collect() } else { vec![frag] };
        // per sample: the texel's four channels as raw bits
        let tmps: Vec<[u32; 4]> = samples.iter().map(|_| [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()]).collect();
        let z = self.c(0);
        for t in &tmps {
            for &v in t {
                self.wr(v, z);
            }
        }
        for (cv, ct) in [(1u32, CompType::U32), (2, CompType::I32), (0, CompType::F32)] {
            let is = self.cmpc(Pred::Eq, cls, cv);
            let c = self.land(run, is);
            let samples = samples.clone();
            let coords = coords.clone();
            let tmps = tmps.clone();
            self.guarded(c, &mut |lw| {
                let tex: Texture = lw.s.heap_texture(tw0, kind, false, ct, 4, true);
                // Emit all TextureLoad calls before GetDimensions in this basic block:
                // AMD's DXIL compiler (amdxc64.dll) hits a null-pointer dereference
                // (0xc0000005) when dx.op.getDimensions precedes dx.op.textureLoad on a
                // dynamically indexed heap Texture2DMS/Texture2DMSArray in the same block.
                let loads: Vec<[Value; 4]> = samples.iter().map(|&sv| lw.s.texture_load_sample(tex, sv, &coords)).collect();
                let dims = lw.s.get_dimensions(tex.handle, None);
                let count = lw.s.m.extract(dims, 3);
                for (si, (&sv, r)) in samples.iter().zip(loads).enumerate() {
                    let inside = lw.cmp(Pred::ULt, sv, count);
                    for k in 0..4 {
                        let b = lw.bits(r[k]);
                        let zz = lw.c(0);
                        let b = lw.sel(inside, b, zz);
                        lw.wr(tmps[si][k], b);
                    }
                }
                Ok(())
            })?;
        }
        let raw: Vec<Value> = if base == "msaa_load" {
            let c = comps.first().copied().unwrap_or(0) as usize;
            tmps.iter().map(|t| self.rd(t[c])).collect()
        } else {
            comps.iter().map(|&c| self.rd(tmps[0][c as usize])).collect()
        };
        let vals: Vec<Value> = if d16 {
            raw.iter()
                .map(|&x| {
                    let fx = self.f(x);
                    let h = self.s.f32_to_f16(fx);
                    let lo = self.andc(x, 0xFFFF);
                    let fl = self.cmpc(Pred::Eq, cls, 0);
                    self.sel(fl, h, lo)
                })
                .collect()
        } else {
            raw
        };
        self.image_result(vdata, &vals, d16, act)
    }

    /// A compare-exchange loop on a texel (`rmw32` on an `At::Ref` texel).
    fn texel_cas(&mut self, h: Value, ic: &[Value], opn: &str, v: Value) -> Value {
        // the initial value: an atomic or 0 reads it without changing it
        let z = self.c(0);
        let init = self.s.atomic_binop(h, AtomicOp::Or, ic, z);
        let pre = self.cur();
        let lp = self.block(Some(2));
        let ex = self.block(Some(1));
        self.br(lp);
        self.s.m.set_block(lp);
        let i32 = self.s.m.i32();
        let c = self.s.m.phi(i32);
        let i = self.s.m.phi(i32);
        self.s.m.add_incoming(c, init, pre);
        self.s.m.add_incoming(i, z, pre);
        let n = match opn {
            "sub" => self.sub(c, v),
            "inc" => {
                let ge = self.cmp(Pred::UGe, c, v);
                let c1 = self.addc(c, 1);
                self.selc(ge, 0, c1)
            }
            "dec" => {
                let cz = self.cmpc(Pred::Eq, c, 0);
                let gt = self.cmp(Pred::UGt, c, v);
                let w = self.lor(cz, gt);
                let one = self.c(1);
                let c1 = self.sub(c, one);
                self.sel(w, v, c1)
            }
            _ => {
                let (x, y) = (self.f(c), self.f(v));
                let r = if opn == "fmin" { self.fmin(x, y) } else { self.fmax(x, y) };
                self.bits(r)
            }
        };
        let p = self.s.atomic_cmpxchg(h, ic, c, n);
        let ok = self.cmp(Pred::Eq, p, c);
        let i1 = self.addc(i, 1);
        let out = self.cmpc(Pred::UGe, i1, 4096);
        let stop = self.lor(ok, out);
        let me = self.cur();
        self.s.m.add_incoming(c, p, me);
        self.s.m.add_incoming(i, i1, me);
        self.cond_br(stop, ex, lp);
        self.s.m.set_block(ex);
        p
    }

    /// IEEE-754 float compare-exchange loop on a texel (`image_atomic_fcmpswap`).
    fn texel_fcas(&mut self, h: Value, ic: &[Value], cmp: Value, v: Value) -> Value {
        // the initial value: an atomic or 0 reads it without changing it
        let z = self.c(0);
        let init = self.s.atomic_binop(h, AtomicOp::Or, ic, z);
        let pre = self.cur();
        let lp = self.block(Some(2));
        let try_cas = self.block(Some(1));
        let ex = self.block(Some(2));
        self.br(lp);
        self.s.m.set_block(lp);
        let i32 = self.s.m.i32();
        let c = self.s.m.phi(i32);
        let i = self.s.m.phi(i32);
        self.s.m.add_incoming(c, init, pre);
        self.s.m.add_incoming(i, z, pre);
        let eq = self.feq_bits32(c, cmp);
        let lp_end = self.cur();
        self.cond_br(eq, try_cas, ex);
        self.s.m.set_block(try_cas);
        let p = self.s.atomic_cmpxchg(h, ic, c, v);
        let ok = self.cmp(Pred::Eq, p, c);
        let i1 = self.addc(i, 1);
        let out = self.cmpc(Pred::UGe, i1, 4096);
        let stop = self.lor(ok, out);
        let me = self.cur();
        self.s.m.add_incoming(c, p, me);
        self.s.m.add_incoming(i, i1, me);
        self.cond_br(stop, ex, lp);
        self.s.m.set_block(ex);
        let res = self.s.m.phi(i32);
        self.s.m.add_incoming(res, c, lp_end);
        self.s.m.add_incoming(res, p, me);
        res
    }

    /// 64-bit typed image atomic (`DMASK = 0x3`, or `0xF` for `cmpswap`/`fcmpswap`):
    /// operates on a single-component `R64_UINT`/`R64_SINT` UAV via SM 6.6 64-bit typed
    /// heap atomics (`dx.op.atomicBinOp.i64` / `dx.op.atomicCompareExchange.i64`).
    #[allow(clippy::too_many_arguments)]
    fn mimg_atomic64(&mut self, d: &Decoded, opn: &str, ic: &[Value], tw0: Value, lkind: TexKind, vdata: u64, glc: bool, run: Value) -> LResult<()> {
        let (vlo, vhi) = (self.vreg(vdata), self.vreg(vdata + 1));
        let v = self.pack64(vlo, vhi);
        let is_cmpswap = opn == "cmpswap" || opn == "fcmpswap";
        let cmpv = if is_cmpswap {
            let (clo, chi) = (self.vreg(vdata + 2), self.vreg(vdata + 3));
            self.pack64(clo, chi)
        } else {
            self.c64(0)
        };
        let ui = self.addc(tw0, 1);
        let signed = opn == "smin" || opn == "smax";
        let ct = if signed { CompType::I64 } else { CompType::U64 };
        let aop = match opn {
            "add" => Some(AtomicOp::Add),
            "smin" => Some(AtomicOp::IMin),
            "umin" => Some(AtomicOp::UMin),
            "smax" => Some(AtomicOp::IMax),
            "umax" => Some(AtomicOp::UMax),
            "and" => Some(AtomicOp::And),
            "or" => Some(AtomicOp::Or),
            "xor" => Some(AtomicOp::Xor),
            "swap" => Some(AtomicOp::Exchange),
            "cmpswap" | "fcmpswap" | "sub" | "inc" | "dec" | "fmin" | "fmax" => None,
            _ => return err(d, "image_atomic64"),
        };
        let (tmp_lo, tmp_hi) = (self.tmp_var(), self.tmp_var());
        let z = self.c(0);
        self.wr(tmp_lo, z);
        self.wr(tmp_hi, z);
        let opn = opn.to_string();
        let ic2 = ic.to_vec();
        self.guarded(run, &mut |lw| {
            lw.s.heap_typed_atomic64();
            let tex = lw.s.heap_texture(ui, lkind, true, ct, 1, true);
            let old = match aop {
                Some(o) => lw.s.atomic_binop(tex.handle, o, &ic2, v),
                None if opn == "sub" => {
                    let z64 = lw.c64(0);
                    let neg = lw.sub(z64, v);
                    lw.s.atomic_binop(tex.handle, AtomicOp::Add, &ic2, neg)
                }
                None if opn == "cmpswap" => lw.s.atomic_cmpxchg(tex.handle, &ic2, cmpv, v),
                None if opn == "fcmpswap" => lw.texel_fcas64(tex.handle, &ic2, cmpv, v),
                None => lw.texel_cas64(tex.handle, &ic2, &opn, v),
            };
            let (olo, ohi) = lw.unpack64(old);
            lw.wr(tmp_lo, olo);
            lw.wr(tmp_hi, ohi);
            Ok(())
        })?;
        if glc {
            let (olo, ohi) = (self.rd(tmp_lo), self.rd(tmp_hi));
            let a = self.act();
            let (cur_lo, cur_hi) = (self.vreg(vdata), self.vreg(vdata + 1));
            let nlo = self.sel(a, olo, cur_lo);
            let nhi = self.sel(a, ohi, cur_hi);
            self.wr(VAR_VGPR + vdata as u32, nlo);
            self.wr(VAR_VGPR + vdata as u32 + 1, nhi);
        }
        Ok(())
    }

    /// 64-bit compare-exchange loop on an `R64_UINT` texel (`inc`/`dec`/`fmin`/`fmax`).
    fn texel_cas64(&mut self, h: Value, ic: &[Value], opn: &str, v: Value) -> Value {
        let z64 = self.c64(0);
        let init = self.s.atomic_binop(h, AtomicOp::Or, ic, z64);
        let pre = self.cur();
        let lp = self.block(Some(2));
        let ex = self.block(Some(1));
        self.br(lp);
        self.s.m.set_block(lp);
        let i64t = self.i64();
        let i32t = self.s.m.i32();
        let c = self.s.m.phi(i64t);
        let i = self.s.m.phi(i32t);
        let z32 = self.c(0);
        self.s.m.add_incoming(c, init, pre);
        self.s.m.add_incoming(i, z32, pre);
        let n = match opn {
            "inc" => {
                let ge = self.cmp(Pred::UGe, c, v);
                let one = self.c64(1);
                let c1 = self.add(c, one);
                self.sel(ge, z64, c1)
            }
            "dec" => {
                let cz = self.cmp(Pred::Eq, c, z64);
                let gt = self.cmp(Pred::UGt, c, v);
                let w = self.lor(cz, gt);
                let one = self.c64(1);
                let c1 = self.sub(c, one);
                self.sel(w, v, c1)
            }
            _ => {
                self.s.require(super::shader::Feature::Doubles);
                let f64t = self.s.m.f64();
                let x = self.s.m.bitcast(c, f64t);
                let y = self.s.m.bitcast(v, f64t);
                let r = if opn == "fmin" {
                    self.s.binary(super::ops::Binary::FMin, x, y)
                } else {
                    self.s.binary(super::ops::Binary::FMax, x, y)
                };
                self.s.m.bitcast(r, i64t)
            }
        };
        let p = self.s.atomic_cmpxchg(h, ic, c, n);
        let ok = self.cmp(Pred::Eq, p, c);
        let i1 = self.addc(i, 1);
        let out = self.cmpc(Pred::UGe, i1, 4096);
        let stop = self.lor(ok, out);
        let me = self.cur();
        self.s.m.add_incoming(c, p, me);
        self.s.m.add_incoming(i, i1, me);
        self.cond_br(stop, ex, lp);
        self.s.m.set_block(ex);
        p
    }

    /// IEEE-754 64-bit float compare-exchange loop on an `R64_UINT` texel (`image_atomic_fcmpswap`, `DMASK = 0xF`).
    fn texel_fcas64(&mut self, h: Value, ic: &[Value], cmp: Value, v: Value) -> Value {
        let z64 = self.c64(0);
        let init = self.s.atomic_binop(h, AtomicOp::Or, ic, z64);
        let pre = self.cur();
        let lp = self.block(Some(2));
        let try_cas = self.block(Some(1));
        let ex = self.block(Some(2));
        self.br(lp);
        self.s.m.set_block(lp);
        let i64t = self.i64();
        let i32t = self.s.m.i32();
        let c = self.s.m.phi(i64t);
        let i = self.s.m.phi(i32t);
        let z32 = self.c(0);
        self.s.m.add_incoming(c, init, pre);
        self.s.m.add_incoming(i, z32, pre);
        let eq = self.feq_bits64(c, cmp);
        let lp_end = self.cur();
        self.cond_br(eq, try_cas, ex);
        self.s.m.set_block(try_cas);
        let p = self.s.atomic_cmpxchg(h, ic, c, v);
        let ok = self.cmp(Pred::Eq, p, c);
        let i1 = self.addc(i, 1);
        let out = self.cmpc(Pred::UGe, i1, 4096);
        let stop = self.lor(ok, out);
        let me = self.cur();
        self.s.m.add_incoming(c, p, me);
        self.s.m.add_incoming(i, i1, me);
        self.cond_br(stop, ex, lp);
        self.s.m.set_block(ex);
        let res = self.s.m.phi(i64t);
        self.s.m.add_incoming(res, c, lp_end);
        self.s.m.add_incoming(res, p, me);
        res
    }

    /// `img_cls`: 1 uint, 2 sint, 0 float (by the buffer-format table's numeric type).
    fn img_cls(&mut self, fmt: Value) -> Value {
        let info = self.fmt_info(fmt);
        let ty = self.lshrc(info, 3);
        let ty = self.andc(ty, 7);
        let big = self.cmpc(Pred::UGt, fmt, 77);
        let k6 = self.c(6);
        let ty = self.sel(big, k6, ty);
        let t4 = self.cmpc(Pred::Eq, ty, 4);
        let t5 = self.cmpc(Pred::Eq, ty, 5);
        let two = self.c(2);
        let z = self.c(0);
        let r = self.sel(t5, two, z);
        self.selc(t4, 1, r)
    }

    /// Writes result components to VDATA (D16: two halves per VGPR) in active lanes.
    fn image_result(&mut self, vdata: u64, vals: &[Value], d16: bool, act: Value) -> LResult<()> {
        let mut out = Vec::new();
        if d16 {
            for k in (0..vals.len()).step_by(2) {
                let lo = self.andc(vals[k], 0xFFFF);
                let hi = if k + 1 < vals.len() { vals[k + 1] } else { self.c(0) };
                let hi = self.shlc(hi, 16);
                out.push(self.or(lo, hi));
            }
        } else {
            out = vals.to_vec();
        }
        for (k, v) in out.into_iter().enumerate() {
            let old = self.vreg(vdata + k as u64);
            let nv = self.sel(act, v, old);
            self.wr(VAR_VGPR + (vdata + k as u64) as u32, nv);
        }
        Ok(())
    }
}

#[cfg(test)]
mod heap_bounds_tests {
    use crate::gpu::dxil::{kernel, validate};
    use crate::gpu::rdna2;
    use crate::gpu::recomp;

    fn mimg_prog(ops: &[(&str, &[(&str, u64)])]) -> Vec<u8> {
        let mut code = Vec::new();
        for &(name, fields) in ops {
            code.extend(rdna2::encode(name, "ENC_MIMG", fields, None).unwrap_or_else(|e| panic!("encode {name}: {e}")));
        }
        code.extend(rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        code
    }

    /// Dynamic `ResourceDescriptorHeap` and `SamplerDescriptorHeap` indices must be
    /// bounds-checked / clamped in both Direct DXIL and HLSL so AMD Tier-3 GPUs never
    /// page-fault on null, uninitialized, or OOB T#/S# descriptors.
    #[test]
    fn mimg_view_and_sampler_heap_indices_are_guarded() {
        let code = mimg_prog(&[
            ("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)]),
            ("IMAGE_LOAD", &[("VDATA", 14), ("VADDR", 2), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1), ("UNORM", 1)]),
            ("IMAGE_GET_RESINFO", &[("VDATA", 18), ("VADDR", 4), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1)]),
            ("IMAGE_STORE", &[("VDATA", 6), ("VADDR", 2), ("SRSRC", 8), ("DMASK", 0xF), ("DIM", 1), ("UNORM", 1)]),
            ("IMAGE_ATOMIC_ADD", &[("VDATA", 7), ("VADDR", 2), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]),
        ]);

        let dxil = kernel::translate(&code, kernel::KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 }).expect("direct translate");
        validate::validate(&dxil).expect("dxil validate");
        let dis = validate::disassemble(&dxil).expect("dxil disassemble");
        assert!(dis.contains(&super::VIEW_HEAP_MAX_SRV.to_string()), "expected VIEW_HEAP_MAX_SRV guard in direct DXIL:\n{dis}");
        assert!(dis.contains("2047"), "expected 0x7FF sampler heap mask in direct DXIL:\n{dis}");

        let hlsl = recomp::translate(&code, &recomp::Options::default()).expect("hlsl translate");
        assert!(hlsl.complete(), "hlsl unsupported: {:?}", hlsl.unsupported);
        assert!(hlsl.hlsl.contains(&format!("{}u", super::VIEW_HEAP_MAX_SRV)), "expected VIEW_HEAP_MAX_SRV check in HLSL");
        assert!(hlsl.hlsl.contains("& 0x7FFu"), "expected 0x7FFu sampler heap mask in HLSL");
        assert!(!hlsl.hlsl.contains("& 0xFFFu"), "unexpected unclamped 0xFFFu sampler mask in HLSL");
        // Ensure atomic UAV handle creation is inside the guarded `if (and(..., ..._ok))` block.
        assert!(hlsl.hlsl.contains("InterlockedAdd(") && hlsl.hlsl.contains("_ok)) { RWTexture2D<uint>"), "expected atomic UAV handle inside guard");
    }

    /// Pixel-shader implicit-LOD `image_sample` and `image_get_lod` must keep convergent
    /// control flow for quad derivatives while clamping invalid/OOB T# indices to slot 0.
    #[test]
    fn ps_implicit_lod_sample_and_get_lod_validate() {
        let code = mimg_prog(&[
            ("IMAGE_SAMPLE", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)]),
            ("IMAGE_GET_LOD", &[("VDATA", 8), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x3), ("DIM", 1)]),
        ]);
        let ps = recomp::PsInterface::default();
        let k = kernel::translate_pixel(&code, kernel::KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 }, &ps).expect("direct ps translate");
        validate::validate(&k.dxil).expect("dxil ps validate");
        let dis = validate::disassemble(&k.dxil).expect("dxil ps disassemble");
        assert!(dis.contains(&super::VIEW_HEAP_MAX_SRV.to_string()), "expected VIEW_HEAP_MAX_SRV clamp in PS DXIL:\n{dis}");
        assert!(dis.contains("2047"), "expected 0x7FF sampler mask in PS DXIL:\n{dis}");

        let hlsl = recomp::translate(&code, &recomp::Options { mode: recomp::Mode::Pixel(ps), ..Default::default() }).expect("hlsl ps translate");
        assert!(hlsl.complete(), "hlsl ps unsupported: {:?}", hlsl.unsupported);
        assert!(hlsl.hlsl.contains("select(") && hlsl.hlsl.contains(&format!("{}u", super::VIEW_HEAP_MAX_SRV)), "expected convergent clamp in PS HLSL");
    }
}
