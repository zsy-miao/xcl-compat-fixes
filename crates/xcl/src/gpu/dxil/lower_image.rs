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

impl Lw {
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
    fn load3f(&mut self, at: Value) -> [Value; 3] {
        let mut r = [at; 3];
        for (k, slot) in r.iter_mut().enumerate() {
            let a = self.addc(at, 4 * k as u32);
            let v = self.mem_ld(a);
            *slot = self.f(v);
        }
        r
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
                let mut child = [ones; 4];
                for (k, c) in child.iter_mut().enumerate() {
                    let a = lw.addc(at, 4 * k as u32);
                    *c = lw.mem_ld(a);
                }
                let ninf = lw.s.m.f32c(0xFF80_0000);
                let pinf = lw.s.m.f32c(0x7F80_0000);
                let mut tn = [pinf; 4];
                let mut id = [ones; 4];
                for k in 0..4u32 {
                    let (mn, mx) = if box32 {
                        let a = lw.addc(at, 16 + 24 * k);
                        let b = lw.addc(at, 28 + 24 * k);
                        (lw.load3f(a), lw.load3f(b))
                    } else {
                        let a = lw.addc(at, 16 + 12 * k);
                        let mut h = [ones; 3];
                        for (j, slot) in h.iter_mut().enumerate() {
                            let x = lw.addc(a, 4 * j as u32);
                            *slot = lw.mem_ld(x);
                        }
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
                    let cv = lw.cmpc(Pred::Ne, child[k as usize], u32::MAX);
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
                    tn[k as usize] = lw.sel(hit, a, pinf);
                    id[k as usize] = lw.sel(hit, child[k as usize], ones);
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
            let t1 = lw.cmpc(Pred::Eq, ty, 1);
            let k0 = lw.c(0);
            let a0 = lw.selc(t1, 12, k0);
            let k36 = lw.c(36);
            let k12 = lw.c(12);
            let a1 = lw.sel(t1, k36, k12);
            let p0 = lw.add(at, a0);
            let p1 = lw.add(at, a1);
            let p2 = lw.addc(at, 24);
            let v0 = lw.load3f(p0);
            let v1 = lw.load3f(p1);
            let v2 = lw.load3f(p2);
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
            let p60 = lw.addc(at, 60);
            let tw = lw.mem_ld(p60);
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
        let act = self.act();
        let comps: Vec<u64> = (0..4).filter(|c| dmask >> c & 1 == 1).collect();
        let kind = match dim {
            0 => TexKind::D1,
            1 => TexKind::D2,
            2 => TexKind::D3,
            // cube T#s are cube arrays (face + 8 · cube), as the HLSL path's TextureCubeArray
            3 => TexKind::CubeArray,
            4 => TexKind::D1Array,
            5 => TexKind::D2Array,
            _ => return err(d, "MIMG dim"),
        };

        if sampling {
            if f("UNORM") != 0 && dim != 3 {
                return err(d, "MIMG unorm");
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
            let mut coords: Vec<Value> = (0..body as usize).map(|k| compf(self, i_body + k)).collect();
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
            let spatial: Vec<Value> = coords[..kind.grads().min(coords.len())].to_vec();
            let tex = self.s.heap_texture(tw0, kind, false, CompType::F32, 4, true);
            let smp: Sampler = match self.static_samplers.get(&d.pc).copied() {
                // the console's static samplers: bound at s<i>, space1 in order of first use
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
                    let si = self.andc(s3, 0xFFF);
                    self.s.heap_sampler(si, has_c, true)
                }
            };
            if let Some(o) = ovgpr {
                // texel offset / size of the sampled mip (0 for lz / gather; else the LOD given, else 0)
                let level = match (&lod, has_lz || gather) {
                    (_, true) => self.c(0),
                    (Some(l), _) => {
                        let r = self.un(op::ROUND_NE, *l);
                        let z = self.fc(0.0);
                        let r = self.fmax(r, z);
                        self.ftoui(r)
                    }
                    // the implicit LOD of a pixel shader's sample, rounded
                    (None, false) if ps && grads.is_none() && !has_c => {
                        let l = self.s.calculate_lod(tex, smp, &spatial, true);
                        let r = self.un(op::ROUND_NE, l);
                        let z = self.fc(0.0);
                        let r = self.fmax(r, z);
                        self.ftoui(r)
                    }
                    _ => self.c(0),
                };
                let dims = self.s.get_dimensions(tex.handle, Some(level));
                let one = self.c(1);
                let (w, hh, dd) = match dim {
                    0 => (self.s.m.extract(dims, 0), one, one),
                    1 => (self.s.m.extract(dims, 0), self.s.m.extract(dims, 1), one),
                    2 => (self.s.m.extract(dims, 0), self.s.m.extract(dims, 1), self.s.m.extract(dims, 2)),
                    4 => (self.s.m.extract(dims, 0), one, one),
                    _ => (self.s.m.extract(dims, 0), self.s.m.extract(dims, 1), one),
                };
                let mut ofs = Vec::new();
                for (sh, size) in [(26u32, w), (18, hh), (10, dd)] {
                    let x = self.shlc(o, sh);
                    let x = self.ashrc(x, 26);
                    let xf = self.sitof(x);
                    let sf = self.uitof(size);
                    ofs.push(self.fdiv(xf, sf));
                }
                for (k, c) in coords.iter_mut().enumerate() {
                    if (dim == 4 && k == 1) || (dim == 5 && k == 2) {
                        continue;
                    }
                    *c = self.fadd(*c, ofs[k]);
                }
            }
            // array layers are rounded; cube coordinates become a direction and a cube index
            let coord: Vec<Value> = match dim {
                3 => {
                    let mut v = self.cube_dir(coords[0], coords[1], coords[2]).to_vec();
                    v.push(self.cube_slice(coords[2]));
                    v
                }
                4 => vec![coords[0], self.un(op::ROUND_NE, coords[1])],
                5 => vec![coords[0], coords[1], self.un(op::ROUND_NE, coords[2])],
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
                    Some(z) => self.s.gather_cmp(tex, smp, &coord, &[], comps[0] as u32, z),
                    None => self.s.gather(tex, smp, &coord, &[], comps[0] as u32),
                }
            } else {
                match (&cmp, &grads, &lod, &bias) {
                    (Some(z), _, _, _) if has_lz || !ps => {
                        let v = self.s.sample_cmp_level_zero(tex, smp, &coord, &[], *z);
                        [v[0]; 4]
                    }
                    (Some(z), _, _, _) => {
                        let v = self.s.sample_cmp(tex, smp, &coord, &[], *z, if dim == 3 { None } else { cl });
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
                        let z = self.c(0);
                        let dims = self.s.get_dimensions(tex.handle, Some(z));
                        let size = self.s.m.extract(dims, 0);
                        let lx = len(self, dx);
                        let ly = len(self, dy);
                        let m = self.fmax(lx, ly);
                        let sf = self.uitof(size);
                        let m = self.fmul(m, sf);
                        let eps = self.fc(1e-8);
                        let m = self.fmax(m, eps);
                        let l = self.un(op::LOG, m);
                        let l = floor(self, l);
                        self.s.sample_level(tex, smp, &coord, &[], l)
                    }
                    (None, Some((dx, dy)), _, _) => self.s.sample_grad(tex, smp, &coord, &[], dx, dy, cl),
                    (None, None, Some(l), _) => self.s.sample_level(tex, smp, &coord, &[], *l),
                    (None, None, None, Some(b)) if ps => self.s.sample_bias(tex, smp, &coord, &[], *b, if dim == 3 { None } else { cl }),
                    (None, None, None, Some(b)) => {
                        let l = floor(self, *b);
                        self.s.sample_level(tex, smp, &coord, &[], l)
                    }
                    _ if has_lz || !ps => {
                        let z = self.fc(0.0);
                        let l = floor(self, z);
                        self.s.sample_level(tex, smp, &coord, &[], l)
                    }
                    _ => self.s.sample(tex, smp, &coord, &[], if dim == 3 { None } else { cl }),
                }
            };
            let rb: Vec<Value> = r.iter().map(|&v| self.bits(v)).collect();
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
            if self.static_samplers.contains_key(&d.pc) {
                return err(d, "image_get_lod with a static sampler");
            }
            let coords: Vec<Value> = (0..body as usize).map(|k| compf(self, i_body + k)).collect();
            let spatial: Vec<Value> = match dim {
                3 => self.cube_dir(coords[0], coords[1], coords[2]).to_vec(),
                _ => coords[..kind.grads().min(coords.len())].to_vec(),
            };
            let s3 = self.sreg(ssamp + 3);
            let tex = self.s.heap_texture(tw0, kind, false, CompType::F32, 4, true);
            let si = self.andc(s3, 0xFFF);
            let smp = self.s.heap_sampler(si, false, true);
            let a = self.s.calculate_lod(tex, smp, &spatial, true);
            let b = self.s.calculate_lod(tex, smp, &spatial, false);
            let z = self.c(0);
            let r = [self.bits(a), self.bits(b), z, z];
            let vals: Vec<Value> = comps.iter().map(|&c| if d16 {
                let f = self.f(r[c as usize]);
                self.s.f32_to_f16(f)
            } else {
                r[c as usize]
            }).collect();
            return self.image_result(vdata, &vals, d16, act);
        }

        if base == "get_resinfo" {
            let mipv = compu(self, 0);
            let tex = self.s.heap_texture(tw0, kind, false, CompType::F32, 4, true);
            let dims = self.s.get_dimensions(tex.handle, Some(mipv));
            let one = self.c(1);
            let e = |lw: &mut Lw, i: u32| lw.s.m.extract(dims, i);
            // HLSL GetDimensions outputs per dim; missing ones are 1
            let (w, hh, dd, n) = match dim {
                0 => (e(self, 0), one, one, e(self, 3)),
                1 => (e(self, 0), e(self, 1), one, e(self, 3)),
                // a cube array: its layers are faces, 6 per cube
                3 => {
                    let n = e(self, 2);
                    {
                        let k = self.c(6);
                        (e(self, 0), e(self, 1), self.mul(n, k), e(self, 3))
                    }
                }
                2 | 5 => (e(self, 0), e(self, 1), e(self, 2), e(self, 3)),
                _ => (e(self, 0), one, e(self, 1), e(self, 3)),
            };
            let r = [w, hh, dd, n];
            let vals: Vec<Value> = comps.iter().map(|&c| r[c as usize]).collect();
            return self.image_result(vdata, &vals, false, act);
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
        match base {
            "load" | "load_mip" => {
                let tmps = [self.tmp_var(), self.tmp_var(), self.tmp_var(), self.tmp_var()];
                let z = self.c(0);
                for &t in &tmps {
                    self.wr(t, z);
                }
                for (cv, ct) in [(1u32, CompType::U32), (2, CompType::I32), (0, CompType::F32)] {
                    let is = self.cmpc(Pred::Eq, cls, cv);
                    self.guarded(is, &mut |lw| {
                        let tex: Texture = lw.s.heap_texture(tw0, lkind, false, ct, 4, true);
                        let r = lw.s.texture_load(tex, Some(mip), &ic, &[]);
                        for k in 0..4 {
                            let b = lw.bits(r[k]);
                            lw.wr(tmps[k], b);
                        }
                        Ok(())
                    })?;
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
                    let c = self.land(act, is);
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
                let (v, cmpv) = (self.vreg(vdata), self.vreg(vdata + 1));
                let glc = f("GLC") == 1;
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
                self.guarded(act, &mut |lw| {
                    let tex = lw.s.heap_texture(ui, lkind, true, ct, 1, true);
                    let old = match aop {
                        Some(o) => lw.s.atomic_binop(tex.handle, o, &ic2, v),
                        None if opn == "cmpswap" || opn == "fcmpswap" => lw.s.atomic_cmpxchg(tex.handle, &ic2, cmpv, v),
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
