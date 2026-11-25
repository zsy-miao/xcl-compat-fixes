//! The recompiler's HLSL helper functions (`recomp/emit.rs` `PRELUDE`) as
//! DXIL builders, operation for operation (the HLSL intrinsics as DXC lowers
//! them), so both backends compute the same bits.

use super::lower::Lw;
use super::module::{BinOp, Pred, Value};
use super::ops_core::op;

impl Lw {
    /// `cvt_u32`: NaN / ≤ 0 → 0, ≥ 2^32 → 0xFFFFFFFF.
    pub fn cvt_u32(&mut self, f: Value) -> Value {
        let n = self.isnan(f);
        let z = self.fc(0.0);
        let le = self.fcmp(Pred::FOLe, f, z);
        let low = self.lor(n, le);
        let big = self.fc(4294967296.0);
        let ge = self.fcmp(Pred::FOGe, f, big);
        let v = self.ftoui(f);
        let v = self.selc(ge, u32::MAX, v);
        self.selc(low, 0, v)
    }
    /// `cvt_i32`: NaN → 0, saturating.
    pub fn cvt_i32(&mut self, f: Value) -> Value {
        let n = self.isnan(f);
        let lo = self.fc(-2147483648.0);
        let hi = self.fc(2147483648.0);
        let le = self.fcmp(Pred::FOLe, f, lo);
        let ge = self.fcmp(Pred::FOGe, f, hi);
        let v = self.ftosi(f);
        let v = self.selc(ge, 0x7FFF_FFFF, v);
        let v = self.selc(le, 0x8000_0000, v);
        self.selc(n, 0, v)
    }
    /// `f32tof16(e) & 0xFFFF`
    /// The recompiler's `to_f16_expr`: `f32tof16(e) & 0xFFFF`, or under a program-set
    /// MODE[3:2] the software conversion with the current mode.
    pub fn to_f16(&mut self, e: Value) -> Value {
        if self.dyn_rm {
            return self.to_f16_rounded(e);
        }
        let h = self.s.f32_to_f16(e);
        self.andc(h, 0xFFFF)
    }
    /// Round an exact binary32 input to binary16 in the selected RDNA mode
    /// (`recomp/emit.rs` `to_f16_rounded_expr`). Legacy `f32tof16` uses
    /// round-toward-zero; the default RDNA mode is round-to-nearest-even.
    pub fn to_f16_rounded(&mut self, e: Value) -> Value {
        if self.dyn_rm {
            let rm = self.rd(super::lower::VAR_FP16_RM);
            self.f32_to_f16_rm(e, rm)
        } else {
            self.f32_to_f16_rne(e)
        }
    }
    /// `f16_rtz`: binary16 rounded toward zero.
    pub fn f16_rtz(&mut self, f: Value) -> Value {
        let h = self.s.f32_to_f16(f);
        let h = self.andc(h, 0xFFFF);
        let n = self.isnan(f);
        let nn = self.lnot(n);
        let mag = self.andc(h, 0x7FFF);
        let nz = self.nz(mag);
        let back = self.s.f16_to_f32(h);
        let ab = self.fabs(back);
        let af = self.fabs(f);
        let gt = self.fcmp(Pred::FOGt, ab, af);
        let c = self.land(nn, nz);
        let c = self.land(c, gt);
        let one = self.u(c);
        self.sub(h, one)
    }
    /// `fract_f`: x - floor(x), below 1.0.
    pub fn fract_f(&mut self, x: Value) -> Value {
        let fl = self.un(op::ROUND_NI, x);
        let r = self.fsub(x, fl);
        let one = self.fc(1.0);
        let ge = self.fcmp(Pred::FOGe, r, one);
        let k = self.s.m.f32c(0x3F7F_FFFF);
        self.sel(ge, k, r)
    }
    /// HLSL `frexp` as DXC lowers it: (mantissa, exponent as float).
    fn frexp_dxc(&mut self, x: Value) -> (Value, Value) {
        let i = self.bits(x);
        let e = self.andc(i, 0x7F80_0000);
        let mant = self.andc(i, 0x007F_FFFF); // DXC drops the sign
        let e = self.addc(e, (-0x3F00_0000i32) as u32);
        let e = self.ashrc(e, 23);
        let mant = self.orc(mant, 0x3F00_0000);
        let z = self.fc(0.0);
        let nz = self.fcmp(Pred::FUNe, x, z);
        let ef = self.sitof(e);
        let ef = self.sel(nz, ef, z);
        let mf = self.f(mant);
        let mf = self.sel(nz, mf, z);
        (mf, ef)
    }
    pub fn frexp_mant(&mut self, x: Value) -> Value {
        let (m, _) = self.frexp_dxc(x);
        let a = self.isinf(x);
        let b = self.isnan(x);
        let s = self.lor(a, b);
        self.sel(s, x, m)
    }
    pub fn frexp_exp(&mut self, x: Value) -> Value {
        let (_, e) = self.frexp_dxc(x);
        let a = self.isinf(x);
        let b = self.isnan(x);
        let s = self.lor(a, b);
        let ei = self.ftosi(e);
        self.selc(s, 0, ei)
    }
    pub fn frexp_exp_f32(&mut self, bits: Value, legacy_src: impl FnOnce(&mut Self) -> Value) -> Value {
        use super::lower_round::{F32_DENORM_INPUT_WRITTEN, VAR_F32_MODE};
        let mode = if self.track_frexp_denorm { self.rd(VAR_F32_MODE) } else { self.c(0) };
        let written = self.andc(mode, F32_DENORM_INPUT_WRITTEN);
        let known = self.nz(written);
        let legacy = |lw: &mut Self| {
            let x = legacy_src(lw);
            lw.frexp_exp(x)
        };
        match self.s.m.const_u64(known) {
            Some(0) => legacy(self),
            Some(1) => self.frexp_exp_f32_bits(bits, mode),
            _ => {
                let result = self.frexp_exp_f32_bits(bits, mode);
                let old = legacy(self);
                self.sel(known, result, old)
            }
        }
    }

    fn frexp_exp_f32_bits(&mut self, bits: Value, mode: Value) -> Value {
        let e8 = self.lshrc(bits, 23);
        let exp = self.andc(e8, 0xFF);
        let fraction = self.andc(bits, 0x7F_FFFF);
        let normal = self.addc(exp, (-126i32) as u32);
        // firstbithigh(0) returns -1; a zero fraction always produces 0 below.
        let top = self.firstbithigh(fraction);
        let subnormal = self.addc(top, (-148i32) as u32);
        let empty = self.cmpc(Pred::Eq, fraction, 0);
        let keep = self.andc(mode, 0x10);
        let flush = self.cmpc(Pred::Eq, keep, 0);
        let zero = self.lor(empty, flush);
        let low = self.selc(zero, 0, subnormal);
        let no_exp = self.cmpc(Pred::Eq, exp, 0);
        let finite = self.sel(no_exp, low, normal);
        let special = self.cmpc(Pred::Eq, exp, 0xFF);
        self.selc(special, 0, finite)
    }
    /// `ldexp_f`: exact x · 2^n on the bits.
    pub fn ldexp_f(&mut self, x: Value, n: Value) -> Value {
        let u = self.bits(x);
        let sign = self.andc(u, 0x8000_0000);
        let e8 = self.lshrc(u, 23);
        let exp = self.andc(e8, 0xFF);
        let mant = self.andc(u, 0x7F_FFFF);
        // exp == 0xFF or (exp == 0 and mant == 0): x
        let special = self.cmpc(Pred::Eq, exp, 0xFF);
        let ez = self.cmpc(Pred::Eq, exp, 0);
        let mz = self.cmpc(Pred::Eq, mant, 0);
        let zero = self.land(ez, mz);
        let keep = self.lor(special, zero);
        // denormal: normalise
        let top = self.firstbithigh(mant);
        let k23 = self.c(23);
        let sh = self.sub(k23, top);
        let nm = self.shl(mant, sh);
        let nm = self.andc(nm, 0x7F_FFFF);
        let k22 = self.c(22);
        let ne = self.sub(top, k22);
        let mant = self.sel(ez, nm, mant);
        let e = self.sel(ez, ne, exp);
        let lo = self.c(-500i32 as u32);
        let hi = self.c(500);
        let nc = self.imax(n, lo);
        let nc = self.imin(nc, hi);
        let eo = self.add(e, nc);
        // eo >= 255: inf
        let inf = self.cmpc(Pred::SGe, eo, 255);
        let infv = self.orc(sign, 0x7F80_0000);
        // eo >= 1: normal
        let norm = self.cmpc(Pred::SGe, eo, 1);
        let eb = self.shlc(eo, 23);
        let nv = self.or(sign, eb);
        let nv = self.or(nv, mant);
        // denormal result
        let one = self.c(1);
        let shd = self.sub(one, eo);
        let tiny = self.cmpc(Pred::UGt, shd, 24);
        let sig = self.orc(mant, 0x80_0000);
        let shc = { let k = self.c(31); self.umin(shd, k) };
        let q = self.bin(BinOp::LShr, sig, shc);
        let shm1 = self.sub(shc, one);
        let shm1 = self.andc(shm1, 31);
        let half = self.bin(BinOp::Shl, one, shm1);
        let full = self.bin(BinOp::Shl, one, shc);
        let rm = self.sub(full, one);
        let rem = self.and(sig, rm);
        let gt = self.cmp(Pred::UGt, rem, half);
        let eqh = self.cmp(Pred::Eq, rem, half);
        let odd = self.andc(q, 1);
        let odd = self.nz(odd);
        let tie = self.land(eqh, odd);
        let inc = self.lor(gt, tie);
        let incv = self.u(inc);
        let qd = self.add(q, incv);
        let dv = self.or(sign, qd);
        let dv = self.sel(tiny, sign, dv);
        let r = self.sel(norm, nv, dv);
        let r = self.sel(inf, infv, r);
        let r = self.sel(keep, u, r);
        self.f(r)
    }
    /// `mul_legacy`: 0 · anything = 0.
    pub fn mul_legacy(&mut self, a: Value, b: Value) -> Value {
        let z = self.fc(0.0);
        let az = self.fcmp(Pred::FOEq, a, z);
        let bz = self.fcmp(Pred::FOEq, b, z);
        let c = self.lor(az, bz);
        let p = self.fmul(a, b);
        self.sel(c, z, p)
    }
    /// `mad_nf`: separately rounded multiply and add (`precise`).
    pub fn mad_nf(&mut self, a: Value, b: Value, c: Value) -> Value {
        self.s.require(super::shader::Feature::DisableMathRefactoring);
        let p = self.fmul(a, b);
        self.fadd(p, c)
    }
    /// `fma_legacy`: (a == 0 || b == 0) ? c : mad(a, b, c).
    pub fn fma_legacy(&mut self, a: Value, b: Value, c: Value) -> Value {
        let z = self.fc(0.0);
        let az = self.fcmp(Pred::FOEq, a, z);
        let bz = self.fcmp(Pred::FOEq, b, z);
        let cnd = self.lor(az, bz);
        let m = self.fmad(a, b, c);
        self.sel(cnd, c, m)
    }
    pub fn med3f(&mut self, a: Value, b: Value, c: Value) -> Value {
        let x = self.fmin(a, b);
        let y = self.fmax(a, b);
        let y = self.fmin(y, c);
        self.fmax(x, y)
    }
    /// `div_fixup`
    pub fn div_fixup(&mut self, q: Value, d: Value, n: Value) -> Value {
        let a = self.isnan(d);
        let b = self.isnan(n);
        let z = self.fc(0.0);
        let dz = self.fcmp(Pred::FOEq, d, z);
        let di = self.isinf(d);
        let ni = self.isinf(n);
        let x = self.lor(a, b);
        let y = self.lor(di, ni);
        let y = self.lor(dz, y);
        let c = self.lor(x, y);
        let r = self.fdiv(n, d);
        self.sel(c, r, q)
    }
    /// cube_id / cube_sc / cube_tc / cube_ma
    pub fn cube(&mut self, name: &str, x: Value, y: Value, z: Value) -> Value {
        let (ax, ay, az) = (self.fabs(x), self.fabs(y), self.fabs(z));
        let zx = self.fcmp(Pred::FOGe, az, ax);
        let zy = self.fcmp(Pred::FOGe, az, ay);
        let zmaj = self.land(zx, zy);
        let ymaj = self.fcmp(Pred::FOGe, ay, ax);
        let zero = self.fc(0.0);
        let zneg = self.fcmp(Pred::FOLt, z, zero);
        let yneg = self.fcmp(Pred::FOLt, y, zero);
        let xneg = self.fcmp(Pred::FOLt, x, zero);
        let (zr, yr, xr) = match name {
            "v_cubeid_f32" => {
                let (f5, f4, f3, f2, f1, f0) = (self.fc(5.0), self.fc(4.0), self.fc(3.0), self.fc(2.0), self.fc(1.0), self.fc(0.0));
                (self.sel(zneg, f5, f4), self.sel(yneg, f3, f2), self.sel(xneg, f1, f0))
            }
            "v_cubesc_f32" => {
                let nx = self.fneg(x);
                let nz = self.fneg(z);
                (self.sel(zneg, nx, x), x, self.sel(xneg, z, nz))
            }
            "v_cubetc_f32" => {
                let ny = self.fneg(y);
                let nz = self.fneg(z);
                (ny, self.sel(yneg, nz, z), ny)
            }
            _ => {
                let two = self.fc(2.0);
                (self.fmul(two, z), self.fmul(two, y), self.fmul(two, x))
            }
        };
        let r = self.sel(ymaj, yr, xr);
        self.sel(zmaj, zr, r)
    }
    /// `i24`: sign-extended low 24 bits.
    pub fn i24(&mut self, v: Value) -> Value {
        let s = self.shlc(v, 8);
        self.ashrc(s, 8)
    }
    pub fn alignbit(&mut self, s0: Value, s1: Value, s: Value) -> Value {
        let s = self.andc(s, 31);
        let a = self.bin(BinOp::LShr, s1, s);
        let k = self.c(32);
        let t = self.sub(k, s);
        let t = self.andc(t, 31);
        let b = self.bin(BinOp::Shl, s0, t);
        let r = self.or(a, b);
        let z = self.cmpc(Pred::Eq, s, 0);
        self.sel(z, s1, r)
    }
    pub fn perm_b32(&mut self, s0: Value, s1: Value, sel: Value) -> Value {
        let mut r = self.c(0);
        for i in 0..4u32 {
            let k = self.lshrc(sel, 8 * i);
            let k = self.andc(k, 0xFF);
            // bytes 0..7: s1 bytes 0..3 then s0 bytes 0..3
            let lo4 = self.cmpc(Pred::ULt, k, 4);
            let k3 = self.andc(k, 3);
            let sh = self.shlc(k3, 3);
            let w = self.sel(lo4, s1, s0);
            let byte = self.bin(BinOp::LShr, w, sh);
            let byte = self.andc(byte, 0xFF);
            // 8..11: sign of s1 bit 15 / 31, s0 bit 15 / 31
            let b15 = self.lshrc(s1, 15);
            let b31 = self.lshrc(s1, 31);
            let c15 = self.lshrc(s0, 15);
            let c31 = self.lshrc(s0, 31);
            let is8 = self.cmpc(Pred::Eq, k, 8);
            let is9 = self.cmpc(Pred::Eq, k, 9);
            let is10 = self.cmpc(Pred::Eq, k, 10);
            let t = self.sel(is10, c15, c31);
            let t = self.sel(is9, b31, t);
            let t = self.sel(is8, b15, t);
            let t = self.andc(t, 1);
            let tn = self.nz(t);
            let ff = self.c(0xFF);
            let z = self.c(0);
            let sgn = self.sel(tn, ff, z);
            let ge8 = self.cmpc(Pred::UGe, k, 8);
            let b = self.sel(ge8, sgn, byte);
            let is12 = self.cmpc(Pred::Eq, k, 12);
            let b = self.sel(is12, z, b);
            let ge13 = self.cmpc(Pred::UGe, k, 13);
            let b = self.sel(ge13, ff, b);
            let b = self.shlc(b, 8 * i);
            r = self.or(r, b);
        }
        r
    }
    /// |a[k..] - b[k..]| of one byte / half (`absd8` / `absd16`).
    pub fn absd(&mut self, a: Value, b: Value, k: u32, mask: u32) -> Value {
        let x = self.lshrc(a, k);
        let x = self.andc(x, mask);
        let y = self.lshrc(b, k);
        let y = self.andc(y, mask);
        let hi = self.umax(x, y);
        let lo = self.umin(x, y);
        self.sub(hi, lo)
    }
    pub fn sad_u8(&mut self, a: Value, b: Value) -> Value {
        let mut r = self.absd(a, b, 0, 0xFF);
        for k in [8, 16, 24] {
            let x = self.absd(a, b, k, 0xFF);
            r = self.add(r, x);
        }
        r
    }
    pub fn msad_u8(&mut self, a: Value, b: Value) -> Value {
        let mut r = self.c(0);
        for k in [0, 8, 16, 24] {
            let x = self.absd(a, b, k, 0xFF);
            let bb = self.lshrc(b, k);
            let bb = self.andc(bb, 0xFF);
            let nz = self.nz(bb);
            let z = self.c(0);
            let x = self.sel(nz, x, z);
            r = self.add(r, x);
        }
        r
    }
    pub fn lerp_u8(&mut self, a: Value, b: Value, c: Value) -> Value {
        let mut r = self.c(0);
        for k in [0, 8, 16, 24] {
            let x = self.lshrc(a, k);
            let x = self.andc(x, 0xFF);
            let y = self.lshrc(b, k);
            let y = self.andc(y, 0xFF);
            let z = self.lshrc(c, k);
            let z = self.andc(z, 1);
            let s = self.add(x, y);
            let s = self.add(s, z);
            let s = self.lshrc(s, 1);
            let s = self.shlc(s, k);
            r = self.or(r, s);
        }
        r
    }
    /// `sat_i32` of an i64.
    pub fn sat_i32(&mut self, r: Value) -> Value {
        let hi = self.c64(0x7FFF_FFFF);
        let lo = self.c64(0xFFFF_FFFF_8000_0000);
        let over = self.cmp(Pred::SGt, r, hi);
        let under = self.cmp(Pred::SLt, r, lo);
        let t = self.trunc32(r);
        let t = self.selc(under, 0x8000_0000, t);
        self.selc(over, 0x7FFF_FFFF, t)
    }
    /// `cvt_norm_i16_f32`: round(clamp(nan ? 0 : f, -1, 1) * 32767) & 0xFFFF.
    pub fn cvt_norm_i16_f32(&mut self, f: Value) -> Value {
        let n = self.isnan(f);
        let z = self.fc(0.0);
        let x = self.sel(n, z, f);
        let (lo, hi) = (self.fc(-1.0), self.fc(1.0));
        let x = self.fmax(x, lo);
        let x = self.fmin(x, hi);
        let k = self.fc(32767.0);
        let x = self.fmul(x, k);
        let x = self.un(op::ROUND_NE, x);
        let i = self.ftosi(x);
        self.andc(i, 0xFFFF)
    }
    /// `(uint)round(saturate(f) * 65535.0f)`
    pub fn cvt_norm_u16_f32(&mut self, f: Value) -> Value {
        let x = self.un(op::SATURATE, f);
        let k = self.fc(65535.0);
        let x = self.fmul(x, k);
        let x = self.un(op::ROUND_NE, x);
        self.ftoui(x)
    }
    /// `cvt_norm_f16`: exact binary16 → unorm / snorm magnitude.
    fn cvt_norm_f16(&mut self, h: Value, scale: u32) -> Value {
        let mag = self.andc(h, 0x7FFF);
        let nan = self.cmpc(Pred::UGt, mag, 0x7C00);
        let e = self.lshrc(h, 10);
        let exp = self.andc(e, 0x1F);
        let frac = self.andc(h, 0x3FF);
        let big = self.cmpc(Pred::UGe, exp, 15);
        let ez = self.cmpc(Pred::Eq, exp, 0);
        let im = self.orc(frac, 1024);
        let m = self.sel(ez, frac, im);
        let k25 = self.c(25);
        let s1 = self.sub(k25, exp);
        let s = self.selc(ez, 24, s1);
        let s = self.andc(s, 31);
        let k = self.c(scale);
        let prod = self.mul(m, k);
        let q = self.bin(BinOp::LShr, prod, s);
        let one = self.c(1);
        let full = self.bin(BinOp::Shl, one, s);
        let msk = self.sub(full, one);
        let rem = self.and(prod, msk);
        let sm1 = self.sub(s, one);
        let sm1 = self.andc(sm1, 31);
        let half = self.bin(BinOp::Shl, one, sm1);
        let gt = self.cmp(Pred::UGt, rem, half);
        let eq = self.cmp(Pred::Eq, rem, half);
        let odd = self.andc(q, 1);
        let odd = self.nz(odd);
        let tie = self.land(eq, odd);
        let inc = self.lor(gt, tie);
        let inc = self.u(inc);
        let r = self.add(q, inc);
        let r = self.selc(big, scale, r);
        self.selc(nan, 0, r)
    }
    pub fn cvt_norm_u16_f16(&mut self, h: Value) -> Value {
        let r = self.cvt_norm_f16(h, 65535);
        let s = self.andc(h, 0x8000);
        let neg = self.nz(s);
        self.selc(neg, 0, r)
    }
    pub fn cvt_norm_i16_f16(&mut self, h: Value) -> Value {
        let r = self.cvt_norm_f16(h, 32767);
        let s = self.andc(h, 0x8000);
        let neg = self.nz(s);
        let z = self.c(0);
        let n = self.sub(z, r);
        let v = self.sel(neg, n, r);
        self.andc(v, 0xFFFF)
    }
    /// `fclass16`: the class bit index of a binary16 value.
    pub fn fclass16(&mut self, h: Value) -> Value {
        let e = self.lshrc(h, 10);
        let e = self.andc(e, 0x1F);
        let m = self.andc(h, 0x3FF);
        let s = self.andc(h, 0x8000);
        self.fclass_parts(e, 0x1F, m, 0x200, s)
    }
    /// `fclass`: the class bit index of a binary32 value.
    pub fn fclass(&mut self, f: Value) -> Value {
        let u = self.bits(f);
        let e = self.lshrc(u, 23);
        let e = self.andc(e, 0xFF);
        let m = self.andc(u, 0x7F_FFFF);
        let s = self.lshrc(u, 31);
        self.fclass_parts(e, 0xFF, m, 0x40_0000, s)
    }
    /// class bit index from the exponent field, its all-ones value, mantissa != 0,
    /// the quiet bit and the sign
    pub fn fclass_bits(&mut self, e: Value, emax: u32, mnz: Value, qn: Value, n: Value) -> Value {
        let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9) = (self.c(0), self.c(1), self.c(2), self.c(3), self.c(4), self.c(5), self.c(6), self.c(7), self.c(8), self.c(9));
        let nanc = self.sel(qn, c1, c0);
        let infc = self.sel(n, c2, c9);
        let special = self.sel(mnz, nanc, infc);
        let zc = self.sel(n, c5, c6);
        let dc = self.sel(n, c4, c7);
        let zero = self.sel(mnz, dc, zc);
        let normal = self.sel(n, c3, c8);
        let ez = self.cmpc(Pred::Eq, e, 0);
        let emx = self.cmpc(Pred::Eq, e, emax);
        let r = self.sel(ez, zero, normal);
        self.sel(emx, special, r)
    }
    fn fclass_parts(&mut self, e: Value, emax: u32, m: Value, quiet: u32, sign: Value) -> Value {
        let n = self.nz(sign);
        let mnz = self.nz(m);
        let q = self.andc(m, quiet);
        let qn = self.nz(q);
        self.fclass_bits(e, emax, mnz, qn, n)
    }
    /// `isnan_d` from the bits.
    pub fn isnan_d(&mut self, x: Value) -> Value {
        let (l, h) = self.s.split_double(x);
        let h = self.andc(h, 0x7FFF_FFFF);
        let gt = self.cmpc(Pred::UGt, h, 0x7FF0_0000);
        let eq = self.cmpc(Pred::Eq, h, 0x7FF0_0000);
        let lnz = self.nz(l);
        let e = self.land(eq, lnz);
        self.lor(gt, e)
    }
    /// `isinf_d` from the bits.
    pub fn isinf_d(&mut self, x: Value) -> Value {
        let (l, h) = self.s.split_double(x);
        let h = self.andc(h, 0x7FFF_FFFF);
        let eq = self.cmpc(Pred::Eq, h, 0x7FF0_0000);
        let lz = self.cmpc(Pred::Eq, l, 0);
        self.land(eq, lz)
    }
}

#[cfg(test)]
mod disable_math_refactoring_tests {
    use super::super::kernel::{self, KernelOptions};
    use super::super::validate;
    use crate::gpu::rdna2;
    use crate::gpu::recomp::PsInterface;

    fn opts() -> KernelOptions {
        KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }
    }

    /// Parses the entry-point shader flags (`!dx.entryPoints` property tag `0`)
    /// from DXIL disassembly. Returns `0` if tag `0` is absent.
    fn entry_point_flags(dis: &str) -> u64 {
        let props_id = dis
            .lines()
            .find_map(|line| {
                if !line.contains("!\"main\"") {
                    return None;
                }
                let start = line.rfind('!')?;
                let rest = &line[start..];
                let end = rest.find(|c: char| !c.is_ascii_digit() && c != '!').unwrap_or(rest.len());
                Some(&rest[..end])
            })
            .expect("entry point metadata node");
        let props_line = dis
            .lines()
            .find(|line| line.trim_start().starts_with(&format!("{props_id} =")))
            .expect("entry point properties node");
        let marker = "i32 0, i64 ";
        let Some(idx) = props_line.find(marker) else {
            return 0;
        };
        let rest = &props_line[idx + marker.len()..];
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        rest[..end].parse::<u64>().expect("valid u64 flags")
    }

    fn v_mad_f32(vdst: u64, s0: u64, s1: u64, s2: u64) -> Vec<u8> {
        rdna2::encode("V_MAD_F32", "ENC_VOP3", &[("VDST", vdst), ("SRC0", 256 + s0), ("SRC1", 256 + s1), ("SRC2", 256 + s2)], None).unwrap()
    }

    fn v_mac_f32(vdst: u64, s0: u64, s1: u64) -> Vec<u8> {
        rdna2::encode("V_MAC_F32", "ENC_VOP2", &[("VDST", vdst), ("SRC0", 256 + s0), ("VSRC1", s1)], None).unwrap()
    }

    fn v_fma_f32(vdst: u64, s0: u64, s1: u64, s2: u64) -> Vec<u8> {
        rdna2::encode("V_FMA_F32", "ENC_VOP3", &[("VDST", vdst), ("SRC0", 256 + s0), ("SRC1", 256 + s1), ("SRC2", 256 + s2)], None).unwrap()
    }

    fn v_add_f32(vdst: u64, s0: u64, s1: u64) -> Vec<u8> {
        rdna2::encode("V_ADD_F32", "ENC_VOP2", &[("VDST", vdst), ("SRC0", 256 + s0), ("VSRC1", s1)], None).unwrap()
    }

    fn s_endpgm() -> Vec<u8> {
        rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap()
    }

    #[test]
    fn compute_unfused_mad_sets_disable_math_refactoring() {
        for inst in [v_mad_f32(3, 0, 1, 2), v_mac_f32(2, 0, 1)] {
            let mut code = inst;
            code.extend(s_endpgm());
            let dxil = kernel::translate(&code, opts()).expect("translate compute unfused mad");
            validate::validate(&dxil).expect("dxil validate compute unfused mad");
            let dis = validate::disassemble(&dxil).expect("disassemble compute unfused mad");
            let flags = entry_point_flags(&dis);
            assert_ne!(flags & (1 << 1), 0, "expected DisableMathRefactoring (bit 1) in flags={flags:#x}:\n{dis}");
        }
    }

    fn ps_with_pos() -> PsInterface {
        use crate::gpu::dxbc::SigElement;
        // SPI_PS_INPUT_ENA bits 8..11: SV_Position.xyzw in v0..v3; SV_Target0 so EXP TGT=0 stays live.
        PsInterface {
            inputs: vec![SigElement { name: "SV_Position".into(), index: 0, system_value: 1, component_type: 3, register: 0, mask: 0xF, rw_mask: 0xF, stream: 0 }],
            outputs: vec![SigElement { name: "SV_Target".into(), index: 0, system_value: 64, component_type: 3, register: 0, mask: 0xF, rw_mask: 0xF, stream: 0 }],
            input_ena: 0x0F00,
            attrs: None,
        }
    }

    #[test]
    fn pixel_unfused_mad_sets_disable_math_refactoring() {
        let mut code = v_mad_f32(3, 0, 1, 2);
        code.extend(rdna2::encode("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 3), ("VSRC1", 3), ("VSRC2", 3), ("VSRC3", 3), ("DONE", 1), ("VM", 1)], None).unwrap());
        code.extend(s_endpgm());
        let ps = ps_with_pos();
        let k = kernel::translate_pixel(&code, opts(), &ps).expect("translate pixel unfused mad");
        validate::validate(&k.dxil).expect("dxil validate pixel unfused mad");
        let dis = validate::disassemble(&k.dxil).expect("disassemble pixel unfused mad");
        assert!(dis.contains("fmul float") && dis.contains("fadd float"), "expected live fmul+fadd:\n{dis}");
        let flags = entry_point_flags(&dis);
        assert_ne!(flags & (1 << 1), 0, "expected DisableMathRefactoring (bit 1) in pixel flags={flags:#x}:\n{dis}");
    }

    #[test]
    fn fmad_suppresses_disable_math_refactoring_in_compute_and_pixel() {
        let mut cs_code = v_mad_f32(3, 0, 1, 2);
        cs_code.extend(v_fma_f32(4, 0, 1, 3));
        cs_code.extend(s_endpgm());
        let dxil = kernel::translate(&cs_code, opts()).expect("translate compute mad+fma");
        validate::validate(&dxil).expect("dxil validate compute mad+fma");
        let dis = validate::disassemble(&dxil).expect("disassemble compute mad+fma");
        assert!(dis.contains("dx.op.tertiary.f32(i32 46,"), "expected live FMad in compute shader:\n{dis}");
        let flags = entry_point_flags(&dis);
        assert_eq!(flags & (1 << 1), 0, "DisableMathRefactoring must be omitted when FMad is used (flags={flags:#x}):\n{dis}");

        let mut ps_code = v_mad_f32(3, 0, 1, 2);
        ps_code.extend(v_fma_f32(4, 0, 1, 3));
        ps_code.extend(rdna2::encode("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 4), ("VSRC2", 4), ("VSRC3", 4), ("DONE", 1), ("VM", 1)], None).unwrap());
        ps_code.extend(s_endpgm());
        let ps = ps_with_pos();
        let k = kernel::translate_pixel(&ps_code, opts(), &ps).expect("translate pixel mad+fma");
        validate::validate(&k.dxil).expect("dxil validate pixel mad+fma");
        let dis = validate::disassemble(&k.dxil).expect("disassemble pixel mad+fma");
        assert!(dis.contains("dx.op.tertiary.f32(i32 46,"), "expected live FMad in pixel shader:\n{dis}");
        let flags = entry_point_flags(&dis);
        assert_eq!(flags & (1 << 1), 0, "DisableMathRefactoring must be omitted in pixel shader when FMad is used (flags={flags:#x}):\n{dis}");
    }

    #[test]
    fn plain_float_add_does_not_set_disable_math_refactoring() {
        let mut code = v_add_f32(2, 0, 1);
        code.extend(s_endpgm());
        let dxil = kernel::translate(&code, opts()).expect("translate compute v_add_f32");
        validate::validate(&dxil).expect("dxil validate compute v_add_f32");
        let dis = validate::disassemble(&dxil).expect("disassemble compute v_add_f32");
        let flags = entry_point_flags(&dis);
        assert_eq!(flags & (1 << 1), 0, "plain v_add_f32 must not set DisableMathRefactoring (flags={flags:#x}):\n{dis}");
    }

    #[test]
    fn dead_fmad_eliminated_by_dce_still_allows_disable_math_refactoring() {
        // v5 is written by v_fma_f32 and never read, so DCE eliminates the FMad
        // while v4's v_mad_f32 stays live through the pixel export.
        let mut code = v_fma_f32(5, 0, 1, 2);
        code.extend(v_mad_f32(4, 0, 1, 2));
        code.extend(rdna2::encode("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 4), ("VSRC2", 4), ("VSRC3", 4), ("DONE", 1), ("VM", 1)], None).unwrap());
        code.extend(s_endpgm());
        let ps = ps_with_pos();
        let k = kernel::translate_pixel(&code, opts(), &ps).expect("translate pixel dead-fma+mad");
        validate::validate(&k.dxil).expect("dxil validate pixel dead-fma+mad");
        let dis = validate::disassemble(&k.dxil).expect("disassemble pixel dead-fma+mad");
        assert!(!dis.contains("dx.op.tertiary.f32(i32 46,"), "expected dead FMad to be eliminated by DCE:\n{dis}");
        let flags = entry_point_flags(&dis);
        assert_ne!(flags & (1 << 1), 0, "post-DCE dead FMad should not suppress DisableMathRefactoring (flags={flags:#x}):\n{dis}");
    }
}
