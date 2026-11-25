//! Directed f32 rounding: MODE.FP_ROUND[1:0] (single precision) = 1 (toward +inf), 2 (toward
//! -inf) or 3 (toward zero), which the HLSL path cannot express (DXIL float operations always
//! round to nearest even).
//!
//! Only programs that write a non-default single-precision rounding mode somewhere
//! (`f32_round_writes`) get this: MODE[1:0] and MODE[5:4] (FP_DENORM single) live in
//! `VAR_F32_MODE` (initially 0x30: nearest even, denormals kept), updated by `s_round_mode`,
//! `s_denorm_mode` and `s_setreg*` of MODE. Each rounding f32 instruction computes its result
//! exactly and rounds it by the current mode; with the mode a constant (the usual
//! `s_round_mode` immediate) the selects fold to one rounding.
//!
//! Exact values: an f32 product is exact in f64; a sum is `s + e` from TwoSum in f64 (exact,
//! as f32 inputs cannot overflow f64); an fma is TwoSum(a·b, c). The f64 → f32 rounding is done
//! on the bits (no float operation sees an f32 denormal, so the GPU's f32 denormal handling
//! cannot interfere), with `e` deciding the direction when `s` itself is an f32 value.
//!
//! Denormals as RDNA defines FP_DENORM (bit 4: keep input denormals, bit 5: keep output
//! denormals): flushed inputs become ±0 before the operation, flushed results become ±0 after
//! rounding. Round to nearest even keeps the translator's usual (native) code.

use super::lower::{LResult, Lw};
use super::module::{BinOp, CastOp, Pred, Value};

/// MODE[1:0] and MODE[5:4] (single-precision round and denormal modes).
pub const VAR_F32_MODE: u32 = super::lower::VAR_BUDGET + 5;
/// The MODE bits `VAR_F32_MODE` mirrors.
pub const MODE_BITS: u32 = 0x33;
/// Initial MODE (nearest even, f32 denormals kept: what the translator assumes elsewhere).
pub const MODE_INIT: u32 = 0x30;
/// Translator-only flag: an instruction on this path has explicitly written MODE[4].
pub(super) const F32_DENORM_INPUT_WRITTEN: u32 = 1 << 31;

const MASK52: u64 = (1 << 52) - 1;

impl Lw {
    /// The current single-precision MODE bits (None: the program never directs f32 rounding).
    pub fn f32_mode(&mut self) -> Option<Value> {
        if self.f32_rm {
            Some(self.rd(VAR_F32_MODE))
        } else {
            None
        }
    }

    /// MODE ← (MODE & !mask) | (v & mask), on the mirrored bits.
    pub fn mode_write(&mut self, v: Value, mask: u32) {
        let mask = mask & MODE_BITS;
        if mask == 0 || !(self.f32_rm || self.track_frexp_denorm) {
            return;
        }
        let cur = self.rd(VAR_F32_MODE);
        let keep = self.andc(cur, !mask);
        let nv = self.andc(v, mask);
        let r = self.or(keep, nv);
        let r = if self.track_frexp_denorm && mask & 0x10 != 0 {
            self.orc(r, F32_DENORM_INPUT_WRITTEN)
        } else {
            r
        };
        self.wr(VAR_F32_MODE, r);
    }

    fn bits64(&mut self, x: Value) -> Value {
        let t = self.i64();
        self.s.m.bitcast(x, t)
    }
    fn dbl(&mut self, b: Value) -> Value {
        let t = self.f64ty();
        self.s.m.bitcast(b, t)
    }
    fn sh64(&mut self, op: BinOp, a: Value, k: u64) -> Value {
        let k = self.c64(k);
        self.bin(op, a, k)
    }
    fn and64(&mut self, a: Value, k: u64) -> Value {
        let k = self.c64(k);
        self.and(a, k)
    }

    /// f32 bits with denormals flushed to ±0 when MODE bit `bit` is clear.
    fn flush_den(&mut self, u: Value, mode: Value, bit: u32) -> Value {
        let keep = self.andc(mode, 1 << bit);
        let keep = self.nz(keep);
        let e = self.andc(u, 0x7F80_0000);
        let den = self.cmpc(Pred::Eq, e, 0);
        let sign = self.andc(u, 0x8000_0000);
        let f = self.sel(den, sign, u);
        self.sel(keep, u, f)
    }

    /// The exact f64 of f32 bits `u` (input denormals flushed by MODE bit 4). Denormals are
    /// built from their bits: no f32 float operation sees them.
    fn ext_exact(&mut self, u: Value, mode: Value) -> Value {
        let u = self.flush_den(u, mode, 4);
        let exp = self.andc(u, 0x7F80_0000);
        let mant = self.andc(u, 0x7F_FFFF);
        let ez = self.cmpc(Pred::Eq, exp, 0);
        let mnz = self.nz(mant);
        let den = self.land(ez, mnz);
        let fu = self.f(u);
        let f64t = self.f64ty();
        let n = self.s.m.cast(CastOp::FPExt, fu, f64t);
        // denormal: mant · 2^-149 = 2^(top - 149) · 1.f, top = the highest set bit
        let top = self.firstbithigh(mant);
        let ex = self.addc(top, 874);
        let ex = self.zext64(ex);
        let ex = self.sh64(BinOp::Shl, ex, 52);
        let sh = self.c(52);
        let sh = self.sub(sh, top);
        let sh = self.andc(sh, 63);
        let sh = self.zext64(sh);
        let m = self.zext64(mant);
        let m = self.bin(BinOp::Shl, m, sh);
        let m = self.and64(m, MASK52);
        let sg = self.andc(u, 0x8000_0000);
        let sg = self.zext64(sg);
        let sg = self.sh64(BinOp::Shl, sg, 32);
        let b = self.or(sg, ex);
        let b = self.or(b, m);
        let dv = self.dbl(b);
        self.sel(den, dv, n)
    }

    /// The exact f64 of an i32 / u32 (built from its bits).
    fn int_exact(&mut self, a: Value, signed: bool) -> Value {
        let (neg, mag) = if signed {
            let neg = self.cmpc(Pred::SLt, a, 0);
            let z = self.c(0);
            let n = self.sub(z, a);
            (Some(neg), self.sel(neg, n, a))
        } else {
            (None, a)
        };
        let top = self.firstbithigh(mag);
        let ex = self.addc(top, 1023);
        let ex = self.zext64(ex);
        let ex = self.sh64(BinOp::Shl, ex, 52);
        let sh = self.c(52);
        let sh = self.sub(sh, top);
        let sh = self.andc(sh, 63);
        let sh = self.zext64(sh);
        let m = self.zext64(mag);
        let m = self.bin(BinOp::Shl, m, sh);
        let m = self.and64(m, MASK52);
        let mut b = self.or(ex, m);
        if let Some(neg) = neg {
            let sg = self.c64(1 << 63);
            let z = self.c64(0);
            let sg = self.sel(neg, sg, z);
            b = self.or(b, sg);
        }
        let zero = self.cmpc(Pred::Eq, mag, 0);
        let z = self.c64(0);
        let b = self.sel(zero, z, b);
        self.dbl(b)
    }

    /// f32 bits of the exact value `s + e` (`s` its f64 rounding, `e` the remainder or +0)
    /// rounded by MODE (`rm` 1: +inf, 2: -inf, 3: zero; 0 is not used here), output denormals
    /// flushed by MODE bit 5. `s` NaN / inf: truncated directly from `s` (preserving the
    /// input denormal mode already applied to `s`).
    fn round_exact(&mut self, s: Value, e: Value, mode: Value) -> Value {
        let sb = self.bits64(s);
        let sg = self.sh64(BinOp::LShr, sb, 63);
        let neg = self.trunc32(sg);
        let neg = self.nz(neg);
        let ex = self.sh64(BinOp::LShr, sb, 52);
        let ex = self.trunc32(ex);
        let ex = self.andc(ex, 0x7FF);
        let mant = self.and64(sb, MASK52);
        let ez = self.cmpc(Pred::Eq, ex, 0);
        let one = self.c(1);
        let exe = self.sel(ez, one, ex);
        let imp = self.c64(1 << 52);
        let mi = self.or(mant, imp);
        let m = self.sel(ez, mant, mi);
        // f32 biased exponent
        let k896 = self.c(896);
        let e32 = self.sub(exe, k896);
        let normal = self.cmpc(Pred::SGe, e32, 1);
        let k30 = self.c(30);
        let sub = self.sub(k30, e32);
        let k29 = self.c(29);
        let sh = self.sel(normal, k29, sub);
        let big = self.cmpc(Pred::UGt, sh, 63);
        let k63 = self.c(63);
        let sh = self.sel(big, k63, sh);
        let sh = self.zext64(sh);
        let keep = self.bin(BinOp::LShr, m, sh);
        let one64 = self.c64(1);
        let lowm = self.bin(BinOp::Shl, one64, sh);
        let lowm = self.sub(lowm, one64);
        let rest = self.and(m, lowm);
        let em1 = self.sub(e32, one);
        let em1 = self.shlc(em1, 23);
        let z32 = self.c(0);
        let base = self.sel(normal, em1, z32);
        let base = self.zext64(base);
        let mag = self.add(base, keep);
        let over = self.cmpc(Pred::SGe, e32, 255);
        let maxf = self.c64(0x7F7F_FFFF);
        let mag = self.sel(over, maxf, mag);
        let rnz = self.nz(rest);
        let rnz = self.lor(rnz, over);
        // the remainder: nonzero, and its sign
        let eb = self.bits64(e);
        let emag = self.and64(eb, !(1u64 << 63));
        let enz = self.nz(emag);
        let es = self.sh64(BinOp::LShr, eb, 63);
        let es = self.trunc32(es);
        let eneg = self.nz(es);
        // |exact| < |s| = an f32 value: truncation is one below it
        let opp = self.xor(eneg, neg);
        let nr = self.lnot(rnz);
        let below = self.land(nr, enz);
        let below = self.land(below, opp);
        let bl = self.u(below);
        let bl = self.zext64(bl);
        let mag = self.sub(mag, bl);
        let inexact = self.lor(rnz, enz);
        let rm = self.andc(mode, 3);
        let is_p = self.cmpc(Pred::Eq, rm, 1);
        let is_n = self.cmpc(Pred::Eq, rm, 2);
        let pos = self.lnot(neg);
        let up_p = self.land(is_p, pos);
        let up_n = self.land(is_n, neg);
        let up = self.lor(up_p, up_n);
        let up = self.land(up, inexact);
        let upv = self.u(up);
        let upv = self.zext64(upv);
        let mag = self.add(mag, upv);
        let mag = self.trunc32(mag);
        let sgn = self.u(neg);
        let sgn = self.shlc(sgn, 31);
        let r = self.or(sgn, mag);
        let r = self.flush_den(r, mode, 5);
        let special = self.cmpc(Pred::Eq, ex, 0x7FF);
        let f32t = self.s.m.f32();
        let sf = self.s.m.cast(CastOp::FPTrunc, s, f32t);
        let sp = self.bits(sf);
        self.sel(special, sp, r)
    }

    /// The final f32 (float) of a directed operation: `native` (round to nearest even) when
    /// MODE[1:0] = 0, else `directed()`. Folds when the mode is a constant.
    fn by_mode(&mut self, native: Value, mode: Value, directed: &mut dyn FnMut(&mut Lw) -> Value) -> Value {
        let rm = self.andc(mode, 3);
        let is_ne = self.cmpc(Pred::Eq, rm, 0);
        if self.s.m.const_u64(is_ne) == Some(1) {
            return native;
        }
        let d = directed(self);
        let d = self.f(d);
        self.sel(is_ne, native, d)
    }

    /// TwoSum: (s, e) with s = fl(a + b), e = a + b - s exactly.
    fn two_sum(&mut self, a: Value, b: Value) -> (Value, Value) {
        let s = self.fadd(a, b);
        let bb = self.fsub(s, a);
        let ab = self.fsub(s, bb);
        let da = self.fsub(a, ab);
        let db = self.fsub(b, bb);
        let e = self.fadd(da, db);
        (s, e)
    }

    /// Sum `a + b` (f64, exact inputs) rounded by MODE; an exact zero from a sum is -0 toward
    /// -inf unless both addends are +0 (IEEE 754 6.3).
    fn round_sum(&mut self, a: Value, b: Value, mode: Value) -> Value {
        let (s, e) = self.two_sum(a, b);
        let r = self.round_exact(s, e, mode);
        let mag = self.andc(r, 0x7FFF_FFFF);
        let zero = self.cmpc(Pred::Eq, mag, 0);
        let ab = self.bits64(a);
        let bb = self.bits64(b);
        let o = self.or(ab, bb);
        let z = self.c64(0);
        let both_pos = self.cmp(Pred::Eq, o, z);
        let rm = self.andc(mode, 3);
        let is_n = self.cmpc(Pred::Eq, rm, 2);
        // the sum's own zero (a flushed tiny result has s nonzero)
        let sb = self.bits64(s);
        let sm = self.and64(sb, !(1u64 << 63));
        let sz = self.cmp(Pred::Eq, sm, z);
        let c = self.land(zero, sz);
        let c = self.land(c, is_n);
        let np = self.lnot(both_pos);
        let c = self.land(c, np);
        let negz = self.c(0x8000_0000);
        self.sel(c, negz, r)
    }

    /// `a + b` (f32 bits, modifiers applied), `native` the nearest-even float result.
    pub fn rm_add(&mut self, a: Value, b: Value, native: Value) -> Value {
        let Some(mode) = self.f32_mode() else { return native };
        self.by_mode(native, mode, &mut |lw| {
            let x = lw.ext_exact(a, mode);
            let y = lw.ext_exact(b, mode);
            lw.round_sum(x, y, mode)
        })
    }

    /// `a · b`.
    pub fn rm_mul(&mut self, a: Value, b: Value, native: Value) -> Value {
        let Some(mode) = self.f32_mode() else { return native };
        self.by_mode(native, mode, &mut |lw| {
            let x = lw.ext_exact(a, mode);
            let y = lw.ext_exact(b, mode);
            let p = lw.fmul(x, y);
            let z = lw.dc(0.0);
            lw.round_exact(p, z, mode)
        })
    }

    /// `a · b + c`, one rounding.
    pub fn rm_fma(&mut self, a: Value, b: Value, c: Value, native: Value) -> Value {
        let Some(mode) = self.f32_mode() else { return native };
        self.by_mode(native, mode, &mut |lw| {
            let x = lw.ext_exact(a, mode);
            let y = lw.ext_exact(b, mode);
            let w = lw.ext_exact(c, mode);
            let p = lw.fmul(x, y);
            lw.round_sum(p, w, mode)
        })
    }

    /// `v_mad_f32` / `v_mac_f32`: the product rounded, then the sum rounded.
    pub fn rm_mad(&mut self, a: Value, b: Value, c: Value, native: Value) -> Value {
        if self.f32_mode().is_none() {
            return native;
        }
        let fa = self.f(a);
        let fb = self.f(b);
        let pn = self.fmul(fa, fb);
        let p = self.rm_mul(a, b, pn);
        let pb = self.bits(p);
        let fc = self.f(c);
        let sn = self.fadd(p, fc);
        let r = self.rm_add(pb, c, sn);
        // nearest even: the native mad as before
        let mode = self.rd(VAR_F32_MODE);
        let rm = self.andc(mode, 3);
        let is_ne = self.cmpc(Pred::Eq, rm, 0);
        self.sel(is_ne, native, r)
    }

    /// An integer converted to f32 (`v_cvt_f32_i32` / `_u32`).
    pub fn rm_cvt_int(&mut self, a: Value, signed: bool, native: Value) -> Value {
        let Some(mode) = self.f32_mode() else { return native };
        self.by_mode(native, mode, &mut |lw| {
            let x = lw.int_exact(a, signed);
            let z = lw.dc(0.0);
            lw.round_exact(x, z, mode)
        })
    }

    /// An f64 converted to f32 (`v_cvt_f32_f64`).
    pub fn rm_cvt_f64(&mut self, a: Value, native: Value) -> Value {
        let Some(mode) = self.f32_mode() else { return native };
        self.by_mode(native, mode, &mut |lw| {
            let z = lw.dc(0.0);
            lw.round_exact(a, z, mode)
        })
    }

    /// `a · 2^n` (`v_ldexp_f32`): rounds only into the denormal range.
    pub fn rm_ldexp(&mut self, a: Value, n: Value, native: Value) -> Value {
        let Some(mode) = self.f32_mode() else { return native };
        self.by_mode(native, mode, &mut |lw| {
            let x = lw.ext_exact(a, mode);
            // n clamped to ±400 keeps the product exact in f64 and the f32 result the same
            let lo = lw.c(-400i32 as u32);
            let hi = lw.c(400);
            let n = lw.imax(n, lo);
            let n = lw.imin(n, hi);
            let e = lw.addc(n, 1023);
            let e = lw.zext64(e);
            let e = lw.sh64(BinOp::Shl, e, 52);
            let p2 = lw.dbl(e);
            let p = lw.fmul(x, p2);
            let z = lw.dc(0.0);
            lw.round_exact(p, z, mode)
        })
    }

    /// Square root, correctly rounded in the directed modes (nearest even keeps the native
    /// `sqrt`). The input is scaled into the normal range (· 2^100, exact), the native root is
    /// corrected to the largest f32 whose square (exact in f64) is ≤ the input, stepped up
    /// toward +inf when inexact, and scaled back (· 2^-50).
    pub fn rm_sqrt(&mut self, a: Value, native: Value) -> Value {
        let Some(mode) = self.f32_mode() else { return native };
        self.by_mode(native, mode, &mut |lw| {
            let u = lw.flush_den(a, mode, 4);
            let x = lw.ext_exact(u, mode);
            let exp = lw.andc(u, 0x7F80_0000);
            let small = lw.cmpc(Pred::Eq, exp, 0);
            let k100 = lw.dc(2f64.powi(100));
            let one = lw.dc(1.0);
            let k = lw.sel(small, k100, one);
            let y = lw.fmul(x, k);
            let f32t = lw.s.m.f32();
            let yf = lw.s.m.cast(CastOp::FPTrunc, y, f32t);
            let r = lw.un(super::ops::FUnary::Sqrt, yf);
            let mut lo = lw.bits(r);
            for _ in 0..2 {
                let q = lw.square64(lo);
                let gt = lw.fcmp(Pred::FOGt, q, y);
                let dn = lw.addc(lo, u32::MAX);
                lo = lw.sel(gt, dn, lo);
            }
            for _ in 0..2 {
                let upb = lw.addc(lo, 1);
                let q = lw.square64(upb);
                let le = lw.fcmp(Pred::FOLe, q, y);
                lo = lw.sel(le, upb, lo);
            }
            let q = lw.square64(lo);
            let exact = lw.fcmp(Pred::FOEq, q, y);
            let rm = lw.andc(mode, 3);
            let is_p = lw.cmpc(Pred::Eq, rm, 1);
            let inex = lw.lnot(exact);
            let up = lw.land(is_p, inex);
            let upv = lw.u(up);
            let res = lw.add(lo, upv);
            let back = lw.c(50 << 23);
            let scaled = lw.sub(res, back);
            let res = lw.sel(small, scaled, res);
            // ±0 (flushed denormals too), negative, inf, NaN: sqrt(±0) = ±0, else native
            let mag = lw.andc(u, 0x7FFF_FFFF);
            let z = lw.cmpc(Pred::Eq, mag, 0);
            let negv = lw.cmpc(Pred::SLt, u, 0);
            let infnan = lw.cmpc(Pred::UGe, mag, 0x7F80_0000);
            let sp = lw.lor(z, negv);
            let sp = lw.lor(sp, infnan);
            let nb = lw.bits(native);
            let sg = lw.andc(u, 0x8000_0000);
            let zr = lw.sel(z, sg, nb);
            lw.sel(sp, zr, res)
        })
    }

    /// The square of a normal f32 (bits) in f64 (exact: 48 significant bits).
    fn square64(&mut self, b: Value) -> Value {
        let f = self.f(b);
        let t = self.f64ty();
        let d = self.s.m.cast(CastOp::FPExt, f, t);
        self.fmul(d, d)
    }

    /// `s_denorm_mode`: FP_DENORM single (MODE[5:4]) from SIMM16[1:0].
    pub fn denorm_mode(&mut self, simm: u32) -> LResult<()> {
        let v = self.c((simm & 3) << 4);
        self.mode_write(v, 0x30);
        Ok(())
    }
}

/// Whether the program writes a non-default single-precision rounding mode (MODE[1:0]):
/// `s_round_mode` / `s_setreg_imm32_b32` with a nonzero value there, or any `s_setreg_b32`
/// covering those bits (a run-time value).
pub fn f32_round_writes(insts: &[crate::gpu::rdna2::Decoded]) -> bool {
    use super::lower::{hwreg_imm_field, hwreg_mode_round_masks};
    insts.iter().any(|d| {
        let name = d.inst.name.to_ascii_lowercase();
        let simm = crate::gpu::rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) as u16;
        match name.as_str() {
            "s_round_mode" => simm & 3 != 0,
            "s_setreg_imm32_b32" => {
                let (sp, _, off) = hwreg_mode_round_masks(simm);
                sp != 0 && hwreg_imm_field(d.literal.unwrap_or(0), off, 0, sp) != 0
            }
            "s_setreg_b32" => hwreg_mode_round_masks(simm).0 != 0,
            _ => false,
        }
    })
}

/// Instructions whose f32 rounding obeys MODE[1:0] but that the directed path does not
/// implement: a translation error in programs that direct f32 rounding.
pub fn f32_round_unsupported(name: &str) -> bool {
    matches!(name, "v_fma_mix_f32" | "v_dot2_f32_f16" | "v_dot2c_f32_f16" | "v_interp_p1_f32" | "v_interp_p2_f32")
}
