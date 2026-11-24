//! VALU lowering of 16-bit, 64-bit and packed operations and relative moves
//! (port of `recomp/emit.rs` `valu16` / `valu64` / `valu_pk` / `movrel`).

use super::lower::{err, Dst, LResult, Lw, VAR_VGPR};
use super::lower_valu::Src;
use super::module::{BinOp, CastOp, Pred, Value};
use super::ops::{Binary, FUnary, Tertiary};
use super::ops_core::op;
use super::shader::Feature;
use crate::gpu::rdna2::{self, Decoded, Variant};

impl Lw {
    fn hsel(&mut self, s: Value, hi: bool) -> Value {
        if hi {
            self.lshrc(s, 16)
        } else {
            self.andc(s, 0xFFFF)
        }
    }
    /// sign-extended 16-bit value
    pub fn sx16(&mut self, h: Value) -> Value {
        let x = self.shlc(h, 16);
        self.ashrc(x, 16)
    }

    pub fn valu_pk(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        if d.variant != Variant::Plain {
            return err(d, "VOP3P SDWA/DPP");
        }
        let (outs, ins) = Self::ops(d);
        let vop3p = d.enc.name == "ENC_VOP3P";
        let fv = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let (sel, sel_hi) = if vop3p { (fv("OP_SEL"), fv("OP_SEL_HI") | fv("OP_SEL_HI_2") << 2) } else { (0, 0b111) };
        let (neg, neg_hi) = if vop3p { (fv("NEG"), fv("NEG_HI")) } else { (0, 0) };
        let clamp = vop3p && fv("CLAMP") == 1;
        let u: Vec<Value> = self.sources(d, &ins, name.ends_with("_f16"))?.into_iter().map(Src::w).collect();
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        let z = self.c(0);
        let cur = match &dst {
            Dst::V(n) => self.vreg(*n),
            _ => z,
        };
        let src = |i: usize| u.get(i).copied().unwrap_or(z);
        let h = |s: &mut Self, i: usize, hi: bool| -> Value {
            let x = src(i);
            let pick = (if hi { sel_hi } else { sel }) >> i & 1 == 1;
            s.hsel(x, pick)
        };
        let fh = |s: &mut Self, i: usize, hi: bool| -> Value {
            let x = h(s, i, hi);
            let v = s.s.f16_to_f32(x);
            if (if hi { neg_hi } else { neg }) >> i & 1 == 1 {
                s.fneg(v)
            } else {
                v
            }
        };
        let si = |s: &mut Self, i: usize, hi: bool| {
            let x = h(s, i, hi);
            s.sx16(x)
        };
        let fr = |s: &mut Self, e: Value| {
            let e = if clamp { s.un(op::SATURATE, e) } else { e };
            s.to_f16(e)
        };
        let dot: Option<Value> = match name {
            "v_dot2_f32_f16" | "v_dot2c_f32_f16" => {
                let c = if name == "v_dot2_f32_f16" {
                    let c = self.f(src(2));
                    if neg >> 2 & 1 == 1 {
                        self.fneg(c)
                    } else {
                        c
                    }
                } else {
                    self.f(cur)
                };
                let (a0, b0) = (fh(self, 0, false), fh(self, 1, false));
                let inner = self.fmad(a0, b0, c);
                let (a1, b1) = (fh(self, 0, true), fh(self, 1, true));
                let e = self.fmad(a1, b1, inner);
                let e = if clamp && name == "v_dot2_f32_f16" { self.un(op::SATURATE, e) } else { e };
                Some(self.bits(e))
            }
            "v_dot4c_i32_i8" => Some(self.dot_n(src(0), src(1), cur, 8, true, false)),
            "v_dot2_i32_i16" => {
                let (a0, b0, a1, b1) = (si(self, 0, false), si(self, 1, false), si(self, 0, true), si(self, 1, true));
                let p = self.mul(a0, b0);
                let q = self.mul(a1, b1);
                Some(self.dot_sum(&[p, q], src(2), true, clamp))
            }
            "v_dot2_u32_u16" => {
                let (a0, b0, a1, b1) = (h(self, 0, false), h(self, 1, false), h(self, 0, true), h(self, 1, true));
                let p = self.mul(a0, b0);
                let q = self.mul(a1, b1);
                Some(self.dot_sum(&[p, q], src(2), false, clamp))
            }
            "v_dot4_u32_u8" => Some(self.dot_n(src(0), src(1), src(2), 8, false, clamp)),
            "v_dot4_i32_i8" => Some(self.dot_n(src(0), src(1), src(2), 8, true, clamp)),
            "v_dot8_u32_u4" => Some(self.dot_n(src(0), src(1), src(2), 4, false, clamp)),
            "v_dot8_i32_i4" => Some(self.dot_n(src(0), src(1), src(2), 4, true, clamp)),
            _ => None,
        };
        if let Some(e) = dot {
            self.write32(&dst, e);
            return Ok(());
        }
        let mut halves = [z, z];
        for (k, hi) in [false, true].into_iter().enumerate() {
            let (a, b, c) = (h(self, 0, hi), h(self, 1, hi), h(self, 2, hi));
            let v = match name {
                "v_pk_add_f16" | "v_pk_mul_f16" | "v_pk_min_f16" | "v_pk_max_f16" => {
                    let (x, y) = (fh(self, 0, hi), fh(self, 1, hi));
                    let e = match name {
                        "v_pk_add_f16" => self.fadd(x, y),
                        "v_pk_mul_f16" => self.fmul(x, y),
                        "v_pk_min_f16" => self.fmin(x, y),
                        _ => self.fmax(x, y),
                    };
                    fr(self, e)
                }
                "v_pk_fma_f16" => {
                    let (x, y, w) = (fh(self, 0, hi), fh(self, 1, hi), fh(self, 2, hi));
                    let e = self.fmad(x, y, w);
                    fr(self, e)
                }
                "v_pk_fmac_f16" => {
                    let (x, y) = (fh(self, 0, hi), fh(self, 1, hi));
                    let ch = self.hsel(cur, hi);
                    let w = self.s.f16_to_f32(ch);
                    let e = self.fmad(x, y, w);
                    fr(self, e)
                }
                // clamp: saturate to the 16-bit range (computed exactly in 32 bits)
                "v_pk_add_u16" | "v_pk_sub_u16" | "v_pk_mad_u16" if clamp => {
                    let t = match name {
                        "v_pk_add_u16" => self.add(a, b),
                        "v_pk_sub_u16" => self.sub(a, b),
                        _ => {
                            let t = self.mul(a, b);
                            self.add(t, c)
                        }
                    };
                    let t = if name == "v_pk_sub_u16" {
                        let under = self.cmp(Pred::ULt, a, b);
                        self.selc(under, 0, t)
                    } else {
                        t
                    };
                    let k = self.c(0xFFFF);
                    self.umin(t, k)
                }
                "v_pk_add_i16" | "v_pk_sub_i16" | "v_pk_mad_i16" if clamp => {
                    let (x, y) = (si(self, 0, hi), si(self, 1, hi));
                    let t = match name {
                        "v_pk_add_i16" => self.add(x, y),
                        "v_pk_sub_i16" => self.sub(x, y),
                        _ => {
                            let w = si(self, 2, hi);
                            let t = self.mul(x, y);
                            self.add(t, w)
                        }
                    };
                    let (lo, hi) = (self.c(0xFFFF_8000), self.c(0x7FFF));
                    let t = self.imin(t, hi);
                    let t = self.imax(t, lo);
                    self.andc(t, 0xFFFF)
                }
                "v_pk_add_u16" | "v_pk_add_i16" => {
                    let t = self.add(a, b);
                    self.andc(t, 0xFFFF)
                }
                "v_pk_sub_u16" | "v_pk_sub_i16" => {
                    let t = self.sub(a, b);
                    self.andc(t, 0xFFFF)
                }
                "v_pk_mul_lo_u16" => {
                    let t = self.mul(a, b);
                    self.andc(t, 0xFFFF)
                }
                "v_pk_mad_u16" | "v_pk_mad_i16" => {
                    let t = self.mul(a, b);
                    let t = self.add(t, c);
                    self.andc(t, 0xFFFF)
                }
                "v_pk_lshlrev_b16" => {
                    let s = self.andc(a, 15);
                    let t = self.bin(BinOp::Shl, b, s);
                    self.andc(t, 0xFFFF)
                }
                "v_pk_lshrrev_b16" => {
                    let s = self.andc(a, 15);
                    self.bin(BinOp::LShr, b, s)
                }
                "v_pk_ashrrev_i16" => {
                    let x = si(self, 1, hi);
                    let s = self.andc(a, 15);
                    let t = self.bin(BinOp::AShr, x, s);
                    self.andc(t, 0xFFFF)
                }
                "v_pk_max_u16" => self.umax(a, b),
                "v_pk_min_u16" => self.umin(a, b),
                "v_pk_max_i16" | "v_pk_min_i16" => {
                    let (x, y) = (si(self, 0, hi), si(self, 1, hi));
                    let t = if name == "v_pk_max_i16" { self.imax(x, y) } else { self.imin(x, y) };
                    self.andc(t, 0xFFFF)
                }
                _ => return err(d, "VOP3P"),
            };
            halves[k] = v;
        }
        let hi = self.shlc(halves[1], 16);
        let r = self.or(halves[0], hi);
        self.write32(&dst, r);
        Ok(())
    }

    /// dot4_u8 / dot4_i8 / dot8_u4 / dot8_i4: c + Σ a[i]·b[i] over `bits`-wide fields.
    /// `clamp`: the sum saturates to the 32-bit range.
    fn dot_n(&mut self, a: Value, b: Value, c: Value, bits: u32, signed: bool, clamp: bool) -> Value {
        let mut prods = Vec::new();
        let mut i = 0;
        while i < 32 {
            let (x, y) = if signed {
                let x = self.shlc(a, 32 - bits - i);
                let x = self.ashrc(x, 32 - bits);
                let y = self.shlc(b, 32 - bits - i);
                let y = self.ashrc(y, 32 - bits);
                (x, y)
            } else {
                let x = self.lshrc(a, i);
                let x = self.andc(x, (1 << bits) - 1);
                let y = self.lshrc(b, i);
                let y = self.andc(y, (1 << bits) - 1);
                (x, y)
            };
            prods.push(self.mul(x, y));
            i += bits;
        }
        self.dot_sum(&prods, c, signed, clamp)
    }

    /// c + Σ products (each exact in 32 bits); `clamp`: in 64 bits, saturated to 32.
    fn dot_sum(&mut self, prods: &[Value], c: Value, signed: bool, clamp: bool) -> Value {
        if !clamp {
            return prods.iter().fold(c, |r, &p| self.add(r, p));
        }
        let ext = |s: &mut Self, v: Value| if signed { s.sext64(v) } else { s.zext64(v) };
        let mut r = ext(self, c);
        for &p in prods {
            let p = ext(self, p);
            r = self.add(r, p);
        }
        if signed {
            self.sat_i32(r)
        } else {
            let k = self.c64(0xFFFF_FFFF);
            let r = self.umin(r, k);
            self.trunc32(r)
        }
    }

    /// The highest VGPR the program names (relative moves index up to it).
    fn top_vgpr(&self) -> u64 {
        self.init.touched.iter().filter(|&&v| (VAR_VGPR..VAR_VGPR + 256).contains(&v)).map(|&v| (v - VAR_VGPR) as u64).max().unwrap_or(0)
    }

    pub fn movrel(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        if d.variant != Variant::Plain {
            return err(d, "movrel SDWA/DPP");
        }
        let src_code = Self::fv(d, "SRC0");
        let vdst = Self::fv(d, "VDST");
        let m0 = self.sreg(124);
        let top = self.top_vgpr().max(vdst).max(src_code.saturating_sub(256));
        let (rd_idx, wr_idx) = match name {
            "v_movrels_b32" => (Some(m0), None),
            "v_movreld_b32" => (None, Some(m0)),
            "v_movrelsd_b32" => (Some(m0), Some(m0)),
            "v_movrelsd_2_b32" => {
                let r = self.andc(m0, 0x3FF);
                let w = self.lshrc(m0, 16);
                let w = self.andc(w, 0x3FF);
                (Some(r), Some(w))
            }
            _ => return err(d, "movrel"),
        };
        let value = match rd_idx {
            Some(idx) => {
                if src_code < 256 {
                    return err(d, "movrels: the source is not a VGPR");
                }
                let base = src_code - 256;
                let mut v = self.c(0);
                for k in 0..=top.saturating_sub(base) {
                    let r = self.vreg(base + k);
                    let is = self.cmpc(Pred::Eq, idx, k as u32);
                    v = self.sel(is, r, v);
                }
                v
            }
            None => self.code32(src_code, d.literal, false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?,
        };
        let a = self.act();
        match wr_idx {
            Some(idx) => {
                for k in 0..=top.saturating_sub(vdst) {
                    let is = self.cmpc(Pred::Eq, idx, k as u32);
                    let c = self.land(a, is);
                    let old = self.vreg(vdst + k);
                    let v = self.sel(c, value, old);
                    self.wr(VAR_VGPR + (vdst + k) as u32, v);
                }
            }
            None => {
                let old = self.vreg(vdst);
                let v = self.sel(a, value, old);
                self.wr(VAR_VGPR + vdst as u32, v);
            }
        }
        Ok(())
    }

    pub fn swaprel(&mut self, d: &Decoded) -> LResult<()> {
        if d.variant != Variant::Plain {
            return err(d, "swaprel SDWA/DPP");
        }
        let vdst = Self::fv(d, "VDST");
        let src_code = Self::fv(d, "SRC0");
        if src_code < 256 {
            return err(d, "swaprel: the source is not a VGPR");
        }
        let base = src_code - 256;
        let top = self.top_vgpr().max(vdst).max(base);
        let m0 = self.sreg(124);
        let a = self.act();
        let td = self.vreg(vdst);
        let mut ts = self.c(0);
        for k in 0..=top.saturating_sub(base) {
            let r = self.vreg(base + k);
            let is = self.cmpc(Pred::Eq, m0, k as u32);
            ts = self.sel(is, r, ts);
        }
        let nv = self.sel(a, ts, td);
        self.wr(VAR_VGPR + vdst as u32, nv);
        for k in 0..=top.saturating_sub(base) {
            let is = self.cmpc(Pred::Eq, m0, k as u32);
            let c = self.land(a, is);
            let old = self.vreg(base + k);
            let v = self.sel(c, td, old);
            self.wr(VAR_VGPR + (base + k) as u32, v);
        }
        Ok(())
    }

    pub fn valu16(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let (outs, ins) = Self::ops(d);
        let m = Self::mods(d);
        let u: Vec<Value> = self.sources(d, &ins, name.contains("f16"))?.into_iter().map(Src::w).collect();
        let plain = d.variant == Variant::Plain;
        let z = self.c(0);
        let half = |s: &mut Self, i: usize| -> Value {
            let x = u.get(i).copied().unwrap_or(z);
            s.hsel(x, plain && m.opsel >> i & 1 == 1)
        };
        let fh = |s: &mut Self, i: usize| -> Value {
            let x = half(s, i);
            let mut v = s.s.f16_to_f32(x);
            if m.abs[i.min(2)] {
                v = s.fabs(v);
            }
            if m.neg[i.min(2)] {
                v = s.fneg(v);
            }
            v
        };
        let hb = |s: &mut Self, i: usize| -> Value {
            let mut x = half(s, i);
            if m.abs[i.min(2)] {
                x = s.andc(x, 0x7FFF);
            }
            if m.neg[i.min(2)] {
                let k = s.c(0x8000);
                x = s.xor(x, k);
            }
            x
        };
        let si = |s: &mut Self, i: usize| {
            let x = half(s, i);
            s.sx16(x)
        };
        let fr = |s: &mut Self, e: Value| -> Value {
            let mut v = match m.omod {
                1 => {
                    let k = s.fc(2.0);
                    s.fmul(e, k)
                }
                2 => {
                    let k = s.fc(4.0);
                    s.fmul(e, k)
                }
                3 => {
                    let k = s.fc(0.5);
                    s.fmul(e, k)
                }
                _ => e,
            };
            if m.clamp {
                v = s.un(op::SATURATE, v);
            }
            s.to_f16(v)
        };
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        let cur = match &dst {
            Dst::V(n) => self.vreg(*n),
            _ => z,
        };
        let curh = self.hsel(cur, plain && m.opsel >> 3 & 1 == 1);
        let (a, b, c) = (half(self, 0), half(self, 1), half(self, 2));
        macro_rules! f1 {
            ($f:expr) => {{
                let x = fh(self, 0);
                let e = $f(self, x);
                fr(self, e)
            }};
        }
        macro_rules! f2 {
            ($f:expr) => {{
                let x = fh(self, 0);
                let y = fh(self, 1);
                let e = $f(self, x, y);
                fr(self, e)
            }};
        }
        macro_rules! f3 {
            ($f:expr) => {{
                let x = fh(self, 0);
                let y = fh(self, 1);
                let w = fh(self, 2);
                let e = $f(self, x, y, w);
                fr(self, e)
            }};
        }
        let r: Value = match name {
            "v_add_f16" => f2!(|s: &mut Self, x, y| s.fadd(x, y)),
            "v_sub_f16" => f2!(|s: &mut Self, x, y| s.fsub(x, y)),
            "v_subrev_f16" => f2!(|s: &mut Self, x, y| s.fsub(y, x)),
            "v_mul_f16" => f2!(|s: &mut Self, x, y| s.fmul(x, y)),
            "v_min_f16" => f2!(|s: &mut Self, x, y| s.fmin(x, y)),
            "v_max_f16" => f2!(|s: &mut Self, x, y| s.fmax(x, y)),
            "v_fma_f16" => f3!(|s: &mut Self, x, y, w| s.fmad(x, y, w)),
            "v_med3_f16" => f3!(|s: &mut Self, x, y, w| s.med3f(x, y, w)),
            "v_min3_f16" => f3!(|s: &mut Self, x, y, w| {
                let t = s.fmin(x, y);
                s.fmin(t, w)
            }),
            "v_max3_f16" => f3!(|s: &mut Self, x, y, w| {
                let t = s.fmax(x, y);
                s.fmax(t, w)
            }),
            "v_med3_u16" => {
                let (x, y) = (self.umin(a, b), self.umax(a, b));
                let y = self.umin(y, c);
                self.umax(x, y)
            }
            "v_min3_u16" => {
                let t = self.umin(a, b);
                self.umin(t, c)
            }
            "v_max3_u16" => {
                let t = self.umax(a, b);
                self.umax(t, c)
            }
            "v_med3_i16" | "v_min3_i16" | "v_max3_i16" => {
                let (x, y, w) = (si(self, 0), si(self, 1), si(self, 2));
                let t = match name {
                    "v_med3_i16" => {
                        let (p, q) = (self.imin(x, y), self.imax(x, y));
                        let q = self.imin(q, w);
                        self.imax(p, q)
                    }
                    "v_min3_i16" => {
                        let t = self.imin(x, y);
                        self.imin(t, w)
                    }
                    _ => {
                        let t = self.imax(x, y);
                        self.imax(t, w)
                    }
                };
                self.andc(t, 0xFFFF)
            }
            "v_frexp_mant_f16" => f1!(|s: &mut Self, x| s.frexp_mant(x)),
            "v_frexp_exp_i16_f16" => {
                let x = fh(self, 0);
                let e = self.frexp_exp(x);
                self.andc(e, 0xFFFF)
            }
            "v_mad_f16" => {
                let (x, y, w) = (fh(self, 0), fh(self, 1), fh(self, 2));
                let p = self.fmul(x, y);
                let ph = if self.dyn_rm {
                    let rm = self.rd(super::lower::VAR_FP16_RM);
                    self.f32_to_f16_rm(p, rm)
                } else {
                    self.f32_to_f16_rne(p)
                };
                let p16 = self.s.f16_to_f32(ph);
                let e = self.fadd(p16, w);
                fr(self, e)
            }
            "v_fmac_f16" => {
                let (x, y) = (fh(self, 0), fh(self, 1));
                let w = self.s.f16_to_f32(curh);
                let e = self.fmad(x, y, w);
                fr(self, e)
            }
            "v_fmamk_f16" => {
                let (x, w) = (fh(self, 0), fh(self, 2));
                let k = self.andc(u[1], 0xFFFF);
                let k = self.s.f16_to_f32(k);
                let e = self.fmad(x, k, w);
                fr(self, e)
            }
            "v_fmaak_f16" => {
                let (x, y) = (fh(self, 0), fh(self, 1));
                let k = self.andc(u[2], 0xFFFF);
                let k = self.s.f16_to_f32(k);
                let e = self.fmad(x, y, k);
                fr(self, e)
            }
            "v_ldexp_f16" => {
                let x = fh(self, 0);
                let n = si(self, 1);
                let nf = self.sitof(n);
                let p = self.un(op::EXP, nf);
                let e = self.fmul(x, p);
                fr(self, e)
            }
            "v_rcp_f16" => f1!(|s: &mut Self, x| {
                let one = s.fc(1.0);
                s.fdiv(one, x)
            }),
            "v_rsq_f16" => f1!(|s: &mut Self, x| s.un(op::RSQRT, x)),
            "v_sqrt_f16" => f1!(|s: &mut Self, x| s.un(op::SQRT, x)),
            "v_log_f16" => f1!(|s: &mut Self, x| s.un(op::LOG, x)),
            "v_exp_f16" => f1!(|s: &mut Self, x| s.un(op::EXP, x)),
            "v_sin_f16" | "v_cos_f16" => {
                let x = fh(self, 0);
                let k = self.fc(6.283_185_3);
                let x = self.fmul(x, k);
                let e = self.un(if name == "v_sin_f16" { op::SIN } else { op::COS }, x);
                fr(self, e)
            }
            "v_fract_f16" => f1!(|s: &mut Self, x| s.fract_f(x)),
            "v_floor_f16" => f1!(|s: &mut Self, x| s.un(op::ROUND_NI, x)),
            "v_ceil_f16" => f1!(|s: &mut Self, x| s.un(op::ROUND_PI, x)),
            "v_trunc_f16" => f1!(|s: &mut Self, x| s.un(op::ROUND_Z, x)),
            "v_rndne_f16" => f1!(|s: &mut Self, x| s.un(op::ROUND_NE, x)),
            "v_cvt_f16_u16" => {
                let e = self.uitof(a);
                fr(self, e)
            }
            "v_cvt_f16_i16" => {
                let x = si(self, 0);
                let e = self.sitof(x);
                fr(self, e)
            }
            "v_cvt_u16_f16" => {
                let x = fh(self, 0);
                let v = self.cvt_u32(x);
                let big = self.cmpc(Pred::UGt, v, 0xFFFF);
                self.selc(big, 0xFFFF, v)
            }
            "v_cvt_i16_f16" => {
                let x = fh(self, 0);
                let v = self.cvt_i32(x);
                let lo = self.c(-32768i32 as u32);
                let hi = self.c(32767);
                let v = self.imax(v, lo);
                let v = self.imin(v, hi);
                self.andc(v, 0xFFFF)
            }
            "v_cvt_norm_u16_f16" => {
                let x = hb(self, 0);
                self.cvt_norm_u16_f16(x)
            }
            "v_cvt_norm_i16_f16" => {
                let x = hb(self, 0);
                self.cvt_norm_i16_f16(x)
            }
            "v_add_nc_u16" | "v_add_u16" | "v_add_nc_i16" => {
                let t = self.add(a, b);
                self.andc(t, 0xFFFF)
            }
            "v_sub_nc_u16" | "v_sub_u16" | "v_sub_nc_i16" => {
                let t = self.sub(a, b);
                self.andc(t, 0xFFFF)
            }
            "v_subrev_nc_u16" | "v_subrev_u16" => {
                let t = self.sub(b, a);
                self.andc(t, 0xFFFF)
            }
            "v_mul_lo_u16" => {
                let t = self.mul(a, b);
                self.andc(t, 0xFFFF)
            }
            "v_lshlrev_b16" => {
                let s = self.andc(a, 15);
                let t = self.bin(BinOp::Shl, b, s);
                self.andc(t, 0xFFFF)
            }
            "v_lshrrev_b16" => {
                let s = self.andc(a, 15);
                self.bin(BinOp::LShr, b, s)
            }
            "v_ashrrev_i16" => {
                let x = si(self, 1);
                let s = self.andc(a, 15);
                let t = self.bin(BinOp::AShr, x, s);
                self.andc(t, 0xFFFF)
            }
            "v_max_u16" => self.umax(a, b),
            "v_min_u16" => self.umin(a, b),
            "v_max_i16" | "v_min_i16" => {
                let (x, y) = (si(self, 0), si(self, 1));
                let t = if name == "v_max_i16" { self.imax(x, y) } else { self.imin(x, y) };
                self.andc(t, 0xFFFF)
            }
            "v_mad_u16" | "v_mad_i16" => {
                let t = self.mul(a, b);
                let t = self.add(t, c);
                self.andc(t, 0xFFFF)
            }
            _ => return err(d, "VALU 16-bit"),
        };
        if d.variant == Variant::Sdwa {
            self.vwrite(d, &dst, r);
        } else {
            let merged = if m.opsel >> 3 & 1 == 1 {
                let l = self.andc(cur, 0xFFFF);
                let h = self.shlc(r, 16);
                self.or(l, h)
            } else {
                let h = self.andc(cur, 0xFFFF_0000);
                self.or(h, r)
            };
            self.vwrite(d, &dst, merged);
        }
        Ok(())
    }

    /// `f32_to_f16_rm(f, 0)`: binary32 → binary16, round to nearest even (software).
    pub fn f32_to_f16_rne(&mut self, f: Value) -> Value {
        let z = self.c(0);
        self.f32_to_f16_rm(f, z)
    }

    /// `f32_to_f16_rm`: binary32 → binary16 under MODE[3:2] `rm` (0 RNE, 1 +inf, 2 -inf, 3 RTZ).
    pub fn f32_to_f16_rm(&mut self, f: Value, rm: Value) -> Value {
        let x = self.bits(f);
        let sign = self.lshrc(x, 16);
        let sign = self.andc(sign, 0x8000);
        let ax = self.andc(x, 0x7FFF_FFFF);
        let isinfnan = self.cmpc(Pred::UGe, ax, 0x7F80_0000);
        let isinf = self.cmpc(Pred::Eq, ax, 0x7F80_0000);
        let pl = self.lshrc(ax, 13);
        let pl = self.andc(pl, 0x1FF);
        let nanv = self.orc(pl, 0x7E00);
        let infnan = self.selc(isinf, 0x7C00, nanv);
        let infnan = self.or(sign, infnan);
        let zero = self.cmpc(Pred::Eq, ax, 0);
        let rne = self.cmpc(Pred::Eq, rm, 0);
        let pos = self.cmpc(Pred::Eq, sign, 0);
        let neg = self.lnot(pos);
        let up = self.cmpc(Pred::Eq, rm, 1);
        let down = self.cmpc(Pred::Eq, rm, 2);
        let dir_up = self.land(up, pos);
        let dir_dn = self.land(down, neg);
        let dir = self.lor(dir_up, dir_dn);
        let away = self.lor(rne, dir);
        let ovf = self.cmpc(Pred::UGe, ax, 0x4780_0000);
        let k7bff = self.c(0x7BFF);
        let ovm = self.selc(away, 0x7C00, k7bff);
        let ovfv = self.or(sign, ovm);
        let tiny = self.cmpc(Pred::ULt, ax, 0x3380_0000);
        let half_t = self.cmpc(Pred::UGt, ax, 0x3300_0000);
        let inc_t = self.sel(rne, half_t, dir);
        let inc_t = self.u(inc_t);
        let tinyv = self.or(sign, inc_t);
        let normal = self.cmpc(Pred::UGe, ax, 0x3880_0000);
        let e = self.lshrc(ax, 23);
        let k112 = self.c(112);
        let e = self.sub(e, k112);
        let e = self.shlc(e, 10);
        let mm = self.lshrc(ax, 13);
        let mm = self.andc(mm, 0x3FF);
        let hn = self.or(e, mm);
        let remn = self.andc(ax, 0x1FFF);
        let halfn = self.c(0x1000);
        let k126 = self.c(126);
        let ex = self.lshrc(ax, 23);
        let sh = self.sub(k126, ex);
        let sh = self.andc(sh, 31);
        let sig = self.andc(ax, 0x7F_FFFF);
        let sig = self.orc(sig, 0x80_0000);
        let hs = self.bin(BinOp::LShr, sig, sh);
        let one = self.c(1);
        let shm = self.sub(sh, one);
        let shm = self.andc(shm, 31);
        let halfs = self.bin(BinOp::Shl, one, shm);
        let full = self.bin(BinOp::Shl, one, sh);
        let msk = self.sub(full, one);
        let rems = self.and(sig, msk);
        let h = self.sel(normal, hn, hs);
        let rem = self.sel(normal, remn, rems);
        let hb = self.sel(normal, halfn, halfs);
        let gt = self.cmp(Pred::UGt, rem, hb);
        let eq = self.cmp(Pred::Eq, rem, hb);
        let odd = self.andc(h, 1);
        let odd = self.nz(odd);
        let tie = self.land(eq, odd);
        let rne_inc = self.lor(gt, tie);
        let rnz = self.nz(rem);
        let dir_inc = self.land(rnz, dir);
        let inc = self.sel(rne, rne_inc, dir_inc);
        let inc = self.u(inc);
        let r = self.add(h, inc);
        let r = self.or(sign, r);
        let r = self.sel(tiny, tinyv, r);
        let r = self.sel(ovf, ovfv, r);
        let r = self.sel(zero, sign, r);
        self.sel(isinfnan, infnan, r)
    }

    // ------------------------------------------------------------ 64-bit

    pub fn f64ty(&mut self) -> super::module::Ty {
        self.s.require(Feature::Doubles);
        self.s.m.f64()
    }
    pub fn dc(&mut self, x: f64) -> Value {
        let t = self.f64ty();
        self.s.m.const_float(t, x.to_bits())
    }
    pub fn dbits(&mut self, lo: u32, hi: u32) -> Value {
        let t = self.f64ty();
        self.s.m.const_float(t, (hi as u64) << 32 | lo as u64)
    }
    pub fn mkd(&mut self, lo: Value, hi: Value) -> Value {
        self.s.make_double(lo, hi)
    }

    pub fn valu64(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let (outs, ins) = Self::ops(d);
        let m = Self::mods(d);
        if d.variant != Variant::Plain {
            return err(d, "64-bit SDWA/DPP");
        }
        let src_f64 = name.ends_with("_f64");
        let mut lh: Vec<(Value, Value)> = Vec::new();
        for o in &ins {
            if o.size == 64 {
                lh.push(self.read64(d, o, src_f64).ok_or_else(|| format!("{}: op", rdna2::format(d)))?);
            } else {
                let v = self.read(d, o, false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
                let z = self.c(0);
                lh.push((v, z));
            }
        }
        let u64v = |s: &mut Self, i: usize| s.pack64(lh[i].0, lh[i].1);
        let f64v = |s: &mut Self, i: usize| {
            let mut hi = lh[i].1;
            if m.abs[i.min(2)] {
                hi = s.andc(hi, 0x7FFF_FFFF);
            }
            if m.neg[i.min(2)] {
                let k = s.c(0x8000_0000);
                hi = s.xor(hi, k);
            }
            s.mkd(lh[i].0, hi)
        };
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        if let Some(rest) = name.strip_prefix("v_cmpx_").or_else(|| name.strip_prefix("v_cmp_")) {
            let (opn, ty) = rest.rsplit_once('_').ok_or_else(|| format!("{}: VOPC", rdna2::format(d)))?;
            let dst = if name.starts_with("v_cmpx_") { Dst::S(126) } else { dst };
            if opn == "class" && ty == "f64" {
                let x = f64v(self, 0);
                let cls = self.fclass64(x);
                let s = self.bin(BinOp::LShr, lh[1].0, cls);
                let s = self.andc(s, 1);
                let cond = self.nz(s);
                self.write_mask(&dst, cond);
                return Ok(());
            }
            let cond = match ty {
                "f64" => {
                    let (a, b) = (f64v(self, 0), f64v(self, 1));
                    self.float_compare(opn, a, b).ok_or_else(|| format!("{}: VOPC 64", rdna2::format(d)))?
                }
                "u64" | "i64" => {
                    let (a, b) = (u64v(self, 0), u64v(self, 1));
                    let sg = ty == "i64";
                    match opn {
                        "f" => self.b(false),
                        "t" | "tru" => self.b(true),
                        "lt" => self.cmp(if sg { Pred::SLt } else { Pred::ULt }, a, b),
                        "eq" => self.cmp(Pred::Eq, a, b),
                        "le" => self.cmp(if sg { Pred::SLe } else { Pred::ULe }, a, b),
                        "gt" => self.cmp(if sg { Pred::SGt } else { Pred::UGt }, a, b),
                        "ne" | "lg" => self.cmp(Pred::Ne, a, b),
                        "ge" => self.cmp(if sg { Pred::SGe } else { Pred::UGe }, a, b),
                        _ => return err(d, "VOPC 64"),
                    }
                }
                _ => return err(d, "VOPC 64"),
            };
            self.write_mask(&dst, cond);
            return Ok(());
        }
        let r64: Value = match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let v = u64v(self, 1);
                let s = self.andc(lh[0].0, 63);
                let s = self.zext64(s);
                match name {
                    "v_lshlrev_b64" => self.bin(BinOp::Shl, v, s),
                    "v_lshrrev_b64" => self.bin(BinOp::LShr, v, s),
                    _ => self.bin(BinOp::AShr, v, s),
                }
            }
            "v_mad_u64_u32" | "v_mad_i64_i32" => {
                let p = if name == "v_mad_u64_u32" {
                    let (x, y) = (self.zext64(lh[0].0), self.zext64(lh[1].0));
                    self.mul(x, y)
                } else {
                    let (x, y) = (self.sext64(lh[0].0), self.sext64(lh[1].0));
                    self.mul(x, y)
                };
                let c = u64v(self, 2);
                let w = self.add(p, c);
                let carry = self.cmp(Pred::ULt, w, c);
                let sdst = outs.get(1).map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
                self.write_mask(&sdst, carry);
                let (lo, hi) = self.unpack64(w);
                self.write64(&dst, lo, hi);
                return Ok(());
            }
            _ => {
                let e = match name {
                    "v_add_f64" | "v_mul_f64" | "v_min_f64" | "v_max_f64" => {
                        let (a, b) = (f64v(self, 0), f64v(self, 1));
                        match name {
                            "v_add_f64" => self.fadd(a, b),
                            "v_mul_f64" => self.fmul(a, b),
                            "v_min_f64" => self.s.binary(Binary::FMin, a, b),
                            _ => self.s.binary(Binary::FMax, a, b),
                        }
                    }
                    "v_fma_f64" => {
                        let (a, b, c) = (f64v(self, 0), f64v(self, 1), f64v(self, 2));
                        self.s.tertiary(Tertiary::Fma, a, b, c)
                    }
                    "v_rcp_f64" => {
                        let a = f64v(self, 0);
                        let one = self.dc(1.0);
                        self.ddiv(one, a)
                    }
                    "v_cvt_f64_f32" => {
                        let t = self.f64ty();
                        let x = self.f(lh[0].0);
                        self.s.m.cast(CastOp::FPExt, x, t)
                    }
                    "v_cvt_f64_i32" => {
                        self.s.require(Feature::DoubleExtensions);
                        let t = self.f64ty();
                        self.s.m.cast(CastOp::SIToFP, lh[0].0, t)
                    }
                    "v_cvt_f64_u32" => {
                        self.s.require(Feature::DoubleExtensions);
                        let t = self.f64ty();
                        self.s.m.cast(CastOp::UIToFP, lh[0].0, t)
                    }
                    "v_floor_f64" => {
                        let a = f64v(self, 0);
                        self.floor_d(a)
                    }
                    "v_ceil_f64" => {
                        let a = f64v(self, 0);
                        self.ceil_d(a)
                    }
                    "v_fract_f64" => {
                        let a = f64v(self, 0);
                        self.fract_d(a)
                    }
                    "v_rndne_f64" => {
                        let a = f64v(self, 0);
                        self.round_ne_d(a)
                    }
                    "v_frexp_mant_f64" => {
                        let a = f64v(self, 0);
                        self.frexp_mant_d(a)
                    }
                    "v_sqrt_f64" => {
                        let a = f64v(self, 0);
                        self.sqrt_d(a)
                    }
                    "v_rsq_f64" => {
                        let a = f64v(self, 0);
                        let r = self.sqrt_d(a);
                        let one = self.dc(1.0);
                        self.ddiv(one, r)
                    }
                    "v_trunc_f64" => {
                        let a = f64v(self, 0);
                        self.trunc_d(a)
                    }
                    "v_ldexp_f64" => {
                        let a = f64v(self, 0);
                        self.ldexp_d(a, lh[1].0)
                    }
                    "v_cvt_f32_f64" => {
                        let a = f64v(self, 0);
                        let t = self.s.m.f32();
                        let f = self.s.m.cast(CastOp::FPTrunc, a, t);
                        let f = match m.omod {
                            1..=3 => {
                                let k = self.fc([2.0, 4.0, 0.5][m.omod as usize - 1]);
                                self.fmul(f, k)
                            }
                            _ => f,
                        };
                        let f = if m.clamp { self.s.funary(FUnary::Saturate, f) } else { f };
                        let b = self.bits(f);
                        self.write32(&dst, b);
                        return Ok(());
                    }
                    "v_cvt_i32_f64" | "v_cvt_u32_f64" => {
                        let a = f64v(self, 0);
                        let n = self.isnan_d(a);
                        let (lo, hi) = if name == "v_cvt_i32_f64" { (self.dc(-2147483648.0), self.dc(2147483647.0)) } else { (self.dc(0.0), self.dc(4294967295.0)) };
                        let x = self.s.binary(Binary::FMax, a, lo);
                        let x = self.s.binary(Binary::FMin, x, hi);
                        let i32 = self.s.m.i32();
                        self.s.require(Feature::DoubleExtensions);
                        let v = self.s.m.cast(if name == "v_cvt_i32_f64" { CastOp::FPToSI } else { CastOp::FPToUI }, x, i32);
                        let r = self.selc(n, 0, v);
                        self.write32(&dst, r);
                        return Ok(());
                    }
                    "v_frexp_exp_i32_f64" => {
                        let a = f64v(self, 0);
                        let r = self.frexp_exp_d(a);
                        self.write32(&dst, r);
                        return Ok(());
                    }
                    _ => return err(d, "VALU 64-bit"),
                };
                // output modifier, then clamp (as the f32 path)
                let e = match m.omod {
                    1..=3 => {
                        let k = self.dc([2.0, 4.0, 0.5][m.omod as usize - 1]);
                        self.fmul(e, k)
                    }
                    _ => e,
                };
                let e = if m.clamp { self.s.funary(FUnary::Saturate, e) } else { e };
                let (lo, hi) = self.s.split_double(e);
                self.write64(&dst, lo, hi);
                return Ok(());
            }
        };
        let (lo, hi) = self.unpack64(r64);
        self.write64(&dst, lo, hi);
        Ok(())
    }

    /// Double division (D3D11.1 double extensions).
    pub fn ddiv(&mut self, a: Value, b: Value) -> Value {
        self.s.require(Feature::DoubleExtensions);
        self.fdiv(a, b)
    }

    pub fn trunc_d(&mut self, x: Value) -> Value {
        let (l, h) = self.s.split_double(x);
        let e = self.lshrc(h, 20);
        let e = self.andc(e, 0x7FF);
        let big = self.cmpc(Pred::UGe, e, 1075);
        let small = self.cmpc(Pred::ULt, e, 1023);
        let sgn = self.andc(h, 0x8000_0000);
        let z = self.c(0);
        let zero = self.mkd(z, sgn);
        let k = self.c(1075);
        let s = self.sub(k, e);
        let s = self.andc(s, 63);
        let s64 = self.zext64(s);
        let one = self.c64(1);
        let m = self.bin(BinOp::Shl, one, s64);
        let m = self.sub(m, one);
        let m = self.not(m);
        let bits = self.pack64(l, h);
        let bits = self.and(bits, m);
        let (bl, bh) = self.unpack64(bits);
        let r = self.mkd(bl, bh);
        let r = self.sel(small, zero, r);
        self.sel(big, x, r)
    }
    pub fn floor_d(&mut self, x: Value) -> Value {
        let t = self.trunc_d(x);
        let gt = self.fcmp(Pred::FOGt, t, x);
        let one = self.dc(1.0);
        let t1 = self.fsub(t, one);
        self.sel(gt, t1, t)
    }
    pub fn ceil_d(&mut self, x: Value) -> Value {
        let t = self.trunc_d(x);
        let lt = self.fcmp(Pred::FOLt, t, x);
        let one = self.dc(1.0);
        let t1 = self.fadd(t, one);
        self.sel(lt, t1, t)
    }
    pub fn fract_d(&mut self, x: Value) -> Value {
        let n = self.isnan_d(x);
        let i = self.isinf_d(x);
        let fl = self.floor_d(x);
        let f = self.fsub(x, fl);
        let z = self.dc(0.0);
        let iz = self.fcmp(Pred::FOEq, f, z);
        let lim = self.dbits(0xFFFF_FFFF, 0x3FEF_FFFF);
        let mn = self.s.binary(Binary::FMin, f, lim);
        let r = self.sel(iz, z, mn);
        let qnan = self.dbits(0, 0x7FF8_0000);
        let r = self.sel(i, qnan, r);
        self.sel(n, x, r)
    }
    pub fn round_ne_d(&mut self, x: Value) -> Value {
        let (l, h) = self.s.split_double(x);
        let sign = self.andc(h, 0x8000_0000);
        let e = self.lshrc(h, 20);
        let e = self.andc(e, 0x7FF);
        let hm = self.andc(h, 0x7FFF_FFFF);
        let mag = self.pack64(l, hm);
        let keep = {
            let a = self.cmpc(Pred::Eq, e, 0x7FF);
            let b = self.cmpc(Pred::UGe, e, 1075);
            self.lor(a, b)
        };
        let z = self.c(0);
        let szero = self.mkd(z, sign);
        let tiny = self.cmpc(Pred::ULt, e, 1022);
        let is1022 = self.cmpc(Pred::Eq, e, 1022);
        let fm = self.c64(0x000F_FFFF_FFFF_FFFF);
        let fr = self.and(mag, fm);
        let z64 = self.c64(0);
        let above = self.cmp(Pred::Ne, fr, z64);
        let so = self.orc(sign, 0x3FF0_0000);
        let sone = self.mkd(z, so);
        let half_r = self.sel(above, sone, szero);
        let k = self.c(1075);
        let sh = self.sub(k, e);
        let sh = self.andc(sh, 63);
        let sh64 = self.zext64(sh);
        let one = self.c64(1);
        let unit = self.bin(BinOp::Shl, one, sh64);
        let mask = self.sub(unit, one);
        let k1 = self.c(1);
        let shm = self.sub(sh, k1);
        let shm = self.andc(shm, 63);
        let shm64 = self.zext64(shm);
        let half = self.bin(BinOp::Shl, one, shm64);
        let fraction = self.and(mag, mask);
        let nmask = self.not(mask);
        let whole = self.and(mag, nmask);
        let gt = self.cmp(Pred::UGt, fraction, half);
        let eq = self.cmp(Pred::Eq, fraction, half);
        let wb = self.bin(BinOp::LShr, whole, sh64);
        let wb = self.and(wb, one);
        let odd = self.cmp(Pred::Ne, wb, z64);
        let tie = self.land(eq, odd);
        let up = self.lor(gt, tie);
        let wu = self.add(whole, unit);
        let whole = self.sel(up, wu, whole);
        let (wl, wh) = self.unpack64(whole);
        let wh = self.or(sign, wh);
        let r = self.mkd(wl, wh);
        let r = self.sel(is1022, half_r, r);
        let r = self.sel(tiny, szero, r);
        self.sel(keep, x, r)
    }
    /// position of the top set bit of the 52-bit fraction (m:l)
    fn top52(&mut self, m: Value, l: Value) -> Value {
        let a = self.firstbithigh(m);
        let a = self.addc(a, 32);
        let b = self.firstbithigh(l);
        let mnz = self.nz(m);
        self.sel(mnz, a, b)
    }
    pub fn frexp_mant_d(&mut self, x: Value) -> Value {
        let (l, h) = self.s.split_double(x);
        let e = self.lshrc(h, 20);
        let e = self.andc(e, 0x7FF);
        let m = self.andc(h, 0xF_FFFF);
        let s = self.andc(h, 0x8000_0000);
        let special = self.cmpc(Pred::Eq, e, 0x7FF);
        let ez = self.cmpc(Pred::Eq, e, 0);
        let ml = self.or(m, l);
        let mz = self.cmpc(Pred::Eq, ml, 0);
        let zero = self.land(ez, mz);
        let keep = self.lor(special, zero);
        let nh = self.or(s, m);
        let nh = self.orc(nh, 0x3FE0_0000);
        let normal = self.mkd(l, nh);
        let top = self.top52(m, l);
        let f = self.pack64(l, m);
        let k = self.c(52);
        let sh = self.sub(k, top);
        let sh = self.andc(sh, 63);
        let sh = self.zext64(sh);
        let ff = self.bin(BinOp::Shl, f, sh);
        let fm = self.c64(0x000F_FFFF_FFFF_FFFF);
        let ff = self.and(ff, fm);
        let (fl, fh) = self.unpack64(ff);
        let dh = self.orc(s, 0x3FE0_0000);
        let dh = self.or(dh, fh);
        let den = self.mkd(fl, dh);
        let r = self.sel(ez, den, normal);
        self.sel(keep, x, r)
    }
    pub fn frexp_exp_d(&mut self, x: Value) -> Value {
        let (l, h) = self.s.split_double(x);
        let e = self.lshrc(h, 20);
        let e = self.andc(e, 0x7FF);
        let m = self.andc(h, 0xF_FFFF);
        let special = self.cmpc(Pred::Eq, e, 0x7FF);
        let ez = self.cmpc(Pred::Eq, e, 0);
        let ml = self.or(m, l);
        let mz = self.cmpc(Pred::Eq, ml, 0);
        let zero = self.land(ez, mz);
        let keep = self.lor(special, zero);
        let k = self.c(1022);
        let norm = self.sub(e, k);
        let top = self.top52(m, l);
        let k = self.c(1073);
        let den = self.sub(top, k);
        let r = self.sel(ez, den, norm);
        self.selc(keep, 0, r)
    }
    /// `ldexp_d`: x · 2^n rounded once (scaling steps that stay normal).
    pub fn ldexp_d(&mut self, x: Value, n: Value) -> Value {
        let n0 = self.isnan_d(x);
        let i0 = self.isinf_d(x);
        let z = self.dc(0.0);
        let x0 = self.fcmp(Pred::FOEq, x, z);
        let keep = self.lor(n0, i0);
        let keep = self.lor(keep, x0);
        let lo = self.c(-2200i32 as u32);
        let hi = self.c(2200);
        let n = self.imax(n, lo);
        let n = self.imin(n, hi);
        let up = self.dbits(0, 0x7FE0_0000);
        let down = self.dbits(0, 0x0360_0000);
        let k1023 = self.c(1023);
        let k969 = self.c(969);
        // first step
        let big = self.cmpc(Pred::SGt, n, 1023);
        let small = self.cmpc(Pred::SLt, n, -1022i32 as u32);
        let ru = self.fmul(x, up);
        let rd = self.fmul(x, down);
        let r1 = self.sel(small, rd, x);
        let r1 = self.sel(big, ru, r1);
        let nu = self.sub(n, k1023);
        let nd = self.add(n, k969);
        let n1 = self.sel(small, nd, n);
        let n1 = self.sel(big, nu, n1);
        // second step
        let big2 = self.cmpc(Pred::SGt, n1, 1023);
        let small2 = self.cmpc(Pred::SLt, n1, -1022i32 as u32);
        let big2 = self.land(big, big2);
        let small2 = self.land(small, small2);
        let ru2 = self.fmul(r1, up);
        let rd2 = self.fmul(r1, down);
        let r2 = self.sel(small2, rd2, r1);
        let r2 = self.sel(big2, ru2, r2);
        let nu2 = self.sub(n1, k1023);
        let nu2 = self.imin(nu2, k1023);
        let nd2 = self.add(n1, k969);
        let km = self.c(-1022i32 as u32);
        let nd2 = self.imax(nd2, km);
        let n2 = self.sel(small2, nd2, n1);
        let n2 = self.sel(big2, nu2, n2);
        // last step
        let e = self.add(n2, k1023);
        let e = self.shlc(e, 20);
        let zz = self.c(0);
        let p = self.mkd(zz, e);
        let r = self.fmul(r2, p);
        self.sel(keep, x, r)
    }
    /// `sqrt_d`: f32 root refined by two Newton steps in f64.
    pub fn sqrt_d(&mut self, x: Value) -> Value {
        let n = self.isnan_d(x);
        let z = self.dc(0.0);
        let zero = self.fcmp(Pred::FOEq, x, z);
        let inf = self.isinf_d(x);
        let pos = self.fcmp(Pred::FOGt, x, z);
        let pinf = self.land(inf, pos);
        let keep = self.lor(n, zero);
        let keep = self.lor(keep, pinf);
        let neg = self.fcmp(Pred::FOLt, x, z);
        let fe = self.frexp_exp_d(x);
        let k = self.ashrc(fe, 1);
        let zz = self.c(0);
        let k2 = self.sub(zz, k);
        let k2 = self.shlc(k2, 1);
        let s = self.ldexp_d(x, k2);
        let f32 = self.s.m.f32();
        let sf = self.s.m.cast(CastOp::FPTrunc, s, f32);
        let rf = self.un(op::SQRT, sf);
        let f64 = self.f64ty();
        let y = self.s.m.cast(CastOp::FPExt, rf, f64);
        // y = y + (s - y*y) / (y + y)
        let yy = self.fmul(y, y);
        let num = self.fsub(s, yy);
        let den = self.fadd(y, y);
        let q = self.ddiv(num, den);
        let y = self.fadd(y, q);
        // y = y + fma(-y, y, s) / (y + y)
        let ny = self.fneg(y);
        let res = self.s.tertiary(Tertiary::Fma, ny, y, s);
        let den = self.fadd(y, y);
        let q = self.ddiv(res, den);
        let y = self.fadd(y, q);
        let r = self.ldexp_d(y, k);
        let qnan = self.dbits(0, 0x7FF8_0000);
        let r = self.sel(neg, qnan, r);
        self.sel(keep, x, r)
    }
    pub fn fclass64(&mut self, f: Value) -> Value {
        let (l, h) = self.s.split_double(f);
        let e = self.lshrc(h, 20);
        let e = self.andc(e, 0x7FF);
        let m = self.andc(h, 0xF_FFFF);
        let m = self.or(m, l);
        let n = self.lshrc(h, 31);
        let q = self.andc(h, 0x8_0000);
        let qn = self.nz(q);
        let mnz = self.nz(m);
        let nn = self.nz(n);
        self.fclass_bits(e, 0x7FF, mnz, qn, nn)
    }
}
