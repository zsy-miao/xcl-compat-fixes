//! Vector ALU lowering: 32-bit integer and float operations, compares,
//! source / output modifiers, SDWA and DPP (port of `recomp/emit.rs` `valu`).

use super::lower::{err, ext_encoding, Dst, LResult, Lw, Mods, VAR_VGPR};
use super::module::{BinOp, Pred, Value};
use super::ops_core::op;
use crate::gpu::rdna2::{self, Decoded, Encoding, Operand, Variant};

/// A source operand: one dword, or a mask pair (SGPR pair / VCC).
#[derive(Clone, Copy)]
pub enum Src {
    W(Value),
    Pair(Value, Value),
}

impl Src {
    pub fn w(self) -> Value {
        match self {
            Src::W(v) => v,
            Src::Pair(l, _) => l,
        }
    }
}

impl Lw {
    pub fn mods(d: &Decoded) -> Mods {
        let fv = |e: &Encoding, n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
        let bit = |e: &Encoding, n: &str| fv(e, n) == 1;
        match d.variant {
            Variant::Sdwa => match ext_encoding(d) {
                Some(e) => Mods {
                    abs: [bit(e, "SRC0_ABS"), bit(e, "SRC1_ABS"), false],
                    neg: [bit(e, "SRC0_NEG"), bit(e, "SRC1_NEG"), false],
                    omod: fv(e, "OMOD"),
                    clamp: bit(e, "CLAMP"),
                    opsel: 0,
                },
                None => Mods::default(),
            },
            Variant::Dpp16 => match ext_encoding(d) {
                Some(e) => Mods {
                    abs: [bit(e, "SRC0_ABS"), bit(e, "SRC1_ABS"), false],
                    neg: [bit(e, "SRC0_NEG"), bit(e, "SRC1_NEG"), false],
                    ..Default::default()
                },
                None => Mods::default(),
            },
            Variant::Dpp8 => Mods::default(),
            Variant::Plain => {
                let e = d.enc;
                let (abs, neg) = (fv(e, "ABS"), fv(e, "NEG"));
                Mods {
                    abs: [abs & 1 == 1, abs >> 1 & 1 == 1, abs >> 2 & 1 == 1],
                    neg: [neg & 1 == 1, neg >> 1 & 1 == 1, neg >> 2 & 1 == 1],
                    omod: fv(e, "OMOD"),
                    clamp: bit(e, "CLAMP"),
                    opsel: fv(e, "OP_SEL"),
                }
            }
        }
    }

    /// Float view of a dword with abs / neg of source `i`.
    pub fn fsrc(&mut self, m: &Mods, i: usize, u: Value) -> Value {
        let mut s = self.f(u);
        if m.abs[i] {
            s = self.fabs(s);
        }
        if m.neg[i] {
            s = self.fneg(s);
        }
        s
    }

    /// Float result with output modifiers, as dword bits.
    pub fn fres(&mut self, m: &Mods, e: Value) -> Value {
        let mut s = match m.omod {
            1 => {
                let k = self.fc(2.0);
                self.fmul(e, k)
            }
            2 => {
                let k = self.fc(4.0);
                self.fmul(e, k)
            }
            3 => {
                let k = self.fc(0.5);
                self.fmul(e, k)
            }
            _ => e,
        };
        if m.clamp {
            s = self.un(op::SATURATE, s);
        }
        self.bits(s)
    }

    /// DPP16 source lane (i32, −1 = none) of lane `lane`.
    fn dpp16_src(&mut self, ctrl: u64) -> Option<Value> {
        let n = (ctrl & 15) as u32;
        let l = self.lane;
        Some(match ctrl {
            0x000..=0x0FF => {
                let a = self.andc(l, !3);
                let q = self.andc(l, 3);
                let sh = self.shlc(q, 1);
                let k = self.c(ctrl as u32);
                let s = self.bin(BinOp::LShr, k, sh);
                let s = self.andc(s, 3);
                self.or(a, s)
            }
            0x101..=0x10F => {
                let r = self.andc(l, 15);
                let r = self.addc(r, n);
                let ok = self.cmpc(Pred::ULt, r, 16);
                let t = self.addc(l, n);
                let m1 = self.c(u32::MAX);
                self.sel(ok, t, m1)
            }
            0x111..=0x11F => {
                let r = self.andc(l, 15);
                let ok = self.cmpc(Pred::UGe, r, n);
                let k = self.c(n);
                let t = self.sub(l, k);
                let m1 = self.c(u32::MAX);
                self.sel(ok, t, m1)
            }
            0x121..=0x12F => {
                let a = self.andc(l, !15);
                let r = self.andc(l, 15);
                let k = self.c(n);
                let r = self.sub(r, k);
                let r = self.andc(r, 15);
                self.or(a, r)
            }
            0x140 => {
                let a = self.andc(l, !15);
                let r = self.andc(l, 15);
                let k = self.c(15);
                let r = self.sub(k, r);
                self.or(a, r)
            }
            0x141 => {
                let a = self.andc(l, !15);
                let b8 = self.andc(l, 8);
                let r = self.andc(l, 7);
                let k = self.c(7);
                let r = self.sub(k, r);
                let x = self.or(a, b8);
                self.or(x, r)
            }
            0x150..=0x15F => {
                let a = self.andc(l, !15);
                let k = self.c(n);
                self.or(a, k)
            }
            0x160..=0x16F => {
                let k = self.c(n);
                self.xor(l, k)
            }
            _ => return None,
        })
    }

    /// Source dwords in operand order; SGPR-pair masks as `Src::Pair`. SDWA
    /// selects / extends its two sources, DPP reads src0 from another lane.
    pub fn sources(&mut self, d: &Decoded, ins: &[&Operand], src_f16: bool) -> LResult<Vec<Src>> {
        let mut u = Vec::new();
        for o in ins {
            if d.variant != Variant::Plain && o.field == Some("SRC0") {
                u.push(Src::W(self.c(0)));
                continue;
            }
            match o.ty {
                "OPR_SREG" | "OPR_VCC" if o.size == 64 => {
                    let (l, h) = self.read64(d, o, false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
                    u.push(Src::Pair(l, h));
                }
                _ if o.size == 64 => return err(d, "operand 64-bit"),
                _ => u.push(Src::W(self.read(d, o, src_f16).ok_or_else(|| format!("{}: op", rdna2::format(d)))?)),
            }
        }
        match d.variant {
            Variant::Plain => {}
            Variant::Sdwa => {
                let e = ext_encoding(d).ok_or_else(|| format!("{}: SDWA", rdna2::format(d)))?;
                let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
                let s0 = if fv("S0") == 1 { self.code32(fv("VSRC0"), None, src_f16).ok_or("op")? } else { self.vreg(fv("VSRC0")) };
                if !u.is_empty() {
                    u[0] = Src::W(self.sdwa_sel(s0, fv("SRC0_SEL") as u32, fv("SRC0_SEXT") == 1));
                }
                if e.field("VSRC1").is_some() && u.len() > 1 {
                    let v1 = fv("VSRC1");
                    let s1 = if fv("S1") == 1 { self.code32(v1, None, src_f16).ok_or("op")? } else { self.vreg(v1) };
                    u[1] = Src::W(self.sdwa_sel(s1, fv("SRC1_SEL") as u32, fv("SRC1_SEXT") == 1));
                }
            }
            Variant::Dpp16 | Variant::Dpp8 => {
                let e = ext_encoding(d).ok_or_else(|| format!("{}: DPP", rdna2::format(d)))?;
                let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
                let v = self.vreg(fv("VSRC0"));
                let (el, eh) = self.exec_mask();
                let t;
                if d.variant == Variant::Dpp16 {
                    let ctrl = fv("DPP_CTRL");
                    let (fi, bound) = (fv("FI") == 1, fv("BOUND_CTRL") == 1);
                    let l = self.dpp16_src(ctrl).ok_or_else(|| format!("{}: DPP_CTRL", rdna2::format(d)))?;
                    let z = self.c(0);
                    let valid = self.cmp(Pred::SGe, l, z);
                    let lc = self.imax(l, z);
                    let on = self.lanebit_at(el, eh, lc);
                    let fib = self.b(fi);
                    let ok0 = self.lor(fib, on);
                    let ok = self.land(valid, ok0);
                    let r = self.read_lane_half(v, lc);
                    let (rm, bm) = (fv("ROW_MASK") as u32, fv("BANK_MASK") as u32);
                    let lane = self.lane;
                    let row = self.lshrc(lane, 4);
                    let k = self.c(rm);
                    let rb = self.bin(BinOp::LShr, k, row);
                    let rb = self.andc(rb, 1);
                    let rok = self.nz(rb);
                    let bank = self.lshrc(lane, 2);
                    let bank = self.andc(bank, 3);
                    let k = self.c(bm);
                    let bb = self.bin(BinOp::LShr, k, bank);
                    let bb = self.andc(bb, 1);
                    let bok = self.nz(bb);
                    let mut en = self.land(rok, bok);
                    if bound {
                        t = self.sel(ok, r, z);
                    } else {
                        en = self.land(en, ok);
                        t = r;
                    }
                    self.dpp_enable = Some(en);
                } else {
                    let sel: u64 = (0..8).map(|i| fv(&format!("LANE_SEL_{i}")) << (3 * i)).sum();
                    let fi = rdna2::field_value(d.enc, &d.words, "SRC0").unwrap_or(0) == 0xEA;
                    let lane = self.lane;
                    let a = self.andc(lane, !7);
                    let q = self.andc(lane, 7);
                    let k3 = self.c(3);
                    let sh = self.mul(q, k3);
                    let k = self.c(sel as u32);
                    let s = self.bin(BinOp::LShr, k, sh);
                    let s = self.andc(s, 7);
                    let l = self.or(a, s);
                    let r = self.read_lane_half(v, l);
                    t = if !fi {
                        let on = self.lanebit_at(el, eh, l);
                        let z = self.c(0);
                        self.sel(on, r, z)
                    } else {
                        r
                    };
                }
                if !u.is_empty() {
                    u[0] = Src::W(t);
                }
            }
        }
        Ok(u)
    }

    pub fn sdwa_sel(&mut self, x: Value, sel: u32, sext: bool) -> Value {
        if sel >= 6 {
            return x;
        }
        let (sh, w) = if sel < 4 { (8 * sel, 8) } else { (16 * (sel - 4), 16) };
        let v = self.lshrc(x, sh);
        let v = self.andc(v, (1u32 << w) - 1);
        if sext {
            let s = self.shlc(v, 32 - w);
            self.ashrc(s, 32 - w)
        } else {
            v
        }
    }

    fn sdwa_dst(&mut self, r: Value, old: Value, sel: u32, unused: u32) -> Value {
        if sel >= 6 {
            return r;
        }
        let (sh, w) = if sel < 4 { (8 * sel, 8u32) } else { (16 * (sel - 4), 16u32) };
        let m = (((1u64 << w) - 1) as u32) << sh;
        let s = self.shlc(r, sh);
        let v = self.andc(s, m);
        match unused {
            2 => {
                let o = self.andc(old, !m);
                self.or(o, v)
            }
            1 => {
                let top = sh + w;
                if top >= 32 {
                    return v;
                }
                let b = self.lshrc(r, w - 1);
                let b = self.andc(b, 1);
                let neg = self.nz(b);
                let z = self.c(0);
                let ext = self.c(u32::MAX << top);
                let e = self.sel(neg, ext, z);
                self.or(v, e)
            }
            _ => v,
        }
    }

    /// Writes a VALU result, honouring SDWA dst_sel / dst_unused and DPP write enables.
    pub fn vwrite(&mut self, d: &Decoded, dst: &Dst, val: Value) {
        let mut val = val;
        if d.variant == Variant::Sdwa {
            if let (Some(e), Dst::V(n)) = (ext_encoding(d), dst) {
                let fv = |k: &str| rdna2::field_value(e, &d.words, k).unwrap_or(0);
                if e.field("DST_SEL").is_some() {
                    let cur = self.vreg(*n);
                    val = self.sdwa_dst(val, cur, fv("DST_SEL") as u32, fv("DST_UNUSED") as u32);
                }
            }
        }
        match (self.dpp_enable.take(), dst) {
            (Some(en), Dst::V(n)) => {
                let old = self.vreg(*n);
                let a = self.act();
                let c = self.land(a, en);
                let v = self.sel(c, val, old);
                self.wr(VAR_VGPR + *n as u32, v);
            }
            _ => self.write32(dst, val),
        }
    }

    pub fn valu(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        if name.starts_with("v_interp_") {
            return self.interp_gfx(d, name);
        }
        let (outs, ins) = Self::ops(d);
        const WIDE16: &[&str] = &[
            "v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32", "v_cvt_pknorm_i16_f16", "v_cvt_pknorm_u16_f16", "v_cvt_pk_u16_u32",
            "v_cvt_pk_i16_i32", "v_mad_u32_u16", "v_mad_i32_i16", "v_sad_u16", "v_sat_pk_u8_i16",
        ];
        let f16 = (name.contains("f16") || name.contains("_i16") || name.contains("_u16") || name.contains("_b16")) && !WIDE16.contains(&name);
        let cmp = name.starts_with("v_cmp");
        let src_f16 = matches!(name, "v_cvt_f32_f16" | "v_pack_b32_f16" | "v_cvt_pknorm_i16_f16" | "v_cvt_pknorm_u16_f16") || (cmp && name.ends_with("_f16"));
        let mix = name.starts_with("v_fma_mix");
        if name.starts_with("v_pk_") || name.starts_with("v_dot") {
            return self.valu_pk(d, name);
        }
        if name.starts_with("v_movrel") {
            return self.movrel(d, name);
        }
        if name == "v_swaprel_b32" {
            return self.swaprel(d);
        }
        if name == "v_swap_b32" && d.variant == Variant::Plain {
            let s0 = Self::fv(d, "SRC0");
            if s0 < 256 {
                return err(d, "v_swap_b32: SRC0 is not a VGPR");
            }
            let (xn, yn) = (Self::fv(d, "VDST"), s0 - 256);
            let (x, y) = (self.vreg(xn), self.vreg(yn));
            let a = self.act();
            let nx = self.sel(a, y, x);
            let ny = self.sel(a, x, y);
            self.wr(VAR_VGPR + xn as u32, nx);
            self.wr(VAR_VGPR + yn as u32, ny);
            return Ok(());
        }
        if d.enc.name == "ENC_VOP3P" && !mix {
            return err(d, "VOP3P");
        }
        if d.variant != Variant::Plain && ext_encoding(d).is_none() {
            return err(d, "SDWA/DPP in this encoding");
        }
        self.dpp_enable = None;
        if name.contains("f64") || name.contains("_i64") || name.contains("_u64") || name.contains("_b64") {
            return self.valu64(d, name);
        }
        if f16 && !mix && !cmp && !matches!(name, "v_cvt_f32_f16" | "v_cvt_f16_f32" | "v_cvt_pkrtz_f16_f32" | "v_pack_b32_f16") {
            return self.valu16(d, name);
        }
        let m = Self::mods(d);
        let u = self.sources(d, &ins, src_f16)?;
        if let Some(rest) = name.strip_prefix("v_cmpx_").or_else(|| name.strip_prefix("v_cmp_")) {
            return self.vopc(d, rest, name.starts_with("v_cmpx_"), &outs, &u, &m);
        }
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        let z = self.c(0);
        let cur = match &dst {
            Dst::V(n) => self.vreg(*n),
            _ => z,
        };
        let a = u.first().map(|s| s.w()).unwrap_or(z);
        let b = u.get(1).map(|s| s.w()).unwrap_or(z);
        let c = u.get(2).map(|s| s.w()).unwrap_or(z);
        let wmask = if self.wave64 { 63 } else { 31 };
        macro_rules! fs {
            ($i:expr) => {{
                let x = u.get($i).map(|s| s.w()).unwrap_or(z);
                self.fsrc(&m, $i, x)
            }};
        }
        macro_rules! fr {
            ($e:expr) => {{
                let e = $e;
                self.fres(&m, e)
            }};
        }
        let res: Value = match name {
            "v_mov_b32" | "v_mov_fed_b32" => a,
            "v_not_b32" => self.not(a),
            "v_bfrev_b32" => self.reversebits(a),
            "v_ffbh_u32" => self.lzcnt(a),
            "v_ffbl_b32" => self.firstbitlow(a),
            "v_ffbh_i32" => self.ffbh_i(a),
            "v_cvt_f32_i32" => fr!(self.sitof(a)),
            "v_cvt_f32_u32" => fr!(self.uitof(a)),
            "v_cvt_u32_f32" => {
                let x = fs!(0);
                self.cvt_u32(x)
            }
            "v_cvt_i32_f32" => {
                let x = fs!(0);
                self.cvt_i32(x)
            }
            "v_cvt_rpi_i32_f32" => {
                let x = fs!(0);
                let h = self.fc(0.5);
                let x = self.fadd(x, h);
                let x = self.un(op::ROUND_NI, x);
                self.cvt_i32(x)
            }
            "v_cvt_flr_i32_f32" => {
                let x = fs!(0);
                let x = self.un(op::ROUND_NI, x);
                self.cvt_i32(x)
            }
            "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1" | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let k = (name.as_bytes()[name.len() - 1] - b'0') as u32;
                let x = self.lshrc(a, 8 * k);
                let x = self.andc(x, 0xFF);
                fr!(self.uitof(x))
            }
            "v_cvt_f32_f16" => {
                let h = if m.opsel & 1 == 1 { self.lshrc(a, 16) } else { self.andc(a, 0xFFFF) };
                fr!(self.s.f16_to_f32(h))
            }
            "v_cvt_f16_f32" => {
                let x = fs!(0);
                // output modifier and clamp in f32, before the conversion
                let x = if m.omod != 0 || m.clamp {
                    let b = self.fres(&m, x);
                    self.f(b)
                } else {
                    x
                };
                let h = self.to_f16(x);
                if m.opsel >> 3 & 1 == 1 {
                    let l = self.andc(cur, 0xFFFF);
                    let hh = self.shlc(h, 16);
                    self.or(l, hh)
                } else if d.variant == Variant::Sdwa {
                    h
                } else {
                    let hi = self.andc(cur, 0xFFFF_0000);
                    self.or(hi, h)
                }
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                let l = self.f16_rtz(x);
                let h = self.f16_rtz(y);
                let h = self.shlc(h, 16);
                self.or(l, h)
            }
            "v_pack_b32_f16" => {
                let lo = if m.opsel & 1 == 1 { self.lshrc(a, 16) } else { self.andc(a, 0xFFFF) };
                let hi = if m.opsel & 2 == 2 { self.lshrc(b, 16) } else { self.andc(b, 0xFFFF) };
                let hi = self.shlc(hi, 16);
                self.or(lo, hi)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                let fv = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
                let (sel, hi2) = (fv("OP_SEL"), fv("OP_SEL_HI") | fv("OP_SEL_HI_2") << 2);
                let (neg, abs) = (fv("NEG"), fv("NEG_HI"));
                let mut srcs = Vec::new();
                for i in 0..3 {
                    let raw = u.get(i).map(|s| s.w()).unwrap_or(z);
                    let mut s = if hi2 >> i & 1 == 1 {
                        let h = if sel >> i & 1 == 1 { self.lshrc(raw, 16) } else { self.andc(raw, 0xFFFF) };
                        self.s.f16_to_f32(h)
                    } else {
                        self.f(raw)
                    };
                    if abs >> i & 1 == 1 {
                        s = self.fabs(s);
                    }
                    if neg >> i & 1 == 1 {
                        s = self.fneg(s);
                    }
                    srcs.push(s);
                }
                let mut e = self.fmad(srcs[0], srcs[1], srcs[2]);
                if fv("CLAMP") == 1 {
                    e = self.un(op::SATURATE, e);
                }
                match name {
                    "v_fma_mix_f32" => self.bits(e),
                    "v_fma_mixlo_f16" => {
                        let h = self.to_f16(e);
                        let hi = self.andc(cur, 0xFFFF_0000);
                        self.or(hi, h)
                    }
                    _ => {
                        let h = self.to_f16(e);
                        let h = self.shlc(h, 16);
                        let lo = self.andc(cur, 0xFFFF);
                        self.or(lo, h)
                    }
                }
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                let x = fs!(0);
                let one = self.fc(1.0);
                fr!(self.fdiv(one, x))
            }
            "v_rsq_f32" => fr!({
                let x = fs!(0);
                self.un(op::RSQRT, x)
            }),
            "v_sqrt_f32" => fr!({
                let x = fs!(0);
                self.un(op::SQRT, x)
            }),
            "v_exp_f32" => fr!({
                let x = fs!(0);
                self.un(op::EXP, x)
            }),
            "v_log_f32" => fr!({
                let x = fs!(0);
                self.un(op::LOG, x)
            }),
            "v_sin_f32" | "v_cos_f32" => {
                let x = fs!(0);
                let k = self.fc(6.283_185_3);
                let x = self.fmul(x, k);
                fr!(self.un(if name == "v_sin_f32" { op::SIN } else { op::COS }, x))
            }
            "v_fract_f32" => {
                let x = fs!(0);
                fr!(self.fract_f(x))
            }
            "v_trunc_f32" => fr!({
                let x = fs!(0);
                self.un(op::ROUND_Z, x)
            }),
            "v_floor_f32" => fr!({
                let x = fs!(0);
                self.un(op::ROUND_NI, x)
            }),
            "v_ceil_f32" => fr!({
                let x = fs!(0);
                self.un(op::ROUND_PI, x)
            }),
            "v_rndne_f32" => fr!({
                let x = fs!(0);
                self.un(op::ROUND_NE, x)
            }),
            "v_frexp_mant_f32" => {
                let x = fs!(0);
                fr!(self.frexp_mant(x))
            }
            "v_frexp_exp_i32_f32" => {
                let x = fs!(0);
                self.frexp_exp(x)
            }
            "v_ldexp_f32" => {
                let x = fs!(0);
                fr!(self.ldexp_f(x, b))
            }
            "v_add_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                fr!(self.fadd(x, y))
            }
            "v_sub_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                fr!(self.fsub(x, y))
            }
            "v_subrev_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                fr!(self.fsub(y, x))
            }
            "v_mul_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                fr!(self.fmul(x, y))
            }
            "v_mul_legacy_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                fr!(self.mul_legacy(x, y))
            }
            "v_min_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                fr!(self.fmin(x, y))
            }
            "v_max_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                fr!(self.fmax(x, y))
            }
            "v_fmac_f32" | "v_mac_f32" | "v_fmac_legacy_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                let cf = self.f(cur);
                fr!(match name {
                    "v_fmac_f32" => self.fmad(x, y, cf),
                    "v_mac_f32" => self.mad_nf(x, y, cf),
                    _ => self.fma_legacy(x, y, cf),
                })
            }
            "v_fma_f32" | "v_mad_f32" | "v_fma_legacy_f32" | "v_div_fmas_f32" => {
                let (x, y, w) = (fs!(0), fs!(1), fs!(2));
                fr!(match name {
                    "v_fma_f32" | "v_div_fmas_f32" => self.fmad(x, y, w),
                    "v_mad_f32" => self.mad_nf(x, y, w),
                    _ => self.fma_legacy(x, y, w),
                })
            }
            "v_fmamk_f32" | "v_madmk_f32" => {
                let (x, w) = (fs!(0), fs!(2));
                let k = self.f(b);
                fr!(if name == "v_fmamk_f32" { self.fmad(x, k, w) } else { self.mad_nf(x, k, w) })
            }
            "v_fmaak_f32" | "v_madak_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                let k = self.f(c);
                fr!(if name == "v_fmaak_f32" { self.fmad(x, y, k) } else { self.mad_nf(x, y, k) })
            }
            "v_min3_f32" => {
                let (x, y, w) = (fs!(0), fs!(1), fs!(2));
                let t = self.fmin(x, y);
                fr!(self.fmin(t, w))
            }
            "v_max3_f32" => {
                let (x, y, w) = (fs!(0), fs!(1), fs!(2));
                let t = self.fmax(x, y);
                fr!(self.fmax(t, w))
            }
            "v_med3_f32" => {
                let (x, y, w) = (fs!(0), fs!(1), fs!(2));
                fr!(self.med3f(x, y, w))
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                let (x, y, w) = (fs!(0), fs!(1), fs!(2));
                fr!(self.cube(name, x, y, w))
            }
            "v_div_fixup_f32" => {
                let (q, dd, nn) = (fs!(0), fs!(1), fs!(2));
                fr!(self.div_fixup(q, dd, nn))
            }
            // clamp: unsigned saturation
            "v_add_nc_u32" | "v_add3_u32" if m.clamp => {
                let t = self.add(a, b);
                let over = self.cmp(Pred::ULt, t, a);
                let (t, over) = if name == "v_add3_u32" {
                    let u = self.add(t, c);
                    let o2 = self.cmp(Pred::ULt, u, t);
                    (u, self.lor(over, o2))
                } else {
                    (t, over)
                };
                let m1 = self.c(u32::MAX);
                self.sel(over, m1, t)
            }
            "v_sub_nc_u32" | "v_subrev_nc_u32" if m.clamp => {
                let (x, y) = if name == "v_sub_nc_u32" { (a, b) } else { (b, a) };
                let under = self.cmp(Pred::ULt, x, y);
                let t = self.sub(x, y);
                let z = self.c(0);
                self.sel(under, z, t)
            }
            "v_add_nc_u32" => self.add(a, b),
            "v_sub_nc_u32" => self.sub(a, b),
            "v_subrev_nc_u32" => self.sub(b, a),
            "v_mul_lo_u32" => self.mul(a, b),
            "v_mul_hi_u32" => self.mul_hi(a, b, false),
            "v_mul_hi_i32" => self.mul_hi(a, b, true),
            "v_mul_u32_u24" => {
                let (x, y) = (self.andc(a, 0xFF_FFFF), self.andc(b, 0xFF_FFFF));
                self.mul(x, y)
            }
            "v_mul_hi_u32_u24" => {
                let (x, y) = (self.andc(a, 0xFF_FFFF), self.andc(b, 0xFF_FFFF));
                self.mul_hi(x, y, false)
            }
            "v_mul_i32_i24" => {
                let (x, y) = (self.i24(a), self.i24(b));
                self.mul(x, y)
            }
            "v_mul_hi_i32_i24" => {
                let (x, y) = (self.i24(a), self.i24(b));
                self.mul_hi(x, y, true)
            }
            "v_mad_u32_u24" => {
                let (x, y) = (self.andc(a, 0xFF_FFFF), self.andc(b, 0xFF_FFFF));
                let p = self.mul(x, y);
                self.add(p, c)
            }
            "v_mad_i32_i24" => {
                let (x, y) = (self.i24(a), self.i24(b));
                let p = self.mul(x, y);
                self.add(p, c)
            }
            "v_and_b32" => self.and(a, b),
            "v_or_b32" => self.or(a, b),
            "v_xor_b32" => self.xor(a, b),
            "v_xnor_b32" => {
                let x = self.xor(a, b);
                self.not(x)
            }
            "v_lshlrev_b32" => self.shl(b, a),
            "v_lshrrev_b32" => self.lshr(b, a),
            "v_ashrrev_i32" => self.ashr(b, a),
            "v_lshl_add_u32" => {
                let x = self.shl(a, b);
                self.add(x, c)
            }
            "v_add_lshl_u32" => {
                let x = self.add(a, b);
                self.shl(x, c)
            }
            "v_lshl_or_b32" => {
                let x = self.shl(a, b);
                self.or(x, c)
            }
            "v_and_or_b32" => {
                let x = self.and(a, b);
                self.or(x, c)
            }
            "v_or3_b32" => {
                let x = self.or(a, b);
                self.or(x, c)
            }
            "v_xor3_b32" => {
                let x = self.xor(a, b);
                self.xor(x, c)
            }
            "v_add3_u32" => {
                let x = self.add(a, b);
                self.add(x, c)
            }
            "v_xad_u32" => {
                let x = self.xor(a, b);
                self.add(x, c)
            }
            "v_bfe_u32" => self.bfe_u(a, b, c),
            "v_bfe_i32" => self.bfe_i(a, b, c),
            "v_bfi_b32" => {
                let x = self.and(a, b);
                let n = self.not(a);
                let y = self.and(n, c);
                self.or(x, y)
            }
            "v_bfm_b32" => self.bfm(a, b),
            "v_bcnt_u32_b32" => {
                let n = self.countbits(a);
                self.add(n, b)
            }
            "v_perm_b32" => self.perm_b32(a, b, c),
            "v_alignbit_b32" => self.alignbit(a, b, c),
            "v_alignbyte_b32" => {
                let s = self.andc(c, 3);
                let s = self.shlc(s, 3);
                self.alignbit(a, b, s)
            }
            "v_min_u32" => self.umin(a, b),
            "v_max_u32" => self.umax(a, b),
            "v_min_i32" => self.imin(a, b),
            "v_max_i32" => self.imax(a, b),
            "v_min3_u32" => {
                let t = self.umin(a, b);
                self.umin(t, c)
            }
            "v_max3_u32" => {
                let t = self.umax(a, b);
                self.umax(t, c)
            }
            "v_med3_u32" => {
                let (x, y) = (self.umin(a, b), self.umax(a, b));
                let y = self.umin(y, c);
                self.umax(x, y)
            }
            "v_min3_i32" => {
                let t = self.imin(a, b);
                self.imin(t, c)
            }
            "v_max3_i32" => {
                let t = self.imax(a, b);
                self.imax(t, c)
            }
            "v_med3_i32" => {
                let (x, y) = (self.imin(a, b), self.imax(a, b));
                let y = self.imin(y, c);
                self.imax(x, y)
            }
            "v_sad_u32" => {
                let (x, y) = (self.umax(a, b), self.umin(a, b));
                let t = self.sub(x, y);
                self.add(t, c)
            }
            "v_sad_u8" | "v_sad_hi_u8" => {
                let s = self.sad_u8(a, b);
                let s = if name == "v_sad_hi_u8" { self.shlc(s, 16) } else { s };
                self.add(s, c)
            }
            "v_sad_u16" => {
                let x = self.absd(a, b, 0, 0xFFFF);
                let y = self.absd(a, b, 16, 0xFFFF);
                let t = self.add(x, y);
                self.add(t, c)
            }
            "v_msad_u8" => {
                let s = self.msad_u8(a, b);
                self.add(s, c)
            }
            "v_lerp_u8" => self.lerp_u8(a, b, c),
            "v_add_nc_i32" | "v_sub_nc_i32" if m.clamp => {
                let (x, y) = (self.sext64(a), self.sext64(b));
                let r = if name == "v_add_nc_i32" { self.add(x, y) } else { self.sub(x, y) };
                self.sat_i32(r)
            }
            "v_add_nc_i32" => self.add(a, b),
            "v_sub_nc_i32" => self.sub(a, b),
            "v_mad_u32_u16" => {
                let x = self.lshrc(a, 16 * (m.opsel & 1) as u32);
                let x = self.andc(x, 0xFFFF);
                let y = self.lshrc(b, 8 * (m.opsel & 2) as u32);
                let y = self.andc(y, 0xFFFF);
                let p = self.mul(x, y);
                self.add(p, c)
            }
            "v_mad_i32_i16" => {
                let x = self.lshrc(a, 16 * (m.opsel & 1) as u32);
                let x = self.shlc(x, 16);
                let x = self.ashrc(x, 16);
                let y = self.lshrc(b, 8 * (m.opsel & 2) as u32);
                let y = self.shlc(y, 16);
                let y = self.ashrc(y, 16);
                let p = self.mul(x, y);
                self.add(p, c)
            }
            "v_cvt_pk_u16_u32" => {
                let k = self.c(0xFFFF);
                let x = self.umin(a, k);
                let y = self.umin(b, k);
                let y = self.shlc(y, 16);
                self.or(x, y)
            }
            "v_cvt_pk_i16_i32" => {
                let (lo, hi) = (self.c(-32768i32 as u32), self.c(32767));
                let x = self.imax(a, lo);
                let x = self.imin(x, hi);
                let x = self.andc(x, 0xFFFF);
                let y = self.imax(b, lo);
                let y = self.imin(y, hi);
                let y = self.andc(y, 0xFFFF);
                let y = self.shlc(y, 16);
                self.or(x, y)
            }
            "v_cvt_pknorm_i16_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                let l = self.cvt_norm_i16_f32(x);
                let h = self.cvt_norm_i16_f32(y);
                let h = self.shlc(h, 16);
                self.or(l, h)
            }
            "v_cvt_pknorm_u16_f32" => {
                let (x, y) = (fs!(0), fs!(1));
                let l = self.cvt_norm_u16_f32(x);
                let h = self.cvt_norm_u16_f32(y);
                let h = self.shlc(h, 16);
                self.or(l, h)
            }
            "v_cvt_pknorm_i16_f16" | "v_cvt_pknorm_u16_f16" => {
                let mut hb = [z, z];
                for (i, slot) in hb.iter_mut().enumerate() {
                    let s = u.get(i).map(|s| s.w()).unwrap_or(z);
                    let mut s = if m.opsel >> i & 1 == 1 { self.lshrc(s, 16) } else { self.andc(s, 0xFFFF) };
                    if m.abs[i] {
                        s = self.andc(s, 0x7FFF);
                    }
                    if m.neg[i] {
                        let k = self.c(0x8000);
                        s = self.xor(s, k);
                    }
                    *slot = s;
                }
                let (l, h) = if name == "v_cvt_pknorm_i16_f16" {
                    (self.cvt_norm_i16_f16(hb[0]), self.cvt_norm_i16_f16(hb[1]))
                } else {
                    (self.cvt_norm_u16_f16(hb[0]), self.cvt_norm_u16_f16(hb[1]))
                };
                let h = self.shlc(h, 16);
                self.or(l, h)
            }
            "v_sat_pk_u8_i16" => {
                let (lo, hi) = (self.c(0), self.c(255));
                let x = self.shlc(a, 16);
                let x = self.ashrc(x, 16);
                let x = self.imax(x, lo);
                let x = self.imin(x, hi);
                let y = self.ashrc(a, 16);
                let y = self.imax(y, lo);
                let y = self.imin(y, hi);
                let y = self.shlc(y, 8);
                self.or(x, y)
            }
            "v_cvt_pk_u8_f32" => {
                let x = fs!(0);
                let r = self.un(op::ROUND_NE, x);
                let (lo, hi) = (self.fc(0.0), self.fc(255.0));
                let r = self.fmax(r, lo);
                let r = self.fmin(r, hi);
                let v = self.ftoui(r);
                let sel = self.andc(b, 3);
                let sh = self.shlc(sel, 3);
                let k = self.c(0xFF);
                let msk = self.shl(k, sh);
                let nm = self.not(msk);
                let keep = self.and(c, nm);
                let v = self.shl(v, sh);
                self.or(keep, v)
            }
            "v_mbcnt_lo_u32_b32" => {
                let m = self.lanemask_lt(false);
                let x = self.and(a, m);
                let n = self.countbits(x);
                self.add(n, b)
            }
            "v_mbcnt_hi_u32_b32" => {
                let m = self.lanemask_lt(true);
                let x = self.and(a, m);
                let n = self.countbits(x);
                self.add(n, b)
            }
            "v_cndmask_b32" => {
                let Some(Src::Pair(l, h)) = u.get(2).copied() else { return err(d, "mask") };
                let bit = self.lane_of(l, h);
                self.sel(bit, b, a)
            }
            "v_permlane16_b32" | "v_permlanex16_b32" => {
                let (fi, bound) = (m.opsel & 1 == 1, m.opsel & 2 == 2);
                let lane = self.lane;
                let r = self.andc(lane, 15);
                let lo8 = self.cmpc(Pred::ULt, r, 8);
                let w = self.sel(lo8, b, c);
                let q = self.andc(lane, 7);
                let sh = self.shlc(q, 2);
                let nsel = self.bin(BinOp::LShr, w, sh);
                let nsel = self.andc(nsel, 15);
                let base = self.andc(lane, !31);
                let row = self.andc(lane, 16);
                let row = if name == "v_permlanex16_b32" {
                    let k = self.c(16);
                    let x = self.xor(row, k);
                    self.andc(x, 16)
                } else {
                    row
                };
                let l = self.or(base, row);
                let l = self.or(l, nsel);
                let (el, eh) = self.exec_mask();
                let on = self.lanebit_at(el, eh, l);
                let fib = self.b(fi);
                let ok = self.lor(fib, on);
                let v = self.read_lane_half(a, l);
                if bound {
                    self.sel(ok, v, z)
                } else {
                    self.sel(ok, v, cur)
                }
            }
            "v_readfirstlane_b32" => {
                let (el, eh) = self.exec_mask();
                let l = self.first_lane(el, eh);
                let t = self.read_lane_uniform(a, l);
                self.write32(&dst, t);
                return Ok(());
            }
            "v_readlane_b32" => {
                let l = self.andc(b, wmask);
                let t = self.read_lane_uniform(a, l);
                self.write32(&dst, t);
                return Ok(());
            }
            "v_writelane_b32" => {
                if let Dst::V(n) = dst {
                    let l = self.andc(b, wmask);
                    let lane = self.lane;
                    let me = self.cmp(Pred::Eq, lane, l);
                    let v = self.sel(me, a, cur);
                    self.wr(VAR_VGPR + n as u32, v);
                }
                return Ok(());
            }
            "v_div_scale_f32" => {
                let sdst = outs.get(1).map(|o| Self::dst_of(d, o)).unwrap_or(Dst::S(106));
                let f = self.b(false);
                self.write_mask(&sdst, f);
                let x = fs!(0);
                let r = self.bits(x);
                self.vwrite(d, &dst, r);
                return Ok(());
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" => {
                let (x, y) = if name == "v_subrev_co_u32" { (b, a) } else { (a, b) };
                let (t, carry) = if name == "v_add_co_u32" {
                    let t = self.add(x, y);
                    (t, self.cmp(Pred::ULt, t, x))
                } else {
                    let t = self.sub(x, y);
                    (t, self.cmp(Pred::UGt, y, x))
                };
                let sdst = outs.get(1).map(|o| Self::dst_of(d, o)).unwrap_or(Dst::S(106));
                self.write_mask(&sdst, carry);
                self.vwrite(d, &dst, t);
                return Ok(());
            }
            "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                let Some(Src::Pair(l, h)) = u.get(2).copied() else { return err(d, "mask") };
                let cb = self.lane_of(l, h);
                let cin = self.u(cb);
                let (x, y) = if name == "v_subrev_co_ci_u32" { (b, a) } else { (a, b) };
                let (x6, y6, c6) = (self.zext64(x), self.zext64(y), self.zext64(cin));
                let w = if name == "v_add_co_ci_u32" {
                    let s = self.add(x6, y6);
                    self.add(s, c6)
                } else {
                    let s = self.sub(x6, y6);
                    self.sub(s, c6)
                };
                let (t, hi) = self.unpack64(w);
                let carry = self.nz(hi);
                let sdst = outs.get(1).map(|o| Self::dst_of(d, o)).unwrap_or(Dst::S(106));
                self.write_mask(&sdst, carry);
                self.vwrite(d, &dst, t);
                return Ok(());
            }
            _ => return err(d, "VALU"),
        };
        self.vwrite(d, &dst, res);
        Ok(())
    }

    pub fn vopc(&mut self, d: &Decoded, rest: &str, cmpx: bool, outs: &[&Operand], u: &[Src], m: &Mods) -> LResult<()> {
        let (opn, ty) = rest.rsplit_once('_').ok_or_else(|| format!("{}: VOPC", rdna2::format(d)))?;
        if u.len() < 2 {
            return err(d, "VOPC");
        }
        let (mut ua, mut ub) = (u[0].w(), u[1].w());
        let half = |s: &mut Self, i: u64, v: Value| if m.opsel >> i & 1 == 1 { s.lshrc(v, 16) } else { s.andc(v, 0xFFFF) };
        let ty = match ty {
            "f16" => {
                if opn == "class" {
                    let mut h = half(self, 0, ua);
                    if m.abs[0] {
                        h = self.andc(h, 0x7FFF);
                    }
                    if m.neg[0] {
                        let k = self.c(0x8000);
                        h = self.xor(h, k);
                    }
                    let cls = self.fclass16(h);
                    let s = self.bin(BinOp::LShr, ub, cls);
                    let s = self.andc(s, 1);
                    let cond = self.nz(s);
                    return self.vopc_write(d, cmpx, outs, cond);
                }
                let ha = half(self, 0, ua);
                let fa = self.s.f16_to_f32(ha);
                ua = self.bits(fa);
                let hb = half(self, 1, ub);
                let fb = self.s.f16_to_f32(hb);
                ub = self.bits(fb);
                "f32"
            }
            "u16" => {
                ua = half(self, 0, ua);
                ub = half(self, 1, ub);
                "u32"
            }
            "i16" => {
                let x = half(self, 0, ua);
                let x = self.shlc(x, 16);
                ua = self.ashrc(x, 16);
                let y = half(self, 1, ub);
                let y = self.shlc(y, 16);
                ub = self.ashrc(y, 16);
                "i32"
            }
            t => t,
        };
        let cond = match ty {
            "f32" => {
                let a = self.fsrc(m, 0, ua);
                if opn == "class" {
                    let cls = self.fclass(a);
                    let s = self.bin(BinOp::LShr, ub, cls);
                    let s = self.andc(s, 1);
                    self.nz(s)
                } else {
                    let b = self.fsrc(m, 1, ub);
                    self.float_compare(opn, a, b).ok_or_else(|| format!("{}: VOPC f32", rdna2::format(d)))?
                }
            }
            "i32" | "u32" => {
                let s = ty == "i32";
                match opn {
                    "f" => self.b(false),
                    "t" | "tru" => self.b(true),
                    "lt" => self.cmp(if s { Pred::SLt } else { Pred::ULt }, ua, ub),
                    "eq" => self.cmp(Pred::Eq, ua, ub),
                    "le" => self.cmp(if s { Pred::SLe } else { Pred::ULe }, ua, ub),
                    "gt" => self.cmp(if s { Pred::SGt } else { Pred::UGt }, ua, ub),
                    "ne" | "lg" => self.cmp(Pred::Ne, ua, ub),
                    "ge" => self.cmp(if s { Pred::SGe } else { Pred::UGe }, ua, ub),
                    _ => return err(d, "VOPC int"),
                }
            }
            _ => return err(d, "VOPC type"),
        };
        self.vopc_write(d, cmpx, outs, cond)
    }

    /// The recompiler's `float_compare`: NaN tested explicitly.
    pub fn float_compare(&mut self, opn: &str, a: Value, b: Value) -> Option<Value> {
        let rel = |s: &mut Self, r: &str| -> Option<Value> {
            Some(match r {
                "lt" => s.fcmp(Pred::FOLt, a, b),
                "eq" => s.fcmp(Pred::FOEq, a, b),
                "le" => s.fcmp(Pred::FOLe, a, b),
                "gt" => s.fcmp(Pred::FOGt, a, b),
                "ge" => s.fcmp(Pred::FOGe, a, b),
                "lg" => {
                    let x = s.fcmp(Pred::FOLt, a, b);
                    let y = s.fcmp(Pred::FOGt, a, b);
                    s.lor(x, y)
                }
                _ => return None,
            })
        };
        let nan = |s: &mut Self| {
            let x = s.isnan_any(a);
            let y = s.isnan_any(b);
            s.lor(x, y)
        };
        Some(match opn {
            "f" => self.b(false),
            "t" | "tru" => self.b(true),
            "o" => {
                let n = nan(self);
                self.lnot(n)
            }
            "u" => nan(self),
            _ => match opn.strip_prefix('n').filter(|r| matches!(*r, "lt" | "eq" | "le" | "gt" | "ge" | "lg")) {
                Some(r) => {
                    let n = nan(self);
                    let x = rel(self, r)?;
                    let nx = self.lnot(x);
                    self.lor(n, nx)
                }
                None => {
                    let n = nan(self);
                    let nn = self.lnot(n);
                    let x = rel(self, opn)?;
                    self.land(nn, x)
                }
            },
        })
    }

    /// IsNaN of a float or double (doubles from their bits: DXC narrows `isnan`).
    pub fn isnan_any(&mut self, x: Value) -> Value {
        let t = self.s.m.type_of(x);
        if t == self.s.m.f64() {
            self.isnan_d(x)
        } else {
            self.isnan(x)
        }
    }

    pub fn vopc_write(&mut self, d: &Decoded, cmpx: bool, outs: &[&Operand], cond: Value) -> LResult<()> {
        let mut dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::S(106));
        if d.variant == Variant::Sdwa {
            if let Some(e) = ext_encoding(d) {
                if rdna2::field_value(e, &d.words, "SD") == Some(1) {
                    let sd = rdna2::field_value(e, &d.words, "SDST").unwrap_or(106);
                    dst = if sd == 125 { Dst::Null } else { Dst::S(sd) };
                }
            }
        }
        if cmpx {
            dst = Dst::S(126);
        }
        self.write_mask(&dst, cond);
        Ok(())
    }

    /// lanemask_lt_lo / _hi words for this lane.
    pub fn lanemask_lt(&mut self, hi: bool) -> Value {
        let l = self.lane;
        let all = self.c(u32::MAX);
        if !hi {
            // lane >= 32 ? ~0 : (lane == 0 ? 0 : ~0 >> (32 - lane))
            let k = self.c(32);
            let s = self.sub(k, l);
            let s = self.andc(s, 31);
            let m = self.bin(BinOp::LShr, all, s);
            let z = self.cmpc(Pred::Eq, l, 0);
            let m = self.selc(z, 0, m);
            let up = self.cmpc(Pred::UGe, l, 32);
            self.sel(up, all, m)
        } else {
            // lane <= 32 ? 0 : ~0 >> (64 - lane)
            let k = self.c(64);
            let s = self.sub(k, l);
            let s = self.andc(s, 31);
            let m = self.bin(BinOp::LShr, all, s);
            let le = self.cmpc(Pred::ULe, l, 32);
            self.selc(le, 0, m)
        }
    }
}
