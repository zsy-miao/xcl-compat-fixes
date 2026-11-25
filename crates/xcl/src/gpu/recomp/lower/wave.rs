//! Cross-lane operations of the lowering: `v_readlane`, `v_readfirstlane`, `v_writelane`,
//! DPP16 / DPP8 sources, `v_permlane(x)16`, `ds_swizzle_b32`, `v_mbcnt`, and the counts of a
//! mask as a number (`s_bcnt1`, `s_ff1`).
//!
//! A lane reads another through `Ir::read_lane` (`WaveReadLaneAt`), which needs that lane to
//! be running the same instruction. Here a lane is where the console has it except across a
//! lane branch: the lanes that went the other way are elsewhere. So:
//! - a read of a lane in EXEC is always fine: that lane is here (readfirstlane, and DPP,
//!   permlane and swizzle without fetch-inactive, whose source outside EXEC reads 0 or keeps
//!   the destination);
//! - a read of any lane (readlane, writelane, fetch-inactive) needs every lane here, so it is
//!   refused in a block not every lane reaches (`partial`), and the program stays on HLSL;
//! - a mask as a number is the ballot of its bits, which counts the lanes that are here: exact
//!   for EXEC (a lane elsewhere is outside it), and for any mask where every lane is here.
//!
//! DPP, permlane(x)16, ds_swizzle and `v_mbcnt_hi` only need this lane's 32-lane half, so they
//! use `Ir::read_lane_half` / `Ir::ballot_half`; readfirstlane, readlane, writelane, `v_mbcnt_lo`
//! and a mask as a number need the whole wave (`read_lane`, `ballot`). With one hardware wave per
//! console wave they are the same; a wave64 split into two 32-lane hardware waves exchanges data
//! between the halves only for the whole-wave forms.
//!
//! Lanes outside EXEC when a kernel-mode program starts leave at once (`lower_kernel`), so a
//! read of one of those is not the console's; compute waves start with every lane in EXEC.

use super::{Bin, Cmp, Ir, Lower, Var, EXEC, R};
use crate::gpu::rdna2::{self, Decoded, Variant};

impl<I: Ir> Lower<'_, I> {
    fn all_lanes(&self, b: usize, what: &str) -> R<()> {
        if self.partial[b] {
            Err(format!("{what} where some lanes are elsewhere"))
        } else {
            Ok(())
        }
    }

    /// Whether operand `code` names a register holding a mask here, and the pair's register.
    fn mask_reg(&mut self, b: usize, code: u64) -> Option<(u8, bool)> {
        if code > 127 || code == 124 || code == 125 {
            return None;
        }
        let r = code as u8;
        if self.holds_mask(b, r) {
            return Some((r, false));
        }
        // the upper dword of a wave64 mask pair
        (self.wave64 && r > 0 && self.holds_mask(b, r - 1)).then(|| (r - 1, true))
    }

    /// The mask in s<r> as numbers (lanes 0–31, 32–63): the ballot of its bits; with `half`,
    /// only this lane's half of it is right (`Ir::ballot_half`).
    fn mask_words(&mut self, b: usize, r: u8, half: bool) -> R<[I::V; 2]> {
        if r != EXEC || (!half && self.ir.whole_needs_exchange()) {
            self.all_lanes(b, "a mask read as a number")?;
        }
        let bit = self.read_bit(b, r)?;
        let [lo, hi] = if half { self.ir.ballot_half(bit) } else { self.ir.ballot(bit) };
        Ok(if self.wave64 { [lo, hi] } else { [lo, self.ir.int(0)] })
    }

    /// A 32-bit scalar operand as a number, also where it holds (half of) a mask.
    fn word(&mut self, b: usize, d: &Decoded, code: u64) -> R<I::V> {
        self.word_in(b, d, code, false)
    }

    /// `word`; with `half`, a mask's word is only right for the lanes of the half it covers.
    fn word_in(&mut self, b: usize, d: &Decoded, code: u64, half: bool) -> R<I::V> {
        match self.mask_reg(b, code) {
            Some((r, hi)) => Ok(self.mask_words(b, r, half)?[hi as usize]),
            None => self.src(b, d, code),
        }
    }

    /// A 64-bit scalar operand as numbers, also where it holds a mask.
    fn words(&mut self, b: usize, d: &Decoded, code: u64) -> R<[I::V; 2]> {
        match self.mask_reg(b, code) {
            Some((r, false)) => self.mask_words(b, r, false),
            Some((_, true)) => Err("a 64-bit operand across two masks".into()),
            None => {
                let (lo, hi) = self.src64(b, d, code)?;
                Ok([lo, hi])
            }
        }
    }

    /// The lowest set bit of x, 32 if none.
    fn ctz(&mut self, x: I::V) -> I::V {
        let (z, one) = (self.ir.int(0), self.ir.int(1));
        let neg = self.ir.bin(Bin::Sub, z, x);
        let low = self.ir.bin(Bin::And, x, neg);
        let below = self.ir.bin(Bin::Sub, low, one);
        self.ir.count_bits(below)
    }

    /// Leading zero bits of `x` (32 if `x == 0`).
    fn clz32(&mut self, mut x: I::V) -> I::V {
        for sh in [1, 2, 4, 8, 16] {
            let s = self.c(sh);
            let t = self.ir.bin(Bin::LShr, x, s);
            x = self.ir.bin(Bin::Or, x, t);
        }
        let all = self.c(u32::MAX);
        let inv = self.ir.bin(Bin::Xor, x, all);
        self.ir.count_bits(inv)
    }

    /// The lowest set bit of the 64-bit (lo, hi), `none` if none.
    fn first_set(&mut self, [lo, hi]: [I::V; 2], none: u32) -> I::V {
        let z = self.ir.int(0);
        let (cl, ch) = (self.ctz(lo), self.ctz(hi));
        let c32 = self.ir.int(32);
        let ch = self.ir.bin(Bin::Add, ch, c32);
        let none = self.ir.int(none);
        let hz = self.ir.icmp(Cmp::Eq, hi, z);
        let h = self.ir.select(hz, none, ch);
        let lz = self.ir.icmp(Cmp::Eq, lo, z);
        self.ir.select(lz, h, cl)
    }

    /// Bit `l` of the 64-bit (lo, hi).
    fn bit_at(&mut self, [lo, hi]: [I::V; 2], l: I::V) -> I::V {
        let (c31, c32, one, z) = (self.ir.int(31), self.ir.int(32), self.ir.int(1), self.ir.int(0));
        let low = self.ir.icmp(Cmp::Ult, l, c32);
        let w = self.ir.select(low, lo, hi);
        let sh = self.ir.bin(Bin::And, l, c31);
        let w = self.ir.bin(Bin::LShr, w, sh);
        let w = self.ir.bin(Bin::And, w, one);
        self.ir.icmp(Cmp::Ne, w, z)
    }

    /// The lanes in EXEC, as numbers; with `half`, only this lane's half is right.
    fn exec_words(&mut self, b: usize, half: bool) -> R<[I::V; 2]> {
        // split: the other half's word comes through an exchange, which both halves must reach
        if !half && self.ir.whole_needs_exchange() {
            self.all_lanes(b, "EXEC as a number in the split form")?;
        }
        let e = self.read_bit(b, EXEC)?;
        let [lo, hi] = if half { self.ir.ballot_half(e) } else { self.ir.ballot(e) };
        Ok([lo, if self.wave64 { hi } else { self.ir.int(0) }])
    }

    /// `v` of lane `l` (in this lane's 32-lane half) if that lane is in EXEC (or
    /// `inactive_too`); whether it was.
    fn read_from(&mut self, b: usize, v: I::V, l: I::V, inactive_too: bool, what: &str) -> R<(I::V, I::V)> {
        let ok = if inactive_too {
            self.all_lanes(b, what)?;
            self.ir.boolean(true)
        } else {
            let e = self.exec_words(b, true)?;
            self.bit_at(e, l)
        };
        // a lane outside EXEC may be elsewhere: read this lane's own value instead, unused
        let me = self.lane();
        let at = self.ir.select(ok, l, me);
        Ok((self.ir.read_lane_half(v, at), ok))
    }

    fn c(&mut self, v: u32) -> I::V {
        self.ir.int(v)
    }

    /// `(lane & !keep) | x`
    fn lane_with(&mut self, keep_not: u32, x: I::V) -> I::V {
        let me = self.lane();
        let k = self.c(!keep_not);
        let hi = self.ir.bin(Bin::And, me, k);
        self.ir.bin(Bin::Or, hi, x)
    }

    /// `(k >> (bits * (lane & group))) & field`: a lane's entry of a packed selector.
    fn packed(&mut self, k: I::V, bits: u32, group: u32, field: u32) -> I::V {
        let me = self.lane();
        let (g, w, f) = (self.c(group), self.c(bits), self.c(field));
        let i = self.ir.bin(Bin::And, me, g);
        let sh = self.ir.bin(Bin::Mul, i, w);
        let x = self.ir.bin(Bin::LShr, k, sh);
        self.ir.bin(Bin::And, x, f)
    }

    /// The source lane of DPP16 control `ctrl`, and whether it is inside the row.
    fn dpp16_lane(&mut self, ctrl: u64) -> R<(I::V, I::V)> {
        let me = self.lane();
        let n = (ctrl & 15) as u32;
        let t = self.ir.boolean(true);
        let row = |l: &mut Self| {
            let c = l.c(15);
            l.ir.bin(Bin::And, me, c)
        };
        Ok(match ctrl {
            0x000..=0x0FF => {
                let k = self.c(ctrl as u32);
                let x = self.packed(k, 2, 3, 3);
                (self.lane_with(3, x), t)
            }
            0x101..=0x10F => {
                let (r, cn, c16) = (row(self), self.c(n), self.c(16));
                let to = self.ir.bin(Bin::Add, r, cn);
                let ok = self.ir.icmp(Cmp::Ult, to, c16);
                (self.ir.bin(Bin::Add, me, cn), ok)
            }
            0x111..=0x11F => {
                let (r, cn) = (row(self), self.c(n));
                let ok = self.ir.icmp(Cmp::Uge, r, cn);
                (self.ir.bin(Bin::Sub, me, cn), ok)
            }
            0x121..=0x12F => {
                let (r, cn, c15) = (row(self), self.c(n), self.c(15));
                let x = self.ir.bin(Bin::Sub, r, cn);
                let x = self.ir.bin(Bin::And, x, c15);
                (self.lane_with(15, x), t)
            }
            0x140 => {
                let (r, c15) = (row(self), self.c(15));
                let x = self.ir.bin(Bin::Sub, c15, r);
                (self.lane_with(15, x), t)
            }
            0x141 => {
                let (c7, c8) = (self.c(7), self.c(8));
                let x = self.ir.bin(Bin::And, me, c7);
                let x = self.ir.bin(Bin::Sub, c7, x);
                let h = self.ir.bin(Bin::And, me, c8);
                let x = self.ir.bin(Bin::Or, x, h);
                (self.lane_with(15, x), t)
            }
            0x150..=0x15F => {
                let x = self.c(n);
                (self.lane_with(15, x), t)
            }
            0x160..=0x16F => {
                let cn = self.c(n);
                (self.ir.bin(Bin::Xor, me, cn), t)
            }
            _ => return Err(format!("DPP_CTRL {ctrl:#x}")),
        })
    }

    /// A DPP instruction's first source, read from its source lane, and its write enable
    /// (`dpp_src`, `dpp_en`, taken by `src` and `write_v`).
    pub(super) fn dpp(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        if name.starts_with("V_CMP") {
            return Err(format!("{name}: DPP compare"));
        }
        let e = super::super::emit::ext_encoding(d).ok_or("DPP encoding")?;
        let f = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
        if ["SRC0_ABS", "SRC0_NEG", "SRC1_ABS", "SRC1_NEG"].iter().any(|m| f(m) != 0) {
            return Err(format!("{name}: DPP modifiers"));
        }
        let v = self.read(b, Var::V(f("VSRC0") as u16));
        let (val, en) = if d.variant == Variant::Dpp16 {
            let (l, inside) = self.dpp16_lane(f("DPP_CTRL"))?;
            // outside the row: this lane's own, unused (the lane number may be out of range)
            let me = self.lane();
            let l = self.ir.select(inside, l, me);
            let (val, ok) = self.read_from(b, v, l, f("FI") == 1, "DPP fetch-inactive")?;
            let ok = self.ir.bin(Bin::And, ok, inside);
            let z = self.c(0);
            let mut en = None;
            let val = if f("BOUND_CTRL") == 1 {
                self.ir.select(ok, val, z)
            } else {
                en = Some(ok);
                val
            };
            let (rm, bm) = (f("ROW_MASK") as u32, f("BANK_MASK") as u32);
            if rm != 0xF || bm != 0xF {
                let me = self.lane();
                let (k, s4, s2, c3) = (self.c(rm), self.c(4), self.c(2), self.c(3));
                let r = self.ir.bin(Bin::LShr, me, s4);
                let r = self.ir.bin(Bin::LShr, k, r);
                let k = self.c(bm);
                let q = self.ir.bin(Bin::LShr, me, s2);
                let q = self.ir.bin(Bin::And, q, c3);
                let q = self.ir.bin(Bin::LShr, k, q);
                let both = self.ir.bin(Bin::And, r, q);
                let one = self.c(1);
                let both = self.ir.bin(Bin::And, both, one);
                let on = self.ir.icmp(Cmp::Ne, both, z);
                en = Some(match en {
                    Some(ok) => self.ir.bin(Bin::And, ok, on),
                    None => on,
                });
            }
            (val, en)
        } else {
            let sel: u64 = (0..8).map(|i| f(&format!("LANE_SEL_{i}")) << (3 * i)).sum();
            let fi = rdna2::field_value(d.enc, &d.words, "SRC0") == Some(0xEA);
            let k = self.c(sel as u32);
            let x = self.packed(k, 3, 7, 7);
            let l = self.lane_with(7, x);
            let (val, ok) = self.read_from(b, v, l, fi, "DPP8 fetch-inactive")?;
            let z = self.c(0);
            (self.ir.select(ok, val, z), None)
        };
        self.dpp_src = Some(val);
        self.dpp_en = en;
        Ok(())
    }

    /// The cross-lane VALU instructions; `false` for any other.
    pub(super) fn wave_valu(&mut self, b: usize, d: &Decoded) -> R<bool> {
        let name = d.inst.name;
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0);
        let (s0, s1, vdst) = (fv("SRC0"), fv("SRC1"), fv("VDST"));
        let wmask = if self.wave64 { 63 } else { 31 };
        match name {
            "V_READFIRSTLANE_B32" => {
                let v = self.src(b, d, s0)?;
                let r = if s0 >= 256 {
                    let e = self.exec_words(b, false)?;
                    let l = self.first_set(e, 0);
                    self.ir.read_lane(v, l)
                } else {
                    v
                };
                self.write_num(b, vdst as u8, r);
            }
            "V_READLANE_B32" => {
                self.all_lanes(b, "v_readlane")?;
                let (v, sel) = (self.src(b, d, s0)?, self.word(b, d, s1)?);
                let m = self.c(wmask);
                let l = self.ir.bin(Bin::And, sel, m);
                let r = self.ir.read_lane(v, l);
                self.write_num(b, vdst as u8, r);
            }
            "V_WRITELANE_B32" => {
                // whatever EXEC holds
                self.all_lanes(b, "v_writelane")?;
                let (a, sel) = (self.word(b, d, s0)?, self.word(b, d, s1)?);
                let m = self.c(wmask);
                let l = self.ir.bin(Bin::And, sel, m);
                let me = self.lane();
                let hit = self.ir.icmp(Cmp::Eq, me, l);
                let old = self.read(b, Var::V(vdst as u16));
                let v = self.ir.select(hit, a, old);
                self.ssa.write(b, Var::V(vdst as u16), v);
            }
            "V_MBCNT_LO_U32_B32" | "V_MBCNT_HI_U32_B32" => {
                // mbcnt_hi counts the upper lanes' own half only (0 below lane 32); mbcnt_lo
                // gives the upper lanes all of the lower half
                let hi_half = name == "V_MBCNT_HI_U32_B32";
                let (m, add) = (self.word_in(b, d, s0, hi_half)?, self.src(b, d, s1)?);
                // the lanes below this one, in this half of the mask
                let me = self.lane();
                let (c31, c32, one, z, all) = (self.c(31), self.c(32), self.c(1), self.c(0), self.c(u32::MAX));
                let sh = self.ir.bin(Bin::And, me, c31);
                let below = self.ir.bin(Bin::Shl, one, sh);
                let below = self.ir.bin(Bin::Sub, below, one);
                let upper = self.ir.icmp(Cmp::Uge, me, c32);
                let lt = if name == "V_MBCNT_LO_U32_B32" { self.ir.select(upper, all, below) } else { self.ir.select(upper, below, z) };
                let x = self.ir.bin(Bin::And, m, lt);
                let n = self.ir.count_bits(x);
                let r = self.ir.bin(Bin::Add, n, add);
                self.write_v(b, d, vdst as u16, r)?;
            }
            "V_PERMLANE16_B32" | "V_PERMLANEX16_B32" => {
                let op_sel = fv("OP_SEL");
                if ["ABS", "NEG", "CLAMP", "OMOD"].iter().any(|f| fv(f) != 0) || op_sel & !3 != 0 {
                    return Err(format!("{name}: VOP3 modifiers"));
                }
                let (fi, bound) = (op_sel & 1 == 1, op_sel & 2 == 2);
                let v = self.src(b, d, s0)?;
                let (lo, hi) = (self.word(b, d, s1)?, self.word(b, d, fv("SRC2"))?);
                let me = self.lane();
                let (c8, c15, c16) = (self.c(8), self.c(15), self.c(16));
                let r = self.ir.bin(Bin::And, me, c15);
                let first8 = self.ir.icmp(Cmp::Ult, r, c8);
                let k = self.ir.select(first8, lo, hi);
                let n = self.packed(k, 4, 7, 15);
                // the same row (permlane16) or the other row of the 32-lane half
                let row = self.ir.bin(Bin::And, me, c16);
                let row = if name == "V_PERMLANEX16_B32" { self.ir.bin(Bin::Xor, row, c16) } else { row };
                let x = self.ir.bin(Bin::Or, row, n);
                let l = self.lane_with(31, x);
                let (val, ok) = self.read_from(b, v, l, fi, "v_permlane fetch-inactive")?;
                let other = if bound { self.c(0) } else { self.read(b, Var::V(vdst as u16)) };
                let r = self.ir.select(ok, val, other);
                self.write_v(b, d, vdst as u16, r)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// `ds_swizzle_b32`: a lane reads another of its 32-lane half (`emit::swizzle_lane`).
    pub(super) fn swizzle(&mut self, b: usize, d: &Decoded) -> R<()> {
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0);
        let off = fv("OFFSET0") | fv("OFFSET1") << 8;
        let v = self.read(b, Var::V(fv("ADDR") as u16));
        let me = self.lane();
        let c31 = self.c(31);
        let i = self.ir.bin(Bin::And, me, c31);
        let x = if off >= 0xE000 {
            return Err("ds_swizzle FFT mode".into());
        } else if off >= 0xC000 {
            // rotate by offset[9:5] lanes, left or right (offset[10]); masked bits stay
            let mask = (off & 0x1F) as u32;
            let r = ((off >> 5) & 0x1F) as u32;
            let r = if off & 0x400 != 0 { r.wrapping_neg() } else { r };
            let (cm, cn, cr) = (self.c(mask), self.c(!mask & 31), self.c(r));
            let kept = self.ir.bin(Bin::And, i, cm);
            let moved = self.ir.bin(Bin::Add, i, cr);
            let moved = self.ir.bin(Bin::And, moved, cn);
            self.ir.bin(Bin::Or, kept, moved)
        } else if off & 0x8000 != 0 {
            let k = self.c(off as u32);
            let q = self.packed(k, 2, 3, 3);
            let c28 = self.c(28);
            let base = self.ir.bin(Bin::And, i, c28);
            self.ir.bin(Bin::Or, base, q)
        } else {
            let (and, or, xor) = (self.c((off & 0x1F) as u32), self.c(((off >> 5) & 0x1F) as u32), self.c(((off >> 10) & 0x1F) as u32));
            let x = self.ir.bin(Bin::And, i, and);
            let x = self.ir.bin(Bin::Or, x, or);
            self.ir.bin(Bin::Xor, x, xor)
        };
        let l = self.lane_with(31, x);
        let (val, ok) = self.read_from(b, v, l, false, "")?;
        let z = self.c(0);
        let r = self.ir.select(ok, val, z);
        self.write_v(b, d, fv("VDST") as u16, r)
    }

    /// Scalar counts of a mask; `false` for any other instruction.
    pub(super) fn wave_salu(&mut self, b: usize, d: &Decoded) -> R<bool> {
        let name = d.inst.name;
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0);
        let (s0, sd) = (fv("SSRC0"), fv("SDST") as u8);
        // `s_wqm` of a mask (`s_wqm_b64 exec, exec`): a lane's bit is set when any lane of its
        // quad has its bit; the quad is in the lane's own half of the mask
        // (the launch EXEC's, and any in a VS / PS, are `salu`'s)
        let wqm = if self.wave64 { "S_WQM_B64" } else { "S_WQM_B32" };
        if name == wqm && !self.graphics && !self.sum.wqm_all.contains(&d.pc) {
            let Some((r, false)) = self.mask_reg(b, s0) else { return Err(format!("unsupported {name} of a number")) };
            let [lo, hi] = self.mask_words(b, r, true)?;
            let lane = self.lane();
            let (c32, c28, f) = (self.c(32), self.c(28), self.c(0xF));
            let upper = self.ir.icmp(Cmp::Uge, lane, c32);
            let w = self.ir.select(upper, hi, lo);
            let q = self.ir.bin(Bin::And, lane, c28);
            let w = self.ir.bin(Bin::LShr, w, q);
            let w = self.ir.bin(Bin::And, w, f);
            let z = self.c(0);
            let bit = self.ir.icmp(Cmp::Ne, w, z);
            self.write_bit(b, sd, bit);
            let scc = self.mask_nonzero(bit);
            self.write_scc(b, scc);
            return Ok(true);
        }
        enum MaskOp {
            Bcnt(bool),
            Ff(bool),
            Flbit,
        }
        let op = match name {
            "S_BCNT1_I32_B32" | "S_BCNT1_I32_B64" => MaskOp::Bcnt(true),
            "S_BCNT0_I32_B32" | "S_BCNT0_I32_B64" => MaskOp::Bcnt(false),
            "S_FF1_I32_B32" | "S_FF1_I32_B64" => MaskOp::Ff(true),
            "S_FF0_I32_B32" | "S_FF0_I32_B64" => MaskOp::Ff(false),
            "S_FLBIT_I32_B32" | "S_FLBIT_I32_B64" => MaskOp::Flbit,
            _ => return Ok(false),
        };
        // only where the source is a mask: numbers take the usual path
        if self.mask_reg(b, s0).is_none() {
            return Ok(false);
        }
        let b64 = name.ends_with("_B64");
        let w = if b64 {
            self.words(b, d, s0)?
        } else {
            [self.word(b, d, s0)?, self.c(0)]
        };
        let r = match op {
            MaskOp::Bcnt(ones) => {
                let w = if ones {
                    w
                } else {
                    let all = self.c(u32::MAX);
                    let hi = if b64 { self.ir.bin(Bin::Xor, w[1], all) } else { w[1] };
                    [self.ir.bin(Bin::Xor, w[0], all), hi]
                };
                let (lo, hi) = (self.ir.count_bits(w[0]), self.ir.count_bits(w[1]));
                let r = self.ir.bin(Bin::Add, lo, hi);
                let z = self.c(0);
                let c = self.ir.icmp(Cmp::Ne, r, z);
                self.ssa.write(b, Var::Scc, c);
                r
            }
            MaskOp::Ff(ones) => {
                let w = if ones {
                    w
                } else {
                    let all = self.c(u32::MAX);
                    let hi = if b64 { self.ir.bin(Bin::Xor, w[1], all) } else { w[1] };
                    [self.ir.bin(Bin::Xor, w[0], all), hi]
                };
                self.first_set(w, u32::MAX)
            }
            MaskOp::Flbit => {
                let (z, none) = (self.c(0), self.c(u32::MAX));
                let cl = self.clz32(w[0]);
                if b64 {
                    let ch = self.clz32(w[1]);
                    let c32 = self.c(32);
                    let cl64 = self.ir.bin(Bin::Add, cl, c32);
                    let lz = self.ir.icmp(Cmp::Eq, w[0], z);
                    let lo_r = self.ir.select(lz, none, cl64);
                    let hz = self.ir.icmp(Cmp::Eq, w[1], z);
                    self.ir.select(hz, lo_r, ch)
                } else {
                    let lz = self.ir.icmp(Cmp::Eq, w[0], z);
                    self.ir.select(lz, none, cl)
                }
            }
        };
        self.write_num(b, sd, r);
        Ok(true)
    }
}
