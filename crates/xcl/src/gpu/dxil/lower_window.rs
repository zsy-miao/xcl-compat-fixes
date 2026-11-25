//! Memory window selection and the windows taken from the descriptor heap: the direct
//! translator's form of the HLSL kernel's `window_read` / `window_write` and of its
//! `xcl_m*` accessors with `Options::heap_windows` (`recomp::emit::mem_accessors`).

use super::lower::{Lw, VAR_WIN};
use super::module::{CastOp, Pred, Value};
use crate::gpu::rdna2::Decoded;

/// A term of a 64-bit effective address (`window_at`).
#[derive(Clone, Copy)]
pub enum Ea {
    /// a 32-bit value, zero-extended
    U(Value),
    /// a 32-bit constant, zero-extended
    K(u32),
    /// a 32-bit value, sign-extended
    S(Value),
    /// the 64-bit product of two 32-bit values
    Mul(Value, Value),
}

impl Lw {
    /// The program may address several memory windows.
    pub fn windowed(&self) -> bool {
        self.mems.len() > 1
    }

    /// The high dword of the UMA base (`IO_UMA_BASE_HI` of the io buffer, `xcl_uma_base`),
    /// hoisted to the entry block so multi-window shaders load it once rather than on every
    /// memory operation.
    fn uma_base(&mut self) -> Value {
        if let Some(v) = self.uma_base_val {
            return v;
        }
        let io = self.init.io.expect("io");
        let off = self.c(crate::gpu::recomp::IO_UMA_BASE_HI as u32 * 4);
        let cur = self.cur();
        let entry = self.init.entry;
        let v = if cur == entry {
            self.s.raw_load(io, off, 1)[0]
        } else {
            let prev = self.s.m.set_insert_before_term(true);
            self.s.m.set_block(entry);
            let v = self.s.raw_load(io, off, 1)[0];
            self.s.m.set_block(cur);
            self.s.m.set_insert_before_term(prev);
            v
        };
        self.uma_base_val = Some(v);
        v
    }

    /// Selects the window of an access whose 64-bit effective address is `hi:lo` plus
    /// `adds` (carry / borrow into the high dword), as the HLSL kernel: a read with
    /// `hi == 0` (32-bit console descriptor table pointers) goes to window 0; any other
    /// high dword outside `[uma_base, uma_base + windows)` selects no window (reads 0,
    /// stores and atomics are dropped). Several windows only.
    pub fn window_at(&mut self, hi: Value, lo: Value, adds: &[Ea], write: bool) {
        if !self.windowed() {
            return;
        }
        let non_zero = |t: &Ea| match *t {
            Ea::K(0) => false,
            Ea::U(v) | Ea::S(v) => self.s.m.const_u64(v) != Some(0),
            Ea::Mul(a, b) => self.s.m.const_u64(a) != Some(0) && self.s.m.const_u64(b) != Some(0),
            Ea::K(_) => true,
        };
        let terms: Vec<Ea> = adds.iter().copied().filter(non_zero).collect();
        let (mut lo, mut hi) = (lo, hi);
        let uadd = |lw: &mut Self, lo: &mut Value, hi: &mut Value, add_lo: Value, add_hi: Option<Value>| {
            if let Some(ah) = add_hi {
                *hi = lw.add(*hi, ah);
            }
            if lw.s.m.const_u64(*lo) == Some(0) {
                *lo = add_lo;
            } else {
                let sum = lw.add(*lo, add_lo);
                let carry = lw.cmp(Pred::ULt, sum, *lo);
                *lo = sum;
                let i32 = lw.s.m.i32();
                let c32 = lw.s.m.cast(CastOp::ZExt, carry, i32);
                *hi = lw.add(*hi, c32);
            }
        };
        for (idx, t) in terms.iter().copied().enumerate() {
            let has_more = idx + 1 < terms.len();
            match t {
                Ea::K(k) => {
                    let kv = self.c(k);
                    uadd(self, &mut lo, &mut hi, kv, None);
                }
                Ea::U(v) => uadd(self, &mut lo, &mut hi, v, None),
                Ea::S(v) => match self.s.m.const_u64(v) {
                    Some(k) if (k as i32) < 0 => {
                        let neg_k = (k as u32).wrapping_neg();
                        if self.s.m.const_u64(lo) == Some(0) {
                            lo = v;
                            hi = self.addc(hi, u32::MAX);
                        } else {
                            let borrow = self.cmpc(Pred::ULt, lo, neg_k);
                            if has_more {
                                lo = self.add(lo, v);
                            }
                            let i32 = self.s.m.i32();
                            let b32 = self.s.m.cast(CastOp::ZExt, borrow, i32);
                            hi = self.sub(hi, b32);
                        }
                    }
                    Some(_) => uadd(self, &mut lo, &mut hi, v, None),
                    None => {
                        let sign = self.ashrc(v, 31);
                        uadd(self, &mut lo, &mut hi, v, Some(sign));
                    }
                },
                Ea::Mul(a, b) => match (self.s.m.const_u64(a), self.s.m.const_u64(b)) {
                    (Some(1), _) => uadd(self, &mut lo, &mut hi, b, None),
                    (_, Some(1)) => uadd(self, &mut lo, &mut hi, a, None),
                    (Some(ka), Some(kb)) => {
                        let prod = (ka as u32 as u64) * (kb as u32 as u64);
                        let (plo, phi) = (prod as u32, (prod >> 32) as u32);
                        if phi != 0 {
                            hi = self.addc(hi, phi);
                        }
                        if plo != 0 {
                            let plov = self.c(plo);
                            uadd(self, &mut lo, &mut hi, plov, None);
                        }
                    }
                    _ => {
                        let (phi, plo) = self.s.umul_wide(a, b);
                        uadd(self, &mut lo, &mut hi, plo, Some(phi));
                    }
                },
            }
        }
        self.window_hi(hi, write);
    }

    /// `window_at` of a high dword already computed (`window_read` / `window_write`).
    pub fn window_hi(&mut self, hi: Value, write: bool) {
        if !self.windowed() {
            return;
        }
        let base = self.uma_base();
        let mut w = self.sub(hi, base);
        if !write {
            let z = self.cmpc(Pred::Eq, hi, 0);
            let zero = self.c(0);
            w = self.sel(z, zero, w);
        }
        self.wr(VAR_WIN, w);
    }

    /// The window of a buffer access: V# base `d1[15:0]:d0` plus `adds` (the stride bits
    /// `d1[29:16]` are not part of the address).
    pub fn buffer_window(&mut self, d0: Value, d1: Value, adds: &[Ea], write: bool) {
        if self.windowed() {
            let hi = self.andc(d1, 0xFFFF);
            self.window_at(hi, d0, adds, write);
        }
    }

    /// The window of a FLAT / global access: SADDR pair + VGPR offset, or the 64-bit VGPR
    /// address, plus the instruction offset (signed for global).
    pub fn flat_window(&mut self, d: &Decoded, write: bool) {
        if !self.windowed() {
            return;
        }
        let global = d.enc.name == "ENC_FLAT_GLBL";
        let saddr = Self::fv(d, "SADDR");
        let addr = Self::fv(d, "ADDR");
        let raw = Self::fv(d, "OFFSET") as u32;
        let off = if global {
            Ea::S(self.c((((raw << 20) as i32) >> 20) as u32))
        } else {
            Ea::U(self.c(raw))
        };
        if global && saddr != 0x7D && saddr != 0x7F {
            let (lo, hi, v) = (self.sreg(saddr), self.sreg(saddr + 1), self.vreg(addr));
            self.window_at(hi, lo, &[Ea::U(v), off], write);
        } else {
            let (lo, hi) = (self.vreg(addr), self.vreg(addr + 1));
            self.window_at(hi, lo, &[off], write);
        }
    }

    /// The window of a `global_*_addtid` access: SADDR pair (or 0) + 4 × lane + offset.
    pub fn addtid_window(&mut self, d: &Decoded, write: bool) {
        if !self.windowed() {
            return;
        }
        let global = d.enc.name == "ENC_FLAT_GLBL";
        let saddr = Self::fv(d, "SADDR");
        let raw = Self::fv(d, "OFFSET") as u32;
        let off = if global {
            Ea::S(self.c((((raw << 20) as i32) >> 20) as u32))
        } else {
            Ea::U(self.c(raw))
        };
        let lane = self.lane;
        let l4 = self.shlc(lane, 2);
        let (lo, hi) = match saddr {
            0x7D | 0x7F => (self.c(0), self.c(0)),
            _ => (self.sreg(saddr), self.sreg(saddr + 1)),
        };
        self.window_at(hi, lo, &[Ea::U(l4), off], write);
    }
}
