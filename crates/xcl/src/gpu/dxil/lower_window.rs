//! Memory window selection and the windows taken from the descriptor heap: the direct
//! translator's form of the HLSL kernel's `window_read` / `window_write` and of its
//! `xcl_m*` accessors with `Options::heap_windows` (`recomp::emit::mem_accessors`).

use super::lower::{Lw, VAR_WIN};
use super::module::{Pred, Value};
use super::shader::RawUav;
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

    /// The high dword of the UMA base (`IO_UMA_BASE_HI` of the io buffer, `xcl_uma_base`).
    fn uma_base(&mut self) -> Value {
        let io = self.init.io.expect("io");
        let off = self.c(crate::gpu::recomp::IO_UMA_BASE_HI as u32 * 4);
        self.s.raw_load(io, off, 1)[0]
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
        let mut ea = self.pack64(lo, hi);
        for &t in adds {
            let x = match t {
                Ea::U(v) => self.zext64(v),
                Ea::K(k) => self.c64(k as u64),
                Ea::S(v) => self.sext64(v),
                Ea::Mul(a, b) => {
                    let (a, b) = (self.zext64(a), self.zext64(b));
                    self.mul(a, b)
                }
            };
            ea = self.add(ea, x);
        }
        let (_, h) = self.unpack64(ea);
        self.window_hi(h, write);
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

    /// `with_mem` with the windows in the descriptor heap (`Options::heap_windows`): window
    /// `w` is the raw UAV `ResourceDescriptorHeap[NonUniformResourceIndex(MEM_HEAP_BASE + w)]`
    /// (`globallycoherent` where the root windows are); a `w` outside the windows does what
    /// the HLSL accessor does (reads 0, stores and atomics dropped).
    pub(super) fn with_heap_mem(
        &mut self,
        coherent: bool,
        f: &mut dyn FnMut(&mut Lw, RawUav) -> Option<Value>,
    ) -> Option<Value> {
        let windows = self.mems.len() as u32;
        let w = self.rd(VAR_WIN);
        let (tl, th) = (self.tmp_var(), self.tmp_var());
        let z = self.c(0);
        self.wr(tl, z);
        self.wr(th, z);
        let mut kind = 0u8; // 0 none, 1 i32, 2 i64
        let inside = self.cmpc(Pred::ULt, w, windows);
        let _ = self.guarded(inside, &mut |lw| {
            let i = lw.addc(w, crate::gpu::recomp::MEM_HEAP_BASE);
            let m = lw.s.heap_raw_buffer_uav(i, coherent);
            if let Some(v) = f(lw, m) {
                if lw.s.m.type_of(v) == lw.s.m.i64() {
                    let (l, h) = lw.unpack64(v);
                    lw.wr(tl, l);
                    lw.wr(th, h);
                    kind = 2;
                } else {
                    lw.wr(tl, v);
                    kind = 1;
                }
            }
            Ok(())
        });
        match kind {
            0 => None,
            1 => Some(self.rd(tl)),
            _ => {
                let (l, h) = (self.rd(tl), self.rd(th));
                Some(self.pack64(l, h))
            }
        }
    }
}
