//! Memory lowering: scalar loads, buffer / typed buffer, flat / global, LDS,
//! atomics and kernel exports on the kernel ABI's flat `mem` window (port of
//! `recomp/emit.rs` `smem` / `mubuf` / `flat` / `ds` / `export` and
//! `PRELUDE_MEM`). Accesses of disabled lanes are real branches.

use super::lower::{err, Dst, LResult, Lw, VAR_VGPR};
use super::module::{BinOp, Pred, RmwOp, Value};
use super::ops::AtomicOp;
use super::lower_window::Ea;
use super::shader::RawUav;
use crate::gpu::rdna2::{self, Decoded};
use crate::gpu::recomp::emit::buf_format_info;
use crate::gpu::recomp::{KERNEL_EXPORTS, KERNEL_LANES};

/// Temporary SSA variables (merged across guarded blocks, never loaded / stored).
pub const VAR_TMP: u32 = 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BufKind {
    Load,
    Store,
    FormatLoad,
    FormatStore,
}

fn buffer_op(name: &str) -> Option<(BufKind, u64)> {
    let n = |suffix: &str| -> Option<u64> {
        Some(match suffix {
            "dword" | "x" => 1,
            "dwordx2" | "xy" => 2,
            "dwordx3" | "xyz" => 3,
            "dwordx4" | "xyzw" => 4,
            _ => return None,
        })
    };
    if let Some(r) = name.strip_prefix("buffer_load_format_").or_else(|| name.strip_prefix("tbuffer_load_format_")) {
        return Some((BufKind::FormatLoad, n(r)?));
    }
    if let Some(r) = name.strip_prefix("buffer_store_format_").or_else(|| name.strip_prefix("tbuffer_store_format_")) {
        return Some((BufKind::FormatStore, n(r)?));
    }
    if let Some(r) = name.strip_prefix("buffer_load_") {
        return Some((BufKind::Load, n(r)?));
    }
    if let Some(r) = name.strip_prefix("buffer_store_") {
        return Some((BufKind::Store, n(r)?));
    }
    None
}

fn buffer_format_d16(name: &str) -> Option<(BufKind, u64, bool)> {
    let (kind, r) = if let Some(r) = name.strip_prefix("buffer_load_format_d16_").or_else(|| name.strip_prefix("tbuffer_load_format_d16_")) {
        (BufKind::FormatLoad, r)
    } else if let Some(r) = name.strip_prefix("buffer_store_format_d16_").or_else(|| name.strip_prefix("tbuffer_store_format_d16_")) {
        (BufKind::FormatStore, r)
    } else {
        return None;
    };
    let (components, high) = match r {
        "x" => (1, false),
        "xy" => (2, false),
        "xyz" => (3, false),
        "xyzw" => (4, false),
        "hi_x" => (1, true),
        _ => return None,
    };
    Some((kind, components, high))
}

fn d16_small(op: &str) -> Option<(u32, bool, bool, u32)> {
    Some(match op {
        "load_ubyte_d16" => (1, false, false, 0),
        "load_ubyte_d16_hi" => (1, false, false, 1),
        "load_sbyte_d16" => (1, true, false, 0),
        "load_sbyte_d16_hi" => (1, true, false, 1),
        "load_short_d16" => (2, false, false, 0),
        "load_short_d16_hi" => (2, false, false, 1),
        "store_byte_d16_hi" => (1, false, true, 1),
        "store_short_d16_hi" => (2, false, true, 1),
        _ => return None,
    })
}

fn flat_small(op: &str) -> Option<(u32, bool, bool, u32)> {
    Some(match op {
        "load_ubyte" => (1, false, false, 2),
        "load_sbyte" => (1, true, false, 2),
        "load_ushort" => (2, false, false, 2),
        "load_sshort" => (2, true, false, 2),
        "store_byte" => (1, false, true, 2),
        "store_short" => (2, false, true, 2),
        _ => return d16_small(op),
    })
}

fn ds_d16(name: &str) -> Option<(u32, bool, bool, u32)> {
    Some(match name {
        "ds_read_u8_d16" => (8, false, false, 0),
        "ds_read_u8_d16_hi" => (8, false, false, 1),
        "ds_read_i8_d16" => (8, true, false, 0),
        "ds_read_i8_d16_hi" => (8, true, false, 1),
        "ds_read_u16_d16" => (16, false, false, 0),
        "ds_read_u16_d16_hi" => (16, false, false, 1),
        "ds_write_b8_d16_hi" => (8, false, true, 1),
        "ds_write_b16_d16_hi" => (16, false, true, 1),
        _ => return None,
    })
}

/// Where a 32-bit atomic works.
#[derive(Clone, Copy)]
enum At {
    Mem(Value),
    Lds(Value),
}

/// Hoisted descriptor-level format info for `FormatLoad` / `FormatStore`.
#[derive(Clone, Copy)]
pub struct FmtDesc {
    pub info: Value,
    pub n: Value,
    pub ty: Value,
    pub isint: Value,
    pub nb: Value,
    pub has: Value,
    pub t: [Value; 6],
}

/// Hoisted single-component format decode parameters for `BUFFER_LOAD_FORMAT_X`
/// (and 1-component `d16` format loads), computed once in `d3`'s defining block.
#[derive(Clone, Copy)]
pub struct FmtComp1 {
    pub desc: FmtDesc,
    pub s0: Value,
    pub s1: Value,
    pub one_val: Value,
    pub off_bytes: Value,
    pub off_bits: Value,
    pub big: Value,
    pub sh: Value,
    pub mu: Value,
    pub maxu: Value,
    pub maxs: Value,
    pub w16: Value,
    pub w32: Value,
    pub mb: Value,
    pub mmask: Value,
    pub mshift: Value,
    pub den_scale: Value,
    pub cge: Value,
    pub dflt: Value,
}

impl Lw {
    /// Runs `f` on the active memory window (`mem`, `mem1`.. at u2.., or descriptor heap
    /// when `self.heap_mem` is set via `XCL_HEAP_WINDOWS`), returning `out_tys.len()` values
    /// merged across the window arms via a single `switch` (or single guarded heap lookup)
    /// and direct phis (`f` must not write SSA variables). Out-of-range windows (`w >= n`)
    /// jump to `join` (reads 0, stores and atomics dropped), matching `#1693` semantics.
    pub(crate) fn with_mem_many_tys(
        &mut self,
        mut guard: Option<Value>,
        out_tys: &[super::module::Ty],
        f: &mut dyn FnMut(&mut Lw, RawUav) -> Vec<Value>,
    ) -> Vec<Value> {
        if let Some(c) = guard {
            match self.s.m.const_u64(c) {
                Some(1) => guard = None,
                Some(0) => return out_tys.iter().map(|&t| self.s.m.const_int(t, 0)).collect(),
                _ => {}
            }
        }
        // Descriptor-heap windows (`#1693`, `XCL_HEAP_WINDOWS=1`): single guarded heap lookup
        // combined with `guard`, stacking cleanly with vector loads/stores & FormatLoad opt.
        if let Some(coherent) = self.heap_mem {
            let windows = self.mems.len() as u32;
            let w = self.rd(super::lower::VAR_WIN);
            let inside = self.cmpc(Pred::ULt, w, windows);
            let cond = match guard {
                Some(g) => self.and(g, inside),
                None => inside,
            };
            if self.s.m.const_u64(cond) == Some(0) {
                return out_tys.iter().map(|&t| self.s.m.const_int(t, 0)).collect();
            }
            if self.s.m.const_u64(cond) == Some(1) {
                let i = self.addc(w, crate::gpu::recomp::MEM_HEAP_BASE);
                let m = self.s.heap_raw_buffer_uav(i, coherent);
                let vals = f(self, m);
                debug_assert_eq!(vals.len(), out_tys.len());
                return vals;
            }
            let from = self.cur();
            let then = self.block(None);
            let join = self.block(None);
            self.cond_br(cond, then, join);
            self.seal(then);
            self.s.m.set_block(then);
            #[cfg(debug_assertions)]
            self.ssa.begin_log();
            let i = self.addc(w, crate::gpu::recomp::MEM_HEAP_BASE);
            let m = self.s.heap_raw_buffer_uav(i, coherent);
            let vals = f(self, m);
            #[cfg(debug_assertions)]
            debug_assert!(self.ssa.end_log().is_empty(), "with_mem closure wrote SSA var");
            debug_assert_eq!(vals.len(), out_tys.len());
            let end = self.cur();
            self.s.m.br(join);
            self.ssa.seal_region_join(join, from, from, Vec::new());
            self.s.m.set_block(join);
            let mut out = Vec::with_capacity(out_tys.len());
            for (j, &ty) in out_tys.iter().enumerate() {
                let z = self.s.m.const_int(ty, 0);
                let phi = self.s.m.phi(ty);
                self.s.m.add_incoming(phi, z, from);
                self.s.m.add_incoming(phi, vals[j], end);
                out.push(phi);
            }
            return out;
        }
        // Single window, or window known at translation time: at most one branch for `guard`.
        let fixed_mem = if self.mems.len() <= 1 {
            Some(self.mems[0])
        } else {
            let w = self.rd(super::lower::VAR_WIN);
            if let Some(k) = self.s.m.const_u64(w) {
                if (k as usize) < self.mems.len() {
                    Some(self.mems[k as usize])
                } else {
                    return out_tys.iter().map(|&t| self.s.m.const_int(t, 0)).collect();
                }
            } else {
                None
            }
        };
        if let Some(m) = fixed_mem {
            let Some(cond) = guard else {
                let vals = f(self, m);
                debug_assert_eq!(vals.len(), out_tys.len());
                return vals;
            };
            let from = self.cur();
            let then = self.block(None);
            let join = self.block(None);
            self.cond_br(cond, then, join);
            self.seal(then);
            self.s.m.set_block(then);
            #[cfg(debug_assertions)]
            self.ssa.begin_log();
            let vals = f(self, m);
            #[cfg(debug_assertions)]
            debug_assert!(self.ssa.end_log().is_empty(), "with_mem closure wrote SSA var");
            debug_assert_eq!(vals.len(), out_tys.len());
            let end = self.cur();
            self.s.m.br(join);
            self.ssa.seal_region_join(join, from, from, Vec::new());
            self.s.m.set_block(join);
            let mut out = Vec::with_capacity(out_tys.len());
            for (j, &ty) in out_tys.iter().enumerate() {
                let z = self.s.m.const_int(ty, 0);
                let phi = self.s.m.phi(ty);
                self.s.m.add_incoming(phi, z, from);
                self.s.m.add_incoming(phi, vals[j], end);
                out.push(phi);
            }
            return out;
        }
        // Multi-window dispatch via a single LLVM `switch` (default -> `join` for out-of-range
        // or inactive lane, matching `#1693` window semantics).
        let w = self.rd(super::lower::VAR_WIN);
        let mems = self.mems.clone();
        let n = mems.len();
        let from = self.cur();
        let arms: Vec<super::module::Block> = (0..n).map(|_| self.block(None)).collect();
        let join = self.block(None);
        let w_eff = if let Some(cond) = guard {
            let skip = self.c(n as u32);
            self.sel(cond, w, skip)
        } else {
            w
        };
        let cases: Vec<(Value, super::module::Block)> = (0..n).map(|k| (self.c(k as u32), arms[k])).collect();
        self.s.m.switch(w_eff, join, &cases);
        for &arm in &arms {
            self.edge(from, arm);
            self.seal(arm);
        }
        let mut arm_vals: Vec<Vec<Value>> = Vec::with_capacity(n);
        let mut arm_ends: Vec<super::module::Block> = Vec::with_capacity(n);
        for (k, &m) in mems.iter().enumerate() {
            self.s.m.set_block(arms[k]);
            #[cfg(debug_assertions)]
            self.ssa.begin_log();
            let vals = f(self, m);
            #[cfg(debug_assertions)]
            debug_assert!(self.ssa.end_log().is_empty(), "with_mem closure wrote SSA var");
            debug_assert_eq!(vals.len(), out_tys.len());
            arm_ends.push(self.cur());
            arm_vals.push(vals);
            self.s.m.br(join);
        }
        self.ssa.seal_region_join(join, from, from, Vec::new());
        self.s.m.set_block(join);
        let mut out = Vec::with_capacity(out_tys.len());
        for (j, &ty) in out_tys.iter().enumerate() {
            let phi = self.s.m.phi(ty);
            let z = self.s.m.const_int(ty, 0);
            self.s.m.add_incoming(phi, z, from);
            for k in 0..n {
                self.s.m.add_incoming(phi, arm_vals[k][j], arm_ends[k]);
            }
            out.push(phi);
        }
        out
    }

    pub(crate) fn with_mem_i32s(
        &mut self,
        guard: Option<Value>,
        n: usize,
        f: &mut dyn FnMut(&mut Lw, RawUav) -> Vec<Value>,
    ) -> Vec<Value> {
        let i32 = self.s.m.i32();
        let tys = vec![i32; n];
        self.with_mem_many_tys(guard, &tys, f)
    }

    pub(crate) fn with_mem_void(
        &mut self,
        guard: Option<Value>,
        f: &mut dyn FnMut(&mut Lw, RawUav),
    ) {
        let _ = self.with_mem_many_tys(guard, &[], &mut |lw, m| {
            f(lw, m);
            Vec::new()
        });
    }
    fn mraw_atomic(&mut self, op: AtomicOp, a: Value, v: Value) -> Value {
        let ty = self.s.m.type_of(v);
        self.with_mem_many_tys(None, &[ty], &mut |lw, m| vec![lw.s.raw_atomic(m, op, a, v)])[0]
    }
    fn mraw_cmpxchg(&mut self, a: Value, c: Value, n: Value) -> Value {
        let ty = self.s.m.type_of(c);
        self.with_mem_many_tys(None, &[ty], &mut |lw, m| vec![lw.s.raw_cmpxchg(m, a, c, n)])[0]
    }
    /// `mem.Load(a & ~3)`
    pub fn mem_ld(&mut self, a: Value) -> Value {
        let a = self.andc(a, !3);
        self.with_mem_i32s(None, 1, &mut |lw, m| lw.s.raw_load(m, a, 1))[0]
    }
    /// `mem.Load2/3/4(a & ~3)` (1..=4 consecutive dwords in one window switch).
    pub(crate) fn mem_ld_n(&mut self, guard: Option<Value>, a: Value, count: u32) -> Vec<Value> {
        debug_assert!((1..=4).contains(&count));
        let a = self.andc(a, !3);
        let mask = ((1u32 << count) - 1) as u8;
        self.with_mem_i32s(guard, count as usize, &mut |lw, m| lw.s.raw_load(m, a, mask))
    }
    /// `mem.Store(a & ~3, v)`
    pub fn mem_st(&mut self, a: Value, v: Value) {
        let a = self.andc(a, !3);
        self.with_mem_void(None, &mut |lw, m| lw.s.raw_store(m, a, &[v]));
    }
    /// `mem.Store1..4(a & ~3, vals)` in one window switch.
    fn mem_st_n(&mut self, guard: Option<Value>, a: Value, vals: &[Value]) {
        debug_assert!((1..=4).contains(&vals.len()));
        let a = self.andc(a, !3);
        self.with_mem_void(guard, &mut |lw, m| lw.s.raw_store(m, a, vals));
    }

    /// `ld_raw`: four dwords from an unaligned byte address (one Load4 + one Load1
    /// in a single window switch, optionally combined with a lane guard).
    fn ld_raw_guarded(&mut self, guard: Option<Value>, a: Value) -> [Value; 4] {
        let b = self.andc(a, !3);
        let b16 = self.addc(b, 16);
        let s = self.andc(a, 3);
        let s = self.shlc(s, 3);
        let w = self.with_mem_i32s(guard, 5, &mut |lw, m| {
            let mut v = lw.s.raw_load(m, b, 0xF);
            v.push(lw.s.raw_load(m, b16, 1)[0]);
            v
        });
        let k32 = self.c(32);
        let inv = self.sub(k32, s);
        let inv = self.andc(inv, 31);
        let sz = self.cmpc(Pred::Eq, s, 0);
        let mut r = [w[0]; 4];
        for i in 0..4 {
            let lo = self.bin(BinOp::LShr, w[i], s);
            let hi = self.bin(BinOp::Shl, w[i + 1], inv);
            let x = self.or(lo, hi);
            r[i] = self.sel(sz, w[i], x);
        }
        r
    }

    /// One unaligned dword from byte address `a` (one Load2 in a single window switch).
    fn ld_raw_u32_guarded(&mut self, guard: Option<Value>, a: Value) -> Value {
        let b = self.andc(a, !3);
        let s = self.andc(a, 3);
        let s = self.shlc(s, 3);
        let w = self.with_mem_i32s(guard, 2, &mut |lw, m| lw.s.raw_load(m, b, 0x3));
        let k32 = self.c(32);
        let inv = self.sub(k32, s);
        let inv = self.andc(inv, 31);
        let sz = self.cmpc(Pred::Eq, s, 0);
        let lo = self.bin(BinOp::LShr, w[0], s);
        let hi = self.bin(BinOp::Shl, w[1], inv);
        let x = self.or(lo, hi);
        self.sel(sz, w[0], x)
    }
    /// `st_part`: atomically replace the bits `m` of the dword at `b` by `v`.
    fn st_part(&mut self, b: Value, m: Value, v: Value) {
        let nm = self.not(m);
        self.mraw_atomic(AtomicOp::And, b, nm);
        let vm = self.and(v, m);
        self.mraw_atomic(AtomicOp::Or, b, vm);
    }
    /// `st_bytes`: store the low `n` bytes of `r` at byte address `a` (n ≤ 16; partial
    /// dwords through atomics so neighbours stay intact).
    fn st_bytes(&mut self, a: Value, r: [Value; 4], n: Value) -> LResult<()> {
        let s = self.andc(a, 3);
        let s = self.shlc(s, 3);
        let b = self.andc(a, !3);
        let sz = self.cmpc(Pred::Eq, s, 0);
        let n3 = self.andc(n, 3);
        let nz3 = self.cmpc(Pred::Eq, n3, 0);
        let whole = self.land(sz, nz3);
        self.guarded(whole, &mut |lw| {
            for k in 0..4u32 {
                let has = lw.cmpc(Pred::UGt, n, k * 4);
                let at = lw.addc(b, 4 * k);
                let v = r[k as usize];
                lw.guarded(has, &mut |lw| {
                    lw.mem_st(at, v);
                    Ok(())
                })?;
            }
            Ok(())
        })?;
        let part = self.lnot(whole);
        self.guarded(part, &mut |lw| {
            // dword k of `r` holds bytes 4k..min(4k + 4, n); each lands at b + 4k with shift s
            // and spills into the next dword when s + its bit width passes 32
            let one = lw.c(1);
            let all = lw.c(u32::MAX);
            let k4 = lw.c(4);
            let k32 = lw.c(32);
            let snz = lw.cmpc(Pred::Ne, s, 0);
            let inv = lw.sub(k32, s);
            let inv = lw.andc(inv, 31);
            for k in 0..4u32 {
                let has = lw.cmpc(Pred::UGt, n, k * 4);
                let rk = r[k as usize];
                lw.guarded(has, &mut |lw| {
                    let rem = lw.addc(n, (k * 4).wrapping_neg());
                    let big = lw.cmpc(Pred::UGe, rem, 4);
                    let chunk = lw.sel(big, k4, rem);
                    let c8 = lw.shlc(chunk, 3);
                    let c8m = lw.andc(c8, 31);
                    let m = lw.bin(BinOp::Shl, one, c8m);
                    let m = lw.sub(m, one);
                    let m = lw.sel(big, all, m);
                    let at = lw.addc(b, 4 * k);
                    let ms = lw.bin(BinOp::Shl, m, s);
                    let vs = lw.bin(BinOp::Shl, rk, s);
                    lw.st_part(at, ms, vs);
                    let top = lw.add(s, c8);
                    let over = lw.cmpc(Pred::UGt, top, 32);
                    let spill = lw.land(snz, over);
                    lw.guarded(spill, &mut |lw| {
                        let at4 = lw.addc(at, 4);
                        let m2 = lw.bin(BinOp::LShr, m, inv);
                        let v2 = lw.bin(BinOp::LShr, rk, inv);
                        lw.st_part(at4, m2, v2);
                        Ok(())
                    })
                })?;
            }
            Ok(())
        })
    }

    // ------------------------------------------------------------ formats

    /// `FMT_INFO[min(fmt, 77)]` (a constant table) for a dynamic format; constant-folded
    /// when `fmt` is known at translation time.
    pub(crate) fn fmt_info(&mut self, fmt: Value) -> Value {
        if let Some(k) = self.s.m.const_u64(fmt) {
            return self.c(buf_format_info(k.min(77) as usize));
        }
        let i32 = self.s.m.i32();
        let g = match self.fmt_table {
            Some(g) => g,
            None => {
                let vals: Vec<Value> = (0..78).map(|i| self.s.m.i32c(buf_format_info(i))).collect();
                let at = self.s.m.array(78, i32);
                let init = self.s.m.aggregate(at, &vals);
                let g = self.s.m.global("xcl_fmt_info", at, 0, Some(init), true, 4);
                self.fmt_table = Some(g);
                g
            }
        };
        let k = self.c(77);
        let i = self.umin(fmt, k);
        let z = self.c(0);
        let p = self.s.m.gep(g, &[z, i]);
        self.s.m.load(p, 4)
    }
    /// Emits pure instructions into `v`'s defining block (before its terminator if already
    /// terminated) so every use of `v` can share them without duplicating across branches.
    fn in_def_block<T>(&mut self, v: Value, f: impl FnOnce(&mut Lw) -> T) -> T {
        let Some(b) = self.s.m.def_block(v) else {
            return f(self);
        };
        let cur = self.cur();
        if b == cur {
            return f(self);
        }
        let prev_ins = self.s.m.set_insert_before_term(true);
        self.s.m.set_block(b);
        let r = f(self);
        self.s.m.set_block(cur);
        self.s.m.set_insert_before_term(prev_ins);
        r
    }
    fn fmt_desc(&mut self, d3: Value, mtbuf_fmt: Option<u32>) -> FmtDesc {
        let key = (d3, mtbuf_fmt);
        if let Some(&cached) = self.fmt_desc_cache.get(&key) {
            return cached;
        }
        let anchor = if mtbuf_fmt.is_some() { self.c(0) } else { d3 };
        let desc = self.in_def_block(anchor, |lw| {
            let fmt = match mtbuf_fmt {
                Some(k) => lw.c(k),
                None => {
                    let x = lw.lshrc(d3, 12);
                    lw.andc(x, 0x7F)
                }
            };
            let info = lw.fmt_info(fmt);
            let n = lw.andc(info, 7);
            let ty = lw.lshrc(info, 3);
            let ty = lw.andc(ty, 7);
            let t = [0, 1, 2, 3, 4, 5].map(|k| lw.cmpc(Pred::Eq, ty, k));
            let isint = lw.lor(t[4], t[5]);
            let nb = lw.fmt_bytes(info);
            let has = lw.nz(n);
            FmtDesc { info, n, ty, isint, nb, has, t }
        });
        self.fmt_desc_cache.insert(key, desc);
        desc
    }
    fn fmt_comp1(&mut self, d3: Value, mtbuf_fmt: Option<u32>) -> FmtComp1 {
        let key = (d3, mtbuf_fmt);
        if let Some(&cached) = self.fmt_comp1_cache.get(&key) {
            return cached;
        }
        let desc = self.fmt_desc(d3, mtbuf_fmt);
        let c1 = self.in_def_block(d3, |lw| {
            let sel = lw.andc(d3, 7);
            let s0 = lw.cmpc(Pred::Eq, sel, 0);
            let s1 = lw.cmpc(Pred::Eq, sel, 1);
            let one = lw.c(1);
            let onef = lw.c(0x3F80_0000);
            let one_val = lw.sel(desc.isint, one, onef);
            let k4 = lw.c(4);
            let ci = lw.sub(sel, k4);
            let ci = lw.andc(ci, 3);
            let z = lw.c(0);
            let w0 = lw.lshrc(desc.info, 8);
            let w0 = lw.andc(w0, 63);
            let w1 = lw.lshrc(desc.info, 14);
            let w1 = lw.andc(w1, 63);
            let w2 = lw.lshrc(desc.info, 20);
            let w2 = lw.andc(w2, 63);
            let gt0 = lw.cmpc(Pred::UGt, ci, 0);
            let gt1 = lw.cmpc(Pred::UGt, ci, 1);
            let gt2 = lw.cmpc(Pred::UGt, ci, 2);
            let o0 = lw.sel(gt0, w0, z);
            let o1 = lw.sel(gt1, w1, z);
            let o2 = lw.sel(gt2, w2, z);
            let off = lw.add(o0, o1);
            let off = lw.add(off, o2);
            let off_bytes = lw.lshrc(off, 3);
            let off_bits = lw.andc(off, 7);
            let k6 = lw.c(6);
            let c6 = lw.mul(ci, k6);
            let sh_info = lw.addc(c6, 8);
            let sh_info = lw.andc(sh_info, 31);
            let w = lw.bin(BinOp::LShr, desc.info, sh_info);
            let w = lw.andc(w, 63);
            let big = lw.cmpc(Pred::UGe, w, 32);
            let k32 = lw.c(32);
            let sh = lw.sub(k32, w);
            let sh = lw.andc(sh, 31);
            let w31 = lw.andc(w, 31);
            let mu = lw.bin(BinOp::Shl, one, w31);
            let mu = lw.sub(mu, one);
            let maxu = lw.uitof(mu);
            let fmaxu = lw.fc(4294967295.0);
            let maxu = lw.sel(big, fmaxu, maxu);
            let wm1 = lw.sub(w, one);
            let wm1 = lw.andc(wm1, 31);
            let ms = lw.bin(BinOp::Shl, one, wm1);
            let ms = lw.sub(ms, one);
            let maxs = lw.uitof(ms);
            let fmaxs = lw.fc(2147483647.0);
            let maxs = lw.sel(big, fmaxs, maxs);
            let w11 = lw.cmpc(Pred::Eq, w, 11);
            let w16 = lw.cmpc(Pred::Eq, w, 16);
            let w32 = lw.cmpc(Pred::Eq, w, 32);
            let k5 = lw.c(5);
            let mb = lw.sel(w11, k6, k5);
            let k63 = lw.c(63);
            let k31 = lw.c(31);
            let mmask = lw.sel(w11, k63, k31);
            let k17 = lw.c(17);
            let k18 = lw.c(18);
            let mshift = lw.sel(w11, k17, k18);
            let sc6 = lw.s.m.f32c(0x3580_0000); // 2^-20
            let sc5 = lw.s.m.f32c(0x3600_0000); // 2^-19
            let den_scale = lw.sel(w11, sc6, sc5);
            let cge = lw.cmp(Pred::ULe, desc.n, ci);
            let is3 = lw.cmpc(Pred::Eq, ci, 3);
            let dflt = lw.sel(is3, one_val, z);
            FmtComp1 { desc, s0, s1, one_val, off_bytes, off_bits, big, sh, mu, maxu, maxs, w16, w32, mb, mmask, mshift, den_scale, cge, dflt }
        });
        self.fmt_comp1_cache.insert(key, c1);
        c1
    }
    fn fmt_bytes(&mut self, info: Value) -> Value {
        let mut t = self.c(0);
        for sh in [8, 14, 20, 26] {
            let w = self.lshrc(info, sh);
            let w = self.andc(w, 63);
            t = self.add(t, w);
        }
        self.lshrc(t, 3)
    }
    /// `r[i]` for a dynamic i (0..3)
    fn pick4(&mut self, r: &[Value; 4], i: Value) -> Value {
        let mut v = r[3];
        for k in (0..3).rev() {
            let is = self.cmpc(Pred::Eq, i, k);
            v = self.sel(is, r[k as usize], v);
        }
        v
    }
    fn bits_at(&mut self, r: &[Value; 4], off: Value, w: Value) -> Value {
        let i = self.lshrc(off, 5);
        let s = self.andc(off, 31);
        let lo = self.pick4(r, i);
        let i1 = self.addc(i, 1);
        let k3 = self.c(3);
        let i1 = self.umin(i1, k3);
        let hi = self.pick4(r, i1);
        let in3 = self.cmpc(Pred::ULt, i, 3);
        let hi = {
            let z = self.c(0);
            let t = self.lnot(in3);
            self.sel(t, z, hi)
        };
        let a = self.bin(BinOp::LShr, lo, s);
        let k32 = self.c(32);
        let inv = self.sub(k32, s);
        let inv = self.andc(inv, 31);
        let b = self.bin(BinOp::Shl, hi, inv);
        let x = self.or(a, b);
        let sz = self.cmpc(Pred::Eq, s, 0);
        let v = self.sel(sz, lo, x);
        let w31 = self.andc(w, 31);
        let one = self.c(1);
        let m = self.bin(BinOp::Shl, one, w31);
        let m = self.sub(m, one);
        let vm = self.and(v, m);
        let big = self.cmpc(Pred::UGe, w, 32);
        self.sel(big, v, vm)
    }
    /// Exact IEEE-754 conversion of 10-bit (mb=5) or 11-bit (mb=6) unsigned float to f32 bits.
    fn uf_small_bits(&mut self, v: Value, mb: Value, mmask: Value, mshift: Value, den_scale: Value) -> Value {
        let e = self.bin(BinOp::LShr, v, mb);
        let m = self.and(v, mmask);
        let mnz = self.nz(m);
        let qn = self.c(0x7FC0_0000);
        let inf = self.c(0x7F80_0000);
        let special = self.sel(mnz, qn, inf);
        let fm = self.uitof(m);
        let den = self.fmul(fm, den_scale);
        let den = self.bits(den);
        let eb = self.addc(e, 112);
        let exp_bits = self.shlc(eb, 23);
        let mant_bits = self.bin(BinOp::Shl, m, mshift);
        let norm = self.or(exp_bits, mant_bits);
        let e0 = self.cmpc(Pred::Eq, e, 0);
        let r = self.sel(e0, den, norm);
        let e31 = self.cmpc(Pred::Eq, e, 31);
        self.sel(e31, special, r)
    }
    /// Decodes raw component bits `v` using pre-computed format parameters.
    #[allow(clippy::too_many_arguments)]
    fn fmt_decode_val(
        &mut self,
        v: Value,
        desc: &FmtDesc,
        big: Value,
        sh: Value,
        maxu: Value,
        maxs: Value,
        w16: Value,
        w32: Value,
        mb: Value,
        mmask: Value,
        mshift: Value,
        den_scale: Value,
        cge: Value,
        dflt: Value,
    ) -> Value {
        let x = self.bin(BinOp::Shl, v, sh);
        let x = self.bin(BinOp::AShr, x, sh);
        let sv = self.sel(big, v, x);
        let fv = self.uitof(v);
        let fsv = self.sitof(sv);
        let zf = self.fc(0.0);
        let onef = self.fc(1.0);
        let vz = self.cmpc(Pred::Eq, v, 0);
        let ge = self.fcmp(Pred::FOGe, fv, maxu);
        let q = self.fdiv(fv, maxu);
        let un = self.sel(ge, onef, q);
        let un = self.sel(vz, zf, un);
        let f_un = self.bits(un);
        let svz = self.cmpc(Pred::Eq, sv, 0);
        let ges = self.fcmp(Pred::FOGe, fsv, maxs);
        let qs = self.fdiv(fsv, maxs);
        let m1 = self.fc(-1.0);
        let qs = self.fmax(qs, m1);
        let sn = self.sel(ges, onef, qs);
        let sn = self.sel(svz, zf, sn);
        let f_sn = self.bits(sn);
        let h = self.s.f16_to_f32(v);
        let hb = self.bits(h);
        let ufb = self.uf_small_bits(v, mb, mmask, mshift, den_scale);
        let flt = self.sel(w16, hb, ufb);
        let flt = self.sel(w32, v, flt);
        let fvb = self.bits(fv);
        let fsvb = self.bits(fsv);
        let mut res = flt;
        for (k, val) in [(5usize, sv), (4, v), (3, fsvb), (2, fvb), (1, f_sn), (0, f_un)] {
            res = self.sel(desc.t[k], val, res);
        }
        self.sel(cge, dflt, res)
    }
    fn fmt_comp(&mut self, r: &[Value; 4], desc: &FmtDesc, c: u32) -> Value {
        let mut off = self.c(0);
        for k in 0..c {
            let w = self.lshrc(desc.info, 8 + 6 * k);
            let w = self.andc(w, 63);
            off = self.add(off, w);
        }
        let w = self.lshrc(desc.info, (8 + 6 * c) & 31);
        let w = self.andc(w, 63);
        let v = self.bits_at(r, off, w);
        let k32 = self.c(32);
        let sh = self.sub(k32, w);
        let sh = self.andc(sh, 31);
        let big = self.cmpc(Pred::UGe, w, 32);
        let w31 = self.andc(w, 31);
        let one = self.c(1);
        let mu = self.bin(BinOp::Shl, one, w31);
        let mu = self.sub(mu, one);
        let maxu = self.uitof(mu);
        let fmaxu = self.fc(4294967295.0);
        let maxu = self.sel(big, fmaxu, maxu);
        let wm1 = self.sub(w, one);
        let wm1 = self.andc(wm1, 31);
        let ms = self.bin(BinOp::Shl, one, wm1);
        let ms = self.sub(ms, one);
        let maxs = self.uitof(ms);
        let fmaxs = self.fc(2147483647.0);
        let maxs = self.sel(big, fmaxs, maxs);
        let w11 = self.cmpc(Pred::Eq, w, 11);
        let w16 = self.cmpc(Pred::Eq, w, 16);
        let w32 = self.cmpc(Pred::Eq, w, 32);
        let k6 = self.c(6);
        let k5 = self.c(5);
        let mb = self.sel(w11, k6, k5);
        let k63 = self.c(63);
        let k31 = self.c(31);
        let mmask = self.sel(w11, k63, k31);
        let k17 = self.c(17);
        let k18 = self.c(18);
        let mshift = self.sel(w11, k17, k18);
        let sc6 = self.s.m.f32c(0x3580_0000);
        let sc5 = self.s.m.f32c(0x3600_0000);
        let den_scale = self.sel(w11, sc6, sc5);
        let cge = self.cmpc(Pred::ULe, desc.n, c);
        let dflt = if c == 3 {
            let a = self.c(1);
            let b = self.c(0x3F80_0000);
            self.sel(desc.isint, a, b)
        } else {
            self.c(0)
        };
        self.fmt_decode_val(v, desc, big, sh, maxu, maxs, w16, w32, mb, mmask, mshift, den_scale, cge, dflt)
    }
    /// Decodes a single dynamically chosen component `ci` (0..3) from `r` (used for
    /// 2- and 3-component `FormatLoad`s so unused channels are not decoded).
    fn fmt_comp_dyn(&mut self, r: &[Value; 4], desc: &FmtDesc, ci: Value) -> Value {
        let z = self.c(0);
        let w0 = self.lshrc(desc.info, 8);
        let w0 = self.andc(w0, 63);
        let w1 = self.lshrc(desc.info, 14);
        let w1 = self.andc(w1, 63);
        let w2 = self.lshrc(desc.info, 20);
        let w2 = self.andc(w2, 63);
        let gt0 = self.cmpc(Pred::UGt, ci, 0);
        let gt1 = self.cmpc(Pred::UGt, ci, 1);
        let gt2 = self.cmpc(Pred::UGt, ci, 2);
        let o0 = self.sel(gt0, w0, z);
        let o1 = self.sel(gt1, w1, z);
        let o2 = self.sel(gt2, w2, z);
        let off = self.add(o0, o1);
        let off = self.add(off, o2);
        let k6 = self.c(6);
        let c6 = self.mul(ci, k6);
        let sh_info = self.addc(c6, 8);
        let sh_info = self.andc(sh_info, 31);
        let w = self.bin(BinOp::LShr, desc.info, sh_info);
        let w = self.andc(w, 63);
        let v = self.bits_at(r, off, w);
        let k32 = self.c(32);
        let sh = self.sub(k32, w);
        let sh = self.andc(sh, 31);
        let big = self.cmpc(Pred::UGe, w, 32);
        let w31 = self.andc(w, 31);
        let one = self.c(1);
        let mu = self.bin(BinOp::Shl, one, w31);
        let mu = self.sub(mu, one);
        let maxu = self.uitof(mu);
        let fmaxu = self.fc(4294967295.0);
        let maxu = self.sel(big, fmaxu, maxu);
        let wm1 = self.sub(w, one);
        let wm1 = self.andc(wm1, 31);
        let ms = self.bin(BinOp::Shl, one, wm1);
        let ms = self.sub(ms, one);
        let maxs = self.uitof(ms);
        let fmaxs = self.fc(2147483647.0);
        let maxs = self.sel(big, fmaxs, maxs);
        let w11 = self.cmpc(Pred::Eq, w, 11);
        let w16 = self.cmpc(Pred::Eq, w, 16);
        let w32 = self.cmpc(Pred::Eq, w, 32);
        let k5 = self.c(5);
        let mb = self.sel(w11, k6, k5);
        let k63 = self.c(63);
        let k31 = self.c(31);
        let mmask = self.sel(w11, k63, k31);
        let k17 = self.c(17);
        let k18 = self.c(18);
        let mshift = self.sel(w11, k17, k18);
        let sc6 = self.s.m.f32c(0x3580_0000);
        let sc5 = self.s.m.f32c(0x3600_0000);
        let den_scale = self.sel(w11, sc6, sc5);
        let cge = self.cmp(Pred::ULe, desc.n, ci);
        let is3 = self.cmpc(Pred::Eq, ci, 3);
        let onef = self.c(0x3F80_0000);
        let one_val = self.sel(desc.isint, one, onef);
        let dflt = self.sel(is3, one_val, z);
        self.fmt_decode_val(v, desc, big, sh, maxu, maxs, w16, w32, mb, mmask, mshift, den_scale, cge, dflt)
    }
    fn dst_sel(&mut self, c: &[Value; 4], sel: Value, isint: Value) -> Value {
        let k4 = self.c(4);
        let i = self.sub(sel, k4);
        let i = self.andc(i, 3);
        let v = self.pick4(c, i);
        let one = self.c(1);
        let onef = self.c(0x3F80_0000);
        let o = self.sel(isint, one, onef);
        let s1 = self.cmpc(Pred::Eq, sel, 1);
        let v = self.sel(s1, o, v);
        let s0 = self.cmpc(Pred::Eq, sel, 0);
        self.selc(s0, 0, v)
    }
    fn fmt_encode(&mut self, x: Value, info: Value, c: u32) -> Value {
        let ty = self.lshrc(info, 3);
        let ty = self.andc(ty, 7);
        let w = self.lshrc(info, 8 + 6 * c);
        let w = self.andc(w, 63);
        let f = self.f(x);
        let big = self.cmpc(Pred::UGe, w, 32);
        let one = self.c(1);
        let w31 = self.andc(w, 31);
        let mu = self.bin(BinOp::Shl, one, w31);
        let mu = self.sub(mu, one);
        let maxu = self.uitof(mu);
        let fmaxu = self.fc(4294967295.0);
        let maxu = self.sel(big, fmaxu, maxu);
        let wm1 = self.sub(w, one);
        let wm1 = self.andc(wm1, 31);
        let ms = self.bin(BinOp::Shl, one, wm1);
        let ms = self.sub(ms, one);
        let maxs = self.uitof(ms);
        let fmaxs = self.fc(2147483647.0);
        let maxs = self.sel(big, fmaxs, maxs);
        // ty 0: (uint)round(saturate(f) * maxu)
        let sat = self.un(super::ops_core::op::SATURATE, f);
        let u0 = self.fmul(sat, maxu);
        let u0 = self.un(super::ops_core::op::ROUND_NE, u0);
        let e0 = self.ftoui(u0);
        // ty 1: (uint)(int)round((isnan(f) ? 0 : clamp(f, -1, 1)) * maxs)
        let n = self.isnan(f);
        let z = self.fc(0.0);
        let m1 = self.fc(-1.0);
        let p1 = self.fc(1.0);
        let cl = self.fmax(f, m1);
        let cl = self.fmin(cl, p1);
        let cl = self.sel(n, z, cl);
        let s1 = self.fmul(cl, maxs);
        let s1 = self.un(super::ops_core::op::ROUND_NE, s1);
        let e1 = self.ftosi(s1);
        let e2 = self.ftoui(f);
        let e3 = self.ftosi(f);
        // ty 4/5: saturate UINT to [0, mu] and SINT to [!ms, ms] when w < 32
        let u4 = self.umin(x, mu);
        let e4 = self.sel(big, x, u4);
        let lo = self.not(ms);
        let s5 = self.imax(x, lo);
        let s5 = self.imin(s5, ms);
        let e5 = self.sel(big, x, s5);
        // floats (RTZ matches RDNA2 / HLSL fmt_encode so finite > 65504 saturates to max finite)
        let h = self.f16_rtz(f);
        let w11 = self.cmpc(Pred::Eq, w, 11);
        let k3f0 = self.c(0x3F0);
        let nanv = self.selc(w11, 0x7E0, k3f0);
        let h4 = self.lshrc(h, 4);
        let h4 = self.andc(h4, 0x7FF);
        let h5 = self.lshrc(h, 5);
        let h5 = self.andc(h5, 0x3FF);
        let small = self.sel(w11, h4, h5);
        let le0 = self.fcmp(Pred::FOLe, f, z);
        let small = self.selc(le0, 0, small);
        let small = self.sel(n, nanv, small);
        let w16 = self.cmpc(Pred::Eq, w, 16);
        let fl = self.sel(w16, h, small);
        let w32 = self.cmpc(Pred::Eq, w, 32);
        let r = self.sel(w32, x, fl);
        let mut res = r;
        for (k, val) in [(5u32, e5), (4, e4), (3, e3), (2, e2), (1, e1), (0, e0)] {
            let is = self.cmpc(Pred::Eq, ty, k);
            res = self.sel(is, val, res);
        }
        res
    }
    fn fmt_pack(&mut self, v: [Value; 4], info: Value) -> [Value; 4] {
        let z = self.c(0);
        let mut r = [z; 4];
        let n = self.andc(info, 7);
        let mut off = self.c(0);
        for c in 0..4u32 {
            let active = self.cmpc(Pred::UGt, n, c);
            let w = self.lshrc(info, 8 + 6 * c);
            let w = self.andc(w, 63);
            let e = self.fmt_encode(v[c as usize], info, c);
            // put_bits(r, off, w, e)
            let big = self.cmpc(Pred::UGe, w, 32);
            let one = self.c(1);
            let w31 = self.andc(w, 31);
            let m = self.bin(BinOp::Shl, one, w31);
            let m = self.sub(m, one);
            let all = self.c(u32::MAX);
            let m = self.sel(big, all, m);
            let e = self.and(e, m);
            let e = self.sel(active, e, z);
            let i = self.lshrc(off, 5);
            let s = self.andc(off, 31);
            let lo = self.bin(BinOp::Shl, e, s);
            let k32 = self.c(32);
            let inv = self.sub(k32, s);
            let inv = self.andc(inv, 31);
            let hi = self.bin(BinOp::LShr, e, inv);
            let snz = self.cmpc(Pred::Ne, s, 0);
            let sw = self.add(s, w);
            let over = self.cmpc(Pred::UGt, sw, 32);
            let spill = self.land(snz, over);
            let i3 = self.cmpc(Pred::ULt, i, 3);
            let spill = self.land(spill, i3);
            for j in 0..4u32 {
                let at = self.cmpc(Pred::Eq, i, j);
                let x = self.sel(at, lo, z);
                let mut nv = self.or(r[j as usize], x);
                if j > 0 {
                    let at1 = self.cmpc(Pred::Eq, i, j - 1);
                    let sp = self.land(at1, spill);
                    let y = self.sel(sp, hi, z);
                    nv = self.or(nv, y);
                }
                r[j as usize] = nv;
            }
            let wa = self.sel(active, w, z);
            off = self.add(off, wa);
        }
        r
    }

    // ------------------------------------------------------------ d16

    fn d16_data(&mut self, v: Value, half: u32) -> Value {
        if half == 1 {
            self.lshrc(v, 16)
        } else {
            v
        }
    }
    fn d16_merge(&mut self, v: Value, raw: Value, n: u32, signed: bool, half: u32) -> Value {
        let bits = 8 * n;
        let val = if signed {
            let x = self.shlc(raw, 32 - bits);
            self.ashrc(x, 32 - bits)
        } else {
            self.andc(raw, ((1u64 << bits) - 1) as u32)
        };
        match half {
            0 => {
                let a = self.andc(v, 0xFFFF_0000);
                let b = self.andc(val, 0xFFFF);
                self.or(a, b)
            }
            1 => {
                let a = self.andc(v, 0xFFFF);
                let b = self.shlc(val, 16);
                self.or(a, b)
            }
            _ => val,
        }
    }

    /// Writes VGPR `n` in a block where this lane is known active.
    fn vset(&mut self, n: u64, v: Value) {
        self.wr(VAR_VGPR + n as u32, v);
    }

    // ------------------------------------------------------------ SMEM

    pub fn smem(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        if name == "s_memtime" || name == "s_memrealtime" {
            let sdata = Self::field(d, "SDATA");
            let z = self.c(0);
            self.write64(&Dst::S(sdata), z, z);
            return Ok(());
        }
        let count = match name.trim_start_matches("s_buffer_load_").trim_start_matches("s_load_") {
            "dword" => 1,
            "dwordx2" => 2,
            "dwordx4" => 4,
            "dwordx8" => 8,
            "dwordx16" => 16,
            _ => return err(d, "SMEM"),
        };
        let sdata = Self::field(d, "SDATA");
        let sbase = Self::field(d, "SBASE");
        let imm = Self::fv(d, "OFFSET") as u32;
        let soff = rdna2::field_value(d.enc, &d.words, "SOFFSET").unwrap_or(125);
        let soff = if soff == 125 { self.c(0) } else { self.code32(soff, None, false).ok_or_else(|| format!("{}: op", rdna2::format(d)))? };
        let mut vals = if name.starts_with("s_buffer_load") {
            let (d0, d1, d2) = (self.sreg(sbase), self.sreg(sbase + 1), self.sreg(sbase + 2));
            let k = self.c(imm);
            let off = self.add(k, soff);
            self.buffer_window(d0, d1, &[Ea::U(k), Ea::U(soff)], false);
            let st = self.lshrc(d1, 16);
            let st = self.andc(st, 0x3FFF);
            let sz = self.cmpc(Pred::Eq, st, 0);
            let lim = self.mul(d2, st);
            let lim = self.sel(sz, d2, lim);
            let base = self.add(d0, off);
            let base_aligned = self.andc(base, !3);
            let raw = self.with_mem_i32s(None, count as usize, &mut |lw, m| {
                let mut out = Vec::with_capacity(count as usize);
                let mut done = 0u32;
                while done < count {
                    let chunk = (count - done).min(4);
                    let at = lw.addc(base_aligned, 4 * done);
                    let mask = ((1u32 << chunk) - 1) as u8;
                    out.extend(lw.s.raw_load(m, at, mask));
                    done += chunk;
                }
                out
            });
            let z = self.c(0);
            (0..count)
                .map(|i| {
                    let o = self.addc(off, 4 * i);
                    let ok = self.cmp(Pred::ULt, o, lim);
                    self.sel(ok, raw[i as usize], z)
                })
                .collect::<Vec<_>>()
        } else {
            // a repeat of a load nothing has changed since: its values (`reload`)
            let slot = self.reload.map(|r| super::reload::VAR_RELOAD + 16 * r.slot);
            if let (Some(var), Some(true)) = (slot, self.reload.map(|r| r.available)) {
                for i in 0..count {
                    let v = self.rd(var + i);
                    self.write32(&Dst::S(sdata + i as u64), v);
                }
                return Ok(());
            }
            let imm = ((imm << 11) as i32 >> 11) as u32;
            let b = self.sreg(sbase);
            let hi = self.sreg(sbase + 1);
            let k = self.c(imm);
            self.window_at(hi, b, &[Ea::S(k), Ea::U(soff)], false);
            let a = self.add(b, k);
            let a = self.add(a, soff);
            let base_aligned = self.andc(a, !3);
            self.with_mem_i32s(None, count as usize, &mut |lw, m| {
                let mut out = Vec::with_capacity(count as usize);
                let mut done = 0u32;
                while done < count {
                    let chunk = (count - done).min(4);
                    let at = lw.addc(base_aligned, 4 * done);
                    let mask = ((1u32 << chunk) - 1) as u8;
                    out.extend(lw.s.raw_load(m, at, mask));
                    done += chunk;
                }
                out
            })
        };
        let coherent = ["GLC", "DLC"].iter().any(|f| rdna2::field_value(d.enc, &d.words, f) == Some(1));
        if self.split && coherent {
            // both halves take the lower half's values
            let other = self.exchange_n(&vals);
            let up = self.upper;
            for (v, o) in vals.iter_mut().zip(other) {
                *v = self.sel(up, o, *v);
            }
        }
        // (only an `s_load_dword*` without GLC / DLC has a slot)
        if let Some(r) = self.reload {
            for (i, &v) in vals.iter().enumerate() {
                self.wr(super::reload::VAR_RELOAD + 16 * r.slot + i as u32, v);
            }
        }
        for (i, v) in vals.into_iter().enumerate() {
            self.write32(&Dst::S(sdata + i as u64), v);
        }
        Ok(())
    }

    // ------------------------------------------------------------ MUBUF / MTBUF

    pub fn mubuf(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        if name.starts_with("buffer_atomic_") {
            return self.buffer_atomic(d, name);
        }
        let small = match name {
            "buffer_load_ubyte" => Some((1u32, false, false)),
            "buffer_load_sbyte" => Some((1, true, false)),
            "buffer_load_ushort" => Some((2, false, false)),
            "buffer_load_sshort" => Some((2, true, false)),
            "buffer_store_byte" => Some((1, false, true)),
            "buffer_store_short" => Some((2, false, true)),
            _ => None,
        };
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let d16s = d16_small(name.trim_start_matches("buffer_"));
        let d16_format = buffer_format_d16(name);
        let (kind, count) = match (small, d16_format) {
            (Some(_), _) => (BufKind::Load, 1),
            (None, Some((kind, components, _))) => (kind, (components + 1) / 2),
            (None, None) if d16s.is_some() => (BufKind::Load, 1),
            (None, None) => buffer_op(name).ok_or_else(|| format!("{}: MUBUF", rdna2::format(d)))?,
        };
        if f("LDS") != 0 || f("TFE") != 0 {
            return err(d, "MUBUF lds/tfe");
        }
        let (vdata, vaddr, srsrc) = (f("VDATA"), f("VADDR"), Self::field(d, "SRSRC"));
        let soff = self.code32(f("SOFFSET"), None, false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
        let (offen, idxen) = (f("OFFEN") != 0, f("IDXEN") != 0);
        let (d0, d1, d2, d3) = (self.sreg(srsrc), self.sreg(srsrc + 1), self.sreg(srsrc + 2), self.sreg(srsrc + 3));
        let index = if idxen { self.vreg(vaddr) } else { self.c(0) };
        let voff = if offen { self.vreg(vaddr + idxen as u64) } else { self.c(0) };
        let stride = self.lshrc(d1, 16);
        let stride = self.andc(stride, 0x3FFF);
        let off = self.addc(voff, f("OFFSET") as u32);
        let a = self.add(d0, soff);
        let a = self.add(a, off);
        let si = self.mul(stride, index);
        let a = self.add(a, si);
        self.buffer_window(d0, d1, &[Ea::U(soff), Ea::U(voff), Ea::K(f("OFFSET") as u32), Ea::Mul(stride, index)], name.contains("store"));
        let act = self.act();
        let szero = self.cmpc(Pred::Eq, stride, 0);
        let idx_ok = self.cmp(Pred::ULt, index, d2);
        let in_range = |lw: &mut Lw, bytes: Value| -> Value {
            let e = lw.add(off, bytes);
            let ok = lw.cmp(Pred::ULe, e, d2);
            lw.sel(szero, ok, idx_ok)
        };
        if let Some((n, signed, store, half)) = d16s {
            let nb = self.c(n);
            let ok = in_range(self, nb);
            let v = self.vreg(vdata);
            if store {
                let data = self.d16_data(v, half);
                let c = self.land(act, ok);
                let z = self.c(0);
                let nv = self.c(n);
                return self.guarded(c, &mut |lw| lw.st_bytes(a, [data, z, z, z], nv));
            }
            let r0 = self.ld_raw_u32_guarded(Some(act), a);
            let z = self.c(0);
            let raw = self.sel(ok, r0, z);
            let nv = self.d16_merge(v, raw, n, signed, half);
            self.write32(&Dst::V(vdata), nv);
            return Ok(());
        }
        if let Some((n, signed, store)) = small {
            let nb = self.c(n);
            let ok = in_range(self, nb);
            let v = self.vreg(vdata);
            if store {
                let c = self.land(act, ok);
                let z = self.c(0);
                let nv = self.c(n);
                return self.guarded(c, &mut |lw| lw.st_bytes(a, [v, z, z, z], nv));
            }
            let r0 = self.ld_raw_u32_guarded(Some(act), a);
            let bits = 8 * n;
            let val = if signed {
                let x = self.shlc(r0, 32 - bits);
                self.ashrc(x, 32 - bits)
            } else {
                self.andc(r0, ((1u64 << bits) - 1) as u32)
            };
            let z = self.c(0);
            let val = self.sel(ok, val, z);
            self.write32(&Dst::V(vdata), val);
            return Ok(());
        }
        match kind {
            BufKind::Load => {
                let oks: Vec<Value> = (0..count)
                    .map(|i| {
                        let b = self.c(4 * i as u32 + 4);
                        in_range(self, b)
                    })
                    .collect();
                let loaded = self.mem_ld_n(Some(act), a, count as u32);
                let z = self.c(0);
                for i in 0..count {
                    let v = self.sel(oks[i as usize], loaded[i as usize], z);
                    self.write32(&Dst::V(vdata + i), v);
                }
                Ok(())
            }
            BufKind::Store => {
                let vals: Vec<Value> = (0..count).map(|i| self.vreg(vdata + i)).collect();
                let oks: Vec<Value> = (0..count)
                    .map(|i| {
                        let b = self.c(4 * i as u32 + 4);
                        in_range(self, b)
                    })
                    .collect();
                for i in 0..count as usize {
                    let at = self.addc(a, 4 * i as u32);
                    let c = self.land(act, oks[i]);
                    self.mem_st_n(Some(c), at, &[vals[i]]);
                }
                Ok(())
            }
            BufKind::FormatLoad => {
                let mtbuf_fmt = (d.enc.name == "ENC_MTBUF").then_some(f("FORMAT") as u32);
                let single_comp = match d16_format {
                    Some((_, components, _)) => components == 1,
                    None => count == 1,
                };
                if single_comp {
                    let c1 = self.fmt_comp1(d3, mtbuf_fmt);
                    let rng = in_range(self, c1.desc.nb);
                    let ok = self.land(c1.desc.has, rng);
                    let ea = self.add(a, c1.off_bytes);
                    let eb = self.andc(ea, !3);
                    let ea3 = self.andc(ea, 3);
                    let ea3_8 = self.shlc(ea3, 3);
                    let s = self.or(ea3_8, c1.off_bits);
                    let w = self.with_mem_i32s(Some(act), 2, &mut |lw, m| lw.s.raw_load(m, eb, 0x3));
                    let lo = self.bin(BinOp::LShr, w[0], s);
                    let k32 = self.c(32);
                    let inv = self.sub(k32, s);
                    let inv = self.andc(inv, 31);
                    let hi = self.bin(BinOp::Shl, w[1], inv);
                    let x = self.or(lo, hi);
                    let sz = self.cmpc(Pred::Eq, s, 0);
                    let v_raw = self.sel(sz, w[0], x);
                    let vm = self.and(v_raw, c1.mu);
                    let v = self.sel(c1.big, v_raw, vm);
                    let dec = self.fmt_decode_val(
                        v,
                        &c1.desc,
                        c1.big,
                        c1.sh,
                        c1.maxu,
                        c1.maxs,
                        c1.w16,
                        c1.w32,
                        c1.mb,
                        c1.mmask,
                        c1.mshift,
                        c1.den_scale,
                        c1.cge,
                        c1.dflt,
                    );
                    let s_val = self.sel(c1.s1, c1.one_val, dec);
                    let s_val = self.selc(c1.s0, 0, s_val);
                    let z = self.c(0);
                    if let Some((_, _, high)) = d16_format {
                        let cur0 = self.vreg(vdata);
                        let lo16 = self.andc(s_val, 0xFFFF);
                        let sf = self.f(s_val);
                        let h16 = self.s.f32_to_f16(sf);
                        let h = self.sel(c1.desc.isint, lo16, h16);
                        if high {
                            let h = self.shlc(h, 16);
                            let h = self.sel(ok, h, z);
                            let l = self.andc(cur0, 0xFFFF);
                            let res = self.or(l, h);
                            self.write32(&Dst::V(vdata), res);
                        } else {
                            let res = self.sel(ok, h, z);
                            self.write32(&Dst::V(vdata), res);
                        }
                    } else {
                        let res = self.sel(ok, s_val, z);
                        self.write32(&Dst::V(vdata), res);
                    }
                    return Ok(());
                }
                let desc = self.fmt_desc(d3, mtbuf_fmt);
                let rng = in_range(self, desc.nb);
                let ok = self.land(desc.has, rng);
                let r = self.ld_raw_guarded(Some(act), a);
                let z = self.c(0);
                let num_comp = d16_format.map_or(count, |(_, components, _)| components);
                if num_comp < 4 {
                    let decode_sel = |lw: &mut Lw, comp: u64| -> Value {
                        let sel = lw.lshrc(d3, 3 * comp as u32);
                        let sel = lw.andc(sel, 7);
                        let k4 = lw.c(4);
                        let ci = lw.sub(sel, k4);
                        let ci = lw.andc(ci, 3);
                        let dec = lw.fmt_comp_dyn(&r, &desc, ci);
                        let one = lw.c(1);
                        let onef = lw.c(0x3F80_0000);
                        let o = lw.sel(desc.isint, one, onef);
                        let s1 = lw.cmpc(Pred::Eq, sel, 1);
                        let v = lw.sel(s1, o, dec);
                        let s0 = lw.cmpc(Pred::Eq, sel, 0);
                        lw.selc(s0, 0, v)
                    };
                    if let Some((_, components, _)) = d16_format {
                        let half = |lw: &mut Lw, comp: u64| -> Value {
                            let s = decode_sel(lw, comp);
                            let lo = lw.andc(s, 0xFFFF);
                            let sf = lw.f(s);
                            let h = lw.s.f32_to_f16(sf);
                            lw.sel(desc.isint, lo, h)
                        };
                        for i in 0..count {
                            let lo = half(self, 2 * i);
                            let hi = if 2 * i + 1 < components { half(self, 2 * i + 1) } else { z };
                            let hi = self.shlc(hi, 16);
                            let v = self.or(lo, hi);
                            let v = self.sel(ok, v, z);
                            self.write32(&Dst::V(vdata + i), v);
                        }
                    } else {
                        for i in 0..count {
                            let s = decode_sel(self, i);
                            let v = self.sel(ok, s, z);
                            self.write32(&Dst::V(vdata + i), v);
                        }
                    }
                    return Ok(());
                }
                let c = [self.fmt_comp(&r, &desc, 0), self.fmt_comp(&r, &desc, 1), self.fmt_comp(&r, &desc, 2), self.fmt_comp(&r, &desc, 3)];
                if let Some((_, components, _)) = d16_format {
                    let half = |lw: &mut Lw, comp: u64| -> Value {
                        let sel = lw.lshrc(d3, 3 * comp as u32);
                        let sel = lw.andc(sel, 7);
                        let s = lw.dst_sel(&c, sel, desc.isint);
                        let lo = lw.andc(s, 0xFFFF);
                        let sf = lw.f(s);
                        let h = lw.s.f32_to_f16(sf);
                        lw.sel(desc.isint, lo, h)
                    };
                    for i in 0..count {
                        let lo = half(self, 2 * i);
                        let hi = if 2 * i + 1 < components { half(self, 2 * i + 1) } else { z };
                        let hi = self.shlc(hi, 16);
                        let v = self.or(lo, hi);
                        let v = self.sel(ok, v, z);
                        self.write32(&Dst::V(vdata + i), v);
                    }
                } else {
                    for i in 0..count {
                        let sel = self.lshrc(d3, 3 * i as u32);
                        let sel = self.andc(sel, 7);
                        let s = self.dst_sel(&c, sel, desc.isint);
                        let v = self.sel(ok, s, z);
                        self.write32(&Dst::V(vdata + i), v);
                    }
                }
                Ok(())
            }
            BufKind::FormatStore => {
                let mtbuf_fmt = (d.enc.name == "ENC_MTBUF").then_some(f("FORMAT") as u32);
                let desc = self.fmt_desc(d3, mtbuf_fmt);
                let info = desc.info;
                let nb = desc.nb;
                let rng = in_range(self, nb);
                let ok = self.land(desc.has, rng);
                let z = self.c(0);
                let vals: [Value; 4] = if let Some((_, components, high)) = d16_format {
                    let ty = self.lshrc(info, 3);
                    let ty = self.andc(ty, 7);
                    let mut out = [z; 4];
                    for (i, slot) in out.iter_mut().enumerate() {
                        if (i as u64) >= components {
                            continue;
                        }
                        let v = self.vreg(vdata + if high { 0 } else { i as u64 / 2 });
                        let hsh = if high || i & 1 != 0 { self.lshrc(v, 16) } else { v };
                        let h = self.andc(hsh, 0xFFFF);
                        let sx = self.sx16(h);
                        let fh = self.s.f16_to_f32(h);
                        let fb = self.bits(fh);
                        let t5 = self.cmpc(Pred::Eq, ty, 5);
                        let t4 = self.cmpc(Pred::Eq, ty, 4);
                        let x = self.sel(t5, sx, fb);
                        *slot = self.sel(t4, h, x);
                    }
                    out
                } else {
                    let mut out = [z; 4];
                    for (i, slot) in out.iter_mut().enumerate().take(count as usize) {
                        *slot = self.vreg(vdata + i as u64);
                    }
                    out
                };
                let c = self.land(act, ok);
                self.guarded(c, &mut |lw| {
                    let packed = lw.fmt_pack(vals, info);
                    lw.st_bytes(a, packed, nb)
                })
            }
        }
    }

    // ------------------------------------------------------------ atomics

    /// A 32-bit atomic; returns the old value (valid in the guarded block it runs in).
    fn rmw32(&mut self, d: &Decoded, op: &str, at: At, v: Value, v2: Value) -> LResult<Value> {
        let direct = |lw: &mut Lw, o: AtomicOp, rm: RmwOp, val: Value| -> Value {
            match at {
                At::Mem(a) => {
                    lw.mraw_atomic(o, a, val)
                }
                At::Lds(i) => {
                    let g = lw.lds.expect("lds");
                    lw.s.gs_atomic(rm, g, i, val)
                }
            }
        };
        let z = self.c(0);
        Ok(match op {
            "add" | "add_u32" => direct(self, AtomicOp::Add, RmwOp::Add, v),
            "sub" | "sub_u32" => {
                let n = self.sub(z, v);
                direct(self, AtomicOp::Add, RmwOp::Add, n)
            }
            "and" | "and_b32" => direct(self, AtomicOp::And, RmwOp::And, v),
            "or" | "or_b32" => direct(self, AtomicOp::Or, RmwOp::Or, v),
            "xor" | "xor_b32" => direct(self, AtomicOp::Xor, RmwOp::Xor, v),
            "swap" | "wrxchg_b32" => direct(self, AtomicOp::Exchange, RmwOp::Xchg, v),
            "umin" | "min_u32" => direct(self, AtomicOp::UMin, RmwOp::UMin, v),
            "umax" | "max_u32" => direct(self, AtomicOp::UMax, RmwOp::UMax, v),
            "smin" | "min_i32" => direct(self, AtomicOp::IMin, RmwOp::Min, v),
            "smax" | "max_i32" => direct(self, AtomicOp::IMax, RmwOp::Max, v),
            "cmpswap" | "cmpst_b32" => self.cmpxchg(at, v2, v),
            "fcmpswap" | "cmpst_f32" => self.fcas32(at, v2, v),
            "fmin" | "fmax" | "min_f32" | "max_f32" => {
                let min = op == "fmin" || op == "min_f32";
                match at {
                    At::Mem(a) => self.float_minmax_mem(min, a, v)?,
                    At::Lds(_) => self.cas(at, &mut |lw, c| {
                        let (x, y) = (lw.f(c), lw.f(v));
                        let r = if min { lw.fmin(x, y) } else { lw.fmax(x, y) };
                        lw.bits(r)
                    }),
                }
            }
            "rsub" | "rsub_u32" => self.cas(at, &mut |lw, c| lw.sub(v, c)),
            "inc" | "inc_u32" => self.cas(at, &mut |lw, c| {
                let ge = lw.cmp(Pred::UGe, c, v);
                let c1 = lw.addc(c, 1);
                lw.selc(ge, 0, c1)
            }),
            "dec" | "dec_u32" => self.cas(at, &mut |lw, c| {
                let z = lw.cmpc(Pred::Eq, c, 0);
                let gt = lw.cmp(Pred::UGt, c, v);
                let w = lw.lor(z, gt);
                let one = lw.c(1);
                let c1 = lw.sub(c, one);
                lw.sel(w, v, c1)
            }),
            "csub" => self.cas(at, &mut |lw, c| {
                let ge = lw.cmp(Pred::UGe, c, v);
                let s = lw.sub(c, v);
                let z = lw.c(0);
                lw.sel(ge, s, z)
            }),
            "add_f32" => self.cas(at, &mut |lw, c| {
                let (x, y) = (lw.f(c), lw.f(v));
                let r = lw.fadd(x, y);
                lw.bits(r)
            }),
            "mskor_b32" => self.cas(at, &mut |lw, c| {
                let n = lw.not(v);
                let x = lw.and(c, n);
                lw.or(x, v2)
            }),
            _ => return err(d, "atomic"),
        })
    }

    fn cmpxchg(&mut self, at: At, cmp: Value, new: Value) -> Value {
        match at {
            At::Mem(a) => {
                self.mraw_cmpxchg(a, cmp, new)
            }
            At::Lds(i) => {
                let g = self.lds.expect("lds");
                self.s.gs_cmpxchg(g, i, cmp, new)
            }
        }
    }

    /// A compare-exchange loop (at most 4096 rounds) computing `new(old)`; returns the old value.
    fn cas(&mut self, at: At, new: &mut dyn FnMut(&mut Lw, Value) -> Value) -> Value {
        let init = match at {
            At::Mem(a) => self.mem_ld(a),
            At::Lds(i) => {
                let g = self.lds.expect("lds");
                self.s.gs_load(g, i)
            }
        };
        let pre = self.cur();
        let lp = self.block(Some(2));
        let ex = self.block(Some(1));
        self.br(lp);
        self.s.m.set_block(lp);
        let i32 = self.s.m.i32();
        let c = self.s.m.phi(i32);
        let i = self.s.m.phi(i32);
        self.s.m.add_incoming(c, init, pre);
        let z = self.c(0);
        self.s.m.add_incoming(i, z, pre);
        let n = new(self, c);
        let p = self.cmpxchg(at, c, n);
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

    /// Bitwise IEEE-754 f32 ordered equality: `-0.0 == +0.0`, `NaN != NaN` (denormals kept).
    pub(super) fn feq_bits32(&mut self, a: Value, b: Value) -> Value {
        let abs_a = self.andc(a, 0x7FFF_FFFF);
        let not_nan = self.cmpc(Pred::ULe, abs_a, 0x7F80_0000);
        let same = self.cmp(Pred::Eq, a, b);
        let both = self.or(a, b);
        let mag = self.andc(both, 0x7FFF_FFFF);
        let both_zero = self.cmpc(Pred::Eq, mag, 0);
        let eq = self.lor(same, both_zero);
        self.land(not_nan, eq)
    }

    /// Bitwise IEEE-754 f64 ordered equality in pure i64 ops (no DoubleExtensions needed).
    fn feq_bits64(&mut self, a: Value, b: Value) -> Value {
        let mask = self.c64(0x7FFF_FFFF_FFFF_FFFF);
        let inf = self.c64(0x7FF0_0000_0000_0000);
        let z = self.c64(0);
        let abs_a = self.and(a, mask);
        let not_nan = self.cmp(Pred::ULe, abs_a, inf);
        let same = self.cmp(Pred::Eq, a, b);
        let both = self.or(a, b);
        let mag = self.and(both, mask);
        let both_zero = self.cmp(Pred::Eq, mag, z);
        let eq = self.lor(same, both_zero);
        self.land(not_nan, eq)
    }

    /// IEEE-754 32-bit float compare-exchange loop (at most 4096 rounds); returns the old value.
    fn fcas32(&mut self, at: At, cmp: Value, new: Value) -> Value {
        let z = self.c(0);
        let init = match at {
            At::Mem(a) => self.mraw_atomic(AtomicOp::Or, a, z),
            At::Lds(i) => {
                let g = self.lds.expect("lds");
                self.s.gs_atomic(RmwOp::Or, g, i, z)
            }
        };
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
        let p = self.cmpxchg(at, c, new);
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

    /// IEEE-754 64-bit float compare-exchange loop (at most 4096 rounds); returns the old value.
    fn fcas64(&mut self, a: Value, cmp: Value, new: Value) -> Value {
        let z64 = self.c64(0);
        let init = self.mraw_atomic(AtomicOp::Or, a, z64);
        let pre = self.cur();
        let lp = self.block(Some(2));
        let try_cas = self.block(Some(1));
        let ex = self.block(Some(2));
        self.br(lp);
        self.s.m.set_block(lp);
        let i64t = self.i64();
        let i32 = self.s.m.i32();
        let c = self.s.m.phi(i64t);
        let i = self.s.m.phi(i32);
        self.s.m.add_incoming(c, init, pre);
        let z = self.c(0);
        self.s.m.add_incoming(i, z, pre);
        let eq = self.feq_bits64(c, cmp);
        let lp_end = self.cur();
        self.cond_br(eq, try_cas, ex);
        self.s.m.set_block(try_cas);
        let p = self.mraw_cmpxchg(a, c, new);
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

    /// Float min / max on `mem` as one integer atomic (the recompiler's `float_minmax_mem`).
    fn float_minmax_mem(&mut self, min: bool, a: Value, v: Value) -> LResult<Value> {
        let (pos, neg) = if min { (AtomicOp::IMin, AtomicOp::UMax) } else { (AtomicOp::IMax, AtomicOp::UMin) };
        let tmp = self.tmp_var();
        let z = self.c(0);
        self.wr(tmp, z);
        let pp = self.cmpc(Pred::ULe, v, 0x7F80_0000);
        self.guarded(pp, &mut |lw| {
            let o = lw.mraw_atomic(pos, a, v);
            lw.wr(tmp, o);
            Ok(())
        })?;
        let np = self.lnot(pp);
        self.guarded(np, &mut |lw| {
            let o = lw.mraw_atomic(neg, a, v);
            lw.wr(tmp, o);
            Ok(())
        })?;
        Ok(self.rd(tmp))
    }

    pub fn tmp_var(&mut self) -> u32 {
        self.ntmp += 1;
        VAR_TMP + self.ntmp
    }

    /// 64-bit atomic at `a`; data at VGPRs `data`/+1, compare at `cmp`/+1; old → `dst` (GLC).
    #[allow(clippy::too_many_arguments)]
    fn atomic64(&mut self, d: &Decoded, op: &str, a: Value, data: u64, cmp: u64, glc: bool, dst: u64, active: Value) -> LResult<()> {
        let (lo, hi) = (self.vreg(data), self.vreg(data + 1));
        let (clo, chi) = (self.vreg(cmp), self.vreg(cmp + 1));
        let v = self.pack64(lo, hi);
        let cv = self.pack64(clo, chi);
        let ops = op.to_string();
        let mut failed = None;
        self.guarded(active, &mut |lw| {
            let z64 = lw.c64(0);
            let old = match ops.as_str() {
                "add" => lw.mraw_atomic(AtomicOp::Add, a, v),
                "sub" => {
                    let n = lw.sub(z64, v);
                    lw.mraw_atomic(AtomicOp::Add, a, n)
                }
                "and" => lw.mraw_atomic(AtomicOp::And, a, v),
                "or" => lw.mraw_atomic(AtomicOp::Or, a, v),
                "xor" => lw.mraw_atomic(AtomicOp::Xor, a, v),
                "swap" => lw.mraw_atomic(AtomicOp::Exchange, a, v),
                "umin" => lw.mraw_atomic(AtomicOp::UMin, a, v),
                "umax" => lw.mraw_atomic(AtomicOp::UMax, a, v),
                "smin" => lw.mraw_atomic(AtomicOp::IMin, a, v),
                "smax" => lw.mraw_atomic(AtomicOp::IMax, a, v),
                "cmpswap" => lw.mraw_cmpxchg(a, cv, v),
                "fcmpswap" => lw.fcas64(a, cv, v),
                "fmin" | "fmax" => {
                    let min = ops == "fmin";
                    let (pos, neg) = if min { (AtomicOp::IMin, AtomicOp::UMax) } else { (AtomicOp::IMax, AtomicOp::UMin) };
                    let (tl, th) = (lw.tmp_var(), lw.tmp_var());
                    let z = lw.c(0);
                    lw.wr(tl, z);
                    lw.wr(th, z);
                    let lim = lw.c64(0x7FF0_0000_0000_0000);
                    let pp = lw.cmp(Pred::ULe, v, lim);
                    for (cond, o) in [(pp, pos), (lw.lnot(pp), neg)] {
                        lw.guarded(cond, &mut |lw| {
                            let r = lw.mraw_atomic(o, a, v);
                            let (l, h) = lw.unpack64(r);
                            lw.wr(tl, l);
                            lw.wr(th, h);
                            Ok(())
                        })?;
                    }
                    let (l, h) = (lw.rd(tl), lw.rd(th));
                    lw.pack64(l, h)
                }
                "inc" | "dec" => {
                    let inc = ops == "inc";
                    let l0 = lw.mem_ld(a);
                    let a4 = lw.addc(a, 4);
                    let h0 = lw.mem_ld(a4);
                    let init = lw.pack64(l0, h0);
                    let pre = lw.cur();
                    let lp = lw.block(Some(2));
                    let ex = lw.block(Some(1));
                    lw.br(lp);
                    lw.s.m.set_block(lp);
                    let i64t = lw.i64();
                    let i32 = lw.s.m.i32();
                    let c = lw.s.m.phi(i64t);
                    let i = lw.s.m.phi(i32);
                    lw.s.m.add_incoming(c, init, pre);
                    let z = lw.c(0);
                    lw.s.m.add_incoming(i, z, pre);
                    let one = lw.c64(1);
                    let n = if inc {
                        let ge = lw.cmp(Pred::UGe, c, v);
                        let c1 = lw.add(c, one);
                        lw.sel(ge, z64, c1)
                    } else {
                        let cz = lw.cmp(Pred::Eq, c, z64);
                        let gt = lw.cmp(Pred::UGt, c, v);
                        let w = lw.lor(cz, gt);
                        let c1 = lw.sub(c, one);
                        lw.sel(w, v, c1)
                    };
                    let p = lw.mraw_cmpxchg(a, c, n);
                    let ok = lw.cmp(Pred::Eq, p, c);
                    let i1 = lw.addc(i, 1);
                    let out = lw.cmpc(Pred::UGe, i1, 4096);
                    let stop = lw.lor(ok, out);
                    let me = lw.cur();
                    lw.s.m.add_incoming(c, p, me);
                    lw.s.m.add_incoming(i, i1, me);
                    lw.cond_br(stop, ex, lp);
                    lw.s.m.set_block(ex);
                    p
                }
                _ => {
                    failed = Some("atomic 64");
                    return Ok(());
                }
            };
            if glc {
                let (l, h) = lw.unpack64(old);
                lw.vset(dst, l);
                lw.vset(dst + 1, h);
            }
            Ok(())
        })?;
        if let Some(why) = failed {
            return err(d, why);
        }
        Ok(())
    }

    fn buffer_atomic(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let op = name.trim_start_matches("buffer_atomic_");
        let (vdata, vaddr, srsrc) = (f("VDATA"), f("VADDR"), Self::field(d, "SRSRC"));
        let soff = self.code32(f("SOFFSET"), None, false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
        let (offen, idxen) = (f("OFFEN") != 0, f("IDXEN") != 0);
        let (d0, d1, d2) = (self.sreg(srsrc), self.sreg(srsrc + 1), self.sreg(srsrc + 2));
        let index = if idxen { self.vreg(vaddr) } else { self.c(0) };
        let voff = if offen { self.vreg(vaddr + idxen as u64) } else { self.c(0) };
        let x2 = op.ends_with("_x2");
        let op = op.trim_end_matches("_x2");
        let bytes = if x2 { 8 } else { 4 };
        let stride = self.lshrc(d1, 16);
        let stride = self.andc(stride, 0x3FFF);
        let off = self.addc(voff, f("OFFSET") as u32);
        let a = self.add(d0, soff);
        let a = self.add(a, off);
        let si = self.mul(stride, index);
        let a = self.add(a, si);
        self.buffer_window(d0, d1, &[Ea::U(soff), Ea::U(voff), Ea::K(f("OFFSET") as u32), Ea::Mul(stride, index)], true);
        let a = self.andc(a, !(bytes - 1));
        let szero = self.cmpc(Pred::Eq, stride, 0);
        let e = self.addc(off, bytes);
        let r1 = self.cmp(Pred::ULe, e, d2);
        let r2 = self.cmp(Pred::ULt, index, d2);
        let ok = self.sel(szero, r1, r2);
        let act = self.act();
        let active = self.land(act, ok);
        if x2 {
            return self.atomic64(d, op, a, vdata, vdata + 2, f("GLC") == 1, vdata, active);
        }
        let (v, cmp) = (self.vreg(vdata), self.vreg(vdata + 1));
        let glc = f("GLC") == 1;
        let ops = op.to_string();
        let tmp = self.tmp_var();
        let z = self.c(0);
        self.wr(tmp, z);
        let mut e = None;
        self.guarded(active, &mut |lw| {
            match lw.rmw32(d, &ops, At::Mem(a), v, cmp) {
                Ok(o) => lw.wr(tmp, o),
                Err(x) => e = Some(x),
            }
            Ok(())
        })?;
        if let Some(x) = e {
            return Err(x);
        }
        if glc {
            // select(act, select(ok, old, 0), v)
            let old = self.rd(tmp);
            let z = self.c(0);
            let r = self.sel(ok, old, z);
            let r = self.sel(act, r, v);
            self.vset(vdata, r);
        }
        Ok(())
    }

    // ------------------------------------------------------------ FLAT / global

    fn flat_base(&mut self, d: &Decoded, write: bool) -> (Value, bool) {
        let global = d.enc.name == "ENC_FLAT_GLBL";
        let addr = self.vreg(Self::fv(d, "ADDR"));
        let saddr = Self::fv(d, "SADDR");
        self.flat_window(d, write);
        let base = if global && saddr != 0x7D && saddr != 0x7F {
            let s = self.sreg(saddr);
            self.add(s, addr)
        } else {
            addr
        };
        let raw = Self::fv(d, "OFFSET") as u32;
        let off = if global { (((raw << 20) as i32) >> 20) as u32 } else { raw };
        (self.addc(base, off), global)
    }

    pub fn flat(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let op = name.trim_start_matches("global_").trim_start_matches("flat_").trim_start_matches("scratch_");
        if d.enc.name == "ENC_FLAT_SCRATCH" {
            return self.scratch(d, op);
        }
        if let Some(a) = op.strip_prefix("atomic_") {
            return self.flat_atomic(d, a);
        }
        let act = self.act();
        if let Some((n, signed, store, half)) = flat_small(op) {
            let (a, _) = self.flat_base(d, store);
            if store {
                let v = self.vreg(Self::fv(d, "DATA"));
                let data = self.d16_data(v, half);
                let z = self.c(0);
                let nv = self.c(n);
                return self.guarded(act, &mut |lw| lw.st_bytes(a, [data, z, z, z], nv));
            }
            let vd = Self::fv(d, "VDST");
            let v = self.vreg(vd);
            let r0 = self.ld_raw_u32_guarded(Some(act), a);
            let nv = self.d16_merge(v, r0, n, signed, half);
            self.write32(&Dst::V(vd), nv);
            return Ok(());
        }
        let (store, count) = match op {
            "load_dword" | "load_dword_addtid" => (false, 1),
            "load_dwordx2" => (false, 2),
            "load_dwordx3" => (false, 3),
            "load_dwordx4" => (false, 4),
            "store_dword" | "store_dword_addtid" => (true, 1),
            "store_dwordx2" => (true, 2),
            "store_dwordx3" => (true, 3),
            "store_dwordx4" => (true, 4),
            _ => return err(d, "FLAT"),
        };
        let a = if op.ends_with("_addtid") {
            self.addtid_window(d, store);
            let saddr = Self::fv(d, "SADDR");
            let lane = self.lane;
            let l4 = self.shlc(lane, 2);
            let base = match saddr {
                0x7D | 0x7F => l4,
                _ => {
                    let s = self.sreg(saddr);
                    self.add(s, l4)
                }
            };
            let raw = Self::fv(d, "OFFSET") as u32;
            let off = if d.enc.name == "ENC_FLAT_GLBL" { (((raw << 20) as i32) >> 20) as u32 } else { raw };
            self.addc(base, off)
        } else {
            self.flat_base(d, store).0
        };
        if store {
            let vals: Vec<Value> = (0..count).map(|i| self.vreg(Self::fv(d, "DATA") + i)).collect();
            self.mem_st_n(Some(act), a, &vals);
            Ok(())
        } else {
            let vd = Self::fv(d, "VDST");
            let loaded = self.mem_ld_n(Some(act), a, count as u32);
            for i in 0..count {
                self.write32(&Dst::V(vd + i), loaded[i as usize]);
            }
            Ok(())
        }
    }

    fn flat_atomic(&mut self, d: &Decoded, op: &str) -> LResult<()> {
        let (a, _) = self.flat_base(d, true);
        let x2 = op.ends_with("_x2");
        let op = op.trim_end_matches("_x2");
        let a = self.andc(a, if x2 { !7 } else { !3 });
        let act = self.act();
        let glc = Self::fv(d, "GLC") == 1;
        let data = Self::fv(d, "DATA");
        if x2 {
            return self.atomic64(d, op, a, data, data + 2, glc, Self::fv(d, "VDST"), act);
        }
        let v = self.vreg(data);
        let cmpv = self.vreg(data + 1);
        let vd = Self::fv(d, "VDST");
        let ops = op.to_string();
        let mut e = None;
        self.guarded(act, &mut |lw| {
            match lw.rmw32(d, &ops, At::Mem(a), v, cmpv) {
                Ok(o) => {
                    if glc {
                        lw.vset(vd, o);
                    }
                }
                Err(x) => e = Some(x),
            }
            Ok(())
        })?;
        match e {
            Some(x) => Err(x),
            None => Ok(()),
        }
    }

    // ------------------------------------------------------------ LDS / DS

    /// LDS dword index of byte address `at` (`(at >> 2) & 8191`, below the exchange words
    /// in the split form).
    fn lds_index(&mut self, at: Value) -> Value {
        let i = self.lshrc(at, 2);
        let i = self.andc(i, 8191);
        if self.split {
            let k = self.c(self.lds_top);
            self.umin(i, k)
        } else {
            i
        }
    }
    fn lds_ld(&mut self, at: Value) -> Value {
        let i = self.lds_index(at);
        let g = self.lds.expect("lds");
        self.s.gs_load(g, i)
    }
    fn lds_st(&mut self, at: Value, v: Value) {
        self.geom_note_store(at);
        let i = self.lds_index(at);
        let g = self.lds.expect("lds");
        self.s.gs_store(g, i, v);
    }

    pub fn ds(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let (o0, o1) = (f("OFFSET0"), f("OFFSET1"));
        let off16 = (o0 | o1 << 8) as u32;
        match name {
            "ds_swizzle_b32" => {
                let a = self.vreg(f("ADDR"));
                let l = self.swizzle_lane(off16 as u64);
                let (el, eh) = self.exec_mask();
                let v = self.read_lane_half(a, l);
                let on = self.lanebit_at(el, eh, l);
                let z = self.c(0);
                let t = self.sel(on, v, z);
                self.write32(&Dst::V(f("VDST")), t);
                return Ok(());
            }
            "ds_bpermute_b32" => {
                let (a, v) = (self.vreg(f("ADDR")), self.vreg(f("DATA0")));
                let m = if self.wave64 { 63 } else { 31 };
                let (el, eh) = self.exec_mask();
                let x = self.addc(a, off16);
                let x = self.lshrc(x, 2);
                let l = self.andc(x, m);
                let r = self.read_lane_any(v, l);
                let on = self.lanebit_at(el, eh, l);
                let z = self.c(0);
                let t = self.sel(on, r, z);
                self.write32(&Dst::V(f("VDST")), t);
                return Ok(());
            }
            _ => {}
        }
        if f("GDS") != 0 {
            return err(d, "GDS");
        }
        let act = self.act();
        if matches!(name, "ds_write_addtid_b32" | "ds_read_addtid_b32") {
            let m0 = self.sreg(124);
            let base = self.andc(m0, 0xFFFF);
            let base = self.addc(base, off16);
            let lane = self.lane;
            let l4 = self.shlc(lane, 2);
            let at = self.add(base, l4);
            if name == "ds_write_addtid_b32" {
                let src = self.vreg(f("DATA0"));
                return self.guarded(act, &mut |lw| {
                    lw.lds_st(at, src);
                    Ok(())
                });
            }
            let vd = f("VDST");
            return self.guarded(act, &mut |lw| {
                let v = lw.lds_ld(at);
                lw.vset(vd, v);
                Ok(())
            });
        }
        let a = self.vreg(f("ADDR"));
        if let Some((bits, signed, write, half)) = ds_d16(name) {
            let at = self.addc(a, off16);
            let sh = self.andc(at, 3);
            let sh = self.shlc(sh, 3);
            let mask = ((1u64 << bits) - 1) as u32;
            if write {
                let v = self.vreg(f("DATA0"));
                let v = self.d16_data(v, half);
                return self.guarded(act, &mut |lw| lw.lds_bits_store(at, sh, mask, v));
            }
            let vd = f("VDST");
            let v = self.vreg(vd);
            return self.guarded(act, &mut |lw| {
                let w = lw.lds_ld(at);
                let raw = lw.bin(BinOp::LShr, w, sh);
                let nv = lw.d16_merge(v, raw, bits / 8, signed, half);
                lw.vset(vd, nv);
                Ok(())
            });
        }
        let small = match name {
            "ds_write_b8" => Some((8u32, false, true)),
            "ds_write_b16" => Some((16, false, true)),
            "ds_read_u8" => Some((8, false, false)),
            "ds_read_i8" => Some((8, true, false)),
            "ds_read_u16" => Some((16, false, false)),
            "ds_read_i16" => Some((16, true, false)),
            _ => None,
        };
        if let Some((bits, signed, write)) = small {
            let at = self.addc(a, off16);
            let sh = self.andc(at, 3);
            let sh = self.shlc(sh, 3);
            let mask = ((1u64 << bits) - 1) as u32;
            if write {
                let v = self.vreg(f("DATA0"));
                return self.guarded(act, &mut |lw| lw.lds_bits_store(at, sh, mask, v));
            }
            let vd = f("VDST");
            return self.guarded(act, &mut |lw| {
                let w = lw.lds_ld(at);
                let x = lw.bin(BinOp::LShr, w, sh);
                let x = lw.andc(x, mask);
                let val = if signed {
                    let y = lw.shlc(x, 32 - bits);
                    lw.ashrc(y, 32 - bits)
                } else {
                    x
                };
                lw.vset(vd, val);
                Ok(())
            });
        }
        let rw = match name {
            "ds_read_b32" => Some((false, 1, 1, false)),
            "ds_read_b64" => Some((false, 2, 1, false)),
            "ds_read_b96" => Some((false, 3, 1, false)),
            "ds_read_b128" => Some((false, 4, 1, false)),
            "ds_write_b32" => Some((true, 1, 1, false)),
            "ds_write_b64" => Some((true, 2, 1, false)),
            "ds_write_b96" => Some((true, 3, 1, false)),
            "ds_write_b128" => Some((true, 4, 1, false)),
            "ds_read2_b32" => Some((false, 1, 4, true)),
            "ds_read2_b64" => Some((false, 2, 8, true)),
            "ds_read2st64_b32" => Some((false, 1, 256, true)),
            "ds_read2st64_b64" => Some((false, 2, 512, true)),
            "ds_write2_b32" => Some((true, 1, 4, true)),
            "ds_write2_b64" => Some((true, 2, 8, true)),
            "ds_write2st64_b32" => Some((true, 1, 256, true)),
            "ds_write2st64_b64" => Some((true, 2, 512, true)),
            _ => None,
        };
        if let Some((write, n, stride, two)) = rw {
            let addrs: Vec<Value> = if two {
                vec![self.addc(a, (o0 * stride) as u32), self.addc(a, (o1 * stride) as u32)]
            } else {
                vec![self.addc(a, off16)]
            };
            let (vd, d0, d1) = (f("VDST"), f("DATA0"), f("DATA1"));
            let srcs: Vec<Vec<Value>> = if write {
                (0..addrs.len()).map(|e| (0..n).map(|k| self.vreg(if e == 0 { d0 } else { d1 } + k)).collect()).collect()
            } else {
                Vec::new()
            };
            return self.guarded(act, &mut |lw| {
                for (e, &at) in addrs.iter().enumerate() {
                    for k in 0..n {
                        let slot = lw.addc(at, 4 * k as u32);
                        if write {
                            lw.lds_st(slot, srcs[e][k as usize]);
                        } else {
                            let v = lw.lds_ld(slot);
                            lw.vset(vd + e as u64 * n + k, v);
                        }
                    }
                }
                Ok(())
            });
        }
        let (op, rtn) = match name.strip_prefix("ds_") {
            Some(r) if r.contains("_rtn_") => (r.replacen("_rtn", "", 1), true),
            Some(r) => (r.to_string(), false),
            None => return err(d, "DS"),
        };
        if op == "wrxchg2_b32" || op == "wrxchg2st64_b32" {
            let stride = if op == "wrxchg2_b32" { 4 } else { 256 };
            let (a0, a1) = (self.addc(a, (o0 * stride) as u32), self.addc(a, (o1 * stride) as u32));
            let (v0, v1) = (self.vreg(f("DATA0")), self.vreg(f("DATA1")));
            let vd = f("VDST");
            return self.guarded(act, &mut |lw| {
                let g = lw.lds.expect("lds");
                let i0 = lw.lds_index(a0);
                let r0 = lw.s.gs_atomic(RmwOp::Xchg, g, i0, v0);
                let i1 = lw.lds_index(a1);
                let r1 = lw.s.gs_atomic(RmwOp::Xchg, g, i1, v1);
                if rtn {
                    lw.vset(vd, r0);
                    lw.vset(vd + 1, r1);
                }
                Ok(())
            });
        }
        let (d0, d1) = (self.vreg(f("DATA0")), self.vreg(f("DATA1")));
        let (v, v2) = if op.starts_with("cmpst_") { (d1, d0) } else { (d0, d1) };
        let at = self.addc(a, off16);
        let vd = f("VDST");
        let mut e = None;
        self.guarded(act, &mut |lw| {
            let i = lw.lds_index(at);
            match lw.rmw32(d, &op, At::Lds(i), v, v2) {
                Ok(o) => {
                    if rtn {
                        lw.vset(vd, o);
                    }
                }
                Err(x) => e = Some(x),
            }
            Ok(())
        })?;
        match e {
            Some(x) => Err(x),
            None => Ok(()),
        }
    }

    /// `InterlockedAnd(slot, ~(mask << sh)); InterlockedOr(slot, (v & mask) << sh)`
    fn lds_bits_store(&mut self, at: Value, sh: Value, mask: u32, v: Value) -> LResult<()> {
        let g = self.lds.expect("lds");
        let i = self.lds_index(at);
        let k = self.c(mask);
        let m = self.bin(BinOp::Shl, k, sh);
        let nm = self.not(m);
        self.s.gs_atomic(RmwOp::And, g, i, nm);
        let vm = self.andc(v, mask);
        let vs = self.bin(BinOp::Shl, vm, sh);
        self.s.gs_atomic(RmwOp::Or, g, i, vs);
        Ok(())
    }

    /// `ds_swizzle_b32`'s source lane of this lane (`emit::swizzle_lane`; the upper
    /// 32 lanes of a wave64 swizzle among themselves).
    fn swizzle_lane(&mut self, offset: u64) -> Value {
        let l = self.lane;
        if offset >= 0xC000 {
            // a permutation table: select chain over the 32 lanes
            let mut v = self.c(0);
            let l31 = self.andc(l, 0x1F);
            for i in 0..32u32 {
                let src = crate::gpu::recomp::emit::swizzle_lane(offset, i);
                let is = self.cmpc(Pred::Eq, l31, i);
                let k = self.c(src);
                v = self.sel(is, k, v);
            }
            let up = self.andc(l, 0x20);
            return self.or(up, v);
        }
        if offset & 0x8000 != 0 {
            let a = self.andc(l, !3);
            let q = self.andc(l, 3);
            let sh = self.shlc(q, 1);
            let k = self.c(offset as u32);
            let s = self.bin(BinOp::LShr, k, sh);
            let s = self.andc(s, 3);
            return self.or(a, s);
        }
        let (and, or, xor) = ((offset & 0x1F) as u32, ((offset >> 5) & 0x1F) as u32, ((offset >> 10) & 0x1F) as u32);
        let up = self.andc(l, 0x20);
        let x = self.andc(l, 0x1F);
        let x = self.andc(x, and);
        let x = self.orc(x, or);
        let k = self.c(xor);
        let x = self.xor(x, k);
        self.or(up, x)
    }

    // ------------------------------------------------------------ scratch

    /// The lane's private memory (`xcl_scratch`, 1024 dwords), allocated in the entry block.
    fn scratch_ptr(&mut self, i: Value) -> Value {
        let arr = match self.scratch {
            Some(a) => a,
            None => {
                let cur = self.cur();
                self.s.m.set_block(self.init.entry);
                let i32 = self.s.m.i32();
                let t = self.s.m.array(1024, i32);
                let a = self.s.m.alloca(t, 4);
                self.s.m.set_block(cur);
                self.scratch = Some(a);
                a
            }
        };
        let z = self.c(0);
        self.s.m.gep(arr, &[z, i])
    }
    fn scr_slot(&mut self, a: Value) -> Value {
        let b = self.lshrc(a, 2);
        self.andc(b, 1023)
    }
    fn scr_ld(&mut self, a: Value) -> Value {
        let b = self.scr_slot(a);
        let s = self.andc(a, 3);
        let s = self.shlc(s, 3);
        let p = self.scratch_ptr(b);
        let lo = self.s.m.load(p, 4);
        let b1 = self.addc(b, 1);
        let b1 = self.andc(b1, 1023);
        let p1 = self.scratch_ptr(b1);
        let hi = self.s.m.load(p1, 4);
        let x = self.bin(BinOp::LShr, lo, s);
        let k32 = self.c(32);
        let inv = self.sub(k32, s);
        let inv = self.andc(inv, 31);
        let y = self.bin(BinOp::Shl, hi, inv);
        let xy = self.or(x, y);
        let sz = self.cmpc(Pred::Eq, s, 0);
        self.sel(sz, lo, xy)
    }
    fn scr_st(&mut self, a: Value, v: Value, n: u32) -> LResult<()> {
        let b = self.scr_slot(a);
        let s = self.andc(a, 3);
        let s = self.shlc(s, 3);
        let m = if n >= 4 { self.c(u32::MAX) } else { self.c((1u32 << (8 * n)) - 1) };
        let p = self.scratch_ptr(b);
        let old = self.s.m.load(p, 4);
        let ms = self.bin(BinOp::Shl, m, s);
        let nm = self.not(ms);
        let keep = self.and(old, nm);
        let vm = self.and(v, m);
        let vs = self.bin(BinOp::Shl, vm, s);
        let nv = self.or(keep, vs);
        self.s.m.store(p, nv, 4);
        let snz = self.cmpc(Pred::Ne, s, 0);
        let top = self.addc(s, 8 * n);
        let over = self.cmpc(Pred::UGt, top, 32);
        let spill = self.land(snz, over);
        self.guarded(spill, &mut |lw| {
            let b2 = lw.addc(b, 1);
            let b2 = lw.andc(b2, 1023);
            let p2 = lw.scratch_ptr(b2);
            let old = lw.s.m.load(p2, 4);
            let k32 = lw.c(32);
            let inv = lw.sub(k32, s);
            let inv = lw.andc(inv, 31);
            let m2 = lw.bin(BinOp::LShr, m, inv);
            let nm2 = lw.not(m2);
            let keep = lw.and(old, nm2);
            let vm = lw.and(v, m);
            let v2 = lw.bin(BinOp::LShr, vm, inv);
            let nv = lw.or(keep, v2);
            lw.s.m.store(p2, nv, 4);
            Ok(())
        })
    }

    fn scratch(&mut self, d: &Decoded, op: &str) -> LResult<()> {
        let saddr = Self::fv(d, "SADDR");
        let base = if saddr != 0x7D && saddr != 0x7F { self.sreg(saddr) } else { self.vreg(Self::fv(d, "ADDR")) };
        let raw = Self::fv(d, "OFFSET") as u32;
        let off = (((raw << 20) as i32) >> 20) as u32;
        let a = self.addc(base, off);
        let act = self.act();
        if let Some((n, signed, store, half)) = flat_small(op) {
            if store {
                let v = self.vreg(Self::fv(d, "DATA"));
                let data = self.d16_data(v, half);
                return self.guarded(act, &mut |lw| lw.scr_st(a, data, n));
            }
            let vd = Self::fv(d, "VDST");
            let v = self.vreg(vd);
            return self.guarded(act, &mut |lw| {
                let raw = lw.scr_ld(a);
                let nv = lw.d16_merge(v, raw, n, signed, half);
                lw.vset(vd, nv);
                Ok(())
            });
        }
        let (store, count) = match op {
            "load_dword" => (false, 1),
            "load_dwordx2" => (false, 2),
            "load_dwordx3" => (false, 3),
            "load_dwordx4" => (false, 4),
            "store_dword" => (true, 1),
            "store_dwordx2" => (true, 2),
            "store_dwordx3" => (true, 3),
            "store_dwordx4" => (true, 4),
            _ => return err(d, "SCRATCH"),
        };
        let vals: Vec<Value> = if store { (0..count).map(|i| self.vreg(Self::fv(d, "DATA") + i)).collect() } else { Vec::new() };
        let vd = Self::fv(d, "VDST");
        self.guarded(act, &mut |lw| {
            for i in 0..count {
                let at = lw.addc(a, 4 * i as u32);
                let slot = lw.scr_slot(at);
                let p = lw.scratch_ptr(slot);
                if store {
                    lw.s.m.store(p, vals[i as usize], 4);
                } else {
                    let v = lw.s.m.load(p, 4);
                    lw.vset(vd + i, v);
                }
            }
            Ok(())
        })
    }

    // ------------------------------------------------------------ exports

    /// Mesh mode `s_sendmsg`: GS_ALLOC_REQ (M0 = primitives << 12 | vertices) into the
    /// record header by lane 0 of each wave.
    pub fn mesh_sendmsg(&mut self, d: &Decoded) -> LResult<()> {
        if Self::fv(d, "SIMM16") & 0xF != 9 {
            return Ok(());
        }
        let m0 = self.sreg(124);
        let rec = self.mesh.as_ref().expect("mesh").rec;
        let res = self.res.expect("res");
        let lane = self.lane;
        let first = self.cmpc(Pred::Eq, lane, 0);
        self.guarded(first, &mut |lw| {
            let v = lw.andc(m0, 0x3FF);
            let p = lw.lshrc(m0, 12);
            let p = lw.andc(p, 0x7FF);
            let a = lw.shlc(rec, 2);
            lw.s.raw_store(res, a, &[v]);
            let a4 = lw.addc(a, 4);
            lw.s.raw_store(res, a4, &[p]);
            Ok(())
        })
    }

    /// Mesh mode exports: position / params into this thread's vertex slot, `prim` into its
    /// primitive slot.
    fn mesh_export(&mut self, d: &Decoded, en: u64, tgt: u64, compr: bool, srcs: &[Value]) -> LResult<()> {
        let comp = |lw: &mut Lw, c: usize| -> Value {
            if compr {
                let r = srcs[c / 2];
                if c % 2 == 0 {
                    lw.andc(r, 0xFFFF)
                } else {
                    lw.lshrc(r, 16)
                }
            } else {
                srcs[c]
            }
        };
        let enabled = |c: usize| if compr { en >> (c / 2 * 2) & 3 != 0 } else { en >> c & 1 == 1 };
        let (vtx, prm) = {
            let m = self.mesh.as_ref().expect("mesh");
            (m.vtx, m.prm)
        };
        // (dword index in res, value)
        let mut st: Vec<(Value, Value)> = Vec::new();
        match tgt {
            12 => {
                for c in 0..4 {
                    if enabled(c) {
                        let at = self.addc(vtx, c as u32);
                        let v = comp(self, c);
                        st.push((at, v));
                    }
                }
            }
            13..=15 | 9 => {}
            32..=63 => {
                let k = (tgt - 32) as usize;
                let outputs = self.mesh.as_ref().expect("mesh").outputs.clone();
                let at = crate::gpu::recomp::param_registers(&outputs, true).get(k).map(|&r| crate::gpu::recomp::elements_at(&outputs, true, r)).unwrap_or_default();
                for j in at {
                    for c in 0..4 {
                        if enabled(c) {
                            let a = self.addc(vtx, (4 * (j + 1) + c) as u32);
                            let v = comp(self, c);
                            st.push((a, v));
                        }
                    }
                }
            }
            20 => st.push((prm, srcs[0])),
            _ => return err(d, "export target"),
        }
        if st.is_empty() {
            return Ok(());
        }
        let res = self.res.expect("res");
        let act = self.act();
        self.guarded(act, &mut |lw| {
            for &(at, v) in &st {
                let off = lw.shlc(at, 2);
                lw.s.raw_store(res, off, &[v]);
            }
            Ok(())
        })
    }

    pub fn export(&mut self, d: &Decoded) -> LResult<()> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let (en, tgt, compr) = (f("EN"), f("TGT"), f("COMPR") != 0);
        let srcs: Vec<Value> = (0..4).map(|i| self.vreg(f(&format!("VSRC{i}")))).collect();
        if self.mesh.is_some() {
            return self.mesh_export(d, en, tgt, compr, &srcs);
        }
        if self.gfx.is_some() {
            return self.export_gfx(d, en, tgt, compr, &srcs);
        }
        let act = self.act();
        let mut st = Vec::new();
        for c in 0..4u64 {
            let val = if compr {
                let r = srcs[(c / 2) as usize];
                if c % 2 == 0 {
                    self.andc(r, 0xFFFF)
                } else {
                    self.lshrc(r, 16)
                }
            } else {
                srcs[c as usize]
            };
            let bit = if compr { 1 << (c / 2 * 2) | 2 << (c / 2 * 2) } else { 1 << c };
            if en & bit == 0 {
                continue;
            }
            st.push((c, val));
        }
        if st.is_empty() {
            return Ok(());
        }
        let res = self.res.expect("res");
        let lane = self.lane;
        self.guarded(act, &mut |lw| {
            for &(c, val) in &st {
                // (KERNEL_EXPORTS + ((tgt * 64 + lane) * 4 + c)) * 4
                let k = lw.c((tgt as usize * KERNEL_LANES) as u32);
                let x = lw.add(k, lane);
                let x = lw.shlc(x, 2);
                let x = lw.addc(x, (KERNEL_EXPORTS as u64 + c) as u32);
                let off = lw.shlc(x, 2);
                lw.s.raw_store(res, off, &[val]);
            }
            Ok(())
        })
    }
}
