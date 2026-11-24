//! Memory lowering: scalar loads, buffer / typed buffer, flat / global, LDS,
//! atomics and kernel exports on the kernel ABI's flat `mem` window (port of
//! `recomp/emit.rs` `smem` / `mubuf` / `flat` / `ds` / `export` and
//! `PRELUDE_MEM`). Accesses of disabled lanes are real branches.

use super::lower::{err, Dst, LResult, Lw, VAR_VGPR};
use super::module::{BinOp, Pred, RmwOp, Value};
use super::ops::AtomicOp;
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

impl Lw {
    /// Runs `f` on the memory window the last `window` selected (`mem`, `mem1`.. at u2..):
    /// a switch over the windows when the program may address several, as the HLSL kernel's
    /// `xcl_m*` accessors; returns `f`'s value (i32 or i64).
    fn with_mem(&mut self, f: &mut dyn FnMut(&mut Lw, RawUav) -> Option<Value>) -> Option<Value> {
        if self.mems.len() <= 1 {
            let m = self.mems[0];
            return f(self, m);
        }
        let w = self.rd(super::lower::VAR_WIN);
        let (tl, th) = (self.tmp_var(), self.tmp_var());
        let z = self.c(0);
        self.wr(tl, z);
        self.wr(th, z);
        let mut kind = 0u8; // 0 none, 1 i32, 2 i64
        let mems = self.mems.clone();
        for (k, &m) in mems.iter().enumerate() {
            // window 0 is the default (`switch` default: any value but 1..)
            let is = if k == 0 {
                let mut any = self.b(false);
                for j in 1..mems.len() as u32 {
                    let e = self.cmpc(Pred::Eq, w, j);
                    any = self.lor(any, e);
                }
                self.lnot(any)
            } else {
                self.cmpc(Pred::Eq, w, k as u32)
            };
            let _ = self.guarded(is, &mut |lw| {
                if let Some(v) = f(lw, m) {
                    let t = lw.s.m.type_of(v);
                    if t == lw.s.m.i64() {
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
        }
        match kind {
            0 => None,
            1 => Some(self.rd(tl)),
            _ => {
                let (l, h) = (self.rd(tl), self.rd(th));
                Some(self.pack64(l, h))
            }
        }
    }
    /// `xcl_w = hi & 3` (several memory windows only).
    pub fn window(&mut self, hi: Value) {
        if self.mems.len() > 1 {
            let v = self.andc(hi, 3);
            self.wr(super::lower::VAR_WIN, v);
        }
    }
    fn mraw_atomic(&mut self, op: AtomicOp, a: Value, v: Value) -> Value {
        self.with_mem(&mut |lw, m| Some(lw.s.raw_atomic(m, op, a, v))).expect("atomic value")
    }
    fn mraw_cmpxchg(&mut self, a: Value, c: Value, n: Value) -> Value {
        self.with_mem(&mut |lw, m| Some(lw.s.raw_cmpxchg(m, a, c, n))).expect("cmpxchg value")
    }
    /// `mem.Load(a & ~3)`
    pub fn mem_ld(&mut self, a: Value) -> Value {
        let a = self.andc(a, !3);
        self.with_mem(&mut |lw, m| Some(lw.s.raw_load(m, a, 1)[0])).expect("load value")
    }
    /// `mem.Store(a & ~3, v)`
    pub fn mem_st(&mut self, a: Value, v: Value) {
        let a = self.andc(a, !3);
        self.with_mem(&mut |lw, m| {
            lw.s.raw_store(m, a, &[v]);
            None
        });
    }

    /// `ld_raw`: four dwords from an unaligned byte address.
    fn ld_raw(&mut self, a: Value) -> [Value; 4] {
        let b = self.andc(a, !3);
        let s = self.andc(a, 3);
        let s = self.shlc(s, 3);
        let mut w = Vec::new();
        for k in 0..5 {
            let at = self.addc(b, 4 * k);
            w.push(self.mem_ld(at));
        }
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
            let big = lw.cmpc(Pred::UGe, n, 4);
            let n8 = lw.shlc(n, 3);
            let n8 = lw.andc(n8, 31);
            let one = lw.c(1);
            let m = lw.bin(BinOp::Shl, one, n8);
            let m = lw.sub(m, one);
            let all = lw.c(u32::MAX);
            let m = lw.sel(big, all, m);
            let ms = lw.bin(BinOp::Shl, m, s);
            let vs = lw.bin(BinOp::Shl, r[0], s);
            lw.st_part(b, ms, vs);
            let snz = lw.cmpc(Pred::Ne, s, 0);
            let top = lw.add(s, n8);
            // s + 8n with n = 4 gives s + 32 (n8 above is masked to 31)
            let is4 = lw.cmpc(Pred::UGe, n, 4);
            let k = lw.c(32);
            let z = lw.c(0);
            let t = lw.sel(is4, k, z);
            let top = lw.add(top, t);
            let over = lw.cmpc(Pred::UGt, top, 32);
            let spill = lw.land(snz, over);
            lw.guarded(spill, &mut |lw| {
                let b4 = lw.addc(b, 4);
                let k32 = lw.c(32);
                let inv = lw.sub(k32, s);
                let inv = lw.andc(inv, 31);
                let m2 = lw.bin(BinOp::LShr, m, inv);
                let v2 = lw.bin(BinOp::LShr, r[0], inv);
                lw.st_part(b4, m2, v2);
                Ok(())
            })
        })
    }

    // ------------------------------------------------------------ formats

    /// `FMT_INFO[min(fmt, 77)]` (a constant table) for a dynamic format.
    pub(crate) fn fmt_info(&mut self, fmt: Value) -> Value {
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
    fn exp2f(&mut self, x: Value) -> Value {
        self.un(super::ops_core::op::EXP, x)
    }
    fn uf_small(&mut self, v: Value, mb: Value) -> Value {
        let e = self.bin(BinOp::LShr, v, mb);
        let one = self.c(1);
        let p = self.bin(BinOp::Shl, one, mb);
        let pm = self.sub(p, one);
        let m = self.and(v, pm);
        let fm = self.uitof(m);
        let fp = self.uitof(p);
        let fm = self.fdiv(fm, fp);
        let mnz = self.nz(m);
        let qn = self.s.m.f32c(0x7FC0_0000);
        let inf = self.s.m.f32c(0x7F80_0000);
        let special = self.sel(mnz, qn, inf);
        let m14 = self.fc(-14.0);
        let p14 = self.exp2f(m14);
        let den = self.fmul(fm, p14);
        let ef = self.uitof(e);
        let k15 = self.fc(15.0);
        let eb = self.fsub(ef, k15);
        let pe = self.exp2f(eb);
        let one_f = self.fc(1.0);
        let mant = self.fadd(one_f, fm);
        let norm = self.fmul(pe, mant);
        let e0 = self.cmpc(Pred::Eq, e, 0);
        let r = self.sel(e0, den, norm);
        let e31 = self.cmpc(Pred::Eq, e, 31);
        self.sel(e31, special, r)
    }
    fn fmt_comp(&mut self, r: &[Value; 4], info: Value, c: u32) -> Value {
        let n = self.andc(info, 7);
        let ty = self.lshrc(info, 3);
        let ty = self.andc(ty, 7);
        let t4 = self.cmpc(Pred::Eq, ty, 4);
        let t5 = self.cmpc(Pred::Eq, ty, 5);
        let isint = self.lor(t4, t5);
        let mut off = self.c(0);
        for k in 0..c {
            let w = self.lshrc(info, 8 + 6 * k);
            let w = self.andc(w, 63);
            off = self.add(off, w);
        }
        let w = self.lshrc(info, (8 + 6 * c) & 31);
        let w = self.andc(w, 63);
        let v = self.bits_at(r, off, w);
        let k32 = self.c(32);
        let sh = self.sub(k32, w);
        let sh = self.andc(sh, 31);
        let x = self.bin(BinOp::Shl, v, sh);
        let x = self.bin(BinOp::AShr, x, sh);
        let big = self.cmpc(Pred::UGe, w, 32);
        let sv = self.sel(big, v, x);
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
        let w11 = self.cmpc(Pred::Eq, w, 11);
        let k6 = self.c(6);
        let k5 = self.c(5);
        let mb = self.sel(w11, k6, k5);
        let uf = self.uf_small(v, mb);
        let ufb = self.bits(uf);
        let w16 = self.cmpc(Pred::Eq, w, 16);
        let w32 = self.cmpc(Pred::Eq, w, 32);
        let flt = self.sel(w16, hb, ufb);
        let flt = self.sel(w32, v, flt);
        let fvb = self.bits(fv);
        let fsvb = self.bits(fsv);
        let mut res = flt;
        for (k, val) in [(5u32, sv), (4, v), (3, fsvb), (2, fvb), (1, f_sn), (0, f_un)] {
            let is = self.cmpc(Pred::Eq, ty, k);
            res = self.sel(is, val, res);
        }
        let cge = self.cmpc(Pred::ULe, n, c);
        let dflt = if c == 3 {
            let a = self.c(1);
            let b = self.c(0x3F80_0000);
            self.sel(isint, a, b)
        } else {
            self.c(0)
        };
        self.sel(cge, dflt, res)
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
        // floats
        let h = self.s.f32_to_f16(f);
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
        let t4 = self.cmpc(Pred::Eq, ty, 4);
        let t5 = self.cmpc(Pred::Eq, ty, 5);
        let w32 = self.cmpc(Pred::Eq, w, 32);
        let raw = self.lor(t4, t5);
        let raw = self.lor(raw, w32);
        let r = self.sel(raw, x, fl);
        let mut res = r;
        for (k, val) in [(3u32, e3), (2, e2), (1, e1), (0, e0)] {
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
        let mut vals = Vec::new();
        if name.starts_with("s_buffer_load") {
            let (d0, d1, d2) = (self.sreg(sbase), self.sreg(sbase + 1), self.sreg(sbase + 2));
            self.window(d1);
            let k = self.c(imm);
            let off = self.add(k, soff);
            let st = self.lshrc(d1, 16);
            let st = self.andc(st, 0x3FFF);
            let sz = self.cmpc(Pred::Eq, st, 0);
            let lim = self.mul(d2, st);
            let lim = self.sel(sz, d2, lim);
            for i in 0..count {
                let o = self.addc(off, 4 * i);
                let ok = self.cmp(Pred::ULt, o, lim);
                let a = self.add(d0, o);
                let v = self.mem_ld(a);
                let z = self.c(0);
                vals.push(self.sel(ok, v, z));
            }
        } else {
            let imm = ((imm << 11) as i32 >> 11) as u32;
            let b = self.sreg(sbase);
            let hi = self.sreg(sbase + 1);
            self.window(hi);
            let k = self.c(imm);
            let a = self.add(b, k);
            let a = self.add(a, soff);
            for i in 0..count {
                let at = self.addc(a, 4 * i);
                vals.push(self.mem_ld(at));
            }
        }
        let coherent = ["GLC", "DLC"].iter().any(|f| rdna2::field_value(d.enc, &d.words, f) == Some(1));
        if self.split && coherent {
            // both halves take the lower half's values
            let other = self.exchange_n(&vals);
            let up = self.upper;
            for (v, o) in vals.iter_mut().zip(other) {
                *v = self.sel(up, o, *v);
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
        self.window(d1);
        let index = if idxen { self.vreg(vaddr) } else { self.c(0) };
        let voff = if offen { self.vreg(vaddr + idxen as u64) } else { self.c(0) };
        let stride = self.lshrc(d1, 16);
        let stride = self.andc(stride, 0x3FFF);
        let off = self.addc(voff, f("OFFSET") as u32);
        let a = self.add(d0, soff);
        let a = self.add(a, off);
        let si = self.mul(stride, index);
        let a = self.add(a, si);
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
            return self.guarded(act, &mut |lw| {
                let r = lw.ld_raw(a);
                let z = lw.c(0);
                let raw = lw.sel(ok, r[0], z);
                let nv = lw.d16_merge(v, raw, n, signed, half);
                lw.vset(vdata, nv);
                Ok(())
            });
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
            return self.guarded(act, &mut |lw| {
                let r = lw.ld_raw(a);
                let bits = 8 * n;
                let val = if signed {
                    let x = lw.shlc(r[0], 32 - bits);
                    lw.ashrc(x, 32 - bits)
                } else {
                    lw.andc(r[0], ((1u64 << bits) - 1) as u32)
                };
                let z = lw.c(0);
                let val = lw.sel(ok, val, z);
                lw.vset(vdata, val);
                Ok(())
            });
        }
        match kind {
            BufKind::Load => {
                let oks: Vec<Value> = (0..count).map(|i| {
                    let b = self.c(4 * i as u32 + 4);
                    in_range(self, b)
                }).collect();
                self.guarded(act, &mut |lw| {
                    for i in 0..count {
                        let at = lw.addc(a, 4 * i as u32);
                        let v = lw.mem_ld(at);
                        let z = lw.c(0);
                        let v = lw.sel(oks[i as usize], v, z);
                        lw.vset(vdata + i, v);
                    }
                    Ok(())
                })
            }
            BufKind::Store => {
                let vals: Vec<Value> = (0..count).map(|i| self.vreg(vdata + i)).collect();
                let oks: Vec<Value> = (0..count).map(|i| {
                    let b = self.c(4 * i as u32 + 4);
                    in_range(self, b)
                }).collect();
                self.guarded(act, &mut |lw| {
                    for i in 0..count as usize {
                        let at = lw.addc(a, 4 * i as u32);
                        let v = vals[i];
                        lw.guarded(oks[i], &mut |lw| {
                            lw.mem_st(at, v);
                            Ok(())
                        })?;
                    }
                    Ok(())
                })
            }
            BufKind::FormatLoad => {
                let fmt = if d.enc.name == "ENC_MTBUF" {
                    self.c(f("FORMAT") as u32)
                } else {
                    let x = self.lshrc(d3, 12);
                    self.andc(x, 0x7F)
                };
                let info = self.fmt_info(fmt);
                let ty = self.lshrc(info, 3);
                let ty = self.andc(ty, 7);
                let t4 = self.cmpc(Pred::Eq, ty, 4);
                let t5 = self.cmpc(Pred::Eq, ty, 5);
                let isint = self.lor(t4, t5);
                let nb = self.fmt_bytes(info);
                let rng = in_range(self, nb);
                let n7 = self.andc(info, 7);
                let has = self.nz(n7);
                let ok = self.land(has, rng);
                let cur: Vec<Value> = (0..count).map(|i| self.vreg(vdata + i)).collect();
                self.guarded(act, &mut |lw| {
                    let r = lw.ld_raw(a);
                    let c = [lw.fmt_comp(&r, info, 0), lw.fmt_comp(&r, info, 1), lw.fmt_comp(&r, info, 2), lw.fmt_comp(&r, info, 3)];
                    let z = lw.c(0);
                    if let Some((_, components, high)) = d16_format {
                        let half = |lw: &mut Lw, comp: u64| -> Value {
                            let sel = lw.lshrc(d3, 3 * comp as u32);
                            let sel = lw.andc(sel, 7);
                            let s = lw.dst_sel(&c, sel, isint);
                            let lo = lw.andc(s, 0xFFFF);
                            let sf = lw.f(s);
                            let h = lw.s.f32_to_f16(sf);
                            lw.sel(isint, lo, h)
                        };
                        if high {
                            let h = half(lw, 0);
                            let h = lw.shlc(h, 16);
                            let h = lw.sel(ok, h, z);
                            let l = lw.andc(cur[0], 0xFFFF);
                            let v = lw.or(l, h);
                            lw.vset(vdata, v);
                        } else {
                            for i in 0..count {
                                let lo = half(lw, 2 * i);
                                let hi = if 2 * i + 1 < components { half(lw, 2 * i + 1) } else { z };
                                let hi = lw.shlc(hi, 16);
                                let v = lw.or(lo, hi);
                                let v = lw.sel(ok, v, z);
                                lw.vset(vdata + i, v);
                            }
                        }
                        return Ok(());
                    }
                    for i in 0..count {
                        let sel = lw.lshrc(d3, 3 * i as u32);
                        let sel = lw.andc(sel, 7);
                        let s = lw.dst_sel(&c, sel, isint);
                        let v = lw.sel(ok, s, z);
                        lw.vset(vdata + i, v);
                    }
                    Ok(())
                })
            }
            BufKind::FormatStore => {
                let fmt = if d.enc.name == "ENC_MTBUF" {
                    self.c(f("FORMAT") as u32)
                } else {
                    let x = self.lshrc(d3, 12);
                    self.andc(x, 0x7F)
                };
                let info = self.fmt_info(fmt);
                let nb = self.fmt_bytes(info);
                let rng = in_range(self, nb);
                let n7 = self.andc(info, 7);
                let has = self.nz(n7);
                let ok = self.land(has, rng);
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
            "cmpswap" | "fcmpswap" | "cmpst_b32" | "cmpst_f32" => self.cmpxchg(at, v2, v),
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
                "cmpswap" | "fcmpswap" => lw.mraw_cmpxchg(a, cv, v),
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
        self.window(d1);
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

    fn flat_base(&mut self, d: &Decoded) -> (Value, bool) {
        let global = d.enc.name == "ENC_FLAT_GLBL";
        let addr = self.vreg(Self::fv(d, "ADDR"));
        let saddr = Self::fv(d, "SADDR");
        self.flat_window(d);
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

    /// `flat_window`: the window of a FLAT / global address (SADDR pair or the 64-bit VGPR address).
    fn flat_window(&mut self, d: &Decoded) {
        if self.mems.len() > 1 {
            let global = d.enc.name == "ENC_FLAT_GLBL";
            let saddr = Self::fv(d, "SADDR");
            let hi = if global && saddr != 0x7D && saddr != 0x7F { self.sreg(saddr + 1) } else { self.vreg(Self::fv(d, "ADDR") + 1) };
            self.window(hi);
        }
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
            let (a, _) = self.flat_base(d);
            if store {
                let v = self.vreg(Self::fv(d, "DATA"));
                let data = self.d16_data(v, half);
                let z = self.c(0);
                let nv = self.c(n);
                return self.guarded(act, &mut |lw| lw.st_bytes(a, [data, z, z, z], nv));
            }
            let vd = Self::fv(d, "VDST");
            let v = self.vreg(vd);
            return self.guarded(act, &mut |lw| {
                let r = lw.ld_raw(a);
                let nv = lw.d16_merge(v, r[0], n, signed, half);
                lw.vset(vd, nv);
                Ok(())
            });
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
            self.flat_window(d);
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
            self.flat_base(d).0
        };
        if store {
            let vals: Vec<Value> = (0..count).map(|i| self.vreg(Self::fv(d, "DATA") + i)).collect();
            self.guarded(act, &mut |lw| {
                for (i, &v) in vals.iter().enumerate() {
                    let at = lw.addc(a, 4 * i as u32);
                    lw.mem_st(at, v);
                }
                Ok(())
            })
        } else {
            let vd = Self::fv(d, "VDST");
            self.guarded(act, &mut |lw| {
                for i in 0..count {
                    let at = lw.addc(a, 4 * i as u32);
                    let v = lw.mem_ld(at);
                    lw.vset(vd + i, v);
                }
                Ok(())
            })
        }
    }

    fn flat_atomic(&mut self, d: &Decoded, op: &str) -> LResult<()> {
        let (a, _) = self.flat_base(d);
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
