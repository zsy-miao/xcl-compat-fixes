//! Randomised differential testing of the emitter: a seeded generator of
//! random-but-valid compute kernels (straight-line and branchy integer and
//! float arithmetic with phis, loops, switches and selects; group-shared
//! loads, stores, barriers and atomics; wave reads; raw buffer I/O). Every
//! module must pass DXC's validator; a subset also runs on the GPU and is
//! compared with a CPU interpreter of the same program.
//!
//! Exactness: float values are small integers (|x| < 64, at most two
//! arithmetic steps from a conversion), so every add / sub / mul result is
//! exactly representable — contraction and denormal modes cannot change any
//! result. Fast-math flags are generated too, but modules using `nsz` or
//! reassociation (which may flip the sign of a zero) are only validated.
//!
//! Kernel ABI as in `ops_tests`: 32 threads (one wave32), inputs in `u2`
//! (`mem`, slot `k` of lane `l` at dword `64·k + l`), results in `u1`.

use super::container::Stage;
use super::module::{fmf, BinOp, CastOp, Pred, RmwOp, Value};
use super::ops::{barrier, BitCount, Binary, GroupShared};
use super::shader::{RawUav, Shader};
use super::validate;
use crate::gpu::recomp::runner;

const LANES: usize = 32;
const IN_SLOTS: u32 = 6;

// --------------------------------------------------------------- PRNG

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.next() % den < num
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
    /// Interesting 32/64-bit patterns as well as uniform ones.
    fn bits(&mut self, w: u32) -> u64 {
        let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
        let v = match self.below(6) {
            0 => self.below(8) as u64,
            1 => (self.below(8) as u64).wrapping_neg(),
            2 => 1u64 << (w - 1),
            3 => (1u64 << (w - 1)) - 1,
            _ => self.next(),
        };
        v & mask
    }
}

// ------------------------------------------------------------------ IR

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum T {
    I1,
    I32,
    I64,
    F32,
    F64,
}

impl T {
    fn width(self) -> u32 {
        match self {
            T::I1 => 1,
            T::I32 | T::F32 => 32,
            T::I64 | T::F64 => 64,
        }
    }
    fn mask(self) -> u64 {
        if self.width() == 64 {
            u64::MAX
        } else {
            (1u64 << self.width()) - 1
        }
    }
    fn is_float(self) -> bool {
        matches!(self, T::F32 | T::F64)
    }
}

#[derive(Debug, Clone, Copy)]
enum Opnd {
    V(usize),
    C(T, u64),
}

#[derive(Debug, Clone, Copy)]
enum DxUn {
    Count,
    Lo,
    Hi,
    Rev,
}

#[derive(Debug, Clone)]
enum Op {
    Bin(BinOp, Opnd, Opnd),
    FBin(BinOp, Opnd, Opnd, u64),
    Cmp(Pred, Opnd, Opnd),
    Select(Opnd, Opnd, Opnd),
    Cast(CastOp, Opnd, T),
    DxBin(Binary, Opnd, Opnd),
    DxUn(DxUn, Opnd),
}

#[derive(Debug, Clone)]
struct Inst {
    dst: usize,
    op: Op,
}

#[derive(Debug, Clone)]
enum Seg {
    Insts(Vec<Inst>),
    /// `if (cond) then else` (no else block when `triangle`), phis at the join.
    Diamond { cond: Opnd, then_: Vec<Inst>, else_: Vec<Inst>, triangle: bool, phis: Vec<(usize, Opnd, Opnd)> },
    /// `switch (sel)` with one phi at the join.
    Switch { sel: Opnd, cases: Vec<(u64, Vec<Inst>, Opnd)>, default: (Vec<Inst>, Opnd), phi: usize },
    /// `for (ctr = 0; ctr < trip; ctr++) { body; accs = next }`.
    Loop { trip: Opnd, ctr: usize, ctr_next: usize, cond: usize, accs: Vec<(usize, Opnd, Opnd)>, body: Vec<Inst> },
    /// `gs[lane] = v; barrier; dst = gs[(lane + k) & 31]`.
    GsExchange { v: Opnd, k: u32, dst: usize },
    /// `cnt[i] = init(i); barrier; atomic op(cnt[lane & 7], v); cmpxchg on a
    /// scratch slot; barrier; dst = cnt[(lane >> 2) & 7]`.
    GsAtomic { op: RmwOp, v: Opnd, k: u32, dst: usize },
    /// `WaveReadLaneAt(v, lane)` (`None`: `WaveReadLaneFirst`).
    Wave { v: Opnd, lane: Option<u32>, dst: usize },
    /// `rawBufferLoad` of `n` dwords from input slot `slot`.
    Load { slot: u32, dsts: Vec<usize> },
}

struct Prog {
    tys: Vec<T>,
    segs: Vec<Seg>,
    lane: usize,
    outs: Vec<usize>,
    gpu_safe: bool,
}

// ------------------------------------------------------------ generator

struct Gen {
    r: Rng,
    tys: Vec<T>,
    /// Float values: arithmetic steps since the integer conversion.
    depth: Vec<u8>,
    f64s: bool,
    i64s: bool,
    gpu_safe: bool,
}

const INT_OPS: [BinOp; 13] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::UDiv,
    BinOp::SDiv,
    BinOp::URem,
    BinOp::SRem,
    BinOp::Shl,
    BinOp::LShr,
    BinOp::AShr,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
];
const INT_PREDS: [Pred; 10] = [Pred::Eq, Pred::Ne, Pred::UGt, Pred::UGe, Pred::ULt, Pred::ULe, Pred::SGt, Pred::SGe, Pred::SLt, Pred::SLe];
const F_PREDS: [Pred; 14] = [
    Pred::FOEq,
    Pred::FOGt,
    Pred::FOGe,
    Pred::FOLt,
    Pred::FOLe,
    Pred::FONe,
    Pred::FOrd,
    Pred::FUno,
    Pred::FUEq,
    Pred::FUGt,
    Pred::FUGe,
    Pred::FULt,
    Pred::FULe,
    Pred::FUNe,
];

impl Gen {
    fn val(&mut self, t: T, depth: u8) -> usize {
        self.tys.push(t);
        self.depth.push(depth);
        self.tys.len() - 1
    }
    fn of(&self, pool: &[usize], t: T) -> Vec<usize> {
        pool.iter().copied().filter(|&v| self.tys[v] == t).collect()
    }
    fn fconst(&mut self, t: T) -> Opnd {
        let x = self.r.below(127) as i32 - 63;
        Opnd::C(t, if t == T::F32 { (x as f32).to_bits() as u64 } else { (x as f64).to_bits() })
    }
    /// An operand of type `t`: a pool value or a constant. Floats only with
    /// depth ≤ `max_depth`.
    fn opnd(&mut self, pool: &[usize], t: T, max_depth: u8) -> Opnd {
        let cands: Vec<usize> = self.of(pool, t).into_iter().filter(|&v| !t.is_float() || self.depth[v] <= max_depth).collect();
        if cands.is_empty() || self.r.chance(1, 5) {
            return match t {
                T::F32 | T::F64 => self.fconst(t),
                _ => Opnd::C(t, self.r.bits(t.width())),
            };
        }
        Opnd::V(self.r.pick(&cands))
    }
    fn odepth(&self, o: Opnd) -> u8 {
        match o {
            Opnd::V(v) => self.depth[v],
            Opnd::C(..) => 0,
        }
    }
    fn push(&mut self, out: &mut Vec<Inst>, pool: &mut Vec<usize>, t: T, depth: u8, op: Op) -> Opnd {
        let dst = self.val(t, depth);
        out.push(Inst { dst, op });
        pool.push(dst);
        Opnd::V(dst)
    }
    fn int_ty(&mut self) -> T {
        if self.i64s && self.r.chance(1, 3) {
            T::I64
        } else {
            T::I32
        }
    }
    fn any_ty(&mut self) -> T {
        let mut ts = vec![T::I1, T::I32, T::I32, T::F32];
        if self.i64s {
            ts.push(T::I64);
        }
        if self.f64s {
            ts.push(T::F64);
        }
        self.r.pick(&ts)
    }
    fn float_ty(&mut self) -> T {
        if self.f64s && self.r.chance(1, 3) {
            T::F64
        } else {
            T::F32
        }
    }

    /// One random instruction (plus guards keeping it well defined).
    fn inst(&mut self, out: &mut Vec<Inst>, pool: &mut Vec<usize>) {
        match self.r.below(12) {
            0..=3 => {
                let t = self.int_ty();
                let op = self.r.pick(&INT_OPS);
                let a = self.opnd(pool, t, 0);
                let mut b = self.opnd(pool, t, 0);
                let w = t.width() as u64;
                match op {
                    BinOp::Shl | BinOp::LShr | BinOp::AShr => b = self.push(out, pool, t, 0, Op::Bin(BinOp::And, b, Opnd::C(t, w - 1))),
                    BinOp::UDiv | BinOp::URem => b = self.push(out, pool, t, 0, Op::Bin(BinOp::Or, b, Opnd::C(t, 1))),
                    BinOp::SDiv | BinOp::SRem => {
                        let pos = self.push(out, pool, t, 0, Op::Bin(BinOp::And, b, Opnd::C(t, t.mask() >> 1)));
                        b = self.push(out, pool, t, 0, Op::Bin(BinOp::Or, pos, Opnd::C(t, 1)));
                    }
                    _ => {}
                }
                self.push(out, pool, t, 0, Op::Bin(op, a, b));
            }
            4 => {
                let op = self.r.pick(&[BinOp::And, BinOp::Or, BinOp::Xor]);
                let (a, b) = (self.opnd(pool, T::I1, 0), self.opnd(pool, T::I1, 0));
                self.push(out, pool, T::I1, 0, Op::Bin(op, a, b));
            }
            5 => {
                let t = self.int_ty();
                let p = self.r.pick(&INT_PREDS);
                let (a, b) = (self.opnd(pool, t, 0), self.opnd(pool, t, 0));
                self.push(out, pool, T::I1, 0, Op::Cmp(p, a, b));
            }
            6 => {
                let t = self.any_ty();
                let c = self.opnd(pool, T::I1, 0);
                let (a, b) = (self.opnd(pool, t, 2), self.opnd(pool, t, 2));
                let d = self.odepth(a).max(self.odepth(b));
                self.push(out, pool, t, d, Op::Select(c, a, b));
            }
            7 => {
                let choices: &[(CastOp, T, T)] = &[
                    (CastOp::ZExt, T::I1, T::I32),
                    (CastOp::SExt, T::I1, T::I32),
                    (CastOp::Trunc, T::I32, T::I1),
                    (CastOp::ZExt, T::I32, T::I64),
                    (CastOp::SExt, T::I32, T::I64),
                    (CastOp::Trunc, T::I64, T::I32),
                    (CastOp::SExt, T::I1, T::I64),
                ];
                let (op, from, to) = self.r.pick(choices);
                if !self.i64s && (from == T::I64 || to == T::I64) {
                    return;
                }
                let v = self.opnd(pool, from, 0);
                self.push(out, pool, to, 0, Op::Cast(op, v, to));
            }
            8 => {
                // int → float of a 6-bit value (signed: sign-extended from 6 bits).
                let t = self.float_ty();
                let x = self.opnd(pool, T::I32, 0);
                let (op, v) = if self.r.chance(1, 2) {
                    (CastOp::UIToFP, self.push(out, pool, T::I32, 0, Op::Bin(BinOp::And, x, Opnd::C(T::I32, 63))))
                } else {
                    let sh = self.push(out, pool, T::I32, 0, Op::Bin(BinOp::Shl, x, Opnd::C(T::I32, 26)));
                    (CastOp::SIToFP, self.push(out, pool, T::I32, 0, Op::Bin(BinOp::AShr, sh, Opnd::C(T::I32, 26))))
                };
                self.push(out, pool, t, 0, Op::Cast(op, v, t));
            }
            9 => {
                let t = self.float_ty();
                match self.r.below(4) {
                    0 | 1 => {
                        let op = self.r.pick(&[BinOp::Add, BinOp::Sub, BinOp::Mul]);
                        let (a, b) = (self.opnd(pool, t, 1), self.opnd(pool, t, 1));
                        let flags = self.r.pick(&[fmf::NONE, fmf::NONE, fmf::FAST, fmf::NO_NANS | fmf::NO_INFS, fmf::NO_SIGNED_ZEROS]);
                        let d = self.odepth(a).max(self.odepth(b)) + 1;
                        // `nsz` / reassociation may flip the sign of a zero result.
                        if flags & (fmf::NO_SIGNED_ZEROS | fmf::UNSAFE_ALGEBRA) != 0 {
                            self.gpu_safe = false;
                        }
                        self.push(out, pool, t, d, Op::FBin(op, a, b, flags));
                    }
                    2 => {
                        let p = self.r.pick(&F_PREDS);
                        let (a, b) = (self.opnd(pool, t, 2), self.opnd(pool, t, 2));
                        self.push(out, pool, T::I1, 0, Op::Cmp(p, a, b));
                    }
                    _ => {
                        let (it, op) = if t == T::F32 { (T::I32, CastOp::BitCast) } else { (T::I64, CastOp::BitCast) };
                        if it == T::I64 && !self.i64s {
                            return;
                        }
                        let v = self.opnd(pool, t, 2);
                        self.push(out, pool, it, 0, Op::Cast(op, v, it));
                    }
                }
            }
            10 => {
                if !self.f64s {
                    return;
                }
                if self.r.chance(1, 2) {
                    let v = self.opnd(pool, T::F32, 2);
                    let d = self.odepth(v);
                    self.push(out, pool, T::F64, d, Op::Cast(CastOp::FPExt, v, T::F64));
                } else {
                    let v = self.opnd(pool, T::F64, 2);
                    let d = self.odepth(v);
                    self.push(out, pool, T::F32, d, Op::Cast(CastOp::FPTrunc, v, T::F32));
                }
            }
            _ => {
                if self.r.chance(1, 2) {
                    let op = self.r.pick(&[Binary::IMax, Binary::IMin, Binary::UMax, Binary::UMin]);
                    let t = self.int_ty();
                    let (a, b) = (self.opnd(pool, t, 0), self.opnd(pool, t, 0));
                    self.push(out, pool, t, 0, Op::DxBin(op, a, b));
                } else {
                    let u = self.r.pick(&[DxUn::Count, DxUn::Lo, DxUn::Hi, DxUn::Rev]);
                    let t = self.int_ty();
                    let a = self.opnd(pool, t, 0);
                    let rt = if matches!(u, DxUn::Rev) { t } else { T::I32 };
                    self.push(out, pool, rt, 0, Op::DxUn(u, a));
                }
            }
        }
    }

    fn insts(&mut self, pool: &mut Vec<usize>, n: usize) -> Vec<Inst> {
        let mut out = Vec::new();
        for _ in 0..n {
            self.inst(&mut out, pool);
        }
        out
    }

    /// A value of type `t` from `pool` for a phi / output.
    fn result(&mut self, pool: &[usize], t: T) -> Opnd {
        self.opnd(pool, t, 2)
    }
}

fn generate(seed: u64) -> Prog {
    let mut g = Gen { r: Rng::new(seed), tys: Vec::new(), depth: Vec::new(), f64s: false, i64s: false, gpu_safe: true };
    g.f64s = g.r.chance(1, 4);
    g.i64s = g.r.chance(1, 2);
    let lane = g.val(T::I32, 0);
    let mut pool = vec![lane];
    let mut segs = Vec::new();
    // Inputs first.
    for slot in 0..g.r.below(3) as u32 + 1 {
        let n = g.r.pick(&[1usize, 2, 3, 4]);
        let dsts: Vec<usize> = (0..n).map(|_| g.val(T::I32, 0)).collect();
        pool.extend(&dsts);
        segs.push(Seg::Load { slot: slot % IN_SLOTS, dsts });
    }
    let n_segs = if g.r.chance(1, 8) { g.r.below(40) + 10 } else { g.r.below(10) + 2 };
    for _ in 0..n_segs {
        match g.r.below(9) {
            0..=2 => {
                let n = g.r.below(12) + 1;
                segs.push(Seg::Insts(g.insts(&mut pool, n)));
            }
            3 => {
                let cond = g.opnd(&pool, T::I1, 0);
                let triangle = g.r.chance(1, 3);
                let (mut tp, mut ep) = (pool.clone(), pool.clone());
                let n = g.r.below(6);
                let then_ = g.insts(&mut tp, n);
                let n = g.r.below(6);
                let else_ = if triangle { Vec::new() } else { g.insts(&mut ep, n) };
                let mut phis = Vec::new();
                for _ in 0..g.r.below(3) + 1 {
                    let t = g.any_ty();
                    let (a, b) = (g.result(&tp, t), g.result(&ep, t));
                    let d = g.odepth(a).max(g.odepth(b));
                    let dst = g.val(t, d);
                    phis.push((dst, a, b));
                }
                pool.extend(phis.iter().map(|p| p.0));
                segs.push(Seg::Diamond { cond, then_, else_, triangle, phis });
            }
            4 => {
                let raw = g.opnd(&pool, T::I32, 0);
                let mut pre = Vec::new();
                let sel = if g.r.chance(1, 2) { g.push(&mut pre, &mut pool, T::I32, 0, Op::Bin(BinOp::And, raw, Opnd::C(T::I32, 7))) } else { raw };
                segs.push(Seg::Insts(pre));
                let t = g.any_ty();
                let mut vals: Vec<u64> = Vec::new();
                let mut cases = Vec::new();
                for _ in 0..g.r.below(5) + 1 {
                    let v = if g.r.chance(2, 3) { g.r.below(8) as u64 } else { g.r.bits(32) };
                    if vals.contains(&v) {
                        continue;
                    }
                    vals.push(v);
                    let mut cp = pool.clone();
                    let n = g.r.below(4);
                    let body = g.insts(&mut cp, n);
                    let out = g.result(&cp, t);
                    cases.push((v, body, out));
                }
                let mut dp = pool.clone();
                let n = g.r.below(4);
                let dbody = g.insts(&mut dp, n);
                let dout = g.result(&dp, t);
                let d = cases.iter().map(|c| g.odepth(c.2)).max().unwrap_or(0).max(g.odepth(dout));
                let phi = g.val(t, d);
                pool.push(phi);
                segs.push(Seg::Switch { sel, cases, default: (dbody, dout), phi });
            }
            5 => {
                let x = g.opnd(&pool, T::I32, 0);
                let mut pre = Vec::new();
                let trip = g.push(&mut pre, &mut pool, T::I32, 0, Op::Bin(BinOp::And, x, Opnd::C(T::I32, 7)));
                segs.push(Seg::Insts(pre));
                let ctr = g.val(T::I32, 0);
                let cond = g.val(T::I1, 0);
                // Accumulators: integers only (float values would grow).
                let mut accs: Vec<(usize, Opnd)> = Vec::new();
                for _ in 0..g.r.below(3) + 1 {
                    let t = if g.r.chance(1, 4) { T::I1 } else { g.int_ty() };
                    let init = g.opnd(&pool, t, 0);
                    accs.push((g.val(t, 0), init));
                }
                let mut bp = pool.clone();
                bp.push(ctr);
                bp.extend(accs.iter().map(|a| a.0));
                let n = g.r.below(8) + 1;
                let body = g.insts(&mut bp, n);
                let ctr_next = g.val(T::I32, 0);
                let accs = accs
                    .into_iter()
                    .map(|(dst, init)| {
                        let t = g.tys[dst];
                        let next = g.opnd(&bp, t, 0);
                        (dst, init, next)
                    })
                    .collect::<Vec<_>>();
                pool.extend(accs.iter().map(|a| a.0));
                segs.push(Seg::Loop { trip, ctr, ctr_next, cond, accs, body });
            }
            6 => {
                let v = g.opnd(&pool, T::I32, 0);
                let dst = g.val(T::I32, 0);
                pool.push(dst);
                segs.push(Seg::GsExchange { v, k: g.r.below(32) as u32, dst });
            }
            7 => {
                let op = g.r.pick(&[RmwOp::Add, RmwOp::Sub, RmwOp::And, RmwOp::Or, RmwOp::Xor, RmwOp::Max, RmwOp::Min, RmwOp::UMax, RmwOp::UMin]);
                let v = g.opnd(&pool, T::I32, 0);
                let dst = g.val(T::I32, 0);
                pool.push(dst);
                segs.push(Seg::GsAtomic { op, v, k: g.r.next() as u32, dst });
            }
            _ => {
                let t = g.r.pick(&[T::I32, T::I32, T::F32, T::I1]);
                let v = g.opnd(&pool, t, 2);
                let d = g.odepth(v);
                let dst = g.val(t, d);
                pool.push(dst);
                let lane = if g.r.chance(3, 4) { Some(g.r.below(LANES) as u32) } else { None };
                segs.push(Seg::Wave { v, lane, dst });
            }
        }
    }
    // Outputs: up to 12 distinct values, latest first.
    let mut outs: Vec<usize> = Vec::new();
    for &v in pool.iter().rev() {
        if outs.len() < 12 && (g.r.chance(1, 2) || outs.len() < 3) {
            outs.push(v);
        }
    }
    Prog { tys: g.tys, segs, lane, outs, gpu_safe: g.gpu_safe }
}

// -------------------------------------------------------------- emitter

struct Emit<'a> {
    s: Shader,
    p: &'a Prog,
    vals: Vec<Option<Value>>,
    lane: Value,
    gs_n: u32,
}

impl Emit<'_> {
    fn ty(&mut self, t: T) -> super::module::Ty {
        match t {
            T::I1 => self.s.m.i1(),
            T::I32 => self.s.m.i32(),
            T::I64 => self.s.m.i64(),
            T::F32 => self.s.m.f32(),
            T::F64 => self.s.m.f64(),
        }
    }
    fn o(&mut self, o: Opnd) -> Value {
        match o {
            Opnd::V(v) => self.vals[v].unwrap_or_else(|| panic!("value {v} used before definition")),
            Opnd::C(t, bits) => match t {
                T::F32 => self.s.m.f32c(bits as u32),
                T::F64 => {
                    let ty = self.s.m.f64();
                    self.s.m.const_float(ty, bits)
                }
                _ => {
                    let ty = self.ty(t);
                    self.s.m.const_int(ty, bits)
                }
            },
        }
    }
    fn insts(&mut self, is: &[Inst]) {
        for i in is {
            let v = match &i.op {
                Op::Bin(op, a, b) => {
                    let (a, b) = (self.o(*a), self.o(*b));
                    self.s.m.bin(*op, a, b)
                }
                Op::FBin(op, a, b, f) => {
                    let (a, b) = (self.o(*a), self.o(*b));
                    self.s.m.fbin(*op, a, b, *f)
                }
                Op::Cmp(p, a, b) => {
                    let (a, b) = (self.o(*a), self.o(*b));
                    self.s.m.cmp(*p, a, b)
                }
                Op::Select(c, a, b) => {
                    let (c, a, b) = (self.o(*c), self.o(*a), self.o(*b));
                    self.s.m.select(c, a, b)
                }
                Op::Cast(op, v, t) => {
                    let v = self.o(*v);
                    let t = self.ty(*t);
                    self.s.m.cast(*op, v, t)
                }
                Op::DxBin(op, a, b) => {
                    let (a, b) = (self.o(*a), self.o(*b));
                    self.s.binary(*op, a, b)
                }
                Op::DxUn(u, a) => {
                    let a = self.o(*a);
                    match u {
                        DxUn::Count => self.s.bit_count(BitCount::Countbits, a),
                        DxUn::Lo => self.s.bit_count(BitCount::FirstbitLo, a),
                        DxUn::Hi => self.s.bit_count(BitCount::FirstbitHi, a),
                        DxUn::Rev => self.s.bfrev(a),
                    }
                }
            };
            self.vals[i.dst] = Some(v);
        }
    }
    fn gs(&mut self, n: u64) -> GroupShared {
        let i32 = self.s.m.i32();
        self.gs_n += 1;
        let name = format!("gs{}", self.gs_n);
        self.s.groupshared(&name, i32, n)
    }
    fn seg(&mut self, seg: &Seg, mem: RawUav) {
        match seg {
            Seg::Insts(is) => self.insts(is),
            Seg::Diamond { cond, then_, else_, triangle, phis } => {
                let c = self.o(*cond);
                let pre = self.s.m.current_block();
                let tb = self.s.m.new_block();
                let eb = if *triangle { None } else { Some(self.s.m.new_block()) };
                let join = self.s.m.new_block();
                self.s.m.cond_br(c, tb, eb.unwrap_or(join));
                self.s.m.set_block(tb);
                self.insts(then_);
                let tv: Vec<Value> = phis.iter().map(|p| self.o(p.1)).collect();
                self.s.m.br(join);
                let from_else = match eb {
                    Some(eb) => {
                        self.s.m.set_block(eb);
                        self.insts(else_);
                        self.s.m.br(join);
                        eb
                    }
                    None => pre,
                };
                let ev: Vec<Value> = phis.iter().map(|p| self.o(p.2)).collect();
                self.s.m.set_block(join);
                for (k, p) in phis.iter().enumerate() {
                    let t = self.ty(self.p.tys[p.0]);
                    let phi = self.s.m.phi(t);
                    self.s.m.add_incoming(phi, tv[k], tb);
                    self.s.m.add_incoming(phi, ev[k], from_else);
                    self.vals[p.0] = Some(phi);
                }
            }
            Seg::Switch { sel, cases, default, phi } => {
                let v = self.o(*sel);
                let blocks: Vec<_> = cases.iter().map(|_| self.s.m.new_block()).collect();
                let db = self.s.m.new_block();
                let join = self.s.m.new_block();
                let cv: Vec<(Value, _)> = cases.iter().zip(&blocks).map(|(c, &b)| (self.s.m.i32c(c.0 as u32), b)).collect();
                self.s.m.switch(v, db, &cv);
                let mut inc = Vec::new();
                for (c, &b) in cases.iter().zip(&blocks).chain(std::iter::once((&(0u64, default.0.clone(), default.1), &db))) {
                    self.s.m.set_block(b);
                    self.insts(&c.1);
                    let r = self.o(c.2);
                    self.s.m.br(join);
                    inc.push((r, b));
                }
                self.s.m.set_block(join);
                let t = self.ty(self.p.tys[*phi]);
                let ph = self.s.m.phi(t);
                for (r, b) in inc {
                    self.s.m.add_incoming(ph, r, b);
                }
                self.vals[*phi] = Some(ph);
            }
            Seg::Loop { trip, ctr, ctr_next, cond, accs, body } => {
                let trip = self.o(*trip);
                let inits: Vec<Value> = accs.iter().map(|a| self.o(a.1)).collect();
                let pre = self.s.m.current_block();
                let header = self.s.m.new_block();
                let bb = self.s.m.new_block();
                let exit = self.s.m.new_block();
                self.s.m.br(header);
                self.s.m.set_block(header);
                let i32 = self.s.m.i32();
                let cp = self.s.m.phi(i32);
                self.vals[*ctr] = Some(cp);
                let mut aps = Vec::new();
                for a in accs {
                    let t = self.ty(self.p.tys[a.0]);
                    let ph = self.s.m.phi(t);
                    self.vals[a.0] = Some(ph);
                    aps.push(ph);
                }
                let c = self.s.m.cmp(Pred::ULt, cp, trip);
                self.vals[*cond] = Some(c);
                self.s.m.cond_br(c, bb, exit);
                self.s.m.set_block(bb);
                self.insts(body);
                let one = self.s.m.i32c(1);
                let n = self.s.m.bin(BinOp::Add, cp, one);
                self.vals[*ctr_next] = Some(n);
                let nexts: Vec<Value> = accs.iter().map(|a| self.o(a.2)).collect();
                self.s.m.br(header);
                let zero = self.s.m.i32c(0);
                self.s.m.add_incoming(cp, zero, pre);
                self.s.m.add_incoming(cp, n, bb);
                for (k, ph) in aps.into_iter().enumerate() {
                    self.s.m.add_incoming(ph, inits[k], pre);
                    self.s.m.add_incoming(ph, nexts[k], bb);
                }
                self.s.m.set_block(exit);
            }
            Seg::GsExchange { v, k, dst } => {
                let g = self.gs(LANES as u64);
                let v = self.o(*v);
                self.s.gs_store(g, self.lane, v);
                self.s.barrier(barrier::GROUP_SYNC);
                let kc = self.s.m.i32c(*k);
                let i = self.s.m.bin(BinOp::Add, self.lane, kc);
                let m = self.s.m.i32c(LANES as u32 - 1);
                let i = self.s.m.bin(BinOp::And, i, m);
                let r = self.s.gs_load(g, i);
                self.vals[*dst] = Some(r);
            }
            Seg::GsAtomic { op, v, k, dst } => {
                let g = self.gs(9);
                let v = self.o(*v);
                let seven = self.s.m.i32c(7);
                let idx = self.s.m.bin(BinOp::And, self.lane, seven);
                let kc = self.s.m.i32c(*k);
                let init = self.s.m.bin(BinOp::Mul, idx, kc);
                let init = self.s.m.bin(BinOp::Xor, init, kc);
                self.s.gs_store(g, idx, init);
                self.s.barrier(barrier::GROUP_SYNC);
                // `atomicrmw sub` as an add of the negation, the form DXC and the translator
                // emit: NVIDIA Blackwell drivers refuse pipelines with a groupshared
                // `atomicrmw sub` (CreateComputePipelineState E_OUTOFMEMORY).
                let (op, v) = if *op == RmwOp::Sub {
                    let zero = self.s.m.i32c(0);
                    (RmwOp::Add, self.s.m.bin(BinOp::Sub, zero, v))
                } else {
                    (*op, v)
                };
                self.s.gs_atomic(op, g, idx, v);
                let eight = self.s.m.i32c(8);
                self.s.gs_cmpxchg(g, eight, v, init);
                self.s.barrier(barrier::GROUP_SYNC);
                let two = self.s.m.i32c(2);
                let i = self.s.m.bin(BinOp::LShr, self.lane, two);
                let i = self.s.m.bin(BinOp::And, i, seven);
                let r = self.s.gs_load(g, i);
                self.vals[*dst] = Some(r);
            }
            Seg::Wave { v, lane, dst } => {
                let v = self.o(*v);
                let r = match lane {
                    Some(l) => {
                        let lc = self.s.m.i32c(*l);
                        self.s.wave_read_lane_at(v, lc)
                    }
                    None => self.s.wave_read_lane_first(v),
                };
                self.vals[*dst] = Some(r);
            }
            Seg::Load { slot, dsts } => {
                let off = self.slot(*slot);
                let mask = (1u8 << dsts.len()) - 1;
                let r = self.s.raw_load(mem, off, mask);
                for (d, v) in dsts.iter().zip(r) {
                    self.vals[*d] = Some(v);
                }
            }
        }
    }
    /// Byte offset of dword `64·k + lane`.
    fn slot(&mut self, k: u32) -> Value {
        let base = self.s.m.i32c(64 * k);
        let i = self.s.m.bin(BinOp::Add, base, self.lane);
        let two = self.s.m.i32c(2);
        self.s.m.bin(BinOp::Shl, i, two)
    }
}

/// Output words of value `v`: one dword, two for 64-bit values (lo, hi).
fn out_words(t: T) -> usize {
    if t.width() == 64 {
        2
    } else {
        1
    }
}

fn emit(p: &Prog) -> Vec<u8> {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let res = s.raw_uav(1, 0);
    let mem = s.raw_uav(2, 0);
    let lane = s.flattened_thread_id_in_group();
    let mut e = Emit { s, p, vals: vec![None; p.tys.len()], lane, gs_n: 0 };
    e.vals[p.lane] = Some(lane);
    for seg in &p.segs {
        e.seg(seg, mem);
    }
    let mut k = 0u32;
    let i32 = e.s.m.i32();
    let i64 = e.s.m.i64();
    for &o in &p.outs {
        let v = e.vals[o].unwrap();
        let words = match p.tys[o] {
            T::I1 => vec![e.s.m.cast(CastOp::ZExt, v, i32)],
            T::I32 => vec![v],
            T::F32 => vec![e.s.m.bitcast(v, i32)],
            T::I64 | T::F64 => {
                let v = e.s.m.bitcast(v, i64);
                let lo = e.s.m.cast(CastOp::Trunc, v, i32);
                let c = e.s.m.const_int(i64, 32);
                let hi = e.s.m.bin(BinOp::LShr, v, c);
                vec![lo, e.s.m.cast(CastOp::Trunc, hi, i32)]
            }
        };
        for w in words {
            let off = e.slot(k);
            e.s.raw_store(res, off, &[w]);
            k += 1;
        }
    }
    // A four-component store (stride 4 dwords per lane) of lane-derived words.
    let n = k.min(4) as u32;
    if n > 0 {
        let base = e.s.m.i32c(64 * k);
        let four = e.s.m.i32c(4);
        let i = e.s.m.bin(BinOp::Mul, lane, four);
        let i = e.s.m.bin(BinOp::Add, base, i);
        let two = e.s.m.i32c(2);
        let off = e.s.m.bin(BinOp::Shl, i, two);
        let words: Vec<Value> = (0..n)
            .map(|c| {
                let cc = e.s.m.i32c(c * 0x1111);
                e.s.m.bin(BinOp::Xor, lane, cc)
            })
            .collect();
        e.s.raw_store(res, off, &words);
    }
    e.s.finish([LANES as u32, 1, 1], Some(LANES as u32))
}

// ----------------------------------------------------------- interpreter

fn sext(v: u64, w: u32) -> i64 {
    if w == 64 {
        v as i64
    } else {
        ((v << (64 - w)) as i64) >> (64 - w)
    }
}

fn f(t: T, bits: u64) -> f64 {
    if t == T::F32 {
        f32::from_bits(bits as u32) as f64
    } else {
        f64::from_bits(bits)
    }
}

fn fbits(t: T, x: f64) -> u64 {
    if t == T::F32 {
        (x as f32).to_bits() as u64
    } else {
        x.to_bits()
    }
}

struct Interp<'a> {
    p: &'a Prog,
    env: Vec<Vec<u64>>,
}

impl Interp<'_> {
    fn o(&self, l: usize, o: Opnd) -> u64 {
        match o {
            Opnd::V(v) => self.env[l][v],
            Opnd::C(t, b) => b & t.mask(),
        }
    }
    fn ot(&self, o: Opnd) -> T {
        match o {
            Opnd::V(v) => self.p.tys[v],
            Opnd::C(t, _) => t,
        }
    }
    fn insts(&mut self, l: usize, is: &[Inst]) {
        for i in is {
            let t = self.p.tys[i.dst];
            let r = match &i.op {
                Op::Bin(op, a, b) => {
                    let (x, y) = (self.o(l, *a), self.o(l, *b));
                    let w = t.width();
                    let (sx, sy) = (sext(x, w), sext(y, w));
                    match op {
                        BinOp::Add => x.wrapping_add(y),
                        BinOp::Sub => x.wrapping_sub(y),
                        BinOp::Mul => x.wrapping_mul(y),
                        BinOp::UDiv => x / y,
                        BinOp::URem => x % y,
                        BinOp::SDiv => sx.wrapping_div(sy) as u64,
                        BinOp::SRem => sx.wrapping_rem(sy) as u64,
                        BinOp::Shl => x << y,
                        BinOp::LShr => x >> y,
                        BinOp::AShr => (sx >> y) as u64,
                        BinOp::And => x & y,
                        BinOp::Or => x | y,
                        BinOp::Xor => x ^ y,
                    }
                }
                Op::FBin(op, a, b, _) => {
                    let (x, y) = (f(t, self.o(l, *a)), f(t, self.o(l, *b)));
                    let r = match op {
                        BinOp::Add => x + y,
                        BinOp::Sub => x - y,
                        BinOp::Mul => x * y,
                        _ => unreachable!(),
                    };
                    fbits(t, r)
                }
                Op::Cmp(p, a, b) => {
                    let at = self.ot(*a);
                    let (x, y) = (self.o(l, *a), self.o(l, *b));
                    let r = if at.is_float() {
                        let (x, y) = (f(at, x), f(at, y));
                        let uno = x.is_nan() || y.is_nan();
                        let (lt, eq, gt) = (x < y, x == y, x > y);
                        match p {
                            Pred::FOEq => eq,
                            Pred::FOGt => gt,
                            Pred::FOGe => gt || eq,
                            Pred::FOLt => lt,
                            Pred::FOLe => lt || eq,
                            Pred::FONe => lt || gt,
                            Pred::FOrd => !uno,
                            Pred::FUno => uno,
                            Pred::FUEq => uno || eq,
                            Pred::FUGt => uno || gt,
                            Pred::FUGe => uno || gt || eq,
                            Pred::FULt => uno || lt,
                            Pred::FULe => uno || lt || eq,
                            Pred::FUNe => uno || lt || gt,
                            _ => unreachable!(),
                        }
                    } else {
                        let w = at.width();
                        let (sx, sy) = (sext(x, w), sext(y, w));
                        match p {
                            Pred::Eq => x == y,
                            Pred::Ne => x != y,
                            Pred::UGt => x > y,
                            Pred::UGe => x >= y,
                            Pred::ULt => x < y,
                            Pred::ULe => x <= y,
                            Pred::SGt => sx > sy,
                            Pred::SGe => sx >= sy,
                            Pred::SLt => sx < sy,
                            Pred::SLe => sx <= sy,
                            _ => unreachable!(),
                        }
                    };
                    r as u64
                }
                Op::Select(c, a, b) => {
                    if self.o(l, *c) != 0 {
                        self.o(l, *a)
                    } else {
                        self.o(l, *b)
                    }
                }
                Op::Cast(op, v, _) => {
                    let ft = self.ot(*v);
                    let x = self.o(l, *v);
                    match op {
                        CastOp::ZExt | CastOp::Trunc | CastOp::BitCast => x,
                        CastOp::SExt => sext(x, ft.width()) as u64,
                        CastOp::UIToFP => fbits(t, x as f64),
                        CastOp::SIToFP => fbits(t, sext(x, ft.width()) as f64),
                        CastOp::FPExt | CastOp::FPTrunc => fbits(t, f(ft, x)),
                        _ => unreachable!(),
                    }
                }
                Op::DxBin(op, a, b) => {
                    let (x, y) = (self.o(l, *a), self.o(l, *b));
                    let w = t.width();
                    let (sx, sy) = (sext(x, w), sext(y, w));
                    match op {
                        Binary::IMax => sx.max(sy) as u64,
                        Binary::IMin => sx.min(sy) as u64,
                        Binary::UMax => x.max(y),
                        Binary::UMin => x.min(y),
                        _ => unreachable!(),
                    }
                }
                Op::DxUn(u, a) => {
                    let at = self.ot(*a);
                    let x = self.o(l, *a);
                    let w = at.width();
                    match u {
                        DxUn::Count => x.count_ones() as u64,
                        DxUn::Lo => {
                            if x == 0 {
                                u32::MAX as u64
                            } else {
                                x.trailing_zeros() as u64
                            }
                        }
                        DxUn::Hi => {
                            if x == 0 {
                                u32::MAX as u64
                            } else {
                                (x.leading_zeros() - (64 - w)) as u64
                            }
                        }
                        DxUn::Rev => x.reverse_bits() >> (64 - w),
                    }
                }
            };
            self.env[l][i.dst] = r & t.mask();
        }
    }

    fn seg(&mut self, seg: &Seg, mem: &[u32]) {
        match seg {
            Seg::Insts(is) => (0..LANES).for_each(|l| self.insts(l, is)),
            Seg::Diamond { cond, then_, else_, phis, .. } => {
                for l in 0..LANES {
                    let taken = self.o(l, *cond) != 0;
                    self.insts(l, if taken { then_ } else { else_ });
                    let vs: Vec<u64> = phis.iter().map(|p| self.o(l, if taken { p.1 } else { p.2 })).collect();
                    for (p, v) in phis.iter().zip(vs) {
                        self.env[l][p.0] = v;
                    }
                }
            }
            Seg::Switch { sel, cases, default, phi } => {
                for l in 0..LANES {
                    let v = self.o(l, *sel);
                    let (body, out) = cases.iter().find(|c| c.0 & 0xffff_ffff == v).map_or((&default.0, default.1), |c| (&c.1, c.2));
                    self.insts(l, body);
                    self.env[l][*phi] = self.o(l, out);
                }
            }
            Seg::Loop { trip, ctr, accs, body, .. } => {
                for l in 0..LANES {
                    let trip = self.o(l, *trip);
                    let mut vals: Vec<u64> = accs.iter().map(|a| self.o(l, a.1)).collect();
                    let mut i = 0u64;
                    loop {
                        self.env[l][*ctr] = i;
                        for (a, v) in accs.iter().zip(&vals) {
                            self.env[l][a.0] = *v;
                        }
                        if i >= trip {
                            break;
                        }
                        self.insts(l, body);
                        vals = accs.iter().map(|a| self.o(l, a.2)).collect();
                        i += 1;
                    }
                }
            }
            Seg::GsExchange { v, k, dst } => {
                let vs: Vec<u64> = (0..LANES).map(|l| self.o(l, *v)).collect();
                for l in 0..LANES {
                    self.env[l][*dst] = vs[(l + *k as usize) & (LANES - 1)];
                }
            }
            Seg::GsAtomic { op, v, k, dst } => {
                let mut cnt: Vec<u32> = (0..8u32).map(|i| i.wrapping_mul(*k) ^ *k).collect();
                for l in 0..LANES {
                    let x = self.o(l, *v) as u32;
                    let c = &mut cnt[l & 7];
                    *c = match op {
                        RmwOp::Add => c.wrapping_add(x),
                        RmwOp::Sub => c.wrapping_sub(x),
                        RmwOp::And => *c & x,
                        RmwOp::Or => *c | x,
                        RmwOp::Xor => *c ^ x,
                        RmwOp::Max => (*c as i32).max(x as i32) as u32,
                        RmwOp::Min => (*c as i32).min(x as i32) as u32,
                        RmwOp::UMax => (*c).max(x),
                        RmwOp::UMin => (*c).min(x),
                        _ => unreachable!(),
                    };
                }
                for l in 0..LANES {
                    self.env[l][*dst] = cnt[(l >> 2) & 7] as u64;
                }
            }
            Seg::Wave { v, lane, dst } => {
                let src = lane.unwrap_or(0) as usize;
                let x = self.o(src, *v);
                for l in 0..LANES {
                    self.env[l][*dst] = x;
                }
            }
            Seg::Load { slot, dsts } => {
                for l in 0..LANES {
                    for (c, d) in dsts.iter().enumerate() {
                        self.env[l][*d] = mem[64 * *slot as usize + l + c] as u64;
                    }
                }
            }
        }
    }
}

/// Expected `res` dwords: `(index, value)` pairs.
fn interpret(p: &Prog, mem: &[u32]) -> Vec<(usize, u32)> {
    let mut it = Interp { p, env: vec![vec![0; p.tys.len()]; LANES] };
    for l in 0..LANES {
        it.env[l][p.lane] = l as u64;
    }
    for seg in &p.segs {
        it.seg(seg, mem);
    }
    let mut out = Vec::new();
    let mut k = 0;
    for &o in &p.outs {
        let n = out_words(p.tys[o]);
        for wi in 0..n {
            for l in 0..LANES {
                let v = it.env[l][o];
                out.push((64 * k + l, (v >> (32 * wi)) as u32));
            }
            k += 1;
        }
    }
    for l in 0..LANES {
        for c in 0..k.min(4) {
            out.push((64 * k + 4 * l + c, (l as u32) ^ (c as u32 * 0x1111)));
        }
    }
    out
}

fn inputs(seed: u64) -> Vec<u32> {
    let mut r = Rng::new(seed ^ 0xDEAD_BEEF);
    (0..64 * IN_SLOTS as usize).map(|_| r.bits(32) as u32).collect()
}

// ---------------------------------------------------------------- tests

fn check_valid(seed: u64, c: &[u8]) {
    if let Err(e) = validate::validate(c) {
        let dis = validate::disassemble(c).unwrap_or_else(|d| d);
        panic!("seed {seed}: validator rejected the module:\n{e}\n--- disassembly ---\n{dis}");
    }
}

fn run_seeds(range: std::ops::Range<u64>, gpu_every: u64) -> (usize, usize) {
    let gpu = matches!(runner::wave32_supported(), Ok(true));
    let mut ran = 0;
    // GPU-run containers whose digest takes each final-block layout.
    let mut tails = [0usize; 2];
    for seed in range.clone() {
        let p = generate(seed);
        let c = emit(&p);
        check_valid(seed, &c);
        if gpu && p.gpu_safe && seed % gpu_every == 0 {
            let mem = inputs(seed);
            let r = runner::run_with(&c, &runner::KernelState::default(), &mem, &[], &[]).unwrap_or_else(|e| panic!("seed {seed}: GPU run ({} bytes, {} instructions): {e}", c.len(), instructions(&c)));
            for (i, v) in interpret(&p, &mem) {
                if r.raw[i] != v {
                    let dis = validate::disassemble(&c).unwrap_or_else(|d| d);
                    panic!("seed {seed}: res[{i}] = {:#x}, CPU {v:#x}\nprogram: {:#?}\n--- disassembly ---\n{dis}", r.raw[i], p.segs);
                }
            }
            ran += 1;
            tails[((c.len() - 20) % 64 >= 56) as usize] += 1;
        }
    }
    eprintln!("fuzz seeds {range:?}: validated {}, GPU-compared {ran} (digest tails < 56 / >= 56: {tails:?})", range.clone().count());
    (range.count(), ran)
}

/// Instructions in the module's disassembly (for failure reports).
fn instructions(c: &[u8]) -> usize {
    validate::disassemble(c).map(|d| d.lines().filter(|l| l.starts_with("  ") && !l.trim_start().starts_with(';')).count()).unwrap_or(0)
}

#[test]
fn fuzz_validate_and_gpu_0() {
    run_seeds(0..500, 1);
}
#[test]
fn fuzz_validate_and_gpu_1() {
    run_seeds(500..1000, 1);
}
#[test]
fn fuzz_validate_and_gpu_2() {
    run_seeds(1000..1500, 1);
}
#[test]
fn fuzz_validate_and_gpu_3() {
    run_seeds(1500..2000, 1);
}
/// Prints the disassembly of one generated module (`FUZZ_SEED=n`).
#[test]
#[ignore]
fn debug_seed() {
    let seed: u64 = std::env::var("FUZZ_SEED").map(|s| s.parse().unwrap()).unwrap_or(0);
    let p = generate(seed);
    let c = emit(&p);
    println!("{}", validate::disassemble(&c).unwrap());
    println!("{:?}", validate::validate(&c));
    println!("program: {:#?}
outs: {:?}", p.segs, p.outs);
    if let Ok(w) = std::env::var("FUZZ_WATCH") {
        let i: usize = w.parse().unwrap();
        let mem = inputs(seed);
        let exp = interpret(&p, &mem).into_iter().find(|&(j, _)| j == i);
        println!("watch res[{i}]: CPU {exp:x?}; mem[0..70] = {:x?}", &mem[..70.min(mem.len())]);
    }
    println!("seed {seed}: {} bytes, {} instructions, gpu_safe {}", c.len(), instructions(&c), p.gpu_safe);
    if p.gpu_safe && matches!(runner::wave32_supported(), Ok(true)) {
        let mem = inputs(seed);
        match runner::run_with(&c, &runner::KernelState::default(), &mem, &[], &[]) {
            Ok(r) => {
                let bad = interpret(&p, &mem).into_iter().filter(|&(i, v)| r.raw[i] != v).count();
                println!("GPU run: {bad} mismatching results");
            }
            Err(e) => println!("GPU run: {e}"),
        }
    }
}

/// Regression: i64 / double instructions built with plain [`super::module::Module`]
/// methods (not the `Shader` helpers) must still set `Int64Ops`, `Doubles`
/// and — for int ↔ double conversions and double division —
/// `DoubleExtensions` ("Flags must match usage" otherwise).
#[test]
fn shader_flags_follow_plain_instruction_types() {
    type Build = fn(&mut Shader, Value) -> Value;
    let cases: [Build; 4] = [
        |s, x| {
            let i64 = s.m.i64();
            let w = s.m.cast(CastOp::ZExt, x, i64);
            let p = s.m.bin(BinOp::Mul, w, w);
            let i32 = s.m.i32();
            s.m.cast(CastOp::Trunc, p, i32)
        },
        |s, x| {
            let (f32, f64) = (s.m.f32(), s.m.f64());
            let f = s.m.bitcast(x, f32);
            let d = s.m.cast(CastOp::FPExt, f, f64);
            let d = s.m.fbin(BinOp::Add, d, d, 0);
            let f = s.m.cast(CastOp::FPTrunc, d, f32);
            let i32 = s.m.i32();
            s.m.bitcast(f, i32)
        },
        |s, x| {
            let f64 = s.m.f64();
            let d = s.m.cast(CastOp::UIToFP, x, f64);
            let i32 = s.m.i32();
            s.m.cast(CastOp::FPToSI, d, i32)
        },
        |s, x| {
            let (f32, f64) = (s.m.f32(), s.m.f64());
            let f = s.m.bitcast(x, f32);
            let d = s.m.cast(CastOp::FPExt, f, f64);
            let q = s.m.fbin(BinOp::SDiv, d, d, 0);
            let f = s.m.cast(CastOp::FPTrunc, q, f32);
            let i32 = s.m.i32();
            s.m.bitcast(f, i32)
        },
    ];
    for (i, build) in cases.iter().enumerate() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let res = s.raw_uav(1, 0);
        let lane = s.flattened_thread_id_in_group();
        let two = s.m.i32c(2);
        let off = s.m.bin(BinOp::Shl, lane, two);
        let v = build(&mut s, lane);
        s.raw_store(res, off, &[v]);
        let c = s.finish([32, 1, 1], None);
        check_valid(i as u64, &c);
    }
}

/// Hand-reduced forms of the fuzz seeds that fail only on AMD (722 and 1599
/// on RDNA4, amdxc64; 540 on RDNA2, a gfx1036 iGPU), for bisecting on that
/// hardware: `cargo test -p xcl --lib
/// gpu::dxil::fuzz_tests::amd_reduced_seeds -- --ignored --nocapture`.
/// Prints one PASS / FAIL line per variant (first mismatching dword) and
/// panics at the end if any failed. `T::I32` value 0 is the lane index.
#[test]
#[ignore]
fn amd_reduced_seeds() {
    use Opnd::{C, V};
    let gpu = matches!(runner::wave32_supported(), Ok(true));
    if !gpu {
        eprintln!("SKIPPED: no wave32 GPU");
        return;
    }
    let ins = |dst, op| Inst { dst, op };
    // Seed 722 shape: x = in[lane]; w = WaveReadLaneFirst(x); a phi over a
    // divergent switch on x where the default and `case 1` take w (so a lane
    // alone in `case 1` must still see lane 0's x).
    let s722 = |countbits_case: bool| {
        let mut cases = vec![(1u64, vec![], V(3)), (2, vec![], V(0))];
        let mut tys = vec![T::I32, T::I32, T::I32, T::I32, T::I32];
        if countbits_case {
            cases.push((227944867, vec![ins(5, Op::DxUn(DxUn::Count, C(T::I64, 10215786766456627865)))], V(1)));
            tys.push(T::I32);
        }
        Prog { tys, segs: vec![Seg::Load { slot: 0, dsts: vec![1, 2] }, Seg::Wave { v: V(1), lane: None, dst: 3 }, Seg::Switch { sel: V(1), cases, default: (vec![], V(3)), phi: 4 }], lane: 0, outs: vec![4], gpu_safe: true }
    };
    // Seed 1599 shape: c = Countbits.i64(2); out ((c >> (lane & 31)) | Bfrev(c)) & 31.
    let s1599 = |dynamic: bool, rev: bool| {
        let src = if dynamic { V(2) } else { C(T::I64, 2) };
        let mut insts = vec![];
        if dynamic {
            // in[lane] & 0 | 2, as i64 (2 on every lane, unknown to the driver).
            insts.push(ins(6, Op::Bin(BinOp::And, V(1), C(T::I32, 0))));
            insts.push(ins(7, Op::Bin(BinOp::Or, V(6), C(T::I32, 2))));
            insts.push(ins(2, Op::Cast(CastOp::ZExt, V(7), T::I64)));
        }
        insts.push(ins(3, Op::DxUn(DxUn::Count, src)));
        insts.push(ins(8, Op::Bin(BinOp::And, V(0), C(T::I32, 31))));
        insts.push(ins(9, Op::Bin(BinOp::LShr, V(3), V(8))));
        insts.push(ins(4, Op::DxUn(DxUn::Rev, V(3))));
        insts.push(ins(10, Op::Bin(BinOp::Or, V(9), V(4))));
        insts.push(ins(5, Op::Bin(BinOp::And, V(10), C(T::I32, 31))));
        let tys = vec![T::I32, T::I32, T::I64, T::I32, T::I32, T::I32, T::I32, T::I32, T::I32, T::I32, T::I32];
        let outs = if rev { vec![5, 3, 9, 4] } else { vec![3] };
        Prog { tys, segs: vec![Seg::Load { slot: 0, dsts: vec![1] }, Seg::Insts(insts)], lane: 0, outs, gpu_safe: true }
    };
    // Seed 540 shape: b = trunc in[lane] to i1; s = sext b to i64 (RDNA2 gives
    // its high dword 0: 0x00000000_ffffffff for a set b); seed 540 itself goes
    // on to ashr(s, 63) - (-7), which then reads 7 instead of 6.
    let s540 = |shift: bool| {
        let mut insts = vec![ins(2, Op::Cast(CastOp::Trunc, V(1), T::I1)), ins(3, Op::Cast(CastOp::SExt, V(2), T::I64))];
        if shift {
            insts.push(ins(4, Op::Bin(BinOp::AShr, V(3), C(T::I64, 63))));
            insts.push(ins(5, Op::Bin(BinOp::Sub, V(4), C(T::I64, (-7i64) as u64))));
        }
        let tys = vec![T::I32, T::I32, T::I1, T::I64, T::I64, T::I64];
        Prog { tys, segs: vec![Seg::Load { slot: 0, dsts: vec![1] }, Seg::Insts(insts)], lane: 0, outs: vec![if shift { 5 } else { 3 }], gpu_safe: true }
    };
    let progs = [
        ("722 switch + WaveReadLaneFirst", s722(false)),
        ("722 switch + WaveReadLaneFirst + Countbits.i64 case", s722(true)),
        ("1599 Countbits.i64(const 2)", s1599(false, false)),
        ("1599 Countbits.i64(const 2) -> shift / Bfrev", s1599(false, true)),
        ("1599 Countbits.i64(dynamic 2) -> shift / Bfrev", s1599(true, true)),
        ("540 sext i1 -> i64", s540(false)),
        ("540 sext i1 -> i64 -> ashr 63 - (-7)", s540(true)),
    ];
    let mut failed = 0;
    for (name, p) in &progs {
        let c = emit(p);
        check_valid(0, &c);
        let mem = inputs(722);
        let r = runner::run_with(&c, &runner::KernelState::default(), &mem, &[], &[]).unwrap_or_else(|e| panic!("{name}: GPU run: {e}"));
        match interpret(p, &mem).into_iter().find(|&(i, v)| r.raw[i] != v) {
            None => eprintln!("PASS {name}"),
            Some((i, v)) => {
                failed += 1;
                eprintln!("FAIL {name}: res[{i}] = {:#x}, CPU {v:#x}\n{}", r.raw[i], validate::disassemble(&c).unwrap_or_else(|d| d));
            }
        }
    }
    for seed in [540u64, 722, 1599] {
        let p = generate(seed);
        let c = emit(&p);
        let mem = inputs(seed);
        let r = runner::run_with(&c, &runner::KernelState::default(), &mem, &[], &[]).unwrap_or_else(|e| panic!("seed {seed}: GPU run: {e}"));
        let bad = interpret(&p, &mem).into_iter().filter(|&(i, v)| r.raw[i] != v).count();
        eprintln!("{} full seed {seed}: {bad} mismatching dwords", if bad == 0 { "PASS" } else { "FAIL" });
        failed += (bad != 0) as usize;
    }
    assert_eq!(failed, 0, "{failed} AMD repro variants failed");
}
