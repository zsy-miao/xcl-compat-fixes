//! Tests of the `dx.op` helpers in [`super::ops`]: every emitted kernel must
//! pass DXC's validator; where the runner can bind the resources, the kernel
//! also runs on the GPU (wave32, 32 threads) and is compared with a CPU
//! reference. Inputs come from `u2` (memory), results go to `u1` (`res`):
//! value `k` of lane `l` at dword `64·k + l`.

use super::container::Stage;
use super::module::{BinOp, CastOp, Pred, RmwOp, Value};
use super::ops::*;
use super::shader::{RawUav, Shader};
use super::validate;
use crate::gpu::recomp::runner;

struct K {
    s: Shader,
    lane: Value,
    res: RawUav,
    mem: RawUav,
}

fn kernel() -> K {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let res = s.raw_uav(1, 0);
    let mem = s.raw_uav(2, 0);
    let lane = s.flattened_thread_id_in_group();
    K { s, lane, res, mem }
}

impl K {
    fn slot(&mut self, k: u32) -> Value {
        let base = self.s.m.i32c(64 * k);
        let i = self.s.m.bin(BinOp::Add, base, self.lane);
        let two = self.s.m.i32c(2);
        self.s.m.bin(BinOp::Shl, i, two)
    }
    /// mem[64k + lane] as i32.
    fn input(&mut self, k: u32) -> Value {
        let off = self.slot(k);
        self.s.raw_load(self.mem, off, 1)[0]
    }
    fn input_f(&mut self, k: u32) -> Value {
        let v = self.input(k);
        let f = self.s.m.f32();
        self.s.m.bitcast(v, f)
    }
    /// res[64k + lane] = v (f32 bitcast, i1 zero-extended).
    fn out(&mut self, k: u32, v: Value) {
        let i32 = self.s.m.i32();
        let t = self.s.m.type_of(v);
        let v = if t == self.s.m.i1() {
            self.s.m.cast(CastOp::ZExt, v, i32)
        } else if t == self.s.m.f32() {
            self.s.m.bitcast(v, i32)
        } else {
            v
        };
        let off = self.slot(k);
        self.s.raw_store(self.res, off, &[v]);
    }
    fn finish(self, nt: [u32; 3]) -> Vec<u8> {
        let c = self.s.finish(nt, Some(32));
        check(&c);
        c
    }
}

fn check(c: &[u8]) {
    if let Err(e) = validate::validate(c) {
        let dis = validate::disassemble(c).unwrap_or_else(|d| d);
        panic!("validator rejected the kernel:\n{e}\n--- disassembly ---\n{dis}");
    }
}

/// Runs a validated kernel; `None` when there is no wave32 GPU.
fn gpu(c: &[u8], mem: &[u32], tex: &[runner::TestTexture], smp: &[runner::TestSampler]) -> Option<runner::KernelResult> {
    match runner::wave32_supported() {
        Ok(true) => {}
        Ok(false) => {
            runner::skip_test("no wave32");
            return None;
        }
        Err(e) => {
            runner::skip_test(&e);
            return None;
        }
    }
    Some(runner::run_with(c, &runner::KernelState::default(), mem, tex, smp).expect("GPU run"))
}

fn outv(r: &runner::KernelResult, k: usize) -> &[u32] {
    &r.raw[64 * k..64 * k + 32]
}

fn mem_of(inputs: &[Vec<u32>]) -> Vec<u32> {
    let mut m = vec![0u32; 64 * inputs.len().max(1)];
    for (k, v) in inputs.iter().enumerate() {
        m[64 * k..64 * k + v.len()].copy_from_slice(v);
    }
    m
}

// ------------------------------------------------------------------- math

const FVALS: [f32; 32] = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.25, 3.75, -3.75, 100.3, -100.7, 1e-3, 7.0, 0.9999, 1e10, -1e10, 0.1, 2.0, 4.0, 9.0,
    f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.25, 33.5, -0.75, 1e-30, 6.5,
];
const UVALS: [u32; 32] = [
    0, 1, 2, 3, 0xffff_ffff, 0x8000_0000, 0x7fff_ffff, 0x1234_5678, 0xdead_beef, 0x0f0f_0f0f, 16, 255, 256, 0x10000, 0xfffe, 7, 1000, 12345, 0xffff_0000, 0x8000_0001, 5, 6, 9,
    100, 0x4000_0000, 0x0000_ffff, 31, 32, 33, 0xc000_0000, 0x5555_5555, 0xaaaa_aaaa,
];

fn uvals2() -> Vec<u32> {
    UVALS.iter().map(|&v| v.rotate_left(7) ^ 0x9e37_79b9).collect()
}

#[test]
fn exact_float_unary_ops_match_cpu() {
    let ops = [FUnary::FAbs, FUnary::Saturate, FUnary::Frc, FUnary::RoundNe, FUnary::RoundNi, FUnary::RoundPi, FUnary::RoundZ];
    let mut k = kernel();
    let f = k.input_f(0);
    for (i, &op) in ops.iter().enumerate() {
        let r = k.s.funary(op, f);
        k.out(i as u32, r);
    }
    let n = ops.len() as u32;
    for (j, c) in [FClass::IsNaN, FClass::IsInf, FClass::IsFinite].into_iter().enumerate() {
        let r = k.s.fclass(c, f);
        k.out(n + j as u32, r);
    }
    let c = k.finish([32, 1, 1]);
    let fin: Vec<u32> = FVALS.iter().map(|f| f.to_bits()).collect();
    let Some(r) = gpu(&c, &mem_of(&[fin]), &[], &[]) else { return };
    let cpu: [fn(f32) -> f32; 7] = [f32::abs, |x| if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }, |x| x - x.floor(), f32::round_ties_even, f32::floor, f32::ceil, f32::trunc];
    for (i, f) in cpu.iter().enumerate() {
        for (l, &x) in FVALS.iter().enumerate() {
            let want = f(x);
            let got = f32::from_bits(outv(&r, i)[l]);
            // Denormals may flush; Frc(±inf) is NaN; compare NaN-ness then bits.
            let ok = (want.is_nan() && got.is_nan()) || want.to_bits() == got.to_bits() || (want == 0.0 && got == 0.0 && i == 2);
            assert!(ok || (x.abs() < 1e-20), "op {:?} of {x}: got {got}, want {want}", ops[i]);
        }
    }
    for (l, &x) in FVALS.iter().enumerate() {
        assert_eq!(outv(&r, 7)[l], x.is_nan() as u32, "isnan {x}");
        assert_eq!(outv(&r, 8)[l], x.is_infinite() as u32, "isinf {x}");
        assert_eq!(outv(&r, 9)[l], x.is_finite() as u32, "isfinite {x}");
    }
}

#[test]
fn approximate_float_unary_ops_are_within_tolerance() {
    let ops = [FUnary::Cos, FUnary::Sin, FUnary::Exp, FUnary::Log, FUnary::Sqrt, FUnary::Rsqrt];
    let mut k = kernel();
    let f = k.input_f(0);
    for (i, &op) in ops.iter().enumerate() {
        let r = k.s.funary(op, f);
        k.out(i as u32, r);
    }
    let c = k.finish([32, 1, 1]);
    let xs: Vec<f32> = (0..32).map(|l| 0.05 + l as f32 * 0.37).collect();
    let Some(r) = gpu(&c, &mem_of(&[xs.iter().map(|x| x.to_bits()).collect()]), &[], &[]) else { return };
    let cpu: [fn(f32) -> f32; 6] = [f32::cos, f32::sin, f32::exp2, f32::log2, f32::sqrt, |x| 1.0 / x.sqrt()];
    for (i, f) in cpu.iter().enumerate() {
        for (l, &x) in xs.iter().enumerate() {
            let (want, got) = (f(x), f32::from_bits(outv(&r, i)[l]));
            let tol = 1e-3f32.max(want.abs() * 1e-5);
            assert!((want - got).abs() <= tol, "{:?}({x}) = {got}, want {want}", ops[i]);
        }
    }
}

#[test]
fn integer_ops_match_cpu() {
    let mut k = kernel();
    let a = k.input(0);
    let b = k.input(1);
    let mut outs = Vec::new();
    outs.push(k.s.bfrev(a));
    for op in [BitCount::Countbits, BitCount::FirstbitLo, BitCount::FirstbitHi, BitCount::FirstbitSHi] {
        outs.push(k.s.bit_count(op, a));
    }
    for op in [Binary::IMax, Binary::IMin, Binary::UMax, Binary::UMin] {
        outs.push(k.s.binary(op, a, b));
    }
    outs.push(k.s.tertiary(Tertiary::IMad, a, b, a));
    outs.push(k.s.tertiary(Tertiary::UMad, a, b, b));
    let (w, o) = {
        let m31 = k.s.m.i32c(31);
        let w = k.s.m.bin(BinOp::And, b, m31);
        let o = { let c = k.s.m.i32c(5); k.s.m.bin(BinOp::LShr, b, c) };
        let o = k.s.m.bin(BinOp::And, o, m31);
        (w, o)
    };
    outs.push(k.s.tertiary(Tertiary::Ibfe, w, o, a));
    outs.push(k.s.tertiary(Tertiary::Ubfe, w, o, a));
    outs.push(k.s.bfi(w, o, a, b));
    let (hi, lo) = k.s.umul_wide(a, b);
    outs.extend([hi, lo]);
    let (hi, lo) = k.s.imul_wide(a, b);
    outs.extend([hi, lo]);
    let (q, rm) = k.s.udiv_rem(a, b);
    outs.extend([q, rm]);
    for (i, &v) in outs.iter().enumerate() {
        k.out(i as u32, v);
    }
    let c = k.finish([32, 1, 1]);
    let bs = uvals2();
    let Some(r) = gpu(&c, &mem_of(&[UVALS.to_vec(), bs.clone()]), &[], &[]) else { return };
    let fhi = |x: u32| if x == 0 { u32::MAX } else { x.leading_zeros() };
    let bfe = |w: u32, o: u32, v: u32, signed: bool| -> u32 {
        if w == 0 {
            0
        } else if w + o < 32 {
            let s = v << (32 - w - o);
            if signed { ((s as i32) >> (32 - w)) as u32 } else { s >> (32 - w) }
        } else if signed {
            ((v as i32) >> o) as u32
        } else {
            v >> o
        }
    };
    for l in 0..32 {
        let (a, b) = (UVALS[l], bs[l]);
        let (w, o) = (b & 31, (b >> 5) & 31);
        let bfi = { let m = ((1u64 << w) - 1) as u32; let m = m << o; ((a << o) & m) | (b & !m) };
        let um = a as u64 * b as u64;
        let im = (a as i32 as i64 * b as i32 as i64) as u64;
        let want = [
            a.reverse_bits(),
            a.count_ones(),
            if a == 0 { u32::MAX } else { a.trailing_zeros() },
            fhi(a),
            fhi(if (a as i32) < 0 { !a } else { a }),
            (a as i32).max(b as i32) as u32,
            (a as i32).min(b as i32) as u32,
            a.max(b),
            a.min(b),
            a.wrapping_mul(b).wrapping_add(a),
            a.wrapping_mul(b).wrapping_add(b),
            bfe(w, o, a, true),
            bfe(w, o, a, false),
            bfi,
            (um >> 32) as u32,
            um as u32,
            (im >> 32) as u32,
            im as u32,
            a.checked_div(b).unwrap_or(u32::MAX),
            a.checked_rem(b).unwrap_or(u32::MAX),
        ];
        for (i, &wv) in want.iter().enumerate() {
            assert_eq!(outv(&r, i)[l], wv, "int op #{i} lane {l}: a={a:#x} b={b:#x}");
        }
    }
}

#[test]
fn float_binary_tertiary_and_half_conversions_match_cpu() {
    let mut k = kernel();
    let a = k.input_f(0);
    let b = k.input_f(1);
    let bits = k.input(0);
    let mx = k.s.binary(Binary::FMax, a, b);
    let mn = k.s.binary(Binary::FMin, a, b);
    let h = k.s.f32_to_f16(a);
    let back = k.s.f16_to_f32(bits);
    let mad = k.s.tertiary(Tertiary::FMad, a, b, a);
    for (i, v) in [mx, mn, h, back, mad].into_iter().enumerate() {
        k.out(i as u32, v);
    }
    let c = k.finish([32, 1, 1]);
    let bv: Vec<f32> = FVALS.iter().rev().copied().collect();
    let Some(r) = gpu(&c, &mem_of(&[FVALS.iter().map(|f| f.to_bits()).collect(), bv.iter().map(|f| f.to_bits()).collect()]), &[], &[]) else { return };
    for l in 0..32 {
        let (x, y) = (FVALS[l], bv[l]);
        let fl = |v: f32| if v.abs() < 1e-37 { 0.0 } else { v };
        assert_eq!(fl(f32::from_bits(outv(&r, 0)[l])).to_bits(), fl(x.max(y)).to_bits(), "fmax {x} {y}");
        assert_eq!(fl(f32::from_bits(outv(&r, 1)[l])).to_bits(), fl(x.min(y)).to_bits(), "fmin {x} {y}");
        if !x.is_nan() {
            // Rounding is implementation-defined: nearest-even or towards zero
            // (the RTX 4090 truncates).
            let got = outv(&r, 2)[l];
            assert!(got == f32_to_f16(x, false) as u32 || got == f32_to_f16(x, true) as u32, "f32tof16 {x}: {got:#x}");
        }
        let hb = (x.to_bits() & 0xffff) as u16;
        let hf = f16_to_f32(hb);
        if !hf.is_nan() {
            assert_eq!(outv(&r, 3)[l], hf.to_bits(), "f16tof32 {hb:#x}");
        }
        let m = f32::from_bits(outv(&r, 4)[l]);
        let w = x * y + x;
        assert!((m.is_nan() && w.is_nan()) || m == w || (m - w).abs() <= w.abs() * 1e-6, "fmad {x} {y}: {m} vs {w}");
    }
}

#[test]
fn uaddc_and_usubb_match_cpu() {
    // HLSL never emits UAddc/USubb, but DXC 1.9's validator accepts them in cs_6_6.
    let mut k = kernel();
    let a = k.input(0);
    let b = k.input(1);
    let (sum, carry) = k.s.uaddc(a, b);
    let (diff, borrow) = k.s.usubb(a, b);
    for (i, v) in [sum, carry, diff, borrow].into_iter().enumerate() {
        k.out(i as u32, v);
    }
    let c = k.finish([32, 1, 1]);
    let bs = uvals2();
    let Some(r) = gpu(&c, &mem_of(&[UVALS.to_vec(), bs.clone()]), &[], &[]) else { return };
    for l in 0..32 {
        let (a, b) = (UVALS[l], bs[l]);
        let (s, c) = a.overflowing_add(b);
        let (d, w) = a.overflowing_sub(b);
        assert_eq!([outv(&r, 0)[l], outv(&r, 1)[l], outv(&r, 2)[l], outv(&r, 3)[l]], [s, c as u32, d, w as u32], "lane {l}: {a:#x} {b:#x}");
    }
}

#[test]
fn doubles_match_cpu() {
    let mut k = kernel();
    let lo = k.input(0);
    let hi = k.input(1);
    let d = k.s.make_double(lo, hi);
    let two = { let f64 = k.s.m.f64(); k.s.m.const_float(f64, 2.0f64.to_bits()) };
    let one = { let f64 = k.s.m.f64(); k.s.m.const_float(f64, 0.1f64.to_bits()) };
    let f = k.s.tertiary(Tertiary::Fma, d, two, one);
    let mx = k.s.binary(Binary::FMax, f, d);
    let (l0, h0) = k.s.split_double(f);
    let (l1, h1) = k.s.split_double(mx);
    for (i, v) in [l0, h0, l1, h1].into_iter().enumerate() {
        k.out(i as u32, v);
    }
    let c = k.finish([32, 1, 1]);
    let ds: Vec<f64> = (0..32).map(|l| (l as f64 - 15.5) * 1.000_000_000_1).collect();
    let Some(r) = gpu(&c, &mem_of(&[ds.iter().map(|d| d.to_bits() as u32).collect(), ds.iter().map(|d| (d.to_bits() >> 32) as u32).collect()]), &[], &[]) else { return };
    for (l, &x) in ds.iter().enumerate() {
        let f = x.mul_add(2.0, 0.1);
        let got = |i: usize| (outv(&r, i)[l] as u64) | ((outv(&r, i + 1)[l] as u64) << 32);
        assert_eq!(got(0), f.to_bits(), "fma({x})");
        assert_eq!(got(2), f.max(x).to_bits(), "max({x})");
    }
}

// ----------------------------------------------------------------- memory

#[test]
fn groupshared_barrier_and_atomics_run() {
    let mut k = kernel();
    let i32 = k.s.m.i32();
    let i64 = k.s.m.i64();
    let gs = k.s.groupshared("gs", i32, 32);
    let acc = k.s.groupshared("acc", i32, 2);
    let acc64 = k.s.groupshared("acc64", i64, 1);
    let v = k.input(0);
    let zero = k.s.m.i32c(0);
    let one = k.s.m.i32c(1);
    // Lane 0 clears the accumulators.
    let is0 = k.s.m.cmp(Pred::Eq, k.lane, zero);
    let init = k.s.m.new_block();
    let join = k.s.m.new_block();
    k.s.m.cond_br(is0, init, join);
    k.s.m.set_block(init);
    k.s.gs_store(acc, zero, zero);
    k.s.gs_store(acc, one, zero);
    let z64 = k.s.m.const_int(i64, 0);
    k.s.gs_store(acc64, zero, z64);
    k.s.m.br(join);
    k.s.m.set_block(join);
    k.s.gs_store(gs, k.lane, v);
    k.s.barrier(barrier::GROUP_SYNC);
    let m31 = k.s.m.i32c(31);
    let rev = k.s.m.bin(BinOp::Sub, m31, k.lane);
    let other = k.s.gs_load(gs, rev);
    k.s.gs_atomic(RmwOp::Add, acc, zero, v);
    k.s.gs_atomic(RmwOp::UMax, acc, one, v);
    let v64 = k.s.m.cast(CastOp::ZExt, v, i64);
    k.s.gs_atomic(RmwOp::Add, acc64, zero, v64);
    // cmpxchg: only the lane that sees 0 in gs[0] after barrier... use a
    // deterministic one: every lane tries gs[lane] == v → v + 1.
    let vp1 = k.s.m.bin(BinOp::Add, v, one);
    let old = k.s.gs_cmpxchg(gs, k.lane, v, vp1);
    k.s.barrier(barrier::GROUP_SYNC);
    let sum = k.s.gs_load(acc, zero);
    let max = k.s.gs_load(acc, one);
    let s64 = k.s.gs_load(acc64, zero);
    let s64hi = { let c = k.s.m.const_int(i64, 32); k.s.m.bin(BinOp::LShr, s64, c) };
    let s64hi = k.s.m.cast(CastOp::Trunc, s64hi, i32);
    let after = k.s.gs_load(gs, k.lane);
    for (i, x) in [other, sum, max, s64hi, old, after].into_iter().enumerate() {
        k.out(i as u32, x);
    }
    let c = k.finish([32, 1, 1]);
    let vals: Vec<u32> = (0..32).map(|l| 0x9000_0000u32.wrapping_add(l * 0x0100_0001)).collect();
    let Some(r) = gpu(&c, &mem_of(&[vals.clone()]), &[], &[]) else { return };
    let sum: u32 = vals.iter().fold(0u32, |a, &b| a.wrapping_add(b));
    let s64: u64 = vals.iter().map(|&v| v as u64).sum();
    for l in 0..32 {
        assert_eq!(outv(&r, 0)[l], vals[31 - l]);
        assert_eq!(outv(&r, 1)[l], sum);
        assert_eq!(outv(&r, 2)[l], *vals.iter().max().unwrap());
        assert_eq!(outv(&r, 3)[l], (s64 >> 32) as u32);
        assert_eq!(outv(&r, 4)[l], vals[l]);
        assert_eq!(outv(&r, 5)[l], vals[l] + 1);
    }
}

#[test]
fn raw_and_structured_buffer_atomics_run() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let sb = s.structured_uav(1, 0, 4);
    let mem = s.raw_uav(2, 0);
    let lane = s.flattened_thread_id_in_group();
    let i32 = s.m.i32();
    let i64 = s.m.i64();
    let off = { let c = s.m.i32c(2); s.m.bin(BinOp::Shl, lane, c) };
    let v = s.raw_load(mem, off, 1)[0];
    // mem[64..]: accumulators (dword 64 add, 65 or, 66 umax; 68..69 i64 add; 70 cmpxchg target).
    let c256 = s.m.i32c(256);
    let o_add = s.raw_atomic(mem, AtomicOp::Add, c256, v);
    let c260 = s.m.i32c(260);
    s.raw_atomic(mem, AtomicOp::Or, c260, v);
    let c264 = s.m.i32c(264);
    s.raw_atomic(mem, AtomicOp::UMax, c264, v);
    let c272 = s.m.i32c(272);
    let v64 = s.m.cast(CastOp::ZExt, v, i64);
    let sh = s.m.const_int(i64, 20);
    let v64 = s.m.bin(BinOp::Shl, v64, sh);
    s.raw_atomic(mem, AtomicOp::Add, c272, v64);
    // Lane-ordered cmpxchg: exactly one lane (the one whose value is 0) wins first.
    let c280 = s.m.i32c(280);
    let zero = s.m.i32c(0);
    let won = s.raw_cmpxchg(mem, c280, zero, v);
    // structured: sb[lane] = v; sb[32] += 1 for each lane (structured atomic).
    s.structured_store(sb, lane, 0, &[v]);
    let c32 = s.m.i32c(32);
    let one = s.m.i32c(1);
    s.structured_atomic(sb, AtomicOp::Add, c32, 0, one);
    let c33 = s.m.i32c(33);
    let o = s.m.bin(BinOp::Add, lane, c33);
    let _ = (o_add, i32);
    s.structured_store(sb, o, 0, &[won]);
    let c = s.finish([32, 1, 1], Some(32));
    check(&c);
    let dis = validate::disassemble(&c).unwrap();
    assert!(dis.contains("@dx.op.atomicBinOp.i64(i32 78"), "{dis}");
    let vals: Vec<u32> = (0..32).map(|l| (l * 37 + 11) & 0xff).collect();
    let mut m = vals.clone();
    m.resize(128, 0);
    m[70] = 0;
    let Some(r) = gpu(&c, &m, &[], &[]) else { return };
    let sum: u32 = vals.iter().sum();
    let or = vals.iter().fold(0, |a, b| a | b);
    let s64: u64 = vals.iter().map(|&v| (v as u64) << 20).sum();
    assert_eq!(r.mem[64], sum);
    assert_eq!(r.mem[65], or);
    assert_eq!(r.mem[66], *vals.iter().max().unwrap());
    assert_eq!(r.mem[68] as u64 | ((r.mem[69] as u64) << 32), s64);
    assert!(vals.contains(&r.mem[70]));
    assert_eq!(&r.raw[..32], &vals[..]);
    assert_eq!(r.raw[32], 32);
    // every lane but the first swapper sees the winner's value
    // No input is 0: exactly one lane swapped (saw 0), the others saw the winner.
    let olds = &r.raw[33..65];
    assert_eq!(olds.iter().filter(|&&o| o == 0).count(), 1);
    assert!(olds.iter().all(|&o| o == 0 || o == r.mem[70]));
}

#[test]
fn typed_buffer_ops_validate() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let tb = s.typed_uav(3, 0, CompType::U32);
    let lane = s.flattened_thread_id_in_group();
    let v = s.buffer_load(tb, lane);
    let one = s.m.i32c(1);
    let x = s.m.bin(BinOp::Add, v[0], one);
    let n = s.m.bin(BinOp::Add, lane, one);
    s.buffer_store(tb, n, &[x, x, x, x]);
    s.typed_atomic(tb, AtomicOp::Add, lane, one);
    let i64 = s.m.i64();
    let tb64 = {
        let idx = s.m.i32c(3);
        s.heap_typed_buffer(idx, true, CompType::U64, false)
    };
    let big = s.m.const_int(i64, 1 << 40);
    s.typed_atomic(tb64, AtomicOp::Or, lane, big);
    let d = s.get_dimensions(tb.handle, None);
    let n = s.m.extract(d, 0);
    s.buffer_store(tb, n, &[x, x, x, x]);
    let c = s.finish([32, 1, 1], None);
    check(&c);
    let dis = validate::disassemble(&c).unwrap();
    for needle in ["@dx.op.bufferLoad.i32(i32 68", "@dx.op.bufferStore.i32(i32 69", "@dx.op.atomicBinOp.i64(i32 78", "RWTypedBuffer<U32>"] {
        assert!(dis.contains(needle), "missing {needle}:\n{dis}");
    }
}

/// Typed buffers on the GPU: heap 0 `Buffer<uint>` (SRV, R32_UINT, 32
/// elements), heap 3 `RWBuffer<uint>` (UAV, R32_UINT, 64 elements).
/// `uav[l] = srv[l] * 3 + uav[l]`; `InterlockedAdd(uav[32 + (l & 3)], l)`
/// returns the old value; `res0 = srv[31 - l]`.
#[test]
fn typed_buffer_ops_run() {
    use windows::Win32::Graphics::Dxgi::Common::*;
    let mut k = kernel();
    let (i0, i3) = (k.s.m.i32c(0), k.s.m.i32c(3));
    let srv = k.s.heap_typed_buffer(i0, false, CompType::U32, false);
    let uav = k.s.heap_typed_buffer(i3, true, CompType::U32, false);
    let lane = k.lane;
    let a = k.s.buffer_load(srv, lane)[0];
    let b = k.s.buffer_load(uav, lane)[0];
    let three = k.s.m.i32c(3);
    let a3 = k.s.m.bin(BinOp::Mul, a, three);
    let v = k.s.m.bin(BinOp::Add, a3, b);
    k.s.buffer_store(uav, lane, &[v, v, v, v]);
    let c32 = k.s.m.i32c(32);
    let q = k.s.m.bin(BinOp::And, lane, three);
    let slot = k.s.m.bin(BinOp::Add, c32, q);
    let old = k.s.typed_atomic(uav, AtomicOp::Add, slot, lane);
    let c31 = k.s.m.i32c(31);
    let r = k.s.m.bin(BinOp::Sub, c31, lane);
    let rv = k.s.buffer_load(srv, r)[0];
    k.out(0, rv);
    k.out(1, old);
    let c = k.finish([32, 1, 1]);
    let srv_data: Vec<u32> = (0..32).map(|i| i * 7 + 1).collect();
    let uav_data: Vec<u32> = (0..64).map(|i| 1000 + i).collect();
    let buf = |data: &[u32], uav: bool| runner::TestTexture {
        dim: runner::TexDim::Buffer,
        format: DXGI_FORMAT_R32_UINT,
        width: data.len() as u32,
        height: 1,
        depth: 1,
        texel_bytes: 4,
        data: data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        uav,
    };
    let Some(r) = gpu(&c, &[], &[buf(&srv_data, false), buf(&uav_data, true)], &[]) else { return };
    let want: Vec<u32> = (0..32).map(|l| srv_data[31 - l]).collect();
    assert_eq!(outv(&r, 0), &want[..]);
    let out: Vec<u32> = r.textures[0].chunks(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    for l in 0..32 {
        assert_eq!(out[l], srv_data[l] * 3 + uav_data[l], "uav[{l}]");
    }
    for q in 0..4u32 {
        let sum: u32 = (0..32).filter(|l| l & 3 == q).sum();
        assert_eq!(out[32 + q as usize], uav_data[32 + q as usize] + sum, "atomic slot {q}");
    }
    // Old values: each lane saw a partial sum of its slot.
    for l in 0..32usize {
        let o = outv(&r, 1)[l];
        let base = uav_data[32 + (l & 3)];
        let total: u32 = (0..32u32).filter(|x| (x & 3) as usize == l & 3).sum();
        assert!(o >= base && o - base <= total, "old value of lane {l}: {o}");
    }
}

// --------------------------------------------------------------- textures

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

fn tex_data(w: u32, h: u32, d: u32) -> Vec<f32> {
    (0..w * h * d).map(|i| i as f32 * 0.25 + 1.0).collect()
}

fn ftex(dim: runner::TexDim, w: u32, h: u32, d: u32) -> runner::TestTexture {
    runner::TestTexture {
        dim,
        format: DXGI_FORMAT_R32_FLOAT,
        width: w,
        height: h,
        depth: d,
        texel_bytes: 4,
        data: tex_data(w, h, d).iter().flat_map(|f| f.to_le_bytes()).collect(),
        uav: false,
    }
}

/// Textures: t0 2D 8×8, t1 2D-array 8×8×2, t2 3D 4×4×4, t3 cube 4×4, t4 UAV
/// 2D R32_UINT 8×4 (heap 9); samplers: s0 point clamp, s1 point clamp
/// comparison LESS_EQUAL.
fn textures() -> (Vec<runner::TestTexture>, Vec<runner::TestSampler>) {
    let mut u = ftex(runner::TexDim::D2, 8, 4, 1);
    u.format = DXGI_FORMAT_R32_UINT;
    u.uav = true;
    u.data = vec![0; 8 * 4 * 4];
    (
        vec![ftex(runner::TexDim::D2, 8, 8, 1), ftex(runner::TexDim::D2Array, 8, 8, 2), ftex(runner::TexDim::D3, 4, 4, 4), ftex(runner::TexDim::Cube, 4, 4, 6), u],
        vec![
            runner::TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None },
            runner::TestSampler { filter: D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: Some(D3D12_COMPARISON_FUNC_LESS_EQUAL) },
        ],
    )
}

#[test]
fn texture_ops_run() {
    let mut k = kernel();
    let x = k.s.thread_id_in_group(0);
    let y = k.s.thread_id_in_group(1);
    let f32 = k.s.m.f32();
    let i32 = k.s.m.i32();
    let idx = |k: &mut K, i: u32| k.s.m.i32c(i);
    let i0 = idx(&mut k, 0);
    let t2 = k.s.heap_texture(i0, TexKind::D2, false, CompType::F32, 1, false);
    let i2 = idx(&mut k, 2);
    let ta = k.s.heap_texture(i2, TexKind::D2Array, false, CompType::F32, 1, false);
    let i4 = idx(&mut k, 4);
    let t3 = k.s.heap_texture(i4, TexKind::D3, false, CompType::F32, 1, false);
    let i6 = idx(&mut k, 6);
    let tc = k.s.heap_texture(i6, TexKind::Cube, false, CompType::F32, 1, false);
    let i9 = idx(&mut k, 9);
    let uv = k.s.heap_texture(i9, TexKind::D2, true, CompType::U32, 1, false);
    let s0 = k.s.heap_sampler(i0, false, false);
    let one = idx(&mut k, 1);
    let sc = k.s.heap_sampler(one, true, false);
    // uv = (x + 0.5) / 8, (y + 0.5) / 8 → texel (x, y) of an 8×8 texture
    let fx = k.s.m.cast(CastOp::UIToFP, x, f32);
    let fy = k.s.m.cast(CastOp::UIToFP, y, f32);
    let half = k.s.m.f32c(0.5f32.to_bits());
    let eighth = k.s.m.f32c(0.125f32.to_bits());
    let u = k.s.m.fbin(BinOp::Add, fx, half, 0);
    let u = k.s.m.fbin(BinOp::Mul, u, eighth, 0);
    let v = k.s.m.fbin(BinOp::Add, fy, half, 0);
    let v = k.s.m.fbin(BinOp::Mul, v, eighth, 0);
    let zero = k.s.m.i32c(0);
    let fzero = k.s.m.f32c(0);
    let fone = k.s.m.f32c(1.0f32.to_bits());
    let ld = k.s.texture_load(t2, Some(zero), &[x, y], &[]);
    let ldo = k.s.texture_load(t2, Some(zero), &[x, y], &[1, 0]);
    let smp = k.s.sample(t2, s0, &[u, v], &[], None);
    let sl = k.s.sample_level(t2, s0, &[u, v], &[], fzero);
    let sg = k.s.sample_grad(t2, s0, &[u, v], &[], &[fzero, fzero], &[fzero, fzero], None);
    let sb = k.s.sample_bias(t2, s0, &[u, v], &[], fzero, None);
    let g = k.s.gather(t2, s0, &[u, v], &[], 0);
    // ref value 10: texels ≤ 10 compare → 1.0 (LESS_EQUAL: ref <= texel)
    let refv = k.s.m.f32c(10.0f32.to_bits());
    let cmp = k.s.sample_cmp(t2, sc, &[u, v], &[], refv, None);
    let cmpz = k.s.sample_cmp_level_zero(t2, sc, &[u, v], &[], refv);
    let gc = k.s.gather_cmp(t2, sc, &[u, v], &[], 0, refv);
    let sa = k.s.sample_level(ta, s0, &[u, v, fone], &[], fzero);
    let s3 = k.s.sample_level(t3, s0, &[u, v, half], &[], fzero);
    let cube_dir = [fone, fzero, fzero];
    let scube = k.s.sample_level(tc, s0, &cube_dir, &[], fzero);
    let lod = k.s.calculate_lod(t2, s0, &[u, v], true);
    let d = k.s.get_dimensions(t2.handle, Some(zero));
    let dims = [0, 1, 2, 3].map(|i| if i == 2 { d } else { k.s.m.extract(d, i) });
    let d = k.s.get_dimensions(uv.handle, None);
    let udims = [0, 1].map(|i| k.s.m.extract(d, i));
    let bits = k.s.m.bitcast(ld[0], i32);
    k.s.texture_store(uv, &[x, y], [bits, bits, bits, bits]);
    let outs = [ld[0], ldo[0], smp[0], sl[0], sg[0], sb[0], g[0], g[1], g[2], g[3], cmp[0], cmpz[0], gc[0], sa[0], s3[0], scube[0], lod, dims[0], dims[1], dims[3], udims[0], udims[1]];
    for (i, &o) in outs.iter().enumerate() {
        k.out(i as u32, o);
    }
    let c = k.finish([8, 4, 1]);
    let (tex, smp) = textures();
    let Some(r) = gpu(&c, &[0; 64], &tex, &smp) else { return };
    let t2d = tex_data(8, 8, 1);
    let at = |x: usize, y: usize| t2d[y.min(7) * 8 + x.min(7)];
    let lane_xy = |l: usize| (l % 8, l / 8);
    let f = |i: usize, l: usize| f32::from_bits(outv(&r, i)[l]);
    for l in 0..32 {
        let (x, y) = lane_xy(l);
        let t = at(x, y);
        assert_eq!(f(0, l), t, "load lane {l}");
        assert_eq!(f(1, l), if x < 7 { at(x + 1, y) } else { 0.0 }, "load+offset lane {l} (out of bounds → 0)");
        for i in 2..6 {
            assert_eq!(f(i, l), t, "sample variant {i} lane {l}");
        }
        // gather at the texel centre: lower-right quad (x..x+1, y..y+1) in
        // (-u+v, +u+v, +u-v, -u-v) order — point at centre picks (x-0.5..): the
        // 2×2 footprint around (x+0.5, y+0.5) is texels x..x+1 with coords floored at x+0.5-0.5.
        let gw = [at(x, y + 1), at(x + 1, y + 1), at(x + 1, y), at(x, y)];
        for j in 0..4 {
            assert_eq!(f(6 + j, l), gw[j], "gather[{j}] lane {l}");
        }
        let c1 = |v: f32| if 10.0 <= v { 1.0 } else { 0.0 };
        assert_eq!(f(10, l), c1(t), "samplecmp lane {l}");
        assert_eq!(f(11, l), c1(t), "samplecmplevelzero lane {l}");
        assert_eq!(f(12, l), c1(gw[0]), "gathercmp lane {l}");
        assert_eq!(f(13, l), tex_data(8, 8, 2)[64 + y * 8 + x], "array slice 1 lane {l}");
        let (x3, y3) = (x * 4 / 8, y * 4 / 8);
        assert_eq!(f(14, l), tex_data(4, 4, 4)[2 * 16 + y3 * 4 + x3], "3D lane {l}");
        // +X face (face 0) centre → texel (2, 2) or nearby of face 0.
        let fc = f(15, l);
        assert!((0..16).any(|i| tex_data(4, 4, 6)[i] == fc), "cube +X lane {l}: {fc}");
        assert_eq!(f(16, l), 0.0, "lod lane {l}");
        assert_eq!(&[outv(&r, 17)[l], outv(&r, 18)[l], outv(&r, 19)[l], outv(&r, 20)[l], outv(&r, 21)[l]], &[8, 8, 1, 8, 4], "dims lane {l}");
    }
    let stored: Vec<u32> = r.textures[0].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for l in 0..32 {
        let (x, y) = lane_xy(l);
        assert_eq!(stored[y * 8 + x], at(x, y).to_bits(), "texture store {x},{y}");
    }
}

#[test]
fn texture_variants_validate() {
    // Offsets, clamp (tiled resources), typed UAV loads of float4, texture
    // atomics, unclamped LOD, 3D / array UAV stores.
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let x = s.thread_id_in_group(0);
    let f = s.m.f32c(0.5f32.to_bits());
    let i0 = s.m.i32c(0);
    let t = s.heap_texture(i0, TexKind::D2, false, CompType::F32, 4, false);
    let smp = s.heap_sampler(i0, false, false);
    let r1 = s.sample(t, smp, &[f, f], &[1, -1], Some(f));
    let r2 = s.sample_bias(t, smp, &[f, f], &[0, 1], f, Some(f));
    let _ = s.calculate_lod(t, smp, &[f, f], false);
    let i9 = s.m.i32c(9);
    let u4 = s.heap_texture(i9, TexKind::D2, true, CompType::F32, 4, false);
    s.typed_uav_load_additional_formats();
    let l = s.texture_load(u4, None, &[x, x], &[]);
    let a = s.m.fbin(BinOp::Add, l[1], r1[2], 0);
    let a = s.m.fbin(BinOp::Add, a, r2[3], 0);
    s.texture_store(u4, &[x, x], [a, a, a, a]);
    let i11 = s.m.i32c(11);
    let ua = s.heap_texture(i11, TexKind::D2Array, true, CompType::U32, 1, false);
    let one = s.m.i32c(1);
    let old = s.atomic_binop(ua.handle, AtomicOp::Add, &[x, x, one], one);
    let cx = s.atomic_cmpxchg(ua.handle, &[x, x, i0], old, one);
    s.texture_store(ua, &[x, x, one], [cx, cx, cx, cx]);
    let i13 = s.m.i32c(13);
    let u3 = s.heap_texture(i13, TexKind::D3, true, CompType::F32, 1, false);
    s.texture_store(u3, &[x, x, one], [a, a, a, a]);
    let c = s.finish([8, 4, 1], None);
    check(&c);
}

// ------------------------------------------------------------------- wave

/// `WaveGetLaneIndex` is declared as DXC 1.9 declares it.
#[test]
fn wave_lane_index_is_declared_readonly() {
    let mut k = kernel();
    let lane = k.s.wave_lane_index();
    k.out(0, lane);
    let dis = validate::disassemble(&k.finish([32, 1, 1])).expect("disassembly");
    let at = dis.lines().position(|l| l.starts_with("declare i32 @dx.op.waveGetLaneIndex(")).expect("the declaration");
    assert_eq!(dis.lines().nth(at - 1), Some("; Function Attrs: nounwind readonly"), "{dis}");
}

#[test]
fn wave_ops_match_cpu() {
    let mut k = kernel();
    let u = k.input(0);
    let fv = k.input_f(1);
    let i64 = k.s.m.i64();
    let three = k.s.m.i32c(3);
    let gt3 = k.s.m.cmp(Pred::UGt, u, three);
    let mut o = vec![
        k.s.wave_is_first_lane(),
        k.s.wave_lane_count(),
        k.s.wave_all_true(gt3),
        k.s.wave_any_true(gt3),
        k.s.wave_all_equal(u),
    ];
    for (op, uns) in [(WaveOp::Sum, true), (WaveOp::Product, true), (WaveOp::Min, true), (WaveOp::Max, true), (WaveOp::Min, false), (WaveOp::Max, false)] {
        o.push(k.s.wave_active_op(u, op, uns));
    }
    for b in [WaveBit::And, WaveBit::Or, WaveBit::Xor] {
        o.push(k.s.wave_active_bit(u, b));
    }
    o.push(k.s.wave_prefix_op(u, WaveOp::Sum, true));
    o.push(k.s.wave_prefix_op(u, WaveOp::Product, false));
    o.push(k.s.wave_all_bit_count(gt3));
    o.push(k.s.wave_prefix_bit_count(gt3));
    let two = k.s.m.i32c(2);
    o.push(k.s.quad_read_lane_at(u, two));
    for d in [QuadDir::ReadAcrossX, QuadDir::ReadAcrossY, QuadDir::ReadAcrossDiagonal] {
        o.push(k.s.quad_op(u, d));
    }
    let m7 = k.s.m.i32c(7);
    let key = k.s.m.bin(BinOp::And, u, m7);
    let mask = k.s.wave_match(key);
    o.push(mask[0]);
    o.push(k.s.wave_multi_prefix_op(u, mask, multi_prefix::SUM, true));
    o.push(k.s.wave_multi_prefix_op(u, mask, multi_prefix::XOR, false));
    o.push(k.s.wave_multi_prefix_bit_count(gt3, mask));
    let u64v = k.s.m.cast(CastOp::ZExt, u, i64);
    let sh = k.s.m.const_int(i64, 8);
    let u64v = k.s.m.bin(BinOp::Shl, u64v, sh);
    let s64 = k.s.wave_active_op(u64v, WaveOp::Sum, true);
    let s64 = { let c = k.s.m.const_int(i64, 32); k.s.m.bin(BinOp::LShr, s64, c) };
    let i32 = k.s.m.i32();
    o.push(k.s.m.cast(CastOp::Trunc, s64, i32));
    o.push(k.s.wave_active_op(fv, WaveOp::Sum, false));
    o.push(k.s.wave_active_op(fv, WaveOp::Max, false));
    o.push(k.s.wave_prefix_op(fv, WaveOp::Sum, false));
    for (i, &v) in o.iter().enumerate() {
        k.out(i as u32, v);
    }
    let c = k.finish([32, 1, 1]);
    let us: Vec<u32> = (0..32u32).map(|l| l.wrapping_mul(2_654_435_761u32) >> 24 | 1).collect();
    let fs: Vec<f32> = (0..32).map(|l| l as f32 * 0.5 + 1.0).collect();
    let Some(r) = gpu(&c, &mem_of(&[us.clone(), fs.iter().map(|f| f.to_bits()).collect()]), &[], &[]) else { return };
    let sum = us.iter().fold(0u32, |a, &b| a.wrapping_add(b));
    let prod = us.iter().fold(1u32, |a, &b| a.wrapping_mul(b));
    let all = us.iter().all(|&v| v > 3) as u32;
    let any = us.iter().any(|&v| v > 3) as u32;
    let cnt = us.iter().filter(|&&v| v > 3).count() as u32;
    let s64: u64 = us.iter().map(|&v| (v as u64) << 8).sum();
    for l in 0..32 {
        let q = l & !3;
        let key = us[l] & 7;
        let grp: Vec<usize> = (0..32).filter(|&j| us[j] & 7 == key).collect();
        let mmask = grp.iter().fold(0u32, |a, &j| a | (1 << j));
        let before: Vec<usize> = grp.iter().copied().filter(|&j| j < l).collect();
        let want = [
            (l == 0) as u32,
            32,
            all,
            any,
            us.iter().all(|&v| v == us[0]) as u32,
            sum,
            prod,
            *us.iter().min().unwrap(),
            *us.iter().max().unwrap(),
            *us.iter().min().unwrap(),
            *us.iter().max().unwrap(),
            us.iter().fold(!0, |a, &b| a & b),
            us.iter().fold(0, |a, &b| a | b),
            us.iter().fold(0, |a, &b| a ^ b),
            us[..l].iter().fold(0u32, |a, &b| a.wrapping_add(b)),
            us[..l].iter().fold(1u32, |a, &b| a.wrapping_mul(b)),
            cnt,
            us[..l].iter().filter(|&&v| v > 3).count() as u32,
            us[q + 2],
            us[l ^ 1],
            us[l ^ 2],
            us[l ^ 3],
            mmask,
            before.iter().fold(0u32, |a, &j| a.wrapping_add(us[j])),
            before.iter().fold(0u32, |a, &j| a ^ us[j]),
            before.iter().filter(|&&j| us[j] > 3).count() as u32,
            (s64 >> 32) as u32,
        ];
        for (i, &w) in want.iter().enumerate() {
            assert_eq!(outv(&r, i)[l], w, "wave op #{i} lane {l}");
        }
        let n = want.len();
        let fsum: f32 = fs.iter().sum();
        assert!((f32::from_bits(outv(&r, n)[l]) - fsum).abs() < 1e-3, "float sum");
        assert_eq!(f32::from_bits(outv(&r, n + 1)[l]), 16.5, "float max");
        let fp: f32 = fs[..l].iter().sum();
        assert!((f32::from_bits(outv(&r, n + 2)[l]) - fp).abs() < 1e-3, "float prefix lane {l}");
    }
}

// ------------------------------------------------------------ derivatives

#[test]
fn derivatives_in_compute_run() {
    // SM 6.6 compute derivatives: with numthreads (8, 4, 1) quads are 2×2
    // blocks in thread-ID space.
    let mut k = kernel();
    let x = k.s.thread_id_in_group(0);
    let y = k.s.thread_id_in_group(1);
    let f32 = k.s.m.f32();
    let fx = k.s.m.cast(CastOp::UIToFP, x, f32);
    let fy = k.s.m.cast(CastOp::UIToFP, y, f32);
    // f = x*x + 10*y*x
    let xx = k.s.m.fbin(BinOp::Mul, fx, fx, 0);
    let ten = k.s.m.f32c(10.0f32.to_bits());
    let ty = k.s.m.fbin(BinOp::Mul, fy, ten, 0);
    let txy = k.s.m.fbin(BinOp::Mul, ty, fx, 0);
    let f = k.s.m.fbin(BinOp::Add, xx, txy, 0);
    for (i, d) in [Deriv::CoarseX, Deriv::CoarseY, Deriv::FineX, Deriv::FineY].into_iter().enumerate() {
        let v = k.s.deriv(d, f);
        k.out(i as u32, v);
    }
    let c = k.finish([8, 4, 1]);
    let Some(r) = gpu(&c, &[0; 64], &[], &[]) else { return };
    let fv = |x: usize, y: usize| (x * x + 10 * y * x) as f32;
    for l in 0..32 {
        let (x, y) = (l % 8, l / 8);
        let (qx, qy) = (x & !1, y & !1);
        let fine_x = fv(qx + 1, y) - fv(qx, y);
        let fine_y = fv(x, qy + 1) - fv(x, qy);
        let got = |i: usize| f32::from_bits(outv(&r, i)[l]);
        assert_eq!(got(2), fine_x, "ddx_fine lane {l}");
        assert_eq!(got(3), fine_y, "ddy_fine lane {l}");
        // coarse: any row / column of the quad (implementation-defined)
        assert!([fv(qx + 1, qy) - fv(qx, qy), fv(qx + 1, qy + 1) - fv(qx, qy + 1)].contains(&got(0)), "ddx_coarse lane {l}");
        assert!([fv(qx, qy + 1) - fv(qx, qy), fv(qx + 1, qy + 1) - fv(qx + 1, qy)].contains(&got(1)), "ddy_coarse lane {l}");
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as i32, (h & 0x3ff) as u32);
    let mag = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 => if m == 0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15),
    };
    if s == 1 { -mag } else { mag }
}

/// f32 → f16 bits, rounding to nearest even or towards zero (NaN not handled).
fn f32_to_f16(x: f32, rtz: bool) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = x.abs();
    if a.is_infinite() || a >= 65520.0 || (!rtz && a >= 65520.0) {
        return sign | if rtz && a.is_finite() { 0x7bff } else { 0x7c00 };
    }
    // Scale to units of the result's ULP and round half to even.
    let e = if a < 2f32.powi(-14) { -14 } else { a.log2().floor() as i32 };
    let ulp = 2f64.powi(e - 10);
    let q = if rtz { (a as f64 / ulp).floor() } else { (a as f64 / ulp).round_ties_even() } as u32;
    let (e, q) = if q >= 2048 { (e + 1, q / 2) } else { (e, q) };
    if q < 1024 {
        sign | q as u16 // subnormal (e == -14)
    } else {
        sign | (((e + 15) as u16) << 10) | (q as u16 & 0x3ff)
    }
}

#[test]
fn double_unary_legality_matches_dxil_tables() {
    if !validate::validator_supported() {
        return validate::skip_validation();
    }
    // Which float unary ops take an f64 overload (DXIL.rst: FAbs, Saturate only).
    let all = [
        FUnary::FAbs, FUnary::Saturate, FUnary::Cos, FUnary::Sin, FUnary::Exp, FUnary::Frc, FUnary::Log, FUnary::Sqrt, FUnary::Rsqrt, FUnary::RoundNe,
        FUnary::RoundNi, FUnary::RoundPi, FUnary::RoundZ,
    ];
    let mut legal = Vec::new();
    for op in all {
        let mut k = kernel();
        let lo = k.input(0);
        let d = k.s.make_double(lo, lo);
        let r = k.s.funary(op, d);
        let (a, _) = k.s.split_double(r);
        k.out(0, a);
        if validate::validate(&k.s.finish([32, 1, 1], Some(32))).is_ok() {
            legal.push(op);
        }
    }
    assert_eq!(legal, DOUBLE_UNARY);
}

#[test]
fn int64_bit_ops_match_cpu() {
    let mut k = kernel();
    let lo = k.input(0);
    let hi = k.input(1);
    let i64 = k.s.m.i64();
    let i32 = k.s.m.i32();
    let l64 = k.s.m.cast(CastOp::ZExt, lo, i64);
    let h64 = k.s.m.cast(CastOp::ZExt, hi, i64);
    let sh = k.s.m.const_int(i64, 32);
    let h64 = k.s.m.bin(BinOp::Shl, h64, sh);
    let v = k.s.m.bin(BinOp::Or, h64, l64);
    let mut outs = vec![];
    for op in [BitCount::Countbits, BitCount::FirstbitLo, BitCount::FirstbitHi] {
        outs.push(k.s.bit_count(op, v));
    }
    let rev = k.s.bfrev(v);
    outs.push(k.s.m.cast(CastOp::Trunc, rev, i32));
    let mx = k.s.binary(Binary::UMax, v, h64);
    let mx = { let c = k.s.m.const_int(i64, 32); k.s.m.bin(BinOp::LShr, mx, c) };
    outs.push(k.s.m.cast(CastOp::Trunc, mx, i32));
    for (i, &o) in outs.iter().enumerate() {
        k.out(i as u32, o);
    }
    let c = k.finish([32, 1, 1]);
    let his: Vec<u32> = UVALS.iter().map(|v| v.rotate_right(9) & 0x00ff_ffff).collect();
    let Some(r) = gpu(&c, &mem_of(&[UVALS.to_vec(), his.clone()]), &[], &[]) else { return };
    for l in 0..32 {
        let v = UVALS[l] as u64 | (his[l] as u64) << 32;
        let want = [
            v.count_ones(),
            if v == 0 { u32::MAX } else { v.trailing_zeros() },
            if v == 0 { u32::MAX } else { v.leading_zeros() },
            v.reverse_bits() as u32,
            his[l],
        ];
        for (i, &w) in want.iter().enumerate() {
            assert_eq!(outv(&r, i)[l], w, "i64 op #{i} of {v:#x}");
        }
    }
}

#[test]
fn uav_array_3d_stores_loads_and_texture_atomics_run() {
    // u: heap 3 = RWTexture2DArray<uint> 4×4×2, heap 5 = RWTexture3D<float> 4×4×2.
    let mut k = kernel();
    let i1 = k.s.m.i32c(3);
    let i3 = k.s.m.i32c(5);
    let ua = k.s.heap_texture(i1, TexKind::D2Array, true, CompType::U32, 1, false);
    let u3 = k.s.heap_texture(i3, TexKind::D3, true, CompType::F32, 1, false);
    let lane = k.lane;
    let c3 = k.s.m.i32c(3);
    let c2 = k.s.m.i32c(2);
    let x = k.s.m.bin(BinOp::And, lane, c3);
    let y = k.s.m.bin(BinOp::LShr, lane, c2);
    let y = k.s.m.bin(BinOp::And, y, c3);
    let z = { let c = k.s.m.i32c(4); k.s.m.bin(BinOp::LShr, lane, c) };
    let old = k.s.texture_load(ua, None, &[x, y, z], &[])[0];
    k.s.texture_store(ua, &[x, y, z], [lane, lane, lane, lane]);
    let f32 = k.s.m.f32();
    let fl = k.s.m.cast(CastOp::UIToFP, lane, f32);
    k.s.texture_store(u3, &[x, y, z], [fl, fl, fl, fl]);
    k.s.barrier(barrier::UAV_FENCE_THREAD_GROUP | barrier::SYNC_THREAD_GROUP);
    // Every lane adds 1 to texel (0, 0, 1) and swaps texel (1, 0, 1) if it holds 17.
    let zero = k.s.m.i32c(0);
    let one = k.s.m.i32c(1);
    let before = k.s.atomic_binop(ua.handle, AtomicOp::Add, &[zero, zero, one], one);
    let c17 = k.s.m.i32c(17);
    let c99 = k.s.m.i32c(99);
    let cx = k.s.atomic_cmpxchg(ua.handle, &[one, zero, one], c17, c99);
    k.out(0, old);
    k.out(1, before);
    k.out(2, cx);
    let c = k.finish([32, 1, 1]);
    let mk = |dim, format, data: Vec<u8>| runner::TestTexture { dim, format, width: 4, height: 4, depth: 2, texel_bytes: 4, data, uav: true };
    let init: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
    let tex = [
        ftex(runner::TexDim::D2, 1, 1, 1),
        mk(runner::TexDim::D2Array, DXGI_FORMAT_R32_UINT, init.iter().flat_map(|v| v.to_le_bytes()).collect()),
        mk(runner::TexDim::D3, DXGI_FORMAT_R32_FLOAT, vec![0; 128]),
    ];
    let Some(r) = gpu(&c, &[0; 64], &tex, &[]) else { return };
    assert_eq!(outv(&r, 0), &init[..], "loads before stores");
    let mut before: Vec<u32> = outv(&r, 1).to_vec();
    before.sort();
    assert_eq!(before, (16..48).collect::<Vec<u32>>(), "atomic add sees 16, 17, … in some order");
    let arr: Vec<u32> = r.textures[0].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(arr[16], 16 + 32);
    assert_eq!(arr[17], 99);
    assert_eq!(outv(&r, 2).iter().filter(|&&v| v == 17).count(), 1);
    let vol: Vec<f32> = r.textures[1].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(vol, (0..32).map(|i| i as f32).collect::<Vec<_>>());
}

fn native_16bit_supported() -> bool {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory6, DXGI_CREATE_FACTORY_FLAGS};
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) else { return false };
        let sel = crate::gpu::adapter::Selector::from_env_or("auto");
        let a: Option<IDXGIAdapter1> = if sel == crate::gpu::adapter::Selector::Warp {
            factory.EnumWarpAdapter().ok()
        } else {
            crate::gpu::adapter::candidates(&factory, &sel).into_iter().next().map(|(a, _)| a)
        };
        let Some(a) = a else { return false };
        let Ok(dev) = crate::gpu::agility::create_device(&a, D3D_FEATURE_LEVEL_12_0) else { return false };
        let mut o4 = D3D12_FEATURE_DATA_D3D12_OPTIONS4::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &mut o4 as *mut _ as _, std::mem::size_of_val(&o4) as u32).is_ok()
            && o4.Native16BitShaderOpsSupported.as_bool()
    }
}

#[test]
fn half_precision_overloads_run() {
    // Native 16-bit floats (SM 6.2+ `-enable-16bit-types`).
    let mut k = kernel();
    k.s.require(super::shader::Feature::NativeLowPrecision);
    let a = k.input_f(0);
    let b = k.input_f(1);
    let f16 = k.s.m.f16();
    let f32 = k.s.m.f32();
    let ha = k.s.m.cast(CastOp::FPTrunc, a, f16);
    let hb = k.s.m.cast(CastOp::FPTrunc, b, f16);
    let mx = k.s.binary(Binary::FMax, ha, hb);
    let fl = k.s.funary(FUnary::RoundNi, ha);
    let ab = k.s.funary(FUnary::FAbs, hb);
    let mad = k.s.tertiary(Tertiary::FMad, ha, hb, ab);
    // (f16 isSpecialFloat needs SM 6.9.)
    let nan = k.s.fclass(FClass::IsNaN, a);
    // Reduce `ha`, not `ab`: AMD's driver (RDNA4, 32.0.3xxxx) drops an
    // `FAbs.f16` feeding `WaveActiveMax.f16` (it returns max(y), 5.0, for
    // max(|y|), 10.5); DXC's own DXIL for the HLSL does the same. `ab` is
    // checked on its own above.
    let sum = k.s.wave_active_op(ha, WaveOp::Max, false);
    for (i, v) in [mx, fl, ab, mad, sum].into_iter().enumerate() {
        let e = k.s.m.cast(CastOp::FPExt, v, f32);
        k.out(i as u32, e);
    }
    k.out(5, nan);
    let c = k.finish([32, 1, 1]);
    match runner::wave32_supported() {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no wave32"),
        Err(e) => return runner::skip_test(&e),
    }
    if !native_16bit_supported() {
        runner::skip_test("no native 16-bit ops");
        return;
    }
    let xs: Vec<f32> = (0..32).map(|l| l as f32 * 0.75 - 9.0).collect();
    let ys: Vec<f32> = (0..32).map(|l| 5.0 - l as f32 * 0.5).collect();
    let r = runner::run_with(&c, &runner::KernelState::default(), &mem_of(&[xs.iter().map(|f| f.to_bits()).collect(), ys.iter().map(|f| f.to_bits()).collect()]), &[], &[]).expect("GPU run");
    let xmax = xs.iter().fold(f32::MIN, |m, &x| m.max(x));
    for l in 0..32 {
        let (x, y) = (xs[l], ys[l]); // all exactly representable in f16
        let got = |i: usize| f32::from_bits(outv(&r, i)[l]);
        assert_eq!(got(0), x.max(y));
        assert_eq!(got(1), x.floor());
        assert_eq!(got(2), y.abs());
        assert!((got(3) - (x * y + y.abs())).abs() <= 0.05, "half mad {x} {y}: {}", got(3));
        assert_eq!(got(4), xmax, "half wave max");
        assert_eq!(outv(&r, 5)[l], 0);
    }
}

/// Every literal `op(<opcode>, "<name>", .., FnAttrs::<X>)` in the backend
/// declares the attributes DXC uses for that opcode (table in
/// `shader::dxc_fn_attrs`, from DXC 1.9 `-Fc` listings).
#[test]
fn dx_op_attrs_match_dxc() {
    use super::module::FnAttrs;
    let srcs = [include_str!("ops.rs"), include_str!("graphics.rs"), include_str!("lower_image.rs"), include_str!("shader.rs")];
    let mut n = 0;
    for src in srcs {
        for (i, _) in src.match_indices("op(") {
            let rest = &src[i + 3..];
            let Some((num, rest)) = rest.split_once(", \"") else { continue };
            let Ok(op) = num.trim().parse::<u32>() else { continue };
            let Some(end) = rest.find(')') .and_then(|_| rest.find("FnAttrs::")) else { continue };
            if rest[..end].contains(';') {
                continue;
            }
            let a = &rest[end + 9..];
            let a: String = a.chars().take_while(|c| c.is_alphanumeric()).collect();
            let got = match a.as_str() {
                "ReadNone" => FnAttrs::ReadNone,
                "ReadOnly" => FnAttrs::ReadOnly,
                "NoUnwind" => FnAttrs::NoUnwind,
                "NoDuplicate" => FnAttrs::NoDuplicate,
                _ => FnAttrs::None,
            };
            let want = super::shader::dxc_fn_attrs(op).unwrap_or_else(|| panic!("opcode {op} missing from dxc_fn_attrs"));
            assert_eq!(got, want, "opcode {op}: {}", &rest[..rest.find('"').unwrap_or(0)]);
            n += 1;
        }
    }
    assert!(n > 50, "only {n} declarations found");
}