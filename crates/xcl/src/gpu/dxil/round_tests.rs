//! Directed f32 rounding in the direct translator (`lower_round`), on the GPU against a CPU
//! reference that rounds exact results (integer arithmetic with a sticky bit, independent of
//! the translator's f64 method).

use crate::gpu::rdna2::encode;
use crate::gpu::recomp::runner::{self, KernelState};

const V: u64 = 256;

fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported()) {
        (Ok(_), Ok(true)) => true,
        (a, _) => {
            runner::skip_test(if a.is_err() { "no GPU" } else { "no wave64" });
            false
        }
    }
}

// ------------------------------------------------------------------ reference

/// RDNA FP_ROUND: 0 nearest even, 1 +inf, 2 -inf, 3 zero.
#[derive(Clone, Copy, Debug)]
struct Mode {
    rm: u32,
    keep_in: bool,
    keep_out: bool,
}

/// f32 → (sign, m, e) with value m · 2^e (m < 2^24); None for inf / NaN.
fn parts(x: u32, keep_in: bool) -> Option<(bool, u128, i32)> {
    let s = x >> 31 == 1;
    let e = ((x >> 23) & 0xFF) as i32;
    let f = (x & 0x7F_FFFF) as u128;
    match e {
        255 => None,
        0 if !keep_in => Some((s, 0, 0)),
        0 => Some((s, f, -149)),
        _ => Some((s, f | 1 << 23, e - 150)),
    }
}

fn bitlen(v: u128) -> i32 {
    128 - v.leading_zeros() as i32
}

/// Round sign · (mag · 2^exp + sticky·ε) (0 < ε ≪ 2^exp) to f32 bits.
fn round(sign: bool, mag: u128, exp: i32, sticky: bool, m: Mode) -> u32 {
    let sg = (sign as u32) << 31;
    if mag == 0 && !sticky {
        return sg;
    }
    // a lone sticky: some value far below every f32
    let (mag, exp) = if mag == 0 { (1u128, -400) } else { (mag, exp) };
    let top = bitlen(mag) - 1 + exp; // value in [2^top, 2^(top+1))
    let mut lsb = if top < -126 { -149 } else { top - 23 };
    let sh = lsb - exp;
    let (mut kept, rem, half) = if sh >= 128 {
        (0u128, 1u128, u128::MAX)
    } else if sh > 0 {
        (mag >> sh, mag & ((1u128 << sh) - 1), 1u128 << (sh - 1))
    } else {
        (mag << (-sh), 0, 1)
    };
    let inexact = rem != 0 || sticky;
    let up = match m.rm {
        0 => rem > half || (rem == half && (sticky || kept & 1 == 1)),
        1 => inexact && !sign,
        2 => inexact && sign,
        _ => false,
    };
    kept += up as u128;
    if kept >= 1 << 24 {
        kept >>= 1;
        lsb += 1;
    }
    let away = match m.rm {
        0 => true,
        1 => !sign,
        2 => sign,
        _ => false,
    };
    if lsb + 23 > 127 {
        return sg | if away { 0x7F80_0000 } else { 0x7F7F_FFFF };
    }
    let bits = if kept < 1 << 23 { kept as u32 } else { (((lsb + 23 + 127) as u32) << 23) | (kept as u32 - (1 << 23)) };
    if !m.keep_out && bits < 0x80_0000 {
        return sg;
    }
    sg | bits
}

/// Exact sum of ±ma·2^ea and ±mb·2^eb → round. `both_pos_zero`: both addends are +0.
fn sum(sa: bool, ma: u128, ea: i32, sb: bool, mb: u128, eb: i32, m: Mode, both_pos_zero: bool) -> u32 {
    if ma == 0 && mb == 0 {
        let neg = if m.rm == 2 { !both_pos_zero } else { sa && sb };
        return (neg as u32) << 31;
    }
    let tops = [if ma == 0 { i32::MIN } else { ea + bitlen(ma) }, if mb == 0 { i32::MIN } else { eb + bitlen(mb) }];
    let t = tops[0].max(tops[1]) - 100;
    // an operand entirely below 2^t is < 2^(T-52): only a sticky bit for the result
    let al = |mm: u128, e: i32| -> (u128, bool) {
        if mm == 0 {
            (0, false)
        } else if e >= t {
            (mm << (e - t), false)
        } else {
            (0, true)
        }
    };
    let (a, ast) = al(ma, ea);
    let (b, bst) = al(mb, eb);
    let va = if sa { -(a as i128) } else { a as i128 };
    let vb = if sb { -(b as i128) } else { b as i128 };
    let s = va + vb;
    let st_sign = if ast { Some(sa) } else if bst { Some(sb) } else { None };
    if s == 0 && st_sign.is_none() {
        // exact cancellation: +0, or -0 toward -inf
        return ((m.rm == 2) as u32) << 31;
    }
    let neg = s < 0 || (s == 0 && st_sign == Some(true));
    let mag = s.unsigned_abs();
    match st_sign {
        None => round(neg, mag, t, false, m),
        Some(ss) if ss == neg || mag == 0 => round(neg, mag, t, true, m),
        // |s| - ε = (|s| - 1) + (1 - ε)
        Some(_) => round(neg, mag - 1, t, true, m),
    }
}

fn is_nan(x: u32) -> bool {
    x & 0x7FFF_FFFF > 0x7F80_0000
}

/// The CPU result of an operation on f32 bits; None: a special case left to native (inf/NaN).
fn ref_add(a: u32, b: u32, m: Mode) -> Option<u32> {
    let (sa, ma, ea) = parts(a, m.keep_in)?;
    let (sb, mb, eb) = parts(b, m.keep_in)?;
    let bpz = !sa && !sb;
    Some(sum(sa, ma, ea, sb, mb, eb, m, bpz))
}

fn ref_mul(a: u32, b: u32, m: Mode) -> Option<u32> {
    let (sa, ma, ea) = parts(a, m.keep_in)?;
    let (sb, mb, eb) = parts(b, m.keep_in)?;
    Some(round(sa ^ sb, ma * mb, ea + eb, false, m))
}

fn ref_fma(a: u32, b: u32, c: u32, m: Mode) -> Option<u32> {
    let (sa, ma, ea) = parts(a, m.keep_in)?;
    let (sb, mb, eb) = parts(b, m.keep_in)?;
    let (sc, mc, ec) = parts(c, m.keep_in)?;
    let sp = sa ^ sb;
    let bpz = !sp && !sc;
    Some(sum(sp, ma * mb, ea + eb, sc, mc, ec, m, bpz))
}

fn ref_sqrt(a: u32, m: Mode) -> Option<u32> {
    let (s, ma, ea) = parts(a, m.keep_in)?;
    if ma == 0 {
        return Some((s as u32) << 31);
    }
    if s {
        return None;
    }
    // sqrt(ma · 2^ea): make the exponent even and the integer large (≥ 2^100)
    let (mut mm, mut e) = (ma, ea);
    if e & 1 != 0 {
        mm <<= 1;
        e -= 1;
    }
    let k = (100 - bitlen(mm)).max(0) & !1;
    mm <<= k;
    e -= k;
    let r = isqrt(mm);
    Some(round(false, r, e / 2, r * r != mm, m))
}

fn isqrt(n: u128) -> u128 {
    let mut x = (n as f64).sqrt() as u128;
    while x * x > n {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
}

fn ref_cvt_int(a: u32, signed: bool, m: Mode) -> u32 {
    let (neg, mag) = if signed { ((a as i32) < 0, (a as i32).unsigned_abs()) } else { (false, a) };
    round(neg, mag as u128, 0, false, m)
}

fn ref_ldexp(a: u32, n: i32, m: Mode) -> Option<u32> {
    let (s, ma, ea) = parts(a, m.keep_in)?;
    if ma == 0 {
        return Some((s as u32) << 31);
    }
    Some(round(s, ma, ea + n.clamp(-400, 400), false, m))
}

fn ref_cvt_f64(d: u64, m: Mode) -> Option<u32> {
    let s = d >> 63 == 1;
    let e = ((d >> 52) & 0x7FF) as i32;
    let f = (d & ((1 << 52) - 1)) as u128;
    match e {
        0x7FF => None,
        0 => Some(round(s, f, -1074, false, m)),
        _ => Some(round(s, f | 1 << 52, e - 1075, false, m)),
    }
}

// ------------------------------------------------------------------ inputs

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// An f32 with an exponent from a chosen band (denormal, small, unit, large, near overflow).
    fn f32(&mut self) -> u32 {
        let r = self.next();
        let sign = r & 0x8000_0000;
        let mant = self.next() & 0x7F_FFFF;
        let exp = match r % 8 {
            0 => 0,
            1 => 1 + r / 8 % 30,
            2 => 254 - r / 8 % 4,
            _ => 110 + r / 8 % 36,
        };
        sign | exp << 23 | mant
    }
}

/// Lane inputs: v0 a, v1 b, v2 c, v3 integer, v4 ldexp exponent, v[6:7] f64.
fn inputs(seed: u64) -> Vec<[u32; 8]> {
    let mut g = Rng(seed);
    let mut v = Vec::new();
    for l in 0..64 {
        let a = g.f32();
        let mut b = g.f32();
        let mut c = g.f32();
        match l % 8 {
            // exact cancellation (zero signs), nearby magnitudes, tiny addend
            0 => b = a ^ 0x8000_0000,
            1 => b = (a & 0xFF80_0000) | (g.next() & 0x7F_FFFF),
            2 => b = (a & 0x8000_0000) | 0x0000_0001,
            3 => c = 0x8000_0000 | (g.next() & 0x7F_FFFF),
            _ => {}
        }
        if l == 5 {
            b = 0x8000_0000; // +x + -0
        }
        let i = g.next();
        let n = (g.next() % 300) as i32 - 200;
        let d = ((g.next() as u64) << 32 | g.next() as u64) & !(0x7FFu64 << 52) | ((800 + g.next() % 450) as u64) << 52;
        v.push([a, b, c, i, n as u32, 0, d as u32, (d >> 32) as u32]);
    }
    v
}

fn ops_program(prefix: &[Vec<u8>]) -> Vec<u8> {
    let mut code: Vec<u8> = prefix.concat();
    let ins: &[(&str, &str, &[(&str, u64)])] = &[
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V), ("VSRC1", 1)]),
        ("V_SUB_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V), ("VSRC1", 1)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 12), ("SRC0", V), ("VSRC1", 1)]),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2)]),
        ("V_SQRT_F32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V)]),
        ("V_CVT_F32_I32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 3)]),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 16), ("SRC0", V + 3)]),
        ("V_LDEXP_F32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V), ("SRC1", V + 4)]),
        ("V_MAD_F32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2)]),
        ("V_CVT_F32_F64", "ENC_VOP1", &[("VDST", 19), ("SRC0", V + 6)]),
        // abs / neg source modifiers
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", V + 1), ("ABS", 1), ("NEG", 2)]),
    ];
    for (n, e, f) in ins {
        code.extend(encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    code
}

fn check_lane(l: usize, v: &[u32], inp: &[u32; 8], m: Mode) {
    let (a, b, c, i, n) = (inp[0], inp[1], inp[2], inp[3], inp[4] as i32);
    let d = inp[6] as u64 | (inp[7] as u64) << 32;
    let neg = |x: u32| x ^ 0x8000_0000;
    let cases: [(&str, usize, Option<u32>); 11] = [
        ("add", 10, ref_add(a, b, m)),
        ("sub", 11, ref_add(a, neg(b), m)),
        ("mul", 12, ref_mul(a, b, m)),
        ("fma", 13, ref_fma(a, b, c, m)),
        ("sqrt", 14, ref_sqrt(a, m)),
        ("cvt_f32_i32", 15, Some(ref_cvt_int(i, true, m))),
        ("cvt_f32_u32", 16, Some(ref_cvt_int(i, false, m))),
        ("ldexp", 17, ref_ldexp(a, n, m)),
        // mad: the product rounded, then the sum
        ("mad", 18, ref_mul(a, b, m).and_then(|p| if is_nan(p) || p & 0x7FFF_FFFF == 0x7F80_0000 { None } else { ref_add(p, c, m) })),
        ("cvt_f32_f64", 19, ref_cvt_f64(d, m)),
        ("add |a| + -b", 20, ref_add(a & 0x7FFF_FFFF, neg(b), m)),
    ];
    for (name, r, want) in cases {
        let Some(want) = want else { continue };
        let got = v[r];
        if is_nan(want) && is_nan(got) {
            continue;
        }
        assert_eq!(got, want, "lane {l} {name} mode {m:?}: inputs a {a:#010x} b {b:#010x} c {c:#010x} i {i:#x} n {n} d {d:#018x}: got {:e} want {:e}", f32::from_bits(got), f32::from_bits(want));
    }
}

fn run(code: &[u8], st: &KernelState) -> runner::KernelResult {
    let k = runner::direct_kernel_full(code, true).unwrap_or_else(|e| panic!("direct translation: {e}"));
    if let Err(e) = super::validate::validate(&k.dxil) {
        panic!("validator: {e}");
    }
    runner::run_with_statics(&k.dxil, st, &[], &[], &[], &k.static_samplers).expect("GPU run")
}

fn state(inp: &[[u32; 8]]) -> KernelState {
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[..8].copy_from_slice(&inp[l]);
    }
    st
}

/// The reference itself on hand-checked values.
#[test]
fn directed_reference_basics() {
    let m = |rm| Mode { rm, keep_in: true, keep_out: true };
    let one = 1.0f32.to_bits();
    let tiny = 1;
    assert_eq!(ref_add(one, tiny, m(1)), Some(one + 1));
    assert_eq!(ref_add(one, tiny, m(2)), Some(one));
    assert_eq!(ref_add(one, tiny | 0x8000_0000, m(3)), Some(one - 1));
    assert_eq!(ref_add(one, tiny | 0x8000_0000, m(0)), Some(one));
    assert_eq!(ref_add(one, one | 0x8000_0000, m(2)), Some(0x8000_0000));
    assert_eq!(ref_add(one, one | 0x8000_0000, m(1)), Some(0));
    assert_eq!(ref_mul(f32::MAX.to_bits(), 2.0f32.to_bits(), m(3)), Some(f32::MAX.to_bits()));
    assert_eq!(ref_mul(f32::MAX.to_bits(), 2.0f32.to_bits(), m(1)), Some(0x7F80_0000));
    assert_eq!(ref_mul(f32::MAX.to_bits() | 0x8000_0000, 2.0f32.to_bits(), m(1)), Some(f32::MAX.to_bits() | 0x8000_0000));
    assert_eq!(ref_sqrt(2.0f32.to_bits(), m(3)), Some(2f32.sqrt().to_bits()));
    assert_eq!(ref_sqrt(2.0f32.to_bits(), m(1)), Some(2f32.sqrt().to_bits() + 1));
    assert_eq!(ref_sqrt(4.0f32.to_bits(), m(1)), Some(2.0f32.to_bits()));
    assert_eq!(ref_cvt_int(0x0100_0001, false, m(1)), 16777218f32.to_bits());
    assert_eq!(ref_cvt_int(0x0100_0001, false, m(3)), 16777216f32.to_bits());
    // RNE reference agrees with the host on random sums / products
    let mut g = Rng(7);
    for _ in 0..20000 {
        let (a, b, c) = (g.f32(), g.f32(), g.f32());
        let (fa, fb, fc) = (f32::from_bits(a), f32::from_bits(b), f32::from_bits(c));
        if let Some(r) = ref_add(a, b, m(0)) {
            assert!(r == (fa + fb).to_bits() || is_nan(r), "add {a:#x} {b:#x}");
        }
        if let Some(r) = ref_mul(a, b, m(0)) {
            assert_eq!(r, (fa * fb).to_bits(), "mul {a:#x} {b:#x}");
        }
        if let Some(r) = ref_fma(a, b, c, m(0)) {
            assert!(r == fa.mul_add(fb, fc).to_bits() || is_nan(r), "fma {a:#x} {b:#x} {c:#x}");
        }
        if let Some(r) = ref_sqrt(a, m(0)) {
            assert_eq!(r, fa.sqrt().to_bits(), "sqrt {a:#x}");
        }
        // directed results bracket the exact value: RTN ≤ RNE ≤ RTP, RTZ toward zero
        if let (Some(lo), Some(hi)) = (ref_add(a, b, m(2)), ref_add(a, b, m(1))) {
            let (lo, hi) = (f32::from_bits(lo), f32::from_bits(hi));
            assert!(lo <= fa + fb && fa + fb <= hi, "bracket {a:#x} {b:#x}");
        }
    }
}

/// Constant modes from `s_round_mode`: every rounding op, all three directions.
#[test]
fn gpu_direct_f32_round_modes_static() {
    if !gpu_ready() {
        return;
    }
    for rm in [1u32, 2, 3] {
        let code = ops_program(&[encode("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", rm as u64)], None).unwrap()]);
        let k = runner::direct_kernel_full(&code, true).unwrap_or_else(|e| panic!("direct translation: {e}"));
        for seed in 0..16u64 {
            let inp = inputs(rm as u64 * 977 + seed * 13);
            let r = runner::run_with_statics(&k.dxil, &state(&inp), &[], &[], &[], &k.static_samplers).expect("GPU run");
            let m = Mode { rm, keep_in: true, keep_out: true };
            for l in 0..64 {
                check_lane(l, &r.vgpr[l], &inp[l], m);
            }
        }
    }
}

/// A run-time mode (`s_setreg_b32` of MODE[1:0] from s0) and denormal flushing
/// (`s_denorm_mode 0`), plus a return to nearest even.
#[test]
fn gpu_direct_f32_round_modes_dynamic_and_denormals() {
    if !gpu_ready() {
        return;
    }
    let hw = |id: u64, off: u64, w: u64| id | off << 6 | (w - 1) << 11;
    for rm in [0u32, 1, 2, 3] {
        let code = ops_program(&[encode("S_SETREG_B32", "ENC_SOPK", &[("SIMM16", hw(1, 0, 2)), ("SDST", 0)], None).unwrap()]);
        let inp = inputs(31 + rm as u64);
        let mut st = state(&inp);
        st.sgpr[0] = rm;
        let r = run(&code, &st);
        let m = Mode { rm, keep_in: true, keep_out: true };
        for l in 0..64 {
            if rm == 0 {
                // nearest even: the translator's usual code; check the exactly rounded ops
                let v = &r.vgpr[l];
                let (a, b) = (inp[l][0], inp[l][1]);
                let want = ref_add(a, b, m);
                if let Some(w) = want {
                    let den = |x: u32| x & 0x7F80_0000 == 0 && x & 0x7F_FFFF != 0;
                    if !den(a) && !den(b) && !den(w) {
                        assert_eq!(v[10], w, "lane {l} add RNE");
                    }
                }
            } else {
                check_lane(l, &r.vgpr[l], &inp[l], m);
            }
        }
    }
    // flushed denormals (FP_DENORM single = 0) with round toward +inf
    let code = ops_program(&[
        encode("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 1)], None).unwrap(),
        encode("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap(),
    ]);
    let inp = inputs(4242);
    let r = run(&code, &state(&inp));
    let m = Mode { rm: 1, keep_in: false, keep_out: false };
    for l in 0..64 {
        check_lane(l, &r.vgpr[l], &inp[l], m);
    }
}

#[test]
fn gpu_frexp_exp_f32_denorm_with_directed_rounding() {
    if !gpu_ready() {
        return;
    }
    let mut code = encode("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 3)], None).unwrap();
    for (name, enc, fields) in [
        ("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 1)][..]),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)]),
        ("S_DENORM_MODE", "ENC_SOPP", &[("SIMM16", 2)]),
        ("V_FREXP_EXP_I32_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("S_ENDPGM", "ENC_SOPP", &[]),
    ] {
        code.extend(encode(name, enc, fields, None).unwrap());
    }
    let mut st = KernelState::default();
    for v in &mut st.vgpr {
        v[..3].copy_from_slice(&[1, 0x3F80_0000, 0x3380_0000]);
    }
    let r = run(&code, &st);
    for (lane, v) in r.vgpr.iter().enumerate() {
        assert_eq!(v[4] as i32, -148, "lane {lane}: preserve after S_ROUND_MODE");
        assert_eq!(v[5], 0, "lane {lane}: flush after S_DENORM_MODE");
        assert_eq!(v[6], 0x3F80_0001, "lane {lane}: round toward +inf after S_DENORM_MODE");
    }
}

/// Unimplemented directed ops are translation errors, not silent nearest-even results.
#[test]
fn direct_f32_round_mode_unsupported_op_rejected() {
    let mut code = encode("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 3)], None).unwrap();
    code.extend(encode("V_DOT2_F32_F16", "ENC_VOP3P", &[("VDST", 1), ("SRC0", V), ("SRC1", V), ("SRC2", V)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let e = runner::direct_kernel(&code, true).unwrap_err();
    assert!(e.contains("directed f32 rounding"), "{e}");
}

/// In a directed rounding mode with input denormals preserved (the default `MODE_INIT`,
/// `s_denorm_mode 3`), `denormal * +inf` and `fma(denormal, +inf, 1.0)` must produce `+inf`,
/// not fall back to the D3D12 native f32 op (which flushes f32 denormals and yields NaN).
#[test]
fn gpu_direct_f32_round_preserved_denormal_times_inf_is_inf() {
    if !gpu_ready() {
        return;
    }
    let mut code = Vec::new();
    code.extend(encode("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 1)], None).unwrap());
    code.extend(encode("V_MUL_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V), ("VSRC1", 1)], None).unwrap());
    code.extend(encode("V_FMA_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 2)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.vgpr[0][0] = 0x0000_0001; // smallest positive f32 denormal (preserved in MODE_INIT)
    st.vgpr[0][1] = 0x7F80_0000; // +inf
    st.vgpr[0][2] = 0x3F80_0000; // 1.0
    let r = run(&code, &st);
    assert_eq!(r.vgpr[0][10], 0x7F80_0000, "v_mul_f32(denorm, +inf) with preserve = {:#010x}, expected +inf", r.vgpr[0][10]);
    assert_eq!(r.vgpr[0][11], 0x7F80_0000, "v_fma_f32(denorm, +inf, 1.0) with preserve = {:#010x}, expected +inf", r.vgpr[0][11]);
}

/// `s_setreg_b32 hwreg(HW_REG_MODE, 0, 2), s0` (`sp_mask = 3`, `dp_mask = 0`) only updates
/// f32 rounding (`MODE[1:0]`); it must leave `VAR_FP16_RM` (`MODE[3:2]`) untouched on the GPU.
#[test]
fn gpu_s_setreg_b32_f32_mode_preserves_fp16_round_mode() {
    let mut code = Vec::new();
    // Set MODE[3:2] (f16/f64 rounding) = 3 (RTZ), MODE[1:0] = 0 (RNE) -> SIMM16 = 0b1100 = 0x0C.
    code.extend(encode("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0x0C)], None).unwrap());
    // Write ONLY MODE[1:0] from s0 (HW_REG_MODE=1, offset=0, size-1=1 -> 0x0801; sp_mask=3, dp_mask=0).
    code.extend(encode("S_SETREG_B32", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 0x0801)], None).unwrap());
    // Directed f32 op using MODE[1:0] = s0 (RTN = 2), followed by f16 conversion using MODE[3:2] = 3 (RTZ).
    code.extend(encode("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 2), ("VSRC1", 3)], None).unwrap());
    code.extend(encode("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 1)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    if !gpu_ready() {
        let _ = runner::direct_kernel(&code, true).expect("direct translation");
        return;
    }
    let mut st = KernelState::default();
    st.sgpr[0] = 2; // f32 RTN (bit 0 == 0; previously clobbered VAR_FP16_RM from 3 -> 1 [RTP])
    // Lane 0: +1.0 + 2^-23 -> f16 RTZ is 0x3C00 (corrupted RTP would produce 0x3C01)
    st.vgpr[0][1] = 0x3F80_0001;
    st.vgpr[0][2] = 0x3F80_0000; // +1.0
    st.vgpr[0][3] = 0x3380_0000; // +2^-24 (under f32 RTN: 1.0 + 2^-24 == 1.0)
    // Lane 1: -1.0 - 2^-23 -> f16 RTZ is 0xBC00; f32 RTN: -1.0 + (-2^-24) == -1.0 - 2^-23
    st.vgpr[1][1] = 0xBF80_0001;
    st.vgpr[1][2] = 0xBF80_0000; // -1.0
    st.vgpr[1][3] = 0xB380_0000; // -2^-24

    let r = run(&code, &st);
    assert_eq!(r.vgpr[0][10], 0x3F80_0000, "lane 0 f32 RTN");
    assert_eq!(r.vgpr[1][10], 0xBF80_0001, "lane 1 f32 RTN");
    assert_eq!(r.vgpr[0][11], 0x3C00, "lane 0 f16 RTZ preserved after s_setreg_b32 MODE[1:0]");
    assert_eq!(r.vgpr[1][11], 0xBC00, "lane 1 f16 RTZ preserved after s_setreg_b32 MODE[1:0]");
}

/// `s_setreg_b32` writing only `MODE[1:0]` (`dp_mask == 0`) does not touch f64 rounding, so a
/// shader containing an f64 rounding op must still translate cleanly.
#[test]
fn direct_s_setreg_b32_f32_mode_allows_f64_ops() {
    for simm in [0x0801u64, 0x0001, 0x0041] {
        let mut code = Vec::new();
        code.extend(encode("S_SETREG_B32", "ENC_SOPK", &[("SDST", 0), ("SIMM16", simm)], None).unwrap());
        code.extend(encode("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", V), ("SRC1", V + 2)], None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let dxil = runner::direct_kernel(&code, true)
            .unwrap_or_else(|e| panic!("SIMM16={simm:#06x} with V_ADD_F64 should translate: {e}"));
        super::validate::validate(&dxil).unwrap_or_else(|e| panic!("validator (SIMM16={simm:#06x}): {e}"));
    }
}
