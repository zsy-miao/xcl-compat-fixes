//! Instruction semantics: one RDNA2 instruction → HLSL statements.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::{elements_at, param_registers, ps_attribute_slot, ps_input_vgpr, ps_slot_attribute, ranked, vs_param_register, Mode, MsInterface, Options, KERNEL_EXPORTS, KERNEL_LANES, KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_SPLIT_MAGIC, KERNEL_SPLIT_SGPRS, KERNEL_VGPRS, MESH_HEADER};
use super::upper_lanes::Read;
use crate::gpu::dxbc::SigElement;
use crate::gpu::rdna2::{self, Decoded, Encoding, Operand, Variant};

pub(super) enum Flow {
    Next,
    Jump(u64),
    JumpExpr(String),
    Cond(String, u64),
    End,
}

pub(super) struct Em {
    opt: Options,
    out: String,
    pub indent: usize,
    sgprs: BTreeSet<u64>,
    vgprs: BTreeSet<u64>,
    pub unsupported: BTreeMap<String, usize>,
    /// "mnemonic: reason" → count
    pub unsupported_why: BTreeMap<String, usize>,
    pub ignored: BTreeMap<String, usize>,
    tmp: usize,
    /// extra write enable of the current DPP16 instruction
    dpp_enable: Option<String>,
    /// PS: interpolation modifier per attribute ("", "centroid", "nointerpolation" …)
    pub ps_modes: BTreeMap<u32, String>,
    /// PS: an export with DONE was emitted (lanes without it are discarded)
    ps_done: bool,
    /// LDS used (compute): declare the groupshared array
    uses_lds: bool,
    /// scratch_* used: declare the per-lane private memory
    uses_scratch: bool,
    /// split wave64 (`x64`): words one exchange carries (1: masks and lane reads, up to 16:
    /// coherent scalar loads, 32: `ds_bpermute`)
    x64_words: usize,
    /// split wave64: pc → SGPRs whose whole value may be read after that instruction
    /// (`x64_masks`); a mask written to none of them stays in its half (no exchange)
    pub x64_whole_live: BTreeMap<u64, u128>,
    /// wave64 VS / PS on a GPU with 32-lane waves only: what the console's lanes 32..63 hold
    /// where the program reads them (`upper_lanes`), used when the wave has 32 lanes
    pub upper: Option<super::upper_lanes::UpperLanes>,
    /// pc of the instruction being emitted
    cur_pc: u64,
    /// MIMG instruction address → its sampler's S# when the code builds it from constants
    pub static_ss: BTreeMap<u64, [u32; 4]>,
    /// the static samplers sampled with: (S#, comparison), in register order
    pub static_used: Vec<([u32; 4], bool)>,
    /// highest VGPR the program references (v_movrel* ranges)
    pub max_vgpr: u64,
    /// program contains an f16 VALU/interp instruction whose rounding obeys `MODE[3:2]`
    has_f16_round_op: bool,
    /// program contains an f64 instruction whose rounding obeys `MODE[3:2]`
    has_f64_round_op: bool,
    /// program writes a non-default `MODE[3:2]` and uses f16 ops (emits `xcl_fp16_rm`)
    dynamic_f16_rm: bool,
}

/// Instructions without an effect in this execution model.
const NOOPS: &[&str] = &[
    "s_nop", "s_waitcnt", "s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt", "s_waitcnt_depctr",
    "s_inst_prefetch", "s_clause", "s_code_end", "s_setprio", "s_sleep", "s_icache_inv", "s_dcache_inv", "s_gl1_inv", "s_ttracedata",
    "s_ttracedata_imm", "v_nop", "s_denorm_mode", "s_sendmsg", "s_sendmsghalt",
    "s_set_gpr_idx_off", "s_version", "buffer_gl0_inv", "buffer_gl1_inv", "ds_nop",
];

fn sname(code: u64) -> Option<String> {
    Some(match code {
        0..=105 => format!("s{code}"),
        106 => "vcc_lo".into(),
        107 => "vcc_hi".into(),
        108..=123 => format!("ttmp{}", code - 108),
        124 => "m0".into(),
        126 => "exec_lo".into(),
        127 => "exec_hi".into(),
        _ => return None,
    })
}

fn f32_const(code: u64) -> Option<f32> {
    Some(match code {
        240 => 0.5,
        241 => -0.5,
        242 => 1.0,
        243 => -1.0,
        244 => 2.0,
        245 => -2.0,
        246 => 4.0,
        247 => -4.0,
        248 => 0.159_154_94,
        _ => return None,
    })
}

fn f16_const(code: u64) -> Option<u32> {
    Some(match code {
        240 => 0x3800,
        241 => 0xB800,
        242 => 0x3C00,
        243 => 0xBC00,
        244 => 0x4000,
        245 => 0xC000,
        246 => 0x4400,
        247 => 0xC400,
        248 => 0x3118,
        _ => return None,
    })
}

fn hex(v: u32) -> String {
    format!("0x{v:08X}u")
}

#[derive(Default, Clone, Copy)]
struct Mods {
    abs: [bool; 3],
    neg: [bool; 3],
    omod: u64,
    clamp: bool,
    opsel: u64,
}

/// A float VOPC compare of `a` and `b`, f32 or f64, with NaN tested explicitly. `isnan` names
/// the NaN test for the type. The ordered compares lt, eq, le, gt, ge, lg and o are false and
/// the unordered ones, n… and u, are true when either operand is NaN.
/// The plain HLSL compare is not enough. After `!(a >= b)` in a shader, the AMD driver gave a
/// later `a < b` of the same operands that unordered result, which is true for a NaN. The
/// DXIL holds a correct `fcmp olt`, with or without -Gis. Seen with driver 32.0.32029.2010 on
/// an RX 7900 XTX and on an RDNA2 iGPU.
/// `and` and `or` replace `&&` and `||`, so the compare has no branches and DXC computes the
/// NaN tests of operands that several compares share only once.
fn float_compare(op: &str, a: &str, b: &str, isnan: &str) -> Option<String> {
    let rel = |r: &str| {
        Some(match r {
            "lt" => format!("{a} < {b}"),
            "eq" => format!("{a} == {b}"),
            "le" => format!("{a} <= {b}"),
            "gt" => format!("{a} > {b}"),
            "ge" => format!("{a} >= {b}"),
            "lg" => format!("or({a} < {b}, {a} > {b})"),
            _ => return None,
        })
    };
    let nan = format!("or({isnan}({a}), {isnan}({b}))");
    Some(match op {
        "f" => "false".to_string(),
        "t" | "tru" => "true".to_string(),
        "o" => format!("!{nan}"),
        "u" => nan,
        _ => match op.strip_prefix('n').and_then(rel) {
            Some(r) => format!("or({nan}, !({r}))"),
            None => format!("and(!{nan}, {})", rel(op)?),
        },
    })
}

/// Layout of the SDWA / DPP extension dword for this instruction's base encoding.
pub(super) fn ext_encoding(d: &Decoded) -> Option<&'static Encoding> {
    let base = match d.enc.name {
        "ENC_VOP1" => "VOP1",
        "ENC_VOP2" => "VOP2",
        "ENC_VOPC" => "VOPC",
        _ => return None,
    };
    let name = match (d.variant, base) {
        (Variant::Sdwa, "VOPC") => "VOPC_VOP_SDWA_SDST_ENC".to_string(),
        (Variant::Sdwa, b) => format!("{b}_VOP_SDWA"),
        (Variant::Dpp16, b) => format!("{b}_VOP_DPP16"),
        (Variant::Dpp8, b) => format!("{b}_VOP_DPP8"),
        _ => return None,
    };
    rdna2::encoding(&name)
}

/// Source lane of a DPP16 control (−1: no source), as an HLSL expression.
fn dpp16_src(ctrl: u64) -> Option<String> {
    let n = ctrl & 15;
    Some(match ctrl {
        0x000..=0x0FF => format!("(int)((lane_id & ~3u) | (({ctrl}u >> (2u * (lane_id & 3u))) & 3u))"),
        0x101..=0x10F => format!("select((lane_id & 15u) + {n}u < 16u, (int)(lane_id + {n}u), -1)"),
        0x111..=0x11F => format!("select((lane_id & 15u) >= {n}u, (int)(lane_id - {n}u), -1)"),
        0x121..=0x12F => format!("(int)((lane_id & ~15u) | (((lane_id & 15u) - {n}u) & 15u))"),
        0x140 => "(int)((lane_id & ~15u) | (15u - (lane_id & 15u)))".into(),
        0x141 => "(int)((lane_id & ~15u) | (lane_id & 8u) | (7u - (lane_id & 7u)))".into(),
        0x150..=0x15F => format!("(int)((lane_id & ~15u) | {n}u)"),
        0x160..=0x16F => format!("(int)(lane_id ^ {n}u)"),
        _ => return None,
    })
}

/// The lane `lane` of a 32-lane half reads with `ds_swizzle_b32` (the RDNA ISA's DS_SWIZZLE_B32):
/// offset ≥ 0xE000 FFT, ≥ 0xC000 rotate, bit 15 quad permute, otherwise the and/or/xor bit
/// mode (xor = offset[14:10], so 0x401F swaps the 16-lane halves).
/// MIMG sample coordinates per DIM as HLSL takes them (array layer rounded); empty for cubes.
fn sample_coord(dim: u64, c: &[String]) -> String {
    match dim {
        0 => c[0].clone(),
        1 => format!("float2({}, {})", c[0], c[1]),
        2 => format!("float3({}, {}, {})", c[0], c[1], c[2]),
        4 => format!("float2({}, round({}))", c[0], c[1]),
        5 => format!("float3({}, {}, round({}))", c[0], c[1], c[2]),
        _ => String::new(),
    }
}

/// The coordinates moved by the offset vector `o` (from `texel_offset`); array layers stay.
fn moved_coords(dim: u64, c: &[String], o: &str) -> Vec<String> {
    c.iter().enumerate().map(|(k, x)| if (dim == 4 && k == 1) || (dim == 5 && k == 2) { x.clone() } else { format!("({x} + {o}[{k}])") }).collect()
}

pub(crate) fn swizzle_lane(offset: u64, lane: u32) -> u32 {
    let i = lane & 0x1F;
    let mask = (offset & 0x1F) as u32;
    if offset >= 0xE000 {
        // FFT: the lane's five bits reversed, shifted right by the mask's population, masked bits kept
        ((i.reverse_bits() >> 27) >> mask.count_ones()) | (i & mask)
    } else if offset >= 0xC000 {
        // rotate by offset[9:5] lanes, left (offset[10] = 0) or right; masked bits stay
        let r = ((offset >> 5) & 0x1F) as u32;
        let r = if offset & 0x400 != 0 { r.wrapping_neg() } else { r };
        ((i & mask) | (i.wrapping_add(r) & !mask)) & 0x1F
    } else if offset & 0x8000 != 0 {
        (i & !3) | ((offset as u32 >> (2 * (i & 3))) & 3)
    } else {
        let (or, xor) = (((offset >> 5) & 0x1F) as u32, ((offset >> 10) & 0x1F) as u32);
        ((i & mask) | or) ^ xor
    }
}

/// `ds_swizzle_b32`'s source lane as an HLSL expression of `lane_id` (`swizzle_lane`; the
/// upper 32 lanes of a wave64 swizzle among themselves), and the declaration it needs first:
/// FFT and rotate (rare) read the permutation from a constant table named `table`.
fn swizzle_source(offset: u64, table: &str) -> (Option<String>, String) {
    if offset >= 0xC000 {
        let lanes: Vec<String> = (0..32).map(|i| format!("{}u", swizzle_lane(offset, i))).collect();
        let decl = format!("static const uint {table}[32] = {{ {} }};", lanes.join(", "));
        return (Some(decl), format!("(lane_id & 0x20u) | {table}[lane_id & 0x1Fu]"));
    }
    if offset & 0x8000 != 0 {
        return (None, format!("(lane_id & ~3u) | (({offset}u >> (2u * (lane_id & 3u))) & 3u)"));
    }
    let (and, or, xor) = (offset & 0x1F, (offset >> 5) & 0x1F, (offset >> 10) & 0x1F);
    (None, format!("(lane_id & 0x20u) | ((((lane_id & 0x1Fu) & {and}u) | {or}u) ^ {xor}u)"))
}

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

/// D16 typed-buffer operations pack two converted components into each VGPR.  The HI_X
/// forms update/read only the high half of VDATA.
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

/// GFX10 unified buffer formats (index = FORMAT value; LLVM `UfmtSymbolicGFX10`).
pub const BUF_FORMATS: [&str; 78] = [
    "INVALID", "8_UNORM", "8_SNORM", "8_USCALED", "8_SSCALED", "8_UINT", "8_SINT", "16_UNORM", "16_SNORM", "16_USCALED",
    "16_SSCALED", "16_UINT", "16_SINT", "16_FLOAT", "8_8_UNORM", "8_8_SNORM", "8_8_USCALED", "8_8_SSCALED", "8_8_UINT", "8_8_SINT",
    "32_UINT", "32_SINT", "32_FLOAT", "16_16_UNORM", "16_16_SNORM", "16_16_USCALED", "16_16_SSCALED", "16_16_UINT", "16_16_SINT",
    "16_16_FLOAT", "10_11_11_UNORM", "10_11_11_SNORM", "10_11_11_USCALED", "10_11_11_SSCALED", "10_11_11_UINT", "10_11_11_SINT",
    "10_11_11_FLOAT", "11_11_10_UNORM", "11_11_10_SNORM", "11_11_10_USCALED", "11_11_10_SSCALED", "11_11_10_UINT", "11_11_10_SINT",
    "11_11_10_FLOAT", "10_10_10_2_UNORM", "10_10_10_2_SNORM", "10_10_10_2_USCALED", "10_10_10_2_SSCALED", "10_10_10_2_UINT",
    "10_10_10_2_SINT", "2_10_10_10_UNORM", "2_10_10_10_SNORM", "2_10_10_10_USCALED", "2_10_10_10_SSCALED", "2_10_10_10_UINT",
    "2_10_10_10_SINT", "8_8_8_8_UNORM", "8_8_8_8_SNORM", "8_8_8_8_USCALED", "8_8_8_8_SSCALED", "8_8_8_8_UINT", "8_8_8_8_SINT",
    "32_32_UINT", "32_32_SINT", "32_32_FLOAT", "16_16_16_16_UNORM", "16_16_16_16_SNORM", "16_16_16_16_USCALED",
    "16_16_16_16_SSCALED", "16_16_16_16_UINT", "16_16_16_16_SINT", "16_16_16_16_FLOAT", "32_32_32_UINT", "32_32_32_SINT",
    "32_32_32_FLOAT", "32_32_32_32_UINT", "32_32_32_32_SINT", "32_32_32_32_FLOAT",
];

/// Packed format info for the HLSL table: component count [2:0], numeric type [5:3]
/// (0 unorm, 1 snorm, 2 uscaled, 3 sscaled, 4 uint, 5 sint, 6 float), component widths
/// (x first = least significant bits; names list the most significant component first)
/// at [8 + 6k +: 6].
pub fn buf_format_info(idx: usize) -> u32 {
    let name = BUF_FORMATS[idx];
    let Some((sizes, ty)) = name.rsplit_once('_') else { return 0 };
    let ty = match ty {
        "UNORM" => 0,
        "SNORM" => 1,
        "USCALED" => 2,
        "SSCALED" => 3,
        "UINT" => 4,
        "SINT" => 5,
        "FLOAT" => 6,
        _ => return 0,
    };
    let mut w: Vec<u32> = sizes.split('_').filter_map(|x| x.parse().ok()).collect();
    w.reverse();
    let mut info = w.len() as u32 | ty << 3;
    for (k, x) in w.iter().enumerate() {
        info |= x << (8 + 6 * k);
    }
    info
}

/// A destination register (pair).
#[derive(Clone)]
enum Dst {
    V(u64),
    S(u64),
    Null,
}

/// Decode `OPR_HWREG` (`SIMM16` of `s_setreg*`): `(id, offset, width)`.
/// `id == 1` is `HW_REG_MODE`, where `[1:0]` is `FP_ROUND_MODE_SP` (f32) and `[3:2]` is
/// `FP_ROUND_MODE_DP` (f64 and f16).
fn decode_hwreg(simm16: u16) -> (u32, u32, u32) {
    let id = (simm16 & 0x3F) as u32;
    let offset = ((simm16 >> 6) & 0x1F) as u32;
    let width = (((simm16 >> 11) & 0x1F) + 1) as u32;
    (id, offset, width)
}

/// Bits of `MODE[1:0]` (sp) and `MODE[3:2]` (dp/f16) touched by a `s_setreg*` `SIMM16`.
/// Returns `(sp_mask, dp_mask)` (each 2 bits) and `offset`. Non-`MODE` registers return `(0, 0, 0)`.
fn hwreg_mode_round_masks(simm16: u16) -> (u32, u32, u32) {
    let (id, offset, width) = decode_hwreg(simm16);
    if id != 1 {
        return (0, 0, 0);
    }
    let covers = |b: u32| b >= offset && b < offset + width;
    let sp = (covers(0) as u32) | ((covers(1) as u32) << 1);
    let dp = (covers(2) as u32) | ((covers(3) as u32) << 1);
    (sp, dp, offset)
}

/// Immediate value written to `MODE[base +: 2]` (masked to `mask`) by `s_setreg_imm32_b32`.
fn hwreg_imm_field(imm: u32, offset: u32, base: u32, mask: u32) -> u32 {
    let mut v = 0u32;
    if mask & 1 != 0 {
        v |= (imm >> (base - offset)) & 1;
    }
    if mask & 2 != 0 {
        v |= ((imm >> (base + 1 - offset)) & 1) << 1;
    }
    v
}

/// Instructions whose f16 output rounding obeys `MODE[3:2]` (`FP_ROUND_MODE_DP`).
fn is_f16_rounded_op(name: &str) -> bool {
    matches!(
        name,
        "v_interp_p2_f16"
            | "v_cvt_f16_f32"
            | "v_fma_mixlo_f16"
            | "v_fma_mixhi_f16"
            | "v_pk_add_f16"
            | "v_pk_mul_f16"
            | "v_pk_fma_f16"
            | "v_pk_fmac_f16"
            | "v_pk_min_f16"
            | "v_pk_max_f16"
            | "v_add_f16"
            | "v_sub_f16"
            | "v_subrev_f16"
            | "v_mul_f16"
            | "v_min_f16"
            | "v_max_f16"
            | "v_fma_f16"
            | "v_med3_f16"
            | "v_min3_f16"
            | "v_max3_f16"
            | "v_frexp_mant_f16"
            | "v_mad_f16"
            | "v_fmac_f16"
            | "v_fmamk_f16"
            | "v_fmaak_f16"
            | "v_ldexp_f16"
            | "v_rcp_f16"
            | "v_rsq_f16"
            | "v_sqrt_f16"
            | "v_log_f16"
            | "v_exp_f16"
            | "v_sin_f16"
            | "v_cos_f16"
            | "v_fract_f16"
            | "v_floor_f16"
            | "v_ceil_f16"
            | "v_trunc_f16"
            | "v_rndne_f16"
            | "v_cvt_f16_u16"
            | "v_cvt_f16_i16"
    )
}

/// Instructions whose f64 rounding obeys `MODE[3:2]` (`FP_ROUND_MODE_DP`).
fn is_f64_rounded_op(name: &str) -> bool {
    name.contains("_f64")
        && !name.starts_with("v_cmp_")
        && !name.starts_with("v_cmpx_")
        && !matches!(name, "v_cvt_f32_f64" | "v_cvt_i32_f64" | "v_cvt_u32_f64" | "v_frexp_exp_i32_f64")
}

/// HLSL expression converting float expression `e` to a 16-bit half (`0..=0xFFFF`).
fn to_f16_expr(dynamic_rm: bool, e: &str) -> String {
    if dynamic_rm {
        to_f16_rounded_expr(true, e)
    } else {
        format!("(f32tof16({e}) & 0xFFFFu)")
    }
}

/// Round an exact binary32 input in the selected RDNA mode. Legacy `f32tof16`
/// uses round-toward-zero. The default RDNA mode is round-to-nearest-even.
fn to_f16_rounded_expr(dynamic_rm: bool, e: &str) -> String {
    let rm = if dynamic_rm { "xcl_fp16_rm" } else { "0u" };
    format!("f32_to_f16_rm({e}, {rm})")
}

impl Em {
    pub fn new(opt: Options) -> Self {
        Em {
            opt,
            out: String::new(),
            indent: 1,
            sgprs: BTreeSet::new(),
            vgprs: BTreeSet::new(),
            unsupported: BTreeMap::new(),
            unsupported_why: BTreeMap::new(),
            ignored: BTreeMap::new(),
            tmp: 0,
            dpp_enable: None,
            ps_modes: BTreeMap::new(),
            ps_done: false,
            uses_lds: false,
            uses_scratch: false,
            x64_words: 1,
            x64_whole_live: BTreeMap::new(),
            upper: None,
            cur_pc: 0,
            static_ss: BTreeMap::new(),
            static_used: Vec::new(),
            max_vgpr: 0,
            has_f16_round_op: false,
            has_f64_round_op: false,
            dynamic_f16_rm: false,
        }
    }

    /// Pre-scan the program for f16/f64 rounding-sensitive ops and non-default `MODE[3:2]` writes.
    /// Shaders that set `MODE[3:2]` without f16/f64 ops pay zero HLSL overhead; shaders that set
    /// `MODE[3:2]` with f16 ops (and no f64 ops) track `xcl_fp16_rm` dynamically.
    pub fn configure_round_modes(&mut self, insts: &[Decoded], stop_at_endpgm: bool) {
        let mut writes_nonzero_dp = false;
        for d in insts {
            let name = d.inst.name.to_ascii_lowercase();
            if stop_at_endpgm && matches!(name.as_str(), "s_endpgm" | "s_endpgm_saved" | "s_endpgm_ordered_ps_done") {
                break;
            }
            if is_f16_rounded_op(&name) {
                self.has_f16_round_op = true;
            }
            if is_f64_rounded_op(&name) {
                self.has_f64_round_op = true;
            }
            match name.as_str() {
                "s_round_mode" => {
                    let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) as u16;
                    if (simm >> 2) & 3 != 0 {
                        writes_nonzero_dp = true;
                    }
                }
                "s_setreg_imm32_b32" => {
                    let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) as u16;
                    let (_, dp_mask, off) = hwreg_mode_round_masks(simm);
                    if dp_mask != 0 && hwreg_imm_field(d.literal.unwrap_or(0), off, 2, dp_mask) != 0 {
                        writes_nonzero_dp = true;
                    }
                }
                "s_setreg_b32" => {
                    let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) as u16;
                    let (_, dp_mask, _) = hwreg_mode_round_masks(simm);
                    if dp_mask != 0 {
                        writes_nonzero_dp = true;
                    }
                }
                _ => {}
            }
        }
        self.dynamic_f16_rm = writes_nonzero_dp && self.has_f16_round_op && !self.has_f64_round_op;
    }

    pub fn take(&mut self) -> String {
        std::mem::take(&mut self.out)
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn t(&mut self) -> String {
        self.tmp += 1;
        format!("t{}", self.tmp)
    }

    /// A wave64 program on a GPU with 32-lane waves only: each console wave runs as two
    /// hardware wave32 halves that exchange masks and lane values (`PRELUDE_X64`,
    /// docs/wave64-on-wave32.md).
    fn x64(&self) -> bool {
        self.opt.wave64 && self.opt.split_wave64 && matches!(self.opt.mode, Mode::Kernel | Mode::Compute(_) | Mode::Mesh(_) | Mode::Geometry(_))
    }

    /// `ballot` of the program's wave (both halves' words in the split form).
    fn ballot_fn(&self) -> &'static str {
        if self.x64() {
            "xcl_x_ballot"
        } else {
            "ballot"
        }
    }

    /// Lane read whose source lane the ISA keeps in the reader's 32-lane half (DPP16, DPP8,
    /// ds_swizzle, v_permlane(x)16).
    fn rla_half_fn(&self) -> &'static str {
        if self.x64() {
            "xcl_x_rla_half"
        } else {
            "rla"
        }
    }

    /// Lane read with the same lane index in every lane (v_readlane, v_readfirstlane).
    fn rla_uniform_fn(&self) -> &'static str {
        if self.x64() {
            "xcl_x_rla_uniform"
        } else {
            "rla"
        }
    }

    fn act(&self) -> &'static str {
        // no short-circuit `?:` (HLSL 2021 makes it a branch – one per instruction at -O0)
        if self.opt.wave64 {
            "lanebit(exec_lo, exec_hi)"
        } else {
            "(((exec_lo >> lane_id) & 1u) != 0u)"
        }
    }

    fn unsupported(&mut self, d: &Decoded, why: &str) {
        let m = rdna2::mnemonic(d);
        *self.unsupported_why.entry(format!("{m}: {why}")).or_default() += 1;
        *self.unsupported.entry(m).or_default() += 1;
        let txt = rdna2::format(d);
        self.line(&format!("/* UNSUPPORTED ({why}): {txt} */"));
    }

    // ---------------------------------------------------------------- registers

    fn sreg(&mut self, code: u64) -> String {
        match sname(code) {
            Some(n) => {
                self.sgprs.insert(code);
                n
            }
            None => "0u".into(),
        }
    }

    fn vreg(&mut self, n: u64) -> String {
        self.vgprs.insert(n);
        format!("v{n}")
    }

    fn mask_zero(&mut self, lo: u64) -> String {
        let l = self.sreg(lo);
        if self.opt.wave64 {
            let h = self.sreg(lo + 1);
            format!("({l} | {h}) == 0u")
        } else {
            format!("{l} == 0u")
        }
    }

    /// 32-bit value of a scalar source code (0..255) or VALU source (256+ = VGPR).
    fn code32(&mut self, code: u64, lit: Option<u32>, f16: bool) -> Option<String> {
        Some(match code {
            0..=124 | 126 | 127 => self.sreg(code),
            125 => "0u".into(),
            128..=192 => format!("{}u", code - 128),
            193..=208 => hex((-((code - 192) as i64)) as u32),
            240..=248 => {
                if f16 {
                    hex(f16_const(code)?)
                } else {
                    hex(f32_const(code)?.to_bits())
                }
            }
            251 => format!("(uint)({})", self.mask_zero(106)),
            252 => format!("(uint)({})", self.mask_zero(126)),
            253 => "scc".into(),
            255 => hex(lit?),
            256..=511 => self.vreg(code - 256),
            // flat apertures (shared/private base and limit): the flat memory model has none
            235..=238 => "0u".into(),
            // POPS exiting wave id: every overlapping earlier wave has already left, so the
            // ordered-section wait passes at once (no ROV ordering yet – known limitation)
            239 => "0x7FFFFFFFu".into(),
            _ => return None,
        })
    }

    /// 64-bit value (lo, hi) of a scalar source code.
    /// In RDNA2 ISA, a 32-bit literal (`code == 255`) goes in the high 32 bits for f64
    /// operands (`{0, lit}`) and in the low 32 bits for integer/bitmask operands (`{lit, 0}`).
    fn code64(&mut self, code: u64, lit: Option<u32>, f64: bool) -> Option<(String, String)> {
        Some(match code {
            0..=123 | 126 => (self.sreg(code), self.sreg(code + 1)),
            125 => ("0u".into(), "0u".into()),
            128..=192 => (format!("{}u", code - 128), "0u".into()),
            193..=208 => (hex((-((code - 192) as i64)) as u32), hex(0xFFFF_FFFF)),
            248 => (hex(0x6DC9_C882), hex(0x3FC4_5F30)), // 1/(2*pi) as a double
            240..=247 => {
                let b = (f32_const(code)? as f64).to_bits();
                (hex(b as u32), hex((b >> 32) as u32))
            }
            251 => (format!("(uint)({})", self.mask_zero(106)), "0u".into()),
            252 => (format!("(uint)({})", self.mask_zero(126)), "0u".into()),
            253 => ("scc".into(), "0u".into()),
            255 => if f64 { ("0u".into(), hex(lit?)) } else { (hex(lit?), "0u".into()) },
            256..=510 => (self.vreg(code - 256), self.vreg(code - 255)),
            _ => return None,
        })
    }

    fn field(d: &Decoded, f: &str) -> u64 {
        rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f)
    }

    /// Value of an input operand as one 32-bit expression.
    fn read(&mut self, d: &Decoded, o: &Operand, f16: bool) -> Option<String> {
        match o.field {
            Some("LITERAL") => Some(hex(d.literal?)),
            Some(f) => {
                let v = Self::field(d, f);
                let width = d.enc.field_width(f);
                if o.ty.starts_with("OPR_VGPR") && width <= 8 {
                    return Some(self.vreg(v));
                }
                match o.ty {
                    "OPR_SIMM16" => Some(format!("{}u", v)),
                    "OPR_SIMM32" => Some(hex(d.literal?)),
                    _ if width >= 9 => self.code32(v, d.literal, f16),
                    _ => self.code32(v, d.literal, f16),
                }
            }
            None => match o.ty {
                "OPR_VCC" => Some(self.sreg(106)),
                "OPR_EXEC" | "OPR_SDST_EXEC" => Some(self.sreg(126)),
                "OPR_SDST_M0" => Some(self.sreg(124)),
                "OPR_SSRC_SPECIAL_SCC" => Some("scc".into()),
                _ => None,
            },
        }
    }

    fn read64(&mut self, d: &Decoded, o: &Operand, f64: bool) -> Option<(String, String)> {
        match o.field {
            Some(f) => {
                let v = Self::field(d, f);
                if o.ty.starts_with("OPR_VGPR") && d.enc.field_width(f) <= 8 {
                    return Some((self.vreg(v), self.vreg(v + 1)));
                }
                self.code64(v, d.literal, f64)
            }
            None => match o.ty {
                "OPR_VCC" => Some((self.sreg(106), self.sreg(107))),
                "OPR_EXEC" | "OPR_SDST_EXEC" => Some((self.sreg(126), self.sreg(127))),
                _ => None,
            },
        }
    }

    fn dst_of(d: &Decoded, o: &Operand) -> Dst {
        match o.field {
            Some(f) => {
                let v = Self::field(d, f);
                match o.ty {
                    "OPR_EXEC" => Dst::S(126),
                    "OPR_VCC" => Dst::S(106),
                    t if t.starts_with("OPR_VGPR") => Dst::V(v),
                    _ if v == 125 => Dst::Null,
                    _ if v > 127 => Dst::Null,
                    _ => Dst::S(v),
                }
            }
            None => match o.ty {
                "OPR_VCC" => Dst::S(106),
                "OPR_EXEC" => Dst::S(126),
                _ => Dst::Null,
            },
        }
    }

    /// Explicit operands: (outputs, inputs) in spec order.
    fn ops(d: &Decoded) -> (Vec<&'static Operand>, Vec<&'static Operand>) {
        let mut outs = Vec::new();
        let mut ins = Vec::new();
        for o in d.ienc.operands {
            if o.implicit && !(o.field.is_none() && o.ty == "OPR_VCC") {
                continue;
            }
            if o.output {
                outs.push(o);
            } else if o.input {
                ins.push(o);
            }
        }
        (outs, ins)
    }

    fn write32(&mut self, dst: &Dst, val: &str) {
        match dst {
            // select, not `if`: straight-line code (a branch per VALU write was more than the
            // driver's compiler could take on a 15 000-instruction console shader)
            Dst::V(n) => {
                let v = self.vreg(*n);
                let a = self.act();
                self.line(&format!("{v} = select({a}, {val}, {v});"));
            }
            Dst::S(n) => {
                let s = self.sreg(*n);
                self.line(&format!("{s} = {val};"));
            }
            Dst::Null => {}
        }
    }

    fn write64(&mut self, dst: &Dst, lo: &str, hi: &str) {
        match dst {
            Dst::V(n) => {
                let (a, b) = (self.vreg(*n), self.vreg(*n + 1));
                let act = self.act();
                self.line(&format!("{a} = select({act}, {lo}, {a}); {b} = select({act}, {hi}, {b});"));
            }
            Dst::S(n) => {
                let (a, b) = (self.sreg(*n), self.sreg(*n + 1));
                self.line(&format!("{a} = {lo}; {b} = {hi};"));
            }
            Dst::Null => {}
        }
    }

    /// Lane mask from a per-lane condition (inactive lanes give 0) into a register pair.
    fn write_mask(&mut self, dst: &Dst, cond: &str) {
        let a = self.act();
        let m = self.t();
        let mut ballot = self.ballot_fn();
        // no exchange for a mask that is only ever read lane by lane (the half's own word is
        // enough), that goes nowhere, or that is 0 in every lane anyway
        let local = match *dst {
            Dst::S(n) => n < 126 && self.x64_whole_live.get(&self.cur_pc).is_some_and(|live| live & (3u128 << n) == 0),
            Dst::Null => true,
            Dst::V(_) => false,
        };
        if self.x64() && (local || cond == "false") {
            ballot = "xcl_x_ballot_local";
        }
        self.line(&format!("uint2 {m} = {ballot}(and({a}, ({cond})));"));
        if self.opt.wave64 {
            self.write64(dst, &format!("{m}.x"), &format!("{m}.y"));
        } else {
            self.write32(dst, &format!("{m}.x"));
        }
    }

    fn lane_of(&mut self, lo: &str, hi: &str) -> String {
        if self.opt.wave64 {
            format!("lanebit({lo}, {hi})")
        } else {
            format!("lanebit({lo}, 0u)")
        }
    }

    // ---------------------------------------------------------------- one instruction

    pub fn inst(&mut self, d: &Decoded, next: u64) -> Flow {
        let name = d.inst.name.to_ascii_lowercase();
        let text = rdna2::format(d);
        self.line(&format!("// {:04x}: {text}", d.pc));
        self.cur_pc = d.pc;
        // a far jump (`rdna2::resolve_far_jumps`) is an unconditional branch
        if let Some(t) = d.far_target {
            return Flow::Jump(t);
        }
        // geometry mode: the ES's s_setpc_b64 continues into the GS code (ES alone: the end)
        if name == "s_setpc_b64" {
            if let Mode::Geometry(g) = &self.opt.mode {
                return match g.gs_start {
                    Some(s) if s > 0 && self.x64() => Flow::JumpExpr(format!("select(xcl_es_phase, 0xFFFFFFFFu, {s}u)")),
                    Some(s) if s > 0 => Flow::Jump(s),
                    _ => Flow::End,
                };
            }
        }
        // s_sendmsg is a no-op except GS_ALLOC_REQ (mesh) and GS emit/cut (geometry)
        let mesh_msg = name == "s_sendmsg" && matches!(self.opt.mode, Mode::Mesh(_) | Mode::Geometry(_));
        if NOOPS.contains(&name.as_str()) && !mesh_msg {
            *self.ignored.entry(name).or_default() += 1;
            return Flow::Next;
        }
        let r = match d.enc.name {
            "ENC_SOPP" => return self.sopp(d, &name, next),
            "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" => self.salu(d, &name),
            "ENC_SMEM" => self.smem(d, &name),
            "ENC_MUBUF" | "ENC_MTBUF" => self.mubuf(d, &name),
            "ENC_EXP" => self.export(d),
            "ENC_DS" => self.ds(d, &name),
            "ENC_VINTRP" => self.interp(d, &name),
            "ENC_MIMG" => self.mimg(d, &name),
            "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" => self.flat(d, &name),
            e if e.starts_with("ENC_VOP") || e.starts_with("VOP") => self.valu(d, &name),
            _ => Err("encoding"),
        };
        if let Err(why) = r {
            self.unsupported(d, why);
        }
        Flow::Next
    }

    fn sopp(&mut self, d: &Decoded, name: &str, _next: u64) -> Flow {
        match name {
            "s_endpgm" | "s_endpgm_saved" | "s_endpgm_ordered_ps_done" => Flow::End,
            "s_barrier" => {
                if self.x64() && matches!(self.opt.mode, Mode::Geometry(_)) {
                    // split ES+GS: the GS phase runs in the lower hardware wave alone (one console
                    // wave: nothing to wait for), a group barrier there would wait for the other
                    self.line("if (xcl_es_phase) GroupMemoryBarrierWithGroupSync();");
                } else if matches!(self.opt.mode, Mode::Kernel | Mode::Mesh(_) | Mode::Geometry(_) | Mode::Compute(_)) {
                    self.line("GroupMemoryBarrierWithGroupSync();");
                }
                Flow::Next
            }
            "s_sendmsg" if matches!(self.opt.mode, Mode::Geometry(_)) => {
                // MSG_GS (2): op 1 cut, 2 emit, 3 emit + cut (stream 0); GS_DONE (3) – nothing to do
                let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0);
                let (msg, op, stream) = (simm & 0xF, (simm >> 4) & 3, (simm >> 8) & 3);
                if let (2, 0, Mode::Geometry(g)) = (msg, stream, &self.opt.mode) {
                    let maxt = g.max_vertex_count.saturating_sub(2).max(1);
                    let act = self.act();
                    if op & 2 != 0 {
                        // strip triangle j = (j, j+1, j+2), odd j with the first two swapped
                        self.line(&format!(
                            "if ({act}) {{ uint k = xcl_emitted; if (xcl_strip >= 2u && xcl_ntri < {maxt}u) {{ xcl_tri[xcl_ntri] = ((xcl_strip & 1u) == 0u) ? ((k - 2u) | ((k - 1u) << 10) | (k << 20)) : ((k - 1u) | ((k - 2u) << 10) | (k << 20)); xcl_ntri++; }} xcl_emitted++; xcl_strip++; }}"
                        ));
                    }
                    if op & 1 != 0 {
                        self.line(&format!("if ({act}) xcl_strip = 0u;"));
                    }
                }
                Flow::Next
            }
            "s_sendmsg" if matches!(self.opt.mode, Mode::Mesh(_)) => {
                // GS_ALLOC_REQ: M0 = primitives << 12 | vertices (scalar: whatever EXEC is)
                if rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) & 0xF == 9 {
                    let m0 = self.sreg(124);
                    self.line(&format!(
                        "if (lane_id == 0u) {{ res.Store(xcl_rec * 4u, {m0} & 0x3FFu); res.Store(xcl_rec * 4u + 4u, ({m0} >> 12) & 0x7FFu); }}"
                    ));
                }
                Flow::Next
            }
            "s_round_mode" => {
                // MODE[3:0] = SIMM16[3:0]: [1:0] = f32 (SP), [3:2] = f64/f16 (DP).
                // 0 = RNE (HLSL default), 1 = +inf, 2 = -inf, 3 = RTZ.
                let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) as u16;
                let sp = (simm & 3) as u32;
                let dp = ((simm >> 2) & 3) as u32;
                if sp != 0 {
                    self.unsupported(d, "non-default f32 round mode");
                } else if dp != 0 && self.has_f64_round_op {
                    self.unsupported(d, "non-default f64 round mode");
                } else if self.dynamic_f16_rm {
                    self.line(&format!("xcl_fp16_rm = {dp}u;"));
                } else {
                    *self.ignored.entry(name.to_string()).or_default() += 1;
                }
                Flow::Next
            }
            _ => {
                let Some(t) = rdna2::branch_target(d) else {
                    if NOOPS.contains(&name) {
                        return Flow::Next;
                    }
                    self.unsupported(d, "SOPP");
                    return Flow::Next;
                };
                match name {
                    "s_branch" => Flow::Jump(t),
                    "s_cbranch_scc0" => Flow::Cond("scc == 0u".into(), t),
                    "s_cbranch_scc1" => Flow::Cond("scc != 0u".into(), t),
                    "s_cbranch_vccz" => Flow::Cond(self.mask_zero(106), t),
                    "s_cbranch_vccnz" => Flow::Cond(format!("!({})", self.mask_zero(106)), t),
                    "s_cbranch_execz" => Flow::Cond(self.mask_zero(126), t),
                    "s_cbranch_execnz" => Flow::Cond(format!("!({})", self.mask_zero(126)), t),
                    _ => {
                        self.unsupported(d, "conditional branch");
                        Flow::Next
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- scalar ALU

    fn setreg(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) as u16;
        let (sp_mask, dp_mask, off) = hwreg_mode_round_masks(simm);
        if sp_mask == 0 && dp_mask == 0 {
            *self.ignored.entry(name.to_string()).or_default() += 1;
            return Ok(());
        }
        if name == "s_setreg_imm32_b32" {
            let imm = d.literal.ok_or("literal")?;
            let sp = hwreg_imm_field(imm, off, 0, sp_mask);
            let dp = hwreg_imm_field(imm, off, 2, dp_mask);
            if sp != 0 {
                return Err("non-default f32 round mode");
            }
            if dp_mask != 0 && dp != 0 && self.has_f64_round_op {
                return Err("non-default f64 round mode");
            }
            if dp_mask != 0 && self.dynamic_f16_rm {
                if dp_mask == 3 {
                    self.line(&format!("xcl_fp16_rm = {dp}u;"));
                } else {
                    let keep = (!dp_mask) & 3;
                    self.line(&format!("xcl_fp16_rm = (xcl_fp16_rm & {keep}u) | {dp}u;"));
                }
            } else {
                *self.ignored.entry(name.to_string()).or_default() += 1;
            }
            return Ok(());
        }
        if sp_mask != 0 {
            return Err("dynamic f32 round mode");
        }
        if self.has_f64_round_op {
            return Err("dynamic f64 round mode");
        }
        if self.dynamic_f16_rm {
            let sd = Self::field(d, "SDST");
            let src = self.code32(sd, None, false).ok_or("register")?;
            let stmt = match dp_mask {
                3 => {
                    let sh = 2 - off;
                    let expr = if sh == 0 { format!("({src} & 3u)") } else { format!("(({src} >> {sh}u) & 3u)") };
                    format!("xcl_fp16_rm = {expr};")
                }
                1 => {
                    let sh = 2 - off;
                    let bit = if sh == 0 { format!("({src} & 1u)") } else { format!("(({src} >> {sh}u) & 1u)") };
                    format!("xcl_fp16_rm = (xcl_fp16_rm & 2u) | {bit};")
                }
                _ => format!("xcl_fp16_rm = (xcl_fp16_rm & 1u) | (({src} & 1u) << 1);"),
            };
            self.line(&stmt);
        } else {
            *self.ignored.entry(name.to_string()).or_default() += 1;
        }
        Ok(())
    }

    fn salu(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        if matches!(name, "s_setreg_imm32_b32" | "s_setreg_b32") {
            return self.setreg(d, name);
        }
        if matches!(name, "s_movrels_b32" | "s_movrels_b64" | "s_movreld_b32" | "s_movreld_b64" | "s_movrelsd_2_b32") {
            return self.smovrel(d, name);
        }
        let (outs, ins) = Self::ops(d);
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        let simm = || rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) as u16;
        let base = name.trim_end_matches("_b32").trim_end_matches("_b64");
        // SOPK: SDST is also the first source
        if d.enc.name == "ENC_SOPK" {
            let k_s = simm() as i16 as i32 as u32;
            let k_u = simm() as u32;
            let sd = Self::field(d, "SDST");
            let cur = self.code32(sd, None, false).ok_or("register")?;
            let dst = Dst::S(sd);
            match name {
                "s_movk_i32" => self.write32(&dst, &hex(k_s)),
                "s_cmovk_i32" => self.write32(&dst, &format!("select(scc != 0u, {}, {cur})", hex(k_s))),
                "s_addk_i32" => {
                    let t = self.t();
                    self.line(&format!("uint {t} = {cur} + {};", hex(k_s)));
                    self.line(&format!("scc = ((({cur} ^ {t}) & ({} ^ {t})) >> 31);", hex(k_s)));
                    self.write32(&dst, &t);
                }
                "s_mulk_i32" => self.write32(&dst, &format!("(uint)((int){cur} * {})", k_s as i32)),
                _ if name.starts_with("s_cmpk_") => {
                    let (op, signed) = cmp_op(&name[7..]).ok_or("comparison")?;
                    let (a, b) = if signed { (format!("(int){cur}"), format!("{}", k_s as i32)) } else { (cur.clone(), format!("{k_u}u")) };
                    self.line(&format!("scc = (uint)({a} {op} {b});"));
                }
                _ => return Err("SOPK"),
            }
            return Ok(());
        }
        if d.enc.name == "ENC_SOPC" {
            if matches!(name, "s_bitcmp0_b64" | "s_bitcmp1_b64") {
                let (a0, a1) = self.read64(d, ins[0], false).ok_or("op")?;
                let b = self.read(d, ins[1], false).ok_or("op")?;
                let bit = format!("((select(({b} & 32u) == 0u, {a0}, {a1}) >> ({b} & 31u)) & 1u)");
                if name == "s_bitcmp0_b64" {
                    self.line(&format!("scc = (uint)({bit} == 0u);"));
                } else {
                    self.line(&format!("scc = {bit};"));
                }
                return Ok(());
            }
            let a64 = name.ends_with("_u64");
            if a64 {
                let (a, b) = (self.read64(d, ins[0], false).ok_or("op")?, self.read64(d, ins[1], false).ok_or("op")?);
                let op = match name {
                    "s_cmp_eq_u64" => "==",
                    "s_cmp_lg_u64" => "!=",
                    _ => return Err("SOPC"),
                };
                let j = if op == "==" { "&&" } else { "||" };
                self.line(&format!("scc = (uint)({} {op} {} {j} {} {op} {});", a.0, b.0, a.1, b.1));
                return Ok(());
            }
            let (a, b) = (self.read(d, ins[0], false).ok_or("op")?, self.read(d, ins[1], false).ok_or("op")?);
            if let Some(rest) = name.strip_prefix("s_cmp_") {
                let (op, signed) = cmp_op(rest).ok_or("comparison")?;
                let (a, b) = if signed { (format!("(int){a}"), format!("(int){b}")) } else { (a, b) };
                self.line(&format!("scc = (uint)({a} {op} {b});"));
                return Ok(());
            }
            match name {
                "s_bitcmp0_b32" => self.line(&format!("scc = (uint)((({a} >> ({b} & 31u)) & 1u) == 0u);")),
                "s_bitcmp1_b32" => self.line(&format!("scc = (({a} >> ({b} & 31u)) & 1u);")),
                _ => return Err("SOPC"),
            }
            return Ok(());
        }
        // SOP1 / SOP2
        let saveexec = name.contains("saveexec") || name.contains("wrexec");
        if saveexec {
            let op = |a: &str, e: &str| -> Option<String> {
                Some(match base.split('_').nth(1)? {
                    "and" => format!("{a} & {e}"),
                    "or" => format!("{a} | {e}"),
                    "xor" => format!("{a} ^ {e}"),
                    "andn2" => format!("{a} & ~{e}"),
                    "orn2" => format!("{a} | ~{e}"),
                    "nand" => format!("~({a} & {e})"),
                    "nor" => format!("~({a} | {e})"),
                    "xnor" => format!("~({a} ^ {e})"),
                    "andn1" => format!("~{a} & {e}"),
                    "orn1" => format!("~{a} | {e}"),
                    _ => return None,
                })
            };
            if !self.opt.wave64 || name.ends_with("_b32") {
                let s0 = self.read(d, ins[0], false).ok_or("op")?;
                let el = self.sreg(126);
                let nl = op(&s0, &el).ok_or("saveexec")?;
                let (t0, o0) = (self.t(), self.t());
                self.line(&format!("uint {o0} = {el};"));
                self.line(&format!("uint {t0} = {nl};"));
                self.line(&format!("{el} = {t0};"));
                self.line(&format!("scc = (uint)({t0} != 0u);"));
                let v = if name.contains("wrexec") { t0 } else { o0 };
                self.write32(&dst, &v);
            } else {
                let s0 = self.read64(d, ins[0], false).ok_or("op")?;
                let (el, eh) = (self.sreg(126), self.sreg(127));
                let (nl, nh) = (op(&s0.0, &el).ok_or("saveexec")?, op(&s0.1, &eh).ok_or("saveexec")?);
                let (t0, t1, o0, o1) = (self.t(), self.t(), self.t(), self.t());
                self.line(&format!("uint {o0} = {el}; uint {o1} = {eh};"));
                self.line(&format!("uint {t0} = {nl}; uint {t1} = {nh};"));
                self.line(&format!("{el} = {t0}; {eh} = {t1};"));
                self.line(&format!("scc = (uint)(({t0} | {t1}) != 0u);"));
                let (a, b) = if name.contains("wrexec") { (t0, t1) } else { (o0, o1) };
                self.write64(&dst, &a, &b);
            }
            return Ok(());
        }
        if name == "s_swappc_b64" || name == "s_setpc_b64" {
            // s_swappc_b64 s[n:n+1], s[n:n+1] at the start of a VS: call of the fetch shader
            // the console driver builds from the input layout; here the attributes are
            // already in their VGPRs (kernel mode) – other indirect jumps are not supported
            let (sd, s0) = (Self::field(d, "SDST"), Self::field(d, "SSRC0"));
            if name == "s_swappc_b64" && sd == s0 {
                self.line("/* fetch shader call – the input attributes are already in VGPRs */");
                *self.ignored.entry("s_swappc_b64 (fetch shader)".into()).or_default() += 1;
                return Ok(());
            }
            return Err("indirect branch");
        }
        if name == "s_getpc_b64" {
            // the code region is addressed from 0 in this model
            let pc = d.pc + d.len as u64;
            self.write64(&dst, &hex(pc as u32), "0u");
            return Ok(());
        }
        if matches!(name, "s_bitset0_b64" | "s_bitset1_b64") {
            let i = self.read(d, ins[0], false).ok_or("op")?;
            let (lo, hi) = match &dst {
                Dst::S(n) => (self.sreg(*n), self.sreg(*n + 1)),
                _ => ("0u".into(), "0u".into()),
            };
            let mask = format!("(1u << ({i} & 31u))");
            let (nl, nh) = if name == "s_bitset1_b64" {
                (format!("{lo} | select(({i} & 32u) == 0u, {mask}, 0u)"), format!("{hi} | select(({i} & 32u) != 0u, {mask}, 0u)"))
            } else {
                (format!("{lo} & ~select(({i} & 32u) == 0u, {mask}, 0u)"), format!("{hi} & ~select(({i} & 32u) != 0u, {mask}, 0u)"))
            };
            self.write64(&dst, &nl, &nh);
            return Ok(());
        }
        if wide {
            return self.salu64(d, name, &dst, &ins);
        }
        let a = match ins.first() {
            Some(o) => self.read(d, o, false).ok_or("op")?,
            None => String::new(),
        };
        let b = match ins.get(1) {
            Some(o) => self.read(d, o, false).ok_or("op")?,
            None => String::new(),
        };
        let t = self.t();
        // (result expression, SCC expression over the result `t` or None)
        let (res, scc): (String, Option<String>) = match name {
            "s_mov_b32" => (a.clone(), None),
            "s_cmov_b32" => {
                let cur = match &dst {
                    Dst::S(n) => self.sreg(*n),
                    _ => "0u".into(),
                };
                (format!("select(scc != 0u, {a}, {cur})"), None)
            }
            "s_not_b32" => (format!("~{a}"), Some(format!("{t} != 0u"))),
            "s_brev_b32" => (format!("reversebits({a})"), None),
            "s_bcnt0_i32_b32" => (format!("32u - countbits({a})"), Some(format!("{t} != 0u"))),
            "s_bcnt1_i32_b32" => (format!("countbits({a})"), Some(format!("{t} != 0u"))),
            "s_ff0_i32_b32" => (format!("firstbitlow(~{a})"), None),
            "s_ff1_i32_b32" => (format!("firstbitlow({a})"), None),
            "s_flbit_i32_b32" => (format!("(({a}) == 0u ? 0xFFFFFFFFu : 31u - firstbithigh({a}))"), None),
            "s_flbit_i32" => (format!("ffbh_i({a})"), None),
            "s_abs_i32" => (format!("(uint)abs((int){a})"), Some(format!("{t} != 0u"))),
            "s_sext_i32_i8" => (format!("(uint)(((int)({a} << 24)) >> 24)"), None),
            "s_sext_i32_i16" => (format!("(uint)(((int)({a} << 16)) >> 16)"), None),
            "s_and_b32" => (format!("{a} & {b}"), Some(format!("{t} != 0u"))),
            "s_or_b32" => (format!("{a} | {b}"), Some(format!("{t} != 0u"))),
            "s_xor_b32" => (format!("{a} ^ {b}"), Some(format!("{t} != 0u"))),
            "s_andn2_b32" => (format!("{a} & ~{b}"), Some(format!("{t} != 0u"))),
            "s_orn2_b32" => (format!("{a} | ~{b}"), Some(format!("{t} != 0u"))),
            "s_nand_b32" => (format!("~({a} & {b})"), Some(format!("{t} != 0u"))),
            "s_nor_b32" => (format!("~({a} | {b})"), Some(format!("{t} != 0u"))),
            "s_xnor_b32" => (format!("~({a} ^ {b})"), Some(format!("{t} != 0u"))),
            "s_add_u32" => (format!("{a} + {b}"), Some(format!("{t} < {a}"))),
            "s_sub_u32" => (format!("{a} - {b}"), Some(format!("{b} > {a}"))),
            "s_addc_u32" => (format!("{a} + {b} + scc"), Some(format!("((uint64_t){a} + (uint64_t){b} + (uint64_t)scc) > 0xFFFFFFFFull"))),
            "s_subb_u32" => (format!("{a} - {b} - scc"), Some(format!("((uint64_t){b} + (uint64_t)scc) > (uint64_t){a}"))),
            "s_add_i32" => (format!("{a} + {b}"), Some(format!("((({a} ^ {t}) & ({b} ^ {t})) >> 31) != 0u"))),
            "s_sub_i32" => (format!("{a} - {b}"), Some(format!("((({a} ^ {b}) & ({a} ^ {t})) >> 31) != 0u"))),
            "s_mul_i32" => (format!("(uint)((int){a} * (int){b})"), None),
            "s_mul_hi_u32" => (format!("mul_hi_u32({a}, {b})"), None),
            "s_mul_hi_i32" => (format!("(uint)(((int64_t)(int){a} * (int64_t)(int){b}) >> 32)"), None),
            "s_lshl_b32" => (format!("{a} << ({b} & 31u)"), Some(format!("{t} != 0u"))),
            "s_lshr_b32" => (format!("{a} >> ({b} & 31u)"), Some(format!("{t} != 0u"))),
            "s_ashr_i32" => (format!("(uint)((int){a} >> ({b} & 31u))"), Some(format!("{t} != 0u"))),
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let k = &name[6..7];
                (format!("({a} << {k}u) + {b}"), Some(format!("((((uint64_t){a}) << {k}) + (uint64_t){b}) > 0xFFFFFFFFull")))
            }
            "s_bfm_b32" => (format!("((({a} & 31u) == 0u ? 0u : (0xFFFFFFFFu >> (32u - ({a} & 31u))))) << ({b} & 31u)"), None),
            "s_bfe_u32" => (format!("sbfe_u({a}, {b})"), Some(format!("{t} != 0u"))),
            "s_bfe_i32" => (format!("(uint)sbfe_i({a}, {b})"), Some(format!("{t} != 0u"))),
            "s_min_u32" => (format!("min({a}, {b})"), Some(format!("{a} < {b}"))),
            "s_max_u32" => (format!("max({a}, {b})"), Some(format!("{a} > {b}"))),
            "s_min_i32" => (format!("(uint)min((int){a}, (int){b})"), Some(format!("(int){a} < (int){b}"))),
            "s_max_i32" => (format!("(uint)max((int){a}, (int){b})"), Some(format!("(int){a} > (int){b}"))),
            "s_cselect_b32" => (format!("select(scc != 0u, {a}, {b})"), None),
            "s_wqm_b32" => (format!("wqm({a})"), Some(format!("{t} != 0u"))),
            "s_bitset0_b32" | "s_bitset1_b32" => {
                let cur = match &dst {
                    Dst::S(n) => self.sreg(*n),
                    _ => "0u".into(),
                };
                if name == "s_bitset1_b32" {
                    (format!("{cur} | (1u << ({a} & 31u))"), None)
                } else {
                    (format!("{cur} & ~(1u << ({a} & 31u))"), None)
                }
            }
            "s_absdiff_i32" => (format!("(uint)abs((int){a} - (int){b})"), Some(format!("{t} != 0u"))),
            "s_pack_ll_b32_b16" => (format!("({a} & 0xFFFFu) | ({b} << 16)"), None),
            "s_pack_lh_b32_b16" => (format!("({a} & 0xFFFFu) | ({b} & 0xFFFF0000u)"), None),
            "s_pack_hh_b32_b16" => (format!("({a} >> 16) | ({b} & 0xFFFF0000u)"), None),
            _ => return Err("SALU"),
        };
        self.line(&format!("uint {t} = {res};"));
        if let Some(c) = scc {
            self.line(&format!("scc = (uint)({c});"));
        }
        self.write32(&dst, &t);
        Ok(())
    }

    /// Scalar relative moves: uniform M0 selects an SGPR from the encoded base. HLSL has
    /// no dynamically indexed scalar locals, so use the same bounded switch representation
    /// as v_movrel. Reads are captured before writes to preserve overlapping moves.
    fn smovrel(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let dst = Self::field(d, "SDST");
        let src = Self::field(d, "SSRC0");
        if dst > 123 || src > 123 {
            return Err("scalar movrel register");
        }
        let m0 = self.sreg(124);
        let (rd, wr, wide) = match name {
            "s_movrels_b32" => (Some(m0.clone()), None, false),
            "s_movrels_b64" => (Some(m0.clone()), None, true),
            "s_movreld_b32" => (None, Some(m0.clone()), false),
            "s_movreld_b64" => (None, Some(m0.clone()), true),
            "s_movrelsd_2_b32" => (Some(format!("({m0} & 0x3FFu)")), Some(format!("(({m0} >> 16) & 0x3FFu)")), false),
            _ => return Err("scalar movrel"),
        };
        let words = if wide { 2 } else { 1 };
        let mut vals = Vec::new();
        for word in 0..words {
            let t = self.t();
            if let Some(idx) = &rd {
                let mut cases = String::new();
                for k in 0..=123u64.saturating_sub(src + word) {
                    cases.push_str(&format!(" case {k}u: {t} = {}; break;", self.sreg(src + word + k)));
                }
                self.line(&format!("uint {t} = 0u; switch ({idx}) {{{cases} default: break; }}"));
            } else {
                let v = self.sreg(src + word);
                self.line(&format!("uint {t} = {v};"));
            }
            vals.push(t);
        }
        if let Some(idx) = &wr {
            let mut cases = String::new();
            for k in 0..=123u64.saturating_sub(dst + words - 1) {
                let writes = (0..words).map(|w| format!("{} = {};", self.sreg(dst + k + w), vals[w as usize])).collect::<Vec<_>>().join(" ");
                cases.push_str(&format!(" case {k}u: {writes} break;"));
            }
            self.line(&format!("switch ({idx}) {{{cases} default: break; }}"));
        } else {
            for word in 0..words {
                let d = self.sreg(dst + word);
                self.line(&format!("{d} = {};", vals[word as usize]));
            }
        }
        Ok(())
    }

    fn salu64(&mut self, d: &Decoded, name: &str, dst: &Dst, ins: &[&Operand]) -> Result<(), &'static str> {
        let a = match ins.first() {
            Some(o) => Some(self.read64(d, o, false).ok_or("op")?),
            None => None,
        };
        let b = match ins.get(1) {
            Some(o) => {
                // shift amounts / bfe controls are 32-bit
                if o.size == 32 {
                    let v = self.read(d, o, false).ok_or("op")?;
                    Some((v, "0u".into()))
                } else {
                    Some(self.read64(d, o, false).ok_or("op")?)
                }
            }
            None => None,
        };
        let (al, ah) = a.clone().unwrap_or_default();
        let (bl, bh) = b.clone().unwrap_or_default();
        let (t0, t1) = (self.t(), self.t());
        let nz = format!("({t0} | {t1}) != 0u");
        let bitwise = |op: &str| -> Option<(String, String)> {
            Some(match op {
                "and" => (format!("{al} & {bl}"), format!("{ah} & {bh}")),
                "or" => (format!("{al} | {bl}"), format!("{ah} | {bh}")),
                "xor" => (format!("{al} ^ {bl}"), format!("{ah} ^ {bh}")),
                "andn2" => (format!("{al} & ~{bl}"), format!("{ah} & ~{bh}")),
                "orn2" => (format!("{al} | ~{bl}"), format!("{ah} | ~{bh}")),
                "nand" => (format!("~({al} & {bl})"), format!("~({ah} & {bh})")),
                "nor" => (format!("~({al} | {bl})"), format!("~({ah} | {bh})")),
                "xnor" => (format!("~({al} ^ {bl})"), format!("~({ah} ^ {bh})")),
                _ => return None,
            })
        };
        let (lo, hi, scc): (String, String, Option<String>) = match name {
            "s_mov_b64" => (al.clone(), ah.clone(), None),
            "s_cmov_b64" => {
                let cur = match dst {
                    Dst::S(n) => (self.sreg(*n), self.sreg(*n + 1)),
                    _ => ("0u".into(), "0u".into()),
                };
                (format!("select(scc != 0u, {al}, {})", cur.0), format!("select(scc != 0u, {ah}, {})", cur.1), None)
            }
            "s_not_b64" => (format!("~{al}"), format!("~{ah}"), Some(nz.clone())),
            "s_cselect_b64" => (format!("select(scc != 0u, {al}, {bl})"), format!("select(scc != 0u, {ah}, {bh})"), None),
            "s_wqm_b64" => (format!("wqm({al})"), format!("wqm({ah})"), Some(nz.clone())),
            "s_brev_b64" => (format!("reversebits({ah})"), format!("reversebits({al})"), None),
            "s_bcnt1_i32_b64" => (format!("countbits({al}) + countbits({ah})"), "0u".into(), Some(format!("{t0} != 0u"))),
            "s_bcnt0_i32_b64" => (format!("64u - countbits({al}) - countbits({ah})"), "0u".into(), Some(format!("{t0} != 0u"))),
            "s_ff1_i32_b64" => {
                (format!("select({al} != 0u, firstbitlow({al}), select({ah} != 0u, 32u + firstbitlow({ah}), 0xFFFFFFFFu))"), "0u".into(), None)
            }
            "s_ff0_i32_b64" => {
                (format!("select(~{al} != 0u, firstbitlow(~{al}), select(~{ah} != 0u, 32u + firstbitlow(~{ah}), 0xFFFFFFFFu))"), "0u".into(), None)
            }
            "s_flbit_i32_b64" => (
                format!("select({ah} != 0u, 31u - firstbithigh({ah}), select({al} != 0u, 63u - firstbithigh({al}), 0xFFFFFFFFu))"),
                "0u".into(),
                None,
            ),
            "s_flbit_i32_i64" => {
                // signed: invert both halves when the i64 sign is set, then count from bit 63
                let (nl, nh) = (self.t(), self.t());
                self.line(&format!("uint {nl} = select((int){ah} < 0, ~{al}, {al}); uint {nh} = select((int){ah} < 0, ~{ah}, {ah});"));
                (
                    format!("select({nh} != 0u, 31u - firstbithigh({nh}), select({nl} != 0u, 63u - firstbithigh({nl}), 0xFFFFFFFFu))"),
                    "0u".into(),
                    None,
                )
            }
            "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                let v = format!("(((uint64_t){ah} << 32) | (uint64_t){al})");
                let e = match name {
                    "s_lshl_b64" => format!("({v} << ({bl} & 63u))"),
                    "s_lshr_b64" => format!("({v} >> ({bl} & 63u))"),
                    _ => format!("(uint64_t)((int64_t){v} >> ({bl} & 63u))"),
                };
                let w = self.t();
                self.line(&format!("uint64_t {w} = {e};"));
                (format!("(uint){w}"), format!("(uint)({w} >> 32)"), Some(nz.clone()))
            }
            "s_bfm_b64" => {
                let w = self.t();
                self.line(&format!(
                    "uint64_t {w} = ((({al} & 63u) == 0u) ? 0ull : (0xFFFFFFFFFFFFFFFFull >> (64u - ({al} & 63u)))) << ({bl} & 63u);"
                ));
                (format!("(uint){w}"), format!("(uint)({w} >> 32)"), None)
            }
            "s_bfe_u64" | "s_bfe_i64" => {
                let v = format!("(((uint64_t){ah} << 32) | (uint64_t){al})");
                let w = self.t();
                let o = self.t();
                let x = self.t();
                let r = self.t();
                if name == "s_bfe_u64" {
                    self.line(&format!("uint {w} = ({bl} >> 16) & 0x7Fu; uint {o} = {bl} & 63u; uint64_t {x} = {v} >> {o};"));
                    self.line(&format!(
                        "uint64_t {r} = ({w} == 0u) ? 0ull : (({w} >= 64u) ? {x} : ({x} & (0xFFFFFFFFFFFFFFFFull >> (64u - {w}))));"
                    ));
                } else {
                    self.line(&format!("uint {w} = ({bl} >> 16) & 0x7Fu; uint {o} = {bl} & 63u; int64_t {x} = ((int64_t){v}) >> {o};"));
                    self.line(&format!(
                        "uint64_t {r} = ({w} == 0u) ? 0ull : (({w} >= 64u) ? (uint64_t){x} : (uint64_t)(((int64_t)({x} << ((64u - {w}) & 63u))) >> ((64u - {w}) & 63u)));"
                    ));
                }
                (format!("(uint){r}"), format!("(uint)({r} >> 32)"), Some(nz.clone()))
            }
            _ => {
                let op = name.trim_start_matches("s_").trim_end_matches("_b64");
                match bitwise(op) {
                    Some((l, h)) => (l, h, Some(nz.clone())),
                    None => return Err("SALU64"),
                }
            }
        };
        self.line(&format!("uint {t0} = {lo}; uint {t1} = {hi};"));
        if let Some(c) = scc {
            self.line(&format!("scc = (uint)({c});"));
        }
        if matches!(name, "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" | "s_ff0_i32_b64" | "s_ff1_i32_b64" | "s_flbit_i32_b64" | "s_flbit_i32_i64") {
            self.write32(dst, &t0);
        } else {
            self.write64(dst, &t0, &t1);
        }
        Ok(())
    }

    // ---------------------------------------------------------------- memory (flat model)

    fn smem(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        if name == "s_memtime" || name == "s_memrealtime" {
            // no clock in this model: a constant time (profiling deltas come out 0)
            let sdata = Self::field(d, "SDATA");
            let (lo, hi) = (self.sreg(sdata), self.sreg(sdata + 1));
            self.line(&format!("{lo} = 0u; {hi} = 0u;"));
            return Ok(());
        }
        let count: usize = match name.trim_start_matches("s_buffer_load_").trim_start_matches("s_load_") {
            "dword" => 1,
            "dwordx2" => 2,
            "dwordx4" => 4,
            "dwordx8" => 8,
            "dwordx16" => 16,
            _ => return Err("SMEM"),
        };
        let sdata = Self::field(d, "SDATA");
        let sbase = Self::field(d, "SBASE");
        let imm = rdna2::field_value(d.enc, &d.words, "OFFSET").unwrap_or(0) as u32;
        let soff = rdna2::field_value(d.enc, &d.words, "SOFFSET").unwrap_or(125);
        let soff = if soff == 125 { "0u".to_string() } else { self.code32(soff, None, false).ok_or("op")? };
        let t = self.t();
        let mut vals = Vec::new();
        if name.starts_with("s_buffer_load") {
            // V#: base = dw0 | dw1[15:0] << 32, stride = dw1[29:16], num_records = dw2
            let (d0, d1, d2) = (self.sreg(sbase), self.sreg(sbase + 1), self.sreg(sbase + 2));
            self.window(&d1);
            self.line(&format!("uint {t}_off = {} + {soff};", hex(imm)));
            self.line(&format!("uint {t}_st = ({d1} >> 16) & 0x3FFFu;"));
            self.line(&format!("uint {t}_lim = select({t}_st == 0u, {d2}, {d2} * {t}_st);"));
            self.line(&format!("uint {t}_a = ({d0} + {t}_off) & ~3u;"));
            // Branch-free: a dword past num_records reads offset 0 (always in the window) and
            // comes out 0. HLSL 2021's `?:` is a branch, and with several memory windows every
            // access is a switch between them (`windowed`): one per dword made DXC several
            // times slower on big programs. Four dwords share one switch (`xcl_loadx4`).
            if count == 1 {
                self.line(&format!("bool {t}_in = {t}_off < {t}_lim;"));
                self.line(&format!("uint {t}_0 = select({t}_in, mem.Load(select({t}_in, {t}_a, 0u)), 0u);"));
                vals.push(format!("{t}_0"));
            } else {
                for g in 0..count.div_ceil(4) {
                    let n = (count - 4 * g).min(4);
                    let o = (0..4).map(|k| format!("{}u", 16 * g + 4 * k.min(n - 1))).collect::<Vec<_>>().join(", ");
                    self.line(&format!("bool4 {t}_in{g} = {t}_off + uint4({o}) < {t}_lim;"));
                    self.line(&format!("uint4 {t}_g{g} = select({t}_in{g}, xcl_loadx4(select({t}_in{g}, {t}_a + uint4({o}), 0u)), 0u);"));
                    vals.extend(["x", "y", "z", "w"][..n].iter().map(|c| format!("{t}_g{g}.{c}")));
                }
            }
        } else {
            let imm = ((imm << 11) as i32 >> 11) as u32; // 21-bit signed
            let b = self.sreg(sbase);
            let hi = self.sreg(sbase + 1);
            self.window(&hi);
            // the dwords are contiguous: Load2 / Load4 (one window switch per four dwords)
            self.line(&format!("uint {t}_a = ({b} + {} + {soff}) & ~3u;", hex(imm)));
            match count {
                1 => {
                    self.line(&format!("uint {t}_0 = mem.Load({t}_a);"));
                    vals.push(format!("{t}_0"));
                }
                2 => {
                    self.line(&format!("uint2 {t}_g0 = mem.Load2({t}_a);"));
                    vals.extend([format!("{t}_g0.x"), format!("{t}_g0.y")]);
                }
                _ => {
                    for g in 0..count / 4 {
                        self.line(&format!("uint4 {t}_g{g} = mem.Load4({t}_a + {}u);", 16 * g));
                        vals.extend(["x", "y", "z", "w"].iter().map(|c| format!("{t}_g{g}.{c}")));
                    }
                }
            }
        }
        // split wave64, a coherent load (GLC / DLC: memory other waves may be writing): both
        // halves take the lower half's values, so they cannot see two different ones
        let coherent = ["GLC", "DLC"].iter().any(|f| rdna2::field_value(d.enc, &d.words, f) == Some(1));
        if self.x64() && coherent {
            self.x64_words = self.x64_words.max(vals.len());
            for (i, v) in vals.iter().enumerate() {
                self.line(&format!("xcl_x_put({i}u, {v});"));
            }
            let get: Vec<String> = vals.iter().enumerate().map(|(i, v)| format!("{v} = xcl_x_get({i}u);")).collect();
            self.line(&format!("if (xcl_x_meet() && xcl_half != 0u) {{ {} }}", get.join(" ")));
        }
        for (i, v) in vals.iter().enumerate() {
            let s = self.sreg(sdata + i as u64);
            self.line(&format!("{s} = {v};"));
        }
        Ok(())
    }

    /// Several memory windows: the accesses that follow go to the window of the address
    /// whose high dword (bits 32..63) is `hi` (`xcl_w`, read by the `mem` accessors).
    fn window(&mut self, hi: &str) {
        if self.opt.mem_windows > 1 {
            self.line(&format!("xcl_w = ({hi}) & 3u;"));
        }
    }

    /// `window` for a FLAT / global address: SADDR pair (global with SADDR) or the 64-bit
    /// VGPR address.
    fn flat_window(&mut self, global: bool, saddr: u64, addr: u64) {
        if self.opt.mem_windows > 1 {
            let hi = if global && saddr != 0x7D && saddr != 0x7F { self.sreg(saddr + 1) } else { self.vreg(addr + 1) };
            self.window(&hi);
        }
    }

    /// The sampler of the MIMG instruction at `pc`: a static sampler when its S# is constant
    /// (register `static_sampler_base + i`, space 1), else the sampler heap entry S# word 3
    /// [11:0] names (XCL shader ABI).
    fn sampler_ref(&mut self, pc: u64, comparison: bool, s3: &str) -> String {
        let Some(&w) = self.static_ss.get(&pc) else { return format!("SamplerDescriptorHeap[NonUniformResourceIndex({s3} & 0xFFFu)]") };
        let i = match self.static_used.iter().position(|&x| x == (w, comparison)) {
            Some(i) => i,
            None => {
                self.static_used.push((w, comparison));
                self.static_used.len() - 1
            }
        };
        format!("xcl_ss{}", self.opt.static_sampler_base as usize + i)
    }

    /// MUBUF / MTBUF: raw loads and stores, typed (format) loads.
    fn mubuf(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        if name.starts_with("buffer_atomic_") {
            return self.buffer_atomic(d, name);
        }
        // sub-dword accesses: (bytes, signed load?, store?)
        let small = match name {
            "buffer_load_ubyte" => Some((1u32, false, false)),
            "buffer_load_sbyte" => Some((1, true, false)),
            "buffer_load_ushort" => Some((2, false, false)),
            "buffer_load_sshort" => Some((2, true, false)),
            "buffer_store_byte" => Some((1, false, true)),
            "buffer_store_short" => Some((2, false, true)),
            _ => None,
        };
        if let Some((n, signed, store, half)) = d16_small(name.trim_start_matches("buffer_")) {
            return self.mubuf_d16(d, n, signed, store, half);
        }
        let d16_format = buffer_format_d16(name);
        let (kind, count) = match (small, d16_format) {
            (Some(_), _) => (BufKind::Load, 1),
            (None, Some((kind, components, _))) => (kind, (components + 1) / 2),
            (None, None) => buffer_op(name).ok_or("MUBUF")?,
        };
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        if f("LDS") != 0 || f("TFE") != 0 {
            return Err("MUBUF lds/tfe");
        }
        let (vdata, vaddr, srsrc) = (f("VDATA"), f("VADDR"), Self::field(d, "SRSRC"));
        let soff = self.code32(f("SOFFSET"), None, false).ok_or("op")?;
        let (offen, idxen) = (f("OFFEN") != 0, f("IDXEN") != 0);
        let (d0, d1, d2, d3) = (self.sreg(srsrc), self.sreg(srsrc + 1), self.sreg(srsrc + 2), self.sreg(srsrc + 3));
        self.window(&d1);
        let index = if idxen { self.vreg(vaddr) } else { "0u".into() };
        let voff = if offen { self.vreg(vaddr + idxen as u64) } else { "0u".into() };
        let t = self.t();
        self.line(&format!("uint {t}_stride = ({d1} >> 16) & 0x3FFFu;"));
        self.line(&format!("uint {t}_off = {voff} + {}u;", f("OFFSET")));
        self.line(&format!("uint {t}_a = {d0} + {soff} + {t}_off + {t}_stride * {index};"));
        let act = self.act();
        let in_range = |bytes: String| format!("select({t}_stride == 0u, {t}_off + {bytes} <= {d2}, {index} < {d2})");
        if let Some((n, signed, store)) = small {
            let ok = in_range(format!("{n}u"));
            let v = self.vreg(vdata);
            if store {
                self.line(&format!("if (and({act}, {ok})) st_bytes({t}_a, uint4({v}, 0u, 0u, 0u), {n}u);"));
            } else {
                let bits = 8 * n;
                let val = if signed {
                    format!("(uint)(((int)(ld_raw({t}_a).x << {}u)) >> {}u)", 32 - bits, 32 - bits)
                } else {
                    format!("(ld_raw({t}_a).x & {}u)", (1u64 << bits) - 1)
                };
                self.line(&format!("if ({act}) {v} = {ok} ? {val} : 0u;"));
            }
            return Ok(());
        }
        match kind {
            BufKind::Load => {
                // branch-free (see `smem`): a lane that is inactive or out of range reads offset
                // 0 and keeps its register / gets 0; one window switch per four dwords
                self.line(&format!("bool {t}_act = {act};"));
                if count == 1 {
                    let ok = in_range("4u".into());
                    let v = self.vreg(vdata);
                    self.line(&format!("bool {t}_ok = {ok};"));
                    self.line(&format!("uint {t}_v = mem.Load(select(and({t}_act, {t}_ok), {t}_a & ~3u, 0u));"));
                    self.line(&format!("{v} = select({t}_act, select({t}_ok, {t}_v, 0u), {v});"));
                } else {
                    let oks: Vec<String> = (0..4).map(|i| in_range(format!("{}u", 4 * i.min(count - 1) + 4))).collect();
                    let o = (0..4).map(|i| format!("{}u", 4 * i.min(count - 1))).collect::<Vec<_>>().join(", ");
                    self.line(&format!("bool4 {t}_ok = bool4({});", oks.join(", ")));
                    self.line(&format!("uint4 {t}_v = xcl_loadx4(select(and({t}_act, {t}_ok), ({t}_a & ~3u) + uint4({o}), 0u));"));
                    for i in 0..count {
                        let v = self.vreg(vdata + i);
                        let c = ["x", "y", "z", "w"][i as usize];
                        self.line(&format!("{v} = select({t}_act, select({t}_ok.{c}, {t}_v.{c}, 0u), {v});"));
                    }
                }
            }
            BufKind::Store => {
                let mut st = Vec::new();
                for i in 0..count {
                    let ok = in_range(format!("{}u", 4 * i + 4));
                    let v = self.vreg(vdata + i);
                    st.push(format!("if ({ok}) mem.Store(({t}_a + {}u) & ~3u, {v});", 4 * i));
                }
                self.line(&format!("if ({act}) {{ {} }}", st.join(" ")));
            }
            BufKind::FormatLoad => {
                // format from the instruction (tbuffer) or from the V# (dword3 [18:12]); DST_SEL from the V#
                let fmt = if d.enc.name == "ENC_MTBUF" { format!("{}u", f("FORMAT")) } else { format!("(({d3} >> 12) & 0x7Fu)") };
                self.line(&format!("uint {t}_info = FMT_INFO[min({fmt}, 77u)];"));
                self.line(&format!("bool {t}_int = ((({t}_info >> 3) & 7u) == 4u) || ((({t}_info >> 3) & 7u) == 5u);"));
                self.line(&format!("bool {t}_ok = and(({t}_info & 7u) != 0u, {});", in_range(format!("fmt_bytes({t}_info)"))));
                self.line(&format!("uint4 {t}_r = ld_raw({t}_a);"));
                self.line(&format!(
                    "uint4 {t}_c = uint4(fmt_comp({t}_r, {t}_info, 0u), fmt_comp({t}_r, {t}_info, 1u), fmt_comp({t}_r, {t}_info, 2u), fmt_comp({t}_r, {t}_info, 3u));"
                ));
                if let Some((_, components, high)) = d16_format {
                    let half = |component: u64| {
                        let selected = format!("dst_sel({t}_c, ({d3} >> {}u) & 7u, {t}_int)", 3 * component);
                        format!("select({t}_int, {selected} & 0xFFFFu, f32tof16(asfloat({selected})))")
                    };
                    let mut assigns = Vec::new();
                    if high {
                        let v = self.vreg(vdata);
                        assigns.push(format!("{v} = ({v} & 0xFFFFu) | ({t}_ok ? ({} << 16) : 0u);", half(0)));
                    } else {
                        for i in 0..count {
                            let lo = half(2 * i);
                            let hi = if 2 * i + 1 < components { half(2 * i + 1) } else { "0u".into() };
                            let v = self.vreg(vdata + i);
                            assigns.push(format!("{v} = {t}_ok ? ({lo} | ({hi} << 16)) : 0u;"));
                        }
                    }
                    self.line(&format!("if ({act}) {{ {} }}", assigns.join(" ")));
                    return Ok(());
                }
                let mut assigns = Vec::new();
                for i in 0..count {
                    let v = self.vreg(vdata + i);
                    assigns.push(format!("{v} = select({t}_ok, dst_sel({t}_c, ({d3} >> {}u) & 7u, {t}_int), 0u);", 3 * i));
                }
                self.line(&format!("if ({act}) {{ {} }}", assigns.join(" ")));
            }
            BufKind::FormatStore => {
                let fmt = if d.enc.name == "ENC_MTBUF" { format!("{}u", f("FORMAT")) } else { format!("(({d3} >> 12) & 0x7Fu)") };
                self.line(&format!("uint {t}_info = FMT_INFO[min({fmt}, 77u)];"));
                self.line(&format!("bool {t}_ok = and(({t}_info & 7u) != 0u, {});", in_range(format!("fmt_bytes({t}_info)"))));
                if let Some((_, components, high)) = d16_format {
                    self.line(&format!("uint {t}_ty = ({t}_info >> 3) & 7u;"));
                    let mut source = |component: u64| {
                        let v = self.vreg(vdata + if high { 0 } else { component / 2 });
                        let shift = if high || component & 1 != 0 { " >> 16" } else { "" };
                        let h = format!("(({v}{shift}) & 0xFFFFu)");
                        format!("select({t}_ty == 4u, {h}, select({t}_ty == 5u, (uint)(((int)({h} << 16)) >> 16), asuint(f16tof32({h}))))")
                    };
                    let vals: Vec<String> = (0..4).map(|i| if i < components { source(i) } else { "0u".into() }).collect();
                    self.line(&format!("if (and({act}, {t}_ok)) st_bytes({t}_a, fmt_pack(uint4({}), {t}_info), fmt_bytes({t}_info));", vals.join(", ")));
                    return Ok(());
                }
                let vals: Vec<String> = (0..4).map(|i| if i < count { self.vreg(vdata + i) } else { "0u".into() }).collect();
                self.line(&format!("if (and({act}, {t}_ok)) st_bytes({t}_a, fmt_pack(uint4({}), {t}_info), fmt_bytes({t}_info));", vals.join(", ")));
            }
        }
        Ok(())
    }

    fn export(&mut self, d: &Decoded) -> Result<(), &'static str> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let (en, tgt, compr) = (f("EN"), f("TGT"), f("COMPR") != 0);
        let srcs: Vec<String> = (0..4).map(|i| self.vreg(f(&format!("VSRC{i}")))).collect();
        if !matches!(self.opt.mode, Mode::Kernel) {
            return self.export_graphics(d, en, tgt, compr, &srcs);
        }
        let act = self.act();
        let mut st = Vec::new();
        for c in 0..4u64 {
            let val = if compr {
                let r = &srcs[(c / 2) as usize];
                if c % 2 == 0 {
                    format!("({r} & 0xFFFFu)")
                } else {
                    format!("({r} >> 16)")
                }
            } else {
                srcs[c as usize].clone()
            };
            let bit = if compr { 1 << (c / 2 * 2) | 2 << (c / 2 * 2) } else { 1 << c };
            if en & bit == 0 {
                continue;
            }
            st.push(format!("res.Store(({}u + (({tgt}u * {}u + xcl_hw_lane) * 4u + {c}u)) * 4u, {val});", KERNEL_EXPORTS, KERNEL_LANES));
        }
        if !st.is_empty() {
            self.line(&format!("if ({act}) {{ {} }}", st.join(" ")));
        }
        Ok(())
    }

    /// Raw component `c` of an export (compressed: 16-bit halves of two registers).
    fn exp_comp(srcs: &[String], compr: bool, c: usize) -> String {
        if compr {
            let r = &srcs[c / 2];
            if c % 2 == 0 {
                format!("({r} & 0xFFFFu)")
            } else {
                format!("({r} >> 16)")
            }
        } else {
            srcs[c].clone()
        }
    }

    fn export_graphics(&mut self, d: &Decoded, en: u64, tgt: u64, compr: bool, srcs: &[String]) -> Result<(), &'static str> {
        let act = self.act();
        let done = rdna2::field_value(d.enc, &d.words, "DONE") == Some(1);
        let enabled = |c: usize| if compr { en >> (c / 2 * 2) & 3 != 0 } else { en >> c & 1 == 1 };
        let mut st = Vec::new();
        match (&self.opt.mode, tgt) {
            (Mode::Vertex(_), 12) => {
                for c in 0..4 {
                    if enabled(c) {
                        st.push(format!("xo.pos[{c}] = asfloat({});", Self::exp_comp(srcs, compr, c)));
                    }
                }
            }
            (Mode::Vertex(_), 13..=15) => {} // misc vector / clip distances: SV_ClipDistance comes from its param
            (Mode::Vertex(vs), 32..=63) => {
                // param N is a register: channel c goes to component c of every element there
                let k = (tgt - 32) as usize;
                let params = ranked(&vs.outputs, true);
                let at = vs_param_register(vs, k).map(|r| elements_at(&vs.outputs, true, r)).unwrap_or_default();
                if at.is_empty() {
                    self.line("/* param without an OSG1 element (dummy) */");
                    return Ok(());
                }
                for j in at {
                    let e = params[j];
                    let n = out_comps(e);
                    for c in 0..n {
                        if enabled(c) {
                            let v = Self::exp_comp(srcs, compr, c);
                            let v = if e.component_type == 3 { format!("asfloat({v})") } else { format!("{v}") };
                            let comp = if n == 1 { String::new() } else { format!("[{c}]") };
                            st.push(format!("xo.p{j}{comp} = {v};"));
                        }
                    }
                }
            }
            (Mode::Pixel(ps), 0..=7) => {
                let Some(e) = ps.outputs.iter().find(|e| e.system_value == 64 && e.index as u64 == tgt) else {
                    self.line("/* export to an undeclared SV_Target */");
                    if done {
                        self.ps_done = true;
                        self.line(&format!("if ({act}) xcl_live = true;"));
                    }
                    return Ok(());
                };
                for c in 0..4 {
                    if !enabled(c) {
                        continue;
                    }
                    let raw = Self::exp_comp(srcs, compr, c);
                    let v = match (e.component_type, compr) {
                        (3, true) => format!("f16tof32({raw})"),
                        (3, false) => format!("asfloat({raw})"),
                        (2, true) => format!("(((int)({raw} << 16)) >> 16)"),
                        (2, false) => format!("(int){raw}"),
                        (_, _) => raw,
                    };
                    st.push(format!("xo.t{tgt}[{c}] = {v};"));
                }
            }
            (Mode::Pixel(ps), 8) => {
                if en & 1 == 1 && ps.outputs.iter().any(|e| matches!(e.system_value, 65 | 67 | 68)) {
                    st.push(format!("xo.xcl_depth = asfloat({});", srcs[0]));
                }
            }
            (Mode::Pixel(_), 9) => {}
            (Mode::Mesh(_), 12) => {
                for c in 0..4 {
                    if enabled(c) {
                        st.push(format!("res.Store((xcl_vtx + {c}u) * 4u, {});", Self::exp_comp(srcs, compr, c)));
                    }
                }
            }
            (Mode::Mesh(_), 13..=15) => {} // misc vector / clip distances (from their params)
            (Mode::Mesh(ms), 32..=63) => {
                // param N is a register: its record slot is that of every element there (the
                // pass-through reads an element's slot as the whole register, as for the GS)
                let k = (tgt - 32) as usize;
                let at = param_registers(&ms.outputs, true).get(k).map(|&r| elements_at(&ms.outputs, true, r)).unwrap_or_default();
                if at.is_empty() {
                    self.line("/* param without an OSG1 element (dummy) */");
                    return Ok(());
                }
                for j in at {
                    for c in 0..4 {
                        if enabled(c) {
                            st.push(format!("res.Store((xcl_vtx + {}u) * 4u, {});", 4 * (j + 1) + c, Self::exp_comp(srcs, compr, c)));
                        }
                    }
                }
            }
            (Mode::Mesh(_), 20) => st.push(format!("res.Store(xcl_prm * 4u, {});", srcs[0])),
            (Mode::Mesh(_), 9) => {}
            _ => return Err("export target"),
        }
        if !st.is_empty() {
            self.line(&format!("if ({act}) {{ {} }}", st.join(" ")));
        }
        if done && matches!(self.opt.mode, Mode::Pixel(_)) {
            self.ps_done = true;
            self.line(&format!("if ({act}) xcl_live = true;"));
        }
        Ok(())
    }

    /// (vdst, I/J VGPR or interp_mov parameter, attribute, channel)
    fn interp_fields(d: &Decoded) -> (u64, u64, u64, u64) {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        if d.enc.name == "ENC_VINTRP" {
            (f("VDST"), f("VSRC"), f("ATTR"), f("ATTRCHAN"))
        } else {
            let s1 = f("SRC1");
            (f("VDST"), if s1 >= 256 { s1 - 256 } else { s1 }, f("SRC0") & 0x3F, f("SRC0") >> 6 & 3)
        }
    }

    /// v_interp_*: attributes come interpolated from the PC rasteriser (PS mode).
    fn interp(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let Mode::Pixel(ps) = &self.opt.mode else { return Err("interpolation outside PS") };
        let (vdst, ij, attr, chan) = Self::interp_fields(d);
        let Some(slot) = ps_attribute_slot(ps, attr as usize, chan as usize) else {
            return Err("attribute outside ISG1");
        };
        let src = format!("xi.a{slot}[{chan}]");
        let act = self.act();
        match name {
            "v_interp_p1_f32" | "v_interp_p1ll_f16" | "v_interp_p1lv_f16" => {
                self.line("/* first half of the interpolation – p2 produces the value */");
            }
            "v_interp_p2_f32" => {
                let v = self.vreg(vdst);
                // (float): the value of an int / uint input, not its bits (see v_interp_mov_f32)
                self.line(&format!("{v} = select({act}, asuint((float){src}), {v});"));
            }
            "v_interp_mov_f32" => {
                if ij != 2 {
                    return Err("v_interp_mov p10/p20");
                }
                let v = self.vreg(vdst);
                // The value of an int / uint input, converted (#107 made it a bit copy, which is what
                // the ISA does): XDefiant's UI PS reads its RenderType (an int input) this way and
                // compares it with 1 / 2 – with the bits its text took those branches and drew
                // nothing, with the converted value (never 1 or 2) it draws the text (red; whether
                // the console's is another colour is not known). Open: why those branches draw
                // nothing (they sample the font with gradients from ds_swizzle derivatives).
                self.line(&format!("{v} = select({act}, asuint((float){src}), {v});"));
            }
            "v_interp_p2_f16" => {
                let v = self.vreg(vdst);
                let hi = rdna2::field_value(d.enc, &d.words, "OP_SEL").unwrap_or(0) >> 3 & 1 == 1;
                let h = to_f16_expr(self.dynamic_f16_rm, &format!("(float){src}"));
                let e = if hi { format!("({v} & 0xFFFFu) | ({h} << 16)") } else { format!("({v} & 0xFFFF0000u) | {h}") };
                self.line(&format!("{v} = select({act}, {e}, {v});"));
            }
            _ => return Err("interpolation"),
        }
        Ok(())
    }

    // ---------------------------------------------------------------- images (MIMG)

    /// MIMG → SM 6.6 dynamic resources. XCL image descriptors (T#, 8 dwords) keep the GFX10
    /// layout, but BASE_ADDRESS (word 0) holds the index of the SRV in the CBV/SRV/UAV heap
    /// (UAV at +1); FORMAT (word 1 [28:20]) selects the typed view (float/uint/sint).
    /// Sampler descriptors (S#, 4 dwords) hold the sampler heap index in BORDER_COLOR_PTR
    /// (word 3 [11:0]). Address layout with A16 (16-bit bias/coordinates/lod, 32-bit offset,
    /// compare and gradients) as in AnyPS5's `componentShape` (GPL-2.0, ported 2026-11-20).
    /// `float3 {t}_o{sfx}`: an MIMG texel offset (`o`, three 6-bit signed fields) in the texture
    /// coordinates of mip `level` – offset / that level's size (array layers do not move).
    fn texel_offset(&mut self, t: &str, dim: u64, o: &str, level: &str, sfx: &str) {
        let (w, h, z, n) = (format!("{t}_w{sfx}"), format!("{t}_h{sfx}"), format!("{t}_d{sfx}"), format!("{t}_n{sfx}"));
        let dims = match dim {
            0 => format!("{t}_T.GetDimensions({level}, {w}, {n});"),
            1 => format!("{t}_T.GetDimensions({level}, {w}, {h}, {n});"),
            2 => format!("{t}_T.GetDimensions({level}, {w}, {h}, {z}, {n});"),
            4 => format!("{t}_T.GetDimensions({level}, {w}, {z}, {n}); {z} = 1u;"),
            _ => format!("{t}_T.GetDimensions({level}, {w}, {h}, {z}, {n}); {z} = 1u;"),
        };
        self.line(&format!("uint {w} = 1u, {h} = 1u, {z} = 1u, {n}; {dims}"));
        self.line(&format!(
            "float3 {t}_o{sfx} = float3((float)(((int)({o} << 26)) >> 26) / (float){w}, (float)(((int)({o} << 18)) >> 26) / (float){h}, (float)(((int)({o} << 10)) >> 26) / (float){z});"
        ));
    }

    fn mimg(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        if f("TFE") != 0 || f("LWE") != 0 {
            return Err("MIMG tfe/lwe");
        }
        let (a16, d16) = (f("A16") == 1, f("D16") == 1);
        let dim = f("DIM");
        let dmask = f("DMASK");
        let (vdata, srsrc, ssamp) = (f("VDATA"), Self::field(d, "SRSRC"), Self::field(d, "SSAMP"));
        let nsa = f("NSA");
        // non-sequential addresses: VADDR, then VADDRA, VADDRB … (4 per extra dword)
        let nsa_enc = rdna2::encoding(&format!("MIMG_NSA{nsa}"));
        let addr_reg = |j: u64| -> u64 {
            match nsa_enc {
                Some(e) if nsa > 0 && j > 0 => {
                    let letter = (b'A' + (j - 1).min(25) as u8) as char;
                    rdna2::field_value(e, &d.words, &format!("VADDR{letter}")).unwrap_or(0)
                }
                _ => f("VADDR") + j,
            }
        };
        let base = name.trim_start_matches("image_");
        if base == "bvh_intersect_ray" || base == "bvh64_intersect_ray" {
            // ray tracing: one BVH node per instruction (public RDNA2 / RADV node format)
            let r = |s: &mut Self, j: u64| s.vreg(addr_reg(j));
            let wide = base == "bvh64_intersect_ray";
            let node = r(self, 0);
            let node_hi = if wide { Some(r(self, 1)) } else { None };
            let k = if wide { 2 } else { 1 };
            let fl = |s: &mut Self, j: u64| format!("asfloat({})", r(s, j));
            let extent = fl(self, k);
            let origin = format!("float3({}, {}, {})", fl(self, k + 1), fl(self, k + 2), fl(self, k + 3));
            let (dir, inv) = if a16 {
                // A16: dir.xy | dir.z inv.x | inv.yz as half pairs
                let (h0, h1, h2) = (r(self, k + 4), r(self, k + 5), r(self, k + 6));
                (
                    format!("float3(f16tof32({h0}), f16tof32({h0} >> 16), f16tof32({h1}))"),
                    format!("float3(f16tof32({h1} >> 16), f16tof32({h2}), f16tof32({h2} >> 16))"),
                )
            } else {
                (
                    format!("float3({}, {}, {})", fl(self, k + 4), fl(self, k + 5), fl(self, k + 6)),
                    format!("float3({}, {}, {})", fl(self, k + 7), fl(self, k + 8), fl(self, k + 9)),
                )
            };
            let (d0, d1) = (self.sreg(srsrc), self.sreg(srsrc + 1));
            let a = self.t();
            let t = self.t();
            let act = self.act();
            // descriptor: 48-bit base address in 256-byte units (word 0 + word 1 [7:0]),
            // box sorting = word 1 bit 31; node offset is 32-bit (bvh) or 64-bit (bvh64) in 8-byte units
            let off = match node_hi {
                Some(hi) => format!("((((uint64_t){hi} << 32) | (uint64_t)({node} & ~7u)) << 3)"),
                None => format!("((uint64_t)({node} & ~7u) << 3)"),
            };
            self.line(&format!("uint64_t {a} = ((((uint64_t)({d1} & 0xFFu) << 32) | (uint64_t){d0}) << 8) + {off};"));
            self.window(&format!("(uint)({a} >> 32)"));
            self.line(&format!("uint4 {t} = bvh_intersect({node} & 7u, (uint){a}, {extent}, {origin}, {dir}, {inv}, ({d1} >> 31) != 0u);"));
            let st: Vec<String> = (0..4).map(|c| format!("{} = {t}.{};", self.vreg(vdata + c), ["x", "y", "z", "w"][c as usize])).collect();
            self.line(&format!("if ({act}) {{ {} }}", st.join(" ")));
            return Ok(());
        }
        // body coordinates per DIM: 1D, 2D, 3D, CUBE, 1D_ARRAY, 2D_ARRAY, 2D_MSAA, 2D_MSAA_ARRAY
        let body = [1u64, 2, 3, 3, 2, 3, 3, 4][dim as usize];
        let sampling = base.starts_with("sample") || base.starts_with("gather4");
        let rest = base.trim_start_matches("sample").trim_start_matches("gather4");
        let mods: Vec<&str> = if sampling { rest.split('_').filter(|x| !x.is_empty()).collect() } else { Vec::new() };
        // `cd` (coarse derivatives) samples like `d`: the gradients are the instruction's own
        let (has_cl, has_o) = (mods.contains(&"cl"), mods.contains(&"o"));
        let (has_b, has_c, has_d, has_l, has_lz) =
            (mods.contains(&"b"), mods.contains(&"c"), mods.contains(&"d") || mods.contains(&"cd"), mods.contains(&"l"), mods.contains(&"lz"));
        let grad_n = if has_d { [1u64, 2, 3, 2, 1, 2, 2, 2][dim as usize] } else { 0 };
        // address components: (16-bit?, starts a dword?)
        let h = if a16 { 16 } else { 32 };
        let mut shape: Vec<(u32, bool)> = Vec::new();
        let (mut i_bias, mut i_cmp, mut i_grad, mut i_body, mut i_extra, mut i_clamp) = (None, None, 0usize, 0usize, None, None);
        if sampling {
            // address order: offset, bias, z-compare, derivatives, body, lod / clamp
            if has_o {
                shape.push((32, true));
            }
            if has_b {
                i_bias = Some(shape.len());
                shape.push((h, !a16));
            }
            if has_c {
                i_cmp = Some(shape.len());
                shape.push((32, true));
            }
            i_grad = shape.len();
            for _ in 0..2 * grad_n {
                shape.push((32, true));
            }
            i_body = shape.len();
            for k in 0..body {
                shape.push((h, !a16 || (grad_n != 0 && k == 0)));
            }
            if has_l {
                i_extra = Some(shape.len());
                shape.push((h, !a16));
            }
            if has_cl {
                i_clamp = Some(shape.len());
                shape.push((h, !a16));
            }
        } else if base == "get_resinfo" {
            i_extra = Some(0);
            shape.push((32, true));
        } else {
            i_body = 0;
            for _ in 0..body {
                shape.push((h, !a16));
            }
            if base.ends_with("_mip") {
                i_extra = Some(shape.len());
                shape.push((h, !a16));
            }
        }
        let mut pos: Vec<(u64, Option<u64>)> = Vec::new();
        let mut cursor = 0u64; // in 16-bit units
        for &(w, starts) in &shape {
            if (w == 32 || starts) && cursor % 2 == 1 {
                cursor += 1;
            }
            pos.push((cursor / 2, (w == 16).then_some(cursor % 2)));
            cursor += if w == 32 { 2 } else { 1 };
        }
        let compf = |s: &mut Self, i: usize| -> String {
            let (dw, half) = pos[i];
            let v = s.vreg(addr_reg(dw));
            match half {
                None => format!("asfloat({v})"),
                Some(hh) => format!("f16tof32(({v} >> {}u) & 0xFFFFu)", 16 * hh),
            }
        };
        let compu = |s: &mut Self, i: usize| -> String {
            let (dw, half) = pos[i];
            let v = s.vreg(addr_reg(dw));
            match half {
                None => v,
                Some(hh) => format!("(({v} >> {}u) & 0xFFFFu)", 16 * hh),
            }
        };
        let t = self.t();
        let tw0 = self.sreg(srsrc);
        let tw1 = self.sreg(srsrc + 1);
        let act = self.act();
        let ps = matches!(self.opt.mode, Mode::Pixel(_));
        let comps: Vec<u64> = (0..4).filter(|c| dmask >> c & 1 == 1).collect();
        // write the selected result components (`r` is a uint4 of raw bits); D16 packs halves
        let write_result = |s: &mut Self, r: &str, n: usize, sel: &dyn Fn(usize) -> u64, half: &dyn Fn(&str) -> String| {
            // selects, not `if (act)`: a branch per image instruction is work for DXC on big
            // programs, and the result is computed for every lane anyway
            let a = format!("{r}_act");
            s.line(&format!("bool {a} = {act};"));
            if d16 {
                for k in (0..n).step_by(2) {
                    let v = s.vreg(vdata + k as u64 / 2);
                    let lo = half(&format!("{r}.{}", ["x", "y", "z", "w"][sel(k) as usize]));
                    let hi = if k + 1 < n { half(&format!("{r}.{}", ["x", "y", "z", "w"][sel(k + 1) as usize])) } else { "0u".into() };
                    s.line(&format!("{v} = select({a}, ({lo} & 0xFFFFu) | ({hi} << 16), {v});"));
                }
            } else {
                for k in 0..n {
                    let v = s.vreg(vdata + k as u64);
                    s.line(&format!("{v} = select({a}, {r}.{}, {v});", ["x", "y", "z", "w"][sel(k) as usize]));
                }
            }
        };
        let f16 = |e: &str| format!("f32tof16(asfloat({e}))");

        if sampling {
            if matches!(dim, 6 | 7) {
                return Err("MIMG sample MSAA");
            }
            let bias = i_bias.map(|i| compf(self, i));
            let cmp = i_cmp.map(|i| compf(self, i));
            let grads = if has_d {
                let dx: Vec<String> = (0..grad_n as usize).map(|k| compf(self, i_grad + k)).collect();
                let dy: Vec<String> = (0..grad_n as usize).map(|k| compf(self, i_grad + grad_n as usize + k)).collect();
                Some((dx, dy))
            } else {
                None
            };
            let coords: Vec<String> = (0..body as usize).map(|k| compf(self, i_body + k)).collect();
            let lod = i_extra.map(|i| compf(self, i));
            let cl = i_clamp.map(|i| compf(self, i));
            // the clamp overloads take an offset first (none for cubes)
            let zoff = match dim {
                0 | 4 => Some("0"),
                1 | 5 => Some("int2(0, 0)"),
                2 => Some("int3(0, 0, 0)"),
                _ => None,
            };
            let tail = |args: String| -> String {
                match (&cl, zoff) {
                    (Some(c), Some(z)) => format!("{args}, {z}, {c}"),
                    (Some(c), None) => format!("{args}, {c}"),
                    _ => args,
                }
            };
            let s3 = self.sreg(ssamp + 3);
            // texel offsets come from a VGPR (6-bit fields); SM 6.6 takes only immediate offsets, so
            // the coordinates move by offset / size of the sampled mip (exact for lz / integer l /
            // gather; an implicit LOD with a linear mip filter is sampled per mip further down)
            let offset = if has_o {
                if dim == 3 {
                    return Err("MIMG offset cube");
                }
                let o = compu(self, 0);
                Some(o)
            } else {
                None
            };
            let coord0 = sample_coord(dim, &coords);
            let coords_unmoved = coords.clone();
            let coords: Vec<String> = if offset.is_some() { moved_coords(dim, &coords, &format!("{t}_o")) } else { coords };
            let (ty, coord) = match dim {
                0 => ("Texture1D<float4>", coords[0].clone()),
                1 => ("Texture2D<float4>", format!("float2({}, {})", coords[0], coords[1])),
                2 => ("Texture3D<float4>", format!("float3({}, {}, {})", coords[0], coords[1], coords[2])),
                // every cube T# is a cube array view; the face coordinate is face + 8 · slice
                3 => ("TextureCubeArray<float4>", format!("float4(cube_dir({0}, {1}, {2}), cube_slice({2}))", coords[0], coords[1], coords[2])),
                4 => ("Texture1DArray<float4>", format!("float2({}, round({}))", coords[0], coords[1])),
                5 => ("Texture2DArray<float4>", format!("float3({}, {}, round({}))", coords[0], coords[1], coords[2])),
                _ => return Err("MIMG dim"),
            };
            if f("UNORM") != 0 && dim != 3 {
                return Err("MIMG unorm");
            }
            let samp = if has_c { "SamplerComparisonState" } else { "SamplerState" };
            self.line(&format!("{ty} {t}_T = ResourceDescriptorHeap[NonUniformResourceIndex({tw0})];"));
            let sampler = self.sampler_ref(d.pc, has_c, &s3);
            self.line(&format!("{samp} {t}_S = {sampler};"));
            if let Some(o) = &offset {
                let level = match (&lod, has_lz || base.starts_with("gather4"), ps && grads.is_none() && !has_c) {
                    (_, true, _) => "0u".to_string(),
                    (Some(l), _, _) => format!("(uint)max(round({l}), 0.0f)"),
                    (None, _, true) => format!("(uint)max(round({t}_T.CalculateLevelOfDetail({t}_S, {coord0})), 0.0f)"),
                    _ => "0u".to_string(),
                };
                self.texel_offset(&t, dim, o, &level, "");
            }
            let gather = base.starts_with("gather4");
            let call = if gather {
                if comps.len() != 1 {
                    return Err("gather4 dmask");
                }
                let ch = ["Red", "Green", "Blue", "Alpha"][comps[0] as usize];
                match &cmp {
                    Some(z) => format!("{t}_T.GatherCmp{ch}({t}_S, {coord}, {z})"),
                    None => format!("{t}_T.Gather{ch}({t}_S, {coord})"),
                }
            } else {
                // outside PS there are no implicit derivatives: LOD 0 (bias = LOD), raised by the clamp
                let floor = |l: String| match &cl {
                    Some(c) => format!("max({l}, {c})"),
                    None => l,
                };
                match (&cmp, &grads, &lod, &bias) {
                    (Some(z), _, _, _) if has_lz || !ps => format!("float4({t}_T.SampleCmpLevelZero({t}_S, {coord}, {z}).xxxx)"),
                    (Some(z), _, _, _) => format!("float4({t}_T.SampleCmp({t}_S, {}).xxxx)", tail(format!("{coord}, {z}"))),
                    // cube maps: RDNA takes face-space 2D gradients, HLSL wants direction-space 3D
                    // ones (and has no clamp form) – the LOD from the gradients and the face size
                    (None, Some((dx, dy)), _, _) if dim == 3 => {
                        let len = |x: &[String]| format!("length(float2({}, {}))", x[0], x.get(1).cloned().unwrap_or("0.0f".into()));
                        let size = format!("{t}_cw");
                        self.line(&format!("uint {size} = 1u, {t}_ch = 1u, {t}_ce = 1u, {t}_cn = 1u; {t}_T.GetDimensions(0u, {size}, {t}_ch, {t}_ce, {t}_cn);"));
                        let lod = format!("log2(max(max({}, {}) * (float){size}, 1e-8f))", len(dx), len(dy));
                        format!("{t}_T.SampleLevel({t}_S, {coord}, {})", floor(lod))
                    }
                    (None, Some((dx, dy)), _, _) => {
                        let v = |x: &[String]| if x.len() == 1 { x[0].clone() } else { format!("float{}({})", x.len(), x.join(", ")) };
                        format!("{t}_T.SampleGrad({t}_S, {})", tail(format!("{coord}, {}, {}", v(dx), v(dy))))
                    }
                    (None, None, Some(l), _) => format!("{t}_T.SampleLevel({t}_S, {coord}, {l})"),
                    (None, None, None, Some(b)) if ps => format!("{t}_T.SampleBias({t}_S, {})", tail(format!("{coord}, {b}"))),
                    (None, None, None, Some(b)) => format!("{t}_T.SampleLevel({t}_S, {coord}, {})", floor(b.clone())),
                    _ if has_lz || !ps => format!("{t}_T.SampleLevel({t}_S, {coord}, {})", floor("0.0f".into())),
                    // implicit LOD with an offset and a linear mip filter: the console moves each of
                    // the two blended mips by the offset in its own texels – sampled per mip and
                    // blended here. Anisotropic or point-mip samplers keep the one-mip form above.
                    _ if offset.is_some() => {
                        let o = offset.clone().unwrap_or_default();
                        let (w0, w2) = match self.static_ss.get(&d.pc) {
                            Some(w) => (format!("{}u", w[0]), format!("{}u", w[2])),
                            None => (self.sreg(ssamp), self.sreg(ssamp + 2)),
                        };
                        // the level a clamp past the last mip asks for is the last mip ({t}_n: the view's levels)
                        let lod = floor(format!("{t}_T.CalculateLevelOfDetail({t}_S, {coord0})"));
                        self.line(&format!("float {t}_L = min({lod}, (float)(max({t}_n, 1u) - 1u)); float {t}_l0 = floor({t}_L);"));
                        let l1 = format!("min((uint){t}_l0 + 1u, max({t}_n, 1u) - 1u)");
                        self.texel_offset(&t, dim, &o, &format!("(uint){t}_l0"), "a");
                        self.texel_offset(&t, dim, &o, &l1, "b");
                        let c0 = sample_coord(dim, &moved_coords(dim, &coords_unmoved, &format!("{t}_oa")));
                        let c1 = sample_coord(dim, &moved_coords(dim, &coords_unmoved, &format!("{t}_ob")));
                        self.line(&format!("float4 {t}_m0 = {t}_T.SampleLevel({t}_S, {c0}, {t}_l0);"));
                        self.line(&format!("float4 {t}_m1 = {t}_T.SampleLevel({t}_S, {c1}, (float)({l1}));"));
                        self.line(&format!("float4 {t}_s1 = {t}_T.Sample({t}_S, {});", tail(coord.clone())));
                        // S# word 2: XY_MAG_FILTER [21:20], XY_MIN_FILTER [23:22], MIP_FILTER [27:26];
                        // word 0: MAX_ANISO_RATIO [11:9] – anisotropic only with an aniso min / mag
                        // filter, as `desc::sampler_d3d12` decides. A level-0 SampleLevel filters with
                        // the mag filter, so mip 0 of a minified sample needs min = mag.
                        self.line(&format!(
                            "uint {t}_mag = (({w2}) >> 20) & 3u, {t}_min = (({w2}) >> 22) & 3u; bool {t}_pm = ((({w2}) >> 26) & 3u) == 2u && !(({t}_mag >= 2u || {t}_min >= 2u) && ((({w0}) >> 9) & 7u) != 0u) && ({t}_l0 >= 1.0f || (({t}_mag | ({t}_mag >> 1)) & 1u) == (({t}_min | ({t}_min >> 1)) & 1u));"
                        ));
                        format!("({t}_pm ? lerp({t}_m0, {t}_m1, {t}_L - {t}_l0) : {t}_s1)")
                    }
                    _ => format!("{t}_T.Sample({t}_S, {})", tail(coord.clone())),
                }
            };
            self.line(&format!("uint4 {t}_r = asuint({call});"));
            let n = if gather { 4 } else { comps.len() };
            let comps2 = comps.clone();
            let sel = move |k: usize| if gather { k as u64 } else { comps2[k] };
            write_result(self, &format!("{t}_r"), n, &sel, &f16);
            return Ok(());
        }

        if base == "get_lod" {
            if !ps {
                return Err("image_get_lod outside PS");
            }
            let coords: Vec<String> = (0..body as usize).map(|k| compf(self, i_body + k)).collect();
            let (ty, coord) = match dim {
                0 => ("Texture1D<float4>", coords[0].clone()),
                1 => ("Texture2D<float4>", format!("float2({}, {})", coords[0], coords[1])),
                2 => ("Texture3D<float4>", format!("float3({}, {}, {})", coords[0], coords[1], coords[2])),
                // a cube array's level of detail takes the direction only
                3 => ("TextureCubeArray<float4>", format!("cube_dir({}, {}, {})", coords[0], coords[1], coords[2])),
                4 => ("Texture1DArray<float4>", format!("float2({}, round({}))", coords[0], coords[1])),
                5 => ("Texture2DArray<float4>", format!("float3({}, {}, round({}))", coords[0], coords[1], coords[2])),
                _ => return Err("MIMG dim"),
            };
            let s3 = self.sreg(ssamp + 3);
            self.line(&format!("{ty} {t}_T = ResourceDescriptorHeap[NonUniformResourceIndex({tw0})];"));
            let sampler = self.sampler_ref(d.pc, false, &s3);
            self.line(&format!("SamplerState {t}_S = {sampler};"));
            self.line(&format!(
                "uint4 {t}_r = asuint(float4({t}_T.CalculateLevelOfDetail({t}_S, {coord}), {t}_T.CalculateLevelOfDetailUnclamped({t}_S, {coord}), 0.0f, 0.0f));"
            ));
            let comps2 = comps.clone();
            write_result(self, &format!("{t}_r"), comps.len(), &move |k| comps2[k], &f16);
            return Ok(());
        }

        if base == "get_resinfo" {
            let mipv = compu(self, 0);
            let (ty, call) = match dim {
                0 => ("Texture1D", format!("{t}_T.GetDimensions({mipv}, {t}_w, {t}_n); {t}_h = 1u; {t}_d = 1u;")),
                1 => ("Texture2D", format!("{t}_T.GetDimensions({mipv}, {t}_w, {t}_h, {t}_n); {t}_d = 1u;")),
                2 => ("Texture3D", format!("{t}_T.GetDimensions({mipv}, {t}_w, {t}_h, {t}_d, {t}_n);")),
                // the hardware returns the layers (6 per cube), as for a 2D array
                3 => ("TextureCubeArray", format!("{t}_T.GetDimensions({mipv}, {t}_w, {t}_h, {t}_d, {t}_n); {t}_d *= 6u;")),
                4 => ("Texture1DArray", format!("{t}_T.GetDimensions({mipv}, {t}_w, {t}_d, {t}_n); {t}_h = 1u;")),
                5 => ("Texture2DArray", format!("{t}_T.GetDimensions({mipv}, {t}_w, {t}_h, {t}_d, {t}_n);")),
                _ => return Err("get_resinfo dim"),
            };
            self.line(&format!("{ty}<float4> {t}_T = ResourceDescriptorHeap[NonUniformResourceIndex({tw0})];"));
            self.line(&format!("uint {t}_w, {t}_h, {t}_d, {t}_n; {call}"));
            self.line(&format!("uint4 {t}_r = uint4({t}_w, {t}_h, {t}_d, {t}_n);"));
            let comps2 = comps.clone();
            write_result(self, &format!("{t}_r"), comps.len(), &move |k| comps2[k], &|e: &str| e.to_string());
            return Ok(());
        }

        // integer coordinates for load / store / atomics
        let ic: Vec<String> = (0..body as usize).map(|k| compu(self, i_body + k)).collect();
        let mip = i_extra.map(|i| compu(self, i)).unwrap_or("0u".into());
        let (rty, icoord, load_arg) = match dim {
            0 => ("1D", format!("int({})", ic[0]), format!("int2({}, {mip})", ic[0])),
            1 => ("2D", format!("int2({}, {})", ic[0], ic[1]), format!("int3({}, {}, {mip})", ic[0], ic[1])),
            2 => ("3D", format!("int3({}, {}, {})", ic[0], ic[1], ic[2]), format!("int4({}, {}, {}, {mip})", ic[0], ic[1], ic[2])),
            4 => ("1DArray", format!("int2({}, {})", ic[0], ic[1]), format!("int3({}, {}, {mip})", ic[0], ic[1])),
            5 | 3 => ("2DArray", format!("int3({}, {}, {})", ic[0], ic[1], ic[2]), format!("int4({}, {}, {}, {mip})", ic[0], ic[1], ic[2])),
            6 => ("2DMS", format!("int2({}, {})", ic[0], ic[1]), String::new()),
            _ => return Err("MIMG dim"),
        };
        if dim == 3 && base.starts_with("load") {
            return Err("image_load cube");
        }
        self.line(&format!("uint {t}_cls = img_cls(({tw1} >> 20) & 0x1FFu);"));
        match base {
            "load" | "load_mip" => {
                self.line(&format!("uint4 {t}_r;"));
                let ld = if dim == 6 { format!("{t}_T.Load({icoord}, {})", ic.get(2).cloned().unwrap_or("0u".into())) } else { format!("{t}_T.Load({load_arg})") };
                let tex = |vt: &str| format!("Texture{rty}<{vt}4> {t}_T = ResourceDescriptorHeap[NonUniformResourceIndex({tw0})];");
                self.line(&format!(
                    "if ({t}_cls == 1u) {{ {} {t}_r = {ld}; }} else if ({t}_cls == 2u) {{ {} {t}_r = asuint({ld}); }} else {{ {} {t}_r = asuint({ld}); }}",
                    tex("uint"),
                    tex("int"),
                    tex("float")
                ));
                let cls = format!("{t}_cls");
                let half = move |e: &str| format!("(({cls} == 0u) ? f32tof16(asfloat({e})) : ({e} & 0xFFFFu))");
                let comps2 = comps.clone();
                write_result(self, &format!("{t}_r"), comps.len(), &move |k| comps2[k], &half);
            }
            "store" | "store_mip" => {
                if base == "store_mip" {
                    return Err("image_store_mip");
                }
                let mut vals = Vec::new();
                for c in 0..4u64 {
                    vals.push(match comps.iter().position(|&x| x == c) {
                        Some(k) if d16 => {
                            let v = self.vreg(vdata + k as u64 / 2);
                            let hpart = format!("(({v} >> {}u) & 0xFFFFu)", 16 * (k % 2));
                            format!("(({t}_cls == 0u) ? asuint(f16tof32({hpart})) : {hpart})")
                        }
                        Some(k) => self.vreg(vdata + k as u64),
                        None => "0u".into(),
                    });
                }
                let v4 = format!("uint4({})", vals.join(", "));
                let uav = |vt: &str| format!("RWTexture{rty}<{vt}4> {t}_U = ResourceDescriptorHeap[NonUniformResourceIndex({tw0} + 1u)];");
                self.line(&format!(
                    "if ({act}) {{ if ({t}_cls == 1u) {{ {} {t}_U[{icoord}] = {v4}; }} else if ({t}_cls == 2u) {{ {} {t}_U[{icoord}] = (int4){v4}; }} else {{ {} {t}_U[{icoord}] = asfloat({v4}); }} }}",
                    uav("uint"),
                    uav("int"),
                    uav("float")
                ));
            }
            _ if base.starts_with("atomic_") && matches!(&base[7..], "cmpswap" | "sub" | "inc" | "dec" | "fcmpswap" | "fmin" | "fmax") => {
                // the operations without an int view: on the uint texel (CAS loops where needed)
                let (v, cmp) = (self.vreg(vdata), self.vreg(vdata + 1));
                self.line(&format!("RWTexture{rty}<uint> {t}_U = ResourceDescriptorHeap[NonUniformResourceIndex({tw0} + 1u)];"));
                self.line(&format!("uint {t}_old = 0u;"));
                let body = rmw32(&base[7..], &At::Ref(format!("{t}_U[{icoord}]")), &v, &cmp, &format!("{t}_old"), &t).ok_or("image_atomic")?;
                self.line(&format!("if ({act}) {{ {body} }}"));
                if f("GLC") == 1 {
                    self.line(&format!("if ({act}) {v} = {t}_old;"));
                }
            }
            _ if base.starts_with("atomic_") => {
                let op = match &base[7..] {
                    "add" => "InterlockedAdd",
                    "smin" | "umin" => "InterlockedMin",
                    "smax" | "umax" => "InterlockedMax",
                    "and" => "InterlockedAnd",
                    "or" => "InterlockedOr",
                    "xor" => "InterlockedXor",
                    "swap" => "InterlockedExchange",
                    _ => return Err("image_atomic"),
                };
                let signed = base.ends_with("smin") || base.ends_with("smax");
                let v = self.vreg(vdata);
                let vt = if signed { "int" } else { "uint" };
                let val = if signed { format!("(int){v}") } else { v.clone() };
                self.line(&format!("RWTexture{rty}<{vt}> {t}_U = ResourceDescriptorHeap[NonUniformResourceIndex({tw0} + 1u)];"));
                self.line(&format!("{vt} {t}_old = 0;"));
                self.line(&format!("if ({act}) {{ {op}({t}_U[{icoord}], {val}, {t}_old); }}"));
                if f("GLC") == 1 {
                    self.line(&format!("if ({act}) {v} = (uint){t}_old;"));
                }
            }
            _ => return Err("MIMG"),
        }
        Ok(())
    }

    // ---------------------------------------------------------------- vector ALU

    /// Source/output modifiers of a VALU instruction, wherever the encoding keeps them.
    fn mods(d: &Decoded) -> Mods {
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

    /// Float view of source `i` with abs/neg.
    fn fsrc(m: &Mods, i: usize, u: &str) -> String {
        let mut s = format!("asfloat({u})");
        if m.abs[i] {
            s = format!("abs({s})");
        }
        if m.neg[i] {
            s = format!("(-{s})");
        }
        s
    }

    /// Float result with output modifiers, as u32.
    fn fres(m: &Mods, e: &str) -> String {
        format!("asuint({})", Self::out_mods(m, e))
    }

    /// The output modifiers on an f32 expression: omod, then clamp.
    fn out_mods(m: &Mods, e: &str) -> String {
        let s = match m.omod {
            1 => format!("(({e}) * 2.0f)"),
            2 => format!("(({e}) * 4.0f)"),
            3 => format!("(({e}) * 0.5f)"),
            _ => format!("({e})"),
        };
        if m.clamp {
            format!("saturate({s})")
        } else {
            s
        }
    }

    /// u32 source expressions in operand order; SGPR-pair masks come as "lo|hi".
    /// SDWA selects/extends its two sources, DPP reads src0 from another lane.
    fn sources(&mut self, d: &Decoded, ins: &[&Operand], src_f16: bool) -> Result<Vec<String>, &'static str> {
        let mut u = Vec::new();
        for o in ins {
            if (d.variant != Variant::Plain && o.field == Some("SRC0")) || (d.variant == Variant::Sdwa && o.field == Some("VSRC1")) {
                u.push("0u".into()); // replaced below
                continue;
            }
            match o.ty {
                "OPR_SREG" | "OPR_VCC" if o.size == 64 => {
                    let (l, h) = self.read64(d, o, false).ok_or("op")?;
                    u.push(format!("{l}|{h}"));
                }
                _ if o.size == 64 => return Err("operand 64-bit"),
                _ => u.push(self.read(d, o, src_f16).ok_or("op")?),
            }
        }
        match d.variant {
            Variant::Plain => {}
            Variant::Sdwa => {
                let e = ext_encoding(d).ok_or("SDWA")?;
                let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
                let s0 = if fv("S0") == 1 { self.code32(fv("VSRC0"), None, src_f16).ok_or("op")? } else { self.vreg(fv("VSRC0")) };
                if !u.is_empty() {
                    u[0] = format!("sdwa_sel({s0}, {}u, {})", fv("SRC0_SEL"), fv("SRC0_SEXT") == 1);
                }
                if d.enc.field("VSRC1").is_some() && u.len() > 1 {
                    let v1 = fv("VSRC1");
                    let s1 = if fv("S1") == 1 { self.code32(v1, None, src_f16).ok_or("op")? } else { self.vreg(v1) };
                    u[1] = format!("sdwa_sel({s1}, {}u, {})", fv("SRC1_SEL"), fv("SRC1_SEXT") == 1);
                }
            }
            Variant::Dpp16 | Variant::Dpp8 => {
                let e = ext_encoding(d).ok_or("DPP")?;
                let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
                let v = self.vreg(fv("VSRC0"));
                let (el, eh) = (self.sreg(126), self.sreg(127));
                let eh = if self.opt.wave64 { eh } else { "0u".into() };
                let t = self.t();
                if d.variant == Variant::Dpp16 {
                    let ctrl = fv("DPP_CTRL");
                    let (fi, bound) = (fv("FI") == 1, fv("BOUND_CTRL") == 1);
                    let src = dpp16_src(ctrl).ok_or("DPP_CTRL")?;
                    self.line(&format!("int {t}_l = {src};"));
                    self.line(&format!(
                        "bool {t}_ok = and({t}_l >= 0, or({fi}, lanebit_at({el}, {eh}, (uint){t}_l)));"
                    ));
                    let rla = self.rla_half_fn();
                    self.line(&format!("uint {t} = {rla}({v}, (uint)max({t}_l, 0));"));
                    let (rm, bm) = (fv("ROW_MASK"), fv("BANK_MASK"));
                    // write enable: row/bank masks, and an invalid source either reads 0 (bound_ctrl) or disables
                    let mut en = format!("((({rm}u >> (lane_id >> 4)) & 1u) != 0u && (({bm}u >> ((lane_id >> 2) & 3u)) & 1u) != 0u)");
                    if bound {
                        self.line(&format!("{t} = select({t}_ok, {t}, 0u);"));
                    } else {
                        en = format!("({en} && {t}_ok)");
                    }
                    self.dpp_enable = Some(en);
                } else {
                    let sel: u64 = (0..8).map(|i| fv(&format!("LANE_SEL_{i}")) << (3 * i)).sum();
                    let fi = rdna2::field_value(d.enc, &d.words, "SRC0").unwrap_or(0) == 0xEA;
                    self.line(&format!("uint {t}_l = (lane_id & ~7u) | (({sel}u >> (3u * (lane_id & 7u))) & 7u);"));
                    let rla = self.rla_half_fn();
                    self.line(&format!("uint {t} = {rla}({v}, {t}_l);"));
                    if !fi {
                        self.line(&format!("{t} = select(lanebit_at({el}, {eh}, {t}_l), {t}, 0u);"));
                    }
                }
                if !u.is_empty() {
                    u[0] = t;
                }
            }
        }
        Ok(u)
    }

    /// Write a VALU result, honouring SDWA dst_sel/dst_unused and DPP write enables.
    fn vwrite(&mut self, d: &Decoded, dst: &Dst, val: &str) {
        let mut val = val.to_string();
        if d.variant == Variant::Sdwa {
            if let (Some(e), Dst::V(n)) = (ext_encoding(d), dst) {
                let fv = |k: &str| rdna2::field_value(e, &d.words, k).unwrap_or(0);
                if e.field("DST_SEL").is_some() {
                    let cur = self.vreg(*n);
                    val = format!("sdwa_dst({val}, {cur}, {}u, {}u)", fv("DST_SEL"), fv("DST_UNUSED"));
                }
            }
        }
        match (self.dpp_enable.take(), dst) {
            (Some(en), Dst::V(n)) => {
                let v = self.vreg(*n);
                let a = self.act();
                self.line(&format!("{v} = select(and({a}, {en}), {val}, {v});"));
            }
            _ => self.write32(dst, &val),
        }
    }

    /// VOP3P packed math (two 16-bit halves; op_sel picks the source half for the low result,
    /// op_sel_hi for the high one; neg / neg_hi per half), v_pk_fmac_f16 (VOP2) and the dots.
    fn valu_pk(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        if d.variant != Variant::Plain {
            return Err("VOP3P SDWA/DPP");
        }
        let (outs, ins) = Self::ops(d);
        let vop3p = d.enc.name == "ENC_VOP3P";
        let fv = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let (sel, sel_hi) = if vop3p { (fv("OP_SEL"), fv("OP_SEL_HI") | fv("OP_SEL_HI_2") << 2) } else { (0, 0b111) };
        let (neg, neg_hi) = if vop3p { (fv("NEG"), fv("NEG_HI")) } else { (0, 0) };
        let clamp = vop3p && fv("CLAMP") == 1;
        let u = self.sources(d, &ins, name.ends_with("_f16"))?;
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        let cur = match &dst {
            Dst::V(n) => self.vreg(*n),
            _ => "0u".into(),
        };
        let src = |i: usize| u.get(i).cloned().unwrap_or_else(|| "0u".into());
        let h = |i: usize, hi: bool| -> String {
            let s = src(i);
            if (if hi { sel_hi } else { sel }) >> i & 1 == 1 {
                format!("(({s}) >> 16)")
            } else {
                format!("(({s}) & 0xFFFFu)")
            }
        };
        let fh = |i: usize, hi: bool| -> String {
            let v = format!("f16tof32({})", h(i, hi));
            if (if hi { neg_hi } else { neg }) >> i & 1 == 1 {
                format!("(-{v})")
            } else {
                v
            }
        };
        let si = |i: usize, hi: bool| format!("(((int)({} << 16)) >> 16)", h(i, hi));
        let dyn_rm = self.dynamic_f16_rm;
        let fr = |e: String| {
            let s = if clamp { format!("saturate({e})") } else { e };
            // The product of two binary16 operands fits exactly in binary32.
            if name == "v_pk_mul_f16" {
                to_f16_rounded_expr(dyn_rm, &s)
            } else {
                to_f16_expr(dyn_rm, &s)
            }
        };
        let t = self.t();
        // dot products: one 32-bit result
        let dot: Option<String> = match name {
            "v_dot2_f32_f16" => {
                let mut c = format!("asfloat({})", src(2));
                if neg >> 2 & 1 == 1 {
                    c = format!("(-{c})");
                }
                let e = format!("mad({}, {}, mad({}, {}, {c}))", fh(0, true), fh(1, true), fh(0, false), fh(1, false));
                Some(if clamp { format!("asuint(saturate({e}))") } else { format!("asuint({e})") })
            }
            "v_dot2c_f32_f16" => Some(format!("asuint(mad({}, {}, mad({}, {}, asfloat({cur}))))", fh(0, true), fh(1, true), fh(0, false), fh(1, false))),
            // the packed 8-bit dot products as SM 6.4's intrinsics (the GPU's own instruction): FSR4's
            // network passes are thousands of them, and the shift-and-multiply form made one pass
            // 1.5 MB of DXIL that the AMD driver could not compile (E_OUTOFMEMORY)
            "v_dot4c_i32_i8" => Some(format!("(uint)dot4add_i8packed({}, {}, (int){cur})", src(0), src(1))),
            // with clamp the sum saturates: the products alone stay in range (a 16-bit pair
            // product up to 2^31 is summed in 64 bits)
            "v_dot2_i32_i16" if clamp => Some(format!(
                "sat_i32((int64_t){} * (int64_t){} + (int64_t){} * (int64_t){} + (int64_t)(int){})",
                si(0, false),
                si(1, false),
                si(0, true),
                si(1, true),
                src(2)
            )),
            "v_dot2_u32_u16" if clamp => Some(format!(
                "sat_u32((uint64_t)({} * {}) + (uint64_t)({} * {}) + (uint64_t){})",
                h(0, false),
                h(1, false),
                h(0, true),
                h(1, true),
                src(2)
            )),
            "v_dot4_u32_u8" if clamp => Some(format!("sat_u32((uint64_t)dot4_u8({}, {}, 0u) + (uint64_t){})", src(0), src(1), src(2))),
            "v_dot4_i32_i8" if clamp => Some(format!("sat_i32((int64_t)(int)dot4_i8({}, {}, 0u) + (int64_t)(int){})", src(0), src(1), src(2))),
            "v_dot8_u32_u4" if clamp => Some(format!("sat_u32((uint64_t)dot8_u4({}, {}, 0u) + (uint64_t){})", src(0), src(1), src(2))),
            "v_dot8_i32_i4" if clamp => Some(format!("sat_i32((int64_t)(int)dot8_i4({}, {}, 0u) + (int64_t)(int){})", src(0), src(1), src(2))),
            "v_dot2_i32_i16" => Some(format!("(uint)({} * {} + {} * {} + (int){})", si(0, false), si(1, false), si(0, true), si(1, true), src(2))),
            "v_dot2_u32_u16" => Some(format!("({} * {} + {} * {} + {})", h(0, false), h(1, false), h(0, true), h(1, true), src(2))),
            "v_dot4_u32_u8" => Some(format!("dot4add_u8packed({}, {}, {})", src(0), src(1), src(2))),
            "v_dot4_i32_i8" => Some(format!("(uint)dot4add_i8packed({}, {}, (int){})", src(0), src(1), src(2))),
            "v_dot8_u32_u4" => Some(format!("dot8_u4({}, {}, {})", src(0), src(1), src(2))),
            "v_dot8_i32_i4" => Some(format!("dot8_i4({}, {}, {})", src(0), src(1), src(2))),
            _ => None,
        };
        if let Some(e) = dot {
            self.line(&format!("uint {t} = {e};"));
            self.write32(&dst, &t);
            return Ok(());
        }
        let half = |hi: bool| -> Option<String> {
            let (a, b, c) = (h(0, hi), h(1, hi), h(2, hi));
            let curh = if hi { format!("({cur} >> 16)") } else { format!("({cur} & 0xFFFFu)") };
            Some(match name {
                "v_pk_add_f16" => fr(format!("{} + {}", fh(0, hi), fh(1, hi))),
                "v_pk_mul_f16" => fr(format!("{} * {}", fh(0, hi), fh(1, hi))),
                "v_pk_fma_f16" => fr(format!("mad({}, {}, {})", fh(0, hi), fh(1, hi), fh(2, hi))),
                "v_pk_fmac_f16" => fr(format!("mad({}, {}, f16tof32({curh}))", fh(0, hi), fh(1, hi))),
                "v_pk_min_f16" => fr(format!("min({}, {})", fh(0, hi), fh(1, hi))),
                "v_pk_max_f16" => fr(format!("max({}, {})", fh(0, hi), fh(1, hi))),
                // integer clamp saturates each half
                "v_pk_add_u16" if clamp => format!("min({a} + {b}, 0xFFFFu)"),
                "v_pk_sub_u16" if clamp => format!("select({a} < {b}, 0u, {a} - {b})"),
                "v_pk_add_i16" if clamp => format!("((uint)clamp({} + {}, -32768, 32767) & 0xFFFFu)", si(0, hi), si(1, hi)),
                "v_pk_sub_i16" if clamp => format!("((uint)clamp({} - {}, -32768, 32767) & 0xFFFFu)", si(0, hi), si(1, hi)),
                "v_pk_mad_u16" if clamp => format!("min({a} * {b} + {c}, 0xFFFFu)"),
                "v_pk_mad_i16" if clamp => format!("((uint)clamp({} * {} + {}, -32768, 32767) & 0xFFFFu)", si(0, hi), si(1, hi), si(2, hi)),
                "v_pk_add_u16" => format!("(({a} + {b}) & 0xFFFFu)"),
                "v_pk_sub_u16" => format!("(({a} - {b}) & 0xFFFFu)"),
                "v_pk_add_i16" => format!("((uint)({} + {}) & 0xFFFFu)", si(0, hi), si(1, hi)),
                "v_pk_sub_i16" => format!("((uint)({} - {}) & 0xFFFFu)", si(0, hi), si(1, hi)),
                "v_pk_mul_lo_u16" => format!("(({a} * {b}) & 0xFFFFu)"),
                "v_pk_mad_u16" => format!("(({a} * {b} + {c}) & 0xFFFFu)"),
                "v_pk_mad_i16" => format!("((uint)({} * {} + {}) & 0xFFFFu)", si(0, hi), si(1, hi), si(2, hi)),
                "v_pk_lshlrev_b16" => format!("(({b} << ({a} & 15u)) & 0xFFFFu)"),
                "v_pk_lshrrev_b16" => format!("({b} >> ({a} & 15u))"),
                "v_pk_ashrrev_i16" => format!("((uint)({} >> ({a} & 15u)) & 0xFFFFu)", si(1, hi)),
                "v_pk_max_u16" => format!("max({a}, {b})"),
                "v_pk_min_u16" => format!("min({a}, {b})"),
                "v_pk_max_i16" => format!("((uint)max({}, {}) & 0xFFFFu)", si(0, hi), si(1, hi)),
                "v_pk_min_i16" => format!("((uint)min({}, {}) & 0xFFFFu)", si(0, hi), si(1, hi)),
                _ => return None,
            })
        };
        let (lo, hi) = (half(false).ok_or("VOP3P")?, half(true).ok_or("VOP3P")?);
        self.line(&format!("uint {t} = {lo} | ({hi} << 16);"));
        self.write32(&dst, &t);
        Ok(())
    }

    /// v_movrel*: VGPR indexing with the (uniform) M0 – a switch over the registers the
    /// program uses from the base upwards.
    fn movrel(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        if d.variant != Variant::Plain {
            return Err("movrel SDWA/DPP");
        }
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let vop3 = d.enc.name == "ENC_VOP3";
        let src_code = f("SRC0");
        let vdst = f("VDST");
        let m0 = self.sreg(124);
        let act = self.act();
        let top = self.max_vgpr.max(vdst).max(src_code.saturating_sub(256));
        let _ = vop3;
        let t = self.t();
        let (rd_idx, wr_idx) = match name {
            "v_movrels_b32" => (Some(m0.clone()), None),
            "v_movreld_b32" => (None, Some(m0.clone())),
            "v_movrelsd_b32" => (Some(m0.clone()), Some(m0.clone())),
            "v_movrelsd_2_b32" => (Some(format!("({m0} & 0x3FFu)")), Some(format!("(({m0} >> 16) & 0x3FFu)"))),
            _ => return Err("movrel"),
        };
        let value = match &rd_idx {
            Some(idx) => {
                if src_code < 256 {
                    return Err("movrels: the source is not a VGPR");
                }
                let base = src_code - 256;
                let mut cases = String::new();
                for k in 0..=top.saturating_sub(base) {
                    let v = self.vreg(base + k);
                    cases.push_str(&format!(" case {k}u: {t} = {v}; break;"));
                }
                self.line(&format!("uint {t} = 0u; switch ({idx}) {{{cases} default: break; }}"));
                t.clone()
            }
            None => self.code32(src_code, d.literal, false).ok_or("op")?,
        };
        match &wr_idx {
            Some(idx) => {
                let mut cases = String::new();
                for k in 0..=top.saturating_sub(vdst) {
                    let v = self.vreg(vdst + k);
                    cases.push_str(&format!(" case {k}u: {v} = {value}; break;"));
                }
                self.line(&format!("if ({act}) {{ switch ({idx}) {{{cases} default: break; }} }}"));
            }
            None => {
                let v = self.vreg(vdst);
                self.line(&format!("if ({act}) {v} = {value};"));
            }
        }
        Ok(())
    }

    /// v_swaprel_b32: exchange the fixed VDST with the VGPR at SRC0 + M0. The spec marks
    /// both read-write, so both old values are captured before either write – an
    /// overlapping pair (VDST = SRC0 + M0) degenerates to a no-op. Writes honor EXEC.
    fn swaprel(&mut self, d: &Decoded) -> Result<(), &'static str> {
        if d.variant != Variant::Plain {
            return Err("swaprel SDWA/DPP");
        }
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let vdst = f("VDST");
        let src_code = f("SRC0");
        if src_code < 256 {
            return Err("swaprel: the source is not a VGPR");
        }
        let base = src_code - 256;
        let top = self.max_vgpr.max(vdst).max(base);
        let m0 = self.sreg(124);
        let act = self.act();
        let (td, ts) = (self.t(), self.t());
        let vd = self.vreg(vdst);
        self.line(&format!("uint {td} = {vd};"));
        let mut cases = String::new();
        for k in 0..=top.saturating_sub(base) {
            cases.push_str(&format!(" case {k}u: {ts} = {}; break;", self.vreg(base + k)));
        }
        self.line(&format!("uint {ts} = 0u; switch ({m0}) {{{cases} default: break; }}"));
        let mut cases = String::new();
        for k in 0..=top.saturating_sub(base) {
            let v = self.vreg(base + k);
            cases.push_str(&format!(" case {k}u: {v} = {td}; break;"));
        }
        self.line(&format!("if ({act}) {{ {vd} = {ts}; switch ({m0}) {{{cases} default: break; }} }}"));
        Ok(())
    }

    /// 16-bit VALU (f16 / u16 / i16 / b16): sources are the low halves (VOP3 op_sel picks the
    /// high half per source), results go to the low half (op_sel[3]: high half) and the other
    /// half is preserved – AnyPS5 `write16Bits` (GPL-2.0, ported 2026-11-20).
    fn valu16(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let (outs, ins) = Self::ops(d);
        let m = Self::mods(d);
        let src_f16 = name.contains("f16") && !matches!(name, "v_cvt_f16_u16" | "v_cvt_f16_i16");
        let u = self.sources(d, &ins, src_f16)?;
        let plain = d.variant == Variant::Plain;
        let half = |i: usize| -> String {
            let s = u.get(i).map(String::as_str).unwrap_or("0u");
            if plain && m.opsel >> i & 1 == 1 {
                format!("(({s}) >> 16)")
            } else {
                format!("(({s}) & 0xFFFFu)")
            }
        };
        let fh = |i: usize| -> String {
            let mut s = format!("f16tof32({})", half(i));
            if m.abs[i.min(2)] {
                s = format!("abs({s})");
            }
            if m.neg[i.min(2)] {
                s = format!("(-{s})");
            }
            s
        };
        let hb = |i: usize| -> String {
            let mut s = half(i);
            if m.abs[i.min(2)] {
                s = format!("({s} & 0x7FFFu)");
            }
            if m.neg[i.min(2)] {
                s = format!("({s} ^ 0x8000u)");
            }
            s
        };
        let si = |i: usize| format!("(((int)({} << 16)) >> 16)", half(i));
        let dyn_rm = self.dynamic_f16_rm;
        let fr = |e: String| -> String {
            let mut s = match m.omod {
                1 => format!("(({e}) * 2.0f)"),
                2 => format!("(({e}) * 4.0f)"),
                3 => format!("(({e}) * 0.5f)"),
                _ => format!("({e})"),
            };
            if m.clamp {
                s = format!("saturate({s})");
            }
            if name == "v_mul_f16" {
                to_f16_rounded_expr(dyn_rm, &s)
            } else {
                to_f16_expr(dyn_rm, &s)
            }
        };
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        let cur = match &dst {
            Dst::V(n) => self.vreg(*n),
            _ => "0u".into(),
        };
        let curh = if plain && m.opsel >> 3 & 1 == 1 { format!("({cur} >> 16)") } else { format!("({cur} & 0xFFFFu)") };
        let (a, b, c) = (half(0), half(1), half(2));
        let r: String = match name {
            "v_add_f16" => fr(format!("{} + {}", fh(0), fh(1))),
            "v_sub_f16" => fr(format!("{} - {}", fh(0), fh(1))),
            "v_subrev_f16" => fr(format!("{} - {}", fh(1), fh(0))),
            "v_mul_f16" => fr(format!("{} * {}", fh(0), fh(1))),
            "v_min_f16" => fr(format!("min({}, {})", fh(0), fh(1))),
            "v_max_f16" => fr(format!("max({}, {})", fh(0), fh(1))),
            "v_fma_f16" => fr(format!("mad({}, {}, {})", fh(0), fh(1), fh(2))),
            "v_med3_f16" => fr(format!("med3f({}, {}, {})", fh(0), fh(1), fh(2))),
            "v_min3_f16" => fr(format!("min(min({}, {}), {})", fh(0), fh(1), fh(2))),
            "v_max3_f16" => fr(format!("max(max({}, {}), {})", fh(0), fh(1), fh(2))),
            "v_med3_u16" => format!("med3u({a}, {b}, {c})"),
            "v_min3_u16" => format!("min(min({a}, {b}), {c})"),
            "v_max3_u16" => format!("max(max({a}, {b}), {c})"),
            "v_med3_i16" => format!("((uint)med3i({}, {}, {}) & 0xFFFFu)", si(0), si(1), si(2)),
            "v_min3_i16" => format!("((uint)min(min({}, {}), {}) & 0xFFFFu)", si(0), si(1), si(2)),
            "v_max3_i16" => format!("((uint)max(max({}, {}), {}) & 0xFFFFu)", si(0), si(1), si(2)),
            "v_frexp_mant_f16" => fr(format!("frexp_mant({})", fh(0))),
            "v_frexp_exp_i16_f16" => format!("((uint)frexp_exp({}) & 0xFFFFu)", fh(0)),
            "v_cndmask_b16" => return Err("VALU 16-bit"),
            "v_mad_f16" => {
                let rm = if dyn_rm { "xcl_fp16_rm" } else { "0u" };
                fr(format!("mad_nf_f16({}, {}, {}, {rm})", fh(0), fh(1), fh(2)))
            }
            "v_fmac_f16" => fr(format!("mad({}, {}, f16tof32({curh}))", fh(0), fh(1))),
            "v_fmamk_f16" => fr(format!("mad({}, f16tof32({} & 0xFFFFu), {})", fh(0), u[1], fh(2))),
            "v_fmaak_f16" => fr(format!("mad({}, {}, f16tof32({} & 0xFFFFu))", fh(0), fh(1), u[2])),
            "v_ldexp_f16" => fr(format!("ldexp({}, (float){})", fh(0), si(1))),
            "v_rcp_f16" => fr(format!("rcp({})", fh(0))),
            "v_rsq_f16" => fr(format!("rsqrt({})", fh(0))),
            "v_sqrt_f16" => fr(format!("sqrt({})", fh(0))),
            "v_log_f16" => fr(format!("log2({})", fh(0))),
            "v_exp_f16" => fr(format!("exp2({})", fh(0))),
            "v_sin_f16" => fr(format!("sin({} * 6.2831853f)", fh(0))),
            "v_cos_f16" => fr(format!("cos({} * 6.2831853f)", fh(0))),
            "v_fract_f16" => fr(format!("fract_f({})", fh(0))),
            "v_floor_f16" => fr(format!("floor({})", fh(0))),
            "v_ceil_f16" => fr(format!("ceil({})", fh(0))),
            "v_trunc_f16" => fr(format!("trunc({})", fh(0))),
            "v_rndne_f16" => fr(format!("round({})", fh(0))),
            "v_cvt_f16_u16" => fr(format!("(float){a}")),
            "v_cvt_f16_i16" => fr(format!("(float){}", si(0))),
            "v_cvt_u16_f16" => format!("(cvt_u32({}) > 0xFFFFu ? 0xFFFFu : cvt_u32({}))", fh(0), fh(0)),
            "v_cvt_i16_f16" => format!("((uint)clamp(cvt_i32({}), -32768, 32767) & 0xFFFFu)", fh(0)),
            // f16 → unorm16 / snorm16: exact RNE on raw binary16 bits (avoids fp32 double-rounding & clamp(NaN))
            "v_cvt_norm_u16_f16" => format!("cvt_norm_u16_f16({})", hb(0)),
            "v_cvt_norm_i16_f16" => format!("cvt_norm_i16_f16({})", hb(0)),
            "v_add_nc_u16" | "v_add_u16" => format!("(({a} + {b}) & 0xFFFFu)"),
            "v_sub_nc_u16" | "v_sub_u16" => format!("(({a} - {b}) & 0xFFFFu)"),
            "v_subrev_nc_u16" | "v_subrev_u16" => format!("(({b} - {a}) & 0xFFFFu)"),
            "v_add_nc_i16" => format!("((uint)({} + {}) & 0xFFFFu)", si(0), si(1)),
            "v_sub_nc_i16" => format!("((uint)({} - {}) & 0xFFFFu)", si(0), si(1)),
            "v_mul_lo_u16" => format!("(({a} * {b}) & 0xFFFFu)"),
            "v_lshlrev_b16" => format!("(({b} << ({a} & 15u)) & 0xFFFFu)"),
            "v_lshrrev_b16" => format!("({b} >> ({a} & 15u))"),
            "v_ashrrev_i16" => format!("((uint)({} >> ({a} & 15u)) & 0xFFFFu)", si(1)),
            "v_max_u16" => format!("max({a}, {b})"),
            "v_min_u16" => format!("min({a}, {b})"),
            "v_max_i16" => format!("((uint)max({}, {}) & 0xFFFFu)", si(0), si(1)),
            "v_min_i16" => format!("((uint)min({}, {}) & 0xFFFFu)", si(0), si(1)),
            "v_mad_u16" => format!("(({a} * {b} + {c}) & 0xFFFFu)"),
            "v_mad_i16" => format!("((uint)({} * {} + {}) & 0xFFFFu)", si(0), si(1), si(2)),
            _ => return Err("VALU 16-bit"),
        };
        let t = self.t();
        self.line(&format!("uint {t} = {r};"));
        if d.variant == Variant::Sdwa {
            self.vwrite(d, &dst, &t);
        } else {
            let merged = if m.opsel >> 3 & 1 == 1 { format!("({cur} & 0xFFFFu) | ({t} << 16)") } else { format!("({cur} & 0xFFFF0000u) | {t}") };
            self.vwrite(d, &dst, &merged);
        }
        Ok(())
    }

    /// 64-bit VALU: integer compares / shifts / v_mad_u64_u32 and the basic f64 operations.
    fn valu64(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let (outs, ins) = Self::ops(d);
        let m = Self::mods(d);
        if d.variant != Variant::Plain {
            return Err("64-bit SDWA/DPP");
        }
        let src_f64 = name.ends_with("_f64");
        let mut lo_hi: Vec<(String, String)> = Vec::new();
        for o in &ins {
            if o.size == 64 {
                lo_hi.push(self.read64(d, o, src_f64).ok_or("op")?);
            } else {
                let v = self.read(d, o, false).ok_or("op")?;
                lo_hi.push((v, "0u".into()));
            }
        }
        let u64v = |i: usize| format!("(((uint64_t){} << 32) | (uint64_t){})", lo_hi[i].1, lo_hi[i].0);
        // abs / neg on the sign bit, as the hardware applies them: `-x` is `fsub -0.0, x` in
        // DXIL, which leaves the sign of a NaN to the driver
        let f64v = |i: usize| {
            let mut hi = lo_hi[i].1.clone();
            if m.abs[i.min(2)] {
                hi = format!("({hi} & 0x7FFFFFFFu)");
            }
            if m.neg[i.min(2)] {
                hi = format!("({hi} ^ 0x80000000u)");
            }
            format!("asdouble({}, {hi})", lo_hi[i].0)
        };
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        let t = self.t();
        if let Some(rest) = name.strip_prefix("v_cmpx_").or_else(|| name.strip_prefix("v_cmp_")) {
            let (op, ty) = rest.rsplit_once('_').ok_or("VOPC")?;
            if op == "class" && ty == "f64" {
                let cond = format!("((({}) >> fclass64({})) & 1u) != 0u", lo_hi[1].0, f64v(0));
                let dst = if name.starts_with("v_cmpx_") { Dst::S(126) } else { dst };
                self.write_mask(&dst, &cond);
                return Ok(());
            }
            let (a, b) = match ty {
                "u64" => (u64v(0), u64v(1)),
                "i64" => (format!("(int64_t){}", u64v(0)), format!("(int64_t){}", u64v(1))),
                "f64" => (f64v(0), f64v(1)),
                _ => return Err("VOPC 64"),
            };
            let cond = if ty == "f64" {
                float_compare(op, &a, &b, "isnan_d").ok_or("VOPC 64")?
            } else {
                match op {
                    "f" => "false".to_string(),
                    "lt" => format!("{a} < {b}"),
                    "eq" => format!("{a} == {b}"),
                    "le" => format!("{a} <= {b}"),
                    "gt" => format!("{a} > {b}"),
                    "ne" | "lg" => format!("{a} != {b}"),
                    "ge" => format!("{a} >= {b}"),
                    "t" | "tru" => "true".to_string(),
                    _ => return Err("VOPC 64"),
                }
            };
            let dst = if name.starts_with("v_cmpx_") { Dst::S(126) } else { dst };
            self.write_mask(&dst, &cond);
            return Ok(());
        }
        let r64: String = match name {
            "v_lshlrev_b64" => format!("({} << ({} & 63u))", u64v(1), lo_hi[0].0),
            "v_lshrrev_b64" => format!("({} >> ({} & 63u))", u64v(1), lo_hi[0].0),
            "v_ashrrev_i64" => format!("(uint64_t)((int64_t){} >> ({} & 63u))", u64v(1), lo_hi[0].0),
            "v_mad_u64_u32" | "v_mad_i64_i32" => {
                let p = if name == "v_mad_u64_u32" {
                    format!("((uint64_t){} * (uint64_t){})", lo_hi[0].0, lo_hi[1].0)
                } else {
                    format!("(uint64_t)((int64_t)(int){} * (int64_t)(int){})", lo_hi[0].0, lo_hi[1].0)
                };
                self.line(&format!("uint64_t {t}_p = {p}; uint64_t {t}_c = {};", u64v(2)));
                self.line(&format!("uint64_t {t}_w = {t}_p + {t}_c;"));
                let sdst = outs.get(1).map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
                self.write_mask(&sdst, &format!("{t}_w < {t}_c"));
                self.write64(&dst, &format!("(uint){t}_w"), &format!("(uint)({t}_w >> 32)"));
                return Ok(());
            }
            _ => {
                // f64 results
                let e = match name {
                    "v_add_f64" => format!("{} + {}", f64v(0), f64v(1)),
                    "v_mul_f64" => format!("{} * {}", f64v(0), f64v(1)),
                    "v_fma_f64" => format!("fma({}, {}, {})", f64v(0), f64v(1), f64v(2)),
                    "v_min_f64" => format!("min({}, {})", f64v(0), f64v(1)),
                    "v_max_f64" => format!("max({}, {})", f64v(0), f64v(1)),
                    "v_rcp_f64" => format!("(1.0 / {})", f64v(0)),
                    "v_cvt_f64_f32" => format!("(double)asfloat({})", lo_hi[0].0),
                    "v_cvt_f64_i32" => format!("(double)(int){}", lo_hi[0].0),
                    "v_cvt_f64_u32" => format!("(double){}", lo_hi[0].0),
                    "v_floor_f64" => format!("floor_d({})", f64v(0)),
                    "v_ceil_f64" => format!("ceil_d({})", f64v(0)),
                    "v_fract_f64" => format!("fract_d({})", f64v(0)),
                    "v_rndne_f64" => format!("round_ne_d({})", f64v(0)),
                    "v_frexp_mant_f64" => format!("frexp_mant_d({})", f64v(0)),
                    "v_sqrt_f64" => format!("sqrt_d({})", f64v(0)),
                    "v_rsq_f64" => format!("((double)1 / sqrt_d({}))", f64v(0)),
                    "v_trunc_f64" => format!("trunc_d({})", f64v(0)),
                    "v_ldexp_f64" => format!("ldexp_d({}, (int){})", f64v(0), lo_hi[1].0),
                    "v_cvt_f32_f64" => {
                        self.line(&format!("uint {t} = {};", Self::fres(&m, &format!("(float){}", f64v(0)))));
                        self.write32(&dst, &t);
                        return Ok(());
                    }
                    "v_cvt_i32_f64" => {
                        // saturating, and NaN gives 0 where clamp would give −2^31
                        self.line(&format!("double {t}_x = {};", f64v(0)));
                        self.line(&format!("uint {t} = isnan_d({t}_x) ? 0u : (uint)(int)clamp({t}_x, -2147483648.0, 2147483647.0);"));
                        self.write32(&dst, &t);
                        return Ok(());
                    }
                    "v_cvt_u32_f64" => {
                        self.line(&format!("double {t}_x = {};", f64v(0)));
                        self.line(&format!("uint {t} = isnan_d({t}_x) ? 0u : (uint)clamp({t}_x, 0.0, 4294967295.0);"));
                        self.write32(&dst, &t);
                        return Ok(());
                    }
                    "v_frexp_exp_i32_f64" => {
                        self.line(&format!("uint {t} = (uint)frexp_exp_d({});", f64v(0)));
                        self.write32(&dst, &t);
                        return Ok(());
                    }
                    _ => return Err("VALU 64-bit"),
                };
                // omod, then clamp (DXIL has a double saturate)
                let mut e = match m.omod {
                    1 => format!("(({e}) * 2.0)"),
                    2 => format!("(({e}) * 4.0)"),
                    3 => format!("(({e}) * 0.5)"),
                    _ => e,
                };
                if m.clamp {
                    e = format!("saturate({e})");
                }
                self.line(&format!("double {t}_d = {e}; uint {t}_l, {t}_h; asuint({t}_d, {t}_l, {t}_h);"));
                self.write64(&dst, &format!("{t}_l"), &format!("{t}_h"));
                return Ok(());
            }
        };
        self.line(&format!("uint64_t {t}_w = {r64};"));
        self.write64(&dst, &format!("(uint){t}_w"), &format!("(uint)({t}_w >> 32)"));
        Ok(())
    }

    fn valu(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        if name.starts_with("v_interp_") {
            return self.interp(d, name);
        }
        let (outs, ins) = Self::ops(d);
        // 32-bit operations whose names mention 16-bit halves
        const WIDE16: &[&str] = &[
            "v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32", "v_cvt_pknorm_i16_f16", "v_cvt_pknorm_u16_f16", "v_cvt_pk_u16_u32",
            "v_cvt_pk_i16_i32", "v_mad_u32_u16", "v_mad_i32_i16", "v_sad_u16", "v_sat_pk_u8_i16",
        ];
        let f16 = (name.contains("f16") || name.contains("_i16") || name.contains("_u16") || name.contains("_b16")) && !WIDE16.contains(&name);
        let cmp = name.starts_with("v_cmp");
        // inline constants are f16 patterns only where the source itself is 16-bit
        let src_f16 = matches!(name, "v_cvt_f32_f16" | "v_pack_b32_f16" | "v_cvt_pknorm_i16_f16" | "v_cvt_pknorm_u16_f16")
            || (cmp && name.ends_with("_f16"));
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
            let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
            let s0 = f("SRC0");
            if s0 < 256 {
                return Err("v_swap_b32: SRC0 is not a VGPR");
            }
            let (x, y) = (self.vreg(f("VDST")), self.vreg(s0 - 256));
            let (t, act) = (self.t(), self.act());
            self.line(&format!("uint {t} = {x}; {x} = select({act}, {y}, {x}); {y} = select({act}, {t}, {y});"));
            return Ok(());
        }
        if d.enc.name == "ENC_VOP3P" && !mix {
            return Err("VOP3P");
        }
        if d.variant != Variant::Plain && ext_encoding(d).is_none() {
            return Err("SDWA/DPP in this encoding");
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
        let cur = match &dst {
            Dst::V(n) => self.vreg(*n),
            _ => "0u".into(),
        };
        let a = u.first().cloned().unwrap_or_default();
        let b = u.get(1).cloned().unwrap_or_default();
        let c = u.get(2).cloned().unwrap_or_default();
        let fs = |i: usize| Self::fsrc(&m, i, u.get(i).map(String::as_str).unwrap_or("0u"));
        let fr = |e: String| Self::fres(&m, &e);
        let wmask = if self.opt.wave64 { 63 } else { 31 };
        let res: String = match name {
            "v_mov_b32" | "v_mov_fed_b32" => a.clone(),
            "v_not_b32" => format!("~{a}"),
            "v_bfrev_b32" => format!("reversebits({a})"),
            "v_ffbh_u32" => format!("(({a}) == 0u ? 0xFFFFFFFFu : 31u - firstbithigh({a}))"),
            "v_ffbl_b32" => format!("(({a}) == 0u ? 0xFFFFFFFFu : firstbitlow({a}))"),
            "v_ffbh_i32" => format!("ffbh_i({a})"),
            "v_cvt_f32_i32" => fr(format!("(float)(int){a}")),
            "v_cvt_f32_u32" => fr(format!("(float){a}")),
            "v_cvt_u32_f32" => format!("cvt_u32({})", fs(0)),
            "v_cvt_i32_f32" => format!("(uint)cvt_i32({})", fs(0)),
            "v_cvt_rpi_i32_f32" => format!("(uint)cvt_i32(floor({} + 0.5f))", fs(0)),
            "v_cvt_flr_i32_f32" => format!("(uint)cvt_i32(floor({}))", fs(0)),
            "v_cvt_f32_ubyte0" => fr(format!("(float)({a} & 0xFFu)")),
            "v_cvt_f32_ubyte1" => fr(format!("(float)(({a} >> 8) & 0xFFu)")),
            "v_cvt_f32_ubyte2" => fr(format!("(float)(({a} >> 16) & 0xFFu)")),
            "v_cvt_f32_ubyte3" => fr(format!("(float)({a} >> 24)")),
            "v_cvt_f32_f16" => {
                let h = if m.opsel & 1 == 1 { format!("({a} >> 16)") } else { format!("({a} & 0xFFFFu)") };
                fr(format!("f16tof32({h})"))
            }
            "v_cvt_f16_f32" => {
                // the modifiers in f32 before the rounding: the same f16 (scaling by a power of
                // two is exact, and 0 and 1 are f16 values), except where the f16 is denormal
                let h = to_f16_rounded_expr(self.dynamic_f16_rm, &Self::out_mods(&m, &fs(0)));
                if m.opsel >> 3 & 1 == 1 {
                    format!("({cur} & 0xFFFFu) | ({h} << 16)")
                } else if d.variant == Variant::Sdwa {
                    h
                } else {
                    format!("({cur} & 0xFFFF0000u) | {h}")
                }
            }
            "v_cvt_pkrtz_f16_f32" => format!("f16_rtz({}) | (f16_rtz({}) << 16)", fs(0), fs(1)),
            "v_pack_b32_f16" => {
                let lo = if m.opsel & 1 == 1 { format!("({a} >> 16)") } else { format!("({a} & 0xFFFFu)") };
                let hi = if m.opsel & 2 == 2 { format!("({b} >> 16)") } else { format!("({b} & 0xFFFFu)") };
                format!("{lo} | ({hi} << 16)")
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                let fv = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
                let (sel, hi2) = (fv("OP_SEL"), fv("OP_SEL_HI") | fv("OP_SEL_HI_2") << 2);
                let (neg, abs) = (fv("NEG"), fv("NEG_HI"));
                let src = |i: usize| -> String {
                    let raw = &u[i];
                    let mut s = if hi2 >> i & 1 == 1 {
                        if sel >> i & 1 == 1 {
                            format!("f16tof32({raw} >> 16)")
                        } else {
                            format!("f16tof32({raw} & 0xFFFFu)")
                        }
                    } else {
                        format!("asfloat({raw})")
                    };
                    if abs >> i & 1 == 1 {
                        s = format!("abs({s})");
                    }
                    if neg >> i & 1 == 1 {
                        s = format!("(-{s})");
                    }
                    s
                };
                let mut e = format!("mad({}, {}, {})", src(0), src(1), src(2));
                if fv("CLAMP") == 1 {
                    e = format!("saturate({e})");
                }
                match name {
                    "v_fma_mix_f32" => format!("asuint({e})"),
                    "v_fma_mixlo_f16" => format!("({cur} & 0xFFFF0000u) | {}", to_f16_expr(self.dynamic_f16_rm, &e)),
                    _ => format!("({cur} & 0xFFFFu) | ({} << 16)", to_f16_expr(self.dynamic_f16_rm, &e)),
                }
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => fr(format!("rcp({})", fs(0))),
            "v_rsq_f32" => fr(format!("rsqrt({})", fs(0))),
            "v_sqrt_f32" => fr(format!("sqrt({})", fs(0))),
            "v_exp_f32" => fr(format!("exp2({})", fs(0))),
            "v_log_f32" => fr(format!("log2({})", fs(0))),
            "v_sin_f32" => fr(format!("sin({} * 6.2831853f)", fs(0))),
            "v_cos_f32" => fr(format!("cos({} * 6.2831853f)", fs(0))),
            "v_fract_f32" => fr(format!("fract_f({})", fs(0))),
            "v_trunc_f32" => fr(format!("trunc({})", fs(0))),
            "v_floor_f32" => fr(format!("floor({})", fs(0))),
            "v_ceil_f32" => fr(format!("ceil({})", fs(0))),
            "v_rndne_f32" => fr(format!("round({})", fs(0))),
            "v_frexp_mant_f32" => fr(format!("frexp_mant({})", fs(0))),
            "v_frexp_exp_i32_f32" => format!("(uint)frexp_exp({})", fs(0)),
            "v_ldexp_f32" => fr(format!("ldexp_f({}, (int){b})", fs(0))),
            "v_add_f32" => fr(format!("{} + {}", fs(0), fs(1))),
            "v_sub_f32" => fr(format!("{} - {}", fs(0), fs(1))),
            "v_subrev_f32" => fr(format!("{} - {}", fs(1), fs(0))),
            "v_mul_f32" => fr(format!("{} * {}", fs(0), fs(1))),
            "v_mul_legacy_f32" => fr(format!("mul_legacy({}, {})", fs(0), fs(1))),
            "v_min_f32" => fr(format!("min({}, {})", fs(0), fs(1))),
            "v_max_f32" => fr(format!("max({}, {})", fs(0), fs(1))),
            "v_fmac_f32" => fr(format!("mad({}, {}, asfloat({cur}))", fs(0), fs(1))),
            "v_mac_f32" => fr(format!("mad_nf({}, {}, asfloat({cur}))", fs(0), fs(1))),
            "v_fmac_legacy_f32" => fr(format!("fma_legacy({}, {}, asfloat({cur}))", fs(0), fs(1))),
            "v_fma_f32" => fr(format!("mad({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_mad_f32" => fr(format!("mad_nf({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_fma_legacy_f32" => fr(format!("fma_legacy({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_fmamk_f32" => fr(format!("mad({}, asfloat({b}), {})", fs(0), fs(2))),
            "v_fmaak_f32" => fr(format!("mad({}, {}, asfloat({c}))", fs(0), fs(1))),
            "v_madmk_f32" => fr(format!("mad_nf({}, asfloat({b}), {})", fs(0), fs(2))),
            "v_madak_f32" => fr(format!("mad_nf({}, {}, asfloat({c}))", fs(0), fs(1))),
            "v_min3_f32" => fr(format!("min(min({}, {}), {})", fs(0), fs(1), fs(2))),
            "v_max3_f32" => fr(format!("max(max({}, {}), {})", fs(0), fs(1), fs(2))),
            "v_med3_f32" => fr(format!("med3f({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_cubeid_f32" => fr(format!("cube_id({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_cubesc_f32" => fr(format!("cube_sc({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_cubetc_f32" => fr(format!("cube_tc({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_cubema_f32" => fr(format!("cube_ma({}, {}, {})", fs(0), fs(1), fs(2))),
            // integer clamp saturates (VOP3 only)
            "v_add_nc_u32" if m.clamp => format!("sat_u32((uint64_t){a} + (uint64_t){b})"),
            "v_sub_nc_u32" if m.clamp => format!("select({a} < {b}, 0u, {a} - {b})"),
            "v_subrev_nc_u32" if m.clamp => format!("select({b} < {a}, 0u, {b} - {a})"),
            "v_add_nc_u32" => format!("{a} + {b}"),
            "v_sub_nc_u32" => format!("{a} - {b}"),
            "v_subrev_nc_u32" => format!("{b} - {a}"),
            "v_mul_lo_u32" => format!("{a} * {b}"),
            "v_mul_hi_u32" => format!("mul_hi_u32({a}, {b})"),
            "v_mul_hi_i32" => format!("(uint)(((int64_t)(int){a} * (int64_t)(int){b}) >> 32)"),
            "v_mul_u32_u24" => format!("({a} & 0xFFFFFFu) * ({b} & 0xFFFFFFu)"),
            "v_mul_hi_u32_u24" => format!("(uint)(((uint64_t)({a} & 0xFFFFFFu) * (uint64_t)({b} & 0xFFFFFFu)) >> 32)"),
            "v_mul_i32_i24" => format!("(uint)(i24({a}) * i24({b}))"),
            "v_mad_u32_u24" => format!("({a} & 0xFFFFFFu) * ({b} & 0xFFFFFFu) + {c}"),
            "v_mad_i32_i24" => format!("(uint)(i24({a}) * i24({b}) + (int){c})"),
            "v_and_b32" => format!("{a} & {b}"),
            "v_or_b32" => format!("{a} | {b}"),
            "v_xor_b32" => format!("{a} ^ {b}"),
            "v_xnor_b32" => format!("~({a} ^ {b})"),
            "v_lshlrev_b32" => format!("{b} << ({a} & 31u)"),
            "v_lshrrev_b32" => format!("{b} >> ({a} & 31u)"),
            "v_ashrrev_i32" => format!("(uint)((int){b} >> ({a} & 31u))"),
            "v_lshl_add_u32" => format!("({a} << ({b} & 31u)) + {c}"),
            "v_add_lshl_u32" => format!("({a} + {b}) << ({c} & 31u)"),
            "v_lshl_or_b32" => format!("({a} << ({b} & 31u)) | {c}"),
            "v_and_or_b32" => format!("({a} & {b}) | {c}"),
            "v_or3_b32" => format!("{a} | {b} | {c}"),
            "v_xor3_b32" => format!("{a} ^ {b} ^ {c}"),
            "v_add3_u32" if m.clamp => format!("sat_u32((uint64_t){a} + (uint64_t){b} + (uint64_t){c})"),
            "v_add3_u32" => format!("{a} + {b} + {c}"),
            "v_xad_u32" => format!("({a} ^ {b}) + {c}"),
            "v_bfe_u32" => format!("bfe_u({a}, {b}, {c})"),
            "v_bfe_i32" => format!("(uint)bfe_i({a}, {b}, {c})"),
            "v_bfi_b32" => format!("({a} & {b}) | (~{a} & {c})"),
            "v_bfm_b32" => format!("select(({a} & 31u) == 0u, 0u, 0xFFFFFFFFu >> (32u - ({a} & 31u))) << ({b} & 31u)"),
            "v_bcnt_u32_b32" => format!("countbits({a}) + {b}"),
            "v_perm_b32" => format!("perm_b32({a}, {b}, {c})"),
            "v_alignbit_b32" => format!("alignbit({a}, {b}, {c})"),
            "v_alignbyte_b32" => format!("alignbit({a}, {b}, ({c} & 3u) * 8u)"),
            "v_min_u32" => format!("min({a}, {b})"),
            "v_max_u32" => format!("max({a}, {b})"),
            "v_min_i32" => format!("(uint)min((int){a}, (int){b})"),
            "v_max_i32" => format!("(uint)max((int){a}, (int){b})"),
            "v_min3_u32" => format!("min(min({a}, {b}), {c})"),
            "v_max3_u32" => format!("max(max({a}, {b}), {c})"),
            "v_med3_u32" => format!("med3u({a}, {b}, {c})"),
            "v_min3_i32" => format!("(uint)min(min((int){a}, (int){b}), (int){c})"),
            "v_max3_i32" => format!("(uint)max(max((int){a}, (int){b}), (int){c})"),
            "v_med3_i32" => format!("(uint)med3i((int){a}, (int){b}, (int){c})"),
            "v_sad_u32" => format!("(max({a}, {b}) - min({a}, {b})) + {c}"),
            "v_sad_u8" => format!("sad_u8({a}, {b}) + {c}"),
            "v_sad_hi_u8" => format!("(sad_u8({a}, {b}) << 16) + {c}"),
            "v_sad_u16" => format!("absd16({a}, {b}, 0u) + absd16({a}, {b}, 16u) + {c}"),
            "v_msad_u8" => format!("msad_u8({a}, {b}) + {c}"),
            "v_lerp_u8" => format!("lerp_u8({a}, {b}, {c})"),
            "v_add_nc_i32" if m.clamp => format!("sat_i32((int64_t)(int){a} + (int64_t)(int){b})"),
            "v_sub_nc_i32" if m.clamp => format!("sat_i32((int64_t)(int){a} - (int64_t)(int){b})"),
            "v_add_nc_i32" => format!("{a} + {b}"),
            "v_sub_nc_i32" => format!("{a} - {b}"),
            "v_mad_u32_u16" => format!("(({a} >> {}u) & 0xFFFFu) * (({b} >> {}u) & 0xFFFFu) + {c}", 16 * (m.opsel & 1), 8 * (m.opsel & 2)),
            "v_mad_i32_i16" => format!(
                "(uint)((((int)(({a} >> {}u) << 16)) >> 16) * (((int)(({b} >> {}u) << 16)) >> 16) + (int){c})",
                16 * (m.opsel & 1),
                8 * (m.opsel & 2)
            ),
            "v_cvt_pk_u16_u32" => format!("min({a}, 0xFFFFu) | (min({b}, 0xFFFFu) << 16)"),
            "v_cvt_pk_i16_i32" => format!("((uint)clamp((int){a}, -32768, 32767) & 0xFFFFu) | (((uint)clamp((int){b}, -32768, 32767) & 0xFFFFu) << 16)"),
            "v_cvt_pknorm_i16_f32" => format!("(cvt_norm_i16_f32({}) | (cvt_norm_i16_f32({}) << 16))", fs(0), fs(1)),
            "v_cvt_pknorm_u16_f32" => format!("(uint)round(saturate({}) * 65535.0f) | ((uint)round(saturate({}) * 65535.0f) << 16)", fs(0), fs(1)),
            // packed f16 sources: op_sel[0]/[1] pick each source's half; result fills all 32 bits
            "v_cvt_pknorm_i16_f16" | "v_cvt_pknorm_u16_f16" => {
                let hb = |i: usize| {
                    let s = u.get(i).map(String::as_str).unwrap_or("0u");
                    let mut s = if m.opsel >> i & 1 == 1 { format!("({s} >> 16)") } else { format!("({s} & 0xFFFFu)") };
                    if m.abs[i] {
                        s = format!("({s} & 0x7FFFu)");
                    }
                    if m.neg[i] {
                        s = format!("({s} ^ 0x8000u)");
                    }
                    s
                };
                if name == "v_cvt_pknorm_i16_f16" {
                    format!("(cvt_norm_i16_f16({}) | (cvt_norm_i16_f16({}) << 16))", hb(0), hb(1))
                } else {
                    format!("(cvt_norm_u16_f16({}) | (cvt_norm_u16_f16({}) << 16))", hb(0), hb(1))
                }
            }
            "v_mul_hi_i32_i24" => format!("(uint)(((int64_t)i24({a}) * (int64_t)i24({b})) >> 32)"),
            "v_sat_pk_u8_i16" => format!("(uint)clamp(((int)({a} << 16)) >> 16, 0, 255) | ((uint)clamp((int){a} >> 16, 0, 255) << 8)"),
            "v_div_fmas_f32" => fr(format!("mad({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_div_fixup_f32" => fr(format!("div_fixup({}, {}, {})", fs(0), fs(1), fs(2))),
            "v_cvt_pk_u8_f32" => format!(
                "({c} & ~(0xFFu << (8u * ({b} & 3u)))) | (((uint)clamp(round({}), 0.0f, 255.0f)) << (8u * ({b} & 3u)))",
                fs(0)
            ),
            "v_mbcnt_lo_u32_b32" => format!("countbits({a} & lanemask_lt_lo()) + {b}"),
            "v_mbcnt_hi_u32_b32" => format!("countbits({a} & lanemask_lt_hi()) + {b}"),
            "v_cndmask_b32" => {
                let (l, h) = u.get(2).ok_or("mask")?.split_once('|').ok_or("mask")?;
                let bit = self.lane_of(l, h);
                format!("select({bit}, {b}, {a})")
            }
            "v_permlane16_b32" | "v_permlanex16_b32" => {
                // lane i of each row reads lane sel[i] of the same / the other row (32-lane halves)
                let (fi, bound) = (m.opsel & 1 == 1, m.opsel & 2 == 2);
                let t = self.t();
                let other = if name == "v_permlanex16_b32" { "^ 16u" } else { "" };
                self.line(&format!(
                    "uint {t}_n = (select((lane_id & 15u) < 8u, {b}, {c}) >> (4u * (lane_id & 7u))) & 15u;"
                ));
                self.line(&format!("uint {t}_l = (lane_id & ~31u) | (((lane_id & 16u) {other}) & 16u) | {t}_n;"));
                let (el, eh) = (self.sreg(126), self.sreg(127));
                let eh = if self.opt.wave64 { eh } else { "0u".into() };
                self.line(&format!("bool {t}_ok = or({fi}, lanebit_at({el}, {eh}, {t}_l));"));
                let rla = self.rla_half_fn();
                self.line(&format!("uint {t} = {rla}({a}, {t}_l);"));
                if bound {
                    format!("select({t}_ok, {t}, 0u)")
                } else {
                    format!("select({t}_ok, {t}, {cur})")
                }
            }
            "v_readfirstlane_b32" => {
                let (el, eh) = (self.sreg(126), self.sreg(127));
                let eh = if self.opt.wave64 { eh } else { "0u".into() };
                let t = self.t();
                let rla = self.rla_uniform_fn();
                self.line(&format!("uint {t} = {rla}({a}, first_lane({el}, {eh}));"));
                self.write32(&dst, &t);
                return Ok(());
            }
            "v_readlane_b32" => {
                let t = self.t();
                let rla = self.rla_uniform_fn();
                let read = format!("{rla}({a}, {b} & {wmask}u)");
                // wave64 VS / PS (`upper_lanes`): in a 32-lane wave, lanes 32..63 without a lane
                // read; a spill slot < 32 from its shadow whatever the wave size (its lane may
                // not have been launched)
                let slot = super::upper_lanes::slot(d);
                match (self.upper.as_ref().and_then(|u| u.reads.get(&d.pc)).copied(), slot) {
                    (Some(Read::Const(c)), Some(_)) => self.line(&format!("uint {t} = {c:#010x}u; if (!xcl_hw32) {t} = {read};")),
                    (Some(Read::Const(c)), None) => {
                        self.line(&format!("uint {t} = {read};"));
                        self.line(&format!("{t} = select(and(xcl_hw32, ({b} & 63u) >= 32u), {c:#010x}u, {t});"));
                    }
                    (Some(Read::Shadow), Some((v, l))) if l < 32 => self.line(&format!("uint {t} = xcl_sh{v}_{l};")),
                    (Some(Read::Shadow), Some((v, l))) => self.line(&format!("uint {t} = xcl_sh{v}_{l}; if (!xcl_hw32) {t} = {read};")),
                    _ => self.line(&format!("uint {t} = {read};")),
                }
                self.write32(&dst, &t);
                return Ok(());
            }
            "v_writelane_b32" => {
                // ignores EXEC
                if let Dst::V(n) = &dst {
                    let v = self.vreg(*n);
                    self.line(&format!("if (lane_id == ({b} & {wmask}u)) {v} = {a};"));
                    // a spill slot some read takes from its shadow (`upper_lanes`)
                    let slot = super::upper_lanes::slot(d).filter(|s| self.upper.as_ref().is_some_and(|u| u.shadows.contains(s)));
                    if let Some((v, l)) = slot {
                        self.line(&format!("xcl_sh{v}_{l} = {a};"));
                    }
                }
                return Ok(());
            }
            "v_div_scale_f32" => {
                // no pre-scaling (the quotient sequence still gives a/b; only extreme ranges lose
                // precision): src0 unchanged, VCC 0 = no post-scale in v_div_fmas
                let sdst = outs.get(1).map(|o| Self::dst_of(d, o)).unwrap_or(Dst::S(106));
                self.write_mask(&sdst, "false");
                let r = format!("asuint({})", fs(0));
                self.vwrite(d, &dst, &r);
                return Ok(());
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" => {
                let (x, y) = if name == "v_subrev_co_u32" { (b.clone(), a.clone()) } else { (a.clone(), b.clone()) };
                let t = self.t();
                let carry = if name == "v_add_co_u32" {
                    self.line(&format!("uint {t} = {x} + {y};"));
                    format!("{t} < {x}")
                } else {
                    self.line(&format!("uint {t} = {x} - {y};"));
                    format!("{y} > {x}")
                };
                let sdst = outs.get(1).map(|o| Self::dst_of(d, o)).unwrap_or(Dst::S(106));
                self.write_mask(&sdst, &carry);
                self.vwrite(d, &dst, &t);
                return Ok(());
            }
            "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                let (l, h) = u.get(2).ok_or("mask")?.split_once('|').ok_or("mask")?;
                let cin = format!("(uint)({})", self.lane_of(l, h));
                let (x, y) = if name == "v_subrev_co_ci_u32" { (b.clone(), a.clone()) } else { (a.clone(), b.clone()) };
                let t = self.t();
                if name == "v_add_co_ci_u32" {
                    self.line(&format!("uint64_t {t}w = (uint64_t){x} + (uint64_t){y} + (uint64_t){cin};"));
                } else {
                    self.line(&format!("uint64_t {t}w = (uint64_t){x} - (uint64_t){y} - (uint64_t){cin};"));
                }
                self.line(&format!("uint {t} = (uint){t}w;"));
                let sdst = outs.get(1).map(|o| Self::dst_of(d, o)).unwrap_or(Dst::S(106));
                self.write_mask(&sdst, &format!("({t}w >> 32) != 0ull"));
                self.vwrite(d, &dst, &t);
                return Ok(());
            }
            _ => return Err("VALU"),
        };
        // the value straight into the write (a named temporary is one more local per
        // instruction – tens of thousands on big programs at -O0)
        self.vwrite(d, &dst, &format!("({res})"));
        Ok(())
    }

    fn vopc(&mut self, d: &Decoded, rest: &str, cmpx: bool, outs: &[&Operand], u: &[String], m: &Mods) -> Result<(), &'static str> {
        let (op, ty) = rest.rsplit_once('_').ok_or("VOPC")?;
        if u.len() < 2 {
            return Err("VOPC");
        }
        let (mut ua, mut ub) = (u[0].clone(), u[1].clone());
        let half = |i: u64, s: &str| if m.opsel >> i & 1 == 1 { format!("(({s}) >> 16)") } else { format!("(({s}) & 0xFFFFu)") };
        let ty = match ty {
            "f16" => {
                // compare in f32: exact for every f16 value (class keeps the f16 view)
                if op == "class" {
                    let (mut h, c) = (half(0, &ua), ub.clone());
                    if m.abs[0] {
                        h = format!("({h} & 0x7FFFu)");
                    }
                    if m.neg[0] {
                        h = format!("({h} ^ 0x8000u)");
                    }
                    let cond = format!("((({c}) >> fclass16({h})) & 1u) != 0u");
                    return self.vopc_write(d, cmpx, outs, &cond);
                }
                ua = format!("asuint(f16tof32({}))", half(0, &ua));
                ub = format!("asuint(f16tof32({}))", half(1, &ub));
                "f32"
            }
            "u16" => {
                ua = half(0, &ua);
                ub = half(1, &ub);
                "u32"
            }
            "i16" => {
                ua = format!("(uint)(((int)({} << 16)) >> 16)", half(0, &ua));
                ub = format!("(uint)(((int)({} << 16)) >> 16)", half(1, &ub));
                "i32"
            }
            t => t,
        };
        let cond = match ty {
            "f32" => {
                let (a, b) = (Self::fsrc(m, 0, &ua), Self::fsrc(m, 1, &ub));
                match op {
                    "class" => format!("((({ub}) >> fclass({a})) & 1u) != 0u"),
                    _ => float_compare(op, &a, &b, "isnan").ok_or("VOPC f32")?,
                }
            }
            "i32" | "u32" => {
                let (a, b) = if ty == "i32" { (format!("(int){ua}"), format!("(int){ub}")) } else { (ua.clone(), ub.clone()) };
                match op {
                    "f" => "false".to_string(),
                    "lt" => format!("{a} < {b}"),
                    "eq" => format!("{a} == {b}"),
                    "le" => format!("{a} <= {b}"),
                    "gt" => format!("{a} > {b}"),
                    "ne" | "lg" => format!("{a} != {b}"),
                    "ge" => format!("{a} >= {b}"),
                    "t" | "tru" => "true".to_string(),
                    _ => return Err("VOPC int"),
                }
            }
            _ => return Err("VOPC type"),
        };
        self.vopc_write(d, cmpx, outs, &cond)
    }

    /// Destination of a compare: VCC / SDST (SDWA SD) / EXEC (cmpx).
    fn vopc_write(&mut self, d: &Decoded, cmpx: bool, outs: &[&Operand], cond: &str) -> Result<(), &'static str> {
        let mut dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::S(106));
        if d.variant == Variant::Sdwa {
            // SDWA compares write VCC unless SD selects SDST
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

    // ---------------------------------------------------------------- cross-lane (DS)

    /// buffer_atomic_*: interlocked operations on the flat memory (GLC returns the old value).
    fn buffer_atomic(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let op = name.trim_start_matches("buffer_atomic_");
        let (vdata, vaddr, srsrc) = (f("VDATA"), f("VADDR"), Self::field(d, "SRSRC"));
        let soff = self.code32(f("SOFFSET"), None, false).ok_or("op")?;
        let (offen, idxen) = (f("OFFEN") != 0, f("IDXEN") != 0);
        let (d0, d1, d2) = (self.sreg(srsrc), self.sreg(srsrc + 1), self.sreg(srsrc + 2));
        self.window(&d1);
        let index = if idxen { self.vreg(vaddr) } else { "0u".into() };
        let voff = if offen { self.vreg(vaddr + idxen as u64) } else { "0u".into() };
        let t = self.t();
        let x2 = op.ends_with("_x2");
        let op = op.trim_end_matches("_x2");
        let bytes = if x2 { 8 } else { 4 };
        self.line(&format!("uint {t}_stride = ({d1} >> 16) & 0x3FFFu;"));
        self.line(&format!("uint {t}_off = {voff} + {}u;", f("OFFSET")));
        self.line(&format!("uint {t}_a = ({d0} + {soff} + {t}_off + {t}_stride * {index}) & ~{}u;", bytes - 1));
        self.line(&format!("bool {t}_ok = select({t}_stride == 0u, {t}_off + {bytes}u <= {d2}, {index} < {d2});"));
        let act = self.act();
        if x2 {
            return self.atomic64(op, &format!("{t}_a"), vdata, vdata + 2, f("GLC") == 1, vdata, &format!("and({act}, {t}_ok)"));
        }
        let (v, cmp) = (self.vreg(vdata), self.vreg(vdata + 1));
        let body = rmw32(op, &At::Mem(format!("{t}_a")), &v, &cmp, &format!("{t}_old"), &t).ok_or("buffer_atomic")?;
        self.line(&format!("uint {t}_old = 0u;"));
        self.line(&format!("if (and({act}, {t}_ok)) {{ {body} }}"));
        if f("GLC") == 1 {
            self.line(&format!("{v} = select({act}, select({t}_ok, {t}_old, 0u), {v});"));
        }
        Ok(())
    }

    /// A 64-bit atomic (`_x2`) at byte address `a` of the flat memory; data in VGPRs
    /// `data`, `data + 1` (compare value of cmpswap at `cmp`); the old value to `dst` (GLC).
    #[allow(clippy::too_many_arguments)]
    fn atomic64(&mut self, op: &str, a: &str, data: u64, cmp: u64, glc: bool, dst: u64, active: &str) -> Result<(), &'static str> {
        let t = self.t();
        let (lo, hi) = (self.vreg(data), self.vreg(data + 1));
        let (clo, chi) = (self.vreg(cmp), self.vreg(cmp + 1));
        self.line(&format!("uint64_t {t}_v = (uint64_t){lo} | ((uint64_t){hi} << 32); uint64_t {t}_cmp = (uint64_t){clo} | ((uint64_t){chi} << 32); uint64_t {t}_old = 0;"));
        let body = rmw64(op, a, &format!("{t}_v"), &format!("{t}_cmp"), &format!("{t}_old"), &t).ok_or("atomic 64")?;
        self.line(&format!("if ({active}) {{ {body} }}"));
        if glc {
            let (r0, r1) = (self.vreg(dst), self.vreg(dst + 1));
            self.line(&format!("if ({active}) {{ {r0} = (uint){t}_old; {r1} = (uint)({t}_old >> 32); }}"));
        }
        Ok(())
    }

    /// buffer_* d16 sub-dword loads / stores (half of the VGPR, the other half kept).
    fn mubuf_d16(&mut self, d: &Decoded, n: u32, signed: bool, store: bool, half: u32) -> Result<(), &'static str> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let (vdata, vaddr, srsrc) = (f("VDATA"), f("VADDR"), Self::field(d, "SRSRC"));
        let soff = self.code32(f("SOFFSET"), None, false).ok_or("op")?;
        let (offen, idxen) = (f("OFFEN") != 0, f("IDXEN") != 0);
        let (d0, d1, d2) = (self.sreg(srsrc), self.sreg(srsrc + 1), self.sreg(srsrc + 2));
        self.window(&d1);
        let index = if idxen { self.vreg(vaddr) } else { "0u".into() };
        let voff = if offen { self.vreg(vaddr + idxen as u64) } else { "0u".into() };
        let t = self.t();
        self.line(&format!("uint {t}_stride = ({d1} >> 16) & 0x3FFFu;"));
        self.line(&format!("uint {t}_off = {voff} + {}u;", f("OFFSET")));
        self.line(&format!("uint {t}_a = {d0} + {soff} + {t}_off + {t}_stride * {index};"));
        let ok = format!("select({t}_stride == 0u, {t}_off + {n}u <= {d2}, {index} < {d2})");
        let act = self.act();
        let v = self.vreg(vdata);
        if store {
            self.line(&format!("if (and({act}, {ok})) st_bytes({t}_a, uint4({}, 0u, 0u, 0u), {n}u);", d16_data(&v, half)));
        } else {
            let raw = format!("select({ok}, ld_raw({t}_a).x, 0u)");
            self.line(&format!("if ({act}) {v} = {};", d16_merge(&v, &raw, n, signed, half)));
        }
        Ok(())
    }

    /// FLAT / global: the flat memory model (address = low 32 bits).
    fn flat(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let op = name.trim_start_matches("global_").trim_start_matches("flat_").trim_start_matches("scratch_");
        let scratch = d.enc.name == "ENC_FLAT_SCRATCH";
        if let Some(a) = op.strip_prefix("atomic_") {
            return self.flat_atomic(d, a);
        }
        if scratch {
            return self.scratch(d, op);
        }
        if let Some((n, signed, store, half)) = flat_small(op) {
            let global = d.enc.name == "ENC_FLAT_GLBL";
            let addr = self.vreg(f("ADDR"));
            let saddr = f("SADDR");
            let base = if global && saddr != 0x7D && saddr != 0x7F { format!("{} + {addr}", self.sreg(saddr)) } else { addr };
            self.flat_window(global, saddr, f("ADDR"));
            let raw = f("OFFSET") as u32;
            let off = if global { (((raw << 20) as i32) >> 20) as u32 } else { raw };
            let t = self.t();
            self.line(&format!("uint {t}_a = {base} + {};", hex(off)));
            let act = self.act();
            if store {
                let v = self.vreg(f("DATA"));
                self.line(&format!("if ({act}) st_bytes({t}_a, uint4({}, 0u, 0u, 0u), {n}u);", d16_data(&v, half)));
            } else {
                let v = self.vreg(f("VDST"));
                self.line(&format!("if ({act}) {v} = {};", d16_merge(&v, &format!("ld_raw({t}_a).x"), n, signed, half)));
            }
            return Ok(());
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
            // the *_addtid forms have no ADDR operand in the spec: scalar base only
            "load_dword_addtid" => (false, 1),
            "store_dword_addtid" => (true, 1),
            _ => return Err("FLAT"),
        };
        let global = d.enc.name == "ENC_FLAT_GLBL";
        let saddr = f("SADDR");
        let base = if op.ends_with("_addtid") {
            // EA = SADDR ("off"/null sentinel → 0) + 4 * lane index + signed OFFSET
            match saddr {
                0x7D | 0x7F => "4u * lane_id".to_string(),
                _ => format!("{} + 4u * lane_id", self.sreg(saddr)),
            }
        } else {
            let addr = self.vreg(f("ADDR"));
            if global && saddr != 0x7D && saddr != 0x7F {
                let s = self.sreg(saddr);
                format!("{s} + {addr}")
            } else {
                addr
            }
        };
        self.flat_window(global, saddr, f("ADDR"));
        let raw = f("OFFSET") as u32;
        let off = if global { (((raw << 20) as i32) >> 20) as u32 } else { raw };
        let t = self.t();
        self.line(&format!("uint {t}_a = {base} + {};", hex(off)));
        let act = self.act();
        let mut st = Vec::new();
        for i in 0..count {
            if store {
                let v = self.vreg(f("DATA") + i);
                st.push(format!("mem.Store(({t}_a + {}u) & ~3u, {v});", 4 * i));
            } else {
                let v = self.vreg(f("VDST") + i);
                st.push(format!("{v} = mem.Load(({t}_a + {}u) & ~3u);", 4 * i));
            }
        }
        self.line(&format!("if ({act}) {{ {} }}", st.join(" ")));
        Ok(())
    }

    /// scratch_*: per-lane private memory (`xcl_scratch`); address = SADDR or VADDR + offset.
    fn scratch(&mut self, d: &Decoded, op: &str) -> Result<(), &'static str> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        self.uses_scratch = true;
        let saddr = f("SADDR");
        let base = if saddr != 0x7D && saddr != 0x7F { self.sreg(saddr) } else { self.vreg(f("ADDR")) };
        let raw = f("OFFSET") as u32;
        let off = (((raw << 20) as i32) >> 20) as u32;
        let t = self.t();
        self.line(&format!("uint {t}_a = {base} + {};", hex(off)));
        let act = self.act();
        if let Some((n, signed, store, half)) = flat_small(op) {
            if store {
                let v = self.vreg(f("DATA"));
                self.line(&format!("if ({act}) scr_st({t}_a, {}, {n}u);", d16_data(&v, half)));
            } else {
                let v = self.vreg(f("VDST"));
                self.line(&format!("if ({act}) {v} = {};", d16_merge(&v, &format!("scr_ld({t}_a)"), n, signed, half)));
            }
            return Ok(());
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
            _ => return Err("SCRATCH"),
        };
        let mut st = Vec::new();
        for i in 0..count {
            let slot = format!("xcl_scratch[(({t}_a + {}u) >> 2) & 1023u]", 4 * i);
            if store {
                st.push(format!("{slot} = {};", self.vreg(f("DATA") + i)));
            } else {
                st.push(format!("{} = {slot};", self.vreg(f("VDST") + i)));
            }
        }
        self.line(&format!("if ({act}) {{ {} }}", st.join(" ")));
        Ok(())
    }

    /// FLAT / global atomics on the flat memory (dword ops; float min / max by a CAS loop).
    fn flat_atomic(&mut self, d: &Decoded, op: &str) -> Result<(), &'static str> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let global = d.enc.name == "ENC_FLAT_GLBL";
        let addr = self.vreg(f("ADDR"));
        let saddr = f("SADDR");
        let base = if global && saddr != 0x7D && saddr != 0x7F { format!("{} + {addr}", self.sreg(saddr)) } else { addr };
        self.flat_window(global, saddr, f("ADDR"));
        let raw = f("OFFSET") as u32;
        let off = if global { (((raw << 20) as i32) >> 20) as u32 } else { raw };
        let t = self.t();
        let x2 = op.ends_with("_x2");
        let op = op.trim_end_matches("_x2");
        self.line(&format!("uint {t}_a = ({base} + {}) & ~{}u;", hex(off), if x2 { 7 } else { 3 }));
        let act = self.act();
        if x2 {
            let data = f("DATA");
            return self.atomic64(op, &format!("{t}_a"), data, data + 2, f("GLC") == 1, f("VDST"), &act);
        }
        let v = self.vreg(f("DATA"));
        self.line(&format!("uint {t}_old = 0u;"));
        let cmpv = self.vreg(f("DATA") + 1);
        let body = match op {
            "inc" | "dec" | "csub" | "fcmpswap" => rmw32(op, &At::Mem(format!("{t}_a")), &v, &cmpv, &format!("{t}_old"), &t).ok_or("FLAT atomic")?,
            "add" => format!("mem.InterlockedAdd({t}_a, {v}, {t}_old);"),
            "sub" => format!("mem.InterlockedAdd({t}_a, 0u - {v}, {t}_old);"),
            "and" => format!("mem.InterlockedAnd({t}_a, {v}, {t}_old);"),
            "or" => format!("mem.InterlockedOr({t}_a, {v}, {t}_old);"),
            "xor" => format!("mem.InterlockedXor({t}_a, {v}, {t}_old);"),
            "swap" => format!("mem.InterlockedExchange({t}_a, {v}, {t}_old);"),
            "umin" => format!("mem.InterlockedMin({t}_a, {v}, {t}_old);"),
            "umax" => format!("mem.InterlockedMax({t}_a, {v}, {t}_old);"),
            "smin" => format!("int {t}_s; mem.InterlockedMin({t}_a, (int){v}, {t}_s); {t}_old = (uint){t}_s;"),
            "smax" => format!("int {t}_s; mem.InterlockedMax({t}_a, (int){v}, {t}_s); {t}_old = (uint){t}_s;"),
            "cmpswap" => {
                let cmp = self.vreg(f("DATA") + 1);
                format!("mem.InterlockedCompareExchange({t}_a, {cmp}, {v}, {t}_old);")
            }
            "fmin" | "fmax" => float_minmax_mem(op == "fmin", false, &format!("{t}_a"), &v, &format!("{t}_old"), &t),
            _ => return Err("FLAT atomic"),
        };
        self.line(&format!("if ({act}) {{ {body} }}"));
        if f("GLC") == 1 {
            let dst = self.vreg(f("VDST"));
            self.line(&format!("if ({act}) {dst} = {t}_old;"));
        }
        Ok(())
    }

    /// LDS (`groupshared`, compute only) and the DS cross-lane permutes.
    fn ds(&mut self, d: &Decoded, name: &str) -> Result<(), &'static str> {
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let act = self.act();
        let t = self.t();
        let (o0, o1) = (f("OFFSET0"), f("OFFSET1"));
        let off16 = o0 | o1 << 8;
        match name {
            "ds_swizzle_b32" => {
                let a = self.vreg(f("ADDR"));
                let (decl, src) = swizzle_source(off16, &format!("{t}_lanes"));
                if let Some(decl) = decl {
                    self.line(&decl);
                }
                let (el, eh) = (self.sreg(126), self.sreg(127));
                let eh = if self.opt.wave64 { eh } else { "0u".into() };
                self.line(&format!("uint {t}_l = {src};"));
                let rla = self.rla_half_fn();
                self.line(&format!("uint {t} = {rla}({a}, {t}_l);"));
                self.line(&format!("{t} = select(lanebit_at({el}, {eh}, {t}_l), {t}, 0u);"));
                let dst = Dst::V(f("VDST"));
                self.write32(&dst, &t);
                return Ok(());
            }
            "ds_bpermute_b32" => {
                let (a, v) = (self.vreg(f("ADDR")), self.vreg(f("DATA0")));
                let m = if self.opt.wave64 { 63 } else { 31 };
                let (el, eh) = (self.sreg(126), self.sreg(127));
                let eh = if self.opt.wave64 { eh } else { "0u".into() };
                self.line(&format!("uint {t}_l = (({a} + {off16}u) >> 2) & {m}u;"));
                let rla = if self.x64() {
                    self.x64_words = 32;
                    "xcl_x_rla_any"
                } else {
                    "rla"
                };
                self.line(&format!("uint {t} = {rla}({v}, {t}_l);"));
                // a 32-lane wave of a wave64 VS / PS: lanes 32..63 hold a constant (`upper_lanes`)
                if let Some(Read::Const(c)) = self.upper.as_ref().and_then(|u| u.reads.get(&d.pc)).copied() {
                    self.line(&format!("{t} = select(and(xcl_hw32, {t}_l >= 32u), {c:#010x}u, {t});"));
                }
                self.line(&format!("{t} = select(lanebit_at({el}, {eh}, {t}_l), {t}, 0u);"));
                let dst = Dst::V(f("VDST"));
                self.write32(&dst, &t);
                return Ok(());
            }
            _ => {}
        }
        if !matches!(self.opt.mode, Mode::Kernel | Mode::Mesh(_) | Mode::Geometry(_) | Mode::Compute(_)) {
            return Err("LDS outside compute");
        }
        // geometry mode: where each GS lane puts its output vertices (lowest LDS address written)
        let gs_phase = match &self.opt.mode {
            Mode::Geometry(g) => g.gs_start.is_some_and(|s| d.pc >= s),
            _ => false,
        };
        // ADDTID forms have no ADDR VGPR: byte address = M0[15:0] + offset16 + 4 * lane_id.
        if matches!(name, "ds_write_addtid_b32" | "ds_read_addtid_b32") {
            if f("GDS") != 0 {
                return Err("GDS");
            }
            self.uses_lds = true;
            let m0 = self.sreg(124);
            let at = format!("(({m0} & 0xFFFFu) + {off16}u + 4u * lane_id)");
            let slot = if self.x64() { format!("xcl_lds[min((({at}) >> 2) & 8191u, XCL_X_LDS_TOP)]") } else { format!("xcl_lds[(({at}) >> 2) & 8191u]") };
            if name == "ds_write_addtid_b32" {
                let src = self.vreg(f("DATA0"));
                self.line(&format!("if ({act}) {{ {slot} = {src}; }}"));
                if gs_phase {
                    self.line(&format!("if ({act}) xcl_gs_base = min(xcl_gs_base, {at});"));
                }
            } else {
                let dst = self.vreg(f("VDST"));
                self.line(&format!("if ({act}) {{ {dst} = {slot}; }}"));
            }
            return Ok(());
        }
        self.uses_lds = true;
        // the address is read before any result is written (ds_read2_b32 v[0:1], v0 …)
        let addr = self.vreg(f("ADDR"));
        let a = format!("{t}_ad");
        self.line(&format!("uint {a} = {addr};"));
        // split wave64: the exchange lives in the top of the LDS block (`XCL_X_LDS_TOP` = the
        // last dword below it)
        let x64 = self.x64();
        let lds = |at: String| if x64 { format!("xcl_lds[min((({at}) >> 2) & 8191u, XCL_X_LDS_TOP)]") } else { format!("xcl_lds[(({at}) >> 2) & 8191u]") };
        let mut st = Vec::new();
        let rw = |n: &str| -> Option<(bool, u64, u64, bool)> {
            // (write?, dwords per element, stride of offsets (bytes), two elements?)
            Some(match n {
                "ds_read_b32" => (false, 1, 1, false),
                "ds_read_b64" => (false, 2, 1, false),
                "ds_read_b96" => (false, 3, 1, false),
                "ds_read_b128" => (false, 4, 1, false),
                "ds_write_b32" => (true, 1, 1, false),
                "ds_write_b64" => (true, 2, 1, false),
                "ds_write_b96" => (true, 3, 1, false),
                "ds_write_b128" => (true, 4, 1, false),
                "ds_read2_b32" => (false, 1, 4, true),
                "ds_read2_b64" => (false, 2, 8, true),
                "ds_read2st64_b32" => (false, 1, 256, true),
                "ds_read2st64_b64" => (false, 2, 512, true),
                "ds_write2_b32" => (true, 1, 4, true),
                "ds_write2_b64" => (true, 2, 8, true),
                "ds_write2st64_b32" => (true, 1, 256, true),
                "ds_write2st64_b64" => (true, 2, 512, true),
                _ => return None,
            })
        };
        // d16 8 / 16-bit LDS accesses (half of the VGPR)
        if let Some((bits, signed, write, half)) = ds_d16(name) {
            let at = format!("{a} + {off16}u");
            let slot = lds(at.clone());
            let sh = format!("((({at}) & 3u) * 8u)");
            let mask = (1u64 << bits) - 1;
            if write {
                let v = d16_data(&self.vreg(f("DATA0")), half);
                self.line(&format!("if ({act}) {{ InterlockedAnd({slot}, ~({mask}u << {sh})); InterlockedOr({slot}, ({v} & {mask}u) << {sh}); }}"));
            } else {
                let v = self.vreg(f("VDST"));
                let raw = format!("({slot} >> {sh})");
                self.line(&format!("if ({act}) {v} = {};", d16_merge(&v, &raw, bits / 8, signed, half)));
            }
            return Ok(());
        }
        // 8 / 16-bit LDS accesses: atomic and / or keep the rest of the dword intact
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
            let at = format!("{a} + {off16}u");
            let slot = lds(at.clone());
            let sh = format!("((({at}) & 3u) * 8u)");
            let mask = (1u64 << bits) - 1;
            if write {
                let v = self.vreg(f("DATA0"));
                self.line(&format!(
                    "if ({act}) {{ InterlockedAnd({slot}, ~({mask}u << {sh})); InterlockedOr({slot}, ({v} & {mask}u) << {sh}); }}"
                ));
            } else {
                let v = self.vreg(f("VDST"));
                let val = if signed {
                    format!("(uint)(((int)((({slot} >> {sh}) & {mask}u) << {}u)) >> {}u)", 32 - bits, 32 - bits)
                } else {
                    format!("(({slot} >> {sh}) & {mask}u)")
                };
                self.line(&format!("if ({act}) {v} = {val};"));
            }
            return Ok(());
        }
        if let Some((write, n, stride, two)) = rw(name) {
            let addrs: Vec<String> = if two {
                vec![format!("{a} + {}u", o0 * stride), format!("{a} + {}u", o1 * stride)]
            } else {
                vec![format!("{a} + {off16}u")]
            };
            let (vd, d0, d1) = (f("VDST"), f("DATA0"), f("DATA1"));
            for (e, at) in addrs.iter().enumerate() {
                for k in 0..n {
                    let slot = lds(format!("{at} + {}u", 4 * k));
                    if write {
                        let src = self.vreg(if e == 0 { d0 } else { d1 } + k);
                        st.push(format!("{slot} = {src};"));
                    } else {
                        let v = self.vreg(vd + e as u64 * n + k);
                        st.push(format!("{v} = {slot};"));
                    }
                }
            }
            self.line(&format!("if ({act}) {{ {} }}", st.join(" ")));
            if write && gs_phase {
                let lowest = addrs.iter().map(|x| format!("({x})")).reduce(|p, q| format!("min({p}, {q})")).unwrap_or_default();
                self.line(&format!("if ({act}) xcl_gs_base = min(xcl_gs_base, {lowest});"));
            }
            return Ok(());
        }
        let (op, rtn) = match name.strip_prefix("ds_") {
            Some(r) if r.contains("_rtn_") => (r.replacen("_rtn", "", 1), true),
            Some(r) => (r.to_string(), false),
            None => return Err("DS"),
        };
        // two exchanges (ds_wrxchg2(st64)_rtn_b32): old values to VDST, VDST + 1
        if op == "wrxchg2_b32" || op == "wrxchg2st64_b32" {
            let stride = if op == "wrxchg2_b32" { 4 } else { 256 };
            let (s0, s1) = (lds(format!("{a} + {}u", o0 * stride)), lds(format!("{a} + {}u", o1 * stride)));
            let (v0, v1) = (self.vreg(f("DATA0")), self.vreg(f("DATA1")));
            self.line(&format!("uint {t}_o0 = 0u, {t}_o1 = 0u; if ({act}) {{ InterlockedExchange({s0}, {v0}, {t}_o0); InterlockedExchange({s1}, {v1}, {t}_o1); }}"));
            if rtn {
                let (r0, r1) = (self.vreg(f("VDST")), self.vreg(f("VDST") + 1));
                self.line(&format!("if ({act}) {{ {r0} = {t}_o0; {r1} = {t}_o1; }}"));
            }
            return Ok(());
        }
        // operands: cmpst = (compare DATA0, new DATA1), mskor = (mask DATA0, or DATA1)
        let (d0, d1) = (self.vreg(f("DATA0")), self.vreg(f("DATA1")));
        let (v, v2) = if op.starts_with("cmpst_") { (d1, d0) } else { (d0, d1) };
        let slot = lds(format!("{a} + {off16}u"));
        let body = rmw32(&op, &At::Ref(slot), &v, &v2, &format!("{t}_o"), &t).ok_or("DS")?;
        self.line(&format!("uint {t}_o = 0u; if ({act}) {{ {body} }}"));
        if rtn {
            let r = self.vreg(f("VDST"));
            self.line(&format!("if ({act}) {r} = {t}_o;"));
        }
        Ok(())
    }

    // ---------------------------------------------------------------- whole shader

    /// Split wave64: threads of the console's thread group.
    fn x64_threads(&self) -> u32 {
        match &self.opt.mode {
            Mode::Compute(cs) => cs.threads.iter().product(),
            Mode::Mesh(ms) => ms.threads,
            _ => 64,
        }
    }

    /// Split wave64: whether the pairs meet through arrival counters (several console waves
    /// per group, or `XCL_WAVE64_SPIN` for testing) instead of group barriers.
    fn x64_spin(&self) -> bool {
        self.x64_threads() > 64 || std::env::var_os("XCL_WAVE64_SPIN").is_some()
    }

    /// The console LDS block (`xcl_lds`) is declared before the split-wave64 prelude, which
    /// keeps its exchange in the top of it.
    fn x64_lds(&self) -> bool {
        // geometry mode always has the LDS (the ES ring and the GS output)
        self.x64() && (self.uses_lds || matches!(self.opt.mode, Mode::Geometry(_)))
    }

    /// Storage and functions of the split wave64 form (`PRELUDE_X64`).
    fn x64_prelude(&self, h: &mut String) {
        let threads = self.x64_threads();
        let waves = threads.div_ceil(64).max(1) as usize;
        let slot = self.x64_words;
        // slots (console wave, half, parity), the arrival counters (console wave, half), the
        // wave counter of `xcl_x_thread`
        let (slots, counters) = (waves * 4 * slot, waves * 2);
        let _ = writeln!(h, "#define XCL_X_THREADS {threads}u\n#define XCL_X_WAVES {waves}u\n#define XCL_X_SLOT {slot}u\n#define XCL_X_SPIN {}", self.x64_spin() as u32);
        if self.x64_lds() {
            // the D3D12 limit (32 KB) is the whole block already: the exchange takes its top,
            // the program's LDS accesses stop below it (`XCL_X_LDS_TOP`)
            let base = 8192 - slots - counters - 1;
            h.push_str("groupshared uint xcl_lds[8192]; // D3D12 limit: 32 KB (the console allows 64 KB); the top holds the wave64 exchange\n");
            let _ = writeln!(h, "#define XCL_X_LDS_TOP {}u", base - 1);
            let _ = writeln!(h, "#define XCL_XS(i) xcl_lds[{base}u + (i)]\n#define XCL_XC(i) xcl_lds[{}u + (i)]\n#define XCL_XW xcl_lds[8191u]", base + slots);
        } else {
            let _ = writeln!(h, "groupshared uint xcl_xs[{slots}];\ngroupshared uint xcl_xc[{counters}];\ngroupshared uint xcl_xw;");
            h.push_str("#define XCL_XS(i) xcl_xs[(i)]\n#define XCL_XC(i) xcl_xc[(i)]\n#define XCL_XW xcl_xw\n");
        }
        h.push_str(PRELUDE_X64);
    }

    /// Split wave64, at the start of `main`: the console thread this thread stands for
    /// (`xcl_tid`, `xcl_x_thread`), its lane, half and wave, and the size of the other half.
    fn x64_entry(&self, h: &mut String) {
        h.push_str("    uint xcl_tid = xcl_x_thread(gi);\n    lane_id = xcl_tid & 63u;\n    xcl_half = (xcl_tid >> 5) & 1u;\n    xcl_cw = xcl_tid >> 6;\n");
        h.push_str("    xcl_xn = XCL_X_THREADS > ((xcl_tid & ~31u) ^ 32u) ? min(32u, XCL_X_THREADS - ((xcl_tid & ~31u) ^ 32u)) : 0u;\n");
    }

    pub fn finish(&self, body: &str) -> String {
        let mut h = String::new();
        h.push_str("// generated by XCL (RDNA2 -> HLSL), do not edit\n");
        if matches!(self.opt.mode, Mode::Pixel(_)) {
            // lane reads within a quad see helper lanes (`rla`)
            h.push_str("#define XCL_PIXEL 1\n");
        }
        h.push_str("RWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\n");
        // compute work across workgroups (bottom-up BVH builds: atomics, then reading what
        // other groups wrote) relies on GLC / cache invalidation on the console: keep the flat
        // memory coherent across the GPU there; graphics keeps the cached (faster) view
        let coherent = if matches!(self.opt.mode, Mode::Compute(_)) { "globallycoherent " } else { "" };
        let windows = self.opt.mem_windows.clamp(1, 4);
        let _ = writeln!(h, "{coherent}RWByteAddressBuffer mem : register(u2);");
        if windows > 1 {
            for k in 1..windows {
                let _ = writeln!(h, "{coherent}RWByteAddressBuffer mem{k} : register(u{});", 2 + k);
            }
            h.push_str("static uint xcl_w = 0u;\n");
            h.push_str(MEM_ACCESSORS_HERE);
        } else {
            h.push_str(LOADX4_ONE_WINDOW);
        }
        for (i, (_, cmp)) in self.static_used.iter().enumerate() {
            let k = self.opt.static_sampler_base as usize + i;
            let ty = if *cmp { "SamplerComparisonState" } else { "SamplerState" };
            let _ = writeln!(h, "{ty} xcl_ss{k} : register(s{k}, space1);");
        }
        h.push_str(PRELUDE);
        if self.x64() {
            self.x64_prelude(&mut h);
        }
        if let Some(u) = &self.upper {
            // wave64 VS / PS on a GPU with 32-lane waves only (`upper_lanes`)
            h.push_str("// the wave has 32 lanes: it is a console wave whose lanes 32..63 are not populated\nstatic bool xcl_hw32 = false;\n");
            for (v, l) in &u.shadows {
                let _ = writeln!(h, "static uint xcl_sh{v}_{l} = 0u; // v_writelane slot: v{v} lane {l}");
            }
        }
        let infos: Vec<String> = (0..BUF_FORMATS.len()).map(|i| format!("0x{:08X}u", buf_format_info(i))).collect();
        let _ = writeln!(h, "static const uint FMT_INFO[78] = {{ {} }};", infos.join(", "));
        h.push_str(PRELUDE_MEM);
        if self.uses_scratch {
            h.push_str(PRELUDE_SCRATCH);
        }
        if self.dynamic_f16_rm {
            h.push_str("static uint xcl_fp16_rm = 0u;\n");
        }
        let mut sg: BTreeSet<u64> = self.sgprs.clone();
        sg.extend([106, 107, 124, 126, 127]);
        let load_sgprs = |h: &mut String| {
            for &c in &sg {
                let _ = writeln!(h, "    uint {} = io.Load({}u);", sname(c).unwrap(), c * 4);
            }
        };
        match &self.opt.mode {
            Mode::Kernel => {
                // hw_wave 64 with a wave32 program: the wave-size independent form (two halves)
                let split = self.opt.hw_wave == 64 && !self.opt.wave64;
                let lanes = if self.opt.wave64 || split { 64 } else { 32 };
                if self.uses_lds && !self.x64_lds() {
                    h.push_str("groupshared uint xcl_lds[8192]; // D3D12 limit: 32 KB (the console allows 64 KB)\n");
                }
                // split wave64: 64 threads (the console wave) in two hardware wave32s
                let hw_lanes = if self.x64() { 32 } else { lanes };
                let _ = writeln!(h, "[WaveSize({hw_lanes})]\n[numthreads({lanes}, 1, 1)]\nvoid main(uint gi : SV_GroupIndex)\n{{");
                if split {
                    h.push_str("    uint xcl_hw_lane = WaveGetLaneIndex();\n    xcl_w32 = WaveGetLaneCount() > 32u;\n    xcl_half = xcl_w32 ? xcl_hw_lane >> 5 : 0u;\n    lane_id = xcl_hw_lane & 31u;\n");
                } else if self.x64() {
                    self.x64_entry(&mut h);
                    // the console lane: its registers in io / res
                    h.push_str("    uint xcl_hw_lane = xcl_tid;\n");
                } else {
                    h.push_str("    uint xcl_hw_lane = WaveGetLaneIndex();\n    lane_id = xcl_hw_lane;\n");
                }
                load_sgprs(&mut h);
                let _ = writeln!(h, "    uint scc = io.Load({}u);", KERNEL_SCC_SLOT * 4);
                for &v in &self.vgprs {
                    let _ = writeln!(h, "    uint v{v} = io.Load(({}u + xcl_hw_lane * {}u + {v}u) * 4u);", KERNEL_SGPRS, KERNEL_VGPRS);
                }
                h.push_str(body);
                h.push_str("    if (xcl_hw_lane == 0u) {\n");
                for &c in &sg {
                    let _ = writeln!(h, "        res.Store({}u, {});", c * 4, sname(c).unwrap());
                }
                let _ = writeln!(h, "        res.Store({}u, scc);", KERNEL_SCC_SLOT * 4);
                h.push_str("    }\n");
                if self.x64() {
                    // the upper half's scalar state too: the runner checks both halves agree
                    h.push_str("    if (xcl_tid == 32u) {\n");
                    for &c in &sg {
                        let _ = writeln!(h, "        res.Store({}u, {});", (KERNEL_SPLIT_SGPRS + c as usize) * 4, sname(c).unwrap());
                    }
                    let _ = writeln!(h, "        res.Store({}u, scc);", (KERNEL_SPLIT_SGPRS + KERNEL_SCC_SLOT) * 4);
                    let _ = writeln!(h, "        res.Store({}u, {KERNEL_SPLIT_MAGIC}u);", (KERNEL_SPLIT_SGPRS + KERNEL_SGPRS) * 4);
                    h.push_str("    }\n");
                }
                for &v in &self.vgprs {
                    let _ = writeln!(h, "    res.Store(({}u + xcl_hw_lane * {}u + {v}u) * 4u, v{v});", KERNEL_SGPRS, KERNEL_VGPRS);
                }
                h.push_str("}\n");
            }
            Mode::Vertex(vs) => {
                h.push_str("struct XclIn {\n");
                let inputs: Vec<&SigElement> = vs.inputs.iter().filter(|e| e.system_value == 0 && !e.name.starts_with("SV_")).collect();
                for e in &inputs {
                    let _ = writeln!(h, "    {}4 a{} : {};", hlsl_type(e), e.register, e.semantic());
                }
                h.push_str("    uint xcl_vid : SV_VertexID;\n    uint xcl_iid : SV_InstanceID;\n};\n");
                out_struct(&mut h, "XclOut", &vs.outputs);
                h.push_str("XclOut main(XclIn xi)\n{\n    XclOut xo = (XclOut)0;\n");
                h.push_str(if self.opt.wave64 { "    lane_id = WaveGetLaneIndex();\n" } else { "    uint xcl_hw_lane = WaveGetLaneIndex();\n    xcl_w32 = WaveGetLaneCount() > 32u;\n    xcl_half = xcl_w32 ? xcl_hw_lane >> 5 : 0u;\n    lane_id = xcl_hw_lane & 31u;\n" });
                if self.upper.is_some() {
                    h.push_str("    xcl_hw32 = WaveGetLaneCount() < 64u;\n");
                }
                load_sgprs(&mut h);
                h.push_str("    uint scc = 0u;\n    { uint2 e0 = ballot(true); exec_lo = e0.x; exec_hi = e0.y; }\n");
                for &v in &self.vgprs {
                    // SV_VertexID / SV_InstanceID listed in ISG1 come through the fetch shader like an
                    // attribute (retail VInstanced* read the instance id in v8 = r1)
                    let sv_at = vs.inputs.iter().find(|e| matches!(e.system_value, 6 | 8) && vs.input_vgpr(e.register) == Some(v));
                    let init = if v == 0 {
                        "xi.xcl_vid".to_string()
                    } else if v == 3 {
                        "xi.xcl_iid".to_string()
                    } else if let Some(e) = sv_at {
                        if e.system_value == 6 { "xi.xcl_vid" } else { "xi.xcl_iid" }.to_string()
                    } else {
                        // the fetch shader's VGPRs: four per register the program reads (`fetched_registers`)
                        inputs
                            .iter()
                            .find_map(|e| vs.input_vgpr(e.register).filter(|&b| v >= b && v < b + 4).map(|b| format!("asuint(xi.a{}[{}])", e.register, v - b)))
                            .unwrap_or_else(|| "0u".into())
                    };
                    let _ = writeln!(h, "    uint v{v} = {init};");
                }
                h.push_str(body);
                h.push_str("    return xo;\n}\n");
            }
            Mode::Compute(cs) => {
                let lanes = if self.opt.wave64 { 64 } else { 32 };
                if self.uses_lds && !self.x64_lds() {
                    h.push_str("groupshared uint xcl_lds[8192]; // D3D12 limit: 32 KB (the console allows 64 KB)\n");
                }
                let [x, y, z] = cs.threads;
                // split wave64: the same group in hardware wave32s (a console wave = two of them)
                let hw_lanes = if self.x64() { 32 } else { lanes };
                // 2-D / 3-D groups: the PC driver need not form its waves from consecutive
                // thread indices (AMD RDNA4 lays them out in 2 x 2 quads: lanes 0, 1, 4, 5, 8 …
                // of an 8 x 8 group; NVIDIA differs too), the console does – a hardware wave
                // takes the next `lanes` console threads as it arrives, lane l the l-th of them
                let reorder = !self.x64() && y * z > 1;
                if reorder {
                    // console threads handed to waves so far: the program's top LDS word while
                    // the waves take their threads (the console's LDS starts undefined; D3D12's
                    // 32 KB has no room for another word next to a full `xcl_lds`)
                    if self.uses_lds && !self.x64_lds() {
                        h.push_str("#define XCL_WT xcl_lds[8191u]\n");
                    } else {
                        h.push_str("groupshared uint xcl_wt;\n#define XCL_WT xcl_wt\n");
                    }
                }
                let _ = writeln!(
                    h,
                    "[WaveSize({hw_lanes})]\n[numthreads({x}, {y}, {z})]\nvoid main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex)\n{{"
                );
                if self.x64() {
                    // the console thread's ids (`xcl_x_thread`), not this hardware thread's
                    self.x64_entry(&mut h);
                    let _ = writeln!(h, "    gtid = uint3(xcl_tid % {x}u, (xcl_tid / {x}u) % {y}u, xcl_tid / {}u);\n    gi = xcl_tid;", x * y);
                } else if reorder {
                    let t = x * y * z;
                    let _ = writeln!(
                        h,
                        "    if (gi == 0u) XCL_WT = 0u;\n    GroupMemoryBarrierWithGroupSync();\n    lane_id = WaveGetLaneIndex();\n    {{\n        bool full = WaveActiveCountBits(true) == {lanes}u;\n        uint base = {}u;\n        if (WaveIsFirstLane() && full) InterlockedAdd(XCL_WT, {lanes}u, base);\n        uint xcl_tid = WaveReadLaneFirst(base) + lane_id;\n        gtid = uint3(xcl_tid % {x}u, (xcl_tid / {x}u) % {y}u, xcl_tid / {}u);\n        gi = xcl_tid;\n    }}\n    GroupMemoryBarrierWithGroupSync(); // all waves have their threads before the program writes LDS",
                        t & !(lanes - 1),
                        x * y
                    );
                } else {
                    h.push_str("    lane_id = WaveGetLaneIndex();\n");
                }
                load_sgprs(&mut h);
                // system SGPRs after the user data: enabled group ids, then TG_SIZE
                let mut next = cs.user_sgprs as u64;
                for (k, c) in ["x", "y", "z"].iter().enumerate() {
                    if cs.tgid[k] {
                        if sg.contains(&next) {
                            let _ = writeln!(h, "    {} = gid.{c};", sname(next).unwrap());
                        }
                        next += 1;
                    }
                }
                if cs.tg_size && sg.contains(&next) {
                    // waves in the group: a last, partial wave counts (96 threads: 2 of 64)
                    let _ = writeln!(h, "    {} = {}u | ((gi / {lanes}u) << 6);", sname(next).unwrap(), (x * y * z).div_ceil(lanes));
                }
                let _ = writeln!(h, "    uint scc = 0u;\n    {{ uint2 e0 = {}(true); exec_lo = e0.x; exec_hi = e0.y; }}", self.ballot_fn());
                for &v in &self.vgprs {
                    let init = match v {
                        0 => "gtid.x",
                        1 if cs.tid_components >= 2 => "gtid.y",
                        2 if cs.tid_components >= 3 => "gtid.z",
                        _ => "0u",
                    };
                    let _ = writeln!(h, "    uint v{v} = {init};");
                }
                h.push_str(body);
                h.push_str("}\n");
            }
            Mode::Mesh(ms) => {
                // split wave64: each console wave in two hardware wave32s (`PRELUDE_X64`)
                let native_lanes = if !self.opt.wave64 || self.x64() { 32 } else { 64 };
                let lanes = if self.opt.wave64 { 64 } else { 32 };
                if self.uses_lds && !self.x64_lds() {
                    h.push_str("groupshared uint xcl_lds[8192]; // D3D12 limit: 32 KB (the console allows 64 KB)\n");
                }
                let t = ms.threads;
                let _ = writeln!(h, "[WaveSize({native_lanes})]\n[numthreads({t}, 1, 1)]\nvoid main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID)\n{{");
                if self.x64() {
                    // the console thread's index (`xcl_x_thread`), not this hardware thread's
                    self.x64_entry(&mut h);
                    h.push_str("    gi = xcl_tid;\n");
                } else {
                    h.push_str("    lane_id = WaveGetLaneIndex();\n");
                }
                // thread index in the group as the console numbers it: wave · lanes + lane
                let _ = writeln!(h, "    uint xcl_wave = gi / {lanes}u;\n    uint xcl_slot = xcl_wave * {lanes}u + lane_id;");
                let _ = writeln!(h, "    uint xcl_rec = gid.x * {}u;", ms.record_dwords());
                let _ = writeln!(h, "    uint xcl_vtx = xcl_rec + {MESH_HEADER}u + xcl_slot * {}u;", ms.vert_dwords());
                let _ = writeln!(h, "    uint xcl_prm = xcl_rec + {}u + xcl_slot;", MESH_HEADER + ms.max_vertices as usize * ms.vert_dwords());
                load_sgprs(&mut h);
                if sg.contains(&3) {
                    h.push_str("    s3 = (s3 & 0xF0FFFFFFu) | (xcl_wave << 24);\n");
                }
                let _ = writeln!(h, "    uint scc = 0u;\n    {{ uint2 e0 = {}(true); exec_lo = e0.x; exec_hi = e0.y; }}", self.ballot_fn());
                for &v in &self.vgprs {
                    let init = if v == 5 { format!("gid.x * {t}u + xcl_slot") } else { "0u".into() };
                    let _ = writeln!(h, "    uint v{v} = {init};");
                }
                h.push_str(body);
                h.push_str("}\n");
            }
            Mode::Geometry(g) => {
                if !self.x64() {
                    let pg = super::GS_PRIMS_PER_GROUP;
                    let rec = g.record();
                    let (vd, vb) = (rec.vert_dwords(), g.vertex_bytes());
                    let (maxv, maxp) = (rec.max_vertices, rec.max_primitives);
                    let maxt = g.max_vertex_count.saturating_sub(2).max(1);
                    let nreg = g.input_registers();
                    let fetched = super::fetched_registers(&g.es_inputs);
                    h.push_str("groupshared uint xcl_lds[8192]; // D3D12 limit: 32 KB (the console allows 64 KB)\n");
                    h.push_str("[WaveSize(64)]\n[numthreads(64, 1, 1)]\nvoid main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID)\n{\n");
                    h.push_str("    lane_id = WaveGetLaneIndex();\n");
                    let _ = writeln!(h, "    uint xcl_total = io.Load({}u);\n    uint xcl_vin = io.Load({}u);", super::IO_GS_PRIMS * 4, super::IO_GS_VERTICES * 4);
                    let _ = writeln!(h, "    uint xcl_p0 = gid.x * {pg}u;\n    uint xcl_np = xcl_total > xcl_p0 ? min({pg}u, xcl_total - xcl_p0) : 0u;\n    uint xcl_nv = 3u * xcl_np;");
                    let _ = writeln!(h, "    uint xcl_rec = gid.x * {}u;", rec.record_dwords());
                    let _ = writeln!(h, "    uint xcl_emitted = 0u, xcl_strip = 0u, xcl_ntri = 0u, xcl_gs_base = 0xFFFFFFFFu;\n    uint xcl_tri[{maxt}];");
                    load_sgprs(&mut h);
                    if sg.contains(&2) {
                        let _ = writeln!(h, "    s2 = {}u; // GS output area in LDS (after the ES items)", g.es_ring_bytes());
                    }
                    let item = g.es_item_bytes() / 4;
                    if sg.contains(&3) {
                        h.push_str("    s3 = xcl_nv | (xcl_np << 8); // merged wave info: ES vertices, GS primitives, wave 0\n");
                    }
                    h.push_str("    uint scc = 0u;\n    { uint2 e0 = ballot(true); exec_lo = e0.x; exec_hi = e0.y; }\n");
                    for &v in &self.vgprs {
                        let init = match v {
                            0 => format!("(lane_id < xcl_np) ? ((lane_id * 3u * {item}u) | ((lane_id * 3u + 1u) * {item}u << 16)) : 0u"),
                            1 => format!("(lane_id < xcl_np) ? ((lane_id * 3u + 2u) * {item}u) : 0u"),
                            2 => "xcl_p0 + lane_id".to_string(),
                            5 => "xcl_p0 * 3u + lane_id".to_string(),
                            // the fetch shader's VGPRs: four per register the ES reads (`fetched_registers`);
                            // the pre-fetched vertices hold one float4 per register
                            v if v >= 9 && ((v - 9) / 4) < fetched.len() as u64 && fetched[((v - 9) / 4) as usize] < nreg => {
                                let (r, c) = (fetched[((v - 9) / 4) as usize], (v - 9) % 4);
                                format!("(lane_id < xcl_nv) ? mem.Load(xcl_vin + ((xcl_p0 * 3u + lane_id) * {nreg}u + {r}u) * 16u + {}u) : 0u", c * 4)
                            }
                            _ => "0u".to_string(),
                        };
                        let _ = writeln!(h, "    uint v{v} = {init};");
                    }
                    h.push_str(body);
                    let pos_reg = g.outputs.iter().find(|e| e.system_value == 1).map(|e| e.register).unwrap_or(0);
                    let regs: Vec<u32> = std::iter::once(pos_reg).chain(ranked(&g.outputs, true).iter().map(|e| e.register)).collect();
                    h.push_str("    uint xcl_vb = WavePrefixSum(xcl_emitted);\n    uint xcl_tb = WavePrefixSum(xcl_ntri);\n");
                    h.push_str("    uint xcl_tv = WaveActiveSum(xcl_emitted);\n    uint xcl_tt = WaveActiveSum(xcl_ntri);\n");
                    let _ = writeln!(h, "    if (lane_id == 0u) {{ res.Store(xcl_rec * 4u, min(xcl_tv, {maxv}u)); res.Store(xcl_rec * 4u + 4u, min(xcl_tt, {maxp}u)); }}");
                    let _ = writeln!(h, "    for (uint k = 0u; k < xcl_emitted && xcl_gs_base != 0xFFFFFFFFu; k++) {{\n        uint vi = xcl_vb + k;\n        if (vi >= {maxv}u) break;");
                    let _ = writeln!(h, "        uint src = xcl_gs_base + k * {vb}u;\n        uint dst = xcl_rec + {MESH_HEADER}u + vi * {vd}u;");
                    for (slot, r) in regs.iter().enumerate() {
                        for c in 0..4 {
                            let _ = writeln!(h, "        res.Store((dst + {}u) * 4u, xcl_lds[((src + {}u) >> 2) & 8191u]);", 4 * slot + c, 16 * r + 4 * c as u32);
                        }
                    }
                    h.push_str("    }\n");
                    let _ = writeln!(h, "    for (uint j = 0u; j < xcl_ntri; j++) {{\n        uint t = xcl_tri[j];\n        if (xcl_tb + j >= {maxp}u) break;");
                    let _ = writeln!(
                        h,
                        "        res.Store((xcl_rec + {}u + xcl_tb + j) * 4u, ((t & 0x3FFu) + xcl_vb) | ((((t >> 10) & 0x3FFu) + xcl_vb) << 10) | ((((t >> 20) & 0x3FFu) + xcl_vb) << 20));\n    }}",
                        MESH_HEADER + maxv as usize * vd
                    );
                    h.push_str("}\n");
                } else {
                let pg = super::GS_PRIMS_PER_GROUP;
                let rec = g.record();
                let (vd, vb) = (rec.vert_dwords(), g.vertex_bytes());
                let (maxv, maxp) = (rec.max_vertices, rec.max_primitives);
                let maxt = g.max_vertex_count.saturating_sub(2).max(1);
                let nreg = g.input_registers();
                let fetched = super::fetched_registers(&g.es_inputs);
                // split wave64 (the native form is above): the ES in both hardware wave32s, the GS
                // (at most 21 primitives) in the lower one alone; `xcl_lds` comes with `PRELUDE_X64`
                h.push_str("[WaveSize(32)]\n[numthreads(64, 1, 1)]\nvoid main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID)\n{\n");
                h.push_str("    uint xcl_tid = xcl_x_thread(gi);\n    gi = xcl_tid;\n");
                let _ = writeln!(h, "    uint xcl_total = io.Load({}u);\n    uint xcl_vin = io.Load({}u);", super::IO_GS_PRIMS * 4, super::IO_GS_VERTICES * 4);
                let _ = writeln!(h, "    uint xcl_p0 = gid.x * {pg}u;\n    uint xcl_np = xcl_total > xcl_p0 ? min({pg}u, xcl_total - xcl_p0) : 0u;\n    uint xcl_nv = 3u * xcl_np;");
                let _ = writeln!(h, "    uint xcl_rec = gid.x * {}u;", rec.record_dwords());
                let item = g.es_item_bytes() / 4; // dwords per ES vertex
                let gs_start = g.gs_start.unwrap_or(0);
                let phase = |h: &mut String, es: bool| {
                    h.push_str("    {\n");
                    let _ = writeln!(h, "        bool xcl_es_phase = {};", if es { "true" } else { "false" });
                    let _ = writeln!(h, "        uint xcl_entry_pc = {}u;", if es { 0 } else { gs_start });
                    // the GS phase has no other half to exchange with
                    let _ = writeln!(h, "        lane_id = gi; xcl_half = gi >> 5; xcl_xn = {}u;", if es { 32 } else { 0 });
                    let _ = writeln!(h, "        uint xcl_emitted = 0u, xcl_strip = 0u, xcl_ntri = 0u, xcl_gs_base = 0xFFFFFFFFu;\n        uint xcl_tri[{maxt}];");
                    load_sgprs(h);
                    if sg.contains(&2) {
                        let _ = writeln!(h, "        s2 = {}u; // GS output area in LDS (after the ES items)", g.es_ring_bytes());
                    }
                    if sg.contains(&3) {
                        h.push_str("        s3 = xcl_nv | (xcl_np << 8); // merged wave info: ES vertices, GS primitives, wave 0\n");
                    }
                    h.push_str("        uint scc = 0u;\n        { uint2 e0 = xcl_x_ballot(true); exec_lo = e0.x; exec_hi = e0.y; }\n");
                    for &v in &self.vgprs {
                        let init = match v {
                            // LDS dword offsets of the triangle's vertices (ES vertex i at i · item)
                            0 => format!("(lane_id < xcl_np) ? ((lane_id * 3u * {item}u) | ((lane_id * 3u + 1u) * {item}u << 16)) : 0u"),
                            1 => format!("(lane_id < xcl_np) ? ((lane_id * 3u + 2u) * {item}u) : 0u"),
                            2 => "xcl_p0 + lane_id".to_string(),
                            5 => "xcl_p0 * 3u + lane_id".to_string(),
                            // the fetch shader's VGPRs: four per register the ES reads (`fetched_registers`);
                            // the pre-fetched vertices hold one float4 per register
                            v if v >= 9 && ((v - 9) / 4) < fetched.len() as u64 && fetched[((v - 9) / 4) as usize] < nreg => {
                                let (r, c) = (fetched[((v - 9) / 4) as usize], (v - 9) % 4);
                                format!("(lane_id < xcl_nv) ? mem.Load(xcl_vin + ((xcl_p0 * 3u + lane_id) * {nreg}u + {r}u) * 16u + {}u) : 0u", c * 4)
                            }
                            _ => "0u".to_string(),
                        };
                        let _ = writeln!(h, "        uint v{v} = {init};");
                    }
                    h.push_str(body);
                    if es {
                        h.push_str("    }\n");
                    }
                };
                phase(&mut h, true);
                h.push_str("    GroupMemoryBarrierWithGroupSync();\n    if (gi < 32u)\n");
                phase(&mut h, false);
                // emitted vertices (LDS, register r at +16·r) and strip triangles → the record
                let pos_reg = g.outputs.iter().find(|e| e.system_value == 1).map(|e| e.register).unwrap_or(0);
                let regs: Vec<u32> = std::iter::once(pos_reg).chain(ranked(&g.outputs, true).iter().map(|e| e.register)).collect();
                h.push_str("    uint xcl_vb = WavePrefixSum(xcl_emitted);\n    uint xcl_tb = WavePrefixSum(xcl_ntri);\n");
                // wave sums with every lane active (not inside the lane-0 branch)
                h.push_str("    uint xcl_tv = WaveActiveSum(xcl_emitted);\n    uint xcl_tt = WaveActiveSum(xcl_ntri);\n");
                let _ = writeln!(h, "    if (lane_id == 0u) {{ res.Store(xcl_rec * 4u, min(xcl_tv, {maxv}u)); res.Store(xcl_rec * 4u + 4u, min(xcl_tt, {maxp}u)); }}");
                let _ = writeln!(h, "    for (uint k = 0u; k < xcl_emitted && xcl_gs_base != 0xFFFFFFFFu; k++) {{\n        uint vi = xcl_vb + k;\n        if (vi >= {maxv}u) break;");
                let _ = writeln!(h, "        uint src = xcl_gs_base + k * {vb}u;\n        uint dst = xcl_rec + {MESH_HEADER}u + vi * {vd}u;");
                for (slot, r) in regs.iter().enumerate() {
                    for c in 0..4 {
                        let _ = writeln!(h, "        res.Store((dst + {}u) * 4u, xcl_lds[((src + {}u) >> 2) & 8191u]);", 4 * slot + c, 16 * r + 4 * c as u32);
                    }
                }
                h.push_str("    }\n");
                let _ = writeln!(h, "    for (uint j = 0u; j < xcl_ntri; j++) {{\n        uint t = xcl_tri[j];\n        if (xcl_tb + j >= {maxp}u) break;");
                let _ = writeln!(
                    h,
                    "        res.Store((xcl_rec + {}u + xcl_tb + j) * 4u, ((t & 0x3FFu) + xcl_vb) | ((((t >> 10) & 0x3FFu) + xcl_vb) << 10) | ((((t >> 20) & 0x3FFu) + xcl_vb) << 20));\n    }}",
                    MESH_HEADER + maxv as usize * vd
                );
                h.push_str("    }\n}\n");
                }
            }
            Mode::Pixel(ps) => {
                // conservative depth output (SV_DepthGreater/LessEqual): DXIL wants the position
                // input non-perspective centroid (or sample)
                let conservative = ps.outputs.iter().any(|e| matches!(e.system_value, 67 | 68));
                let pos_mode = if conservative { "noperspective centroid " } else { "" };
                let _ = writeln!(h, "struct XclIn {{\n    {pos_mode}float4 xcl_pos : SV_Position;");
                for (k, e) in ranked(&ps.inputs, false).iter().enumerate() {
                    // the mode of the attribute this input is read through (none: never read)
                    let attr = ps_slot_attribute(ps, k);
                    let mut m = attr.and_then(|a| self.ps_modes.get(&(a as u32))).cloned().unwrap_or_default();
                    if e.component_type != 3 {
                        m = "nointerpolation".into();
                    }
                    let _ = writeln!(h, "    {m} {}4 a{k} : {};", hlsl_type(e), e.semantic());
                }
                if ps.input_ena >> 12 & 1 == 1 {
                    h.push_str("    bool xcl_ff : SV_IsFrontFace;\n");
                }
                if ps.input_ena >> 14 & 1 == 1 {
                    h.push_str("    uint xcl_cov : SV_Coverage;\n");
                }
                h.push_str("};\nstruct XclOut {\n");
                for e in &ps.outputs {
                    match e.system_value {
                        64 => {
                            let _ = writeln!(h, "    {}4 t{} : SV_Target{};", hlsl_type(e), e.index, e.index);
                        }
                        65 => h.push_str("    float xcl_depth : SV_Depth;\n"),
                        67 => h.push_str("    float xcl_depth : SV_DepthGreaterEqual;\n"),
                        68 => h.push_str("    float xcl_depth : SV_DepthLessEqual;\n"),
                        _ => {}
                    }
                }
                if !ps.outputs.iter().any(|e| matches!(e.system_value, 64 | 65 | 67 | 68)) {
                    h.push_str("    float4 xcl_unused : SV_Target7;\n");
                }
                h.push_str("};\nXclOut main(XclIn xi)\n{\n    XclOut xo = (XclOut)0;\n    bool xcl_live = false;\n");
                h.push_str(if self.opt.wave64 { "    lane_id = WaveGetLaneIndex();\n" } else { "    uint xcl_hw_lane = WaveGetLaneIndex();\n    xcl_w32 = WaveGetLaneCount() > 32u;\n    xcl_half = xcl_w32 ? xcl_hw_lane >> 5 : 0u;\n    lane_id = xcl_hw_lane & 31u;\n" });
                if self.upper.is_some() {
                    h.push_str("    xcl_hw32 = WaveGetLaneCount() < 64u;\n");
                }
                load_sgprs(&mut h);
                h.push_str("    uint scc = 0u;\n    { uint2 e0 = ballot(true); exec_lo = e0.x; exec_hi = e0.y; }\n");
                let sys = |bit: u32| ps_input_vgpr(ps.input_ena, bit);
                for &v in &self.vgprs {
                    let init = if Some(v) == sys(8) {
                        "asuint(xi.xcl_pos.x)"
                    } else if Some(v) == sys(9) {
                        "asuint(xi.xcl_pos.y)"
                    } else if Some(v) == sys(10) {
                        "asuint(xi.xcl_pos.z)"
                    } else if Some(v) == sys(11) {
                        "asuint(xi.xcl_pos.w)"
                    } else if Some(v) == sys(12) {
                        "asuint(xi.xcl_ff ? 1.0f : 0.0f)"
                    } else if Some(v) == sys(14) {
                        "xi.xcl_cov"
                    } else {
                        "0u"
                    };
                    let _ = writeln!(h, "    uint v{v} = {init};");
                }
                h.push_str(body);
                if self.ps_done {
                    h.push_str("    if (!xcl_live) discard;\n");
                }
                h.push_str("    return xo;\n}\n");
            }
        }
        if windows > 1 {
            let coherent = if matches!(self.opt.mode, Mode::Compute(_)) { "globallycoherent " } else { "" };
            h = windowed(&h, windows, self.opt.heap_windows.then_some(coherent));
        }
        h
    }
}

/// Vertex output struct shared by the VS mode and the mesh pass-through: SV_Position, then
/// the params in `exp param` order (non-system OSG1 elements, then clip/cull distances).
fn out_struct(h: &mut String, name: &str, outputs: &[SigElement]) {
    let _ = writeln!(h, "struct {name} {{\n    float4 pos : SV_Position;");
    for (k, e) in ranked(outputs, true).iter().enumerate() {
        let n = out_comps(e);
        let ty = if n == 1 { hlsl_type(e).to_string() } else { format!("{}{n}", hlsl_type(e)) };
        let _ = writeln!(h, "    {ty} p{k} : {};", e.semantic());
    }
    h.push_str("};\n");
}

/// Geometry shader for the console's rect lists (IASetPrimitiveTopology 17, AMD's
/// DI_PT_RECTLIST; PSO topology type 19) behind a translated VS with these outputs: the PC has
/// no such topology, the draw arrives as a triangle list and each rect's three vertices come
/// here. They are three corners of a screen-aligned rectangle (AMD PAL, `palCmdBuffer.h`:
/// "Each rect is three 2D axis-aligned rectangle vertices"), and the corner between the other
/// two is not always the same vertex: Mesa's radeonsi blits send (x1,y1), (x1,y2), (x2,y1) – v0
/// – and xf86-video-ati's r600 EXA (x1,y1), (x1,y2), (x2,y2) – v1. The corner is the vertex
/// sharing x with one of the others and y with the other; the fourth vertex is the opposite
/// corner, a + b - corner, with every float output taken the same way (both drivers' texture
/// coordinates rely on the implied corner continuing them linearly). Only for three vertices
/// with one positive w – then a + b - corner is the same in clip and screen space and every
/// interpolation mode agrees; three that are not such corners, or differ in w, stay the one
/// triangle they make.
///
/// Flat values are the rect's first vertex's: both triangles start with v[0], which is where
/// the PC takes a flat input from (an integer input, or what the translation declares
/// `nointerpolation` for `v_interp_mov_f32`), and the fourth vertex's integer outputs are
/// v[0]'s. On the console the first vertex is the provoking one of every primitive under
/// D3D's setting – AMD's register references (R6xx, SI, CIK), PA_SU_SC_MODE_CNTL
/// .PROVOKING_VTX_LAST: "Defines which vertex of a primitive is used for attribute components
/// when flat shading is enabled", "0 = First Vtx (D3D)"; the Evergreen / Cayman programming
/// guide, 5.2: a flat attribute is the provoking vertex's value with gradients of 0. Neither
/// says so of a rect in particular, and it is not measured on one.
pub fn rect_list_gs(outputs: &[SigElement]) -> String {
    let mut h = String::new();
    h.push_str("// generated by XCL: a console rect list's rectangles (three corners, the fourth implied)\n");
    out_struct(&mut h, "XclOut", outputs);
    h.push_str("[maxvertexcount(6)]\nvoid main(triangle XclOut v[3], inout TriangleStream<XclOut> s)\n{\n");
    // the corner c: x shared with a and y with b, or the other way round (a = c + 1, b = c + 2);
    // one w: clip-space x and y compare as the screen's do
    h.push_str("    uint c = 3u;\n");
    h.push_str("    if (v[0].pos.w > 0.0 && v[1].pos.w == v[0].pos.w && v[2].pos.w == v[0].pos.w) {\n");
    h.push_str("        float2 q[3] = { v[0].pos.xy, v[1].pos.xy, v[2].pos.xy };\n");
    h.push_str("        [unroll] for (uint i = 0u; i < 3u; i++) {\n");
    h.push_str("            float2 p = q[i], a = q[(i + 1u) % 3u], b = q[(i + 2u) % 3u];\n");
    h.push_str("            bool corner = (p.x == a.x && p.y == b.y) || (p.x == b.x && p.y == a.y);\n");
    h.push_str("            bool degenerate = all(p == a) || all(p == b) || all(a == b);\n");
    h.push_str("            if (c == 3u && corner && !degenerate)\n                c = i;\n        }\n    }\n");
    h.push_str("    if (c == 3u) {\n        s.Append(v[0]);\n        s.Append(v[1]);\n        s.Append(v[2]);\n        return;\n    }\n");
    h.push_str("    XclOut k = v[c], a = v[(c + 1u) % 3u], b = v[(c + 2u) % 3u];\n    XclOut d = v[0];\n");
    h.push_str("    d.pos = a.pos + b.pos - k.pos;\n");
    for (k, e) in ranked(outputs, true).iter().enumerate() {
        if hlsl_type(e) == "float" {
            let _ = writeln!(h, "    d.p{k} = a.p{k} + b.p{k} - k.p{k};");
        }
    }
    // (corner, a, b) is v0 v1 v2 in its order or rotated; (corner, a, d), (corner, d, b) and
    // (a, d, b) wind as it does. Of those, the two halves that both hold v0, v0 first:
    // v0 v1 d, v0 d v2 (corner v0) – v0 v1 v2, v0 v2 d (v1) – v0 v1 v2, v0 d v1 (v2)
    h.push_str("    s.Append(v[0]);\n    s.Append(v[1]);\n");
    h.push_str("    if (c == 0u)\n        s.Append(d);\n    else\n        s.Append(v[2]);\n");
    h.push_str("    s.RestartStrip();\n    s.Append(v[0]);\n");
    h.push_str("    if (c == 1u) {\n        s.Append(v[2]);\n        s.Append(d);\n    } else {\n        s.Append(d);\n");
    h.push_str("        if (c == 0u)\n            s.Append(v[2]);\n        else\n            s.Append(v[1]);\n    }\n}\n");
    h
}

/// Second pass of the mesh mode: a mesh shader that turns the records written by the
/// translated program (one per group, see `MsInterface::record_dwords`) into vertices and
/// triangles. Null primitives (bit 31) and out-of-range indices are culled.
pub fn mesh_passthrough(ms: &MsInterface) -> String {
    mesh_passthrough_for(ms, None)
}

/// `mesh_passthrough` whose vertices carry exactly what a pixel shader with `ps_inputs`
/// (its input signature) reads, declared as the translated PS declares them – same
/// elements, order and types, so both signatures pack alike and link. Each comes from the
/// record slot of the program's output with that semantic; outputs the PS does not read are
/// dropped (the 32 KB mesh output limit) and inputs the program does not write read 0 (on
/// the console the driver links attributes by index and tolerates both). Clip / cull
/// distances stay.
pub fn mesh_passthrough_for(ms: &MsInterface, ps_inputs: Option<&[SigElement]>) -> String {
    let mut h = String::new();
    h.push_str("// generated by XCL: mesh pass-through for a translated NGG program\n");
    h.push_str("RWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\n");
    let slots = ranked(&ms.outputs, true);
    // (declaration, record slot or None = 0)
    let elems: Vec<(String, &SigElement, Option<usize>)> = match ps_inputs {
        None => slots.iter().enumerate().map(|(k, e)| (e.semantic(), *e, Some(k))).collect(),
        Some(ins) => {
            let mut v: Vec<(String, &SigElement, Option<usize>)> = ranked(ins, false)
                .into_iter()
                .map(|e| {
                    let k = slots.iter().position(|o| o.system_value == 0 && o.name.eq_ignore_ascii_case(&e.name) && o.index == e.index);
                    (e.semantic(), e, k)
                })
                .collect();
            v.extend(slots.iter().enumerate().filter(|(_, o)| matches!(o.system_value, 2 | 3)).map(|(k, o)| (o.semantic(), *o, Some(k))));
            v
        }
    };
    let _ = writeln!(h, "struct XclVert {{\n    float4 pos : SV_Position;");
    for (k, (sem, e, _)) in elems.iter().enumerate() {
        let n = if ps_inputs.is_some() && !matches!(e.system_value, 2 | 3) { 4 } else { out_comps(e) };
        let ty = if n == 1 { hlsl_type(e).to_string() } else { format!("{}{n}", hlsl_type(e)) };
        let _ = writeln!(h, "    {ty} p{k} : {sem};");
    }
    h.push_str("};\n");
    let (mv, mp, vd, rec) = (ms.max_vertices, ms.max_primitives, ms.vert_dwords(), ms.record_dwords());
    let t = ms.threads.clamp(1, 128);
    // one vertex from record dword address `b` into `o`
    let mut fill = String::from("        XclVert o;\n        o.pos = asfloat(res.Load4(b));\n");
    for (k, (_, e, slot)) in elems.iter().enumerate() {
        let n = if ps_inputs.is_some() && !matches!(e.system_value, 2 | 3) { 4 } else { out_comps(e) };
        let sw = ["x", "y", "z", "w"][..n].concat();
        let raw = match slot {
            Some(s) => format!("res.Load4(b + {}u).{sw}", 16 * (s + 1)),
            None => format!("uint4(0u, 0u, 0u, 0u).{sw}"),
        };
        let v = match e.component_type {
            1 => raw,
            2 => format!("asint({raw})"),
            _ => format!("asfloat({raw})"),
        };
        let _ = writeln!(fill, "        o.p{k} = {v};");
    }
    let tri_base = MESH_HEADER + mv as usize * vd;
    let split = passthrough_split(ms, elems.len());
    if split == 1 {
        let _ = writeln!(
            h,
            "[outputtopology(\"triangle\")]\n[numthreads({t}, 1, 1)]\nvoid main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID, out vertices XclVert verts[{mv}], out indices uint3 tris[{mp}])\n{{"
        );
        let _ = writeln!(h, "    uint rec = gid.x * {rec}u;");
        let _ = writeln!(h, "    uint nv = min(res.Load(rec * 4u), {mv}u);\n    uint np = min(res.Load(rec * 4u + 4u), {mp}u);");
        h.push_str("    SetMeshOutputCounts(nv, np);\n");
        let _ = writeln!(h, "    for (uint i = gi; i < nv; i += {t}u) {{");
        let _ = writeln!(h, "        uint b = (rec + {MESH_HEADER}u + i * {vd}u) * 4u;");
        h.push_str(&fill);
        h.push_str("        verts[i] = o;\n    }\n");
        let _ = writeln!(h, "    for (uint j = gi; j < np; j += {t}u) {{");
        let _ = writeln!(h, "        uint p = res.Load((rec + {tri_base}u + j) * 4u);");
        h.push_str("        uint3 x = uint3(p & 0x3FFu, (p >> 10) & 0x3FFu, (p >> 20) & 0x3FFu);\n");
        h.push_str("        bool bad = or((p >> 31) != 0u, any(x >= nv));\n");
        h.push_str("        tris[j] = select(bad, uint3(0u, 0u, 0u), x);\n    }\n}\n");
        return h;
    }
    // over the mesh output limit: `split` mesh groups per record, each a run of its
    // triangles with their own copies of the three vertices
    let ppg = mp.div_ceil(split);
    let _ = writeln!(
        h,
        "[outputtopology(\"triangle\")]\n[numthreads({t}, 1, 1)]\nvoid main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID, out vertices XclVert verts[{}], out indices uint3 tris[{ppg}])\n{{",
        3 * ppg
    );
    let _ = writeln!(h, "    uint rec = (gid.x / {split}u) * {rec}u;\n    uint first = (gid.x % {split}u) * {ppg}u;");
    let _ = writeln!(h, "    uint nv = min(res.Load(rec * 4u), {mv}u);\n    uint np = min(res.Load(rec * 4u + 4u), {mp}u);");
    let _ = writeln!(h, "    uint n = select(np > first, min(np - first, {ppg}u), 0u);");
    h.push_str("    SetMeshOutputCounts(3u * n, n);\n");
    let _ = writeln!(h, "    for (uint j = gi; j < n; j += {t}u) {{");
    let _ = writeln!(h, "        uint p = res.Load((rec + {tri_base}u + first + j) * 4u);");
    h.push_str("        uint3 x = uint3(p & 0x3FFu, (p >> 10) & 0x3FFu, (p >> 20) & 0x3FFu);\n");
    h.push_str("        bool bad = or((p >> 31) != 0u, any(x >= nv));\n");
    h.push_str("        [unroll] for (uint c = 0u; c < 3u; c++) {\n");
    let _ = writeln!(h, "        uint b = (rec + {MESH_HEADER}u + select(bad, 0u, x[c]) * {vd}u) * 4u;");
    h.push_str(&fill);
    h.push_str("        verts[3u * j + c] = o;\n        }\n");
    h.push_str("        tris[j] = select(bad, uint3(0u, 0u, 0u), uint3(3u * j, 3u * j + 1u, 3u * j + 2u));\n    }\n}\n");
    h
}

/// D3D12's limit on what one mesh shader group outputs (vertex + primitive attributes).
const MESH_OUTPUT_BYTES: u32 = 32768;

/// Mesh groups the pass-through spends on one record: 1 when the record's vertices fit the
/// mesh output limit as they are, else the triangles are split into runs whose own vertex
/// copies (3 per triangle) fit. `elements`: output elements besides SV_Position.
pub fn passthrough_split(ms: &MsInterface, elements: usize) -> u32 {
    let vbytes = 16 * (1 + elements as u32);
    if vbytes * ms.max_vertices <= MESH_OUTPUT_BYTES {
        return 1;
    }
    let ppg = (MESH_OUTPUT_BYTES / (3 * vbytes)).max(1);
    ms.max_primitives.max(1).div_ceil(ppg)
}

/// Output elements (besides SV_Position) of `mesh_passthrough_for` for these inputs.
pub fn passthrough_elements(ms: &MsInterface, ps_inputs: Option<&[SigElement]>) -> usize {
    let slots = ranked(&ms.outputs, true);
    match ps_inputs {
        None => slots.len(),
        Some(ins) => ranked(ins, false).len() + slots.iter().filter(|o| matches!(o.system_value, 2 | 3)).count(),
    }
}

fn hlsl_type(e: &SigElement) -> &'static str {
    match e.component_type {
        1 => "uint",
        2 => "int",
        _ => "float",
    }
}

/// Components of an output element: clip/cull distances by their mask, the rest as 4-vectors.
fn out_comps(e: &SigElement) -> usize {
    if matches!(e.system_value, 2 | 3) {
        (8 - e.mask.leading_zeros()) as usize
    } else {
        4
    }
}

/// Interpolation modifier of every PS attribute from the I/J pair its v_interp uses.
pub fn ps_attr_modes(insts: &[Decoded], ena: u32) -> BTreeMap<u32, String> {
    let mut out = BTreeMap::new();
    for d in insts {
        let name = d.inst.name.to_ascii_lowercase();
        if !name.starts_with("v_interp_") {
            continue;
        }
        let (_, ij, attr, _) = Em::interp_fields(d);
        let mode = if name == "v_interp_mov_f32" {
            "nointerpolation"
        } else {
            let pair = [(0u32, "sample"), (1, ""), (2, "centroid"), (4, "noperspective sample"), (5, "noperspective"), (6, "noperspective centroid")]
                .into_iter()
                .find(|&(bit, _)| ps_input_vgpr(ena, bit).is_some_and(|b| ij == b || ij == b + 1));
            match pair {
                Some((_, m)) => m,
                None => continue,
            }
        };
        out.entry(attr as u32).or_insert_with(|| mode.to_string());
    }
    out
}

/// `lt` → ("<", signed) for s_cmp_* / s_cmpk_* suffixes like "eq_i32".
fn cmp_op(rest: &str) -> Option<(&'static str, bool)> {
    let (op, ty) = rest.rsplit_once('_')?;
    let signed = ty == "i32";
    Some((
        match op {
            "eq" => "==",
            "lg" => "!=",
            "gt" => ">",
            "ge" => ">=",
            "lt" => "<",
            "le" => "<=",
            _ => return None,
        },
        signed,
    ))
}

/// Where a 32-bit atomic works: a byte address in `mem`, or an lvalue (LDS slot, texel).
enum At {
    Mem(String),
    Ref(String),
}

/// A 32-bit atomic read-modify-write: HLSL statements leaving the old value in `old` (a
/// declared uint). `v` = data, `v2` = second operand (cmpswap: the compare value; DS mskor:
/// the or value). Operations without an HLSL intrinsic run as a bounded compare-exchange
/// loop. Operation names as in buffer / flat / image (`umin`) and DS (`min_u32`).
fn rmw32(op: &str, at: &At, v: &str, v2: &str, old: &str, t: &str) -> Option<String> {
    let direct = |f: &str, val: &str| match at {
        At::Mem(a) => format!("mem.{f}({a}, {val}, {old});"),
        At::Ref(r) => format!("{f}({r}, {val}, {old});"),
    };
    let cas = |new: String| {
        let (load, cx) = match at {
            At::Mem(a) => (format!("mem.Load({a})"), format!("mem.InterlockedCompareExchange({a}, {t}_c, {t}_n, {t}_p);")),
            At::Ref(r) => (r.clone(), format!("InterlockedCompareExchange({r}, {t}_c, {t}_n, {t}_p);")),
        };
        format!(
            "uint {t}_c = {load}; [allow_uav_condition] for (uint {t}_i = 0u; {t}_i < 4096u; {t}_i++) {{ uint {t}_n = {new}; uint {t}_p = 0u; {cx} if ({t}_p == {t}_c) break; {t}_c = {t}_p; }} {old} = {t}_c;"
        )
    };
    let c = format!("{t}_c");
    let signed = |f: &str, pick: &str| match at {
        At::Mem(a) => format!("int {t}_s = 0; mem.{f}({a}, (int){v}, {t}_s); {old} = (uint){t}_s;"),
        At::Ref(_) => cas(format!("(uint){pick}((int){c}, (int){v})")),
    };
    Some(match op {
        "add" | "add_u32" => direct("InterlockedAdd", v),
        "sub" | "sub_u32" => direct("InterlockedAdd", &format!("(0u - {v})")),
        "and" | "and_b32" => direct("InterlockedAnd", v),
        "or" | "or_b32" => direct("InterlockedOr", v),
        "xor" | "xor_b32" => direct("InterlockedXor", v),
        "swap" | "wrxchg_b32" => direct("InterlockedExchange", v),
        "umin" | "min_u32" => direct("InterlockedMin", v),
        "umax" | "max_u32" => direct("InterlockedMax", v),
        // float compare-swap compares bits (differs only for -0 / NaN)
        "cmpswap" | "fcmpswap" | "cmpst_b32" | "cmpst_f32" => match at {
            At::Mem(a) => format!("mem.InterlockedCompareExchange({a}, {v2}, {v}, {old});"),
            At::Ref(r) => format!("InterlockedCompareExchange({r}, {v2}, {v}, {old});"),
        },
        "smin" | "min_i32" => signed("InterlockedMin", "min"),
        "smax" | "max_i32" => signed("InterlockedMax", "max"),
        "fmin" | "fmax" | "min_f32" | "max_f32" => {
            let min = op == "fmin" || op == "min_f32";
            match at {
                At::Mem(a) => float_minmax_mem(min, false, a, v, old, t),
                At::Ref(_) => cas(format!("asuint({}(asfloat({c}), asfloat({v})))", if min { "min" } else { "max" })),
            }
        }
        "rsub" | "rsub_u32" => cas(format!("{v} - {c}")),
        "inc" | "inc_u32" => cas(format!("select({c} >= {v}, 0u, {c} + 1u)")),
        "dec" | "dec_u32" => cas(format!("select(or({c} == 0u, {c} > {v}), {v}, {c} - 1u)")),
        "csub" => cas(format!("select({c} >= {v}, {c} - {v}, 0u)")),
        "add_f32" => cas(format!("asuint(asfloat({c}) + asfloat({v}))")),
        "mskor_b32" => cas(format!("({c} & ~{v}) | {v2}")),
        _ => return None,
    })
}

/// A 64-bit atomic at byte address `a` of the flat memory (SM 6.6 InterlockedXxx64);
/// `v`, `cmp`, `old` are uint64_t variables.
fn rmw64(op: &str, a: &str, v: &str, cmp: &str, old: &str, t: &str) -> Option<String> {
    if matches!(op, "inc" | "dec") {
        let c = format!("{t}_c");
        let new = if op == "inc" {
            format!("({c} >= {v}) ? (uint64_t)0 : {c} + (uint64_t)1")
        } else {
            format!("({c} == (uint64_t)0 || {c} > {v}) ? {v} : {c} - (uint64_t)1")
        };
        return Some(format!(
            "uint2 {t}_l = mem.Load2({a}); uint64_t {t}_c = (uint64_t){t}_l.x | ((uint64_t){t}_l.y << 32); [allow_uav_condition] for (uint {t}_i = 0u; {t}_i < 4096u; {t}_i++) {{ uint64_t {t}_n = {new}; uint64_t {t}_p = 0; mem.InterlockedCompareExchange64({a}, {t}_c, {t}_n, {t}_p); if ({t}_p == {t}_c) break; {t}_c = {t}_p; }} {old} = {t}_c;"
        ));
    }
    let f = match op {
        "add" | "sub" => "InterlockedAdd64",
        "and" => "InterlockedAnd64",
        "or" => "InterlockedOr64",
        "xor" => "InterlockedXor64",
        "swap" => "InterlockedExchange64",
        "umin" => "InterlockedMin64",
        "umax" => "InterlockedMax64",
        "smin" | "smax" => {
            let f = if op == "smin" { "InterlockedMin64" } else { "InterlockedMax64" };
            return Some(format!("int64_t {t}_s = 0; mem.{f}({a}, (int64_t){v}, {t}_s); {old} = (uint64_t){t}_s;"));
        }
        // float compare-swap compares the whole 64-bit payload bitwise (like rmw32's
        // fcmpswap): differs from a numeric f64 compare only for -0 / NaN
        "cmpswap" | "fcmpswap" => return Some(format!("mem.InterlockedCompareExchange64({a}, {cmp}, {v}, {old});")),
        "fmin" | "fmax" => return Some(float_minmax_mem(op == "fmin", true, a, v, old, t)),
        _ => return None,
    };
    let val = if op == "sub" { format!("((uint64_t)0 - {v})") } else { v.to_string() };
    Some(format!("mem.{f}({a}, {val}, {old});"))
}

/// Float min / max at byte address `a` of the flat memory as one integer atomic (no CAS loop:
/// a program with 168 such loops was too much for the driver's compiler). Non-negative floats
/// order like signed ints, negative ones like unsigned ints in reverse. The operand takes the
/// signed path iff its bits are at most those of +inf: +0.0 … +inf. -0.0 (sign bit set)
/// compares equal to 0.0 as a float but orders below every non-negative value here, so it
/// takes the unsigned path, as do negative values and NaNs. `x2`: f64 (`v`, `old` are
/// uint64_t), else f32 (uint).
fn float_minmax_mem(min: bool, x2: bool, a: &str, v: &str, old: &str, t: &str) -> String {
    let (pos, neg) = if min { ("InterlockedMin", "InterlockedMax") } else { ("InterlockedMax", "InterlockedMin") };
    if x2 {
        format!(
            "if ({v} <= 0x7FF0000000000000ull) {{ int64_t {t}_s = 0; mem.{pos}64({a}, (int64_t){v}, {t}_s); {old} = (uint64_t){t}_s; }} else {{ mem.{neg}64({a}, {v}, {old}); }}"
        )
    } else {
        format!("if ({v} <= 0x7F800000u) {{ int {t}_s = 0; mem.{pos}({a}, asint({v}), {t}_s); {old} = asuint({t}_s); }} else {{ mem.{neg}({a}, {v}, {old}); }}")
    }
}

/// Placeholder where the memory window accessors go (after the `mem` declarations).
const MEM_ACCESSORS_HERE: &str = "//XCL_MEM_ACCESSORS\n";

/// Four dwords at four addresses (scalar buffer loads with their bounds checks); with several
/// windows `mem_accessors` defines it, with one switch for all four.
const LOADX4_ONE_WINDOW: &str = "uint4 xcl_loadx4(uint4 a) { return uint4(mem.Load(a.x), mem.Load(a.y), mem.Load(a.z), mem.Load(a.w)); }\n";

/// Several memory windows: every `mem.Method(` of the program (and of the preludes) becomes
/// an accessor `xcl_mMethod(` that switches on `xcl_w` between `mem` and `mem1`.., whose
/// definitions replace the placeholder.
/// `heap`: Some(storage class) – the accessors take window k from the descriptor heap
/// (`Options::heap_windows`) instead of switching over the root UAVs.
fn windowed(h: &str, windows: u32, heap: Option<&str>) -> String {
    let mut out = String::with_capacity(h.len() + 8192);
    let mut rest = h;
    while let Some(i) = rest.find("mem.") {
        let before_ok = rest[..i].chars().next_back().is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
        let tail = &rest[i + 4..];
        let name_len = tail.find(|c: char| !c.is_ascii_alphanumeric()).unwrap_or(tail.len());
        if before_ok && tail[name_len..].starts_with('(') {
            out.push_str(&rest[..i]);
            out.push_str("xcl_m");
            out.push_str(&tail[..name_len]);
            rest = &tail[name_len..];
        } else {
            out.push_str(&rest[..i + 4]);
            rest = tail;
        }
    }
    out.push_str(rest);
    out.replacen(MEM_ACCESSORS_HERE, &mem_accessors(windows, heap), 1)
}

/// The accessor functions: one per `mem` method (and overload) the translator emits.
fn mem_accessors(windows: u32, heap: Option<&str>) -> String {
    let mut sigs: Vec<(&str, String, &str, &str)> = vec![
        ("uint", "Load".into(), "uint a", "a"),
        ("uint2", "Load2".into(), "uint a", "a"),
        ("uint3", "Load3".into(), "uint a", "a"),
        ("uint4", "Load4".into(), "uint a", "a"),
        ("void", "Store".into(), "uint a, uint v", "a, v"),
        ("void", "Store2".into(), "uint a, uint2 v", "a, v"),
        ("void", "Store3".into(), "uint a, uint3 v", "a, v"),
        ("void", "Store4".into(), "uint a, uint4 v", "a, v"),
        ("void", "InterlockedCompareExchange".into(), "uint a, uint c, uint v, out uint o", "a, c, v, o"),
        ("void", "InterlockedCompareExchange64".into(), "uint a, uint64_t c, uint64_t v, out uint64_t o", "a, c, v, o"),
    ];
    for op in ["Add", "And", "Or", "Xor", "Exchange", "Min", "Max"] {
        sigs.push(("void", format!("Interlocked{op}"), "uint a, uint v, out uint o", "a, v, o"));
        if op != "Exchange" {
            sigs.push(("void", format!("Interlocked{op}"), "uint a, uint v", "a, v"));
        }
        sigs.push(("void", format!("Interlocked{op}64"), "uint a, uint64_t v, out uint64_t o", "a, v, o"));
        if op == "Min" || op == "Max" {
            sigs.push(("void", format!("Interlocked{op}"), "uint a, int v, out int o", "a, v, o"));
            sigs.push(("void", format!("Interlocked{op}64"), "uint a, int64_t v, out int64_t o", "a, v, o"));
        }
    }
    let mut s = String::new();
    if let Some(coherent) = heap {
        // one access: window k is descriptor MEM_HEAP_BASE + k (a window the program does not
        // have reads window 0, as the switch's default did)
        let _ = writeln!(s, "#define XCL_MW {coherent}RWByteAddressBuffer m = ResourceDescriptorHeap[NonUniformResourceIndex({}u + (xcl_w < {windows}u ? xcl_w : 0u))]", super::MEM_HEAP_BASE);
        let _ = writeln!(s, "uint4 xcl_loadx4(uint4 a) {{ XCL_MW; return uint4(m.Load(a.x), m.Load(a.y), m.Load(a.z), m.Load(a.w)); }}");
        for (ret, name, params, args) in sigs {
            let r = if ret == "void" { "" } else { "return " };
            let _ = writeln!(s, "{ret} xcl_m{name}({params}) {{ XCL_MW; {r}m.{name}({args}); }}");
        }
        return s;
    }
    let _ = write!(s, "uint4 xcl_loadx4(uint4 a) {{ [branch] switch (xcl_w) {{ ");
    for k in 1..windows {
        let _ = write!(s, "case {k}u: return uint4(mem{k}.Load(a.x), mem{k}.Load(a.y), mem{k}.Load(a.z), mem{k}.Load(a.w)); ");
    }
    let _ = writeln!(s, "default: return uint4(mem.Load(a.x), mem.Load(a.y), mem.Load(a.z), mem.Load(a.w)); }} }}");
    for (ret, name, params, args) in sigs {
        let r = if ret == "void" { "" } else { "return " };
        let _ = write!(s, "{ret} xcl_m{name}({params}) {{ [branch] switch (xcl_w) {{ ");
        for k in 1..windows {
            let _ = write!(s, "case {k}u: {r}mem{k}.{name}({args}); break; ");
        }
        let _ = writeln!(s, "default: {r}mem.{name}({args}); break; }} }}");
    }
    s
}

/// Sub-dword d16 forms: (bytes, sign-extend, store, half: 0 low / 1 high) by the name after
/// the `buffer_` / `flat_` / `global_` prefix.
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

/// FLAT / global sub-dword forms (plain and d16): (bytes, sign-extend, store, half: 0 low,
/// 1 high, 2 = the whole VGPR).
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

/// DS d16 forms: (bits, sign-extend, write, half).
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

/// The data a d16 store takes from VGPR `v` (half 1: the high 16 bits).
fn d16_data(v: &str, half: u32) -> String {
    if half == 1 {
        format!("({v} >> 16)")
    } else {
        v.to_string()
    }
}

/// VGPR `v` after loading `n` bytes (`raw`, low bits) as half 0 / 1 (the other half kept)
/// or as the whole register (half 2).
fn d16_merge(v: &str, raw: &str, n: u32, signed: bool, half: u32) -> String {
    let bits = 8 * n;
    let val = if signed { format!("(uint)(((int)(({raw}) << {}u)) >> {}u)", 32 - bits, 32 - bits) } else { format!("(({raw}) & {}u)", (1u64 << bits) - 1) };
    match half {
        0 => format!("({v} & 0xFFFF0000u) | ({val} & 0xFFFFu)"),
        1 => format!("({v} & 0xFFFFu) | ({val} << 16)"),
        _ => val,
    }
}

/// Per-lane private memory of scratch_* (static = thread-private in HLSL; 4 KB per lane).
const PRELUDE_SCRATCH: &str = r#"
static uint xcl_scratch[1024];
uint scr_ld(uint a) { uint b = (a >> 2) & 1023u; uint s = (a & 3u) * 8u; uint lo = xcl_scratch[b]; uint hi = xcl_scratch[(b + 1u) & 1023u]; return select(s == 0u, lo, (lo >> s) | (hi << ((32u - s) & 31u))); }
void scr_st(uint a, uint v, uint n) {
    uint b = (a >> 2) & 1023u; uint s = (a & 3u) * 8u; uint m = select(n >= 4u, 0xFFFFFFFFu, (1u << ((8u * n) & 31u)) - 1u);
    xcl_scratch[b] = (xcl_scratch[b] & ~(m << s)) | ((v & m) << s);
    if (s != 0u && s + 8u * n > 32u) { uint b2 = (b + 1u) & 1023u; xcl_scratch[b2] = (xcl_scratch[b2] & ~(m >> (32u - s))) | ((v & m) >> (32u - s)); }
}
"#;

const PRELUDE_MEM: &str = r#"
uint4 ld_raw(uint a) {
    // one Load4 + one Load (with several memory windows each is a switch) and no branch
    uint b = a & ~3u; uint s = (a & 3u) * 8u;
    uint4 w = mem.Load4(b); uint w4 = mem.Load(b + 16u);
    uint4 n = uint4(w.yzw, w4);
    return select((s == 0u).xxxx, w, (w >> s) | (n << ((32u - s) & 31u)));
}
// format helpers: no loops, no branches (inlined at every buffer access; `?:` and `if`
// on scalars become control flow in HLSL 2021, which DXC is slow to clean up)
uint bits_at(uint4 r, uint off, uint w) {
    uint i = off >> 5; uint s = off & 31u;
    uint lo = r[i]; uint hi = select(i < 3u, r[min(i + 1u, 3u)], 0u);
    uint v = select(s == 0u, lo, (lo >> s) | (hi << ((32u - s) & 31u)));
    return select(w >= 32u, v, v & ((1u << (w & 31u)) - 1u));
}
float uf_small(uint v, uint mb) {
    uint e = v >> mb; uint m = v & ((1u << mb) - 1u); float fm = (float)m / (float)(1u << mb);
    float special = select(m != 0u, asfloat(0x7FC00000u), asfloat(0x7F800000u));
    return select(e == 31u, special, select(e == 0u, fm * exp2(-14.0f), exp2((float)e - 15.0f) * (1.0f + fm)));
}
uint fmt_bytes(uint info) { return (((info >> 8) & 63u) + ((info >> 14) & 63u) + ((info >> 20) & 63u) + ((info >> 26) & 63u)) / 8u; }
uint fmt_comp(uint4 r, uint info, uint c) {
    uint n = info & 7u; uint ty = (info >> 3) & 7u;
    bool isint = or(ty == 4u, ty == 5u);
    uint off = select(c > 0u, (info >> 8) & 63u, 0u) + select(c > 1u, (info >> 14) & 63u, 0u) + select(c > 2u, (info >> 20) & 63u, 0u);
    uint w = (info >> ((8u + 6u * c) & 31u)) & 63u;
    uint v = bits_at(r, off, w);
    uint sh = (32u - w) & 31u;
    int sv = select(w >= 32u, (int)v, ((int)(v << sh)) >> sh);
    float maxu = select(w >= 32u, 4294967295.0f, (float)((1u << (w & 31u)) - 1u));
    float maxs = select(w >= 32u, 2147483647.0f, (float)((1u << ((w - 1u) & 31u)) - 1u));
    float fv = (float)v; float fsv = (float)sv;
    // exact endpoints (the hardware conversion is exact; DXC divides approximately)
    uint f_un = asuint(select(v == 0u, 0.0f, select(fv >= maxu, 1.0f, fv / maxu)));
    uint f_sn = asuint(select(sv == 0, 0.0f, select(fsv >= maxs, 1.0f, max(fsv / maxs, -1.0f))));
    uint flt = select(w == 32u, v, select(w == 16u, asuint(f16tof32(v)), asuint(uf_small(v, select(w == 11u, 6u, 5u)))));
    uint res = select(ty == 0u, f_un, select(ty == 1u, f_sn, select(ty == 2u, asuint(fv), select(ty == 3u, asuint(fsv), select(ty == 4u, v, select(ty == 5u, (uint)sv, flt))))));
    return select(c >= n, select(c == 3u, select(isint, 1u, 0x3F800000u), 0u), res);
}
// image_bvh(64)_intersect_ray: type = node & 7, at = low 32 bits of base + (node & ~7) * 8.
// box32 (5) / box16 (4): 4 child ids + boxes → hit children (closest first when sorting),
// misses 0xFFFFFFFF; triangle (0..3): (t numerator, denominator, i, j numerators), miss t = inf;
// instance / AABB nodes are left to the shader (all 0xFFFFFFFF).
uint4 bvh_intersect(uint type, uint at, float tmax, float3 o, float3 d, float3 inv, bool sort) {
    if (type == 4u || type == 5u) {
        uint4 child = mem.Load4(at);
        float tn[4]; uint id[4];
        float3 ninf = asfloat(0xFF800000u), pinf = asfloat(0x7F800000u);
        [unroll] for (uint k = 0u; k < 4u; k++) {
            float3 mn, mx;
            if (type == 5u) { mn = asfloat(mem.Load3(at + 16u + 24u * k)); mx = asfloat(mem.Load3(at + 28u + 24u * k)); }
            else { uint3 h = mem.Load3(at + 16u + 12u * k); mn = float3(f16tof32(h.x), f16tof32(h.x >> 16), f16tof32(h.y)); mx = float3(f16tof32(h.y >> 16), f16tof32(h.z), f16tof32(h.z >> 16)); }
            float3 t1 = (mn - o) * inv, t2 = (mx - o) * inv;
            bool3 in_slab = and(o >= mn, o <= mx);
            float3 lo = select(isinf(inv), select(in_slab, ninf, pinf), min(t1, t2));
            float3 hi = select(isinf(inv), select(in_slab, pinf, ninf), max(t1, t2));
            float a = max(max(lo.x, lo.y), lo.z);
            float b = min(min(hi.x, hi.y), hi.z);
            bool valid = and(child[k] != 0xFFFFFFFFu, all(mn <= mx));
            bool hit = and(and(valid, a <= b), and(b >= 0.0f, a <= tmax));
            tn[k] = hit ? a : asfloat(0x7F800000u);
            id[k] = hit ? child[k] : 0xFFFFFFFFu;
        }
        if (sort) {
            // sorting network for 4: (0,1) (2,3) (0,2) (1,3) (1,2)
            uint2 pairs[5] = { uint2(0, 1), uint2(2, 3), uint2(0, 2), uint2(1, 3), uint2(1, 2) };
            [unroll] for (uint p = 0u; p < 5u; p++) {
                uint i = pairs[p].x, j = pairs[p].y;
                if (tn[j] < tn[i]) { float ft = tn[i]; tn[i] = tn[j]; tn[j] = ft; uint ui = id[i]; id[i] = id[j]; id[j] = ui; }
            }
        }
        return uint4(id[0], id[1], id[2], id[3]);
    }
    if (type <= 3u) {
        // GPURT gfx10 node: a pair of triangles, type 0 = (v0, v1, v2), type 1 = (v1, v3, v2)
        bool t1 = type == 1u;
        float3 v0 = asfloat(mem.Load3(at + (t1 ? 12u : 0u))), v1 = asfloat(mem.Load3(at + (t1 ? 36u : 12u))), v2 = asfloat(mem.Load3(at + 24u));
        float3 e1 = v1 - v0, e2 = v2 - v0, p = cross(d, e2);
        float det = dot(e1, p);
        float3 sv = o - v0, q = cross(sv, e1);
        float i = dot(sv, p), j = dot(d, q), t = dot(e2, q);
        // inside when i, j, 1 - i - j >= 0 relative to det (either winding); avoid det * det underflow/overflow
        bool hit = det > 0.0f ? (i >= 0.0f && j >= 0.0f && i + j <= det) : (det < 0.0f && i <= 0.0f && j <= 0.0f && i + j >= det);
        // barycentrics back to the original vertex order: triangle id @60, per triangle I / J source
        float bc[4] = { det - i - j, i, j, 0.0f };
        uint tid = mem.Load(at + 60u) >> (8u * type);
        return hit ? uint4(asuint(t), asuint(det), asuint(bc[tid & 3u]), asuint(bc[(tid >> 2) & 3u])) : uint4(0x7F800000u, asuint(1.0f), 0u, 0u);
    }
    return uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
}
uint img_cls(uint fmt) { uint ty = select(fmt <= 77u, (FMT_INFO[min(fmt, 77u)] >> 3) & 7u, 6u); return select(ty == 4u, 1u, select(ty == 5u, 2u, 0u)); }
void put_bits(inout uint4 r, uint off, uint w, uint v) {
    uint m = w >= 32u ? 0xFFFFFFFFu : ((1u << w) - 1u); v &= m;
    uint i = off >> 5; uint s = off & 31u;
    r[i] |= v << s;
    if (s != 0u && s + w > 32u && i < 3u) r[i + 1u] |= v >> (32u - s);
}
uint fmt_encode(uint x, uint info, uint c) {
    uint ty = (info >> 3) & 7u; uint w = (info >> (8u + 6u * c)) & 63u; float f = asfloat(x);
    float maxu = w >= 32u ? 4294967295.0f : (float)((1u << w) - 1u);
    float maxs = w >= 32u ? 2147483647.0f : (float)((1u << (w - 1u)) - 1u);
    if (ty == 0u) return (uint)round(saturate(f) * maxu);
    if (ty == 1u) return (uint)(int)round((isnan(f) ? 0.0f : clamp(f, -1.0f, 1.0f)) * maxs);
    if (ty == 2u) return (uint)f;
    if (ty == 3u) return (uint)(int)f;
    if (ty == 4u || ty == 5u || w == 32u) return x;
    if (w == 16u) return f32tof16(f);
    if (isnan(f)) return w == 11u ? 0x7E0u : 0x3F0u;
    if (f <= 0.0f) return 0u;
    uint h = f32tof16(f);
    return w == 11u ? ((h >> 4) & 0x7FFu) : ((h >> 5) & 0x3FFu);
}
uint4 fmt_pack(uint4 v, uint info) {
    uint4 r = uint4(0u, 0u, 0u, 0u); uint n = info & 7u; uint off = 0u;
    [unroll] for (uint c = 0u; c < 4u; c++) { if (c < n) { uint w = (info >> (8u + 6u * c)) & 63u; put_bits(r, off, w, fmt_encode(v[c], info, c)); off += w; } }
    return r;
}
void st_part(uint b, uint m, uint v) { mem.InterlockedAnd(b, ~m); mem.InterlockedOr(b, v & m); }
void st_bytes(uint a, uint4 r, uint n) {
    uint s = (a & 3u) * 8u; uint b = a & ~3u;
    if (s == 0u && (n & 3u) == 0u) { [unroll] for (uint k = 0u; k < 4u; k++) if (k * 4u < n) mem.Store(b + 4u * k, r[k]); return; }
    uint m = n >= 4u ? 0xFFFFFFFFu : ((1u << (8u * n)) - 1u);
    st_part(b, m << s, r.x << s);
    if (s != 0u && s + 8u * n > 32u) st_part(b + 4u, m >> (32u - s), r.x >> (32u - s));
}
uint dst_sel(uint4 c, uint sel, bool isint) { return select(sel == 0u, 0u, select(sel == 1u, select(isint, 1u, 0x3F800000u), c[(sel - 4u) & 3u])); }
"#;

/// A console wave64 as two hardware wave32 halves of one thread group (docs/wave64-on-wave32.md):
/// per-lane code is unchanged, and the halves meet ("exchange") wherever a value has to be
/// the whole wave's – masks, lane reads, coherent scalar loads – so both hold the same SGPRs
/// and take the same branches. Follows `PRELUDE` (`lane_id`, `xcl_half`); the emitter
/// defines `XCL_X_SLOT` (words per exchange), `XCL_X_SPIN` and the storage `XCL_XS(i)` /
/// `XCL_XC(i)` before it.
const PRELUDE_X64: &str = r#"
// console wave in the group; exchanges so far (the same in both halves); lanes of the other
// half (0: it does not exist – nothing to exchange with)
static uint xcl_cw = 0u;
static uint xcl_xk = 0u;
static uint xcl_xn = 0u;
// the other half did not arrive in time (a desynchronised pair): no more exchanges, the
// wave finishes with its own half's values instead of hanging the GPU
static bool xcl_x_dead = false;
// The console thread (index in the group) this hardware thread stands for. The threads of a
// group are interchangeable, but the console forms its waves from consecutive thread
// indices and the PC GPU need not (NVIDIA does not for 3-D groups such as 8 x 8 x 4): each
// hardware wave takes the next 32 console threads as it arrives (a partial wave the last
// ones), and the program sees those threads' ids. Also zeroes the arrival counters.
uint xcl_x_thread(uint gi) {
    if (gi == 0u) XCL_XW = 0u;
    if (gi < 2u * XCL_X_WAVES) XCL_XC(gi) = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint n = WaveActiveCountBits(true);
    uint base = XCL_X_THREADS & ~31u;
    if (WaveIsFirstLane() && n == 32u) InterlockedAdd(XCL_XW, 32u, base);
    return WaveReadLaneFirst(base) + WavePrefixCountBits(true);
}
// a slot per (console wave, half, parity of the exchange): the other half can be at most one
// exchange ahead, so it never overwrites the slot being read
uint xcl_x_at(uint half, uint k) { return ((xcl_cw * 2u + half) * 2u + (k & 1u)) * XCL_X_SLOT; }
void xcl_x_put(uint i, uint v) { XCL_XS(xcl_x_at(xcl_half, xcl_xk) + i) = v; }
uint xcl_x_get(uint i) { return XCL_XS(xcl_x_at(xcl_half ^ 1u, xcl_xk - 1u) + i); }
// meet the other half after `xcl_x_put`: true = its words are readable (`xcl_x_get`)
bool xcl_x_meet() {
    uint k = xcl_xk;
    xcl_xk = k + 1u;
    if (xcl_xn == 0u) return false;
#if XCL_X_SPIN
    // several console waves per group: a group barrier would wait for the other console
    // waves too, whose control flow may differ – the pair meets through arrival counters
    // instead. Every lane fences its slot store; the wave's lane count (a wave operation:
    // all its lanes got here) goes into this half's counter in one atomic, and one lane
    // waits for the other half's count (one atomic per poll), the others for its verdict.
    if (xcl_x_dead) return false;
    GroupMemoryBarrier();
    uint lanes = WaveActiveCountBits(true);
    uint ok = 1u;
    if (WaveIsFirstLane()) {
        uint o;
        InterlockedAdd(XCL_XC(xcl_cw * 2u + xcl_half), lanes, o);
        uint need = (k + 1u) * xcl_xn;
        [loop] for (uint n = 0u; ; n++) {
            uint c;
            InterlockedAdd(XCL_XC(xcl_cw * 2u + (xcl_half ^ 1u)), 0u, c);
            if ((int)(c - need) >= 0) break;
            if (n >= (1u << 20)) { ok = 0u; break; }
        }
    }
    if (WaveReadLaneFirst(ok) == 0u) { xcl_x_dead = true; return false; }
    GroupMemoryBarrier();
#else
    // one console wave per group: its two halves are the whole group and reach every
    // exchange together
    GroupMemoryBarrierWithGroupSync();
#endif
    return true;
}
uint2 xcl_x_ballot(bool b) {
    uint m = WaveActiveBallot(b).x;
    xcl_x_put(0u, m);
    uint o = xcl_x_meet() ? xcl_x_get(0u) : 0u;
    return xcl_half != 0u ? uint2(o, m) : uint2(m, o);
}
// v_readlane / v_readfirstlane: the same lane index in both halves; the owner half's value
uint xcl_x_rla_uniform(uint v, uint l) {
    uint m = WaveReadLaneAt(v, l & 31u);
    xcl_x_put(0u, m);
    uint o = xcl_x_meet() ? xcl_x_get(0u) : 0u;
    return (l >> 5) == xcl_half ? m : o;
}
// ds_bpermute: any lane of the wave
uint xcl_x_rla_any(uint v, uint l) {
    xcl_x_put(WaveGetLaneIndex(), v);
    uint o = xcl_x_meet() ? xcl_x_get(l & 31u) : 0u;
    uint m = WaveReadLaneAt(v, l & 31u);
    return (l >> 5) == xcl_half ? m : o;
}
// a mask only ever read lane by lane (v_cndmask, carry-ins; `x64_masks`): each half's own
// word, the other half's 0 – no exchange
uint2 xcl_x_ballot_local(bool b) {
    uint m = WaveActiveBallot(b).x;
    return xcl_half != 0u ? uint2(0u, m) : uint2(m, 0u);
}
// DPP16, DPP8, ds_swizzle, v_permlane(x)16: the ISA keeps the source in the reader's half
uint xcl_x_rla_half(uint v, uint l) { return WaveReadLaneAt(v, l & 31u); }
"#;

const PRELUDE: &str = r#"
static uint lane_id;
// loop iterations a lane may run (`LOOP_GUARD`): far above what a working program needs
static uint xcl_budget = 1u << 16;
// the high half of a 32 x 32-bit product, from 16-bit parts: DXC turns
// `(uint)(((uint64_t)a * b) >> 32) != 0` (the console's integer division code) into
// umul.with.overflow, which DXIL does not allow (validation fails)
uint mul_hi_u32(uint a, uint b) {
    uint al = a & 0xFFFFu, ah = a >> 16, bl = b & 0xFFFFu, bh = b >> 16;
    uint lh = al * bh, hl = ah * bl;
    uint mid = ((al * bl) >> 16) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    return ah * bh + (lh >> 16) + (hl >> 16) + (mid >> 16);
}
// wave32 program in a 64-lane hardware wave: each half is its own wave (xcl_half = 0 / 1)
static bool xcl_w32 = false;
static uint xcl_half = 0u;
bool lanebit(uint lo, uint hi) { return ((select(lane_id < 32u, lo, hi) >> (lane_id & 31u)) & 1u) != 0u; }
uint2 ballot(bool b) {
    uint4 m = WaveActiveBallot(b);
    return uint2(select(and(xcl_w32, xcl_half != 0u), m.y, m.x), select(xcl_w32, 0u, m.y));
}
uint rla(uint v, uint l) {
    uint w = WaveReadLaneAt(v, l + 32u * xcl_half);
#ifdef XCL_PIXEL
    // A lane of the reader's own quad through a quad read: those see helper lanes (wave reads
    // do not before SM 6.7), and the console's derivatives (DPP quad_perm, ds_swizzle) read the
    // helper pixels of a quad a primitive's edge splits.
    uint q0 = QuadReadLaneAt(v, 0u), q1 = QuadReadLaneAt(v, 1u), q2 = QuadReadLaneAt(v, 2u), q3 = QuadReadLaneAt(v, 3u);
    uint q = (l & 2u) != 0u ? ((l & 1u) != 0u ? q3 : q2) : ((l & 1u) != 0u ? q1 : q0);
    // A helper lane of another quad: after s_wqm the console's v_readfirstlane often picks
    // one (the first lane of EXEC on a primitive's edge), and a wave read of it is undefined
    // here (0 in practice: every other quad of the wave got the wrong value, e.g. a shader's
    // draw type, and took the wrong branch). Its own quad sees it through a quad read; the
    // reader takes that from a live lane of the same quad.
    uint h = l + 32u * xcl_half;
    uint4 xcl_lv = WaveActiveBallot(true);
    uint nib = (xcl_lv[min(h >> 5, 3u)] >> (h & 28u)) & 15u;
    uint src = (h & ~3u) + firstbitlow(nib);
    uint r0 = WaveReadLaneAt(q0, src), r1 = WaveReadLaneAt(q1, src), r2 = WaveReadLaneAt(q2, src), r3 = WaveReadLaneAt(q3, src);
    uint r = (l & 2u) != 0u ? ((l & 1u) != 0u ? r3 : r2) : ((l & 1u) != 0u ? r1 : r0);
    if (nib != 0u && ((nib >> (h & 3u)) & 1u) == 0u) w = r;
    if ((l >> 2) == (lane_id >> 2)) w = q;
#endif
    return w;
}
bool lanebit_at(uint lo, uint hi, uint l) { return ((select(l < 32u, lo, hi) >> (l & 31u)) & 1u) != 0u; }
uint sdwa_sel(uint x, uint sel, bool sext) {
    if (sel >= 6u) return x;
    uint sh = sel < 4u ? 8u * sel : 16u * (sel - 4u); uint w = sel < 4u ? 8u : 16u;
    uint v = (x >> sh) & ((1u << w) - 1u);
    return sext ? (uint)(((int)(v << (32u - w))) >> (32u - w)) : v;
}
uint sdwa_dst(uint r, uint old, uint sel, uint unused) {
    if (sel >= 6u) return r;
    uint sh = sel < 4u ? 8u * sel : 16u * (sel - 4u); uint w = sel < 4u ? 8u : 16u;
    uint m = ((1u << w) - 1u) << sh; uint v = (r << sh) & m;
    if (unused == 2u) return (old & ~m) | v;
    if (unused == 1u) { uint top = sh + w; bool s = ((r >> (w - 1u)) & 1u) != 0u; return v | ((s && top < 32u) ? (0xFFFFFFFFu << top) : 0u); }
    return v;
}
float cube_id(float x, float y, float z) { if (abs(z) >= abs(x) && abs(z) >= abs(y)) return z < 0.0f ? 5.0f : 4.0f; if (abs(y) >= abs(x)) return y < 0.0f ? 3.0f : 2.0f; return x < 0.0f ? 1.0f : 0.0f; }
float cube_sc(float x, float y, float z) { if (abs(z) >= abs(x) && abs(z) >= abs(y)) return z < 0.0f ? -x : x; if (abs(y) >= abs(x)) return x; return x < 0.0f ? z : -z; }
float cube_tc(float x, float y, float z) { if (abs(z) >= abs(x) && abs(z) >= abs(y)) return -y; if (abs(y) >= abs(x)) return y < 0.0f ? -z : z; return -y; }
float cube_ma(float x, float y, float z) { if (abs(z) >= abs(x) && abs(z) >= abs(y)) return 2.0f * z; if (abs(y) >= abs(x)) return 2.0f * y; return 2.0f * x; }
// AMD cube coordinates (sc/|2ma| + 1.5, tc/|2ma| + 1.5, face) back to a direction
float3 cube_dir(float s, float t, float face) {
    float u = (s - 1.5f) * 2.0f; float v = (t - 1.5f) * 2.0f; uint f = ((uint)round(face)) % 8u;
    if (f == 0u) return float3(1.0f, -v, -u);
    if (f == 1u) return float3(-1.0f, -v, u);
    if (f == 2u) return float3(u, 1.0f, v);
    if (f == 3u) return float3(u, -1.0f, -v);
    if (f == 4u) return float3(u, -v, 1.0f);
    return float3(-u, -v, -1.0f);
}
// the cube of a cube array the face coordinate (face + 8 · slice) is in
float cube_slice(float face) { return (float)(((uint)round(face)) / 8u); }
uint first_lane(uint lo, uint hi) { return select(lo != 0u, firstbitlow(lo), select(hi != 0u, 32u + firstbitlow(hi), 0u)); }
uint lanemask_lt_lo() { return lane_id >= 32u ? 0xFFFFFFFFu : (lane_id == 0u ? 0u : (0xFFFFFFFFu >> (32u - lane_id))); }
uint lanemask_lt_hi() { return lane_id <= 32u ? 0u : (0xFFFFFFFFu >> (64u - lane_id)); }
uint wqm(uint m) { uint q = m | (m >> 1); q = q | (q >> 2); q &= 0x11111111u; return q * 15u; }
float mul_legacy(float a, float b) { return select(or(a == 0.0f, b == 0.0f), 0.0f, a * b); }
uint absd8(uint a, uint b, uint k) { uint x = (a >> k) & 0xFFu; uint y = (b >> k) & 0xFFu; return max(x, y) - min(x, y); }
uint absd16(uint a, uint b, uint k) { uint x = (a >> k) & 0xFFFFu; uint y = (b >> k) & 0xFFFFu; return max(x, y) - min(x, y); }
uint sad_u8(uint a, uint b) { return absd8(a, b, 0u) + absd8(a, b, 8u) + absd8(a, b, 16u) + absd8(a, b, 24u); }
uint msad_u8(uint a, uint b) { return select((b & 0xFFu) != 0u, absd8(a, b, 0u), 0u) + select(((b >> 8) & 0xFFu) != 0u, absd8(a, b, 8u), 0u) + select(((b >> 16) & 0xFFu) != 0u, absd8(a, b, 16u), 0u) + select((b >> 24) != 0u, absd8(a, b, 24u), 0u); }
uint lerp8(uint a, uint b, uint c, uint k) { return ((((a >> k) & 0xFFu) + ((b >> k) & 0xFFu) + ((c >> k) & 1u)) >> 1) << k; }
uint lerp_u8(uint a, uint b, uint c) { return lerp8(a, b, c, 0u) | lerp8(a, b, c, 8u) | lerp8(a, b, c, 16u) | lerp8(a, b, c, 24u); }
uint sat_u32(uint64_t r) { return select(r > (uint64_t)0xFFFFFFFFu, 0xFFFFFFFFu, (uint)r); }
uint sat_i32(int64_t r) { return select(r > (int64_t)0x7FFFFFFF, 0x7FFFFFFFu, select(r < -(int64_t)0x80000000u, 0x80000000u, (uint)r)); }
float div_fixup(float q, float d, float n) { return select(or(or(isnan(d), isnan(n)), or(d == 0.0f, or(isinf(d), isinf(n)))), n / d, q); }
float fma_legacy(float a, float b, float c) { return (a == 0.0f || b == 0.0f) ? c : mad(a, b, c); }
float mad_nf(float a, float b, float c) { precise float p = a * b; precise float r = p + c; return r; }
// IEEE-754 binary32 -> binary16 under RDNA2 MODE[3:2]: 0 = RNE, 1 = +inf, 2 = -inf, 3 = RTZ.
uint f32_to_f16_rm(float f, uint rm) {
    uint x = asuint(f);
    uint sign = (x >> 16) & 0x8000u;
    uint ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) return sign | select(ax == 0x7F800000u, 0x7C00u, 0x7E00u | ((ax >> 13) & 0x01FFu));
    if (ax == 0u) return sign;
    bool away = or(rm == 0u, or(and(rm == 1u, sign == 0u), and(rm == 2u, sign != 0u)));
    if (ax >= 0x47800000u) return sign | select(away, 0x7C00u, 0x7BFFu);
    if (ax < 0x33800000u) {
        bool inc = select(rm == 0u, ax > 0x33000000u, or(and(rm == 1u, sign == 0u), and(rm == 2u, sign != 0u)));
        return sign | (uint)inc;
    }
    uint h, rem, half_bit;
    if (ax >= 0x38800000u) {
        h = (((ax >> 23) - 112u) << 10) | ((ax >> 13) & 0x3FFu);
        rem = ax & 0x1FFFu;
        half_bit = 0x1000u;
    } else {
        uint sh = 126u - (ax >> 23);
        uint sig = 0x00800000u | (ax & 0x007FFFFFu);
        h = sig >> sh;
        half_bit = 1u << (sh - 1u);
        rem = sig & ((1u << sh) - 1u);
    }
    bool rne_inc = or(rem > half_bit, and(rem == half_bit, (h & 1u) != 0u));
    bool dir_inc = and(rem != 0u, or(and(rm == 1u, sign == 0u), and(rm == 2u, sign != 0u)));
    bool inc = select(rm == 0u, rne_inc, dir_inc);
    return sign | (h + (uint)inc);
}
float mad_nf_f16(float a, float b, float c, uint rm) { precise float p = a * b; float p16 = f16tof32(f32_to_f16_rm(p, rm)); precise float r = p16 + c; return r; }
float med3f(float a, float b, float c) { return max(min(a, b), min(max(a, b), c)); }
uint med3u(uint a, uint b, uint c) { return max(min(a, b), min(max(a, b), c)); }
int med3i(int a, int b, int c) { return max(min(a, b), min(max(a, b), c)); }
// x - floor(x) below 1.0 (RDNA: a NaN stays NaN and ±inf gives NaN – min() would return the
// clamp for both: HLSL min returns its non-NaN operand)
float fract_f(float x) { float r = x - floor(x); return select(r >= 1.0f, asfloat(0x3F7FFFFFu), r); }
uint cvt_u32(float f) { return select(or(isnan(f), f <= 0.0f), 0u, select(f >= 4294967296.0f, 0xFFFFFFFFu, (uint)f)); }
int cvt_i32(float f) { return select(isnan(f), 0, select(f <= -2147483648.0f, (int)0x80000000u, select(f >= 2147483648.0f, 0x7FFFFFFF, (int)f))); }
int i24(uint v) { return ((int)(v << 8)) >> 8; }
uint bfe_u(uint v, uint o, uint w) { o &= 31u; w &= 31u; return select(w == 0u, 0u, (v >> o) & (0xFFFFFFFFu >> ((32u - w) & 31u))); }
int bfe_i(uint v, uint o, uint w) { o &= 31u; w &= 31u; uint s = (32u - w) & 31u; return select(w == 0u, 0, ((int)(((uint)(((int)v) >> o)) << s)) >> s); }
uint sbfe_u(uint v, uint c) { uint o = c & 31u; uint w = (c >> 16) & 0x7Fu; return select(w == 0u, 0u, select(w >= 32u, v >> o, (v >> o) & (0xFFFFFFFFu >> ((32u - w) & 31u)))); }
int sbfe_i(uint v, uint c) { uint o = c & 31u; uint w = (c >> 16) & 0x7Fu; uint s = (32u - w) & 31u; return select(w == 0u, 0, select(w >= 32u, ((int)v) >> o, ((int)(((uint)(((int)v) >> o)) << s)) >> s)); }
uint ffbh_i(uint v) { uint x = select((int)v < 0, ~v, v); return select(x == 0u, 0xFFFFFFFFu, 31u - firstbithigh(x)); }
uint alignbit(uint s0, uint s1, uint s) { s &= 31u; return select(s == 0u, s1, (s1 >> s) | (s0 << ((32u - s) & 31u))); }
uint perm_b32(uint s0, uint s1, uint sel) {
    uint r = 0u;
    [unroll] for (uint i = 0u; i < 4u; i++) {
        uint k = (sel >> (8u * i)) & 0xFFu; uint b;
        if (k >= 13u) b = 0xFFu; else if (k == 12u) b = 0u;
        else if (k >= 8u) { uint s = (k == 8u) ? (s1 >> 15) : ((k == 9u) ? (s1 >> 31) : ((k == 10u) ? (s0 >> 15) : (s0 >> 31))); b = (s & 1u) != 0u ? 0xFFu : 0u; }
        else b = ((k < 4u) ? (s1 >> (8u * k)) : (s0 >> (8u * (k - 4u)))) & 0xFFu;
        r |= b << (8u * i);
    }
    return r;
}
uint f16_rtz(float f) { uint h = f32tof16(f) & 0xFFFFu; return h - (uint)and(and(!isnan(f), (h & 0x7FFFu) != 0u), abs(f16tof32(h)) > abs(f)); }
// Exact binary16 → unorm16/snorm16 (round-to-nearest-even, NaN → 0, ±Inf/≥1 → ±scale, subnormals exact).
// Avoids fp32 double-rounding (e.g. 0x3801 * 65535 = 32799.4995 → 32799, not 32800) and DXIL clamp(NaN).
uint cvt_norm_f16(uint h, uint scale) {
    if ((h & 0x7FFFu) > 0x7C00u) return 0u;
    uint exp = (h >> 10) & 0x1Fu;
    uint frac = h & 0x3FFu;
    if (exp >= 15u) return scale;
    uint m = exp == 0u ? frac : (1024u | frac);
    uint s = exp == 0u ? 24u : (25u - exp);
    uint prod = m * scale;
    uint q = prod >> s;
    uint rem = prod & ((1u << s) - 1u);
    uint half_bit = 1u << (s - 1u);
    return q + (uint)(rem > half_bit || (rem == half_bit && (q & 1u) != 0u));
}
uint cvt_norm_u16_f16(uint h) {
    if ((h & 0x8000u) != 0u) return 0u;
    return cvt_norm_f16(h, 65535u);
}
uint cvt_norm_i16_f16(uint h) {
    uint r = cvt_norm_f16(h, 32767u);
    return ((h & 0x8000u) != 0u ? (uint)(-(int)r) : r) & 0xFFFFu;
}
uint cvt_norm_i16_f32(float f) {
    return (uint)(int)round((isnan(f) ? 0.0f : clamp(f, -1.0f, 1.0f)) * 32767.0f) & 0xFFFFu;
}
float frexp_mant(float x) { float e; return (isinf(x) || isnan(x)) ? x : frexp(x, e); }
int frexp_exp(float x) { float e; frexp(x, e); return (isinf(x) || isnan(x)) ? 0 : (int)e; }
// Exact f32 x * 2^n on the IEEE-754 bits: handles |n| >= 127, ±0, ±inf, NaN, and subnormals
// without exp2() overflow/underflow or FTZ flushing.
float ldexp_f(float x, int n) {
    uint u = asuint(x);
    uint sign = u & 0x80000000u;
    uint exp = (u >> 23) & 0xFFu;
    uint mant = u & 0x7FFFFFu;
    if (exp == 0xFFu || (exp == 0u && mant == 0u)) return x;
    int e = (int)exp;
    if (exp == 0u) {
        uint top = firstbithigh(mant);
        mant = (mant << (23u - top)) & 0x7FFFFFu;
        e = (int)top - 22;
    }
    int eo = e + clamp(n, -500, 500);
    if (eo >= 255) return asfloat(sign | 0x7F800000u);
    if (eo >= 1) return asfloat(sign | ((uint)eo << 23) | mant);
    uint sh = (uint)(1 - eo);
    if (sh > 24u) return asfloat(sign);
    uint sig = 0x00800000u | mant;
    uint q = sig >> sh;
    uint half_bit = 1u << (sh - 1u);
    uint rem = sig & ((1u << sh) - 1u);
    bool inc = rem > half_bit || (rem == half_bit && (q & 1u) != 0u);
    return asfloat(sign | (q + (uint)inc));
}
// f64 frexp pair: input = mant * 2^exp with |mant| in [0.5, 1); denormals are normalized.
// 0/±inf/NaN: mant = input, exp = 0 (per the ISA and the amdgcn frexp intrinsics).
// Bit-exact by construction – HLSL frexp() only takes float and would narrow the input.
double frexp_mant_d(double x) {
    uint l, h; asuint(x, l, h);
    uint e = (h >> 20) & 0x7FFu;
    uint m = h & 0xFFFFFu;
    if (e == 0x7FFu || (e == 0u && (m | l) == 0u)) return x;
    uint s = h & 0x80000000u;
    if (e != 0u) return asdouble(l, s | m | 0x3FE00000u);
    uint top = m != 0u ? 32u + firstbithigh(m) : firstbithigh(l);
    uint64_t f = ((uint64_t)m << 32) | (uint64_t)l;
    uint64_t ff = (f << (52u - top)) & 0x000FFFFFFFFFFFFFull;
    return asdouble((uint)ff, s | 0x3FE00000u | (uint)(ff >> 32));
}
int frexp_exp_d(double x) {
    uint l, h; asuint(x, l, h);
    uint e = (h >> 20) & 0x7FFu;
    uint m = h & 0xFFFFFu;
    if (e == 0x7FFu || (e == 0u && (m | l) == 0u)) return 0;
    if (e != 0u) return (int)e - 1022;
    uint top = m != 0u ? 32u + firstbithigh(m) : firstbithigh(l);
    return (int)top - 1073;
}
// f64 classes from the bits. DXC narrows isinf and isnan of a double to f32, so 1e300 tested as inf.
bool isnan_d(double x) { uint l, h; asuint(x, l, h); h &= 0x7FFFFFFFu; return h > 0x7FF00000u || (h == 0x7FF00000u && l != 0u); }
bool isinf_d(double x) { uint l, h; asuint(x, l, h); return (h & 0x7FFFFFFFu) == 0x7FF00000u && l == 0u; }
// f64 rounding on the bits. An int64_t cast breaks at |x| >= 2^63, on inf and NaN, and on the
// sign of zero. |x| >= 2^52, inf and NaN are already integral, and |x| < 1 gives a zero of x's sign.
double trunc_d(double x) {
    uint l, h; asuint(x, l, h);
    uint e = (h >> 20) & 0x7FFu;
    if (e >= 1075u) return x;
    if (e < 1023u) return asdouble(0u, h & 0x80000000u);
    uint64_t bits = (((uint64_t)h << 32) | (uint64_t)l) & ~((1ull << (1075u - e)) - 1ull);
    return asdouble((uint)bits, (uint)(bits >> 32));
}
// t ± 1 only happens where |x| < 2^52, so it is exact. floor(−0) = −0 and ceil(−0.5) = −0.
double floor_d(double x) { double t = trunc_d(x); return t > x ? t - 1.0 : t; }
// no f64 negation here: -floor_d(-x) gave -0.0 for 1.5625 on an AMD driver
double ceil_d(double x) { double t = trunc_d(x); return t < x ? t + 1.0 : t; }
// x · 2^n for any n, rounded once. 2^n is built from bits in at most three steps whose
// intermediate results stay normal, as in musl's scalbn. exp2() works in f32 and ends at 2^±127.
// The steps are `precise` because DXC otherwise folds them into one product that over- or underflows.
double ldexp_d(double x, int n) {
    if (isnan_d(x) || isinf_d(x) || x == 0.0) return x;
    precise double r = x;
    n = clamp(n, -2200, 2200);
    if (n > 1023) {
        r *= asdouble(0u, 0x7FE00000u); n -= 1023;
        if (n > 1023) { r *= asdouble(0u, 0x7FE00000u); n = min(n - 1023, 1023); }
    } else if (n < -1022) {
        // 2^-1022 · 2^53, so only the last step can round into a denormal
        r *= asdouble(0u, 0x03600000u); n += 969;
        if (n < -1022) { r *= asdouble(0u, 0x03600000u); n = max(n + 969, -1022); }
    }
    r *= asdouble(0u, (uint)(n + 1023) << 20);
    return r;
}
// HLSL has no f64 sqrt, its sqrt works in f32. x is scaled by 2^-2k into [0.5, 2), its f32 root
// is refined by two Newton steps in f64, the second with an exact fma residual, and the result
// is scaled back by 2^k. Denormals and values outside the f32 range work too. The result is
// within 1 ulp of the correctly rounded root, and exact on 98.6 % of 64 000 random inputs.
// ±0, +inf and NaN return x, and x < 0 gives NaN.
double sqrt_d(double x) {
    if (isnan_d(x) || x == 0.0 || (isinf_d(x) && x > 0.0)) return x;
    if (x < 0.0) return asdouble(0u, 0x7FF80000u);
    int k = frexp_exp_d(x) >> 1;
    precise double s = ldexp_d(x, -2 * k);
    precise double y = (double)sqrt((float)s);
    y = y + (s - y * y) / (y + y);
    y = y + fma(-y, y, s) / (y + y);
    return ldexp_d(y, k);
}
// ±inf gives NaN, and NaN gives x
double fract_d(double x) {
    if (isnan_d(x)) return x;
    if (isinf_d(x)) return asdouble(0u, 0x7FF80000u);
    // a zero result is +0, as x - floor(x) gives in IEEE. With the inline constant 0, an AMD
    // driver ran 0.0 - floor_d(0.0) as a negation and gave -0.
    double f = x - floor_d(x);
    return f == 0.0 ? asdouble(0u, 0u) : min(f, asdouble(0xFFFFFFFFu, 0x3FEFFFFFu));
}
double round_ne_d(double x) {
    uint l, h; asuint(x, l, h);
    uint sign = h & 0x80000000u;
    uint e = (h >> 20) & 0x7FFu;
    uint64_t mag = ((uint64_t)(h & 0x7FFFFFFFu) << 32) | (uint64_t)l;
    if (e == 0x7FFu || e >= 1075u) return x;
    if (e < 1022u) return asdouble(0u, sign);
    if (e == 1022u) {
        bool above_half = (mag & 0x000FFFFFFFFFFFFFull) != 0ull;
        return above_half ? asdouble(0u, sign | 0x3FF00000u) : asdouble(0u, sign);
    }
    uint shift = 1075u - e;
    uint64_t mask = (1ull << shift) - 1ull;
    uint64_t half = 1ull << (shift - 1u);
    uint64_t fraction = mag & mask;
    uint64_t whole = mag & ~mask;
    if (fraction > half || (fraction == half && ((whole >> shift) & 1ull) != 0ull)) whole += 1ull << shift;
    return asdouble((uint)whole, sign | (uint)(whole >> 32));
}
uint dot4_u8(uint a, uint b, uint c) { uint r = c; [unroll] for (uint i = 0u; i < 32u; i += 8u) r += ((a >> i) & 0xFFu) * ((b >> i) & 0xFFu); return r; }
uint dot4_i8(uint a, uint b, uint c) { int r = (int)c; [unroll] for (uint i = 0u; i < 32u; i += 8u) r += (((int)(a << (24u - i))) >> 24) * (((int)(b << (24u - i))) >> 24); return (uint)r; }
uint dot8_u4(uint a, uint b, uint c) { uint r = c; [unroll] for (uint i = 0u; i < 32u; i += 4u) r += ((a >> i) & 0xFu) * ((b >> i) & 0xFu); return r; }
uint dot8_i4(uint a, uint b, uint c) { int r = (int)c; [unroll] for (uint i = 0u; i < 32u; i += 4u) r += (((int)(a << (28u - i))) >> 28) * (((int)(b << (28u - i))) >> 28); return (uint)r; }
uint fclass16(uint h) {
    uint e = (h >> 10) & 0x1Fu; uint m = h & 0x3FFu; bool n = (h & 0x8000u) != 0u;
    if (e == 0x1Fu) return m != 0u ? (((m & 0x200u) != 0u) ? 1u : 0u) : (n ? 2u : 9u);
    if (e == 0u) return m == 0u ? (n ? 5u : 6u) : (n ? 4u : 7u);
    return n ? 3u : 8u;
}
uint fclass(float f) {
    uint u = asuint(f); uint e = (u >> 23) & 0xFFu; uint m = u & 0x7FFFFFu; bool n = (u >> 31) != 0u;
    if (e == 0xFFu) return m != 0u ? (((m & 0x400000u) != 0u) ? 1u : 0u) : (n ? 2u : 9u);
    if (e == 0u) return m == 0u ? (n ? 5u : 6u) : (n ? 4u : 7u);
    return n ? 3u : 8u;
}
uint fclass64(double f) {
    uint l, h; asuint(f, l, h);
    uint e = (h >> 20) & 0x7FFu; uint m = (h & 0xFFFFFu) | l; bool n = (h >> 31) != 0u;
    if (e == 0x7FFu) return m != 0u ? (((h & 0x80000u) != 0u) ? 1u : 0u) : (n ? 2u : 9u);
    if (e == 0u) return m == 0u ? (n ? 5u : 6u) : (n ? 4u : 7u);
    return n ? 3u : 8u;
}
"#;
