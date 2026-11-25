//! Split wave64 (docs/wave64-on-wave32.md): which masks need the exchange.
//!
//! In the split form every mask write (`v_cmp*`, carries) exchanges the two halves' ballots so
//! that both halves hold the whole 64-bit mask. A mask that is only ever read lane by lane –
//! `v_cndmask_b32`'s selector, the carry-in of `v_add_co_ci` / `v_sub(rev)_co_ci` – needs just
//! the reading lane's own bit, which its own half has. This is a backward liveness analysis
//! over the SGPRs: a register is *whole-live* after an instruction when some path may read
//! its whole value (any SALU / SMEM / VALU scalar operand, a branch on VCCZ, …) before it is
//! written again. A mask write whose destination is not whole-live needs no exchange.
//! Pixel shaders use the same analysis: such a mask needs only the reading lane's own bit, not
//! the quad reads that give every helper lane's bit (`ballot_px_local` / `ballot_px`, emit.rs).
//!
//! After a mask stays in its half the two halves hold different values in its registers, so a
//! whole read that is missed here – or a write taken for a full overwrite that is not one –
//! lets them diverge. The rules follow what the emitter does, not only the ISA tables:
//! - a write counts as an overwrite only when the emitter replaces the register whatever its
//!   old value: not `s_cmov*` (conditional), `s_addk` / `s_mulk` / `s_bitset*` (read the old
//!   value), `s_movreld` / `s_movrelsd` (write `s[dst + m0]`), nor SDWA compares that write
//!   their SDST (SD = 1) instead of VCC;
//! - `s_movrels*` read `s[src + m0]`: every register from `src` up;
//! - SDWA scalar sources (S0 / S1 bits) and SMEM register offsets are reads.
//!
//! Conservative where it cannot see: an indirect jump makes every SGPR whole-live, EXEC is
//! always whole-live, and so is every SGPR at the end of a kernel-mode program (the runner
//! reads them back). Every emitted loop may also be left where its iteration budget
//! (`LOOP_GUARD`) is checked: what is live after a loop, or at the end, is live before each of
//! its instructions.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::emit::ext_encoding;
use crate::gpu::rdna2::{self, Decoded, Variant, F_CONDITIONAL, F_INDIRECT, F_TERMINATOR};
use crate::gpu::rdna2_spec::ENCODINGS;

/// SGPR set: bit n = s<n> (106/107 VCC, 124 M0, 126/127 EXEC).
pub type Regs = u128;

const VCC: Regs = 3 << 106;
const M0: Regs = 1 << 124;
const EXEC: Regs = 3 << 126;
const ALL: Regs = !0;

#[inline]
fn range(first: u64, n: u64) -> Regs {
    if first >= 128 || n == 0 {
        return 0;
    }
    let count = n.min(128 - first);
    let mask = if count >= 128 { !0u128 } else { (1u128 << count) - 1 };
    mask << first
}

/// SGPRs a scalar source code names (`n` dwords from it): s0–s105, VCC, M0, EXEC; src_vccz /
/// src_execz are derived from the whole VCC / EXEC; NULL, constants and literals name none.
#[inline]
fn scalar_code(v: u64, n: u64) -> Regs {
    match v {
        251 => VCC,
        252 => EXEC,
        125 => 0,
        v if v <= 127 => range(v, n),
        _ => 0,
    }
}

/// Instructions whose mask operand is read lane by lane (the emitter's `lane_of`).
fn reads_mask_per_lane(name: &str) -> bool {
    matches!(name, "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32")
}

/// Instructions that keep (part of) their destination's old value: the destination is read
/// and not overwritten.
fn keeps_destination(name: &str) -> bool {
    name.starts_with("S_CMOV") || matches!(name, "S_ADDK_I32" | "S_MULK_I32" | "S_BITSET0_B32" | "S_BITSET1_B32" | "S_BITSET0_B64" | "S_BITSET1_B64")
}

#[inline]
fn extract_ranges(words: &[u32], ranges: &[(u8, u8)]) -> u64 {
    let mut v = 0u64;
    let mut shift = 0u32;
    for &(off, cnt) in ranges {
        if cnt == 0 {
            continue;
        }
        let w = (off >> 5) as usize;
        let b = (off & 31) as u32;
        let cnt = cnt as u32;
        let mask = if cnt >= 64 { !0u64 } else { (1u64 << cnt) - 1 };
        let lo = words.get(w).copied().unwrap_or(0) as u64;
        let bits = if b + cnt <= 32 {
            (lo >> b) & mask
        } else {
            let hi = words.get(w + 1).copied().unwrap_or(0) as u64;
            ((lo | (hi << 32)) >> b) & mask
        };
        v |= bits << shift;
        shift += cnt;
    }
    v
}

const MAX_DYN_OPS: usize = 6;

#[derive(Clone, Copy)]
struct DynOp {
    ranges: &'static [(u8, u8)],
    scale: u8,
    n: u8,
    is_write: bool,
    is_src0: bool,
}

impl DynOp {
    const EMPTY: Self = Self {
        ranges: &[],
        scale: 1,
        n: 1,
        is_write: false,
        is_src0: false,
    };
}

#[derive(Clone, Copy)]
struct UsePlan {
    ienc_ptr: usize,
    enc_ptr: usize,
    is_movrel: bool,
    movrel_ssrc0: &'static [(u8, u8)],
    const_whole: Regs,
    const_written: Regs,
    ops: [DynOp; MAX_DYN_OPS],
    op_len: u8,
    overflow: bool,
}

impl UsePlan {
    const EMPTY: Self = Self {
        ienc_ptr: 0,
        enc_ptr: 0,
        is_movrel: false,
        movrel_ssrc0: &[],
        const_whole: 0,
        const_written: 0,
        ops: [DynOp::EMPTY; MAX_DYN_OPS],
        op_len: 0,
        overflow: false,
    };
}

struct UsePlanTable {
    entries: Box<[UsePlan]>,
    mask: usize,
}

impl UsePlanTable {
    fn build() -> Self {
        let total: usize = rdna2::all_instructions().map(|i| i.encs.len()).sum();
        let cap = (total * 2).next_power_of_two().max(64);
        let mask = cap - 1;
        let mut entries = vec![UsePlan::EMPTY; cap].into_boxed_slice();
        for inst in rdna2::all_instructions() {
            let name = inst.name;
            let is_movrel = name.starts_with("S_MOVREL");
            let per_lane = reads_mask_per_lane(name);
            let keeps = keeps_destination(name);
            for ienc in inst.encs {
                let enc = &ENCODINGS[ienc.encoding as usize];
                let ienc_ptr = ienc as *const rdna2::InstEnc as usize;
                let enc_ptr = enc as *const rdna2::Encoding as usize;
                let mut plan = UsePlan {
                    ienc_ptr,
                    enc_ptr,
                    is_movrel,
                    movrel_ssrc0: if is_movrel {
                        enc.field("SSRC0").map_or(&[][..], |f| f.ranges)
                    } else {
                        &[]
                    },
                    const_whole: 0,
                    const_written: 0,
                    ops: [DynOp::EMPTY; MAX_DYN_OPS],
                    op_len: 0,
                    overflow: false,
                };
                if !is_movrel {
                    for o in ienc.operands {
                        let n = (o.size as u64).div_ceil(32).max(1);
                        let fixed = match (o.field, o.ty) {
                            (_, "OPR_VCC") => Some(VCC),
                            (_, "OPR_EXEC" | "OPR_SDST_EXEC") => Some(EXEC),
                            (_, "OPR_SDST_M0") => Some(M0),
                            (None, _) | (Some("LITERAL"), _) => continue,
                            (Some(f), ty) => {
                                if o.implicit
                                    || !["SREG", "SSRC", "SDST", "OPR_SRC", "SMEM_OFFSET"]
                                        .iter()
                                        .any(|k| ty.contains(k))
                                {
                                    continue;
                                }
                                if !o.output && per_lane && f == "SRC2" {
                                    continue;
                                }
                                let idx = plan.op_len as usize;
                                if idx < MAX_DYN_OPS && n <= u8::MAX as u64 {
                                    plan.ops[idx] = DynOp {
                                        ranges: enc.field(f).map_or(&[][..], |fld| fld.ranges),
                                        scale: rdna2::register_scale(f) as u8,
                                        n: n as u8,
                                        is_write: o.output && !keeps,
                                        is_src0: f == "SRC0",
                                    };
                                    plan.op_len += 1;
                                } else {
                                    plan.overflow = true;
                                }
                                None
                            }
                        };
                        if let Some(regs) = fixed {
                            if o.output {
                                if keeps {
                                    plan.const_whole |= regs;
                                } else {
                                    plan.const_written |= regs;
                                }
                            } else if !(per_lane && (o.ty == "OPR_VCC" || o.field == Some("SRC2"))) {
                                plan.const_whole |= regs;
                            }
                        }
                    }
                }
                let mut slot = hash_ptr(ienc_ptr) & mask;
                while entries[slot].ienc_ptr != 0 {
                    slot = (slot + 1) & mask;
                }
                entries[slot] = plan;
            }
        }
        Self { entries, mask }
    }

    #[inline]
    fn get(&self, ienc_ptr: usize, enc_ptr: usize) -> Option<&UsePlan> {
        let mut slot = hash_ptr(ienc_ptr) & self.mask;
        loop {
            let e = &self.entries[slot];
            if e.ienc_ptr == ienc_ptr {
                return (e.enc_ptr == enc_ptr && !e.overflow).then_some(e);
            }
            if e.ienc_ptr == 0 {
                return None;
            }
            slot = (slot + 1) & self.mask;
        }
    }
}

#[inline]
fn hash_ptr(p: usize) -> usize {
    let x = (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (x ^ (x >> 32)) as usize
}

fn use_plans() -> &'static UsePlanTable {
    static TABLE: OnceLock<UsePlanTable> = OnceLock::new();
    TABLE.get_or_init(UsePlanTable::build)
}

fn uses_fallback(d: &Decoded) -> (Regs, Regs) {
    let name = d.inst.name;
    if name.starts_with("S_MOVREL") {
        let src = rdna2::field_value(d.enc, &d.words, "SSRC0").unwrap_or(0).min(124);
        return (range(src, 124 - src) | M0 | EXEC, 0);
    }
    let (mut whole, mut written) = (0 as Regs, 0 as Regs);
    let per_lane = reads_mask_per_lane(name);
    let keeps = keeps_destination(name);
    let mut sdwa_dst = None;
    if d.variant == Variant::Sdwa {
        let Some(e) = ext_encoding(d) else { return (ALL, 0) };
        let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
        if fv("S0") == 1 {
            whole |= scalar_code(fv("VSRC0"), 1);
        }
        if d.enc.field("VSRC1").is_some() && fv("S1") == 1 {
            whole |= scalar_code(fv("VSRC1"), 1);
        }
        if d.enc.name == "ENC_VOPC" && !name.starts_with("V_CMPX") {
            sdwa_dst = Some(if fv("SD") == 1 { scalar_code(fv("SDST"), 2) } else { VCC });
        }
    }
    for o in d.ienc.operands {
        let n = (o.size as u64).div_ceil(32).max(1);
        let regs = match (o.field, o.ty) {
            (_, "OPR_VCC") => VCC,
            (_, "OPR_EXEC" | "OPR_SDST_EXEC") => EXEC,
            (_, "OPR_SDST_M0") => M0,
            (None, _) | (Some("LITERAL"), _) => continue,
            (Some(f), ty) => {
                if o.implicit || !["SREG", "SSRC", "SDST", "OPR_SRC", "SMEM_OFFSET"].iter().any(|k| ty.contains(k)) {
                    continue;
                }
                if f == "SRC0" && d.variant != Variant::Plain {
                    continue;
                }
                scalar_code(rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f), n)
            }
        };
        if o.output {
            if keeps {
                whole |= regs;
            } else {
                written |= regs;
            }
        } else if !(per_lane && (o.ty == "OPR_VCC" || o.field == Some("SRC2"))) {
            whole |= regs;
        }
    }
    if let Some(dst) = sdwa_dst {
        written = (written & !VCC) | dst;
    }
    (whole, written)
}

#[inline]
fn uses_with_table(d: &Decoded, table: &UsePlanTable) -> (Regs, Regs) {
    let ienc_ptr = d.ienc as *const rdna2::InstEnc as usize;
    let enc_ptr = d.enc as *const rdna2::Encoding as usize;
    let Some(plan) = table.get(ienc_ptr, enc_ptr) else {
        return uses_fallback(d);
    };
    if plan.is_movrel {
        // `smovrel`: s[src + m0] is read, s[dst + m0] written (s_movrelsd_2: both) – any
        // register from the encoded one up to s123, VCC included; nothing for certain
        // overwritten
        let src = extract_ranges(&d.words, plan.movrel_ssrc0).min(124);
        return (range(src, 124 - src) | M0 | EXEC, 0);
    }
    let mut whole = plan.const_whole;
    let mut written = plan.const_written;
    let mut sdwa_dst = None;
    if d.variant == Variant::Sdwa {
        let Some(e) = ext_encoding(d) else { return (ALL, 0) };
        let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
        if fv("S0") == 1 {
            whole |= scalar_code(fv("VSRC0"), 1);
        }
        if d.enc.field("VSRC1").is_some() && fv("S1") == 1 {
            whole |= scalar_code(fv("VSRC1"), 1);
        }
        if d.enc.name == "ENC_VOPC" && !d.inst.name.starts_with("V_CMPX") {
            sdwa_dst = Some(if fv("SD") == 1 { scalar_code(fv("SDST"), 2) } else { VCC });
        }
    }
    let plain = d.variant == Variant::Plain;
    for op in &plan.ops[..plan.op_len as usize] {
        // SDWA / DPP src0 is in the extension dword (above for SDWA; a VGPR for DPP)
        if op.is_src0 && !plain {
            continue;
        }
        let code = extract_ranges(&d.words, op.ranges) * op.scale as u64;
        let regs = scalar_code(code, op.n as u64);
        if op.is_write {
            written |= regs;
        } else {
            whole |= regs;
        }
    }
    if let Some(dst) = sdwa_dst {
        written = (written & !VCC) | dst;
    }
    (whole, written)
}

/// (read whole, overwritten) SGPRs of one instruction. Lane-by-lane mask reads are left out
/// of the first, writes that keep anything of the old value out of the second.
pub fn uses(d: &Decoded) -> (Regs, Regs) {
    uses_with_table(d, use_plans())
}

#[derive(Clone, Copy)]
enum BlockSucc {
    Indirect,
    Empty,
    One(u32),
    Two(u32, u32),
}

#[derive(Clone, Copy)]
struct Block {
    gen: Regs,
    kill: Regs,
    succ: BlockSucc,
    exit_start: u32,
    exit_len: u32,
    has_exit: bool,
}

fn whole_live_out_unsorted(insts: &[Decoded], end_reads_all: bool, table: &UsePlanTable) -> Vec<Regs> {
    let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    let at_end = if end_reads_all { ALL } else { EXEC };
    let succ: Vec<Option<Vec<usize>>> = insts
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let next = insts.get(i + 1).map(|n| n.pc).and_then(|pc| pcs.get(&pc).copied());
            if let Some(t) = d.far_target {
                return Some(vec![*pcs.get(&t)?]);
            }
            if d.inst.flags & F_INDIRECT != 0 || d.inst.name == "S_SETPC_B64" || d.inst.name == "S_SWAPPC_B64" {
                return None;
            }
            let mut out = Vec::new();
            if let Some(t) = rdna2::branch_target(d) {
                out.push(*pcs.get(&t)?);
                if d.inst.flags & F_CONDITIONAL != 0 {
                    out.extend(next);
                }
            } else if d.inst.flags & F_TERMINATOR == 0 {
                out.extend(next);
            }
            Some(out)
        })
        .collect();
    let loops = super::loop_ranges(insts);
    let exits: Vec<Vec<Option<usize>>> = insts
        .iter()
        .map(|d| loops.iter().filter(|&&(a, b)| a <= d.pc && d.pc < b).map(|&(_, b)| pcs.range(b..).next().map(|(_, &j)| j)).collect())
        .collect();
    let gen_kill: Vec<(Regs, Regs)> = insts.iter().map(|d| uses_with_table(d, table)).collect();
    let mut live_in = vec![0 as Regs; insts.len()];
    let mut live_out = vec![0 as Regs; insts.len()];
    loop {
        let mut changed = false;
        for i in (0..insts.len()).rev() {
            let out = match &succ[i] {
                None => ALL,
                Some(s) if s.is_empty() => at_end,
                Some(s) => s.iter().fold(EXEC, |m, &j| m | live_in[j]),
            };
            let left = exits[i].iter().fold(0, |m, e| m | at_end | e.map_or(0, |j| live_in[j]));
            let (gen, kill) = gen_kill[i];
            let inn = gen | (out & !kill) | left | EXEC;
            if out != live_out[i] || inn != live_in[i] {
                live_out[i] = out;
                live_in[i] = inn;
                changed = true;
            }
        }
        if !changed {
            return live_out;
        }
    }
}

/// Merge partly overlapping backward-jump loop ranges `(a, b)` until all ranges nest,
/// sorted and deduplicated — exact equivalent of `super::loop_ranges`, in O(L log L) for
/// properly nested loops (the common case).
fn nested_loop_ranges(mut raw: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    if raw.len() <= 1 {
        return raw;
    }
    let mut sorted = raw.clone();
    sorted.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| y.1.cmp(&x.1)));
    sorted.dedup();
    // Fast check: verify in O(L) whether all intervals already nest properly.
    let mut stack: Vec<u64> = Vec::new();
    let mut has_partial_overlap = false;
    for &(a, b) in &sorted {
        while let Some(&top_end) = stack.last() {
            if top_end <= a {
                stack.pop();
            } else {
                break;
            }
        }
        if let Some(&top_end) = stack.last() {
            if top_end < b {
                has_partial_overlap = true;
                break;
            }
        }
        stack.push(b);
    }
    if !has_partial_overlap {
        sorted.sort_unstable();
        sorted.dedup();
        return sorted;
    }
    // Rare irreducible CFG with partial overlaps: preserve exact merge order of `loop_ranges`.
    loop {
        let mut merged = false;
        'scan: for i in 0..raw.len() {
            for j in 0..raw.len() {
                let (a, b) = (raw[i], raw[j]);
                if i != j && a.0 < b.0 && b.0 < a.1 && a.1 < b.1 {
                    raw[i] = (a.0, b.1);
                    raw.remove(j);
                    merged = true;
                    break 'scan;
                }
            }
        }
        if !merged {
            break;
        }
    }
    raw.sort_unstable();
    raw.dedup();
    raw
}

/// For each instruction (index in `insts`): the SGPRs whose whole value may be read after it.
/// `end_reads_all`: the program's end reads every SGPR (kernel mode).
pub fn whole_live_out(insts: &[Decoded], end_reads_all: bool) -> Vec<Regs> {
    let n = insts.len();
    if n == 0 {
        return Vec::new();
    }
    let table = use_plans();
    if !insts.windows(2).all(|w| w[0].pc < w[1].pc) {
        return whole_live_out_unsorted(insts, end_reads_all, table);
    }

    let at_end = if end_reads_all { ALL } else { EXEC };
    let lookup_pc = |pc: u64| insts.binary_search_by_key(&pc, |d| d.pc).ok();
    let first_ge = |pc: u64| {
        let idx = insts.partition_point(|d| d.pc < pc);
        (idx < n).then_some(idx)
    };

    // Single pass over `insts`: collect backward-jump loop ranges and basic-block leaders.
    let mut is_leader = vec![false; n];
    is_leader[0] = true;
    let mut raw_loops: Vec<(u64, u64)> = Vec::new();
    for (i, d) in insts.iter().enumerate() {
        let flags = d.inst.flags;
        let has_far = d.far_target.is_some();
        let indirect = !has_far
            && (flags & F_INDIRECT != 0 || d.inst.name == "S_SETPC_B64" || d.inst.name == "S_SWAPPC_B64");
        let branch = if has_far {
            d.far_target
        } else if !indirect && flags & rdna2::F_BRANCH != 0 {
            rdna2::branch_target(d)
        } else {
            None
        };
        if let Some(t) = branch {
            if t <= d.pc {
                raw_loops.push((t, d.pc + d.len as u64));
            }
            if let Some(j) = lookup_pc(t) {
                is_leader[j] = true;
            }
            if i + 1 < n {
                is_leader[i + 1] = true;
            }
        } else if indirect || flags & F_TERMINATOR != 0 {
            if i + 1 < n {
                is_leader[i + 1] = true;
            }
        }
    }

    // `LOOP_GUARD`: a loop may be left where its guard runs – at its top, before the head
    // instruction, and in the `pc` dispatcher form before every block of it – to the code
    // after it (structured) or the end of the program (dispatcher: no later block matches
    // the `pc` it leaves with). Taken as: before any instruction of a loop, whatever is live
    // after the loop or at the end may be read.
    let loops = nested_loop_ranges(raw_loops);
    for &(a, b) in &loops {
        if let Some(j) = first_ge(a) {
            is_leader[j] = true;
        }
        if let Some(j) = first_ge(b) {
            is_leader[j] = true;
        }
    }

    let mut block_starts: Vec<usize> = Vec::new();
    let mut inst_to_block = vec![0u32; n];
    for i in 0..n {
        if is_leader[i] {
            block_starts.push(i);
        }
        inst_to_block[i] = (block_starts.len() - 1) as u32;
    }
    let num_blocks = block_starts.len();

    let gen_kill: Vec<(Regs, Regs)> = insts.iter().map(|d| uses_with_table(d, table)).collect();

    // Summarize each basic block: because `(X & !k) | X == X` for bitmasks, the constant
    // per-block `left | EXEC` term collapses out of the intra-block composition into:
    //   block_in[b] = block_gen[b] | (block_out[b] & !block_kill[b]) | block_left[b] | EXEC.
    let mut blocks: Vec<Block> = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = block_starts[b];
        let end = if b + 1 < num_blocks { block_starts[b + 1] } else { n };
        let (mut bgen, mut bkill) = (0 as Regs, 0 as Regs);
        for &(g, k) in gen_kill[start..end].iter().rev() {
            bgen = g | (bgen & !k);
            bkill |= k;
        }
        let d = &insts[end - 1];
        let next_block = (end < n).then(|| inst_to_block[end]);
        let succ = if let Some(t) = d.far_target {
            match lookup_pc(t) {
                Some(j) => BlockSucc::One(inst_to_block[j]),
                None => BlockSucc::Indirect,
            }
        } else if d.inst.flags & F_INDIRECT != 0 || d.inst.name == "S_SETPC_B64" || d.inst.name == "S_SWAPPC_B64" {
            BlockSucc::Indirect
        } else if let Some(t) = rdna2::branch_target(d) {
            match lookup_pc(t) {
                None => BlockSucc::Indirect,
                Some(j) => {
                    let tb = inst_to_block[j];
                    if d.inst.flags & F_CONDITIONAL != 0 {
                        match next_block {
                            Some(nb) if nb != tb => BlockSucc::Two(tb, nb),
                            _ => BlockSucc::One(tb),
                        }
                    } else {
                        BlockSucc::One(tb)
                    }
                }
            }
        } else if d.inst.flags & F_TERMINATOR == 0 {
            match next_block {
                Some(nb) => BlockSucc::One(nb),
                None => BlockSucc::Empty,
            }
        } else {
            BlockSucc::Empty
        };
        blocks.push(Block {
            gen: bgen,
            kill: bkill,
            succ,
            exit_start: 0,
            exit_len: 0,
            has_exit: false,
        });
    }

    // Sweep sorted loops once across basic blocks to collect each block's loop-exit targets.
    let mut exit_targets: Vec<u32> = Vec::new();
    if !loops.is_empty() {
        let mut active: Vec<(u64, Option<u32>)> = Vec::new();
        let mut li = 0usize;
        for b in 0..num_blocks {
            let pc = insts[block_starts[b]].pc;
            while li < loops.len() && loops[li].0 <= pc {
                let (_, b_end) = loops[li];
                if pc < b_end {
                    let exit_blk = first_ge(b_end).map(|j| inst_to_block[j]);
                    active.push((b_end, exit_blk));
                }
                li += 1;
            }
            active.retain(|&(b_end, _)| pc < b_end);
            if !active.is_empty() {
                let ex_start = exit_targets.len() as u32;
                for &(_, eb) in &active {
                    if let Some(eb) = eb {
                        if !exit_targets[ex_start as usize..].contains(&eb) {
                            exit_targets.push(eb);
                        }
                    }
                }
                blocks[b].has_exit = true;
                blocks[b].exit_start = ex_start;
                blocks[b].exit_len = (exit_targets.len() as u32) - ex_start;
            }
        }
    }

    // Solve backward liveness at basic-block granularity.
    let mut block_in = vec![0 as Regs; num_blocks];
    let mut block_out = vec![0 as Regs; num_blocks];
    loop {
        let mut changed = false;
        for b in (0..num_blocks).rev() {
            let blk = &blocks[b];
            let out = match blk.succ {
                BlockSucc::Indirect => ALL,
                BlockSucc::Empty => at_end,
                BlockSucc::One(s0) => EXEC | block_in[s0 as usize],
                BlockSucc::Two(s0, s1) => EXEC | block_in[s0 as usize] | block_in[s1 as usize],
            };
            let left = if blk.has_exit {
                let es = blk.exit_start as usize;
                let mut m = at_end;
                for &eb in &exit_targets[es..es + blk.exit_len as usize] {
                    m |= block_in[eb as usize];
                }
                m
            } else {
                0
            };
            let inn = blk.gen | (out & !blk.kill) | left | EXEC;
            if out != block_out[b] || inn != block_in[b] {
                block_out[b] = out;
                block_in[b] = inn;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Reconstruct per-instruction live_out in a single linear backward pass per block.
    let mut live_out = vec![0 as Regs; n];
    for b in 0..num_blocks {
        let blk = &blocks[b];
        let start = block_starts[b];
        let end = if b + 1 < num_blocks { block_starts[b + 1] } else { n };
        let left = if blk.has_exit {
            let es = blk.exit_start as usize;
            let mut m = at_end;
            for &eb in &exit_targets[es..es + blk.exit_len as usize] {
                m |= block_in[eb as usize];
            }
            m
        } else {
            0
        };
        let mut out = block_out[b];
        for i in (start..end).rev() {
            live_out[i] = out;
            let (g, k) = gen_kill[i];
            out = g | (out & !k) | left | EXEC;
        }
    }
    live_out
}

/// pc → SGPRs whole-live after that instruction, for the emitter.
pub fn by_pc(insts: &[Decoded], end_reads_all: bool) -> BTreeMap<u64, Regs> {
    insts.iter().map(|d| d.pc).zip(whole_live_out(insts, end_reads_all)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(bytes: &[u8]) -> Vec<Decoded> {
        let mut out = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        out
    }

    fn mask_needed(live: &[Regs], i: usize, pair: u64) -> bool {
        live[i] & range(pair, 2) != 0
    }

    const V: u64 = 256;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)]);
    const ENDPGM: I = ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]);
    const CMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]);
    const CMP_LT: I = ("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]);
    const CNDMASK: I = ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]);
    const MOV_S2_VCC: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)]);

    fn asm(lines: &[I]) -> Vec<Decoded> {
        let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        program(&bytes)
    }

    #[test]
    fn a_selector_only_mask_needs_no_exchange() {
        let p = asm(&[CMP, CNDMASK, ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 106));
        // the kernel runner reads VCC at the end: still exchanged there
        assert!(mask_needed(&whole_live_out(&p, true), 0, 106));
    }

    #[test]
    fn a_carry_chain_needs_no_exchange() {
        let p = asm(&[
            ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 0), ("SDST", 106), ("SRC0", V), ("SRC1", V + 2)]),
            ("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 1), ("VSRC1", 3)]),
            ENDPGM,
        ]);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 106));
        assert!(!mask_needed(&live, 1, 106));
    }

    #[test]
    fn scalar_reads_and_branches_keep_the_exchange() {
        let readers: [I; 5] = [
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
            MOV_S2_VCC,
            ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 0)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 106)]),
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
        ];
        for reader in readers {
            let p = asm(&[CMP, reader, ENDPGM]);
            assert!(mask_needed(&whole_live_out(&p, false), 0, 106), "{}", reader.0);
        }
    }

    #[test]
    fn a_whole_read_on_one_path_or_around_a_loop_keeps_the_exchange() {
        // v_cmp; s_cbranch_scc0 → skip; v_cndmask; s_branch → end; skip: s_mov s[2:3], vcc; end
        let p = asm(&[
            CMP,
            ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)]),
            CNDMASK,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)]),
            MOV_S2_VCC,
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        // loop: the mask written at the bottom is read whole at the top of the next iteration
        let p = asm(&[MOV_S2_VCC, CMP, ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)]), ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 1, 106));
    }

    #[test]
    fn overwritten_before_any_whole_read() {
        let p = asm(&[CMP, CNDMASK, CMP_LT, MOV_S2_VCC, ENDPGM]);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 106));
        assert!(mask_needed(&live, 2, 106));
    }

    #[test]
    fn exec_is_always_exchanged() {
        let p = asm(&[("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]), ENDPGM]);
        assert!(whole_live_out(&p, false)[0] & EXEC == EXEC);
    }

    const CMP_S20: I = ("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", V + 1)]);
    const BCNT_S20: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 20)]);
    const BCNT_VCC: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]);

    // review findings: writes that are not full overwrites, reads the operand tables miss

    #[test]
    fn an_sdwa_compare_into_its_sdst_leaves_vcc() {
        let sdwa = |sd: u64| -> I {
            let f: &'static [(&'static str, u64)] = if sd == 1 {
                &[("SRC0", 0xF9), ("VSRC0", 0), ("VSRC1", 0), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("SD", 1), ("SDST", 22)]
            } else {
                &[("SRC0", 0xF9), ("VSRC0", 0), ("VSRC1", 0), ("SRC0_SEL", 6), ("SRC1_SEL", 6)]
            };
            ("V_CMP_EQ_U32", "VOPC_VOP_SDWA_SDST_ENC", f)
        };
        let p = asm(&[CMP, sdwa(1), BCNT_VCC, ENDPGM]);
        assert_eq!(uses(&p[1]).1, range(22, 2), "SD = 1 writes s[22:23], not VCC");
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        // SD = 0: VCC is written, the first compare's mask is dead
        let p = asm(&[CMP, sdwa(0), BCNT_VCC, ENDPGM]);
        assert_eq!(uses(&p[1]).1, VCC);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 106) && mask_needed(&live, 1, 106));
    }

    #[test]
    fn read_modify_write_destinations_are_reads() {
        let cases: [&[I]; 4] = [
            &[("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 191)])],
            &[("S_BITSET0_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 129)]), ("S_BITSET0_B32", "ENC_SOP1", &[("SDST", 21), ("SSRC0", 129)])],
            &[("S_ADDK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 1)]), ("S_MULK_I32", "ENC_SOPK", &[("SDST", 21), ("SIMM16", 3)])],
            &[("S_CMOVK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 1)]), ("S_CMOV_B32", "ENC_SOP1", &[("SDST", 21), ("SSRC0", 129)])],
        ];
        for rmw in cases {
            let mut l = vec![CMP_S20];
            l.extend_from_slice(rmw);
            l.extend([BCNT_S20, ENDPGM]);
            assert!(mask_needed(&whole_live_out(&asm(&l), false), 0, 20), "{}", rmw[0].0);
        }
    }

    #[test]
    fn scalar_relative_moves_reach_any_register_from_their_base() {
        // s_movrels_b64 s[0:1], s[100 + m0 : 101 + m0] – VCC with m0 = 6
        let p = asm(&[CMP, ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 100)]), ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)]), ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)]), ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        // s_movreld_b64 s[20 + m0 : 21 + m0]: s[20:21] itself only when m0 = 0
        let p = asm(&[CMP_S20, ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 0)]), BCNT_S20, ENDPGM]);
        assert_eq!(uses(&p[1]).1, 0);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
    }

    #[test]
    fn smem_register_offsets_and_sdwa_scalar_sources_are_reads() {
        let readers: [I; 3] = [
            ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 0), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 20)]),
            ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 20), ("S1", 1), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6)]),
            ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 20), ("S0", 1), ("VSRC1", 2), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6)]),
        ];
        for r in readers {
            let p = asm(&[CMP_S20, r, ENDPGM]);
            assert!(uses(&p[1]).0 & range(20, 1) != 0, "{} {}", r.0, r.1);
            assert!(mask_needed(&whole_live_out(&p, false), 0, 20), "{} {}", r.0, r.1);
        }
        // a VGPR src1 is not
        let p = asm(&[CMP_S20, ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 20), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6)]), ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 20));
    }

    #[test]
    fn a_selector_that_is_also_a_scalar_source_is_read_whole() {
        // v_cndmask_b32 v2, vcc_lo, v3, vcc: the selector per lane, src0 the whole VCC_LO
        let p = asm(&[CMP, ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 106), ("SRC1", V + 3), ("SRC2", 106)]), ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        let p = asm(&[CMP, ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 3), ("SRC2", 106)]), ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn a_far_jump_is_a_branch_not_an_unknown_jump() {
        // v_cmp vcc; s_getpc s[30:31]; s_add_u32 s30, s30, 20; s_addc_u32 s31, s31, 0;
        // s_setpc s[30:31] → +28 (past the s_mov that reads VCC whole); +28: v_cndmask; end
        let mut bytes = Vec::new();
        for (n, e, f, l) in [
            (CMP.0, CMP.1, CMP.2, None),
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 30)][..], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 30), ("SSRC0", 30), ("SSRC1", 255)][..], Some(20)),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 31), ("SSRC0", 31), ("SSRC1", 128)][..], None),
            ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 30)][..], None),
            (MOV_S2_VCC.0, MOV_S2_VCC.1, MOV_S2_VCC.2, None),
            (CNDMASK.0, CNDMASK.1, CNDMASK.2, None),
            (ENDPGM.0, ENDPGM.1, ENDPGM.2, None),
        ] {
            bytes.extend(rdna2::encode(n, e, f, l).unwrap());
        }
        let mut p = program(&bytes);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106), "unresolved: an indirect jump");
        rdna2::resolve_far_jumps(&mut p);
        assert_eq!(p[4].far_target, Some(p[6].pc));
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn a_loop_may_be_left_at_its_top() {
        // loop: s_mov_b64 vcc, s[4:5]; v_cmp vcc; v_cndmask; s_branch loop – then s_bcnt1 vcc:
        // only the emitter's iteration budget leaves the loop, at its top, with the v_cmp's VCC
        let p = asm(&[
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 4)]),
            CMP,
            CNDMASK,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
            BCNT_VCC,
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, false), 1, 106));
    }

    #[test]
    fn a_loop_left_before_its_head_overwrites() {
        // v_cmp vcc; v_cndmask; loop: s_mov_b64 vcc, s[4:5]; v_cndmask; s_branch loop – then
        // s_bcnt1 vcc: a budget already spent leaves the loop before its s_mov, with the
        // first v_cmp's VCC
        let p = asm(&[CMP, CNDMASK, ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 4)]), CNDMASK, ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFD)]), BCNT_VCC, ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn a_dispatcher_loop_left_by_its_budget_ends_the_program() {
        // loop: v_cmp vcc; v_cndmask; s_mov_b64 vcc, s[4:5]; s_branch loop – then s_mov_b64
        // vcc, 0: structured, the loop's exit overwrites VCC; in the `pc` dispatcher form the
        // budget's break ends the program, whose end the kernel runner reads
        let p = asm(&[
            CMP,
            CNDMASK,
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 4)]),
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)]),
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, true), 0, 106));
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn s_movrelsd_2_reads_from_its_base_and_overwrites_nothing() {
        let mov = ("S_MOVRELSD_2_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 100)][..]);
        let p = asm(&[CMP_S20, mov, BCNT_S20, ENDPGM]);
        assert_eq!(uses(&p[1]).1, 0);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
        // its source s[100 + (m0 & 0x3FF)] reaches VCC
        let p = asm(&[CMP, mov, ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)]), ENDPGM]);
        assert!(uses(&p[1]).0 & VCC == VCC);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    /// Reference instruction-level solver matching the original implementation verbatim.
    fn whole_live_out_reference(insts: &[Decoded], end_reads_all: bool) -> Vec<Regs> {
        let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
        let at_end = if end_reads_all { ALL } else { EXEC };
        let succ: Vec<Option<Vec<usize>>> = insts
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let next = insts.get(i + 1).map(|n| n.pc).and_then(|pc| pcs.get(&pc).copied());
                if let Some(t) = d.far_target {
                    return Some(vec![*pcs.get(&t)?]);
                }
                if d.inst.flags & F_INDIRECT != 0 || d.inst.name == "S_SETPC_B64" || d.inst.name == "S_SWAPPC_B64" {
                    return None;
                }
                let mut out = Vec::new();
                if let Some(t) = rdna2::branch_target(d) {
                    out.push(*pcs.get(&t)?);
                    if d.inst.flags & F_CONDITIONAL != 0 {
                        out.extend(next);
                    }
                } else if d.inst.flags & F_TERMINATOR == 0 {
                    out.extend(next);
                }
                Some(out)
            })
            .collect();
        let loops = super::super::loop_ranges(insts);
        let exits: Vec<Vec<Option<usize>>> = insts
            .iter()
            .map(|d| loops.iter().filter(|&&(a, b)| a <= d.pc && d.pc < b).map(|&(_, b)| pcs.range(b..).next().map(|(_, &j)| j)).collect())
            .collect();
        let gen_kill: Vec<(Regs, Regs)> = insts.iter().map(uses_fallback).collect();
        let mut live_in = vec![0 as Regs; insts.len()];
        let mut live_out = vec![0 as Regs; insts.len()];
        loop {
            let mut changed = false;
            for i in (0..insts.len()).rev() {
                let out = match &succ[i] {
                    None => ALL,
                    Some(s) if s.is_empty() => at_end,
                    Some(s) => s.iter().fold(EXEC, |m, &j| m | live_in[j]),
                };
                let left = exits[i].iter().fold(0, |m, e| m | at_end | e.map_or(0, |j| live_in[j]));
                let (gen, kill) = gen_kill[i];
                let inn = gen | (out & !kill) | left | EXEC;
                if out != live_out[i] || inn != live_in[i] {
                    live_out[i] = out;
                    live_in[i] = inn;
                    changed = true;
                }
            }
            if !changed {
                return live_out;
            }
        }
    }

    /// Build a realistic synthetic shader with `outer_loops` nested loops, diamond branches,
    /// scalar/vector ALU chains, compares, cndmasks, and carry chains (~`outer_loops * 120` insts).
    fn synthetic_shader(outer_loops: usize) -> Vec<Decoded> {
        let mut bytes: Vec<u8> = Vec::with_capacity(outer_loops * 600 + 32);
        let push = |bytes: &mut Vec<u8>, name: &str, enc: &str, fields: &[(&str, u64)]| {
            bytes.extend(rdna2::encode(name, enc, fields, None).unwrap());
        };
        for k in 0..outer_loops {
            let outer_start = bytes.len();
            push(&mut bytes, "S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)]);
            push(&mut bytes, "S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 130)]);
            for inner in 0..2 {
                let inner_start = bytes.len();
                for step in 0..20 {
                    if step % 5 == 0 {
                        push(&mut bytes, CMP.0, CMP.1, CMP.2);
                        push(&mut bytes, CNDMASK.0, CNDMASK.1, CNDMASK.2);
                    } else if step % 7 == 0 {
                        push(&mut bytes, "V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 1), ("VSRC1", 2)]);
                        push(&mut bytes, "S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)]);
                    } else {
                        push(&mut bytes, "V_MUL_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 2), ("VSRC1", 3)]);
                        push(&mut bytes, "V_ADD_F32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 4)]);
                    }
                }
                // Diamond branch over 12 4-byte instructions
                push(&mut bytes, "S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]);
                push(&mut bytes, "S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 12)]);
                for _ in 0..6 {
                    push(&mut bytes, CMP.0, CMP.1, CMP.2);
                    push(&mut bytes, CNDMASK.0, CNDMASK.1, CNDMASK.2);
                }
                if inner == 1 && k % 3 == 0 {
                    push(&mut bytes, MOV_S2_VCC.0, MOV_S2_VCC.1, MOV_S2_VCC.2);
                }
                // Back-edge to inner_start (offset in dwords from PC + 4)
                let back_dw = -(((bytes.len() - inner_start) / 4 + 1) as i16) as u16 as u64;
                push(&mut bytes, "S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", back_dw)]);
            }
            // Tail of outer loop with carry chain + conditional whole-VCC read
            push(&mut bytes, "V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 0), ("SDST", 106), ("SRC0", V), ("SRC1", V + 2)]);
            push(&mut bytes, "V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 1), ("VSRC1", 3)]);
            if k % 2 == 0 {
                push(&mut bytes, BCNT_VCC.0, BCNT_VCC.1, BCNT_VCC.2);
            }
            let outer_back_dw = -(((bytes.len() - outer_start) / 4 + 1) as i16) as u16 as u64;
            push(&mut bytes, "S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", outer_back_dw)]);
        }
        push(&mut bytes, ENDPGM.0, ENDPGM.1, ENDPGM.2);
        program(&bytes)
    }

    #[test]
    fn use_plan_matches_fallback_across_all_rdna2_instructions() {
        let mut seed = 0x517c_c1b7_2722_0a95u64;
        let mut next_u32 = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };
        for inst in rdna2::all_instructions() {
            for ienc in inst.encs {
                let enc = &ENCODINGS[ienc.encoding as usize];
                let nwords = (enc.bits as usize).div_ceil(32).max(1);
                for variant in [Variant::Plain, Variant::Dpp16, Variant::Dpp8] {
                    let words: Vec<u32> = (0..nwords).map(|_| next_u32()).collect();
                    let d = Decoded {
                        pc: 0,
                        inst,
                        ienc,
                        enc,
                        words,
                        len: nwords * 4,
                        literal: None,
                        variant,
                        far_target: None,
                    };
                    assert_eq!(uses(&d), uses_fallback(&d), "{} ({}, {variant:?})", inst.name, enc.name);
                }
            }
        }
    }

    #[test]
    fn nested_loop_ranges_matches_quadratic_merge_on_overlapping_intervals() {
        fn merge_ref(mut loops: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
            loop {
                let mut merged = false;
                'scan: for i in 0..loops.len() {
                    for j in 0..loops.len() {
                        let (a, b) = (loops[i], loops[j]);
                        if i != j && a.0 < b.0 && b.0 < a.1 && a.1 < b.1 {
                            loops[i] = (a.0, b.1);
                            loops.remove(j);
                            merged = true;
                            break 'scan;
                        }
                    }
                }
                if !merged {
                    break;
                }
            }
            loops.sort();
            loops.dedup();
            loops
        }
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % 40
        };
        for _ in 0..200 {
            let len = (next() % 18) as usize;
            let mut ivs = Vec::with_capacity(len);
            for _ in 0..len {
                let a = next() * 4;
                let b = a + (next() % 12 + 1) * 4;
                ivs.push((a, b));
            }
            assert_eq!(nested_loop_ranges(ivs.clone()), merge_ref(ivs));
        }
    }

    #[test]
    fn block_liveness_matches_reference_bit_for_bit() {
        for outer in [0, 1, 4, 16] {
            let p = synthetic_shader(outer);
            for end_reads_all in [false, true] {
                let got = whole_live_out(&p, end_reads_all);
                let want = whole_live_out_reference(&p, end_reads_all);
                assert_eq!(got, want, "outer={outer}, end_reads_all={end_reads_all}");
            }
        }
        // Empty input
        assert!(whole_live_out(&[], false).is_empty());
        assert!(whole_live_out(&[], true).is_empty());
    }

    #[test]
    #[ignore = "manual release benchmark: cargo test --release --lib gpu::recomp::x64_masks::tests::bench_whole_live_out -- --ignored --nocapture"]
    fn bench_whole_live_out() {
        use std::hint::black_box;
        use std::time::Instant;
        // Warm up OnceLock table
        let _ = use_plans();
        for &(outer, iters) in &[(8usize, 50usize), (40, 20), (80, 10)] {
            let p = synthetic_shader(outer);
            assert_eq!(whole_live_out(&p, false), whole_live_out_reference(&p, false));
            let t0 = Instant::now();
            for _ in 0..iters {
                black_box(whole_live_out_reference(black_box(&p), false));
            }
            let ref_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let t1 = Instant::now();
            for _ in 0..iters {
                black_box(whole_live_out(black_box(&p), false));
            }
            let opt_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "x64_masks bench (insts={}, outer_loops={outer}): ref={ref_us:.1} us, opt={opt_us:.1} us, speedup={:.2}x",
                p.len(),
                ref_us / opt_us
            );
        }
    }
}
