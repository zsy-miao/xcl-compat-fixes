//! Whether a wave64 VS / PS gives the console's result when a GPU runs each 32-lane half of its
//! console wave as a wave of its own (#336). Classification only: nothing translates by it yet.
//!
//! `upper_lanes` answers what a lane ≥ 32 holds when it is read. This answers whether the
//! program's result depends on how its wave is cut: a value that depends on the cut must not
//! reach what the program leaves behind (an export, a store, a branch, EXEC).
//! - A wave-wide vote (`s_cbranch_vcc*`, `exec*`, `scc*` on a mask) is taken per half: the half
//!   with no lane for a side takes the other one, which the console wave did not run for those
//!   lanes. That is exact when the vote only skips lane-masked work: the side taken with the
//!   mask empty is the join (the branch's post-dominator), and for a VCC vote the other side
//!   starts by narrowing EXEC to VCC – or the compare is of values every lane has (each half
//!   then votes as the wave does).
//! - Lane counts, ranks and first-lane indices of a mask (`s_bcnt1`, `v_mbcnt`, `s_ff1`), a mask
//!   used as a number, and another lane's varying value (`v_readfirstlane`, `v_readlane`) are
//!   different in a half than in the wave; so is everything they reach.
//!
//! A forward data-flow pass over the program's blocks (`cfg`) carries those as taint bits per
//! register; values every lane has are followed too, since reading one from another lane is
//! exact.

use std::collections::BTreeSet;

use super::cfg::{Cfg, PostDom};
use crate::gpu::rdna2::{self, Decoded};

const LANE_COUNT: u8 = 1;
const RANK: u8 = 2;
const FIRST_LANE: u8 = 4;
const LANE_ID: u8 = 8;
const MASK_DATA: u8 = 16;
const CROSS_LANE: u8 = 32;
const TAINTED_EXEC: u8 = 64;
const STALE_SCALAR: u8 = 128;

const VCC: usize = 106;
const EXEC: usize = 126;

/// How a wave64 VS / PS fares on 32-lane waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// what it leaves behind does not depend on how the wave is cut
    Exact,
    /// it may (the reasons say how; the analysis is conservative: unknown counts as this)
    MayDiffer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    pub tier: Tier,
    pub reasons: BTreeSet<&'static str>,
}

/// What SCC holds, for a branch on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SccVote {
    /// not a vote: a scalar compare / result
    No,
    /// whether the EXEC just written is not empty (`s_and_saveexec_b64`, `s_and_b64 exec, …`)
    Exec,
    /// whether some other mask is not empty (the register it was written to, if any)
    Mask(Option<usize>),
}

#[derive(Clone, PartialEq, Eq)]
struct State {
    /// taint bits per SGPR (s0..s105, VCC, M0, EXEC: codes 0..127)
    s: Vec<u8>,
    scc: u8,
    scc_vote: SccVote,
    /// taint bits per VGPR
    v: Vec<u8>,
    /// the VGPR holds one value in every lane
    uniform: Vec<bool>,
    /// the SGPR holds a word of a lane mask
    mask: Vec<bool>,
    /// … from a compare of values every lane has, under the launch EXEC (each half votes as
    /// the wave does)
    uniform_mask: Vec<bool>,
    /// EXEC is what the program started with (or its whole-quad form)
    exec_launch: bool,
    /// the SGPR holds a word of that launch EXEC (saved by `s_mov_b64` / saveexec)
    launch: Vec<bool>,
    /// the SGPR holds a word of a set of lanes: a mask, or all / none (0 / -1, a scalar condition)
    lanes: Vec<bool>,
}

impl State {
    fn entry() -> State {
        let mut mask = vec![false; 128];
        mask[EXEC] = true;
        mask[EXEC + 1] = true;
        State {
            s: vec![0; 128],
            scc: 0,
            scc_vote: SccVote::No,
            v: vec![0; 256],
            uniform: vec![false; 256],
            uniform_mask: vec![false; 128],
            exec_launch: true,
            launch: vec![false; 128],
            lanes: mask.clone(),
            mask,
        }
    }

    fn join(&self, o: &State) -> State {
        let or = |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(x, y)| x | y).collect();
        let and = |a: &[bool], b: &[bool]| a.iter().zip(b).map(|(x, y)| *x && *y).collect();
        let any = |a: &[bool], b: &[bool]| a.iter().zip(b).map(|(x, y)| *x || *y).collect();
        State {
            s: or(&self.s, &o.s),
            scc: self.scc | o.scc,
            scc_vote: if self.scc_vote == o.scc_vote { self.scc_vote } else { SccVote::Mask(None) },
            v: or(&self.v, &o.v),
            uniform: and(&self.uniform, &o.uniform),
            mask: any(&self.mask, &o.mask),
            uniform_mask: and(&self.uniform_mask, &o.uniform_mask),
            exec_launch: self.exec_launch && o.exec_launch,
            launch: and(&self.launch, &o.launch),
            lanes: and(&self.lanes, &o.lanes),
        }
    }

    fn exec_taint(&self) -> u8 {
        self.s[EXEC] | self.s[EXEC + 1]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reg {
    S(usize),
    V(usize),
    Scc,
}

/// What an instruction reads and writes.
struct Uses {
    /// registers whose whole value is read (`x64_masks::uses` for the SGPRs: destinations a write
    /// keeps part of, SDWA scalar sources, SMEM offsets, `src_vccz` / `src_execz`; SCC, also as
    /// `src_scc`; the VGPRs, NSA addresses included)
    reads: Vec<Reg>,
    /// the SGPRs of those read as an explicit operand (not an implicit EXEC / VCC / M0)
    explicit: Vec<usize>,
    /// a mask read lane by lane as a selector or carry (`v_cndmask`, `v_add_co_ci`, …)
    lane_mask: Vec<usize>,
    /// registers written (a write that keeps part of the old value reads it too); image loads
    /// by their DMASK
    writes: Vec<Reg>,
}

fn uses(d: &Decoded) -> Uses {
    let n = d.inst.name;
    let (whole, _) = super::x64_masks::uses(d);
    let (explicit_r, explicit_w) = rdna2::sgpr_uses(d);
    let mut implicit_in: u128 = 0;
    let mut reads: Vec<Reg> = Vec::new();
    let mut writes: Vec<Reg> = explicit_w.iter().map(|&s| Reg::S(s as usize)).collect();
    for o in d.ienc.operands {
        let regs: &[Reg] = match o.ty {
            "OPR_VCC" => &[Reg::S(VCC), Reg::S(VCC + 1)],
            "OPR_SDST_EXEC" | "OPR_EXEC" => &[Reg::S(EXEC), Reg::S(EXEC + 1)],
            "OPR_SSRC_SPECIAL_SCC" => &[Reg::Scc],
            "OPR_SDST_M0" => &[Reg::S(124)],
            _ => continue,
        };
        if o.output {
            for &g in regs {
                if !writes.contains(&g) {
                    writes.push(g);
                }
            }
        } else if o.field.is_none() {
            for &g in regs {
                match g {
                    Reg::S(s) => implicit_in |= 1 << s,
                    Reg::Scc => reads.push(Reg::Scc),
                    Reg::V(_) => {}
                }
            }
        }
    }
    let lane_mask: Vec<usize> = if matches!(n, "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32") {
        match rdna2::field_value(d.enc, &d.words, "SRC2") {
            Some(s) if s < 126 => vec![s as usize, s as usize + 1],
            _ => vec![VCC, VCC + 1],
        }
    } else {
        Vec::new()
    };
    for s in 0..128usize {
        if whole >> s & 1 == 1 {
            reads.push(Reg::S(s));
        }
    }
    // src_scc as an operand
    for f in ["SRC0", "SRC1", "SRC2", "SSRC0", "SSRC1"] {
        if rdna2::field_value(d.enc, &d.words, f) == Some(253) && d.enc.field(f).is_some() && !reads.contains(&Reg::Scc) {
            reads.push(Reg::Scc);
        }
    }
    let explicit: Vec<usize> = explicit_r.iter().map(|&s| s as usize).chain((0..128usize).filter(|&s| whole >> s & 1 == 1 && implicit_in >> s & 1 == 0)).collect();
    // VGPRs: an image load by its DMASK, NSA addresses
    let (vr, _) = rdna2::vgpr_uses(d);
    reads.extend(vr.iter().map(|&v| Reg::V(v as usize)));
    if d.enc.name.starts_with("ENC_MIMG") {
        let nsa = rdna2::field_value(d.enc, &d.words, "NSA").unwrap_or(0);
        if let Some(e) = (nsa > 0).then(|| rdna2::encoding(&format!("MIMG_NSA{nsa}"))).flatten() {
            for f in e.fields.iter().filter(|f| f.name.starts_with("VADDR") && f.name != "VADDR") {
                if let Some(v) = rdna2::field_value(e, &d.words, f.name) {
                    reads.push(Reg::V(v as usize));
                }
            }
        }
    }
    writes.extend(super::upper_lanes::vgpr_writes(d).iter().map(|&v| Reg::V(v as usize)));
    Uses { reads, explicit, lane_mask, writes }
}

fn is_valu(d: &Decoded) -> bool {
    d.enc.name.starts_with("ENC_VOP") || d.enc.name.starts_with("VOP")
}

fn is_vcmp(d: &Decoded) -> bool {
    d.inst.name.starts_with("V_CMP")
}

/// Reads a lane mask as a mask (bit by bit, or as a whole): not as a number.
fn mask_op(n: &str) -> bool {
    mask_result(n)
        || n.starts_with("S_CMP_") && n.ends_with("_U64")
        || n.starts_with("S_CBRANCH")
        || n.starts_with("V_CNDMASK")
        || n.contains("_CO_CI_")
        || matches!(n, "S_BCNT0_I32_B64" | "S_BCNT1_I32_B64" | "S_FF1_I32_B64" | "S_FF0_I32_B64" | "S_FLBIT_I32_B64")
}

/// Makes a lane mask (or a word of one) of lane masks (bitwise, moves, selects, saveexec): a
/// number of one is not.
fn mask_result(n: &str) -> bool {
    let bitwise = ["S_AND_", "S_OR_", "S_XOR_", "S_ANDN1_", "S_ANDN2_", "S_ORN1_", "S_ORN2_", "S_NAND_", "S_NOR_", "S_XNOR_"];
    let word = n.ends_with("_B64") || n.ends_with("_B32");
    (word && (bitwise.iter().any(|p| n.starts_with(p)) || n.starts_with("S_NOT_") || n.starts_with("S_MOV_B") || n.starts_with("S_CSELECT_B") || n.starts_with("S_CMOV_B") || n == "S_WQM_B64"))
        || n.contains("SAVEEXEC")
}

/// The partition-dependent value an instruction makes of its own (not from its sources).
fn source_taint(d: &Decoded, st: &State, u: &Uses) -> u8 {
    let n = d.inst.name;
    let reads = &u.reads;
    let mut t = 0;
    // explicit operands only: a vector or memory instruction's implicit EXEC is its lane mask
    let reads_mask = u.explicit.iter().any(|&s| st.mask[s]);
    // a vote's outcome used as a number (s_cselect, s_addc, … on SCC; not the branch on it)
    if reads.contains(&Reg::Scc) && st.scc_vote != SccVote::No && !n.starts_with("S_CBRANCH") {
        t |= MASK_DATA;
    }
    if matches!(n, "S_BCNT0_I32_B64" | "S_BCNT1_I32_B64") || (reads_mask && matches!(n, "S_BCNT0_I32_B32" | "S_BCNT1_I32_B32")) {
        t |= LANE_COUNT;
    }
    if matches!(n, "S_FF1_I32_B64" | "S_FF0_I32_B64" | "S_FLBIT_I32_B64" | "S_FLBIT_I32_I64") || (reads_mask && matches!(n, "S_FF1_I32_B32" | "S_FLBIT_I32_B32")) {
        t |= FIRST_LANE;
    }
    if n.starts_with("V_MBCNT") {
        // a constant SRC0 (-1): the lane's index in the wave; a mask: its rank in it
        let constant = matches!(rdna2::field_value(d.enc, &d.words, "SRC0"), Some(128..=208 | 240..=248 | 255));
        t |= if constant { LANE_ID } else { RANK };
    }
    // DPP16: a row or bank mask names rows of the wave (a 32-lane wave has rows 0 and 1 only);
    // row_bcast15 / row_bcast31 read across rows
    if d.variant == rdna2::Variant::Dpp16 {
        if let Some(e) = super::emit::ext_encoding(d) {
            let f = |name: &str| rdna2::field_value(e, &d.words, name);
            if f("ROW_MASK").is_some_and(|m| m != 0xF) || f("BANK_MASK").is_some_and(|m| m != 0xF) {
                t |= LANE_ID;
            }
            if matches!(f("DPP_CTRL"), Some(0x142 | 0x143)) {
                t |= CROSS_LANE;
            }
        }
    }
    // one lane written: lane k of each half is a different console lane
    if n == "V_WRITELANE_B32" {
        t |= LANE_ID;
    }
    // another lane's value: exact only for a value every lane has, read from the first active
    // lane (a given lane may be one the half did not launch)
    if (n == "V_READFIRSTLANE_B32" && reads.iter().any(|r| matches!(r, Reg::V(v) if !st.uniform[*v]))) || n == "V_READLANE_B32" {
        t |= CROSS_LANE;
    }
    if reads_mask && !mask_op(n) && !is_vcmp(d) {
        t |= MASK_DATA;
    }
    // a number combined with a mask (`s_and_b64 exec, exec, 1`): lane k of each half is a
    // different console lane. Masks, and 0 / -1 (a scalar condition as a mask), are not numbers
    let bitwise = ["S_AND_", "S_OR_", "S_XOR_", "S_ANDN1_", "S_ANDN2_", "S_ORN1_", "S_ORN2_", "S_NAND_", "S_NOR_", "S_XNOR_"].iter().any(|p| n.starts_with(p));
    if bitwise && (reads_mask || n.contains("SAVEEXEC")) {
        let wide = usize::from(n.ends_with("_B64"));
        let set = |a: Option<u64>| match a {
            None | Some(128 | 193) => true,
            Some(s @ 0..=127) => (s as usize..=s as usize + wide).all(|x| x < 128 && st.lanes[x]),
            _ => false,
        };
        let f = |name: &str| rdna2::field_value(d.enc, &d.words, name);
        if !set(f("SSRC0")) || !set(f("SSRC1")) {
            t |= MASK_DATA;
        }
    }
    t
}

fn reasons_of(t: u8, out: &mut BTreeSet<&'static str>) {
    for (bit, name) in [
        (LANE_COUNT, "lane-count"),
        (RANK, "rank"),
        (FIRST_LANE, "first-lane"),
        (LANE_ID, "lane-id"),
        (MASK_DATA, "mask-data"),
        (CROSS_LANE, "cross-lane"),
        (TAINTED_EXEC, "tainted-exec"),
        (STALE_SCALAR, "stale-scalar"),
    ] {
        if t & bit != 0 {
            out.insert(name);
        }
    }
}

/// What the program leaves behind: exports, memory writes (LDS / GDS ones included).
fn is_store(n: &str) -> bool {
    let ds_read = n.starts_with("DS_READ") || matches!(n, "DS_SWIZZLE_B32" | "DS_PERMUTE_B32" | "DS_BPERMUTE_B32" | "DS_NOP");
    n == "EXP" || n.contains("STORE") || n.contains("ATOMIC") || (n.starts_with("DS_") && !ds_read) || n.starts_with("S_SENDMSG")
}

/// One instruction: the state after it; what it leaves behind that depends on the cut goes to
/// `out`.
fn step(d: &Decoded, st: &mut State, out: &mut BTreeSet<&'static str>) {
    let n = d.inst.name;
    let u = uses(d);
    let (reads, writes) = (&u.reads, &u.writes);
    let mut t = source_taint(d, st, &u);
    for r in reads {
        t |= match *r {
            Reg::S(s) => st.s[s],
            Reg::V(v) => st.v[v],
            Reg::Scc => st.scc,
        };
    }
    let valu = is_valu(d) || !d.enc.name.starts_with("ENC_S");
    if valu && st.exec_taint() != 0 {
        t |= TAINTED_EXEC | st.exec_taint();
    }
    if is_store(n) && t != 0 {
        reasons_of(t, out);
    }
    if n.starts_with("S_CBRANCH") && t != 0 {
        reasons_of(t, out);
    }
    let writes_exec = writes.contains(&Reg::S(EXEC)) || writes.contains(&Reg::S(EXEC + 1));
    if writes_exec && t != 0 {
        reasons_of(t, out);
    }
    // one word of EXEC written on its own: a 32-lane half's lanes are all under its low word
    if writes.contains(&Reg::S(EXEC)) != writes.contains(&Reg::S(EXEC + 1)) {
        out.insert("mask-data");
    }
    // a number written to EXEC (not all lanes, not none): lane k of each half is a different
    // console lane
    if writes_exec && matches!(n, "S_MOV_B64" | "S_MOV_B32") && !matches!(rdna2::field_value(d.enc, &d.words, "SSRC0"), Some(128 | 193) | Some(0..=127)) {
        out.insert("mask-data");
    }
    let reads_mask_or_exec = reads.iter().any(|r| matches!(r, Reg::S(s) if st.mask[*s])) || u.lane_mask.iter().any(|&s| st.mask[s]);
    // a per-lane selector counts as a source: it is every lane's only if all or none of them
    let all_uniform = reads.iter().all(|r| !matches!(r, Reg::V(v) if !st.uniform[*v])) && u.lane_mask.iter().all(|&s| st.uniform_mask[s]);
    // a mask that is all or none of the active lanes (EXEC itself, a compare of values every lane
    // has, a scalar condition selected into a mask): a half with active lanes votes on it as the
    // wave does. A half with none may take the other side: it writes no lane there, and its
    // scalars are followed as stale where the sides join. Kept by these ops, not by complements.
    let vcmp_uniform = is_vcmp(d) && all_uniform;
    let keeps_uniform = matches!(n, "S_AND_B64" | "S_OR_B64" | "S_ANDN2_B64" | "S_MOV_B64" | "S_CSELECT_B64" | "S_WQM_B64") || n.contains("SAVEEXEC");
    let sgpr_reads: Vec<usize> = reads.iter().filter_map(|r| if let Reg::S(s) = r { Some(*s) } else { None }).collect();
    let is_launch = |s: usize| if s == EXEC || s == EXEC + 1 { st.exec_launch } else { st.launch[s] };
    let mask_uniform = keeps_uniform && sgpr_reads.iter().all(|&s| !st.mask[s] || st.uniform_mask[s] || s == EXEC || s == EXEC + 1 || is_launch(s));
    let from_launch = matches!(n, "S_MOV_B64" | "S_WQM_B64") && !sgpr_reads.is_empty() && sgpr_reads.iter().all(|&s| is_launch(s));
    let or_launch = n == "S_OR_B64" && sgpr_reads.iter().any(|&s| is_launch(s));
    let saved_exec = n.contains("SAVEEXEC") && st.exec_launch;
    // a set of lanes made of sets of lanes and 0 / -1
    let lane_src = |a: Option<u64>| match a {
        None | Some(128 | 193) => true,
        Some(s @ 0..=127) => st.lanes[s as usize] && (s == 127 || st.lanes[s as usize + 1] || !n.ends_with("_B64")),
        _ => false,
    };
    let lanes_made = mask_result(n) && lane_src(rdna2::field_value(d.enc, &d.words, "SSRC0")) && lane_src(rdna2::field_value(d.enc, &d.words, "SSRC1"));
    let mut new_scc = None;
    for w in writes {
        match *w {
            Reg::S(s) => {
                st.s[s] = t;
                // (a carry-out is a mask too)
                st.mask[s] = s == EXEC || s == EXEC + 1 || is_vcmp(d) || (is_valu(d) && n.contains("_CO_")) || (mask_result(n) && reads_mask_or_exec);
                st.uniform_mask[s] = vcmp_uniform || (st.mask[s] && mask_uniform);
                st.lanes[s] = st.mask[s] || lanes_made;
                st.launch[s] = s != EXEC && s != EXEC + 1 && (from_launch || (saved_exec && Some(&Reg::S(s)) != writes.iter().find(|w| **w == Reg::S(EXEC))));
            }
            Reg::V(v) => {
                st.v[v] = t;
                // a value every lane has: written under the launch EXEC (a narrowed one leaves the
                // other lanes their old value) by a plain VALU op of such values (not DPP / SDWA,
                // which read other lanes or parts), or a load from an address every lane has
                let load = matches!(d.enc.name, "ENC_MUBUF" | "ENC_MTBUF" | "ENC_MIMG") && !n.contains("ATOMIC");
                let plain = d.enc.name.starts_with("ENC_VOP") && d.variant == rdna2::Variant::Plain && !n.starts_with("V_PERMLANE");
                st.uniform[v] = st.exec_launch && all_uniform && t & (RANK | LANE_ID | CROSS_LANE) == 0 && n != "V_WRITELANE_B32" && (plain || load);
            }
            Reg::Scc => {
                new_scc = Some(t);
            }
        }
    }
    // SCC no longer says whether the mask it was about is empty once that mask is rewritten
    if new_scc.is_none() {
        let stale_vote = match st.scc_vote {
            SccVote::Exec => writes_exec,
            SccVote::Mask(Some(m)) => writes.contains(&Reg::S(m)) || writes.contains(&Reg::S(m + 1)),
            _ => false,
        };
        if stale_vote {
            st.scc_vote = SccVote::Mask(None);
        }
    }
    if let Some(sc) = new_scc {
        st.scc = sc;
        st.scc_vote = if writes_exec {
            SccVote::Exec
        } else if reads_mask_or_exec && (mask_result(n) || n.starts_with("S_CMP")) {
            let to = writes.iter().find_map(|w| if let Reg::S(s) = w { Some(*s) } else { None });
            SccVote::Mask(if mask_result(n) { to } else { None })
        } else {
            SccVote::No
        };
    }
    if writes_exec {
        // EXEC back at the launch mask: its whole-quad form, a copy of it, or OR-ed with one
        st.exec_launch = from_launch || or_launch;
        // whole-quad mode turns on helper lanes, which got none of the values written before it
        if n == "S_WQM_B64" {
            st.uniform.iter_mut().for_each(|u| *u = false);
            st.uniform_mask.iter_mut().for_each(|u| *u = false);
        }
    }
}

/// The scalar registers an instruction writes, as a mask (bit s: register s).
fn scalar_writes(d: &Decoded) -> u128 {
    uses(d).writes.iter().fold(0, |m, w| match *w {
        Reg::S(s) if s < 128 => m | 1 << s,
        _ => m,
    })
}

/// Whether the 64-bit mask in registers `m`, `m + 1` is in `z`.
fn has_pair(z: u128, m: usize) -> bool {
    m < 127 && z & 3 << m == 3 << m
}

/// What a half whose lanes a vote's side does not run knows of its scalar registers (EXEC and
/// VCC included), as masks of registers: `z`, they hold zero there; `r`, they hold the EXEC it
/// had at the vote.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Half {
    z: u128,
    r: u128,
}

impl Half {
    fn meet(self, o: Half) -> Half {
        Half { z: self.z & o.z, r: self.r & o.r }
    }
}

/// Updates `h` for one instruction (`writes`: what it writes).
fn zero_step(d: &Decoded, writes: u128, h: &mut Half) {
    if writes == 0 {
        return;
    }
    let n = d.inst.name;
    let f = |name: &str| rdna2::field_value(d.enc, &d.words, name);
    // (the constant 0 is zero)
    let zero = |a: Option<u64>| a == Some(128) || a.is_some_and(|s| has_pair(h.z, s as usize));
    let vote = |a: Option<u64>| a.is_some_and(|s| has_pair(h.r, s as usize));
    let (ez, er) = (has_pair(h.z, EXEC), has_pair(h.r, EXEC));
    let (z0, z1) = (zero(f("SSRC0")), zero(f("SSRC1")));
    let (r0, r1) = (vote(f("SSRC0")), vote(f("SSRC1")));
    // x | y and x ^ y are the vote's EXEC when one is and the other is zero (x | x too)
    let or = |xr: bool, xz: bool, yr: bool, yz: bool| xr && (yr || yz) || yr && xz;
    let xor = |xr: bool, xz: bool, yr: bool, yz: bool| xr && yz || yr && xz;
    // (zero, the vote's EXEC) for the destination, and for a write of EXEC beside it, EXEC
    let (dst, exec) = match n {
        "S_MOV_B64" => ((z0, r0), None),
        "S_WQM_B64" => ((z0, false), None),
        "S_ANDN2_B64" => ((z0, r0 && z1), None),
        "S_AND_B64" => ((z0 || z1, r0 && r1), None),
        "S_OR_B64" => ((z0 && z1, or(r0, z0, r1, z1)), None),
        "S_XOR_B64" => ((z0 && z1, xor(r0, z0, r1, z1)), None),
        // saveexec: the destination is the old EXEC
        "S_AND_SAVEEXEC_B64" => ((ez, er), Some((z0 || ez, r0 && er))),
        "S_OR_SAVEEXEC_B64" => ((ez, er), Some((z0 && ez, or(r0, z0, er, ez)))),
        "S_XOR_SAVEEXEC_B64" => ((ez, er), Some((z0 && ez, xor(r0, z0, er, ez)))),
        // EXEC = s0 & ~EXEC (and the destination); EXEC = ~s0 & EXEC
        "S_ANDN2_SAVEEXEC_B64" => ((ez, er), Some((z0, r0 && ez))),
        "S_ANDN1_SAVEEXEC_B64" => ((ez, er), Some((ez, false))),
        "S_ANDN2_WREXEC_B64" => ((z0, r0 && ez), Some((z0, r0 && ez))),
        "S_ANDN1_WREXEC_B64" => ((ez, false), Some((ez, false))),
        // a compare writes 0 for the lanes not in EXEC
        _ if n.starts_with("V_CMPX") => ((false, false), Some((ez, false))),
        _ if n.starts_with("V_CMP") => ((ez, false), None),
        _ => ((false, false), None),
    };
    let exec_bits = 3u128 << EXEC;
    let (ex_z, ex_r) = exec.unwrap_or(dst);
    let set = |d: bool, e: bool| (if d { writes & !exec_bits } else { 0 }) | (if e { writes & exec_bits } else { 0 });
    h.z = h.z & !writes | set(dst.0, ex_z);
    h.r = h.r & !writes | set(dst.1, ex_r);
}

/// What a half that skips the blocks from `run` up to `stop` (exclusive; None: the program's end)
/// knows on entering each of them (None: not one of them), from `at_vote` on entering `run` from
/// the vote's block `from`, to a fixed point. (A wave that entered them stays in them until
/// `stop`: an edge from another block is not on its way.)
fn zero_entries(insts: &[Decoded], sw: &[u128], g: &Cfg, from: usize, run: usize, stop: Option<usize>, at_vote: Half) -> Vec<Option<Half>> {
    let nb = g.start.len();
    let mut inside = vec![false; nb];
    let mut todo = vec![run];
    while let Some(x) = todo.pop() {
        if Some(x) != stop && !inside[x] {
            inside[x] = true;
            todo.extend(g.succ[x].iter().copied());
        }
    }
    let entry = |x: usize, out: &[Option<Half>]| {
        let ins = g.pred[x].iter().filter_map(|&p| if x == run && p == from { Some(at_vote) } else { out[p] });
        ins.reduce(Half::meet)
    };
    // (only what holds on every way in; the sets only shrink)
    let mut out: Vec<Option<Half>> = vec![None; nb];
    loop {
        let mut changed = false;
        for x in (0..nb).filter(|&x| inside[x]) {
            let Some(mut h) = entry(x, &out) else { continue };
            for i in g.range(x) {
                zero_step(&insts[i], sw[i], &mut h);
            }
            if out[x] != Some(h) {
                out[x] = Some(h);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    (0..nb).map(|x| if inside[x] { entry(x, &out) } else { None }).collect()
}

/// A vote's side, from `run` up to `stop`, differs for a half that skips it: it writes EXEC a value
/// not zero in that half (a write on the way into the join with no lane work after it is the
/// join's restore), or reaches the join with an EXEC other than the vote's (zero too, if
/// `exec_zero`: the half had no lane at the vote). `at_vote`: what the half knows there.
#[allow(clippy::too_many_arguments)]
fn widens_after(insts: &[Decoded], sw: &[u128], g: &Cfg, from: usize, run: usize, stop: Option<usize>, at_vote: Half, exec_zero: bool) -> bool {
    for (x, h) in zero_entries(insts, sw, g, from, run, stop, at_vote).into_iter().enumerate() {
        let Some(mut h) = h else { continue };
        let range: Vec<usize> = g.range(x).collect();
        for (k, &i) in range.iter().enumerate() {
            zero_step(&insts[i], sw[i], &mut h);
            if sw[i] & 3 << EXEC == 0 || has_pair(h.z, EXEC) {
                continue;
            }
            let trailing = stop.is_some() && g.succ[x].iter().all(|&s| Some(s) == stop) && range[k + 1..].iter().all(|&j| insts[j].enc.name.starts_with("ENC_SOP"));
            if !trailing {
                return true;
            }
        }
        if stop.is_some_and(|s| g.succ[x].contains(&s)) && !(has_pair(h.r, EXEC) || exec_zero && has_pair(h.z, EXEC)) {
            return true;
        }
    }
    false
}

/// What a half whose mask `m` (a 64-bit register pair) is zero at the end of the vote's block `b`
/// knows there: `m` is zero, and what is written from it after its last write in the block; EXEC
/// is the vote's, and what is copied from it after its last write.
fn zero_at_vote(insts: &[Decoded], sw: &[u128], g: &Cfg, b: usize, m: usize) -> Half {
    let range = g.range(b);
    let branch = range.end - 1;
    let after_last = |reg: usize| (range.start..branch).rev().find(|&i| sw[i] & 3 << reg != 0).map_or(range.start, |i| i + 1);
    let mut z = Half { z: 3 << m, r: 0 };
    for i in after_last(m)..branch {
        zero_step(&insts[i], sw[i], &mut z);
    }
    let mut r = Half { z: 0, r: 3 << EXEC };
    for i in after_last(EXEC)..branch {
        zero_step(&insts[i], sw[i], &mut r);
    }
    Half { z: z.z, r: r.r }
}

/// A conditional branch's sides: (the side taken with the mask empty, the other side, the mask's
/// register; None for EXEC). None if it is not a vote on a known mask.
fn sides(g: &Cfg, b: usize, d: &Decoded, st: &State) -> Option<(usize, usize, Option<usize>)> {
    let n = d.inst.name;
    // a conditional branch's successors: its target, then the next block (one if they are the same)
    let [t, next] = g.succ[b][..] else { return None };
    let mask = match n {
        "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ" => None,
        "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" => Some(VCC),
        "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" => match st.scc_vote {
            SccVote::Exec => None,
            // SCC from some other mask: the register it was written to is the one to narrow EXEC by
            SccVote::Mask(Some(m)) => Some(m),
            _ => return None,
        },
        _ => return None,
    };
    let empty_is_target = matches!(n, "S_CBRANCH_EXECZ" | "S_CBRANCH_VCCZ" | "S_CBRANCH_SCC0");
    Some(if empty_is_target { (t, next, mask) } else { (next, t, mask) })
}

/// Block `b` ends the program and leaves nothing behind: at most a null export (a pixel shader's
/// way out when every pixel is discarded) – a half whose mask is empty that goes there writes
/// nothing, as its lanes write nothing on the console.
fn leaves_nothing(insts: &[Decoded], g: &Cfg, b: usize) -> bool {
    g.succ[b].is_empty()
        && g.range(b).all(|i| {
            let d = &insts[i];
            let n = d.inst.name;
            if n == "EXP" {
                return rdna2::field_value(d.enc, &d.words, "TGT") == Some(9);
            }
            !is_store(n) && (d.enc.name.starts_with("ENC_SOP") || n == "S_ENDPGM")
        })
}

/// Block `b`, a vote's side, turns EXEC off in a half whose mask is empty before it does any lane
/// work: from what that half knows at the vote (`at_vote`), EXEC is zero in it once the block's
/// leading scalar instructions are done (`s_and_saveexec_b64 s, m`, `s_wqm_b64 exec, m`, or a
/// compiler's `s_wqm_b64 s, m; s_and_b64 exec, exec, s`). Before that only scalar ALU ops, scalar
/// loads and waits: their writes are the wave's, followed as stale where the sides join.
fn narrows_before_lane_work(insts: &[Decoded], sw: &[u128], g: &Cfg, b: usize, at_vote: Half) -> bool {
    let mut h = at_vote;
    for i in g.range(b) {
        if has_pair(h.z, EXEC) {
            return true;
        }
        let d = &insts[i];
        let n = d.inst.name;
        let scalar = match d.enc.name {
            "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" => !n.starts_with("S_SETREG") && !n.starts_with("S_GETREG") && !n.contains("PC_B64"),
            "ENC_SMEM" => n.contains("LOAD"),
            "ENC_SOPP" => n.starts_with("S_WAITCNT") || n == "S_NOP",
            _ => false,
        };
        if !scalar {
            return false;
        }
        zero_step(d, sw[i], &mut h);
    }
    has_pair(h.z, EXEC)
}

/// Classify a wave64 VS / PS (instructions in address order, far jumps resolved).
pub fn classify(insts: &[Decoded]) -> Partition {
    let mut reasons = BTreeSet::new();
    let Some(g) = Cfg::new(insts) else {
        reasons.insert("no-graph");
        return Partition { tier: Tier::MayDiffer, reasons };
    };
    let nb = g.start.len();
    if nb == 0 {
        return Partition { tier: Tier::Exact, reasons };
    }
    let ipdom = g.post_dominators();
    let sw: Vec<u128> = insts.iter().map(scalar_writes).collect();
    // first without stale scalars, to find the votes; then with the scalars each vote's
    // dependent blocks write marked stale where its sides join
    let none = vec![Vec::new(); nb];
    let at = fixed_point(insts, &g, &none);
    let stale = stale_at_joins(insts, &sw, &g, &ipdom, &at);
    let at = fixed_point(insts, &g, &stale);
    // the sinks, and the votes, from the fixed point
    for b in 0..nb {
        let Some(mut st) = at[b].clone() else { continue };
        mark_stale(&mut st, &stale[b]);
        let range = g.range(b);
        let last = range.end - 1;
        for i in range {
            if i == last {
                vote(insts, &sw, &g, &ipdom, b, &insts[i], &st, &mut reasons);
            }
            step(&insts[i], &mut st, &mut reasons);
        }
    }
    let tier = if reasons.is_empty() { Tier::Exact } else { Tier::MayDiffer };
    Partition { tier, reasons }
}

/// Marks scalar registers (and SCC) stale.
fn mark_stale(st: &mut State, regs: &[Reg]) {
    for r in regs {
        match *r {
            Reg::S(s) => st.s[s] |= STALE_SCALAR,
            Reg::Scc => st.scc |= STALE_SCALAR,
            Reg::V(_) => {}
        }
    }
}

/// Block-entry states to a fixed point (the bits only grow, the flags only fall), with the
/// registers of `stale[b]` marked stale on entering block `b`.
fn fixed_point(insts: &[Decoded], g: &Cfg, stale: &[Vec<Reg>]) -> Vec<Option<State>> {
    let nb = g.start.len();
    let mut at: Vec<Option<State>> = vec![None; nb];
    at[0] = Some(State::entry());
    let mut work = BTreeSet::from([0usize]);
    let mut scratch = BTreeSet::new();
    while let Some(b) = work.pop_first() {
        let mut st = at[b].clone().expect("queued with a state");
        mark_stale(&mut st, &stale[b]);
        for i in g.range(b) {
            step(&insts[i], &mut st, &mut scratch);
        }
        for &s in &g.succ[b] {
            let new = match &at[s] {
                None => st.clone(),
                Some(old) => old.join(&st),
            };
            if at[s].as_ref() != Some(&new) {
                at[s] = Some(new);
                work.insert(s);
            }
        }
    }
    at
}

/// Whether the conditional branch `d` (state `st` before it) is a vote a half can take
/// differently from the wave: on a mask, unless every half has lanes and votes as the wave does
/// (the launch EXEC, or a mask that is all or none of its lanes).
fn is_vote(d: &Decoded, st: &State) -> bool {
    let mask = match d.inst.name {
        "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ" => EXEC,
        "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" => VCC,
        "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" => match st.scc_vote {
            SccVote::No => return false,
            SccVote::Exec => EXEC,
            SccVote::Mask(Some(m)) => m,
            SccVote::Mask(None) => return true,
        },
        _ => return false,
    };
    let uniform = mask == EXEC || (mask + 1 < 128 && st.uniform_mask[mask] && st.uniform_mask[mask + 1]);
    !(st.exec_launch && uniform)
}

/// For each block, the scalars (SGPRs, VCC, M0, SCC; not EXEC, which the join restores) that
/// the blocks a vote decides write, when the vote's sides join there: a half that took the
/// other side has not written them, so what it reads there is not the wave's.
fn stale_at_joins(insts: &[Decoded], sw: &[u128], g: &Cfg, ipdom: &[PostDom], at: &[Option<State>]) -> Vec<Vec<Reg>> {
    let nb = g.start.len();
    let cd = g.control_dependence(ipdom);
    // branch block -> the blocks that depend on it
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (x, on) in cd.iter().enumerate() {
        for &c in on {
            dependents[c].push(x);
        }
    }
    let mut stale: Vec<Vec<Reg>> = vec![Vec::new(); nb];
    for b in 0..nb {
        let (Some(st), PostDom::Block(join)) = (&at[b], ipdom[b]) else { continue };
        let mut st = st.clone();
        let range = g.range(b);
        let last = range.end - 1;
        let mut scratch = BTreeSet::new();
        for i in range.start..last {
            step(&insts[i], &mut st, &mut scratch);
        }
        if g.succ[b].len() < 2 || !is_vote(&insts[last], &st) {
            continue;
        }
        // what is zero, on the side the wave runs, in a half that skipped it
        let run_side = sides(g, b, &insts[last], &st)
            .filter(|&(_, run, _)| run != join)
            .map(|(_, run, mask)| zero_entries(insts, sw, g, b, run, Some(join), zero_at_vote(insts, sw, g, b, mask.unwrap_or(EXEC))));
        // the blocks it decides, nested branches' blocks included
        let mut region: BTreeSet<usize> = BTreeSet::new();
        let mut todo = dependents[b].clone();
        while let Some(x) = todo.pop() {
            if region.insert(x) {
                todo.extend(dependents[x].iter().copied());
            }
        }
        for &x in &region {
            let mut z = run_side.as_ref().and_then(|e| e[x]);
            for i in g.range(x) {
                let d = &insts[i];
                let f = |name: &str| rdna2::field_value(d.enc, &d.words, name);
                // s &= ~m, s |= m with m zero in that half: its bits stay what they were
                let accumulates = matches!(d.inst.name, "S_ANDN2_B64" | "S_OR_B64")
                    && f("SDST") == f("SSRC0")
                    && f("SSRC1").zip(z).is_some_and(|(m, h)| has_pair(h.z, m as usize));
                if let Some(z) = z.as_mut() {
                    zero_step(d, sw[i], z);
                }
                for w in uses(d).writes {
                    if accumulates && matches!(w, Reg::S(_)) {
                        continue;
                    }
                    let scalar = matches!(w, Reg::Scc) || matches!(w, Reg::S(s) if s != EXEC && s != EXEC + 1);
                    if scalar && !stale[join].contains(&w) {
                        stale[join].push(w);
                    }
                }
            }
        }
    }
    stale
}

/// A conditional branch at the end of block `b` (state `st` before it): a vote that may run a
/// side the console wave did not run for the half's lanes.
#[allow(clippy::too_many_arguments)]
fn vote(insts: &[Decoded], sw: &[u128], g: &Cfg, ipdom: &[PostDom], b: usize, d: &Decoded, st: &State, out: &mut BTreeSet<&'static str>) {
    let Some((empty, run, mask)) = sides(g, b, d, st) else {
        // SCC from a mask in no register: counted as a vote that may differ
        if g.succ[b].len() == 2 && matches!(d.inst.name, "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1") && st.scc_vote == SccVote::Mask(None) {
            out.insert("vote-branch");
        }
        return;
    };
    // a mask every half votes on as the wave does: all or none of the active lanes. Under the
    // launch EXEC every half has lanes; under a narrowed one a half may have none and take the
    // other side, which is exact only if neither side turns lanes on again before the join
    let m = mask.unwrap_or(EXEC);
    let uniform = m + 1 < 128 && st.uniform_mask[m] && st.uniform_mask[m + 1];
    if mask.is_none() && st.exec_launch {
        return;
    }
    if uniform {
        if st.exec_launch {
            return;
        }
        if let PostDom::Block(join) = ipdom[b] {
            let zero = zero_at_vote(insts, sw, g, b, EXEC);
            let side = |s: usize| s == join || !widens_after(insts, sw, g, b, s, Some(join), zero, true);
            if side(run) && side(empty) {
                return;
            }
        }
    }
    let joins = ipdom[b] == PostDom::Block(empty);
    let skips = joins || leaves_nothing(insts, g, empty);
    // … and keeps it narrowed: up to the join, or for a way out that ends the program, to the end
    // (a half whose mask is empty must not miss lanes the wave turns on again)
    let at_vote = zero_at_vote(insts, sw, g, b, mask.unwrap_or(EXEC));
    let narrows = (mask.is_none() || narrows_before_lane_work(insts, sw, g, run, at_vote))
        && !widens_after(insts, sw, g, b, run, if joins { Some(empty) } else { None }, at_vote, mask.is_none());
    if !(skips && narrows) {
        out.insert(if mask.is_none() { "unsafe-execz-branch" } else { "vote-branch" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;

    const V: u64 = 256;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);

    fn program(lines: &[I]) -> Vec<Decoded> {
        let mut bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect();
        bytes.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let mut out = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        rdna2::resolve_far_jumps(&mut out);
        out
    }

    fn reasons(lines: &[I]) -> Vec<&'static str> {
        let p = classify(&program(lines));
        assert_eq!(p.tier == Tier::Exact, p.reasons.is_empty());
        p.reasons.into_iter().collect()
    }

    // export v0..v3 (a pixel's colour)
    const EXP: I = ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1), ("VM", 1)], None);
    // vcc = v4 > v5 (per pixel)
    const CMP: I = ("V_CMP_GT_F32", "ENC_VOPC", &[("SRC0", V + 4), ("VSRC1", 5)], None);
    // s[10:11] = exec; exec &= vcc
    const NARROW: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", VCC as u64)], None);
    const RESTORE: I = ("S_OR_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 10)], None);
    // v0 = 1.0
    const BODY: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 242)], None);

    #[test]
    fn per_lane_code_is_exact() {
        assert_eq!(reasons(&[BODY, EXP]), Vec::<&str>::new());
    }

    #[test]
    fn an_if_that_narrows_exec_and_skips_to_the_join_is_exact() {
        // 0 cmp, 1 saveexec, 2 execz → 4; 3 body; 4 restore, 5 exp
        let p: &[I] = &[CMP, NARROW, ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, RESTORE, EXP];
        assert_eq!(reasons(&p), Vec::<&str>::new());
    }

    #[test]
    fn a_vcc_vote_whose_side_runs_with_the_whole_exec_may_differ() {
        // 0 cmp, 1 vccz → 3; 2 body (every lane of the wave, whatever it voted); 3 exp
        let p: &[I] = &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(&p), vec!["vote-branch"]);
    }

    #[test]
    fn a_vcc_vote_that_narrows_exec_first_is_exact() {
        // 0 cmp, 1 vccz → 5; 2 saveexec vcc, 3 body, 4 restore; 5 exp
        let p: &[I] = &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 3)], None), NARROW, BODY, RESTORE, EXP];
        assert_eq!(reasons(&p), Vec::<&str>::new());
    }

    #[test]
    fn a_vote_on_a_compare_of_values_every_lane_has_is_exact() {
        // vcc = s2 > s3 for every lane, under the launch exec: each half votes as the wave does
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
        let p: &[I] = &[UV7, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(&p), Vec::<&str>::new());
    }

    #[test]
    fn whole_quad_mode_at_the_start_keeps_the_launch_mask() {
        // s[88:89] = exec; exec = wqm(exec); a uniform compare then votes as the wave does
        const SAVE: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None);
        const WQM: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64)], None);
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
        let p: &[I] = &[SAVE, WQM, UV7, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    #[test]
    fn an_scc_vote_on_a_mask_is_exact_when_its_side_narrows_exec_to_it() {
        // s[0:1] = vcc & exec (scc: not empty); scc0 → join; exec = s[0:1] (saved in s[2:3]); body; restore
        const AND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", VCC as u64), ("SSRC1", EXEC as u64)], None);
        const SAVE: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)], None);
        const BACK: I = ("S_OR_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 2)], None);
        let p: &[I] = &[CMP, AND, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 3)], None), SAVE, BODY, BACK, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
        // the same vote with the side running under the whole exec
        let q: &[I] = &[CMP, AND, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(q), vec!["vote-branch"]);
    }

    #[test]
    fn an_early_out_when_every_pixel_is_discarded_is_exact() {
        // 0 s[88:89] = exec, 1 cmp, 2 s[88:89] &= ~vcc (discard), 3 scc0 → 7; 4 exec = wqm(s[88:89]),
        // 5 exp (8 bytes), 6 end; 7 null export, 8 end
        const SAVE: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None);
        const KILL: I = ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None);
        const LIVE: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 88)], None);
        const NULL: I = ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("VSRC0", 0), ("VSRC1", 0), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1), ("VM", 1)], None);
        let p: &[I] = &[SAVE, CMP, KILL, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 4)], None), LIVE, EXP, ("S_ENDPGM", "ENC_SOPP", &[], None), NULL];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    #[test]
    fn a_discard_early_out_that_narrows_through_a_whole_quad_copy_is_exact() {
        // as above, with the compiler's form of the narrowing: s[90:91] = wqm(s[88:89]) (and a
        // scalar load), then exec &= s[90:91]
        const SAVE: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None);
        const KILL: I = ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None);
        const WQM: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", 90), ("SSRC0", 88)], None);
        const LOAD: I = ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)], None);
        const LIVE: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 90)], None);
        const NULL: I = ("EXP", "ENC_EXP", &[("TGT", 9), ("EN", 0), ("VSRC0", 0), ("VSRC1", 0), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1), ("VM", 1)], None);
        const END: I = ("S_ENDPGM", "ENC_SOPP", &[], None);
        // (branch offsets in dwords: a scalar load, an export and an imm32 are two)
        let scc0 = |n: u64| -> I {
            ("S_CBRANCH_SCC0", "ENC_SOPP", match n { 7 => &[("SIMM16", 7)], 6 => &[("SIMM16", 6)], _ => &[("SIMM16", 5)] }, None)
        };
        let p: &[I] = &[SAVE, CMP, KILL, scc0(7), WQM, LOAD, LIVE, EXP, END, NULL];
        assert_eq!(reasons(p), Vec::<&str>::new());
        // lane work before the narrowing: the half whose pixels are all discarded would skip it
        let q: &[I] = &[SAVE, CMP, KILL, scc0(6), WQM, BODY, LIVE, EXP, END, NULL];
        assert_eq!(reasons(q), vec!["vote-branch"]);
        // a narrowing by a mask that is not the vote's (the compare's lanes: not empty in that half)
        const OTHER: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", 90), ("SSRC0", VCC as u64)], None);
        let r: &[I] = &[SAVE, CMP, KILL, scc0(5), OTHER, LIVE, EXP, END, NULL];
        assert_eq!(reasons(r), vec!["vote-branch"]);
        // a mode register written before the narrowing is the wave's, not the half's
        const MODE: I = ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", 1), ("SDST", 0)], Some(0));
        let s: &[I] = &[SAVE, CMP, KILL, scc0(7), WQM, MODE, LIVE, EXP, END, NULL];
        assert_eq!(reasons(s), vec!["vote-branch"]);
    }

    #[test]
    fn a_load_from_an_address_every_lane_has_gives_a_value_every_lane_has() {
        // v7 = buffer[s[4:7]] at offset 0 (no VGPR address); vcc = s2 > v7; vccz over code
        const LOAD: I = ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 7), ("SRSRC", 1), ("SOFFSET", 128)], None);
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        let p: &[I] = &[LOAD, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    #[test]
    fn a_scalar_condition_selected_into_a_mask_is_a_vote_every_half_takes_as_the_wave() {
        // scc = s0 < s1; s[10:11] = scc ? exec : 0; vcc = exec & s[10:11]; vccnz → 6 over code on both sides
        const SCMP: I = ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 1)], None);
        const SEL: I = ("S_CSELECT_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", EXEC as u64), ("SSRC1", 128)], None);
        const VAND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", VCC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 10)], None);
        const ELSE: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 240)], None);
        let p: &[I] = &[SCMP, SEL, VAND, ("S_CBRANCH_VCCNZ", "ENC_SOPP", &[("SIMM16", 2)], None), ELSE, ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    #[test]
    fn a_scalar_written_behind_a_vote_and_read_after_the_join_may_differ() {
        // 0 cmp, 1 saveexec, 2 execz → 4; 3 s5 = 1 (only where some lane voted yes); 4 restore,
        // 5 v0 = s5, 6 exp
        const DEF: I = ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 129)], None);
        const USE: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 5)], None);
        let p: &[I] = &[CMP, NARROW, ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None), DEF, RESTORE, USE, EXP];
        assert_eq!(reasons(p), vec!["stale-scalar"]);
        // not read after the join: exact
        let q: &[I] = &[CMP, NARROW, ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None), DEF, RESTORE, EXP];
        assert_eq!(reasons(q), Vec::<&str>::new());
    }

    #[test]
    fn a_scalar_behind_a_vote_every_half_takes_as_the_wave_is_exact() {
        // a compare of values every lane has, under the launch exec: every half takes the same side
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
        const DEF: I = ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 129)], None);
        const USE: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 5)], None);
        let p: &[I] = &[UV7, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), DEF, USE, EXP];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    // ----- counter-examples from the review of #517 (each was Exact and exports something else
    // when the wave is cut)

    #[test]
    fn a_value_written_under_a_narrowed_exec_is_not_every_lanes() {
        // v6 = s5; if (v4 > v5) v6 = s6; s0 = first lane's v6; export it
        let p: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 5)], None),
            CMP,
            NARROW,
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 6)], None),
            RESTORE,
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 6)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None),
            EXP,
        ];
        assert!(reasons(p).contains(&"cross-lane"));
    }

    #[test]
    fn a_per_lane_select_is_not_every_lanes() {
        // v7 = s3; v6 = vcc ? v7 : 0 (vcc per pixel); s0 = first lane's v6; export it
        let p: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None),
            CMP,
            ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 7)], None),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 6)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None),
            EXP,
        ];
        assert!(reasons(p).contains(&"cross-lane"));
    }

    #[test]
    fn a_vote_read_as_a_number_may_differ() {
        // scc = (vcc != 0); s20 = scc ? s6 : s5; export s20
        let p: &[I] = &[
            CMP,
            ("S_CMP_LG_U64", "ENC_SOPC", &[("SSRC0", VCC as u64), ("SSRC1", 128)], None),
            ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 6), ("SSRC1", 5)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None),
            EXP,
        ];
        assert!(!reasons(p).is_empty());
    }

    #[test]
    fn a_mask_word_moved_into_another_register_is_still_a_mask() {
        // s20 = exec_lo; v0 = s20; export
        let p: &[I] = &[("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None), ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None), EXP];
        assert_eq!(reasons(p), vec!["mask-data"]);
    }

    #[test]
    fn a_write_that_keeps_the_old_value_keeps_its_taint() {
        // s20 = popcount(exec); s20 += 1 (s_addk); v0 = s20; export
        let p: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("S_ADDK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None),
            EXP,
        ];
        assert_eq!(reasons(p), vec!["lane-count"]);
    }

    #[test]
    fn a_vote_whose_side_widens_exec_again_may_differ() {
        // 0 cmp, 1 vccz → 6; 2 exec &= vcc (saved in s[10:11]), 3 exec = s[10:11] (widened again),
        // 4 v0 = 1.0 under the whole exec, 5 s_nop; 6 exp
        const WIDEN: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 10)], None);
        let p: &[I] = &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 4)], None), NARROW, WIDEN, BODY, ("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None), EXP];
        assert!(reasons(p).contains(&"vote-branch"));
    }

    #[test]
    fn an_scc_vote_is_forgotten_when_its_mask_is_rewritten() {
        // 0 cmp, 1 s[0:1] = vcc & exec (scc), 2 s[0:1] = s[2:3] (scc is no longer about s[0:1]),
        // 3 scc0 → 6; 4 exec = s[0:1] & exec, 5 body; 6 exp
        const AND: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", VCC as u64), ("SSRC1", EXEC as u64)], None);
        const OTHER: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 2)], None);
        const RUN: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", 0), ("SSRC1", EXEC as u64)], None);
        let p: &[I] = &[CMP, AND, OTHER, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)], None), RUN, BODY, EXP];
        assert!(reasons(p).contains(&"vote-branch"));
    }

    #[test]
    fn the_high_word_of_exec_and_a_number_in_exec_are_seen() {
        // exec_hi = 0: in a 32-lane half its lanes are not under exec_hi – a sink of its own
        let hi: &[I] = &[("S_MOV_B32", "ENC_SOP1", &[("SDST", EXEC as u64 + 1), ("SSRC0", 128)], None), BODY, EXP];
        assert!(!reasons(hi).is_empty());
        // exec = lane 0 only: lane 0 of each half is a different console lane
        let one: &[I] = &[("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 129)], None), BODY, EXP];
        assert!(!reasons(one).is_empty());
    }

    #[test]
    fn an_image_load_writes_only_the_registers_its_dmask_names() {
        // v3 = popcount(exec) (tainted); v0 = sample (DMASK 1: v0 only); export v0..v3 – v3 still tainted
        let p: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 20)], None),
            ("IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 0), ("VADDR", 4), ("SRSRC", 2), ("SSAMP", 6), ("DMASK", 1), ("DIM", 1)], None),
            EXP,
        ];
        assert_eq!(reasons(p), vec!["lane-count"]);
    }

    #[test]
    fn a_ds_write_is_a_store() {
        // m0 = popcount(exec); ds_write_b32 v4, v0 with v0 = m0
        let p: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None),
            ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 4), ("DATA0", 0)], None),
        ];
        assert_eq!(reasons(p), vec!["lane-count"]);
        let q: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None),
            ("DS_MAX_U32", "ENC_DS", &[("ADDR", 4), ("DATA0", 0)], None),
        ];
        assert_eq!(reasons(q), vec!["lane-count"]);
    }

    #[test]
    fn a_dpp_row_mask_names_rows_a_half_does_not_have() {
        // v0 = v6 in rows 2 and 3 only (quad_perm identity, row_mask 0xc); export
        let p: &[I] = &[("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 0), ("SRC0", 0xFA), ("VSRC0", 6), ("DPP_CTRL", 0xE4), ("ROW_MASK", 0xC), ("BANK_MASK", 0xF)], None), EXP];
        assert_eq!(reasons(p), vec!["lane-id"]);
    }

    #[test]
    fn a_uniform_compare_under_a_narrowed_exec_needs_both_sides_to_keep_it() {
        // exec &= (v4 > v5); vcc = s2 > v7 (every lane's); vccz → join; exec = saved (wider again);
        // v0 = 1.0; join: exec = saved; export – a half with no lane left skips the body
        const UV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None);
        const UCMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None);
        let p: &[I] = &[UV7, CMP, NARROW, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 2)], None), RESTORE, BODY, RESTORE, EXP];
        assert!(!reasons(p).is_empty());
        // the same without widening EXEC in the body: exact
        let q: &[I] = &[UV7, CMP, NARROW, UCMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None), BODY, RESTORE, EXP];
        assert_eq!(reasons(q), Vec::<&str>::new());
    }

    #[test]
    fn one_lane_written_is_a_different_console_lane_in_each_half() {
        // v0[lane 5] = s5; export v0
        let p: &[I] = &[("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 5), ("SRC1", 128 + 5)], None), EXP];
        assert_eq!(reasons(p), vec!["lane-id"]);
    }

    #[test]
    fn a_mask_accumulated_behind_a_vote_is_not_stale() {
        // live = exec; if (v4 > v5) { live &= ~(2 > v9); exec &= live; v0 = 1.0 }; exec = saved & live
        let p: &[I] = &[
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None),
            CMP,
            NARROW,
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 4)], None),
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 130), ("VSRC1", 9)], None),
            ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None),
            ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 88)], None),
            BODY,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", 10), ("SSRC1", 88)], None),
            EXP,
        ];
        assert_eq!(reasons(p), Vec::<&str>::new());
    }

    // ----- a side of a vote may turn lanes on again only from a mask that is zero in a half that
    // skipped it, on every way to that write (third review)

    const SAVE20: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None);
    const EXEC_S20: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 20)], None);
    const BODY1: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 242)], None);
    const fn execz(n: u64) -> I {
        ("S_CBRANCH_EXECZ", "ENC_SOPP", match n {
            1 => &[("SIMM16", 1)],
            2 => &[("SIMM16", 2)],
            4 => &[("SIMM16", 4)],
            _ => &[("SIMM16", 5)],
        }, None)
    }

    #[test]
    fn the_first_write_of_an_exec_votes_side_is_looked_at() {
        let p: &[I] = &[CMP, SAVE20, NARROW, execz(2), EXEC_S20, BODY, RESTORE, EXP];
        assert!(reasons(p).contains(&"unsafe-execz-branch"));
        for first in [("S_ANDN2_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 20)][..], None), ("S_ANDN2_WREXEC_B64", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 20)][..], None)] {
            let p: &[I] = &[CMP, SAVE20, NARROW, execz(2), first, BODY, EXEC_S20, EXP];
            assert!(reasons(p).contains(&"unsafe-execz-branch"), "{}", first.0);
        }
    }

    #[test]
    fn a_votes_mask_rewritten_in_its_side_no_longer_bounds_it() {
        let p: &[I] = &[
            CMP,
            ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 5)], None),
            NARROW,
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", VCC as u64), ("SSRC0", 10)], None),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", VCC as u64)], None),
            BODY,
            RESTORE,
            EXP,
        ];
        assert!(reasons(p).contains(&"vote-branch"));
    }

    #[test]
    fn a_save_counts_only_until_it_is_overwritten_and_only_after_it_is_made() {
        const SAVE12: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", EXEC as u64)], None);
        const EXEC_S12: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", 12)], None);
        let p: &[I] = &[CMP, SAVE20, NARROW, execz(5), BODY, SAVE12, ("S_MOV_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 20)], None), EXEC_S12, BODY, RESTORE, EXP];
        assert!(reasons(p).contains(&"unsafe-execz-branch"));
        // without the rewrite: exact
        let q: &[I] = &[CMP, SAVE20, NARROW, execz(4), BODY, SAVE12, EXEC_S12, BODY, RESTORE, EXP];
        assert_eq!(reasons(q), Vec::<&str>::new());
        // a save after the write does not bound it
        let r: &[I] = &[CMP, SAVE20, NARROW, execz(4), BODY, EXEC_S20, BODY, SAVE20, RESTORE, EXP];
        assert!(reasons(r).contains(&"unsafe-execz-branch"));
    }

    #[test]
    fn an_else_flip_of_the_exec_from_before_the_vote_turns_its_lanes_on() {
        const OR_SAVE: I = ("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 10)], None);
        const FLIP: I = ("S_XOR_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 10)], None);
        let p: &[I] = &[CMP, SAVE20, NARROW, execz(4), BODY, OR_SAVE, FLIP, BODY1, EXEC_S20, EXP];
        assert!(reasons(p).contains(&"unsafe-execz-branch"));
        // the flip at the join, as compilers emit it: exact
        let q: &[I] = &[CMP, SAVE20, NARROW, execz(1), BODY, OR_SAVE, FLIP, BODY1, EXEC_S20, EXP];
        assert_eq!(reasons(q), Vec::<&str>::new());
    }

    #[test]
    fn a_mask_accumulated_from_a_vcc_wider_than_exec_is_stale() {
        let p: &[I] = &[
            CMP,
            SAVE20,
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None),
            NARROW,
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", VCC as u64), ("SSRC0", 20)], None),
            execz(2),
            BODY,
            ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None),
            EXEC_S20,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 88)], None),
            BODY1,
            EXEC_S20,
            EXP,
        ];
        assert!(reasons(p).contains(&"stale-scalar"));
        // with VCC from a compare under the narrowed EXEC: exact
        let q: &[I] = &[
            CMP,
            SAVE20,
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 88), ("SSRC0", EXEC as u64)], None),
            NARROW,
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 130), ("VSRC1", 9)], None),
            execz(2),
            BODY,
            ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 88), ("SSRC0", 88), ("SSRC1", VCC as u64)], None),
            EXEC_S20,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 88)], None),
            BODY1,
            EXEC_S20,
            EXP,
        ];
        assert_eq!(reasons(q), Vec::<&str>::new());
    }

    // ----- fourth review

    #[test]
    fn a_vcc_votes_side_must_reach_the_join_with_the_votes_exec() {
        // vccz → join; exec &= vcc (saved in s10, never restored); body; join: exp
        let p: &[I] = &[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 2)], None), NARROW, BODY, EXP];
        assert!(reasons(p).contains(&"vote-branch"));
        let q: &[I] = &[
            CMP,
            ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 2)], None),
            ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", VCC as u64)], None),
            BODY,
            EXP,
        ];
        assert!(reasons(q).contains(&"vote-branch"));
    }

    #[test]
    fn whole_quad_mode_after_a_write_turns_on_lanes_without_it() {
        // v4 = s3 (every launched lane); exec = wqm(exec); vcc = s2 > v4; vccz → exp; body; exp
        let p: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 3)], None),
            ("S_WQM_B64", "ENC_SOP1", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64)], None),
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 4)], None),
            ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None),
            BODY,
            EXP,
        ];
        assert!(reasons(p).contains(&"vote-branch"));
    }

    #[test]
    fn a_number_combined_into_a_mask_is_mask_data() {
        // exec &= 1: lane 0 of the wave, lanes 0 and 32 of the halves
        let p: &[I] = &[("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 129)], None), BODY, EXP];
        assert!(reasons(p).contains(&"mask-data"));
        // the same through s[20:21] = 1
        let q: &[I] = &[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 129)], None),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 21), ("SSRC0", 128)], None),
            ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 20)], None),
            BODY,
            EXP,
        ];
        assert!(reasons(q).contains(&"mask-data"));
        // a uniform vcc &= 1, then a vote on it
        let r: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 3)], None),
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 7)], None),
            ("S_AND_B64", "ENC_SOP2", &[("SDST", VCC as u64), ("SSRC0", VCC as u64), ("SSRC1", 129)], None),
            ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None),
            BODY,
            EXP,
        ];
        assert!(!reasons(r).is_empty());
        // a scalar condition as a mask (-1 / 0) is not a number
        let s: &[I] = &[
            ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 2), ("SSRC1", 3)], None),
            ("S_CSELECT_B64", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 193), ("SSRC1", 128)], None),
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 22), ("SSRC0", EXEC as u64), ("SSRC1", 20)], None),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 22)], None),
            EXP,
        ];
        assert_eq!(reasons(s), Vec::<&str>::new());
    }

    // ----- every reason has a test

    #[test]
    fn a_first_lane_index_may_differ() {
        let p: &[I] = &[("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None), ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 20)], None), EXP];
        assert_eq!(reasons(p), vec!["first-lane"]);
    }

    #[test]
    fn work_under_an_exec_that_depends_on_the_cut_may_differ() {
        // s[20:21] = 1 << popcount(exec) → exec (a number, and a lane count); v0 = 1.0 under it
        let p: &[I] = &[
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", EXEC as u64)], None),
            ("S_LSHL_B64", "ENC_SOP2", &[("SDST", 22), ("SSRC0", 129), ("SSRC1", 20)], None),
            ("S_AND_B64", "ENC_SOP2", &[("SDST", EXEC as u64), ("SSRC0", EXEC as u64), ("SSRC1", 22)], None),
            BODY,
            EXP,
        ];
        assert!(reasons(p).contains(&"tainted-exec"));
    }

    #[test]
    fn an_execz_that_does_not_skip_to_the_join_may_differ() {
        // 0 cmp, 1 saveexec, 2 execz → 5 (past the join's code); 3 body, 4 s_branch → 6; 5 v0 = 0; 6 exp
        let p: &[I] = &[
            CMP,
            NARROW,
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None),
            BODY,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 128)], None),
            EXP,
        ];
        assert!(reasons(p).contains(&"unsafe-execz-branch"));
    }

    #[test]
    fn an_indirect_jump_is_not_classified() {
        let p: &[I] = &[("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None)];
        assert_eq!(reasons(p), vec!["no-graph"]);
    }

    #[test]
    fn a_lane_count_that_reaches_an_export_may_differ() {
        // s0 = popcount(exec); v0 = s0; exp
        let p: &[I] = &[("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None), ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None), EXP];
        assert_eq!(reasons(&p), vec!["lane-count"]);
    }

    #[test]
    fn a_lane_count_that_reaches_nothing_is_exact() {
        let p: &[I] = &[("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", EXEC as u64)], None), BODY, EXP];
        assert_eq!(reasons(&p), Vec::<&str>::new());
    }

    #[test]
    fn the_lane_index_and_a_rank_may_differ() {
        let lane: &[I] = &[("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None), EXP];
        assert_eq!(reasons(&lane), vec!["lane-id"]);
        let rank: &[I] = &[("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 20), ("SRC1", 128)], None), EXP];
        // s20 is not known as a mask here, but mbcnt over it is still a rank
        assert_eq!(reasons(&rank), vec!["rank"]);
    }

    #[test]
    fn reading_another_lanes_value_may_differ_unless_every_lane_has_it() {
        // s0 = v4 of the first active lane; v0 = s0; exp
        let varying: &[I] = &[("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None), ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 0)], None), EXP];
        assert_eq!(reasons(&varying), vec!["cross-lane"]);
        // v4 = s5 first: every lane has it
        let uniform: &[I] = &[("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 5)], None), varying[0], varying[1], EXP];
        assert_eq!(reasons(&uniform), Vec::<&str>::new());
    }

    #[test]
    fn a_mask_used_as_a_number_may_differ() {
        // v0 = exec_lo
        let p: &[I] = &[("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", EXEC as u64)], None), EXP];
        assert_eq!(reasons(&p), vec!["mask-data"]);
    }
}
