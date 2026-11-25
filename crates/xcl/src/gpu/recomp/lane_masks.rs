//! Lane masks as per-lane booleans: what a backend that writes DXIL itself (no HLSL, so no
//! structured control flow) needs in order to run console code without emulating EXEC.
//!
//! The HLSL path keeps EXEC as a wave-uniform value and guards every vector write with the
//! lane's bit of it. With an unstructured CFG each lane can instead branch on its own bit and
//! leave reconvergence to the host driver. Per program this works out:
//! - every lane mask (EXEC, VCC, any SGPR a mask is kept in) as a boolean function of *atoms* –
//!   the lane's result of a VALU compare or carry, a wave-uniform SCC, a mask it cannot follow,
//!   a join it cannot resolve – as a reduced ordered BDD;
//! - every branch on a mask (`s_cbranch_execz/execnz/vccz/vccnz`, SCC of a mask op) as a branch
//!   of each lane on its own bit, and a branch on any other SCC as a uniform one;
//! - for every block, what holds in each lane that reaches it: a forward dataflow over the
//!   branch conditions; facts a back edge carries about atoms its loop redefines are dropped;
//! - which vector instructions still need a guard: those where reaching the instruction does
//!   not imply the lane's EXEC bit.
//!
//! Branching lane by lane on a mask assumes what compiled code does: it jumps over work that
//! inactive lanes would not change anyway, or it tests a uniform value. A join of a mask is
//! resolved when every lane coming in has it set (or clear) – EXEC restored where the branches
//! meet – and becomes an atom otherwise. A loop head's join is guessed from its forward edges
//! and checked against its back edges; a wrong guess becomes an atom and the analysis restarts.
//!
//! Nothing translates with this yet: `xcl shaders --recomp` reports its counts.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::x64_masks;
use crate::gpu::rdna2::{self, Decoded, F_CONDITIONAL, F_INDIRECT, F_TERMINATOR};

type R<T> = Result<T, &'static str>;

/// A BDD node: 0 = false, 1 = true.
type B = u32;
const FALSE: B = 0;
const TRUE: B = 1;
/// More nodes than this: the program fails with "lane conditions too large".
const MAX_NODES: usize = 1 << 20;
/// New nodes one reach fact may take before it is weakened.
const STEP_NODES: usize = 1 << 9;
/// Passes of the reach fixpoint, restarts after wrong loop-head guesses included.
const MAX_ROUNDS: usize = 256;
const EXEC: usize = 126;
const VCC: usize = 106;
/// The SCC slot of a join.
const SCC: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Op {
    And,
    Or,
    Xor,
}

struct Bdd {
    /// (atom, low, high); the terminals have atom u32::MAX, so atoms created first are on top
    nodes: Vec<(u32, B, B)>,
    unique: HashMap<(u32, B, B), B>,
    memo: HashMap<(Op, B, B), B>,
    /// node count `mk` fails at
    limit: usize,
}

impl Bdd {
    fn new() -> Self {
        Bdd { nodes: vec![(u32::MAX, 0, 0), (u32::MAX, 1, 1)], unique: HashMap::new(), memo: HashMap::new(), limit: MAX_NODES }
    }

    /// `f`'s result, or None when it needs more than `STEP_NODES` new nodes (those it made stay:
    /// they are valid nodes, and the memo stays right). For results that can be weakened.
    fn within<T>(&mut self, f: impl FnOnce(&mut Self) -> R<T>) -> Option<T> {
        self.limit = (self.nodes.len() + STEP_NODES).min(MAX_NODES);
        let r = f(self);
        self.limit = MAX_NODES;
        r.ok()
    }

    /// The function "atom v", whatever the limits: what a weakened result falls back on.
    fn atom(&mut self, v: u32) -> B {
        if let Some(&n) = self.unique.get(&(v, FALSE, TRUE)) {
            return n;
        }
        let n = self.nodes.len() as B;
        self.nodes.push((v, FALSE, TRUE));
        self.unique.insert((v, FALSE, TRUE), n);
        n
    }

    fn mk(&mut self, v: u32, lo: B, hi: B) -> R<B> {
        if lo == hi {
            return Ok(lo);
        }
        if let Some(&n) = self.unique.get(&(v, lo, hi)) {
            return Ok(n);
        }
        if self.nodes.len() >= self.limit {
            return Err("lane conditions too large");
        }
        let n = self.nodes.len() as B;
        self.nodes.push((v, lo, hi));
        self.unique.insert((v, lo, hi), n);
        Ok(n)
    }

    fn apply(&mut self, op: Op, a: B, b: B) -> R<B> {
        let (a, b) = (a.min(b), a.max(b));
        match (op, a) {
            (Op::And, FALSE) => return Ok(FALSE),
            (Op::Or, TRUE) => return Ok(TRUE),
            (Op::And, TRUE) | (Op::Or | Op::Xor, FALSE) => return Ok(b),
            _ if a == b => return Ok(if op == Op::Xor { FALSE } else { a }),
            _ => {}
        }
        if let Some(&n) = self.memo.get(&(op, a, b)) {
            return Ok(n);
        }
        let (na, nb) = (self.nodes[a as usize], self.nodes[b as usize]);
        let v = na.0.min(nb.0);
        let (a0, a1) = if na.0 == v { (na.1, na.2) } else { (a, a) };
        let (b0, b1) = if nb.0 == v { (nb.1, nb.2) } else { (b, b) };
        let lo = self.apply(op, a0, b0)?;
        let hi = self.apply(op, a1, b1)?;
        let n = self.mk(v, lo, hi)?;
        self.memo.insert((op, a, b), n);
        Ok(n)
    }

    fn and(&mut self, a: B, b: B) -> R<B> {
        self.apply(Op::And, a, b)
    }

    fn or(&mut self, a: B, b: B) -> R<B> {
        self.apply(Op::Or, a, b)
    }

    fn not(&mut self, a: B) -> R<B> {
        self.apply(Op::Xor, a, TRUE)
    }

    /// a → b, without building a ∧ ¬b.
    fn implies(&self, a: B, b: B) -> bool {
        self.implies_in(a, b, &mut HashSet::new())
    }

    fn implies_in(&self, a: B, b: B, seen: &mut HashSet<(B, B)>) -> bool {
        if a == FALSE || b == TRUE || a == b {
            return true;
        }
        if a == TRUE && b == FALSE {
            return false;
        }
        if !seen.insert((a, b)) {
            return true;
        }
        let (na, nb) = (self.nodes[a as usize], self.nodes[b as usize]);
        let v = na.0.min(nb.0);
        let (a0, a1) = if na.0 == v { (na.1, na.2) } else { (a, a) };
        let (b0, b1) = if nb.0 == v { (nb.1, nb.2) } else { (b, b) };
        self.implies_in(a0, b0, seen) && self.implies_in(a1, b1, seen)
    }

    /// a → ¬b, i.e. a ∧ b is false, without building it.
    fn disjoint(&self, a: B, b: B, seen: &mut HashSet<(B, B)>) -> bool {
        if a == FALSE || b == FALSE {
            return true;
        }
        if a == TRUE && b == TRUE {
            return false;
        }
        if a == b || !seen.insert((a, b)) {
            return a != b;
        }
        let (na, nb) = (self.nodes[a as usize], self.nodes[b as usize]);
        let v = na.0.min(nb.0);
        let (a0, a1) = if na.0 == v { (na.1, na.2) } else { (a, a) };
        let (b0, b1) = if nb.0 == v { (nb.1, nb.2) } else { (b, b) };
        self.disjoint(a0, b0, seen) && self.disjoint(a1, b1, seen)
    }

    /// `a` with the atoms `drop` names quantified away (∃).
    fn exists(&mut self, a: B, drop: &dyn Fn(u32) -> bool, memo: &mut HashMap<B, B>) -> R<B> {
        if a <= TRUE {
            return Ok(a);
        }
        if let Some(&n) = memo.get(&a) {
            return Ok(n);
        }
        let (v, lo, hi) = self.nodes[a as usize];
        let (lo, hi) = (self.exists(lo, drop, memo)?, self.exists(hi, drop, memo)?);
        let n = if drop(v) { self.or(lo, hi)? } else { self.mk(v, lo, hi)? };
        memo.insert(a, n);
        Ok(n)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Atom {
    /// the lane's result of the VALU compare / carry at this pc (before EXEC)
    Lane(u64),
    /// SCC written at this pc: uniform
    Scc(u64),
    /// a mask the analysis cannot follow, read at this pc from this register
    Opaque(u64, usize),
    /// an unresolved join of this slot (SGPR, `SCC`) at the start of this block
    Join(usize, usize),
    /// EXEC at launch, where it need not hold every lane that runs (`LaunchExec::Unknown`)
    Launch,
}

/// EXEC at the program's start, as the analysis takes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaunchExec {
    /// every lane that runs is in EXEC: kernel mode (lanes outside it leave at once) and compute
    All,
    /// unknown per lane: a PS, whose helper lanes start outside EXEC and run on. A vector
    /// instruction keeps its guard until EXEC is written.
    Unknown,
}

/// How a tracked SCC bit relates to the hardware's wave-wide SCC.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SccKind {
    /// Uniform scalar SCC (independent of per-lane mask bits).
    Uniform,
    /// Hardware SCC = (mask != 0): 1 iff any lane in the wave has bit 1.
    Any,
    /// Hardware SCC = (a == b): 1 iff every lane in the wave has bit 1 (`!(a ^ b)`).
    All,
    /// Joined from incompatible mask-op SCC kinds or overflowed: never sound as a per-lane read.
    Unknown,
}

/// The masks at a program point. `regs[r]`: the lane function of the mask SGPR r holds (wave64:
/// in the pair's first register), None when it holds nothing the analysis knows as a mask.
/// EXEC always has one. `scc`: (lane function, how hardware reduces the wave's bits to SCC).
#[derive(Clone, PartialEq)]
struct State {
    regs: Vec<Option<B>>,
    scc: Option<(B, SccKind)>,
}

/// How a block's successor is reached: unconditionally, or by its branch taken / not taken.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Edge {
    Always,
    Taken,
    NotTaken,
}

/// A basic block of the program, as the lowering walks it.
#[derive(Clone, Debug)]
pub struct Block {
    /// its instructions: indices into the decoded program
    pub insts: std::ops::Range<usize>,
    pub succ: Vec<(usize, Edge)>,
    /// predecessors that the entry reaches, back edges included
    pub preds: Vec<usize>,
    /// it ends in a branch on a lane mask: each lane takes it on its own bit
    pub lane_branch: bool,
}

/// What the analysis found in a program.
#[derive(Debug, Default)]
pub struct Summary {
    /// vector instructions (VALU writes, memory, LDS, exports)
    pub vector: usize,
    /// pcs of those that still need a guard on the lane's EXEC bit
    pub guarded: Vec<u64>,
    /// pcs of the mask-width `s_wqm` whose result is every lane (whole-quad mode of the launch
    /// EXEC, `LaunchExec::Unknown`)
    pub wqm_all: Vec<u64>,
    /// conditional branches on a lane mask (each lane decides) and on uniform values
    pub lane_branches: usize,
    pub uniform_branches: usize,
    /// the CFG (block 0 is the entry) and the order to lower it in (reverse postorder of the
    /// blocks the entry reaches: every block after its forward predecessors)
    pub blocks: Vec<Block>,
    pub order: Vec<usize>,
}

/// Whether `d` reads SCC as a non-branch input (either implicitly via `OPR_SSRC_SPECIAL_SCC`
/// or explicitly via source encoding 253 / `SRC_SCC`).
fn reads_scc(d: &Decoded) -> bool {
    if matches!(d.inst.name, "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1") {
        return false;
    }
    if d.ienc.operands.iter().any(|o| o.input && o.ty == "OPR_SSRC_SPECIAL_SCC") {
        return true;
    }
    ["SSRC0", "SSRC1", "SRC0", "SRC1", "SRC2", "SOFFSET"]
        .iter()
        .any(|f| rdna2::field_value(d.enc, &d.words, f) == Some(253))
}

/// Executed with the lane's EXEC bit (VOPC and the lane reads / writes are mask producers).
fn is_vector(d: &Decoded) -> bool {
    let n = d.inst.name;
    !n.starts_with("S_") && !n.starts_with("V_CMP") && !matches!(n, "V_READLANE_B32" | "V_READFIRSTLANE_B32" | "V_WRITELANE_B32")
}

struct Lanes<'a> {
    insts: &'a [Decoded],
    wave64: bool,
    launch: LaunchExec,
    wave_branches: bool,
    bdd: Bdd,
    atoms: HashMap<Atom, u32>,
    /// block each atom is born in
    born: Vec<usize>,
    /// block → its instruction index range, successors (block, edge), predecessors (block,
    /// index in its successors)
    blocks: Vec<(usize, usize)>,
    succ: Vec<Vec<(usize, Edge)>>,
    preds: Vec<Vec<(usize, usize)>>,
    /// reverse postorder, and for each block: Some(blocks of its loop) when it heads one
    order: Vec<usize>,
    body: Vec<Option<Vec<bool>>>,
    /// per instruction: SGPRs it overwrites (`x64_masks::uses`), SGPRs it may change, whether
    /// it writes SCC
    effects: Vec<(u128, u128, bool)>,
}

impl<'a> Lanes<'a> {
    fn new(insts: &'a [Decoded], wave64: bool, launch: LaunchExec) -> R<Self> {
        let starts: HashMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
        let mut leaders = BTreeSet::from([0usize]);
        let mut target = vec![None; insts.len()];
        for (i, d) in insts.iter().enumerate() {
            let indirect = d.far_target.is_none() && (d.inst.flags & F_INDIRECT != 0 || matches!(d.inst.name, "S_SETPC_B64" | "S_SWAPPC_B64"));
            if indirect {
                return Err("indirect jump");
            }
            if let Some(t) = rdna2::branch_target(d) {
                let &j = starts.get(&t).ok_or("branch into an instruction")?;
                if j == 0 {
                    return Err("branch to the entry");
                }
                target[i] = Some(j);
                leaders.extend([j, i + 1]);
            } else if d.inst.flags & F_TERMINATOR != 0 {
                leaders.insert(i + 1);
            }
        }
        leaders.retain(|&i| i < insts.len());
        let firsts: Vec<usize> = leaders.into_iter().collect();
        let block_of: HashMap<usize, usize> = firsts.iter().enumerate().map(|(b, &i)| (i, b)).collect();
        let blocks: Vec<(usize, usize)> = firsts.iter().enumerate().map(|(b, &i)| (i, firsts.get(b + 1).copied().unwrap_or(insts.len()))).collect();
        let succ: Vec<Vec<(usize, Edge)>> = blocks
            .iter()
            .map(|&(_, end)| {
                let d = &insts[end - 1];
                let next = block_of.get(&end).copied();
                match target[end - 1] {
                    Some(t) if d.inst.flags & F_CONDITIONAL != 0 => [Some((block_of[&t], Edge::Taken)), next.map(|n| (n, Edge::NotTaken))].into_iter().flatten().collect(),
                    Some(t) => vec![(block_of[&t], Edge::Always)],
                    None if d.inst.flags & F_TERMINATOR != 0 => vec![],
                    None => next.map(|n| (n, Edge::Always)).into_iter().collect(),
                }
            })
            .collect();
        let n = blocks.len();
        // reverse postorder of what the entry reaches
        let (mut order, mut seen, mut stack) = (Vec::new(), vec![false; n], vec![(0usize, 0usize)]);
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, i) = *top;
            if let Some(&(s, _)) = succ[b].get(i) {
                top.1 += 1;
                if !seen[s] {
                    seen[s] = true;
                    stack.push((s, 0));
                }
            } else {
                order.push(b);
                stack.pop();
            }
        }
        order.reverse();
        let mut rpo = vec![usize::MAX; n];
        for (i, &b) in order.iter().enumerate() {
            rpo[b] = i;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &order {
            for (k, &(s, _)) in succ[b].iter().enumerate() {
                preds[s].push((b, k));
            }
        }
        // dominators (Cooper, Harvey, Kennedy) – a retreating edge must go to a dominator
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
            while a != b {
                while rpo[a] > rpo[b] {
                    a = idom[a];
                }
                while rpo[b] > rpo[a] {
                    b = idom[b];
                }
            }
            a
        };
        loop {
            let mut changed = false;
            for &b in &order[1..] {
                let new = preds[b].iter().map(|p| p.0).filter(|&p| idom[p] != usize::MAX).reduce(|a, p| intersect(&idom, a, p)).unwrap_or(usize::MAX);
                changed |= idom[b] != new;
                idom[b] = new;
            }
            if !changed {
                break;
            }
        }
        let dominates = |h: usize, mut b: usize| loop {
            if b == h {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = idom[b];
        };
        let mut body: Vec<Option<Vec<bool>>> = vec![None; n];
        for &h in &order {
            for &(p, _) in &preds[h] {
                if rpo[p] < rpo[h] {
                    continue;
                }
                if !dominates(h, p) {
                    return Err("irreducible control flow");
                }
                let inside = body[h].get_or_insert_with(|| vec![false; n]);
                inside[h] = true;
                let mut work = vec![p];
                while let Some(b) = work.pop() {
                    if !inside[b] {
                        inside[b] = true;
                        work.extend(preds[b].iter().map(|q| q.0));
                    }
                }
            }
        }
        let effects = insts
            .iter()
            .map(|d| {
                let written = x64_masks::uses(d).1;
                let explicit = rdna2::sgpr_uses(d).1.iter().fold(0u128, |m, &r| m | 1 << r);
                let kill = if d.inst.name.starts_with("S_MOVREL") { !0 } else { written | explicit };
                (written, kill, d.ienc.operands.iter().any(|o| o.output && o.ty == "OPR_SSRC_SPECIAL_SCC"))
            })
            .collect();
        Ok(Lanes { insts, wave64, launch, wave_branches: false, bdd: Bdd::new(), atoms: HashMap::new(), born: Vec::new(), blocks, succ, preds, order, body, effects })
    }

    fn atom(&mut self, a: Atom, block: usize) -> R<B> {
        let fresh = self.born.len() as u32;
        let v = *self.atoms.entry(a).or_insert(fresh);
        if v == fresh {
            self.born.push(block);
        }
        Ok(self.bdd.atom(v))
    }

    /// `step` within `STEP_NODES` new nodes; past that, what the instruction writes becomes
    /// unknown: EXEC and SCC atoms, no mask in the other registers (weaker, still sound).
    fn step_bounded(&mut self, st: &mut State, i: usize, block: usize) -> R<()> {
        let mut next = st.clone();
        self.bdd.limit = (self.bdd.nodes.len() + STEP_NODES).min(MAX_NODES);
        let r = self.step(&mut next, i, block);
        self.bdd.limit = MAX_NODES;
        if r.is_ok() {
            *st = next;
            return Ok(());
        }
        let d = &self.insts[i];
        let (_, kill, writes_scc) = self.effects[i];
        for r in (0..128).filter(|&r| kill >> r & 1 != 0) {
            self.kill(st, r);
        }
        if st.regs[EXEC].is_none() || d.inst.name.starts_with("V_CMPX") {
            st.regs[EXEC] = Some(self.atom(Atom::Opaque(d.pc, EXEC), block)?);
        }
        if writes_scc {
            let kind = if self.wave_branches { SccKind::Uniform } else { SccKind::Unknown };
            st.scc = Some((self.atom(Atom::Scc(d.pc), block)?, kind));
        }
        Ok(())
    }

    /// A scalar source read as a mask: a register's lane function, all lanes, none, or None.
    fn mask_src(&self, st: &State, d: &Decoded, code: u64) -> Option<B> {
        match code {
            128 => Some(FALSE),
            193 => Some(TRUE),
            255 if d.literal == Some(0) => Some(FALSE),
            255 if d.literal == Some(!0) && !self.wave64 => Some(TRUE),
            c if c < 128 => st.regs[c as usize],
            _ => None,
        }
    }

    /// … and an atom where the analysis cannot follow it.
    fn need(&mut self, st: &State, d: &Decoded, code: u64, block: usize) -> R<B> {
        match self.mask_src(st, d, code) {
            Some(v) => Ok(v),
            None => self.atom(Atom::Opaque(d.pc, code as usize), block),
        }
    }

    fn combine(&mut self, op: &str, a: B, b: B) -> R<Option<B>> {
        let bdd = &mut self.bdd;
        Ok(Some(match op {
            "AND" => bdd.and(a, b)?,
            "OR" => bdd.or(a, b)?,
            "XOR" => bdd.apply(Op::Xor, a, b)?,
            "ANDN2" => (bdd.not(b).and_then(|nb| bdd.and(a, nb)))?,
            "ORN2" => (bdd.not(b).and_then(|nb| bdd.or(a, nb)))?,
            "ANDN1" => (bdd.not(a).and_then(|na| bdd.and(na, b)))?,
            "ORN1" => (bdd.not(a).and_then(|na| bdd.or(na, b)))?,
            "NAND" => (bdd.and(a, b).and_then(|x| bdd.not(x)))?,
            "NOR" => (bdd.or(a, b).and_then(|x| bdd.not(x)))?,
            "XNOR" => (bdd.apply(Op::Xor, a, b).and_then(|x| bdd.not(x)))?,
            _ => return Ok(None),
        }))
    }

    /// One instruction's effect on the masks.
    fn step(&mut self, st: &mut State, i: usize, block: usize) -> R<()> {
        let d = &self.insts[i];
        let (written, kill, writes_scc) = self.effects[i];
        let name = d.inst.name;
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f);
        let e = st.regs[EXEC].expect("EXEC");
        let mut set: Vec<(usize, B)> = Vec::new();
        let mut scc = None;
        if name.starts_with("V_") {
            if name.starts_with("V_CMPX") {
                let l = self.atom(Atom::Lane(d.pc), block)?;
                set.push((EXEC, self.bdd.and(l, e)?));
            } else if written != 0 && !matches!(name, "V_READLANE_B32" | "V_READFIRSTLANE_B32") {
                // compares and carries: the lane's result, 0 in inactive lanes
                let l = self.atom(Atom::Lane(d.pc), block)?;
                set.push((written.trailing_zeros() as usize, self.bdd.and(l, e)?));
            }
        } else if let Some(base) = name.strip_suffix(if self.wave64 { "_B64" } else { "_B32" }) {
            let (s0, s1, sd) = (fv("SSRC0").unwrap_or(0), fv("SSRC1").unwrap_or(0), fv("SDST").map(|v| v as usize));
            let op = base.trim_start_matches("S_");
            if let Some(op) = op.strip_suffix("_SAVEEXEC").or_else(|| op.strip_suffix("_WREXEC")) {
                let a = self.need(st, d, s0, block)?;
                let new = self.combine(op, a, e)?.ok_or("unknown saveexec")?;
                set.push((EXEC, new));
                if let Some(sd) = sd.filter(|&r| r < 128) {
                    set.push((sd, if base.ends_with("_WREXEC") { new } else { e }));
                }
                scc = Some((new, SccKind::Any));
            } else {
                let (mut a, mut b) = (self.mask_src(st, d, s0), self.mask_src(st, d, s1));
                // A bitwise scalar instruction can mix a tracked mask and an SGPR
                // holding a number. The lowering extracts each lane's bit of that
                // number. Its unknown bits must not turn the result's SCC uniform.
                if matches!(op, "AND" | "OR" | "XOR" | "ANDN2" | "ORN2" | "ANDN1" | "NAND" | "NOR" | "XNOR")
                    && (sd == Some(EXEC) || (s0 < 128 && a.is_some()) || (s1 < 128 && b.is_some()))
                {
                    a = Some(self.need(st, d, s0, block)?);
                    b = Some(self.need(st, d, s1, block)?);
                }
                let value = match (op, a, b) {
                    ("MOV", Some(a), _) => Some(a),
                    ("NOT", Some(a), _) => Some(self.bdd.not(a)?),
                    ("WQM", Some(a), _) => self.wqm(a),
                    ("CSELECT", Some(a), Some(b)) => match st.scc {
                        Some((u, SccKind::Uniform)) => {
                            let (ua, nu) = (self.bdd.and(u, a)?, self.bdd.not(u)?);
                            let nub = self.bdd.and(nu, b)?;
                            Some(self.bdd.or(ua, nub)?)
                        }
                        _ => None,
                    },
                    (_, Some(a), Some(b)) => self.combine(op, a, b)?,
                    _ => None,
                };
                if let (Some(v), Some(sd)) = (value, sd.filter(|&r| r < 128)) {
                    set.push((sd, v));
                    if !matches!(op, "MOV" | "CSELECT") {
                        scc = Some((v, SccKind::Any));
                    }
                }
            }
        }
        // mask compares (`s_cmp_lg_u64 exec, 0`): whether the lane's bits differ / agree
        let cmp = if self.wave64 { ("S_CMP_EQ_U64", "S_CMP_LG_U64") } else { ("S_CMP_EQ_U32", "S_CMP_LG_U32") };
        if name == cmp.0 || name == cmp.1 {
            let (s0, s1) = (fv("SSRC0").unwrap_or(255), fv("SSRC1").unwrap_or(255));
            if (s0 < 128 && st.regs[s0 as usize].is_some()) || (s1 < 128 && st.regs[s1 as usize].is_some()) {
                let a = self.need(st, d, s0, block)?;
                let b = self.need(st, d, s1, block)?;
                let x = self.bdd.apply(Op::Xor, a, b)?;
                scc = Some(if name == cmp.0 { (self.bdd.not(x)?, SccKind::All) } else { (x, SccKind::Any) });
            }
        }
        // whatever else the instruction writes is no longer a known mask
        for r in (0..128).filter(|&r| kill >> r & 1 != 0) {
            self.kill(st, r);
        }
        for (r, v) in set {
            self.kill(st, r);
            st.regs[r] = Some(v);
        }
        if st.regs[EXEC].is_none() {
            st.regs[EXEC] = Some(self.atom(Atom::Opaque(d.pc, EXEC), block)?);
        }
        if writes_scc {
            st.scc = match scc.filter(|_| !self.wave_branches) {
                Some(s) => Some(s),
                None => Some((self.atom(Atom::Scc(d.pc), block)?, SccKind::Uniform)),
            };
        }
        Ok(())
    }

    /// `s_wqm` of a mask: the lanes whose quad has a lane in it. Followed only where that is a
    /// mask the analysis knows: none or every lane, and the launch EXEC (`LaunchExec::Unknown`),
    /// since a lane that runs outside it is a helper lane of a quad with a lane in it.
    fn wqm(&mut self, a: B) -> Option<B> {
        if a <= TRUE {
            return Some(a);
        }
        let launch = *self.atoms.get(&Atom::Launch)?;
        (a == self.bdd.atom(launch)).then_some(TRUE)
    }

    fn kill(&self, st: &mut State, r: usize) {
        st.regs[r] = None;
        // wave64: the pair starting one register lower
        if self.wave64 && r > 0 {
            st.regs[r - 1] = None;
        }
    }

    /// Whether each active lane's bit `c` agrees with the hardware's wave-wide reduction `kind`
    /// (including bits in inactive lanes): then a lane deciding by its own bit decides as the
    /// whole wave does. Unknown (too large to build) is no.
    fn uniform_over(&mut self, reach: B, e: B, c: B, kind: SccKind) -> bool {
        if reach == FALSE || kind == SccKind::Uniform {
            return true;
        }
        let Some(re) = self.bdd.within(|bdd| bdd.and(reach, e)) else { return false };
        match kind {
            SccKind::Uniform => true,
            SccKind::Unknown => false,
            SccKind::Any => c == FALSE || (re != FALSE && self.bdd.implies(re, c)),
            SccKind::All => {
                c == TRUE
                    || (re != FALSE
                        && self.bdd.within(|bdd| bdd.not(c)).is_some_and(|nc| self.bdd.implies(re, nc)))
            }
        }
    }

    /// The unnegated condition, its wave-wide reduction kind, and whether the branch negates it.
    fn cond(&mut self, st: &State, d: &Decoded, block: usize) -> R<(B, SccKind, bool)> {
        let e = st.regs[EXEC].expect("EXEC");
        // All hardware lanes take the same edge. A wave decision carries no fact about
        // this lane's mask bit, even when the wave took the nonzero edge.
        if self.wave_branches {
            return Ok((self.atom(Atom::Scc(d.pc), block)?, SccKind::Uniform, false));
        }
        Ok(match d.inst.name {
            n @ ("S_CBRANCH_SCC0" | "S_CBRANCH_SCC1") => {
                let (c, kind) = match st.scc {
                    Some(s) => s,
                    None => (self.atom(Atom::Scc(d.pc), block)?, SccKind::Uniform),
                };
                (c, kind, n == "S_CBRANCH_SCC0")
            }
            n @ ("S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ") => {
                let v = match st.regs[VCC] {
                    Some(v) => v,
                    None => self.atom(Atom::Opaque(d.pc, VCC), block)?,
                };
                (v, SccKind::Any, n == "S_CBRANCH_VCCZ")
            }
            n @ ("S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ") => (e, SccKind::Any, n == "S_CBRANCH_EXECZ"),
            _ => (self.atom(Atom::Scc(d.pc), block)?, SccKind::Uniform, false),
        })
    }

    /// `cond` within `STEP_NODES` new nodes; past that, an unknown per-lane condition.
    fn cond_bounded(&mut self, st: &State, d: &Decoded, block: usize) -> R<(B, SccKind, bool)> {
        self.bdd.limit = (self.bdd.nodes.len() + STEP_NODES).min(MAX_NODES);
        let r = self.cond(st, d, block);
        self.bdd.limit = MAX_NODES;
        match r {
            Ok(c) => Ok(c),
            Err(_) => Ok((self.atom(Atom::Scc(d.pc), block)?, SccKind::Unknown, false)),
        }
    }

    /// Simulates a block from `st`: the facts and states on its outgoing edges.
    fn run_block(&mut self, b: usize, reach: B, mut st: State) -> R<Vec<(B, State)>> {
        let (first, end) = self.blocks[b];
        for i in first..end {
            self.step_bounded(&mut st, i, b)?;
        }
        let last = &self.insts[end - 1];
        let mut out = Vec::new();
        for k in 0..self.succ[b].len() {
            let fact = match self.succ[b][k].1 {
                Edge::Always => reach,
                edge => {
                    // a fact too large to build is left without the branch's condition (weaker)
                    let (raw_c, _, negate) = self.cond_bounded(&st, last, b)?;
                    let take_neg = (edge == Edge::Taken) == negate;
                    self.bdd
                        .within(|bdd| {
                            let c = if take_neg { bdd.not(raw_c)? } else { raw_c };
                            bdd.and(reach, c)
                        })
                        .unwrap_or(reach)
                }
            };
            out.push((fact, st.clone()));
        }
        Ok(out)
    }

    /// Block b's reach and masks from its predecessors' edges. Back edges add their facts
    /// with the loop's atoms dropped; their masks only count with `back_masks` (the check of a
    /// loop head's guess).
    fn join(&mut self, b: usize, outs: &[Vec<(B, State)>], forced: &HashSet<(usize, usize)>, back_masks: bool) -> R<(B, State)> {
        let mut reach = FALSE;
        let mut edges: Vec<(B, State)> = Vec::new();
        for &(p, k) in &self.preds[b].clone() {
            let Some((fact, st)) = outs[p].get(k) else { continue };
            let back = self.body[b].as_ref().is_some_and(|inside| inside[p]);
            let carried = if back {
                let (inside, born) = (self.body[b].as_ref().unwrap(), &self.born);
                self.bdd.within(|bdd| bdd.exists(*fact, &|v| inside[born[v as usize]], &mut HashMap::new())).unwrap_or(TRUE)
            } else {
                *fact
            };
            reach = self.bdd.within(|bdd| bdd.or(reach, carried)).unwrap_or(TRUE);
            if *fact != FALSE && (!back || back_masks) {
                edges.push((*fact, st.clone()));
            }
        }
        let mut st = State { regs: vec![None; 128], scc: None };
        st.regs[EXEC] = Some(FALSE);
        if edges.is_empty() {
            return Ok((reach, st));
        }
        // an SCC a mask op set on any incoming edge, back edges included (their masks may not
        // count here, but a lane's own SCC carried around a loop is still a lane's: thehollowedone,
        // #1745)
        let mut scc_kind = SccKind::Uniform;
        for &(p, k) in &self.preds[b] {
            if let Some((f, s)) = outs[p].get(k) {
                if *f != FALSE {
                    if let Some((_, knd)) = s.scc {
                        scc_kind = match (scc_kind, knd) {
                            (SccKind::Uniform, k) | (k, SccKind::Uniform) => k,
                            (a, b) if a == b => a,
                            _ => SccKind::Unknown,
                        };
                    }
                }
            }
        }
        for slot in 0..=SCC {
            let get = |s: &State| if slot == SCC { s.scc.map(|x| x.0) } else { s.regs[slot] };
            let v = if forced.contains(&(b, slot)) {
                Some(self.atom(Atom::Join(b, slot), b)?)
            } else if edges.iter().any(|(_, s)| get(s).is_none()) {
                None
            } else {
                let first = get(&edges[0].1).unwrap();
                Some(if edges.iter().all(|(_, s)| get(s) == Some(first)) {
                    first
                } else if slot == SCC && scc_kind != SccKind::Uniform {
                    self.atom(Atom::Join(b, slot), b)?
                } else if edges.iter().all(|(fact, s)| self.bdd.implies(*fact, get(s).unwrap())) {
                    TRUE
                } else if edges.iter().all(|(fact, s)| self.bdd.disjoint(*fact, get(s).unwrap(), &mut HashSet::new())) {
                    FALSE
                } else {
                    self.atom(Atom::Join(b, slot), b)?
                })
            };
            if slot == SCC {
                // unknown but a lane's on some edge: an unknown lane condition, not a uniform one
                let v = match v {
                    None if scc_kind != SccKind::Uniform => Some(self.atom(Atom::Join(b, SCC), b)?),
                    v => v,
                };
                st.scc = v.map(|v| (v, scc_kind));
            } else {
                st.regs[slot] = v;
            }
        }
        if st.regs[EXEC].is_none() {
            st.regs[EXEC] = Some(self.atom(Atom::Join(b, EXEC), b)?);
        }
        Ok((reach, st))
    }

    fn analyze(mut self) -> R<Summary> {
        let n = self.blocks.len();
        let mut entry = State { regs: vec![None; 128], scc: None };
        entry.regs[EXEC] = Some(match self.launch {
            LaunchExec::All => TRUE,
            LaunchExec::Unknown => self.atom(Atom::Launch, 0)?,
        });
        let mut forced = HashSet::new();
        let mut reach = vec![FALSE; n];
        let mut ins: Vec<Option<State>> = vec![None; n];
        let mut outs: Vec<Vec<(B, State)>> = vec![Vec::new(); n];
        // blocks to recompute: every block first, then those an incoming edge changed for. Any
        // fixpoint of the facts is sound (it is weaker than the least one), so a wrong loop-head
        // guess only makes that head recompute instead of restarting from nothing.
        let order = self.order.clone();
        let mut dirty = vec![false; n];
        for &b in &order {
            dirty[b] = true;
        }
        let heads: Vec<usize> = order.iter().copied().filter(|&h| self.body[h].is_some()).collect();
        let mut rounds = 0;
        loop {
            while order.iter().any(|&b| dirty[b]) {
                rounds += 1;
                if rounds > MAX_ROUNDS {
                    return Err("no fixpoint");
                }
                for &b in &order {
                    if !std::mem::take(&mut dirty[b]) {
                        continue;
                    }
                    let (r, st) = if b == 0 { (TRUE, entry.clone()) } else { self.join(b, &outs, &forced, false)? };
                    if ins[b].as_ref() == Some(&st) && reach[b] == r {
                        continue;
                    }
                    reach[b] = r;
                    ins[b] = Some(st.clone());
                    let out = self.run_block(b, r, st)?;
                    if out != outs[b] {
                        for &(s, _) in &self.succ[b] {
                            dirty[s] = true;
                        }
                        outs[b] = out;
                    }
                }
            }
            // loop heads: does the guess from the forward edges hold on the back edges too?
            let mut wrong = false;
            for &h in &heads {
                let (_, full) = self.join(h, &outs, &forced, true)?;
                let guess = ins[h].as_ref().unwrap();
                for slot in 0..=SCC {
                    let changed = if slot == SCC {
                        guess.scc != full.scc
                    } else {
                        guess.regs[slot] != full.regs[slot]
                    };
                    if changed && forced.insert((h, slot)) {
                        wrong = true;
                        dirty[h] = true;
                    }
                }
            }
            if !wrong {
                break;
            }
        }
        let mut sum = Summary::default();
        let mut implied: HashMap<(B, B), bool> = HashMap::new();
        let mut lane_branch = vec![false; self.blocks.len()];
        for &b in &self.order.clone() {
            let mut st = ins[b].clone().unwrap();
            let (first, end) = self.blocks[b];
            for i in first..end {
                let d = &self.insts[i];
                if is_vector(d) {
                    sum.vector += 1;
                    let key = (reach[b], st.regs[EXEC].unwrap());
                    let covered = match implied.get(&key) {
                        Some(&c) => c,
                        None => *implied.entry(key).or_insert(self.bdd.implies(key.0, key.1)),
                    };
                    if !covered {
                        sum.guarded.push(d.pc);
                    }
                }
                // an SCC a mask op set, read by a scalar or vector instruction (implicit or
                // SRC_SCC = 253): the console reads the whole wave's, each lane here its own bit,
                // the same only where it agrees wave-wide
                if reads_scc(d) {
                    if let Some((c, kind)) = st.scc {
                        if kind != SccKind::Uniform {
                            let e = st.regs[EXEC].unwrap();
                            if !self.uniform_over(reach[b], e, c, kind) {
                                return Err("a scalar read of the SCC of a mask op, with lanes in EXEC without its bit");
                            }
                        }
                    }
                }
                if d.inst.name == if self.wave64 { "S_WQM_B64" } else { "S_WQM_B32" } {
                    let s0 = rdna2::field_value(d.enc, &d.words, "SSRC0").unwrap_or(0);
                    if self.mask_src(&st, d, s0).and_then(|a| self.wqm(a)) == Some(TRUE) {
                        sum.wqm_all.push(d.pc);
                    }
                }
                self.step_bounded(&mut st, i, b)?;
            }
            let last = &self.insts[end - 1];
            if self.succ[b].iter().any(|s| s.1 != Edge::Always) {
                let (raw_c, kind, _) = self.cond_bounded(&st, last, b)?;
                let lane = kind != SccKind::Uniform;
                // a lane branch is each lane's only where the condition agrees with the whole
                // wave's reduction (`s_and_saveexec` then `s_cbranch_execz`); `s_cbranch_vccz` on a
                // mask with bits in inactive lanes or lacking active lanes takes the whole wave's
                // way on the console (Kalmarv, #1745)
                if lane && !self.uniform_over(reach[b], st.regs[EXEC].unwrap(), raw_c, kind) {
                    return Err("a branch on whether any lane of a mask is set, with lanes in EXEC without its bit");
                }
                if lane {
                    sum.lane_branches += 1;
                } else {
                    sum.uniform_branches += 1;
                }
                lane_branch[b] = lane;
            }
        }
        sum.blocks = (0..self.blocks.len())
            .map(|b| Block {
                insts: self.blocks[b].0..self.blocks[b].1,
                succ: self.succ[b].clone(),
                preds: self.preds[b].iter().map(|p| p.0).collect(),
                lane_branch: lane_branch[b],
            })
            .collect();
        sum.order = self.order.clone();
        Ok(sum)
    }
}

/// Analyses a decoded program (far jumps resolved) that starts with every lane in EXEC.
pub fn analyze(insts: &[Decoded], wave64: bool) -> R<Summary> {
    analyze_launch(insts, wave64, LaunchExec::All)
}

/// Analyses a decoded program (far jumps resolved) that starts with EXEC as `launch` says.
pub fn analyze_launch(insts: &[Decoded], wave64: bool, launch: LaunchExec) -> R<Summary> {
    analyze_branches(insts, wave64, launch, false)
}

pub(super) fn analyze_branches(insts: &[Decoded], wave64: bool, launch: LaunchExec, wave_branches: bool) -> R<Summary> {
    if insts.is_empty() {
        return Ok(Summary::default());
    }
    let mut lanes = Lanes::new(insts, wave64, launch)?;
    lanes.wave_branches = wave_branches;
    lanes.analyze()
}

/// Decodes and analyses a program.
pub fn analyze_code(code: &[u8], wave64: bool) -> R<Summary> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else { return Err("cannot decode") };
        off += d.len;
        insts.push(d);
    }
    rdna2::resolve_far_jumps(&mut insts);
    analyze(&insts, wave64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: u64 = 256;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)]);
    const ENDPGM: I = ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]);
    const CMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]);
    const MOV2: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 129)]);
    const MOV3: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 130)]);
    const RESTORE: I = ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]);

    /// Every instruction here is one dword: a branch's SIMM16 is (target − index − 1).
    fn run(lines: &[I]) -> R<Summary> {
        let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        assert_eq!(bytes.len(), lines.len() * 4);
        analyze_code(&bytes, true)
    }

    #[test]
    fn implication_tests_agree_with_the_built_formulas() {
        let mut bdd = Bdd::new();
        let x: Vec<B> = (0..4).map(|v| bdd.mk(v, FALSE, TRUE).unwrap()).collect();
        let mut fs = vec![FALSE, TRUE];
        for &a in &x {
            for &b in &x {
                let (and, or, xor) = (bdd.and(a, b).unwrap(), bdd.or(a, b).unwrap(), bdd.apply(Op::Xor, a, b).unwrap());
                let na = bdd.not(a).unwrap();
                fs.extend([a, and, or, xor, na]);
            }
        }
        for &a in &fs {
            for &b in &fs {
                let nb = bdd.not(b).unwrap();
                let (built_imp, built_dis) = (bdd.and(a, nb).unwrap() == FALSE, bdd.and(a, b).unwrap() == FALSE);
                assert_eq!(bdd.implies(a, b), built_imp, "{a} → {b}");
                assert_eq!(bdd.disjoint(a, b, &mut HashSet::new()), built_dis, "{a} ∧ {b}");
            }
        }
    }

    #[test]
    fn a_skipped_if_needs_no_guard() {
        // v_cmp; s_and_saveexec s[0:1], vcc; s_cbranch_execz L; v_mov; L: exec |= s[0:1]; v_mov
        let s = run(&[
            CMP,
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]),
            MOV2,
            RESTORE,
            MOV3,
            ENDPGM,
        ])
        .unwrap();
        assert_eq!((s.vector, s.guarded.len(), s.lane_branches, s.uniform_branches), (2, 0, 1, 0));
        // the CFG the lowering walks: entry (to the skip), the v_mov, the join
        let b = &s.blocks;
        assert_eq!(b.iter().map(|b| b.insts.clone()).collect::<Vec<_>>(), vec![0..3, 3..4, 4..7]);
        assert_eq!(b[0].succ, vec![(2, Edge::Taken), (1, Edge::NotTaken)]);
        assert_eq!((b[1].succ.clone(), b[2].preds.clone()), (vec![(2, Edge::Always)], vec![0, 1]));
        assert_eq!(b.iter().map(|b| b.lane_branch).collect::<Vec<_>>(), vec![true, false, false]);
        assert_eq!(s.order, vec![0, 1, 2]);
    }

    #[test]
    fn an_if_without_its_skip_is_guarded() {
        let s = run(&[CMP, ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]), MOV2, RESTORE, MOV3, ENDPGM]).unwrap();
        assert_eq!(s.vector, 2);
        assert_eq!(s.guarded, vec![8], "only the v_mov under the narrowed EXEC");
    }

    #[test]
    fn lanes_leaving_a_loop_keep_it_unguarded() {
        // s_mov s[0:1], exec; L: v_cmp; v_add; exec &= ~vcc; s_cbranch_execnz L; exec |= s[0:1]
        let s = run(&[
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
            CMP,
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 5)]),
            ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
            ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
            RESTORE,
            MOV3,
            ENDPGM,
        ])
        .unwrap();
        assert_eq!((s.vector, s.guarded.len(), s.lane_branches), (2, 0, 1));
    }

    #[test]
    fn a_loop_that_narrows_exec_is_found_on_its_back_edge() {
        // s4 = 0; L: v_mov; v_cmp; exec &= vcc; s4 += 1; s_cmp_lt s4, 10; s_cbranch_scc1 L; v_mov
        // – the head's guess (all lanes) is wrong once the back edge is seen
        let s = run(&[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 128)]),
            MOV2,
            CMP,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)]),
            ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 138)]),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFA)]),
            MOV3,
            ENDPGM,
        ])
        .unwrap();
        assert_eq!(s.guarded, vec![4, 28]);
        assert_eq!((s.lane_branches, s.uniform_branches), (0, 1));
    }

    #[test]
    fn a_vccz_branch_is_a_lane_branch_only_where_vcc_is_the_exec() {
        // v_cmp; s_cbranch_vccz L: the console skips only when no lane in EXEC has the bit, so
        // lanes in EXEC without it run the v_mov too; deciding per lane would skip it: refused
        let err = run(&[CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)]), MOV2, MOV3, ENDPGM]).err();
        assert_eq!(err, Some("a branch on whether any lane of a mask is set, with lanes in EXEC without its bit"));
        // v_cmp; s_and_saveexec s0, vcc; s_cbranch_vccz L: VCC ∧ EXEC is the new EXEC, the same
        // as each lane's bit for every lane in it: a lane branch
        let save: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]);
        let s = run(&[CMP, save, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)]), MOV2, MOV3, ENDPGM]).unwrap();
        assert_eq!((s.vector, s.guarded.len(), s.lane_branches), (2, 1, 1));
    }

    #[test]
    fn unsupported_control_flow_is_refused() {
        // a loop entered at two places
        let two_entries = run(&[
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]),
            MOV2,
            MOV3,
            ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0xFFFD)]),
            ENDPGM,
        ]);
        assert_eq!(two_entries.unwrap_err(), "irreducible control flow");
        assert_eq!(run(&[("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 0)]), ENDPGM]).unwrap_err(), "indirect jump");
        assert_eq!(run(&[MOV2, ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFE)]), ENDPGM]).unwrap_err(), "branch to the entry");
    }

    fn run_launch(lines: &[I], launch: LaunchExec) -> R<Summary> {
        let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        let mut insts = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            insts.push(d);
        }
        rdna2::resolve_far_jumps(&mut insts);
        analyze_launch(&insts, true, launch)
    }

    #[test]
    fn a_lane_outside_exec_at_launch_keeps_its_guard_until_exec_is_written() {
        // v_mov (a helper lane must not write); exec = -1; v_mov (every lane writes)
        let p = [MOV2, ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]), MOV3, ENDPGM];
        let all = run_launch(&p, LaunchExec::All).unwrap();
        assert_eq!((all.vector, all.guarded.len()), (2, 0));
        let unknown = run_launch(&p, LaunchExec::Unknown).unwrap();
        assert_eq!(unknown.guarded, vec![0], "only the v_mov before EXEC is written");
    }

    #[test]
    fn a_lane_branch_still_proves_exec_when_the_launch_exec_is_unknown() {
        // as `a_skipped_if_needs_no_guard`: inside the if only lanes with their bit run, but the
        // restored EXEC after the join is the launch EXEC again
        let p = [
            CMP,
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]),
            MOV2,
            RESTORE,
            MOV3,
            ENDPGM,
        ];
        assert_eq!(run_launch(&p, LaunchExec::All).unwrap().guarded.len(), 0);
        let s = run_launch(&p, LaunchExec::Unknown).unwrap();
        assert_eq!((s.vector, s.guarded.clone(), s.lane_branches), (2, vec![20], 1));
    }

    #[test]
    fn whole_quad_mode_of_the_launch_exec_is_every_lane() {
        // s[0:1] = exec; s_wqm exec, exec; v_mov (helper lanes too); exec &= s[0:1]; v_mov
        let wqm: I = ("S_WQM_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)]);
        let p: [I; 6] = [
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
            wqm,
            MOV2,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            MOV3,
            ENDPGM,
        ];
        let s = run_launch(&p, LaunchExec::Unknown).unwrap();
        assert_eq!(s.guarded, vec![16], "only the v_mov after EXEC is narrowed to the launch EXEC again");
        assert_eq!(s.wqm_all, vec![4]);
        assert_eq!(run_launch(&p, LaunchExec::All).unwrap().guarded.len(), 0);
        // the quads of any other mask are not followed
        let s = run_launch(&[CMP, ("S_WQM_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 106)]), MOV2, ENDPGM], LaunchExec::All).unwrap();
        assert_eq!((s.guarded, s.wqm_all), (vec![8], vec![]));
    }

    #[test]
    fn whole_wave_scc_accounts_for_inactive_lanes() {
        let save_and: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]);
        let save_andn2: I = ("S_ANDN2_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]);
        let cmp_exec_all: I = ("S_CMP_EQ_U64", "ENC_SOPC", &[("SSRC0", 126), ("SSRC1", 193)]);
        let cmp_exec_zero: I = ("S_CMP_EQ_U64", "ENC_SOPC", &[("SSRC0", 126), ("SSRC1", 128)]);
        let not_exec: I = ("S_NOT_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 126)]);
        let cmp_vcc_eq0: I = ("S_CMP_EQ_U64", "ENC_SOPC", &[("SSRC0", 106), ("SSRC1", 128)]);
        let cmp_vcc_lg0: I = ("S_CMP_LG_U64", "ENC_SOPC", &[("SSRC0", 106), ("SSRC1", 128)]);
        let br_scc1: I = ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]);
        let br_scc0: I = ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)]);

        // partial EXEC != -1: hardware SCC = 0, active lanes see 1
        assert!(run(&[CMP, save_and, cmp_exec_all, br_scc1, MOV2, ENDPGM]).is_err());
        // full EXEC == -1 wave-wide: hardware SCC = 1, every lane sees 1
        assert!(run(&[cmp_exec_all, br_scc1, MOV2, ENDPGM]).is_ok());
        // !EXEC with partial EXEC is nonzero in inactive lanes: hardware SCC = 1, active lanes see 0
        assert!(run(&[CMP, save_and, not_exec, br_scc0, MOV2, ENDPGM]).is_err());
        // !EXEC with full EXEC is 0 wave-wide: hardware SCC = 0, every lane sees 0
        assert!(run(&[not_exec, br_scc0, MOV2, ENDPGM]).is_ok());
        // VCC has bits in inactive lanes after s_andn2_saveexec: active lanes have VCC == 0, wave has VCC != 0
        assert!(run(&[CMP, save_andn2, cmp_vcc_eq0, br_scc1, MOV2, ENDPGM]).is_err());
        assert!(run(&[CMP, save_andn2, cmp_vcc_lg0, br_scc1, MOV2, ENDPGM]).is_err());
        // EXEC == 0 check on narrowed EXEC matches per-lane (!EXEC)
        assert!(run(&[CMP, save_and, cmp_exec_zero, br_scc1, MOV2, ENDPGM]).is_ok());
    }

    #[test]
    fn vccz_branch_checks_bits_outside_exec() {
        let save_andn2: I = ("S_ANDN2_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]);
        // active lanes have VCC == 0, but inactive lanes have VCC == 1: hardware sees VCC != 0
        assert!(run(&[CMP, save_andn2, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)]), MOV2, ENDPGM]).is_err());
        assert!(run(&[CMP, save_andn2, ("S_CBRANCH_VCCNZ", "ENC_SOPP", &[("SIMM16", 1)]), MOV2, ENDPGM]).is_err());
        // VCC == 0 wave-wide: sound
        let zero_vcc: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)]);
        assert!(run(&[zero_vcc, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)]), MOV2, ENDPGM]).is_ok());
    }

    #[test]
    fn explicit_src_scc_reads_are_validated() {
        let and_mask: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 106), ("SSRC1", 126)]);
        let save_and: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]);
        // explicit SRC_SCC (253) on SALU and VALU reads of a non-uniform mask SCC must be refused
        let smov_scc: I = ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 253)]);
        let vmov_scc: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 253)]);
        assert_eq!(run(&[CMP, and_mask, smov_scc, ENDPGM]).err(), Some("a scalar read of the SCC of a mask op, with lanes in EXEC without its bit"));
        assert_eq!(run(&[CMP, and_mask, vmov_scc, ENDPGM]).err(), Some("a scalar read of the SCC of a mask op, with lanes in EXEC without its bit"));
        // when SCC agrees across every active lane (s_and_saveexec), explicit SRC_SCC is sound
        assert!(run(&[CMP, save_and, smov_scc, ENDPGM]).is_ok());
    }
}
