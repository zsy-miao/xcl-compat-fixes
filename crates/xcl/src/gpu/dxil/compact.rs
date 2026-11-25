//! The compact form: each distinct block of a program lowered once, in a loop over a table.

use super::cfg::{Cfg, Term};
use super::lower::{ext_encoding, LResult, Lw, VAR_BUDGET};
use super::lower_mem::VAR_TMP;
use super::module::{Block, Pred, Value};
use std::collections::{BTreeSet, HashMap};

pub struct Plan {
    /// the reachable blocks in address order: the loop's block numbers
    pub order: Vec<usize>,
    /// the class of each
    pub class: Vec<u32>,
    /// a block of each class (a block number)
    pub rep: Vec<usize>,
    /// where each block goes: (branch taken or jump, branch not taken), as block numbers
    pub next: Vec<(u32, u32)>,
    /// the distinct pairs of offsets from a block to its two successors, sorted
    pub steps: Vec<(u32, u32)>,
}

/// Most 32-bit words of tables a program may need to take the form by itself.
const MAX_TABLE_WORDS: usize = 8192;

impl Plan {
    /// Bits of a table entry: the class, and the index of the pair of offsets.
    fn widths(&self) -> (u32, u32) {
        (bits(self.rep.len()), bits(self.steps.len()))
    }

    /// 32-bit words of the two tables.
    pub fn words(&self) -> usize {
        let (cb, pb) = self.widths();
        (self.order.len() * (cb + pb) as usize).div_ceil(32) + 1 + self.steps.len() * 2
    }
}

/// Whether a program takes the form by itself: large, mostly repeated blocks, small tables.
pub fn wanted(cfg: &Cfg, p: &Plan) -> bool {
    let insts = |n: usize| cfg.blocks[p.order[n]].insts.len();
    let all: usize = (0..p.order.len()).map(insts).sum();
    let once: usize = p.rep.iter().map(|&n| insts(n)).sum();
    large(cfg) && once * 10 < all * 3 && p.words() <= MAX_TABLE_WORDS
}

/// Whether a program is large enough for `wanted` to be asked.
pub fn large(cfg: &Cfg) -> bool {
    cfg.rpo.iter().map(|&b| cfg.blocks[b].insts.len()).sum::<usize>() >= 20_000
}

pub const END: u32 = u32::MAX;

/// Sorts the reachable blocks into classes the lowering would emit the same for.
pub fn plan(cfg: &Cfg, lw: &Lw) -> Option<Plan> {
    let mut order = cfg.rpo.clone();
    order.sort_by_key(|&b| (cfg.blocks[b].pc, b));
    let mut number = vec![END; cfg.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        number[b] = n as u32;
    }
    let mut ids: HashMap<(Vec<u64>, &'static str), u32> = HashMap::new();
    let (mut class, mut rep, mut next) = (Vec::new(), Vec::new(), Vec::new());
    for (n, &b) in order.iter().enumerate() {
        let blk = &cfg.blocks[b];
        let mut key = vec![cfg.headers.contains(&b) as u64];
        for &i in &blk.insts {
            let d = &cfg.insts[i];
            key.push(d.words.len() as u64);
            key.extend(d.words.iter().map(|&w| w as u64));
            key.push(d.literal.map_or(u64::MAX, |l| l as u64));
            if let Some(s) = lw.static_samplers.get(&d.pc) {
                key.extend(s.iter().map(|&w| w as u64));
            }
            // what `Lw::write_mask` asks of this address, for any register a mask can go to
            if let Some(&l) = lw.whole_live.get(&d.pc) {
                let sdwa = ext_encoding(d).and_then(|e| crate::gpu::rdna2::field_value(e, &d.words, "SDST"));
                let to = crate::gpu::rdna2::sgpr_uses(d).1.into_iter().chain([106]).chain(sdwa);
                key.extend(to.filter(|&r| r < 126).map(|r| (l >> r) as u64 & 3));
            }
            if d.inst.name == "S_GETPC_B64" {
                key.push(d.pc);
            }
        }
        let (end, to) = match blk.term {
            Term::End => ("end", (END, END)),
            Term::Jump(t) => ("jump", (number[t], number[t])),
            Term::Cond(i, t, f) => (cfg.insts[i].inst.name, (number[t], number[f])),
        };
        let fresh = ids.len() as u32;
        let c = *ids.entry((key, end)).or_insert(fresh);
        if c == fresh {
            rep.push(n);
        }
        class.push(c);
        next.push(to);
    }
    let steps: BTreeSet<(u32, u32)> = next.iter().enumerate().map(|(b, &to)| offsets(b, to)).collect();
    let p = Plan { order, class, rep, next, steps: steps.into_iter().collect() };
    let (cb, pb) = p.widths();
    (cb + pb <= 32).then_some(p)
}

/// A block's successors as offsets from its own number.
fn offsets(b: usize, (t, f): (u32, u32)) -> (u32, u32) {
    if t == END { (0, 0) } else { (t.wrapping_sub(b as u32), f.wrapping_sub(b as u32)) }
}

fn bits(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).max(1)
}

/// A constant table of 32-bit words.
fn table(lw: &mut Lw, name: &str, words: &[u32]) -> Value {
    let vals: Vec<Value> = words.iter().map(|&w| lw.c(w)).collect();
    let i32 = lw.s.m.i32();
    let at = lw.s.m.array(words.len() as u64, i32);
    let init = lw.s.m.aggregate(at, &vals);
    lw.s.m.global(name, at, 0, Some(init), true, 4)
}

fn word(lw: &mut Lw, table: Value, i: Value) -> Value {
    let z = lw.c(0);
    let p = lw.s.m.gep(table, &[z, i]);
    lw.s.m.load(p, 4)
}

fn slot(lw: &mut Lw, var: u32) -> Value {
    let Lw { s, init, .. } = lw;
    init.slot(s, var)
}

/// Lowers the program as one loop from `head` to `exit` that switches to each block's class.
pub fn emit(lw: &mut Lw, cfg: &Cfg, p: &Plan, head: Block, exit: Block) -> LResult<()> {
    let n = p.order.len();
    let (cb, pb) = p.widths();
    let width = cb + pb;
    let mut packed = vec![0u32; p.words() - p.steps.len() * 2];
    for (b, &to) in p.next.iter().enumerate() {
        let pair = p.steps.binary_search(&offsets(b, to)).expect("a pair of the plan") as u32;
        let e = (p.class[b] | pair << cb) as u64;
        let at = b * width as usize;
        packed[at / 32] |= (e << (at % 32)) as u32;
        packed[at / 32 + 1] |= (e << (at % 32) >> 32) as u32;
    }
    let steps: Vec<u32> = p.steps.iter().flat_map(|&(t, f)| [t, f]).collect();
    let blocks = table(lw, "xcl_blocks", &packed);
    let steps = table(lw, "xcl_steps", &steps);

    let (at, to) = (lw.tmp_var(), lw.tmp_var());
    let (at, to, budget) = (slot(lw, at), slot(lw, to), slot(lw, VAR_BUDGET));
    lw.s.m.set_block(head);
    let b = lw.s.m.load(at, 4);
    let w = lw.c(width);
    let bit = lw.mul(b, w);
    let i = lw.lshrc(bit, 5);
    let sh = lw.andc(bit, 31);
    let lo = word(lw, blocks, i);
    let i1 = lw.addc(i, 1);
    let hi = word(lw, blocks, i1);
    let lo = lw.lshr(lo, sh);
    let hi = lw.shlc(hi, 1);
    let k31 = lw.c(31);
    let up = lw.sub(k31, sh);
    let hi = lw.shl(hi, up);
    let e = lw.or(lo, hi);
    let class = lw.andc(e, (1u32 << cb) - 1);
    let pair = lw.lshrc(e, cb);
    let pair = lw.andc(pair, (1u32 << pb) - 1);
    let pi = lw.shlc(pair, 1);
    let step_t = word(lw, steps, pi);
    let pi1 = lw.addc(pi, 1);
    let step_f = word(lw, steps, pi1);
    let (taken, not_taken) = (lw.add(b, step_t), lw.add(b, step_f));
    let cases: Vec<Block> = p.rep.iter().map(|_| lw.block(Some(0))).collect();
    let latch = lw.block(Some(0));
    let arms: Vec<(Value, Block)> = cases.iter().enumerate().skip(1).map(|(k, &c)| (lw.c(k as u32), c)).collect();
    lw.s.m.switch(class, cases[0], &arms);
    for (&c, &r) in cases.iter().zip(&p.rep) {
        lw.s.m.set_block(c);
        lw.init.mem = Some(c);
        lw.ssa.begin_log();
        let blk = &cfg.blocks[p.order[r]];
        if cfg.headers.contains(&p.order[r]) {
            // a loop head spends the loop budget
            let left = lw.s.m.load(budget, 4);
            let none = lw.cmpc(Pred::Eq, left, 0);
            let (out, body) = (lw.block(Some(1)), lw.block(Some(1)));
            lw.cond_br(none, out, body);
            lw.s.m.set_block(out);
            let end = lw.c(END);
            lw.s.m.store(to, end, 4);
            lw.s.m.br(latch);
            lw.s.m.set_block(body);
            let one = lw.c(1);
            let l = lw.sub(left, one);
            lw.s.m.store(budget, l, 4);
        }
        for &i in &blk.insts {
            lw.inst(&cfg.insts[i])?;
        }
        let next = match blk.term {
            Term::End => lw.c(END),
            Term::Jump(_) => taken,
            Term::Cond(i, ..) => {
                let c = lw.branch_cond(&cfg.insts[i])?;
                lw.sel(c, taken, not_taken)
            }
        };
        for var in lw.ssa.end_log().into_iter().filter(|&v| v < VAR_TMP) {
            let (v, p) = (lw.rd(var), slot(lw, var));
            lw.s.m.store(p, v, 4);
        }
        lw.s.m.store(to, next, 4);
        lw.s.m.br(latch);
    }
    // the epilogue's reads load here
    lw.init.mem = Some(latch);
    lw.s.m.set_block(latch);
    let next = lw.s.m.load(to, 4);
    let stop = lw.cmpc(Pred::UGe, next, n as u32);
    let cont = lw.block(Some(1));
    lw.cond_br(stop, exit, cont);
    lw.s.m.set_block(cont);
    lw.s.m.store(at, next, 4);
    lw.br(head);
    lw.seal(head);
    Ok(())
}
