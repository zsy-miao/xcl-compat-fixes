//! Basic blocks of a decoded RDNA2 program, their edges and a reverse
//! post-order (the emission order: every block after its dominators).

use std::collections::{BTreeMap, BTreeSet};

use crate::gpu::rdna2::{self, Decoded};

/// How a block ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// falls through / jumps to the block (index)
    Jump(usize),
    /// conditional branch: (instruction index of the branch, taken, not taken)
    Cond(usize, usize, usize),
    /// s_endpgm or the end of the code
    End,
}

pub struct CfgBlock {
    /// byte offset of the block's first instruction
    pub pc: u64,
    /// instruction indices (the terminating branch excluded for `Cond`/`Jump` by branch)
    pub insts: Vec<usize>,
    pub term: Term,
    pub preds: Vec<usize>,
}

pub struct Cfg {
    pub insts: Vec<Decoded>,
    pub blocks: Vec<CfgBlock>,
    /// reachable blocks in reverse post-order (blocks[rpo[0]] is the entry)
    pub rpo: Vec<usize>,
    /// targets of back edges (loop headers)
    pub headers: BTreeSet<usize>,
}

fn is_end(name: &str) -> bool {
    matches!(name, "S_ENDPGM" | "S_ENDPGM_SAVED" | "S_ENDPGM_ORDERED_PS_DONE")
}

impl Cfg {
    pub fn build(code: &[u8]) -> Result<Cfg, String> {
        Self::build_with(code, None)
    }

    /// Geometry mode: the ES's `s_setpc_b64` continues at `gs_start` (the GS code) when
    /// that is past the start, else ends the program (`recomp::Mode::Geometry`).
    pub fn build_geometry(code: &[u8], gs_start: Option<u64>) -> Result<Cfg, String> {
        Self::build_with(code, Some(gs_start.filter(|&s| s > 0)))
    }

    fn build_with(code: &[u8], setpc: Option<Option<u64>>) -> Result<Cfg, String> {
        let mut insts: Vec<Decoded> = Vec::new();
        let mut off = 0usize;
        while off + 4 <= code.len() {
            let d = rdna2::decode(&code[off..], off as u64).map_err(|e| format!("+{off:#x}: cannot decode ({e:?})"))?;
            off += d.len;
            insts.push(d);
        }
        rdna2::resolve_far_jumps(&mut insts);
        let index: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
        let mut leaders: BTreeSet<usize> = BTreeSet::from([0]);
        let is_setpc = |d: &Decoded| setpc.is_some() && d.inst.name == "S_SETPC_B64" && d.far_target.is_none();
        let setpc_target = match setpc {
            Some(Some(t)) => Some(*index.get(&t).ok_or_else(|| format!("GS start {t:#x} is not the start of an instruction"))?),
            _ => None,
        };
        if let Some(t) = setpc_target {
            leaders.insert(t);
        }
        for (i, d) in insts.iter().enumerate() {
            if is_setpc(d) {
                leaders.insert(i + 1);
                continue;
            }
            if let Some(t) = rdna2::branch_target(d) {
                let ti = *index.get(&t).ok_or_else(|| format!("+{:#x}: branch to {t:#x}, which is not the start of an instruction", d.pc))?;
                leaders.insert(ti);
                leaders.insert(i + 1);
            } else if d.inst.flags & rdna2::F_TERMINATOR != 0 || is_end(d.inst.name) {
                leaders.insert(i + 1);
            }
        }
        let starts: Vec<usize> = leaders.into_iter().filter(|&l| l < insts.len()).collect();
        let block_of: BTreeMap<usize, usize> = starts.iter().enumerate().map(|(b, &i)| (i, b)).collect();
        let mut blocks = Vec::new();
        for (b, &s) in starts.iter().enumerate() {
            let e = starts.get(b + 1).copied().unwrap_or(insts.len());
            let mut list = Vec::new();
            let mut term = if e < insts.len() { Term::Jump(b + 1) } else { Term::End };
            for i in s..e {
                let d = &insts[i];
                if is_end(d.inst.name) {
                    term = Term::End;
                    break;
                }
                if is_setpc(d) {
                    term = match setpc_target {
                        Some(t) => Term::Jump(block_of[&t]),
                        None => Term::End,
                    };
                    break;
                }
                if let Some(t) = rdna2::branch_target(d) {
                    let tb = block_of[&index[&t]];
                    let name = d.inst.name;
                    term = if name == "S_BRANCH" || d.far_target.is_some() {
                        Term::Jump(tb)
                    } else {
                        let next = if e < insts.len() { b + 1 } else { usize::MAX };
                        if next == usize::MAX {
                            return Err(format!("+{:#x}: conditional branch at the end of the code", d.pc));
                        }
                        Term::Cond(i, tb, next)
                    };
                    break;
                }
                list.push(i);
            }
            blocks.push(CfgBlock { pc: insts[s].pc, insts: list, term, preds: Vec::new() });
        }
        // a conditional branch whose both edges meet is a jump
        for b in &mut blocks {
            if let Term::Cond(_, t, f) = b.term {
                if t == f {
                    b.term = Term::Jump(t);
                }
            }
        }
        // DXIL requires reducible control flow (AMD's compiler faults on an irreducible
        // CFG; the console compiler emits loops entered in the middle): split nodes until
        // every retreating edge is a back edge to a dominator
        let (post, headers) = make_reducible(&mut blocks)?;
        let succ = succ_of;
        let reach: BTreeSet<usize> = post.iter().copied().collect();
        for &b in &post {
            for t in succ(&blocks[b]) {
                if !blocks[t].preds.contains(&b) {
                    blocks[t].preds.push(b);
                }
            }
        }
        let _ = reach;
        Ok(Cfg { insts, blocks, rpo: post, headers })
    }
}

/// Makes the CFG reducible by splitting: for an irreducible edge's innermost multi-entry
/// cycle region, the region minus its first entry (in RPO) is copied for the edges entering
/// it elsewhere. The reverse post-order and loop headers of the result; gives up past four
/// times the original block count.
fn make_reducible(blocks: &mut Vec<CfgBlock>) -> Result<(Vec<usize>, BTreeSet<usize>), String> {
    let limit = 4 * blocks.len() + 256;
    loop {
        let (post, headers) = order(blocks);
        let Some((_, t)) = irreducible_edge(blocks, &post) else { return Ok((post, headers)) };
        let n = blocks.len();
        let mut pos = vec![usize::MAX; n];
        for (i, &b) in post.iter().enumerate() {
            pos[b] = i;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &post {
            for s in succ_of(&blocks[b]) {
                preds[s].push(b);
            }
        }
        // nodes reachable from `from` within `inside`, along successors or predecessors
        let reach = |from: usize, inside: &[bool], fwd: bool| {
            let mut seen = vec![false; n];
            let mut stack = vec![from];
            seen[from] = true;
            while let Some(x) = stack.pop() {
                let next = if fwd { succ_of(&blocks[x]) } else { preds[x].clone() };
                for y in next {
                    if inside[y] && !seen[y] {
                        seen[y] = true;
                        stack.push(y);
                    }
                }
            }
            seen
        };
        // shrink to the innermost cycle region around `t` with more than one entry
        let mut inside: Vec<bool> = (0..n).map(|b| pos[b] != usize::MAX).collect();
        let (region, entries) = loop {
            let (f, r) = (reach(t, &inside, true), reach(t, &inside, false));
            let region: Vec<bool> = (0..n).map(|b| f[b] && r[b]).collect();
            let entries: Vec<usize> = (0..n).filter(|&b| region[b] && (b == post[0] || preds[b].iter().any(|&p| !region[p]))).collect();
            if entries.len() != 1 {
                break (region, entries);
            }
            inside = region;
            inside[entries[0]] = false;
            if !inside[t] {
                return Err("irreducible control flow (no multi-entry region)".into());
            }
        };
        if blocks.len() >= limit || entries.is_empty() {
            return Err(format!("irreducible control flow (still after splitting to {} blocks)", blocks.len()));
        }
        let h = *entries.iter().min_by_key(|&&e| pos[e]).unwrap();
        let mut copy = vec![usize::MAX; n];
        for b in (0..n).filter(|&b| region[b] && b != h) {
            copy[b] = blocks.len();
            blocks.push(CfgBlock { pc: blocks[b].pc, insts: blocks[b].insts.clone(), term: blocks[b].term.clone(), preds: Vec::new() });
        }
        let redirect = |term: &mut Term, to: &dyn Fn(usize) -> usize| match term {
            Term::Jump(x) => *x = to(*x),
            Term::Cond(_, x, y) => {
                *x = to(*x);
                *y = to(*y)
            }
            Term::End => {}
        };
        // the copies stay among themselves except for `h` and the region's exits
        for b in (0..n).filter(|&b| copy[b] != usize::MAX) {
            redirect(&mut blocks[copy[b]].term, &|x| if copy[x] != usize::MAX { copy[x] } else { x });
        }
        // edges from outside the region into its other entries go to the copies
        for p in (0..n).filter(|&p| !region[p] && pos[p] != usize::MAX) {
            redirect(&mut blocks[p].term, &|x| if region[x] && x != h { copy[x] } else { x });
        }
    }
}

fn succ_of(b: &CfgBlock) -> Vec<usize> {
    match b.term {
        Term::Jump(t) => vec![t],
        Term::Cond(_, t, f) => vec![t, f],
        Term::End => vec![],
    }
}

/// Reachable blocks in reverse post-order from the entry (iterative DFS) and the targets of
/// retreating edges.
fn order(blocks: &[CfgBlock]) -> (Vec<usize>, BTreeSet<usize>) {
    let n = blocks.len();
    let succ = succ_of;
    let mut state = vec![0u8; n]; // 0 new, 1 on stack, 2 done
    let mut post = Vec::new();
    let mut headers = BTreeSet::new();
    if n > 0 {
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            let s = succ(&blocks[b]);
            if *k < s.len() {
                let t = s[*k];
                *k += 1;
                match state[t] {
                    0 => {
                        state[t] = 1;
                        stack.push((t, 0));
                    }
                    1 => {
                        headers.insert(t);
                    }
                    _ => {}
                }
            } else {
                state[b] = 2;
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    (post, headers)
}

/// A retreating edge (a, t) of the reverse post-order `rpo` whose target does not dominate
/// its source (the CFG is irreducible), if any.
fn irreducible_edge(blocks: &[CfgBlock], rpo: &[usize]) -> Option<(usize, usize)> {
    let mut pos = vec![usize::MAX; blocks.len()];
    for (i, &b) in rpo.iter().enumerate() {
        pos[b] = i;
    }
    let mut preds = vec![Vec::new(); blocks.len()];
    for &b in rpo {
        for t in succ_of(&blocks[b]) {
            preds[t].push(b);
        }
    }
    // Cooper–Harvey–Kennedy dominators over the RPO
    let mut idom = vec![usize::MAX; blocks.len()];
    let entry = *rpo.first()?;
    idom[entry] = entry;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo[1..] {
            let mut nd = usize::MAX;
            for &p in preds[b].iter().filter(|&&p| idom[p] != usize::MAX) {
                nd = if nd == usize::MAX {
                    p
                } else {
                    let (mut x, mut y) = (p, nd);
                    while x != y {
                        while pos[x] > pos[y] {
                            x = idom[x];
                        }
                        while pos[y] > pos[x] {
                            y = idom[y];
                        }
                    }
                    x
                };
            }
            if idom[b] != nd {
                idom[b] = nd;
                changed = true;
            }
        }
    }
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if b == entry {
            return false;
        }
        b = idom[b];
    };
    rpo.iter().find_map(|&a| succ_of(&blocks[a]).into_iter().find(|&t| pos[t] <= pos[a] && !dominates(t, a)).map(|t| (a, t)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(term: Term) -> CfgBlock {
        CfgBlock { pc: 0, insts: Vec::new(), term, preds: Vec::new() }
    }

    /// The console's loop entered in the middle (`62b584e061c05fc5`'s shape): 0 → {1 | 3},
    /// 1 → 2, cycle 2 → 3 → {2 | 4}; split into a single-entry loop.
    #[test]
    fn loop_entered_in_the_middle_is_split() {
        let mut b = vec![blk(Term::Cond(0, 3, 1)), blk(Term::Jump(2)), blk(Term::Jump(3)), blk(Term::Cond(0, 2, 4)), blk(Term::End)];
        let (rpo, headers) = order(&b);
        assert!(irreducible_edge(&b, &rpo).is_some());
        let (rpo, headers2) = make_reducible(&mut b).unwrap();
        assert!(irreducible_edge(&b, &rpo).is_none());
        assert_eq!(b.len(), 6);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers2.len(), 1);
    }
}
