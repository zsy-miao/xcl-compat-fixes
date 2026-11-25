//! A console program's control flow over its basic blocks: successors, post-dominators and
//! control dependence – which blocks run (or not) because of which branch, for analyses that
//! follow what a wave-wide vote decides.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::gpu::rdna2::Decoded;

/// Basic blocks of a program (instructions in address order, far jumps resolved).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cfg {
    /// first instruction of each block; block `b` is `start[b]..start[b + 1]` (the last one ends
    /// at the program's end)
    pub start: Vec<usize>,
    /// the block of each instruction
    pub block_of: Vec<usize>,
    /// successors of each block (in the order the branch gives them: target first)
    pub succ: Vec<Vec<usize>>,
    /// predecessors of each block
    pub pred: Vec<Vec<usize>>,
    len: usize,
}

/// `Cfg::post_dominators`: the immediate post-dominator of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostDom {
    /// this block
    Block(usize),
    /// the program's end: every path from the block ends the program without a common block
    Exit,
    /// no path from the block ends the program (a loop with no way out)
    None,
}

impl Cfg {
    /// None when the program has an indirect jump (its successors are not known).
    pub fn new(insts: &[Decoded]) -> Option<Cfg> {
        let n = insts.len();
        let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
        let succ_i: Vec<Vec<usize>> = super::upper_lanes::successors(insts, &pcs).into_iter().collect::<Option<_>>()?;
        let mut leader = vec![false; n];
        if n > 0 {
            leader[0] = true;
        }
        for (i, s) in succ_i.iter().enumerate() {
            // anything but a plain fall-through ends a block, and every target starts one
            if s.as_slice() != [i + 1] {
                if i + 1 < n {
                    leader[i + 1] = true;
                }
                for &t in s {
                    leader[t] = true;
                }
            }
        }
        let start: Vec<usize> = (0..n).filter(|&i| leader[i]).collect();
        let mut block_of = vec![0; n];
        for (b, &s) in start.iter().enumerate() {
            let end = start.get(b + 1).copied().unwrap_or(n);
            block_of[s..end].fill(b);
        }
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); start.len()];
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); start.len()];
        for b in 0..start.len() {
            let last = start.get(b + 1).copied().unwrap_or(n) - 1;
            for &t in &succ_i[last] {
                let tb = block_of[t];
                if !succ[b].contains(&tb) {
                    succ[b].push(tb);
                    pred[tb].push(b);
                }
            }
        }
        Some(Cfg { start, block_of, succ, pred, len: n })
    }

    /// The instructions of block `b`.
    pub fn range(&self, b: usize) -> Range<usize> {
        self.start[b]..self.start.get(b + 1).copied().unwrap_or(self.len)
    }

    /// The immediate post-dominator of each block (Cooper, Harvey and Kennedy's iteration on the
    /// reversed graph, with one exit node after every block that ends the program). A successor
    /// from which no path ends the program counts as the exit: the branch to it does not pass
    /// through the other side's blocks.
    pub fn post_dominators(&self) -> Vec<PostDom> {
        let nb = self.start.len();
        let exit = nb;
        // the reversed graph's successors of a node: the blocks that can go to it
        let rsucc = |v: usize| -> Vec<usize> {
            if v == exit {
                (0..nb).filter(|&b| self.succ[b].is_empty()).collect()
            } else {
                self.pred[v].clone()
            }
        };
        // post-order of the reversed graph from the exit
        let mut order = vec![usize::MAX; nb + 1];
        let mut post = Vec::with_capacity(nb + 1);
        let mut seen = vec![false; nb + 1];
        let mut stack = vec![(exit, rsucc(exit), 0usize)];
        seen[exit] = true;
        while let Some((v, next, k)) = stack.last_mut() {
            if let Some(&w) = next.get(*k) {
                *k += 1;
                if !seen[w] {
                    seen[w] = true;
                    let ws = rsucc(w);
                    stack.push((w, ws, 0));
                }
            } else {
                order[*v] = post.len();
                post.push(*v);
                stack.pop();
            }
        }
        // the reversed graph's predecessors of a block: its successors, and the exit after a last block
        let rpred = |v: usize| -> Vec<usize> {
            if self.succ[v].is_empty() {
                vec![exit]
            } else {
                self.succ[v].clone()
            }
        };
        let mut idom = vec![usize::MAX; nb + 1];
        idom[exit] = exit;
        let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
            while a != b {
                while order[a] < order[b] {
                    a = idom[a];
                }
                while order[b] < order[a] {
                    b = idom[b];
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &v in post.iter().rev().filter(|&&v| v != exit) {
                let mut new = usize::MAX;
                for p in rpred(v) {
                    let p = if seen[p] { p } else { exit };
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX { p } else { intersect(&idom, p, new) };
                }
                if new != usize::MAX && idom[v] != new {
                    idom[v] = new;
                    changed = true;
                }
            }
        }
        (0..nb)
            .map(|b| match idom[b] {
                usize::MAX => PostDom::None,
                d if d == exit => PostDom::Exit,
                d => PostDom::Block(d),
            })
            .collect()
    }

    /// For each block, the blocks with a branch it is control-dependent on: the branch decides
    /// whether the block runs (Ferrante, Ottenstein and Warren: from each successor of a branch up
    /// the post-dominator tree to the branch's own post-dominator). A block on a path that never
    /// ends the program counts as dependent on every branch that leads to it.
    pub fn control_dependence(&self, ipdom: &[PostDom]) -> Vec<Vec<usize>> {
        let mut cd: Vec<Vec<usize>> = vec![Vec::new(); self.start.len()];
        for (a, s) in self.succ.iter().enumerate() {
            if s.len() < 2 {
                continue;
            }
            let stop = ipdom[a];
            for &first in s {
                let mut run = first;
                loop {
                    if PostDom::Block(run) == stop {
                        break;
                    }
                    if !cd[run].contains(&a) {
                        cd[run].push(a);
                    } else {
                        break;
                    }
                    match ipdom[run] {
                        PostDom::Block(next) => run = next,
                        PostDom::Exit => break,
                        PostDom::None => {
                            // nothing after it ends the program either: every block from here on
                            // runs only on this side
                            let mut todo = self.succ[run].clone();
                            while let Some(b) = todo.pop() {
                                if !cd[b].contains(&a) {
                                    cd[b].push(a);
                                    todo.extend(self.succ[b].iter().copied());
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        cd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::{self, encode};

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

    const CMP: I = ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 1)], None);
    const MOV2: I = ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 129)], None);
    const MOV3: I = ("S_MOV_B32", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 130)], None);

    #[test]
    fn an_if_without_else() {
        // 0 cmp, 1 scc0 → 3; 2 skipped by it; 3 the join; 4 s_endpgm
        let p = program(&[CMP, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)], None), MOV2, MOV3]);
        let g = Cfg::new(&p).unwrap();
        assert_eq!(g.start, vec![0, 2, 3]);
        assert_eq!(g.succ, vec![vec![2, 1], vec![2], vec![]]);
        let pd = g.post_dominators();
        assert_eq!(pd, vec![PostDom::Block(2), PostDom::Block(2), PostDom::Exit]);
        assert_eq!(g.control_dependence(&pd), vec![vec![], vec![0], vec![]]);
    }

    #[test]
    fn an_if_with_else() {
        // 0 cmp, 1 scc0 → 4 (else); 2 then, 3 s_branch → 5; 4 else; 5 join
        let p = program(&[
            CMP,
            ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)], None),
            MOV2,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None),
            MOV3,
            MOV2,
        ]);
        let g = Cfg::new(&p).unwrap();
        assert_eq!(g.start, vec![0, 2, 4, 5]);
        let pd = g.post_dominators();
        assert_eq!(pd, vec![PostDom::Block(3), PostDom::Block(3), PostDom::Block(3), PostDom::Exit]);
        assert_eq!(g.control_dependence(&pd), vec![vec![], vec![0], vec![0], vec![]]);
    }

    #[test]
    fn a_loop_depends_on_its_own_branch() {
        // 0 mov; 1 add, 2 cmp, 3 scc1 → 1; 4 s_endpgm
        let p = program(&[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
            ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 132)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
        ]);
        let g = Cfg::new(&p).unwrap();
        assert_eq!(g.start, vec![0, 1, 4]);
        assert_eq!(g.succ[1], vec![1, 2]);
        let pd = g.post_dominators();
        assert_eq!(pd, vec![PostDom::Block(1), PostDom::Block(2), PostDom::Exit]);
        assert_eq!(g.control_dependence(&pd), vec![vec![], vec![1], vec![]]);
    }

    #[test]
    fn a_loop_with_no_way_out_has_no_post_dominator() {
        // 0 mov; 1 add, 2 s_branch → 1 (never ends); 3 s_endpgm, unreachable
        let p = program(&[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFE)], None),
        ]);
        let g = Cfg::new(&p).unwrap();
        assert_eq!(g.start, vec![0, 1, 3]);
        assert_eq!(g.post_dominators(), vec![PostDom::None, PostDom::None, PostDom::Exit]);
    }

    #[test]
    fn a_branch_into_a_loop_with_no_way_out_decides_both_sides() {
        // 0 cmp, 1 scc0 → 4; 2 add, 3 s_branch → 2 (never ends); 4 s_endpgm
        let p = program(&[
            CMP,
            ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFE)], None),
        ]);
        let g = Cfg::new(&p).unwrap();
        assert_eq!(g.start, vec![0, 2, 4]);
        let pd = g.post_dominators();
        assert_eq!(pd, vec![PostDom::Exit, PostDom::None, PostDom::Exit]);
        assert_eq!(g.control_dependence(&pd), vec![vec![], vec![0], vec![0]]);
    }

    #[test]
    fn every_block_of_a_side_with_no_way_out_depends_on_the_branch() {
        // 0 cmp, 1 scc0 → 5; 2 mov; 3 add, 4 s_branch → 3 (never ends); 5 s_endpgm
        let p = program(&[
            CMP,
            ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 3)], None),
            MOV2,
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFE)], None),
        ]);
        let g = Cfg::new(&p).unwrap();
        assert_eq!(g.start, vec![0, 2, 3, 5]);
        let pd = g.post_dominators();
        assert_eq!(pd, vec![PostDom::Exit, PostDom::None, PostDom::None, PostDom::Exit]);
        assert_eq!(g.control_dependence(&pd), vec![vec![], vec![0], vec![0], vec![0]]);
    }

    #[test]
    fn an_indirect_jump_gives_no_graph() {
        let p = program(&[("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None)]);
        assert_eq!(Cfg::new(&p), None);
    }
}
