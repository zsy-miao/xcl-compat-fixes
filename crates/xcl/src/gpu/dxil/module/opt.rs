//! Cheap cleanup of the entry function before it is written: unreachable
//! blocks emptied, trivial phis replaced by their single value, common
//! subexpressions merged (pure instructions along the dominator tree, wave
//! intrinsics within a block), dead instructions removed.
//!
//! The translator emits each guest instruction's per-lane expansion as is
//! (repeated lane-bit / ballot / extract sequences, operands of discarded
//! results); this pass removes the redundancy without changing what any
//! remaining instruction computes.

use std::collections::HashMap;

use super::{BinOp, Block, CastOp, FnAttrs, Module, Op, Pred, Ty, Value};

impl Op {
    /// Calls `f` on every value operand.
    pub(super) fn for_each_operand(&self, mut f: impl FnMut(Value)) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                f(*a);
                f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                f(*a);
                f(*b);
                f(*c)
            }
            Op::Call(_, args) => args.iter().copied().for_each(f),
            Op::Phi(inc) => inc.iter().for_each(|p| f(p.0)),
            Op::Gep(p, idx) => {
                f(*p);
                idx.iter().copied().for_each(f)
            }
            Op::Switch(v, _, cases) => {
                f(*v);
                cases.iter().for_each(|c| f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }

    /// Rewrites every value operand through `f`.
    pub(super) fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                *a = f(*a);
                *b = f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => *v = f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                *a = f(*a);
                *b = f(*b);
                *c = f(*c)
            }
            Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
            Op::Phi(inc) => inc.iter_mut().for_each(|p| p.0 = f(p.0)),
            Op::Gep(p, idx) => {
                *p = f(*p);
                idx.iter_mut().for_each(|v| *v = f(*v))
            }
            Op::Switch(v, _, cases) => {
                *v = f(*v);
                cases.iter_mut().for_each(|c| c.0 = f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }
}

/// How an instruction may be merged with an identical one.
#[derive(PartialEq)]
enum Cse {
    No,
    /// pure: any dominating copy will do
    Dom,
    /// a wave intrinsic: same result only under the same active lanes (same block)
    Block,
}

impl Module {
    fn wave_call(&self, f: u32) -> bool {
        let n = &self.funcs[f as usize].name;
        n.starts_with("dx.op.wave") || n.starts_with("dx.op.quad")
    }

    fn is_const_global_ptr(&self, v: Value) -> bool {
        match v {
            Value::Global(g) => self.globals[g as usize].constant,
            Value::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(base, _) => self.is_const_global_ptr(*base),
                _ => false,
            },
            _ => false,
        }
    }

    fn cse_kind(&self, op: &Op) -> Cse {
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) => Cse::Dom,
            Op::Load(p, _) if self.is_const_global_ptr(*p) => Cse::Dom,
            Op::Call(f, _) if self.funcs[*f as usize].attrs == FnAttrs::ReadNone => Cse::Dom,
            Op::Call(f, _) if self.wave_call(*f) && !self.funcs[*f as usize].name.starts_with("dx.op.waveIsFirstLane") => Cse::Block,
            _ => Cse::No,
        }
    }

    /// Cheap algebraic & constant simplification on already-resolved operands.
    fn simplify_op(&mut self, op: &Op, ty: Ty) -> Option<Value> {
        // Keep 64-bit instructions intact so Int64Ops / Doubles shader flags stay consistent.
        if matches!(self.types[ty.0 as usize], super::TypeDef::Int(64) | super::TypeDef::Double) {
            return None;
        }
        match *op {
            Op::Select(c, t, f) => {
                if t == f {
                    return Some(t);
                }
                match self.const_u64(c) {
                    Some(1) => Some(t),
                    Some(0) => Some(f),
                    _ => None,
                }
            }
            Op::Bin(bop, a, b, 0) => {
                let w = self.int_width(ty)?;
                // avoid dropping 64-bit ops that keep Int64Ops shader flags consistent
                if w > 32 {
                    return None;
                }
                let mask = (1u64 << w) - 1;
                let (ca, cb) = (self.const_u64(a), self.const_u64(b));
                if let (Some(x), Some(y)) = (ca, cb) {
                    let r = match bop {
                        BinOp::Add => Some(x.wrapping_add(y)),
                        BinOp::Sub => Some(x.wrapping_sub(y)),
                        BinOp::Mul => Some(x.wrapping_mul(y)),
                        BinOp::And => Some(x & y),
                        BinOp::Or => Some(x | y),
                        BinOp::Xor => Some(x ^ y),
                        BinOp::Shl if y < w as u64 => Some(x << y),
                        BinOp::LShr if y < w as u64 => Some(x >> y),
                        BinOp::AShr if y < w as u64 => Some((((x << (64 - w)) as i64) >> (64 - w + y as u32)) as u64),
                        BinOp::UDiv if y != 0 => Some(x / y),
                        BinOp::URem if y != 0 => Some(x % y),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r & mask));
                    }
                }
                match (bop, cb) {
                    (BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::And, Some(m)) if m == mask => return Some(a),
                    (BinOp::And | BinOp::Mul, Some(0)) => return Some(b),
                    (BinOp::Mul, Some(1)) => return Some(a),
                    _ => {}
                }
                match (bop, ca) {
                    (BinOp::Add | BinOp::Or | BinOp::Xor, Some(0)) => return Some(b),
                    (BinOp::And, Some(m)) if m == mask => return Some(b),
                    (BinOp::And | BinOp::Mul | BinOp::Shl | BinOp::LShr, Some(0)) => return Some(a),
                    (BinOp::Mul, Some(1)) => return Some(b),
                    _ => {}
                }
                if a == b {
                    match bop {
                        BinOp::And | BinOp::Or => return Some(a),
                        BinOp::Sub | BinOp::Xor => return Some(self.const_int(ty, 0)),
                        _ => {}
                    }
                }
                None
            }
            Op::Cmp(p, a, b) => {
                let at = self.value_ty(a);
                let w = self.int_width(at)?;
                if w > 32 {
                    return None;
                }
                if let (Some(x), Some(y)) = (self.const_u64(a), self.const_u64(b)) {
                    let sx = |v: u64| ((v << (64 - w)) as i64) >> (64 - w);
                    let r = match p {
                        Pred::Eq => Some(x == y),
                        Pred::Ne => Some(x != y),
                        Pred::UGt => Some(x > y),
                        Pred::UGe => Some(x >= y),
                        Pred::ULt => Some(x < y),
                        Pred::ULe => Some(x <= y),
                        Pred::SGt => Some(sx(x) > sx(y)),
                        Pred::SGe => Some(sx(x) >= sx(y)),
                        Pred::SLt => Some(sx(x) < sx(y)),
                        Pred::SLe => Some(sx(x) <= sx(y)),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                if a == b {
                    let r = match p {
                        Pred::Eq | Pred::ULe | Pred::UGe | Pred::SLe | Pred::SGe => Some(true),
                        Pred::Ne | Pred::ULt | Pred::UGt | Pred::SLt | Pred::SGt => Some(false),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                None
            }
            Op::Cast(CastOp::BitCast, Value::Inst(j), to) => {
                if matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Int(64) | super::TypeDef::Double) {
                    return None;
                }
                if let Op::Cast(CastOp::BitCast, inner, _) = self.insts[j as usize].op {
                    if self.value_ty(inner) == to {
                        return Some(inner);
                    }
                }
                None
            }
            Op::Cast(cop @ (CastOp::ZExt | CastOp::SExt | CastOp::Trunc), v, to) => {
                let sw = self.int_width(self.value_ty(v))?;
                let dw = self.int_width(to)?;
                if sw > 32 || dw > 32 {
                    return None;
                }
                let x = self.const_u64(v)?;
                let r = match cop {
                    CastOp::ZExt | CastOp::Trunc => x & ((1u64 << dw) - 1),
                    CastOp::SExt => ((((x << (64 - sw)) as i64) >> (64 - sw)) as u64) & ((1u64 << dw) - 1),
                    _ => unreachable!(),
                };
                Some(self.const_int(to, r))
            }
            _ => None,
        }
    }

    /// Whether an unused instruction can go. 64-bit values stay: helpers already set
    /// the Int64Ops / Doubles / DoubleExtensions flags for them.
    fn removable(&self, op: &Op, ty: Ty, wide: &[Ty]) -> bool {
        if wide.contains(&ty) {
            return false;
        }
        let mut w = false;
        op.for_each_operand(|v| w |= wide.contains(&self.value_ty(v)));
        if w {
            return false;
        }
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) | Op::Phi(_) | Op::Load(..) | Op::Alloca(..) => true,
            // only calls whose presence sets no shader flag (the flags are collected while
            // building and must match the code: no wave ops, f64 math, typed / raw loads)
            Op::Call(f, _) => {
                let d = &self.funcs[*f as usize];
                matches!(d.attrs, FnAttrs::ReadNone | FnAttrs::ReadOnly)
                    && !d.name.ends_with(".f64")
                    && ["dx.op.unary.", "dx.op.binary.", "dx.op.tertiary.", "dx.op.loadInput.", "dx.op.annotateHandle", "dx.op.createHandleFromBinding", "dx.op.cbufferLoadLegacy."]
                        .iter()
                        .any(|p| d.name.starts_with(p))
            }
            _ => false,
        }
    }

    /// Whether every non-phi operand of the reachable instructions is written before its
    /// use and has a value (what the writer checks).
    fn uses_well_formed(&self, order: &[u32]) -> bool {
        let mut at = vec![usize::MAX; self.insts.len()];
        let mut k = 0;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                at[i as usize] = k;
                k += 1;
            }
        }
        let mut ok = true;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                let op = &self.insts[i as usize].op;
                if matches!(op, Op::Phi(_)) {
                    continue;
                }
                op.for_each_operand(|v| {
                    if let Value::Inst(j) = v {
                        ok &= at[j as usize] < at[i as usize] && !matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Void);
                    }
                });
            }
        }
        ok
    }

    /// Immediate dominators of the blocks in `order` (reverse postorder), by
    /// Cooper, Harvey and Kennedy's iteration ("A Simple, Fast Dominance Algorithm").
    pub(super) fn idoms(&self, order: &[u32]) -> Vec<Option<usize>> {
        let n = self.blocks.len();
        let mut rpo = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            rpo[b as usize] = k;
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &b in order {
            for s in self.successors(b as usize) {
                preds[s.0 as usize].push(b as usize);
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        idom[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().skip(1) {
                let b = b as usize;
                let mut new: Option<usize> = None;
                for &p in &preds[b] {
                    if idom[p].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(mut x) => {
                            let mut y = p;
                            while x != y {
                                while rpo[x] > rpo[y] {
                                    x = idom[x].unwrap();
                                }
                                while rpo[y] > rpo[x] {
                                    y = idom[y].unwrap();
                                }
                            }
                            x
                        }
                    });
                }
                if new.is_some() && idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        idom
    }

    /// Cleans up the entry function (see the module docs of `opt`). Values
    /// keep their identities; replaced ones simply lose their uses.
    ///
    /// `XCL_DXIL_NO_OPT=phi,cse,dce,merge` (any subset, or `all`) skips passes, for bisecting.
    pub fn optimize(&mut self) {
        let off = std::env::var("XCL_DXIL_NO_OPT").unwrap_or_default();
        let skip = |p: &str| off.split(',').any(|x| x.trim() == p || x.trim() == "all");
        let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
        // malformed IR (a use before its definition or of a void result) is left for the
        // writer to report, not cleaned away
        if !self.uses_well_formed(&order) {
            return;
        }
        // unreachable blocks: emptied (the writer drops them)
        let mut reach = vec![false; self.blocks.len()];
        for &b in &order {
            reach[b as usize] = true;
        }
        for (b, r) in reach.iter().enumerate() {
            if !r {
                self.blocks[b].clear();
            }
        }
        for &b in &order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k] as usize;
                if let Op::Phi(inc) = &mut self.insts[i].op {
                    inc.retain(|p| reach[p.1 .0 as usize]);
                }
            }
        }

        let mut repl: Vec<Option<Value>> = vec![None; self.insts.len()];

        // trivial phis (all incoming values the phi itself or one value), to a fixed point
        while !skip("phi") {
            let mut changed = false;
            for &b in &order {
                for &i in &self.blocks[b as usize] {
                    let i = i as usize;
                    if repl[i].is_some() {
                        continue;
                    }
                    let Op::Phi(inc) = &self.insts[i].op else { break };
                    let me = Value::Inst(i as u32);
                    let mut same: Option<Value> = None;
                    let mut trivial = true;
                    for &(v, _) in inc {
                        let v = resolve(&mut repl, v);
                        if v == me || Some(v) == same {
                            continue;
                        }
                        if same.is_some() {
                            trivial = false;
                            break;
                        }
                        same = Some(v);
                    }
                    if let (true, Some(v)) = (trivial, same) {
                        repl[i] = Some(v);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // CSE along the dominator tree (preorder walk with a scoped table)
        let idom = self.idoms(&order);
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
        for &b in order.iter().skip(1) {
            if let Some(d) = idom[b as usize] {
                kids[d].push(b as usize);
            }
        }
        let mut table: HashMap<(Op, Ty), Value> = HashMap::new();
        // (block, undo log length) frames; undo log holds keys to remove / restore
        let mut undo: Vec<((Op, Ty), Option<Value>)> = Vec::new();
        enum Step {
            Enter(usize),
            Leave(usize),
        }
        let mut stack = if skip("cse") { Vec::new() } else { vec![Step::Enter(0)] };
        let mut cse_replaced = false;
        while let Some(step) = stack.pop() {
            let b = match step {
                Step::Leave(mark) => {
                    while undo.len() > mark {
                        let (k, old) = undo.pop().unwrap();
                        match old {
                            Some(v) => table.insert(k, v),
                            None => table.remove(&k),
                        };
                    }
                    continue;
                }
                Step::Enter(b) => b,
            };
            stack.push(Step::Leave(undo.len()));
            let mut local: HashMap<(Op, Ty), Value> = HashMap::new();
            for k in 0..self.blocks[b].len() {
                let i = self.blocks[b][k] as usize;
                let mut op = std::mem::replace(&mut self.insts[i].op, Op::Ret);
                op.map_operands(|v| resolve(&mut repl, v));
                if repl[i].is_none() {
                    if let Some(s) = self.simplify_op(&op, self.insts[i].ty) {
                        repl[i] = Some(s);
                        cse_replaced = true;
                    } else {
                        let kind = self.cse_kind(&op);
                        if kind != Cse::No {
                            let key = (op.clone(), self.insts[i].ty);
                            let t = if kind == Cse::Dom { &mut table } else { &mut local };
                            if let Some(&v) = t.get(&key) {
                                repl[i] = Some(v);
                                cse_replaced = true;
                            } else {
                                let old = t.insert(key.clone(), Value::Inst(i as u32));
                                if kind == Cse::Dom {
                                    undo.push((key, old));
                                }
                            }
                        } else if matches!(&op, Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.discard") {
                            // `dx.op.discard` demotes lanes mid-block, changing the active lane
                            // set for subsequent wave/quad intrinsics in the same basic block.
                            local.clear();
                        }
                    }
                }
                self.insts[i].op = op;
            }
            for &c in kids[b].iter().rev() {
                stack.push(Step::Enter(c));
            }
        }
        // second trivial-phi fixed point if CSE/simplification collapsed incoming values
        if cse_replaced && !skip("phi") {
            loop {
                let mut changed = false;
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        let i = i as usize;
                        if repl[i].is_some() {
                            continue;
                        }
                        let Op::Phi(inc) = &self.insts[i].op else { break };
                        let me = Value::Inst(i as u32);
                        let mut same: Option<Value> = None;
                        let mut trivial = true;
                        for &(v, _) in inc {
                            let v = resolve(&mut repl, v);
                            if v == me || Some(v) == same {
                                continue;
                            }
                            if same.is_some() {
                                trivial = false;
                                break;
                            }
                            same = Some(v);
                        }
                        if let (true, Some(v)) = (trivial, same) {
                            repl[i] = Some(v);
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        // phis may name values defined later (back edges): resolve once more
        for &b in &order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k] as usize;
                self.insts[i].op.map_operands(|v| resolve(&mut repl, v));
            }
        }

        // dead instructions: live = side effects, terminators and their operands, transitively
        let (i64t, f64t) = self.existing_i64_f64();
        let wide: Vec<Ty> = i64t.into_iter().chain(f64t).collect();
        let mut live = vec![false; self.insts.len()];
        let mut work: Vec<u32> = Vec::new();
        for &b in &order {
            for &i in &self.blocks[b as usize] {
                if repl[i as usize].is_none() && (skip("dce") || !self.removable(&self.insts[i as usize].op, self.insts[i as usize].ty, &wide)) {
                    live[i as usize] = true;
                    work.push(i);
                }
            }
        }
        while let Some(i) = work.pop() {
            self.insts[i as usize].op.for_each_operand(|v| {
                if let Value::Inst(j) = v {
                    if !live[j as usize] {
                        live[j as usize] = true;
                        work.push(j);
                    }
                }
            });
        }
        for b in &mut self.blocks {
            b.retain(|&i| live[i as usize]);
        }

        if !skip("merge") && self.simplify_cfg() {
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            let mut repl2: Vec<Option<Value>> = vec![None; self.insts.len()];
            let mut phi_changed = false;
            while !skip("phi") {
                let mut changed = false;
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        let i = i as usize;
                        if repl2[i].is_some() {
                            continue;
                        }
                        let Op::Phi(inc) = &self.insts[i].op else { break };
                        let me = Value::Inst(i as u32);
                        let mut same: Option<Value> = None;
                        let mut trivial = true;
                        for &(v, _) in inc {
                            let v = resolve(&mut repl2, v);
                            if v == me || Some(v) == same {
                                continue;
                            }
                            if same.is_some() {
                                trivial = false;
                                break;
                            }
                            same = Some(v);
                        }
                        if let (true, Some(v)) = (trivial, same) {
                            repl2[i] = Some(v);
                            changed = true;
                            phi_changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            if phi_changed {
                for &b in &order {
                    for k in 0..self.blocks[b as usize].len() {
                        let i = self.blocks[b as usize][k] as usize;
                        self.insts[i].op.map_operands(|v| resolve(&mut repl2, v));
                    }
                }
            }
            if !skip("dce") {
                live.fill(false);
                work.clear();
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        if repl2[i as usize].is_none() && !self.removable(&self.insts[i as usize].op, self.insts[i as usize].ty, &wide) {
                            live[i as usize] = true;
                            work.push(i);
                        }
                    }
                }
                while let Some(i) = work.pop() {
                    self.insts[i as usize].op.for_each_operand(|v| {
                        if let Value::Inst(j) = v {
                            if !live[j as usize] {
                                live[j as usize] = true;
                                work.push(j);
                            }
                        }
                    });
                }
                for b in &mut self.blocks {
                    b.retain(|&i| live[i as usize]);
                }
            }
            debug_assert!(self.uses_well_formed(&order));
        }
    }

    /// Simplifies the CFG to a fixed point:
    /// 1. Collapses degenerate `CondBr(_, t, t)` and `Switch` cases matching the default target.
    /// 2. Bypasses empty trampoline blocks (`[Br(t)]`) whenever `t`'s phis allow it.
    /// 3. Merges single-predecessor blocks `a -> b` along unconditional branches.
    fn simplify_cfg(&mut self) -> bool {
        let mut ever_changed = false;
        let mut phi_repl: Vec<Option<Value>> = Vec::new();
        loop {
            let mut changed = false;
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();

            // Step 0: simplify degenerate terminators (`CondBr(_, t, t)`, `Switch` cases -> default).
            for &b_u32 in &order {
                let b = b_u32 as usize;
                let Some(&term_i) = self.blocks[b].last() else { continue };
                let b_blk = Block(b_u32);
                match &mut self.insts[term_i as usize].op {
                    Op::CondBr(_, t, f) if *t == *f => {
                        let target = *t;
                        self.insts[term_i as usize].op = Op::Br(target);
                        self.drop_phi_edges(target.0 as usize, b_blk, 1);
                        changed = true;
                    }
                    Op::Switch(_, def, cases) => {
                        let def = *def;
                        let before = cases.len();
                        cases.retain(|&(_, target)| target != def);
                        let dropped = before - cases.len();
                        let became_empty = cases.is_empty();
                        if dropped > 0 {
                            self.drop_phi_edges(def.0 as usize, b_blk, dropped);
                            changed = true;
                        }
                        if became_empty {
                            self.insts[term_i as usize].op = Op::Br(def);
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }

            // Step 1: bypass empty trampoline blocks (`[Br(t)]`, `b != 0`, `t != b`).
            // `preds[x]` is maintained as a superset of live blocks whose terminator targets `x`.
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
            for &b_u32 in &order {
                let b = b_u32 as usize;
                for s in self.successors(b) {
                    let s = s.0 as usize;
                    if !preds[s].contains(&b) {
                        preds[s].push(b);
                    }
                }
            }
            for &b_u32 in &order {
                let b = b_u32 as usize;
                if b == 0 || self.blocks[b].len() != 1 {
                    continue;
                }
                let Op::Br(t_blk) = self.insts[self.blocks[b][0] as usize].op else { continue };
                let t = t_blk.0 as usize;
                if t == b || self.blocks[t].is_empty() {
                    continue;
                }
                let b_blk = Block(b_u32);
                let cand_preds = preds[b].clone();
                let target_has_phi = self.blocks[t]
                    .first()
                    .is_some_and(|&i| matches!(self.insts[i as usize].op, Op::Phi(_)));
                if target_has_phi {
                    // Bypassing a multi-edge trampoline into a phi block would duplicate phi
                    // operands for every phi in `t`; only bypass when `b` has a single incoming edge.
                    let total_edges: usize = cand_preds
                        .iter()
                        .filter(|&&p| p != b && !self.blocks[p].is_empty())
                        .map(|&p| self.count_term_edges(p, b_blk))
                        .sum();
                    if total_edges > 1 {
                        continue;
                    }
                }
                let mut all_redirected = true;
                let mut any_redirected = false;
                for p in cand_preds {
                    if p == b || self.blocks[p].is_empty() {
                        continue;
                    }
                    let edges = self.count_term_edges(p, b_blk);
                    if edges == 0 {
                        continue;
                    }
                    let p_blk = Block(p as u32);
                    if !self.phi_redirect_safe(t, b_blk, p_blk) {
                        all_redirected = false;
                        continue;
                    }
                    self.rewrite_term_target(p, b_blk, t_blk);
                    self.add_redirected_phi_edges(t, b_blk, p_blk, edges);
                    if !preds[t].contains(&p) {
                        preds[t].push(p);
                    }
                    any_redirected = true;
                    changed = true;
                }
                if any_redirected && all_redirected {
                    self.blocks[b].clear();
                    self.remove_all_phi_edges(t, b_blk);
                }
            }

            // Step 2: merge single-predecessor linear blocks (`a -> b`).
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            let mut pred_edges = vec![0u32; self.blocks.len()];
            for &b_u32 in &order {
                for s in self.successors(b_u32 as usize) {
                    pred_edges[s.0 as usize] += 1;
                }
            }
            let mut any_phi_repl = false;
            for &a_u32 in &order {
                let a = a_u32 as usize;
                if self.blocks[a].is_empty() {
                    continue;
                }
                let a_blk = Block(a_u32);
                while let Some(&term_i) = self.blocks[a].last() {
                    let Op::Br(b_blk) = self.insts[term_i as usize].op else { break };
                    let b = b_blk.0 as usize;
                    if b == 0 || b == a || self.blocks[b].is_empty() || pred_edges[b] != 1 {
                        break;
                    }
                    // Verify every leading phi in `b` has an incoming value from `a` before mutating.
                    let mut phis_ok = true;
                    for &inst in &self.blocks[b] {
                        let Op::Phi(inc) = &self.insts[inst as usize].op else { break };
                        if !inc.iter().any(|&(_, bb)| bb == a_blk) {
                            phis_ok = false;
                            break;
                        }
                    }
                    if !phis_ok {
                        break;
                    }
                    // Pop `Br(b)` from `a` and move `b`'s non-phi instructions into `a`.
                    self.blocks[a].pop();
                    let b_insts = std::mem::take(&mut self.blocks[b]);
                    for inst in b_insts {
                        if let Op::Phi(inc) = &self.insts[inst as usize].op {
                            let v = inc.iter().find(|&&(_, bb)| bb == a_blk).unwrap().0;
                            if phi_repl.is_empty() {
                                phi_repl.resize(self.insts.len(), None);
                            }
                            phi_repl[inst as usize] = Some(v);
                            any_phi_repl = true;
                        } else {
                            self.def_block[inst as usize] = a_blk;
                            self.blocks[a].push(inst);
                        }
                    }
                    // Update successor phis from `b` to `a`.
                    let mut seen_s = Vec::new();
                    for s_blk in self.successors(a) {
                        if seen_s.contains(&s_blk) {
                            continue;
                        }
                        seen_s.push(s_blk);
                        let s = s_blk.0 as usize;
                        for k in 0..self.blocks[s].len() {
                            let i = self.blocks[s][k] as usize;
                            let Op::Phi(inc) = &mut self.insts[i].op else { break };
                            for (_, bb) in inc.iter_mut() {
                                if *bb == b_blk {
                                    *bb = a_blk;
                                }
                            }
                        }
                    }
                    pred_edges[b] = 0;
                    changed = true;
                }
            }
            if any_phi_repl {
                let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
                for &b_u32 in &order {
                    let b = b_u32 as usize;
                    for k in 0..self.blocks[b].len() {
                        let i = self.blocks[b][k] as usize;
                        self.insts[i].op.map_operands(|v| resolve(&mut phi_repl, v));
                    }
                }
                phi_repl.fill(None);
            }

            if !changed {
                break;
            }
            ever_changed = true;
        }
        ever_changed
    }

    fn count_term_edges(&self, b: usize, target: Block) -> usize {
        let Some(&term_i) = self.blocks[b].last() else { return 0 };
        match &self.insts[term_i as usize].op {
            Op::Br(t) => usize::from(*t == target),
            Op::CondBr(_, t, f) => usize::from(*t == target) + usize::from(*f == target),
            Op::Switch(_, d, cases) => usize::from(*d == target) + cases.iter().filter(|&&(_, c)| c == target).count(),
            _ => 0,
        }
    }

    fn rewrite_term_target(&mut self, b: usize, from: Block, to: Block) {
        let Some(&term_i) = self.blocks[b].last() else { return };
        match &mut self.insts[term_i as usize].op {
            Op::Br(t) => {
                if *t == from {
                    *t = to;
                }
            }
            Op::CondBr(_, t, f) => {
                if *t == from {
                    *t = to;
                }
                if *f == from {
                    *f = to;
                }
            }
            Op::Switch(_, d, cases) => {
                if *d == from {
                    *d = to;
                }
                for (_, c) in cases.iter_mut() {
                    if *c == from {
                        *c = to;
                    }
                }
            }
            _ => {}
        }
    }

    fn phi_redirect_safe(&self, target: usize, via: Block, pred: Block) -> bool {
        for &i in &self.blocks[target] {
            let Op::Phi(inc) = &self.insts[i as usize].op else { break };
            let Some(&(v_via, _)) = inc.iter().find(|&&(_, bb)| bb == via) else {
                return false;
            };
            if inc.iter().any(|&(v_p, bb)| bb == pred && v_p != v_via) {
                return false;
            }
        }
        true
    }

    fn add_redirected_phi_edges(&mut self, target: usize, via: Block, pred: Block, count: usize) {
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            let v_via = inc.iter().find(|&&(_, bb)| bb == via).unwrap().0;
            for _ in 0..count {
                inc.push((v_via, pred));
            }
        }
    }

    fn remove_all_phi_edges(&mut self, target: usize, from: Block) {
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            inc.retain(|&(_, bb)| bb != from);
        }
    }

    fn drop_phi_edges(&mut self, target: usize, from: Block, count: usize) {
        if count == 0 {
            return;
        }
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            let mut rem = count;
            inc.retain(|&(_, bb)| {
                if rem > 0 && bb == from {
                    rem -= 1;
                    false
                } else {
                    true
                }
            });
        }
    }
}

/// The value `v` stands for after replacements (path-compressing the chain).
fn resolve(repl: &mut [Option<Value>], v: Value) -> Value {
    let mut r = v;
    while let Value::Inst(i) = r {
        match repl[i as usize] {
            Some(n) => r = n,
            None => break,
        }
    }
    let mut c = v;
    while let Value::Inst(i) = c {
        match repl[i as usize] {
            Some(n) if n != r => {
                repl[i as usize] = Some(r);
                c = n;
            }
            _ => break,
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_calls(m: &Module, target_fn: u32) -> usize {
        m.blocks
            .iter()
            .flat_map(|b| b.iter())
            .filter(|&&i| matches!(m.insts[i as usize].op, Op::Call(f, _) if f == target_fn))
            .count()
    }

    #[test]
    fn block_cse_invalidates_wave_and_quad_calls_across_discard() {
        let build = |op_name: &str, with_discard: bool| -> (Module, u32) {
            let mut m = Module::new();
            m.define_entry("main");
            let b = m.entry_block();
            m.set_block(b);

            let void = m.void();
            let i1 = m.i1();
            let i32t = m.i32();

            let wave_ty = m.func_ty(i32t, &[i32t, i1]);
            let wave_fn = m.declare(op_name, wave_ty, FnAttrs::NoUnwind);
            let discard_ty = m.func_ty(void, &[i32t, i1]);
            let discard_fn = m.declare("dx.op.discard", discard_ty, FnAttrs::NoUnwind);
            let sink_ty = m.func_ty(void, &[i32t, i32t]);
            let sink_fn = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

            let c_op = m.i32c(116);
            let c_true = m.i1c(true);
            let c_discard_op = m.i32c(82);

            let first = m.call(wave_fn, &[c_op, c_true]);
            if with_discard {
                m.call(discard_fn, &[c_discard_op, c_true]);
            }
            let second = m.call(wave_fn, &[c_op, c_true]);
            m.call(sink_fn, &[first, second]);
            m.ret();
            m.optimize();
            (m, wave_fn)
        };

        // Without `dx.op.discard`, identical wave/quad calls in the same block still CSE to 1.
        let (m_no_discard, f_no_discard) = build("dx.op.waveActiveBallot", false);
        assert_eq!(count_calls(&m_no_discard, f_no_discard), 1);

        // Across `dx.op.discard`, active lanes change so both calls must be preserved.
        for op_name in ["dx.op.waveActiveBallot", "dx.op.quadReadLaneAt"] {
            let (m_with_discard, f_with_discard) = build(op_name, true);
            assert_eq!(
                count_calls(&m_with_discard, f_with_discard),
                2,
                "{op_name} must not be CSE-merged across dx.op.discard"
            );
        }
    }

    fn reachable_blocks(m: &Module) -> usize {
        m.block_layout().0.len()
    }

    #[test]
    fn simplify_cfg_collapses_trampoline_chain_and_merges_linear_blocks() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let t1 = m.new_block();
        let t2 = m.new_block();
        let mid = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let c7 = m.i32c(7);
        m.br(t1);

        m.set_block(t1);
        m.br(t2);

        m.set_block(t2);
        m.br(mid);

        m.set_block(mid);
        let p = m.phi(i32t);
        m.add_incoming(p, c7, t2);
        let c3 = m.i32c(3);
        let sum = m.bin(BinOp::Add, p, c3);
        m.br(exit);

        m.set_block(exit);
        m.call(sink, &[sum]);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 1, "entire linear trampoline chain should collapse into entry");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        assert_eq!(count_calls(&m, sink), 1);
    }

    #[test]
    fn simplify_cfg_preserves_phi_conflict_trampoline() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let tramp = m.new_block();
        let join = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let cond = m.cmp(Pred::Eq, lane, z);
        let v1 = m.i32c(1);
        let v2 = m.i32c(2);
        m.cond_br(cond, tramp, join);

        m.set_block(tramp);
        m.br(join);

        m.set_block(join);
        let p = m.phi(i32t);
        m.add_incoming(p, v1, entry);
        m.add_incoming(p, v2, tramp);
        m.call(sink, &[p]);
        m.ret();

        m.optimize();
        // `tramp` cannot be bypassed because `entry` already feeds a different value into `join`'s phi.
        assert_eq!(reachable_blocks(&m), 3);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn simplify_cfg_collapses_same_target_cond_br_and_switch() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let mid = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let c1 = m.i32c(1);
        let cond = m.cmp(Pred::Eq, lane, z);
        let v = m.i32c(42);
        m.cond_br(cond, mid, mid);

        m.set_block(mid);
        let p = m.phi(i32t);
        m.add_incoming(p, v, entry);
        m.add_incoming(p, v, entry);
        m.switch(lane, exit, &[(z, exit), (c1, exit)]);

        m.set_block(exit);
        let p2 = m.phi(i32t);
        m.add_incoming(p2, p, mid);
        m.add_incoming(p2, p, mid);
        m.add_incoming(p2, p, mid);
        m.call(sink, &[p2]);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 1, "degenerate CondBr and Switch should collapse and merge");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        assert_eq!(count_calls(&m, sink), 1);
    }

    #[test]
    fn simplify_cfg_merges_loop_latch_and_rewrites_header_phi() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let header = m.new_block();
        let latch = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let z = m.i32c(0);
        let one = m.i32c(1);
        let limit = m.i32c(10);
        m.br(header);

        m.set_block(header);
        let iv = m.phi(i32t);
        m.add_incoming(iv, z, entry);
        m.br(latch);

        m.set_block(latch);
        let next = m.bin(BinOp::Add, iv, one);
        m.add_incoming(iv, next, latch);
        let cont = m.cmp(Pred::ULt, next, limit);
        m.cond_br(cont, header, exit);

        m.set_block(exit);
        m.call(sink, &[next]);
        m.ret();

        m.optimize();
        // `latch` merges into `header`, leaving `entry`, `header` (self-loop), `exit`.
        assert_eq!(reachable_blocks(&m), 3);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        // Header phi's backedge predecessor must now be `header` itself.
        let Value::Inst(iv_i) = iv else { unreachable!() };
        let Op::Phi(inc) = &m.insts[iv_i as usize].op else { panic!("expected phi") };
        assert!(inc.iter().any(|&(_, b)| b == header), "header phi backedge should point to header after latch merge");
        assert!(!inc.iter().any(|&(_, b)| b == latch), "merged latch must not remain in header phi");
    }

    #[test]
    fn simplify_cfg_does_not_inflate_multi_pred_phi_trampoline() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let left = m.new_block();
        let right = m.new_block();
        let other = m.new_block();
        let tramp = m.new_block();
        let join = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let c0 = m.i32c(0);
        let c1 = m.i32c(1);
        let v1 = m.i32c(11);
        let v2 = m.i32c(22);
        m.switch(lane, other, &[(c0, left), (c1, right)]);

        m.set_block(left);
        m.call(sink, &[c0]);
        m.br(tramp);

        m.set_block(right);
        m.call(sink, &[c1]);
        m.br(tramp);

        m.set_block(other);
        m.br(join);

        m.set_block(tramp);
        m.br(join);

        m.set_block(join);
        let p = m.phi(i32t);
        m.add_incoming(p, v1, tramp);
        m.add_incoming(p, v2, other);
        m.call(sink, &[p]);
        m.ret();

        m.optimize();
        // `other` (1 pred -> phi) IS bypassed into `entry -> join`, while `tramp` (2 preds -> phi) stays
        // so `join`'s phi stays at 2 entries instead of growing to 3.
        let Value::Inst(p_i) = p else { unreachable!() };
        let Op::Phi(inc) = &m.insts[p_i as usize].op else { panic!("expected phi") };
        assert_eq!(inc.len(), 2);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }
}
