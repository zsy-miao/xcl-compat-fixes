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

use super::{BinOp, CastOp, FnAttrs, Module, Op, Pred, Ty, Value};

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
    fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
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
    fn idoms(&self, order: &[u32]) -> Vec<Option<usize>> {
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
    /// `XCL_DXIL_NO_OPT=phi,cse,dce` (any subset, or `all`) skips passes, for bisecting.
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
}
