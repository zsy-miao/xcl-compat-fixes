//! Partition the optimized compute entry at CFG edges. A dispatch cannot
//! jump back to an earlier dispatch, so a strongly connected component is
//! indivisible. Each crossing edge has its own compact state layout.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::{BinOp, Block, CastOp, Module, Op, TypeDef, Value};
use crate::gpu::dxil::shader::{Feature, RawUav, Shader};

#[derive(Debug)]
pub(crate) struct Edge {
    pub from: Block,
    pub to: Block,
    pub live: Vec<Value>,
}

pub(crate) struct Plan {
    pub blocks: Vec<Vec<Block>>,
    pub edges: Vec<Edge>,
}

#[derive(Clone)]
pub struct DispatchChain {
    pub pieces: Vec<Vec<u8>>,
    pub dwords_per_group: u32,
}

struct Scratch {
    uav: RawUav,
    base: Value,
    bytes_per_slot: u32,
}

/// The part of a dispatch a chain runs at once (`pso::record_dispatch_chain`): its groups in X
/// and Y, and the dispatch's group its first one is.
struct Slab {
    width: Value,
    height: Value,
    origin: [Value; 3],
}

impl Slab {
    fn new(s: &mut Shader) -> Self {
        let cb = s.cbv(0, 2, 32);
        let ty = s.m.i32();
        let [size, origin] = [0, 1].map(|row| {
            let row = s.m.i32c(row);
            s.cbuffer_load_legacy(cb, row, ty)
        });
        Self { width: size[0], height: size[1], origin: [origin[0], origin[1], origin[2]] }
    }

    /// The program's value for its copied instruction `v`: a group id is the dispatch's, not the slab's.
    fn group_id(&self, src: &Module, op: &Op, m: &mut Module, v: Value) -> Value {
        match op {
            Op::Call(f, args) if src.funcs[*f as usize].name.starts_with("dx.op.groupId") => match args.get(1).and_then(|&a| src.const_u64(a)) {
                Some(axis @ 0..=2) => m.bin(BinOp::Add, v, self.origin[axis as usize]),
                _ => v,
            },
            _ => v,
        }
    }
}

impl Scratch {
    fn new(s: &mut Shader, slab: &Slab, stride: u32, threads: u32, lane: Value) -> Self {
        let uav = s.raw_uav(0, 2);
        let (width, height) = (slab.width, slab.height);
        let [x, y, z] = [0, 1, 2].map(|axis| s.group_id(axis));
        let row = s.m.bin(BinOp::Mul, z, height);
        let row = s.m.bin(BinOp::Add, row, y);
        let first = s.m.bin(BinOp::Mul, row, width);
        let group = s.m.bin(BinOp::Add, first, x);
        // Give consecutive logical lanes consecutive words for a slot, including
        // the continuation tag in slot zero. A workgroup owns all its slots.
        let group_words = s.m.i32c(stride * threads);
        let first = s.m.bin(BinOp::Mul, group, group_words);
        let index = s.m.bin(BinOp::Add, first, lane);
        let four = s.m.i32c(4);
        Self { uav, base: s.m.bin(BinOp::Mul, index, four), bytes_per_slot: threads * 4 }
    }

    fn at(&self, m: &mut Module, slot: u32) -> Value {
        let offset = m.i32c(slot * self.bytes_per_slot);
        m.bin(BinOp::Add, self.base, offset)
    }
}

impl Shader {
    /// Start with full native wave32 groups. A fresh dispatch can assign a
    /// different physical wave to a group; wave-lane index identifies the
    /// continuation within a full, single-wave group.
    pub(crate) fn split_compute(&self, threads: [u32; 3], split_tid: Option<Value>, limit: usize) -> Option<DispatchChain> {
        if self.m.globals.iter().any(|g| !g.constant && !(split_tid.is_some() && matches!(g.name.as_str(), "xcl_xw" | "xcl_xs"))) {
            return None;
        }
        if self.m.blocks.iter().map(Vec::len).sum::<usize>() <= limit { return None; }
        let mut prefix = Vec::new();
        if let Some(tid) = split_tid {
            let end = self.m.blocks[0].iter().position(|&i| Value::Inst(i) == tid)?;
            prefix.extend_from_slice(&self.m.blocks[0][..=end]);
        }
        let mut consumed = HashSet::new();
        for &i in self.m.blocks.iter().flatten() {
            self.m.insts[i as usize].op.for_each_operand(|v| { consumed.insert(v); });
        }
        for (b, block) in self.m.blocks.iter().enumerate() {
            // In a single split console wave each inter-half exchange begins
            // with TGSM stores, one rendezvous, peer loads and a rendezvous.
            // A block ending anywhere inside that sequence cannot be a cut.
            let mut exchange = 0;
            for &i in block {
                if b == 0 && prefix.contains(&i) { continue; }
                match &self.m.insts[i as usize].op {
                    Op::Alloca(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => return None,
                    Op::Store(..) if split_tid.is_some() && exchange <= 1 => exchange = 1,
                    Op::Load(p, _) if self.m.global_root(*p).is_some_and(|g| self.m.globals[g].constant) => {}
                    Op::Load(..) if split_tid.is_some() && matches!(exchange, 2 | 3) => exchange = 3,
                    Op::Load(..) | Op::Store(..) => return None,
                    Op::Call(f, args) if self.m.funcs[*f as usize].name == "dx.op.barrier"
                            && self.m.const_u64(args[1]) == Some(crate::gpu::dxil::ops::barrier::GROUP_SYNC as u64) => {
                        exchange = match exchange { 1 => 2, 3 => 0, _ => return None };
                    }
                    // Sub-dword stores use interlocked AND/OR. They return an
                    // old word that nobody observes; every write stays in place.
                    Op::Call(f, args) if self.m.funcs[*f as usize].name == "dx.op.atomicBinOp.i32"
                        && matches!(self.m.const_u64(args[2]), Some(1 | 2))
                        && !consumed.contains(&Value::Inst(i)) => {}
                    Op::Call(f, _) if ["dx.op.barrier", "dx.op.atomic"].iter().any(|p| self.m.funcs[*f as usize].name.starts_with(p)) => return None,
                    _ => {}
                }
            }
            if exchange != 0 { return None; }
        }
        let plan = self.m.dispatch_plan(limit)?;
        // Slot zero is the tag. Every transported SSA value owns its payload slots
        // until the chain finishes. An outgoing store can never overwrite a value
        // restored by the same piece, even when the restore is sunk past a branch.
        let mut stride = 1u32;
        let mut slots = BTreeMap::new();
        for value in plan.edges.iter().flat_map(|e| e.live.iter().copied()).collect::<BTreeSet<_>>() {
            let words = self.m.transport_words(value)?;
            if words != 0 {
                slots.insert(value, stride);
                stride = stride.checked_add(words)?;
            }
        }
        stride.checked_mul(threads[0])?.checked_mul(4)?;
        let pieces = (0..plan.blocks.len()).map(|i| self.make_piece(&plan, i, stride, &slots, threads, &prefix)).collect::<Option<Vec<_>>>()?;
        Some(DispatchChain { pieces, dwords_per_group: stride.checked_mul(threads[0])? })
    }

    fn make_piece(&self, plan: &Plan, stage: usize, stride: u32, slots: &BTreeMap<Value, u32>, threads: [u32; 3], prefix: &[u32]) -> Option<Vec<u8>> {
        let src = &self.m;
        let mut dst = self.dispatch_piece();
        let mut values = HashMap::new();
        let slab = Slab::new(&mut dst);
        for &id in prefix {
            let def = &src.insts[id as usize];
            let mut op = def.op.clone();
            op.map_operands(|v| *values.get(&v).unwrap_or(&v));
            let v = dst.m.push(op, def.ty);
            values.insert(Value::Inst(id), slab.group_id(src, &def.op, &mut dst.m, v));
        }
        let lane = prefix.last().map(|&i| values[&Value::Inst(i)]).unwrap_or_else(|| dst.wave_lane_index());
        let scratch = Scratch::new(&mut dst, &slab, stride, threads[0], lane);
        let local: HashMap<_, _> = plan.blocks[stage].iter().map(|&b| (b, dst.m.new_block())).collect();
        let mut hops = HashMap::new();
        let mut exits = HashMap::new();
        let mut cases = Vec::new();
        let mut deferred = BTreeMap::new();
        let mut incoming = BTreeSet::new();
        let done = dst.m.new_block();
        let selector = if stage != 0 { Some(dst.raw_load(scratch.uav, scratch.base, 1)[0]) } else { None };
        for (index, edge) in plan.edges.iter().enumerate() {
            let tag = dst.m.i32c(index as u32 + 1);
            if local.contains_key(&edge.from) {
                exits.insert((edge.from, edge.to), (dst.m.new_block(), tag));
            }
            if let Some(&target) = local.get(&edge.to) {
                incoming.extend(edge.live.iter().copied());
                let hop = dst.m.new_block();
                dst.m.set_block(hop);
                dst.m.br(target);
                cases.push((tag, hop));
                hops.insert((edge.from, edge.to), hop);
            }
        }
        dst.m.set_block(Block(0));
        if let Some(sel) = selector {
            for v in incoming {
                if values.contains_key(&v) { continue; }
                if slots.contains_key(&v) {
                    let id = dst.m.detached_phi(src.insts[inst(v)?].ty);
                    deferred.insert(id, v);
                    values.insert(v, id);
                } else {
                    src.rebuild_static(v, &mut dst.m, &mut values)?;
                }
            }
            dst.m.switch(sel, done, &cases);
        } else {
            dst.m.br(local[&Block(0)]);
        }
        dst.m.set_block(done);
        dst.m.ret();

        // Assign all instruction identities before fixing operands. A loop's
        // phi can use an instruction in a block copied later.
        let mut copied = Vec::new();
        for &block in &plan.blocks[stage] {
            dst.m.set_block(local[&block]);
            for &i in &src.blocks[block.0 as usize] {
                if block == Block(0) && prefix.contains(&i) { continue; }
                let old = &src.insts[i as usize];
                if matches!(old.op, Op::Ret) && stage + 1 < plan.blocks.len() {
                    let zero = dst.m.i32c(0);
                    dst.raw_store(scratch.uav, scratch.base, &[zero]);
                }
                let v = dst.m.push(old.op.clone(), old.ty);
                values.insert(Value::Inst(i), slab.group_id(src, &old.op, &mut dst.m, v));
                copied.push((block, inst(v)?));
            }
        }
        for (block, id) in copied {
            let op = &mut dst.m.insts[id].op;
            let mut complete = true;
            op.map_operands(|v| match v {
                Value::Inst(_) => values.get(&v).copied().unwrap_or_else(|| { complete = false; v }),
                _ => v,
            });
            if !complete { return None; }
            let target = |b: Block| local.get(&b).copied().or_else(|| exits.get(&(block, b)).map(|p| p.0));
            match op {
                Op::Phi(inc) => for (_, p) in inc {
                    *p = *local.get(p).or_else(|| hops.get(&(*p, block)))?;
                },
                Op::Br(b) => *b = target(*b)?,
                Op::CondBr(_, t, f) => { *t = target(*t)?; *f = target(*f)?; }
                Op::Switch(_, d, cases) => {
                    *d = target(*d)?;
                    for (_, t) in cases { *t = target(*t)?; }
                }
                _ => {}
            }
        }
        for edge in &plan.edges {
            let Some(&(b, tag)) = exits.get(&(edge.from, edge.to)) else { continue };
            dst.m.set_block(b);
            for &v in &edge.live {
                // The defining piece writes a distinct slot. Later pieces can bypass
                // it or read it without copying it at their own outgoing edges.
                if let Some(&slot) = slots.get(&v) {
                    if !local.contains_key(&src.def_block[inst(v)?]) { continue; }
                    let words = src.pack(v, *values.get(&v)?, &mut dst.m);
                    for (k, word) in words.into_iter().enumerate() {
                        let at = scratch.at(&mut dst.m, slot + k as u32);
                        dst.raw_store(scratch.uav, at, &[word]);
                    }
                }
            }
            dst.raw_store(scratch.uav, scratch.base, &[tag]);
            dst.m.ret();
        }
        src.place_restores(&mut dst, &scratch, slots, &deferred)?;
        dst.m.sink_cheap_piece_ops();
        dst.infer_piece_features();
        // a refused piece leaves the whole program unsplit: say so
        let refused = |e: String| crate::xlog_warn!("compute split: piece {} of {} refused, the program stays whole: {}",
            stage + 1, plan.blocks.len(), e.lines().next().unwrap_or_default());
        let blob = dst.try_finish(threads, Some(32)).map_err(|e| refused(e.to_string())).ok()?;
        crate::gpu::dxil::validate::validate(&blob).map_err(refused).ok()?;
        Some(blob)
    }

    fn infer_piece_features(&mut self) {
        for f in [Feature::Doubles, Feature::Int64Ops, Feature::DoubleExtensions,
                  Feature::ResourceHeapIndexing, Feature::SamplerHeapIndexing] {
            self.features.remove(&f);
        }
        // typed UAV loads (`texture_load` without a mip): the whole program's flag stays only on
        // the pieces that have one, the validator wants flags to match usage
        let uav_load = |args: &[Value], m: &Module| args.get(2).is_some_and(|&mip| matches!(mip, Value::Const(c) if matches!(m.consts[c as usize], super::ConstDef::Undef(_))));
        if !self.m.blocks.iter().flatten().any(|&i| matches!(&self.m.insts[i as usize].op,
            Op::Call(f, args) if self.m.funcs[*f as usize].name.starts_with("dx.op.textureLoad") && uav_load(args, &self.m))) {
            self.features.remove(&Feature::TypedUavLoadAdditionalFormats);
        }
        for def in &self.m.insts {
            if let Op::Call(f, args) = &def.op {
                if self.m.funcs[*f as usize].name == "dx.op.createHandleFromHeap" {
                    self.features.insert(if self.m.const_u64(args[2]) == Some(0) {
                        Feature::ResourceHeapIndexing
                    } else { Feature::SamplerHeapIndexing });
                }
            }
        }
    }
}

fn common_dominator(mut a: usize, mut b: usize, rank: &[Option<u32>], dom: &[Option<usize>]) -> usize {
    while a != b {
        if rank[a] > rank[b] {
            a = dom[a].unwrap();
        } else {
            b = dom[b].unwrap();
        }
    }
    a
}

fn inst(v: Value) -> Option<usize> {
    if let Value::Inst(i) = v { Some(i as usize) } else { None }
}

impl Module {
    fn global_root(&self, v: Value) -> Option<usize> {
        match v {
            Value::Global(g) => Some(g as usize),
            Value::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(p, _) => self.global_root(*p),
                _ => None,
            },
            _ => None,
        }
    }

    // Phi operands are consumed on incoming edges. A forwarding-only input has
    // no placeholder users, so it requires no load in this piece.
    fn place_restores(
        &self,
        dst: &mut Shader,
        scratch: &Scratch,
        slots: &BTreeMap<Value, u32>,
        deferred: &BTreeMap<Value, Value>,
    ) -> Option<()> {
        let (order, rank) = dst.m.block_layout();
        let dom = dst.m.idoms(&order.iter().map(|&b| b as u32).collect::<Vec<_>>());
        let mut uses = HashMap::new();
        for &b in &order {
            for &id in &dst.m.blocks[b] {
                let op = &dst.m.insts[id as usize].op;
                let mut record = |v: Value, p: usize| {
                    if deferred.contains_key(&v) {
                        uses.entry(v).and_modify(|d| *d = common_dominator(*d, p, &rank, &dom)).or_insert(p);
                    }
                };
                if let Op::Phi(edges) = op {
                    for &(v, p) in edges {
                        record(v, p.0 as usize);
                    }
                } else {
                    op.for_each_operand(|v| record(v, b));
                }
            }
        }
        let mut replacements = HashMap::new();
        for (&placeholder, &v) in deferred {
            let Some(&b) = uses.get(&placeholder) else { continue };
            dst.m.set_block(Block(b as u32));
            let terminator = dst.m.blocks[b].pop()?;
            let previous = dst.m.blocks[b].len();
            let slot = slots[&v];
            let words = (slot..slot + self.transport_words(v)?)
                .map(|k| {
                    let at = scratch.at(&mut dst.m, k);
                    dst.raw_load(scratch.uav, at, 1)[0]
                })
                .collect::<Vec<_>>();
            let value = self.unpack(v, &words, &mut dst.m);
            let added = dst.m.blocks[b].split_off(previous);
            // In block zero the scratch address is materialized before the switch.
            // At every other block entry only the leading phis must precede a load.
            let at = if b == 0 {
                previous
            } else {
                dst.m.blocks[b].iter().take_while(|&&i| matches!(dst.m.insts[i as usize].op, Op::Phi(_))).count()
            };
            dst.m.blocks[b].splice(at..at, added);
            dst.m.blocks[b].push(terminator);
            replacements.insert(placeholder, value);
        }
        dst.m.replace_uses(&replacements);
        Some(())
    }

    fn sink_cheap_piece_ops(&mut self) {
        // Only sink a cheap definition whose nonconstant inputs are already used in
        // the destination block. Moving it cannot introduce a new live-in there.
        let (order, _) = self.block_layout();
        let mut uses: HashMap<Value, BTreeSet<usize>> = HashMap::new();
        for &b in &order {
            for &id in &self.blocks[b] {
                let op = &self.insts[id as usize].op;
                if let Op::Phi(inc) = op {
                    for &(v, p) in inc {
                        uses.entry(v).or_default().insert(p.0 as usize);
                    }
                } else {
                    op.for_each_operand(|v| {
                        uses.entry(v).or_default().insert(b);
                    });
                }
            }
        }
        // Moving a producer and then a consumer based on stale use sets may violate
        // dominance. Limit this pass to candidates whose consumers stay in place.
        let eligible = |op: &Op| matches!(op, Op::Cast(CastOp::BitCast | CastOp::Trunc, ..) | Op::Select(..));
        let mut feeds_candidate = HashSet::new();
        for def in &self.insts {
            if eligible(&def.op) {
                def.op.for_each_operand(|v| {
                    feeds_candidate.insert(v);
                });
            }
        }
        for &b in &order {
            for id in self.blocks[b].clone() {
                let val = Value::Inst(id);
                if !eligible(&self.insts[id as usize].op) || feeds_candidate.contains(&val) {
                    continue;
                }
                let Some(sites) = uses.get(&val) else { continue };
                if sites.len() != 1 {
                    continue;
                }
                let t = *sites.first().unwrap();
                if t == b {
                    continue;
                }
                let mut shared = true;
                self.insts[id as usize].op.for_each_operand(|v| {
                    if let Value::Inst(_) = v {
                        shared &= uses.get(&v).is_some_and(|vs| vs.contains(&t))
                            && self.def_block[inst(v).unwrap()].0 as usize != t;
                    }
                });
                if !shared {
                    continue;
                }
                let at =
                    self.blocks[t].iter().take_while(|&&i| matches!(self.insts[i as usize].op, Op::Phi(_))).count();
                self.blocks[b].retain(|&i| i != id);
                self.blocks[t].insert(at, id);
                self.def_block[id as usize] = Block(t as u32);
            }
        }
    }

    /// Only an index is transported. Resource handles and TGSM pointers are
    /// created inside the receiving dispatch, using their original properties.
    fn indexed(&self, v: Value) -> Option<Value> {
        let op = &self.insts[inst(v)?].op;
        if let Op::Gep(Value::Global(g), indices) = op {
            if self.globals[*g as usize].name == "xcl_xs" && indices.len() == 2 && self.const_u64(indices[0]) == Some(0) {
                return Some(indices[1]);
            }
        }
        if let Op::Call(f, a) = op {
            if self.funcs[*f as usize].name == "dx.op.annotateHandle" && matches!(a[2], Value::Const(_)) {
                if let Op::Call(create, args) = &self.insts[inst(a[1])?].op {
                    if self.funcs[*create as usize].name == "dx.op.createHandleFromHeap" && args[2..].iter().all(|v| matches!(v, Value::Const(_))) {
                        return Some(args[1]);
                    }
                }
            }
        }
        None
    }

    fn transport_words(&self, v: Value) -> Option<u32> {
        let def = &self.insts[inst(v)?];
        match &self.types[def.ty.0 as usize] {
            TypeDef::Int(1..=32) | TypeDef::Float => Some(1),
            TypeDef::Int(64) => Some(2),
            TypeDef::Struct(..) if self.static_handle(v) => Some(0),
            _ if self.indexed(v).is_some() => Some(1),
            _ => None,
        }
    }

    fn static_handle(&self, v: Value) -> bool {
        let Value::Inst(i) = v else { return true };
        let Op::Call(f, args) = &self.insts[i as usize].op else { return false };
        match self.funcs[*f as usize].name.as_str() {
            "dx.op.createHandleFromBinding" => args.iter().all(|v| matches!(v, Value::Const(_))),
            "dx.op.annotateHandle" => args.iter().all(|&v| self.static_handle(v)),
            _ => false,
        }
    }

    fn rebuild_static(&self, v: Value, dst: &mut Module, map: &mut HashMap<Value, Value>) -> Option<Value> {
        let Value::Inst(id) = v else { return Some(v) };
        if let Some(&old) = map.get(&v) { return Some(old); }
        let Op::Call(f, args) = &self.insts[id as usize].op else { return None };
        let args: Vec<_> = args.iter().map(|&a| self.rebuild_static(a, dst, map)).collect::<Option<_>>()?;
        let new = dst.call(*f, &args);
        map.insert(v, new);
        Some(new)
    }

    fn pack(&self, original: Value, v: Value, dst: &mut Module) -> Vec<Value> {
        if let Some(index) = dst.indexed(v) { return vec![index]; }
        let ty = self.insts[inst(original).unwrap()].ty;
        let word = dst.i32();
        match self.types[ty.0 as usize] {
            TypeDef::Int(32) => vec![v],
            TypeDef::Int(64) => {
                let shift = dst.const_int(ty, 32);
                let high = dst.bin(BinOp::LShr, v, shift);
                vec![dst.cast(CastOp::Trunc, v, word), dst.cast(CastOp::Trunc, high, word)]
            }
            TypeDef::Float => vec![dst.bitcast(v, word)],
            _ => vec![dst.cast(CastOp::ZExt, v, word)],
        }
    }

    fn unpack(&self, original: Value, words: &[Value], dst: &mut Module) -> Value {
        if self.indexed(original).is_some() {
            match &self.insts[inst(original).unwrap()].op {
                Op::Gep(base, indices) => return dst.gep(*base, &[indices[0], words[0]]),
                Op::Call(f, annotate) => {
                    let Op::Call(create, args) = &self.insts[inst(annotate[1]).unwrap()].op else { unreachable!() };
                    let handle = dst.call(*create, &[args[0], words[0], args[2], args[3]]);
                    return dst.call(*f, &[annotate[0], handle, annotate[2]]);
                }
                _ => unreachable!(),
            }
        }
        let ty = self.insts[inst(original).unwrap()].ty;
        match self.types[ty.0 as usize] {
            TypeDef::Int(32) => words[0],
            TypeDef::Int(64) => {
                let low = dst.cast(CastOp::ZExt, words[0], ty);
                let high = dst.cast(CastOp::ZExt, words[1], ty);
                let shift = dst.const_int(ty, 32);
                let shifted = dst.bin(BinOp::Shl, high, shift);
                dst.bin(BinOp::Or, low, shifted)
            }
            TypeDef::Float => dst.bitcast(words[0], ty),
            _ => dst.cast(CastOp::Trunc, words[0], ty),
        }
    }
}

impl Module {
    /// Preserve symbol and type indices when copying instructions into a
    /// piece. The destination has its own blocks, instructions and metadata.
    pub(in crate::gpu::dxil) fn declarations(&self) -> Self {
        Self {
            types: self.types.clone(),
            type_ids: self.type_ids.clone(),
            consts: self.consts.clone(),
            const_ids: self.const_ids.clone(),
            funcs: self.funcs.clone(),
            func_ids: self.func_ids.clone(),
            globals: self.globals.clone(),
            ..Self::new()
        }
    }

    pub(crate) fn dispatch_plan(&self, limit: usize) -> Option<Plan> {
        let (order, _) = self.block_layout();
        let count = |b: usize| self.blocks[b].len();
        if limit == 0 || order.iter().copied().map(count).sum::<usize>() <= limit {
            return None;
        }
        let mut predecessors = vec![Vec::new(); self.blocks.len()];
        for &b in &order {
            for target in self.successors(b) {
                predecessors[target.0 as usize].push(b);
            }
        }

        // In reverse postorder of the original graph, flooding the transpose
        // visits one component at a time in topological order (Kosaraju).
        let mut visited = vec![false; self.blocks.len()];
        let mut blocks = vec![Vec::new()];
        let mut stages = vec![usize::MAX; self.blocks.len()];
        let mut used = 0;
        for &root in &order {
            if visited[root] {
                continue;
            }
            let mut component = Vec::new();
            let mut cost = 0;
            let mut stack = vec![root];
            visited[root] = true;
            while let Some(b) = stack.pop() {
                cost += count(b);
                component.push(b);
                for &p in &predecessors[b] {
                    if !visited[p] {
                        visited[p] = true;
                        stack.push(p);
                    }
                }
            }
            // The target is soft for an indivisible component. A component
            // above it occupies a piece by itself; bound that exception so a
            // very large loop still selects the whole pipeline.
            if cost > limit.saturating_mul(4) {
                return None;
            }
            if used != 0 && used + cost > limit {
                blocks.push(Vec::new());
                used = 0;
            }
            used += cost;
            for b in component {
                stages[b] = blocks.len() - 1;
            }
        }

        if blocks.len() == 1 { return None; }

        let mut crossing = BTreeSet::new();
        for &b in &order {
            let from = Block(b as u32);
            blocks[stages[b]].push(from);
            for to in self.successors(b) {
                if stages[b] != stages[to.0 as usize] {
                    debug_assert!(stages[b] < stages[to.0 as usize]);
                    crossing.insert((from, to));
                }
            }
        }
        let mut live_in = vec![HashSet::new(); self.blocks.len()];
        let mut phi_input = HashMap::<_, BTreeSet<_>>::new();
        let mut work = Vec::new();
        for &b in &order {
            for &inst in &self.blocks[b] {
                let op = &self.insts[inst as usize].op;
                // A phi reads each operand on its incoming edge, including
                // when the definition is its predecessor's last instruction.
                if let Op::Phi(inputs) = op {
                    for &(value, pred) in inputs {
                        if let Value::Inst(i) = value {
                            let edge = (pred, Block(b as u32));
                            if crossing.contains(&edge) {
                                phi_input.entry(edge).or_default().insert(value);
                            }
                            if self.def_block[i as usize] != pred {
                                work.push((pred.0 as usize, i));
                            }
                        }
                    }
                } else {
                    op.for_each_operand(|v| {
                        if let Value::Inst(i) = v {
                            if self.def_block[i as usize].0 as usize != b {
                                work.push((b, i));
                            }
                        }
                    });
                }
            }
        }
        while let Some((b, value)) = work.pop() {
            if live_in[b].insert(value) {
                for &p in &predecessors[b] {
                    if self.def_block[value as usize].0 as usize != p {
                        work.push((p, value));
                    }
                }
            }
        }
        let edges = crossing.into_iter().map(|(from, to)| {
            let mut live = phi_input.remove(&(from, to)).unwrap_or_default();
            live.extend(live_in[to.0 as usize].iter().copied().map(Value::Inst));
            Edge { from, to, live: live.into_iter().collect() }
        }).collect();
        Some(Plan { blocks, edges })
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::{
        dxil::{kernel::{Entry, KernelOptions}, lower::Lw, module::{BinOp, CastOp, Pred}},
        recomp::{runner, CsInterface},
    };

    #[test]
    fn oversized_loop_occupies_one_piece_up_to_the_hard_ceiling() {
        use super::{Block, Module};

        let mut m = Module::new();
        let entry = m.entry_block();
        let header = m.new_block();
        let body = m.new_block();
        let exit = m.new_block();
        let word = m.i32();
        let zero = m.i32c(0);
        let one = m.i32c(1);
        m.br(header);
        m.set_block(header);
        let count = m.phi(word);
        m.add_incoming(count, zero, entry);
        let more = m.cmp(Pred::ULt, count, one);
        m.cond_br(more, body, exit);
        m.set_block(body);
        let mut next = count;
        for _ in 0..12 { next = m.bin(BinOp::Add, next, one); }
        m.add_incoming(count, next, body);
        m.br(header);
        m.set_block(exit);
        m.ret();

        // Header (3) + body (13) is exactly four times a target of 4.
        let plan = m.dispatch_plan(4).expect("keep the loop in its own piece");
        assert_eq!(plan.blocks, [vec![entry], vec![header, body], vec![exit]]);
        assert!(m.dispatch_plan(3).is_none());

        // An oversized first block must not create an empty leading piece.
        // If the program fits wholly in this one component, splitting has no benefit.
        let mut m = Module::new();
        let one = m.i32c(1);
        for _ in 0..4 { m.bin(BinOp::Add, one, one); }
        let end = m.new_block();
        m.br(end);
        m.set_block(end);
        m.ret();
        assert_eq!(m.dispatch_plan(4).unwrap().blocks, [vec![Block(0)], vec![end]]);
        let mut m = Module::new();
        let one = m.i32c(1);
        for _ in 0..4 { m.bin(BinOp::Add, one, one); }
        m.ret();
        assert!(m.dispatch_plan(4).is_none());
    }

    fn compare(threads: u32, extra_adds: u32) {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let geometry = [threads, 1, 1];
        let split = threads == 64;
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let options = KernelOptions { wave64: split, split, mem_windows: 1, sampler_base: 0 };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let group = lw.s.group_id(0);
        let count = lw.c(threads);
        let base = lw.mul(group, count);
        let index = lw.add(base, tid);
        let offset = lw.shlc(index, 2);
        let original = lw.s.raw_load(mem, offset, 1)[0];
        // Two different dynamic dwords cross every cut as one i64. With
        // slot-major scratch the high word is a wave width past the low word.
        let i64_ty = lw.s.m.i64();
        let lo = lw.s.m.cast(CastOp::ZExt, original, i64_ty);
        let hi = lw.s.m.cast(CastOp::ZExt, tid, i64_ty);
        let shift = lw.s.m.const_int(i64_ty, 32);
        let hi = lw.s.m.bin(BinOp::Shl, hi, shift);
        let pair = lw.s.m.bin(BinOp::Or, lo, hi);
        let trips = lw.andc(tid, 3);
        let trips = lw.addc(trips, 1);
        let mut value = original;
        let i32 = lw.s.m.i32();
        let zero = lw.c(0);

        // Each loop both consumes the previous loop's result and reuses
        // the original per-lane input. All lanes eventually reconverge.
        let mut large_loop = None;
        for round in 1..=16 {
            let pred = lw.cur();
            let body = lw.s.m.new_block();
            let exit = lw.s.m.new_block();
            lw.s.m.br(body);
            lw.s.m.set_block(body);
            let counter = lw.s.m.phi(i32);
            let acc = lw.s.m.phi(i32);
            lw.s.m.add_incoming(counter, zero, pred);
            lw.s.m.add_incoming(acc, value, pred);
            let next = lw.addc(counter, 1);
            value = lw.add(acc, original);
            value = lw.addc(value, round);
            if round == 8 {
                for _ in 0..extra_adds { value = lw.add(value, original); }
                if extra_adds > 0 { large_loop = Some(body); }
            }
            lw.s.m.add_incoming(counter, next, body);
            lw.s.m.add_incoming(acc, value, body);
            let again = lw.cmp(Pred::ULt, next, trips);
            lw.s.m.cond_br(again, body, exit);
            lw.s.m.set_block(exit);
            if split && round == 8 {
                let other_half = lw.exchange(lw.half);
                value = lw.add(value, other_half);
            }
        }
        let pair_lo = lw.s.m.cast(CastOp::Trunc, pair, i32);
        let high = lw.s.m.bin(BinOp::LShr, pair, shift);
        let pair_hi = lw.s.m.cast(CastOp::Trunc, high, i32);
        let both = lw.xor(pair_lo, pair_hi);
        value = lw.add(value, both);
        lw.s.raw_store(mem, offset, &[value]);
        lw.s.m.ret();
        lw.s.m.optimize();
        if let Some(oversized) = large_loop {
            let n = lw.s.m.block_insts(oversized).len();
            assert!(n > 40 && n <= 160);
            let plan = lw.s.m.dispatch_plan(40).expect("partition around oversized loop");
            assert!(plan.blocks.iter().any(|piece| piece == &[oversized]));
        }
        let chain = lw.s.split_compute(geometry, split.then_some(tid), 40);
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        crate::gpu::dxil::validate::validate(&whole).expect("valid unsplit fixture");
        let chain = chain.expect("split the synthetic program");
        assert!(chain.pieces.len() >= 3);
        let initial: Vec<_> = (1..=threads * 3).collect();
        let expected: Vec<_> = initial.iter().enumerate().map(|(i, &v)| {
            let n = (i as u32 % 4) + 1;
            v + n * ((16 + extra_adds) * v + 136) + u32::from(split && i as u32 % threads < 32) + (v ^ (i as u32 % threads))
        }).collect();
        let reference = runner::dispatch(&whole, [3, 1, 1], &[], &initial).unwrap();
        assert_eq!(reference, expected);
        let result = runner::dispatch_split(&chain, [3, 1, 1], &initial).unwrap();
        assert_eq!(result, reference);
    }

    fn check_persistent_payload(reuse_hole: bool) {
        use super::Value;

        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        fn mix(lw: &mut Lw, mut v: Value, seed: u32) -> Value {
            let multiplier = lw.c(33);
            for i in 1..=12 {
                let key = lw.c(seed + i);
                v = lw.xor(v, key);
                v = lw.mul(v, multiplier);
            }
            v
        }
        fn expected_mix(v: u32, seed: u32) -> u32 {
            (1..=12).fold(v, |x, i| (x ^ (seed + i)).wrapping_mul(33))
        }

        let geometry = [32, 1, 1];
        let options = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let offset = lw.shlc(tid, 2);
        let first = lw.s.raw_load(mem, offset, 1)[0];
        let saved = mix(&mut lw, first, 10);
        let middle = lw.s.m.new_block();
        let last = lw.s.m.new_block();
        // every lane takes the middle block; the branch only keeps the optimiser from merging it
        let entry = lw.cur();
        let width = lw.s.wave_lane_count();
        let taken = lw.cmp(Pred::ULt, tid, width);
        lw.s.m.cond_br(taken, middle, last);
        lw.s.m.set_block(middle);
        let replacement = mix(&mut lw, if reuse_hole { first } else { tid }, 30);
        lw.s.raw_store(mem, offset, &[replacement]);
        lw.s.m.br(last);
        lw.s.m.set_block(last);
        let i32 = lw.s.m.i32();
        let arrived = lw.s.m.phi(i32);
        lw.s.m.add_incoming(arrived, replacement, middle);
        lw.s.m.add_incoming(arrived, first, entry);
        let input = if reuse_hole { lw.xor(saved, arrived) } else { saved };
        let output = mix(&mut lw, input, 50);
        lw.s.raw_store(mem, offset, &[output]);
        lw.s.m.ret();
        lw.s.m.optimize();

        let limit = lw.s.m.blocks.iter().map(Vec::len).max().unwrap();
        let plan = lw.s.m.dispatch_plan(limit).unwrap();
        assert_eq!(plan.blocks.len(), 3);
        let incoming = plan.edges.iter().find(|edge| edge.to == middle).unwrap();
        let outgoing = plan.edges.iter().find(|edge| edge.from == middle && edge.to == last).unwrap();
        assert!(incoming.live.contains(&saved) && outgoing.live.contains(&saved));
        if reuse_hole {
            // In compact per-edge order, the replacement occupies saved's old
            // slot. A delayed load of saved from that slot would get replacement.
            let slot = |edge: &super::Edge, v| {
                edge.live.iter().filter(|&&x| lw.s.m.transport_words(x) == Some(1)).position(|&x| x == v).unwrap()
            };
            assert_eq!(slot(incoming, saved), slot(outgoing, replacement));
        }

        let chain = lw.s.split_compute(geometry, None, limit).unwrap();
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        let initial: Vec<u32> = (0..32).map(|i| 1009 * i + 77).collect();
        let mut expected: Vec<_> = initial.iter().map(|&v| {
            let saved = expected_mix(v, 10);
            expected_mix(if reuse_hole { saved ^ expected_mix(v, 30) } else { saved }, 50)
        }).collect();
        // The runners read back at least 64 words. Only lanes 0..32 write.
        expected.resize(64, 0);
        assert_eq!(runner::dispatch(&whole, [1, 1, 1], &[], &initial).unwrap(), expected);
        assert_eq!(runner::dispatch_split(&chain, [1, 1, 1], &initial).unwrap(), expected);
    }

    /// A typed UAV load in one piece only: the whole program declares its flag, the other pieces
    /// must not (the validator wants flags to match usage, and a refused piece leaves it whole).
    #[test]
    fn dispatch_chain_declares_typed_uav_loads_only_where_they_are() {
        use crate::gpu::dxil::ops::{CompType, TexKind};

        let geometry = [32, 1, 1];
        let options = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let offset = lw.shlc(tid, 2);
        let first = lw.s.raw_load(mem, offset, 1)[0];
        let entry = lw.cur();
        let middle = lw.s.m.new_block();
        let last = lw.s.m.new_block();
        let width = lw.s.wave_lane_count();
        let taken = lw.cmp(Pred::ULt, tid, width);
        lw.s.m.cond_br(taken, middle, last);
        lw.s.m.set_block(middle);
        let slot = lw.s.m.i32c(9);
        let texture = lw.s.heap_texture(slot, TexKind::D2, true, CompType::F32, 4, false);
        lw.s.typed_uav_load_additional_formats();
        let texel = lw.s.texture_load(texture, None, &[tid, tid], &[]);
        let i32 = lw.s.m.i32();
        let bits = lw.s.m.cast(CastOp::BitCast, texel[1], i32);
        let loaded = lw.xor(first, bits);
        lw.s.m.br(last);
        lw.s.m.set_block(last);
        let value = lw.s.m.phi(i32);
        lw.s.m.add_incoming(value, loaded, middle);
        lw.s.m.add_incoming(value, first, entry);
        lw.s.raw_store(mem, offset, &[value]);
        lw.s.m.ret();
        lw.s.m.optimize();
        let limit = lw.s.m.blocks.iter().map(Vec::len).max().unwrap();
        let chain = lw.s.split_compute(geometry, None, limit).expect("every piece validates");
        assert!(chain.pieces.len() >= 2);
    }

    #[test]
    fn dispatch_chain_payload_skips_middle_piece() { check_persistent_payload(false); }

    #[test]
    fn dispatch_chain_payload_survives_reused_edge_slot() { check_persistent_payload(true); }

    #[test]
    fn dispatch_chain_wave32_live_loops() { compare(32, 0); }

    /// A guest wave64 split across two hardware wave32s.
    #[test]
    fn dispatch_chain_wave64_live_loops_and_exchange() { compare(64, 0); }
    #[test]
    fn dispatch_chain_oversized_loop_matches_whole() {
        compare(32, 48);
        compare(64, 48);
    }

    /// Early exits and forward skips retain their own continuation tags; group
    /// indexing must keep every lane separate across all three dispatch axes.
    #[test]
    fn dispatch_chain_divergent_forward_edges_3d() {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        println!("CHAIN_GPU={}", runner::adapter().unwrap());
        for threads in [32, 64] {
            let geometry = [threads, 1, 1];
            let split = threads == 64;
            let cs = CsInterface { threads: geometry, ..Default::default() };
            let options = KernelOptions { wave64: split, split, mem_windows: 1, sampler_base: 0 };
            let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
            let (tid, _) = lw.setup_lanes_compute(geometry, false);
            let mem = lw.mems[0];
            let [x, y, z] = [0, 1, 2].map(|axis| lw.s.group_id(axis));
            let three = lw.c(3);
            let row = lw.mul(z, three);
            let row = lw.add(row, y);
            let row = lw.shlc(row, 1);
            let group = lw.add(row, x);
            let count = lw.c(threads);
            let base = lw.mul(group, count);
            let index = lw.add(base, tid);
            let offset = lw.shlc(index, 2);
            let original = lw.s.raw_load(mem, offset, 1)[0];
            let mode = lw.andc(tid, 3);
            let trips = lw.addc(mode, 1);
            let zero = lw.c(0);
            let one = lw.c(1);
            let i32 = lw.s.m.i32();
            let early = lw.s.m.new_block();
            let choose = lw.s.m.new_block();
            let first = lw.s.m.new_block();
            let middle = lw.s.m.new_block();
            let second = lw.s.m.new_block();
            let done = lw.s.m.new_block();
            let exits = lw.cmp(Pred::Eq, mode, zero);
            lw.s.m.cond_br(exits, early, choose);
            lw.s.m.set_block(early);
            let marker = lw.addc(index, 100_000);
            lw.s.raw_store(mem, offset, &[marker]);
            lw.s.m.ret();
            lw.s.m.set_block(choose);
            let skips = lw.cmp(Pred::Eq, mode, one);
            lw.s.m.cond_br(skips, middle, first);
            lw.s.m.set_block(first);
            let mut value = original;
            let mut middle_value = None;
            for round in 1..=16 {
                if round == 9 {
                    let previous = lw.cur();
                    lw.s.m.br(middle);
                    lw.s.m.set_block(middle);
                    let joined = lw.s.m.phi(i32);
                    lw.s.m.add_incoming(joined, original, choose);
                    lw.s.m.add_incoming(joined, value, previous);
                    value = joined;
                    middle_value = Some(joined);
                    let stops = lw.cmp(Pred::Eq, mode, three);
                    lw.s.m.cond_br(stops, done, second);
                    lw.s.m.set_block(second);
                }
                let pred = lw.cur();
                let body = lw.s.m.new_block();
                let exit = lw.s.m.new_block();
                lw.s.m.br(body);
                lw.s.m.set_block(body);
                let counter = lw.s.m.phi(i32);
                let acc = lw.s.m.phi(i32);
                lw.s.m.add_incoming(counter, zero, pred);
                lw.s.m.add_incoming(acc, value, pred);
                let next = lw.addc(counter, 1);
                value = lw.add(acc, original);
                value = lw.addc(value, round);
                lw.s.m.add_incoming(counter, next, body);
                lw.s.m.add_incoming(acc, value, body);
                let again = lw.cmp(Pred::ULt, next, trips);
                lw.s.m.cond_br(again, body, exit);
                lw.s.m.set_block(exit);
            }
            let last = lw.cur();
            lw.s.m.br(done);
            lw.s.m.set_block(done);
            let final_value = lw.s.m.phi(i32);
            lw.s.m.add_incoming(final_value, middle_value.unwrap(), middle);
            lw.s.m.add_incoming(final_value, value, last);
            lw.s.raw_store(mem, offset, &[final_value]);
            lw.s.m.ret();
            lw.s.m.optimize();
            let plan = lw.s.m.dispatch_plan(64).expect("partition branch fixture");
            let stage = |b| plan.blocks.iter().position(|blocks| blocks.contains(&b)).unwrap();
            assert!(plan.edges.iter().any(|edge| stage(edge.to) > stage(edge.from) + 1), "exercise a forward skip over an intermediate piece");
            let chain = lw.s.split_compute(geometry, split.then_some(tid), 64).expect("split branch fixture");
            assert!(chain.pieces.len() >= 3);
            let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
            crate::gpu::dxil::validate::validate(&whole).expect("valid unsplit fixture");
            let words = (threads * 12) as usize;
            let initial: Vec<_> = (0..words + 16).map(|i| if i < words { i as u32 + 17 } else { 0xdeadbeef }).collect();
            let expected: Vec<_> = initial.iter().enumerate().map(|(i, &v)| {
                if i >= words { return v; }
                match i as u32 % 4 {
                    0 => 100_000 + i as u32,
                    1 => v + 2 * (8 * v + 100),
                    2 => v + 3 * (16 * v + 136),
                    _ => v + 4 * (8 * v + 36),
                }
            }).collect();
            let reference = runner::dispatch(&whole, [2, 3, 2], &[], &initial).unwrap();
            assert_eq!(reference, expected, "unsplit {threads}");
            let result = runner::dispatch_split(&chain, [2, 3, 2], &initial).unwrap();
            assert_eq!(result, expected, "split {threads}");
            // in slabs: single groups, a row (and a part of one), a plane, all
            for most in [1, 2, 3, 6, 7, 12] {
                let result = runner::dispatch_split_by(&chain, [2, 3, 2], &initial, most).unwrap();
                assert_eq!(result, expected, "split {threads}, {most} groups at once");
            }
            println!("CHAIN_BRANCH threads={threads} pieces={} scratch_words_per_group={} checked_words={words} guards=16", chain.pieces.len(), chain.dwords_per_group);
        }
    }

}
