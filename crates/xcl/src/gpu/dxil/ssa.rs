//! On-the-fly SSA construction (Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau:
//! "Simple and Efficient Construction of Static Single Assignment Form", CC 2013)
//! over the module's basic blocks. Variables are register dwords (SGPRs, SCC,
//! VCC, M0, EXEC, VGPRs) numbered by the translator.
//!
//! Blocks are registered as they are created; predecessors are added as the
//! branches are emitted. A block whose predecessors are not all known yet is
//! *unsealed*: reads there create incomplete phis, completed on [`Ssa::seal`].
//! A read that reaches a block without predecessors asks the caller for the
//! variable's initial value (the kernel loads it from `io` in the entry block).

use std::collections::HashMap;

use super::module::{Block, Ty, Value};
use super::shader::Shader;

/// Multiplicative hash for the small integer keys of the definition maps (SipHash
/// dominated the reads of long programs).
#[derive(Default, Clone, Copy)]
struct VarHasher(u64);

impl std::hash::Hasher for VarHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.0 = (self.0.rotate_left(5) ^ v as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type VarMap = HashMap<u32, Value, std::hash::BuildHasherDefault<VarHasher>>;

#[derive(Default)]
struct SBlock {
    defs: VarMap,
    preds: Vec<Block>,
    sealed: bool,
    /// predecessors to wait for before sealing automatically (None: sealed explicitly)
    expect: Option<usize>,
    incomplete: Vec<(u32, Value)>,
    /// a region join ([`Ssa::seal_region_join`]): the region's last block and the
    /// variables it writes
    region: Option<(Block, Vec<u32>)>,
}

pub struct Ssa {
    blocks: Vec<SBlock>,
    ty: Ty,
    /// (block, variable) of the sealed joins whose operands are being read, with the
    /// phi created if a read came back around to the join (a loop)
    visiting: HashMap<(u32, u32), Option<Value>>,
    /// open write logs (innermost last)
    logs: Vec<Vec<u32>>,
}

impl Ssa {
    /// All variables have type `ty`.
    pub fn new(ty: Ty) -> Ssa {
        Ssa { blocks: Vec::new(), ty, visiting: HashMap::new(), logs: Vec::new() }
    }

    fn at(&mut self, b: Block) -> &mut SBlock {
        let i = b.0 as usize;
        if self.blocks.len() <= i {
            self.blocks.resize_with(i + 1, SBlock::default);
        }
        &mut self.blocks[i]
    }

    /// `b` seals itself once `n` predecessors have been added.
    pub fn expect_preds(&mut self, s: &mut Shader, b: Block, n: usize, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        self.at(b).expect = Some(n);
        if self.at(b).preds.len() >= n && !self.at(b).sealed {
            self.seal(s, b, init);
        }
    }

    pub fn add_pred(&mut self, s: &mut Shader, b: Block, p: Block, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        let sb = self.at(b);
        assert!(!sb.sealed, "edge into the sealed block {b:?}");
        sb.preds.push(p);
        if let Some(n) = sb.expect {
            if sb.preds.len() >= n {
                self.seal(s, b, init);
            }
        }
    }

    pub fn is_sealed(&mut self, b: Block) -> bool {
        self.at(b).sealed
    }

    pub fn preds(&mut self, b: Block) -> Vec<Block> {
        self.at(b).preds.clone()
    }

    /// A definition by the program (logged for an open [`Ssa::begin_log`]).
    pub fn write(&mut self, b: Block, var: u32, v: Value) {
        if let Some(log) = self.logs.last_mut() {
            log.push(var);
        }
        self.at(b).defs.insert(var, v);
    }

    /// A read's result remembered in `b` (not a definition).
    fn cache(&mut self, b: Block, var: u32, v: Value) {
        self.at(b).defs.insert(var, v);
    }

    /// Starts logging the variables written (e.g. in a guarded region).
    pub fn begin_log(&mut self) {
        self.logs.push(Vec::new());
    }

    /// Ends the innermost log: the variables written since `begin_log` (also
    /// counted in the enclosing log).
    pub fn end_log(&mut self) -> Vec<u32> {
        let mut log = self.logs.pop().expect("begin_log");
        log.sort_unstable();
        log.dedup();
        if let Some(outer) = self.logs.last_mut() {
            outer.extend_from_slice(&log);
        }
        log
    }

    /// Seals the join `j` of a region `cond → (then … end) → j` / `cond → j` whose
    /// only writes are `written` (sorted): a written variable merges over both edges,
    /// every other one reads through `cond` directly, as if `cond` were `j`'s only
    /// predecessor. Reads then skip the region instead of walking both edges of every
    /// guard (guarded memory accesses make thousands of such joins).
    pub fn seal_region_join(&mut self, j: Block, cond: Block, end: Block, written: Vec<u32>) {
        let sb = self.at(j);
        assert!(!sb.sealed && sb.incomplete.is_empty(), "region join already read");
        sb.preds = vec![cond];
        sb.sealed = true;
        if !written.is_empty() {
            sb.region = Some((end, written));
        }
    }

    /// The predecessors a read of `var` in `b` merges over.
    fn preds_of(&mut self, b: Block, var: u32) -> Vec<Block> {
        let sb = self.at(b);
        match &sb.region {
            Some((end, w)) if w.binary_search(&var).is_ok() => vec![sb.preds[0], *end],
            _ => sb.preds.clone(),
        }
    }

    /// The definition of `var` reaching the current end of block `b`.
    pub fn read(&mut self, s: &mut Shader, b: Block, var: u32, init: &mut dyn FnMut(&mut Shader, u32) -> Value) -> Value {
        if let Some(&v) = self.at(b).defs.get(&var) {
            return v;
        }
        // chains of sealed single-predecessor blocks (guarded memory accesses make long ones)
        // are walked iteratively, not recursively
        let mut chain = Vec::new();
        let mut b = b;
        loop {
            let sb = self.at(b);
            if !sb.sealed || sb.preds.len() != 1 || sb.region.as_ref().is_some_and(|(_, w)| w.binary_search(&var).is_ok()) {
                break;
            }
            chain.push(b);
            b = sb.preds[0];
            if let Some(&v) = self.at(b).defs.get(&var) {
                    for c in chain {
                    self.cache(c, var, v);
                }
                return v;
            }
        }
        if !chain.is_empty() {
            let v = self.read(s, b, var, init);
            for c in chain {
                self.cache(c, var, v);
            }
            return v;
        }
        let v = if !self.at(b).sealed {
            let ty = self.ty;
            let phi = s.m.phi_at(b, ty);
            self.at(b).incomplete.push((var, phi));
            phi
        } else {
            let preds = self.preds_of(b, var);
            match preds.len() {
                0 => init(s, var),
                1 => self.read(s, preds[0], var, init),
                _ => {
                    // the operands first, without a phi: most joins a read walks through
                    // (guarded accesses make thousands) see one value from every edge
                    // (Braun et al.'s trivial phi) and need none. A read that comes back
                    // here around a loop gets a phi, which then stays.
                    let key = (b.0, var);
                    if let Some(slot) = self.visiting.get(&key).copied() {
                        let phi = slot.unwrap_or_else(|| s.m.detached_phi(self.ty));
                        self.visiting.insert(key, Some(phi));
                        return phi;
                    }
                    self.visiting.insert(key, None);
                    let vals: Vec<Value> = preds.iter().map(|&p| self.read(s, p, var, init)).collect();
                    let marker = self.visiting.remove(&key).flatten();
                    if marker.is_none() && vals.iter().all(|&v| v == vals[0]) {
                        vals[0]
                    } else {
                        let phi = marker.unwrap_or_else(|| s.m.detached_phi(self.ty));
                        for (&v, &p) in vals.iter().zip(&preds) {
                            s.m.add_incoming(phi, v, p);
                        }
                        s.m.place_phi(b, phi);
                        phi
                    }
                }
            }
        };
        self.cache(b, var, v);
        v
    }

    fn add_operands(&mut self, s: &mut Shader, phi: Value, preds: &[Block], var: u32, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        for &p in preds {
            let v = self.read(s, p, var, init);
            s.m.add_incoming(phi, v, p);
        }
    }

    pub fn seal(&mut self, s: &mut Shader, b: Block, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        if self.at(b).sealed {
            return;
        }
        let preds = self.at(b).preds.clone();
        let inc = std::mem::take(&mut self.at(b).incomplete);
        for (var, phi) in inc {
            self.add_operands(s, phi, &preds, var, init);
        }
        self.at(b).sealed = true;
    }
}
