//! RDNA2 instruction → DXIL lowering for the direct compute translator.
//!
//! The model is the recompiler's (`recomp/emit.rs`, the semantic reference
//! for every instruction): each console lane is one thread; SGPRs, SCC, VCC,
//! M0 and EXEC are SSA values every thread of the console wave computes alike;
//! VGPRs are per-thread SSA values whose writes keep the old value in lanes
//! EXEC disables; lane masks come from ballots (`lanes.rs`). Register dwords
//! are SSA variables ([`super::ssa`]): no register arrays, no scratch.
//!
//! Variables: SGPR code `c` (0..=127; slot 125 = SCC, as in the kernel ABI),
//! VGPR `v` = 128 + v, the loop budget = [`VAR_BUDGET`].

use std::collections::{BTreeSet, HashMap};

use super::module::{BinOp, Block, CastOp, Pred, Value};
use super::ops::{BitCount, Binary, FClass, FUnary, Tertiary};
use super::shader::{Feature, RawUav, Shader};
use super::ssa::Ssa;
use crate::gpu::rdna2::{self, Decoded, Encoding, Operand, Variant};

pub const VAR_SCC: u32 = 125;
pub const VAR_VGPR: u32 = 128;
pub const VAR_BUDGET: u32 = 128 + 256;
/// MODE[3:2] (f16 / f64 rounding) when the program changes it (`Lw::dyn_rm`)
pub const VAR_FP16_RM: u32 = VAR_BUDGET + 1;
/// the memory window the next access uses (`xcl_w`)
pub const VAR_WIN: u32 = VAR_BUDGET + 2;
/// split form, several console waves per group: exchanges so far, and the "partner never
/// arrived" flag (`xcl_xk`, `xcl_x_dead`)
pub const VAR_XK: u32 = VAR_BUDGET + 3;
pub const VAR_XDEAD: u32 = VAR_BUDGET + 4;
/// `tw0 + 1` of the last image store / atomic through the k-th T# of `Lw::img_writes` (16
/// slots; the recompiler's `xcl_iw{srsrc}`)
pub const VAR_IMG_WRITE: u32 = VAR_BUDGET + 8;
/// loop iterations a lane may run (the recompiler's `xcl_budget`)
pub const LOOP_BUDGET: u32 = 1 << 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dst {
    V(u64),
    S(u64),
    Null,
}

/// What the translator could not handle: the instruction and why.
pub type LResult<T> = Result<T, String>;

pub(crate) fn err<T>(d: &Decoded, why: &str) -> LResult<T> {
    Err(format!("{}: {why}", rdna2::format(d)))
}

/// Compute-mode initial values (the real CS ABI): fixed SGPRs (group ids, TG_SIZE,
/// SCC, EXEC) and v0..v2 (thread id in the group); other VGPRs start at 0, other SGPRs
/// (user data) come from `io`.
pub struct ComputeInit {
    pub sgpr: HashMap<u32, Value>,
    pub vgpr: HashMap<u32, Value>,
    pub zero: Value,
}

/// Mesh (NGG) first pass: where this group's record and this thread's vertex /
/// primitive slots are in `res` (dwords), and the per-vertex outputs.
pub struct MeshCtx {
    pub rec: Value,
    pub vtx: Value,
    pub prm: Value,
    pub outputs: Vec<crate::gpu::dxbc::SigElement>,
}

/// Initial register values: loaded from `io` in the entry block on first use.
pub struct Init {
    pub io: Option<RawUav>,
    pub compute: Option<ComputeInit>,
    /// byte offset of this lane's VGPRs in `io`
    pub vgpr_base: Option<Value>,
    pub entry: Block,
    pub touched: BTreeSet<u32>,
    /// the compact form: each register's local variable, and the block that loads them now
    pub slots: HashMap<u32, Value>,
    pub mem: Option<Block>,
}

impl Init {
    pub fn load(&mut self, s: &mut Shader, var: u32) -> Value {
        let Some(b) = self.mem else { return self.first(s, var) };
        let p = self.slot(s, var);
        let cur = s.m.current_block();
        let before = s.m.set_insert_before_term(true);
        s.m.set_block(b);
        let v = s.m.load(p, 4);
        s.m.set_block(cur);
        s.m.set_insert_before_term(before);
        v
    }

    /// The local variable of `var`, made in the entry block with the register's first value.
    pub fn slot(&mut self, s: &mut Shader, var: u32) -> Value {
        if let Some(&p) = self.slots.get(&var) {
            return p;
        }
        let v = self.first(s, var);
        let cur = s.m.current_block();
        let before = s.m.set_insert_before_term(false);
        s.m.set_block(self.entry);
        let i32 = s.m.i32();
        let p = s.m.alloca(i32, 4);
        s.m.store(p, v, 4);
        s.m.set_block(cur);
        s.m.set_insert_before_term(before);
        self.slots.insert(var, p);
        p
    }

    fn first(&mut self, s: &mut Shader, var: u32) -> Value {
        if var == VAR_BUDGET {
            return s.m.i32c(LOOP_BUDGET);
        }
        if var == super::lower_round::VAR_F32_MODE {
            return s.m.i32c(super::lower_round::MODE_INIT);
        }
        if var == VAR_FP16_RM || var == VAR_WIN || var == VAR_XK || var == VAR_XDEAD || (VAR_IMG_WRITE..VAR_IMG_WRITE + 16).contains(&var) {
            return s.m.i32c(0);
        }
        if var >= super::lower_mem::VAR_TMP {
            return s.m.i32c(0);
        }
        if let Some(ci) = &self.compute {
            if let Some(&v) = ci.sgpr.get(&var) {
                return v;
            }
            if var >= VAR_VGPR {
                return ci.vgpr.get(&(var - VAR_VGPR)).copied().unwrap_or(ci.zero);
            }
        }
        let cur = s.m.current_block();
        s.m.set_block(self.entry);
        self.touched.insert(var);
        let io = self.io.expect("io");
        let v = if var < VAR_VGPR {
            let off = s.m.i32c(var * 4);
            s.raw_load(io, off, 1)[0]
        } else {
            let base = self.vgpr_base.expect("vgpr base");
            let k = s.m.i32c((var - VAR_VGPR) * 4);
            let off = s.m.bin(BinOp::Add, base, k);
            s.raw_load(io, off, 1)[0]
        };
        s.m.set_block(cur);
        v
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Partner {
    Present,
    /// several console waves per group: the halves pair up through arrival counters
    Spin,
    Missing,
    Unsupported(&'static str),
}

#[derive(Default, Clone, Copy)]
pub struct Mods {
    pub abs: [bool; 3],
    pub neg: [bool; 3],
    pub omod: u64,
    pub clamp: bool,
    pub opsel: u64,
}

pub struct Lw {
    pub s: Shader,
    pub ssa: Ssa,
    pub init: Init,
    pub wave64: bool,
    /// wave64 program in two hardware wave32 halves
    pub split: bool,
    /// console lane (0..63 / 0..31)
    pub lane: Value,
    pub lane31: Value,
    /// lane >= 32 (i1)
    pub upper: Value,
    /// lane >> 5
    pub half: Value,
    /// the memory windows `mem`, `mem1`.. (u2..)
    pub mems: Vec<RawUav>,
    /// the windows come from the descriptor heap (`lower_window`; `mems` only counts them):
    /// Some(globallycoherent)
    pub heap_mem: Option<bool>,
    pub res: Option<RawUav>,
    /// the buffer format table (`FMT_INFO`), created on first use
    pub fmt_table: Option<Value>,
    /// cached decoded format descriptors hoisted to `d3`'s defining block
    pub fmt_desc_cache: HashMap<(Value, Option<u32>), super::lower_mem::FmtDesc>,
    /// cached single-component format decode parameters hoisted to `d3`'s defining block
    pub fmt_comp1_cache: HashMap<(Value, Option<u32>), super::lower_mem::FmtComp1>,
    /// cached `IO_UMA_BASE_HI` loaded once in the entry block for multi-window address math
    pub uma_base_val: Option<Value>,
    /// temporaries handed out (`lower_mem::VAR_TMP`)
    pub ntmp: u32,
    /// the `s_load_dword*` being lowered repeats an earlier one (`reload`): set by the
    /// translator's loop for that instruction
    pub reload: Option<super::reload::Reload>,
    /// the lane's scratch array (an alloca in the entry block)
    pub scratch: Option<Value>,
    /// MIMG instructions (pc) whose S# is built from constants in the code: the HLSL
    /// kernel binds those as static samplers
    pub static_samplers: std::collections::BTreeMap<u64, [u32; 4]>,
    /// the program writes a non-default MODE[3:2] and has f16 rounding ops (no f64 ones):
    /// f16 results round in software under `VAR_FP16_RM` (the recompiler's `dynamic_f16_rm`)
    pub dyn_rm: bool,
    /// the program writes a non-default MODE[1:0]: rounding f32 instructions round by
    /// `lower_round::VAR_F32_MODE` (`lower_round`)
    pub f32_rm: bool,
    pub track_frexp_denorm: bool,
    pub has_f64_round_op: bool,
    /// the T# SGPRs of the program's image stores and atomics (`recomp::emit::image_write_srsrcs`)
    pub img_writes: Vec<u64>,
    /// static samplers bound so far: (S#, comparison, handle) at `s<i>, space1`
    pub static_used: Vec<([u32; 4], bool, super::ops::Sampler)>,
    pub sampler_base: u32,
    pub exec_cache: HashMap<(Block, Value, Value), Value>,
    /// groupshared exchange words of the split form
    /// split form: the halves' exchange words (storage, first index) and the lane counter
    pub xs: Option<(super::ops::GroupShared, u32)>,
    pub xw: Option<(super::ops::GroupShared, u32)>,
    /// arrival counters of the spin exchange (two per console wave)
    pub xc: Option<(super::ops::GroupShared, u32)>,
    /// console waves per group, this thread's console wave and its partner half's lanes
    pub waves: u32,
    pub cw: Value,
    pub xn: Value,
    /// words per half in an exchange
    pub xslot: u32,
    /// the program's LDS (`xcl_lds`, 8192 dwords) and its last usable dword
    pub lds: Option<super::ops::GroupShared>,
    pub lds_top: u32,
    /// DPP write enable of the instruction being lowered
    pub dpp_enable: Option<Value>,
    /// the split form's partner half: present, missing (a console wave of at most 32
    /// threads: its masks / lane reads see 0), or the exchange is not supported here
    pub partner: Partner,
    /// mesh mode: the record slots
    pub mesh: Option<MeshCtx>,
    /// geometry mode (ES+GS)
    pub geom: Option<super::lower_geom::GeomCtx>,
    /// VS / PS entry
    pub gfx: Option<super::lower_gfx::GfxCtx>,
    /// a wave32 program in 64-lane hardware: this hardware half is the upper one (i1)
    pub gfx_half: Option<Value>,
    /// split form: SGPRs whole-live after each instruction (`recomp::x64_masks::by_pc`)
    pub whole_live: std::collections::BTreeMap<u64, u128>,
    /// set when an exchange was needed where it is not supported
    pub exchange_error: Option<String>,
    pub pc: u64,
    /// Lowering under an externally chosen guard ([`Lw::inst_guarded`]): when set,
    /// `act()` (this lane is active: VGPR write selects, guarded memory) returns it
    /// instead of this lane's bit of the SSA EXEC.
    pub guard_override: Option<Value>,
    /// When set, the wave-wide mask cross-lane instructions see (DPP source lanes,
    /// readfirstlane, ds_swizzle / bpermute) instead of the SSA EXEC.
    pub mask_override: Option<(Value, Value)>,
}

impl Lw {
    // ------------------------------------------------------------ SSA access

    pub fn cur(&self) -> Block {
        self.s.m.current_block()
    }

    pub fn rd(&mut self, var: u32) -> Value {
        self.init.touched.insert(var);
        let b = self.cur();
        let Lw { s, ssa, init, .. } = self;
        ssa.read(s, b, var, &mut |s, v| init.load(s, v))
    }

    pub fn wr(&mut self, var: u32, v: Value) {
        self.init.touched.insert(var);
        let b = self.cur();
        self.ssa.write(b, var, v);
    }

    /// Creates a block (registered with the SSA builder, sealing itself after `preds` edges).
    pub fn block(&mut self, preds: Option<usize>) -> Block {
        let b = self.s.m.new_block();
        if let Some(n) = preds {
            let Lw { s, ssa, init, .. } = self;
            ssa.expect_preds(s, b, n, &mut |s, v| init.load(s, v));
        }
        b
    }

    pub fn edge(&mut self, from: Block, to: Block) {
        let Lw { s, ssa, init, .. } = self;
        ssa.add_pred(s, to, from, &mut |s, v| init.load(s, v));
    }

    pub fn br(&mut self, to: Block) {
        let from = self.cur();
        self.s.m.br(to);
        self.edge(from, to);
    }

    pub fn cond_br(&mut self, c: Value, t: Block, f: Block) {
        if t == f {
            return self.br(t);
        }
        let from = self.cur();
        self.s.m.cond_br(c, t, f);
        self.edge(from, t);
        self.edge(from, f);
    }

    pub fn seal(&mut self, b: Block) {
        let Lw { s, ssa, init, .. } = self;
        ssa.seal(s, b, &mut |s, v| init.load(s, v));
    }

    /// Runs `body` only in lanes where `cond` holds (a real branch: memory
    /// accesses of disabled lanes must not happen). Registers `body` writes
    /// merge through phis at the join.
    pub fn guarded(&mut self, cond: Value, body: &mut dyn FnMut(&mut Lw) -> LResult<()>) -> LResult<()> {
        // a condition known at translation time (e.g. EXEC proven all ones): no branch
        match self.s.m.const_u64(cond) {
            Some(1) => return body(self),
            Some(0) => return Ok(()),
            _ => {}
        }
        let from = self.cur();
        let then = self.block(None);
        let join = self.block(None);
        self.cond_br(cond, then, join);
        self.seal(then);
        self.s.m.set_block(then);
        self.ssa.begin_log();
        let r = body(self);
        let written = self.ssa.end_log();
        r?;
        let end = self.cur();
        self.br(join);
        // only what the body wrote merges at the join; other registers read straight
        // through `from` (no walk through both edges of every guard)
        self.ssa.seal_region_join(join, from, end, written);
        self.s.m.set_block(join);
        Ok(())
    }

    // ------------------------------------------------------------ values

    pub fn c(&mut self, v: u32) -> Value {
        self.s.m.i32c(v)
    }
    pub fn b(&mut self, v: bool) -> Value {
        self.s.m.i1c(v)
    }
    /// Integer binop, folded when both operands are constants (and the trivial
    /// identities with 0 / 1 / all ones), else emitted.
    pub fn bin(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        let t = self.s.m.type_of(a);
        if let Some(w) = self.s.m.int_width(t) {
            let (ca, cb) = (self.s.m.const_u64(a), self.s.m.const_u64(b));
            let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
            if let (Some(x), Some(y)) = (ca, cb) {
                if let Some(r) = fold_bin(op, x, y, w) {
                    return self.s.m.const_int(t, r & mask);
                }
            }
            match (op, cb) {
                (BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return a,
                (BinOp::And, Some(m)) if m == mask => return a,
                (BinOp::And | BinOp::Mul, Some(0)) => return b,
                (BinOp::Mul, Some(1)) => return a,
                _ => {}
            }
            match (op, ca) {
                (BinOp::Add | BinOp::Or | BinOp::Xor, Some(0)) => return b,
                (BinOp::And, Some(m)) if m == mask => return b,
                (BinOp::And | BinOp::Mul | BinOp::Shl | BinOp::LShr, Some(0)) => return a,
                (BinOp::Mul, Some(1)) => return b,
                _ => {}
            }
        }
        self.s.m.bin(op, a, b)
    }
    pub fn add(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Add, a, b)
    }
    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Sub, a, b)
    }
    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Mul, a, b)
    }
    pub fn and(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::And, a, b)
    }
    pub fn or(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Or, a, b)
    }
    pub fn xor(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Xor, a, b)
    }
    pub fn not(&mut self, a: Value) -> Value {
        let t = self.s.m.type_of(a);
        let m = self.s.m.const_int(t, u64::MAX);
        self.xor(a, m)
    }
    pub fn andc(&mut self, a: Value, k: u32) -> Value {
        let k = self.c(k);
        self.and(a, k)
    }
    pub fn addc(&mut self, a: Value, k: u32) -> Value {
        let k = self.c(k);
        self.add(a, k)
    }
    /// `a << (b & 31)` etc. (HLSL masks shift counts)
    pub fn shl(&mut self, a: Value, b: Value) -> Value {
        let b = self.mask_shift(a, b);
        self.bin(BinOp::Shl, a, b)
    }
    pub fn lshr(&mut self, a: Value, b: Value) -> Value {
        let b = self.mask_shift(a, b);
        self.bin(BinOp::LShr, a, b)
    }
    pub fn ashr(&mut self, a: Value, b: Value) -> Value {
        let b = self.mask_shift(a, b);
        self.bin(BinOp::AShr, a, b)
    }
    fn mask_shift(&mut self, a: Value, b: Value) -> Value {
        let t = self.s.m.type_of(a);
        let w = self.s.m.int_width(t).unwrap_or(32);
        let k = self.s.m.const_int(t, (w - 1) as u64);
        self.bin(BinOp::And, b, k)
    }
    pub fn shlc(&mut self, a: Value, k: u32) -> Value {
        let k = self.c(k);
        self.bin(BinOp::Shl, a, k)
    }
    pub fn lshrc(&mut self, a: Value, k: u32) -> Value {
        let k = self.c(k);
        self.bin(BinOp::LShr, a, k)
    }
    pub fn ashrc(&mut self, a: Value, k: u32) -> Value {
        let k = self.c(k);
        self.bin(BinOp::AShr, a, k)
    }
    pub fn cmp(&mut self, p: Pred, a: Value, b: Value) -> Value {
        let t = self.s.m.type_of(a);
        if let (Some(w), Some(x), Some(y)) = (self.s.m.int_width(t), self.s.m.const_u64(a), self.s.m.const_u64(b)) {
            let sx = |v: u64| if w >= 64 { v as i64 } else { ((v << (64 - w)) as i64) >> (64 - w) };
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
                return self.s.m.i1c(r);
            }
        }
        self.s.m.cmp(p, a, b)
    }
    pub fn cmpc(&mut self, p: Pred, a: Value, k: u32) -> Value {
        let k = self.c(k);
        self.cmp(p, a, k)
    }
    pub fn sel(&mut self, c: Value, t: Value, f: Value) -> Value {
        if t == f {
            return t;
        }
        match self.s.m.const_u64(c) {
            Some(1) => t,
            Some(0) => f,
            _ => self.s.m.select(c, t, f),
        }
    }
    pub fn selc(&mut self, c: Value, t: u32, f: Value) -> Value {
        let t = self.c(t);
        self.sel(c, t, f)
    }
    /// i1 → i32 0 / 1
    pub fn u(&mut self, c: Value) -> Value {
        let i32 = self.s.m.i32();
        self.s.m.cast(CastOp::ZExt, c, i32)
    }
    /// i32 != 0
    pub fn nz(&mut self, a: Value) -> Value {
        let t = self.s.m.type_of(a);
        let z = self.s.m.const_int(t, 0);
        self.cmp(Pred::Ne, a, z)
    }
    pub fn lnot(&mut self, c: Value) -> Value {
        let t = self.b(true);
        self.xor(c, t)
    }
    pub fn land(&mut self, a: Value, b: Value) -> Value {
        self.and(a, b)
    }
    pub fn lor(&mut self, a: Value, b: Value) -> Value {
        self.or(a, b)
    }
    pub fn i64(&mut self) -> super::module::Ty {
        self.s.require(Feature::Int64Ops);
        self.s.m.i64()
    }
    pub fn zext64(&mut self, a: Value) -> Value {
        let t = self.i64();
        self.s.m.cast(CastOp::ZExt, a, t)
    }
    pub fn sext64(&mut self, a: Value) -> Value {
        let t = self.i64();
        self.s.m.cast(CastOp::SExt, a, t)
    }
    pub fn trunc32(&mut self, a: Value) -> Value {
        let t = self.s.m.i32();
        self.s.m.cast(CastOp::Trunc, a, t)
    }
    pub fn c64(&mut self, v: u64) -> Value {
        let t = self.i64();
        self.s.m.const_int(t, v)
    }
    /// (lo, hi) → i64
    pub fn pack64(&mut self, lo: Value, hi: Value) -> Value {
        let l = self.zext64(lo);
        let h = self.zext64(hi);
        let k = self.c64(32);
        let h = self.bin(BinOp::Shl, h, k);
        self.or(l, h)
    }
    /// i64 → (lo, hi)
    pub fn unpack64(&mut self, v: Value) -> (Value, Value) {
        let lo = self.trunc32(v);
        let k = self.c64(32);
        let h = self.bin(BinOp::LShr, v, k);
        let hi = self.trunc32(h);
        (lo, hi)
    }
    pub fn f(&mut self, a: Value) -> Value {
        let t = self.s.m.f32();
        self.s.m.bitcast(a, t)
    }
    pub fn bits(&mut self, a: Value) -> Value {
        let t = self.s.m.type_of(a);
        if t == self.s.m.f64() {
            panic!("bits of a double");
        }
        let i = self.s.m.i32();
        self.s.m.bitcast(a, i)
    }
    pub fn fc(&mut self, x: f32) -> Value {
        self.s.m.f32c(x.to_bits())
    }
    pub fn fbin(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        self.s.m.fbin(op, a, b, 0)
    }
    pub fn fadd(&mut self, a: Value, b: Value) -> Value {
        self.fbin(BinOp::Add, a, b)
    }
    pub fn fsub(&mut self, a: Value, b: Value) -> Value {
        self.fbin(BinOp::Sub, a, b)
    }
    pub fn fmul(&mut self, a: Value, b: Value) -> Value {
        self.fbin(BinOp::Mul, a, b)
    }
    /// `fdiv` (bitcode BINOP_SDIV on floats; BINOP_UDIV is invalid there)
    pub fn fdiv(&mut self, a: Value, b: Value) -> Value {
        self.fbin(BinOp::SDiv, a, b)
    }
    /// HLSL `-x` (`fsub -0.0, x`)
    pub fn fneg(&mut self, a: Value) -> Value {
        let t = self.s.m.type_of(a);
        let bits = if t == self.s.m.f64() { 1u64 << 63 } else if t == self.s.m.f16() { 0x8000 } else { 0x8000_0000 };
        let z = self.s.m.const_float(t, bits);
        self.fsub(z, a)
    }
    pub fn fabs(&mut self, a: Value) -> Value {
        self.s.funary(FUnary::FAbs, a)
    }
    pub fn un(&mut self, opc: FUnary, a: Value) -> Value {
        self.s.funary(opc, a)
    }
    pub fn fmin(&mut self, a: Value, b: Value) -> Value {
        self.s.binary(Binary::FMin, a, b)
    }
    pub fn fmax(&mut self, a: Value, b: Value) -> Value {
        self.s.binary(Binary::FMax, a, b)
    }
    pub fn umin(&mut self, a: Value, b: Value) -> Value {
        self.s.binary(Binary::UMin, a, b)
    }
    pub fn umax(&mut self, a: Value, b: Value) -> Value {
        self.s.binary(Binary::UMax, a, b)
    }
    pub fn imin(&mut self, a: Value, b: Value) -> Value {
        self.s.binary(Binary::IMin, a, b)
    }
    pub fn imax(&mut self, a: Value, b: Value) -> Value {
        self.s.binary(Binary::IMax, a, b)
    }
    pub fn fmad(&mut self, a: Value, b: Value, c: Value) -> Value {
        self.s.tertiary(Tertiary::FMad, a, b, c)
    }
    pub fn isnan(&mut self, a: Value) -> Value {
        self.s.fclass(FClass::IsNaN, a)
    }
    pub fn isinf(&mut self, a: Value) -> Value {
        self.s.fclass(FClass::IsInf, a)
    }
    pub fn fcmp(&mut self, p: Pred, a: Value, b: Value) -> Value {
        self.s.m.cmp(p, a, b)
    }
    pub fn countbits(&mut self, a: Value) -> Value {
        self.s.bit_count(BitCount::Countbits, a)
    }
    /// HLSL `firstbitlow` (−1 for 0)
    pub fn firstbitlow(&mut self, a: Value) -> Value {
        self.s.bit_count(BitCount::FirstbitLo, a)
    }
    /// leading zeros from bit 31 (−1 for 0): DXIL `FirstbitHi`
    pub fn lzcnt(&mut self, a: Value) -> Value {
        self.s.bit_count(BitCount::FirstbitHi, a)
    }
    /// HLSL `firstbithigh(uint)`: bit index from bit 0 (−1 for 0)
    pub fn firstbithigh(&mut self, a: Value) -> Value {
        let r = self.lzcnt(a);
        let k = self.c(31);
        let s = self.sub(k, r);
        let none = self.cmpc(Pred::Eq, r, u32::MAX);
        self.selc(none, u32::MAX, s)
    }
    pub fn reversebits(&mut self, a: Value) -> Value {
        self.s.bfrev(a)
    }
    pub fn uitof(&mut self, a: Value) -> Value {
        let t = self.s.m.f32();
        self.s.m.cast(CastOp::UIToFP, a, t)
    }
    pub fn sitof(&mut self, a: Value) -> Value {
        let t = self.s.m.f32();
        self.s.m.cast(CastOp::SIToFP, a, t)
    }
    pub fn ftoui(&mut self, a: Value) -> Value {
        let t = self.s.m.i32();
        self.s.m.cast(CastOp::FPToUI, a, t)
    }
    pub fn ftosi(&mut self, a: Value) -> Value {
        let t = self.s.m.i32();
        self.s.m.cast(CastOp::FPToSI, a, t)
    }

    // ------------------------------------------------------------ registers

    pub fn sreg(&mut self, code: u64) -> Value {
        match code {
            0..=124 | 126 | 127 => self.rd(code as u32),
            _ => self.c(0),
        }
    }
    pub fn vreg(&mut self, n: u64) -> Value {
        self.rd(VAR_VGPR + n as u32)
    }
    pub fn scc(&mut self) -> Value {
        self.rd(VAR_SCC)
    }
    pub fn set_scc(&mut self, c: Value) {
        let v = self.u(c);
        self.wr(VAR_SCC, v);
    }
    pub fn scc_bit(&mut self) -> Value {
        let s = self.scc();
        self.nz(s)
    }

    /// 32-bit value of a scalar source code (0..255) or VALU source (256+ = VGPR).
    pub fn code32(&mut self, code: u64, lit: Option<u32>, f16: bool) -> Option<Value> {
        Some(match code {
            0..=124 | 126 | 127 => self.sreg(code),
            125 => self.c(0),
            128..=192 => self.c((code - 128) as u32),
            193..=208 => self.c((-((code - 192) as i64)) as u32),
            240..=248 => {
                let v = if f16 { f16_const(code)? } else { f32_const(code)?.to_bits() };
                self.c(v)
            }
            251 | 252 => {
                let (l, h) = if code == 251 { (106, 107) } else { (126, 127) };
                // wave32: the high half (vcc_hi / exec_hi) is not part of the mask
                let o = if self.wave64 {
                    let (l, h) = (self.sreg(l), self.sreg(h));
                    self.or(l, h)
                } else {
                    self.sreg(l)
                };
                let z = self.cmpc(Pred::Eq, o, 0);
                self.u(z)
            }
            253 => self.scc(),
            255 => self.c(lit?),
            256..=511 => self.vreg(code - 256),
            235..=238 => self.c(0),
            239 => self.c(0x7FFF_FFFF),
            _ => return None,
        })
    }

    /// 64-bit value (lo, hi) of a scalar source code.
    pub fn code64(&mut self, code: u64, lit: Option<u32>, f64: bool) -> Option<(Value, Value)> {
        Some(match code {
            0..=123 | 126 => (self.sreg(code), self.sreg(code + 1)),
            125 => (self.c(0), self.c(0)),
            128..=192 => (self.c((code - 128) as u32), self.c(0)),
            193..=208 => (self.c((-((code - 192) as i64)) as u32), self.c(u32::MAX)),
            248 => (self.c(0x6DC9_C882), self.c(0x3FC4_5F30)),
            240..=247 => {
                let b = (f32_const(code)? as f64).to_bits();
                (self.c(b as u32), self.c((b >> 32) as u32))
            }
            // src_vccz / src_execz / scc: a 0 / 1 value, zero-extended
            251 | 252 | 253 => (self.code32(code, None, false)?, self.c(0)),
            255 => {
                if f64 {
                    (self.c(0), self.c(lit?))
                } else {
                    (self.c(lit?), self.c(0))
                }
            }
            256..=510 => (self.vreg(code - 256), self.vreg(code - 255)),
            _ => return None,
        })
    }

    pub fn field(d: &Decoded, f: &str) -> u64 {
        rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f)
    }
    pub fn fv(d: &Decoded, f: &str) -> u64 {
        rdna2::field_value(d.enc, &d.words, f).unwrap_or(0)
    }

    pub fn read(&mut self, d: &Decoded, o: &Operand, f16: bool) -> Option<Value> {
        match o.field {
            Some("LITERAL") => Some(self.c(d.literal?)),
            Some(f) => {
                let v = Self::field(d, f);
                let width = d.enc.field_width(f);
                if o.ty.starts_with("OPR_VGPR") && width <= 8 {
                    return Some(self.vreg(v));
                }
                match o.ty {
                    "OPR_SIMM16" => Some(self.c(v as u32)),
                    "OPR_SIMM32" => Some(self.c(d.literal?)),
                    _ => self.code32(v, d.literal, f16),
                }
            }
            None => match o.ty {
                "OPR_VCC" => Some(self.sreg(106)),
                "OPR_EXEC" | "OPR_SDST_EXEC" => Some(self.sreg(126)),
                "OPR_SDST_M0" => Some(self.sreg(124)),
                "OPR_SSRC_SPECIAL_SCC" => Some(self.scc()),
                _ => None,
            },
        }
    }

    pub fn read64(&mut self, d: &Decoded, o: &Operand, f64: bool) -> Option<(Value, Value)> {
        match o.field {
            Some(f) => {
                let v = Self::field(d, f);
                if o.ty.starts_with("OPR_VGPR") && d.enc.field_width(f) <= 8 {
                    return Some((self.vreg(v), self.vreg(v + 1)));
                }
                self.code64(v, d.literal, f64)
            }
            None => match o.ty {
                "OPR_VCC" => Some((self.sreg(106), self.sreg(107))),
                "OPR_EXEC" | "OPR_SDST_EXEC" => Some((self.sreg(126), self.sreg(127))),
                _ => None,
            },
        }
    }

    pub fn dst_of(d: &Decoded, o: &Operand) -> Dst {
        match o.field {
            Some(f) => {
                let v = Self::field(d, f);
                match o.ty {
                    "OPR_EXEC" => Dst::S(126),
                    "OPR_VCC" => Dst::S(106),
                    t if t.starts_with("OPR_VGPR") => Dst::V(v),
                    _ if v == 125 => Dst::Null,
                    _ if v > 127 => Dst::Null,
                    _ => Dst::S(v),
                }
            }
            None => match o.ty {
                "OPR_VCC" => Dst::S(106),
                "OPR_EXEC" => Dst::S(126),
                _ => Dst::Null,
            },
        }
    }

    /// Explicit operands: (outputs, inputs) in spec order.
    pub fn ops(d: &Decoded) -> (Vec<&'static Operand>, Vec<&'static Operand>) {
        let mut outs = Vec::new();
        let mut ins = Vec::new();
        for o in d.ienc.operands {
            if o.implicit && !(o.field.is_none() && o.ty == "OPR_VCC") {
                continue;
            }
            if o.output {
                outs.push(o);
            } else if o.input {
                ins.push(o);
            }
        }
        (outs, ins)
    }

    /// This lane's EXEC bit.
    pub fn act(&mut self) -> Value {
        if let Some(g) = self.guard_override {
            return g;
        }
        let lo = self.sreg(126);
        let hi = if self.wave64 { self.sreg(127) } else { self.c(0) };
        self.lanebit(lo, hi)
    }

    /// The wave-wide active mask (lo, hi) cross-lane instructions consult: the
    /// override if set, else the SSA EXEC (hi = 0 in wave32).
    pub fn exec_mask(&mut self) -> (Value, Value) {
        if let Some(m) = self.mask_override {
            return m;
        }
        let lo = self.sreg(126);
        let hi = if self.wave64 { self.sreg(127) } else { self.c(0) };
        (lo, hi)
    }

    /// Lowers one instruction under a guard chosen by the caller instead of the
    /// SSA EXEC: `guard` (i1) decides whether this lane's VGPR writes and memory
    /// effects happen (a constant `true` folds every select / branch away);
    /// `mask`, if given, is the wave-wide mask cross-lane instructions see (e.g. a
    /// ballot of the guard), else they read the SSA EXEC. Instructions that write
    /// EXEC still update the SSA EXEC registers (126/127); EXEC read as a plain
    /// scalar operand (`s_mov s0, exec`) reads them too. Control flow (branches)
    /// is the caller's business: use this for non-branch instructions.
    pub fn inst_guarded(&mut self, d: &Decoded, guard: Value, mask: Option<(Value, Value)>) -> LResult<()> {
        let (g, m) = (self.guard_override.replace(guard), std::mem::replace(&mut self.mask_override, mask));
        let r = self.inst(d);
        self.guard_override = g;
        self.mask_override = m;
        r
    }

    pub fn write32(&mut self, dst: &Dst, val: Value) {
        match *dst {
            Dst::V(n) => {
                let old = self.vreg(n);
                let a = self.act();
                let v = self.sel(a, val, old);
                self.wr(VAR_VGPR + n as u32, v);
            }
            Dst::S(n) => {
                if n <= 127 && n != 125 {
                    self.wr(n as u32, val);
                }
            }
            Dst::Null => {}
        }
    }

    pub fn write64(&mut self, dst: &Dst, lo: Value, hi: Value) {
        match *dst {
            Dst::V(n) => {
                let a = self.act();
                let (o0, o1) = (self.vreg(n), self.vreg(n + 1));
                let v0 = self.sel(a, lo, o0);
                let v1 = self.sel(a, hi, o1);
                self.wr(VAR_VGPR + n as u32, v0);
                self.wr(VAR_VGPR + n as u32 + 1, v1);
            }
            Dst::S(n) => {
                if n != 125 {
                    self.wr(n as u32, lo);
                }
                if n + 1 <= 127 && n + 1 != 125 {
                    self.wr(n as u32 + 1, hi);
                }
            }
            Dst::Null => {}
        }
    }

    /// Lane mask from a per-lane condition (inactive lanes give 0) into a register (pair).
    pub fn write_mask(&mut self, dst: &Dst, cond: Value) {
        let a = self.act();
        let c = self.land(a, cond);
        // split form: a mask only ever read lane by lane (or going nowhere, or 0 anyway)
        // needs only this half's word (`recomp::x64_masks`, the HLSL `xcl_x_ballot_local`)
        let local = self.split
            && (self.s.m.const_u64(cond) == Some(0)
                || match *dst {
                    Dst::S(n) => n < 126 && self.whole_live.get(&self.pc).is_some_and(|live| live & (3u128 << n) == 0),
                    Dst::Null => true,
                    Dst::V(_) => false,
                });
        let (lo, hi) = if local {
            let w = self.s.wave_ballot(c)[0];
            let z = self.c(0);
            let u = self.upper;
            (self.sel(u, z, w), self.sel(u, w, z))
        } else {
            self.ballot(c)
        };
        if self.wave64 {
            self.write64(dst, lo, hi);
        } else {
            self.write32(dst, lo);
        }
    }

    /// This lane's bit of a mask (lo, hi).
    pub fn lane_of(&mut self, lo: Value, hi: Value) -> Value {
        if self.wave64 {
            self.lanebit(lo, hi)
        } else {
            let z = self.c(0);
            self.lanebit(lo, z)
        }
    }

    // ------------------------------------------------------------ dispatch

    pub fn inst(&mut self, d: &Decoded) -> LResult<()> {
        let name = d.inst.name.to_ascii_lowercase();
        self.pc = d.pc;
        self.dpp_enable = None;
        if name == "s_sendmsg" && self.mesh.is_some() {
            return self.mesh_sendmsg(d);
        }
        if self.geom.is_some() && self.geom_inst(d, &name).is_some() {
            return Ok(());
        }
        if name == "s_denorm_mode" && (self.f32_rm || self.track_frexp_denorm) {
            return self.denorm_mode(Self::fv(d, "SIMM16") as u32);
        }
        if NOOPS.contains(&name.as_str()) {
            return Ok(());
        }
        match d.enc.name {
            "ENC_SOPP" => self.sopp(d, &name),
            "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" => self.salu(d, &name),
            "ENC_SMEM" => self.smem(d, &name),
            "ENC_MUBUF" | "ENC_MTBUF" => self.mubuf(d, &name),
            "ENC_EXP" => self.export(d),
            "ENC_DS" => self.ds(d, &name),
            "ENC_VINTRP" => self.interp_gfx(d, &name),
            "ENC_MIMG" => self.mimg(d, &name),
            "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" => self.flat(d, &name),
            e if e.starts_with("ENC_VOP") || e.starts_with("VOP") => self.valu(d, &name),
            _ => err(d, "encoding"),
        }
    }

    /// The condition of a conditional branch (i1).
    pub fn branch_cond(&mut self, d: &Decoded) -> LResult<Value> {
        let name = d.inst.name.to_ascii_lowercase();
        let zero = |s: &mut Self, l: u64| {
            let lo = s.sreg(l);
            let v = if s.wave64 {
                let hi = s.sreg(l + 1);
                s.or(lo, hi)
            } else {
                lo
            };
            s.cmpc(Pred::Eq, v, 0)
        };
        Ok(match name.as_str() {
            "s_cbranch_scc0" => {
                let s = self.scc();
                self.cmpc(Pred::Eq, s, 0)
            }
            "s_cbranch_scc1" => {
                let s = self.scc();
                self.cmpc(Pred::Ne, s, 0)
            }
            "s_cbranch_vccz" => zero(self, 106),
            "s_cbranch_vccnz" => {
                let z = zero(self, 106);
                self.lnot(z)
            }
            "s_cbranch_execz" => zero(self, 126),
            "s_cbranch_execnz" => {
                let z = zero(self, 126);
                self.lnot(z)
            }
            _ => return err(d, "conditional branch"),
        })
    }

    fn sopp(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        match name {
            "s_barrier" => {
                self.s.barrier(super::ops::barrier::GROUP_SYNC);
                Ok(())
            }
            "s_round_mode" => {
                let simm = Self::fv(d, "SIMM16") as u16;
                let (sp, dp) = ((simm & 3) as u32, ((simm >> 2) & 3) as u32);
                if self.f32_rm {
                    let v = self.c(sp);
                    self.mode_write(v, 3);
                } else if sp != 0 {
                    return err(d, "non-default f32 round mode");
                }
                if dp != 0 && self.has_f64_round_op {
                    return err(d, "non-default f64 round mode");
                }
                if self.dyn_rm {
                    let v = self.c(dp);
                    self.wr(VAR_FP16_RM, v);
                }
                Ok(())
            }
            _ if NOOPS.contains(&name) => Ok(()),
            _ => err(d, "SOPP"),
        }
    }

    // ------------------------------------------------------------ scalar ALU

    fn setreg(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let simm = Self::fv(d, "SIMM16") as u16;
        let (sp_mask, dp_mask, off) = hwreg_mode_round_masks(simm);
        if self.f32_rm || self.track_frexp_denorm {
            // the mirrored MODE bits (single-precision round / denormal modes)
            let (id, offset, width) = decode_hwreg(simm);
            if id == 1 {
                let wmask = (if width == 32 { u32::MAX } else { (1u32 << width) - 1 }) << offset;
                let src = if name == "s_setreg_imm32_b32" {
                    let imm = d.literal.ok_or_else(|| format!("{}: literal", rdna2::format(d)))?;
                    self.c(imm)
                } else {
                    let sd = Self::field(d, "SDST");
                    self.code32(sd, None, false).ok_or_else(|| format!("{}: register", rdna2::format(d)))?
                };
                let v = self.shlc(src, offset);
                self.mode_write(v, wmask);
            }
        }
        if sp_mask == 0 && dp_mask == 0 {
            return Ok(());
        }
        if name == "s_setreg_imm32_b32" {
            let imm = d.literal.ok_or_else(|| format!("{}: literal", rdna2::format(d)))?;
            let sp = hwreg_imm_field(imm, off, 0, sp_mask);
            let dp = hwreg_imm_field(imm, off, 2, dp_mask);
            if sp != 0 && !self.f32_rm {
                return err(d, "non-default f32 round mode");
            }
            if dp_mask != 0 && dp != 0 && self.has_f64_round_op {
                return err(d, "non-default f64 round mode");
            }
            if dp_mask != 0 && self.dyn_rm {
                let v = if dp_mask == 3 {
                    self.c(dp)
                } else {
                    let cur = self.rd(VAR_FP16_RM);
                    let keep = self.andc(cur, !dp_mask & 3);
                    self.orc(keep, dp)
                };
                self.wr(VAR_FP16_RM, v);
            }
            return Ok(());
        }
        if sp_mask != 0 && !self.f32_rm {
            return err(d, "dynamic f32 round mode");
        }
        if self.has_f64_round_op {
            return err(d, "dynamic f64 round mode");
        }
        if self.dyn_rm {
            let sd = Self::field(d, "SDST");
            let src = self.code32(sd, None, false).ok_or_else(|| format!("{}: register", rdna2::format(d)))?;
            let v = match dp_mask {
                3 => {
                    let x = self.lshrc(src, 2 - off);
                    self.andc(x, 3)
                }
                1 => {
                    let x = self.lshrc(src, 2 - off);
                    let bit = self.andc(x, 1);
                    let cur = self.rd(VAR_FP16_RM);
                    let keep = self.andc(cur, 2);
                    self.or(keep, bit)
                }
                _ => {
                    let bit = self.andc(src, 1);
                    let bit = self.shlc(bit, 1);
                    let cur = self.rd(VAR_FP16_RM);
                    let keep = self.andc(cur, 1);
                    self.or(keep, bit)
                }
            };
            self.wr(VAR_FP16_RM, v);
        }
        Ok(())
    }

    fn salu(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        if matches!(name, "s_setreg_imm32_b32" | "s_setreg_b32") {
            return self.setreg(d, name);
        }
        if matches!(name, "s_getreg_b32") {
            return err(d, "s_getreg");
        }
        if matches!(name, "s_movrels_b32" | "s_movrels_b64" | "s_movreld_b32" | "s_movreld_b64" | "s_movrelsd_2_b32") {
            return self.smovrel(d, name);
        }
        let (outs, ins) = Self::ops(d);
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        let dst = outs.first().map(|o| Self::dst_of(d, o)).unwrap_or(Dst::Null);
        let simm = Self::fv(d, "SIMM16") as u16;
        let base = name.trim_end_matches("_b32").trim_end_matches("_b64");
        if d.enc.name == "ENC_SOPK" {
            let k_s = simm as i16 as i32 as u32;
            let k_u = simm as u32;
            let sd = Self::field(d, "SDST");
            let cur = self.code32(sd, None, false).ok_or("register").or_else(|e| err(d, e))?;
            let dst = Dst::S(sd);
            let ks = self.c(k_s);
            match name {
                "s_movk_i32" => self.write32(&dst, ks),
                "s_cmovk_i32" => {
                    let c = self.scc_bit();
                    let v = self.sel(c, ks, cur);
                    self.write32(&dst, v)
                }
                "s_addk_i32" => {
                    let t = self.add(cur, ks);
                    let a = self.xor(cur, t);
                    let b = self.xor(ks, t);
                    let x = self.and(a, b);
                    let x = self.lshrc(x, 31);
                    self.wr(VAR_SCC, x);
                    self.write32(&dst, t);
                }
                "s_mulk_i32" => {
                    let v = self.mul(cur, ks);
                    self.write32(&dst, v)
                }
                "s_version" | "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt" => {}
                _ if name.starts_with("s_cmpk_") => {
                    let (op, signed) = cmp_op(&name[7..]).ok_or("comparison").or_else(|e| err(d, e))?;
                    let k = if signed { ks } else { self.c(k_u) };
                    let c = self.cmp(int_pred(op, signed), cur, k);
                    self.set_scc(c);
                }
                _ => return err(d, "SOPK"),
            }
            return Ok(());
        }
        if d.enc.name == "ENC_SOPC" {
            if matches!(name, "s_bitcmp0_b64" | "s_bitcmp1_b64") {
                let (a0, a1) = self.read64(d, ins[0], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
                let b = self.read(d, ins[1], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
                let b32 = self.andc(b, 32);
                let lowhalf = self.cmpc(Pred::Eq, b32, 0);
                let w = self.sel(lowhalf, a0, a1);
                let sh = self.lshr(w, b);
                let bit = self.andc(sh, 1);
                if name == "s_bitcmp0_b64" {
                    let c = self.cmpc(Pred::Eq, bit, 0);
                    self.set_scc(c);
                } else {
                    self.wr(VAR_SCC, bit);
                }
                return Ok(());
            }
            if name.ends_with("_u64") {
                let a = self.read64(d, ins[0], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
                let b = self.read64(d, ins[1], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
                let (p, eq) = match name {
                    "s_cmp_eq_u64" => (Pred::Eq, true),
                    "s_cmp_lg_u64" => (Pred::Ne, false),
                    _ => return err(d, "SOPC"),
                };
                let c0 = self.cmp(p, a.0, b.0);
                let c1 = self.cmp(p, a.1, b.1);
                let c = if eq { self.land(c0, c1) } else { self.lor(c0, c1) };
                self.set_scc(c);
                return Ok(());
            }
            let a = self.read(d, ins[0], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
            let b = self.read(d, ins[1], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
            if let Some(rest) = name.strip_prefix("s_cmp_") {
                let (op, signed) = cmp_op(rest).ok_or("comparison").or_else(|e| err(d, e))?;
                let c = self.cmp(int_pred(op, signed), a, b);
                self.set_scc(c);
                return Ok(());
            }
            match name {
                "s_bitcmp0_b32" | "s_bitcmp1_b32" => {
                    let sh = self.lshr(a, b);
                    let bit = self.andc(sh, 1);
                    if name == "s_bitcmp0_b32" {
                        let c = self.cmpc(Pred::Eq, bit, 0);
                        self.set_scc(c);
                    } else {
                        self.wr(VAR_SCC, bit);
                    }
                }
                _ => return err(d, "SOPC"),
            }
            return Ok(());
        }
        // SOP1 / SOP2
        let saveexec = name.contains("saveexec") || name.contains("wrexec");
        if saveexec {
            // a b32 source is one register (m0, a 32-bit constant): no 64-bit pair to read
            let s0 = if name.ends_with("_b32") {
                let lo = self.read(d, ins[0], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
                (lo, self.c(0))
            } else {
                self.read64(d, ins[0], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?
            };
            let (el, eh) = (self.sreg(126), self.sreg(127));
            let opn = base.split('_').nth(1).unwrap_or("").to_string();
            let op = |s: &mut Self, a: Value, e: Value| -> Option<Value> {
                Some(match opn.as_str() {
                    "and" => s.and(a, e),
                    "or" => s.or(a, e),
                    "xor" => s.xor(a, e),
                    "andn2" => {
                        let n = s.not(e);
                        s.and(a, n)
                    }
                    "orn2" => {
                        let n = s.not(e);
                        s.or(a, n)
                    }
                    "nand" => {
                        let x = s.and(a, e);
                        s.not(x)
                    }
                    "nor" => {
                        let x = s.or(a, e);
                        s.not(x)
                    }
                    "xnor" => {
                        let x = s.xor(a, e);
                        s.not(x)
                    }
                    "andn1" => {
                        let n = s.not(a);
                        s.and(n, e)
                    }
                    "orn1" => {
                        let n = s.not(a);
                        s.or(n, e)
                    }
                    _ => return None,
                })
            };
            let t0 = op(self, s0.0, el).ok_or_else(|| format!("{}: saveexec", rdna2::format(d)))?;
            let t1 = op(self, s0.1, eh).ok_or_else(|| format!("{}: saveexec", rdna2::format(d)))?;
            let wrexec = name.contains("wrexec");
            if !self.wave64 || name.ends_with("_b32") {
                self.wr(126, t0);
                let c = self.nz(t0);
                self.set_scc(c);
                self.write32(&dst, if wrexec { t0 } else { el });
            } else {
                self.wr(126, t0);
                self.wr(127, t1);
                let o = self.or(t0, t1);
                let c = self.nz(o);
                self.set_scc(c);
                if wrexec {
                    self.write64(&dst, t0, t1);
                } else {
                    self.write64(&dst, el, eh);
                }
            }
            return Ok(());
        }
        if name == "s_swappc_b64" || name == "s_setpc_b64" {
            let (sd, s0) = (Self::field(d, "SDST"), Self::field(d, "SSRC0"));
            if name == "s_swappc_b64" && sd == s0 {
                return Ok(());
            }
            return err(d, "indirect branch");
        }
        if name == "s_getpc_b64" {
            let pc = d.pc + d.len as u64;
            let (lo, hi) = (self.c(pc as u32), self.c(0));
            self.write64(&dst, lo, hi);
            return Ok(());
        }
        if matches!(name, "s_bitset0_b64" | "s_bitset1_b64") {
            let i = self.read(d, ins[0], false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?;
            let (lo, hi) = match &dst {
                Dst::S(n) => (self.sreg(*n), self.sreg(*n + 1)),
                _ => (self.c(0), self.c(0)),
            };
            let one = self.c(1);
            let mask = self.shl(one, i);
            let i32b = self.andc(i, 32);
            let low = self.cmpc(Pred::Eq, i32b, 0);
            let z = self.c(0);
            let ml = self.sel(low, mask, z);
            let mh = self.sel(low, z, mask);
            let (nl, nh) = if name == "s_bitset1_b64" {
                (self.or(lo, ml), self.or(hi, mh))
            } else {
                let a = self.not(ml);
                let b = self.not(mh);
                (self.and(lo, a), self.and(hi, b))
            };
            self.write64(&dst, nl, nh);
            return Ok(());
        }
        if wide {
            return self.salu64(d, name, &dst, &ins);
        }
        let a = match ins.first() {
            Some(o) => Some(self.read(d, o, false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?),
            None => None,
        };
        let b = match ins.get(1) {
            Some(o) => Some(self.read(d, o, false).ok_or_else(|| format!("{}: op", rdna2::format(d)))?),
            None => None,
        };
        let z = self.c(0);
        let a = a.unwrap_or(z);
        let b = b.unwrap_or(z);
        // result, and SCC: None, Some(Nz) = result != 0, Some(Val(i1))
        enum Scc {
            No,
            Nz,
            Is(Value),
        }
        let (res, scc): (Value, Scc) = match name {
            "s_mov_b32" => (a, Scc::No),
            "s_cmov_b32" => {
                let cur = match &dst {
                    Dst::S(n) => self.sreg(*n),
                    _ => self.c(0),
                };
                let c = self.scc_bit();
                (self.sel(c, a, cur), Scc::No)
            }
            "s_not_b32" => (self.not(a), Scc::Nz),
            "s_brev_b32" => (self.reversebits(a), Scc::No),
            "s_bcnt0_i32_b32" => {
                let n = self.countbits(a);
                let k = self.c(32);
                (self.sub(k, n), Scc::Nz)
            }
            "s_bcnt1_i32_b32" => (self.countbits(a), Scc::Nz),
            "s_ff0_i32_b32" => {
                let n = self.not(a);
                (self.firstbitlow(n), Scc::No)
            }
            "s_ff1_i32_b32" => (self.firstbitlow(a), Scc::No),
            "s_flbit_i32_b32" => (self.lzcnt(a), Scc::No),
            "s_flbit_i32" => (self.ffbh_i(a), Scc::No),
            "s_abs_i32" => {
                let n = self.sub(z, a);
                let neg = self.cmp(Pred::SLt, a, z);
                (self.sel(neg, n, a), Scc::Nz)
            }
            "s_sext_i32_i8" => {
                let x = self.shlc(a, 24);
                (self.ashrc(x, 24), Scc::No)
            }
            "s_sext_i32_i16" => {
                let x = self.shlc(a, 16);
                (self.ashrc(x, 16), Scc::No)
            }
            "s_and_b32" => (self.and(a, b), Scc::Nz),
            "s_or_b32" => (self.or(a, b), Scc::Nz),
            "s_xor_b32" => (self.xor(a, b), Scc::Nz),
            "s_andn2_b32" => {
                let n = self.not(b);
                (self.and(a, n), Scc::Nz)
            }
            "s_orn2_b32" => {
                let n = self.not(b);
                (self.or(a, n), Scc::Nz)
            }
            "s_nand_b32" => {
                let x = self.and(a, b);
                (self.not(x), Scc::Nz)
            }
            "s_nor_b32" => {
                let x = self.or(a, b);
                (self.not(x), Scc::Nz)
            }
            "s_xnor_b32" => {
                let x = self.xor(a, b);
                (self.not(x), Scc::Nz)
            }
            "s_add_u32" => {
                let t = self.add(a, b);
                let c = self.cmp(Pred::ULt, t, a);
                (t, Scc::Is(c))
            }
            "s_sub_u32" => {
                let t = self.sub(a, b);
                let c = self.cmp(Pred::UGt, b, a);
                (t, Scc::Is(c))
            }
            "s_addc_u32" => {
                let sc = self.scc();
                let t0 = self.add(a, b);
                let t = self.add(t0, sc);
                let c0 = self.cmp(Pred::ULt, t0, a);
                let c1 = self.cmp(Pred::ULt, t, t0);
                let c = self.lor(c0, c1);
                (t, Scc::Is(c))
            }
            "s_subb_u32" => {
                let sc = self.scc();
                let t0 = self.sub(a, b);
                let t = self.sub(t0, sc);
                let c0 = self.cmp(Pred::ULt, a, b);
                let c1 = self.cmp(Pred::ULt, t0, sc);
                let c = self.lor(c0, c1);
                (t, Scc::Is(c))
            }
            "s_add_i32" => {
                let t = self.add(a, b);
                let x = self.xor(a, t);
                let y = self.xor(b, t);
                let x = self.and(x, y);
                let x = self.lshrc(x, 31);
                (t, Scc::Is(self.nz(x)))
            }
            "s_sub_i32" => {
                let t = self.sub(a, b);
                let x = self.xor(a, b);
                let y = self.xor(a, t);
                let x = self.and(x, y);
                let x = self.lshrc(x, 31);
                (t, Scc::Is(self.nz(x)))
            }
            "s_mul_i32" => (self.mul(a, b), Scc::No),
            "s_mul_hi_u32" => (self.mul_hi(a, b, false), Scc::No),
            "s_mul_hi_i32" => (self.mul_hi(a, b, true), Scc::No),
            "s_lshl_b32" => (self.shl(a, b), Scc::Nz),
            "s_lshr_b32" => (self.lshr(a, b), Scc::Nz),
            "s_ashr_i32" => (self.ashr(a, b), Scc::Nz),
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let k = (name.as_bytes()[6] - b'0') as u32;
                let s = self.shlc(a, k);
                let t = self.add(s, b);
                let hi = self.lshrc(a, 32 - k);
                let c0 = self.nz(hi);
                let c1 = self.cmp(Pred::ULt, t, b);
                let c = self.lor(c0, c1);
                (t, Scc::Is(c))
            }
            "s_bfm_b32" => (self.bfm(a, b), Scc::No),
            "s_bfe_u32" => (self.sbfe_u(a, b), Scc::Nz),
            "s_bfe_i32" => (self.sbfe_i(a, b), Scc::Nz),
            "s_min_u32" => {
                let c = self.cmp(Pred::ULt, a, b);
                (self.umin(a, b), Scc::Is(c))
            }
            "s_max_u32" => {
                let c = self.cmp(Pred::UGt, a, b);
                (self.umax(a, b), Scc::Is(c))
            }
            "s_min_i32" => {
                let c = self.cmp(Pred::SLt, a, b);
                (self.imin(a, b), Scc::Is(c))
            }
            "s_max_i32" => {
                let c = self.cmp(Pred::SGt, a, b);
                (self.imax(a, b), Scc::Is(c))
            }
            "s_cselect_b32" => {
                let c = self.scc_bit();
                (self.sel(c, a, b), Scc::No)
            }
            "s_wqm_b32" => (self.wqm(a), Scc::Nz),
            "s_bitset0_b32" | "s_bitset1_b32" => {
                let cur = match &dst {
                    Dst::S(n) => self.sreg(*n),
                    _ => self.c(0),
                };
                let one = self.c(1);
                let m = self.shl(one, a);
                if name == "s_bitset1_b32" {
                    (self.or(cur, m), Scc::No)
                } else {
                    let n = self.not(m);
                    (self.and(cur, n), Scc::No)
                }
            }
            "s_absdiff_i32" => {
                let t = self.sub(a, b);
                let n = self.sub(z, t);
                let neg = self.cmp(Pred::SLt, t, z);
                (self.sel(neg, n, t), Scc::Nz)
            }
            "s_pack_ll_b32_b16" => {
                let l = self.andc(a, 0xFFFF);
                let h = self.shlc(b, 16);
                (self.or(l, h), Scc::No)
            }
            "s_pack_lh_b32_b16" => {
                let l = self.andc(a, 0xFFFF);
                let h = self.andc(b, 0xFFFF_0000);
                (self.or(l, h), Scc::No)
            }
            "s_pack_hh_b32_b16" => {
                let l = self.lshrc(a, 16);
                let h = self.andc(b, 0xFFFF_0000);
                (self.or(l, h), Scc::No)
            }
            _ => return err(d, "SALU"),
        };
        match scc {
            Scc::No => {}
            Scc::Nz => {
                let c = self.nz(res);
                self.set_scc(c);
            }
            Scc::Is(c) => self.set_scc(c),
        }
        self.write32(&dst, res);
        Ok(())
    }

    fn smovrel(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let dst = Self::field(d, "SDST");
        let src = Self::field(d, "SSRC0");
        if dst > 123 || src > 123 {
            return err(d, "scalar movrel register");
        }
        let m0 = self.sreg(124);
        let (rd, wr, wide) = match name {
            "s_movrels_b32" => (Some(m0), None, false),
            "s_movrels_b64" => (Some(m0), None, true),
            "s_movreld_b32" => (None, Some(m0), false),
            "s_movreld_b64" => (None, Some(m0), true),
            _ => {
                let r = self.andc(m0, 0x3FF);
                let w = self.lshrc(m0, 16);
                let w = self.andc(w, 0x3FF);
                (Some(r), Some(w), false)
            }
        };
        let words = if wide { 2 } else { 1 };
        let mut vals = Vec::new();
        for word in 0..words {
            if let Some(idx) = rd {
                // select chain over every register from src + word up to s123
                let mut v = self.c(0);
                for k in 0..=123u64.saturating_sub(src + word) {
                    let r = self.sreg(src + word + k);
                    let is = self.cmpc(Pred::Eq, idx, k as u32);
                    v = self.sel(is, r, v);
                }
                vals.push(v);
            } else {
                vals.push(self.sreg(src + word));
            }
        }
        if let Some(idx) = wr {
            for k in 0..=123u64.saturating_sub(dst + words - 1) {
                let is = self.cmpc(Pred::Eq, idx, k as u32);
                for w in 0..words {
                    let old = self.sreg(dst + k + w);
                    let v = self.sel(is, vals[w as usize], old);
                    self.wr((dst + k + w) as u32, v);
                }
            }
        } else {
            for w in 0..words {
                self.wr((dst + w) as u32, vals[w as usize]);
            }
        }
        Ok(())
    }

    fn salu64(&mut self, d: &Decoded, name: &str, dst: &Dst, ins: &[&Operand]) -> LResult<()> {
        let e = || format!("{}: op", rdna2::format(d));
        let a = match ins.first() {
            Some(o) => Some(self.read64(d, o, false).ok_or_else(e)?),
            None => None,
        };
        let b = match ins.get(1) {
            Some(o) => {
                if o.size == 32 {
                    let v = self.read(d, o, false).ok_or_else(e)?;
                    let z = self.c(0);
                    Some((v, z))
                } else {
                    Some(self.read64(d, o, false).ok_or_else(e)?)
                }
            }
            None => None,
        };
        let z = self.c(0);
        let (al, ah) = a.unwrap_or((z, z));
        let (bl, bh) = b.unwrap_or((z, z));
        let one_word = matches!(name, "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" | "s_ff0_i32_b64" | "s_ff1_i32_b64" | "s_flbit_i32_b64" | "s_flbit_i32_i64");
        let mut nz = true;
        let (lo, hi): (Value, Value) = match name {
            "s_mov_b64" => {
                nz = false;
                (al, ah)
            }
            "s_cmov_b64" => {
                nz = false;
                let (c0, c1) = match dst {
                    Dst::S(n) => (self.sreg(*n), self.sreg(*n + 1)),
                    _ => (z, z),
                };
                let c = self.scc_bit();
                (self.sel(c, al, c0), self.sel(c, ah, c1))
            }
            "s_not_b64" => (self.not(al), self.not(ah)),
            "s_cselect_b64" => {
                nz = false;
                let c = self.scc_bit();
                (self.sel(c, al, bl), self.sel(c, ah, bh))
            }
            "s_wqm_b64" => (self.wqm(al), self.wqm(ah)),
            "s_brev_b64" => {
                nz = false;
                (self.reversebits(ah), self.reversebits(al))
            }
            "s_bcnt1_i32_b64" => {
                let x = self.countbits(al);
                let y = self.countbits(ah);
                (self.add(x, y), z)
            }
            "s_bcnt0_i32_b64" => {
                let x = self.countbits(al);
                let y = self.countbits(ah);
                let k = self.c(64);
                let t = self.sub(k, x);
                (self.sub(t, y), z)
            }
            "s_ff1_i32_b64" | "s_ff0_i32_b64" => {
                nz = false;
                let (l, h) = if name == "s_ff0_i32_b64" { (self.not(al), self.not(ah)) } else { (al, ah) };
                let fl = self.firstbitlow(l);
                let fh = self.firstbitlow(h);
                let fh32 = self.addc(fh, 32);
                let hnz = self.nz(h);
                let m1 = self.c(u32::MAX);
                let hv = self.sel(hnz, fh32, m1);
                let lnz = self.nz(l);
                (self.sel(lnz, fl, hv), z)
            }
            "s_flbit_i32_b64" | "s_flbit_i32_i64" => {
                nz = false;
                let (l, h) = if name == "s_flbit_i32_i64" {
                    let neg = self.cmp(Pred::SLt, ah, z);
                    let nl = self.not(al);
                    let nh = self.not(ah);
                    (self.sel(neg, nl, al), self.sel(neg, nh, ah))
                } else {
                    (al, ah)
                };
                let lh = self.lzcnt(h);
                let ll = self.lzcnt(l);
                let ll32 = self.addc(ll, 32);
                let lnz = self.nz(l);
                let m1 = self.c(u32::MAX);
                let lv = self.sel(lnz, ll32, m1);
                let hnz = self.nz(h);
                (self.sel(hnz, lh, lv), z)
            }
            "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                let v = self.pack64(al, ah);
                let sh = self.andc(bl, 63);
                let sh = self.zext64(sh);
                let w = match name {
                    "s_lshl_b64" => self.bin(BinOp::Shl, v, sh),
                    "s_lshr_b64" => self.bin(BinOp::LShr, v, sh),
                    _ => self.bin(BinOp::AShr, v, sh),
                };
                self.unpack64(w)
            }
            "s_bfm_b64" => {
                nz = false;
                let wdt = self.andc(al, 63);
                let off = self.andc(bl, 63);
                let all = self.c64(u64::MAX);
                let k64 = self.c(64);
                let sh = self.sub(k64, wdt);
                let sh = self.andc(sh, 63);
                let sh = self.zext64(sh);
                let m = self.bin(BinOp::LShr, all, sh);
                let wz = self.cmpc(Pred::Eq, wdt, 0);
                let z64 = self.c64(0);
                let m = self.sel(wz, z64, m);
                let off = self.zext64(off);
                let r = self.bin(BinOp::Shl, m, off);
                self.unpack64(r)
            }
            "s_bfe_u64" | "s_bfe_i64" => {
                let v = self.pack64(al, ah);
                let w = self.lshrc(bl, 16);
                let w = self.andc(w, 0x7F);
                let o = self.andc(bl, 63);
                let o64 = self.zext64(o);
                let wz = self.cmpc(Pred::Eq, w, 0);
                let wbig = self.cmpc(Pred::UGe, w, 64);
                let k64 = self.c(64);
                let s = self.sub(k64, w);
                let s = self.andc(s, 63);
                let s64 = self.zext64(s);
                let z64 = self.c64(0);
                let r = if name == "s_bfe_u64" {
                    let x = self.bin(BinOp::LShr, v, o64);
                    let all = self.c64(u64::MAX);
                    let m = self.bin(BinOp::LShr, all, s64);
                    let xm = self.and(x, m);
                    let r = self.sel(wbig, x, xm);
                    self.sel(wz, z64, r)
                } else {
                    let x = self.bin(BinOp::AShr, v, o64);
                    let y = self.bin(BinOp::Shl, x, s64);
                    let y = self.bin(BinOp::AShr, y, s64);
                    let r = self.sel(wbig, x, y);
                    self.sel(wz, z64, r)
                };
                self.unpack64(r)
            }
            _ => {
                let op = name.trim_start_matches("s_").trim_end_matches("_b64");
                match op {
                    "and" => (self.and(al, bl), self.and(ah, bh)),
                    "or" => (self.or(al, bl), self.or(ah, bh)),
                    "xor" => (self.xor(al, bl), self.xor(ah, bh)),
                    "andn2" => {
                        let (x, y) = (self.not(bl), self.not(bh));
                        (self.and(al, x), self.and(ah, y))
                    }
                    "orn2" => {
                        let (x, y) = (self.not(bl), self.not(bh));
                        (self.or(al, x), self.or(ah, y))
                    }
                    "nand" => {
                        let (x, y) = (self.and(al, bl), self.and(ah, bh));
                        (self.not(x), self.not(y))
                    }
                    "nor" => {
                        let (x, y) = (self.or(al, bl), self.or(ah, bh));
                        (self.not(x), self.not(y))
                    }
                    "xnor" => {
                        let (x, y) = (self.xor(al, bl), self.xor(ah, bh));
                        (self.not(x), self.not(y))
                    }
                    _ => return err(d, "SALU64"),
                }
            }
        };
        if nz {
            let c = if one_word {
                self.nz(lo)
            } else {
                let o = self.or(lo, hi);
                self.nz(o)
            };
            self.set_scc(c);
        }
        if one_word {
            self.write32(dst, lo);
        } else {
            self.write64(dst, lo, hi);
        }
        Ok(())
    }

    // ------------------------------------------------------------ helpers (PRELUDE)

    pub fn mul_hi(&mut self, a: Value, b: Value, signed: bool) -> Value {
        let (x, y) = if signed { (self.sext64(a), self.sext64(b)) } else { (self.zext64(a), self.zext64(b)) };
        let p = self.mul(x, y);
        let k = self.c64(32);
        let h = self.bin(BinOp::LShr, p, k);
        self.trunc32(h)
    }
    pub fn bfm(&mut self, a: Value, b: Value) -> Value {
        let w = self.andc(a, 31);
        let wz = self.cmpc(Pred::Eq, w, 0);
        let k = self.c(32);
        let s = self.sub(k, w);
        let s = self.andc(s, 31);
        let all = self.c(u32::MAX);
        let m = self.bin(BinOp::LShr, all, s);
        let m = self.selc(wz, 0, m);
        self.shl(m, b)
    }
    pub fn ffbh_i(&mut self, v: Value) -> Value {
        let z = self.c(0);
        let neg = self.cmp(Pred::SLt, v, z);
        let n = self.not(v);
        let x = self.sel(neg, n, v);
        self.lzcnt(x)
    }
    pub fn sbfe_u(&mut self, v: Value, c: Value) -> Value {
        let o = self.andc(c, 31);
        let w = self.lshrc(c, 16);
        let w = self.andc(w, 0x7F);
        let x = self.bin(BinOp::LShr, v, o);
        let k = self.c(32);
        let s = self.sub(k, w);
        let s = self.andc(s, 31);
        let all = self.c(u32::MAX);
        let m = self.bin(BinOp::LShr, all, s);
        let xm = self.and(x, m);
        let big = self.cmpc(Pred::UGe, w, 32);
        let r = self.sel(big, x, xm);
        let wz = self.cmpc(Pred::Eq, w, 0);
        self.selc(wz, 0, r)
    }
    pub fn sbfe_i(&mut self, v: Value, c: Value) -> Value {
        let o = self.andc(c, 31);
        let w = self.lshrc(c, 16);
        let w = self.andc(w, 0x7F);
        let k = self.c(32);
        let s = self.sub(k, w);
        let s = self.andc(s, 31);
        let x = self.bin(BinOp::AShr, v, o);
        let y = self.bin(BinOp::Shl, x, s);
        let y = self.bin(BinOp::AShr, y, s);
        let big = self.cmpc(Pred::UGe, w, 32);
        let r = self.sel(big, x, y);
        let wz = self.cmpc(Pred::Eq, w, 0);
        self.selc(wz, 0, r)
    }
    pub fn bfe_u(&mut self, v: Value, o: Value, w: Value) -> Value {
        let o = self.andc(o, 31);
        let w = self.andc(w, 31);
        let x = self.bin(BinOp::LShr, v, o);
        let k = self.c(32);
        let s = self.sub(k, w);
        let s = self.andc(s, 31);
        let all = self.c(u32::MAX);
        let m = self.bin(BinOp::LShr, all, s);
        let r = self.and(x, m);
        let wz = self.cmpc(Pred::Eq, w, 0);
        self.selc(wz, 0, r)
    }
    pub fn bfe_i(&mut self, v: Value, o: Value, w: Value) -> Value {
        let o = self.andc(o, 31);
        let w = self.andc(w, 31);
        let k = self.c(32);
        let s = self.sub(k, w);
        let s = self.andc(s, 31);
        let x = self.bin(BinOp::AShr, v, o);
        let y = self.bin(BinOp::Shl, x, s);
        let y = self.bin(BinOp::AShr, y, s);
        let wz = self.cmpc(Pred::Eq, w, 0);
        self.selc(wz, 0, y)
    }
    pub fn wqm(&mut self, m: Value) -> Value {
        let a = self.lshrc(m, 1);
        let q = self.or(m, a);
        let b = self.lshrc(q, 2);
        let q = self.or(q, b);
        let q = self.andc(q, 0x1111_1111);
        let k = self.c(15);
        self.mul(q, k)
    }
}

/// Instructions without an effect in this execution model.
pub const NOOPS: &[&str] = &[
    "s_nop", "s_waitcnt", "s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt", "s_waitcnt_depctr",
    "s_inst_prefetch", "s_clause", "s_code_end", "s_setprio", "s_sleep", "s_icache_inv", "s_dcache_inv", "s_gl1_inv", "s_ttracedata",
    "s_ttracedata_imm", "v_nop", "s_denorm_mode", "s_sendmsg", "s_sendmsghalt", "s_set_gpr_idx_off", "s_version", "buffer_gl0_inv",
    "buffer_gl1_inv", "ds_nop",
];

pub fn f32_const(code: u64) -> Option<f32> {
    Some(match code {
        240 => 0.5,
        241 => -0.5,
        242 => 1.0,
        243 => -1.0,
        244 => 2.0,
        245 => -2.0,
        246 => 4.0,
        247 => -4.0,
        248 => 0.159_154_94,
        _ => return None,
    })
}

pub fn f16_const(code: u64) -> Option<u32> {
    Some(match code {
        240 => 0x3800,
        241 => 0xB800,
        242 => 0x3C00,
        243 => 0xBC00,
        244 => 0x4000,
        245 => 0xC000,
        246 => 0x4400,
        247 => 0xC400,
        248 => 0x3118,
        _ => return None,
    })
}

pub fn cmp_op(rest: &str) -> Option<(&'static str, bool)> {
    let (op, ty) = rest.rsplit_once('_')?;
    let signed = ty == "i32";
    Some((
        match op {
            "eq" => "==",
            "lg" => "!=",
            "gt" => ">",
            "ge" => ">=",
            "lt" => "<",
            "le" => "<=",
            _ => return None,
        },
        signed,
    ))
}

pub fn int_pred(op: &str, signed: bool) -> Pred {
    match (op, signed) {
        ("==", _) => Pred::Eq,
        ("!=", _) => Pred::Ne,
        (">", false) => Pred::UGt,
        (">=", false) => Pred::UGe,
        ("<", false) => Pred::ULt,
        ("<=", false) => Pred::ULe,
        (">", true) => Pred::SGt,
        (">=", true) => Pred::SGe,
        ("<", true) => Pred::SLt,
        _ => Pred::SLe,
    }
}

/// Layout of the SDWA / DPP extension dword for this instruction's base encoding.
pub fn ext_encoding(d: &Decoded) -> Option<&'static Encoding> {
    let base = match d.enc.name {
        "ENC_VOP1" => "VOP1",
        "ENC_VOP2" => "VOP2",
        "ENC_VOPC" => "VOPC",
        _ => return None,
    };
    let name = match (d.variant, base) {
        (Variant::Sdwa, "VOPC") => "VOPC_VOP_SDWA_SDST_ENC".to_string(),
        (Variant::Sdwa, b) => format!("{b}_VOP_SDWA"),
        (Variant::Dpp16, b) => format!("{b}_VOP_DPP16"),
        (Variant::Dpp8, b) => format!("{b}_VOP_DPP8"),
        _ => return None,
    };
    rdna2::encoding(&name)
}

impl Lw {
    pub fn orc(&mut self, a: Value, k: u32) -> Value {
        let k = self.c(k);
        self.or(a, k)
    }
}

/// `(id, offset, width)` of an `OPR_HWREG` `SIMM16`.
fn decode_hwreg(simm16: u16) -> (u32, u32, u32) {
    ((simm16 & 0x3F) as u32, ((simm16 >> 6) & 0x1F) as u32, (((simm16 >> 11) & 0x1F) + 1) as u32)
}

/// Bits of MODE[1:0] (sp) and MODE[3:2] (dp / f16) a `s_setreg*` touches, and its offset.
pub fn hwreg_mode_round_masks(simm16: u16) -> (u32, u32, u32) {
    let (id, offset, width) = decode_hwreg(simm16);
    if id != 1 {
        return (0, 0, 0);
    }
    let covers = |b: u32| b >= offset && b < offset + width;
    let sp = (covers(0) as u32) | ((covers(1) as u32) << 1);
    let dp = (covers(2) as u32) | ((covers(3) as u32) << 1);
    (sp, dp, offset)
}

/// The value `s_setreg_imm32_b32` writes to MODE[base +: 2] (masked).
pub fn hwreg_imm_field(imm: u32, offset: u32, base: u32, mask: u32) -> u32 {
    let mut v = 0u32;
    if mask & 1 != 0 {
        v |= (imm >> (base - offset)) & 1;
    }
    if mask & 2 != 0 {
        v |= ((imm >> (base + 1 - offset)) & 1) << 1;
    }
    v
}

/// Instructions whose f16 result rounding obeys MODE[3:2].
pub fn is_f16_rounded_op(name: &str) -> bool {
    matches!(
        name,
        "v_interp_p2_f16" | "v_cvt_f16_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" | "v_pk_add_f16" | "v_pk_mul_f16" | "v_pk_fma_f16"
            | "v_pk_fmac_f16" | "v_pk_min_f16" | "v_pk_max_f16" | "v_add_f16" | "v_sub_f16" | "v_subrev_f16" | "v_mul_f16" | "v_min_f16"
            | "v_max_f16" | "v_fma_f16" | "v_med3_f16" | "v_min3_f16" | "v_max3_f16" | "v_frexp_mant_f16" | "v_mad_f16" | "v_div_fixup_f16" | "v_fmac_f16"
            | "v_fmamk_f16" | "v_fmaak_f16" | "v_ldexp_f16" | "v_rcp_f16" | "v_rsq_f16" | "v_sqrt_f16" | "v_log_f16" | "v_exp_f16"
            | "v_sin_f16" | "v_cos_f16" | "v_fract_f16" | "v_floor_f16" | "v_ceil_f16" | "v_trunc_f16" | "v_rndne_f16" | "v_cvt_f16_u16"
            | "v_cvt_f16_i16"
    )
}

/// Instructions whose f64 rounding obeys MODE[3:2].
pub fn is_f64_rounded_op(name: &str) -> bool {
    name.contains("_f64")
        && !name.starts_with("v_cmp_")
        && !name.starts_with("v_cmpx_")
        && !matches!(name, "v_cvt_f32_f64" | "v_cvt_i32_f64" | "v_cvt_u32_f64" | "v_frexp_exp_i32_f64")
}

/// The recompiler's `configure_round_modes`: (dynamic f16 rounding, has f64 rounding ops).
pub fn round_modes(insts: &[Decoded], stop_at_endpgm: bool) -> (bool, bool) {
    let (mut f16, mut f64, mut writes) = (false, false, false);
    for d in insts {
        let name = d.inst.name.to_ascii_lowercase();
        if stop_at_endpgm && matches!(name.as_str(), "s_endpgm" | "s_endpgm_saved" | "s_endpgm_ordered_ps_done") {
            break;
        }
        f16 |= is_f16_rounded_op(&name);
        f64 |= is_f64_rounded_op(&name);
        let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) as u16;
        match name.as_str() {
            "s_round_mode" => writes |= (simm >> 2) & 3 != 0,
            "s_setreg_imm32_b32" => {
                let (_, dp_mask, off) = hwreg_mode_round_masks(simm);
                writes |= dp_mask != 0 && hwreg_imm_field(d.literal.unwrap_or(0), off, 2, dp_mask) != 0;
            }
            "s_setreg_b32" => writes |= hwreg_mode_round_masks(simm).1 != 0,
            _ => {}
        }
    }
    (writes && f16 && !f64, f64)
}

/// Constant folding of an integer binop on `w`-bit values (None: undefined in LLVM —
/// division by zero, shifts by the width or more — or not folded).
fn fold_bin(op: BinOp, x: u64, y: u64, w: u32) -> Option<u64> {
    let sx = |v: u64| if w >= 64 { v as i64 } else { ((v << (64 - w)) as i64) >> (64 - w) };
    Some(match op {
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::And => x & y,
        BinOp::Or => x | y,
        BinOp::Xor => x ^ y,
        BinOp::Shl if y < w as u64 => x << y,
        BinOp::LShr if y < w as u64 => x >> y,
        BinOp::AShr if y < w as u64 => (sx(x) >> y) as u64,
        BinOp::UDiv if y != 0 => x / y,
        BinOp::URem if y != 0 => x % y,
        _ => return None,
    })
}
