//! A DXIL module in memory (an LLVM 3.7 module restricted to what DXIL uses)
//! and its serialisation to bitcode.
//!
//! The module holds one defined function (the entry point) plus declarations
//! (`dx.op.*` intrinsics). Values are symbolic while the module is built
//! ([`Value`]) and numbered when written, in LLVM's order: global variables,
//! functions, module-level constants, then the entry's instruction results.
//! All constants live at module level, which LLVM permits and keeps function
//! bodies simple.
//!
//! Record layouts follow LLVM 3.7's `LLVMBitCodes.h` / `BitcodeWriter.cpp`
//! (University of Illinois/NCSA licence), checked against DXC's own output.

use std::collections::HashMap;

use super::bitstream::{signed, BitWriter};

mod opt;

// ---------------------------------------------------------------- types

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ty(u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TypeDef {
    Void,
    Label,
    Metadata,
    Half,
    Float,
    Double,
    Int(u32),
    Ptr(Ty, u32),
    Struct(Option<String>, Vec<Ty>),
    Array(u64, Ty),
    Func(Ty, Vec<Ty>),
}

// --------------------------------------------------------------- values

/// A value: global variable, function, module constant or instruction result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    Global(u32),
    Func(u32),
    Const(u32),
    Inst(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ConstDef {
    /// Sign-extended to 64 bits from the type's width.
    Int(Ty, i64),
    /// Raw IEEE bits of the type's width.
    Float(Ty, u64),
    Undef(Ty),
    Null(Ty),
    Aggregate(Ty, Vec<Value>),
}

struct FuncDecl {
    name: String,
    ty: Ty,
    attrs: FnAttrs,
    defined: bool,
}

/// Function attribute sets DXIL intrinsics use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FnAttrs {
    None,
    /// `nounwind readnone`
    ReadNone,
    /// `nounwind readonly`
    ReadOnly,
    /// `nounwind`
    NoUnwind,
    /// `noduplicate nounwind` (barriers)
    NoDuplicate,
}

impl FnAttrs {
    /// Index into the PARAMATTR table (+1; 0 = none).
    fn index(self) -> u64 {
        match self {
            FnAttrs::None => 0,
            FnAttrs::ReadNone => 1,
            FnAttrs::ReadOnly => 2,
            FnAttrs::NoUnwind => 3,
            FnAttrs::NoDuplicate => 4,
        }
    }
}

struct GlobalDef {
    name: String,
    /// Value type (the global itself is a pointer to it).
    ty: Ty,
    ptr: Ty,
    addrspace: u32,
    init: Option<Value>,
    constant: bool,
    align: u32,
}

// --------------------------------------------------------- instructions

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    UDiv = 3,
    SDiv = 4,
    URem = 5,
    SRem = 6,
    Shl = 7,
    LShr = 8,
    AShr = 9,
    And = 10,
    Or = 11,
    Xor = 12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastOp {
    Trunc = 0,
    ZExt = 1,
    SExt = 2,
    FPToUI = 3,
    FPToSI = 4,
    UIToFP = 5,
    SIToFP = 6,
    FPTrunc = 7,
    FPExt = 8,
    BitCast = 11,
}

/// Comparison predicates (LLVM `CmpInst::Predicate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pred {
    FFalse = 0,
    FOEq = 1,
    FOGt = 2,
    FOGe = 3,
    FOLt = 4,
    FOLe = 5,
    FONe = 6,
    FOrd = 7,
    FUno = 8,
    FUEq = 9,
    FUGt = 10,
    FUGe = 11,
    FULt = 12,
    FULe = 13,
    FUNe = 14,
    FTrue = 15,
    Eq = 32,
    Ne = 33,
    UGt = 34,
    UGe = 35,
    ULt = 36,
    ULe = 37,
    SGt = 38,
    SGe = 39,
    SLt = 40,
    SLe = 41,
}

/// LLVM atomicrmw operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RmwOp {
    Xchg = 0,
    Add = 1,
    Sub = 2,
    And = 3,
    Nand = 4,
    Or = 5,
    Xor = 6,
    Max = 7,
    Min = 8,
    UMax = 9,
    UMin = 10,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Op {
    Bin(BinOp, Value, Value, u64),
    Cast(CastOp, Value, Ty),
    Cmp(Pred, Value, Value),
    Select(Value, Value, Value),
    Extract(Value, u32),
    Call(u32, Vec<Value>),
    Phi(Vec<(Value, Block)>),
    Load(Value, u32),
    Store(Value, Value, u32),
    Gep(Value, Vec<Value>),
    AtomicRmw(RmwOp, Value, Value),
    CmpXchg(Value, Value, Value),
    /// allocated type, array size (an i32 constant), alignment
    Alloca(Ty, Value, u32),
    Br(Block),
    CondBr(Value, Block, Block),
    Switch(Value, Block, Vec<(Value, Block)>),
    Ret,
    Unreachable,
}

struct InstDef {
    op: Op,
    ty: Ty,
}

// ------------------------------------------------------------- metadata

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Md(u32);

enum MdDef {
    Str(String),
    Val(Ty, Value),
    Node(Vec<Option<Md>>),
}

// --------------------------------------------------------------- module

/// Fast-math flags (LLVM 3.7 bitcode encoding).
pub mod fmf {
    pub const NONE: u64 = 0;
    pub const UNSAFE_ALGEBRA: u64 = 1;
    pub const NO_NANS: u64 = 2;
    pub const NO_INFS: u64 = 4;
    pub const NO_SIGNED_ZEROS: u64 = 8;
    pub const ALLOW_RECIPROCAL: u64 = 16;
    /// HLSL's default (`fast`).
    pub const FAST: u64 = 31;
}

pub struct Module {
    types: Vec<TypeDef>,
    type_ids: HashMap<TypeDef, Ty>,
    consts: Vec<ConstDef>,
    const_ids: HashMap<ConstDef, u32>,
    funcs: Vec<FuncDecl>,
    func_ids: HashMap<String, u32>,
    globals: Vec<GlobalDef>,
    insts: Vec<InstDef>,
    blocks: Vec<Vec<u32>>,
    cur: Block,
    md: Vec<MdDef>,
    named_md: Vec<(String, Vec<Md>)>,
    md_strings: HashMap<String, Md>,
    /// The first misuse of the builder (see [`Module::error`]); later ones are dropped.
    error: std::cell::RefCell<Option<String>>,
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
}

impl Module {
    pub fn new() -> Self {
        Module {
            types: Vec::new(),
            type_ids: HashMap::new(),
            consts: Vec::new(),
            const_ids: HashMap::new(),
            funcs: Vec::new(),
            func_ids: HashMap::new(),
            globals: Vec::new(),
            insts: Vec::new(),
            blocks: vec![Vec::new()],
            cur: Block(0),
            md: Vec::new(),
            named_md: Vec::new(),
            md_strings: HashMap::new(),
            error: std::cell::RefCell::new(None),
        }
    }

    // ---- errors

    /// Records a misuse of the builder (the first one is kept). The builder carries on with a
    /// placeholder so callers need no error paths; [`Module::try_to_bitcode`] (and the
    /// `Shader::try_finish*` methods) then fail with this message instead of emitting.
    pub fn fail(&self, msg: impl Into<String>) {
        let mut e = self.error.borrow_mut();
        if e.is_none() {
            *e = Some(msg.into());
        }
    }
    /// The first recorded builder error, if any.
    pub fn error(&self) -> Option<String> {
        self.error.borrow().clone()
    }

    // ---- types

    fn ty(&mut self, t: TypeDef) -> Ty {
        if let Some(&id) = self.type_ids.get(&t) {
            return id;
        }
        let id = Ty(self.types.len() as u32);
        self.types.push(t.clone());
        self.type_ids.insert(t, id);
        id
    }
    pub fn void(&mut self) -> Ty {
        self.ty(TypeDef::Void)
    }
    pub fn label(&mut self) -> Ty {
        self.ty(TypeDef::Label)
    }
    pub fn metadata(&mut self) -> Ty {
        self.ty(TypeDef::Metadata)
    }
    pub fn int(&mut self, bits: u32) -> Ty {
        self.ty(TypeDef::Int(bits))
    }
    pub fn i1(&mut self) -> Ty {
        self.int(1)
    }
    pub fn i8(&mut self) -> Ty {
        self.int(8)
    }
    pub fn i32(&mut self) -> Ty {
        self.int(32)
    }
    pub fn i64(&mut self) -> Ty {
        self.int(64)
    }
    pub fn f16(&mut self) -> Ty {
        self.ty(TypeDef::Half)
    }
    pub fn f32(&mut self) -> Ty {
        self.ty(TypeDef::Float)
    }
    pub fn f64(&mut self) -> Ty {
        self.ty(TypeDef::Double)
    }
    pub fn ptr(&mut self, pointee: Ty, addrspace: u32) -> Ty {
        self.ty(TypeDef::Ptr(pointee, addrspace))
    }
    pub fn array(&mut self, n: u64, elem: Ty) -> Ty {
        self.ty(TypeDef::Array(n, elem))
    }
    pub fn named_struct(&mut self, name: &str, elems: &[Ty]) -> Ty {
        self.ty(TypeDef::Struct(Some(name.to_string()), elems.to_vec()))
    }
    pub fn func_ty(&mut self, ret: Ty, params: &[Ty]) -> Ty {
        self.ty(TypeDef::Func(ret, params.to_vec()))
    }

    pub fn is_float(&self, t: Ty) -> bool {
        matches!(self.types[t.0 as usize], TypeDef::Half | TypeDef::Float | TypeDef::Double)
    }
    pub fn int_width(&self, t: Ty) -> Option<u32> {
        match self.types[t.0 as usize] {
            TypeDef::Int(w) => Some(w),
            _ => None,
        }
    }
    fn struct_elem(&mut self, t: Ty, i: u32) -> Ty {
        let r = match &self.types[t.0 as usize] {
            TypeDef::Struct(_, e) => e.get(i as usize).copied(),
            TypeDef::Array(_, e) => Some(*e),
            _ => None,
        };
        r.unwrap_or_else(|| {
            self.fail(format!("extractvalue {i} from {:?}", self.types[t.0 as usize]));
            self.i32()
        })
    }
    fn pointee_of(&self, t: Ty) -> Option<(Ty, u32)> {
        match self.types[t.0 as usize] {
            TypeDef::Ptr(p, a) => Some((p, a)),
            _ => None,
        }
    }
    fn pointee(&mut self, t: Ty) -> (Ty, u32) {
        self.pointee_of(t).unwrap_or_else(|| {
            self.fail(format!("not a pointer: {:?}", self.types[t.0 as usize]));
            (self.i32(), 0)
        })
    }

    // ---- constants

    fn constant(&mut self, c: ConstDef) -> Value {
        if let Some(&id) = self.const_ids.get(&c) {
            return Value::Const(id);
        }
        let id = self.consts.len() as u32;
        self.consts.push(c.clone());
        self.const_ids.insert(c, id);
        Value::Const(id)
    }
    /// An integer constant; `v` is truncated / sign-extended to the type's width.
    pub fn const_int(&mut self, ty: Ty, v: u64) -> Value {
        let w = self.int_width(ty).unwrap_or_else(|| {
            self.fail(format!("const_int of {:?}", self.types[ty.0 as usize]));
            64
        });
        let sext = if w >= 64 { v as i64 } else { ((v << (64 - w)) as i64) >> (64 - w) };
        self.constant(ConstDef::Int(ty, sext))
    }
    pub fn i32c(&mut self, v: u32) -> Value {
        let t = self.i32();
        self.const_int(t, v as u64)
    }
    pub fn i8c(&mut self, v: u8) -> Value {
        let t = self.i8();
        self.const_int(t, v as u64)
    }
    pub fn i1c(&mut self, v: bool) -> Value {
        let t = self.i1();
        self.const_int(t, v as u64)
    }
    pub fn f32c(&mut self, bits: u32) -> Value {
        let t = self.f32();
        self.constant(ConstDef::Float(t, bits as u64))
    }
    /// A floating-point constant of a half / float / double type from raw bits.
    /// Bits above the type's width are ignored (as LLVM's reader truncates
    /// them), so equal constants intern to one value.
    pub fn const_float(&mut self, ty: Ty, bits: u64) -> Value {
        let bits = match self.types[ty.0 as usize] {
            TypeDef::Half => bits & 0xFFFF,
            TypeDef::Float => bits & 0xFFFF_FFFF,
            TypeDef::Double => bits,
            ref other => {
                self.fail(format!("const_float of {other:?}"));
                bits
            }
        };
        self.constant(ConstDef::Float(ty, bits))
    }
    pub fn undef(&mut self, ty: Ty) -> Value {
        self.constant(ConstDef::Undef(ty))
    }
    pub fn null(&mut self, ty: Ty) -> Value {
        self.constant(ConstDef::Null(ty))
    }
    pub fn aggregate(&mut self, ty: Ty, elems: &[Value]) -> Value {
        self.constant(ConstDef::Aggregate(ty, elems.to_vec()))
    }

    // ---- functions and globals

    /// Declares (or finds) a function by name.
    pub fn declare(&mut self, name: &str, ty: Ty, attrs: FnAttrs) -> u32 {
        if let Some(&f) = self.func_ids.get(name) {
            if self.funcs[f as usize].ty != ty {
                self.fail(format!("function {name} redeclared with another type"));
            }
            return f;
        }
        let f = self.funcs.len() as u32;
        self.funcs.push(FuncDecl { name: name.to_string(), ty, attrs, defined: false });
        self.func_ids.insert(name.to_string(), f);
        f
    }

    /// Declares the entry point `void name()`; its body is built with the
    /// instruction methods.
    pub fn define_entry(&mut self, name: &str) -> u32 {
        let v = self.void();
        let ty = self.func_ty(v, &[]);
        let f = self.declare(name, ty, FnAttrs::None);
        if self.funcs.iter().any(|f| f.defined) {
            self.fail(format!("a second defined function ({name})"));
        }
        self.funcs[f as usize].defined = true;
        f
    }

    pub fn func_value(&self, f: u32) -> Value {
        Value::Func(f)
    }

    /// A global variable (e.g. thread-group shared memory in address space 3).
    pub fn global(&mut self, name: &str, ty: Ty, addrspace: u32, init: Option<Value>, constant: bool, align: u32) -> Value {
        let ptr = self.ptr(ty, addrspace);
        let id = self.globals.len() as u32;
        self.globals.push(GlobalDef { name: name.to_string(), ty, ptr, addrspace, init, constant, align });
        Value::Global(id)
    }

    pub fn type_of(&mut self, v: Value) -> Ty {
        match v {
            Value::Global(g) => self.globals[g as usize].ptr,
            Value::Func(f) => {
                let t = self.funcs[f as usize].ty;
                self.ptr(t, 0)
            }
            Value::Const(c) => match &self.consts[c as usize] {
                ConstDef::Int(t, _) | ConstDef::Float(t, _) | ConstDef::Undef(t) | ConstDef::Null(t) | ConstDef::Aggregate(t, _) => *t,
            },
            Value::Inst(i) => self.insts[i as usize].ty,
        }
    }

    // ---- blocks

    pub fn new_block(&mut self) -> Block {
        self.blocks.push(Vec::new());
        Block(self.blocks.len() as u32 - 1)
    }
    pub fn entry_block(&self) -> Block {
        Block(0)
    }
    pub fn set_block(&mut self, b: Block) {
        self.cur = b;
    }
    pub fn current_block(&self) -> Block {
        self.cur
    }
    /// Whether the current block already ends in a terminator.
    pub fn terminated(&self) -> bool {
        self.blocks[self.cur.0 as usize].last().is_some_and(|&i| {
            matches!(self.insts[i as usize].op, Op::Br(_) | Op::CondBr(..) | Op::Switch(..) | Op::Ret | Op::Unreachable)
        })
    }

    fn push(&mut self, op: Op, ty: Ty) -> Value {
        if self.terminated() {
            self.fail(format!("instruction after a terminator in block {:?}", self.cur));
        }
        let id = self.insts.len() as u32;
        self.insts.push(InstDef { op, ty });
        self.blocks[self.cur.0 as usize].push(id);
        Value::Inst(id)
    }

    // ---- instructions

    pub fn bin(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        let t = self.type_of(a);
        debug_assert_eq!(t, self.type_of(b), "binop operand types differ");
        self.push(Op::Bin(op, a, b, 0), t)
    }
    /// Floating-point binop (`Add` = fadd, `Sub` = fsub, `Mul` = fmul,
    /// `SDiv` = fdiv, `SRem` = frem — the bitcode reader rejects `UDiv` /
    /// `URem` on floats) with fast-math flags.
    pub fn fbin(&mut self, op: BinOp, a: Value, b: Value, flags: u64) -> Value {
        let t = self.type_of(a);
        debug_assert!(self.is_float(t));
        self.push(Op::Bin(op, a, b, flags), t)
    }
    /// Integer binop with `nuw` (1) / `nsw` (2) / `exact` (1) flags.
    pub fn bin_flags(&mut self, op: BinOp, a: Value, b: Value, flags: u64) -> Value {
        let t = self.type_of(a);
        self.push(Op::Bin(op, a, b, flags), t)
    }
    pub fn cast(&mut self, op: CastOp, v: Value, to: Ty) -> Value {
        self.push(Op::Cast(op, v, to), to)
    }
    pub fn bitcast(&mut self, v: Value, to: Ty) -> Value {
        if self.type_of(v) == to {
            return v;
        }
        self.cast(CastOp::BitCast, v, to)
    }
    pub fn cmp(&mut self, p: Pred, a: Value, b: Value) -> Value {
        let t = self.i1();
        self.push(Op::Cmp(p, a, b), t)
    }
    pub fn select(&mut self, c: Value, t: Value, f: Value) -> Value {
        let ty = self.type_of(t);
        self.push(Op::Select(c, t, f), ty)
    }
    pub fn extract(&mut self, agg: Value, idx: u32) -> Value {
        let at = self.type_of(agg);
        let t = self.struct_elem(at, idx);
        self.push(Op::Extract(agg, idx), t)
    }
    pub fn call(&mut self, f: u32, args: &[Value]) -> Value {
        let ret = match &self.types[self.funcs[f as usize].ty.0 as usize] {
            TypeDef::Func(r, p) => {
                let r = *r;
                if p.len() != args.len() {
                    self.fail(format!("call to {} with {} arguments", self.funcs[f as usize].name, args.len()));
                }
                r
            }
            _ => {
                self.fail(format!("call to {}, not a function type", self.funcs[f as usize].name));
                self.void()
            }
        };
        self.push(Op::Call(f, args.to_vec()), ret)
    }
    /// A phi of type `ty`; incoming edges are added with [`Module::add_incoming`].
    pub fn phi(&mut self, ty: Ty) -> Value {
        // Phis must lead their block.
        debug_assert!(self.blocks[self.cur.0 as usize].iter().all(|&i| matches!(self.insts[i as usize].op, Op::Phi(_))));
        self.push(Op::Phi(Vec::new()), ty)
    }
    /// A phi of type `ty` placed after the existing phis of block `b`, whatever
    /// the current block is and whether `b` already holds other instructions
    /// (on-the-fly SSA construction creates phis in blocks already filled).
    pub fn phi_at(&mut self, b: Block, ty: Ty) -> Value {
        let id = self.insts.len() as u32;
        self.insts.push(InstDef { op: Op::Phi(Vec::new()), ty });
        let list = &mut self.blocks[b.0 as usize];
        let at = list.iter().position(|&i| !matches!(self.insts[i as usize].op, Op::Phi(_))).unwrap_or(list.len());
        list.insert(at, id);
        Value::Inst(id)
    }
    /// A phi of type `ty` not yet in any block (place it with [`Module::place_phi`];
    /// one never placed is simply not written).
    pub fn detached_phi(&mut self, ty: Ty) -> Value {
        let id = self.insts.len() as u32;
        self.insts.push(InstDef { op: Op::Phi(Vec::new()), ty });
        Value::Inst(id)
    }
    /// Puts a [`Module::detached_phi`] after the existing phis of block `b`.
    pub fn place_phi(&mut self, b: Block, phi: Value) {
        let Value::Inst(id) = phi else { panic!("not a phi") };
        let list = &mut self.blocks[b.0 as usize];
        let at = list.iter().position(|&i| !matches!(self.insts[i as usize].op, Op::Phi(_))).unwrap_or(list.len());
        list.insert(at, id);
    }
    /// Number of incoming edges a phi has so far.
    pub fn phi_incoming(&self, phi: Value) -> Vec<(Value, Block)> {
        let Value::Inst(i) = phi else { return Vec::new() };
        match &self.insts[i as usize].op {
            Op::Phi(list) => list.clone(),
            _ => Vec::new(),
        }
    }
    pub fn add_incoming(&mut self, phi: Value, v: Value, from: Block) {
        if let Value::Inst(i) = phi {
            if let Op::Phi(list) = &mut self.insts[i as usize].op {
                list.push((v, from));
                return;
            }
        }
        self.fail(format!("add_incoming to a non-phi {phi:?}"));
    }
    pub fn load(&mut self, ptr: Value, align: u32) -> Value {
        let pt = self.type_of(ptr);
        let (t, _) = self.pointee(pt);
        self.push(Op::Load(ptr, align), t)
    }
    pub fn store(&mut self, ptr: Value, v: Value, align: u32) {
        let t = self.void();
        self.push(Op::Store(ptr, v, align), t);
    }
    /// `getelementptr inbounds` with constant / variable indices.
    pub fn gep(&mut self, ptr: Value, idx: &[Value]) -> Value {
        let pt = self.type_of(ptr);
        let (mut t, space) = self.pointee(pt);
        for _ in 1..idx.len() {
            t = match &self.types[t.0 as usize] {
                TypeDef::Array(_, e) => *e,
                other => {
                    self.fail(format!("gep into {other:?}"));
                    break;
                }
            };
        }
        let rt = self.ptr(t, space);
        self.push(Op::Gep(ptr, idx.to_vec()), rt)
    }
    pub fn atomic_rmw(&mut self, op: RmwOp, ptr: Value, v: Value) -> Value {
        let t = self.type_of(v);
        self.push(Op::AtomicRmw(op, ptr, v), t)
    }
    /// `cmpxchg` returning the old value (`{ T, i1 }` extracted at index 0).
    pub fn cmpxchg(&mut self, ptr: Value, cmp: Value, new: Value) -> Value {
        let t = self.type_of(cmp);
        let b = self.i1();
        let st = self.ty(TypeDef::Struct(None, vec![t, b]));
        let pair = self.push(Op::CmpXchg(ptr, cmp, new), st);
        self.extract(pair, 0)
    }
    /// `alloca <ty>, align <align>` (a function-local variable, e.g. an indexable array;
    /// emit it in the entry block).
    pub fn alloca(&mut self, ty: Ty, align: u32) -> Value {
        let one = self.i32c(1);
        let p = self.ptr(ty, 0);
        self.push(Op::Alloca(ty, one, align), p)
    }
    pub fn br(&mut self, to: Block) {
        let t = self.void();
        self.push(Op::Br(to), t);
    }
    pub fn cond_br(&mut self, c: Value, t: Block, f: Block) {
        let v = self.void();
        self.push(Op::CondBr(c, t, f), v);
    }
    pub fn switch(&mut self, v: Value, default: Block, cases: &[(Value, Block)]) {
        let t = self.void();
        self.push(Op::Switch(v, default, cases.to_vec()), t);
    }
    pub fn ret(&mut self) {
        let t = self.void();
        self.push(Op::Ret, t);
    }
    pub fn unreachable(&mut self) {
        let t = self.void();
        self.push(Op::Unreachable, t);
    }

    // ---- metadata

    pub fn md_str(&mut self, s: &str) -> Md {
        if let Some(&m) = self.md_strings.get(s) {
            return m;
        }
        let m = Md(self.md.len() as u32);
        self.md.push(MdDef::Str(s.to_string()));
        self.md_strings.insert(s.to_string(), m);
        m
    }
    pub fn md_value(&mut self, v: Value) -> Md {
        let t = self.type_of(v);
        let m = Md(self.md.len() as u32);
        self.md.push(MdDef::Val(t, v));
        m
    }
    pub fn md_i32(&mut self, v: u32) -> Md {
        let c = self.i32c(v);
        self.md_value(c)
    }
    pub fn md_i64(&mut self, v: u64) -> Md {
        let t = self.i64();
        let c = self.const_int(t, v);
        self.md_value(c)
    }
    pub fn md_i1(&mut self, v: bool) -> Md {
        let c = self.i1c(v);
        self.md_value(c)
    }
    pub fn md_node(&mut self, elems: &[Option<Md>]) -> Md {
        let m = Md(self.md.len() as u32);
        self.md.push(MdDef::Node(elems.to_vec()));
        m
    }
    pub fn named_md(&mut self, name: &str, nodes: &[Md]) {
        self.named_md.push((name.to_string(), nodes.to_vec()));
    }

    // ---------------------------------------------------- serialisation

    /// [`Module::to_bitcode`], or the first builder error (recorded while building or
    /// found while writing, e.g. a forward reference outside a phi).
    pub fn try_to_bitcode(&mut self, triple: &str, datalayout: &str) -> Result<Vec<u8>, String> {
        if let Some(e) = self.error() {
            return Err(e);
        }
        let b = self.write_bitcode(triple, datalayout);
        match self.error() {
            Some(e) => Err(e),
            None => Ok(b),
        }
    }

    /// The module as LLVM 3.7 bitcode (`BC\xC0\xDE` …). Panics on a builder error
    /// (recorded or found while writing); [`Module::try_to_bitcode`] reports it instead.
    pub fn to_bitcode(&mut self, triple: &str, datalayout: &str) -> Vec<u8> {
        self.try_to_bitcode(triple, datalayout).unwrap_or_else(|e| panic!("DXIL builder: {e}"))
    }

    fn write_bitcode(&mut self, triple: &str, datalayout: &str) -> Vec<u8> {
        // Every type the writer references must exist before the type table:
        // pointers to the declared functions (call / metadata operands).
        for f in 0..self.funcs.len() {
            self.type_of(Value::Func(f as u32));
        }

        // Blocks are written in reverse post-order of the entry's CFG (entry
        // first), so a lowering may create blocks in any order; unreachable blocks
        // are dropped. Value numbering follows that order.
        let (order, pos) = self.block_layout();

        // Global variables nothing references (e.g. an exchange array the code never
        // needed) are left out: the validator rejects unused internal declarations.
        // Likewise declarations of functions no longer called (after `optimize`).
        let mut gused = vec![false; self.globals.len()];
        let mut fused: Vec<bool> = self.funcs.iter().map(|f| f.defined).collect();
        {
            let mut mark = |v: Value| match v {
                Value::Global(g) => gused[g as usize] = true,
                Value::Func(f) => fused[f as usize] = true,
                _ => {}
            };
            for &b in &order {
                for &i in &self.blocks[b] {
                    let op = &self.insts[i as usize].op;
                    if let Op::Call(f, _) = op {
                        mark(Value::Func(*f));
                    }
                    op.for_each_operand(&mut mark);
                }
            }
            for c in &self.consts {
                if let ConstDef::Aggregate(_, e) = c {
                    e.iter().copied().for_each(&mut mark);
                }
            }
            for m in &self.md {
                if let MdDef::Val(_, v) = m {
                    mark(*v);
                }
            }
        }
        let mut gmap: Vec<Option<u64>> = vec![None; self.globals.len()];
        let mut n_globals = 0u64;
        for (g, used) in gused.iter().enumerate() {
            if *used {
                gmap[g] = Some(n_globals);
                n_globals += 1;
            }
        }
        let gmap = &gmap;
        let mut fmap: Vec<Option<u64>> = vec![None; self.funcs.len()];
        let mut n_funcs = 0u64;
        for (f, used) in fused.iter().enumerate() {
            if *used {
                fmap[f] = Some(n_funcs);
                n_funcs += 1;
            }
        }
        let fmap = &fmap;
        let n_consts = self.consts.len() as u64;
        let const_base = n_globals + n_funcs;
        let inst_base = const_base + n_consts;

        // Absolute value ids of instruction results.
        let mut inst_ids: Vec<Option<u64>> = vec![None; self.insts.len()];
        {
            let mut next = inst_base;
            for &b in &order {
                for &i in &self.blocks[b] {
                    if !self.is_void(self.insts[i as usize].ty) {
                        inst_ids[i as usize] = Some(next);
                        next += 1;
                    }
                }
            }
        }
        let abs = |v: Value| -> u64 {
            match v {
                Value::Global(g) => gmap[g as usize].expect("use of a dropped global"),
                Value::Func(f) => n_globals + fmap[f as usize].expect("use of a dropped function"),
                Value::Const(c) => const_base + c as u64,
                Value::Inst(i) => inst_ids[i as usize].unwrap_or_else(|| {
                    self.fail(format!("use of a void instruction %{i}"));
                    0
                }),
            }
        };

        let mut w = BitWriter::new();
        for b in [b'B', b'C', 0xC0, 0xDE] {
            w.emit(b as u64, 8);
        }
        w.enter_block(BLOCK_MODULE);
        w.record(MODULE_CODE_VERSION, &[1]);

        // Attribute groups: 1 = nounwind readnone, 2 = nounwind readonly,
        // 3 = nounwind, 4 = noduplicate nounwind.
        w.enter_block(BLOCK_PARAMATTR_GROUP);
        const FN: u64 = 0xFFFF_FFFF;
        w.record(PARAMATTR_GRP_CODE_ENTRY, &[1, FN, 0, ATTR_NOUNWIND, 0, ATTR_READNONE]);
        w.record(PARAMATTR_GRP_CODE_ENTRY, &[2, FN, 0, ATTR_NOUNWIND, 0, ATTR_READONLY]);
        w.record(PARAMATTR_GRP_CODE_ENTRY, &[3, FN, 0, ATTR_NOUNWIND]);
        w.record(PARAMATTR_GRP_CODE_ENTRY, &[4, FN, 0, ATTR_NODUPLICATE, 0, ATTR_NOUNWIND]);
        w.end_block();
        w.enter_block(BLOCK_PARAMATTR);
        for g in 1..=4 {
            w.record(PARAMATTR_CODE_ENTRY, &[g]);
        }
        w.end_block();

        self.write_types(&mut w);
        w.record_str(MODULE_CODE_TRIPLE, &[], triple);
        w.record_str(MODULE_CODE_DATALAYOUT, &[], datalayout);

        for g in self.globals.iter().enumerate().filter(|(i, _)| gused[*i]).map(|x| x.1) {
            // [valty, addrspace<<2 | explicit_type<<1 | isconst, initid+1, linkage, align, section,
            //  visibility, tls, unnamed_addr, externally_initialized, dllstorage, comdat]
            let init = g.init.map_or(0, |v| abs(v) + 1);
            let linkage = if g.init.is_some() { LINKAGE_INTERNAL } else { LINKAGE_EXTERNAL };
            let align = if g.align == 0 { 0 } else { g.align.trailing_zeros() as u64 + 1 };
            w.record(
                MODULE_CODE_GLOBALVAR,
                &[g.ty.0 as u64, ((g.addrspace as u64) << 2) | 2 | g.constant as u64, init, linkage, align, 0, 0, 0, 0, 0, 0, 0],
            );
        }
        for f in self.funcs.iter().enumerate().filter(|(i, _)| fused[*i]).map(|x| x.1) {
            // [type, cc, isproto, linkage, paramattr, align, section, visibility,
            //  gc, unnamed_addr, prologue, dllstorage, comdat, prefix, personality]
            w.record(
                MODULE_CODE_FUNCTION,
                &[f.ty.0 as u64, 0, (!f.defined) as u64, LINKAGE_EXTERNAL, f.attrs.index(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            );
        }

        // Module-level constants.
        if !self.consts.is_empty() {
            w.enter_block(BLOCK_CONSTANTS);
            let mut cur_ty = None;
            for c in &self.consts {
                let t = match c {
                    ConstDef::Int(t, _) | ConstDef::Float(t, _) | ConstDef::Undef(t) | ConstDef::Null(t) | ConstDef::Aggregate(t, _) => *t,
                };
                if cur_ty != Some(t) {
                    w.record(CST_CODE_SETTYPE, &[t.0 as u64]);
                    cur_ty = Some(t);
                }
                match c {
                    ConstDef::Int(_, v) => w.record(CST_CODE_INTEGER, &[signed(*v)]),
                    ConstDef::Float(_, bits) => w.record(CST_CODE_FLOAT, &[*bits]),
                    ConstDef::Undef(_) => w.record(CST_CODE_UNDEF, &[]),
                    ConstDef::Null(_) => w.record(CST_CODE_NULL, &[]),
                    ConstDef::Aggregate(_, e) => {
                        let ops: Vec<u64> = e.iter().map(|&v| abs(v)).collect();
                        w.record(CST_CODE_AGGREGATE, &ops)
                    }
                }
            }
            w.end_block();
        }

        // Metadata.
        if !self.md.is_empty() || !self.named_md.is_empty() {
            w.enter_block(BLOCK_METADATA);
            for m in &self.md {
                match m {
                    MdDef::Str(s) => w.record_str(METADATA_STRING, &[], s),
                    MdDef::Val(t, v) => w.record(METADATA_VALUE, &[t.0 as u64, abs(*v)]),
                    MdDef::Node(e) => {
                        let ops: Vec<u64> = e.iter().map(|m| m.map_or(0, |m| m.0 as u64 + 1)).collect();
                        w.record(METADATA_NODE, &ops)
                    }
                }
            }
            for (name, nodes) in &self.named_md {
                w.record_str(METADATA_NAME, &[], name);
                let ops: Vec<u64> = nodes.iter().map(|m| m.0 as u64).collect();
                w.record(METADATA_NAMED_NODE, &ops);
            }
            w.end_block();
        }

        // Module value symbol table: names of globals and functions.
        w.enter_block(BLOCK_VALUE_SYMTAB);
        for (i, g) in self.globals.iter().enumerate() {
            if let Some(k) = gmap[i] {
                w.record_str(VST_CODE_ENTRY, &[k], &g.name);
            }
        }
        for (i, f) in self.funcs.iter().enumerate() {
            if let Some(k) = fmap[i] {
                w.record_str(VST_CODE_ENTRY, &[n_globals + k], &f.name);
            }
        }
        w.end_block();

        // The entry's body.
        w.enter_block(BLOCK_FUNCTION);
        w.record(FUNC_CODE_DECLAREBLOCKS, &[order.len() as u64]);
        let mut inst_id = inst_base;
        for &b in &order {
            for k in 0..self.blocks[b].len() {
                let i = self.blocks[b][k] as usize;
                self.write_inst(&mut w, i, inst_id, &abs, &pos);
                if !self.is_void(self.insts[i].ty) {
                    inst_id += 1;
                }
            }
        }
        w.end_block();

        w.end_block(); // module
        w.finish()
    }

    fn is_void(&self, t: Ty) -> bool {
        matches!(self.types[t.0 as usize], TypeDef::Void)
    }

    fn write_types(&self, w: &mut BitWriter) {
        w.enter_block(BLOCK_TYPE);
        w.record(TYPE_CODE_NUMENTRY, &[self.types.len() as u64]);
        for t in &self.types {
            match t {
                TypeDef::Void => w.record(TYPE_CODE_VOID, &[]),
                TypeDef::Label => w.record(TYPE_CODE_LABEL, &[]),
                TypeDef::Metadata => w.record(TYPE_CODE_METADATA, &[]),
                TypeDef::Half => w.record(TYPE_CODE_HALF, &[]),
                TypeDef::Float => w.record(TYPE_CODE_FLOAT, &[]),
                TypeDef::Double => w.record(TYPE_CODE_DOUBLE, &[]),
                TypeDef::Int(n) => w.record(TYPE_CODE_INTEGER, &[*n as u64]),
                TypeDef::Ptr(p, a) => w.record(TYPE_CODE_POINTER, &[p.0 as u64, *a as u64]),
                TypeDef::Array(n, e) => w.record(TYPE_CODE_ARRAY, &[*n, e.0 as u64]),
                TypeDef::Func(r, p) => {
                    let mut ops = vec![0, r.0 as u64];
                    ops.extend(p.iter().map(|t| t.0 as u64));
                    w.record(TYPE_CODE_FUNCTION, &ops)
                }
                TypeDef::Struct(name, e) => {
                    let mut ops = vec![0];
                    ops.extend(e.iter().map(|t| t.0 as u64));
                    match name {
                        Some(n) => {
                            w.record_str(TYPE_CODE_STRUCT_NAME, &[], n);
                            w.record(TYPE_CODE_STRUCT_NAMED, &ops)
                        }
                        None => w.record(TYPE_CODE_STRUCT_ANON, &ops),
                    }
                }
            }
        }
        w.end_block();
    }

    /// Successor blocks of block `b` (from its terminator, if any).
    fn successors(&self, b: usize) -> Vec<Block> {
        match self.blocks[b].last().map(|&i| &self.insts[i as usize].op) {
            Some(Op::Br(t)) => vec![*t],
            Some(Op::CondBr(_, t, f)) => vec![*t, *f],
            Some(Op::Switch(_, d, cases)) => std::iter::once(*d).chain(cases.iter().map(|c| c.1)).collect(),
            _ => Vec::new(),
        }
    }

    /// The written block order (reverse post-order from the entry block, which
    /// keeps the creation order where that is already a valid layout) and each
    /// block's position in it (`None`: unreachable, not written).
    fn block_layout(&self) -> (Vec<usize>, Vec<Option<u32>>) {
        let n = self.blocks.len();
        let mut seen = vec![false; n];
        let mut post = Vec::with_capacity(n);
        // Iterative DFS; successors pushed so that earlier-created ones are visited
        // last, which makes the reverse post-order prefer creation order.
        let mut stack: Vec<(usize, Vec<Block>)> = vec![(0, {
            let mut s = self.successors(0);
            s.sort_by_key(|b| std::cmp::Reverse(b.0));
            s
        })];
        seen[0] = true;
        while let Some((b, succ)) = stack.last_mut() {
            match succ.pop() {
                Some(t) => {
                    let t = t.0 as usize;
                    if t < n && !seen[t] {
                        seen[t] = true;
                        let mut s = self.successors(t);
                        s.sort_by_key(|b| std::cmp::Reverse(b.0));
                        stack.push((t, s));
                    }
                }
                None => {
                    post.push(*b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        let mut pos = vec![None; n];
        for (k, &b) in post.iter().enumerate() {
            pos[b] = Some(k as u32);
        }
        (post, pos)
    }

    /// Replaces every use of `old` by `new` in the entry's instructions and in
    /// metadata (e.g. to drop a trivial phi). `old` itself stays in its block.
    pub fn replace_all_uses(&mut self, old: Value, new: Value) {
        let r = |v: &mut Value| {
            if *v == old {
                *v = new;
            }
        };
        for inst in &mut self.insts {
            match &mut inst.op {
                Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                    r(a);
                    r(b)
                }
                Op::Cast(_, a, _) | Op::Extract(a, _) | Op::Load(a, _) | Op::CondBr(a, _, _) | Op::Alloca(_, a, _) => r(a),
                Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                    r(a);
                    r(b);
                    r(c)
                }
                Op::Call(_, args) => args.iter_mut().for_each(r),
                Op::Phi(inc) => inc.iter_mut().for_each(|(v, _)| r(v)),
                Op::Gep(p, idx) => {
                    r(p);
                    idx.iter_mut().for_each(r)
                }
                Op::Switch(v, _, cases) => {
                    r(v);
                    cases.iter_mut().for_each(|(c, _)| r(c))
                }
                Op::Br(_) | Op::Ret | Op::Unreachable => {}
            }
        }
        for m in &mut self.md {
            if let MdDef::Val(_, v) = m {
                r(v);
            }
        }
    }

    fn write_inst(&self, w: &mut BitWriter, i: usize, inst_id: u64, abs: &dyn Fn(Value) -> u64, pos: &[Option<u32>]) {
        // Relative operand; forward references are only legal in phis.
        let rel = |v: Value| -> u64 {
            let a = abs(v);
            if a >= inst_id {
                self.fail(format!("forward reference to value {a} at instruction {inst_id} (blocks must be in dominance order)"));
                return 1;
            }
            inst_id - a
        };
        let bb = |b: Block| match pos.get(b.0 as usize).copied().flatten() {
            Some(p) => p as u64,
            None => {
                self.fail(format!("branch to missing block {}", b.0));
                0
            }
        };
        match &self.insts[i].op {
            Op::Bin(op, a, b, flags) => {
                let mut ops = vec![rel(*a), rel(*b), *op as u64];
                if *flags != 0 {
                    ops.push(*flags);
                }
                w.record(FUNC_CODE_INST_BINOP, &ops)
            }
            Op::Cast(op, v, t) => w.record(FUNC_CODE_INST_CAST, &[rel(*v), t.0 as u64, *op as u64]),
            Op::Cmp(p, a, b) => w.record(FUNC_CODE_INST_CMP2, &[rel(*a), rel(*b), *p as u64]),
            Op::Select(c, t, f) => w.record(FUNC_CODE_INST_VSELECT, &[rel(*t), rel(*f), rel(*c)]),
            Op::Extract(a, idx) => w.record(FUNC_CODE_INST_EXTRACTVAL, &[rel(*a), *idx as u64]),
            Op::Call(f, args) => {
                let fd = &self.funcs[*f as usize];
                let mut ops = vec![0, 1 << 15, fd.ty.0 as u64, rel(Value::Func(*f))];
                ops.extend(args.iter().map(|&a| rel(a)));
                w.record(FUNC_CODE_INST_CALL, &ops)
            }
            Op::Phi(inc) => {
                let mut ops = vec![self.insts[i].ty.0 as u64];
                // incoming edges from unreachable (unwritten) blocks are dropped
                for &(v, b) in inc.iter().filter(|(_, b)| pos.get(b.0 as usize).copied().flatten().is_some()) {
                    ops.push(signed(inst_id as i64 - abs(v) as i64));
                    ops.push(bb(b));
                }
                w.record(FUNC_CODE_INST_PHI, &ops)
            }
            Op::Load(p, align) => {
                let a = if *align == 0 { 0 } else { align.trailing_zeros() as u64 + 1 };
                w.record(FUNC_CODE_INST_LOAD, &[rel(*p), self.insts[i].ty.0 as u64, a, 0])
            }
            Op::Store(p, v, align) => {
                let a = if *align == 0 { 0 } else { align.trailing_zeros() as u64 + 1 };
                w.record(FUNC_CODE_INST_STORE, &[rel(*p), rel(*v), a, 0])
            }
            Op::Gep(p, idx) => {
                let pt = self.value_ty(*p);
                let (src, _) = self.pointee_of(pt).unwrap_or((pt, 0));
                let mut ops = vec![1, src.0 as u64, rel(*p)];
                ops.extend(idx.iter().map(|&v| rel(v)));
                w.record(FUNC_CODE_INST_GEP, &ops)
            }
            Op::AtomicRmw(op, p, v) => {
                // [ptr, val, op, vol, ordering, synchscope (crossthread = 1)]
                w.record(FUNC_CODE_INST_ATOMICRMW, &[rel(*p), rel(*v), *op as u64, 0, ORDERING_SEQCST, 1])
            }
            Op::CmpXchg(p, c, n) => {
                // [ptr, cmp, new, vol, success ordering, synchscope, failure ordering, weak]
                w.record(FUNC_CODE_INST_CMPXCHG, &[rel(*p), rel(*c), rel(*n), 0, ORDERING_SEQCST, 1, ORDERING_SEQCST, 0])
            }
            Op::Alloca(t, n, align) => {
                // [instty, opty, size (absolute value id), align record (log2 + 1 | explicit type)]
                let i32t = self.value_ty(*n);
                let a = align.trailing_zeros() as u64 + 1;
                w.record(FUNC_CODE_INST_ALLOCA, &[t.0 as u64, i32t.0 as u64, abs(*n), a | 1 << 6])
            }
            Op::Br(b) => w.record(FUNC_CODE_INST_BR, &[bb(*b)]),
            Op::CondBr(c, t, f) => w.record(FUNC_CODE_INST_BR, &[bb(*t), bb(*f), rel(*c)]),
            Op::Switch(v, d, cases) => {
                let mut ops = vec![self.value_ty(*v).0 as u64, rel(*v), bb(*d)];
                for &(c, b) in cases {
                    ops.push(abs(c));
                    ops.push(bb(b));
                }
                w.record(FUNC_CODE_INST_SWITCH, &ops)
            }
            Op::Ret => w.record(FUNC_CODE_INST_RET, &[]),
            Op::Unreachable => w.record(FUNC_CODE_INST_UNREACHABLE, &[]),
        }
    }

    /// `type_of` without interning (writer side; pointer types already exist).
    fn value_ty(&self, v: Value) -> Ty {
        match v {
            Value::Global(g) => self.globals[g as usize].ptr,
            Value::Func(f) => self.type_ids[&TypeDef::Ptr(self.funcs[f as usize].ty, 0)],
            Value::Const(c) => match &self.consts[c as usize] {
                ConstDef::Int(t, _) | ConstDef::Float(t, _) | ConstDef::Undef(t) | ConstDef::Null(t) | ConstDef::Aggregate(t, _) => *t,
            },
            Value::Inst(i) => self.insts[i as usize].ty,
        }
    }
}

// LLVMBitCodes.h (3.7)
const BLOCK_MODULE: u32 = 8;
const BLOCK_PARAMATTR: u32 = 9;
const BLOCK_PARAMATTR_GROUP: u32 = 10;
const BLOCK_CONSTANTS: u32 = 11;
const BLOCK_FUNCTION: u32 = 12;
const BLOCK_VALUE_SYMTAB: u32 = 14;
const BLOCK_METADATA: u32 = 15;
const BLOCK_TYPE: u32 = 17;

const MODULE_CODE_VERSION: u32 = 1;
const MODULE_CODE_TRIPLE: u32 = 2;
const MODULE_CODE_DATALAYOUT: u32 = 3;
const MODULE_CODE_GLOBALVAR: u32 = 7;
const MODULE_CODE_FUNCTION: u32 = 8;

const PARAMATTR_CODE_ENTRY: u32 = 2;
const PARAMATTR_GRP_CODE_ENTRY: u32 = 3;
const ATTR_NODUPLICATE: u64 = 12;
const ATTR_NOUNWIND: u64 = 18;
const ATTR_READNONE: u64 = 20;
const ATTR_READONLY: u64 = 21;

/// `bitc::ORDERING_SEQCST`: the bitcode code, not the in-memory
/// `AtomicOrdering::SequentiallyConsistent` (7), which LLVM's reader only
/// accepts because it maps unknown codes to seq_cst.
const ORDERING_SEQCST: u64 = 6;

const LINKAGE_EXTERNAL: u64 = 0;
const LINKAGE_INTERNAL: u64 = 3;

const TYPE_CODE_NUMENTRY: u32 = 1;
const TYPE_CODE_VOID: u32 = 2;
const TYPE_CODE_FLOAT: u32 = 3;
const TYPE_CODE_DOUBLE: u32 = 4;
const TYPE_CODE_LABEL: u32 = 5;
const TYPE_CODE_INTEGER: u32 = 7;
const TYPE_CODE_POINTER: u32 = 8;
const TYPE_CODE_HALF: u32 = 10;
const TYPE_CODE_ARRAY: u32 = 11;
const TYPE_CODE_METADATA: u32 = 16;
const TYPE_CODE_STRUCT_ANON: u32 = 18;
const TYPE_CODE_STRUCT_NAME: u32 = 19;
const TYPE_CODE_STRUCT_NAMED: u32 = 20;
const TYPE_CODE_FUNCTION: u32 = 21;

const CST_CODE_SETTYPE: u32 = 1;
const CST_CODE_NULL: u32 = 2;
const CST_CODE_UNDEF: u32 = 3;
const CST_CODE_INTEGER: u32 = 4;
const CST_CODE_FLOAT: u32 = 6;
const CST_CODE_AGGREGATE: u32 = 7;

const METADATA_STRING: u32 = 1;
const METADATA_VALUE: u32 = 2;
const METADATA_NODE: u32 = 3;
const METADATA_NAME: u32 = 4;
const METADATA_NAMED_NODE: u32 = 10;

const VST_CODE_ENTRY: u32 = 1;

const FUNC_CODE_DECLAREBLOCKS: u32 = 1;
const FUNC_CODE_INST_BINOP: u32 = 2;
const FUNC_CODE_INST_CAST: u32 = 3;
const FUNC_CODE_INST_RET: u32 = 10;
const FUNC_CODE_INST_BR: u32 = 11;
const FUNC_CODE_INST_SWITCH: u32 = 12;
const FUNC_CODE_INST_UNREACHABLE: u32 = 15;
const FUNC_CODE_INST_PHI: u32 = 16;
const FUNC_CODE_INST_ALLOCA: u32 = 19;
const FUNC_CODE_INST_LOAD: u32 = 20;
const FUNC_CODE_INST_EXTRACTVAL: u32 = 26;
const FUNC_CODE_INST_CMP2: u32 = 28;
const FUNC_CODE_INST_VSELECT: u32 = 29;
const FUNC_CODE_INST_CALL: u32 = 34;
const FUNC_CODE_INST_ATOMICRMW: u32 = 38;
const FUNC_CODE_INST_GEP: u32 = 43;
const FUNC_CODE_INST_STORE: u32 = 44;
const FUNC_CODE_INST_CMPXCHG: u32 = 46;

// ------------------------------------------------------- introspection

/// What an instruction is, for analyses over the built function (e.g. the
/// signature dependency tables PSV0 carries).
#[derive(Debug, Clone)]
pub enum InstInfo<'a> {
    /// A call: callee name and arguments.
    Call(&'a str, &'a [Value]),
    Phi(&'a [(Value, Block)]),
    Load(Value),
    Store(Value, Value),
    /// Any other value-producing instruction with its value operands.
    Other(Vec<Value>),
    /// A terminator: condition operand (if any) and successors.
    Term(Option<Value>, Vec<Block>),
}

impl Module {
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
    pub fn block_insts(&self, b: Block) -> &[u32] {
        &self.blocks[b.0 as usize]
    }
    pub fn inst_info(&self, i: u32) -> InstInfo<'_> {
        match &self.insts[i as usize].op {
            Op::Call(f, args) => InstInfo::Call(&self.funcs[*f as usize].name, args),
            Op::Phi(list) => InstInfo::Phi(list),
            Op::Load(p, _) => InstInfo::Load(*p),
            Op::Store(p, v, _) => InstInfo::Store(*p, *v),
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) => InstInfo::Other(vec![*a, *b]),
            Op::Cast(_, v, _) | Op::Extract(v, _) => InstInfo::Other(vec![*v]),
            Op::Select(c, t, f) => InstInfo::Other(vec![*c, *t, *f]),
            Op::Gep(p, idx) => InstInfo::Other(std::iter::once(*p).chain(idx.iter().copied()).collect()),
            Op::AtomicRmw(_, p, v) => InstInfo::Other(vec![*p, *v]),
            Op::CmpXchg(p, c, n) => InstInfo::Other(vec![*p, *c, *n]),
            Op::Alloca(..) => InstInfo::Other(Vec::new()),
            Op::Br(b) => InstInfo::Term(None, vec![*b]),
            Op::CondBr(c, t, f) => InstInfo::Term(Some(*c), vec![*t, *f]),
            Op::Switch(v, d, cases) => InstInfo::Term(Some(*v), std::iter::once(*d).chain(cases.iter().map(|c| c.1)).collect()),
            Op::Ret | Op::Unreachable => InstInfo::Term(None, Vec::new()),
        }
    }
    /// Whether an instruction of the entry function produces or consumes a
    /// value of type `t` (the shader-flag rules key off instruction types).
    /// The 64-bit integer and double types, if already interned (lookup
    /// without adding them to the type table).
    pub fn existing_i64_f64(&self) -> (Option<Ty>, Option<Ty>) {
        (self.type_ids.get(&TypeDef::Int(64)).copied(), self.type_ids.get(&TypeDef::Double).copied())
    }
    /// Whether the entry uses a D3D11.1 double extension through plain
    /// instructions: double `fdiv`, or conversions between double and
    /// integers (DXC's shader-flag rules; `Fma` is flagged by its helper).
    pub fn uses_double_extensions(&self) -> bool {
        let Some(f64) = self.type_ids.get(&TypeDef::Double).copied() else { return false };
        self.insts.iter().any(|d| match &d.op {
            Op::Bin(BinOp::UDiv | BinOp::SDiv, ..) => d.ty == f64,
            Op::Cast(CastOp::FPToUI | CastOp::FPToSI, v, _) => self.value_ty(*v) == f64,
            Op::Cast(CastOp::UIToFP | CastOp::SIToFP, _, t) => *t == f64,
            _ => false,
        })
    }
    pub fn insts_use_type(&self, t: Ty) -> bool {
        self.blocks.iter().flatten().any(|&i| {
            let d = &self.insts[i as usize];
            if d.ty == t {
                return true;
            }
            let ops: Vec<Value> = match self.inst_info(i) {
                InstInfo::Call(_, a) => a.to_vec(),
                InstInfo::Phi(l) => l.iter().map(|p| p.0).collect(),
                InstInfo::Load(p) => vec![p],
                InstInfo::Store(p, v) => vec![p, v],
                InstInfo::Other(v) => v,
                InstInfo::Term(c, _) => c.into_iter().collect(),
            };
            ops.into_iter().any(|v| self.value_ty(v) == t)
        })
    }
    /// The integer value of a constant (zero-extended from its width).
    pub fn const_u64(&self, v: Value) -> Option<u64> {
        match v {
            Value::Const(c) => match &self.consts[c as usize] {
                ConstDef::Int(t, x) => {
                    let w = self.int_width(*t)?;
                    Some(if w >= 64 { *x as u64 } else { (*x as u64) & ((1u64 << w) - 1) })
                }
                _ => None,
            },
            _ => None,
        }
    }
}

impl Module {
    /// `getelementptr inbounds` through arrays and structs (struct indices
    /// must be constants).
    pub fn gep_fields(&mut self, ptr: Value, idx: &[Value]) -> Value {
        let pt = self.type_of(ptr);
        let (mut t, space) = self.pointee(pt);
        for &i in &idx[1..] {
            let next = match &self.types[t.0 as usize] {
                TypeDef::Array(_, e) => Ok(*e),
                TypeDef::Struct(_, f) => self
                    .const_u64(i)
                    .and_then(|k| f.get(k as usize).copied())
                    .ok_or_else(|| format!("gep: struct field {i:?} of {:?}", self.types[t.0 as usize])),
                other => Err(format!("gep into {other:?}")),
            };
            match next {
                Ok(n) => t = n,
                Err(e) => {
                    self.fail(e);
                    break;
                }
            }
        }
        let rt = self.ptr(t, space);
        self.push(Op::Gep(ptr, idx.to_vec()), rt)
    }

    /// Allocation size of a type under the DXIL data layout (scalars below
    /// 32 bits are 32-bit aligned).
    pub fn alloc_size(&self, t: Ty) -> u64 {
        self.size_align(t, false).0
    }
    /// Allocation size under the native 16-bit data layout (`i8:8`,
    /// `i16:16`, `f16:16`) when `native16`, else the 32-bit-aligned one.
    pub fn alloc_size_layout(&self, t: Ty, native16: bool) -> u64 {
        self.size_align(t, native16).0
    }
    fn size_align(&self, t: Ty, native16: bool) -> (u64, u64) {
        match &self.types[t.0 as usize] {
            TypeDef::Int(w) if *w > 32 => (8, 8),
            TypeDef::Int(8) if native16 => (1, 1),
            TypeDef::Int(16) | TypeDef::Half if native16 => (2, 2),
            TypeDef::Double => (8, 8),
            TypeDef::Int(_) | TypeDef::Half | TypeDef::Float | TypeDef::Ptr(..) => (4, 4),
            TypeDef::Array(n, e) => {
                let (s, a) = self.size_align(*e, native16);
                (n * s, a)
            }
            TypeDef::Struct(_, f) => {
                let (mut off, mut al) = (0u64, 1u64);
                for &x in f {
                    let (s, a) = self.size_align(x, native16);
                    off = off.div_ceil(a) * a + s;
                    al = al.max(a);
                }
                (off.div_ceil(al) * al, al)
            }
            _ => (0, 1),
        }
    }
    /// Total allocation size of the global variables in `addrspace`
    /// (3: thread-group shared memory).
    pub fn global_bytes(&self, addrspace: u32) -> u64 {
        self.global_bytes_layout(addrspace, false)
    }
    /// [`Module::global_bytes`] under the native 16-bit layout when `native16`.
    pub fn global_bytes_layout(&self, addrspace: u32, native16: bool) -> u64 {
        self.globals.iter().filter(|g| g.addrspace == addrspace).map(|g| self.alloc_size_layout(g.ty, native16)).sum()
    }
}
