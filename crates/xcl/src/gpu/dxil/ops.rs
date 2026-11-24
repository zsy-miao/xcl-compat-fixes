//! Typed helpers for the DXIL operation (`dx.op.*`) library on top of
//! [`Shader::op`]: math, memory (group-shared, typed / structured / raw
//! buffers, atomics, barriers), textures and samplers, wave and quad
//! operations, derivatives.
//!
//! Opcode numbers, overload classes, argument layouts and function attributes
//! follow DXC's `docs/DXIL.rst` and DXC 1.9's output for small HLSL programs
//! (NCSA licence); each helper is covered by a validator test below, most by
//! a GPU execution test against a CPU reference.
//!
//! # Exactness
//!
//! None of the helpers add fast-math flags. Per the D3D functional spec and
//! `DXIL.rst`, these DXIL operations are **exact** (bit-exact up to the
//! denorm mode — 32-bit float denormals may be flushed unless the shader
//! runs with preserved denormals): `FAbs`, `Saturate`, `FMax`/`FMin`
//! (IEEE-754-2008 `maxNum`/`minNum`: a NaN operand yields the other operand),
//! `Round_*`, `Frc` (`x - floor(x)`), all integer ops (`IMax`…`Bfi`,
//! `IMul`/`UMul`/`UDiv`, bit counts), `IsNaN`/`IsInf`/`IsFinite`,
//! `MakeDouble`/`SplitDouble`, `Fma` (fused, f64), and double `FMax`/`FMin`.
//! `LegacyF32ToF16`'s rounding is implementation-defined: the RTX 4090
//! rounds towards zero (others round to nearest even); `LegacyF16ToF32` is
//! exact.
//!
//! **Approximate** (implementation-defined within D3D tolerances):
//! `Cos`/`Sin` (absolute error ≤ 0.0008 on [-100π, 100π]), `Exp`/`Log`
//! (base 2, ≈ 3 ULP / absolute 2⁻²¹ near 1), `Sqrt` (≈ 1 ULP… drivers may
//! be less precise than correctly rounded), `Rsqrt` (≈ 2 ULP), the
//! hyperbolic / inverse trig ops, and float division (2.5 ULP).
//! `FMad` may or may not be fused: do not rely on either rounding.

use super::module::{BinOp, CastOp, FnAttrs, RmwOp, Ty, Value};
use super::shader::{kind, Feature, RawUav, ResClass, Shader};

// ------------------------------------------------------------------ enums

/// `dx.op.unary` floating-point operations (overloads f16 / f32; f64 only
/// for those in [`DOUBLE_UNARY`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FUnary {
    FAbs = 6,
    Saturate = 7,
    Cos = 12,
    Sin = 13,
    Tan = 14,
    Acos = 15,
    Asin = 16,
    Atan = 17,
    Hcos = 18,
    Hsin = 19,
    Htan = 20,
    /// 2^x
    Exp = 21,
    /// `x - floor(x)`
    Frc = 22,
    /// log2(x)
    Log = 23,
    Sqrt = 24,
    Rsqrt = 25,
    /// Round to nearest even.
    RoundNe = 26,
    /// Round towards -inf (floor).
    RoundNi = 27,
    /// Round towards +inf (ceil).
    RoundPi = 28,
    /// Round towards zero (trunc).
    RoundZ = 29,
}

/// `dx.op.isSpecialFloat` tests (result i1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FClass {
    IsNaN = 8,
    IsInf = 9,
    IsFinite = 10,
    IsNormal = 11,
}

/// Bit-counting operations (`unaryBits`, result i32 for every overload).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitCount {
    /// Population count.
    Countbits = 31,
    /// Index of the lowest set bit, or -1.
    FirstbitLo = 32,
    /// Index of the highest set bit **counted from the MSB**, or -1
    /// (HLSL's `firstbithigh` is `31 - this` for non-zero inputs).
    FirstbitHi = 33,
    /// Like `FirstbitHi` on the first bit differing from the sign bit.
    FirstbitSHi = 34,
}

/// `dx.op.binary` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binary {
    FMax = 35,
    FMin = 36,
    IMax = 37,
    IMin = 38,
    UMax = 39,
    UMin = 40,
}

/// `dx.op.tertiary` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tertiary {
    /// a*b+c, fusion unspecified (f16/f32/f64).
    FMad = 46,
    /// Fused a*b+c (f64 only; needs `DoubleExtensions`).
    Fma = 47,
    IMad = 48,
    UMad = 49,
    /// `Ibfe(width, offset, value)` (i32): signed bitfield extract; width and
    /// offset use their low 5 bits.
    Ibfe = 51,
    /// `Ubfe(width, offset, value)` (i32).
    Ubfe = 52,
}

/// Atomic operations of `AtomicBinOp` (op 78).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add = 0,
    And = 1,
    Or = 2,
    Xor = 3,
    IMin = 4,
    IMax = 5,
    UMin = 6,
    UMax = 7,
    Exchange = 8,
}

/// `Barrier` mode flags (op 80), combined with `|`.
pub mod barrier {
    /// Execution barrier: wait for every thread of the group.
    pub const SYNC_THREAD_GROUP: u32 = 1;
    /// Device-memory (global UAV) fence.
    pub const UAV_FENCE_GLOBAL: u32 = 2;
    /// Group-scoped UAV fence.
    pub const UAV_FENCE_THREAD_GROUP: u32 = 4;
    /// Group-shared memory fence.
    pub const TGSM_FENCE: u32 = 8;
    /// `GroupMemoryBarrierWithGroupSync()`
    pub const GROUP_SYNC: u32 = SYNC_THREAD_GROUP | TGSM_FENCE;
    /// `AllMemoryBarrierWithGroupSync()`
    pub const ALL_SYNC: u32 = SYNC_THREAD_GROUP | TGSM_FENCE | UAV_FENCE_GLOBAL;
}

/// `DXIL::ComponentType` of typed resource elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompType {
    I32 = 4,
    U32 = 5,
    I64 = 6,
    U64 = 7,
    F16 = 8,
    F32 = 9,
    F64 = 10,
    SNormF32 = 13,
    UNormF32 = 14,
}

impl CompType {
    fn is_float(self) -> bool {
        matches!(self, CompType::F16 | CompType::F32 | CompType::F64 | CompType::SNormF32 | CompType::UNormF32)
    }
}

/// Texture shapes (`DXIL::ResourceKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexKind {
    D1 = 1,
    D2 = 2,
    D3 = 4,
    Cube = 5,
    D1Array = 6,
    D2Array = 7,
    CubeArray = 9,
}

impl TexKind {
    /// Coordinates used to address a texel / sample (array slice included).
    pub fn coords(self) -> usize {
        match self {
            TexKind::D1 => 1,
            TexKind::D2 | TexKind::D1Array => 2,
            TexKind::D3 | TexKind::D2Array | TexKind::Cube => 3,
            TexKind::CubeArray => 4,
        }
    }
    /// Texel-offset components (none for cubes).
    pub fn offsets(self) -> usize {
        match self {
            TexKind::D1 | TexKind::D1Array => 1,
            TexKind::D2 | TexKind::D2Array => 2,
            TexKind::D3 => 3,
            TexKind::Cube | TexKind::CubeArray => 0,
        }
    }
    /// Gradient components for `SampleGrad` (spatial dimensions).
    pub fn grads(self) -> usize {
        match self {
            TexKind::D1 | TexKind::D1Array => 1,
            TexKind::D2 | TexKind::D2Array => 2,
            TexKind::D3 | TexKind::Cube | TexKind::CubeArray => 3,
        }
    }
}

/// A texture handle (heap SRV or UAV) with its element type.
#[derive(Debug, Clone, Copy)]
pub struct Texture {
    pub handle: Value,
    pub kind: TexKind,
    pub uav: bool,
    pub comp: CompType,
}

/// A sampler handle.
#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    pub handle: Value,
    pub comparison: bool,
}

/// A constant buffer handle.
#[derive(Debug, Clone, Copy)]
pub struct Cbv {
    pub handle: Value,
}

/// A typed buffer (`Buffer<T>` / `RWBuffer<T>`).
#[derive(Debug, Clone, Copy)]
pub struct TypedBuffer {
    pub handle: Value,
    pub comp: CompType,
    binding: Option<usize>,
}

/// Wave reduction / prefix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveOp {
    Sum = 0,
    Product = 1,
    Min = 2,
    Max = 3,
}

/// Wave bitwise reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveBit {
    And = 0,
    Or = 1,
    Xor = 2,
}

/// `QuadOp` directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadDir {
    ReadAcrossX = 0,
    ReadAcrossY = 1,
    ReadAcrossDiagonal = 2,
}

/// Derivative operations (`dx.op.unary`, f16/f32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deriv {
    CoarseX = 83,
    CoarseY = 84,
    FineX = 85,
    FineY = 86,
}

/// Typed-buffer / texture resource properties word 1: component type and
/// count (`comp | ncomp << 8`).
fn typed_props(comp: CompType, ncomp: u32) -> u32 {
    comp as u32 | (ncomp << 8)
}

// ------------------------------------------------------------------- math

impl Shader {
    fn is_i64(&mut self, v: Value) -> bool {
        let t = self.m.type_of(v);
        t == self.m.i64()
    }
    fn is_f64(&mut self, v: Value) -> bool {
        let t = self.m.type_of(v);
        t == self.m.f64()
    }

    /// A float unary op on `v` (f16 / f32; f64 only where DXIL allows).
    pub fn funary(&mut self, op: FUnary, v: Value) -> Value {
        let t = self.m.type_of(v);
        if self.is_f64(v) {
            self.require(Feature::Doubles);
        }
        self.op(op as u32, "unary", Some(t), t, &[v], FnAttrs::ReadNone)
    }

    /// `IsNaN` / `IsInf` / `IsFinite` / `IsNormal` → i1 (f32; the f16
    /// overload needs SM 6.9).
    pub fn fclass(&mut self, op: FClass, v: Value) -> Value {
        let t = self.m.type_of(v);
        let i1 = self.m.i1();
        self.op(op as u32, "isSpecialFloat", Some(t), i1, &[v], FnAttrs::ReadNone)
    }

    /// `Bfrev` (bit reverse; i16 / i32 / i64).
    pub fn bfrev(&mut self, v: Value) -> Value {
        let t = self.m.type_of(v);
        if self.is_i64(v) {
            // AMD's driver (amdxc64, RDNA3/4) gets `Bfrev.i64` wrong (it
            // writes the reversed low half into both halves), so reverse
            // each 32-bit half and swap them.
            self.require(Feature::Int64Ops);
            let i32 = self.m.i32();
            let lo = self.m.cast(CastOp::Trunc, v, i32);
            let c32 = self.m.const_int(t, 32);
            let hi = self.m.bin(BinOp::LShr, v, c32);
            let hi = self.m.cast(CastOp::Trunc, hi, i32);
            let rlo = self.op(30, "unary", Some(i32), i32, &[lo], FnAttrs::ReadNone);
            let rhi = self.op(30, "unary", Some(i32), i32, &[hi], FnAttrs::ReadNone);
            let rlo = self.m.cast(CastOp::ZExt, rlo, t);
            let rlo = self.m.bin(BinOp::Shl, rlo, c32);
            let rhi = self.m.cast(CastOp::ZExt, rhi, t);
            return self.m.bin(BinOp::Or, rlo, rhi);
        }
        self.op(30, "unary", Some(t), t, &[v], FnAttrs::ReadNone)
    }

    /// Bit counts (i16 / i32 / i64 input) → i32.
    pub fn bit_count(&mut self, op: BitCount, v: Value) -> Value {
        let t = self.m.type_of(v);
        let i32 = self.m.i32();
        if self.is_i64(v) {
            self.require(Feature::Int64Ops);
            if matches!(op, BitCount::Countbits) {
                // Count each 32-bit half: AMD's driver (amdxc64, RDNA4) gets
                // i64 bit ops wrong (see `bfrev`); fuzz seed 1599 reads a wrong
                // value downstream of `Countbits.i64(i64 2)` there only.
                let lo = self.m.cast(CastOp::Trunc, v, i32);
                let c32 = self.m.const_int(t, 32);
                let hi = self.m.bin(BinOp::LShr, v, c32);
                let hi = self.m.cast(CastOp::Trunc, hi, i32);
                let a = self.op(op as u32, "unaryBits", Some(i32), i32, &[lo], FnAttrs::ReadNone);
                let b = self.op(op as u32, "unaryBits", Some(i32), i32, &[hi], FnAttrs::ReadNone);
                return self.m.bin(BinOp::Add, a, b);
            }
        }
        self.op(op as u32, "unaryBits", Some(t), i32, &[v], FnAttrs::ReadNone)
    }

    /// `FMax`…`UMin` on two values of the same type.
    pub fn binary(&mut self, op: Binary, a: Value, b: Value) -> Value {
        let t = self.m.type_of(a);
        if self.is_f64(a) {
            self.require(Feature::Doubles);
        }
        if self.is_i64(a) {
            self.require(Feature::Int64Ops);
        }
        self.op(op as u32, "binary", Some(t), t, &[a, b], FnAttrs::ReadNone)
    }

    /// `FMad`, `Fma`, `IMad`, `UMad`, `Ibfe`, `Ubfe`.
    pub fn tertiary(&mut self, op: Tertiary, a: Value, b: Value, c: Value) -> Value {
        let t = self.m.type_of(a);
        if self.is_f64(a) {
            self.require(Feature::Doubles);
        }
        if op == Tertiary::Fma {
            self.require(Feature::DoubleExtensions);
        }
        if self.is_i64(a) {
            self.require(Feature::Int64Ops);
        }
        self.op(op as u32, "tertiary", Some(t), t, &[a, b, c], FnAttrs::ReadNone)
    }

    /// `Bfi(width, offset, value, replaced)` (i32): inserts the low `width`
    /// bits of `value` into `replaced` at `offset`.
    pub fn bfi(&mut self, width: Value, offset: Value, value: Value, replaced: Value) -> Value {
        let t = self.m.type_of(value);
        self.op(53, "quaternary", Some(t), t, &[width, offset, value, replaced], FnAttrs::ReadNone)
    }

    fn two_outs(&mut self, op: u32, a: Value, b: Value) -> (Value, Value) {
        let i32 = self.m.i32();
        let st = self.m.named_struct("dx.types.twoi32", &[i32, i32]);
        let r = self.op(op, "binaryWithTwoOuts", Some(i32), st, &[a, b], FnAttrs::ReadNone);
        (self.m.extract(r, 0), self.m.extract(r, 1))
    }
    /// Full 64-bit product of two signed i32 → `(hi, lo)`.
    pub fn imul_wide(&mut self, a: Value, b: Value) -> (Value, Value) {
        self.two_outs(41, a, b)
    }
    /// Full 64-bit product of two unsigned i32 → `(hi, lo)`.
    pub fn umul_wide(&mut self, a: Value, b: Value) -> (Value, Value) {
        self.two_outs(42, a, b)
    }
    /// Unsigned divide → `(quotient, remainder)` (division by zero gives all ones).
    pub fn udiv_rem(&mut self, a: Value, b: Value) -> (Value, Value) {
        self.two_outs(43, a, b)
    }
    fn carry(&mut self, op: u32, a: Value, b: Value) -> (Value, Value) {
        let i32 = self.m.i32();
        let i1 = self.m.i1();
        let st = self.m.named_struct("dx.types.i32c", &[i32, i1]);
        let r = self.op(op, "binaryWithCarryOrBorrow", Some(i32), st, &[a, b], FnAttrs::ReadNone);
        (self.m.extract(r, 0), self.m.extract(r, 1))
    }
    /// `UAddc`: `(a + b, carry)`.
    pub fn uaddc(&mut self, a: Value, b: Value) -> (Value, Value) {
        self.carry(44, a, b)
    }
    /// `USubb`: `(a - b, borrow)`.
    pub fn usubb(&mut self, a: Value, b: Value) -> (Value, Value) {
        self.carry(45, a, b)
    }

    /// `MakeDouble(lo, hi)`.
    pub fn make_double(&mut self, lo: Value, hi: Value) -> Value {
        self.require(Feature::Doubles);
        let f64 = self.m.f64();
        self.op(101, "makeDouble", Some(f64), f64, &[lo, hi], FnAttrs::ReadNone)
    }
    /// `SplitDouble(d)` → `(lo, hi)`.
    pub fn split_double(&mut self, d: Value) -> (Value, Value) {
        self.require(Feature::Doubles);
        let (f64, i32) = (self.m.f64(), self.m.i32());
        let st = self.m.named_struct("dx.types.splitdouble", &[i32, i32]);
        let r = self.op(102, "splitDouble", Some(f64), st, &[d], FnAttrs::ReadNone);
        (self.m.extract(r, 0), self.m.extract(r, 1))
    }
    /// `f32tof16`: half bits in the low 16 bits of an i32.
    pub fn f32_to_f16(&mut self, f: Value) -> Value {
        let i32 = self.m.i32();
        self.op(130, "legacyF32ToF16", None, i32, &[f], FnAttrs::ReadNone)
    }
    /// `f16tof32` of the low 16 bits of an i32.
    pub fn f16_to_f32(&mut self, h: Value) -> Value {
        let f32 = self.m.f32();
        self.op(131, "legacyF16ToF32", None, f32, &[h], FnAttrs::ReadNone)
    }
}

// ----------------------------------------------------------------- memory

/// A group-shared array (`groupshared T name[n]`, address space 3).
#[derive(Debug, Clone, Copy)]
pub struct GroupShared {
    pub global: Value,
    pub elem: Ty,
}

impl Shader {
    /// Declares `groupshared <elem> name[n]`.
    pub fn groupshared(&mut self, name: &str, elem: Ty, n: u64) -> GroupShared {
        let arr = self.m.array(n, elem);
        let align = if elem == self.m.i64() || elem == self.m.f64() { 8 } else { 4 };
        let global = self.m.global(name, arr, 3, None, false, align);
        GroupShared { global, elem }
    }
    /// Pointer to element `idx` (i32).
    pub fn gs_ptr(&mut self, g: GroupShared, idx: Value) -> Value {
        let zero = self.m.i32c(0);
        self.m.gep(g.global, &[zero, idx])
    }
    fn elem_align(&mut self, t: Ty) -> u32 {
        if t == self.m.i64() || t == self.m.f64() {
            8
        } else {
            4
        }
    }
    pub fn gs_load(&mut self, g: GroupShared, idx: Value) -> Value {
        let p = self.gs_ptr(g, idx);
        let a = self.elem_align(g.elem);
        self.m.load(p, a)
    }
    pub fn gs_store(&mut self, g: GroupShared, idx: Value, v: Value) {
        let p = self.gs_ptr(g, idx);
        let a = self.elem_align(g.elem);
        self.m.store(p, v, a);
    }
    /// `atomicrmw` on a group-shared element; returns the old value.
    pub fn gs_atomic(&mut self, op: RmwOp, g: GroupShared, idx: Value, v: Value) -> Value {
        if self.is_i64(v) {
            self.require(Feature::Int64Ops);
            self.require(Feature::AtomicInt64GroupShared);
        }
        let p = self.gs_ptr(g, idx);
        self.m.atomic_rmw(op, p, v)
    }
    /// `cmpxchg` on a group-shared element; returns the old value.
    pub fn gs_cmpxchg(&mut self, g: GroupShared, idx: Value, cmp: Value, new: Value) -> Value {
        if self.is_i64(cmp) {
            self.require(Feature::Int64Ops);
            self.require(Feature::AtomicInt64GroupShared);
        }
        let p = self.gs_ptr(g, idx);
        self.m.cmpxchg(p, cmp, new)
    }

    /// `Barrier(mode)` — see [`barrier`] for the flags.
    pub fn barrier(&mut self, mode: u32) {
        let void = self.m.void();
        let mk = self.m.i32c(mode);
        self.op(80, "barrier", None, void, &[mk], FnAttrs::NoDuplicate);
    }

    // ---- typed and structured buffers

    /// Declares `RWBuffer<comp>` at `u<lower>, space<space>`.
    pub fn typed_uav(&mut self, lower: u32, space: u32, comp: CompType) -> TypedBuffer {
        let id = self.bindings.len();
        self.bindings.push(super::shader::Binding { comp_type: comp as u32, ..super::shader::Binding::new(ResClass::Uav, space, lower, kind::TYPED_BUFFER) });
        let h = self.handle_from_binding(ResClass::Uav, lower, lower, space, lower);
        let handle = self.annotate(h, kind::TYPED_BUFFER | 0x1000, typed_props(comp, 1));
        TypedBuffer { handle, comp, binding: Some(id) }
    }
    /// `Buffer<comp>` / `RWBuffer<comp>` from the descriptor heap.
    pub fn heap_typed_buffer(&mut self, index: Value, uav: bool, comp: CompType, nonuniform: bool) -> TypedBuffer {
        let h = self.handle_from_heap(index, false, nonuniform);
        let handle = self.annotate(h, kind::TYPED_BUFFER | if uav { 0x1000 } else { 0 }, typed_props(comp, 1));
        TypedBuffer { handle, comp, binding: None }
    }
    fn comp_ty(&mut self, c: CompType) -> Ty {
        match c {
            CompType::F16 => self.m.f16(),
            CompType::F64 => self.m.f64(),
            c if c.is_float() => self.m.f32(),
            CompType::I64 | CompType::U64 => self.m.i64(),
            _ => self.m.i32(),
        }
    }
    /// `BufferLoad` (op 68) of element `index` → four components.
    pub fn buffer_load(&mut self, buf: TypedBuffer, index: Value) -> [Value; 4] {
        let t = self.comp_ty(buf.comp);
        let rt = self.res_ret(t);
        let i32 = self.m.i32();
        let undef = self.m.undef(i32);
        let r = self.op(68, "bufferLoad", Some(t), rt, &[buf.handle, index, undef], FnAttrs::ReadOnly);
        [0, 1, 2, 3].map(|c| self.m.extract(r, c))
    }
    /// `BufferStore` (op 69) of up to four components (DXIL requires the
    /// full mask the format has; pass all components of the element).
    pub fn buffer_store(&mut self, buf: TypedBuffer, index: Value, vals: &[Value]) {
        assert!((1..=4).contains(&vals.len()));
        let t = self.m.type_of(vals[0]);
        let void = self.m.void();
        let i32 = self.m.i32();
        let undef = self.m.undef(i32);
        let mut v = [vals[0]; 4];
        v[..vals.len()].copy_from_slice(vals);
        let mk = self.m.i8c(0xf);
        self.op(69, "bufferStore", Some(t), void, &[buf.handle, index, undef, v[0], v[1], v[2], v[3], mk], FnAttrs::NoUnwind);
    }

    /// Declares `Buffer<comp[ncomp]>` at `t<lower>, space<space>`.
    pub fn srv_typed(&mut self, lower: u32, space: u32, comp: CompType, ncomp: u32) -> TypedBuffer {
        let id = self.bindings.len();
        self.bindings.push(super::shader::Binding { comp_type: comp as u32, ..super::shader::Binding::new(ResClass::Srv, space, lower, kind::TYPED_BUFFER) });
        let h = self.handle_from_binding(ResClass::Srv, lower, lower, space, lower);
        let handle = self.annotate(h, kind::TYPED_BUFFER, typed_props(comp, ncomp));
        TypedBuffer { handle, comp, binding: Some(id) }
    }
    /// Declares `ByteAddressBuffer` at `t<lower>, space<space>`.
    pub fn srv_raw(&mut self, lower: u32, space: u32) -> RawUav {
        self.require(Feature::RawAndStructuredBuffers);
        let id = self.bindings.len();
        self.bindings.push(super::shader::Binding::new(ResClass::Srv, space, lower, kind::RAW_BUFFER));
        let h = self.handle_from_binding(ResClass::Srv, lower, lower, space, lower);
        let handle = self.annotate(h, kind::RAW_BUFFER, 0);
        RawUav { handle, binding: Some(id) }
    }
    /// Declares `StructuredBuffer` with `stride` bytes at `t<lower>, space<space>`.
    pub fn srv_structured(&mut self, lower: u32, space: u32, stride: u32) -> RawUav {
        self.require(Feature::RawAndStructuredBuffers);
        let id = self.bindings.len();
        self.bindings.push(super::shader::Binding { stride, ..super::shader::Binding::new(ResClass::Srv, space, lower, kind::STRUCTURED_BUFFER) });
        let h = self.handle_from_binding(ResClass::Srv, lower, lower, space, lower);
        let handle = self.annotate(h, kind::STRUCTURED_BUFFER, stride);
        RawUav { handle, binding: Some(id) }
    }
    /// Declares a texture SRV (`Texture2D<float4>` = `(D2, F32, 4)`) at `t<lower>, space<space>`.
    pub fn srv_texture(&mut self, lower: u32, space: u32, kind: TexKind, comp: CompType, ncomp: u32) -> Texture {
        self.bindings.push(super::shader::Binding { comp_type: comp as u32, ..super::shader::Binding::new(ResClass::Srv, space, lower, kind as u32) });
        let h = self.handle_from_binding(ResClass::Srv, lower, lower, space, lower);
        let handle = self.annotate(h, kind as u32, typed_props(comp, ncomp));
        Texture { handle, kind, uav: false, comp }
    }
    /// Declares a cbuffer of `size` bytes at `b<lower>, space<space>` (a
    /// root-constant block is the same: a small cbuffer).
    pub fn cbv(&mut self, lower: u32, space: u32, size: u32) -> Cbv {
        self.bindings.push(super::shader::Binding { size, ..super::shader::Binding::new(ResClass::Cbv, space, lower, kind::CBUFFER) });
        let h = self.handle_from_binding(ResClass::Cbv, lower, lower, space, lower);
        let handle = self.annotate(h, kind::CBUFFER, size);
        Cbv { handle }
    }
    /// A cbuffer from `ResourceDescriptorHeap[index]` of `size` bytes.
    pub fn heap_cbv(&mut self, index: Value, size: u32, nonuniform: bool) -> Cbv {
        let h = self.handle_from_heap(index, false, nonuniform);
        let handle = self.annotate(h, kind::CBUFFER, size);
        Cbv { handle }
    }
    /// `CBufferLoadLegacy` (op 59): the 16-byte row `row` as four `ty`
    /// values (i32/f32; i64/f64 rows have two components, i16/f16 eight —
    /// only the 32-bit forms are exposed here).
    pub fn cbuffer_load_legacy(&mut self, cb: Cbv, row: Value, ty: Ty) -> [Value; 4] {
        let name = format!("dx.types.CBufRet.{}", self.type_suffix(ty));
        let rt = self.m.named_struct(&name, &[ty, ty, ty, ty]);
        let r = self.op(59, "cbufferLoadLegacy", Some(ty), rt, &[cb.handle, row], FnAttrs::ReadOnly);
        [0, 1, 2, 3].map(|c| self.m.extract(r, c))
    }
    /// Declares a sampler (`SamplerComparisonState` if `comparison`) at `s<lower>, space<space>`.
    pub fn sampler(&mut self, lower: u32, space: u32, comparison: bool) -> Sampler {
        self.bindings.push(super::shader::Binding { comparison, ..super::shader::Binding::new(ResClass::Sampler, space, lower, kind::SAMPLER) });
        let h = self.handle_from_binding(ResClass::Sampler, lower, lower, space, lower);
        let handle = self.annotate(h, kind::SAMPLER | if comparison { 0x8000 } else { 0 }, 0);
        Sampler { handle, comparison }
    }

    /// Declares `RWStructuredBuffer` with `stride` bytes at `u<lower>, space<space>`.
    pub fn structured_uav(&mut self, lower: u32, space: u32, stride: u32) -> RawUav {
        self.require(Feature::RawAndStructuredBuffers);
        let id = self.bindings.len();
        self.bindings.push(super::shader::Binding { stride, ..super::shader::Binding::new(ResClass::Uav, space, lower, kind::STRUCTURED_BUFFER) });
        let h = self.handle_from_binding(ResClass::Uav, lower, lower, space, lower);
        let handle = self.annotate(h, kind::STRUCTURED_BUFFER | 0x1000, stride);
        RawUav { handle, binding: Some(id) }
    }
    /// A structured buffer from the heap.
    pub fn heap_structured_buffer(&mut self, index: Value, uav: bool, stride: u32, nonuniform: bool) -> RawUav {
        self.require(Feature::RawAndStructuredBuffers);
        let h = self.handle_from_heap(index, false, nonuniform);
        let handle = self.annotate(h, kind::STRUCTURED_BUFFER | if uav { 0x1000 } else { 0 }, stride);
        RawUav { handle, binding: None }
    }
    /// Structured load of `mask` consecutive i32s of element `index` at byte `offset`.
    pub fn structured_load(&mut self, buf: RawUav, index: Value, offset: u32, mask: u8) -> Vec<Value> {
        let i32 = self.m.i32();
        let rt = self.res_ret(i32);
        let (off, mk, al) = (self.m.i32c(offset), self.m.i8c(mask), self.m.i32c(4));
        let r = self.op(139, "rawBufferLoad", Some(i32), rt, &[buf.handle, index, off, mk, al], FnAttrs::ReadOnly);
        (0..4).filter(|c| mask & (1 << c) != 0).map(|c| self.m.extract(r, c)).collect()
    }
    /// Structured store of up to four i32s to element `index` at byte `offset`.
    pub fn structured_store(&mut self, buf: RawUav, index: Value, offset: u32, vals: &[Value]) {
        assert!((1..=4).contains(&vals.len()));
        let (i32, void) = (self.m.i32(), self.m.void());
        let undef = self.m.undef(i32);
        let mut v = [undef; 4];
        v[..vals.len()].copy_from_slice(vals);
        let mask = (1u8 << vals.len()) - 1;
        let (off, mk, al) = (self.m.i32c(offset), self.m.i8c(mask), self.m.i32c(4));
        self.op(140, "rawBufferStore", Some(i32), void, &[buf.handle, index, off, v[0], v[1], v[2], v[3], mk, al], FnAttrs::NoUnwind);
    }

    // ---- resource atomics

    /// 64-bit resource atomics: `Int64Ops`; typed resources also need
    /// `AtomicInt64Typed`, heap resources `AtomicInt64Heap`, and static
    /// bindings carry the metadata tag (as DXC sets them).
    fn note_atomic64(&mut self, v: Value, binding: Option<usize>, typed: bool) {
        if self.is_i64(v) {
            self.require(Feature::Int64Ops);
            if typed {
                self.require(Feature::AtomicInt64Typed);
            }
            match binding {
                Some(b) => self.bindings[b].atomic64 = true,
                None => self.require(Feature::AtomicInt64Heap),
            }
        }
    }
    /// Declares the features of a 64-bit atomic on a heap texture/typed
    /// buffer handle used with [`Shader::atomic_binop`] / [`Shader::atomic_cmpxchg`].
    pub fn heap_typed_atomic64(&mut self) {
        self.require(Feature::Int64Ops);
        self.require(Feature::AtomicInt64Typed);
        self.require(Feature::AtomicInt64Heap);
    }
    fn atomic_coords(&mut self, c: &[Value]) -> [Value; 3] {
        assert!((1..=3).contains(&c.len()));
        let i32 = self.m.i32();
        let u = self.m.undef(i32);
        let mut out = [u; 3];
        out[..c.len()].copy_from_slice(c);
        out
    }
    /// `AtomicBinOp` (op 78) on a raw buffer at byte `offset` (i32 or i64 value);
    /// returns the old value.
    pub fn raw_atomic(&mut self, buf: RawUav, op: AtomicOp, offset: Value, v: Value) -> Value {
        self.note_atomic64(v, buf.binding, false);
        self.atomic_binop(buf.handle, op, &[offset], v)
    }
    /// `AtomicBinOp` on a structured buffer element `index` at byte `offset`.
    pub fn structured_atomic(&mut self, buf: RawUav, op: AtomicOp, index: Value, offset: u32, v: Value) -> Value {
        self.note_atomic64(v, buf.binding, false);
        let off = self.m.i32c(offset);
        self.atomic_binop(buf.handle, op, &[index, off], v)
    }
    /// `AtomicBinOp` on a typed buffer element.
    pub fn typed_atomic(&mut self, buf: TypedBuffer, op: AtomicOp, index: Value, v: Value) -> Value {
        self.note_atomic64(v, buf.binding, true);
        self.atomic_binop(buf.handle, op, &[index], v)
    }
    /// Raw `AtomicBinOp` with up to three coordinates (typed UAV textures:
    /// texel coordinates). Callers handle 64-bit feature flags.
    pub fn atomic_binop(&mut self, handle: Value, op: AtomicOp, coords: &[Value], v: Value) -> Value {
        let t = self.m.type_of(v);
        let c = self.atomic_coords(coords);
        let o = self.m.i32c(op as u32);
        self.op(78, "atomicBinOp", Some(t), t, &[handle, o, c[0], c[1], c[2], v], FnAttrs::NoUnwind)
    }
    /// `AtomicCompareExchange` (op 79) on a raw buffer; returns the old value.
    pub fn raw_cmpxchg(&mut self, buf: RawUav, offset: Value, cmp: Value, new: Value) -> Value {
        self.note_atomic64(cmp, buf.binding, false);
        self.atomic_cmpxchg(buf.handle, &[offset], cmp, new)
    }
    /// `AtomicCompareExchange` with up to three coordinates.
    pub fn atomic_cmpxchg(&mut self, handle: Value, coords: &[Value], cmp: Value, new: Value) -> Value {
        let t = self.m.type_of(cmp);
        let c = self.atomic_coords(coords);
        self.op(79, "atomicCompareExchange", Some(t), t, &[handle, c[0], c[1], c[2], cmp, new], FnAttrs::NoUnwind)
    }
}

// --------------------------------------------------------------- textures

impl Shader {
    /// A texture from `ResourceDescriptorHeap[index]` with `ncomp`
    /// components of `comp` (`Texture2D<float4>` = `(D2, false, F32, 4)`).
    pub fn heap_texture(&mut self, index: Value, kind: TexKind, uav: bool, comp: CompType, ncomp: u32, nonuniform: bool) -> Texture {
        let h = self.handle_from_heap(index, false, nonuniform);
        let handle = self.annotate(h, kind as u32 | if uav { 0x1000 } else { 0 }, typed_props(comp, ncomp));
        Texture { handle, kind, uav, comp }
    }
    /// A sampler from `SamplerDescriptorHeap[index]`.
    pub fn heap_sampler(&mut self, index: Value, comparison: bool, nonuniform: bool) -> Sampler {
        let h = self.handle_from_heap(index, true, nonuniform);
        let handle = self.annotate(h, 14 | if comparison { 0x8000 } else { 0 }, 0);
        Sampler { handle, comparison }
    }

    fn tex_ty(&mut self, t: Texture) -> Ty {
        self.comp_ty(t.comp)
    }
    fn ret4(&mut self, r: Value) -> [Value; 4] {
        [0, 1, 2, 3].map(|c| self.m.extract(r, c))
    }
    fn padded(&mut self, vals: &[Value], n: usize, ty: Ty) -> Vec<Value> {
        assert!(vals.len() <= n, "too many coordinates");
        let u = self.m.undef(ty);
        let mut v = vals.to_vec();
        v.resize(n, u);
        v
    }
    /// Offsets: given ones as constants, missing ones 0 for the kind's
    /// dimensions (as DXC emits) and undef past them.
    fn offsets(&mut self, kind: TexKind, offs: &[i32], n: usize) -> Vec<Value> {
        let i32 = self.m.i32();
        let u = self.m.undef(i32);
        (0..n)
            .map(|i| if i < kind.offsets() { self.m.i32c(offs.get(i).copied().unwrap_or(0) as u32) } else { u })
            .collect()
    }
    fn coords_f(&mut self, coords: &[Value]) -> Vec<Value> {
        let f32 = self.m.f32();
        self.padded(coords, 4, f32)
    }

    /// `TextureLoad` (op 66): integer texel coordinates (array slice
    /// included), `mip` for SRVs (`None` on UAVs), immediate offsets.
    pub fn texture_load(&mut self, tex: Texture, mip: Option<Value>, coords: &[Value], offs: &[i32]) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let i32 = self.m.i32();
        let mip = mip.unwrap_or_else(|| self.m.undef(i32));
        let c = self.padded(coords, 3, i32);
        let o = if offs.is_empty() { self.padded(&[], 3, i32) } else { self.offsets(tex.kind, offs, 3) };
        if tex.uav && tex.comp != CompType::U32 && tex.comp != CompType::I32 && tex.comp != CompType::F32 {
            self.require(Feature::TypedUavLoadAdditionalFormats);
        }
        let mut args = vec![tex.handle, mip];
        args.extend(c);
        args.extend(o);
        let r = self.op(66, "textureLoad", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    /// `TextureLoad` on a UAV loading more than one component needs
    /// `TypedUavLoadAdditionalFormats`; call this to declare it.
    pub fn typed_uav_load_additional_formats(&mut self) {
        self.require(Feature::TypedUavLoadAdditionalFormats);
    }
    /// `TextureStore` (op 67) of four components (the full mask) to a UAV texture.
    pub fn texture_store(&mut self, tex: Texture, coords: &[Value], vals: [Value; 4]) {
        let i32 = self.m.i32();
        let void = self.m.void();
        let t = self.m.type_of(vals[0]);
        let c = self.padded(coords, 3, i32);
        let mk = self.m.i8c(0xf);
        let mut args = vec![tex.handle];
        args.extend(c);
        args.extend(vals);
        args.push(mk);
        let t = if t == self.m.f64() { panic!("no f64 texture stores") } else { t };
        self.op(67, "textureStore", Some(t), void, &args, FnAttrs::NoUnwind);
    }

    fn sample_common(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32]) -> Vec<Value> {
        let mut args = vec![tex.handle, s.handle];
        let c = self.coords_f(coords);
        args.extend(c);
        let o = self.offsets(tex.kind, offs, 3);
        args.extend(o);
        args
    }
    fn clamp_arg(&mut self, clamp: Option<Value>) -> Value {
        match clamp {
            Some(c) => {
                self.require(Feature::TiledResources);
                c
            }
            None => {
                let f32 = self.m.f32();
                self.m.undef(f32)
            }
        }
    }
    /// `Sample` (op 60; implicit derivatives: pixel shaders, or compute with
    /// quad-shaped thread groups in SM 6.6). `clamp` = min LOD.
    pub fn sample(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], clamp: Option<Value>) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let mut args = self.sample_common(tex, s, coords, offs);
        let c = self.clamp_arg(clamp);
        args.push(c);
        let r = self.op(60, "sample", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    /// `SampleBias` (op 61).
    pub fn sample_bias(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], bias: Value, clamp: Option<Value>) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let mut args = self.sample_common(tex, s, coords, offs);
        let c = self.clamp_arg(clamp);
        args.extend([bias, c]);
        let r = self.op(61, "sampleBias", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    /// `SampleLevel` (op 62): explicit LOD, legal in every stage.
    pub fn sample_level(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], lod: Value) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let mut args = self.sample_common(tex, s, coords, offs);
        args.push(lod);
        let r = self.op(62, "sampleLevel", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    /// `SampleGrad` (op 63) with explicit gradients (`kind.grads()` each).
    pub fn sample_grad(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], ddx: &[Value], ddy: &[Value], clamp: Option<Value>) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let f32 = self.m.f32();
        let mut args = self.sample_common(tex, s, coords, offs);
        let dx = self.padded(ddx, 3, f32);
        let dy = self.padded(ddy, 3, f32);
        args.extend(dx);
        args.extend(dy);
        let c = self.clamp_arg(clamp);
        args.push(c);
        let r = self.op(63, "sampleGrad", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    /// `SampleCmp` (op 64; implicit derivatives like `Sample`).
    pub fn sample_cmp(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], cmp: Value, clamp: Option<Value>) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let mut args = self.sample_common(tex, s, coords, offs);
        let c = self.clamp_arg(clamp);
        args.extend([cmp, c]);
        let r = self.op(64, "sampleCmp", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    /// `SampleCmpLevelZero` (op 65).
    pub fn sample_cmp_level_zero(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], cmp: Value) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let mut args = self.sample_common(tex, s, coords, offs);
        args.push(cmp);
        let r = self.op(65, "sampleCmpLevelZero", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    fn gather_args(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], channel: u32) -> Vec<Value> {
        let mut args = vec![tex.handle, s.handle];
        let c = self.coords_f(coords);
        args.extend(c);
        let o = self.offsets(tex.kind, offs, 2);
        args.extend(o);
        let ch = self.m.i32c(channel);
        args.push(ch);
        args
    }
    /// `TextureGather` (op 73) of `channel` (0 = red … 3 = alpha); the result
    /// holds the texels in D3D order (-u+v, +u+v, +u-v, -u-v).
    pub fn gather(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], channel: u32) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let args = self.gather_args(tex, s, coords, offs, channel);
        let r = self.op(73, "textureGather", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    /// `TextureGatherCmp` (op 74).
    pub fn gather_cmp(&mut self, tex: Texture, s: Sampler, coords: &[Value], offs: &[i32], channel: u32, cmp: Value) -> [Value; 4] {
        let t = self.tex_ty(tex);
        let rt = self.res_ret(t);
        let mut args = self.gather_args(tex, s, coords, offs, channel);
        args.push(cmp);
        let r = self.op(74, "textureGatherCmp", Some(t), rt, &args, FnAttrs::ReadOnly);
        self.ret4(r)
    }
    /// `GetDimensions` (op 72): a `dx.types.Dimensions` struct of i32
    /// `{width, height, depth / array size, mip levels}`; extract only the
    /// components the resource kind defines (the validator rejects reads of
    /// undefined ones, e.g. `z` of a `Texture2D`). Buffers: `x` = elements
    /// (raw buffers: bytes). `mip` is `None` for UAVs and buffers.
    pub fn get_dimensions(&mut self, handle: Value, mip: Option<Value>) -> Value {
        let i32 = self.m.i32();
        let st = self.m.named_struct("dx.types.Dimensions", &[i32, i32, i32, i32]);
        let mip = mip.unwrap_or_else(|| self.m.undef(i32));
        self.op(72, "getDimensions", None, st, &[handle, mip], FnAttrs::ReadOnly)
    }
    /// `CalculateLOD` (op 81): clamped or unclamped LOD for `coords` (spatial only).
    pub fn calculate_lod(&mut self, tex: Texture, s: Sampler, coords: &[Value], clamped: bool) -> Value {
        let f32 = self.m.f32();
        let c = self.padded(coords, 3, f32);
        let cl = self.m.i1c(clamped);
        self.op(81, "calculateLOD", Some(f32), f32, &[tex.handle, s.handle, c[0], c[1], c[2], cl], FnAttrs::ReadOnly)
    }
}

// ------------------------------------------------------------------- wave

impl Shader {
    pub fn wave_is_first_lane(&mut self) -> Value {
        self.require(Feature::WaveOps);
        let i1 = self.m.i1();
        self.op(110, "waveIsFirstLane", None, i1, &[], FnAttrs::NoUnwind)
    }
    pub fn wave_lane_count(&mut self) -> Value {
        self.require(Feature::WaveOps);
        let i32 = self.m.i32();
        self.op(112, "waveGetLaneCount", None, i32, &[], FnAttrs::ReadNone)
    }
    /// `WaveActiveAnyTrue(cond)`.
    pub fn wave_any_true(&mut self, cond: Value) -> Value {
        self.require(Feature::WaveOps);
        let i1 = self.m.i1();
        self.op(113, "waveAnyTrue", None, i1, &[cond], FnAttrs::NoUnwind)
    }
    /// `WaveActiveAllTrue(cond)`.
    pub fn wave_all_true(&mut self, cond: Value) -> Value {
        self.require(Feature::WaveOps);
        let i1 = self.m.i1();
        self.op(114, "waveAllTrue", None, i1, &[cond], FnAttrs::NoUnwind)
    }
    pub fn wave_all_equal(&mut self, v: Value) -> Value {
        self.require(Feature::WaveOps);
        let t = self.m.type_of(v);
        let i1 = self.m.i1();
        self.op(115, "waveActiveAllEqual", Some(t), i1, &[v], FnAttrs::NoUnwind)
    }
    fn wave_int64(&mut self, v: Value) {
        self.require(Feature::WaveOps);
        if self.is_i64(v) {
            self.require(Feature::Int64Ops);
        }
        if self.is_f64(v) {
            self.require(Feature::Doubles);
        }
    }
    /// `WaveActiveOp` (op 119); `unsigned` selects unsigned integer min/max
    /// (and is ignored for sum/product bits but kept as DXC emits it).
    pub fn wave_active_op(&mut self, v: Value, op: WaveOp, unsigned: bool) -> Value {
        self.wave_int64(v);
        let t = self.m.type_of(v);
        let (o, s) = (self.m.i8c(op as u8), self.m.i8c(unsigned as u8));
        self.op(119, "waveActiveOp", Some(t), t, &[v, o, s], FnAttrs::NoUnwind)
    }
    /// `WaveActiveBitAnd/Or/Xor` (op 120; integers).
    pub fn wave_active_bit(&mut self, v: Value, op: WaveBit) -> Value {
        self.wave_int64(v);
        let t = self.m.type_of(v);
        let o = self.m.i8c(op as u8);
        self.op(120, "waveActiveBit", Some(t), t, &[v, o], FnAttrs::NoUnwind)
    }
    /// `WavePrefixSum/Product` (op 121; exclusive). DXIL only defines sum
    /// and product prefixes.
    pub fn wave_prefix_op(&mut self, v: Value, op: WaveOp, unsigned: bool) -> Value {
        assert!(matches!(op, WaveOp::Sum | WaveOp::Product), "WavePrefixOp only sums and multiplies");
        self.wave_int64(v);
        let t = self.m.type_of(v);
        let (o, s) = (self.m.i8c(op as u8), self.m.i8c(unsigned as u8));
        self.op(121, "wavePrefixOp", Some(t), t, &[v, o, s], FnAttrs::NoUnwind)
    }
    /// `WaveActiveCountBits(cond)` (op 135).
    pub fn wave_all_bit_count(&mut self, cond: Value) -> Value {
        self.require(Feature::WaveOps);
        let i32 = self.m.i32();
        self.op(135, "waveAllOp", None, i32, &[cond], FnAttrs::NoUnwind)
    }
    /// `WavePrefixCountBits(cond)` (op 136; exclusive).
    pub fn wave_prefix_bit_count(&mut self, cond: Value) -> Value {
        self.require(Feature::WaveOps);
        let i32 = self.m.i32();
        self.op(136, "wavePrefixOp", None, i32, &[cond], FnAttrs::NoUnwind)
    }
    /// `QuadReadLaneAt(v, lane)` (op 122; lane 0..3, uniform).
    pub fn quad_read_lane_at(&mut self, v: Value, lane: Value) -> Value {
        self.wave_int64(v);
        let t = self.m.type_of(v);
        self.op(122, "quadReadLaneAt", Some(t), t, &[v, lane], FnAttrs::NoUnwind)
    }
    /// `QuadReadAcrossX/Y/Diagonal` (op 123).
    pub fn quad_op(&mut self, v: Value, dir: QuadDir) -> Value {
        self.wave_int64(v);
        let t = self.m.type_of(v);
        let d = self.m.i8c(dir as u8);
        self.op(123, "quadOp", Some(t), t, &[v, d], FnAttrs::NoUnwind)
    }
    /// `WaveMatch(v)` (op 165, SM 6.5) → lane mask words.
    pub fn wave_match(&mut self, v: Value) -> [Value; 4] {
        self.wave_int64(v);
        let t = self.m.type_of(v);
        let i32 = self.m.i32();
        let fr = self.m.named_struct("dx.types.fouri32", &[i32, i32, i32, i32]);
        let r = self.op(165, "waveMatch", Some(t), fr, &[v], FnAttrs::NoUnwind);
        self.ret4(r)
    }
    /// `WaveMultiPrefixOp` (op 166, SM 6.5): exclusive prefix over the lanes
    /// of `mask` with a [`multi_prefix`] operation.
    pub fn wave_multi_prefix_op(&mut self, v: Value, mask: [Value; 4], op: u8, unsigned: bool) -> Value {
        self.wave_int64(v);
        let t = self.m.type_of(v);
        // DXC always marks the bitwise ops signed; AMD's compiler removes
        // the device on `WaveMultiPrefixBitXor` with sop = unsigned.
        let unsigned = unsigned && !matches!(op, multi_prefix::AND | multi_prefix::OR | multi_prefix::XOR);
        let (o, s) = (self.m.i8c(op), self.m.i8c(unsigned as u8));
        self.op(166, "waveMultiPrefixOp", Some(t), t, &[v, mask[0], mask[1], mask[2], mask[3], o, s], FnAttrs::NoUnwind)
    }
    /// `WaveMultiPrefixCountBits` (op 167, SM 6.5).
    pub fn wave_multi_prefix_bit_count(&mut self, cond: Value, mask: [Value; 4]) -> Value {
        self.require(Feature::WaveOps);
        let i32 = self.m.i32();
        self.op(167, "waveMultiPrefixBitCount", None, i32, &[cond, mask[0], mask[1], mask[2], mask[3]], FnAttrs::NoUnwind)
    }

    // ---- derivatives

    /// `ddx`/`ddy` (coarse or fine). Pixel shaders, or (SM 6.6) compute /
    /// mesh / amplification shaders whose thread groups form quads.
    pub fn deriv(&mut self, op: Deriv, v: Value) -> Value {
        let t = self.m.type_of(v);
        self.op(op as u32, "unary", Some(t), t, &[v], FnAttrs::ReadNone)
    }
}

/// `WaveMultiPrefixOp` operation codes (`DXIL::WaveMultiPrefixOpKind`).
pub mod multi_prefix {
    pub const SUM: u8 = 0;
    pub const AND: u8 = 1;
    pub const OR: u8 = 2;
    pub const XOR: u8 = 3;
    pub const PRODUCT: u8 = 4;
}

/// Float unary operations with an f64 overload (checked against the
/// validator in the tests); the others are f16 / f32 only.
pub const DOUBLE_UNARY: [FUnary; 2] = [FUnary::FAbs, FUnary::Saturate];
