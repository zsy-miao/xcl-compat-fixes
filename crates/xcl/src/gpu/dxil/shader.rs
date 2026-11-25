//! A DXIL shader under construction: the [`Module`] plus what DXIL adds on
//! top of LLVM — `dx.op` intrinsics, resource handles, entry-point metadata,
//! shader flags, PSV0 — and the finished, signed container.
//!
//! Operation numbers and signatures follow DXC's `DxilOperations` tables and
//! `docs/DXIL.rst` (NCSA licence); each helper is checked against DXC output
//! and the validator in the tests.

use std::collections::BTreeSet;

use super::container::{self, Psv, PsvResType, PsvResource, Stage};
use super::module::{FnAttrs, Md, Module, Ty, Value};

pub const TRIPLE: &str = "dxil-ms-dx";
pub const DATALAYOUT: &str = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64";

/// Optional features a shader uses; each sets a metadata shader flag and,
/// for most, a container feature bit (`SFI0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Feature {
    /// Prevents drivers from refactoring floating-point operations (e.g.
    /// contracting separate `fmul` + `fadd` into FMA).
    DisableMathRefactoring,
    Doubles,
    RawAndStructuredBuffers,
    WaveOps,
    Int64Ops,
    NativeLowPrecision,
    ResourceHeapIndexing,
    SamplerHeapIndexing,
    AtomicInt64Typed,
    AtomicInt64GroupShared,
    AtomicInt64Heap,
    /// `fma`, double division / reciprocal (D3D11.1 double extensions).
    DoubleExtensions,
    /// Sample/gather with a `clamp` operand (min-LOD), CheckAccessFullyMapped.
    TiledResources,
    /// Typed UAV loads of multi-component / non-32-bit formats.
    TypedUavLoadAdditionalFormats,
    /// Rasterizer-Ordered Views (ROVs) in pixel shaders.
    Rovs,
}

impl Feature {
    /// `DxilShaderFlags` bit (entry-point metadata tag 0).
    pub(super) fn metadata_bit(self) -> u64 {
        if self == Feature::NativeLowPrecision {
            // UseNativeLowPrecision | LowPrecisionPresent
            return (1 << 23) | (1 << 5);
        }
        1u64 << match self {
            Feature::Doubles => 2,
            Feature::RawAndStructuredBuffers => 4,
            Feature::Rovs => 18,
            Feature::WaveOps => 19,
            Feature::Int64Ops => 20,
            Feature::NativeLowPrecision => 23,
            Feature::AtomicInt64Typed => 27,
            Feature::AtomicInt64GroupShared => 28,
            Feature::ResourceHeapIndexing => 30,
            Feature::SamplerHeapIndexing => 31,
            Feature::AtomicInt64Heap => 32,
            Feature::DoubleExtensions => 6,
            Feature::TiledResources => 12,
            Feature::TypedUavLoadAdditionalFormats => 13,
            Feature::DisableMathRefactoring => 1,
        }
    }
    /// `D3D_SHADER_REQUIRES_*` container feature bit.
    pub(super) fn container_bit(self) -> u64 {
        match self {
            Feature::Doubles => 0x1,
            Feature::RawAndStructuredBuffers => 0,
            Feature::Rovs => 0x1000,
            Feature::WaveOps => 0x4000,
            Feature::Int64Ops => 0x8000,
            Feature::NativeLowPrecision => 0x40000,
            Feature::AtomicInt64Typed => 0x40_0000,
            Feature::AtomicInt64GroupShared => 0x80_0000,
            Feature::ResourceHeapIndexing => 0x200_0000,
            Feature::SamplerHeapIndexing => 0x400_0000,
            Feature::AtomicInt64Heap => 0x1000_0000,
            Feature::DoubleExtensions => 0x20,
            Feature::TiledResources => 0x100,
            Feature::TypedUavLoadAdditionalFormats => 0x800,
            Feature::DisableMathRefactoring => 0,
        }
    }
}

/// Resource classes (`DXIL::ResourceClass`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResClass {
    Srv = 0,
    Uav = 1,
    Cbv = 2,
    Sampler = 3,
}

/// A statically bound resource range.
#[derive(Debug, Clone)]
pub(super) struct Binding {
    pub(super) class: ResClass,
    pub(super) space: u32,
    pub(super) lower: u32,
    /// `DXIL::ResourceKind`.
    pub(super) kind: u32,
    /// Structured stride (0 otherwise).
    pub(super) stride: u32,
    /// Typed resources: `DXIL::ComponentType` of the elements (0 otherwise).
    pub(super) comp_type: u32,
    /// 64-bit atomics are used on the resource (metadata tag 3).
    pub(super) atomic64: bool,
    /// CBVs: size in bytes.
    pub(super) size: u32,
    /// Samplers: `SamplerComparisonState`.
    pub(super) comparison: bool,
    /// UAVs: `globallycoherent`
    pub(super) coherent: bool,
    /// UAVs: Rasterizer-Ordered View (`RasterizerOrdered*`, pixel shaders only).
    pub(super) rov: bool,
}

impl Binding {
    pub(super) fn new(class: ResClass, space: u32, lower: u32, kind: u32) -> Binding {
        Binding { class, space, lower, kind, stride: 0, comp_type: 0, atomic64: false, size: 0, comparison: false, coherent: false, rov: false }
    }
}

/// `DXIL::ResourceKind` values used here.
pub mod kind {
    pub const TEXTURE2D: u32 = 2;
    pub const CBUFFER: u32 = 13;
    pub const SAMPLER: u32 = 14;
    pub const TYPED_BUFFER: u32 = 10;
    pub const RAW_BUFFER: u32 = 11;
    pub const STRUCTURED_BUFFER: u32 = 12;
}

pub struct Shader {
    pub m: Module,
    pub stage: Stage,
    pub(super) sm: (u32, u32),
    pub(super) entry: u32,
    pub(super) features: BTreeSet<Feature>,
    pub(super) bindings: Vec<Binding>,
    pub(super) handle_ty: Ty,
    props_ty: Ty,
    /// When enabled on a pixel shader, UAV handles and static UAV bindings
    /// created afterwards carry the `IsROV` (`0x2000`) property.
    pub(super) rov: bool,
}

/// A raw (byte-address) UAV bound at `u<lower>, space<space>`.
#[derive(Debug, Clone, Copy)]
pub struct RawUav {
    pub(super) handle: Value,
    /// Index into the static bindings (`None` for heap handles).
    pub(super) binding: Option<usize>,
}

impl Shader {
    pub(super) fn dispatch_piece(&self) -> Self {
        Self {
            m: self.m.declarations(),
            stage: self.stage,
            sm: self.sm,
            entry: self.entry,
            features: self.features.clone(),
            bindings: self.bindings.clone(),
            handle_ty: self.handle_ty,
            props_ty: self.props_ty,
            rov: self.rov,
        }
    }

    pub fn new(stage: Stage, sm: (u32, u32)) -> Shader {
        let mut m = Module::new();
        let entry = m.define_entry("main");
        let i8 = m.i8();
        let p8 = m.ptr(i8, 0);
        let handle_ty = m.named_struct("dx.types.Handle", &[p8]);
        let i32 = m.i32();
        let props_ty = m.named_struct("dx.types.ResourceProperties", &[i32, i32]);
        Shader { m, stage, sm, entry, features: BTreeSet::new(), bindings: Vec::new(), handle_ty, props_ty, rov: false }
    }

    /// Enables Rasterizer-Ordered View (ROV) marking for UAV bindings and
    /// heap UAV handles created afterwards in this pixel shader.
    pub fn enable_rov(&mut self) {
        assert_eq!(self.stage, Stage::Pixel, "ROVs are only valid in pixel shaders");
        self.rov = true;
    }

    pub fn require(&mut self, f: Feature) {
        self.features.insert(f);
    }

    /// Calls `dx.op.<name>[.<overload>]` with opcode `op` and the given arguments.
    pub fn op(&mut self, op: u32, name: &str, overload: Option<Ty>, ret: Ty, args: &[Value], attrs: FnAttrs) -> Value {
        let full = match overload {
            Some(t) => format!("dx.op.{name}.{}", self.type_suffix(t)),
            None => format!("dx.op.{name}"),
        };
        let i32 = self.m.i32();
        let mut params = vec![i32];
        for &a in args {
            params.push(self.m.type_of(a));
        }
        if let Some(want) = dxc_fn_attrs(op) {
            debug_assert_eq!(attrs, want, "dx.op {op} ({name}) attributes differ from DXC's");
        }
        let fty = self.m.func_ty(ret, &params);
        let f = self.m.declare(&full, fty, attrs);
        let opc = self.m.i32c(op);
        let mut all = vec![opc];
        all.extend_from_slice(args);
        self.m.call(f, &all)
    }

    pub(super) fn type_suffix(&mut self, t: Ty) -> &'static str {
        if t == self.m.f16() {
            "f16"
        } else if t == self.m.f32() {
            "f32"
        } else if t == self.m.f64() {
            "f64"
        } else if t == self.m.i1() {
            "i1"
        } else if t == self.m.int(16) {
            "i16"
        } else if t == self.m.i32() {
            "i32"
        } else if t == self.m.i64() {
            "i64"
        } else if t == self.m.i8() {
            "i8"
        } else {
            self.m.fail(format!("no dx.op overload suffix for type {t:?}"));
            "i32"
        }
    }

    // ---- system values

    /// `SV_GroupThreadID` component.
    pub fn thread_id_in_group(&mut self, comp: u32) -> Value {
        let i32 = self.m.i32();
        let c = self.m.i32c(comp);
        self.op(95, "threadIdInGroup", Some(i32), i32, &[c], FnAttrs::ReadNone)
    }
    /// `SV_DispatchThreadID` component.
    pub fn thread_id(&mut self, comp: u32) -> Value {
        let i32 = self.m.i32();
        let c = self.m.i32c(comp);
        self.op(93, "threadId", Some(i32), i32, &[c], FnAttrs::ReadNone)
    }
    /// `SV_GroupID` component.
    pub fn group_id(&mut self, comp: u32) -> Value {
        let i32 = self.m.i32();
        let c = self.m.i32c(comp);
        self.op(94, "groupId", Some(i32), i32, &[c], FnAttrs::ReadNone)
    }
    /// `SV_GroupIndex`.
    pub fn flattened_thread_id_in_group(&mut self) -> Value {
        let i32 = self.m.i32();
        self.op(96, "flattenedThreadIdInGroup", Some(i32), i32, &[], FnAttrs::ReadNone)
    }

    // ---- resources

    /// Declares a raw UAV at `u<lower>, space<space>` and creates its handle
    /// in the entry block (call before emitting other code in that block).
    pub fn raw_uav(&mut self, lower: u32, space: u32) -> RawUav {
        self.require(Feature::RawAndStructuredBuffers);
        let id = self.bindings.len();
        self.bindings.push(Binding { rov: self.rov, ..Binding::new(ResClass::Uav, space, lower, kind::RAW_BUFFER) });
        let h = self.handle_from_binding(ResClass::Uav, lower, lower, space, lower);
        let handle = self.annotate(h, kind::RAW_BUFFER | 0x1000, 0);
        RawUav { handle, binding: Some(id) }
    }

    /// `globallycoherent RWByteAddressBuffer` at `u<lower>, space<space>`.
    pub fn raw_uav_coherent(&mut self, lower: u32, space: u32) -> RawUav {
        self.require(Feature::RawAndStructuredBuffers);
        let id = self.bindings.len();
        self.bindings.push(Binding { coherent: true, rov: self.rov, ..Binding::new(ResClass::Uav, space, lower, kind::RAW_BUFFER) });
        let h = self.handle_from_binding(ResClass::Uav, lower, lower, space, lower);
        // resource properties: IsUAV (bit 12) | IsGloballyCoherent (bit 14)
        let handle = self.annotate(h, kind::RAW_BUFFER | 0x1000 | 0x4000, 0);
        RawUav { handle, binding: Some(id) }
    }

    pub(super) fn handle_from_binding(&mut self, class: ResClass, lower: u32, upper: u32, space: u32, index: u32) -> Value {
        let i32 = self.m.i32();
        let i8 = self.m.i8();
        let bind_ty = self.m.named_struct("dx.types.ResBind", &[i32, i32, i32, i8]);
        let (lo, hi, sp, cl) = (self.m.i32c(lower), self.m.i32c(upper), self.m.i32c(space), self.m.i8c(class as u8));
        let bind = self.m.aggregate(bind_ty, &[lo, hi, sp, cl]);
        let idx = self.m.i32c(index);
        let nonuniform = self.m.i1c(false);
        let ht = self.handle_ty;
        self.op(217, "createHandleFromBinding", None, ht, &[bind, idx, nonuniform], FnAttrs::ReadNone)
    }

    /// A handle to `ResourceDescriptorHeap[index]` (or the sampler heap).
    pub fn handle_from_heap(&mut self, index: Value, sampler: bool, nonuniform: bool) -> Value {
        self.require(if sampler { Feature::SamplerHeapIndexing } else { Feature::ResourceHeapIndexing });
        let (s, n) = (self.m.i1c(sampler), self.m.i1c(nonuniform));
        let ht = self.handle_ty;
        self.op(218, "createHandleFromHeap", None, ht, &[index, s, n], FnAttrs::ReadNone)
    }

    /// `annotateHandle` with resource properties (`{kind | uav<<12 | ..., stride/format}`).
    pub fn annotate(&mut self, h: Value, props0: u32, props1: u32) -> Value {
        let props0 = if self.rov && (props0 & 0x1000 != 0) {
            // IsROV (bit 13) for UAV handles in POPS pixel shaders.
            props0 | 0x2000
        } else {
            props0
        };
        let (a, b) = (self.m.i32c(props0), self.m.i32c(props1));
        let pt = self.props_ty;
        let props = self.m.aggregate(pt, &[a, b]);
        let ht = self.handle_ty;
        self.op(216, "annotateHandle", None, ht, &[h, props], FnAttrs::ReadNone)
    }

    /// A raw-buffer handle from the descriptor heap (`RWByteAddressBuffer` if `uav`).
    pub fn heap_raw_buffer(&mut self, index: Value, uav: bool, nonuniform: bool) -> RawUav {
        let h = self.handle_from_heap(index, false, nonuniform);
        let handle = self.annotate(h, kind::RAW_BUFFER | if uav { 0x1000 } else { 0 }, 0);
        RawUav { handle, binding: None }
    }

    /// A `RWByteAddressBuffer` from `ResourceDescriptorHeap[NonUniformResourceIndex(index)]`,
    /// `globallycoherent` if `coherent` (the heap memory windows).
    pub fn heap_raw_buffer_uav(&mut self, index: Value, coherent: bool) -> RawUav {
        self.require(Feature::RawAndStructuredBuffers);
        let h = self.handle_from_heap(index, false, true);
        // resource properties: IsUAV (bit 12) | IsGloballyCoherent (bit 14)
        let handle = self.annotate(h, kind::RAW_BUFFER | 0x1000 | if coherent { 0x4000 } else { 0 }, 0);
        RawUav { handle, binding: None }
    }

    pub(super) fn res_ret(&mut self, t: Ty) -> Ty {
        let name = format!("dx.types.ResRet.{}", self.type_suffix(t));
        let i32 = self.m.i32();
        self.m.named_struct(&name, &[t, t, t, t, i32])
    }

    /// `rawBufferLoad` of `mask` (1..15) consecutive 32-bit values at byte `offset`.
    pub fn raw_load(&mut self, buf: RawUav, offset: Value, mask: u8) -> Vec<Value> {
        let i32 = self.m.i32();
        let rt = self.res_ret(i32);
        let undef = self.m.undef(i32);
        let (mk, al) = (self.m.i8c(mask), self.m.i32c(4));
        let r = self.op(139, "rawBufferLoad", Some(i32), rt, &[buf.handle, offset, undef, mk, al], FnAttrs::ReadOnly);
        (0..4).filter(|c| mask & (1 << c) != 0).map(|c| self.m.extract(r, c)).collect()
    }

    /// `rawBufferStore` of up to four 32-bit values at byte `offset`.
    pub fn raw_store(&mut self, buf: RawUav, offset: Value, vals: &[Value]) {
        if !(1..=4).contains(&vals.len()) {
            return self.m.fail(format!("rawBufferStore of {} values", vals.len()));
        }
        let i32 = self.m.i32();
        let void = self.m.void();
        let undef = self.m.undef(i32);
        let mut v = [undef; 4];
        v[..vals.len()].copy_from_slice(vals);
        let mask = (1u8 << vals.len()) - 1;
        let (mk, al) = (self.m.i8c(mask), self.m.i32c(4));
        self.op(140, "rawBufferStore", Some(i32), void, &[buf.handle, offset, undef, v[0], v[1], v[2], v[3], mk, al], FnAttrs::NoUnwind);
    }

    // ---- wave operations

    pub fn wave_read_lane_at(&mut self, v: Value, lane: Value) -> Value {
        self.require(Feature::WaveOps);
        let t = self.m.type_of(v);
        self.op(117, "waveReadLaneAt", Some(t), t, &[v, lane], FnAttrs::NoUnwind)
    }
    pub fn wave_read_lane_first(&mut self, v: Value) -> Value {
        self.require(Feature::WaveOps);
        let t = self.m.type_of(v);
        self.op(118, "waveReadLaneFirst", Some(t), t, &[v], FnAttrs::NoUnwind)
    }
    pub fn wave_lane_index(&mut self) -> Value {
        self.require(Feature::WaveOps);
        let i32 = self.m.i32();
        self.op(111, "waveGetLaneIndex", None, i32, &[], FnAttrs::ReadOnly)
    }
    /// `WaveActiveBallot` → four i32 words (lanes 0-31, 32-63, …).
    pub fn wave_ballot(&mut self, cond: Value) -> [Value; 4] {
        self.require(Feature::WaveOps);
        let i32 = self.m.i32();
        let fr = self.m.named_struct("dx.types.fouri32", &[i32, i32, i32, i32]);
        let r = self.op(116, "waveActiveBallot", None, fr, &[cond], FnAttrs::NoUnwind);
        [0, 1, 2, 3].map(|i| self.m.extract(r, i))
    }

    // ---- finishing

    /// The `dx.resources` node (`!{srvs, uavs, cbvs, samplers}`, also the
    /// entry's resource field) and the PSV0 resource entries for the
    /// statically bound resources; `None` when nothing is bound. Record ids
    /// count per class in declaration order; PSV0 lists CBVs, samplers,
    /// SRVs, then UAVs (as DXC does).
    pub(super) fn resource_metadata(&mut self) -> (Option<Md>, Vec<PsvResource>) {
        let bindings = self.bindings.clone();
        let m = &mut self.m;
        let mut lists: [Vec<Option<Md>>; 4] = Default::default();
        let mut psv: [Vec<PsvResource>; 4] = Default::default();
        for b in &bindings {
            let ci = b.class as usize;
            let i32 = m.i32();
            let sname = match (b.class, b.kind) {
                (ResClass::Uav, kind::RAW_BUFFER) if b.rov => "struct.RasterizerOrderedByteAddressBuffer",
                (ResClass::Uav, kind::TYPED_BUFFER) if b.rov => "class.RasterizerOrderedBuffer",
                (ResClass::Uav, kind::STRUCTURED_BUFFER) if b.rov => "struct.RasterizerOrderedStructuredBuffer",
                (ResClass::Uav, _) if b.rov => "class.RasterizerOrderedTexture",
                (ResClass::Uav, kind::RAW_BUFFER) => "struct.RWByteAddressBuffer",
                (ResClass::Uav, kind::TYPED_BUFFER) => "class.RWBuffer",
                (ResClass::Uav, kind::STRUCTURED_BUFFER) => "struct.RWStructuredBuffer",
                (ResClass::Uav, _) => "class.RWTexture",
                (ResClass::Srv, kind::RAW_BUFFER) => "struct.ByteAddressBuffer",
                (ResClass::Srv, kind::TYPED_BUFFER) => "class.Buffer",
                (ResClass::Srv, kind::STRUCTURED_BUFFER) => "class.StructuredBuffer",
                (ResClass::Srv, _) => "class.Texture",
                (ResClass::Cbv, _) => "CB",
                (ResClass::Sampler, _) if b.comparison => "struct.SamplerComparisonState",
                (ResClass::Sampler, _) => "struct.SamplerState",
            };
            let st = if b.class == ResClass::Cbv {
                // Register-sized struct of the cbuffer's bytes.
                let a = m.array(b.size.div_ceil(4).max(1) as u64, i32);
                let name = format!("CB.{}.{}", b.space, b.lower);
                m.named_struct(&name, &[a])
            } else {
                m.named_struct(sname, &[i32])
            };
            let pst = m.ptr(st, 0);
            let undef = m.undef(pst);
            let id = m.md_i32(lists[ci].len() as u32);
            let var = m.md_value(undef);
            let name = m.md_str("");
            let space = m.md_i32(b.space);
            let lower = m.md_i32(b.lower);
            let range = m.md_i32(1);
            let mut node = vec![Some(id), Some(var), Some(name), Some(space), Some(lower), Some(range)];
            // Extended properties: tag 0 element type, 1 stride, 3 64-bit atomics.
            let mut ext: Vec<Option<Md>> = Vec::new();
            if b.comp_type != 0 {
                ext.extend([Some(m.md_i32(0)), Some(m.md_i32(b.comp_type))]);
            }
            if b.stride != 0 {
                ext.extend([Some(m.md_i32(1)), Some(m.md_i32(b.stride))]);
            }
            if b.atomic64 {
                ext.extend([Some(m.md_i32(3)), Some(m.md_i32(1))]);
            }
            let extra = if ext.is_empty() { None } else { Some(m.md_node(&ext)) };
            match b.class {
                ResClass::Uav => {
                    let (k, f, gc, rov) = (m.md_i32(b.kind), m.md_i1(false), m.md_i1(b.coherent), m.md_i1(b.rov));
                    // globallycoherent, hasCounter, rasterizerOrdered.
                    node.extend([Some(k), Some(gc), Some(f), Some(rov), extra]);
                }
                ResClass::Srv => {
                    let (k, samples) = (m.md_i32(b.kind), m.md_i32(0));
                    node.extend([Some(k), Some(samples), extra]);
                }
                ResClass::Cbv => node.extend([Some(m.md_i32(b.size)), None]),
                ResClass::Sampler => node.extend([Some(m.md_i32(b.comparison as u32)), None]),
            }
            lists[ci].push(Some(m.md_node(&node)));
            let res_type = match (b.class, b.kind) {
                (ResClass::Cbv, _) => PsvResType::Cbv,
                (ResClass::Sampler, _) => PsvResType::Sampler,
                (ResClass::Srv, kind::RAW_BUFFER) => PsvResType::SrvRaw,
                (ResClass::Srv, kind::STRUCTURED_BUFFER) => PsvResType::SrvStructured,
                (ResClass::Srv, _) => PsvResType::SrvTyped,
                (ResClass::Uav, kind::RAW_BUFFER) => PsvResType::UavRaw,
                (ResClass::Uav, kind::STRUCTURED_BUFFER) => PsvResType::UavStructured,
                (ResClass::Uav, _) => PsvResType::UavTyped,
            };
            // PSVResourceFlags::UsedByAtomic64
            psv[ci].push(PsvResource { res_type, space: b.space, lower: b.lower, upper: b.lower, kind: b.kind, flags: b.atomic64 as u32 });
        }
        if lists.iter().all(|l| l.is_empty()) {
            return (None, Vec::new());
        }
        let parts: Vec<Option<Md>> = lists.iter().map(|l| if l.is_empty() { None } else { Some(m.md_node(l)) }).collect();
        let node = m.md_node(&parts);
        let [srv, uav, cbv, smp] = psv;
        (Some(node), [cbv, smp, srv, uav].concat())
    }

    /// Requires the features implied by instruction types alone (`Int64Ops`
    /// for any i64, `Doubles` for any double, `DoubleExtensions` for double
    /// division and int ↔ double conversions), which the validator derives
    /// from the code and demands in the flags even when the IR was built
    /// with [`Module`] methods directly rather than the [`Shader`] helpers.
    pub(super) fn infer_type_features(&mut self) {
        let (i64, f64) = self.m.existing_i64_f64();
        if i64.is_some_and(|t| self.m.insts_use_type(t)) {
            self.require(Feature::Int64Ops);
        }
        if f64.is_some_and(|t| self.m.insts_use_type(t)) {
            self.require(Feature::Doubles);
        }
        if self.m.uses_double_extensions() {
            self.require(Feature::DoubleExtensions);
        }
        if self.bindings.iter().any(|b| b.rov) || self.m.has_rov_handle() {
            self.require(Feature::Rovs);
        }
    }

    /// Writes metadata, bitcode and the signed container (panics on a builder error: tests;
    /// translators use [`Shader::try_finish`]).
    pub fn finish(self, numthreads: [u32; 3], wave_size: Option<u32>) -> Vec<u8> {
        self.try_finish(numthreads, wave_size).unwrap_or_else(|e| panic!("DXIL builder: {e}"))
    }

    /// [`Shader::finish`], or the first builder error.
    pub fn try_finish(mut self, numthreads: [u32; 3], wave_size: Option<u32>) -> Result<Vec<u8>, String> {
        if self.stage != Stage::Compute {
            return Err(format!("finish of a {:?} entry (only compute; graphics use finish_graphics)", self.stage));
        }
        if !self.m.terminated() {
            self.m.ret();
        }
        self.m.optimize();
        self.finish_optimized(numthreads, wave_size)
    }

    pub(super) fn finish_optimized(mut self, numthreads: [u32; 3], wave_size: Option<u32>) -> Result<Vec<u8>, String> {
        self.infer_type_features();
        let (flags, sfi0) = self.feature_flags();

        let m = &mut self.m;
        let one = m.md_i32(1);
        let ver = m.md_i32(self.sm.1);
        let dx_version = m.md_node(&[Some(one), Some(ver)]);
        let valver = m.md_i32(super::validate::VALVER.1);
        let dx_valver = m.md_node(&[Some(one), Some(valver)]);
        let prefix = m.md_str(self.stage.profile_prefix());
        let smaj = m.md_i32(self.sm.0);
        let shader_model = m.md_node(&[Some(prefix), Some(smaj), Some(ver)]);

        let (resources, psv_res) = self.resource_metadata();
        let m = &mut self.m;

        // Entry properties.
        let mut props: Vec<Option<Md>> = Vec::new();
        if flags != 0 {
            let tag = m.md_i32(0);
            let v = m.md_i64(flags);
            props.extend([Some(tag), Some(v)]);
        }
        let tag = m.md_i32(4);
        let nt: Vec<Option<Md>> = numthreads.iter().map(|&n| Some(m.md_i32(n))).collect();
        let ntn = m.md_node(&nt);
        props.extend([Some(tag), Some(ntn)]);
        if let Some(w) = wave_size {
            let tag = m.md_i32(11);
            let wv = m.md_i32(w);
            let wn = m.md_node(&[Some(wv)]);
            props.extend([Some(tag), Some(wn)]);
        }
        let props_node = m.md_node(&props);
        let fv = m.func_value(self.entry);
        let fmd = m.md_value(fv);
        let name = m.md_str("main");
        let entry = m.md_node(&[Some(fmd), Some(name), None, resources, Some(props_node)]);

        // `!llvm.ident`, as DXC writes first: AMD's driver compiler (amdxc64) crashes at
        // PSO creation on modules without it.
        let ident = m.md_str("xcl");
        let ident = m.md_node(&[Some(ident)]);
        m.named_md("llvm.ident", &[ident]);
        m.named_md("dx.version", &[dx_version]);
        m.named_md("dx.valver", &[dx_valver]);
        m.named_md("dx.shaderModel", &[shader_model]);
        if let Some(r) = resources {
            m.named_md("dx.resources", &[r]);
        }
        m.named_md("dx.entryPoints", &[entry]);

        let bitcode = m.try_to_bitcode(TRIPLE, DATALAYOUT)?;
        let (min_wave, max_wave) = wave_size.map_or((0, u32::MAX), |w| (w, w));
        let psv = Psv { stage: self.stage, min_wave, max_wave, numthreads, resources: psv_res, entry: "main".into() };
        Ok(container::container(self.stage, self.sm, &bitcode, sfi0, &psv))
    }

    /// Computes `(metadata_flags, container_sfi0)` from required features.
    ///
    /// Omits `DisableMathRefactoring` when the optimized module still invokes
    /// `dx.op.tertiary.*` `FMad` (op 46), since DXIL validation forbids `FMad`
    /// together with `DisableMathRefactoring`.
    pub(super) fn feature_flags(&self) -> (u64, u64) {
        let drop_no_refactor = self.features.contains(&Feature::DisableMathRefactoring) && self.uses_fmad();
        let mut flags = 0u64;
        let mut sfi0 = 0u64;
        for &f in &self.features {
            if drop_no_refactor && f == Feature::DisableMathRefactoring {
                continue;
            }
            flags |= f.metadata_bit();
            sfi0 |= f.container_bit();
        }
        (flags, sfi0)
    }

    /// True if any surviving instruction calls `dx.op.tertiary.*` with opcode
    /// `46` (`Tertiary::FMad`).
    fn uses_fmad(&self) -> bool {
        use super::module::{Block, InstInfo};
        for b in 0..self.m.block_count() {
            for &inst in self.m.block_insts(Block(b as u32)) {
                let InstInfo::Call(name, args) = self.m.inst_info(inst) else { continue };
                if name.starts_with("dx.op.tertiary.")
                    && args.first().and_then(|&v| self.m.const_u64(v)) == Some(46)
                {
                    return true;
                }
            }
        }
        false
    }
}

/// The function attributes DXC gives each `dx.op` declaration (DXC 1.9
/// `DxilOperations.cpp`, checked against `dxc -Fc` listings). AMD's compiler
/// is sensitive to them (e.g. `waveGetLaneIndex` must be `readonly`), so
/// [`Shader::op`] checks every declaration against this table in debug
/// builds. `None` for opcodes not listed.
pub fn dxc_fn_attrs(op: u32) -> Option<FnAttrs> {
    use FnAttrs::*;
    Some(match op {
        4 => ReadNone,                         // loadInput
        5 => NoUnwind,                         // storeOutput
        6..=56 => ReadNone,                    // unary/binary/tertiary/quaternary/dot
        57..=66 => ReadOnly,                   // createHandle, cbuffer loads, samples, textureLoad
        67 => NoUnwind,                        // textureStore
        68 => ReadOnly,                        // bufferLoad
        69 | 70 => NoUnwind,                   // bufferStore, bufferUpdateCounter
        71..=77 => ReadOnly,                   // checkAccess, getDimensions, gathers, sample positions
        78 | 79 => NoUnwind,                   // atomics
        80 => NoDuplicate,                     // barrier
        81 => ReadOnly,                        // calculateLOD
        82 => NoUnwind,                        // discard
        83..=96 => ReadNone,                   // derivatives, eval*, sampleIndex, coverage, ids
        97..=99 => NoUnwind,                   // emitStream, cutStream, emitThenCutStream
        100..=102 => ReadNone,                 // gsInstanceID, makeDouble, splitDouble
        110 => NoUnwind,                       // waveIsFirstLane
        111 => ReadOnly,                       // waveGetLaneIndex
        112 => ReadNone,                       // waveGetLaneCount
        113..=123 => NoUnwind,                 // wave / quad ops
        124..=132 => ReadNone,                 // bitcasts, legacy conversions
        135 | 136 => NoUnwind,                 // waveAllOp, wavePrefixBitCount
        137 | 138 => ReadNone,                 // attributeAtVertex, viewID
        139 => ReadOnly,                       // rawBufferLoad
        140 => NoUnwind,                       // rawBufferStore
        165..=169 => NoUnwind,                 // waveMatch, waveMultiPrefix*, mesh outputs
        170 => ReadOnly,                       // getMeshPayload
        171 | 172 => NoUnwind,                 // storeVertexOutput, storePrimitiveOutput
        216..=218 => ReadNone,                 // annotateHandle, createHandleFromBinding/Heap
        221 => ReadOnly,                       // isHelperLane
        _ => return Option::None,
    })
}

#[cfg(test)]
mod pops_rov_tests {
    use super::super::kernel::{self, KernelOptions};
    use super::super::validate;
    use crate::gpu::rdna2;
    use crate::gpu::recomp::{self, PsInterface};

    fn ps_with_pops(pops_wait: bool, write_uav: bool) -> Vec<u8> {
        let mut code = Vec::new();
        if pops_wait {
            // Typical RDNA2 POPS wait reading src_pops_exiting_wave_id (operand 239).
            code.extend(rdna2::encode("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 18), ("SSRC1", 239)], None).unwrap());
        }
        if write_uav {
            // Heap texture UAV store + flat memory store
            code.extend(rdna2::encode("IMAGE_STORE", "ENC_MIMG", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1), ("UNORM", 1)], None).unwrap());
            code.extend(rdna2::encode("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 0), ("DATA", 4), ("SADDR", 0x7D), ("OFFSET", 0)], None).unwrap());
        }
        code.extend(rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        code
    }

    #[test]
    fn pops_pixel_shader_marks_uavs_as_rov_and_validates() {
        let code = ps_with_pops(true, true);
        let ps = PsInterface::default();
        // Both static memory windows (u2) and heap memory windows (XCL_HEAP_WINDOWS=1) must validate.
        for mem_windows in [1u32, 4 | recomp::HEAP_WINDOWS] {
            let k = kernel::translate_pixel(&code, KernelOptions { wave64: true, split: false, mem_windows, sampler_base: 0 }, &ps)
                .expect("translate POPS PS");
            validate::validate(&k.dxil).expect("dxil validate POPS PS");
            let dis = validate::disassemble(&k.dxil).expect("disassemble POPS PS");
            assert!(dis.contains("Raster Ordered UAVs"), "expected Raster Ordered UAVs feature flag (mem_windows={mem_windows:#x}):\n{dis}");
            assert!(dis.contains("ROVTexture2D<"), "expected ROVTexture2D annotation (mem_windows={mem_windows:#x}):\n{dis}");
            assert!(dis.contains("ROVByteAddressBuffer"), "expected ROVByteAddressBuffer annotation (mem_windows={mem_windows:#x}):\n{dis}");
            // u0 (SGPRs) is created before enable_rov() and must remain an ordinary RWByteAddressBuffer.
            assert!(dis.contains("resource: RWByteAddressBuffer"), "expected u0 to remain non-ROV RWByteAddressBuffer:\n{dis}");
        }
    }

    #[test]
    fn pops_pixel_shader_without_uavs_on_heap_windows_validates_cleanly() {
        let code = ps_with_pops(true, false);
        let ps = PsInterface::default();
        let k = kernel::translate_pixel(&code, KernelOptions { wave64: true, split: false, mem_windows: 4 | recomp::HEAP_WINDOWS, sampler_base: 0 }, &ps)
            .expect("translate POPS PS without UAVs");
        validate::validate(&k.dxil).expect("dxil validate POPS PS without UAVs");
        let dis = validate::disassemble(&k.dxil).expect("disassemble POPS PS without UAVs");
        assert!(!dis.contains("Raster Ordered UAVs"), "POPS PS with no ROV handles must not emit dangling ROV flag:\n{dis}");
    }

    #[test]
    fn non_pops_pixel_shader_does_not_mark_rovs() {
        let code = ps_with_pops(false, true);
        let ps = PsInterface::default();
        let k = kernel::translate_pixel(&code, KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 }, &ps)
            .expect("translate normal PS");
        validate::validate(&k.dxil).expect("dxil validate normal PS");
        let dis = validate::disassemble(&k.dxil).expect("disassemble normal PS");
        assert!(!dis.contains("Raster Ordered UAVs"), "non-POPS PS must not require ROVs:\n{dis}");
        assert!(!dis.contains("ROVTexture2D<"), "non-POPS PS must not mark ROVTexture2D:\n{dis}");
    }
}
