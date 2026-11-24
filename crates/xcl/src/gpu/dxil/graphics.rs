//! Graphics stages for the direct DXIL emitter: vertex, pixel and mesh
//! shaders — signature elements and their packing, the I/O intrinsics
//! (`loadInput`, `storeOutput`, mesh outputs, discard, coverage…), the
//! `dx.entryPoints` signature metadata, `dx.viewIdState`, and the
//! `ISG1`/`OSG1`/`PSG1`/`PSV0` container parts.
//!
//! Layouts and rules follow DXC's output for small HLSL programs (NCSA
//! licence; see `docs/specs/2026-11-24-direct-dxil.md`): element
//! packing is DXC's default prefix-stable first fit, the PSV0 dependency
//! tables are recomputed from the emitted code the way the validator does.

use std::collections::{BTreeSet, HashMap};

use super::container::{self, PsvFull, PsvSigElement, SigPartElement, Stage};
use super::module::{Block, FnAttrs, InstInfo, Md, Ty, Value};
use super::shader::{Feature, ResClass, Shader, DATALAYOUT, TRIPLE};

/// The data layout DXC writes with `-enable-16bit-types`.
pub const DATALAYOUT_NATIVE16: &str = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64";

/// `DXIL::SemanticKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemKind {
    Arbitrary = 0,
    VertexId = 1,
    InstanceId = 2,
    Position = 3,
    RenderTargetArrayIndex = 4,
    ViewportArrayIndex = 5,
    ClipDistance = 6,
    CullDistance = 7,
    PrimitiveId = 10,
    SampleIndex = 12,
    IsFrontFace = 13,
    Coverage = 14,
    InnerCoverage = 15,
    Target = 16,
    Depth = 17,
    DepthLessEqual = 18,
    DepthGreaterEqual = 19,
    StencilRef = 20,
    ShadingRate = 29,
    CullPrimitive = 30,
}

impl SemKind {
    const ALL: [SemKind; 20] = [
        SemKind::Arbitrary,
        SemKind::VertexId,
        SemKind::InstanceId,
        SemKind::Position,
        SemKind::RenderTargetArrayIndex,
        SemKind::ViewportArrayIndex,
        SemKind::ClipDistance,
        SemKind::CullDistance,
        SemKind::PrimitiveId,
        SemKind::SampleIndex,
        SemKind::IsFrontFace,
        SemKind::Coverage,
        SemKind::InnerCoverage,
        SemKind::Target,
        SemKind::Depth,
        SemKind::DepthLessEqual,
        SemKind::DepthGreaterEqual,
        SemKind::StencilRef,
        SemKind::ShadingRate,
        SemKind::CullPrimitive,
    ];
    /// The kind for a `D3D_NAME` system value (as console / DXBC signatures carry it).
    pub fn from_d3d_name(name: u32) -> Option<SemKind> {
        Self::ALL.into_iter().find(|k| k.d3d_name() == name)
    }
    /// `D3D_NAME` as signature parts store it.
    pub fn d3d_name(self) -> u32 {
        match self {
            SemKind::Arbitrary => 0,
            SemKind::Position => 1,
            SemKind::ClipDistance => 2,
            SemKind::CullDistance => 3,
            SemKind::RenderTargetArrayIndex => 4,
            SemKind::ViewportArrayIndex => 5,
            SemKind::VertexId => 6,
            SemKind::PrimitiveId => 7,
            SemKind::InstanceId => 8,
            SemKind::IsFrontFace => 9,
            SemKind::SampleIndex => 10,
            SemKind::Target => 64,
            SemKind::Depth => 65,
            SemKind::Coverage => 66,
            SemKind::DepthGreaterEqual => 67,
            SemKind::DepthLessEqual => 68,
            SemKind::StencilRef => 69,
            SemKind::InnerCoverage => 70,
            SemKind::ShadingRate => 24,
            SemKind::CullPrimitive => 25,
        }
    }
}

/// Component types (`DXIL::ComponentType`) supported in signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompType {
    /// `bool` (`SV_CullPrimitive`, …): stored and loaded as `i32`.
    Bool,
    I32,
    U32,
    F32,
    I16,
    U16,
    F16,
}

impl CompType {
    /// From a `D3D_REGISTER_COMPONENT_TYPE` (1 uint, 2 sint, 3 float).
    pub fn from_d3d(t: u32) -> Option<CompType> {
        match t {
            1 => Some(CompType::U32),
            2 => Some(CompType::I32),
            3 => Some(CompType::F32),
            _ => None,
        }
    }
    fn md(self) -> u8 {
        match self {
            CompType::Bool => 1,
            CompType::I16 => 2,
            CompType::U16 => 3,
            CompType::I32 => 4,
            CompType::U32 => 5,
            CompType::F16 => 8,
            CompType::F32 => 9,
        }
    }
    /// `D3D_REGISTER_COMPONENT_TYPE`.
    fn d3d(self) -> u32 {
        match self {
            CompType::U32 | CompType::Bool => 1,
            CompType::I32 => 2,
            CompType::F32 => 3,
            CompType::U16 => 4,
            CompType::I16 => 5,
            CompType::F16 => 6,
        }
    }
    fn is_16bit(self) -> bool {
        matches!(self, CompType::F16 | CompType::I16 | CompType::U16)
    }
    fn psv(self) -> u8 {
        match self {
            CompType::U32 | CompType::Bool => 1,
            CompType::I32 => 2,
            CompType::F32 => 3,
            CompType::U16 => 4,
            CompType::I16 => 5,
            CompType::F16 => 6,
        }
    }
    /// `D3D_MIN_PRECISION` (0 with native 16-bit types).
    fn min_precision(self, native16: bool) -> u32 {
        if native16 {
            return 0;
        }
        match self {
            CompType::F16 => 1,
            CompType::I16 => 4,
            CompType::U16 => 5,
            _ => 0,
        }
    }
    fn is_float(self) -> bool {
        matches!(self, CompType::F32 | CompType::F16)
    }
}

/// `DXIL::InterpolationMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    Undefined = 0,
    Constant = 1,
    Linear = 2,
    LinearCentroid = 3,
    LinearNoperspective = 4,
    LinearNoperspectiveCentroid = 5,
    LinearSample = 6,
    LinearNoperspectiveSample = 7,
}

/// Mesh output topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    Line = 1,
    Triangle = 2,
}

/// A signature element as declared (one HLSL parameter / struct member).
#[derive(Debug, Clone)]
pub struct SigElement {
    pub name: String,
    /// Semantic index of each row.
    pub indices: Vec<u32>,
    pub kind: SemKind,
    pub ty: CompType,
    /// `None`: DXC's default for the stage and type.
    pub interp: Option<Interp>,
    pub cols: u8,
    /// Fixed placement (row, column); `None` packs like DXC.
    pub place: Option<(u32, u8)>,
}

impl SigElement {
    /// A one-row element `name<index>` of `cols` components.
    pub fn new(name: &str, index: u32, kind: SemKind, ty: CompType, cols: u8) -> SigElement {
        SigElement { name: name.into(), indices: vec![index], kind, ty, interp: None, cols, place: None }
    }
    /// An arbitrary (user) semantic.
    pub fn user(name: &str, index: u32, ty: CompType, cols: u8) -> SigElement {
        SigElement::new(name, index, SemKind::Arbitrary, ty, cols)
    }
    /// An element from a DXBC-style signature row (`D3D_NAME`, component type,
    /// component mask): columns span the mask, the start column is kept as a
    /// fixed placement only if `register` is given.
    pub fn from_d3d(name: &str, index: u32, system_value: u32, component_type: u32, mask: u8, register: Option<u32>) -> Option<SigElement> {
        let kind = SemKind::from_d3d_name(system_value)?;
        let ty = CompType::from_d3d(component_type)?;
        if mask & 0xF == 0 {
            return None;
        }
        let start = mask.trailing_zeros() as u8;
        let hi = 7 - (mask & 0xF).leading_zeros() as u8;
        let cols = hi - start + 1;
        let mut e = SigElement::new(name, index, kind, ty, cols);
        e.place = register.map(|r| (r, start));
        Some(e)
    }
    pub fn interp(mut self, i: Interp) -> SigElement {
        self.interp = Some(i);
        self
    }
    /// `rows` rows with consecutive semantic indices from the first.
    pub fn rows(mut self, rows: u32) -> SigElement {
        let first = self.indices[0];
        self.indices = (first..first + rows).collect();
        self
    }
    pub fn at(mut self, row: u32, col: u8) -> SigElement {
        self.place = Some((row, col));
        self
    }
    fn nrows(&self) -> u32 {
        self.indices.len() as u32
    }
}

/// Which signature an element belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    In,
    Out,
    Prim,
}

/// A handle to a declared element (its id within its signature).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sig {
    dir: Dir,
    id: u32,
}

/// How an element is laid out (`DXIL::SemanticInterpretationKind`, simplified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interpretation {
    /// User semantic: packed.
    Arb,
    /// System value packed like a user one, but nothing may follow it in its row.
    Sv,
    /// `SV_ClipDistance` / `SV_CullDistance`: share rows only with each other.
    ClipCull,
    /// Not packed: `register` -1 (depth, coverage, sample index…).
    NotPacked,
    /// `SV_Target`: row = semantic index.
    Target,
}

/// Placed element.
#[derive(Debug, Clone)]
struct Placed {
    e: SigElement,
    interp: Interp,
    /// `None`: unallocated.
    at: Option<(u32, u8)>,
    /// Components read (inputs) / written (outputs), relative to the start column.
    usage: u8,
}

/// Vertex / pixel / mesh I/O declared on a shader.
pub struct GraphicsIo {
    stage: Stage,
    inputs: Vec<SigElement>,
    outputs: Vec<SigElement>,
    prims: Vec<SigElement>,
    usage: HashMap<(u8, u32), u8>,
    uses_sample_index: bool,
}

/// Stage-specific entry information.
#[derive(Debug, Clone, Copy)]
pub enum StageInfo {
    Vertex,
    Pixel { early_depth_stencil: bool },
    Mesh { numthreads: [u32; 3], max_vertices: u32, max_primitives: u32, topology: Topology, payload_bytes: u32 },
}

fn dir_code(d: Dir) -> u8 {
    d as u8
}

impl GraphicsIo {
    pub fn new(stage: Stage) -> GraphicsIo {
        assert!(matches!(stage, Stage::Vertex | Stage::Pixel | Stage::Mesh), "graphics I/O for {stage:?}");
        GraphicsIo { stage, inputs: Vec::new(), outputs: Vec::new(), prims: Vec::new(), usage: HashMap::new(), uses_sample_index: false }
    }

    /// Declares an input element (VS/PS).
    pub fn input(&mut self, e: SigElement) -> Sig {
        assert_ne!(self.stage, Stage::Mesh, "mesh shaders have no input signature");
        self.inputs.push(e);
        Sig { dir: Dir::In, id: self.inputs.len() as u32 - 1 }
    }
    /// Declares an output element (VS/PS outputs, MS vertex outputs).
    pub fn output(&mut self, e: SigElement) -> Sig {
        self.outputs.push(e);
        Sig { dir: Dir::Out, id: self.outputs.len() as u32 - 1 }
    }
    /// Declares a mesh primitive output element.
    pub fn primitive(&mut self, e: SigElement) -> Sig {
        assert_eq!(self.stage, Stage::Mesh);
        self.prims.push(e);
        Sig { dir: Dir::Prim, id: self.prims.len() as u32 - 1 }
    }

    fn elem(&self, s: Sig) -> &SigElement {
        match s.dir {
            Dir::In => &self.inputs[s.id as usize],
            Dir::Out => &self.outputs[s.id as usize],
            Dir::Prim => &self.prims[s.id as usize],
        }
    }

    fn mark(&mut self, s: Sig, col: u8) {
        let e = self.elem(s);
        assert!(col < e.cols, "column {col} outside {} ({} columns)", e.name, e.cols);
        *self.usage.entry((dir_code(s.dir), s.id)).or_default() |= 1 << col;
    }

    fn scalar_ty(s: &mut Shader, t: CompType) -> Ty {
        match t {
            CompType::F32 => s.m.f32(),
            CompType::F16 => s.m.f16(),
            CompType::I32 | CompType::U32 | CompType::Bool => s.m.i32(),
            CompType::I16 | CompType::U16 => s.m.int(16),
        }
    }

    /// `loadInput` of component `col` of row `row` (relative to the element).
    /// PS `SV_SampleIndex` is read with `sampleIndex` instead.
    pub fn load_input(&mut self, s: &mut Shader, sig: Sig, row: u32, col: u8) -> Value {
        assert_eq!(sig.dir, Dir::In);
        let e = self.elem(sig).clone();
        if e.kind == SemKind::SampleIndex {
            self.uses_sample_index = true;
            let i32 = s.m.i32();
            return s.op(90, "sampleIndex", Some(i32), i32, &[], FnAttrs::ReadNone);
        }
        assert!(row < e.nrows());
        self.mark(sig, col);
        let t = Self::scalar_ty(s, e.ty);
        let (id, r, c) = (s.m.i32c(sig.id), s.m.i32c(row), s.m.i8c(col));
        let i32 = s.m.i32();
        let undef = s.m.undef(i32);
        s.op(4, "loadInput", Some(t), t, &[id, r, c, undef], FnAttrs::ReadNone)
    }

    /// `storeOutput` of `v` into component `col` of row `row`.
    pub fn store_output(&mut self, s: &mut Shader, sig: Sig, row: u32, col: u8, v: Value) {
        assert_eq!(sig.dir, Dir::Out);
        assert_ne!(self.stage, Stage::Mesh, "mesh vertex outputs use store_vertex_output");
        let e = self.elem(sig).clone();
        assert!(row < e.nrows());
        self.mark(sig, col);
        let t = Self::scalar_ty(s, e.ty);
        let v = s.m.bitcast(v, t);
        let (id, r, c) = (s.m.i32c(sig.id), s.m.i32c(row), s.m.i8c(col));
        let void = s.m.void();
        s.op(5, "storeOutput", Some(t), void, &[id, r, c, v], FnAttrs::NoUnwind);
    }

    // ---- pixel shader

    /// `discard` when `cond` holds.
    pub fn discard(&mut self, s: &mut Shader, cond: Value) {
        assert_eq!(self.stage, Stage::Pixel);
        let void = s.m.void();
        s.op(82, "discard", None, void, &[cond], FnAttrs::NoUnwind);
    }
    /// The input `SV_Coverage` mask.
    pub fn coverage(&mut self, s: &mut Shader) -> Value {
        assert_eq!(self.stage, Stage::Pixel);
        let i32 = s.m.i32();
        s.op(91, "coverage", Some(i32), i32, &[], FnAttrs::ReadNone)
    }
    /// `SV_InnerCoverage`.
    pub fn inner_coverage(&mut self, s: &mut Shader) -> Value {
        assert_eq!(self.stage, Stage::Pixel);
        let i32 = s.m.i32();
        s.op(92, "innerCoverage", Some(i32), i32, &[], FnAttrs::ReadNone)
    }
    /// `IsHelperLane()`.
    pub fn is_helper_lane(&mut self, s: &mut Shader) -> Value {
        let i1 = s.m.i1();
        s.op(221, "isHelperLane", Some(i1), i1, &[], FnAttrs::ReadOnly)
    }

    // ---- mesh shader

    pub fn set_mesh_output_counts(&mut self, s: &mut Shader, vertices: Value, primitives: Value) {
        assert_eq!(self.stage, Stage::Mesh);
        let void = s.m.void();
        s.op(168, "setMeshOutputCounts", None, void, &[vertices, primitives], FnAttrs::NoUnwind);
    }
    /// Triangle (three indices) or line (two; the third operand is 0, as
    /// DXC emits it) indices of primitive `prim`.
    pub fn emit_indices(&mut self, s: &mut Shader, prim: Value, idx: &[Value]) {
        assert_eq!(self.stage, Stage::Mesh);
        assert!(matches!(idx.len(), 2 | 3), "{} indices per primitive", idx.len());
        let void = s.m.void();
        let mut a = vec![prim];
        a.extend_from_slice(idx);
        if idx.len() == 2 {
            a.push(s.m.i32c(0));
        }
        s.op(169, "emitIndices", None, void, &a, FnAttrs::NoUnwind);
    }
    pub fn store_vertex_output(&mut self, s: &mut Shader, sig: Sig, row: u32, col: u8, v: Value, vertex: Value) {
        assert_eq!((self.stage, sig.dir), (Stage::Mesh, Dir::Out));
        self.store_mesh(s, sig, row, col, v, vertex, 171, "storeVertexOutput");
    }
    pub fn store_primitive_output(&mut self, s: &mut Shader, sig: Sig, row: u32, col: u8, v: Value, prim: Value) {
        assert_eq!(sig.dir, Dir::Prim);
        self.store_mesh(s, sig, row, col, v, prim, 172, "storePrimitiveOutput");
    }
    #[allow(clippy::too_many_arguments)]
    fn store_mesh(&mut self, s: &mut Shader, sig: Sig, row: u32, col: u8, v: Value, index: Value, op: u32, name: &str) {
        let e = self.elem(sig).clone();
        assert!(row < e.nrows());
        self.mark(sig, col);
        let t = Self::scalar_ty(s, e.ty);
        let v = s.m.bitcast(v, t);
        let (id, r, c) = (s.m.i32c(sig.id), s.m.i32c(row), s.m.i8c(col));
        let void = s.m.void();
        s.op(op, name, Some(t), void, &[id, r, c, v, index], FnAttrs::NoUnwind);
    }
    /// `GetMeshPayload` as a pointer to `payload` (a named struct type
    /// `struct.<name>`); fields are read with GEP + load.
    pub fn mesh_payload(&mut self, s: &mut Shader, payload: Ty, struct_name: &str) -> Value {
        assert_eq!(self.stage, Stage::Mesh);
        let p = s.m.ptr(payload, 0);
        let i32 = s.m.i32();
        let fty = s.m.func_ty(p, &[i32]);
        let f = s.m.declare(&format!("dx.op.getMeshPayload.{struct_name}"), fty, FnAttrs::ReadOnly);
        let opc = s.m.i32c(170);
        s.m.call(f, &[opc])
    }

    // ---- layout

    fn interpretation(&self, dir: Dir, k: SemKind) -> Interpretation {
        use SemKind::*;
        match (self.stage, dir, k) {
            (_, _, Arbitrary) => Interpretation::Arb,
            (Stage::Pixel, Dir::Out, Target) => Interpretation::Target,
            (Stage::Pixel, Dir::Out, Depth | DepthLessEqual | DepthGreaterEqual | Coverage | StencilRef) => Interpretation::NotPacked,
            (Stage::Pixel, Dir::In, SampleIndex) => Interpretation::NotPacked,
            (Stage::Pixel, Dir::In, Coverage | InnerCoverage) => panic!("{k:?} is not a signature element (use coverage())"),
            (Stage::Mesh, Dir::Prim, CullPrimitive) => Interpretation::NotPacked,
            (_, _, ClipDistance | CullDistance) => Interpretation::ClipCull,
            _ => Interpretation::Sv,
        }
    }

    fn default_interp(&self, dir: Dir, e: &SigElement) -> Interp {
        use SemKind::*;
        match (self.stage, dir) {
            (Stage::Vertex, Dir::In) | (Stage::Pixel, Dir::Out) => Interp::Undefined,
            (_, Dir::Prim) => Interp::Constant,
            _ => match e.kind {
                Position => Interp::LinearNoperspective,
                Arbitrary | ClipDistance | CullDistance if e.ty.is_float() => Interp::Linear,
                SampleIndex if self.stage == Stage::Pixel => Interp::Constant,
                _ => Interp::Constant,
            },
        }
    }

    /// DXC's default prefix-stable first-fit packing.
    fn place(&self, dir: Dir) -> Vec<Placed> {
        let list = match dir {
            Dir::In => &self.inputs,
            Dir::Out => &self.outputs,
            Dir::Prim => &self.prims,
        };
        // Per row: occupied columns, interpolation, and the first column after a system value.
        #[derive(Clone, Default)]
        struct Row {
            used: u8,
            interp: Option<Interp>,
            sv_end: bool,
            clip_cull: bool,
        }
        let mut rows: Vec<Row> = Vec::new();
        let no_pack = self.stage == Stage::Vertex && dir == Dir::In;
        let mut next_row = 0u32;
        let mut out = Vec::new();
        for (id, e) in list.iter().enumerate() {
            let interp = e.interp.unwrap_or_else(|| self.default_interp(dir, e));
            let usage = self.usage.get(&(dir_code(dir), id as u32)).copied().unwrap_or(0);
            let ip = self.interpretation(dir, e.kind);
            let mask = ((1u16 << e.cols) - 1) as u8;
            let at = match (ip, e.place) {
                (Interpretation::NotPacked, _) => None,
                (_, Some(p)) => Some(p),
                (Interpretation::Target, None) => Some((e.indices[0], 0)),
                _ if no_pack => {
                    next_row += e.nrows();
                    Some((next_row - e.nrows(), 0))
                }
                _ => {
                    let fits = |rows: &Vec<Row>, r: u32, c: u8| {
                        (r..r + e.nrows()).all(|r| match rows.get(r as usize) {
                            None => true,
                            Some(row) => {
                                row.used & (mask << c) == 0
                                    && (row.used >> c) == 0
                                    && row.interp.is_none_or(|i| i == interp)
                                    && (row.used == 0 || row.clip_cull == (ip == Interpretation::ClipCull))
                                    && (!row.sv_end || ip == Interpretation::Sv)
                            }
                        })
                    };
                    let mut found = None;
                    'search: for r in 0.. {
                        for c in 0..=(4 - e.cols) {
                            if fits(&rows, r, c) {
                                found = Some((r, c));
                                break 'search;
                            }
                        }
                    }
                    found
                }
            };
            if let Some((r, c)) = at {
                for r in r..r + e.nrows() {
                    if rows.len() <= r as usize {
                        rows.resize(r as usize + 1, Row::default());
                    }
                    let row = &mut rows[r as usize];
                    row.used |= mask << c;
                    row.interp = Some(interp);
                    row.sv_end |= ip == Interpretation::Sv;
                    row.clip_cull |= ip == Interpretation::ClipCull;
                }
            }
            out.push(Placed { e: e.clone(), interp, at, usage });
        }
        out
    }
}

/// Number of allocated rows (signature vectors).
fn vectors(p: &[Placed]) -> u32 {
    p.iter().filter_map(|e| e.at.map(|(r, _)| r + e.e.nrows())).max().unwrap_or(0)
}

/// Number of allocated scalar components as `dx.viewIdState` counts them.
fn components(p: &[Placed]) -> u32 {
    p.iter().filter_map(|e| e.at.map(|(r, c)| (r + e.e.nrows() - 1) * 4 + c as u32 + e.e.cols as u32)).max().unwrap_or(0)
}

fn sig_part(p: &[Placed], output: bool, native16: bool) -> Vec<u8> {
    let mut rows = Vec::new();
    for e in p {
        let (reg, col) = e.at.map_or((u32::MAX, 0), |(r, c)| (r, c));
        let mask = (((1u16 << e.e.cols) - 1) as u8) << col;
        let used = e.usage << col;
        let rw = if output { !used & 0xF } else { used };
        for (i, &idx) in e.e.indices.iter().enumerate() {
            rows.push(SigPartElement {
                stream: 0,
                name: e.e.name.clone(),
                index: idx,
                system_value: e.e.kind.d3d_name(),
                comp_type: e.e.ty.d3d(),
                register: if reg == u32::MAX { reg } else { reg + i as u32 },
                mask,
                rw_mask: if e.at.is_none() && !output { 0 } else { rw },
                min_precision: e.e.ty.min_precision(native16),
            });
        }
    }
    rows.sort_by_key(|r| (r.register, r.mask.trailing_zeros()));
    let names: Vec<&str> = p.iter().map(|e| e.e.name.as_str()).collect();
    container::signature_part(&rows, &names)
}

fn psv_elems(p: &[Placed]) -> Vec<PsvSigElement> {
    p.iter()
        .map(|e| {
            let (r, c) = e.at.unwrap_or((0, 0));
            PsvSigElement {
                name: if e.e.kind == SemKind::Arbitrary { e.e.name.clone() } else { String::new() },
                indices: e.e.indices.clone(),
                rows: e.e.nrows() as u8,
                start_row: r as u8,
                cols: e.e.cols,
                start_col: c,
                allocated: e.at.is_some(),
                kind: e.e.kind as u8,
                comp_type: e.e.ty.psv(),
                interp: e.interp as u8,
                dynamic_mask: 0,
                stream: 0,
            }
        })
        .collect()
}

/// Input component → set of output components (`row * 4 + col`), from the code.
struct Deps {
    /// For each output component, the input components it depends on.
    out: HashMap<(u8, u32), BTreeSet<u32>>,
}

/// Input components each value depends on, through data flow, memory and
/// control dependence (as DXC's `ComputeViewIdState`).
fn compute_deps(s: &Shader, inputs: &[Placed], outputs: &[Placed], prims: &[Placed]) -> Deps {
    let m = &s.m;
    let nb = m.block_count();
    // Successors and the branch condition of each block.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); nb];
    let mut cond: Vec<Option<Value>> = vec![None; nb];
    for b in 0..nb {
        if let Some(&last) = m.block_insts(Block(b as u32)).last() {
            if let InstInfo::Term(c, ss) = m.inst_info(last) {
                succ[b] = ss.iter().map(|x| x.0 as usize).collect();
                if ss.len() > 1 {
                    cond[b] = c;
                }
            }
        }
    }
    // Immediate post-dominators with a virtual exit `nb` (Cooper, Harvey and Kennedy's
    // iteration over the reverse CFG; sets per block were quadratic in time and memory).
    let rsucc: Vec<Vec<usize>> = (0..nb).map(|b| if succ[b].is_empty() { vec![nb] } else { succ[b].clone() }).collect();
    let mut rpreds: Vec<Vec<usize>> = vec![Vec::new(); nb + 1];
    for (b, ss) in rsucc.iter().enumerate() {
        for &x in ss {
            rpreds[x].push(b);
        }
    }
    // reverse postorder of the reverse CFG from the exit
    let mut order: Vec<usize> = Vec::with_capacity(nb + 1);
    {
        let mut seen = vec![false; nb + 1];
        let mut stack: Vec<(usize, usize)> = vec![(nb, 0)];
        seen[nb] = true;
        while let Some((v, k)) = stack.last_mut() {
            if let Some(&w) = rpreds[*v].get(*k) {
                *k += 1;
                if !seen[w] {
                    seen[w] = true;
                    stack.push((w, 0));
                }
            } else {
                order.push(*v);
                stack.pop();
            }
        }
        order.reverse();
    }
    let mut rank = vec![usize::MAX; nb + 1];
    for (k, &v) in order.iter().enumerate() {
        rank[v] = k;
    }
    let mut ipdom: Vec<Option<usize>> = vec![None; nb + 1];
    ipdom[nb] = Some(nb);
    let mut changed = true;
    while changed {
        changed = false;
        for &v in order.iter().skip(1) {
            let mut new: Option<usize> = None;
            for &x in &rsucc[v] {
                if ipdom[x].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => x,
                    Some(mut p) => {
                        let mut q = x;
                        while p != q {
                            while rank[p] > rank[q] {
                                p = ipdom[p].unwrap();
                            }
                            while rank[q] > rank[p] {
                                q = ipdom[q].unwrap();
                            }
                        }
                        p
                    }
                });
            }
            if new.is_some() && ipdom[v] != new {
                ipdom[v] = new;
                changed = true;
            }
        }
    }
    // Block b is control dependent on branch block a if b post-dominates a
    // successor of a but does not strictly post-dominate a: the blocks from the
    // successor up the post-dominator tree to (excluding) a's immediate post-dominator.
    let mut ctrl: Vec<Vec<usize>> = vec![Vec::new(); nb];
    let mut stamp = vec![usize::MAX; nb + 1];
    for a in 0..nb {
        if cond[a].is_none() {
            continue;
        }
        let stop = ipdom[a];
        let mut marked: Vec<usize> = Vec::new();
        for &x in &succ[a] {
            let mut r = Some(x);
            while let Some(v) = r {
                if Some(v) == stop || v == nb || stamp[v] == a {
                    break;
                }
                stamp[v] = a;
                marked.push(v);
                r = ipdom[v].filter(|&p| p != v);
            }
        }
        marked.sort_unstable();
        for v in marked {
            ctrl[v].push(a);
        }
    }
    // Instruction → block.
    let mut block_of: HashMap<u32, usize> = HashMap::new();
    let mut root_stores: HashMap<Value, Vec<Value>> = HashMap::new();
    for b in 0..nb {
        for &i in m.block_insts(Block(b as u32)) {
            block_of.insert(i, b);
            if let InstInfo::Store(p, v) = m.inst_info(i) {
                root_stores.entry(root(s, p)).or_default().push(v);
            }
            if let InstInfo::Other(ops) = m.inst_info(i) {
                // atomics write memory too
                if ops.len() >= 2 {
                    if let Value::Global(_) = root(s, ops[0]) {
                        root_stores.entry(root(s, ops[0])).or_default().extend_from_slice(&ops[1..]);
                    }
                }
            }
        }
    }
    let in_base: Vec<Option<u32>> = inputs.iter().map(|e| e.at.map(|(r, c)| r * 4 + c as u32)).collect();
    let mut memo: HashMap<Value, BTreeSet<u32>> = HashMap::new();
    let mut ctrl_memo: HashMap<usize, BTreeSet<u32>> = HashMap::new();

    struct Cx<'a> {
        s: &'a Shader,
        block_of: &'a HashMap<u32, usize>,
        ctrl: &'a Vec<Vec<usize>>,
        cond: &'a Vec<Option<Value>>,
        root_stores: &'a HashMap<Value, Vec<Value>>,
        in_base: &'a [Option<u32>],
        inputs: &'a [Placed],
    }
    fn block_deps(cx: &Cx, b: usize, memo: &mut HashMap<Value, BTreeSet<u32>>, cm: &mut HashMap<usize, BTreeSet<u32>>) -> BTreeSet<u32> {
        if let Some(d) = cm.get(&b) {
            return d.clone();
        }
        cm.insert(b, BTreeSet::new());
        let mut d = BTreeSet::new();
        for &a in &cx.ctrl[b] {
            if let Some(c) = cx.cond[a] {
                d.extend(value_deps(cx, c, memo, cm));
            }
            if a != b {
                d.extend(block_deps(cx, a, memo, cm));
            }
        }
        cm.insert(b, d.clone());
        d
    }
    fn value_deps(cx: &Cx, v: Value, memo: &mut HashMap<Value, BTreeSet<u32>>, cm: &mut HashMap<usize, BTreeSet<u32>>) -> BTreeSet<u32> {
        let Value::Inst(i) = v else { return BTreeSet::new() };
        if let Some(d) = memo.get(&v) {
            return d.clone();
        }
        memo.insert(v, BTreeSet::new());
        let m = &cx.s.m;
        let mut d = BTreeSet::new();
        match m.inst_info(i) {
            InstInfo::Call(name, args) if name.starts_with("dx.op.loadInput.") => {
                let id = m.const_u64(args[1]).unwrap() as usize;
                let col = m.const_u64(args[3]).unwrap() as u32;
                if let Some(base) = cx.in_base[id] {
                    match m.const_u64(args[2]) {
                        Some(r) => {
                            d.insert(base + r as u32 * 4 + col);
                        }
                        None => {
                            for r in 0..cx.inputs[id].e.nrows() {
                                d.insert(base + r * 4 + col);
                            }
                            d.extend(value_deps(cx, args[2], memo, cm));
                        }
                    }
                }
            }
            InstInfo::Call(_, args) => {
                for &a in args {
                    d.extend(value_deps(cx, a, memo, cm));
                }
            }
            InstInfo::Phi(list) => {
                for &(x, from) in list {
                    d.extend(value_deps(cx, x, memo, cm));
                    d.extend(block_deps(cx, from.0 as usize, memo, cm));
                    if let Some(c) = cx.cond[from.0 as usize] {
                        d.extend(value_deps(cx, c, memo, cm));
                    }
                }
            }
            InstInfo::Load(p) => {
                d.extend(value_deps(cx, p, memo, cm));
                if let Some(vs) = cx.root_stores.get(&root(cx.s, p)) {
                    for &x in vs {
                        d.extend(value_deps(cx, x, memo, cm));
                    }
                }
            }
            InstInfo::Other(ops) => {
                for x in ops {
                    d.extend(value_deps(cx, x, memo, cm));
                }
            }
            InstInfo::Store(..) | InstInfo::Term(..) => {}
        }
        let _ = cx.block_of;
        memo.insert(v, d.clone());
        d
    }

    let cx = Cx { s, block_of: &block_of, ctrl: &ctrl, cond: &cond, root_stores: &root_stores, in_base: &in_base, inputs };
    let mut out: HashMap<(u8, u32), BTreeSet<u32>> = HashMap::new();
    let bases = |p: &[Placed]| -> Vec<Option<u32>> { p.iter().map(|e| e.at.map(|(r, c)| r * 4 + c as u32)).collect() };
    let (ob, pb) = (bases(outputs), bases(prims));
    for b in 0..nb {
        for &i in m.block_insts(Block(b as u32)) {
            let InstInfo::Call(name, args) = m.inst_info(i) else { continue };
            let (dir, base) = if name.starts_with("dx.op.storeOutput.") || name.starts_with("dx.op.storeVertexOutput.") {
                (0u8, &ob)
            } else if name.starts_with("dx.op.storePrimitiveOutput.") {
                (1u8, &pb)
            } else {
                continue;
            };
            let id = m.const_u64(args[1]).unwrap() as usize;
            let Some(base) = base[id] else { continue };
            let col = m.const_u64(args[3]).unwrap() as u32;
            let row = m.const_u64(args[2]).unwrap() as u32;
            let mut d = value_deps(&cx, args[4], &mut memo, &mut ctrl_memo);
            d.extend(block_deps(&cx, b, &mut memo, &mut ctrl_memo));
            if args.len() > 5 {
                d.extend(value_deps(&cx, args[5], &mut memo, &mut ctrl_memo));
            }
            out.entry((dir, base + row * 4 + col)).or_default().extend(d);
        }
    }
    Deps { out }
}

/// The global (or other base) a pointer is derived from.
fn root(s: &Shader, p: Value) -> Value {
    let mut p = p;
    loop {
        match p {
            Value::Inst(i) => match s.m.inst_info(i) {
                InstInfo::Other(ops) if !ops.is_empty() => p = ops[0],
                _ => return p,
            },
            _ => return p,
        }
    }
}

/// Dependency table: for each input component `0..n_in`, a bitmask over
/// `n_out` output components, `ceil(n_out / 32)` dwords each.
fn dep_table(deps: &Deps, dir: u8, n_in: u32, n_out: u32) -> Vec<u32> {
    let words = n_out.div_ceil(32) as usize;
    let mut t = vec![0u32; n_in as usize * words];
    for (&(d, o), ins) in &deps.out {
        if d != dir || o >= n_out {
            continue;
        }
        for &i in ins {
            if i < n_in {
                t[i as usize * words + (o / 32) as usize] |= 1 << (o % 32);
            }
        }
    }
    t
}

impl Shader {
    /// Writes metadata, bitcode and the signed container for a vertex,
    /// pixel or mesh entry point with the I/O declared in `io`.
    /// Panics on a builder error (tests; translators use [`Shader::try_finish_graphics`]).
    pub fn finish_graphics(self, io: GraphicsIo, info: StageInfo) -> Vec<u8> {
        self.try_finish_graphics(io, info).unwrap_or_else(|e| panic!("DXIL builder: {e}"))
    }

    /// [`Shader::finish_graphics`], or the first builder error.
    pub fn try_finish_graphics(mut self, io: GraphicsIo, info: StageInfo) -> Result<Vec<u8>, String> {
        if self.stage != io.stage {
            return Err(format!("{:?} shader with {:?} I/O", self.stage, io.stage));
        }
        match (self.stage, info) {
            (Stage::Vertex, StageInfo::Vertex) | (Stage::Pixel, StageInfo::Pixel { .. }) | (Stage::Mesh, StageInfo::Mesh { .. }) => {}
            (st, i) => return Err(format!("stage {st:?} with {i:?}")),
        }
        if !self.m.terminated() {
            self.m.ret();
        }
        self.m.optimize();
        self.infer_type_features();
        let ins = io.place(Dir::In);
        let outs = io.place(Dir::Out);
        let prims = io.place(Dir::Prim);

        let mut flags: u64 = self.features.iter().map(|f| f.metadata_bit()).fold(0, |a, b| a | b);
        let mut sfi0: u64 = self.features.iter().map(|f| f.container_bit()).fold(0, |a, b| a | b);
        let rt_vp = outs.iter().chain(&prims).any(|e| matches!(e.e.kind, SemKind::RenderTargetArrayIndex | SemKind::ViewportArrayIndex));
        // DXC sets this for vertex shaders only (not for mesh shaders).
        if rt_vp && self.stage == Stage::Vertex {
            flags |= 1 << 9;
            sfi0 |= 0x2000;
        }
        // 16-bit signature elements: min precision unless native 16-bit types are enabled.
        let native16 = self.features.contains(&Feature::NativeLowPrecision);
        if !native16 && ins.iter().chain(&outs).chain(&prims).any(|e| e.e.ty.is_16bit()) {
            flags |= 1 << 5;
            sfi0 |= 0x10;
        }
        // Bound UAVs outside pixel/compute shaders (DXC: not for mesh shaders,
        // nor for heap-indexed UAVs).
        if self.stage == Stage::Vertex && self.bindings.iter().any(|b| b.class == ResClass::Uav) {
            flags |= 1 << 16;
            sfi0 |= 0x4;
        }
        if self.stage == Stage::Pixel {
            if outs.iter().any(|e| e.e.kind == SemKind::StencilRef) {
                flags |= 1 << 11;
                sfi0 |= 0x200;
            }
            if let StageInfo::Pixel { early_depth_stencil: true } = info {
                flags |= 1 << 3;
            }
        }

        // Dependency tables (from the code, before metadata is added).
        let deps = compute_deps(&self, &ins, &outs, &prims);
        let (n_in, n_out, n_prim) = (components(&ins), components(&outs), components(&prims));

        let stage = self.stage;
        let sm = self.sm;
        let m = &mut self.m;
        let one = m.md_i32(1);
        let ver = m.md_i32(sm.1);
        let dx_version = m.md_node(&[Some(one), Some(ver)]);
        let valver = m.md_i32(super::validate::VALVER.1);
        let dx_valver = m.md_node(&[Some(one), Some(valver)]);
        let prefix = m.md_str(stage.profile_prefix());
        let smaj = m.md_i32(sm.0);
        let shader_model = m.md_node(&[Some(prefix), Some(smaj), Some(ver)]);

        // dx.viewIdState: [inputs, outputs, (prim outputs), tables…].
        let mut vis = vec![n_in, n_out];
        if stage == Stage::Mesh {
            vis.push(n_prim);
        }
        if n_in > 0 && n_out > 0 {
            vis.extend(dep_table(&deps, 0, n_in, n_out));
        }
        let i32 = m.i32();
        let at = m.array(vis.len() as u64, i32);
        let elems: Vec<Value> = vis.iter().map(|&v| m.i32c(v)).collect();
        let arr = m.aggregate(at, &elems);
        let arr_md = m.md_value(arr);
        let view_id = m.md_node(&[Some(arr_md)]);

        // Signature metadata.
        let sig_list = |m: &mut super::module::Module, p: &[Placed]| -> Option<Md> {
            if p.is_empty() {
                return None;
            }
            let nodes: Vec<Option<Md>> = p
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let id = m.md_i32(i as u32);
                    let name = m.md_str(&e.e.name);
                    let ct = m.i8c(e.e.ty.md());
                    let ct = m.md_value(ct);
                    let k = m.i8c(e.e.kind as u8);
                    let k = m.md_value(k);
                    let idx: Vec<Option<Md>> = e.e.indices.iter().map(|&x| Some(m.md_i32(x))).collect();
                    let idx = m.md_node(&idx);
                    let ip = m.i8c(e.interp as u8);
                    let ip = m.md_value(ip);
                    let rows = m.md_i32(e.e.nrows());
                    let cols = m.i8c(e.e.cols);
                    let cols = m.md_value(cols);
                    let (r, c) = e.at.map_or((u32::MAX, 0xFF), |(r, c)| (r, c));
                    let r = m.md_i32(r);
                    let c = m.i8c(c);
                    let c = m.md_value(c);
                    let ext = if e.usage != 0 {
                        let t = m.md_i32(3);
                        let u = m.md_i32(e.usage as u32);
                        Some(m.md_node(&[Some(t), Some(u)]))
                    } else {
                        None
                    };
                    Some(m.md_node(&[Some(id), Some(name), Some(ct), Some(k), Some(idx), Some(ip), Some(rows), Some(cols), Some(r), Some(c), ext]))
                })
                .collect();
            Some(m.md_node(&nodes))
        };
        let in_md = sig_list(m, &ins);
        let out_md = sig_list(m, &outs);
        let prim_md = sig_list(m, &prims);
        let sigs = if in_md.is_none() && out_md.is_none() && prim_md.is_none() { None } else { Some(m.md_node(&[in_md, out_md, prim_md])) };

        let (resources, psv_res) = self.resource_metadata();
        let m = &mut self.m;

        // Entry properties.
        let mut props: Vec<Option<Md>> = Vec::new();
        if flags != 0 {
            let tag = m.md_i32(0);
            let v = m.md_i64(flags);
            props.extend([Some(tag), Some(v)]);
        }
        let mut numthreads = [0u32; 3];
        if let StageInfo::Mesh { numthreads: nt, max_vertices, max_primitives, topology, payload_bytes, .. } = info {
            numthreads = nt;
            let tag = m.md_i32(9);
            let ntv: Vec<Option<Md>> = nt.iter().map(|&n| Some(m.md_i32(n))).collect();
            let ntn = m.md_node(&ntv);
            let (mv, mp, tp, pb) = (m.md_i32(max_vertices), m.md_i32(max_primitives), m.md_i32(topology as u32), m.md_i32(payload_bytes));
            let node = m.md_node(&[Some(ntn), Some(mv), Some(mp), Some(tp), Some(pb)]);
            props.extend([Some(tag), Some(node)]);
        }
        let props_node = if props.is_empty() { None } else { Some(m.md_node(&props)) };
        let fv = m.func_value(self.entry);
        let fmd = m.md_value(fv);
        let name = m.md_str("main");
        let entry = m.md_node(&[Some(fmd), Some(name), sigs, resources, props_node]);

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
        m.named_md("dx.viewIdState", &[view_id]);
        m.named_md("dx.entryPoints", &[entry]);
        let bitcode = m.try_to_bitcode(TRIPLE, if native16 { DATALAYOUT_NATIVE16 } else { DATALAYOUT })?;

        // PSV0.
        let (iv, ov, pv) = (vectors(&ins), vectors(&outs), vectors(&prims));
        let mut stage_info = [0u8; 16];
        let mut info1 = [0u8; 2];
        match info {
            StageInfo::Vertex => stage_info[0] = outs.iter().any(|e| e.e.kind == SemKind::Position) as u8,
            StageInfo::Pixel { .. } => {
                stage_info[0] = outs.iter().any(|e| matches!(e.e.kind, SemKind::Depth | SemKind::DepthLessEqual | SemKind::DepthGreaterEqual)) as u8;
                stage_info[1] = (io.uses_sample_index
                    || ins.iter().any(|e| e.e.kind == SemKind::SampleIndex || matches!(e.interp, Interp::LinearSample | Interp::LinearNoperspectiveSample)))
                    as u8;
            }
            StageInfo::Mesh { max_vertices, max_primitives, topology, payload_bytes, .. } => {
                let groupshared_bytes = self.m.global_bytes_layout(3, native16) as u32;
                stage_info[0..4].copy_from_slice(&groupshared_bytes.to_le_bytes());
                stage_info[8..12].copy_from_slice(&payload_bytes.to_le_bytes());
                stage_info[12..14].copy_from_slice(&(max_vertices as u16).to_le_bytes());
                stage_info[14..16].copy_from_slice(&(max_primitives as u16).to_le_bytes());
                info1 = [pv as u8, topology as u8];
            }
        }
        let mut tables = Vec::new();
        if iv > 0 && ov > 0 {
            tables.push(dep_table(&deps, 0, iv * 4, ov * 4));
        }
        let psv = PsvFull {
            stage,
            stage_info,
            min_wave: 0,
            max_wave: u32::MAX,
            uses_view_id: false,
            info1,
            numthreads,
            resources: psv_res,
            entry: "main".into(),
            inputs: psv_elems(&ins),
            outputs: psv_elems(&outs),
            prims: psv_elems(&prims),
            input_vectors: iv as u8,
            output_vectors: [ov as u8, 0, 0, 0],
            tables,
        };
        let psg = if stage == Stage::Mesh { Some(sig_part(&prims, true, native16)) } else { None };
        let (isg, osg) = (sig_part(&ins, false, native16), sig_part(&outs, true, native16));
        Ok(container::container_with_signatures(stage, sm, &bitcode, sfi0, isg, osg, psg, container::psv_part_full(&psv)))
    }
}

#[cfg(test)]
#[path = "graphics_tests.rs"]
pub(in crate::gpu::dxil) mod tests;
