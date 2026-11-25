//! Graphics-stage checks: container parts byte for byte against DXC 1.9
//! output for the HLSL programs in `tests/data/dxil/dxc_{vs,ps,ms}_*.hlsl`, the
//! validator on every emitted shader, and draws on the GPU.

use super::*;
use crate::gpu::dxil::module::{BinOp, CastOp, Pred};
use crate::gpu::dxil::validate;

pub(super) fn part<'a>(c: &'a [u8], cc: &[u8; 4]) -> Option<&'a [u8]> {
    let n = u32::from_le_bytes(c[28..32].try_into().unwrap()) as usize;
    (0..n).find_map(|i| {
        let off = u32::from_le_bytes(c[32 + 4 * i..36 + 4 * i].try_into().unwrap()) as usize;
        (&c[off..off + 4] == cc).then(|| {
            let len = u32::from_le_bytes(c[off + 4..off + 8].try_into().unwrap()) as usize;
            &c[off + 8..off + 8 + len]
        })
    })
}

fn hex(b: &[u8]) -> String {
    b.chunks(16).enumerate().map(|(i, c)| format!("{:04x}: {}\n", i * 16, c.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" "))).collect()
}

/// Asserts the signature and PSV parts match DXC's.
fn assert_parts_match(ours: &[u8], dxc: &[u8]) {
    for cc in [b"ISG1", b"OSG1", b"PSG1", b"PSV0"] {
        let (a, b) = (part(ours, cc), part(dxc, cc));
        assert_eq!(a.is_some(), b.is_some(), "{} presence", String::from_utf8_lossy(cc));
        if let (Some(a), Some(b)) = (a, b) {
            assert!(a == b, "{} differs\n--- ours\n{}--- dxc\n{}", String::from_utf8_lossy(cc), hex(a), hex(b));
        }
    }
}

pub(super) fn assert_valid(c: &[u8]) {
    if let Err(e) = validate::validate(c) {
        let dis = validate::disassemble(c).unwrap_or_else(|d| d);
        panic!("validator rejected the shader:\n{e}\n--- disassembly ---\n{dis}");
    }
}

/// `tests/data/dxil/dxc_vs_basic.hlsl`.
fn vs_basic() -> Vec<u8> {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let pos = io.input(SigElement::user("POSITION", 0, CompType::F32, 3));
    let col = io.input(SigElement::user("COLOR", 0, CompType::F32, 4));
    let vid = io.input(SigElement::new("SV_VertexID", 0, SemKind::VertexId, CompType::U32, 1));
    let opos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let ocol = io.output(SigElement::user("COLOR", 0, CompType::F32, 4));
    let otc = io.output(SigElement::user("TEXCOORD", 1, CompType::F32, 2));
    let v = io.load_input(&mut s, vid, 0, 0);
    let c: Vec<Value> = (0..4).map(|i| io.load_input(&mut s, col, 0, i)).collect();
    let p: Vec<Value> = (0..3).map(|i| io.load_input(&mut s, pos, 0, i)).collect();
    let f32 = s.m.f32();
    let vf = s.m.cast(CastOp::UIToFP, v, f32);
    let one = s.m.f32c(1.0f32.to_bits());
    let zero = s.m.f32c(0);
    for (i, &x) in p.iter().chain([&one]).enumerate() {
        io.store_output(&mut s, opos, 0, i as u8, x);
    }
    for (i, &x) in c.iter().enumerate() {
        io.store_output(&mut s, ocol, 0, i as u8, x);
    }
    io.store_output(&mut s, otc, 0, 0, vf);
    io.store_output(&mut s, otc, 0, 1, zero);
    s.finish_graphics(io, StageInfo::Vertex)
}

/// `tests/data/dxil/dxc_ps_basic.hlsl`.
fn ps_basic() -> Vec<u8> {
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let _pos = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let col = io.input(SigElement::user("COLOR", 0, CompType::F32, 4));
    let tc = io.input(SigElement::user("TEXCOORD", 1, CompType::F32, 2));
    let tgt = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F32, 4));
    let uv: Vec<Value> = (0..2).map(|i| io.load_input(&mut s, tc, 0, i)).collect();
    let c: Vec<Value> = (0..4).map(|i| io.load_input(&mut s, col, 0, i)).collect();
    for i in 0..4 {
        let r = s.m.fbin(BinOpF::Add.into(), c[i], uv[i % 2], 0);
        io.store_output(&mut s, tgt, 0, i as u8, r);
    }
    s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false })
}

/// `BinOp` for float adds (readability).
enum BinOpF {
    Add,
}
impl From<BinOpF> for crate::gpu::dxil::module::BinOp {
    fn from(_: BinOpF) -> Self {
        crate::gpu::dxil::module::BinOp::Add
    }
}

const DXC_VS_BASIC: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_vs_basic.dxil");
const DXC_PS_BASIC: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ps_basic.dxil");

#[test]
fn vs_basic_parts_match_dxc() {
    assert_parts_match(&vs_basic(), DXC_VS_BASIC);
}

#[test]
fn vs_basic_validates() {
    assert_valid(&vs_basic());
}

#[test]
fn ps_basic_parts_match_dxc() {
    assert_parts_match(&ps_basic(), DXC_PS_BASIC);
}

#[test]
fn ps_basic_validates() {
    assert_valid(&ps_basic());
}


fn f(s: &mut Shader, v: f32) -> Value {
    s.m.f32c(v.to_bits())
}

/// `tests/data/dxil/dxc_ps_modes.hlsl`: interpolation modes, system values,
/// discard, coverage, two targets and depth.
fn ps_modes() -> Vec<u8> {
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let pos = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let mode = io.input(SigElement::user("MODE", 0, CompType::U32, 1));
    let c = io.input(SigElement::user("TEXCOORD", 0, CompType::F32, 2).interp(Interp::LinearCentroid));
    let t = io.input(SigElement::user("TEXCOORD", 1, CompType::F32, 1).interp(Interp::LinearNoperspective));
    let smp = io.input(SigElement::user("TEXCOORD", 2, CompType::F32, 3).interp(Interp::LinearSample));
    let ff = io.input(SigElement::new("SV_IsFrontFace", 0, SemKind::IsFrontFace, CompType::U32, 1));
    let pid = io.input(SigElement::new("SV_PrimitiveID", 0, SemKind::PrimitiveId, CompType::U32, 1));
    let q = io.input(SigElement::user("Q", 0, CompType::F32, 1).interp(Interp::LinearNoperspectiveCentroid));
    let ta = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F32, 4));
    let tb = io.output(SigElement::new("SV_Target", 1, SemKind::Target, CompType::F32, 4));
    let depth = io.output(SigElement::new("SV_Depth", 0, SemKind::Depth, CompType::F32, 1));

    let qv = io.load_input(&mut s, q, 0, 0);
    let pv = io.load_input(&mut s, pid, 0, 0);
    let fv = io.load_input(&mut s, ff, 0, 0);
    let zero_i = s.m.i32c(0);
    let front = s.m.cmp(Pred::Ne, fv, zero_i);
    let sv: Vec<Value> = (0..3).map(|i| io.load_input(&mut s, smp, 0, i)).collect();
    let tv = io.load_input(&mut s, t, 0, 0);
    let cv: Vec<Value> = (0..2).map(|i| io.load_input(&mut s, c, 0, i)).collect();
    let mv = io.load_input(&mut s, mode, 0, 0);
    let pz = io.load_input(&mut s, pos, 0, 2);
    let cov = io.coverage(&mut s);
    let three = s.m.i32c(3);
    let is3 = s.m.cmp(Pred::Eq, mv, three);
    let kill = s.m.new_block();
    let join = s.m.new_block();
    s.m.cond_br(is3, kill, join);
    s.m.set_block(kill);
    let t_ = s.m.i1c(true);
    io.discard(&mut s, t_);
    s.m.br(join);
    s.m.set_block(join);
    let (one, zero) = (f(&mut s, 1.0), f(&mut s, 0.0));
    let sel = s.m.select(front, one, zero);
    let sum = s.m.bin(BinOp::Add, cov, pv);
    let f32 = s.m.f32();
    let sumf = s.m.cast(CastOp::UIToFP, sum, f32);
    let w = s.m.fbin(BinOp::Add, sumf, qv, 0);
    for (i, &x) in [cv[0], cv[1], tv, sv[0]].iter().enumerate() {
        io.store_output(&mut s, ta, 0, i as u8, x);
    }
    for (i, &x) in [sv[1], sv[2], sel, w].iter().enumerate() {
        io.store_output(&mut s, tb, 0, i as u8, x);
    }
    io.store_output(&mut s, depth, 0, 0, pz);
    s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false })
}

/// `tests/data/dxil/dxc_ps_packing.hlsl`: row sharing between user semantics and
/// system values, mixed types, unpacked `SV_SampleIndex`, unused inputs.
fn ps_packing() -> Vec<u8> {
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let a = io.input(SigElement::user("A", 0, CompType::U32, 1));
    let ff = io.input(SigElement::new("SV_IsFrontFace", 0, SemKind::IsFrontFace, CompType::U32, 1));
    let b = io.input(SigElement::user("B", 0, CompType::U32, 1));
    let c = io.input(SigElement::user("C", 0, CompType::F32, 1));
    let d = io.input(SigElement::user("D", 0, CompType::F32, 2));
    let e = io.input(SigElement::user("E", 0, CompType::F32, 1).interp(Interp::Constant));
    let si = io.input(SigElement::new("SV_SampleIndex", 0, SemKind::SampleIndex, CompType::U32, 1));
    let p = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let h = io.input(SigElement::user("H", 0, CompType::I32, 1));
    let _rest = io.input(SigElement::user("REST", 0, CompType::U32, 4));
    let tgt = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F32, 4));
    let f32 = s.m.f32();
    let mut acc = f(&mut s, 0.0);
    for (sig, int) in [(a, true), (ff, true), (b, true), (c, false), (d, false), (e, false), (si, true), (p, false), (h, true)] {
        let v = io.load_input(&mut s, sig, 0, 0);
        let v = if int { s.m.cast(CastOp::UIToFP, v, f32) } else { v };
        acc = s.m.fbin(BinOp::Add, acc, v, 0);
    }
    for i in 0..4 {
        io.store_output(&mut s, tgt, 0, i, acc);
    }
    s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false })
}

/// `tests/data/dxil/dxc_vs_packing.hlsl`: output packing with system values,
/// a two-row matrix, `SV_RenderTargetArrayIndex`, `SV_InstanceID`.
fn vs_packing() -> Vec<u8> {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let vid = io.input(SigElement::new("SV_VertexID", 0, SemKind::VertexId, CompType::U32, 1));
    let iid = io.input(SigElement::new("SV_InstanceID", 0, SemKind::InstanceId, CompType::U32, 1));
    let oa = io.output(SigElement::user("A", 0, CompType::F32, 2));
    let ob = io.output(SigElement::user("B", 0, CompType::F32, 2));
    let oc = io.output(SigElement::user("C", 0, CompType::F32, 1));
    let od = io.output(SigElement::user("D", 0, CompType::U32, 1));
    let op = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let oe = io.output(SigElement::user("E", 0, CompType::F32, 1));
    let ort = io.output(SigElement::new("SV_RenderTargetArrayIndex", 0, SemKind::RenderTargetArrayIndex, CompType::U32, 1));
    let of = io.output(SigElement::user("F", 0, CompType::U32, 2));
    let og = io.output(SigElement::user("G", 0, CompType::F32, 3));
    let om = io.output(SigElement::user("M", 0, CompType::F32, 2).rows(2));
    let oi = io.output(SigElement::user("I", 0, CompType::U32, 1));
    let v = io.load_input(&mut s, vid, 0, 0);
    let i = io.load_input(&mut s, iid, 0, 0);
    let f32 = s.m.f32();
    let vf = s.m.cast(CastOp::UIToFP, v, f32);
    for c in 0..2 {
        io.store_output(&mut s, oa, 0, c, vf);
        let one = f(&mut s, 1.0);
        io.store_output(&mut s, ob, 0, c, one);
    }
    let two = f(&mut s, 2.0);
    io.store_output(&mut s, oc, 0, 0, two);
    io.store_output(&mut s, od, 0, 0, i);
    for c in 0..4 {
        let z = f(&mut s, 0.0);
        io.store_output(&mut s, op, 0, c, z);
    }
    let three = f(&mut s, 3.0);
    io.store_output(&mut s, oe, 0, 0, three);
    io.store_output(&mut s, ort, 0, 0, v);
    io.store_output(&mut s, of, 0, 0, i);
    io.store_output(&mut s, of, 0, 1, i);
    for c in 0..3 {
        let x = f(&mut s, 4.0);
        io.store_output(&mut s, og, 0, c, x);
    }
    let (m1, m2, m3) = (f(&mut s, 1.0), f(&mut s, 2.0), f(&mut s, 3.0));
    io.store_output(&mut s, om, 0, 0, m1);
    io.store_output(&mut s, om, 0, 1, m2);
    io.store_output(&mut s, om, 1, 0, m3);
    io.store_output(&mut s, om, 1, 1, vf);
    let five = s.m.i32c(5);
    io.store_output(&mut s, oi, 0, 0, five);
    s.finish_graphics(io, StageInfo::Vertex)
}

/// `tests/data/dxil/dxc_ms_basic.hlsl`: vertex and primitive outputs.
fn ms_basic() -> Vec<u8> {
    let mut s = Shader::new(Stage::Mesh, (6, 6));
    let mut io = GraphicsIo::new(Stage::Mesh);
    let pos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let col = io.output(SigElement::user("COLOR", 0, CompType::F32, 4));
    let pid = io.primitive(SigElement::new("SV_PrimitiveID", 0, SemKind::PrimitiveId, CompType::U32, 1));
    let extra = io.primitive(SigElement::user("EXTRA", 0, CompType::F32, 2).interp(Interp::Constant));
    let tid = s.thread_id_in_group(0);
    let (three, one) = (s.m.i32c(3), s.m.i32c(1));
    io.set_mesh_output_counts(&mut s, three, one);
    let lt = s.m.cmp(Pred::ULt, tid, three);
    let (bv, test, bp, end) = (s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block());
    s.m.cond_br(lt, bv, test);
    s.m.set_block(bv);
    let f32 = s.m.f32();
    let x = s.m.cast(CastOp::UIToFP, tid, f32);
    let (z, o) = (f(&mut s, 0.0), f(&mut s, 1.0));
    for (c, &v) in [x, z, z, o].iter().enumerate() {
        io.store_vertex_output(&mut s, pos, 0, c as u8, v, tid);
    }
    for (c, &v) in [o, z, z, o].iter().enumerate() {
        io.store_vertex_output(&mut s, col, 0, c as u8, v, tid);
    }
    s.m.br(test);
    s.m.set_block(test);
    let zero = s.m.i32c(0);
    let first = s.m.cmp(Pred::Eq, tid, zero);
    s.m.cond_br(first, bp, end);
    s.m.set_block(bp);
    let two = s.m.i32c(2);
    io.emit_indices(&mut s, zero, &[zero, one, two]);
    let seven = s.m.i32c(7);
    io.store_primitive_output(&mut s, pid, 0, 0, seven, zero);
    let f2 = f(&mut s, 2.0);
    io.store_primitive_output(&mut s, extra, 0, 0, o, zero);
    io.store_primitive_output(&mut s, extra, 0, 1, f2, zero);
    s.m.br(end);
    s.m.set_block(end);
    let info = StageInfo::Mesh { numthreads: [32, 1, 1], max_vertices: 3, max_primitives: 1, topology: Topology::Triangle, payload_bytes: 0 };
    s.finish_graphics(io, info)
}

const DXC_PS_MODES: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ps_modes.dxil");
const DXC_PS_PACKING: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ps_packing.dxil");
const DXC_VS_PACKING: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_vs_packing.dxil");
const DXC_MS_BASIC: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ms_basic.dxil");

#[test]
fn ps_modes_parts_match_dxc() {
    let c = ps_modes();
    assert_parts_match(&c, DXC_PS_MODES);
    assert_eq!(part(&c, b"SFI0"), part(DXC_PS_MODES, b"SFI0"));
    assert_valid(&c);
}

#[test]
fn ps_packing_parts_match_dxc() {
    let c = ps_packing();
    assert_parts_match(&c, DXC_PS_PACKING);
    assert_valid(&c);
}

#[test]
fn vs_packing_parts_match_dxc() {
    let c = vs_packing();
    assert_parts_match(&c, DXC_VS_PACKING);
    assert_eq!(part(&c, b"SFI0"), part(DXC_VS_PACKING, b"SFI0"));
    assert_valid(&c);
}

#[test]
fn ms_basic_parts_match_dxc() {
    let c = ms_basic();
    assert_parts_match(&c, DXC_MS_BASIC);
    assert_valid(&c);
}

// ---- GPU draws

/// The draw test's pixel shader: `SV_Target = float4(COLOR.rgb * mem[0..3], float(ID) + mem[3])`
/// where `ID` is a `nointerpolation uint` (the provoking vertex's id) and
/// `mem` the raw UAV `u2` the draw benches bind to their flat memory.
fn draw_ps() -> Vec<u8> {
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let mem = s.raw_uav(2, 0);
    let _pos = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let col = io.input(SigElement::user("COLOR", 0, CompType::F32, 4));
    let id = io.input(SigElement::user("ID", 0, CompType::U32, 1));
    let tgt = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F32, 4));
    let zero = s.m.i32c(0);
    let k = s.raw_load(mem, zero, 0xF);
    let f32 = s.m.f32();
    for c in 0..3 {
        let x = io.load_input(&mut s, col, 0, c);
        let kc = s.m.bitcast(k[c as usize], f32);
        let r = s.m.fbin(BinOp::Mul, x, kc, 0);
        io.store_output(&mut s, tgt, 0, c, r);
    }
    let i = io.load_input(&mut s, id, 0, 0);
    let fi = s.m.cast(CastOp::UIToFP, i, f32);
    let k3 = s.m.bitcast(k[3], f32);
    let w = s.m.fbin(BinOp::Add, fi, k3, 0);
    io.store_output(&mut s, tgt, 0, 3, w);
    s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false })
}

/// The draw test's vertex shader: `POSITION.xy` (z = 0, w = 1), `COLOR`
/// passed through, `ID = SV_VertexID + 5`.
fn draw_vs() -> Vec<u8> {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let pos = io.input(SigElement::user("POSITION", 0, CompType::F32, 2));
    let col = io.input(SigElement::user("COLOR", 0, CompType::F32, 4));
    let vid = io.input(SigElement::new("SV_VertexID", 0, SemKind::VertexId, CompType::U32, 1));
    let opos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let ocol = io.output(SigElement::user("COLOR", 0, CompType::F32, 4));
    let oid = io.output(SigElement::user("ID", 0, CompType::U32, 1));
    let x = io.load_input(&mut s, pos, 0, 0);
    let y = io.load_input(&mut s, pos, 0, 1);
    let (z, o) = (f(&mut s, 0.0), f(&mut s, 1.0));
    for (c, &v) in [x, y, z, o].iter().enumerate() {
        io.store_output(&mut s, opos, 0, c as u8, v);
    }
    for c in 0..4 {
        let v = io.load_input(&mut s, col, 0, c);
        io.store_output(&mut s, ocol, 0, c, v);
    }
    let v = io.load_input(&mut s, vid, 0, 0);
    let five = s.m.i32c(5);
    let id = s.m.bin(BinOp::Add, v, five);
    io.store_output(&mut s, oid, 0, 0, id);
    s.finish_graphics(io, StageInfo::Vertex)
}

const SIZE: u32 = 8;
const CLEAR: [f32; 4] = [0.0, 0.0, 0.0, 0.5];
const MEM: [f32; 4] = [2.0, 0.5, 1.0, 0.25];
const COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// Pixel (x, y) is covered by the triangle (-1,-1), (-1,1.1), (1.1,-1) iff x <= y.
fn expect_triangle(px: &[[f32; 4]], inside: [f32; 4]) {
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let want = if x <= y { inside } else { CLEAR };
            assert_eq!(px[y * SIZE as usize + x], want, "pixel ({x}, {y})");
        }
    }
}

/// Whether the runner's VS/PS draw path can run here. It is exercised upstream only where
/// wave64 programs run (`recomp::tests::gpu_ready`); on an adapter with neither 32- nor
/// 64-lane waves (WARP / the Basic Render Driver on CI) even a DXC-compiled VS/PS draw
/// removes the device, which would then fail every later GPU test in the process.
pub(in crate::gpu::dxil) fn draw_ready() -> bool {
    use crate::gpu::recomp::runner;
    match runner::wave64_programs_supported() {
        Ok(true) => true,
        Ok(false) => {
            runner::skip_test("no wave64");
            false
        }
        Err(_) => {
            runner::skip_test("no GPU");
            false
        }
    }
}

#[test]
fn emitted_vs_ps_draw_a_triangle() {
    use crate::gpu::recomp::runner;
    use windows::Win32::Graphics::Dxgi::Common::*;
    if !draw_ready() {
        return;
    }
    let (vs, ps) = (draw_vs(), draw_ps());
    assert_valid(&vs);
    assert_valid(&ps);
    let tri = [(-1.0f32, -1.0f32), (-1.0, 1.1), (1.1, -1.0)];
    let verts: Vec<f32> = tri.iter().flat_map(|&(x, y)| [x, y].into_iter().chain(COLOR)).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    let mem: Vec<u32> = MEM.iter().map(|f| f.to_bits()).collect();
    let px = match runner::draw(&vs, &ps, &layout, &verts, 24, &[], &mem, SIZE, CLEAR) {
        Ok(p) => p,
        Err(e) if e.contains("no D3D12") || e.contains("adapter") => return runner::skip_test(&e),
        Err(e) => panic!("draw: {e}"),
    };
    // Provoking vertex 0 → ID 5.
    expect_triangle(&px, [COLOR[0] * MEM[0], COLOR[1] * MEM[1], COLOR[2] * MEM[2], 5.0 + MEM[3]]);
}

/// The mesh test's mesh shader: 3 threads, thread `t` writes vertex `t` of
/// the draw test's triangle (`ID = t + 5`), thread 0 the indices and counts.
fn draw_ms() -> Vec<u8> {
    let mut s = Shader::new(Stage::Mesh, (6, 6));
    let mut io = GraphicsIo::new(Stage::Mesh);
    let opos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let ocol = io.output(SigElement::user("COLOR", 0, CompType::F32, 4));
    let oid = io.output(SigElement::user("ID", 0, CompType::U32, 1));
    let tid = s.thread_id_in_group(0);
    let (zero, one, two, three) = (s.m.i32c(0), s.m.i32c(1), s.m.i32c(2), s.m.i32c(3));
    io.set_mesh_output_counts(&mut s, three, one);
    // x = (t == 2) ? 1.1 : -1;  y = (t == 1) ? 1.1 : -1
    let (m1, big) = (f(&mut s, -1.0), f(&mut s, 1.1));
    let is2 = s.m.cmp(Pred::Eq, tid, two);
    let is1 = s.m.cmp(Pred::Eq, tid, one);
    let x = s.m.select(is2, big, m1);
    let y = s.m.select(is1, big, m1);
    let (z, o) = (f(&mut s, 0.0), f(&mut s, 1.0));
    for (c, &v) in [x, y, z, o].iter().enumerate() {
        io.store_vertex_output(&mut s, opos, 0, c as u8, v, tid);
    }
    for (c, &v) in COLOR.iter().enumerate() {
        let k = f(&mut s, v);
        io.store_vertex_output(&mut s, ocol, 0, c as u8, k, tid);
    }
    let five = s.m.i32c(5);
    let id = s.m.bin(BinOp::Add, tid, five);
    io.store_vertex_output(&mut s, oid, 0, 0, id, tid);
    let first = s.m.cmp(Pred::Eq, tid, zero);
    let (bp, end) = (s.m.new_block(), s.m.new_block());
    s.m.cond_br(first, bp, end);
    s.m.set_block(bp);
    io.emit_indices(&mut s, zero, &[zero, one, two]);
    s.m.br(end);
    s.m.set_block(end);
    let info = StageInfo::Mesh { numthreads: [3, 1, 1], max_vertices: 3, max_primitives: 1, topology: Topology::Triangle, payload_bytes: 0 };
    s.finish_graphics(io, info)
}

/// An empty compute pass for the two-pass mesh bench.
fn empty_cs() -> Vec<u8> {
    Shader::new(Stage::Compute, (6, 6)).finish([1, 1, 1], None)
}

#[test]
fn emitted_ms_ps_draw_a_triangle() {
    use crate::gpu::recomp::runner;
    match runner::mesh_shaders_supported() {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no mesh shaders"),
        Err(e) => return runner::skip_test(&e),
    }
    let (cs, ms, ps) = (empty_cs(), draw_ms(), draw_ps());
    assert_valid(&ms);
    let mem: Vec<u32> = MEM.iter().map(|f| f.to_bits()).collect();
    let (px, _) = runner::mesh_draw(&cs, &ms, &ps, 1, 4, &[], &mem, SIZE, CLEAR).expect("mesh draw");
    expect_triangle(&px, [COLOR[0] * MEM[0], COLOR[1] * MEM[1], COLOR[2] * MEM[2], 5.0 + MEM[3]]);
}

/// `tests/data/dxil/dxc_ms_payload.hlsl`: payload, groupshared memory, line
/// topology, `SV_CullPrimitive` (unpacked bool), `SV_RenderTargetArrayIndex`
/// sharing a primitive row.
fn ms_payload() -> Vec<u8> {
    let mut s = Shader::new(Stage::Mesh, (6, 6));
    let mut io = GraphicsIo::new(Stage::Mesh);
    let pos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let k = io.primitive(SigElement::user("PRIMK", 0, CompType::U32, 1));
    let cull = io.primitive(SigElement::new("SV_CullPrimitive", 0, SemKind::CullPrimitive, CompType::Bool, 1));
    let rt = io.primitive(SigElement::new("SV_RenderTargetArrayIndex", 0, SemKind::RenderTargetArrayIndex, CompType::U32, 1));
    let (i32, f32) = (s.m.i32(), s.m.f32());
    let arr3 = s.m.array(3, f32);
    let pay_ty = s.m.named_struct("struct.Pay", &[i32, arr3]);
    let gs_ty = s.m.array(16, i32);
    let gs = s.m.global("gs", gs_ty, 3, None, false, 4);

    let tid = s.flattened_thread_id_in_group();
    let pay = io.mesh_payload(&mut s, pay_ty, "struct.Pay");
    let (c0, c1, c2, c3, c4, c15) = (s.m.i32c(0), s.m.i32c(1), s.m.i32c(2), s.m.i32c(3), s.m.i32c(4), s.m.i32c(15));
    let pa = s.m.gep_fields(pay, &[c0, c0]);
    let a = s.m.load(pa, 4);
    let slot = s.m.bin(BinOp::And, tid, c15);
    let p = s.m.gep(gs, &[c0, slot]);
    s.m.store(p, a, 4);
    let void = s.m.void();
    let mode = s.m.i32c(9);
    s.op(80, "barrier", None, void, &[mode], FnAttrs::NoDuplicate);
    io.set_mesh_output_counts(&mut s, c4, c2);
    let lt4 = s.m.cmp(Pred::ULt, tid, c4);
    let (bv, test, bp, end) = (s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block());
    s.m.cond_br(lt4, bv, end);
    s.m.set_block(bv);
    let t1 = s.m.bin(BinOp::Add, tid, c1);
    let t1m = s.m.bin(BinOp::And, t1, c15);
    let q = s.m.gep(gs, &[c0, t1m]);
    let g = s.m.load(q, 4);
    let gf = s.m.cast(CastOp::UIToFP, g, f32);
    let r = s.m.bin(BinOp::URem, tid, c3);
    let pb = s.m.gep_fields(pay, &[c0, c1, r]);
    let b = s.m.load(pb, 4);
    let (z, o) = (f(&mut s, 0.0), f(&mut s, 1.0));
    for (c, &v) in [gf, b, z, o].iter().enumerate() {
        io.store_vertex_output(&mut s, pos, 0, c as u8, v, tid);
    }
    let lt2 = s.m.cmp(Pred::ULt, tid, c2);
    s.m.cond_br(lt2, bp, end);
    s.m.set_block(bp);
    io.emit_indices(&mut s, tid, &[tid, t1]);
    io.store_primitive_output(&mut s, k, 0, 0, tid, tid);
    io.store_primitive_output(&mut s, cull, 0, 0, c0, tid);
    io.store_primitive_output(&mut s, rt, 0, 0, c1, tid);
    s.m.br(end);
    s.m.set_block(test);
    s.m.br(end);
    s.m.set_block(end);
    let info = StageInfo::Mesh { numthreads: [16, 2, 1], max_vertices: 4, max_primitives: 2, topology: Topology::Line, payload_bytes: 16 };
    s.finish_graphics(io, info)
}

/// `tests/data/dxil/dxc_ps_outputs.hlsl`: `SV_Target2`, coverage / depth-GE /
/// stencil outputs, centroid position, a primitive attribute and the render
/// target index as inputs.
fn ps_outputs() -> Vec<u8> {
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let p = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4).interp(Interp::LinearNoperspectiveCentroid));
    let k = io.input(SigElement::user("PRIMK", 0, CompType::U32, 1));
    let rt = io.input(SigElement::new("SV_RenderTargetArrayIndex", 0, SemKind::RenderTargetArrayIndex, CompType::U32, 1));
    let c = io.output(SigElement::new("SV_Target", 2, SemKind::Target, CompType::F32, 4));
    let cov = io.output(SigElement::new("SV_Coverage", 0, SemKind::Coverage, CompType::U32, 1));
    let d = io.output(SigElement::new("SV_DepthGreaterEqual", 0, SemKind::DepthGreaterEqual, CompType::F32, 1));
    let st = io.output(SigElement::new("SV_StencilRef", 0, SemKind::StencilRef, CompType::U32, 1));
    let rv = io.load_input(&mut s, rt, 0, 0);
    let kv = io.load_input(&mut s, k, 0, 0);
    let px = io.load_input(&mut s, p, 0, 0);
    let f32 = s.m.f32();
    let kf = s.m.cast(CastOp::UIToFP, kv, f32);
    let rf = s.m.cast(CastOp::UIToFP, rv, f32);
    let four = f(&mut s, 4.0);
    let gt = s.m.cmp(Pred::FOGt, px, four);
    let (two, zero, one, half) = (f(&mut s, 2.0), f(&mut s, 0.0), f(&mut s, 1.0), f(&mut s, 0.5));
    let z = s.m.select(gt, two, zero);
    for (i, &v) in [kf, rf, z, one].iter().enumerate() {
        io.store_output(&mut s, c, 0, i as u8, v);
    }
    let c1 = s.m.i32c(1);
    io.store_output(&mut s, cov, 0, 0, c1);
    io.store_output(&mut s, d, 0, 0, half);
    io.store_output(&mut s, st, 0, 0, kv);
    s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false })
}

/// `tests/data/dxil/dxc_vs_control.hlsl`: outputs that depend on inputs only
/// through branches and a loop (control dependence in the PSV0 tables).
fn vs_control() -> Vec<u8> {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let p = io.input(SigElement::user("POSITION", 0, CompType::F32, 4));
    let c = io.input(SigElement::user("C", 0, CompType::F32, 1));
    let iid = io.input(SigElement::new("SV_InstanceID", 0, SemKind::InstanceId, CompType::U32, 1));
    let opos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let oa = io.output(SigElement::user("A", 0, CompType::F32, 1));
    let ob = io.output(SigElement::user("B", 0, CompType::F32, 1));
    let n = io.load_input(&mut s, iid, 0, 0);
    let cv = io.load_input(&mut s, c, 0, 0);
    let pv: Vec<Value> = (0..4).map(|i| io.load_input(&mut s, p, 0, i)).collect();
    let half = f(&mut s, 0.5);
    let big = s.m.cmp(Pred::FOGt, cv, half);
    let entry = s.m.current_block();
    let (b8, b10, b13, b14, b15, b17) = (s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block());
    s.m.cond_br(big, b8, b10);
    s.m.set_block(b8);
    let two = f(&mut s, 2.0);
    let x2 = s.m.fbin(BinOp::Mul, pv[0], two, 0);
    s.m.br(b10);
    s.m.set_block(b10);
    let f32 = s.m.f32();
    let a = s.m.phi(f32);
    let zero = f(&mut s, 0.0);
    s.m.add_incoming(a, x2, b8);
    s.m.add_incoming(a, zero, entry);
    let c0 = s.m.i32c(0);
    let none = s.m.cmp(Pred::Eq, n, c0);
    s.m.cond_br(none, b15, b13);
    s.m.set_block(b13);
    s.m.br(b17);
    s.m.set_block(b14);
    s.m.br(b15);
    s.m.set_block(b17);
    let i32 = s.m.i32();
    let acc = s.m.phi(f32);
    let i = s.m.phi(i32);
    let next = s.m.fbin(BinOp::Add, acc, pv[1], 0);
    let c1 = s.m.i32c(1);
    let inext = s.m.bin(BinOp::Add, i, c1);
    let done = s.m.cmp(Pred::Eq, inext, n);
    s.m.cond_br(done, b14, b17);
    let one = f(&mut s, 1.0);
    s.m.add_incoming(acc, next, b17);
    s.m.add_incoming(acc, one, b13);
    s.m.add_incoming(i, inext, b17);
    s.m.add_incoming(i, c0, b13);
    s.m.set_block(b15);
    let b = s.m.phi(f32);
    s.m.add_incoming(b, one, b10);
    s.m.add_incoming(b, next, b14);
    for (k, &v) in pv.iter().enumerate() {
        io.store_output(&mut s, opos, 0, k as u8, v);
    }
    io.store_output(&mut s, oa, 0, 0, a);
    io.store_output(&mut s, ob, 0, 0, b);
    s.finish_graphics(io, StageInfo::Vertex)
}


const DXC_MS_PAYLOAD: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ms_payload.dxil");
const DXC_PS_OUTPUTS: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ps_outputs.dxil");
const DXC_VS_CONTROL: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_vs_control.dxil");

#[test]
fn ms_payload_parts_match_dxc() {
    let c = ms_payload();
    assert_parts_match(&c, DXC_MS_PAYLOAD);
    assert_eq!(part(&c, b"SFI0"), part(DXC_MS_PAYLOAD, b"SFI0"));
    assert_valid(&c);
}

#[test]
fn ps_outputs_parts_match_dxc() {
    let c = ps_outputs();
    assert_parts_match(&c, DXC_PS_OUTPUTS);
    assert_eq!(part(&c, b"SFI0"), part(DXC_PS_OUTPUTS, b"SFI0"));
    assert_valid(&c);
}

#[test]
fn vs_control_dependence_matches_dxc() {
    let c = vs_control();
    assert_parts_match(&c, DXC_VS_CONTROL);
    assert_valid(&c);
}

/// A value of a loop depends on what the loop's phi depends on, also when it is first reached
/// through that phi (#1892): `acc = phi [X, entry], [next, loop]`, `next = acc + 1`, and the
/// outputs A = acc, B = next. Asking for `acc` reaches `next` while `acc` is still being worked
/// out; `next` must not keep the empty set it saw then.
#[test]
fn a_loop_value_first_reached_through_its_phi_keeps_the_phis_inputs() {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let x = io.input(SigElement::user("X", 0, CompType::F32, 1));
    let oa = io.output(SigElement::user("A", 0, CompType::F32, 1));
    let ob = io.output(SigElement::user("B", 0, CompType::F32, 1));
    let xv = io.load_input(&mut s, x, 0, 0);
    let entry = s.m.current_block();
    let (lp, exit) = (s.m.new_block(), s.m.new_block());
    s.m.br(lp);
    s.m.set_block(lp);
    let (f32, i32) = (s.m.f32(), s.m.i32());
    let acc = s.m.phi(f32);
    let i = s.m.phi(i32);
    let one = f(&mut s, 1.0);
    let next = s.m.fbin(BinOp::Add, acc, one, 0);
    let (c0, c1, c4) = (s.m.i32c(0), s.m.i32c(1), s.m.i32c(4));
    let inext = s.m.bin(BinOp::Add, i, c1);
    let done = s.m.cmp(Pred::Eq, inext, c4);
    s.m.cond_br(done, exit, lp);
    s.m.add_incoming(acc, xv, entry);
    s.m.add_incoming(acc, next, lp);
    s.m.add_incoming(i, c0, entry);
    s.m.add_incoming(i, inext, lp);
    s.m.set_block(exit);
    io.store_output(&mut s, oa, 0, 0, acc);
    io.store_output(&mut s, ob, 0, 0, next);
    s.m.ret();
    let (ins, outs, prims) = (io.place(Dir::In), io.place(Dir::Out), io.place(Dir::Prim));
    let component = |p: &[Placed], sig: Sig| p[sig.id as usize].at.map(|(r, c)| r * 4 + c as u32).expect("a placed element");
    let deps = compute_deps(&s, &ins, &outs, &prims);
    let x_in = component(&ins, x);
    for (name, out) in [("A", oa), ("B", ob)] {
        let d = deps.out.get(&(0, component(&outs, out))).cloned().unwrap_or_default();
        assert!(d.contains(&x_in), "output {name} depends on input X through the loop; its inputs: {d:?}");
    }
}

/// A pixel shader reading a per-primitive attribute: `SV_Target =
/// float4(COLOR.rgb, PRIMK)`, discarding pixels left of x = 4.
fn prim_ps() -> Vec<u8> {
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let pos = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let col = io.input(SigElement::user("COLOR", 0, CompType::F32, 4));
    let _id = io.input(SigElement::user("ID", 0, CompType::U32, 1));
    let k = io.input(SigElement::user("PRIMK", 0, CompType::U32, 1));
    let tgt = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F32, 4));
    let x = io.load_input(&mut s, pos, 0, 0);
    let four = f(&mut s, 4.0);
    let left = s.m.cmp(Pred::FOLt, x, four);
    io.discard(&mut s, left);
    for c in 0..3 {
        let v = io.load_input(&mut s, col, 0, c);
        io.store_output(&mut s, tgt, 0, c, v);
    }
    let kv = io.load_input(&mut s, k, 0, 0);
    let f32 = s.m.f32();
    let kf = s.m.cast(CastOp::UIToFP, kv, f32);
    io.store_output(&mut s, tgt, 0, 3, kf);
    s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false })
}

/// [`draw_ms`] plus a primitive attribute `PRIMK = 9`.
fn prim_ms() -> Vec<u8> {
    let mut s = Shader::new(Stage::Mesh, (6, 6));
    let mut io = GraphicsIo::new(Stage::Mesh);
    let opos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let ocol = io.output(SigElement::user("COLOR", 0, CompType::F32, 4));
    let oid = io.output(SigElement::user("ID", 0, CompType::U32, 1));
    let ok = io.primitive(SigElement::user("PRIMK", 0, CompType::U32, 1));
    let tid = s.thread_id_in_group(0);
    let (zero, one, two, three) = (s.m.i32c(0), s.m.i32c(1), s.m.i32c(2), s.m.i32c(3));
    io.set_mesh_output_counts(&mut s, three, one);
    let (m1, big) = (f(&mut s, -1.0), f(&mut s, 1.1));
    let is2 = s.m.cmp(Pred::Eq, tid, two);
    let is1 = s.m.cmp(Pred::Eq, tid, one);
    let x = s.m.select(is2, big, m1);
    let y = s.m.select(is1, big, m1);
    let (z, o) = (f(&mut s, 0.0), f(&mut s, 1.0));
    for (c, &v) in [x, y, z, o].iter().enumerate() {
        io.store_vertex_output(&mut s, opos, 0, c as u8, v, tid);
    }
    for (c, &v) in COLOR.iter().enumerate() {
        let k = f(&mut s, v);
        io.store_vertex_output(&mut s, ocol, 0, c as u8, k, tid);
    }
    io.store_vertex_output(&mut s, oid, 0, 0, tid, tid);
    let first = s.m.cmp(Pred::Eq, tid, zero);
    let (bp, end) = (s.m.new_block(), s.m.new_block());
    s.m.cond_br(first, bp, end);
    s.m.set_block(bp);
    io.emit_indices(&mut s, zero, &[zero, one, two]);
    let nine = s.m.i32c(9);
    io.store_primitive_output(&mut s, ok, 0, 0, nine, zero);
    s.m.br(end);
    s.m.set_block(end);
    let info = StageInfo::Mesh { numthreads: [3, 1, 1], max_vertices: 3, max_primitives: 1, topology: Topology::Triangle, payload_bytes: 0 };
    s.finish_graphics(io, info)
}

#[test]
fn emitted_ms_primitive_attribute_reaches_the_pixel_shader() {
    use crate::gpu::recomp::runner;
    match runner::mesh_shaders_supported() {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no mesh shaders"),
        Err(e) => return runner::skip_test(&e),
    }
    let (cs, ms, ps) = (empty_cs(), prim_ms(), prim_ps());
    assert_valid(&ms);
    assert_valid(&ps);
    let (px, _) = runner::mesh_draw(&cs, &ms, &ps, 1, 4, &[], &[0; 4], SIZE, CLEAR).expect("mesh draw");
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let want = if x <= y && x >= 4 { [COLOR[0], COLOR[1], COLOR[2], 9.0] } else { CLEAR };
            assert_eq!(px[y * SIZE as usize + x], want, "pixel ({x}, {y})");
        }
    }
}

/// `storeOutput` of each (element, column, value), row 0.
fn store_all(io: &mut GraphicsIo, s: &mut Shader, stores: &[(Sig, u8, Value)]) {
    for &(sig, c, v) in stores {
        io.store_output(s, sig, 0, c, v);
    }
}

/// `tests/data/dxil/dxc_vs_clip.hlsl`: clip and cull distances (two clip elements
/// sharing a row), `SV_ViewportArrayIndex`.
fn vs_clip() -> Vec<u8> {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let ip = io.input(SigElement::user("POSITION", 0, CompType::F32, 4));
    let pos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let a = io.output(SigElement::user("A", 0, CompType::F32, 2));
    let clip = io.output(SigElement::new("SV_ClipDistance", 0, SemKind::ClipDistance, CompType::F32, 3));
    let c = io.output(SigElement::user("C", 0, CompType::F32, 1));
    let cull = io.output(SigElement::new("SV_CullDistance", 0, SemKind::CullDistance, CompType::F32, 2));
    let clip1 = io.output(SigElement::new("SV_ClipDistance", 1, SemKind::ClipDistance, CompType::F32, 1));
    let vp = io.output(SigElement::new("SV_ViewportArrayIndex", 0, SemKind::ViewportArrayIndex, CompType::U32, 1));
    let p: Vec<Value> = (0..4).map(|i| io.load_input(&mut s, ip, 0, i)).collect();
    let one = s.m.i32c(1);
    let st = [
        (pos, 0, p[0]), (pos, 1, p[1]), (pos, 2, p[2]), (pos, 3, p[3]),
        (a, 0, p[0]), (a, 1, p[1]),
        (clip, 0, p[0]), (clip, 1, p[1]), (clip, 2, p[2]),
        (c, 0, p[3]),
        (cull, 0, p[2]), (cull, 1, p[3]),
        (clip1, 0, p[0]),
        (vp, 0, one),
    ];
    store_all(&mut io, &mut s, &st);
    s.finish_graphics(io, StageInfo::Vertex)
}

/// `tests/data/dxil/dxc_vs_clip_rows.hlsl`: clip/cull distances never share a
/// row with user semantics.
fn vs_clip_rows() -> Vec<u8> {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let ip = io.input(SigElement::user("POSITION", 0, CompType::F32, 4));
    let pos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let a = io.output(SigElement::user("A", 0, CompType::F32, 1));
    let clip = io.output(SigElement::new("SV_ClipDistance", 0, SemKind::ClipDistance, CompType::F32, 1));
    let b = io.output(SigElement::user("B", 0, CompType::F32, 1));
    let cull = io.output(SigElement::new("SV_CullDistance", 0, SemKind::CullDistance, CompType::F32, 1));
    let c = io.output(SigElement::user("C", 0, CompType::F32, 1));
    let d = io.output(SigElement::user("D", 0, CompType::F32, 2));
    let p: Vec<Value> = (0..4).map(|i| io.load_input(&mut s, ip, 0, i)).collect();
    let (one, two) = (f(&mut s, 1.0), f(&mut s, 2.0));
    let st = [
        (pos, 0, p[0]), (pos, 1, p[1]), (pos, 2, p[2]), (pos, 3, p[3]),
        (a, 0, p[0]), (clip, 0, p[1]), (b, 0, p[2]), (cull, 0, p[3]), (c, 0, one), (d, 0, two), (d, 1, two),
    ];
    store_all(&mut io, &mut s, &st);
    s.finish_graphics(io, StageInfo::Vertex)
}

const DXC_VS_CLIP: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_vs_clip.dxil");
const DXC_VS_CLIP_ROWS: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_vs_clip_rows.dxil");

#[test]
fn vs_clip_parts_match_dxc() {
    let c = vs_clip();
    assert_parts_match(&c, DXC_VS_CLIP);
    assert_eq!(part(&c, b"SFI0"), part(DXC_VS_CLIP, b"SFI0"));
    assert_valid(&c);
}

#[test]
fn vs_clip_rows_parts_match_dxc() {
    let c = vs_clip_rows();
    assert_parts_match(&c, DXC_VS_CLIP_ROWS);
    assert_valid(&c);
}

const DXC_PS_EARLY: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ps_early.dxil");

/// `tests/data/dxil/dxc_ps_early.hlsl`: `[earlydepthstencil]` (shader flag only).
#[test]
fn ps_early_depth_matches_dxc() {
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let p = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let t = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F32, 4));
    for c in 0..4 {
        let v = io.load_input(&mut s, p, 0, c);
        io.store_output(&mut s, t, 0, c, v);
    }
    let c = s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: true });
    assert_parts_match(&c, DXC_PS_EARLY);
    assert_valid(&c);
    let dis = validate::disassemble(&c).unwrap();
    assert!(dis.contains("!{i32 0, i64 8}"), "{dis}");
}

const DXC_VS_UAV: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_vs_uav.dxil");

/// `tests/data/dxil/dxc_vs_uav.hlsl`: a bound UAV in a vertex shader needs the
/// UAVs-at-every-stage flag.
#[test]
fn vs_uav_matches_dxc() {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let mem = s.raw_uav(2, 0);
    let vid = io.input(SigElement::new("SV_VertexID", 0, SemKind::VertexId, CompType::U32, 1));
    let pos = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let v = io.load_input(&mut s, vid, 0, 0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Shl, v, four);
    let k = s.raw_load(mem, off, 0xF);
    for (c, &x) in k.iter().enumerate() {
        io.store_output(&mut s, pos, 0, c as u8, x);
    }
    let c = s.finish_graphics(io, StageInfo::Vertex);
    assert_parts_match(&c, DXC_VS_UAV);
    assert_eq!(part(&c, b"SFI0"), part(DXC_VS_UAV, b"SFI0"));
    assert_valid(&c);
}

/// A pixel shader sampling heap texture 0 with sampler 0 at
/// `uv = SV_Position.xy / 8` (implicit LOD), `w = ddx_fine(uv.x) * 8`.
fn sampling_ps() -> Vec<u8> {
    use crate::gpu::dxil::ops::{CompType as OpsComp, Deriv, TexKind};
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let pos = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let tgt = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F32, 4));
    let zero = s.m.i32c(0);
    let tex = s.heap_texture(zero, TexKind::D2, false, OpsComp::F32, 4, false);
    let smp = s.heap_sampler(zero, false, false);
    let x = io.load_input(&mut s, pos, 0, 0);
    let y = io.load_input(&mut s, pos, 0, 1);
    let eighth = f(&mut s, 0.125);
    let u = s.m.fbin(BinOp::Mul, x, eighth, 0);
    let v = s.m.fbin(BinOp::Mul, y, eighth, 0);
    let r = s.sample(tex, smp, &[u, v], &[], None);
    let du = s.deriv(Deriv::FineX, u);
    let eight = f(&mut s, 8.0);
    let w = s.m.fbin(BinOp::Mul, du, eight, 0);
    for (c, &val) in [r[0], r[1], r[2], w].iter().enumerate() {
        io.store_output(&mut s, tgt, 0, c as u8, val);
    }
    s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false })
}

/// Full-screen triangle VS: `SV_Position` from `POSITION.xy`.
fn fullscreen_vs() -> Vec<u8> {
    let mut s = Shader::new(Stage::Vertex, (6, 6));
    let mut io = GraphicsIo::new(Stage::Vertex);
    let pos = io.input(SigElement::user("POSITION", 0, CompType::F32, 2));
    let out = io.output(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let x = io.load_input(&mut s, pos, 0, 0);
    let y = io.load_input(&mut s, pos, 0, 1);
    let (z, o) = (f(&mut s, 0.0), f(&mut s, 1.0));
    for (c, &v) in [x, y, z, o].iter().enumerate() {
        io.store_output(&mut s, out, 0, c as u8, v);
    }
    s.finish_graphics(io, StageInfo::Vertex)
}

/// Implicit-LOD sampling and fine derivatives in an emitted pixel shader
/// (the `ops` library could only test these in compute shaders).
#[test]
fn emitted_ps_samples_a_texture_with_implicit_lod() {
    use crate::gpu::recomp::runner;
    use std::mem::transmute_copy;
    use windows::Win32::Graphics::Direct3D::*;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    if !draw_ready() {
        return;
    }
    let (vs, ps) = (fullscreen_vs(), sampling_ps());
    assert_valid(&vs);
    assert_valid(&ps);
    // 4×4 RGBA32F texture: texel (i, j) = (i, j, 10·i + j, 1).
    let mut data = Vec::new();
    for j in 0..4 {
        for i in 0..4 {
            for v in [i as f32, j as f32, (10 * i + j) as f32, 1.0] {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = runner::ViewTexture {
        res: D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        },
        srv: D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
        },
        data,
        row_bytes: 64,
    };
    let sampler = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    let px = unsafe {
        let factory: IDXGIFactory6 = match CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) {
            Ok(f) => f,
            Err(e) => return runner::skip_test(&format!("{e}")),
        };
        let sel = crate::gpu::adapter::Selector::from_env_or("auto");
        let Some((adapter, _)) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next() else {
            return runner::skip_test("no adapter");
        };
        let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).expect("device");
        let rs = crate::gpu::recomp::pso::abi_root_signature(&dev, true).expect("root signature");
        let names = [c"POSITION"];
        let elems = [D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::PCSTR(names[0].as_ptr() as *const u8),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        }];
        let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&rs),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: 1 },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = true.into();
        pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&pd).expect("graphics PSO");
        let verts = [-1.0f32, -1.0, -1.0, 3.0, 3.0, -1.0];
        let draw = runner::Draw { sgprs: vec![], scissor: [0, 0, SIZE as i32, SIZE as i32] };
        runner::draw_on_multi(&dev, &pso, &rs, &verts, 8, &[draw], &[0; 4], &[tex], &[sampler], SIZE, CLEAR).expect("draw")
    };
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let (i, j) = ((x / 2) as f32, (y / 2) as f32);
            assert_eq!(px[y * SIZE as usize + x], [i, j, 10.0 * i + j, 1.0], "pixel ({x}, {y})");
        }
    }
}

#[test]
fn elements_from_dxbc_rows() {
    let e = SigElement::from_d3d("TEXCOORD", 3, 0, 3, 0b0110, Some(2)).unwrap();
    assert_eq!((e.kind, e.ty, e.cols, e.place), (SemKind::Arbitrary, CompType::F32, 2, Some((2, 1))));
    let e = SigElement::from_d3d("SV_Position", 0, 1, 3, 0xF, None).unwrap();
    assert_eq!((e.kind, e.cols, e.place), (SemKind::Position, 4, None));
    assert_eq!(SigElement::from_d3d("SV_Target", 1, 64, 1, 0x1, None).unwrap().kind, SemKind::Target);
    assert_eq!(SemKind::from_d3d_name(25), Some(SemKind::CullPrimitive));
    assert!(SigElement::from_d3d("X", 0, 999, 3, 1, None).is_none());
}

/// `tests/data/dxil/dxc_ps_min16.hlsl` / `dxc_ps_native16.hlsl` (`-enable-16bit-types`):
/// `half2 a : A, uint16_t b : B → half4 SV_Target`.
fn ps_16bit(native: bool) -> Vec<u8> {
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    if native {
        s.require(crate::gpu::dxil::shader::Feature::NativeLowPrecision);
    }
    let mut io = GraphicsIo::new(Stage::Pixel);
    let a = io.input(SigElement::user("A", 0, CompType::F16, 2));
    let b = io.input(SigElement::user("B", 0, CompType::U16, 1));
    let t = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F16, 4));
    let bv = io.load_input(&mut s, b, 0, 0);
    let a0 = io.load_input(&mut s, a, 0, 0);
    let a1 = io.load_input(&mut s, a, 0, 1);
    let h = s.m.f16();
    let bf = s.m.cast(CastOp::UIToFP, bv, h);
    let one = s.m.const_float(h, 0x3C00);
    for (c, &v) in [a0, a1, bf, one].iter().enumerate() {
        io.store_output(&mut s, t, 0, c as u8, v);
    }
    s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false })
}

const DXC_PS_MIN16: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ps_min16.dxil");
const DXC_PS_NATIVE16: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ps_native16.dxil");

#[test]
fn ps_16bit_signatures_match_dxc() {
    for (native, dxc) in [(false, DXC_PS_MIN16), (true, DXC_PS_NATIVE16)] {
        let c = ps_16bit(native);
        assert_parts_match(&c, dxc);
        assert_eq!(part(&c, b"SFI0"), part(dxc, b"SFI0"), "native {native}");
        assert_valid(&c);
    }
}

/// Groupshared sizes under the native 16-bit layout: DXC reports 18 bytes
/// of PSV0 groupshared memory for `groupshared uint16_t g[6]; groupshared
/// half h[3];` with `-enable-16bit-types` (36 with the 32-bit layout).
#[test]
fn groupshared_bytes_follow_the_16bit_layout() {
    let mut s = Shader::new(Stage::Mesh, (6, 6));
    let (i16, f16) = (s.m.int(16), s.m.f16());
    let (a, b) = (s.m.array(6, i16), s.m.array(3, f16));
    s.m.global("g", a, 3, None, false, 2);
    s.m.global("h", b, 3, None, false, 2);
    assert_eq!(s.m.global_bytes_layout(3, true), 18);
    assert_eq!(s.m.global_bytes(3), 36);
}

const DXC_CS_RESOURCES: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_cs_resources.dxil");
const DXC_PS_RESOURCES: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_ps_resources.dxil");

/// `tests/data/dxil/dxc_cs_resources.hlsl`: statically bound SRVs of every buffer
/// kind and a texture (space 1), a cbuffer, a sampler and a comparison
/// sampler, and a raw UAV — `dx.resources` and PSV0 from the shared
/// resource helper on the compute path.
#[test]
fn cs_static_resources_match_dxc() {
    use crate::gpu::dxil::ops::{CompType as OpsComp, TexKind};
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let rawb = s.srv_raw(0, 0);
    let sb = s.srv_structured(1, 0, 8);
    let tb = s.srv_typed(2, 0, OpsComp::F32, 4);
    let tex = s.srv_texture(3, 1, TexKind::D2, OpsComp::F32, 4);
    let cb = s.cbv(0, 0, 32);
    let smp = s.sampler(0, 0, false);
    let cmp = s.sampler(1, 0, true);
    let res = s.raw_uav(1, 0);
    let id = s.flattened_thread_id_in_group();
    let i32 = s.m.i32();
    let two = s.m.i32c(2);
    let off = s.m.bin(BinOp::Shl, id, two);
    let a = s.raw_load(rawb, off, 1)[0];
    let b = s.structured_load(sb, id, 0, 3)[1];
    let c = s.buffer_load(tb, id)[0];
    let c = s.m.bitcast(c, i32);
    let one = s.m.i32c(1);
    let d = s.cbuffer_load_legacy(cb, one, i32)[1];
    let v = s.m.bin(BinOp::Add, a, b);
    let v = s.m.bin(BinOp::Add, v, c);
    let v = s.m.bin(BinOp::Add, v, d);
    let (h, z) = (f(&mut s, 0.5), f(&mut s, 0.0));
    let x = s.sample_level(tex, smp, &[h, h], &[], z)[0];
    let y = s.sample_cmp_level_zero(tex, cmp, &[h, h], &[], h)[0];
    let fs = s.m.fbin(BinOp::Add, x, y, 0);
    let fi = s.m.cast(CastOp::BitCast, fs, i32);
    let v = s.m.bin(BinOp::Add, v, fi);
    s.raw_store(res, off, &[v]);
    let c = s.finish([32, 1, 1], None);
    assert_valid(&c);
    for cc in [b"PSV0", b"SFI0"] {
        let (a, b) = (part(&c, cc).unwrap(), part(DXC_CS_RESOURCES, cc).unwrap());
        assert!(a == b, "{} differs\n--- ours\n{}--- dxc\n{}", String::from_utf8_lossy(cc), hex(a), hex(b));
    }
    let dis = validate::disassemble(&c).unwrap();
    for needle in [
        "@dx.op.cbufferLoadLegacy.i32(i32 59",
        "i32 0, i32 0, i32 1, i32 11, i32 0, null}",
        "i32 0, i32 1, i32 1, i32 12, i32 0, !",
        "i32 1, i32 3, i32 1, i32 2, i32 0, !",
        "i32 0, i32 0, i32 1, i32 32, null}",
        "i32 0, i32 1, i32 1, i32 1, null}",
    ] {
        assert!(dis.contains(needle), "missing `{needle}` in:\n{dis}");
    }
}

/// `tests/data/dxil/dxc_ps_resources.hlsl`: a root-constant style cbuffer
/// (8 bytes, `b2, space1`), a texture and a sampler on the graphics path.
#[test]
fn ps_static_resources_match_dxc() {
    use crate::gpu::dxil::ops::{CompType as OpsComp, TexKind};
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let pos = io.input(SigElement::new("SV_Position", 0, SemKind::Position, CompType::F32, 4));
    let tgt = io.output(SigElement::new("SV_Target", 0, SemKind::Target, CompType::F32, 4));
    let tex = s.srv_texture(0, 0, TexKind::D2, OpsComp::F32, 4);
    let smp = s.sampler(3, 0, false);
    let rc = s.cbv(2, 1, 8);
    let i32 = s.m.i32();
    let zero = s.m.i32c(0);
    let row = s.cbuffer_load_legacy(rc, zero, i32);
    let f32 = s.m.f32();
    let scale = s.m.bitcast(row[1], f32);
    let k = s.m.cast(CastOp::UIToFP, row[0], f32);
    let x = io.load_input(&mut s, pos, 0, 0);
    let y = io.load_input(&mut s, pos, 0, 1);
    let u = s.m.fbin(BinOp::Mul, x, scale, 0);
    let v = s.m.fbin(BinOp::Mul, y, scale, 0);
    let r = s.sample(tex, smp, &[u, v], &[], None);
    for (c, &val) in r.iter().enumerate() {
        let o = s.m.fbin(BinOp::Add, val, k, 0);
        io.store_output(&mut s, tgt, 0, c as u8, o);
    }
    let c = s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false });
    assert_parts_match(&c, DXC_PS_RESOURCES);
    assert_eq!(part(&c, b"SFI0"), part(DXC_PS_RESOURCES, b"SFI0"));
    assert_valid(&c);
}
