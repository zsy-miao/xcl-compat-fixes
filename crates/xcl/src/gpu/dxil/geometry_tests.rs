//! Geometry mode in the direct translator: a GS with per-lane loops and cuts, validated in
//! every form and its records compared on the GPU with a model (and the HLSL path's native form).

use super::kernel::{translate_geometry, KernelOptions};
use crate::gpu::dxbc::SigElement;
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::{dxc, mesh_passthrough, runner, translate_geometry as hlsl_geometry, GsInterface, HEAP_WINDOWS, IO_GS_PRIMS, IO_GS_VERTICES, MESH_HEADER};

fn sig(name: &str, sv: u32, reg: u32, ty: u32) -> SigElement {
    SigElement { name: name.into(), index: 0, system_value: sv, component_type: ty, register: reg, mask: 0xF, rw_mask: 0xF, stream: 0 }
}

/// ES: nothing but the jump into the GS. GS: EXEC = this wave's primitives, each lane
/// emits 2..5 vertices in a loop (EXEC narrowing per lane, `s_cbranch_execz` out), a cut
/// after the second vertex; vertex = (lane, k, primitive, 1.0), (count, lane, lane, lane).
fn branchy_program() -> (Vec<u8>, GsInterface) {
    let e = |n: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, enc, f, lit).unwrap_or_else(|x| panic!("{n}: {x}"));
    const V: u64 = 256;
    let es = e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None);
    let mut gs: Vec<Vec<u8>> = Vec::new();
    gs.push(e("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)));
    gs.push(e("S_BFM_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)], None));
    gs.push(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    gs.push(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None));
    gs.push(e("V_AND_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 131), ("VSRC1", 2)], None)); // prim & 3
    gs.push(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 130), ("VSRC1", 21)], None)); // + 2
    gs.push(e("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 50), ("SRC0", V + 20), ("SRC1", 255), ("SRC2", 2)], Some(160)));
    gs.push(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 52), ("SRC0", 242)], None)); // 1.0
    gs.push(e("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 128)], None));
    let top = gs.len();
    gs.push(e("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", 4), ("VSRC1", 21)], None));
    gs.push(e("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)], None));
    let execz = gs.len();
    gs.push(Vec::new()); // s_cbranch_execz end
    gs.push(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 51), ("SRC0", 4)], None));
    gs.push(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 20), ("DATA1", 51), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    gs.push(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 2), ("DATA1", 52), ("OFFSET0", 2), ("OFFSET1", 3)], None));
    gs.push(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 21), ("DATA1", 20), ("OFFSET0", 4), ("OFFSET1", 5)], None));
    gs.push(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 20), ("DATA1", 20), ("OFFSET0", 6), ("OFFSET1", 7)], None));
    gs.push(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 50), ("SRC0", 160), ("VSRC1", 50)], None));
    gs.push(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None)); // emit
    gs.push(e("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 129)], None));
    let skip = gs.len();
    gs.push(Vec::new()); // s_cbranch_scc0 past the cut
    gs.push(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x12)], None)); // cut
    let after_cut = gs.len();
    gs.push(e("S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)], None));
    let back = gs.len();
    gs.push(Vec::new()); // s_branch top
    let end = gs.len();
    gs.push(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None));
    gs.push(e("S_ENDPGM", "ENC_SOPP", &[], None));
    let pc = |gs: &Vec<Vec<u8>>, i: usize| gs[..i].iter().map(|p| if p.is_empty() { 4 } else { p.len() }).sum::<usize>() as i64;
    let br = |gs: &Vec<Vec<u8>>, name: &str, at: usize, to: usize| {
        let rel = (pc(gs, to) - pc(gs, at) - 4) / 4;
        e(name, "ENC_SOPP", &[("SIMM16", rel as i16 as u16 as u64)], None)
    };
    gs[execz] = br(&gs, "S_CBRANCH_EXECZ", execz, end);
    gs[skip] = br(&gs, "S_CBRANCH_SCC0", skip, after_cut);
    gs[back] = br(&gs, "S_BRANCH", back, top);
    let g = GsInterface {
        es_inputs: vec![],
        es_outputs: vec![sig("SV_Position", 1, 0, 3)],
        outputs: vec![sig("SV_Position", 1, 0, 3), sig("TEXCOORD", 0, 1, 1)],
        max_vertex_count: 5,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es;
    code.extend(gs.concat());
    (code, g)
}

fn opts(split: bool, windows: u32) -> KernelOptions {
    KernelOptions { wave64: true, split, mem_windows: windows, sampler_base: 0 }
}

/// Both forms, ES alone, GS alone and four memory windows translate and validate.
#[test]
fn geometry_forms_validate() {
    let (code, g) = branchy_program();
    for split in [false, true] {
        translate_geometry(&code, opts(split, 1), &g).unwrap_or_else(|e| panic!("split {split}: {e}"));
        translate_geometry(&code, opts(split, 4), &g).unwrap_or_else(|e| panic!("split {split}, 4 windows: {e}"));
        let es_alone = GsInterface { gs_start: None, ..g.clone() };
        translate_geometry(&code[..4], opts(split, 1), &es_alone).unwrap_or_else(|e| panic!("ES alone: {e}"));
        let gs_alone = GsInterface { gs_start: Some(0), ..g.clone() };
        translate_geometry(&code[4..], opts(split, 1), &gs_alone).unwrap_or_else(|e| panic!("GS alone: {e}"));
    }
}

/// The records of `branchy_program` for `prims` primitives: (group, record dwords).
fn expected_records(g: &GsInterface, prims: usize) -> Vec<u32> {
    let rec = g.record();
    let (rd, vd, maxv) = (rec.record_dwords(), rec.vert_dwords(), rec.max_vertices as usize);
    let groups = prims.div_ceil(21);
    let mut out = vec![0u32; groups * rd];
    for grp in 0..groups {
        let r = &mut out[grp * rd..(grp + 1) * rd];
        let (mut vb, mut tb) = (0usize, 0usize);
        for lane in 0..21.min(prims - grp * 21) {
            let p = grp * 21 + lane;
            let n = (p & 3) + 2;
            for k in 0..n {
                let v = MESH_HEADER + (vb + k) * vd;
                r[v..v + 8].copy_from_slice(&[lane as u32, k as u32, p as u32, 0x3F80_0000, n as u32, lane as u32, lane as u32, lane as u32]);
            }
            if n == 5 {
                let b = vb as u32;
                r[MESH_HEADER + maxv * vd + tb] = (b + 2) | (b + 3) << 10 | (b + 4) << 20;
                tb += 1;
            }
            vb += n;
        }
        r[0] = vb as u32;
        r[1] = tb as u32;
    }
    out
}

/// The direct translation's records (GPU, mesh pass-through included) for per-lane loops,
/// cuts and a partial last group, against a model of the program, in the split form and
/// (64-lane hardware) the native one; the HLSL path's records in the same forms must agree too.
#[test]
fn gpu_geometry_direct_records() {
    if runner::adapter().is_err() || runner::wave64_programs_supported() != Ok(true) || dxc::location().is_err() || runner::mesh_shaders_supported() != Ok(true) {
        runner::skip_test("no GPU / DXC / mesh shaders");
        return;
    }
    let (code, g) = branchy_program();
    let rec = g.record();
    let msb = dxc::compile(&mesh_passthrough(&rec), "main", "ms_6_6", &[]).unwrap();
    let ps_src = "struct V { float4 pos : SV_Position; nointerpolation uint4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return float4(1.0, (float)v.p0.y, 0.0, 1.0); }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();
    let prims = 25;
    let mut io = vec![0u32; 128];
    io[IO_GS_PRIMS] = prims as u32;
    let mem = vec![0u32; 64];
    let want = expected_records(&g, prims);
    let check = |what: &str, got: &[u32]| {
        // the vertex / triangle slots past the counts are not written
        let rd = rec.record_dwords();
        let (vd, maxv) = (rec.vert_dwords(), rec.max_vertices as usize);
        let diff: Vec<usize> = (0..want.len())
            .filter(|&i| {
                let (grp, o) = (i / rd, i % rd);
                let r = &want[grp * rd..];
                let used = o < MESH_HEADER || (o < MESH_HEADER + maxv * vd && (o - MESH_HEADER) / vd < r[0] as usize) || (o >= MESH_HEADER + maxv * vd && o - MESH_HEADER - maxv * vd < r[1] as usize);
                used && want[i] != got[i]
            })
            .take(16)
            .collect();
        assert!(diff.is_empty(), "{what}: records differ at {diff:?}: got {:?}, want {:?}", diff.iter().map(|&i| got[i]).collect::<Vec<_>>(), diff.iter().map(|&i| want[i]).collect::<Vec<_>>());
    };
    let mut forms = vec![true];
    if runner::wave64_supported() == Ok(true) {
        forms.push(false);
    }
    for split in forms {
        let direct = translate_geometry(&code, opts(split, 1), &g).unwrap().dxil;
        let (_, got) = runner::mesh_draw(&direct, &msb, &ps, 2, rec.record_dwords(), &io, &mem, 32, [0.0; 4]).unwrap();
        check(&format!("direct, split {split}"), &got);
        let t = hlsl_geometry(&code, g.clone(), split, 1).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let (_, got) = runner::mesh_draw(&hlsl, &msb, &ps, 2, rec.record_dwords(), &io, &mem, 32, [0.0; 4]).unwrap();
        check(&format!("HLSL, split {split}"), &got);
    }
}

/// The GS epilogue coalesces the 2-dword header store and each 4-dword vertex attribute store
/// into wide `rawBufferStore` calls (mask 3 and mask 15) instead of per-dword scalar stores.
#[test]
fn geometry_epilogue_coalesces_header_and_vertex_stores() {
    let (code, g) = branchy_program();
    let k = translate_geometry(&code, opts(true, 1), &g).expect("translate geometry");
    super::validate::validate(&k.dxil).expect("validate geometry");
    let dis = super::validate::disassemble(&k.dxil).expect("disassemble geometry");
    let stores: Vec<&str> = dis.lines().filter(|l| l.contains("call void @dx.op.rawBufferStore.i32(")).collect();
    // 1 two-dword header store (mask 3), 2 four-dword vertex register stores (mask 15), 1 triangle store (mask 1).
    let mask_count = |mask: u8| stores.iter().filter(|l| l.contains(&format!(", i8 {mask}, i32 4)"))).count();
    assert_eq!(mask_count(3), 1, "expected 1 coalesced 2-dword header store:\n{}", stores.join("\n"));
    assert_eq!(mask_count(15), 2, "expected 2 coalesced 4-dword vertex stores:\n{}", stores.join("\n"));
    assert_eq!(mask_count(1), 1, "expected only 1 scalar store (triangle index):\n{}", stores.join("\n"));
    assert_eq!(stores.len(), 4, "expected 4 total rawBufferStore calls:\n{}", stores.join("\n"));
}

/// ES+GS with pre-fetched ES vertex inputs (`v9..v12`) under `HEAP_WINDOWS`: `geom_init`
/// must load pre-fetched vertices from window 0 in the descriptor heap (`MEM_HEAP_BASE`)
/// rather than the placeholder `lw.mems[0]` handle.
#[test]
fn geometry_es_inputs_validate_and_run_with_heap_windows() {
    let e = |n: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, enc, f, lit).unwrap_or_else(|x| panic!("{n}: {x}"));
    // ES: compute byte offset = 0x100 + v5 * 4, store fetched v9 to window 0 (`hi = base_hi`),
    // then jump to GS (which ends immediately).
    let mut code = Vec::new();
    code.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 20), ("SRC0", 130), ("VSRC1", 5)], None)); // v20 = v5 * 4
    code.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 20), ("SRC0", 255), ("VSRC1", 20)], Some(0x100))); // v20 += 0x100
    code.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 128)], None)); // v21 = 0 (hi = uma_base_hi = 0)
    code.extend(e("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 20), ("DATA", 9), ("SADDR", 0x7D), ("OFFSET", 0)], None));
    code.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None));
    let gs_start = code.len() as u64;
    code.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None));
    code.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));

    let g = GsInterface {
        es_inputs: vec![sig("POSITION", 0, 0, 3)],
        es_outputs: vec![sig("SV_Position", 1, 0, 3)],
        outputs: vec![sig("SV_Position", 1, 0, 3)],
        max_vertex_count: 3,
        gs_start: Some(gs_start),
    };

    for split in [false, true] {
        translate_geometry(&code, opts(split, 4 | HEAP_WINDOWS), &g)
            .unwrap_or_else(|err| panic!("split {split}, 4 heap windows: {err}"));
    }

    if runner::adapter().is_err() || runner::wave64_programs_supported() != Ok(true) {
        runner::skip_test("no GPU / wave64");
        return;
    }
    let split = runner::split_wave64();
    let dxil = translate_geometry(&code, opts(split, 4 | HEAP_WINDOWS), &g).unwrap().dxil;
    let mut io = vec![0u32; 128];
    io[IO_GS_PRIMS] = 1; // 1 primitive = 3 vertices
    io[IO_GS_VERTICES] = 0; // pre-fetched vertices start at byte 0 of window 0
    let mut mem = vec![0u32; 128];
    // 3 vertices, 1 register (16 bytes = 4 dwords) each; component 0 is v9.
    mem[0] = 0x1111_0001;
    mem[4] = 0x2222_0002;
    mem[8] = 0x3333_0003;
    let out = runner::dispatch(&dxil, [1, 1, 1], &io, &mem).unwrap();
    assert_eq!(out[0x100 / 4], 0x1111_0001, "vertex 0 v9 fetched via heap window 0");
    assert_eq!(out[0x100 / 4 + 1], 0x2222_0002, "vertex 1 v9 fetched via heap window 0");
    assert_eq!(out[0x100 / 4 + 2], 0x3333_0003, "vertex 2 v9 fetched via heap window 0");
    assert_eq!(out[0x100 / 4 + 3], 0, "lane past nv gets 0");
}
