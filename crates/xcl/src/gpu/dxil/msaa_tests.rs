//! Multisampled textures: the `dx.op` helpers (`Texture2DMS` / `Texture2DMSArray` loads,
//! sample counts, sample positions) and the direct translator's MSAA image instructions
//! (`image_load` / `image_msaa_load` / `image_get_resinfo` with DIM 2D_MSAA(_ARRAY)), run on
//! the GPU against a CPU reference. The test textures are filled sample by sample
//! (`runner::TexDim::D2Ms`).

use super::container::Stage;
use super::module::{BinOp, CastOp, Value};
use super::ops::*;
use super::shader::{RawUav, Shader};
use super::validate;
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::runner::{self, KernelState, TestTexture, TexDim};
use windows::Win32::Graphics::Dxgi::Common::*;

fn check(c: &[u8]) {
    if let Err(e) = validate::validate(c) {
        let dis = validate::disassemble(c).unwrap_or_else(|d| d);
        panic!("validator rejected the shader:\n{e}\n--- disassembly ---\n{dis}");
    }
}

fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, _, d) => {
            runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            false
        }
    }
}

/// 4×4 RGBA32F, 4 samples: sample s of (x, y) = (x, y, s, x + 4y + 16s).
fn ms_rgba() -> TestTexture {
    let mut data = Vec::new();
    for y in 0..4u32 {
        for x in 0..4u32 {
            for s in 0..4u32 {
                for v in [x as f32, y as f32, s as f32, (x + 4 * y + 16 * s) as f32] {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    TestTexture { dim: TexDim::D2Ms(4), format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data, uav: false }
}

/// 4×4×3 R32_UINT array, 2 samples: 1000z + 100s + x + 4y.
fn ms_array_u32() -> TestTexture {
    let mut data = Vec::new();
    for z in 0..3u32 {
        for y in 0..4u32 {
            for x in 0..4u32 {
                for s in 0..2u32 {
                    data.extend_from_slice(&(1000 * z + 100 * s + x + 4 * y).to_le_bytes());
                }
            }
        }
    }
    TestTexture { dim: TexDim::D2MsArray(2), format: DXGI_FORMAT_R32_UINT, width: 4, height: 4, depth: 3, texel_bytes: 4, data, uav: false }
}

/// The ops: heap `Texture2DMS<float4>` / `Texture2DMSArray<uint>` loads per sample,
/// `GetDimensions` (sample count in `w`) and `GetSamplePosition`.
#[test]
fn msaa_texture_ops_run() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let res: RawUav = s.raw_uav(1, 0);
    let lane = s.flattened_thread_id_in_group();
    let c = |s: &mut Shader, v: u32| s.m.i32c(v);
    // lane → x = l % 4, y = l / 4 % 4, sample = l / 16 (32 lanes: samples 0, 1)
    let x = {
        let k = c(&mut s, 3);
        s.m.bin(BinOp::And, lane, k)
    };
    let y = {
        let k = c(&mut s, 2);
        let t = s.m.bin(BinOp::LShr, lane, k);
        let k3 = c(&mut s, 3);
        s.m.bin(BinOp::And, t, k3)
    };
    let smp = {
        let k = c(&mut s, 4);
        s.m.bin(BinOp::LShr, lane, k)
    };
    let i0 = c(&mut s, 0);
    let i2 = c(&mut s, 2);
    let t = s.heap_texture(i0, TexKind::D2MS, false, CompType::F32, 4, false);
    let ta = s.heap_texture(i2, TexKind::D2MSArray, false, CompType::U32, 1, false);
    let r = s.texture_load_sample(t, smp, &[x, y]);
    // the array: slice = sample index too (0 / 1), sample 1 - smp
    let one = c(&mut s, 1);
    let other = s.m.bin(BinOp::Sub, one, smp);
    let ra = s.texture_load_sample(ta, other, &[x, y, smp]);
    let d = s.get_dimensions(t.handle, None);
    let da = s.get_dimensions(ta.handle, None);
    let pos = s.texture2dms_sample_position(t, smp);
    let i32 = s.m.i32();
    let mut outs: Vec<Value> = r.iter().map(|&v| s.m.bitcast(v, i32)).collect();
    outs.push(ra[0]);
    for i in [0, 1, 3] {
        outs.push(s.m.extract(d, i));
    }
    for i in [0, 1, 2, 3] {
        outs.push(s.m.extract(da, i));
    }
    for p in pos {
        outs.push(s.m.bitcast(p, i32));
    }
    for (k, &v) in outs.iter().enumerate() {
        let base = c(&mut s, 64 * k as u32);
        let i = s.m.bin(BinOp::Add, base, lane);
        let two = c(&mut s, 2);
        let off = s.m.bin(BinOp::Shl, i, two);
        let v = if s.m.type_of(v) == s.m.i1() { s.m.cast(CastOp::ZExt, v, i32) } else { v };
        s.raw_store(res, off, &[v]);
    }
    let dxil = s.finish([32, 1, 1], Some(32));
    check(&dxil);
    if !gpu_ready() || runner::wave32_supported() != Ok(true) {
        return;
    }
    let tex = [ms_rgba(), ms_array_u32()];
    let r = runner::run_with(&dxil, &KernelState::default(), &[0; 64], &tex, &[]).expect("GPU run");
    let o = |k: usize, l: usize| r.raw[64 * k + l];
    let mut seen = Vec::new();
    for l in 0..32 {
        let (x, y, s) = ((l % 4) as u32, (l / 4 % 4) as u32, (l / 16) as u32);
        let got: Vec<f32> = (0..4).map(|k| f32::from_bits(o(k, l))).collect();
        assert_eq!(got, vec![x as f32, y as f32, s as f32, (x + 4 * y + 16 * s) as f32], "Texture2DMS load lane {l}");
        assert_eq!(o(4, l), 1000 * s + 100 * (1 - s) + x + 4 * y, "Texture2DMSArray load lane {l}");
        assert_eq!([o(5, l), o(6, l), o(7, l)], [4, 4, 4], "Texture2DMS dims lane {l}");
        assert_eq!([o(8, l), o(9, l), o(10, l), o(11, l)], [4, 4, 3, 2], "Texture2DMSArray dims lane {l}");
        let p = (f32::from_bits(o(12, l)), f32::from_bits(o(13, l)));
        assert!((-0.5..0.5).contains(&p.0) && (-0.5..0.5).contains(&p.1), "sample position lane {l}: {p:?}");
        if l % 16 == 0 {
            seen.push(p);
        }
    }
    assert_ne!(seen[0], seen[1], "samples 0 and 1 at different positions");
}

/// `RenderTargetGetSampleCount` / `RenderTargetGetSamplePosition` validate in a pixel shader.
#[test]
fn render_target_sample_ops_validate() {
    use super::graphics::{CompType as SigComp, GraphicsIo, SemKind, SigElement, StageInfo};
    let mut s = Shader::new(Stage::Pixel, (6, 6));
    let mut io = GraphicsIo::new(Stage::Pixel);
    let _pos = io.input(SigElement::new("SV_Position", 0, SemKind::Position, SigComp::F32, 4));
    let tgt = io.output(SigElement::new("SV_Target", 0, SemKind::Target, SigComp::F32, 4));
    let n = s.render_target_sample_count();
    let f32 = s.m.f32();
    let nf = s.m.cast(CastOp::UIToFP, n, f32);
    let one = s.m.i32c(1);
    let p = s.render_target_sample_position(one);
    let z = s.m.f32c(0);
    for (c, v) in [nf, p[0], p[1], z].into_iter().enumerate() {
        io.store_output(&mut s, tgt, 0, c as u8, v);
    }
    let c = s.finish_graphics(io, StageInfo::Pixel { early_depth_stencil: false });
    check(&c);
}

fn mimg(name: &str, f: &[(&str, u64)]) -> Vec<u8> {
    encode(name, "ENC_MIMG", f, None).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn tsharp(index: u32, format: u32, ty: u32) -> [u32; 8] {
    [index, format << 20, 0, ty << 28, 0, 0, 0, 0]
}

fn run_direct(code: &[u8], st: &KernelState, tex: &[TestTexture]) -> runner::KernelResult {
    let k = runner::direct_kernel_full(code, true).unwrap_or_else(|e| panic!("direct translation: {e}"));
    check(&k.dxil);
    runner::run_with_statics(&k.dxil, st, &[], tex, &[], &k.static_samplers).expect("GPU run")
}

/// Console MSAA image instructions through the direct translator: `image_load` of one sample
/// (2D MSAA and 2D MSAA array), `image_msaa_load` (one component of four samples; samples
/// past the count read 0), `image_get_resinfo`, D16 `image_msaa_load`.
#[test]
fn gpu_direct_msaa_image_loads() {
    let mut code = Vec::new();
    // T# 0: 2D MSAA RGBA32F (s[16:23], SRSRC 4); T# 1: 2D MSAA array R32_UINT (s[24:31], 6)
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 6), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_MSAA_LOAD", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x8), ("DIM", 6), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 18), ("VADDR", 5), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 6)]));
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 22), ("VADDR", 30), ("SRSRC", 6), ("DMASK", 0x1), ("DIM", 7), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_MSAA_LOAD", &[("VDATA", 24), ("VADDR", 30), ("SRSRC", 6), ("DMASK", 0x1), ("DIM", 7), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 40), ("VADDR", 5), ("SRSRC", 6), ("DMASK", 0xF), ("DIM", 7)]));
    // D16: the y channel (exact small integers) of four samples as halves
    code.extend(mimg("IMAGE_MSAA_LOAD", &[("VDATA", 44), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x2), ("DIM", 6), ("UNORM", 1), ("D16", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    if !gpu_ready() {
        return;
    }
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 14));
    st.sgpr[24..32].copy_from_slice(&tsharp(2, 20, 15));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as u32, (l / 4 % 4) as u32);
        r[0] = x;
        r[1] = y;
        r[2] = (l / 16) as u32; // fragment 0..3
        r[5] = 0;
        r[30] = x;
        r[31] = y;
        r[32] = (l / 16 % 3) as u32; // slice
        r[33] = (l % 2) as u32; // fragment
    }
    let r = run_direct(&code, &st, &[ms_rgba(), ms_array_u32()]);
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (x, y, fr) = ((l % 4) as u32, (l / 4 % 4) as u32, (l / 16) as u32);
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [x as f32, y as f32, fr as f32, (x + 4 * y + 16 * fr) as f32], "lane {l} image_load 2D MSAA");
        for k in 0..4u32 {
            let s = fr + k;
            let want = if s < 4 { (x + 4 * y + 16 * s) as f32 } else { 0.0 };
            assert_eq!(fl(14 + k as usize), want, "lane {l} image_msaa_load sample {s}");
        }
        assert_eq!(&v[18..22], &[4, 4, 1, 1], "lane {l} resinfo 2D MSAA");
        let (z, f2) = ((l / 16 % 3) as u32, (l % 2) as u32);
        assert_eq!(v[22], 1000 * z + 100 * f2 + x + 4 * y, "lane {l} image_load 2D MSAA array");
        for k in 0..4u32 {
            let s = f2 + k;
            let want = if s < 2 { 1000 * z + 100 * s + x + 4 * y } else { 0 };
            assert_eq!(v[24 + k as usize], want, "lane {l} image_msaa_load array sample {s}");
        }
        assert_eq!(&v[40..44], &[4, 4, 3, 1], "lane {l} resinfo 2D MSAA array");
        let h = |s: u32| if s < 4 { half_bits(y as f32) } else { 0 };
        assert_eq!(v[44], h(fr) | h(fr + 1) << 16, "lane {l} d16 msaa_load lo");
        assert_eq!(v[45], h(fr + 2) | h(fr + 3) << 16, "lane {l} d16 msaa_load hi");
    }
}

/// binary16 bits of a small integer-valued f32.
fn half_bits(v: f32) -> u32 {
    if v == 0.0 {
        return 0;
    }
    let b = v.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    ((e as u32) << 10) | ((b >> 13) & 0x3FF)
}

/// Stores and atomics on MSAA T#s need writable MSAA textures (SM 6.7): a translation error.
#[test]
fn direct_msaa_store_is_rejected() {
    let mut code = mimg("IMAGE_STORE", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x1), ("DIM", 6), ("UNORM", 1)]);
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let e = runner::direct_kernel(&code, true).unwrap_err();
    assert!(e.contains("MSAA"), "{e}");
}

/// Regression guard for an AMD DXIL compiler (`amdxc64.dll`) null-pointer dereference
/// (`0xc0000005`): within any basic block, `dx.op.getDimensions` on a dynamically indexed
/// heap `Texture2DMS` / `Texture2DMSArray` must not precede `dx.op.textureLoad`.
#[test]
fn direct_msaa_loads_precede_get_dimensions() {
    let mut code = Vec::new();
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 6), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_MSAA_LOAD", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x8), ("DIM", 6), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 18), ("VADDR", 5), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 6)]));
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 22), ("VADDR", 30), ("SRSRC", 6), ("DMASK", 0x1), ("DIM", 7), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_MSAA_LOAD", &[("VDATA", 24), ("VADDR", 30), ("SRSRC", 6), ("DMASK", 0x1), ("DIM", 7), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 40), ("VADDR", 5), ("SRSRC", 6), ("DMASK", 0xF), ("DIM", 7)]));
    code.extend(mimg("IMAGE_MSAA_LOAD", &[("VDATA", 44), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x2), ("DIM", 6), ("UNORM", 1), ("D16", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let k = runner::direct_kernel_full(&code, true).expect("direct translation");
    check(&k.dxil);
    let dis = validate::disassemble(&k.dxil).expect("disassemble");
    let mut in_fn = false;
    let mut seen_get_dims = false;
    let mut load_blocks = 0usize;
    let mut saw_load_in_block = false;
    for line in dis.lines() {
        let t = line.trim();
        if t.starts_with("define ") {
            in_fn = true;
            seen_get_dims = false;
            saw_load_in_block = false;
        } else if in_fn && t == "}" {
            if saw_load_in_block {
                load_blocks += 1;
            }
            in_fn = false;
        } else if !in_fn {
            continue;
        } else if t.starts_with("; <label>:") {
            if saw_load_in_block {
                load_blocks += 1;
            }
            seen_get_dims = false;
            saw_load_in_block = false;
        } else if t.contains("call ") && t.contains("@dx.op.getDimensions(") {
            seen_get_dims = true;
        } else if t.contains("call ") && t.contains("@dx.op.textureLoad.") {
            assert!(!seen_get_dims, "dx.op.getDimensions preceded dx.op.textureLoad in the same basic block:\n{dis}");
            saw_load_in_block = true;
        }
    }
    assert!(load_blocks >= 15, "expected MSAA textureLoad blocks across all format branches, saw {load_blocks}");
}

/// `image_get_resinfo` with `D16=1` packs two 16-bit dimensions per VGPR and preserves the
/// trailing VGPRs for both standard 2D and MSAA textures (matching HLSL).
#[test]
fn gpu_direct_image_get_resinfo_d16() {
    let mut code = Vec::new();
    // T# 0: 2D 64x32 R32_UINT (s[16:23], SRSRC 4); T# 1: 2D MSAA array 4x4x3 (s[24:31], SRSRC 6)
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 0), ("VADDR", 10), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1), ("D16", 1)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 4), ("VADDR", 10), ("SRSRC", 6), ("DMASK", 0xF), ("DIM", 7), ("D16", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    if !gpu_ready() {
        return;
    }
    let tex2d = TestTexture {
        dim: TexDim::D2,
        format: DXGI_FORMAT_R32_UINT,
        width: 64,
        height: 32,
        depth: 1,
        texel_bytes: 4,
        data: vec![0u8; 64 * 32 * 4],
        uav: false,
    };
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 20, 9));
    st.sgpr[24..32].copy_from_slice(&tsharp(2, 20, 15));
    for r in &mut st.vgpr {
        r[2] = 0xDEAD_BEEF;
        r[3] = 0xCAFE_BABE;
        r[6] = 0x1122_3344;
        r[7] = 0x5566_7788;
        r[10] = 0; // mip 0
    }
    let r = run_direct(&code, &st, &[tex2d, ms_array_u32()]);
    assert_eq!(r.vgpr[0][0], 64 | (32 << 16), "2D D16 resinfo v0: {:#x}", r.vgpr[0][0]);
    assert_eq!(r.vgpr[0][1], 1 | (1 << 16), "2D D16 resinfo v1: {:#x}", r.vgpr[0][1]);
    assert_eq!((r.vgpr[0][2], r.vgpr[0][3]), (0xDEAD_BEEF, 0xCAFE_BABE), "2D D16 resinfo clobbered v2/v3");
    assert_eq!(r.vgpr[0][4], 4 | (4 << 16), "MSAA array D16 resinfo v4: {:#x}", r.vgpr[0][4]);
    assert_eq!(r.vgpr[0][5], 3 | (1 << 16), "MSAA array D16 resinfo v5: {:#x}", r.vgpr[0][5]);
    assert_eq!((r.vgpr[0][6], r.vgpr[0][7]), (0x1122_3344, 0x5566_7788), "MSAA array D16 resinfo clobbered v6/v7");
}
