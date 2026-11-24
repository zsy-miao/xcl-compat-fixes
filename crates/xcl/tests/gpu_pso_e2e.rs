//! End to end without a game executable: real console shaders from the retail pack
//! (`XCL_SAMPLE_RPF`) go through the D3D12.X backend's CreateGraphicsPipelineState hook
//! exactly as a title would hand them over → recompiled VS/PS → D3D12 pipeline → drawn into
//! a texture on the GPU → pixels read back and compared with values computed here from the
//! inputs the tests set explicitly (vertex buffer, user SGPRs, descriptor tables, constants,
//! textures, samplers).
//!
//! The stand-in shaders XCL uses when a console shader cannot be translated draw nothing
//! (degenerate position) or magenta, and the pipeline counters are checked, so these tests
//! cannot pass on the fallback path.

use std::ffi::CStr;
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xcl::gpu::recomp::runner;
use xcl::gpu::{desc, pack};
use xcl::hle::d3d12x::real;

const SIZE: u32 = 64;
const CLEAR: [f32; 4] = [0.1, 0.2, 0.3, 0.4];

/// The tests share the backend's pipeline counters: one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn f(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn sample_pack_read_errors_fail_instead_of_skipping() {
    use std::path::Path;
    use std::process::Command;

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("xcl-sample-path-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let missing = dir.join("missing-shaders.rpf");
    // Separate processes exercise the real setup path without racing environment variables
    // or reusing PACK's cached result. An unset sample needs no GPU, DXC or game data.
    let run = |path: Option<&Path>| {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", "retail_vs_ps_through_the_pso_hook_render_and_read_back", "--nocapture"]);
        cmd.env_remove("XCL_SAMPLE_RPF");
        if let Some(path) = path {
            cmd.env("XCL_SAMPLE_RPF", path);
        }
        cmd.output().unwrap()
    };
    let unset = run(None);
    let missing_output = run(Some(&missing));
    let directory_output = run(Some(&dir));
    std::fs::remove_dir_all(&dir).unwrap();

    let unset_stdout = String::from_utf8_lossy(&unset.stdout);
    assert!(unset.status.success(), "unset sample should skip: {unset:?}");
    assert!(unset_stdout.contains("running 1 test") && unset_stdout.contains("XCL_SAMPLE_RPF not set"));
    for (path, output) in [(&missing, missing_output), (&dir, directory_output)] {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "configured unreadable sample must fail: {output:?}");
        assert!(stdout.contains("running 1 test"), "{stdout}");
        assert!(stderr.contains("cannot read XCL_SAMPLE_RPF") && stderr.contains(&*path.to_string_lossy()), "{stderr}");
        assert!(!stdout.contains("XCL_SAMPLE_RPF not set"), "{stdout}");
    }
}

/// Console GS and NGG programs are drawn through a mesh pass: they need mesh shaders, which
/// not every adapter has (an integrated RDNA2 GPU reports none).
fn mesh_shaders(dev: &ID3D12Device) -> bool {
    let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    let ok = unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut o7 as *mut _ as _, std::mem::size_of_val(&o7) as u32) }.is_ok()
        && o7.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0;
    if !ok {
        println!("GPU without mesh shaders – skipping");
        xcl::gpu::recomp::runner::skip_test("no mesh shaders");
    }
    ok
}

/// Console shader containers by record name (as a title passes them in
/// D3D12_SHADER_BYTECODE) and the backend device. None: no sample pack / DXC / GPU.
/// `setup` holds `SERIAL` for the test: they share the backend device and its global PSO
/// counters.
fn setup(names: &[&str]) -> Option<(Vec<Vec<u8>>, ID3D12Device, std::sync::MutexGuard<'static, ()>)> {
    let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    static PACK: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    let data = PACK.get_or_init(|| {
        std::env::var_os("XCL_SAMPLE_RPF").map(|path| {
            std::fs::read(&path).unwrap_or_else(|error| {
                panic!("cannot read XCL_SAMPLE_RPF '{}': {error}", std::path::Path::new(&path).display())
            })
        })
    }).as_ref();
    let Some(data) = data else {
        println!("XCL_SAMPLE_RPF not set – skipping");
        return None;
    };
    if xcl::gpu::recomp::dxc::location().is_err() {
        println!("no DXC – skipping");
        xcl::gpu::recomp::runner::skip_test("no DXC");
        return None;
    }
    let recs = pack::scan(data, 0x40);
    let blobs: Vec<Vec<u8>> = names
        .iter()
        .map(|n| {
            let r = recs.iter().find(|r| r.name == *n).unwrap_or_else(|| panic!("no record {n}"));
            let b = pack::inflate(data, r).unwrap();
            assert!(xcl::gpu::xbox::is_xbox(&b), "{n}: expected a console container");
            b
        })
        .collect();
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        xcl::gpu::recomp::runner::skip_test("no GPU");
        return None;
    }
    println!("backend device: {}", real::stats().adapter);
    let raw = real::device_raw();
    Some((blobs, unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone(), serial))
}

/// CreateGraphicsPipelineState through the backend hook with the title's own (empty)
/// root signature and input layout; asserts the pipeline came from translated shaders.
fn pso_through_hook(dev: &ID3D12Device, vs: &[u8], ps: &[u8], layout: &[(&CStr, DXGI_FORMAT, u32)]) -> ID3D12PipelineState {
    // a pipeline built in the background behind a stand-in: the real one
    real_pso(&pso_through_hook_raw(dev, vs, ps, layout))
}

/// The real pipeline of `pso` (waits when it is a stand-in still being built).
fn real_pso(pso: &ID3D12PipelineState) -> ID3D12PipelineState {
    let real = xcl::hle::d3d12x::pso_async::resolve_wait(pso.as_raw() as usize) as *mut std::ffi::c_void;
    unsafe { ID3D12PipelineState::from_raw_borrowed(&real) }.unwrap().clone()
}

/// The pipeline the title gets back (a stand-in when its shaders were not compiled yet).
fn pso_through_hook_raw(dev: &ID3D12Device, vs: &[u8], ps: &[u8], layout: &[(&CStr, DXGI_FORMAT, u32)]) -> ID3D12PipelineState {
    let game_rs: ID3D12RootSignature = unsafe {
        let desc = D3D12_ROOT_SIGNATURE_DESC { Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, ..Default::default() };
        let mut ser = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    };
    let elems: Vec<D3D12_INPUT_ELEMENT_DESC> = layout
        .iter()
        .map(|(name, fmt, off)| D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::PCSTR(name.as_ptr() as *const u8),
            Format: *fmt,
            AlignedByteOffset: *off,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            ..Default::default()
        })
        .collect();
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(game_rs)),
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
        SampleMask: u32::MAX,
        InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: elems.len() as u32 },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = true.into();
    desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let before = real::stats();
    let pso: ID3D12PipelineState = unsafe { dev.CreateGraphicsPipelineState(&desc) }.expect("CreateGraphicsPipelineState through the backend hook");
    unsafe { std::mem::ManuallyDrop::drop(&mut desc.pRootSignature) };
    // built in the background behind a stand-in when its shaders were not compiled yet
    xcl::hle::d3d12x::pso_async::resolve_wait(pso.as_raw() as usize);
    let after = real::stats();
    assert_eq!(after.translated_pipelines, before.translated_pipelines + 1, "PSO was not built from the translated shaders: {:?}", after.translation_errors);
    assert_eq!(after.fallback_pipelines, before.fallback_pipelines, "fallback shaders were used");
    pso
}

/// `VS_Rage_TransformPosCpv` (POSITION · world 4×3 · view-projection 4×4, COLOR passed on;
/// s[8:9] = address of a table of two V#s) + `PS_Rage_Untextured` (interpolated COLOR →
/// SV_Target0 through f16): one triangle, interpolated colours checked pixel by pixel.
#[test]
fn retail_vs_ps_through_the_pso_hook_render_and_read_back() {
    let Some((blobs, dev, _serial)) = setup(&["VS_Rage_TransformPosCpv_f020b20e_Wrapped", "PS_Rage_Untextured_14f8529b_Wrapped"]) else { return };
    let pso = pso_through_hook(&dev, &blobs[0], &blobs[1], &UNTEXTURED_LAYOUT);
    check_untextured_triangle(&dev, &pso);
}

const UNTEXTURED_LAYOUT: [(&CStr, DXGI_FORMAT, u32); 2] = [(c"POSITION", DXGI_FORMAT_R32G32B32_FLOAT, 0), (c"COLOR", DXGI_FORMAT_R32G32B32A32_FLOAT, 12)];

/// Draw a triangle with the retail untextured VS / PS pipeline `pso` and check every pixel
/// against the same transform and interpolation on the CPU.
fn check_untextured_triangle(dev: &ID3D12Device, pso: &ID3D12PipelineState) {
    // the pipeline's own root signature (static samplers from the console code)
    let rs_raw = real::pso_root_signature(pso.as_raw() as usize).expect("root signature of the translated PSO") as *mut std::ffi::c_void;
    let rs = unsafe { ID3D12RootSignature::from_raw_borrowed(&rs_raw) }.unwrap().clone();

    // explicit inputs: user SGPR s[8:9] → descriptor table at 0x100 with two V#s
    let mut io = vec![0u32; 128];
    io[8] = 0x100;
    io[9] = 0;
    let mut mem = vec![0u32; 0xC00];
    let vsharp = |base: u32| [base, 0, 0x1000, 0];
    mem[0x100 / 4..0x100 / 4 + 4].copy_from_slice(&vsharp(0x1000)); // world (VS reads 0x00..0x3F)
    mem[0x120 / 4..0x120 / 4 + 4].copy_from_slice(&vsharp(0x2000)); // view-projection (0x160..0x19F)
    // world rows = coefficients of x, y, z, then translation: (0.5·x + 0.25, 0.5·y, z)
    let world: [[f32; 4]; 4] = [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.25, 0.0, 0.0, 0.0]];
    for (r, row) in world.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            mem[0x1000 / 4 + 4 * r + c] = f(*v);
        }
    }
    // view-projection: out.k = Σ m[4·i + k]·world_i + m[12 + k]; flips y
    let mut vp = [0.0f32; 16];
    vp[0] = 1.0;
    vp[5] = -1.0;
    vp[10] = 1.0;
    vp[15] = 1.0;
    for (i, v) in vp.iter().enumerate() {
        mem[(0x2000 + 0x160) / 4 + i] = f(*v);
    }
    let corners = [([-1.0f32, -1.0, 0.5], [1.0f32, 0.0, 0.0, 1.0]), ([-1.0, 1.0, 0.5], [0.0, 1.0, 0.0, 1.0]), ([1.0, -1.0, 0.5], [0.0, 0.0, 1.0, 0.5])];
    let mut vertices = Vec::new();
    for (p, c) in &corners {
        vertices.extend_from_slice(p);
        vertices.extend_from_slice(c);
    }
    let px = runner::draw_on(dev, pso, &rs, &vertices, 28, &io, &mem, SIZE, CLEAR).expect("draw");

    // expected: the same transform on the CPU → pixel-space triangle → interpolated colour
    let to_px = |p: [f32; 3]| {
        let (wx, wy) = (0.5 * p[0] + 0.25, 0.5 * p[1]);
        let (cx, cy) = (wx, -wy);
        ((cx + 1.0) * 0.5 * SIZE as f32, (1.0 - cy) * 0.5 * SIZE as f32)
    };
    let (a, b, c) = (to_px(corners[0].0), to_px(corners[1].0), to_px(corners[2].0));
    let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
    let (mut inside, mut outside) = (0, 0);
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let (qx, qy) = (x as f32 + 0.5, y as f32 + 0.5);
            let l1 = ((b.0 - qx) * (c.1 - qy) - (c.0 - qx) * (b.1 - qy)) / area;
            let l2 = ((c.0 - qx) * (a.1 - qy) - (a.0 - qx) * (c.1 - qy)) / area;
            let l3 = 1.0 - l1 - l2;
            let got = px[y * SIZE as usize + x];
            let m = l1.min(l2).min(l3);
            if m > 0.02 {
                let want: Vec<f32> = (0..4).map(|k| l1 * corners[0].1[k] + l2 * corners[1].1[k] + l3 * corners[2].1[k]).collect();
                let ok = (0..4).all(|k| (got[k] - want[k]).abs() < 0.01);
                assert!(ok, "pixel ({x},{y}) inside: {got:?}, expected {want:?}");
                inside += 1;
            } else if m < -0.02 {
                assert_eq!(got, CLEAR, "pixel ({x},{y}) outside the triangle");
                outside += 1;
            }
        }
    }
    println!("checked pixels: {inside} inside, {outside} outside; triangle {a:?} {b:?} {c:?}");
    assert!(inside > 400 && outside > 2500, "too few checked pixels: {inside} / {outside}");
}

/// Non-blocking pipeline creation with an empty DXIL cache: the call returns a stand-in at
/// once, the real pipeline is built in the background – and it renders exactly like one built
/// in the call; a mesh pipeline's two-pass parts are found through its stand-in once built.
#[test]
fn retail_pipelines_built_in_the_background_behind_a_stand_in() {
    use xcl::hle::d3d12x::pso_async;
    let Some((blobs, dev, _serial)) = setup(&["VS_Rage_TransformPosCpv_f020b20e_Wrapped", "PS_Rage_Untextured_14f8529b_Wrapped"]) else { return };
    /// an empty DXIL cache for this test only (restored also when an assertion fails)
    struct ColdCache(std::path::PathBuf);
    impl Drop for ColdCache {
        fn drop(&mut self) {
            std::env::remove_var("XCL_DXIL_CACHE");
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let cold = ColdCache(std::env::temp_dir().join(format!("xcl-async-pso-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&cold.0);
    std::env::set_var("XCL_DXIL_CACHE", &cold.0);
    let pso = pso_through_hook_raw(&dev, &blobs[0], &blobs[1], &UNTEXTURED_LAYOUT); // waits for the real one
    let raw = pso.as_raw() as usize;
    let real = pso_async::resolve(raw);
    assert_ne!(real, raw, "a stand-in, replaced by the real pipeline built in the background");
    assert!(real::pso_root_signature(raw).is_some(), "the stand-in resolves to the real pipeline's root signature");
    let real_pso = unsafe { ID3D12PipelineState::from_raw_borrowed(&(real as *mut std::ffi::c_void)) }.unwrap().clone();
    check_untextured_triangle(&dev, &real_pso);
    // the shaders are in the cache now: the next pipeline is built in the call
    let again = pso_through_hook(&dev, &blobs[0], &blobs[1], &UNTEXTURED_LAYOUT);
    assert!(!pso_async::is_pending(again.as_raw() as usize) && pso_async::resolve(again.as_raw() as usize) == again.as_raw() as usize);

    // a mesh pipeline (MS + PS of one pass) the same way
    if !mesh_shaders(&dev) {
        return;
    }
    let data = std::fs::read(std::env::var("XCL_SAMPLE_RPF").unwrap()).unwrap();
    let recs = pack::scan(&data, 0x40);
    let (ms, ps) = recs
        .iter()
        .filter(|r| r.name.starts_with("MS"))
        .find_map(|r| {
            let h = pass_hash(&r.name)?;
            let pr = recs.iter().find(|p| p.name.starts_with("PS_") && pass_hash(&p.name) == Some(h))?;
            Some((pack::inflate(&data, r)?.to_vec(), pack::inflate(&data, pr)?.to_vec()))
        })
        .unwrap();
    let game_rs: ID3D12RootSignature = unsafe {
        let mut ser = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    };
    #[repr(C, align(8))]
    struct Sub<T> {
        ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
        v: T,
    }
    #[repr(C)]
    struct Stream {
        rs: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        rt: Sub<D3D12_RT_FORMAT_ARRAY>,
    }
    let code = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
    let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
    rt.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    let mut st = Stream {
        rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: game_rs.as_raw() },
        ms: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, v: code(&ms) },
        ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, v: code(&ps) },
        rt: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rt },
    };
    let d2: ID3D12Device2 = dev.cast().unwrap();
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: std::mem::size_of::<Stream>(), pPipelineStateSubobjectStream: &st as *const _ as *mut _ };
    let t0 = std::time::Instant::now();
    let mesh: ID3D12PipelineState = unsafe { d2.CreatePipelineState(&desc) }.expect("mesh stand-in");
    let call = t0.elapsed();
    let raw = mesh.as_raw() as usize;
    // the title's description is gone (scribbled over); the worker builds from its own copy
    unsafe { std::ptr::write_bytes(&mut st as *mut Stream, 0xCD, 1) };
    assert_ne!(pso_async::resolve_wait(raw), raw);
    assert!(real::two_pass_of(raw).is_some(), "the two-pass parts through the stand-in");
    println!("mesh: the call returned after {call:?}, the real pipeline was ready after {:?}", t0.elapsed());
    // the title releases the stand-in: its real pipeline goes with it, the address is forgotten
    // (the tests here run one at a time: the count is this test's)
    let two_pass = real::two_pass_count();
    drop(mesh);
    assert_eq!(real::two_pass_count(), two_pass - 1, "the released pipeline's compute pass is gone");
    assert_eq!(pso_async::resolve(raw), raw);
    assert!(real::two_pass_of(raw).is_none());
}

/// Texel (x, y) of the two test textures, RGBA8.
fn texel_a(x: usize, y: usize) -> [u8; 4] {
    [(x * 4) as u8, (y * 4) as u8, if (x / 8 + y / 8) % 2 == 0 { 255 } else { 0 }, 255]
}

fn texel_b(x: usize, y: usize) -> [u8; 4] {
    [(255 - y * 4) as u8, 128, (x * 4) as u8, 200]
}

/// Console T# of a linear (SW_MODE 0) 64×64 8_8_8_8_UNORM 2D texture at `address`.
fn console_tsharp(address: u64, dst_sel: [u32; 4]) -> [u32; 8] {
    let (w1, h1) = (SIZE - 1, SIZE - 1);
    [
        (address >> 8) as u32,
        ((address >> 40) as u32 & 0xFF) | 56 << 20 | (w1 & 3) << 30,
        (w1 >> 2) | h1 << 14,
        dst_sel[0] | dst_sel[1] << 3 | dst_sel[2] << 6 | dst_sel[3] << 9 | 9 << 28,
        0,
        0,
        0,
        0,
    ]
}

/// `VS_Rage_TransformPosCpvTex` + `PS_Rage_Textured` (texture · vertex colour; the T# comes
/// from the descriptor table at s[8:9] + 0x20, the S# is built into the code). Two console
/// T#s → our decoder → D3D12 textures and views; one pipeline, two draws in one command list
/// whose only difference is the descriptor table (left half: texture A, right half:
/// texture B with BGRA swizzle). Checked texel by texel, then that the change took effect.
#[test]
fn retail_textured_ps_decoded_descriptors_and_a_descriptor_change_between_draws() {
    let Some((blobs, dev, _serial)) = setup(&["VS_Rage_TransformPosCpvTex_1702efec_Wrapped", "PS_Rage_Textured_1702efec_Wrapped"]) else { return };
    let pso = pso_through_hook(
        &dev,
        &blobs[0],
        &blobs[1],
        &[(c"POSITION", DXGI_FORMAT_R32G32B32_FLOAT, 0), (c"COLOR", DXGI_FORMAT_R32G32B32A32_FLOAT, 12), (c"TEXCOORD", DXGI_FORMAT_R32G32_FLOAT, 28)],
    );
    // the pipeline's own root signature (static samplers from the console code)
    let rs_raw = real::pso_root_signature(pso.as_raw() as usize).expect("root signature of the translated PSO") as *mut std::ffi::c_void;
    let rs = unsafe { ID3D12RootSignature::from_raw_borrowed(&rs_raw) }.unwrap().clone();

    // console memory: texel data at the addresses the T#s name, constants, two descriptor tables
    let (addr_a, addr_b) = (0x1_0000u64, 0x2_0000u64);
    let mut mem = vec![0u32; 0x2_4000 / 4];
    let bytes = |mem: &mut Vec<u32>, at: u64, texel: fn(usize, usize) -> [u8; 4]| {
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                mem[at as usize / 4 + y * SIZE as usize + x] = u32::from_le_bytes(texel(x, y));
            }
        }
    };
    bytes(&mut mem, addr_a, texel_a);
    bytes(&mut mem, addr_b, texel_b);
    let t_a = console_tsharp(addr_a, [4, 5, 6, 7]); // identity
    let t_b = console_tsharp(addr_b, [6, 5, 4, 7]); // r ← z, b ← x
    // the static sampler the PS builds in code: s[0:3] = 0x400, 0xC00000, 0x18F00000, 0
    let s_code = [0x400u32, 0xC0_0000, 0x18F0_0000, 0];

    // our decoder: T# → D3D12 resource + SRV (texel data read from the T#'s address), S# → sampler
    let mut textures = Vec::new();
    for t in [t_a, t_b] {
        let img = desc::image(&t).expect("T#");
        assert_eq!(img.swizzle_mode, 0, "the test uses linear textures");
        let (res, srv) = desc::image_d3d12(&img).expect("format DXGI");
        assert_eq!((res.Width, res.Height, res.Format), (SIZE as u64, SIZE, DXGI_FORMAT_R8G8B8A8_UNORM));
        let at = img.address as usize;
        let data: Vec<u8> = mem[at / 4..at / 4 + (SIZE * SIZE) as usize].iter().flat_map(|w| w.to_le_bytes()).collect();
        textures.push(runner::ViewTexture { res, srv, data, row_bytes: SIZE as usize * 4 });
    }
    let sampler = desc::sampler_d3d12(&s_code, false);
    assert_eq!((sampler.Filter, sampler.AddressU, sampler.MaxAnisotropy), (D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP, 4));

    // descriptor tables in the XCL shader ABI form: +0x00 world V#, +0x20 T#, +0x40 view-projection V#
    let vsharp = |base: u32| [base, 0, 0x1000, 0];
    for (table, t, heap_index) in [(0x100usize, t_a, 0u32), (0x200, t_b, 2)] {
        mem[table / 4..table / 4 + 4].copy_from_slice(&vsharp(0x1000));
        mem[(table + 0x20) / 4..(table + 0x20) / 4 + 8].copy_from_slice(&desc::xcl_image_words(&t, heap_index));
        mem[(table + 0x40) / 4..(table + 0x40) / 4 + 4].copy_from_slice(&vsharp(0x2000));
    }
    assert_eq!(desc::xcl_sampler_words(&s_code, 0), s_code, "the S# from the code points at sampler 0");
    // identity world and view-projection
    for (i, v) in [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0].iter().enumerate() {
        mem[0x1000 / 4 + i] = f(*v);
    }
    for (i, v) in [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0].iter().enumerate() {
        mem[(0x2000 + 0x160) / 4 + i] = f(*v);
    }
    // full-screen quad, white, uv = screen position → every pixel centre hits a texel centre
    let mut vertices = Vec::new();
    for (x, y) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        vertices.extend_from_slice(&[x, y, 0.5, 1.0, 1.0, 1.0, 1.0, (x + 1.0) * 0.5, (1.0 - y) * 0.5]);
    }
    let half = SIZE as i32 / 2;
    let user_data = |table: u32| {
        let mut io = vec![0u32; 128];
        io[8] = table;
        io
    };
    let draws = [
        runner::Draw { sgprs: user_data(0x100), scissor: [0, 0, half, SIZE as i32] },
        runner::Draw { sgprs: user_data(0x200), scissor: [half, 0, SIZE as i32, SIZE as i32] },
    ];
    let px = runner::draw_on_multi(&dev, &pso, &rs, &vertices, 36, &draws, &mem, &textures, &[sampler], SIZE, CLEAR).expect("draw");

    let unorm = |t: [u8; 4]| t.map(|c| c as f32 / 255.0);
    let (mut checked, mut changed) = (0, 0);
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let got = px[y * SIZE as usize + x];
            let a = unorm(texel_a(x, y));
            let b = unorm(texel_b(x, y));
            let want = if x < half as usize { a } else { [b[2], b[1], b[0], b[3]] };
            assert!((0..4).all(|k| (got[k] - want[k]).abs() < 0.005), "pixel ({x},{y}): {got:?}, expected {want:?}");
            checked += 1;
            // right half: the second draw saw the other descriptor, not texture A
            if x >= half as usize && (0..4).any(|k| (got[k] - a[k]).abs() > 0.05) {
                changed += 1;
            }
        }
    }
    let right = (SIZE * SIZE / 2) as usize;
    println!("checked pixels: {checked}; differing from texture A after the descriptor change: {changed} / {right}");
    assert!(changed > right * 9 / 10, "changing the descriptor between draw calls did not change the result: {changed} / {right}");
}

/// Retail legacy GS: `VS_VoxelizeGeneric` (ES variant) + `GS_VoxelizeGeneric` translated as
/// one compute shader (geometry mode) → records → mesh pass-through. The GS projects each
/// triangle onto the plane of its dominant normal axis (voxelisation); checked through
/// properties that follow from its code: 3 vertices / 1 triangle per input triangle,
/// TEXCOORD1 = (-1, -1, 1, 1), TEXCOORD3 = the unit normal, TEXCOORD0 = the axis (same for
/// a triangle, different for differently oriented triangles), position = the two other
/// coordinates (w = 1, z in [0, 1]), and pixels covered.
#[test]
fn retail_es_gs_voxelize_geometry_mode() {
    let Some((blobs, _dev, _serial)) = setup(&["VS_VoxelizeGeneric_6be01b34_Wrapped", "GS_VoxelizeGeneric_6be01b34_Wrapped"]) else { return };
    if !mesh_shaders(&_dev) {
        return;
    }
    use xcl::gpu::recomp::{self, dxc, IO_GS_PRIMS, IO_GS_VERTICES, MESH_HEADER};
    let (code, g) = recomp::geometry_program(&blobs[0], &blobs[1]).expect("ES + GS pair");
    assert_eq!((g.max_vertex_count, g.vertex_bytes()), (3, 64));
    let split_wave32 = recomp::pso::wave32_only_for_device(&_dev);
    let t = recomp::translate_geometry(&code, g.clone(), split_wave32, 1).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let rec = g.record();
    let ms = dxc::compile(&recomp::mesh_passthrough(&rec), "main", "ms_6_6", &[]).unwrap();
    let ps = dxc::compile("float4 main(float4 p : SV_Position) : SV_Target0 { return float4(1, 0, 0, 1); }", "main", "ps_6_6", &[]).unwrap();

    // user data: s[8:9] → X (GS output base, ·256) at +0xB8; s16 → GS table (V# at +0x920 →
    // scale at +0x40); s17 → ES table (V# at +0x60 → world 4×3); s19 = 0 (scale 1)
    let (p_tab, gs_tab, es_tab, world, scale, vin) = (0x100u32, 0x200u32, 0x1200u32, 0x2000u32, 0x2100u32, 0x3000u32);
    let mut mem = vec![0u32; 0x4000 / 4];
    let vsharp = |base: u32| [base, 0, 0x1000, 0];
    mem[(p_tab + 0xB8) as usize / 4] = 0;
    mem[(gs_tab + 0x920) as usize / 4..(gs_tab + 0x920) as usize / 4 + 4].copy_from_slice(&vsharp(scale));
    mem[(es_tab + 0x60) as usize / 4..(es_tab + 0x60) as usize / 4 + 4].copy_from_slice(&vsharp(world));
    for (i, v) in [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0].iter().enumerate() {
        mem[world as usize / 4 + i] = v.to_bits();
    }
    for (i, v) in [1.0f32, 1.0, 1.0, 0.0].iter().enumerate() {
        mem[(scale + 0x40) as usize / 4 + i] = v.to_bits();
    }
    // two triangles: in the plane z = 0.2 (normal z) and x = 0.3 (normal x)
    let tris = [[[-0.5f32, -0.5, 0.2], [0.5, -0.5, 0.2], [-0.5, 0.5, 0.2]], [[0.3, -0.4, -0.4], [0.3, 0.4, -0.4], [0.3, -0.4, 0.4]]];
    for (t, tri) in tris.iter().enumerate() {
        for (i, p) in tri.iter().enumerate() {
            let at = vin as usize / 4 + (t * 3 + i) * 4;
            mem[at..at + 4].copy_from_slice(&[p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), 1.0f32.to_bits()]);
        }
    }
    let mut io = vec![0u32; 128];
    io[8] = p_tab;
    io[16] = gs_tab;
    io[17] = es_tab;
    io[IO_GS_PRIMS] = 2;
    io[IO_GS_VERTICES] = vin;
    let (px, r) = runner::mesh_draw(&cs, &ms, &ps, 1, rec.record_dwords(), &io, &mem, 64, [0.0; 4]).expect("draw");

    let vd = rec.vert_dwords();
    let fl = |i: usize| f32::from_bits(r[i]);
    assert_eq!((r[0], r[1]), (6, 2), "3 vertices and 1 triangle per input triangle");
    let tri_at = MESH_HEADER + rec.max_vertices as usize * vd;
    let mut axes = Vec::new();
    for (t, tri) in tris.iter().enumerate() {
        let base = 3 * t;
        assert_eq!(r[tri_at + t], (base | (base + 1) << 10 | (base + 2) << 20) as u32, "triangle {t}: indices");
        // true normal and the two coordinates that are not the dominant axis
        let (e1, e2) = ([0, 1, 2].map(|k| tri[1][k] - tri[0][k]), [0, 1, 2].map(|k| tri[2][k] - tri[0][k]));
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let dom = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs())).unwrap();
        let others: Vec<usize> = (0..3).filter(|&k| k != dom).collect();
        let mut found = false;
        for i in 0..3 {
            let v = MESH_HEADER + (base + i) * vd;
            let pos = [fl(v), fl(v + 1), fl(v + 2), fl(v + 3)];
            let (axis, tc1, tc3) = (r[v + 4], [fl(v + 8), fl(v + 9), fl(v + 10), fl(v + 11)], [fl(v + 12), fl(v + 13), fl(v + 14)]);
            println!("triangle {t} vertex {i}: pos {pos:?} axis {axis} TEXCOORD1 {tc1:?} TEXCOORD3 {tc3:?}");
            assert_eq!(tc1, [-1.0, -1.0, 1.0, 1.0], "TEXCOORD1");
            let dot = (tc3[0] * n[0] + tc3[1] * n[1] + tc3[2] * n[2]) / len;
            let unit = (tc3[0] * tc3[0] + tc3[1] * tc3[1] + tc3[2] * tc3[2]).sqrt();
            assert!((unit - 1.0).abs() < 1e-3 && (dot.abs() - 1.0).abs() < 1e-3, "TEXCOORD3 = unit normal: {tc3:?} vs {n:?}");
            assert_eq!(pos[3], 1.0, "w");
            assert!((0.0..=1.0).contains(&pos[2]), "z w [0, 1]: {pos:?}");
            if i == 0 {
                axes.push(axis);
            } else {
                assert_eq!(axis, axes[t], "same axis within the triangle");
            }
            // position = the two non-dominant coordinates (any order, any sign)
            let p = tri[i];
            let ok = [(others[0], others[1]), (others[1], others[0])]
                .iter()
                .any(|&(a, b)| (pos[0].abs() - p[a].abs()).abs() < 1e-5 && (pos[1].abs() - p[b].abs()).abs() < 1e-5);
            found |= ok;
            assert!(ok, "position = projection onto the plane of the dominant axis: {pos:?} from {p:?} (axis {dom})");
            // exactly, as the code selects: z-dominant → (x, y), x-dominant → (z, y); depth = 0.5·d + 0.5
            let want = match dom {
                2 => [p[0], p[1], 0.5 * p[2] + 0.5, 1.0],
                _ => [p[2], p[1], 0.5 * p[0] + 0.5, 1.0],
            };
            assert_eq!(pos, want, "triangle {t} vertex {i}");
        }
        assert!(found);
    }
    assert_eq!(axes, vec![4, 0], "dominant axis: z → 4, x → 0");
    let covered = px.iter().filter(|p| p[0] == 1.0).count();
    println!("covered pixels: {covered}");
    assert!(covered > 300, "pass-through drew the GS output: {covered}");
}

/// Hash of a retail record name (`MS_VS_Transform_PositionOnly_6463b8c5_Wrapped` → 6463b8c5):
/// shaders of one effect pass share it.
fn pass_hash(name: &str) -> Option<&str> {
    let mut it = name.rsplit('_');
    it.next()?;
    it.next().filter(|h| h.len() == 8 && h.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Retail mesh pipelines through the CreatePipelineState hook: MS (NGG) + a PS of the same
/// effect pass become a two-pass pipeline (compute pass + pass-through MS with the translated
/// PS; D3D12 checks the MS → PS linkage). A sample of the pack (every `XCL_MS_EVERY`-th MS).
#[test]
fn retail_mesh_pipelines_through_the_hook() {
    let Some((_, dev, _serial)) = setup(&[]) else { return };
    if !mesh_shaders(&dev) {
        return;
    }
    let data = std::fs::read(std::env::var("XCL_SAMPLE_RPF").unwrap()).unwrap();
    let recs = pack::scan(&data, 0x40);
    let every: usize = std::env::var("XCL_MS_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
    let game_rs: ID3D12RootSignature = unsafe {
        let mut ser = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    };
    #[repr(C, align(8))]
    struct Sub<T> {
        ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
        v: T,
    }
    #[repr(C)]
    struct Stream {
        rs: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        rt: Sub<D3D12_RT_FORMAT_ARRAY>,
    }
    let (mut tried, mut ok) = (0, 0);
    let mut failed = Vec::new();
    for r in recs.iter().filter(|r| r.name.starts_with("MS")).step_by(every) {
        let Some(h) = pass_hash(&r.name) else { continue };
        let Some(pr) = recs.iter().find(|p| p.name.starts_with("PS_") && pass_hash(&p.name) == Some(h)) else { continue };
        let (ms, ps) = (pack::inflate(&data, r).unwrap(), pack::inflate(&data, pr).unwrap());
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
        let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
        rt.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let s = Stream {
            rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: game_rs.as_raw() },
            ms: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, v: code(&ms) },
            ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, v: code(&ps) },
            rt: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rt },
        };
        let d2: ID3D12Device2 = dev.cast().unwrap();
        tried += 1;
        let t0 = std::time::Instant::now();
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: std::mem::size_of::<Stream>(), pPipelineStateSubobjectStream: &s as *const _ as *mut _ };
        match unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&desc) } {
            Ok(p) if real::two_pass_of(real_pso(&p).as_raw() as usize).is_some() => ok += 1,
            Ok(_) => failed.push(format!("{} + {}: PSO without a compute pass", r.name, pr.name)),
            Err(e) => {
                failed.push(format!("{} + {}: {e}", r.name, pr.name));
                if std::env::var_os("XCL_MS_SIGS").is_some() {
                    let sig = |v: &[xcl::gpu::dxbc::SigElement]| v.iter().map(|e| format!("{}{}(sv{} t{} r{} m{:x})", e.name, e.index, e.system_value, e.component_type, e.register, e.mask)).collect::<Vec<_>>().join(" ");
                    println!("    MS wyj.: {}
    PS wej.: {}", sig(&xcl::gpu::dxbc::signatures(&ms).1), sig(&xcl::gpu::dxbc::signatures(&ps).0));
                }
            }
        }
        println!("  {:.1} s  {} + {}", t0.elapsed().as_secs_f64(), r.name, pr.name);
    }
    println!("MS + PS pipelines through the hook: {ok} / {tried}");
    for f in failed.iter().take(10) {
        println!("  {f}");
    }
    for e in real::stats().translation_errors.iter().take(10) {
        println!("  translation error: {}", e.lines().take(6).collect::<Vec<_>>().join(" | "));
    }
    assert!(tried >= 3 && ok == tried, "{} failed", failed.len());
}

/// Retail legacy GS pipeline through CreateGraphicsPipelineState: VS (ES variant) + GS
/// become a two-pass pipeline (no PS: depth-only voxelisation pass is fine too).
#[test]
fn retail_gs_pipeline_through_the_hook() {
    let Some((blobs, dev, _serial)) = setup(&["VS_VoxelizeGeneric_6be01b34_Wrapped", "GS_VoxelizeGeneric_6be01b34_Wrapped"]) else { return };
    if !mesh_shaders(&dev) {
        return;
    }
    let code = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
    let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        VS: code(&blobs[0]),
        GS: code(&blobs[1]),
        SampleMask: u32::MAX,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    let p: ID3D12PipelineState = unsafe { dev.CreateGraphicsPipelineState(&d) }.unwrap_or_else(|e| panic!("{e}; {:?}", real::stats().translation_errors));
    let p = real_pso(&p);
    let tp = real::two_pass_of(p.as_raw() as usize).expect("GS pipeline in two passes");
    assert_eq!((tp.kind, tp.record.max_vertices), ("GS", 63));
}
