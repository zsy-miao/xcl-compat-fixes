//! Memory windows from the descriptor heap (`Options::heap_windows`): the raw view a window gets
//! (`runner::raw_window_uav`) on a real 4 GiB reserved buffer.

use std::mem::{transmute_copy, ManuallyDrop};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

const GIB: u64 = 1 << 30;
const TILE: u64 = 0x1_0000;

fn buf(dev: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, uav: bool) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    unsafe { dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap() };
    r.unwrap()
}

fn transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn device() -> Option<(ID3D12Device, String)> {
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let sel = crate::gpu::adapter::Selector::from_env_or("auto");
        if sel == crate::gpu::adapter::Selector::Warp {
            return None;
        }
        let (a, desc) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next()?;
        let dev = crate::gpu::agility::create_device(&a, D3D_FEATURE_LEVEL_12_0).ok()?;
        Some((dev, crate::gpu::adapter::name(&desc)))
    }
}

/// The heap-window form (`Options::heap_windows`) on a real sparse window: a 4 GiB reserved
/// buffer seen through `runner::raw_window_uav` reaches its last tile, which the sparse windows
/// hand out (a reservation across two windows covers it), up to its last 32 bytes (the view
/// ends there: an older D3D12 runtime removes the device for a raw view that ends above).
#[test]
fn gpu_raw_window_view_reaches_its_top() {
    let Some((dev, name)) = device() else {
        super::runner::skip_test("no GPU");
        return;
    };
    if super::dxc::location().is_err() {
        super::runner::skip_test("no DXC");
        return;
    }
    // the shader indexes the descriptor heap (SM 6.6): without it the pipeline is refused
    // (E_INVALIDARG), as on Windows 10's inbox D3D12 – skipped like the runner's tests
    let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_6 };
    let sm_ok = unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as _, std::mem::size_of_val(&sm) as u32) };
    if sm_ok.is_err() || sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
        super::runner::skip_test(&crate::gpu::agility::format_no_sm66_message(&name, crate::gpu::agility::state()));
        return;
    }
    let top = (4 * GIB) - TILE; // the window's last tile
    let src = r#"
#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), UAV(u0)"
RWByteAddressBuffer outb : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint l : SV_GroupThreadID) {
    RWByteAddressBuffer w = ResourceDescriptorHeap[NonUniformResourceIndex(0u)];
    w.Store(0xFFFF0000u + 4u * l, 0x5A000000u | l);
    w.Store(4u * l, 0xA5000000u | l);
    if (l < 56u) w.Store(0xFFFFFF00u + 4u * l, 0x3C000000u | l);
    DeviceMemoryBarrier();
    outb.Store(8u * l, w.Load(0xFFFF0000u + 4u * l));
    outb.Store(8u * l + 4u, w.Load(4u * l));
    outb.Store(512u + 4u * l, l < 56u ? w.Load(0xFFFFFF00u + 4u * l) : 0u);
}
"#;
    unsafe {
        let dxil = super::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
        let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: transmute_copy(&rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                ..Default::default()
            })
            .unwrap();
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 4 * GIB,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut win: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut win) {
            // a software adapter (the CI's) makes no 4 GiB reserved buffer
            println!("{name}: no 4 GiB reserved buffer ({e})");
            super::runner::skip_test("no GPU");
            return;
        }
        let win = win.unwrap();
        let mut heap: Option<ID3D12Heap> = None;
        dev.CreateHeap(&D3D12_HEAP_DESC { SizeInBytes: 2 * TILE, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS }, &mut heap).unwrap();
        let heap = heap.unwrap();
        let coords = [0u64, top].map(|o| D3D12_TILED_RESOURCE_COORDINATE { X: (o / TILE) as u32, ..Default::default() });
        let regions = [D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() }; 2];
        let flags = [D3D12_TILE_RANGE_FLAG_NONE; 2];
        queue.UpdateTileMappings(&win, 2, Some(coords.as_ptr()), Some(regions.as_ptr()), &heap, 2, Some(flags.as_ptr()), Some([0u32, 1].as_ptr()), Some([1u32; 2].as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
        let dh: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
            .unwrap();
        dev.CreateUnorderedAccessView(&win, None::<&ID3D12Resource>, Some(&super::runner::raw_window_uav(4 * GIB)), dh.GetCPUDescriptorHandleForHeapStart());
        let out = buf(&dev, 768, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        let rb = buf(&dev, 768 + 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).unwrap();
        list.SetDescriptorHeaps(&[Some(dh.clone())]);
        list.SetComputeRootSignature(&rs);
        list.SetComputeRootUnorderedAccessView(0, out.GetGPUVirtualAddress());
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[
            transition(&win, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        list.CopyBufferRegion(&rb, 0, &out, 0, 768);
        list.CopyBufferRegion(&rb, 768, &win, top, 256);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 20_000).map_err(|e| format!("{e}")).unwrap();
        let mut q = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut q)).unwrap();
        let w = std::slice::from_raw_parts(q as *const u32, 256).to_vec();
        rb.Unmap(0, None);
        for l in 0..64usize {
            assert_eq!(w[2 * l], 0x5A00_0000 | l as u32, "{name}: lane {l}, the window's top tile through the heap view");
            assert_eq!(w[2 * l + 1], 0xA500_0000 | l as u32, "{name}: lane {l}, the window's first tile");
            assert_eq!(w[192 + l], 0x5A00_0000 | l as u32, "{name}: lane {l}, the top tile in memory");
            if l < 56 {
                assert_eq!(w[128 + l], 0x3C00_0000 | l as u32, "{name}: lane {l}, the window's last dwords");
            }
        }
    }
}

fn gpu_kernel_ready() -> bool {
    if super::runner::adapter().is_err() {
        super::runner::skip_test("no GPU");
        return false;
    }
    if super::runner::wave64_programs_supported() != Ok(true) {
        super::runner::skip_test("no wave64");
        return false;
    }
    if super::dxc::location().is_err() {
        super::runner::skip_test("no DXC");
        return false;
    }
    true
}

fn atomic_int64_on_heap_supported() -> bool {
    let Some((dev, _)) = device() else {
        return false;
    };
    let mut o11 = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &mut o11 as *mut _ as _, std::mem::size_of_val(&o11) as u32).is_ok() && o11.AtomicInt64OnDescriptorHeapResourceSupported.as_bool() }
}

fn compile_variant(code: &[u8], direct: bool, heap: bool, mem_windows: u32) -> Vec<u8> {
    let mut opt = super::runner::options(super::Mode::Kernel);
    opt.mem_windows = mem_windows;
    opt.heap_windows = heap;
    if direct {
        let mw = if heap && mem_windows > 1 { mem_windows | super::HEAP_WINDOWS } else { mem_windows };
        let ko = crate::gpu::dxil::kernel::KernelOptions { wave64: opt.wave64, split: opt.split_wave64, mem_windows: mw, sampler_base: 0 };
        crate::gpu::dxil::kernel::translate(code, ko).unwrap_or_else(|e| panic!("direct heap={heap}: {e}"))
    } else {
        let t = super::translate(code, &opt).unwrap_or_else(|e| panic!("hlsl heap={heap}: {e}"));
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("dxc heap={heap}: {e}\n{}", t.hlsl))
    }
}

/// Divergent per-lane window indexing (`NonUniformResourceIndex`): in a single wave, lanes hit
/// window 0 (`l % 3 == 0`), window 1 (`l % 3 == 1`), and out-of-UMA (`l % 3 == 2`, aliasing w0
/// under `& 3`) simultaneously on loads, stores, and atomics. HLSL and direct DXIL must match
/// the switch form bit-for-bit on both registers and memory.
#[test]
fn gpu_heap_windows_divergent_lanes_match_switch_form() {
    if !gpu_kernel_ready() {
        return;
    }
    let e = |n: &str, enc: &str, f: &[(&str, u64)]| crate::gpu::rdna2::encode(n, enc, f, None).unwrap();
    let mut code = Vec::new();
    code.extend(e("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 4), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)]));
    code.extend(e("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 2), ("DATA", 5), ("SADDR", 0x7D), ("OFFSET", 0)]));
    code.extend(e("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 8), ("ADDR", 6), ("DATA", 9), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]));
    code.extend(e("S_ENDPGM", "ENC_SOPP", &[]));

    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    let base_hi = (w0 >> 32) as u32;
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };

    const CANARY: u32 = 0xCAFE_BABE;
    let mut st = super::runner::KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let hi = match l % 3 {
            0 => base_hi,
            1 => base_hi + 1,
            _ => base_hi + 4, // OOB, shares low bits with w0
        };
        r[0] = (0x1000 + 4 * l) as u32;
        r[1] = hi;
        r[2] = (0x2000 + 4 * l) as u32;
        r[3] = hi;
        r[4] = 0x7777_7777;
        r[5] = 0xA000_0000 | l as u32;
        r[6] = 0x3000;
        r[7] = hi;
        r[8] = 0x7777_7777;
        r[9] = 1;
    }

    let mut baseline: Option<Vec<u32>> = None;
    for (direct, heap) in [(false, false), (false, true), (true, false), (true, true)] {
        for l in 0..64u64 {
            *word(w0 + 0x1000 + 4 * l) = 0x1000 + l as u32;
            *word(w1 + 0x1000 + 4 * l) = 0x2000 + l as u32;
            *word(w0 + 0x2000 + 4 * l) = CANARY;
            *word(w1 + 0x2000 + 4 * l) = CANARY;
        }
        *word(w0 + 0x3000) = 0;
        *word(w1 + 0x3000) = 0;

        let dxil = compile_variant(&code, direct, heap, 2);
        let res = super::runner::run_in_arena(&dxil, &st, &arena).unwrap_or_else(|err| panic!("direct={direct} heap={heap}: {err}"));
        let loads: Vec<u32> = (0..64).map(|l| res.vgpr[l][4]).collect();
        for l in 0..64usize {
            let want_load = match l % 3 {
                0 => 0x1000 + l as u32,
                1 => 0x2000 + l as u32,
                _ => 0,
            };
            assert_eq!(loads[l], want_load, "direct={direct} heap={heap} lane {l}: divergent load");
            let w0_store = *word(w0 + 0x2000 + 4 * l as u64);
            let w1_store = *word(w1 + 0x2000 + 4 * l as u64);
            match l % 3 {
                0 => {
                    assert_eq!(w0_store, 0xA000_0000 | l as u32, "direct={direct} heap={heap} lane {l}: w0 store");
                    assert_eq!(w1_store, CANARY, "direct={direct} heap={heap} lane {l}: w1 untouched");
                }
                1 => {
                    assert_eq!(w1_store, 0xA000_0000 | l as u32, "direct={direct} heap={heap} lane {l}: w1 store");
                    assert_eq!(w0_store, CANARY, "direct={direct} heap={heap} lane {l}: w0 untouched");
                }
                _ => {
                    assert_eq!(w0_store, CANARY, "direct={direct} heap={heap} lane {l}: OOB store must not clobber w0");
                    assert_eq!(w1_store, CANARY, "direct={direct} heap={heap} lane {l}: OOB store must not clobber w1");
                    assert_eq!(res.vgpr[l][8], 0, "direct={direct} heap={heap} lane {l}: OOB atomic returns 0");
                }
            }
        }
        assert_eq!(*word(w0 + 0x3000), 22, "direct={direct} heap={heap}: w0 atomic count (lanes 0,3,..,63)");
        assert_eq!(*word(w1 + 0x3000), 21, "direct={direct} heap={heap}: w1 atomic count (lanes 1,4,..,61)");
        match &baseline {
            Some(b) => assert_eq!(&loads, b, "direct={direct} heap={heap}: must match switch baseline"),
            None => baseline = Some(loads),
        }
    }
}

/// 64-bit atomics (`GLOBAL_ATOMIC_ADD_X2`, `GLOBAL_ATOMIC_CMPSWAP_X2`) on heap windows:
/// in-bounds window-1 operations carry across the 32-bit boundary and swap cleanly, while
/// out-of-UMA 64-bit atomics return 0 and leave live window canaries untouched.
#[test]
fn gpu_heap_windows_64bit_atomics_match_switch_form() {
    if !gpu_kernel_ready() {
        return;
    }
    if !atomic_int64_on_heap_supported() {
        super::runner::skip_test("no 64-bit heap atomics");
        return;
    }
    let e = |n: &str, enc: &str, f: &[(&str, u64)]| crate::gpu::rdna2::encode(n, enc, f, None).unwrap();
    let mut code = Vec::new();
    // 1. In-bounds w1 + 0x1000: 64-bit atomic add (+1 per lane) across 32-bit carry boundary
    code.extend(e("GLOBAL_ATOMIC_ADD_X2", "ENC_FLAT_GLBL", &[("VDST", 10), ("ADDR", 0), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]));
    // 2. In-bounds w1 + 0x1010: 64-bit cmpswap
    code.extend(e("GLOBAL_ATOMIC_CMPSWAP_X2", "ENC_FLAT_GLBL", &[("VDST", 12), ("ADDR", 4), ("DATA", 20), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]));
    // 3. Out-of-UMA (base_hi + 5, aliases w1 + 0x1020 under & 3): 64-bit atomic add dropped
    code.extend(e("GLOBAL_ATOMIC_ADD_X2", "ENC_FLAT_GLBL", &[("VDST", 14), ("ADDR", 6), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]));
    // 4. Out-of-UMA (base_hi + 4, aliases w0 + 0x1020 under & 3): 64-bit cmpswap dropped
    code.extend(e("GLOBAL_ATOMIC_CMPSWAP_X2", "ENC_FLAT_GLBL", &[("VDST", 16), ("ADDR", 8), ("DATA", 20), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]));
    code.extend(e("S_ENDPGM", "ENC_SOPP", &[]));

    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    let base_hi = (w0 >> 32) as u32;
    let qword = |a: u64| unsafe { &mut *(a as *mut u64) };

    const INIT_CAS: u64 = 0x1122_3344_5566_7788;
    const NEW_CAS: u64 = 0x99AA_BBCC_DDEE_FF00;
    const CANARY64: u64 = 0xDEAD_BEEF_CAFE_BABE;

    let mut st = super::runner::KernelState::default();
    for r in st.vgpr.iter_mut() {
        r[0] = 0x1000;
        r[1] = base_hi + 1;
        r[2] = 1;
        r[3] = 0;
        r[4] = 0x1010;
        r[5] = base_hi + 1;
        r[6] = 0x1020;
        r[7] = base_hi + 5;
        r[8] = 0x1020;
        r[9] = base_hi + 4;
        r[14] = 0x7777_7777;
        r[15] = 0x7777_7777;
        r[16] = 0x7777_7777;
        r[17] = 0x7777_7777;
        r[20] = NEW_CAS as u32;
        r[21] = (NEW_CAS >> 32) as u32;
        r[22] = INIT_CAS as u32;
        r[23] = (INIT_CAS >> 32) as u32;
    }

    for (direct, heap) in [(false, false), (false, true), (true, false), (true, true)] {
        *qword(w1 + 0x1000) = 0x0000_0000_FFFF_FFF0;
        *qword(w1 + 0x1010) = INIT_CAS;
        *qword(w1 + 0x1020) = CANARY64;
        *qword(w0 + 0x1020) = CANARY64;

        let dxil = compile_variant(&code, direct, heap, 2);
        let res = super::runner::run_in_arena(&dxil, &st, &arena).unwrap_or_else(|err| panic!("direct={direct} heap={heap}: {err}"));

        assert_eq!(*qword(w1 + 0x1000), 0x0000_0001_0000_0030, "direct={direct} heap={heap}: 64-bit add carry in w1");
        assert_eq!(*qword(w1 + 0x1010), NEW_CAS, "direct={direct} heap={heap}: 64-bit cmpswap in w1");
        let cas_winners = res.vgpr.iter().filter(|v| (v[12] as u64 | ((v[13] as u64) << 32)) == INIT_CAS).count();
        assert_eq!(cas_winners, 1, "direct={direct} heap={heap}: exactly one lane wins 64-bit cmpswap");
        assert_eq!(*qword(w1 + 0x1020), CANARY64, "direct={direct} heap={heap}: OOB 64-bit add must not touch w1");
        assert_eq!(*qword(w0 + 0x1020), CANARY64, "direct={direct} heap={heap}: OOB 64-bit cmpswap must not touch w0");
        assert_eq!((res.vgpr[0][14], res.vgpr[0][15]), (0, 0), "direct={direct} heap={heap}: OOB 64-bit add returns 0");
        assert_eq!((res.vgpr[0][16], res.vgpr[0][17]), (0, 0), "direct={direct} heap={heap}: OOB 64-bit cmpswap returns 0");
    }
}

/// Multi-window `HEAP_WINDOWS` translation and DXIL validation across graphics stages (VS, PS, MS)
/// in both direct DXIL and HLSL+DXC (for 4 and 16 windows).
#[test]
fn heap_windows_graphics_stages_validate() {
    use crate::gpu::dxbc::SigElement;
    use crate::gpu::dxil::kernel::{translate_mesh, translate_pixel, translate_vertex, KernelOptions};
    use crate::gpu::recomp::{Mode, MsInterface, Options, PsInterface, VsInterface, HEAP_WINDOWS};

    let sig = |name: &str, sv: u32, reg: u32| SigElement { name: name.into(), index: 0, system_value: sv, component_type: 3, register: reg, mask: 0xF, rw_mask: 0xF, stream: 0 };
    let e = |n: &str, enc: &str, f: &[(&str, u64)]| crate::gpu::rdna2::encode(n, enc, f, None).unwrap();

    // Minimal shader that performs a global memory load (exercising heap-window accessors) and exports.
    let mut vs_ps_code = Vec::new();
    vs_ps_code.extend(e("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)]));
    vs_ps_code.extend(e("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 2), ("VSRC1", 2), ("VSRC2", 2), ("VSRC3", 2), ("DONE", 1)]));
    vs_ps_code.extend(e("S_ENDPGM", "ENC_SOPP", &[]));

    let mut ps_code = Vec::new();
    ps_code.extend(e("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)]));
    ps_code.extend(e("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 2), ("VSRC1", 2), ("VSRC2", 2), ("VSRC3", 2), ("DONE", 1), ("VM", 1)]));
    ps_code.extend(e("S_ENDPGM", "ENC_SOPP", &[]));

    let vs = VsInterface { inputs: vec![], outputs: vec![sig("SV_Position", 1, 0)], attr_base: 4, params: None };
    let ps = PsInterface { inputs: vec![sig("SV_Position", 1, 0)], outputs: vec![sig("SV_Target", 64, 0)], input_ena: 0, attrs: None };
    let ms = MsInterface { outputs: vec![sig("SV_Position", 1, 0)], threads: 64, max_vertices: 3, max_primitives: 1 };

    let has_dxc = super::dxc::location().is_ok();
    for windows in [4u32, crate::gpu::uma::MAX_HEAP_WINDOWS as u32] {
        let ko = KernelOptions { wave64: true, split: false, mem_windows: windows | HEAP_WINDOWS, sampler_base: 0 };
        translate_vertex(&vs_ps_code, ko, &vs).unwrap_or_else(|err| panic!("direct VS {windows}w: {err}"));
        translate_pixel(&ps_code, ko, &ps).unwrap_or_else(|err| panic!("direct PS {windows}w: {err}"));
        translate_mesh(&vs_ps_code, ko, &ms).unwrap_or_else(|err| panic!("direct MS {windows}w: {err}"));

        for (label, code, mode, target) in [
            ("VS", &vs_ps_code, Mode::Vertex(vs.clone()), "vs_6_6"),
            ("PS", &ps_code, Mode::Pixel(ps.clone()), "ps_6_6"),
            ("MS", &vs_ps_code, Mode::Mesh(ms.clone()), "cs_6_6"),
        ] {
            let opt = Options { mode, mem_windows: windows, heap_windows: true, ..Default::default() };
            let t = super::translate(code, &opt).unwrap_or_else(|err| panic!("hlsl {label} {windows}w: {err}"));
            assert!(t.hlsl.contains("ResourceDescriptorHeap["), "hlsl {label} {windows}w must use heap windows");
            if has_dxc {
                super::dxc::compile(&t.hlsl, "main", target, &[]).unwrap_or_else(|err| panic!("dxc {label} {windows}w: {err}\n{}", t.hlsl));
            }
        }
    }
}
