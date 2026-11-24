//! Experiment E1 (memory aperture): can one binding reach a reserved buffer far above 4 GiB?
//!
//! The memory windows are four root UAVs of 4 GiB each, because a `ByteAddressBuffer` offset
//! is a 32-bit `uint`. A *structured* view with a 2048-byte row addresses
//! `row · 2048 + byte`: 2^32 rows reach 8 TiB through one binding, so a shader could reach all
//! of title memory without a per-access `switch` over windows. This test checks that the
//! drivers actually compute that address beyond 4 GiB, through a root UAV and through a
//! descriptor-heap view, on a 16 GiB reserved buffer with tiles mapped at 0, 4 GiB + 64 KiB,
//! 13 GiB and the last 64 KiB. A tile at 64 KiB (where a 32-bit truncation of 4 GiB + 64 KiB
//! would land) must stay untouched.

use std::mem::{transmute_copy, ManuallyDrop};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

const GIB: u64 = 1 << 30;
const TILE: u64 = 0x1_0000;
/// aperture size
const SIZE: u64 = 16 * GIB;
/// tiles written by the shader (byte offsets in the aperture)
const TILES: [u64; 4] = [0, 4 * GIB + TILE, 13 * GIB, SIZE - TILE];
/// a 32-bit truncation of `TILES[1]` lands here: must stay zero
const ALIAS: u64 = TILE;
/// dwords per lane and tile the shader writes (one every 4 KiB of the tile)
const STEPS: u64 = 16;
const LANES: u64 = 64;

fn shader(heap: bool) -> String {
    let (decl, get) = if heap {
        ("", "RWStructuredBuffer<XclRow> ap = ResourceDescriptorHeap[0];")
    } else {
        ("RWStructuredBuffer<XclRow> ap : register(u0);", "")
    };
    format!(
        r#"
#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), UAV(u0), UAV(u1), RootConstants(num32BitConstants=4, b0)"
struct XclRow {{ uint d[512]; }};
{decl}
RWByteAddressBuffer outb : register(u1);
cbuffer C : register(b0) {{ uint pass_; uint probe; uint pad0; uint pad1; }};
// (hi, lo) of a 64-bit aperture offset → row and byte
uint xcl_row(uint hi, uint lo) {{ return (hi << 21) | (lo >> 11); }}
static const uint THI[4] = {{ {h0}u, {h1}u, {h2}u, {h3}u }};
static const uint TLO[4] = {{ {l0}u, {l1}u, {l2}u, {l3}u }};
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint l : SV_GroupThreadID) {{
    {get}
    for (uint t = 0; t < 4; t++) {{
        for (uint j = 0; j < {steps}u; j++) {{
            uint lo = TLO[t] + j * 4096u + l * 4u;
            uint row = xcl_row(THI[t], lo);
            uint byte = lo & 2047u;
            uint v = 0x80000000u | (t << 24) | (j << 16) | l;
            if (pass_ == 0u) {{
                ap[row].d[byte >> 2] = v;
            }} else {{
                outb.Store(((t * {steps}u + j) * 64u + l) * 4u, ap[row].d[byte >> 2]);
            }}
        }}
    }}
    if (pass_ == 1u && l == 0u) {{
        // probe: a dword past the end of row 0 of tile 0 (d[512] = row 1, d[0]?)
        outb.Store(4u * 4u * {steps}u * 64u, ap[0].d[511u + probe]);
    }}
}}
"#,
        h0 = TILES[0] >> 32,
        h1 = TILES[1] >> 32,
        h2 = TILES[2] >> 32,
        h3 = TILES[3] >> 32,
        l0 = TILES[0] as u32,
        l1 = TILES[1] as u32,
        l2 = TILES[2] as u32,
        l3 = TILES[3] as u32,
        steps = STEPS,
    )
}

fn expected(t: u64, j: u64, l: u64) -> u32 {
    0x8000_0000 | (t << 24 | j << 16 | l) as u32
}

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

fn barrier_uav(r: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: unsafe { transmute_copy(r) } }) },
    }
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

fn aperture_fits(tier: D3D12_TILED_RESOURCES_TIER, bits: u32, bytes: u64) -> bool {
    tier != D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED && bytes != 0 && (bits >= 64 || (bytes - 1) >> bits == 0)
}

fn aperture_device(bytes: u64) -> Option<(ID3D12Device, String)> {
    let Some((dev, name)) = device() else {
        super::runner::skip_test("no GPU");
        return None;
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    let mut va = D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT::default();
    unsafe {
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut options as *mut _ as _, std::mem::size_of_val(&options) as u32).unwrap();
        dev.CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &mut va as *mut _ as _, std::mem::size_of_val(&va) as u32).unwrap();
    }
    println!("{name}: tiled resources {:?}, {} GPU VA bits per resource; aperture {bytes} bytes", options.TiledResourcesTier, va.MaxGPUVirtualAddressBitsPerResource);
    if !aperture_fits(options.TiledResourcesTier, va.MaxGPUVirtualAddressBitsPerResource, bytes) {
        super::runner::skip_test("no tiled-resource address space for this aperture");
        return None;
    }
    Some((dev, name))
}

// Only failure to reserve the aperture itself is a capacity skip. Invalid descriptors,
// device removal and every later allocation or shader/readback failure still fail the test.
fn aperture_reserved(result: windows::core::Result<()>) -> windows::core::Result<bool> {
    match result {
        Ok(()) => Ok(true),
        Err(e) if e.code() == windows::Win32::Foundation::E_OUTOFMEMORY => Ok(false),
        Err(e) => Err(e),
    }
}

#[test]
fn aperture_capability_checks_include_the_last_byte() {
    assert!(!aperture_fits(D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED, 64, 4 * GIB));
    assert!(!aperture_fits(D3D12_TILED_RESOURCES_TIER_2, 31, 4 * GIB));
    assert!(aperture_fits(D3D12_TILED_RESOURCES_TIER_2, 32, 4 * GIB));
    assert!(!aperture_fits(D3D12_TILED_RESOURCES_TIER_2, 32, 4 * GIB + 1));
    assert!(!aperture_fits(D3D12_TILED_RESOURCES_TIER_2, 33, SIZE));
    assert!(aperture_fits(D3D12_TILED_RESOURCES_TIER_2, 34, SIZE));
    assert!(!aperture_fits(D3D12_TILED_RESOURCES_TIER_2, 64, 0));
    assert!(aperture_fits(D3D12_TILED_RESOURCES_TIER_2, 64, u64::MAX));
}

#[test]
fn aperture_reservation_only_skips_out_of_memory() {
    use windows::Win32::Foundation::{E_INVALIDARG, E_OUTOFMEMORY};
    assert!(aperture_reserved(Ok(())).unwrap());
    assert!(!aperture_reserved(Err(windows::core::Error::from_hresult(E_OUTOFMEMORY))).unwrap());
    for code in [E_INVALIDARG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET] {
        assert_eq!(aperture_reserved(Err(windows::core::Error::from_hresult(code))).unwrap_err().code(), code);
    }
}

/// One run (root UAV or heap view): the words read back through the shader and the tiles'
/// contents copied straight out of the aperture.
unsafe fn run(dev: &ID3D12Device, heap_view: bool) -> Result<(Vec<u32>, Vec<Vec<u32>>), String> {
    let dxil = super::dxc::compile(&shader(heap_view), "main", "cs_6_6", &[])?;
    let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).map_err(|e| format!("root signature: {e}"))?;
    let pso: ID3D12PipelineState = dev
        .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&rs),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
            ..Default::default()
        })
        .map_err(|e| format!("pipeline: {e}"))?;
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(|e| e.to_string())?;
    // the aperture: a 16 GiB reserved buffer
    let mut ap: Option<ID3D12Resource> = None;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SIZE,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut ap).map_err(|e| format!("16 GiB reserved buffer: {e}"))?;
    let ap = ap.unwrap();
    let mapped: Vec<u64> = TILES.iter().copied().chain([ALIAS]).collect();
    let mut tiles: Option<ID3D12Heap> = None;
    dev.CreateHeap(
        &D3D12_HEAP_DESC {
            SizeInBytes: TILE * mapped.len() as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        },
        &mut tiles,
    )
    .map_err(|e| e.to_string())?;
    let tiles = tiles.unwrap();
    let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = mapped.iter().map(|&o| D3D12_TILED_RESOURCE_COORDINATE { X: (o / TILE) as u32, ..Default::default() }).collect();
    let regions: Vec<D3D12_TILE_REGION_SIZE> = mapped.iter().map(|_| D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() }).collect();
    let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; mapped.len()];
    let offsets: Vec<u32> = (0..mapped.len() as u32).collect();
    let counts = vec![1u32; mapped.len()];
    queue.UpdateTileMappings(&ap, coords.len() as u32, Some(coords.as_ptr()), Some(regions.as_ptr()), &tiles, flags.len() as u32, Some(flags.as_ptr()), Some(offsets.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);

    let out_words = 4 * STEPS * LANES + 1;
    let out = buf(dev, out_words * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true);
    let rb_bytes = out_words * 4 + TILE * mapped.len() as u64;
    let rb = buf(dev, rb_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
    // shader-visible heap: [0] = the aperture as a structured UAV of 2048-byte rows
    let dh: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
        .map_err(|e| e.to_string())?;
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: (SIZE / 2048) as u32, StructureByteStride: 2048, CounterOffsetInBytes: 0, Flags: D3D12_BUFFER_UAV_FLAG_NONE },
        },
    };
    dev.CreateUnorderedAccessView(&ap, None::<&ID3D12Resource>, Some(&uav), dh.GetCPUDescriptorHandleForHeapStart());

    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(|e| e.to_string())?;
    // fresh tiles hold whatever the heap had: clear them first (copy from a zeroed buffer)
    let zero = buf(dev, TILE, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
    let mut p = std::ptr::null_mut();
    zero.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
    std::ptr::write_bytes(p as *mut u8, 0, TILE as usize);
    zero.Unmap(0, None);
    for &o in &mapped {
        list.CopyBufferRegion(&ap, o, &zero, 0, TILE);
    }
    list.ResourceBarrier(&[
        transition(&ap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    ]);
    list.SetDescriptorHeaps(&[Some(dh.clone())]);
    list.SetComputeRootSignature(&rs);
    list.SetComputeRootUnorderedAccessView(0, ap.GetGPUVirtualAddress());
    list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
    for pass in 0..2u32 {
        list.SetComputeRoot32BitConstants(2, 2, [pass, 1u32].as_ptr() as _, 0);
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[barrier_uav(&ap), barrier_uav(&out)]);
    }
    list.ResourceBarrier(&[
        transition(&ap, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
    ]);
    list.CopyBufferRegion(&rb, 0, &out, 0, out_words * 4);
    for (i, &o) in mapped.iter().enumerate() {
        list.CopyBufferRegion(&rb, out_words * 4 + i as u64 * TILE, &ap, o, TILE);
    }
    list.Close().map_err(|e| e.to_string())?;
    queue.ExecuteCommandLists(&[Some(list.cast().map_err(|e| e.to_string())?)]);
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| e.to_string())?;
    queue.Signal(&fence, 1).map_err(|e| e.to_string())?;
    crate::gpu::fence::wait(&fence, 1, dev, None, 20_000).map_err(|e| format!("{e}"))?;
    let mut q = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut q)).map_err(|e| e.to_string())?;
    let all = std::slice::from_raw_parts(q as *const u32, (rb_bytes / 4) as usize).to_vec();
    rb.Unmap(0, None);
    let read = all[..out_words as usize].to_vec();
    let tiles = (0..mapped.len()).map(|i| all[out_words as usize + i * (TILE as usize / 4)..][..TILE as usize / 4].to_vec()).collect();
    Ok((read, tiles))
}

/// Not an assertion: what the GPU does (RTX 3090, driver 616.92: both bindings fail above
/// 4 GiB – the structured row · stride wraps at 2^32 and a heap view's byte size is 32 bits –
/// so a single aperture binding cannot replace the windows; `gpu_raw_window_view_reaches_its_top`
/// is the form XCL uses).
#[test]
fn gpu_aperture_structured_rows_above_4_gib() {
    let Some((dev, name)) = device() else {
        println!("no GPU – skipping");
        return;
    };
    if super::dxc::location().is_err() {
        println!("no DXC – skipping");
        return;
    }
    let mut va = D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &mut va as *mut _ as _, std::mem::size_of_val(&va) as u32).unwrap() };
    println!("{name}: GPU VA bits per resource {}, per process {}", va.MaxGPUVirtualAddressBitsPerResource, va.MaxGPUVirtualAddressBitsPerProcess);
    let mut fails = Vec::new();
    for heap_view in [false, true] {
        let how = if heap_view { "heap view" } else { "root UAV" };
        let (read, tiles) = match unsafe { run(&dev, heap_view) } {
            Ok(r) => r,
            Err(e) => {
                fails.push(format!("{how}: {e}"));
                continue;
            }
        };
        let mut bad = 0;
        for t in 0..4u64 {
            for j in 0..STEPS {
                for l in 0..LANES {
                    let want = expected(t, j, l);
                    let got_read = read[((t * STEPS + j) * LANES + l) as usize];
                    let got_mem = tiles[t as usize][((j * 4096 + l * 4) / 4) as usize];
                    if got_read != want || got_mem != want {
                        if bad < 5 {
                            println!("{how}: tile {t} step {j} lane {l}: read {got_read:#x}, memory {got_mem:#x}, want {want:#x}");
                        }
                        bad += 1;
                    }
                }
            }
        }
        let alias_dirty = tiles[4].iter().filter(|&&w| w != 0).count();
        let probe = read[(4 * STEPS * LANES) as usize];
        println!(
            "{how}: {bad} wrong words of {}, alias tile dirty words {alias_dirty}; d[512] of row 0 reads {probe:#x} (row 1 d[0] = {:#x})",
            4 * STEPS * LANES,
            tiles[0][512]
        );
        if bad > 0 || alias_dirty > 0 {
            fails.push(format!("{how}: {bad} wrong words, {alias_dirty} words in the alias tile"));
        }
    }
    println!("{name}: structured aperture {}", if fails.is_empty() { "works".to_string() } else { format!("fails: {fails:?}") });
}

/// The heap-window form (`Options::heap_windows`) on a real sparse window: a 4 GiB reserved
/// buffer seen through `runner::raw_window_uav` reaches the original last usable tile. The
/// official `heap_window_tests` additionally cover the widened view's final tile.
#[test]
fn gpu_raw_window_view_reaches_its_top() {
    let Some((dev, name)) = aperture_device(4 * GIB) else { return };
    if super::dxc::location().is_err() {
        super::runner::skip_test("no DXC");
        return;
    }
    let top = (4 * GIB) - 2 * TILE; // the last tile the view covers
    let src = r#"
#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), UAV(u0)"
RWByteAddressBuffer outb : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint l : SV_GroupThreadID) {
    RWByteAddressBuffer w = ResourceDescriptorHeap[NonUniformResourceIndex(0u)];
    w.Store(0xFFFE0000u + 4u * l, 0x5A000000u | l);
    w.Store(4u * l, 0xA5000000u | l);
    DeviceMemoryBarrier();
    outb.Store(8u * l, w.Load(0xFFFE0000u + 4u * l));
    outb.Store(8u * l + 4u, w.Load(4u * l));
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
        if !aperture_reserved(dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut win)).unwrap() {
            super::runner::skip_test("insufficient resources for a 4 GiB reserved aperture");
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
        let out = buf(&dev, 512, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        let rb = buf(&dev, 512 + 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
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
        list.CopyBufferRegion(&rb, 0, &out, 0, 512);
        list.CopyBufferRegion(&rb, 512, &win, top, 256);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 20_000).map_err(|e| format!("{e}")).unwrap();
        let mut q = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut q)).unwrap();
        let w = std::slice::from_raw_parts(q as *const u32, 192).to_vec();
        rb.Unmap(0, None);
        for l in 0..64usize {
            assert_eq!(w[2 * l], 0x5A00_0000 | l as u32, "{name}: lane {l}, the window's top tile through the heap view");
            assert_eq!(w[2 * l + 1], 0xA500_0000 | l as u32, "{name}: lane {l}, the window's first tile");
            assert_eq!(w[128 + l], 0x5A00_0000 | l as u32, "{name}: lane {l}, the top tile in memory");
        }
    }
}

/// Where does each path wrap? One 16 GiB reserved buffer, tiles at 64 KiB (A) and
/// 4 GiB + 64 KiB (B), each pre-filled by the shader through a root UAV whose *VA* already
/// points past 4 GiB (so no shader offset above 4 GiB): then (1) a raw read through a root UAV
/// at VA + 4 GiB, (2) a structured read of row 0x200020 through a root UAV at VA, (3) the copy
/// engine at offset 4 GiB + 64 KiB.
#[test]
fn gpu_aperture_probe_where_it_wraps() {
    let Some((dev, name)) = aperture_device(SIZE) else { return };
    if super::dxc::location().is_err() {
        super::runner::skip_test("no DXC");
        return;
    }
    let src = r#"
#define RS "UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
struct XclRow { uint d[512]; };
RWByteAddressBuffer lo4 : register(u0);      // VA + 0
RWByteAddressBuffer hi4 : register(u1);      // VA + 4 GiB
RWStructuredBuffer<XclRow> rows : register(u2); // VA + 0, rows of 2 KiB
RWByteAddressBuffer outb : register(u3);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    lo4.Store(0x10000, 0xAAAAu);
    hi4.Store(0x10000, 0xBBBBu);
    DeviceMemoryBarrier();
    outb.Store(0, lo4.Load(0x10000));
    outb.Store(4, hi4.Load(0x10000));
    outb.Store(8, rows[0x200020u].d[0]);   // 4 GiB + 64 KiB as row · 2048
    outb.Store(12, rows[0x20u].d[0]);      // 64 KiB as row · 2048
    rows[0x200020u].d[1] = 0xCCCCu;         // write through the structured path
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
            Width: SIZE,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut ap: Option<ID3D12Resource> = None;
        if !aperture_reserved(dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut ap)).unwrap() {
            super::runner::skip_test("insufficient resources for a 16 GiB reserved aperture");
            return;
        }
        let ap = ap.unwrap();
        let mut heap: Option<ID3D12Heap> = None;
        dev.CreateHeap(&D3D12_HEAP_DESC { SizeInBytes: 2 * TILE, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS }, &mut heap).unwrap();
        let heap = heap.unwrap();
        let at = [ALIAS, 4 * GIB + TILE];
        let coords: Vec<_> = at.iter().map(|&o| D3D12_TILED_RESOURCE_COORDINATE { X: (o / TILE) as u32, ..Default::default() }).collect();
        let regions = [D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() }; 2];
        let flags = [D3D12_TILE_RANGE_FLAG_NONE; 2];
        let offs = [0u32, 1];
        let counts = [1u32; 2];
        queue.UpdateTileMappings(&ap, 2, Some(coords.as_ptr()), Some(regions.as_ptr()), &heap, 2, Some(flags.as_ptr()), Some(offs.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
        let out = buf(&dev, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        let rb = buf(&dev, 256 + 2 * 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).unwrap();
        let va = ap.GetGPUVirtualAddress();
        list.SetComputeRootSignature(&rs);
        list.SetComputeRootUnorderedAccessView(0, va);
        list.SetComputeRootUnorderedAccessView(1, va + 4 * GIB);
        list.SetComputeRootUnorderedAccessView(2, va);
        list.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[
            transition(&ap, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        list.CopyBufferRegion(&rb, 0, &out, 0, 256);
        list.CopyBufferRegion(&rb, 256, &ap, ALIAS, 16);
        list.CopyBufferRegion(&rb, 272, &ap, 4 * GIB + TILE, 16);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 20_000).map_err(|e| format!("{e}")).unwrap();
        let mut q = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut q)).unwrap();
        let w = std::slice::from_raw_parts(q as *const u32, (256 + 32) / 4).to_vec();
        rb.Unmap(0, None);
        println!("{name}: raw VA+0 @64K = {:#x} (want AAAA), raw VA+4G @64K = {:#x} (want BBBB)", w[0], w[1]);
        println!("structured row 0x200020 = {:#x} (want BBBB), row 0x20 = {:#x} (want AAAA)", w[2], w[3]);
        println!("copy @64K = {:x?} (want AAAA, 0..), copy @4G+64K = {:x?} (want BBBB, CCCC..)", &w[64..68], &w[68..72]);
        // a second submission (nothing cached from the stores) reads both tiles back through
        // the raw root UAVs: tells a wrapped tile mapping from a wrapping copy engine
        let src2 = r#"
#define RS "UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
RWByteAddressBuffer lo4 : register(u0);
RWByteAddressBuffer hi4 : register(u1);
RWByteAddressBuffer unused : register(u2);
RWByteAddressBuffer outb : register(u3);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { outb.Store4(0, uint4(lo4.Load2(0x10000), hi4.Load2(0x10000))); }
"#;
        let dxil2 = super::dxc::compile(src2, "main", "cs_6_6", &[]).unwrap();
        let rs2: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil2).unwrap();
        let pso2: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: transmute_copy(&rs2),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil2.as_ptr() as _, BytecodeLength: dxil2.len() },
                ..Default::default()
            })
            .unwrap();
        let out2 = buf(&dev, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        let rb2 = buf(&dev, 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
        alloc.Reset().unwrap();
        list.Reset(&alloc, &pso2).unwrap();
        list.ResourceBarrier(&[transition(&ap, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        list.SetComputeRootSignature(&rs2);
        list.SetComputeRootUnorderedAccessView(0, va);
        list.SetComputeRootUnorderedAccessView(1, va + 4 * GIB);
        list.SetComputeRootUnorderedAccessView(2, va);
        list.SetComputeRootUnorderedAccessView(3, out2.GetGPUVirtualAddress());
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[transition(&out2, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&rb2, 0, &out2, 0, 256);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        queue.Signal(&fence, 2).unwrap();
        crate::gpu::fence::wait(&fence, 2, &dev, None, 20_000).map_err(|e| format!("{e}")).unwrap();
        let mut q = std::ptr::null_mut();
        rb2.Map(0, None, Some(&mut q)).unwrap();
        let w2 = std::slice::from_raw_parts(q as *const u32, 4).to_vec();
        rb2.Unmap(0, None);
        println!("second submission: VA+0 @64K = {:x?}, VA+4G @64K = {:x?}", &w2[0..2], &w2[2..4]);
    }
}

/// Experiment E2: DXC + driver time of one large synthetic compute program (~20 000
/// instructions, ~2 800 memory accesses: s_load, buffer load / store, global load) with one
/// window, four windows through the root-UAV `switch`, and four windows from the descriptor
/// heap. A random literal per run keeps the driver's pipeline cache out of it. Run alone:
/// `cargo test --release -p xcl --lib e2_window_compile_time -- --ignored --nocapture`
#[test]
#[ignore]
fn e2_window_compile_time() {
    use crate::gpu::rdna2::encode;
    let Some((dev, name)) = device() else { return };
    if super::dxc::location().is_err() {
        return;
    }
    let blocks: usize = std::env::var("XCL_E2_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(700);
    let reps: usize = std::env::var("XCL_E2_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let program = |salt: u32| -> Vec<u8> {
        let mut v = Vec::new();
        let mut e = |n: &str, enc: &str, f: &[(&str, u64)], l: Option<u32>| v.extend(encode(n, enc, f, l).unwrap_or_else(|x| panic!("{n}: {x}")));
        e("S_MOV_B32", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 255)], Some(salt));
        for i in 0..blocks as u64 {
            e("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", (i * 16) & 0xFFFF)], None);
            e("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None);
            e("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 6), ("ADDR", 4), ("SADDR", 0x7D), ("OFFSET", (i * 4) & 0x7FF)], None);
            for k in 0..12u64 {
                e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 256 + 1 + (k & 1) * 5), ("VSRC1", 2)], None);
                e("V_MUL_LO_U16", "ENC_VOP3", &[("VDST", 3), ("SRC0", 256 + 2), ("SRC1", 30)], None);
            }
            e("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None);
        }
        e("S_ENDPGM", "ENC_SOPP", &[], None);
        v
    };
    let rs = super::pso::abi_root_signature(&dev, false).unwrap();
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 4, ..Default::default() };
    let forms = [("1 window", 1, false), ("4 windows, switch", 4, false), ("4 windows, heap", 4, true)];
    let mut seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u32;
    println!("{name}: {blocks} blocks ({} instructions), {reps} runs each", 1 + blocks * 28);
    for (label, windows, heap) in forms {
        let (mut dxc_t, mut drv_t, mut sizes) = (Vec::new(), Vec::new(), (0, 0));
        for _ in 0..reps {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let t = super::translate(&program(seed), &super::Options { mode: super::Mode::Compute(cs.clone()), mem_windows: windows, heap_windows: heap, split_wave64: super::runner::split_wave64(), ..Default::default() }).unwrap();
            let t0 = std::time::Instant::now();
            let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            dxc_t.push(t0.elapsed().as_secs_f64());
            let t1 = std::time::Instant::now();
            let _pso: ID3D12PipelineState = unsafe {
                dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                    ..Default::default()
                })
            }
            .unwrap();
            drv_t.push(t1.elapsed().as_secs_f64());
            sizes = (t.hlsl.len(), dxil.len());
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (d, r) = (med(&mut dxc_t), med(&mut drv_t));
        println!("{label:>18}: HLSL {:>8} B, DXIL {:>8} B, DXC {d:6.2} s, driver {r:6.2} s, total {:6.2} s  (DXC {dxc_t:.2?}, driver {drv_t:.2?})", sizes.0, sizes.1, d + r);
    }
}

/// Experiment E1b, run alone (it turns on the debug layer for the process):
/// `cargo test --release -p xcl --lib e1b_raw_views_under_debug_layer -- --ignored --nocapture`.
/// (1) a raw view of 4 GiB - 64 KiB over a 4 GiB reserved window (the heap-window form; one
/// audit expected the 2^27-element typed-buffer cap to reject it), (2) a raw 1 GiB view whose
/// FirstElement starts past 4 GiB in a 16 GiB reserved buffer (another audit's 1 GiB slices):
/// both written and read through `ResourceDescriptorHeap`, with what the debug layer says.
#[test]
#[ignore]
fn e1b_raw_views_under_debug_layer() {
    unsafe {
        match crate::gpu::agility::configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug) {
            Ok(d) => d.EnableDebugLayer(),
            Err(e) => {
                println!("no debug layer: {e}");
                return;
            }
        }
    }
    let Some((dev, name)) = device() else { return };
    let src = r#"
#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), UAV(u0), RootConstants(num32BitConstants=4, b0)"
RWByteAddressBuffer outb : register(u0);
cbuffer C : register(b0) { uint slot; uint at; uint tag; uint pad; };
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint l : SV_GroupThreadID) {
    RWByteAddressBuffer w = ResourceDescriptorHeap[NonUniformResourceIndex(slot)];
    w.Store(at + 4u * l, tag | l);
    DeviceMemoryBarrier();
    outb.Store(4u * l, w.Load(at + 4u * l));
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
        let q: ID3D12InfoQueue = dev.cast().expect("info queue (debug layer on)");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let reserved = |bytes: u64| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut r).unwrap();
            r.unwrap()
        };
        let win = reserved(4 * GIB);
        let big = reserved(16 * GIB);
        let mut heap: Option<ID3D12Heap> = None;
        dev.CreateHeap(&D3D12_HEAP_DESC { SizeInBytes: 3 * TILE, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS }, &mut heap).unwrap();
        let heap = heap.unwrap();
        // window: its last covered tile; big: 5 GiB + 64 KiB (the slice's offset 64 KiB) and its 32-bit alias
        let map = |r: &ID3D12Resource, at: u64, tile: u32| {
            let c = D3D12_TILED_RESOURCE_COORDINATE { X: (at / TILE) as u32, ..Default::default() };
            let s = D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() };
            queue.UpdateTileMappings(r, 1, Some(&c), Some(&s), &heap, 1, Some(&D3D12_TILE_RANGE_FLAG_NONE), Some(&tile), Some(&1), D3D12_TILE_MAPPING_FLAG_NONE);
        };
        map(&win, 4 * GIB - 2 * TILE, 0);
        map(&big, 5 * GIB + TILE, 1);
        map(&big, GIB + TILE, 2);
        let dh: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 2, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
            .unwrap();
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let h0 = dh.GetCPUDescriptorHandleForHeapStart();
        let before = q.GetNumStoredMessages();
        dev.CreateUnorderedAccessView(&win, None::<&ID3D12Resource>, Some(&super::runner::raw_window_uav(4 * GIB)), h0);
        let mut slice = super::runner::raw_window_uav(GIB);
        slice.Anonymous.Buffer.FirstElement = 5 * GIB / 4;
        dev.CreateUnorderedAccessView(&big, None::<&ID3D12Resource>, Some(&slice), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h0.ptr + inc });
        let after_views = q.GetNumStoredMessages();
        let out = buf(&dev, 512, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        let rb = buf(&dev, 512, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).unwrap();
        list.SetDescriptorHeaps(&[Some(dh.clone())]);
        list.SetComputeRootSignature(&rs);
        list.SetComputeRootUnorderedAccessView(0, out.GetGPUVirtualAddress());
        // slot 0: top tile of the window (offset 0xFFFE0000); slot 1: offset 64 KiB of the slice at 5 GiB
        list.SetComputeRoot32BitConstants(1, 3, [0u32, 0xFFFE_0000, 0x5A00_0000].as_ptr() as _, 0);
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&rb, 0, &out, 0, 256);
        list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        list.SetComputeRoot32BitConstants(1, 3, [1u32, 0x1_0000, 0x6B00_0000].as_ptr() as _, 0);
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&rb, 256, &out, 0, 256);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 20_000).map_err(|e| format!("{e}")).unwrap();
        // the 32-bit alias of the slice's tile (1 GiB + 64 KiB) through a copy (offsets below
        // 4 GiB copy correctly): untouched, or did the slice wrap?
        let rb2 = buf(&dev, 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
        alloc.Reset().unwrap();
        list.Reset(&alloc, None).unwrap();
        list.ResourceBarrier(&[transition(&big, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&rb2, 0, &big, GIB + TILE, 256);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        queue.Signal(&fence, 2).unwrap();
        crate::gpu::fence::wait(&fence, 2, &dev, None, 20_000).map_err(|e| format!("{e}")).unwrap();
        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).unwrap();
        let w = std::slice::from_raw_parts(p as *const u32, 128).to_vec();
        rb.Unmap(0, None);
        rb2.Map(0, None, Some(&mut p)).unwrap();
        let alias = std::slice::from_raw_parts(p as *const u32, 64).to_vec();
        rb2.Unmap(0, None);
        let n = q.GetNumStoredMessages();
        println!("{name}: debug layer messages: {} at view creation, {} in all", after_views - before, n);
        for i in 0..n {
            let mut len = 0usize;
            let _ = q.GetMessage(i, None, &mut len);
            let mut b = vec![0u8; len];
            let m = b.as_mut_ptr() as *mut D3D12_MESSAGE;
            if q.GetMessage(i, Some(m), &mut len).is_ok() {
                let m = &*m;
                println!("  [{:?}] {}", m.Severity, std::ffi::CStr::from_ptr(m.pDescription as _).to_string_lossy());
            }
        }
        let win_ok = (0..64).all(|l| w[l] == 0x5A00_0000 | l as u32);
        let slice_ok = (0..64).all(|l| w[64 + l] == 0x6B00_0000 | l as u32);
        let alias_clean = alias.iter().all(|&x| x == 0);
        println!("4 GiB - 64 KiB raw view, top tile: {}", if win_ok { "ok" } else { "WRONG" });
        println!("1 GiB raw slice at FirstElement 5 GiB: read back {}, 32-bit alias tile {}", if slice_ok { "ok" } else { "WRONG" }, if alias_clean { "untouched" } else { "WRITTEN (the slice wrapped at 4 GiB)" });
        println!("first words: window {:#x}, slice {:#x}, alias {:#x}", w[0], w[64], alias[0]);
    }
}
