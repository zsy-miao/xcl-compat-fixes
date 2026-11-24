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
