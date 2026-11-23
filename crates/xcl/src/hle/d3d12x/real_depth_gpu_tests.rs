use super::*;
use std::mem::{transmute_copy, ManuallyDrop};
use windows::Win32::Graphics::Dxgi::Common::*;

unsafe fn buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 32, Height: 1,
        DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if heap == D3D12_HEAP_TYPE_DEFAULT { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut out: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out).unwrap();
    out.unwrap()
}

fn transition(res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
            pResource: unsafe { transmute_copy(res) },
            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: before, StateAfter: after,
        }) },
        ..Default::default()
    }
}

/// Clear through the typed DSV and read both depth and stencil through the shader views.
/// Creating views alone returns no HRESULT and would miss an invalid resource/view pairing.
#[test]
fn typeless_depth_resource_accepts_typed_dsv_and_shader_views() {
    assert_eq!(crate::hle::d3d12x::backend(), crate::hle::d3d12x::Backend::Real, "real GPU required");
    unsafe {
        let raw = device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap().clone();
        let mut console = [0u8; 0x38];
        console[0..4].copy_from_slice(&(D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32).to_le_bytes());
        console[0x10..0x18].copy_from_slice(&16u64.to_le_bytes());
        console[0x18..0x1c].copy_from_slice(&16u32.to_le_bytes());
        console[0x1c..0x1e].copy_from_slice(&1u16.to_le_bytes());
        console[0x1e..0x20].copy_from_slice(&1u16.to_le_bytes());
        console[0x20..0x24].copy_from_slice(&118u32.to_le_bytes());
        console[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        console[0x30..0x34].copy_from_slice(&6u32.to_le_bytes());
        let bytes = super::super::device_x::pc_resource_desc(&console);
        let desc = (bytes.as_ptr() as *const D3D12_RESOURCE_DESC).read_unaligned();
        assert_eq!(desc.Format, DXGI_FORMAT_R24G8_TYPELESS);
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D24_UNORM_S8_UINT,
            Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 0.25, Stencil: 17 } },
        };
        let mut texture: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(&clear), &mut texture).unwrap();
        let texture = texture.unwrap();
        let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
        let dsv = pc_depth_dsv(&texture.GetDesc(), None).expect("typed default DSV");
        assert_eq!(dsv.Format, DXGI_FORMAT_D24_UNORM_S8_UINT);
        let dsv_handle = dsv_heap.GetCPUDescriptorHandleForHeapStart();
        dev.CreateDepthStencilView(&texture, Some(&dsv), dsv_handle);
        let views: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 2, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).unwrap();
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let texture_raw = texture.as_raw();
        let depth = pc_depth_srv(texture_raw, std::ptr::null()).unwrap();
        let stencil_request = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(120), ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
        };
        let stencil = pc_depth_srv(texture_raw, &stencil_request).unwrap();
        assert_eq!(depth.Format, DXGI_FORMAT_R24_UNORM_X8_TYPELESS);
        assert_eq!(stencil.Format, DXGI_FORMAT_X24_TYPELESS_G8_UINT);
        assert_eq!(stencil.Anonymous.Texture2D.PlaneSlice, 1);
        let mut handle = views.GetCPUDescriptorHandleForHeapStart();
        dev.CreateShaderResourceView(&texture, Some(&depth), handle);
        handle.ptr += inc;
        dev.CreateShaderResourceView(&texture, Some(&stencil), handle);
        let hlsl = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2)), UAV(u0)"
Texture2D<float> depth : register(t0);
Texture2D<uint4> stencil : register(t1);
RWByteAddressBuffer result : register(u0);
[RootSignature(RS)] [numthreads(1,1,1)]
void main() {
    result.Store(0, asuint(depth.Load(int3(0,0,0))));
    result.Store4(4, stencil.Load(int3(0,0,0)));
}
"#;
        let dxil = crate::gpu::recomp::dxc::compile(hlsl, "main", "cs_6_6", &[]).unwrap();
        let root: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
        let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&root), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() }, ..Default::default() }).unwrap();
        let output = buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        let readback = buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pso).unwrap();
        list.ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 17, None);
        list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
        list.SetDescriptorHeaps(&[Some(views.clone())]);
        list.SetComputeRootSignature(&root);
        list.SetComputeRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
        list.SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress());
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[transition(&output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&readback, 0, &output, 0, 20);
        list.Close().unwrap();
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 5_000).expect("depth view dispatch completes");
        let mut mapped = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut mapped)).unwrap();
        let values = std::slice::from_raw_parts(mapped as *const u32, 5).to_vec();
        readback.Unmap(0, None);
        let depth = f32::from_bits(values[0]);
        assert!((depth - 0.25).abs() < 0.00001, "depth read through the SRV: {depth}, {values:?}");
        assert!(values[1..].contains(&17), "stencil view must read the clear's stencil, not a null view: {values:?}");
        dev.GetDeviceRemovedReason().expect("legal depth resource/view families keep the device");
    }
}
