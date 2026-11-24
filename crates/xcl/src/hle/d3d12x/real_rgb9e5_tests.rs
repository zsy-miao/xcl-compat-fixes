use super::super::{device_x, vram};
use super::*;
use std::mem::{transmute_copy, ManuallyDrop};
use windows::Win32::Graphics::Dxgi::Common::*;

fn console_desc(format: u32, flags: u32) -> [u8; 0x38] {
    let mut desc = [0u8; 0x38];
    desc[0..4].copy_from_slice(&3u32.to_le_bytes());
    desc[0x10..0x18].copy_from_slice(&8u64.to_le_bytes());
    desc[0x18..0x1c].copy_from_slice(&8u32.to_le_bytes());
    desc[0x1c..0x1e].copy_from_slice(&1u16.to_le_bytes());
    desc[0x1e..0x20].copy_from_slice(&1u16.to_le_bytes());
    desc[0x20..0x24].copy_from_slice(&format.to_le_bytes());
    desc[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
    desc[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
    desc
}

unsafe fn placed(
    dev: *mut c_void,
    address: u64,
    format: u32,
    flags: u32,
    clear: &D3D12_CLEAR_VALUE,
) -> ID3D12Resource {
    let desc = console_desc(format, flags);
    let mut raw = std::ptr::null_mut();
    let hr = device_x::create_placed_resource_x(
        dev,
        address,
        desc.as_ptr(),
        0,
        clear as *const _ as _,
        &pc_iid(Kind::Resource),
        &mut raw,
    );
    assert!(
        hr >= 0 && !raw.is_null(),
        "console resource {format}, flags {flags:#x}, clear {}: {hr:#x}",
        clear.Format.0
    );
    ID3D12Resource::from_raw(raw)
}

unsafe fn buffer(
    dev: &ID3D12Device,
    heap: D3D12_HEAP_TYPE,
    bytes: u64,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if heap == D3D12_HEAP_TYPE_DEFAULT {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let mut resource = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES {
            Type: heap,
            ..Default::default()
        },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        state,
        None,
        &mut resource,
    )
    .unwrap();
    resource.unwrap()
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
        ..Default::default()
    }
}

unsafe fn copy_texture(
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    readback: &ID3D12Resource,
    offset: u64,
) {
    let source = D3D12_TEXTURE_COPY_LOCATION {
        pResource: transmute_copy(texture),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let destination = D3D12_TEXTURE_COPY_LOCATION {
        pResource: transmute_copy(readback),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: offset,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: DXGI_FORMAT_R11G11B10_FLOAT,
                    Width: 8,
                    Height: 8,
                    Depth: 1,
                    RowPitch: 256,
                },
            },
        },
    };
    list.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
}

/// Exercise console creation and view hooks, then use both views on the GPU. Creation alone
/// and void descriptor creation calls would not detect a mismatched format or null view.
/// The HDR powers of two are exactly representable in both formats; the format promotion
/// itself is not lossless for arbitrary shared-exponent values.
#[test]
fn shared_exponent_target_clear_copy_and_shader_views_preserve_hdr_values() {
    if crate::gpu::recomp::dxc::location().is_err() {
        crate::gpu::recomp::runner::skip_test("no DXC");
        return;
    }
    if crate::hle::d3d12x::backend() != crate::hle::d3d12x::Backend::Real {
        crate::gpu::recomp::runner::skip_test("no GPU");
        return;
    }
    unsafe {
        let raw = device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&raw).unwrap().clone();
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                Color: [2.0, 4.0, 8.0, 1.0],
            },
        };
        let texture = placed(raw, 0x0065_6510_0000, 67, 0x30_0005, &clear);
        assert_eq!(texture.GetDesc().Format, DXGI_FORMAT_R11G11B10_FLOAT);
        let mut graphics = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            NumRenderTargets: 1,
            ..Default::default()
        };
        graphics.RTVFormats[0] = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        pc_topology_type(&mut graphics);
        assert_eq!(
            graphics.RTVFormats[0], DXGI_FORMAT_R11G11B10_FLOAT,
            "production PSO target mapper agrees with its resource"
        );

        let rtvs: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap();
        let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
        let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        dev_create_rtv(raw, texture.as_raw(), &rtv_desc as *const _ as _, rtv.ptr);
        let views: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .unwrap();
        let cpu_views: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap();
        let increment =
            dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let visible_uav = views.GetCPUDescriptorHandleForHeapStart();
        let cpu_uav = cpu_views.GetCPUDescriptorHandleForHeapStart();
        dev_create_uav_x(
            raw,
            texture.as_raw(),
            std::ptr::null_mut(),
            &uav,
            visible_uav.ptr as u64,
        );
        dev_create_uav_x(
            raw,
            texture.as_raw(),
            std::ptr::null_mut(),
            &uav,
            cpu_uav.ptr as u64,
        );
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        dev_create_srv_x(
            raw,
            texture.as_raw(),
            &srv,
            (visible_uav.ptr + increment) as u64,
        );
        let hlsl = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0)"
Texture2D<float4> image : register(t0);
RWByteAddressBuffer result : register(u0);
[RootSignature(RS)] [numthreads(1,1,1)]
void main() { result.Store3(0, asuint(image.Load(int3(0,0,0)).xyz)); }
"#;
        let dxil = crate::gpu::recomp::dxc::compile(hlsl, "main", "cs_6_0", &[]).unwrap();
        let root: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: dxil.as_ptr() as _,
                    BytecodeLength: dxil.len(),
                },
                ..Default::default()
            })
            .unwrap();
        let output = buffer(
            &dev,
            D3D12_HEAP_TYPE_DEFAULT,
            16,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let readback = buffer(
            &dev,
            D3D12_HEAP_TYPE_READBACK,
            4128,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let allocator: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pso)
            .unwrap();
        list.ResourceBarrier(&[transition(
            &texture,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        )]);
        list.ClearRenderTargetView(rtv, &[2.0, 4.0, 8.0, 1.0], None);
        list.ResourceBarrier(&[transition(
            &texture,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        copy_texture(&list, &texture, &readback, 0);
        list.ResourceBarrier(&[transition(
            &texture,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        list.SetDescriptorHeaps(&[Some(views.clone())]);
        list.ClearUnorderedAccessViewFloat(
            views.GetGPUDescriptorHandleForHeapStart(),
            cpu_uav,
            &texture,
            &[4.0, 2.0, 1.0, 1.0],
            &[],
        );
        list.ResourceBarrier(&[transition(
            &texture,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        copy_texture(&list, &texture, &readback, 2048);
        list.ResourceBarrier(&[transition(
            &texture,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )]);
        list.SetComputeRootSignature(&root);
        let mut gpu_srv = views.GetGPUDescriptorHandleForHeapStart();
        gpu_srv.ptr += increment as u64;
        list.SetComputeRootDescriptorTable(0, gpu_srv);
        list.SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress());
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[transition(
            &output,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(&readback, 4096, &output, 0, 12);
        list.Close().unwrap();
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())
            .unwrap();
        vram::before_execute(&queue);
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 5_000)
            .expect("HDR view and copy work completes");
        let mut mapped = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut mapped)).unwrap();
        let bytes = std::slice::from_raw_parts(mapped as *const u8, 4128);
        for (offset, expected) in [(0, 0x9022_0400u32), (2048, 0x7820_0440u32)] {
            for y in 0..8 {
                for x in 0..8 {
                    let at = offset + 256 * y + 4 * x;
                    assert_eq!(
                        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()),
                        expected,
                        "HDR clear/copy pixel ({x},{y}) at {offset}"
                    );
                }
            }
        }
        let sampled = std::slice::from_raw_parts((mapped as *const u8).add(4096).cast::<u32>(), 3);
        assert_eq!(
            sampled,
            [4.0f32.to_bits(), 2.0f32.to_bits(), 1.0f32.to_bits()],
            "SRV reads the UAV's HDR values"
        );
        readback.Unmap(0, None);

        // The later refusal seen in the same startup chain: typeless depth clear follows its
        // DSV family while the resource itself remains typeless.
        let depth_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_R24G8_TYPELESS,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: 0.5,
                    Stencil: 3,
                },
            },
        };
        let depth = placed(raw, 0x0065_6520_0000, 118, 0x180_0002, &depth_clear);
        assert_eq!(depth.GetDesc().Format, DXGI_FORMAT_R24G8_TYPELESS);
        dev.GetDeviceRemovedReason()
            .expect("matching console resource/view/clear formats keep the device");
    }
}
