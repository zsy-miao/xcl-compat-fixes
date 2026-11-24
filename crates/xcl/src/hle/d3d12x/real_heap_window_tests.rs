use super::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// Exercise the runtime hooks, not runner::dispatch_on (which makes its own heap views).
#[test]
fn heap_window_runtime_dispatch_without_title_heaps_reads_real_memory() {
    unsafe {
        if std::env::var("XCL_HEAP_WINDOWS").is_ok_and(|v| v == "0") {
            crate::gpu::recomp::runner::skip_test("heap windows explicitly disabled");
            return;
        }
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let Some(u) = crate::gpu::uma::get() else {
            crate::gpu::recomp::runner::skip_test("no unified memory");
            return;
        };
        if ADAPTER.lock().unwrap().contains("WARP")
            || uma_window_vas().len() < 2
            || !(0..uma_window_vas().len()).any(|k| u.is_committed_window(k))
            || !UMA_IMPORT_CHECK.lock().unwrap().starts_with("ok")
        {
            crate::gpu::recomp::runner::skip_test("no verified multi-window coherent import");
            return;
        }
        if !crate::gpu::recomp::pso::heap_windows() {
            crate::gpu::recomp::runner::skip_test("memory-window descriptors unavailable");
            return;
        }
        let raw_dev = device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&raw_dev).unwrap();
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        if dev
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                &mut sm as *mut _ as *mut c_void,
                std::mem::size_of_val(&sm) as u32,
            )
            .is_err()
            || sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0
        {
            crate::gpu::recomp::runner::skip_test("shader model 6.6 unavailable");
            return;
        }
        assert!(
            super::super::console_heap::global_heap().is_some(),
            "runtime initialization must create the memory-window descriptor heap"
        );
        let heap = super::super::console_heap::global_heap().unwrap();
        let at = u
            .alloc(0x1_0000, 0x1_0000)
            .expect("coherent sentinel memory") as usize;
        struct Memory(usize);
        impl Drop for Memory {
            fn drop(&mut self) {
                crate::gpu::uma::get().unwrap().free(self.0);
            }
        }
        let _memory = Memory(at);
        let k = u.window_of(at).unwrap();
        assert!(u.is_committed_window(k));
        let offset = at - u.window(k).0;
        (at as *mut u32).write_volatile(0x1234_5678);
        ((at + 4) as *mut u32).write_volatile(0);
        let hlsl = format!(
            "[numthreads(1,1,1)] void main() {{ RWByteAddressBuffer m = ResourceDescriptorHeap[{}u]; uint v = m.Load({offset}u); m.Store({}u, v ^ 0xA5A55A5Au); }}",
            crate::gpu::recomp::MEM_HEAP_BASE + k as u32,
            offset + 4
        );
        let dxil = crate::gpu::recomp::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .expect("heap-window shader");
        let rs_raw = abi_rs_raw(raw_dev).expect("runtime ABI root signature");
        let rs = ID3D12RootSignature::from_raw_borrowed(&rs_raw).unwrap();
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(rs),
                CS: bytecode(&dxil),
                ..Default::default()
            })
            .expect("heap-window pipeline");
        let key = pso.as_raw() as usize;
        register_translated_compute_for_test(key);
        note_pso_rs(pso.as_raw(), rs_raw);
        struct Pipeline(usize);
        impl Drop for Pipeline {
            fn drop(&mut self) {
                forget_pso_for_test(self.0);
            }
        }
        let _pipeline = Pipeline(key);
        let allocator: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            .unwrap();
        let raw = list.as_raw();
        adopt(raw, Kind::CommandList);
        super::super::list_x::on_create(
            raw,
            allocator.as_raw() as usize,
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
        );
        super::super::bind::reset(raw, allocator.as_raw() as usize, 0);
        // Deliberately no title SetDescriptorHeaps: bind_abi must provide the shared heap.
        super::super::bind::set_pipeline_state(raw, pso.as_raw());
        super::super::bind::dispatch(raw, 1, 1, 1);
        // Check the recorder before submission, even without a system debug layer.
        assert!(
            super::super::list_x::bound_heaps_for_test(raw).contains(&(heap.as_raw() as usize)),
            "runtime ABI dispatch must bind the shared descriptor heap"
        );
        let (window_raw, source_offset) = uma_window_buffer(at, 8).unwrap();
        let window_raw = window_raw as *mut c_void;
        let window = ID3D12Resource::from_raw_borrowed(&window_raw).unwrap();
        let mut readback: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut readback,
        )
        .unwrap();
        let readback = readback.unwrap();
        let barrier = |before, after| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(window),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
            ..Default::default()
        };
        list.ResourceBarrier(&[barrier(
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(&readback, 0, window, source_offset, 8);
        list.ResourceBarrier(&[barrier(
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
        list.Close()
            .expect("runtime must have bound the heap required by the shader");
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())
            .unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, dev, None, 5_000).unwrap();
        let mut mapped = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut mapped)).unwrap();
        let got = std::slice::from_raw_parts(mapped as *const u32, 2);
        assert_eq!(
            got,
            &[0x1234_5678, 0x1234_5678 ^ 0xA5A5_5A5A],
            "the runtime's descriptors and heap bind must address the genuine coherent bytes"
        );
        readback.Unmap(0, None);
        assert_eq!(
            ((at + 4) as *const u32).read_volatile(),
            0x1234_5678 ^ 0xA5A5_5A5A
        );
    }
}
