//! What the GPU wrote into a sparse window is in the CPU's view when the title's own fence after
//! that work completes (`sparse::readback_after_execute`): a title waits for its fence and reads.

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use xcl::hle::d3d12x::sparse;

/// 20 times: a list writes 8 bytes of a sparse window, the title waits for its fence after it
/// and reads them at once.
#[test]
fn gpu_writes_are_there_when_the_titles_fence_completes() {
    unsafe {
        std::env::set_var("XCL_UMA_MB", "64");
        std::env::set_var("XCL_COHERENT_MB", "64");
        let Ok(dev) = xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            println!("SKIPPED: no GPU");
            return;
        };
        // the title's device: its queues and fences come with XCL's methods
        xcl::hle::d3d12x::real::adopt(dev.as_raw(), xcl::hle::d3d12x::real::Kind::Device);
        let windows = sparse::init(&dev);
        let Some(&(k, raw, _)) = windows.first() else {
            println!("SKIPPED: no sparse windows");
            return;
        };
        let base = xcl::gpu::uma::get().unwrap().window(k).0;
        let (a, _) = sparse::alloc_in(k, 0x1_0000, 0x1_0000).expect("sparse allocation");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let ev = CreateEventW(None, false, false, None).unwrap();
        let mut value = 0u64;
        let mut wait = || {
            value += 1;
            queue.Signal(&fence, value).unwrap();
            fence.SetEventOnCompletion(value, ev).unwrap();
            assert_eq!(WaitForSingleObject(ev, 5000), WAIT_OBJECT_0, "the queue did not finish");
        };
        // as before each of the title's submits (the tile mappings), done
        sparse::before_execute(&queue, &[], 0x1000);
        wait();

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut src: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut src).unwrap();
        let src = src.unwrap();
        let mut p = std::ptr::null_mut();
        src.Map(0, None, Some(&mut p)).unwrap();
        let raw = raw as *mut std::ffi::c_void;
        let window = ID3D12Resource::from_raw_borrowed(&raw).unwrap();
        let al: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &al, None).unwrap();
        list.Close().unwrap();
        for i in 0..20u64 {
            let want = 0x1122_3344_0000_0000 | i;
            *(p as *mut u64) = want;
            al.Reset().unwrap();
            list.Reset(&al, None).unwrap();
            list.CopyBufferRegion(window, (a - base) as u64, &src, 0, 8);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            sparse::readback_after_execute(&queue, &[(a, 8)]);
            wait();
            let got = *(a as *const u64);
            assert_eq!(got, want, "round {i}: the CPU's view after the title's fence wait");
        }
        let _ = CloseHandle(ev);
    }
}
