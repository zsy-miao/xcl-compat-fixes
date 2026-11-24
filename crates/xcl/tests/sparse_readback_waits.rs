//! A title's queue may GPU-wait for a fence that a later submit signals (async compute). When
//! its lists write sparse memory on the GPU, the copy back to the CPU's view
//! (`sparse::readback_after_execute`, after its ExecuteCommandLists) must not wait for them on
//! the CPU: the later submit, which lets them run, would never come.

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use xcl::hle::d3d12x::sparse;

/// The title's thread: queue A waits for F, runs a list writing 8 bytes of a sparse window, and
/// is read back; then queue B submits and signals F. A finishes within 5 s.
#[test]
fn a_readback_behind_a_wait_for_a_later_signal_does_not_hang() {
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
        let qa: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let qb: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
        let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

        // A's list: 8 bytes from a buffer of the title's into the window
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
        *(p as *mut u64) = 0x1122_3344_5566_7788;
        let raw = raw as *mut std::ffi::c_void;
        let window = ID3D12Resource::from_raw_borrowed(&raw).unwrap();
        let al: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let la: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &al, None).unwrap();
        la.CopyBufferRegion(window, (a - base) as u64, &src, 0, 8);
        la.Close().unwrap();
        let bl: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
        let lb: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &bl, None).unwrap();
        lb.Close().unwrap();
        sparse::before_execute(&qa, &[], 0x1000);

        // the title's thread
        let (qa2, qb2, f2, done2) = (qa.clone(), qb.clone(), f.clone(), done.clone());
        let (la, lb) = (la.cast::<ID3D12CommandList>().unwrap(), lb.cast::<ID3D12CommandList>().unwrap());
        let title = std::thread::spawn(move || {
            qa2.Wait(&f2, 1).unwrap();
            qa2.ExecuteCommandLists(&[Some(la)]);
            sparse::readback_after_execute(&qa2, &[(a, 8)]);
            qa2.Signal(&done2, 1).unwrap();
            sparse::before_execute(&qb2, &[], 0x1000);
            qb2.ExecuteCommandLists(&[Some(lb)]);
            qb2.Signal(&f2, 1).unwrap();
        });

        let ev = CreateEventW(None, false, false, None).unwrap();
        done.SetEventOnCompletion(1, ev).unwrap();
        let finished = WaitForSingleObject(ev, 5000) == WAIT_OBJECT_0;
        if !finished {
            // let the queues go so the process can end
            f.Signal(1).unwrap();
        }
        title.join().unwrap();
        let _ = CloseHandle(ev);
        assert!(finished, "queue A did not finish in 5 s: the readback waits on the CPU for A, which waits for B's later submit");
        // the GPU's write reaches the CPU's view
        let got = *(a as *const u64);
        let ev = CreateEventW(None, false, false, None).unwrap();
        let mut seen = got == 0x1122_3344_5566_7788;
        for _ in 0..100 {
            if seen {
                break;
            }
            let _ = WaitForSingleObject(ev, 10);
            seen = *(a as *const u64) == 0x1122_3344_5566_7788;
        }
        let _ = CloseHandle(ev);
        assert!(seen, "the GPU's write did not reach the CPU's view: {got:#x}");
    }
}
