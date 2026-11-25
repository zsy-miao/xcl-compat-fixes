//! Host-driver parity and console ABI/contract tests for `ID3D12Resource` (`Map`, `Unmap`,
//! `GetGPUVirtualAddress`, `WriteToSubresource`, `Release`), `ID3D12DescriptorHeap` handle-start
//! slots (#9 / #10), `ID3D12CommandAllocator::Reset`, `ID3D12Fence` (`Signal`, console #12
//! completed-value cell), and `ID3D12CommandQueue` runtime sync/timing/execution methods
//! (`ExecuteCommandLists`, `Signal`, `Wait`, `GetTimestampFrequency`, `GetClockCalibration`,
//! `GetDesc`, console `SignalX` #23). Part of #480.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use windows::core::{Interface, GUID};
use windows::Win32::Foundation::{CloseHandle, E_INVALIDARG, S_OK, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows_sys::Win32::System::Diagnostics::Debug::RtlAddFunctionTable;
use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE};

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{console_heap, real};
use xcl::modules::{self, MappedModule};
use xcl::pe::OwnedExports;

type TrampolineFn = unsafe extern "system" fn(
    target: usize,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
    a7: usize,
) -> usize;

struct ConsoleCaller {
    trampoline: TrampolineFn,
}

/// Allocates a single executable page registered with `xcl::modules` and Windows x64 unwind
/// metadata so calls routed through `trampoline` appear to originate inside a console image.
fn console_caller() -> &'static ConsoleCaller {
    static CALLER: OnceLock<ConsoleCaller> = OnceLock::new();
    CALLER.get_or_init(|| unsafe {
        let base = VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
        assert!(!base.is_null(), "VirtualAlloc for console caller trampoline failed");

        // Windows x64 trampoline: shifts (target, a0..a7) -> target(a0..a7) with 16-byte stack
        // alignment (`sub rsp, 0x48`) and a return address inside the registered module range.
        let code: &[u8] = &[
            0x48, 0x83, 0xEC, 0x48,                         // sub rsp, 0x48
            0x48, 0x89, 0xC8,                               // mov rax, rcx
            0x48, 0x89, 0xD1,                               // mov rcx, rdx
            0x4C, 0x89, 0xC2,                               // mov rdx, r8
            0x4D, 0x89, 0xC8,                               // mov r8, r9
            0x4C, 0x8B, 0x4C, 0x24, 0x70,                   // mov r9, [rsp+0x70]
            0x4C, 0x8B, 0x54, 0x24, 0x78,                   // mov r10, [rsp+0x78]
            0x4C, 0x89, 0x54, 0x24, 0x20,                   // mov [rsp+0x20], r10
            0x4C, 0x8B, 0x94, 0x24, 0x80, 0x00, 0x00, 0x00, // mov r10, [rsp+0x80]
            0x4C, 0x89, 0x54, 0x24, 0x28,                   // mov [rsp+0x28], r10
            0x4C, 0x8B, 0x94, 0x24, 0x88, 0x00, 0x00, 0x00, // mov r10, [rsp+0x88]
            0x4C, 0x89, 0x54, 0x24, 0x30,                   // mov [rsp+0x30], r10
            0x4C, 0x8B, 0x94, 0x24, 0x90, 0x00, 0x00, 0x00, // mov r10, [rsp+0x90]
            0x4C, 0x89, 0x54, 0x24, 0x38,                   // mov [rsp+0x38], r10
            0xFF, 0xD0,                                     // call rax
            0x48, 0x83, 0xC4, 0x48,                         // add rsp, 0x48
            0xC3,                                           // ret
        ];
        std::ptr::copy_nonoverlapping(code.as_ptr(), base, code.len());

        // UNWIND_INFO at +0x100: version=1, prolog=4, codes=1 (UWOP_ALLOC_SMALL 0x48).
        let unwind: [u8; 8] = [0x01, 0x04, 0x01, 0x00, 0x04, 0x82, 0x00, 0x00];
        std::ptr::copy_nonoverlapping(unwind.as_ptr(), base.add(0x100), unwind.len());
        // RUNTIME_FUNCTION (BeginAddress, EndAddress, UnwindInfoAddress) at +0x110.
        let rf: [u32; 3] = [0, code.len() as u32, 0x100];
        std::ptr::copy_nonoverlapping(rf.as_ptr(), base.add(0x110) as *mut u32, rf.len());
        assert!(RtlAddFunctionTable(base.add(0x110) as *const _, 1, base as u64), "RtlAddFunctionTable failed");

        modules::register_mapped(Arc::new(MappedModule {
            name: "test_d3d12_sync_caller.exe".into(),
            path: PathBuf::from("test_d3d12_sync_caller.exe"),
            base: base as usize,
            size: 0x1000,
            entry_rva: 0,
            is_dll: false,
            exports: OwnedExports::default(),
            tls_callbacks: std::sync::Mutex::new(Vec::new()),
            no_thread_calls: AtomicBool::new(false),
            initialized: AtomicBool::new(true),
            stack_reserve: 0,
        }));

        ConsoleCaller {
            trampoline: std::mem::transmute(base),
        }
    })
}

unsafe fn vslot(obj: *mut c_void, slot: usize) -> usize {
    *(*(obj as *const *const usize)).add(slot)
}

unsafe fn call_console(obj: *mut c_void, slot: usize, args: &[usize]) -> usize {
    assert!(args.len() <= 7, "call_console supports up to 7 arguments after this");
    let mut a = [0usize; 7];
    a[..args.len()].copy_from_slice(args);
    (console_caller().trampoline)(vslot(obj, slot), obj as usize, a[0], a[1], a[2], a[3], a[4], a[5], a[6])
}

fn adopted_device() -> Option<ID3D12Device> {
    static INIT: OnceLock<Result<String, String>> = OnceLock::new();
    if INIT.get_or_init(real::init).is_err() {
        runner::skip_test("no GPU");
        return None;
    }
    unsafe {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let hr = real::create_device(&ID3D12Device::IID as *const GUID as *const c_void, &mut raw);
        assert_eq!(hr, S_OK.0, "real::create_device failed: {hr:#x}");
        assert!(!raw.is_null());
        Some(ID3D12Device::from_raw(raw))
    }
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    }
}

fn tex2d_desc(w: u64, h: u32, layout: D3D12_TEXTURE_LAYOUT) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: w,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: layout,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

fn heap_props(ty: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: ty, ..Default::default() }
}

unsafe fn create_committed(
    dev: &ID3D12Device,
    props: &D3D12_HEAP_PROPERTIES,
    flags: D3D12_HEAP_FLAGS,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut out: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(props, flags, desc, state, None, &mut out)
        .expect("CreateCommittedResource failed");
    out.unwrap()
}

unsafe fn wait_fence_event(fence: &ID3D12Fence, value: u64) {
    if fence.GetCompletedValue() >= value {
        return;
    }
    let ev = CreateEventW(None, false, false, windows::core::PCWSTR::null()).expect("CreateEventW failed");
    fence.SetEventOnCompletion(value, ev).expect("SetEventOnCompletion");
    let r = WaitForSingleObject(ev, 5000);
    let _ = CloseHandle(ev);
    assert_eq!(r, WAIT_OBJECT_0, "timed out waiting for fence value {value}");
}

#[test]
fn resource_map_unmap_write_and_gpu_va_match_host_and_console_contracts() {
    let Some(dev) = adopted_device() else { return };
    unsafe {
        type MapFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RANGE, *mut *mut c_void) -> i32;
        type UnmapFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RANGE);
        type GpuVaFn = unsafe extern "system" fn(*mut c_void) -> u64;
        type WriteSubFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_BOX, *const c_void, u32, u32) -> i32;

        // 1. UPLOAD & READBACK buffers: Map / Unmap / GetGPUVirtualAddress parity with orig_fn.
        let upload = create_committed(&dev, &heap_props(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &buffer_desc(4096, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ);
        let readback = create_committed(&dev, &heap_props(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE, &buffer_desc(4096, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST);

        let orig_gpu_va: GpuVaFn = std::mem::transmute(real::orig_fn(upload.as_raw(), 11));
        assert_ne!(upload.GetGPUVirtualAddress(), 0);
        assert_eq!(upload.GetGPUVirtualAddress(), orig_gpu_va(upload.as_raw()));
        let console_up_va = call_console(upload.as_raw(), 11, &[]) as u64;
        assert_eq!(console_up_va, upload.GetGPUVirtualAddress());

        let empty_range = D3D12_RANGE { Begin: 0, End: 0 };
        let mut p_hook: *mut c_void = std::ptr::null_mut();
        upload.Map(0, Some(&empty_range), Some(&mut p_hook)).expect("upload Map");
        assert!(!p_hook.is_null());
        std::ptr::write_bytes(p_hook as *mut u8, 0x3C, 64);
        upload.Unmap(0, None);

        let orig_map: MapFn = std::mem::transmute(real::orig_fn(upload.as_raw(), 8));
        let orig_unmap: UnmapFn = std::mem::transmute(real::orig_fn(upload.as_raw(), 9));
        let mut p_orig: *mut c_void = std::ptr::null_mut();
        assert_eq!(orig_map(upload.as_raw(), 0, &empty_range, &mut p_orig), S_OK.0);
        assert_eq!(std::slice::from_raw_parts(p_orig as *const u8, 64), &[0x3Cu8; 64]);
        orig_unmap(upload.as_raw(), 0, std::ptr::null());

        let mut p_rb: *mut c_void = std::ptr::null_mut();
        readback.Map(0, Some(&empty_range), Some(&mut p_rb)).expect("readback Map");
        assert!(!p_rb.is_null());
        readback.Unmap(0, Some(&empty_range));

        // 2. Unmappable resources (pc_can_map): DEFAULT buffer, DEFAULT texture, CUSTOM
        //    CPU_PAGE_PROPERTY_NOT_AVAILABLE texture, and reserved texture return E_INVALIDARG.
        let def_buf = create_committed(&dev, &heap_props(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &buffer_desc(4096, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON);
        let def_tex = create_committed(&dev, &heap_props(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &tex2d_desc(16, 16, D3D12_TEXTURE_LAYOUT_UNKNOWN), D3D12_RESOURCE_STATE_COMMON);
        let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &mut arch as *mut _ as *mut c_void, std::mem::size_of_val(&arch) as u32)
            .expect("CheckFeatureSupport(ARCHITECTURE)");
        let custom_no_cpu = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: if arch.UMA.as_bool() { D3D12_MEMORY_POOL_L0 } else { D3D12_MEMORY_POOL_L1 },
            ..Default::default()
        };
        let custom_unmap_tex = create_committed(&dev, &custom_no_cpu, D3D12_HEAP_FLAG_NONE, &tex2d_desc(16, 16, D3D12_TEXTURE_LAYOUT_UNKNOWN), D3D12_RESOURCE_STATE_COMMON);
        let mut reserved_tex: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&tex2d_desc(128, 128, D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved_tex)
            .expect("CreateReservedResource");
        let reserved_tex = reserved_tex.unwrap();

        for r in [&def_buf, &def_tex, &custom_unmap_tex, &reserved_tex] {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let err = r.Map(0, None, Some(&mut ptr)).unwrap_err();
            assert_eq!(err.code(), E_INVALIDARG);
            assert!(ptr.is_null());
        }
        let orig_tex_va: GpuVaFn = std::mem::transmute(real::orig_fn(def_tex.as_raw(), 11));
        assert_eq!(def_tex.GetGPUVirtualAddress(), 0);
        assert_eq!(orig_tex_va(def_tex.as_raw()), 0);

        // 3. WriteToSubresource & CUSTOM WRITE_BACK / L0 Map parity (buffer + UNKNOWN-layout texture).
        let custom_wb = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let wb_buf = create_committed(&dev, &custom_wb, D3D12_HEAP_FLAG_NONE, &buffer_desc(4096, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON);
        let mut p_wb_hook: *mut c_void = std::ptr::null_mut();
        let mut p_wb_orig: *mut c_void = std::ptr::null_mut();
        let orig_wb_map: MapFn = std::mem::transmute(real::orig_fn(wb_buf.as_raw(), 8));
        let orig_wb_unmap: UnmapFn = std::mem::transmute(real::orig_fn(wb_buf.as_raw(), 9));
        assert_eq!(wb_buf.Map(0, Some(&empty_range), Some(&mut p_wb_hook)).map_or_else(|e| e.code().0, |_| 0), S_OK.0);
        assert_eq!(orig_wb_map(wb_buf.as_raw(), 0, &empty_range, &mut p_wb_orig), S_OK.0);
        assert!(!p_wb_hook.is_null() && !p_wb_orig.is_null());
        orig_wb_unmap(wb_buf.as_raw(), 0, std::ptr::null());
        wb_buf.Unmap(0, None);

        let wb_tex = create_committed(&dev, &custom_wb, D3D12_HEAP_FLAG_NONE, &tex2d_desc(8, 8, D3D12_TEXTURE_LAYOUT_UNKNOWN), D3D12_RESOURCE_STATE_COMMON);
        let src_a = [0x1122_3344u32; 8 * 8];
        let src_b = [0x5566_7788u32; 8 * 8];
        let dst_box = D3D12_BOX { left: 0, top: 0, front: 0, right: 8, bottom: 8, back: 1 };
        let hook_write = wb_tex.WriteToSubresource(0, Some(&dst_box), src_a.as_ptr() as *const c_void, 8 * 4, 8 * 8 * 4);
        let orig_write: WriteSubFn = std::mem::transmute(real::orig_fn(wb_tex.as_raw(), 12));
        let orig_hr = orig_write(wb_tex.as_raw(), 0, &dst_box, src_b.as_ptr() as *const c_void, 8 * 4, 8 * 8 * 4);
        assert_eq!(hook_write.map_or_else(|e| e.code().0, |_| 0), orig_hr);
        // Opaque UNKNOWN-layout CPU textures only accept Map(..., None) for Write/ReadFromSubresource.
        let orig_tex_map: MapFn = std::mem::transmute(real::orig_fn(wb_tex.as_raw(), 8));
        let orig_tex_unmap: UnmapFn = std::mem::transmute(real::orig_fn(wb_tex.as_raw(), 9));
        let hook_map_null = wb_tex.Map(0, None, None).map_or_else(|e| e.code().0, |_| 0);
        let orig_map_null = orig_tex_map(wb_tex.as_raw(), 0, std::ptr::null(), std::ptr::null_mut());
        assert_eq!(hook_map_null, orig_map_null);
        if hook_map_null == S_OK.0 {
            wb_tex.Unmap(0, None);
            orig_tex_unmap(wb_tex.as_raw(), 0, std::ptr::null());
        }

        // 4. Console #11 GetGPUVirtualAddress + Map/Unmap fallback (`texture_memory` + `TITLE_ADDRESS`).
        let title_va = call_console(def_tex.as_raw(), 11, &[]) as u64;
        assert_ne!(title_va, 0, "console GetGPUVirtualAddress on texture must reserve lazy title memory");
        assert_eq!(title_va & 0xFFFF, 0);
        assert_eq!(call_console(def_tex.as_raw(), 11, &[]) as u64, title_va);
        let mut tex_map_ptr: *mut c_void = std::ptr::null_mut();
        def_tex.Map(0, None, Some(&mut tex_map_ptr)).expect("Map on TITLE_ADDRESS texture");
        assert_eq!(tex_map_ptr as u64, title_va);
        std::ptr::write_volatile(tex_map_ptr as *mut u32, 0xCAFE_BABE);
        assert_eq!(std::ptr::read_volatile(tex_map_ptr as *const u32), 0xCAFE_BABE);
        def_tex.Unmap(0, None);

        // 5. Console CreateCommittedResource (#76) buffer placement in UMA when UMA is active.
        if real::uma_gpu_va() != 0 {
            let props_def = heap_props(D3D12_HEAP_TYPE_DEFAULT);
            let buf_d = buffer_desc(4096, D3D12_RESOURCE_FLAG_NONE);
            let mut raw_buf: *mut c_void = std::ptr::null_mut();
            let hr = call_console(
                dev.as_raw(),
                76,
                &[
                    &props_def as *const _ as usize,
                    0,
                    &buf_d as *const _ as usize,
                    D3D12_RESOURCE_STATE_COMMON.0 as usize,
                    0,
                    &ID3D12Resource::IID as *const GUID as usize,
                    &mut raw_buf as *mut _ as usize,
                ],
            ) as i32;
            assert_eq!(hr, S_OK.0, "console CreateCommittedResource buffer failed: {hr:#x}");
            let console_buf = ID3D12Resource::from_raw(raw_buf);
            let (uma_addr, uma_bytes) = real::console_buffer(console_buf.as_raw() as usize)
                .expect("UMA-backed console buffer expected when uma_gpu_va != 0");
            assert_eq!(uma_bytes, 4096);
            assert_eq!(call_console(console_buf.as_raw(), 11, &[]) as usize, uma_addr);
            let mut mapped: *mut c_void = std::ptr::null_mut();
            console_buf.Map(0, None, Some(&mut mapped)).expect("console buffer Map");
            assert_eq!(mapped as usize, uma_addr);
            std::ptr::write_volatile(mapped as *mut u32, 0xDEAD_BEEF);
            assert_eq!(std::ptr::read_volatile(uma_addr as *const u32), 0xDEAD_BEEF);
            console_buf.Unmap(0, None);
            drop(console_buf);
        }
    }
}

#[test]
fn descriptor_heap_console_start_handles_match_host_and_console_heap() {
    let Some(dev) = adopted_device() else { return };
    unsafe {
        type OrigStartFn = unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64;

        // 1. Host-created RTV and CBV_SRV_UAV heaps: console #9/#10 rax-return ABI matches PC orig_fn.
        let rtv_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
            .expect("RTV heap");
        let orig_cpu: OrigStartFn = std::mem::transmute(real::orig_fn(rtv_heap.as_raw(), 9));
        let mut expected_rtv_cpu = 0u64;
        orig_cpu(rtv_heap.as_raw(), &mut expected_rtv_cpu);
        assert_ne!(expected_rtv_cpu, 0);
        assert_eq!(call_console(rtv_heap.as_raw(), 9, &[]) as u64, expected_rtv_cpu);

        let pc_srv_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .expect("host shader-visible CBV_SRV_UAV heap");
        let orig_srv_cpu: OrigStartFn = std::mem::transmute(real::orig_fn(pc_srv_heap.as_raw(), 9));
        let orig_srv_gpu: OrigStartFn = std::mem::transmute(real::orig_fn(pc_srv_heap.as_raw(), 10));
        let (mut pc_cpu, mut pc_gpu) = (0u64, 0u64);
        orig_srv_cpu(pc_srv_heap.as_raw(), &mut pc_cpu);
        orig_srv_gpu(pc_srv_heap.as_raw(), &mut pc_gpu);
        assert_eq!(call_console(pc_srv_heap.as_raw(), 9, &[]) as u64, pc_cpu);
        assert_eq!(call_console(pc_srv_heap.as_raw(), 10, &[]) as u64, pc_gpu);
        assert!(console_heap::start(pc_srv_heap.as_raw() as usize).is_none());

        // 2. Console-created CBV_SRV_UAV (XBOX_EXTRA_CBV_SRV_UAV = 4) and SAMPLER heaps via #14.
        for (ty, count) in [(4u32, 16u32), (D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32, 8u32)] {
            let desc = [ty, count, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32, 0u32];
            let mut raw_heap: *mut c_void = std::ptr::null_mut();
            let hr = call_console(
                dev.as_raw(),
                14,
                &[
                    desc.as_ptr() as usize,
                    &ID3D12DescriptorHeap::IID as *const GUID as usize,
                    &mut raw_heap as *mut _ as usize,
                ],
            ) as i32;
            assert_eq!(hr, S_OK.0, "console CreateDescriptorHeap(type={ty}) failed: {hr:#x}");
            let heap = ID3D12DescriptorHeap::from_raw(raw_heap);
            let raw_addr = heap.as_raw() as usize;
            let cpu_x = call_console(heap.as_raw(), 9, &[]) as u64;
            let gpu_x = call_console(heap.as_raw(), 10, &[]) as u64;
            if let Some(shadow) = console_heap::start(raw_addr) {
                assert_ne!(shadow, 0);
                assert_eq!(cpu_x, shadow);
                assert_eq!(gpu_x, shadow);
            } else {
                let o_cpu: OrigStartFn = std::mem::transmute(real::orig_fn(heap.as_raw(), 9));
                let o_gpu: OrigStartFn = std::mem::transmute(real::orig_fn(heap.as_raw(), 10));
                let (mut e_cpu, mut e_gpu) = (0u64, 0u64);
                o_cpu(heap.as_raw(), &mut e_cpu);
                o_gpu(heap.as_raw(), &mut e_gpu);
                assert_eq!(cpu_x, e_cpu);
                assert_eq!(gpu_x, e_gpu);
            }
            drop(heap);
            assert!(console_heap::start(raw_addr).is_none(), "heap_release must unregister console heap");
        }
    }
}

#[test]
fn fence_and_command_queue_sync_timing_and_allocator_reset_match_host_and_console() {
    let Some(dev) = adopted_device() else { return };
    unsafe {
        type TsFreqFn = unsafe extern "system" fn(*mut c_void, *mut u64) -> i32;
        type CalibFn = unsafe extern "system" fn(*mut c_void, *mut u64, *mut u64) -> i32;
        type QueueDescFn = unsafe extern "system" fn(*mut c_void, *mut D3D12_COMMAND_QUEUE_DESC) -> *mut D3D12_COMMAND_QUEUE_DESC;
        type AllocResetFn = unsafe extern "system" fn(*mut c_void) -> i32;

        let q_direct: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("DIRECT queue");
        let q_compute: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).expect("COMPUTE queue");
        let q_copy: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() }).expect("COPY queue");

        // 1. Host-driver GetDesc (#18), GetTimestampFrequency (#16), GetClockCalibration (#17).
        for q in [&q_direct, &q_compute, &q_copy] {
            let orig_get_desc: QueueDescFn = std::mem::transmute(real::orig_fn(q.as_raw(), 18));
            let mut orig_desc = D3D12_COMMAND_QUEUE_DESC::default();
            orig_get_desc(q.as_raw(), &mut orig_desc);
            assert_eq!(q.GetDesc(), orig_desc);
        }
        for q in [&q_direct, &q_compute] {
            let orig_freq_fn: TsFreqFn = std::mem::transmute(real::orig_fn(q.as_raw(), 16));
            let mut orig_freq = 0u64;
            let orig_freq_hr = orig_freq_fn(q.as_raw(), &mut orig_freq);
            let hook_freq = q.GetTimestampFrequency();
            assert_eq!(hook_freq.as_ref().map_or_else(|e| e.code().0, |_| 0), orig_freq_hr);
            if let Ok(freq) = hook_freq {
                assert_eq!(freq, orig_freq);
                assert!(freq > 0);
            }

            let orig_calib_fn: CalibFn = std::mem::transmute(real::orig_fn(q.as_raw(), 17));
            let (mut og, mut oc) = (0u64, 0u64);
            let orig_calib_hr = orig_calib_fn(q.as_raw(), &mut og, &mut oc);
            let (mut hg, mut hc) = (0u64, 0u64);
            let hook_calib = q.GetClockCalibration(&mut hg, &mut hc);
            assert_eq!(hook_calib.as_ref().map_or_else(|e| e.code().0, |_| 0), orig_calib_hr);
            if hook_calib.is_ok() {
                assert!(hg > 0 && hc > 0);
            }
        }

        // 2. Console #18 GetDesc before layout is established writes at least Type + Priority (8 bytes).
        let mut pre_desc_bytes = [0xAAu8; 16];
        let ret_ptr = call_console(q_compute.as_raw(), 18, &[pre_desc_bytes.as_mut_ptr() as usize]);
        assert_eq!(ret_ptr, pre_desc_bytes.as_ptr() as usize);
        let pre_desc = std::ptr::read_unaligned(pre_desc_bytes.as_ptr() as *const D3D12_COMMAND_QUEUE_DESC);
        assert_eq!(pre_desc.Type, D3D12_COMMAND_LIST_TYPE_COMPUTE);
        assert_eq!(pre_desc.Priority, 0);

        // 3. Fence #12 completed_value_address + GPU Wait/ExecuteCommandLists/Copy/Signal/SignalX/CPU Signal.
        let fence_a: ID3D12Fence = dev.CreateFence(5, D3D12_FENCE_FLAG_NONE).expect("fence_a");
        let fence_b: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence_b");
        let cell_ptr = call_console(fence_a.as_raw(), 12, &[]) as *const AtomicU64;
        assert!(!cell_ptr.is_null(), "Fence #12 returned null cell");
        assert_eq!(cell_ptr as usize % 64, 0, "Fence #12 cell must be 64-byte aligned");
        assert_eq!(call_console(fence_a.as_raw(), 12, &[]) as *const AtomicU64, cell_ptr);
        assert_eq!((*cell_ptr).load(Ordering::Acquire), 5);

        let upload = create_committed(&dev, &heap_props(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &buffer_desc(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ);
        let gpu_buf = create_committed(&dev, &heap_props(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &buffer_desc(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON);
        let readback = create_committed(&dev, &heap_props(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE, &buffer_desc(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST);

        let mut p_up: *mut c_void = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p_up)).expect("Map upload");
        std::ptr::write_bytes(p_up as *mut u8, 0xA5, 256);
        upload.Unmap(0, None);

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("command list");
        list.CopyBufferRegion(&gpu_buf, 0, &upload, 0, 256);
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&gpu_buf),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        };
        list.ResourceBarrier(&[barrier]);
        list.CopyBufferRegion(&readback, 0, &gpu_buf, 0, 256);
        list.Close().expect("list Close");

        // Host Wait(fence_b, 1), ExecuteCommandLists, and console #14 Signal(fence_a, 6) (sets QUEUE_OLDER).
        q_direct.Wait(&fence_b, 1).expect("queue Wait");
        q_direct.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
        let hr_sig = call_console(q_direct.as_raw(), 14, &[fence_a.as_raw() as usize, 6]) as i32;
        assert_eq!(hr_sig, S_OK.0);
        assert_eq!(fence_a.GetCompletedValue(), 5, "queue must stay blocked until fence_b reaches 1");

        // Unblock via hooked CPU ID3D12Fence::Signal (#10) and wait for GPU copy + xcl-fences refresh.
        fence_b.Signal(1).expect("CPU fence_b.Signal(1)");
        wait_fence_event(&fence_a, 6);
        assert_eq!(fence_a.GetCompletedValue(), 6);

        let wait_cell = |target: u64| {
            let t0 = Instant::now();
            while (*cell_ptr).load(Ordering::Acquire) < target {
                assert!(t0.elapsed() < Duration::from_secs(5), "fence cell did not reach {target}");
                std::thread::yield_now();
            }
        };
        wait_cell(6);

        let mut p_rb: *mut c_void = std::ptr::null_mut();
        let rb_range = D3D12_RANGE { Begin: 0, End: 256 };
        readback.Map(0, Some(&rb_range), Some(&mut p_rb)).expect("Map readback");
        assert_eq!(std::slice::from_raw_parts(p_rb as *const u8, 256), &[0xA5u8; 256]);
        let empty = D3D12_RANGE { Begin: 0, End: 0 };
        readback.Unmap(0, Some(&empty));

        // 4. Console #23 SignalX and #15 Wait + immediate CPU fence_signal update.
        let hr_sigx = call_console(q_direct.as_raw(), 23, &[fence_a.as_raw() as usize, 7, 0x3]) as i32;
        assert_eq!(hr_sigx, S_OK.0);
        wait_fence_event(&fence_a, 7);
        wait_cell(7);

        let hr_waitx = call_console(q_direct.as_raw(), 15, &[fence_b.as_raw() as usize, 2]) as i32;
        assert_eq!(hr_waitx, S_OK.0);
        let hr_sigx2 = call_console(q_direct.as_raw(), 23, &[fence_a.as_raw() as usize, 8, 0]) as i32;
        assert_eq!(hr_sigx2, S_OK.0);
        assert_eq!(fence_a.GetCompletedValue(), 7);
        fence_b.Signal(2).expect("CPU fence_b.Signal(2)");
        wait_fence_event(&fence_a, 8);
        wait_cell(8);

        fence_a.Signal(11).expect("CPU fence_a.Signal(11)");
        assert_eq!(fence_a.GetCompletedValue(), 11);
        assert_eq!((*cell_ptr).load(Ordering::Acquire), 11, "CPU fence_signal must update cell immediately");

        // 5. Console queue #16 / #17 / #18 (full 16-byte GetDesc) once QUEUE_OLDER is established.
        let mut x_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE(-1),
            Priority: -1,
            Flags: D3D12_COMMAND_QUEUE_FLAGS(-1),
            NodeMask: u32::MAX,
        };
        call_console(q_copy.as_raw(), 18, &[&mut x_desc as *mut _ as usize]);
        assert_eq!(x_desc, q_copy.GetDesc());

        let mut x_freq = 0u64;
        let hr_xfreq = call_console(q_direct.as_raw(), 16, &[&mut x_freq as *mut _ as usize]) as i32;
        let mut host_freq = 0u64;
        let host_freq_hr = q_direct.GetTimestampFrequency().map(|f| host_freq = f).map_or_else(|e| e.code().0, |_| 0);
        assert_eq!(hr_xfreq, host_freq_hr);
        assert_eq!(x_freq, host_freq);

        let (mut xg, mut xc) = (0u64, 0u64);
        let hr_xcalib = call_console(q_direct.as_raw(), 17, &[&mut xg as *mut _ as usize, &mut xc as *mut _ as usize]) as i32;
        let (mut hg, mut hc) = (0u64, 0u64);
        let host_calib_hr = q_direct.GetClockCalibration(&mut hg, &mut hc).map_or_else(|e| e.code().0, |_| 0);
        assert_eq!(hr_xcalib, host_calib_hr);
        if hr_xcalib == S_OK.0 {
            assert!(xg > 0 && xc > 0);
        }

        // 6. ID3D12CommandAllocator::Reset (#8, alloc_reset) parity & re-recording on the allocator.
        let orig_alloc_reset: AllocResetFn = std::mem::transmute(real::orig_fn(alloc.as_raw(), 8));
        alloc.Reset().expect("allocator Reset");
        assert_eq!(orig_alloc_reset(alloc.as_raw()), S_OK.0);
        list.Reset(&alloc, None).expect("list Reset after allocator Reset");
        list.Close().expect("list re-Close");
        q_direct.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
        q_direct.Signal(&fence_a, 12).expect("post-reset Signal");
        wait_fence_event(&fence_a, 12);
    }
}
