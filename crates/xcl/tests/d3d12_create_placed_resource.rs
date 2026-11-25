//! Host D3D12 driver parity tests for `ID3D12Device::CreatePlacedResource` (slot 29,
//! part of #480): compares XCL's hooked path directly against the unhooked PC driver
//! on the same adapter without requiring game files or console DLLs.

use std::ffi::c_void;

use windows::core::{Interface, GUID};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, real};

const SLOT_CREATE_HEAP: usize = 28;
const SLOT_CREATE_PLACED_RESOURCE: usize = 29;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;
const HEAP_BYTES: u64 = 4 * 1024 * 1024;

type CreateHeapFn = unsafe extern "system" fn(*mut c_void, *const D3D12_HEAP_DESC, *const GUID, *mut *mut c_void) -> i32;

type CreatePlacedFn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    u64,
    *const D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_STATES,
    *const D3D12_CLEAR_VALUE,
    *const GUID,
    *mut *mut c_void,
) -> i32;

/// `dev_create_placed_resource_x` can read 56 bytes from `desc` when console-routed;
/// zero-pad the tail so reads remain in bounds across every layout.
#[repr(C)]
#[derive(Clone, Copy)]
struct PaddedDesc {
    pc: D3D12_RESOURCE_DESC,
    console_tail: [u32; 4],
}

impl PaddedDesc {
    fn new(pc: D3D12_RESOURCE_DESC) -> Self {
        Self { pc, console_tail: [0; 4] }
    }
}

fn init_device() -> Option<*mut c_void> {
    std::env::set_var("XCL_UMA_MB", "64");
    if d3d12x::backend() != d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        runner::skip_test("no GPU");
        return None;
    }
    Some(real::device_raw())
}

unsafe fn fns(dev: *mut c_void) -> (CreatePlacedFn, CreatePlacedFn) {
    let vt = *(dev as *const *const usize);
    let hooked_addr = *vt.add(SLOT_CREATE_PLACED_RESOURCE);
    let host_addr = real::orig_fn(dev, SLOT_CREATE_PLACED_RESOURCE);
    assert_ne!(hooked_addr, host_addr, "CreatePlacedResource slot must be hooked");
    (std::mem::transmute(hooked_addr), std::mem::transmute(host_addr))
}

unsafe fn create_raw_heap(dev: *mut c_void, heap_type: D3D12_HEAP_TYPE, flags: D3D12_HEAP_FLAGS) -> ID3D12Heap {
    let create_heap: CreateHeapFn = std::mem::transmute(real::orig_fn(dev, SLOT_CREATE_HEAP));
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: HEAP_BYTES,
        Properties: D3D12_HEAP_PROPERTIES { Type: heap_type, ..Default::default() },
        Alignment: 0,
        Flags: flags,
    };
    let mut raw = std::ptr::null_mut();
    let hr = create_heap(dev, &desc, &ID3D12Heap::IID, &mut raw);
    assert!(hr >= 0 && !raw.is_null(), "raw CreateHeap({heap_type:?}, {flags:?}) failed: {hr:#x}");
    ID3D12Heap::from_raw(raw)
}

fn buf_desc(bytes: u64) -> PaddedDesc {
    PaddedDesc::new(D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    })
}

fn tex2d_desc(w: u64, h: u32, mips: u16, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> PaddedDesc {
    PaddedDesc::new(D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: w,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        ..Default::default()
    })
}

#[test]
fn create_placed_buffers_match_host_driver() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let (hooked, host) = fns(dev);
        let iid = ID3D12Resource::IID;
        for (heap_type, state) in [
            (D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON),
            (D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ),
            (D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST),
        ] {
            let heap = create_raw_heap(dev, heap_type, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
            for (bytes, offset) in [(256u64, 0u64), (65_536, 0x1_0000)] {
                let d = buf_desc(bytes);
                let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
                let hr_hook = hooked(dev, heap.as_raw(), offset, &d.pc, state, std::ptr::null(), &iid, &mut p_hook);
                let hr_host = host(dev, heap.as_raw(), offset, &d.pc, state, std::ptr::null(), &iid, &mut p_host);
                assert_eq!(hr_hook, hr_host, "HRESULT mismatch for {heap_type:?} buffer {bytes}@{offset:#x}");
                assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());

                let r_hook = ID3D12Resource::from_raw(p_hook);
                let r_host = ID3D12Resource::from_raw(p_host);
                assert_eq!(r_hook.GetDesc(), r_host.GetDesc());
                assert_ne!(r_hook.GetGPUVirtualAddress(), 0);
                assert_ne!(r_host.GetGPUVirtualAddress(), 0);

                if heap_type == D3D12_HEAP_TYPE_UPLOAD && offset == 0 {
                    let mut ptr: *mut c_void = std::ptr::null_mut();
                    r_hook.Map(0, None, Some(&mut ptr)).expect("placed upload buffer Map");
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut u32, 4);
                    slice.copy_from_slice(&[0xCAFE_BABEu32, 0x1234_5678, 0xDEAD_BEEF, 0x0BAD_F00D]);
                    assert_eq!(slice[0], 0xCAFE_BABE);
                    r_hook.Unmap(0, None);
                }
            }
        }
    }
}

#[test]
fn create_placed_textures_match_host_driver() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let (hooked, host) = fns(dev);
        let iid = ID3D12Resource::IID;

        let non_rt_heap = create_raw_heap(dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES);
        for (format, mips, offset) in [(DXGI_FORMAT_R8G8B8A8_UNORM, 1u16, 0u64), (DXGI_FORMAT_BC1_UNORM, 4, 0x10_0000)] {
            let d = tex2d_desc(64, 64, mips, format, D3D12_RESOURCE_FLAG_NONE);
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = hooked(dev, non_rt_heap.as_raw(), offset, &d.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut p_hook);
            let hr_host = host(dev, non_rt_heap.as_raw(), offset, &d.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "HRESULT mismatch for non-RT texture {format:?}");
            assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
            let r_hook = ID3D12Resource::from_raw(p_hook);
            let r_host = ID3D12Resource::from_raw(p_host);
            assert_eq!(r_hook.GetDesc(), r_host.GetDesc());
        }

        let rt_heap = create_raw_heap(dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES);
        let rt_desc = tex2d_desc(64, 64, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        let rt_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.25, 0.5, 0.75, 1.0] },
        };
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr_hook = hooked(dev, rt_heap.as_raw(), 0, &rt_desc.pc, D3D12_RESOURCE_STATE_RENDER_TARGET, &rt_clear, &iid, &mut p_hook);
        let hr_host = host(dev, rt_heap.as_raw(), 0, &rt_desc.pc, D3D12_RESOURCE_STATE_RENDER_TARGET, &rt_clear, &iid, &mut p_host);
        assert_eq!(hr_hook, hr_host, "HRESULT mismatch for placed RTV");
        assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
        assert_eq!(ID3D12Resource::from_raw(p_hook).GetDesc(), ID3D12Resource::from_raw(p_host).GetDesc());

        let ds_desc = tex2d_desc(64, 64, 1, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        let ds_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } },
        };
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr_hook = hooked(dev, rt_heap.as_raw(), 0x10_0000, &ds_desc.pc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &ds_clear, &iid, &mut p_hook);
        let hr_host = host(dev, rt_heap.as_raw(), 0x10_0000, &ds_desc.pc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &ds_clear, &iid, &mut p_host);
        assert_eq!(hr_hook, hr_host, "HRESULT mismatch for placed DSV");
        assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
        assert_eq!(ID3D12Resource::from_raw(p_hook).GetDesc(), ID3D12Resource::from_raw(p_host).GetDesc());
    }
}

#[test]
fn create_placed_errors_and_console_iid_fallback() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let (hooked, host) = fns(dev);
        let iid = ID3D12Resource::IID;
        let heap = create_raw_heap(dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
        let buf = buf_desc(4096);

        // Misaligned offset and offset past the end of the heap both fail identically to host.
        for bad_offset in [0x100u64, HEAP_BYTES] {
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = hooked(dev, heap.as_raw(), bad_offset, &buf.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut p_hook);
            let hr_host = host(dev, heap.as_raw(), bad_offset, &buf.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut p_host);
            assert!(hr_hook < 0 && hr_host < 0, "expected failure at offset {bad_offset:#x}: hook={hr_hook:#x}, host={hr_host:#x}");
            assert!(p_hook.is_null() && p_host.is_null());
        }

        // Null desc returns error without dereferencing. (vkd3d-proton, under Wine, dereferences it and
        // the process ends.)
        if !xcl::util::under_wine() {
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = hooked(dev, heap.as_raw(), 0, std::ptr::null(), D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut p_hook);
            let hr_host = host(dev, heap.as_raw(), 0, std::ptr::null(), D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut p_host);
            assert!(hr_hook < 0 && hr_host < 0, "expected failure on null desc: hook={hr_hook:#x}, host={hr_host:#x}");
            assert!(p_hook.is_null() && p_host.is_null());
        }

        // Unknown console IID fails on host with E_NOINTERFACE; XCL falls back to ID3D12Resource.
        let console_iid = GUID::from_u128(0x11223344_5566_7788_99aa_bbccddeeff03);
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr_host = host(dev, heap.as_raw(), 0, &buf.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &console_iid, &mut p_host);
        let hr_hook = hooked(dev, heap.as_raw(), 0, &buf.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &console_iid, &mut p_hook);
        assert_eq!(hr_host, E_NOINTERFACE);
        assert!(p_host.is_null());
        assert!(hr_hook >= 0 && !p_hook.is_null(), "console IID fallback failed: {hr_hook:#x}");

        let (mut p_ref, _) = (std::ptr::null_mut(), 0);
        let hr_ref = host(dev, heap.as_raw(), 0, &buf.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut p_ref);
        assert!(hr_ref >= 0);
        assert_eq!(ID3D12Resource::from_raw(p_hook).GetDesc(), ID3D12Resource::from_raw(p_ref).GetDesc());
    }
}
