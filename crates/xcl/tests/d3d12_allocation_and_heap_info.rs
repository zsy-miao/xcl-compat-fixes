//! Host-driver parity checks for `ID3D12Device::GetResourceAllocationInfo` (slot 25)
//! and `ID3D12Device::GetCustomHeapProperties` (slot 26) (Part of #480).
//!
//! Compares XCL's console-side hooks against the unhooked host `ID3D12Device` methods on
//! the exact same adopted adapter, with no console files or game data:
//! - Calls through a synthetic registered title-image trampoline so `guard::make_split`
//!   dispatches to the real console hooks (`get_resource_allocation_info_x` and
//!   `get_custom_heap_properties_x`), just like a retail title.
//! - Single buffers, multi-buffer arrays, and invalid buffers (`Width == 0`) agree with
//!   the host driver on `{SizeInBytes, Alignment}` in both C++ ABI `(this, ret, ...)` and
//!   C ABI `(ret, this, ...)`.
//! - `GetCustomHeapProperties` agrees with the host driver for `DEFAULT`, `UPLOAD`, and
//!   `READBACK` heap types in both C++ ABI and C ABI.
//! - Standard 2D textures (`RGBA8`, `BC7`) return consistent, power-of-two-aligned,
//!   non-error XG allocation info across both C++ and C ABIs.

use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::{GUID, Interface};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN,
    DXGI_SAMPLE_DESC,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc,
};

use xcl::hle::d3d12x::{self, Backend};
use xcl::modules::{self, MappedModule};
use xcl::pe::OwnedExports;

const GET_RESOURCE_ALLOCATION_INFO_SLOT: usize = 25;
const GET_CUSTOM_HEAP_PROPERTIES_SLOT: usize = 26;

type AllocInfoHostFn = unsafe extern "system" fn(
    this: *mut c_void,
    ret: *mut [u64; 2],
    visible_mask: u32,
    count: u32,
    descs: *const D3D12_RESOURCE_DESC,
) -> *mut [u64; 2];

type CustomHeapHostFn = unsafe extern "system" fn(
    this: *mut c_void,
    ret: *mut [u32; 5],
    node_mask: u32,
    heap_type: u32,
) -> *mut [u32; 5];

/// Generic 5-arg title-caller trampoline: `fn(target, a, b, c, d, e) -> usize`.
type TitleCall5Fn =
    unsafe extern "system" fn(usize, usize, usize, usize, usize, usize) -> usize;

struct TestDevices {
    xcl_raw: usize,
    host_alloc_info: AllocInfoHostFn,
    hooked_alloc_info: usize,
    host_custom_heap: CustomHeapHostFn,
    hooked_custom_heap: usize,
    title_call5: TitleCall5Fn,
}

static DEVICES: OnceLock<Option<TestDevices>> = OnceLock::new();

/// Allocates a 4 KiB executable page containing a tiny x86-64 forwarding stub and
/// registers it as a synthetic title module so `guard::make_split` treats calls from it
/// as title calls (routing to XCL's console hooks).
unsafe fn make_title_caller() -> TitleCall5Fn {
    // fn(rcx=target, rdx=a, r8=b, r9=c, [rsp+0x28]=d, [rsp+0x30]=e) -> rax
    #[rustfmt::skip]
    const STUB: &[u8] = &[
        0x48, 0x83, 0xEC, 0x38,             // sub rsp, 0x38  (16-byte align + shadow + 5th arg)
        0x48, 0x89, 0xC8,                   // mov rax, rcx   (target)
        0x48, 0x89, 0xD1,                   // mov rcx, rdx   (arg1 = a)
        0x4C, 0x89, 0xC2,                   // mov rdx, r8    (arg2 = b)
        0x4D, 0x89, 0xC8,                   // mov r8, r9     (arg3 = c)
        0x4C, 0x8B, 0x4C, 0x24, 0x60,       // mov r9, [rsp+0x60]  (arg4 = d)
        0x4C, 0x8B, 0x54, 0x24, 0x68,       // mov r10, [rsp+0x68] (arg5 = e)
        0x4C, 0x89, 0x54, 0x24, 0x20,       // mov [rsp+0x20], r10
        0xFF, 0xD0,                         // call rax
        0x48, 0x83, 0xC4, 0x38,             // add rsp, 0x38
        0xC3,                               // ret
    ];

    let page = VirtualAlloc(
        ptr::null(),
        4096,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_EXECUTE_READWRITE,
    );
    assert!(!page.is_null(), "VirtualAlloc failed for synthetic title caller");
    ptr::copy_nonoverlapping(STUB.as_ptr(), page as *mut u8, STUB.len());

    modules::register_mapped(Arc::new(MappedModule {
        name: "synthetic_title_test.exe".to_string(),
        path: PathBuf::from("synthetic_title_test.exe"),
        base: page as usize,
        size: 4096,
        entry_rva: 0,
        is_dll: false,
        exports: OwnedExports::default(),
        tls_callbacks: Mutex::new(Vec::new()),
        no_thread_calls: AtomicBool::new(true),
        initialized: AtomicBool::new(true),
        stack_reserve: 0,
    }));

    assert_eq!(
        modules::image_base_for_addr(page as usize + STUB.len() - 1),
        Some(page as usize),
        "synthetic title caller should be recognized as a mapped title image"
    );

    std::mem::transmute(page)
}

fn init_devices() -> Option<&'static TestDevices> {
    let entry = DEVICES.get_or_init(|| unsafe {
        if d3d12x::backend() != Backend::Real {
            return None;
        }
        assert_eq!(
            std::mem::size_of::<D3D12_RESOURCE_DESC>(),
            0x38,
            "D3D12_RESOURCE_DESC must be 0x38 bytes"
        );
        assert_eq!(
            std::mem::size_of::<D3D12_HEAP_PROPERTIES>(),
            20,
            "D3D12_HEAP_PROPERTIES must be 20 bytes"
        );

        let mut xcl_raw: *mut c_void = ptr::null_mut();
        let iid = ID3D12Device::IID;
        let hr = d3d12x::D3D12CreateDevice(
            ptr::null_mut(),
            D3D_FEATURE_LEVEL_11_0.0 as u32,
            &iid as *const GUID as *const c_void,
            &mut xcl_raw,
        );
        if hr < 0 || xcl_raw.is_null() {
            return None;
        }
        assert_eq!(
            xcl_raw,
            d3d12x::real::device_raw(),
            "D3D12CreateDevice should return the adopted XCL backend device"
        );

        let vt = *(xcl_raw as *const *const usize);

        let orig_alloc = d3d12x::real::orig_fn(xcl_raw, GET_RESOURCE_ALLOCATION_INFO_SLOT);
        let hooked_alloc = *vt.add(GET_RESOURCE_ALLOCATION_INFO_SLOT);
        assert_ne!(orig_alloc, 0, "orig_fn returned null for slot 25");
        assert_ne!(
            hooked_alloc, orig_alloc,
            "XCL should have hooked ID3D12Device::GetResourceAllocationInfo (slot 25)"
        );

        let orig_heap = d3d12x::real::orig_fn(xcl_raw, GET_CUSTOM_HEAP_PROPERTIES_SLOT);
        let hooked_heap = *vt.add(GET_CUSTOM_HEAP_PROPERTIES_SLOT);
        assert_ne!(orig_heap, 0, "orig_fn returned null for slot 26");
        assert_ne!(
            hooked_heap, orig_heap,
            "XCL should have hooked ID3D12Device::GetCustomHeapProperties (slot 26)"
        );

        let title_call5 = make_title_caller();

        Some(TestDevices {
            xcl_raw: xcl_raw as usize,
            host_alloc_info: std::mem::transmute(orig_alloc),
            hooked_alloc_info: hooked_alloc,
            host_custom_heap: std::mem::transmute(orig_heap),
            hooked_custom_heap: hooked_heap,
            title_call5,
        })
    });
    if entry.is_none() {
        xcl::gpu::recomp::runner::skip_test("no GPU");
    }
    entry.as_ref()
}

fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

fn tex2d_desc(width: u64, height: u32, mips: u16, format: DXGI_FORMAT) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

unsafe fn query_host_alloc(devs: &TestDevices, descs: &[D3D12_RESOURCE_DESC]) -> [u64; 2] {
    let mut out = [0u64; 2];
    let ret = (devs.host_alloc_info)(
        devs.xcl_raw as *mut c_void,
        &mut out,
        0,
        descs.len() as u32,
        descs.as_ptr(),
    );
    assert_eq!(ret, &mut out as *mut [u64; 2], "host slot 25 must return ret pointer");
    out
}

unsafe fn query_xcl_alloc_cpp(devs: &TestDevices, descs: &[D3D12_RESOURCE_DESC]) -> [u64; 2] {
    let mut out = [0u64; 2];
    let ret = (devs.title_call5)(
        devs.hooked_alloc_info,
        devs.xcl_raw,
        &mut out as *mut [u64; 2] as usize,
        0,
        descs.len(),
        descs.as_ptr() as usize,
    );
    assert_eq!(ret, &mut out as *mut [u64; 2] as usize, "XCL C++ slot 25 must return ret pointer");
    out
}

unsafe fn query_xcl_alloc_c(devs: &TestDevices, descs: &[D3D12_RESOURCE_DESC]) -> [u64; 2] {
    let mut out = [0u64; 2];
    let ret = (devs.title_call5)(
        devs.hooked_alloc_info,
        &mut out as *mut [u64; 2] as usize,
        devs.xcl_raw,
        0,
        descs.len(),
        descs.as_ptr() as usize,
    );
    assert_eq!(ret, &mut out as *mut [u64; 2] as usize, "XCL C slot 25 must return ret pointer");
    out
}

#[test]
fn buffer_allocation_info_matches_host_in_cpp_and_c_abi() {
    let Some(devs) = init_devices() else { return };
    unsafe {
        let single_cases: &[(&str, D3D12_RESOURCE_DESC)] = &[
            ("buf_256b", buffer_desc(256)),
            ("buf_4k", buffer_desc(4096)),
            ("buf_12345b", buffer_desc(12_345)),
            ("buf_64k", buffer_desc(65_536)),
            ("buf_1m", buffer_desc(1024 * 1024)),
        ];

        for (label, desc) in single_cases {
            let host = query_host_alloc(devs, &[*desc]);
            let xcl_cpp = query_xcl_alloc_cpp(devs, &[*desc]);
            let xcl_c = query_xcl_alloc_c(devs, &[*desc]);
            assert_eq!(
                xcl_cpp, host,
                "{label}: XCL C++ ABI {:?} != host {:?}",
                xcl_cpp, host
            );
            assert_eq!(
                xcl_c, host,
                "{label}: XCL C ABI {:?} != host {:?}",
                xcl_c, host
            );
        }

        // Zero-width buffer: native D3D12 returns SizeInBytes == UINT64_MAX (where XCL leaves
        // Alignment = 0), whereas vkd3d-proton returns [0, 65536]. Compare XCL's SizeInBytes
        // (and non-error Alignment) against the host driver so both Windows and Wine/vkd3d pass.
        let invalid = buffer_desc(0);
        let host_inv = query_host_alloc(devs, &[invalid]);
        let cpp_inv = query_xcl_alloc_cpp(devs, &[invalid]);
        let c_inv = query_xcl_alloc_c(devs, &[invalid]);
        assert_eq!(cpp_inv[0], host_inv[0], "zero-width buffer C++ SizeInBytes != host");
        assert_eq!(c_inv[0], host_inv[0], "zero-width buffer C SizeInBytes != host");
        if host_inv[0] != u64::MAX {
            assert_eq!(cpp_inv[1], host_inv[1], "zero-width buffer C++ Alignment != host");
            assert_eq!(c_inv[1], host_inv[1], "zero-width buffer C Alignment != host");
        }

        // Multi-buffer array: combined size & alignment must match the host driver.
        let multi = [buffer_desc(256), buffer_desc(70_000), buffer_desc(131_072)];
        let host = query_host_alloc(devs, &multi);
        let xcl_cpp = query_xcl_alloc_cpp(devs, &multi);
        let xcl_c = query_xcl_alloc_c(devs, &multi);
        assert_eq!(xcl_cpp, host, "multi-buffer C++ ABI {:?} != host {:?}", xcl_cpp, host);
        assert_eq!(xcl_c, host, "multi-buffer C ABI {:?} != host {:?}", xcl_c, host);
        assert_ne!(host[0], u64::MAX, "valid multi-buffer array should not fail");
        assert!(host[0] >= 256 + 70_000 + 131_072);
    }
}

#[test]
fn custom_heap_properties_match_host_in_cpp_and_c_abi() {
    let Some(devs) = init_devices() else { return };
    unsafe {
        let heap_types = [
            ("DEFAULT", D3D12_HEAP_TYPE_DEFAULT.0 as u32),
            ("UPLOAD", D3D12_HEAP_TYPE_UPLOAD.0 as u32),
            ("READBACK", D3D12_HEAP_TYPE_READBACK.0 as u32),
        ];

        // 8-byte aligned return buffer to avoid #1130 (`real::is_adopted` unaligned read).
        #[repr(align(8))]
        struct AlignedHeapProps([u32; 5]);

        for (label, ty) in heap_types {
            let mut host = AlignedHeapProps([0u32; 5]);
            let r_host = (devs.host_custom_heap)(devs.xcl_raw as *mut c_void, &mut host.0, 0, ty);
            assert_eq!(r_host, &mut host.0 as *mut [u32; 5]);

            let mut xcl_cpp = AlignedHeapProps([0u32; 5]);
            let r_cpp = (devs.title_call5)(
                devs.hooked_custom_heap,
                devs.xcl_raw,
                &mut xcl_cpp.0 as *mut [u32; 5] as usize,
                0,
                ty as usize,
                0,
            );
            assert_eq!(r_cpp, &mut xcl_cpp.0 as *mut [u32; 5] as usize);

            let mut xcl_c = AlignedHeapProps([0u32; 5]);
            let r_c = (devs.title_call5)(
                devs.hooked_custom_heap,
                &mut xcl_c.0 as *mut [u32; 5] as usize,
                devs.xcl_raw,
                0,
                ty as usize,
                0,
            );
            assert_eq!(r_c, &mut xcl_c.0 as *mut [u32; 5] as usize);

            assert_eq!(
                xcl_cpp.0, host.0,
                "GetCustomHeapProperties({label}) C++ ABI {:?} != host {:?}",
                xcl_cpp.0, host.0
            );
            assert_eq!(
                xcl_c.0, host.0,
                "GetCustomHeapProperties({label}) C ABI {:?} != host {:?}",
                xcl_c.0, host.0
            );
            assert_eq!(
                host.0[0],
                D3D12_HEAP_TYPE_CUSTOM.0 as u32,
                "GetCustomHeapProperties({label}) should report D3D12_HEAP_TYPE_CUSTOM"
            );
        }
    }
}

#[test]
fn texture_allocation_info_agrees_across_cpp_and_c_abi() {
    let Some(devs) = init_devices() else { return };
    unsafe {
        let cases: &[(&str, D3D12_RESOURCE_DESC)] = &[
            ("tex2d_256_rgba8", tex2d_desc(256, 256, 1, DXGI_FORMAT_R8G8B8A8_UNORM)),
            ("tex2d_256_rgba8_mips", tex2d_desc(256, 256, 4, DXGI_FORMAT_R8G8B8A8_UNORM)),
            ("tex2d_128_bc7", tex2d_desc(128, 128, 1, DXGI_FORMAT_BC7_UNORM)),
        ];

        for (label, desc) in cases {
            let xcl_cpp = query_xcl_alloc_cpp(devs, &[*desc]);
            let xcl_c = query_xcl_alloc_c(devs, &[*desc]);
            assert_eq!(
                xcl_c, xcl_cpp,
                "{label}: C ABI {:?} != C++ ABI {:?}",
                xcl_c, xcl_cpp
            );
            assert_ne!(xcl_cpp[0], u64::MAX, "{label}: texture allocation failed");
            assert!(xcl_cpp[0] > 0, "{label}: expected non-zero SizeInBytes");
            assert!(
                xcl_cpp[1].is_power_of_two(),
                "{label}: expected power-of-two Alignment, got {}",
                xcl_cpp[1]
            );
        }
    }
}
