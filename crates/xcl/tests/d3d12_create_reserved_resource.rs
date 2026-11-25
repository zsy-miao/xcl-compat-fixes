//! Host-driver parity checks for `ID3D12Device::CreateReservedResource` (Part of #480).
//!
//! Compares XCL's hooked `CreateReservedResource` (vtable slot 30) against a raw host
//! `ID3D12Device` on the same machine, with no console files or game data:
//! - Valid buffers and 2D/3D reserved textures agree with the host on `HRESULT`, `GetDesc()`,
//!   and `GetResourceTiling()` (covering both AMD standard-tile and NVIDIA packed-mip-tail
//!   drivers).
//! - Invalid descriptors fail with the same `HRESULT` as the host driver.
//! - Unknown (console-only) resource IIDs return `E_NOINTERFACE` from the host driver, while
//!   XCL falls back to `IID_ID3D12Resource` and returns a usable resource.
//! - Render-target clear values and `ppv == null` match the host driver without crashing.

use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

use windows::core::{GUID, Interface};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN,
    DXGI_SAMPLE_DESC,
};

use xcl::hle::d3d12x::{self, Backend};
use xcl::util::HRESULT;

const E_NOINTERFACE_HR: HRESULT = 0x8000_4002u32 as HRESULT;
const CREATE_RESERVED_RESOURCE_SLOT: usize = 30;

type CreateReservedResourceFn = unsafe extern "system" fn(
    this: *mut c_void,
    desc: *const D3D12_RESOURCE_DESC,
    initial_state: D3D12_RESOURCE_STATES,
    clear_value: *const D3D12_CLEAR_VALUE,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT;

struct TestDevices {
    xcl_raw: usize,
    host_create_reserved: CreateReservedResourceFn,
}

static DEVICES: OnceLock<Option<TestDevices>> = OnceLock::new();

unsafe fn call_host_create_reserved(
    devs: &TestDevices,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
    clear: *const D3D12_CLEAR_VALUE,
    riid: &GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    (devs.host_create_reserved)(devs.xcl_raw as *mut c_void, desc, state, clear, riid, ppv)
}

unsafe fn call_xcl_create_reserved(
    devs: &TestDevices,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
    clear: *const D3D12_CLEAR_VALUE,
    riid: &GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    let dev_raw = devs.xcl_raw as *mut c_void;
    let vt = *(dev_raw as *const *const usize);
    let f: CreateReservedResourceFn =
        std::mem::transmute(*vt.add(CREATE_RESERVED_RESOURCE_SLOT));
    f(dev_raw, desc, state, clear, riid, ppv)
}

unsafe fn host_device(devs: &TestDevices) -> ID3D12Device {
    // Borrowed wrapper on the same adopted device: bump refcount so dropping keeps DEVICES alive.
    let raw = devs.xcl_raw as *mut c_void;
    let vt = *(raw as *const *const usize);
    let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(1));
    add_ref(raw);
    ID3D12Device::from_raw(raw)
}

fn init_devices(_test_name: &str) -> Option<&'static TestDevices> {
    let entry = DEVICES.get_or_init(|| unsafe {
        if d3d12x::backend() != Backend::Real {
            return None;
        }
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
        // Use the adopted XCL backend device for both paths so host and XCL are guaranteed to
        // share the exact same adapter (including on multi-GPU systems with XCL_ADAPTER set).
        assert_eq!(
            xcl_raw,
            d3d12x::real::device_raw(),
            "D3D12CreateDevice should return the adopted XCL backend device"
        );
        let orig_ptr = d3d12x::real::orig_fn(xcl_raw, CREATE_RESERVED_RESOURCE_SLOT);
        assert_ne!(orig_ptr, 0, "orig_fn returned null for CreateReservedResource slot 30");
        let hooked_vt = *(xcl_raw as *const *const usize);
        let hooked_fn = *hooked_vt.add(CREATE_RESERVED_RESOURCE_SLOT);
        assert_ne!(
            hooked_fn,
            orig_ptr,
            "XCL should have hooked ID3D12Device::CreateReservedResource (slot 30)"
        );
        let host_create_reserved: CreateReservedResourceFn = std::mem::transmute(orig_ptr);
        Some(TestDevices {
            xcl_raw: xcl_raw as usize,
            host_create_reserved,
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

fn tex2d_desc(
    w: u64,
    h: u32,
    mips: u16,
    fmt: DXGI_FORMAT,
    layout: D3D12_TEXTURE_LAYOUT,
    flags: D3D12_RESOURCE_FLAGS,
) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: w,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: mips,
        Format: fmt,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: layout,
        Flags: flags,
    }
}

fn tex3d_desc(w: u64, h: u32, d: u16, mips: u16, fmt: DXGI_FORMAT) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Alignment: 0,
        Width: w,
        Height: h,
        DepthOrArraySize: d,
        MipLevels: mips,
        Format: fmt,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

unsafe fn query_tiling(
    dev: &ID3D12Device,
    res: &ID3D12Resource,
) -> (u32, D3D12_PACKED_MIP_INFO, D3D12_TILE_SHAPE, D3D12_SUBRESOURCE_TILING) {
    let mut total = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut shape = D3D12_TILE_SHAPE::default();
    let mut count = 1u32;
    let mut sub0 = D3D12_SUBRESOURCE_TILING::default();
    dev.GetResourceTiling(
        res,
        Some(&mut total),
        Some(&mut packed),
        Some(&mut shape),
        Some(&mut count),
        0,
        &mut sub0,
    );
    (total, packed, shape, sub0)
}

unsafe fn query_tiling_range(
    dev: &ID3D12Device,
    res: &ID3D12Resource,
    start_sub: u32,
    count: u32,
) -> (u32, D3D12_PACKED_MIP_INFO, D3D12_TILE_SHAPE, Vec<D3D12_SUBRESOURCE_TILING>) {
    let mut total = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut shape = D3D12_TILE_SHAPE::default();
    let mut actual_count = count;
    let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); count as usize];
    dev.GetResourceTiling(
        res,
        Some(&mut total),
        Some(&mut packed),
        Some(&mut shape),
        Some(&mut actual_count),
        start_sub,
        tilings.as_mut_ptr(),
    );
    tilings.truncate(actual_count as usize);
    (total, packed, shape, tilings)
}

#[test]
fn pc_reserved_resources_match_host_driver() {
    unsafe {
        let Some(devs) = init_devices("pc_reserved_resources_match_host_driver") else {
            return;
        };
        let dev_for_tiling = host_device(devs);
        let iid = ID3D12Resource::IID;

        let cases: &[(&str, D3D12_RESOURCE_DESC)] = &[
            ("buf_64k", buffer_desc(65536)),
            ("buf_256k", buffer_desc(256 * 1024)),
            (
                "tex2d_8x8_rgba8",
                tex2d_desc(
                    8,
                    8,
                    1,
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                    D3D12_RESOURCE_FLAG_NONE,
                ),
            ),
            (
                "tex2d_256x256_rgba8_mips",
                tex2d_desc(
                    256,
                    256,
                    4,
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                    D3D12_RESOURCE_FLAG_NONE,
                ),
            ),
            (
                "tex2d_128x128_bc7",
                tex2d_desc(
                    128,
                    128,
                    1,
                    DXGI_FORMAT_BC7_UNORM,
                    D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                    D3D12_RESOURCE_FLAG_NONE,
                ),
            ),
            (
                "tex3d_64x64x64_rgba8",
                tex3d_desc(64, 64, 64, 1, DXGI_FORMAT_R8G8B8A8_UNORM),
            ),
            // Invalid descriptors: XCL must refuse them the same way the host driver does.
            ("invalid_buf_zero", buffer_desc(0)),
            (
                "invalid_tex2d_row_major",
                tex2d_desc(
                    64,
                    64,
                    1,
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    D3D12_RESOURCE_FLAG_NONE,
                ),
            ),
        ];

        for (label, desc) in cases {
            let mut host_res_raw: *mut c_void = ptr::null_mut();
            let mut xcl_res_raw: *mut c_void = ptr::null_mut();

            let host_hr = call_host_create_reserved(
                devs,
                desc,
                D3D12_RESOURCE_STATE_COMMON,
                ptr::null(),
                &iid,
                &mut host_res_raw,
            );
            let xcl_hr = call_xcl_create_reserved(
                devs,
                desc,
                D3D12_RESOURCE_STATE_COMMON,
                ptr::null(),
                &iid,
                &mut xcl_res_raw,
            );

            assert_eq!(
                xcl_hr, host_hr,
                "{label}: XCL HRESULT ({xcl_hr:#x}) != host ({host_hr:#x})"
            );

            if host_hr >= 0 {
                assert!(!host_res_raw.is_null(), "{label}: host returned null");
                assert!(!xcl_res_raw.is_null(), "{label}: XCL returned null");

                let host_res = ID3D12Resource::from_raw(host_res_raw);
                let xcl_res = ID3D12Resource::from_raw(xcl_res_raw);

                let hd = host_res.GetDesc();
                let xd = xcl_res.GetDesc();
                assert_eq!(xd.Dimension, hd.Dimension, "{label}: Dimension");
                assert_eq!(xd.Width, hd.Width, "{label}: Width");
                assert_eq!(xd.Height, hd.Height, "{label}: Height");
                assert_eq!(xd.DepthOrArraySize, hd.DepthOrArraySize, "{label}: DepthOrArraySize");
                assert_eq!(xd.MipLevels, hd.MipLevels, "{label}: MipLevels");
                assert_eq!(xd.Format, hd.Format, "{label}: Format");
                assert_eq!(xd.Layout, hd.Layout, "{label}: Layout");
                assert_eq!(xd.Flags, hd.Flags, "{label}: Flags");

                let ht = query_tiling(&dev_for_tiling, &host_res);
                let xt = query_tiling(&dev_for_tiling, &xcl_res);
                assert_eq!(xt.0, ht.0, "{label}: total_tiles");
                assert_eq!(xt.1, ht.1, "{label}: packed_mip_info");
                assert_eq!(xt.2, ht.2, "{label}: tile_shape");
                assert_eq!(xt.3, ht.3, "{label}: subresource_tiling[0]");

                if desc.MipLevels > 1 {
                    let ht_range = query_tiling_range(&dev_for_tiling, &host_res, 1, 2);
                    let xt_range = query_tiling_range(&dev_for_tiling, &xcl_res, 1, 2);
                    assert_eq!(xt_range.0, ht_range.0, "{label}: total_tiles (range)");
                    assert_eq!(xt_range.1, ht_range.1, "{label}: packed_mip_info (range)");
                    assert_eq!(xt_range.2, ht_range.2, "{label}: tile_shape (range)");
                    assert_eq!(xt_range.3, ht_range.3, "{label}: subresource_tiling range");
                }
            } else {
                assert!(xcl_res_raw.is_null(), "{label}: failed call left non-null output");
            }
        }
    }
}

#[test]
fn unknown_console_iid_falls_back_to_pc_resource_interface() {
    unsafe {
        let Some(devs) =
            init_devices("unknown_console_iid_falls_back_to_pc_resource_interface")
        else {
            return;
        };
        let fake_console_iid = GUID::from_u128(0xdeadbeef_1234_5678_9abc_def012345678);
        let desc = tex2d_desc(
            128,
            128,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            D3D12_RESOURCE_FLAG_NONE,
        );

        let mut host_res_raw: *mut c_void = ptr::null_mut();
        let host_hr = call_host_create_reserved(
            devs,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &fake_console_iid,
            &mut host_res_raw,
        );
        assert_eq!(
            host_hr, E_NOINTERFACE_HR,
            "host driver should reject unknown IID with E_NOINTERFACE, got {host_hr:#x}"
        );
        assert!(host_res_raw.is_null());

        let mut xcl_res_raw: *mut c_void = ptr::null_mut();
        let xcl_hr = call_xcl_create_reserved(
            devs,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &fake_console_iid,
            &mut xcl_res_raw,
        );
        assert!(
            xcl_hr >= 0 && !xcl_res_raw.is_null(),
            "XCL CreateReservedResource should retry with IID_ID3D12Resource, got {xcl_hr:#x}"
        );
        let res = ID3D12Resource::from_raw(xcl_res_raw);
        let d = res.GetDesc();
        assert_eq!(d.Width, 128);
        assert_eq!(d.Height, 128);
        assert_eq!(d.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
    }
}

#[test]
fn reserved_resource_clear_value_and_null_ppv_match_host() {
    unsafe {
        let Some(devs) =
            init_devices("reserved_resource_clear_value_and_null_ppv_match_host")
        else {
            return;
        };
        let iid = ID3D12Resource::IID;
        let rt_desc = tex2d_desc(
            128,
            128,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        );
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.25, 0.5, 0.75, 1.0] },
        };

        // 1. Valid RT reserved resource with optimized clear value.
        let mut host_res_raw: *mut c_void = ptr::null_mut();
        let mut xcl_res_raw: *mut c_void = ptr::null_mut();
        let host_hr = call_host_create_reserved(
            devs,
            &rt_desc,
            D3D12_RESOURCE_STATE_COMMON,
            &clear,
            &iid,
            &mut host_res_raw,
        );
        let xcl_hr = call_xcl_create_reserved(
            devs,
            &rt_desc,
            D3D12_RESOURCE_STATE_COMMON,
            &clear,
            &iid,
            &mut xcl_res_raw,
        );
        assert_eq!(xcl_hr, host_hr, "RT with clear value HRESULT mismatch");
        if host_hr >= 0 {
            let _h = ID3D12Resource::from_raw(host_res_raw);
            let x = ID3D12Resource::from_raw(xcl_res_raw);
            assert_eq!(x.GetDesc().Flags, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        }

        // 2. Null ppv (D3D12 allows null ppv to validate args and return S_FALSE / error).
        let host_null_hr = call_host_create_reserved(
            devs,
            &rt_desc,
            D3D12_RESOURCE_STATE_COMMON,
            &clear,
            &iid,
            ptr::null_mut(),
        );
        let xcl_null_hr = call_xcl_create_reserved(
            devs,
            &rt_desc,
            D3D12_RESOURCE_STATE_COMMON,
            &clear,
            &iid,
            ptr::null_mut(),
        );
        assert_eq!(
            xcl_null_hr, host_null_hr,
            "null ppv HRESULT mismatch: XCL={xcl_null_hr:#x}, host={host_null_hr:#x}"
        );
    }
}
