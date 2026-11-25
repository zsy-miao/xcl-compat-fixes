//! Host D3D12 driver parity tests for `ID3D12Device::CreateCommittedResource` (slot 27,
//! part of #480): compares XCL's hooked console path directly against the unhooked PC driver
//! on the same adapter without requiring game files or console DLLs.

use std::ffi::c_void;

use windows::core::{Interface, GUID};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, real};

const SLOT_CREATE_COMMITTED_RESOURCE: usize = 27;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;

type CreateCommittedFn = unsafe extern "system" fn(
    *mut c_void,
    *const D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_FLAGS,
    *const D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_STATES,
    *const D3D12_CLEAR_VALUE,
    *const GUID,
    *mut *mut c_void,
) -> i32;

/// `dev_create_committed_resource_x` reads 56 bytes from `desc` to inspect console-extended
/// layout fields past the 52-byte `D3D12_RESOURCE_DESC`; zero-pad the tail so reads stay in bounds.
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

unsafe fn fns(dev: *mut c_void) -> (CreateCommittedFn, CreateCommittedFn) {
    let vt = *(dev as *const *const usize);
    let hooked_addr = *vt.add(SLOT_CREATE_COMMITTED_RESOURCE);
    let host_addr = real::orig_fn(dev, SLOT_CREATE_COMMITTED_RESOURCE);
    assert_ne!(hooked_addr, host_addr, "CreateCommittedResource slot must be hooked");
    (std::mem::transmute(hooked_addr), std::mem::transmute(host_addr))
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

fn tex3d_desc(w: u64, h: u32, d: u16, format: DXGI_FORMAT) -> PaddedDesc {
    PaddedDesc::new(D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: w,
        Height: h,
        DepthOrArraySize: d,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        ..Default::default()
    })
}

fn assert_same_desc(label: &str, a: &D3D12_RESOURCE_DESC, b: &D3D12_RESOURCE_DESC) {
    assert_eq!(a.Dimension, b.Dimension, "{label}: Dimension");
    assert_eq!(a.Width, b.Width, "{label}: Width");
    assert_eq!(a.Height, b.Height, "{label}: Height");
    assert_eq!(a.DepthOrArraySize, b.DepthOrArraySize, "{label}: DepthOrArraySize");
    assert_eq!(a.MipLevels, b.MipLevels, "{label}: MipLevels");
    assert_eq!(a.Format, b.Format, "{label}: Format");
    assert_eq!(a.SampleDesc.Count, b.SampleDesc.Count, "{label}: SampleDesc.Count");
    assert_eq!(a.SampleDesc.Quality, b.SampleDesc.Quality, "{label}: SampleDesc.Quality");
    assert_eq!(a.Layout, b.Layout, "{label}: Layout");
    assert_eq!(a.Flags, b.Flags, "{label}: Flags");
}

#[test]
fn committed_resources_match_host_driver_desc_and_status() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let (hooked, host) = fns(dev);
        let riid = &ID3D12Resource::IID;

        let rt_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.25, 0.5, 0.75, 1.0] },
        };
        let ds_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } },
        };

        let cases: [(&str, D3D12_HEAP_TYPE, PaddedDesc, D3D12_RESOURCE_STATES, Option<D3D12_CLEAR_VALUE>); 8] = [
            ("buffer 64 KiB DEFAULT", D3D12_HEAP_TYPE_DEFAULT, buf_desc(64 << 10), D3D12_RESOURCE_STATE_COMMON, None),
            ("buffer 4 KiB UPLOAD", D3D12_HEAP_TYPE_UPLOAD, buf_desc(4 << 10), D3D12_RESOURCE_STATE_GENERIC_READ, None),
            ("buffer 4 KiB READBACK", D3D12_HEAP_TYPE_READBACK, buf_desc(4 << 10), D3D12_RESOURCE_STATE_COPY_DEST, None),
            ("2D 64x64 RGBA8 4 mips", D3D12_HEAP_TYPE_DEFAULT, tex2d_desc(64, 64, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None),
            ("2D 128x128 BC1", D3D12_HEAP_TYPE_DEFAULT, tex2d_desc(128, 128, 1, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None),
            ("2D 64x64 RT with clear", D3D12_HEAP_TYPE_DEFAULT, tex2d_desc(64, 64, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), D3D12_RESOURCE_STATE_RENDER_TARGET, Some(rt_clear)),
            ("2D 64x64 D32 with clear", D3D12_HEAP_TYPE_DEFAULT, tex2d_desc(64, 64, 1, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(ds_clear)),
            ("3D 16x16x16 RGBA8", D3D12_HEAP_TYPE_DEFAULT, tex3d_desc(16, 16, 16, DXGI_FORMAT_R8G8B8A8_UNORM), D3D12_RESOURCE_STATE_COMMON, None),
        ];

        for (label, heap_ty, desc, state, clear) in cases {
            let props = D3D12_HEAP_PROPERTIES { Type: heap_ty, ..Default::default() };
            let clear_ptr = clear.as_ref().map_or(std::ptr::null(), |c| c as *const _);
            let (mut p_host, mut p_hook) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_host = host(dev, &props, D3D12_HEAP_FLAG_NONE, &desc.pc, state, clear_ptr, riid, &mut p_host);
            let hr_hook = hooked(dev, &props, D3D12_HEAP_FLAG_NONE, &desc.pc, state, clear_ptr, riid, &mut p_hook);
            assert!(hr_host >= 0 && !p_host.is_null(), "{label}: host hr={hr_host:#x}");
            assert!(hr_hook >= 0 && !p_hook.is_null(), "{label}: hooked hr={hr_hook:#x}");

            let r_host = ID3D12Resource::from_raw(p_host);
            let r_hook = ID3D12Resource::from_raw(p_hook);
            assert_same_desc(label, &r_hook.GetDesc(), &r_host.GetDesc());
            if desc.pc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
                assert_ne!(r_host.GetGPUVirtualAddress(), 0, "{label}: host GPU VA");
                assert_ne!(r_hook.GetGPUVirtualAddress(), 0, "{label}: hooked GPU VA");
            }
        }

        let invalid = [
            ("zero-width buffer", buf_desc(0)),
            ("zero-height 2D texture", tex2d_desc(64, 0, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)),
        ];
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        for (label, desc) in invalid {
            let (mut p_host, mut p_hook) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_host = host(dev, &props, D3D12_HEAP_FLAG_NONE, &desc.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), riid, &mut p_host);
            let hr_hook = hooked(dev, &props, D3D12_HEAP_FLAG_NONE, &desc.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), riid, &mut p_hook);
            assert_eq!(hr_hook < 0, hr_host < 0, "{label}: hooked {hr_hook:#x}, host {hr_host:#x}");
            assert_eq!((p_hook.is_null(), p_host.is_null()), (hr_hook < 0, hr_host < 0), "{label}: output");
            // The PC runtime refuses these; vkd3d-proton (under Wine) accepts some of them.
            assert!(hr_host < 0 || xcl::util::under_wine(), "{label}: expected host failure, got {hr_host:#x}");
            for p in [p_hook, p_host].into_iter().filter(|p| !p.is_null()) {
                drop(ID3D12Resource::from_raw(p));
            }
        }
    }
}

#[test]
fn committed_resource_unknown_console_iid_falls_back_to_pc_resource() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let (hooked, host) = fns(dev);
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = tex2d_desc(64, 64, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        let fake_console_iid = GUID::from_u128(0x1111_2222_3333_4444_5555_6666_7777_8888);

        let mut p_host = std::ptr::null_mut();
        let hr_host = host(dev, &props, D3D12_HEAP_FLAG_NONE, &desc.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &fake_console_iid, &mut p_host);
        assert_eq!(hr_host, E_NOINTERFACE, "host driver must reject unknown IID");
        assert!(p_host.is_null());

        let mut p_hook = std::ptr::null_mut();
        let hr_hook = hooked(dev, &props, D3D12_HEAP_FLAG_NONE, &desc.pc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &fake_console_iid, &mut p_hook);
        assert!(hr_hook >= 0 && !p_hook.is_null(), "XCL must fall back to IID_ID3D12Resource: {hr_hook:#x}");
        let r_hook = ID3D12Resource::from_raw(p_hook);
        assert_eq!(r_hook.GetDesc().Width, 64);
        assert_eq!(r_hook.GetDesc().Height, 64);
    }
}
