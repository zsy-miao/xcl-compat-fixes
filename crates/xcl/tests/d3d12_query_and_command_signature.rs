//! Host D3D12 driver parity tests for `ID3D12Device::CreateQueryHeap` (slot 39) and
//! `ID3D12Device::CreateCommandSignature` (slot 41), part of #480: compares XCL's hooked
//! path directly against the unhooked PC driver without requiring game files or console DLLs.

use std::ffi::c_void;

use windows::core::{Interface, GUID};
use windows::Win32::Graphics::Direct3D12::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, real};

const SLOT_CREATE_QUERY_HEAP: usize = 39;
const SLOT_CREATE_COMMAND_SIGNATURE: usize = 41;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;

type CreateQueryHeapFn = unsafe extern "system" fn(*mut c_void, *const D3D12_QUERY_HEAP_DESC, *const GUID, *mut *mut c_void) -> i32;
type CreateCommandSignatureFn = unsafe extern "system" fn(*mut c_void, *const D3D12_COMMAND_SIGNATURE_DESC, *mut c_void, *const GUID, *mut *mut c_void) -> i32;

fn init_device() -> Option<*mut c_void> {
    std::env::set_var("XCL_UMA_MB", "64");
    if d3d12x::backend() != d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        runner::skip_test("no GPU");
        return None;
    }
    Some(real::device_raw())
}

#[test]
fn create_query_heap_matches_host_driver() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let vt = *(dev as *const *const usize);
        let hooked_addr = *vt.add(SLOT_CREATE_QUERY_HEAP);
        let host_addr = real::orig_fn(dev, SLOT_CREATE_QUERY_HEAP);
        assert_ne!(hooked_addr, host_addr, "CreateQueryHeap slot must be hooked");
        let hooked: CreateQueryHeapFn = std::mem::transmute(hooked_addr);
        let host: CreateQueryHeapFn = std::mem::transmute(host_addr);
        let iid = ID3D12QueryHeap::IID;

        for (ty, count) in [
            (D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1u32),
            (D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 16),
            (D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, 64),
            (D3D12_QUERY_HEAP_TYPE_SO_STATISTICS, 16),
        ] {
            let desc = D3D12_QUERY_HEAP_DESC { Type: ty, Count: count, NodeMask: 0 };
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = hooked(dev, &desc, &iid, &mut p_hook);
            let hr_host = host(dev, &desc, &iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "HRESULT mismatch for {ty:?} count={count}");
            assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
            let _qh_hook = ID3D12QueryHeap::from_raw(p_hook);
            let _qh_host = ID3D12QueryHeap::from_raw(p_host);
        }

        // Invalid descriptors (Count = 0, invalid Type, invalid NodeMask) match host failure.
        for bad in [
            D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 0, NodeMask: 0 },
            D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE(999), Count: 4, NodeMask: 0 },
            D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 4, NodeMask: 0x10 },
        ] {
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = hooked(dev, &bad, &iid, &mut p_hook);
            let hr_host = host(dev, &bad, &iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "HRESULT mismatch for {bad:?}");
            assert_eq!((p_hook.is_null(), p_host.is_null()), (hr_hook < 0, hr_host < 0), "output for {bad:?}");
            // The PC runtime refuses these; vkd3d-proton (under Wine) accepts some of them.
            assert!(hr_host < 0 || xcl::util::under_wine(), "expected failure for {bad:?}: host={hr_host:#x}");
            for p in [p_hook, p_host].into_iter().filter(|p| !p.is_null()) {
                drop(ID3D12QueryHeap::from_raw(p));
            }
        }

        // Unknown console IID fails on host with E_NOINTERFACE; XCL falls back to ID3D12QueryHeap.
        let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 8, NodeMask: 0 };
        let console_iid = GUID::from_u128(0x11223344_5566_7788_99aa_bbccddeeff04);
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr_host = host(dev, &desc, &console_iid, &mut p_host);
        let hr_hook = hooked(dev, &desc, &console_iid, &mut p_hook);
        assert_eq!(hr_host, E_NOINTERFACE);
        assert!(p_host.is_null());
        assert!(hr_hook >= 0 && !p_hook.is_null(), "console IID fallback failed: {hr_hook:#x}");
        let _qh = ID3D12QueryHeap::from_raw(p_hook);
    }
}

#[test]
fn create_command_signature_matches_host_driver() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let vt = *(dev as *const *const usize);
        let hooked_addr = *vt.add(SLOT_CREATE_COMMAND_SIGNATURE);
        let host_addr = real::orig_fn(dev, SLOT_CREATE_COMMAND_SIGNATURE);
        assert_ne!(hooked_addr, host_addr, "CreateCommandSignature slot must be hooked");
        let hooked: CreateCommandSignatureFn = std::mem::transmute(hooked_addr);
        let host: CreateCommandSignatureFn = std::mem::transmute(host_addr);
        let iid = ID3D12CommandSignature::IID;

        for (arg_ty, stride) in [
            (D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 16u32),
            (D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, 20),
            (D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 12),
        ] {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: arg_ty, ..Default::default() };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = hooked(dev, &desc, std::ptr::null_mut(), &iid, &mut p_hook);
            let hr_host = host(dev, &desc, std::ptr::null_mut(), &iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "HRESULT mismatch for {arg_ty:?} stride={stride}");
            assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
            let _cs_hook = ID3D12CommandSignature::from_raw(p_hook);
            let _cs_host = ID3D12CommandSignature::from_raw(p_host);
        }

        // Invalid descriptors (stride too small, zero arguments, invalid NodeMask) match host failure.
        let draw_arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, ..Default::default() };
        let small_stride = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 4, NumArgumentDescs: 1, pArgumentDescs: &draw_arg, NodeMask: 0 };
        let empty_args = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 0, pArgumentDescs: &draw_arg, NodeMask: 0 };
        let bad_node = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 1, pArgumentDescs: &draw_arg, NodeMask: 0x10 };
        for desc in [small_stride, empty_args, bad_node] {
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = hooked(dev, &desc, std::ptr::null_mut(), &iid, &mut p_hook);
            let hr_host = host(dev, &desc, std::ptr::null_mut(), &iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "HRESULT mismatch for an invalid signature");
            assert_eq!((p_hook.is_null(), p_host.is_null()), (hr_hook < 0, hr_host < 0), "output for an invalid signature");
            // The PC runtime refuses these; vkd3d-proton (under Wine) accepts some of them.
            assert!(hr_host < 0 || xcl::util::under_wine(), "expected failure: host={hr_host:#x}");
            for p in [p_hook, p_host].into_iter().filter(|p| !p.is_null()) {
                drop(ID3D12CommandSignature::from_raw(p));
            }
        }

        // Unknown console IID fails on host with E_NOINTERFACE; XCL falls back to ID3D12CommandSignature.
        let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 1, pArgumentDescs: &draw_arg, NodeMask: 0 };
        let console_iid = GUID::from_u128(0x11223344_5566_7788_99aa_bbccddeeff05);
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr_host = host(dev, &desc, std::ptr::null_mut(), &console_iid, &mut p_host);
        let hr_hook = hooked(dev, &desc, std::ptr::null_mut(), &console_iid, &mut p_hook);
        assert_eq!(hr_host, E_NOINTERFACE);
        assert!(p_host.is_null());
        assert!(hr_hook >= 0 && !p_hook.is_null(), "console IID fallback failed: {hr_hook:#x}");
        let _cs = ID3D12CommandSignature::from_raw(p_hook);
    }
}
