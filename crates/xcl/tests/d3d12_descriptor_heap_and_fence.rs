//! Host D3D12 driver parity tests for `ID3D12Device::CreateDescriptorHeap` (slot 14),
//! `GetDescriptorHandleIncrementSize` (slot 15), and `CreateFence` (slot 36) (part of #480):
//! compares XCL's hooked paths directly against the unhooked PC driver on the same adapter
//! without requiring game files or console DLLs.

use std::ffi::c_void;

use windows::core::{Interface, GUID};
use windows::Win32::Graphics::Direct3D12::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, real};

const SLOT_CREATE_DESCRIPTOR_HEAP: usize = 14;
const SLOT_GET_INCREMENT_SIZE: usize = 15;
const SLOT_CREATE_FENCE: usize = 36;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;

type CreateDescriptorHeapFn =
    unsafe extern "system" fn(*mut c_void, *const D3D12_DESCRIPTOR_HEAP_DESC, *const GUID, *mut *mut c_void) -> i32;
type GetIncrementSizeFn = unsafe extern "system" fn(*mut c_void, D3D12_DESCRIPTOR_HEAP_TYPE) -> u32;
type CreateFenceFn =
    unsafe extern "system" fn(*mut c_void, u64, D3D12_FENCE_FLAGS, *const GUID, *mut *mut c_void) -> i32;

fn init_device() -> Option<*mut c_void> {
    std::env::set_var("XCL_UMA_MB", "64");
    if d3d12x::backend() != d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        runner::skip_test("no GPU");
        return None;
    }
    Some(real::device_raw())
}

unsafe fn slot_pair<F: Copy>(dev: *mut c_void, slot: usize, name: &str) -> (F, F) {
    assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<usize>());
    let vt = *(dev as *const *const usize);
    let hooked_addr = *vt.add(slot);
    let host_addr = real::orig_fn(dev, slot);
    assert_ne!(hooked_addr, host_addr, "{name} slot {slot} must be hooked");
    (std::mem::transmute_copy(&hooked_addr), std::mem::transmute_copy(&host_addr))
}

#[test]
fn descriptor_handle_increment_size_pc_path_matches_host_driver() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let (hooked, host): (GetIncrementSizeFn, GetIncrementSizeFn) =
            slot_pair(dev, SLOT_GET_INCREMENT_SIZE, "GetDescriptorHandleIncrementSize");

        // Slot 15 is wrapped with `guard::make_split(dev_increment_size_x, pc)`: host/PC callers
        // must see the exact host driver increment size for every descriptor heap type.
        for ty in [
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        ] {
            let h_host = host(dev, ty);
            let h_hook = hooked(dev, ty);
            assert!(h_host > 0, "heap type {}: host increment size must be > 0", ty.0);
            assert_eq!(h_hook, h_host, "heap type {}: PC-path increment must match host driver", ty.0);
        }
    }
}

#[test]
fn create_descriptor_heap_matches_host_driver_and_handles_console_iid() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let (hooked, host): (CreateDescriptorHeapFn, CreateDescriptorHeapFn) =
            slot_pair(dev, SLOT_CREATE_DESCRIPTOR_HEAP, "CreateDescriptorHeap");
        let riid = &ID3D12DescriptorHeap::IID;

        for (label, ty, count) in [
            ("RTV 8", D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8),
            ("DSV 4", D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 4),
        ] {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: ty,
                NumDescriptors: count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            let (mut p_host, mut p_hook) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_host = host(dev, &desc, riid, &mut p_host);
            let hr_hook = hooked(dev, &desc, riid, &mut p_hook);
            assert!(hr_host >= 0 && !p_host.is_null(), "{label}: host hr={hr_host:#x}");
            assert!(hr_hook >= 0 && !p_hook.is_null(), "{label}: hooked hr={hr_hook:#x}");

            let d_host = ID3D12DescriptorHeap::from_raw(p_host).GetDesc();
            let d_hook = ID3D12DescriptorHeap::from_raw(p_hook).GetDesc();
            assert_eq!(d_hook.Type, d_host.Type, "{label}: Type");
            assert_eq!(d_hook.NumDescriptors, d_host.NumDescriptors, "{label}: NumDescriptors");
            assert_eq!(d_hook.Flags, d_host.Flags, "{label}: Flags");
            assert_eq!(d_hook.NodeMask, d_host.NodeMask, "{label}: NodeMask");
        }

        for (label, ty, count, flags) in [
            ("CBV_SRV_UAV 16 cpu", D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
            ("CBV_SRV_UAV 16 vis", D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
            ("SAMPLER 8 cpu", D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 8, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
            ("SAMPLER 8 vis", D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 8, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
        ] {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: count, Flags: flags, NodeMask: 0 };
            let (mut p_host, mut p_hook) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_host = host(dev, &desc, riid, &mut p_host);
            let hr_hook = hooked(dev, &desc, riid, &mut p_hook);
            assert!(hr_host >= 0 && !p_host.is_null(), "{label}: host hr={hr_host:#x}");
            assert!(hr_hook >= 0 && !p_hook.is_null(), "{label}: hooked hr={hr_hook:#x}");
            let _h_host = ID3D12DescriptorHeap::from_raw(p_host);
            let h_hook = ID3D12DescriptorHeap::from_raw(p_hook);
            assert_ne!(h_hook.GetCPUDescriptorHandleForHeapStart().ptr, 0, "{label}: CPU start");
            if flags == D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE {
                assert_ne!(h_hook.GetGPUDescriptorHandleForHeapStart().ptr, 0, "{label}: GPU start");
            }
        }

        let invalid = [
            ("CBV_SRV_UAV zero", D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 0, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
            ("SAMPLER zero", D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 0, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
            ("RTV zero", D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 0, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
            ("DSV zero", D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 0, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
            ("RTV shader-visible", D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
            ("DSV shader-visible", D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
        ];
        for (label, ty, count, flags) in invalid {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: count, Flags: flags, NodeMask: 0 };
            let (mut p_host, mut p_hook) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_host = host(dev, &desc, riid, &mut p_host);
            let hr_hook = hooked(dev, &desc, riid, &mut p_hook);
            assert_eq!(hr_hook < 0, hr_host < 0, "{label}: host hr={hr_host:#x}, hooked hr={hr_hook:#x}");
            assert_eq!(p_hook.is_null(), p_host.is_null(), "{label}: out pointer parity");
            if !p_host.is_null() {
                drop(ID3D12DescriptorHeap::from_raw(p_host));
            }
            if !p_hook.is_null() {
                drop(ID3D12DescriptorHeap::from_raw(p_hook));
            }
        }

        let rtv = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 2,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let fake_console_iid = GUID::from_u128(0x2222_3333_4444_5555_6666_7777_8888_9999);
        let mut p_host = std::ptr::null_mut();
        assert_eq!(host(dev, &rtv, &fake_console_iid, &mut p_host), E_NOINTERFACE);
        assert!(p_host.is_null());

        let mut p_hook = std::ptr::null_mut();
        let hr_hook = hooked(dev, &rtv, &fake_console_iid, &mut p_hook);
        assert!(hr_hook >= 0 && !p_hook.is_null(), "XCL must fall back to IID_ID3D12DescriptorHeap: {hr_hook:#x}");
        let d_hook = ID3D12DescriptorHeap::from_raw(p_hook).GetDesc();
        assert_eq!(d_hook.Type, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        assert_eq!(d_hook.NumDescriptors, 2);
    }
}

#[test]
fn create_fence_and_signal_match_host_driver_and_handle_console_iid() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let (hooked, host): (CreateFenceFn, CreateFenceFn) = slot_pair(dev, SLOT_CREATE_FENCE, "CreateFence");
        let riid = &ID3D12Fence::IID;

        for init in [0u64, 42, u64::MAX - 1] {
            let (mut p_host, mut p_hook) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_host = host(dev, init, D3D12_FENCE_FLAG_NONE, riid, &mut p_host);
            let hr_hook = hooked(dev, init, D3D12_FENCE_FLAG_NONE, riid, &mut p_hook);
            assert!(hr_host >= 0 && !p_host.is_null(), "init={init}: host hr={hr_host:#x}");
            assert!(hr_hook >= 0 && !p_hook.is_null(), "init={init}: hooked hr={hr_hook:#x}");

            let f_host = ID3D12Fence::from_raw(p_host);
            let f_hook = ID3D12Fence::from_raw(p_hook);
            assert_eq!(f_hook.GetCompletedValue(), f_host.GetCompletedValue());
            assert_eq!(f_hook.GetCompletedValue(), init);

            f_host.Signal(init + 1).expect("host Signal");
            f_hook.Signal(init + 1).expect("hooked Signal");
            assert_eq!(f_hook.GetCompletedValue(), f_host.GetCompletedValue());
            assert_eq!(f_hook.GetCompletedValue(), init + 1);
        }

        let hr_null_host = host(dev, 0, D3D12_FENCE_FLAG_NONE, riid, std::ptr::null_mut());
        let hr_null_hook = hooked(dev, 0, D3D12_FENCE_FLAG_NONE, riid, std::ptr::null_mut());
        assert_eq!(hr_null_hook, hr_null_host, "ppv == null HRESULT parity");

        let fake_console_iid = GUID::from_u128(0x3333_4444_5555_6666_7777_8888_9999_0000);
        let mut p_host = std::ptr::null_mut();
        assert_eq!(host(dev, 7, D3D12_FENCE_FLAG_NONE, &fake_console_iid, &mut p_host), E_NOINTERFACE);
        assert!(p_host.is_null());

        let mut p_hook = std::ptr::null_mut();
        let hr_hook = hooked(dev, 7, D3D12_FENCE_FLAG_NONE, &fake_console_iid, &mut p_hook);
        assert!(hr_hook >= 0 && !p_hook.is_null(), "XCL must fall back to IID_ID3D12Fence: {hr_hook:#x}");
        let f_hook = ID3D12Fence::from_raw(p_hook);
        assert_eq!(f_hook.GetCompletedValue(), 7);
    }
}
