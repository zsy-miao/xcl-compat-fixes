//! Host D3D12 driver parity tests for `CreateCommandQueue` (slot 8), `CreateCommandAllocator`
//! (slot 9), `CreateCommandList` (slot 12), `CreateCommandList1` (slot 51), and
//! `CreateRootSignature` (slot 16), part of #480: compares XCL's hooked path directly against
//! the unhooked PC driver without requiring game files or console DLLs.

use std::ffi::c_void;

use windows::core::{IUnknown, Interface, GUID};
use windows::Win32::Graphics::Direct3D12::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, real};

const SLOT_CREATE_COMMAND_QUEUE: usize = 8;
const SLOT_CREATE_COMMAND_ALLOCATOR: usize = 9;
const SLOT_CREATE_COMMAND_LIST: usize = 12;
const SLOT_CREATE_ROOT_SIGNATURE: usize = 16;
const SLOT_CREATE_COMMAND_LIST1: usize = 51;
const E_NOINTERFACE: i32 = 0x8000_4002u32 as i32;

type CreateQueueFn = unsafe extern "system" fn(*mut c_void, *const D3D12_COMMAND_QUEUE_DESC, *const GUID, *mut *mut c_void) -> i32;
type CreateAllocatorFn = unsafe extern "system" fn(*mut c_void, D3D12_COMMAND_LIST_TYPE, *const GUID, *mut *mut c_void) -> i32;
type CreateListFn = unsafe extern "system" fn(*mut c_void, u32, D3D12_COMMAND_LIST_TYPE, *mut c_void, *mut c_void, *const GUID, *mut *mut c_void) -> i32;
type CreateList1Fn = unsafe extern "system" fn(*mut c_void, u32, D3D12_COMMAND_LIST_TYPE, D3D12_COMMAND_LIST_FLAGS, *const GUID, *mut *mut c_void) -> i32;
type CreateRootSigFn = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const GUID, *mut *mut c_void) -> i32;

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
fn create_command_queue_and_allocator_match_host_driver() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let vt = *(dev as *const *const usize);

        let q_hook_addr = *vt.add(SLOT_CREATE_COMMAND_QUEUE);
        let q_host_addr = real::orig_fn(dev, SLOT_CREATE_COMMAND_QUEUE);
        assert_ne!(q_hook_addr, q_host_addr, "CreateCommandQueue slot must be hooked");
        let q_hook: CreateQueueFn = std::mem::transmute(q_hook_addr);
        let q_host: CreateQueueFn = std::mem::transmute(q_host_addr);
        let q_iid = ID3D12CommandQueue::IID;

        for ty in [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY] {
            let desc = D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() };
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = q_hook(dev, &desc, &q_iid, &mut p_hook);
            let hr_host = q_host(dev, &desc, &q_iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "CreateCommandQueue({ty:?}) HRESULT mismatch");
            assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
            let c_hook = ID3D12CommandQueue::from_raw(p_hook);
            let c_host = ID3D12CommandQueue::from_raw(p_host);
            assert_eq!(c_hook.GetDesc(), c_host.GetDesc());
        }

        // Invalid queue descriptors fail identically to host.
        for bad in [
            D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE(999), ..Default::default() },
            D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, NodeMask: 0x10, ..Default::default() },
        ] {
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = q_hook(dev, &bad, &q_iid, &mut p_hook);
            let hr_host = q_host(dev, &bad, &q_iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "HRESULT mismatch for {bad:?}");
            assert_eq!((p_hook.is_null(), p_host.is_null()), (hr_hook < 0, hr_host < 0), "output for {bad:?}");
            // The PC runtime refuses these; vkd3d-proton (under Wine) accepts some of them.
            assert!(hr_host < 0 || xcl::util::under_wine(), "expected queue failure: host={hr_host:#x}");
            for p in [p_hook, p_host].into_iter().filter(|p| !p.is_null()) {
                drop(ID3D12CommandQueue::from_raw(p));
            }
        }

        let q_console_iid = GUID::from_u128(0x11223344_5566_7788_99aa_bbccddeeff06);
        let q_desc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() };
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(q_host(dev, &q_desc, &q_console_iid, &mut p_host), E_NOINTERFACE);
        assert!(p_host.is_null());
        let hr = q_hook(dev, &q_desc, &q_console_iid, &mut p_hook);
        assert!(hr >= 0 && !p_hook.is_null(), "queue console IID fallback failed: {hr:#x}");
        let _q = ID3D12CommandQueue::from_raw(p_hook);

        let a_hook_addr = *vt.add(SLOT_CREATE_COMMAND_ALLOCATOR);
        let a_host_addr = real::orig_fn(dev, SLOT_CREATE_COMMAND_ALLOCATOR);
        assert_ne!(a_hook_addr, a_host_addr, "CreateCommandAllocator slot must be hooked");
        let a_hook: CreateAllocatorFn = std::mem::transmute(a_hook_addr);
        let a_host: CreateAllocatorFn = std::mem::transmute(a_host_addr);
        let a_iid = ID3D12CommandAllocator::IID;

        for ty in [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY] {
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = a_hook(dev, ty, &a_iid, &mut p_hook);
            let hr_host = a_host(dev, ty, &a_iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "CreateCommandAllocator({ty:?}) HRESULT mismatch");
            assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
            let ca_hook = ID3D12CommandAllocator::from_raw(p_hook);
            let ca_host = ID3D12CommandAllocator::from_raw(p_host);
            assert_eq!(ca_hook.Reset(), ca_host.Reset());
        }

        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr_hook = a_hook(dev, D3D12_COMMAND_LIST_TYPE(999), &a_iid, &mut p_hook);
        let hr_host = a_host(dev, D3D12_COMMAND_LIST_TYPE(999), &a_iid, &mut p_host);
        assert!(hr_hook < 0 && hr_host < 0);
        assert!(p_hook.is_null() && p_host.is_null());

        let a_console_iid = GUID::from_u128(0x11223344_5566_7788_99aa_bbccddeeff07);
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(a_host(dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &a_console_iid, &mut p_host), E_NOINTERFACE);
        assert!(p_host.is_null());
        let hr = a_hook(dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &a_console_iid, &mut p_hook);
        assert!(hr >= 0 && !p_hook.is_null(), "allocator console IID fallback failed: {hr:#x}");
        let _a = ID3D12CommandAllocator::from_raw(p_hook);
    }
}

#[test]
fn create_command_list_and_list1_match_host_driver() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let vt = *(dev as *const *const usize);
        let a_host: CreateAllocatorFn = std::mem::transmute(real::orig_fn(dev, SLOT_CREATE_COMMAND_ALLOCATOR));
        let make_alloc = |ty: D3D12_COMMAND_LIST_TYPE| -> ID3D12CommandAllocator {
            let mut raw = std::ptr::null_mut();
            let hr = a_host(dev, ty, &ID3D12CommandAllocator::IID, &mut raw);
            assert!(hr >= 0 && !raw.is_null());
            ID3D12CommandAllocator::from_raw(raw)
        };

        let cl_hook_addr = *vt.add(SLOT_CREATE_COMMAND_LIST);
        let cl_host_addr = real::orig_fn(dev, SLOT_CREATE_COMMAND_LIST);
        assert_ne!(cl_hook_addr, cl_host_addr, "CreateCommandList slot must be hooked");
        let cl_hook: CreateListFn = std::mem::transmute(cl_hook_addr);
        let cl_host: CreateListFn = std::mem::transmute(cl_host_addr);
        let cl_iid = ID3D12GraphicsCommandList::IID;

        for ty in [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY] {
            let alloc_hook = make_alloc(ty);
            let alloc_host = make_alloc(ty);
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = cl_hook(dev, 0, ty, alloc_hook.as_raw(), std::ptr::null_mut(), &cl_iid, &mut p_hook);
            let hr_host = cl_host(dev, 0, ty, alloc_host.as_raw(), std::ptr::null_mut(), &cl_iid, &mut p_host);
            assert_eq!(hr_hook, hr_host, "CreateCommandList({ty:?}) HRESULT mismatch");
            assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
            let l_hook = ID3D12GraphicsCommandList::from_raw(p_hook);
            let l_host = ID3D12GraphicsCommandList::from_raw(p_host);
            assert_eq!(l_hook.GetType(), l_host.GetType());
            assert_eq!(l_hook.Close(), l_host.Close());
        }

        // Mismatched allocator type fails identically on both.
        let direct_alloc = make_alloc(D3D12_COMMAND_LIST_TYPE_DIRECT);
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr_hook = cl_hook(dev, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE, direct_alloc.as_raw(), std::ptr::null_mut(), &cl_iid, &mut p_hook);
        let hr_host = cl_host(dev, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE, direct_alloc.as_raw(), std::ptr::null_mut(), &cl_iid, &mut p_host);
        assert!(hr_hook < 0 && hr_host < 0);
        assert!(p_hook.is_null() && p_host.is_null());

        let cl_console_iid = GUID::from_u128(0x11223344_5566_7788_99aa_bbccddeeff08);
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(cl_host(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, direct_alloc.as_raw(), std::ptr::null_mut(), &cl_console_iid, &mut p_host), E_NOINTERFACE);
        assert!(p_host.is_null());
        let hr = cl_hook(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, direct_alloc.as_raw(), std::ptr::null_mut(), &cl_console_iid, &mut p_hook);
        assert!(hr >= 0 && !p_hook.is_null(), "CreateCommandList console IID fallback failed: {hr:#x}");
        let _l = ID3D12GraphicsCommandList::from_raw(p_hook);

        // Verify ID3D12Device4::CreateCommandList1 (slot 51) when supported by runtime.
        let dev_unk = IUnknown::from_raw_borrowed(&dev).unwrap();
        if dev_unk.cast::<ID3D12Device4>().is_ok() {
            let cl1_hook_addr = *vt.add(SLOT_CREATE_COMMAND_LIST1);
            let cl1_host_addr = real::orig_fn(dev, SLOT_CREATE_COMMAND_LIST1);
            assert_ne!(cl1_hook_addr, cl1_host_addr, "CreateCommandList1 slot must be hooked");
            let cl1_hook: CreateList1Fn = std::mem::transmute(cl1_hook_addr);
            let cl1_host: CreateList1Fn = std::mem::transmute(cl1_host_addr);

            for ty in [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY] {
                let alloc = make_alloc(ty);
                let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
                let hr_hook = cl1_hook(dev, 0, ty, D3D12_COMMAND_LIST_FLAG_NONE, &cl_iid, &mut p_hook);
                let hr_host = cl1_host(dev, 0, ty, D3D12_COMMAND_LIST_FLAG_NONE, &cl_iid, &mut p_host);
                assert_eq!(hr_hook, hr_host, "CreateCommandList1({ty:?}) HRESULT mismatch");
                assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
                let l_hook = ID3D12GraphicsCommandList::from_raw(p_hook);
                let l_host = ID3D12GraphicsCommandList::from_raw(p_host);
                assert_eq!(l_hook.GetType(), l_host.GetType());
                // CreateCommandList1 starts closed; Reset + Close validates it cleanly.
                assert!(l_hook.Reset(&alloc, None).is_ok());
                assert!(l_hook.Close().is_ok());
            }

            let cl1_console_iid = GUID::from_u128(0x11223344_5566_7788_99aa_bbccddeeff09);
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            assert_eq!(cl1_host(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE, &cl1_console_iid, &mut p_host), E_NOINTERFACE);
            assert!(p_host.is_null());
            let hr = cl1_hook(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE, &cl1_console_iid, &mut p_hook);
            assert!(hr >= 0 && !p_hook.is_null(), "CreateCommandList1 console IID fallback failed: {hr:#x}");
            let _l1 = ID3D12GraphicsCommandList::from_raw(p_hook);
        }
    }
}

#[test]
fn create_root_signature_matches_host_driver_and_fallbacks() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let vt = *(dev as *const *const usize);
        let rs_hook_addr = *vt.add(SLOT_CREATE_ROOT_SIGNATURE);
        let rs_host_addr = real::orig_fn(dev, SLOT_CREATE_ROOT_SIGNATURE);
        assert_ne!(rs_hook_addr, rs_host_addr, "CreateRootSignature slot must be hooked");
        let rs_hook: CreateRootSigFn = std::mem::transmute(rs_hook_addr);
        let rs_host: CreateRootSigFn = std::mem::transmute(rs_host_addr);
        let iid = ID3D12RootSignature::IID;

        // 1) Empty root signature.
        let empty_desc = D3D12_ROOT_SIGNATURE_DESC {
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            ..Default::default()
        };
        let mut empty_blob = None;
        D3D12SerializeRootSignature(&empty_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut empty_blob, None).expect("serialize empty RS");
        let empty_blob = empty_blob.unwrap();

        // 2) Root signature with root constants + root CBV.
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 1, RegisterSpace: 0 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
            },
        ];
        let param_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            ..Default::default()
        };
        let mut param_blob = None;
        D3D12SerializeRootSignature(&param_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut param_blob, None).expect("serialize parameterized RS");
        let param_blob = param_blob.unwrap();

        for blob in [&empty_blob, &param_blob] {
            let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr_hook = rs_hook(dev, 0, blob.GetBufferPointer(), blob.GetBufferSize(), &iid, &mut p_hook);
            let hr_host = rs_host(dev, 0, blob.GetBufferPointer(), blob.GetBufferSize(), &iid, &mut p_host);
            assert_eq!(hr_hook, hr_host);
            assert!(hr_hook >= 0 && !p_hook.is_null() && !p_host.is_null());
            let _r_hook = ID3D12RootSignature::from_raw(p_hook);
            let _r_host = ID3D12RootSignature::from_raw(p_host);
        }

        // Unknown console IID fails on host with E_NOINTERFACE; XCL falls back to ID3D12RootSignature.
        let console_iid = GUID::from_u128(0x11223344_5566_7788_99aa_bbccddeeff0a);
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(rs_host(dev, 0, empty_blob.GetBufferPointer(), empty_blob.GetBufferSize(), &console_iid, &mut p_host), E_NOINTERFACE);
        assert!(p_host.is_null());
        let hr = rs_hook(dev, 0, empty_blob.GetBufferPointer(), empty_blob.GetBufferSize(), &console_iid, &mut p_hook);
        assert!(hr >= 0 && !p_hook.is_null(), "CreateRootSignature console IID fallback failed: {hr:#x}");
        let _r = ID3D12RootSignature::from_raw(p_hook);

        // Deliberate XCL contract: malformed bytecode fails on host but XCL substitutes an empty root signature.
        let garbage = [0xAAu8; 16];
        let (mut p_hook, mut p_host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let hr_host = rs_host(dev, 0, garbage.as_ptr() as *const c_void, garbage.len(), &iid, &mut p_host);
        let hr_hook = rs_hook(dev, 0, garbage.as_ptr() as *const c_void, garbage.len(), &iid, &mut p_hook);
        assert!(hr_host < 0 && p_host.is_null());
        assert!(hr_hook >= 0 && !p_hook.is_null(), "XCL empty RS fallback failed: {hr_hook:#x}");
        let _fallback = ID3D12RootSignature::from_raw(p_hook);
    }
}
