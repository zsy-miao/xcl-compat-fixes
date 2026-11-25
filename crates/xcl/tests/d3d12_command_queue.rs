//! Compare queues created through the console device hook with the host driver on the same device.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry;
use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, device_x::XBOX_TYPE_DMA, real};
use xcl::modules::{self, MappedModule};

const CREATE_COMMAND_QUEUE_SLOT: usize = 8;
const CONSOLE_QUEUE_FLAG: D3D12_COMMAND_QUEUE_FLAGS = D3D12_COMMAND_QUEUE_FLAGS(2);

type CreateQueue = unsafe extern "system" fn(
    *mut c_void,
    *const D3D12_COMMAND_QUEUE_DESC,
    *const GUID,
    *mut *mut c_void,
) -> HRESULT;

#[inline(never)]
unsafe fn create_from_title(
    method: CreateQueue,
    device: *mut c_void,
    desc: &D3D12_COMMAND_QUEUE_DESC,
    output: &mut *mut c_void,
) -> HRESULT {
    let result = method(device, desc, &ID3D12CommandQueue::IID, output);
    // Keep the vtable call's return address inside this function in release builds too.
    std::hint::black_box(result)
}

struct TitleCaller(usize);

impl TitleCaller {
    fn register() -> Self {
        // The slot guard classifies the caller by its return address. Use the x64 unwind range
        // of the call above so the rest of the integration test stays on the host path.
        let caller = create_from_title as *const () as u64;
        let mut image_base = 0;
        let entry = unsafe {
            RtlLookupFunctionEntry(caller, &mut image_base, null_mut())
                .as_ref()
                .expect("unwind entry for the synthetic title caller")
        };
        let base = (image_base + u64::from(entry.BeginAddress)) as usize;
        let size = (entry.EndAddress - entry.BeginAddress) as usize;
        modules::register_mapped(Arc::new(MappedModule {
            name: "d3d12-command-queue-fixture".into(),
            path: Default::default(),
            base,
            size,
            entry_rva: 0,
            is_dll: false,
            exports: Default::default(),
            tls_callbacks: Default::default(),
            no_thread_calls: Default::default(),
            initialized: Default::default(),
            stack_reserve: 0,
        }));
        Self(base)
    }
}

impl Drop for TitleCaller {
    fn drop(&mut self) {
        modules::unregister_mapped(self.0);
    }
}

#[test]
fn console_command_queue_descriptions_agree_with_the_host_driver() {
    if d3d12x::backend() != d3d12x::Backend::Real {
        runner::skip_test("no GPU");
        return;
    }
    let device = real::device_raw();
    let direct = D3D12_COMMAND_QUEUE_DESC::default();
    let compute = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..direct };
    let copy = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..direct };
    let cases = [
        ("direct", direct, direct),
        ("console compute flags", D3D12_COMMAND_QUEUE_DESC { Flags: CONSOLE_QUEUE_FLAG, ..compute }, compute),
        ("console DMA", D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE(XBOX_TYPE_DMA as i32), ..copy }, copy),
    ];

    let _title = TitleCaller::register();
    unsafe {
        let dev = ID3D12Device::from_raw_borrowed(&device).expect("backend device");
        let hooked_create = dev.vtable().CreateCommandQueue;
        let original = real::orig_fn(device, CREATE_COMMAND_QUEUE_SLOT);
        assert_ne!(hooked_create as usize, original, "XCL's device must hook queue creation");
        let native_create: CreateQueue = std::mem::transmute(original);

        for (case, console_desc, pc_desc) in cases {
            let mut expected = null_mut();
            native_create(device, &pc_desc, &ID3D12CommandQueue::IID, &mut expected)
                .ok()
                .unwrap_or_else(|err| panic!("{case} failed on the host driver ({err})"));
            assert!(!expected.is_null(), "no host queue for {case}");
            let expected = ID3D12CommandQueue::from_raw(expected);

            let mut actual = null_mut();
            create_from_title(hooked_create, device, &console_desc, &mut actual)
                .ok()
                .unwrap_or_else(|err| panic!("{case} failed through XCL ({err})"));
            assert!(!actual.is_null(), "no XCL queue for {case}");
            let actual = ID3D12CommandQueue::from_raw(actual);
            assert_eq!(actual.GetDesc(), expected.GetDesc(), "queue description for {case}");
        }
    }
}
