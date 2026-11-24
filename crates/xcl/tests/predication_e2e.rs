//! End-to-end GPU + D3D12 debug-layer test for `SetPredicationBufferX`.
//!
//! Exercises:
//! 1. Predicated copies (`EQUAL_ZERO` skipping/executing, `NOT_EQUAL_ZERO`, `NULL`), interleaved
//!    UMA window writes (`Write32BitValueBottomOfPipeX`) on the same UMA window without state
//!    conflicts, GPU-written predicate updates in the same recording, and direct switching from
//!    one predicate buffer to another without `NULL`.
//! 2. `KickoffX` closing, submitting, and resetting mid-recording while predication is active,
//!    preserving predication afterward, followed by closing the list while still active.
//! 3. `ClearState` unbinding predication while preserving the dedicated predication buffer's
//!    resource state, then re-arming `SetPredicationBufferX` on the same recording.

use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{list_x, real};

#[repr(C)]
struct PredDesc {
    va: u64,
    op: u32,
}

static DEBUG_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "system" fn collect_debug_message(
    _category: D3D12_MESSAGE_CATEGORY,
    severity: D3D12_MESSAGE_SEVERITY,
    id: D3D12_MESSAGE_ID,
    text: windows::core::PCSTR,
    _context: *mut c_void,
) {
    if severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
        let msg = if text.is_null() {
            String::new()
        } else {
            String::from_utf8_lossy(text.as_bytes()).into_owned()
        };
        DEBUG_ERRORS
            .lock()
            .unwrap()
            .push(format!("severity={} id={} {msg}", severity.0, id.0));
    }
}

fn buffer(
    dev: &ID3D12Device,
    bytes: u64,
    heap: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let props = D3D12_HEAP_PROPERTIES {
        Type: heap,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    unsafe {
        dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res)
            .expect("CreateCommittedResource");
    }
    res.unwrap()
}

unsafe fn wait_fence(queue: &ID3D12CommandQueue, fence: &ID3D12Fence, value: &mut u64) {
    *value += 1;
    let target = *value;
    queue.Signal(fence, target).expect("Signal");
    if fence.GetCompletedValue() < target {
        let ev = CreateEventW(None, false, false, None).expect("CreateEventW");
        let res = fence.SetEventOnCompletion(target, ev).map(|()| {
            let s = WaitForSingleObject(ev, 10_000);
            if s != WAIT_OBJECT_0 {
                panic!("WaitForSingleObject returned {s:?}");
            }
        });
        let _ = CloseHandle(ev);
        res.expect("SetEventOnCompletion");
    }
}

struct ReadbackCtx<'a> {
    queue: &'a ID3D12CommandQueue,
    alloc: &'a ID3D12CommandAllocator,
    list: &'a ID3D12GraphicsCommandList,
    fence: &'a ID3D12Fence,
}

unsafe fn read_u32s(
    ctx: &ReadbackCtx<'_>,
    fence_val: &mut u64,
    dst_default: &ID3D12Resource,
    readback: &ID3D12Resource,
    count: usize,
) -> Vec<u32> {
    let bytes = (count * 4) as u64;
    ctx.list.Reset(ctx.alloc, None).expect("readback Reset");
    ctx.list.CopyBufferRegion(readback, 0, dst_default, 0, bytes);
    ctx.list.Close().expect("readback Close");
    let cl: ID3D12CommandList = ctx.list.cast().unwrap();
    ctx.queue.ExecuteCommandLists(&[Some(cl)]);
    wait_fence(ctx.queue, ctx.fence, fence_val);

    let mut p = std::ptr::null_mut();
    readback.Map(0, None, Some(&mut p)).expect("Map readback");
    let out = std::slice::from_raw_parts(p as *const u32, count).to_vec();
    readback.Unmap(0, None);
    out
}

fn take_debug_errors(iq: Option<&ID3D12InfoQueue>) -> Vec<String> {
    let mut errs = std::mem::take(&mut *DEBUG_ERRORS.lock().unwrap());
    if let Some(iq) = iq {
        unsafe {
            let n = iq.GetNumStoredMessages();
            for i in 0..n {
                let mut len = 0usize;
                if iq.GetMessage(i, None, &mut len).is_ok() && len > 0 {
                    let mut buf = vec![0u8; len];
                    let msg_ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                    if iq.GetMessage(i, Some(msg_ptr), &mut len).is_ok() {
                        let msg = &*msg_ptr;
                        if msg.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                            let desc = if msg.pDescription.is_null() {
                                String::new()
                            } else {
                                std::ffi::CStr::from_ptr(msg.pDescription as *const i8)
                                    .to_string_lossy()
                                    .into_owned()
                            };
                            errs.push(format!("stored id={} {desc}", msg.ID.0));
                        }
                    }
                }
            }
            iq.ClearStoredMessages();
        }
    }
    errs
}

#[test]
fn set_predication_buffer_state_and_kickoff() {
    std::env::set_var("XCL_UMA_MB", "64");
    std::env::set_var("XCL_D3D12_DEBUG", "1");

    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        runner::skip_test("no GPU");
        return;
    }
    let Some(uma) = xcl::gpu::uma::get() else {
        println!("no UMA – skipping");
        runner::skip_test("no UMA");
        return;
    };
    if real::uma_gpu_va() == 0 {
        println!("UMA not opened on GPU – skipping");
        runner::skip_test("no GPU UMA");
        return;
    }

    let raw_dev = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw_dev) }
        .expect("backend device")
        .clone();

    let iq = dev.cast::<ID3D12InfoQueue>().ok();
    let iq1 = dev.cast::<ID3D12InfoQueue1>().ok();
    let mut cookie = 0u32;
    if let Some(ref q1) = iq1 {
        unsafe {
            let _ = q1.RegisterMessageCallback(
                Some(collect_debug_message),
                D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                std::ptr::null_mut(),
                &mut cookie,
            );
        }
    }
    let _ = take_debug_errors(iq.as_ref());

    unsafe {
        // Allocate UMA slots in committed coherent UMA (zero-initialized, 8-byte aligned).
        let uma_ptr = uma.alloc(0x1_0000, 0x1_0000).expect("UMA block");
        let slot_zero_va = uma_ptr as u64;
        let slot_one_va = uma_ptr.add(0x100) as u64;
        let slot_write_va = uma_ptr.add(0x200) as u64;
        let slot_gpu_va = uma_ptr.add(0x300) as u64;
        (slot_zero_va as *mut u64).write_volatile(0);
        (slot_one_va as *mut u64).write_volatile(1);
        (slot_write_va as *mut u64).write_volatile(0);
        (slot_gpu_va as *mut u64).write_volatile(0);

        let desc_zero_eq = PredDesc {
            va: slot_zero_va,
            op: D3D12_PREDICATION_OP_EQUAL_ZERO.0 as u32,
        };
        let desc_one_eq = PredDesc {
            va: slot_one_va,
            op: D3D12_PREDICATION_OP_EQUAL_ZERO.0 as u32,
        };
        let desc_one_ne = PredDesc {
            va: slot_one_va,
            op: D3D12_PREDICATION_OP_NOT_EQUAL_ZERO.0 as u32,
        };
        let desc_gpu_eq = PredDesc {
            va: slot_gpu_va,
            op: D3D12_PREDICATION_OP_EQUAL_ZERO.0 as u32,
        };

        const COUNT: usize = 9;
        let bytes = (COUNT * 4) as u64;
        let init_vals: [u32; COUNT] = [
            0x1111_1111,
            0x2222_2222,
            0x3333_3333,
            0x4444_4444,
            0x5555_5555,
            0x6666_6666,
            0x7777_7777,
            0x8888_8888,
            0x9999_9999,
        ];
        let src_vals: [u32; COUNT] = [
            0xAAAA_0000,
            0xBBBB_0001,
            0xCCCC_0002,
            0xDDDD_0003,
            0xEEEE_0004,
            0xFFFF_0005,
            0x9999_0006,
            0x1234_0007,
            0x5678_0008,
        ];

        let init_buf = buffer(
            &dev,
            bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let src_buf = buffer(
            &dev,
            bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let dst_buf = buffer(
            &dev,
            bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let readback = buffer(
            &dev,
            bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let mut p = std::ptr::null_mut();
        init_buf.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(init_vals.as_ptr(), p as *mut u32, COUNT);
        init_buf.Unmap(0, None);

        src_buf.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(src_vals.as_ptr(), p as *mut u32, COUNT);
        src_buf.Unmap(0, None);

        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .expect("CreateCommandQueue");
        list_x::on_queue_create(
            queue.as_raw() as usize,
            dev.as_raw() as usize,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
        );

        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .expect("CreateCommandAllocator");
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .expect("CreateCommandList");
        let list_ptr = list.as_raw();
        list_x::on_create(
            list_ptr,
            alloc.as_raw() as usize,
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
        );

        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let mut fence_val = 0u64;
        let ctx = ReadbackCtx {
            queue: &queue,
            alloc: &alloc,
            list: &list,
            fence: &fence,
        };

        // Seed dst_buf with init_vals.
        list.CopyBufferRegion(&dst_buf, 0, &init_buf, 0, bytes);
        list.Close().unwrap();
        let cl: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cl)]);
        wait_fence(&queue, &fence, &mut fence_val);
        let _ = take_debug_errors(iq.as_ref());

        // Case 1:
        // - dst[0]: EQUAL_ZERO on value 0 -> must SKIP copy
        // - interleaved Write32BitValueBottomOfPipeX on the same UMA window while predication is active
        // - switch directly to EQUAL_ZERO on value 1 WITHOUT calling NULL first -> dst[1] must EXECUTE
        // - NOT_EQUAL_ZERO on value 1 -> dst[2] must SKIP
        // - SetPredicationBufferX(NULL) -> dst[3] must EXECUTE
        list.Reset(&alloc, None).unwrap();
        list_x::set_predication_buffer_x(list_ptr, &desc_zero_eq as *const PredDesc as *const u64);
        list.CopyBufferRegion(&dst_buf, 0, &src_buf, 0, 4);
        list_x::write32_bottom_x(list_ptr, slot_write_va, 0xABCD_1234, 0, 0);
        list_x::set_predication_buffer_x(list_ptr, &desc_one_eq as *const PredDesc as *const u64);
        list.CopyBufferRegion(&dst_buf, 4, &src_buf, 4, 4);
        list_x::set_predication_buffer_x(list_ptr, &desc_one_ne as *const PredDesc as *const u64);
        list.CopyBufferRegion(&dst_buf, 8, &src_buf, 8, 4);
        list_x::set_predication_buffer_x(list_ptr, std::ptr::null());
        list.CopyBufferRegion(&dst_buf, 12, &src_buf, 12, 4);
        list.Close().unwrap();

        let cl: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cl)]);
        wait_fence(&queue, &fence, &mut fence_val);

        let case1_errs = take_debug_errors(iq.as_ref());
        let got1 = read_u32s(&ctx, &mut fence_val, &dst_buf, &readback, COUNT);
        let written_uma = (slot_write_va as *const u32).read_volatile();
        assert!(case1_errs.is_empty(), "Case 1 D3D12 debug errors: {case1_errs:#?}");
        assert_eq!(got1[0], init_vals[0], "EQUAL_ZERO(0) must skip dst[0]");
        assert_eq!(got1[1], src_vals[1], "direct switch to EQUAL_ZERO(1) must execute dst[1]");
        assert_eq!(got1[2], init_vals[2], "NOT_EQUAL_ZERO(1) must skip dst[2]");
        assert_eq!(got1[3], src_vals[3], "SetPredicationBufferX(NULL) must execute dst[3]");
        assert_eq!(
            written_uma, 0xABCD_1234,
            "Write32BitValueBottomOfPipeX on UMA must still write 0xABCD_1234"
        );

        // Case 1b: GPU-written predicate values (`Write32BitValueBottomOfPipeX` -> `SetPredicationBufferX`)
        // within a single command list recording, verifying barrier ordering on the UMA window.
        alloc.Reset().unwrap();
        list.Reset(&alloc, None).unwrap();
        list_x::write32_bottom_x(list_ptr, slot_gpu_va, 0, 0, 0);
        list_x::set_predication_buffer_x(list_ptr, &desc_gpu_eq as *const PredDesc as *const u64);
        list.CopyBufferRegion(&dst_buf, 28, &src_buf, 28, 4);
        list_x::write32_bottom_x(list_ptr, slot_gpu_va, 1, 0, 0);
        list_x::set_predication_buffer_x(list_ptr, &desc_gpu_eq as *const PredDesc as *const u64);
        list.CopyBufferRegion(&dst_buf, 32, &src_buf, 32, 4);
        list_x::set_predication_buffer_x(list_ptr, std::ptr::null());
        list.Close().unwrap();

        let cl: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cl)]);
        wait_fence(&queue, &fence, &mut fence_val);

        let case1b_errs = take_debug_errors(iq.as_ref());
        let got1b = read_u32s(&ctx, &mut fence_val, &dst_buf, &readback, COUNT);
        assert!(case1b_errs.is_empty(), "Case 1b D3D12 debug errors: {case1b_errs:#?}");
        assert_eq!(
            got1b[7], init_vals[7],
            "GPU-written EQUAL_ZERO(0) must skip dst[7]"
        );
        assert_eq!(
            got1b[8], src_vals[8],
            "GPU-written EQUAL_ZERO(1) in the same recording must execute dst[8]"
        );

        // Case 2: KickoffX mid-recording while EQUAL_ZERO(0) predication is active,
        // then predicated CopyBufferRegion on dst[4], then Close while STILL active!
        list.Reset(&alloc, None).unwrap();
        list_x::set_predication_buffer_x(list_ptr, &desc_zero_eq as *const PredDesc as *const u64);
        list_x::kickoff_x(list_ptr);
        list.CopyBufferRegion(&dst_buf, 16, &src_buf, 16, 4);
        list.Close().unwrap();

        let cl: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cl)]);
        wait_fence(&queue, &fence, &mut fence_val);

        let case2_errs = take_debug_errors(iq.as_ref());
        let got2 = read_u32s(&ctx, &mut fence_val, &dst_buf, &readback, COUNT);
        assert!(case2_errs.is_empty(), "Case 2 D3D12 debug errors: {case2_errs:#?}");
        assert_eq!(
            got2[4], init_vals[4],
            "KickoffX must preserve active EQUAL_ZERO(0) predication so dst[4] stays skipped"
        );

        // Case 3: ClearState unbinds predication on PC list without changing pred_buf state:
        // - EQUAL_ZERO(0) -> ClearState(None) -> dst[5] executes unpredicated
        // - Re-arm SetPredicationBufferX(EQUAL_ZERO(0)) in the SAME recording -> dst[6] skipped
        // - Close while active
        alloc.Reset().unwrap();
        list.Reset(&alloc, None).unwrap();
        list_x::set_predication_buffer_x(list_ptr, &desc_zero_eq as *const PredDesc as *const u64);
        list.ClearState(None);
        list.CopyBufferRegion(&dst_buf, 20, &src_buf, 20, 4);
        list_x::set_predication_buffer_x(list_ptr, &desc_zero_eq as *const PredDesc as *const u64);
        list.CopyBufferRegion(&dst_buf, 24, &src_buf, 24, 4);
        list.Close().unwrap();

        let cl: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cl)]);
        wait_fence(&queue, &fence, &mut fence_val);

        let case3_errs = take_debug_errors(iq.as_ref());
        let got3 = read_u32s(&ctx, &mut fence_val, &dst_buf, &readback, COUNT);
        assert!(case3_errs.is_empty(), "Case 3 D3D12 debug errors: {case3_errs:#?}");
        assert_eq!(got3[5], src_vals[5], "ClearState must unbind predication so dst[5] executes");
        assert_eq!(
            got3[6], init_vals[6],
            "re-arming SetPredicationBufferX after ClearState must skip dst[6]"
        );

        // Also test console Reset (`list_x::reset_x`) clears predication while reusing `pred_buf`.
        alloc.Reset().unwrap();
        let hr = list_x::reset_x(list_ptr, alloc.as_raw() as usize, 0);
        assert!(hr >= 0, "reset_x failed: {hr:#x}");
        list.CopyBufferRegion(&dst_buf, 24, &src_buf, 24, 4);
        list.Close().unwrap();
        let cl: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cl)]);
        wait_fence(&queue, &fence, &mut fence_val);
        let got4 = read_u32s(&ctx, &mut fence_val, &dst_buf, &readback, COUNT);
        let reset_errs = take_debug_errors(iq.as_ref());
        assert!(reset_errs.is_empty(), "Reset reuse D3D12 debug errors: {reset_errs:#?}");
        assert_eq!(got4[6], src_vals[6], "reset_x must clear prior predication so dst[6] executes");

        drop(list);
        drop(queue);
        let _ = uma.free(uma_ptr as usize);
    }

    if let Some(q1) = iq1 {
        unsafe {
            let _ = q1.UnregisterMessageCallback(cookie);
        }
    }
}
