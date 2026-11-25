//! Robustness and concurrency checks for `ID3D12Device2::CreatePipelineState`
//! (`d3d12x::real::dev_create_pipeline_state` / `translated_mesh_stream`), with no game data.
//!
//! Runs on any D3D12 adapter (AMD, NVIDIA, Intel, or WARP) with zero vendor-specific skips:
//! 1. Synthetic console MS placed in the AS slot with no MS slot is refused cleanly with
//!    `E_INVALIDARG` and records the `"in the AS slot – refused"` error.
//! 2. Synthetic console MS paired with an AS subobject is refused cleanly with `E_INVALIDARG`.
//! 3. Empty, truncated, unknown-subobject, and unreadable-shader pipeline streams fail cleanly
//!    without panicking or crashing.
//! 4. Five worker threads submitting malformed/refused console mesh streams concurrently complete
//!    without panicking or corrupting state.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::core::Interface;
use windows::Win32::Foundation::E_INVALIDARG;
use windows::Win32::Graphics::Direct3D12::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::real;

fn device2() -> Option<ID3D12Device2> {
    static DEV: OnceLock<Option<ID3D12Device2>> = OnceLock::new();
    DEV.get_or_init(|| unsafe {
        if real::init().is_err() {
            runner::skip_test("d3d12x::real::init failed");
            return None;
        }
        let raw = real::device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&raw)?.clone();
        match dev.cast::<ID3D12Device2>() {
            Ok(d2) => Some(d2),
            Err(_) => {
                runner::skip_test("no ID3D12Device2");
                None
            }
        }
    })
    .clone()
}

/// Minimal synthetic console shader DXBC container recognized by `pso::is_console` and
/// `xbox::parse` (`kind = 13` for `"MS"`, `14` for `"AS"`), with no game data.
fn synthetic_console_shader(kind: u32) -> Vec<u8> {
    let dxil_size = 32u32;
    let total_size = 36u32 + 8 + dxil_size;
    let mut b = Vec::with_capacity(total_size as usize);
    b.extend_from_slice(b"DXBC");
    b.extend_from_slice(&[0u8; 16]); // digest
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&total_size.to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes()); // part count
    b.extend_from_slice(&36u32.to_le_bytes()); // part 0 offset
    b.extend_from_slice(b"DXIL");
    b.extend_from_slice(&dxil_size.to_le_bytes());
    let pv = (kind << 16) | 0x65; // SM 6.5
    b.extend_from_slice(&pv.to_le_bytes());
    b.extend_from_slice(&[0u8; 20]); // rest of 24-byte program header
    b.extend_from_slice(b"XDAT");
    b.extend_from_slice(&0u32.to_le_bytes());
    assert!(xcl::gpu::recomp::pso::is_console(&b));
    b
}

#[repr(C, align(8))]
struct Sub<T> {
    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    v: T,
}

fn bytecode(b: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as *const c_void, BytecodeLength: b.len() }
}

#[test]
fn console_ms_in_as_slot_and_ms_with_as_are_refused_cleanly() {
    let Some(d2) = device2() else { return };
    let ms = synthetic_console_shader(13);
    let as_blob = synthetic_console_shader(14);
    assert_eq!(xcl::gpu::xbox::parse(&ms).unwrap().stage, "MS");
    assert_eq!(xcl::gpu::xbox::parse(&as_blob).unwrap().stage, "AS");

    // Case 1: console MS placed in the AS slot with no MS slot (#1968 context path)
    #[repr(C)]
    struct AsOnly {
        as_slot: Sub<D3D12_SHADER_BYTECODE>,
    }
    let st = AsOnly { as_slot: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, v: bytecode(&ms) } };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: std::mem::size_of::<AsOnly>(),
        pPipelineStateSubobjectStream: &st as *const _ as *mut c_void,
    };
    let err = unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&desc) }.expect_err("MS in AS slot must be refused");
    assert_eq!(err.code(), E_INVALIDARG);
    assert!(
        real::stats().translation_errors.iter().any(|e| e.contains("in the AS slot")),
        "expected AS-slot refusal in translation_errors: {:?}",
        real::stats().translation_errors
    );

    // Case 2: console MS paired with an AS subobject
    #[repr(C)]
    struct AsAndMs {
        as_slot: Sub<D3D12_SHADER_BYTECODE>,
        ms_slot: Sub<D3D12_SHADER_BYTECODE>,
    }
    let st2 = AsAndMs {
        as_slot: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, v: bytecode(&as_blob) },
        ms_slot: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, v: bytecode(&ms) },
    };
    let desc2 = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: std::mem::size_of::<AsAndMs>(),
        pPipelineStateSubobjectStream: &st2 as *const _ as *mut c_void,
    };
    let err2 = unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&desc2) }.expect_err("MS + AS must be refused");
    assert_eq!(err2.code(), E_INVALIDARG);
}

#[test]
fn malformed_and_truncated_pipeline_streams_do_not_panic_or_crash() {
    let Some(d2) = device2() else { return };

    // Empty / null stream pointer
    let empty = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: 0, pPipelineStateSubobjectStream: std::ptr::null_mut() };
    assert!(unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&empty) }.is_err());

    // Truncated MS subobject (header says MS=25, only 16 bytes available instead of 24)
    let mut trunc = [0u8; 16];
    trunc[0..4].copy_from_slice(&(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32).to_le_bytes());
    let desc_trunc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: trunc.len(),
        pPipelineStateSubobjectStream: trunc.as_mut_ptr() as *mut c_void,
    };
    assert!(unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&desc_trunc) }.is_err());

    // Unknown subobject type
    let mut unknown = [0u8; 16];
    unknown[0..4].copy_from_slice(&0xBEEFu32.to_le_bytes());
    let desc_unknown = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: unknown.len(),
        pPipelineStateSubobjectStream: unknown.as_mut_ptr() as *mut c_void,
    };
    assert!(unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&desc_unknown) }.is_err());

    // Unreadable shader pointer in MS subobject
    let mut unreadable = [0u8; 24];
    unreadable[0..4].copy_from_slice(&(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32).to_le_bytes());
    unreadable[8..16].copy_from_slice(&0x1000u64.to_le_bytes());
    unreadable[16..24].copy_from_slice(&64u64.to_le_bytes());
    let desc_unreadable = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: unreadable.len(),
        pPipelineStateSubobjectStream: unreadable.as_mut_ptr() as *mut c_void,
    };
    assert!(unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&desc_unreadable) }.is_err());
}

#[test]
fn concurrent_mesh_stream_refusals_across_five_threads_survive() {
    let Some(d2) = device2() else { return };
    let ms = std::sync::Arc::new(synthetic_console_shader(13));
    let mut handles = Vec::new();
    for _ in 0..5 {
        let d2 = d2.clone();
        let ms = ms.clone();
        handles.push(std::thread::spawn(move || {
            #[repr(C)]
            struct AsOnly {
                as_slot: Sub<D3D12_SHADER_BYTECODE>,
            }
            for i in 0..20 {
                if i % 2 == 0 {
                    let st = AsOnly { as_slot: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, v: bytecode(&ms) } };
                    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
                        SizeInBytes: std::mem::size_of::<AsOnly>(),
                        pPipelineStateSubobjectStream: &st as *const _ as *mut c_void,
                    };
                    let err = unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&desc) }.expect_err("MS in AS slot");
                    assert_eq!(err.code(), E_INVALIDARG);
                } else {
                    let mut trunc = [0u8; 16];
                    trunc[0..4].copy_from_slice(&(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32).to_le_bytes());
                    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
                        SizeInBytes: trunc.len(),
                        pPipelineStateSubobjectStream: trunc.as_mut_ptr() as *mut c_void,
                    };
                    assert!(unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&desc) }.is_err());
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }
}
