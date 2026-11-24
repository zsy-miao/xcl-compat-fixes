//! d3d12_xs.dll – D3D12.X.
//!
//! Backends:
//! * `real` (default): genuine PC D3D12 objects from the GPU driver with extended
//!   vtables (PC methods go straight to the driver, console-only slots are guarded and
//!   logged), shader capture, fallback pipeline states, presentation window.
//! * `null`: stand-in objects that render nothing (`graphics = "null"` in xcl.toml or
//!   when no D3D12 device can be created).

#![allow(non_snake_case)]

mod alloc_pool;
pub mod bind;
pub mod console_heap;
pub mod console_list;
pub mod device_x;
pub mod dxgi_device;
mod cpu_writes;
mod fence_x;
pub mod frame;
pub mod guard;
mod late_release;
pub mod layout;
pub mod list_x;
pub mod null;
pub mod present;
pub mod pso_async;
pub mod real;
mod shader_cache_session;
pub mod sparse;
pub mod title_list;
pub mod title_textures;
pub mod vram;

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::util::{HRESULT, S_OK};

/// data export: titles pass `&IID_ID3D12Device` to D3D12XboxCreateDevice / QueryInterface
static IID_ID3D12_DEVICE: windows::core::GUID = windows::core::GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);

pub fn exports() -> Vec<(&'static str, usize)> {
    let mut v = exports_fns();
    v.push(("IID_ID3D12Device", &IID_ID3D12_DEVICE as *const _ as usize));
    v
}

fn exports_fns() -> Vec<(&'static str, usize)> {
    crate::hle_exports![
        D3D12CreateDevice,
        D3D12XboxCreateDevice,
        D3D12GetDebugInterface,
        D3D12SerializeRootSignature,
        D3D12SerializeVersionedRootSignature,
        D3D12CreateRootSignatureDeserializer,
        D3D12CreateVersionedRootSignatureDeserializer,
        D3D12EnableExperimentalFeatures,
        D3D12GetInterface,
        D3D10CreateBlob,
        D3D12XboxSetProcessDebugFlags,
        D3D12ConfigureVirtualMemoryX,
    ]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Real,
    Null,
}

/// Decided once: the real backend unless disabled or the driver refuses.
pub fn backend() -> Backend {
    static B: OnceLock<Backend> = OnceLock::new();
    *B.get_or_init(|| {
        let cfg = crate::state::config();
        if cfg.graphics.eq_ignore_ascii_case("null") {
            crate::xlog_info!("D3D12.X: backend 'null' (from the config)");
            return Backend::Null;
        }
        match real::init() {
            Ok(desc) => {
                crate::xlog_info!("D3D12.X: backend 'real' – {desc}");
                Backend::Real
            }
            Err(e) => {
                crate::xlog_warn!("D3D12.X: could not create a D3D12 device ({e}) – switching to the fallback device");
                Backend::Null
            }
        }
    })
}

/// Call a function of the system d3d12.dll by name.
unsafe fn sys(name: &str) -> Option<usize> {
    crate::resolver::system_proc("d3d12.dll", &crate::pe::ImportName::Name { hint: 0, name: name.into() })
}

macro_rules! forward {
    ($name:literal, $ty:ty, $null:expr, ($($a:ident),*)) => {{
        if backend() == Backend::Real {
            if let Some(f) = sys($name) {
                let f: $ty = std::mem::transmute(f);
                return f($($a),*);
            }
        }
        $null($($a),*)
    }};
}

pub unsafe extern "system" fn D3D12CreateDevice(adapter: *mut c_void, fl: u32, riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if backend() == Backend::Real {
        return real::create_device(riid, ppv);
    }
    null::D3D12CreateDevice(adapter, fl, riid, ppv)
}

pub unsafe extern "system" fn D3D12XboxCreateDevice(adapter: *mut c_void, params: *const c_void, riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    // D3D12XBOX_CREATE_DEVICE_PARAMETERS: the first word is the SDK version the title was
    // built with (the console's vtable layouts follow it)
    if !params.is_null() && crate::gpu::shader::readable(params as usize, 32) {
        let w: Vec<u32> = (0..8).map(|k| (params as *const u32).add(k).read_unaligned()).collect();
        crate::xlog_info!("D3D12XboxCreateDevice: parameters {w:x?} (SDK version {})", w[0]);
    }
    if backend() == Backend::Real {
        return real::create_device(riid, ppv);
    }
    null::D3D12XboxCreateDevice(adapter, params, riid, ppv)
}

pub unsafe extern "system" fn D3D12GetDebugInterface(riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    forward!("D3D12GetDebugInterface", unsafe extern "system" fn(*const c_void, *mut *mut c_void) -> HRESULT, null::D3D12GetDebugInterface, (riid, ppv))
}

pub unsafe extern "system" fn D3D12SerializeRootSignature(d: *const c_void, v: u32, blob: *mut *mut c_void, err: *mut *mut c_void) -> HRESULT {
    forward!(
        "D3D12SerializeRootSignature",
        unsafe extern "system" fn(*const c_void, u32, *mut *mut c_void, *mut *mut c_void) -> HRESULT,
        null::D3D12SerializeRootSignature,
        (d, v, blob, err)
    )
}

pub unsafe extern "system" fn D3D12SerializeVersionedRootSignature(d: *const c_void, blob: *mut *mut c_void, err: *mut *mut c_void) -> HRESULT {
    let mut own_err: *mut c_void = std::ptr::null_mut();
    let e = if err.is_null() { &mut own_err as *mut _ } else { err };
    // console-only root signature flags (seen: 0x3000000 from XboxBvhBuilder.dll) – Windows
    // rejects unknown bits; the flags sit at +40 in every version of the desc (48 bytes)
    const PC_FLAGS: u32 = 0xFFF;
    let mut copy = [0u8; 48];
    let mut d = d;
    if !d.is_null() && *(d as *const u8).add(40).cast::<u32>() & !PC_FLAGS != 0 {
        std::ptr::copy_nonoverlapping(d as *const u8, copy.as_mut_ptr(), 48);
        let f = copy.as_mut_ptr().add(40).cast::<u32>();
        crate::xlog_info!("D3D12SerializeVersionedRootSignature: console flags {:#x} dropped", *f & !PC_FLAGS);
        *f &= PC_FLAGS;
        d = copy.as_ptr() as *const c_void;
    }
    type Serialize = unsafe extern "system" fn(*const c_void, *mut *mut c_void, *mut *mut c_void) -> HRESULT;
    let hr = match sys("D3D12SerializeVersionedRootSignature").filter(|_| backend() == Backend::Real) {
        Some(f) => std::mem::transmute::<_, Serialize>(f)(d, blob, e),
        None => null::D3D12SerializeVersionedRootSignature(d, blob, e),
    };
    if hr < 0 && !d.is_null() {
        // D3D12_VERSIONED_ROOT_SIGNATURE_DESC: version, then (1.1 / 1.2) parameters, samplers, flags
        let w = |o: usize| *(d as *const u8).add(o).cast::<u32>();
        let text = blob_text(*e);
        crate::xlog_warn!(
            "D3D12SerializeVersionedRootSignature: {hr:#x} (version {}, parameters {}, samplers {}, flags {:#x}): {text}",
            w(0),
            w(8),
            w(24),
            w(40)
        );
    }
    if err.is_null() && !own_err.is_null() {
        blob_release(own_err);
    }
    hr
}

/// text of an ID3DBlob (error messages)
unsafe fn blob_text(b: *mut c_void) -> String {
    if b.is_null() {
        return String::new();
    }
    let vt = *(b as *const *const usize);
    let ptr: unsafe extern "system" fn(*mut c_void) -> *const u8 = std::mem::transmute(*vt.add(3));
    let len: unsafe extern "system" fn(*mut c_void) -> usize = std::mem::transmute(*vt.add(4));
    String::from_utf8_lossy(std::slice::from_raw_parts(ptr(b), len(b))).trim_end_matches('\0').trim().to_string()
}

unsafe fn blob_release(b: *mut c_void) {
    let vt = *(b as *const *const usize);
    let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
    release(b);
}

pub unsafe extern "system" fn D3D12CreateRootSignatureDeserializer(d: *const c_void, l: usize, r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    forward!(
        "D3D12CreateRootSignatureDeserializer",
        unsafe extern "system" fn(*const c_void, usize, *const c_void, *mut *mut c_void) -> HRESULT,
        null::D3D12CreateRootSignatureDeserializer,
        (d, l, r, ppv)
    )
}

pub unsafe extern "system" fn D3D12CreateVersionedRootSignatureDeserializer(d: *const c_void, l: usize, r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    forward!(
        "D3D12CreateVersionedRootSignatureDeserializer",
        unsafe extern "system" fn(*const c_void, usize, *const c_void, *mut *mut c_void) -> HRESULT,
        null::D3D12CreateVersionedRootSignatureDeserializer,
        (d, l, r, ppv)
    )
}

pub unsafe extern "system" fn D3D12EnableExperimentalFeatures(n: u32, i: *const c_void, c: *mut c_void, s: *mut u32) -> HRESULT {
    forward!(
        "D3D12EnableExperimentalFeatures",
        unsafe extern "system" fn(u32, *const c_void, *mut c_void, *mut u32) -> HRESULT,
        null::D3D12EnableExperimentalFeatures,
        (n, i, c, s)
    )
}

pub unsafe extern "system" fn D3D12GetInterface(clsid: *const c_void, riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    forward!(
        "D3D12GetInterface",
        unsafe extern "system" fn(*const c_void, *const c_void, *mut *mut c_void) -> HRESULT,
        null::D3D12GetInterface,
        (clsid, riid, ppv)
    )
}

pub unsafe extern "system" fn D3D10CreateBlob(size: usize, out: *mut *mut c_void) -> HRESULT {
    null::D3D10CreateBlob(size, out)
}

pub unsafe extern "system" fn D3D12XboxSetProcessDebugFlags(_flags: u32) -> HRESULT {
    S_OK
}

/// How the console splits the title's GPU virtual address space (a 32-byte configuration;
/// the retail game's second dword is a count of 16 GiB units + 2). The PC's GPU addresses come from its
/// own device, so there is nothing to configure – but a title asserts (`int3`) on a failure.
pub unsafe extern "system" fn D3D12ConfigureVirtualMemoryX(config: *const c_void) -> HRESULT {
    if !config.is_null() && crate::gpu::shader::readable(config as usize, 32) {
        let w: Vec<u32> = (0..8).map(|k| (config as *const u32).add(k).read_unaligned()).collect();
        crate::xlog_info!("D3D12ConfigureVirtualMemoryX: {w:x?} – accepted (nothing to configure on the PC)");
    }
    S_OK
}
