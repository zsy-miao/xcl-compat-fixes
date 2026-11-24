//! D3D12.X "null device" – fallback backend (design §5.5a).
//!
//! Objects accept every call, log it, and hand back plausible outputs (child objects,
//! descriptor handles, completed fences, mappable memory) so a title can get past
//! graphics initialisation and keep loading data. Nothing is rendered. Used when the
//! real backend is disabled (`graphics = "null"`) or no D3D12 device can be created.
//!
//! Vtable layout guess: public PC D3D12 order (DirectX-Headers). Every slot we do not
//! implement is a logging stub named "<Interface>::#<slot>", so reports from real titles
//! show exactly where the console layout differs.

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::util::{E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, HRESULT, S_FALSE, S_OK};

const VTBL_SLOTS: usize = 256;
const DLL: &str = "d3d12_xs.dll";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Device,
    CommandQueue,
    CommandAllocator,
    CommandList,
    Fence,
    DescriptorHeap,
    Resource,
    Heap,
    PipelineState,
    RootSignature,
    QueryHeap,
    CommandSignature,
    PipelineLibrary,
    StateObject,
    Blob,
}

impl Kind {
    fn iface(self) -> &'static str {
        match self {
            Kind::Device => "ID3D12Device",
            Kind::CommandQueue => "ID3D12CommandQueue",
            Kind::CommandAllocator => "ID3D12CommandAllocator",
            Kind::CommandList => "ID3D12GraphicsCommandList",
            Kind::Fence => "ID3D12Fence",
            Kind::DescriptorHeap => "ID3D12DescriptorHeap",
            Kind::Resource => "ID3D12Resource",
            Kind::Heap => "ID3D12Heap",
            Kind::PipelineState => "ID3D12PipelineState",
            Kind::RootSignature => "ID3D12RootSignature",
            Kind::QueryHeap => "ID3D12QueryHeap",
            Kind::CommandSignature => "ID3D12CommandSignature",
            Kind::PipelineLibrary => "ID3D12PipelineLibrary",
            Kind::StateObject => "ID3D12StateObject",
            Kind::Blob => "ID3DBlob",
        }
    }
}

#[repr(C)]
pub struct NullObj {
    vtbl: *const usize,
    refs: AtomicU32,
    kind: Kind,
    /// Fence value / descriptor heap base / resource GPU VA.
    value: AtomicU64,
    /// Resource backing memory or blob data.
    mem: Mutex<(usize, usize)>,
    /// Raw descriptor (resource desc 56 B, heap desc 16 B, …).
    desc: [u8; 64],
}

unsafe impl Sync for NullObj {}
unsafe impl Send for NullObj {}

static CREATED: AtomicU64 = AtomicU64::new(0);

fn vtable(kind: Kind) -> *const usize {
    static TABLES: OnceLock<Mutex<Vec<(Kind, usize)>>> = OnceLock::new();
    let t = TABLES.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = t.lock().unwrap();
    if let Some(&(_, p)) = g.iter().find(|(k, _)| *k == kind) {
        return p as *const usize;
    }
    let mut v: Vec<usize> = (0..VTBL_SLOTS)
        .map(|i| crate::stubs::make_stub(DLL, &format!("{}::#{i}", kind.iface()), 0))
        .collect();
    v[0] = QueryInterface as usize;
    v[1] = AddRef as usize;
    v[2] = Release as usize;
    if kind == Kind::Blob {
        v[3] = Blob_GetBufferPointer as usize;
        v[4] = Blob_GetBufferSize as usize;
    } else {
        // ID3D12Object
        v[3] = Object_GetPrivateData as usize;
        v[4] = Object_Ok3 as usize; // SetPrivateData
        v[5] = Object_Ok2 as usize; // SetPrivateDataInterface
        v[6] = Object_Ok1 as usize; // SetName
        if kind != Kind::Device {
            v[7] = DeviceChild_GetDevice as usize;
        }
    }
    match kind {
        Kind::Device => {
            v[7] = Device_GetNodeCount as usize;
            v[8] = Device_CreateCommandQueue as usize;
            v[9] = Device_CreateCommandAllocator as usize;
            v[10] = Device_CreateGraphicsPipelineState as usize;
            v[11] = Device_CreateComputePipelineState as usize;
            v[12] = Device_CreateCommandList as usize;
            v[13] = Device_CheckFeatureSupport as usize;
            v[14] = Device_CreateDescriptorHeap as usize;
            v[15] = Device_GetDescriptorHandleIncrementSize as usize;
            v[16] = Device_CreateRootSignature as usize;
            v[25] = Device_GetResourceAllocationInfo as usize;
            v[26] = Device_GetCustomHeapProperties as usize;
            v[27] = Device_CreateCommittedResource as usize;
            v[28] = Device_CreateHeap as usize;
            v[29] = Device_CreatePlacedResource as usize;
            v[30] = Device_CreateReservedResource as usize;
            v[31] = Ret_E_NOTIMPL as usize; // CreateSharedHandle
            v[34] = Ret_S_OK as usize; // MakeResident
            v[35] = Ret_S_OK as usize; // Evict
            v[36] = Device_CreateFence as usize;
            v[37] = Ret_S_OK as usize; // GetDeviceRemovedReason
            v[38] = Device_GetCopyableFootprints as usize;
            v[39] = Device_CreateQueryHeap as usize;
            v[40] = Ret_S_OK as usize; // SetStablePowerState
            v[41] = Device_CreateCommandSignature as usize;
            v[43] = Device_GetAdapterLuid as usize;
            v[44] = Device_CreatePipelineLibrary as usize;
            v[45] = Device_SetEventOnMultipleFenceCompletion as usize;
            v[46] = Ret_S_OK as usize; // SetResidencyPriority
            v[47] = Device_CreatePipelineStateStream as usize; // CreatePipelineState (Device2)
            v[51] = Device_CreateCommandList1 as usize;
            v[53] = Device_CreateCommittedResource1 as usize;
            v[54] = Device_CreateHeap1 as usize;
            v[62] = Device_CreatePipelineState4 as usize; // CreateStateObject (Device5)
        }
        Kind::CommandQueue => {
            v[10] = Ret_Void as usize; // ExecuteCommandLists
            v[14] = Queue_Signal as usize;
            v[15] = Ret_S_OK as usize; // Wait
            v[16] = Queue_GetTimestampFrequency as usize;
            v[17] = Queue_GetClockCalibration as usize;
            v[18] = Hidden_ZeroDesc16 as usize; // GetDesc
        }
        Kind::CommandAllocator => v[8] = Ret_S_OK as usize, // Reset
        Kind::CommandList => {
            v[8] = Ret_Zero as usize; // GetType → DIRECT
            v[9] = Ret_S_OK as usize; // Close
            v[10] = Ret_S_OK as usize; // Reset
        }
        Kind::Fence => {
            v[8] = Fence_GetCompletedValue as usize;
            v[9] = Fence_SetEventOnCompletion as usize;
            v[10] = Fence_Signal as usize;
        }
        Kind::DescriptorHeap => {
            v[8] = Heap_GetDesc16 as usize;
            v[9] = Heap_GetHandleStart as usize; // CPU
            v[10] = Heap_GetHandleStart as usize; // GPU
        }
        Kind::Resource => {
            v[8] = Resource_Map as usize;
            v[9] = Ret_Void as usize; // Unmap
            v[10] = Resource_GetDesc as usize;
            v[11] = Resource_GetGPUVirtualAddress as usize;
            v[12] = Ret_S_OK as usize; // WriteToSubresource
            v[13] = Ret_S_OK as usize; // ReadFromSubresource
            v[14] = Resource_GetHeapProperties as usize;
        }
        Kind::Heap => v[8] = Hidden_ZeroDesc40 as usize,
        Kind::PipelineState => v[8] = Ret_E_NOTIMPL as usize, // GetCachedBlob
        _ => {}
    }
    let leaked: &'static [usize] = Box::leak(v.into_boxed_slice());
    g.push((kind, leaked.as_ptr() as usize));
    leaked.as_ptr()
}

fn new_obj(kind: Kind) -> *mut NullObj {
    let n = CREATED.fetch_add(1, Ordering::Relaxed);
    if n < 64 || n % 10_000 == 0 {
        crate::xlog_debug!("D3D12.X null: new {} (#{n})", kind.iface());
    }
    Box::into_raw(Box::new(NullObj {
        vtbl: vtable(kind),
        refs: AtomicU32::new(1),
        kind,
        value: AtomicU64::new(0),
        mem: Mutex::new((0, 0)),
        desc: [0; 64],
    }))
}

unsafe fn out_obj(ppv: *mut *mut c_void, kind: Kind) -> HRESULT {
    if ppv.is_null() {
        return S_FALSE;
    }
    *ppv = new_obj(kind) as *mut c_void;
    S_OK
}

fn device() -> *mut NullObj {
    static DEV: OnceLock<usize> = OnceLock::new();
    *DEV.get_or_init(|| new_obj(Kind::Device) as usize) as *mut NullObj
}

// ---------------------------------------------------------------------------------------
// common slots
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn QueryInterface(this: *mut NullObj, _riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return E_INVALIDARG;
    }
    (*this).refs.fetch_add(1, Ordering::AcqRel);
    *ppv = this as *mut c_void;
    S_OK
}
unsafe extern "system" fn AddRef(this: *mut NullObj) -> u32 {
    (*this).refs.fetch_add(1, Ordering::AcqRel) + 1
}
unsafe extern "system" fn Release(this: *mut NullObj) -> u32 {
    // Objects are intentionally leaked: a null device must never free memory a title
    // may still reference through a mis-guessed vtable slot.
    (*this).refs.fetch_sub(1, Ordering::AcqRel).saturating_sub(1)
}
unsafe extern "system" fn Object_GetPrivateData(_t: *mut NullObj, _g: *const c_void, size: *mut u32, _d: *mut c_void) -> HRESULT {
    if !size.is_null() {
        *size = 0;
    }
    0x887A_0002u32 as i32 // DXGI_ERROR_NOT_FOUND
}
unsafe extern "system" fn Object_Ok3(_t: *mut NullObj, _a: usize, _b: usize, _c: usize) -> HRESULT {
    S_OK
}
unsafe extern "system" fn Object_Ok2(_t: *mut NullObj, _a: usize, _b: usize) -> HRESULT {
    S_OK
}
unsafe extern "system" fn Object_Ok1(_t: *mut NullObj, _a: usize) -> HRESULT {
    S_OK
}
unsafe extern "system" fn DeviceChild_GetDevice(_t: *mut NullObj, _riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return E_INVALIDARG;
    }
    *ppv = device() as *mut c_void;
    S_OK
}
unsafe extern "system" fn Ret_S_OK(_t: *mut NullObj) -> HRESULT {
    S_OK
}
unsafe extern "system" fn Ret_E_NOTIMPL(_t: *mut NullObj) -> HRESULT {
    E_NOTIMPL
}
unsafe extern "system" fn Ret_Zero(_t: *mut NullObj) -> usize {
    0
}
unsafe extern "system" fn Ret_Void(_t: *mut NullObj) {}
/// Methods returning a 16-byte struct via hidden pointer (MSVC member-function ABI).
unsafe extern "system" fn Hidden_ZeroDesc16(_t: *mut NullObj, ret: *mut u8) -> *mut u8 {
    if !ret.is_null() {
        std::ptr::write_bytes(ret, 0, 16);
    }
    ret
}
unsafe extern "system" fn Hidden_ZeroDesc40(_t: *mut NullObj, ret: *mut u8) -> *mut u8 {
    if !ret.is_null() {
        std::ptr::write_bytes(ret, 0, 40);
    }
    ret
}

// ---------------------------------------------------------------------------------------
// device
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn Device_GetNodeCount(_t: *mut NullObj) -> u32 {
    1
}
unsafe extern "system" fn Device_CreateCommandQueue(_t: *mut NullObj, _d: *const c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::CommandQueue)
}
unsafe extern "system" fn Device_CreateCommandAllocator(_t: *mut NullObj, _ty: u32, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::CommandAllocator)
}
unsafe extern "system" fn Device_CreatePipelineState4(_t: *mut NullObj, _d: *const c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::PipelineState)
}

/// D3D12_SHADER_BYTECODE { const void* pShaderBytecode; SIZE_T BytecodeLength; }
unsafe fn capture_bytecode(stage: &str, at: *const u8) {
    let p = *(at as *const usize);
    let n = *(at.add(8) as *const usize);
    crate::gpu::shader::capture(stage, p, n);
}

/// D3D12_GRAPHICS_PIPELINE_STATE_DESC: pRootSignature @0, VS @8, PS @24, DS @40, HS @56, GS @72.
pub unsafe fn capture_graphics_desc(desc: *const u8) {
    if !crate::gpu::shader::readable(desc as usize, 88) {
        return;
    }
    for (stage, off) in [("vs", 8), ("ps", 24), ("ds", 40), ("hs", 56), ("gs", 72)] {
        capture_bytecode(stage, desc.add(off));
    }
}

/// D3D12_COMPUTE_PIPELINE_STATE_DESC: pRootSignature @0, CS @8.
pub unsafe fn capture_compute_desc(desc: *const u8) {
    if crate::gpu::shader::readable(desc as usize, 24) {
        capture_bytecode("cs", desc.add(8));
    }
}

/// (payload size, payload alignment) of D3D12_PIPELINE_STATE_SUBOBJECT_TYPE values.
pub fn subobject_payload(ty: u32) -> Option<(usize, usize)> {
    Some(match ty {
        0 => (8, 8),                          // ROOT_SIGNATURE
        1..=6 | 24 | 25 => (16, 8),           // VS PS DS HS GS CS AS MS
        7 => (32, 8),                         // STREAM_OUTPUT
        8 => (328, 4),                        // BLEND
        9 | 13 | 14 | 16 | 18 | 20 => (4, 4), // SAMPLE_MASK, IB_STRIP_CUT, TOPOLOGY, DS_FORMAT, NODE_MASK, FLAGS
        10 | 27 => (44, 4),                   // RASTERIZER, RASTERIZER1
        11 => (52, 4),                        // DEPTH_STENCIL
        12 => (16, 8),                        // INPUT_LAYOUT
        15 => (36, 4),                        // RENDER_TARGET_FORMATS
        17 => (8, 4),                         // SAMPLE_DESC
        19 => (16, 8),                        // CACHED_PSO
        21 => (56, 4),                        // DEPTH_STENCIL1
        22 => (24, 8),                        // VIEW_INSTANCING
        26 => (60, 4),                        // DEPTH_STENCIL2
        28 => (40, 4),                        // RASTERIZER2
        _ => return None,
    })
}

/// D3D12_PIPELINE_STATE_STREAM_DESC { SIZE_T SizeInBytes; void* pPipelineStateSubobjectStream; }
pub unsafe fn capture_stream_desc(desc: *const u8) {
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return;
    }
    let size = *(desc as *const usize);
    let base = *(desc.add(8) as *const usize);
    if !crate::gpu::shader::readable(base, size) {
        return;
    }
    let mut off = 0usize;
    while off + 8 <= size {
        let ty = *((base + off) as *const u32);
        let Some((psize, palign)) = subobject_payload(ty) else {
            crate::xlog_debug!("[SHADER] PSO stream: unknown subobject type {ty} – stopping the parse");
            break;
        };
        let poff = if palign == 8 { 8 } else { 4 };
        if off + poff + psize > size {
            break; // SizeInBytes ends inside the payload
        }
        let stage = match ty {
            1 => Some("vs"),
            2 => Some("ps"),
            3 => Some("ds"),
            4 => Some("hs"),
            5 => Some("gs"),
            6 => Some("cs"),
            24 => Some("as"),
            25 => Some("ms"),
            _ => None,
        };
        if let Some(s) = stage {
            capture_bytecode(s, (base + off + poff) as *const u8);
        }
        off += (poff + psize + 7) & !7;
    }
}

unsafe extern "system" fn Device_CreateGraphicsPipelineState(_t: *mut NullObj, d: *const u8, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    capture_graphics_desc(d);
    out_obj(ppv, Kind::PipelineState)
}
unsafe extern "system" fn Device_CreateComputePipelineState(_t: *mut NullObj, d: *const u8, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    capture_compute_desc(d);
    out_obj(ppv, Kind::PipelineState)
}
unsafe extern "system" fn Device_CreatePipelineStateStream(_t: *mut NullObj, d: *const u8, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    capture_stream_desc(d);
    out_obj(ppv, Kind::PipelineState)
}
unsafe extern "system" fn Device_CreateCommandList(_t: *mut NullObj, _m: u32, _ty: u32, _a: *mut c_void, _p: *mut c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::CommandList)
}
unsafe extern "system" fn Device_CreateCommandList1(_t: *mut NullObj, _m: u32, _ty: u32, _f: u32, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::CommandList)
}
unsafe extern "system" fn Device_CheckFeatureSupport(_t: *mut NullObj, feature: u32, data: *mut c_void, size: u32) -> HRESULT {
    crate::xlog_debug!("D3D12.X null: CheckFeatureSupport({feature})");
    // D3D12_FEATURE_D3D12_OPTIONS5 = 27: RaytracingTier sits at +8
    if feature == 27 && !data.is_null() && size >= 12 {
        let tier = crate::state::config().raytracing_tier_override().unwrap_or(0);
        *((data as *mut u8).add(8) as *mut u32) = tier;
    }
    S_OK
}
unsafe extern "system" fn Device_CreateDescriptorHeap(_t: *mut NullObj, desc: *const u8, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return S_FALSE;
    }
    let o = new_obj(Kind::DescriptorHeap);
    if !desc.is_null() {
        std::ptr::copy_nonoverlapping(desc, (*o).desc.as_mut_ptr(), 16);
        let n = *(desc.add(4) as *const u32) as usize;
        // unique, never-dereferenced-by-us address range for the handles
        let mem = vec![0u8; (n.max(1) * 32).min(64 << 20)].leak();
        (*o).value.store(mem.as_ptr() as u64, Ordering::Relaxed);
    }
    *ppv = o as *mut c_void;
    S_OK
}
unsafe extern "system" fn Device_GetDescriptorHandleIncrementSize(_t: *mut NullObj, _ty: u32) -> u32 {
    32
}
unsafe extern "system" fn Device_CreateRootSignature(_t: *mut NullObj, _m: u32, b: *const c_void, l: usize, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    crate::gpu::shader::capture("rootsig", b as usize, l);
    out_obj(ppv, Kind::RootSignature)
}

/// D3D12_RESOURCE_DESC: Dimension u32 @0, Alignment u64 @8, Width u64 @16, Height u32 @24,
/// DepthOrArraySize u16 @28, MipLevels u16 @30, Format u32 @32 … (56 bytes).
unsafe fn approx_size(desc: *const u8) -> usize {
    if desc.is_null() {
        return 0x1000;
    }
    let dim = *(desc as *const u32);
    let width = *(desc.add(16) as *const u64) as usize;
    if dim == 1 {
        return width.max(1);
    }
    let height = (*(desc.add(24) as *const u32)).max(1) as usize;
    let depth = (*(desc.add(28) as *const u16)).max(1) as usize;
    (width.max(1) * height * depth * 4 * 4 / 3).min(256 << 20)
}

unsafe fn new_resource(desc: *const u8, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return S_FALSE;
    }
    let o = new_obj(Kind::Resource);
    if !desc.is_null() {
        std::ptr::copy_nonoverlapping(desc, (*o).desc.as_mut_ptr(), 56);
    }
    let size = approx_size(desc);
    *(*o).mem.lock().unwrap() = (0, size);
    // unique fake GPU VA (64 KiB aligned bump allocator)
    static NEXT_VA: AtomicU64 = AtomicU64::new(0x1_0000_0000);
    let va = NEXT_VA.fetch_add(((size as u64) + 0xFFFF) & !0xFFFF, Ordering::Relaxed);
    (*o).value.store(va, Ordering::Relaxed);
    *ppv = o as *mut c_void;
    S_OK
}

unsafe extern "system" fn Device_GetResourceAllocationInfo(_t: *mut NullObj, ret: *mut u64, _mask: u32, n: u32, descs: *const u8) -> *mut u64 {
    let mut total = 0usize;
    for i in 0..n as usize {
        total += (approx_size(descs.add(i * 56)) + 0xFFFF) & !0xFFFF;
    }
    if !ret.is_null() {
        *ret = total as u64;
        *ret.add(1) = 0x10000;
    }
    ret
}
unsafe extern "system" fn Device_GetCustomHeapProperties(_t: *mut NullObj, ret: *mut u8, _m: u32, ty: u32) -> *mut u8 {
    if !ret.is_null() {
        std::ptr::write_bytes(ret, 0, 20);
        *(ret as *mut u32) = 4; // CUSTOM
        *(ret.add(4) as *mut u32) = if ty == 2 || ty == 3 { 3 } else { 1 }; // CPU page property
        *(ret.add(8) as *mut u32) = 1; // memory pool L0
    }
    ret
}
unsafe extern "system" fn Device_CreateCommittedResource(_t: *mut NullObj, _hp: *const c_void, _hf: u32, desc: *const u8, _st: u32, _cv: *const c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    new_resource(desc, ppv)
}
unsafe extern "system" fn Device_CreateCommittedResource1(_t: *mut NullObj, _hp: *const c_void, _hf: u32, desc: *const u8, _st: u32, _cv: *const c_void, _ps: *mut c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    new_resource(desc, ppv)
}
unsafe extern "system" fn Device_CreateHeap(_t: *mut NullObj, _d: *const c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::Heap)
}
unsafe extern "system" fn Device_CreateHeap1(_t: *mut NullObj, _d: *const c_void, _s: *mut c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::Heap)
}
unsafe extern "system" fn Device_CreatePlacedResource(_t: *mut NullObj, _h: *mut c_void, _o: u64, desc: *const u8, _st: u32, _cv: *const c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    new_resource(desc, ppv)
}
unsafe extern "system" fn Device_CreateReservedResource(_t: *mut NullObj, desc: *const u8, _st: u32, _cv: *const c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    new_resource(desc, ppv)
}
unsafe extern "system" fn Device_CreateFence(_t: *mut NullObj, initial: u64, _f: u32, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return S_FALSE;
    }
    let o = new_obj(Kind::Fence);
    (*o).value.store(initial, Ordering::Relaxed);
    *ppv = o as *mut c_void;
    S_OK
}
unsafe extern "system" fn Device_GetCopyableFootprints(
    _t: *mut NullObj,
    desc: *const u8,
    _first: u32,
    num: u32,
    base: u64,
    layouts: *mut u8,
    num_rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    let size = approx_size(desc) as u64;
    let width = if desc.is_null() { 1 } else { (*(desc.add(16) as *const u64)).max(1) };
    let height = if desc.is_null() { 1 } else { (*(desc.add(24) as *const u32)).max(1) };
    let format = if desc.is_null() { 0 } else { *(desc.add(32) as *const u32) };
    let row = ((width * 4 + 255) & !255) as u32;
    for i in 0..num as usize {
        if !layouts.is_null() {
            let l = layouts.add(i * 32);
            *(l as *mut u64) = base;
            *(l.add(8) as *mut u32) = format;
            *(l.add(12) as *mut u32) = width as u32;
            *(l.add(16) as *mut u32) = height;
            *(l.add(20) as *mut u32) = 1;
            *(l.add(24) as *mut u32) = row;
        }
        if !num_rows.is_null() {
            *num_rows.add(i) = height;
        }
        if !row_sizes.is_null() {
            *row_sizes.add(i) = width * 4;
        }
    }
    if !total.is_null() {
        *total = size.max(row as u64 * height as u64);
    }
}
unsafe extern "system" fn Device_CreateQueryHeap(_t: *mut NullObj, _d: *const c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::QueryHeap)
}
unsafe extern "system" fn Device_CreateCommandSignature(_t: *mut NullObj, _d: *const c_void, _rs: *mut c_void, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::CommandSignature)
}
unsafe extern "system" fn Device_GetAdapterLuid(_t: *mut NullObj, ret: *mut u64) -> *mut u64 {
    if !ret.is_null() {
        *ret = 0x5843_4C00;
    }
    ret
}
unsafe extern "system" fn Device_CreatePipelineLibrary(_t: *mut NullObj, _b: *const c_void, _l: usize, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    out_obj(ppv, Kind::PipelineLibrary)
}
unsafe extern "system" fn Device_SetEventOnMultipleFenceCompletion(_t: *mut NullObj, _f: *const c_void, _v: *const u64, _n: u32, _flags: u32, ev: *mut c_void) -> HRESULT {
    if !ev.is_null() {
        windows_sys::Win32::System::Threading::SetEvent(ev);
    }
    S_OK
}

// ---------------------------------------------------------------------------------------
// queue / fence / heap / resource / blob
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn Queue_Signal(_t: *mut NullObj, fence: *mut NullObj, value: u64) -> HRESULT {
    if !fence.is_null() && (*fence).kind == Kind::Fence {
        (*fence).value.fetch_max(value, Ordering::AcqRel);
    }
    S_OK
}
unsafe extern "system" fn Queue_GetTimestampFrequency(_t: *mut NullObj, f: *mut u64) -> HRESULT {
    if f.is_null() {
        return E_INVALIDARG;
    }
    *f = 10_000_000;
    S_OK
}
unsafe extern "system" fn Queue_GetClockCalibration(_t: *mut NullObj, gpu: *mut u64, cpu: *mut u64) -> HRESULT {
    let mut qpc = 0i64;
    windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut qpc);
    if !gpu.is_null() {
        *gpu = qpc as u64;
    }
    if !cpu.is_null() {
        *cpu = qpc as u64;
    }
    S_OK
}
unsafe extern "system" fn Fence_GetCompletedValue(this: *mut NullObj) -> u64 {
    (*this).value.load(Ordering::Acquire)
}
unsafe extern "system" fn Fence_SetEventOnCompletion(this: *mut NullObj, value: u64, ev: *mut c_void) -> HRESULT {
    // the null GPU finishes everything instantly
    (*this).value.fetch_max(value, Ordering::AcqRel);
    if !ev.is_null() {
        windows_sys::Win32::System::Threading::SetEvent(ev);
    }
    S_OK
}
unsafe extern "system" fn Fence_Signal(this: *mut NullObj, value: u64) -> HRESULT {
    (*this).value.store(value, Ordering::Release);
    S_OK
}
unsafe extern "system" fn Heap_GetDesc16(this: *mut NullObj, ret: *mut u8) -> *mut u8 {
    if !ret.is_null() {
        std::ptr::copy_nonoverlapping((*this).desc.as_ptr(), ret, 16);
    }
    ret
}
/// #9 / #10 GetCPU / GetGPUDescriptorHandleForHeapStart: console code takes the handle in rax
/// and passes only `this` (as `real::heap_cpu_start_x`); rdx holds whatever the caller left
/// there, so it must not be written through as the PC's hidden return pointer.
unsafe extern "system" fn Heap_GetHandleStart(this: *mut NullObj) -> u64 {
    (*this).value.load(Ordering::Relaxed)
}
unsafe extern "system" fn Resource_Map(this: *mut NullObj, _sub: u32, _range: *const c_void, data: *mut *mut c_void) -> HRESULT {
    let mut g = (*this).mem.lock().unwrap();
    if g.0 == 0 {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};
        g.0 = VirtualAlloc(std::ptr::null(), g.1.max(1), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as usize;
    }
    if !data.is_null() {
        *data = g.0 as *mut c_void;
    }
    if g.0 == 0 {
        crate::util::E_OUTOFMEMORY
    } else {
        S_OK
    }
}
unsafe extern "system" fn Resource_GetDesc(this: *mut NullObj, ret: *mut u8) -> *mut u8 {
    if !ret.is_null() {
        std::ptr::copy_nonoverlapping((*this).desc.as_ptr(), ret, 56);
    }
    ret
}
unsafe extern "system" fn Resource_GetGPUVirtualAddress(this: *mut NullObj) -> u64 {
    (*this).value.load(Ordering::Relaxed)
}
unsafe extern "system" fn Resource_GetHeapProperties(_t: *mut NullObj, props: *mut u8, flags: *mut u32) -> HRESULT {
    if !props.is_null() {
        std::ptr::write_bytes(props, 0, 20);
        *(props as *mut u32) = 1; // DEFAULT
    }
    if !flags.is_null() {
        *flags = 0;
    }
    S_OK
}
unsafe extern "system" fn Blob_GetBufferPointer(this: *mut NullObj) -> *mut c_void {
    (*this).mem.lock().unwrap().0 as *mut c_void
}
unsafe extern "system" fn Blob_GetBufferSize(this: *mut NullObj) -> usize {
    (*this).mem.lock().unwrap().1
}

unsafe fn new_blob(size: usize, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let o = new_obj(Kind::Blob);
    let buf = vec![0u8; size.max(1)].leak();
    *(*o).mem.lock().unwrap() = (buf.as_mut_ptr() as usize, size);
    *out = o as *mut c_void;
    S_OK
}

// ---------------------------------------------------------------------------------------
// exports
// ---------------------------------------------------------------------------------------

static ANNOUNCED: AtomicU32 = AtomicU32::new(0);

unsafe fn create_device(ppv: *mut *mut c_void) -> HRESULT {
    if ANNOUNCED.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_warn!("D3D12.X: creating a FALLBACK DEVICE (null device) – graphics are not rendered in milestone M1");
    }
    if ppv.is_null() {
        return S_FALSE;
    }
    let d = device();
    (*d).refs.fetch_add(1, Ordering::AcqRel);
    *ppv = d as *mut c_void;
    S_OK
}

pub unsafe extern "system" fn D3D12CreateDevice(_adapter: *mut c_void, _fl: u32, _riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    create_device(ppv)
}
pub unsafe extern "system" fn D3D12XboxCreateDevice(_adapter: *mut c_void, _params: *const c_void, _riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    create_device(ppv)
}
pub unsafe extern "system" fn D3D12GetDebugInterface(_riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if !ppv.is_null() {
        *ppv = std::ptr::null_mut();
    }
    E_NOINTERFACE
}
pub unsafe extern "system" fn D3D12SerializeRootSignature(_d: *const c_void, _v: u32, blob: *mut *mut c_void, err: *mut *mut c_void) -> HRESULT {
    if !err.is_null() {
        *err = std::ptr::null_mut();
    }
    new_blob(16, blob)
}
pub unsafe extern "system" fn D3D12SerializeVersionedRootSignature(_d: *const c_void, blob: *mut *mut c_void, err: *mut *mut c_void) -> HRESULT {
    if !err.is_null() {
        *err = std::ptr::null_mut();
    }
    new_blob(16, blob)
}
pub unsafe extern "system" fn D3D12CreateRootSignatureDeserializer(_d: *const c_void, _l: usize, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if !ppv.is_null() {
        *ppv = std::ptr::null_mut();
    }
    E_NOTIMPL
}
pub unsafe extern "system" fn D3D12CreateVersionedRootSignatureDeserializer(_d: *const c_void, _l: usize, _r: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if !ppv.is_null() {
        *ppv = std::ptr::null_mut();
    }
    E_NOTIMPL
}
pub unsafe extern "system" fn D3D12EnableExperimentalFeatures(_n: u32, _i: *const c_void, _c: *mut c_void, _s: *mut u32) -> HRESULT {
    E_NOINTERFACE
}
pub unsafe extern "system" fn D3D12GetInterface(_clsid: *const c_void, _riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if !ppv.is_null() {
        *ppv = std::ptr::null_mut();
    }
    E_NOINTERFACE
}
pub unsafe extern "system" fn D3D10CreateBlob(size: usize, out: *mut *mut c_void) -> HRESULT {
    new_blob(size, out)
}
pub unsafe extern "system" fn D3D12XboxSetProcessDebugFlags(_flags: u32) -> HRESULT {
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    type Fn4 = unsafe extern "system" fn(*mut c_void, *const c_void, *const c_void, *mut *mut c_void) -> HRESULT;

    unsafe fn slot(obj: *mut c_void, i: usize) -> usize {
        *(*(obj as *const *const usize)).add(i)
    }

    /// A stream whose SizeInBytes ends inside a shader subobject's payload, right at the end of
    /// readable memory: the bytecode's length past the end is not read (an access violation in
    /// the device hook before).
    #[test]
    fn a_stream_ending_inside_a_shader_payload_is_not_read_past_its_end() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 0x2000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as *mut u8;
            assert!(!page.is_null());
            let mut old = 0u32;
            assert_ne!(VirtualProtect(page.add(0x1000) as *const _, 0x1000, PAGE_NOACCESS, &mut old), 0);
            // VS subobject (type 1) and the pointer half of its D3D12_SHADER_BYTECODE, ending at
            // the unreadable page
            let base = page.add(0x1000 - 16);
            *(base as *mut u64) = 1;
            *(base.add(8) as *mut u64) = 0x1234;
            let desc: [usize; 2] = [16, base as usize];
            capture_stream_desc(desc.as_ptr() as *const u8);
            VirtualFree(page as *mut _, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn device_queue_fence_resource_roundtrip() {
        unsafe {
            let mut dev = std::ptr::null_mut();
            assert_eq!(D3D12CreateDevice(std::ptr::null_mut(), 0xC100, std::ptr::null(), &mut dev), S_OK);
            let create_queue: Fn4 = std::mem::transmute(slot(dev, 8));
            let mut q = std::ptr::null_mut();
            assert_eq!(create_queue(dev, std::ptr::null(), std::ptr::null(), &mut q), S_OK);
            // fence: CreateFence(initial=5)
            let create_fence: unsafe extern "system" fn(*mut c_void, u64, u32, *const c_void, *mut *mut c_void) -> HRESULT =
                std::mem::transmute(slot(dev, 36));
            let mut f = std::ptr::null_mut();
            assert_eq!(create_fence(dev, 5, 0, std::ptr::null(), &mut f), S_OK);
            let signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT = std::mem::transmute(slot(q, 14));
            assert_eq!(signal(q, f, 9), S_OK);
            let completed: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(slot(f, 8));
            assert_eq!(completed(f), 9);
            // buffer resource: Map gives writable memory
            let mut desc = [0u8; 56];
            desc[0] = 1; // BUFFER
            desc[16..24].copy_from_slice(&4096u64.to_le_bytes());
            let create_res: unsafe extern "system" fn(*mut c_void, *const c_void, u32, *const u8, u32, *const c_void, *const c_void, *mut *mut c_void) -> HRESULT =
                std::mem::transmute(slot(dev, 27));
            let mut r = std::ptr::null_mut();
            assert_eq!(create_res(dev, std::ptr::null(), 0, desc.as_ptr(), 0, std::ptr::null(), std::ptr::null(), &mut r), S_OK);
            let map: unsafe extern "system" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> HRESULT = std::mem::transmute(slot(r, 8));
            let mut p = std::ptr::null_mut();
            assert_eq!(map(r, 0, std::ptr::null(), &mut p), S_OK);
            std::ptr::write_bytes(p as *mut u8, 0xAB, 4096);
            let gpu: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(slot(r, 11));
            assert_ne!(gpu(r), 0);
            // unknown slot → logging stub returning 0
            let unknown: unsafe extern "system" fn(*mut c_void) -> usize = std::mem::transmute(slot(dev, 200));
            assert_eq!(unknown(dev), 0);
            assert!(crate::stubs::all_stubs().iter().any(|s| s.name == "ID3D12Device::#200" && s.calls.load(Ordering::Relaxed) > 0));
        }
    }

    /// Console code calls descriptor heap #9 / #10 with only `this` and takes the start in rax;
    /// rdx is left over from the caller and must not be written (the test title's rdx pointed
    /// into its read-only .rdata: an access violation under Wine on the null device).
    #[test]
    fn heap_start_handles_come_back_in_rax() {
        unsafe {
            let mut dev = std::ptr::null_mut();
            assert_eq!(D3D12CreateDevice(std::ptr::null_mut(), 0xC100, std::ptr::null(), &mut dev), S_OK);
            let desc = [0u32, 16, 1, 0]; // CBV_SRV_UAV, 16 slots, shader visible
            let create_heap: Fn4 = std::mem::transmute(slot(dev, 14));
            let mut heaps = [std::ptr::null_mut(); 2];
            for h in &mut heaps {
                assert_eq!(create_heap(dev, desc.as_ptr() as *const c_void, std::ptr::null(), h), S_OK);
            }
            // the start is the heap's own handle range (CreateDescriptorHeap): not 0, one per heap
            let base = heaps.map(|h| (*(h as *mut NullObj)).value.load(Ordering::Relaxed));
            assert_ne!(base[0], 0);
            assert_ne!(base[0], base[1]);
            for (h, base) in heaps.into_iter().zip(base) {
                for i in [9, 10] {
                    #[cfg(all(target_arch = "x86_64", windows))]
                    {
                        let mut sentinel = 0x5A5A_5A5A_5A5A_5A5Au64;
                        let p = call_with_rdx(slot(h, i), h, &mut sentinel);
                        assert_eq!(sentinel, 0x5A5A_5A5A_5A5A_5A5A, "slot {i} wrote through rdx");
                        assert_eq!(p, base, "slot {i} with rdx set");
                    }
                    let start: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(slot(h, i));
                    assert_eq!(start(h), base, "slot {i}");
                }
            }
        }
    }

    /// Calls `f(this)` (Win64) the way console code does, with a writable pointer left in rdx.
    #[cfg(all(target_arch = "x86_64", windows))]
    unsafe fn call_with_rdx(f: usize, this: *mut c_void, rdx: *mut u64) -> u64 {
        let ret: u64;
        // rsp is 16-aligned on entry to the asm block: 32 B of shadow space keep it so at the call
        core::arch::asm!(
            "sub rsp, 32",
            "call {f}",
            "add rsp, 32",
            f = in(reg) f,
            in("rcx") this,
            in("rdx") rdx,
            lateout("rax") ret,
            clobber_abi("system"),
        );
        ret
    }
}
