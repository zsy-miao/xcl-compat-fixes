//! The console device's own methods and the PC ones whose arguments differ on the console.
//!
//! The console's ID3D12Device has the PC ID3D12Device methods at slots 0–43; from 44 on its
//! vtable holds console extensions (as observed on Fortnite): CreatePlacedResourceX #44,
//! CreateCommandQueueX #48, SetDebugFlagsX #52, CreateDerivedPipelineStateX #55, the
//! pipeline-state (de)serialisation
//! #56 / #57 / #66 / #67, CreateGraphicsPipelineStateX #60, CreateSamplerX #61,
//! CreateComputePipelineStateX #62, CreateCommandListX #73, page pools #74 / #75, the frame
//! clock #78–#81 (`frame`), CreatePipelineState #86 (streams, mesh shaders). A title call
//! of any other extension slot is logged and answered S_OK (`real::class_for`) – not run as
//! the PC method at that index.
//!
//! Console queue and list types add DMA (0x10, the copy engine); console queue priorities
//! are mapped onto PC ones.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicI8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::core::GUID;

use super::real;
use crate::util::{HRESULT, S_OK};

const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
const E_NOTIMPL: HRESULT = 0x8000_4001u32 as i32;
const E_ACCESSDENIED: HRESULT = 0x8007_0005u32 as i32;
const E_FAIL: HRESULT = 0x8000_4005u32 as i32;

/// D3D12_COMMAND_LIST_TYPE of the console's DMA (copy) engine
pub const XBOX_TYPE_DMA: u32 = 0x10;
/// IID of the console's DMA command list (created as a PC graphics command list)
const IID_XBOX_DMA_COMMAND_LIST: GUID = GUID::from_u128(0x30bc77b9_8379_4b25_95de_b8588c42275d);
const IID_GRAPHICS_COMMAND_LIST: GUID = GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
const IID_PIPELINE_STATE: GUID = GUID::from_u128(0x765a30f3_f624_4c6f_a828_ace948622445);
/// D3D12_INDIRECT_ARGUMENT_TYPE of the console's DispatchX
const XBOX_ARG_DISPATCH_X: u32 = 105;

fn guid_bytes(g: &GUID) -> (u32, u16, u16, [u8; 8]) {
    (g.data1, g.data2, g.data3, g.data4)
}

fn once(counter: &AtomicU64, n: u64) -> bool {
    counter.fetch_add(1, Ordering::Relaxed) < n
}

/// a console list / allocator type → the PC one
pub fn pc_list_type(t: u32) -> u32 {
    if t == XBOX_TYPE_DMA {
        D3D12_COMMAND_LIST_TYPE_COPY.0 as u32
    } else {
        t
    }
}

/// D3D12_COMMAND_QUEUE_DESC { type, priority, flags, node mask } of the console → PC
pub fn pc_queue_desc(d: [u32; 4]) -> [u32; 4] {
    let priority = match d[1] {
        0 | 100 | 10000 => d[1], // NORMAL, HIGH, GLOBAL_REALTIME
        _ => 100,
    };
    // flags: the PC has only DISABLE_GPU_TIMEOUT (1); the console's others (a retail game: 2, 3 on
    // compute queues) are dropped – the PC refuses them with E_INVALIDARG
    if d[2] & !1 != 0 {
        static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SEEN.swap(true, Ordering::Relaxed) {
            crate::xlog_info!("D3D12.X: command queue flags {:#x} (type {}) – the console-only bits are dropped", d[2], d[0]);
        }
    }
    [pc_list_type(d[0]), priority, d[2] & 1, if d[3] == 0 { 0 } else { 1 }]
}

/// device #8 CreateCommandQueue / #48 CreateCommandQueueX(desc, riid, queue)
pub(super) unsafe extern "system" fn create_command_queue_x(this: *mut c_void, desc: *const [u32; 4], riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let d = if desc.is_null() { [0; 4] } else { pc_queue_desc(desc.read_unaligned()) };
    let hr = real::dev_create_command_queue(this, d.as_ptr() as usize, riid, ppv);
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        super::list_x::on_queue_create(*ppv as usize, this as usize, D3D12_COMMAND_LIST_TYPE(d[0] as i32));
    }
    hr
}

/// device #9 CreateCommandAllocator(type, riid, allocator)
pub(super) unsafe extern "system" fn create_command_allocator_x(this: *mut c_void, ty: u32, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    real::dev_create_command_allocator(this, pc_list_type(ty) as usize, riid, ppv)
}

unsafe fn create_list(this: *mut c_void, node: u32, ty: u32, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let dma = !riid.is_null() && guid_bytes(&*riid) == guid_bytes(&IID_XBOX_DMA_COMMAND_LIST);
    let riid = if dma { &IID_GRAPHICS_COMMAND_LIST as *const GUID } else { riid };
    let (node, ty) = (if node == 0 { 0 } else { 1 }, pc_list_type(ty) as usize);
    let hr = if super::console_list::enabled() {
        real::dev_create_command_list_x(this, node, ty, alloc, pso, riid, ppv)
    } else {
        let resolved_pso = super::pso_async::resolve_for_bind(pso);
        let hr = real::dev_create_command_list(this, node, ty, alloc, resolved_pso, riid, ppv);
        if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
            super::list_x::on_create(*ppv, alloc, pso, D3D12_COMMAND_LIST_TYPE(ty as i32));
        }
        hr
    };
    if dma && hr >= 0 && !ppv.is_null() {
        real::readopt(*ppv, real::Kind::DmaList);
        if std::env::var_os("XCL_WATCH_DMA").is_some() {
            crate::eh::watch_writes(*ppv as usize);
        }
    }
    // the title's inline D3D12.X code writes into the list object: it gets one of the
    // console's shape (`title_list`)
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() && !super::console_list::enabled() {
        super::bind::reset(*ppv, alloc, pso);
        *ppv = super::title_list::wrap(*ppv);
    }
    hr
}

/// device #12 CreateCommandList(node mask, type, allocator, initial pipeline, riid, list):
/// console list types and the DMA list's IID
pub(super) unsafe extern "system" fn create_command_list_pc(this: *mut c_void, node: u32, ty: u32, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    create_list(this, node, ty, alloc, pso, riid, ppv)
}

/// device #73 CreateCommandListX(params { type @0; node mask @8 }, allocator, initial
/// pipeline, riid, list)
pub(super) unsafe extern "system" fn create_command_list_x(this: *mut c_void, params: *const u32, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let (ty, node) = if params.is_null() { (0, 1) } else { (params.read_unaligned(), params.add(2).read_unaligned()) };
    create_list(this, node, ty, alloc, pso, riid, ppv)
}

/// device #60 CreateGraphicsPipelineStateX(desc, extension count, extensions, riid, pipeline):
/// the desc is a PC D3D12_GRAPHICS_PIPELINE_STATE_DESC with console shaders; the extensions
/// are not modelled
pub(super) unsafe extern "system" fn create_graphics_pso_x(this: *mut c_void, desc: *const c_void, extensions: u32, _ext: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static N: AtomicU64 = AtomicU64::new(0);
    if extensions != 0 && once(&N, 1) {
        crate::xlog_info!("D3D12.X: CreateGraphicsPipelineStateX with {extensions} extension(s) – not modelled");
    }
    create_graphics_pso_pc(this, desc, riid, ppv)
}

/// device #10 CreateGraphicsPipelineState(desc, riid, pipeline): remembered as a base for
/// derived pipelines and tagged with `D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS` when supported.
pub(super) unsafe extern "system" fn create_graphics_pso_pc(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    create_graphics_pso_tracked(this, desc, riid, ppv)
}

/// Cached result of `CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16).DynamicDepthBiasSupported`:
/// -1 = not yet probed, 0 = unsupported, 1 = supported.
static DYN_DEPTH_BIAS: AtomicI8 = AtomicI8::new(-1);

/// Query `D3D12_FEATURE_D3D12_OPTIONS16` for native dynamic depth bias support and cache it.
pub(super) fn probe_dynamic_depth_bias(dev: &ID3D12Device) -> bool {
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    let supported = unsafe {
        dev.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            &mut opts as *mut _ as *mut c_void,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32,
        )
        .is_ok()
            && opts.DynamicDepthBiasSupported.as_bool()
    };
    DYN_DEPTH_BIAS.store(if supported { 1 } else { 0 }, Ordering::Release);
    crate::xlog_info!(
        "D3D12.X: native dynamic depth bias (OPTIONS16): {}",
        if supported { "supported (ID3D12GraphicsCommandList9::RSSetDepthBias)" } else { "unsupported (using patched-PSO fallback)" }
    );
    supported
}

/// Whether the D3D12 device supports `D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS` /
/// `ID3D12GraphicsCommandList9::RSSetDepthBias`. Lazily probes `dev` if not yet probed.
pub(super) unsafe fn dynamic_depth_bias_supported(dev: *mut c_void) -> bool {
    match DYN_DEPTH_BIAS.load(Ordering::Acquire) {
        1 => return true,
        0 => return false,
        _ => {}
    }
    if dev.is_null() {
        return false;
    }
    ID3D12Device::from_raw_borrowed(&dev).is_some_and(probe_dynamic_depth_bias)
}

/// Create a graphics PSO, enabling `D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS` when the
/// device supports it (retrying without the flag if creation fails), and record it as a base.
unsafe fn create_graphics_pso_tracked(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut used_dynamic = false;
    let hr = if !desc.is_null() && crate::gpu::shader::readable(desc as usize, GFX_DESC) {
        let orig = GfxDesc((desc as *const [u8; GFX_DESC]).read_unaligned());
        let already_dynamic = rd32(&orig.0, PSO_FLAGS) & FLAG_DYNAMIC_DEPTH_BIAS != 0;
        let want_dynamic = !already_dynamic && rd32(&orig.0, SO_ENTRIES) == 0 && dynamic_depth_bias_supported(this);
        if want_dynamic {
            let mut dyn_desc = orig;
            enable_dynamic_depth_bias(&mut dyn_desc.0);
            let hr = real::dev_create_graphics_pso(this, dyn_desc.0.as_ptr() as *const c_void, riid, ppv);
            if hr >= 0 {
                used_dynamic = true;
                hr
            } else {
                real::dev_create_graphics_pso(this, desc, riid, ppv)
            }
        } else {
            used_dynamic = already_dynamic;
            real::dev_create_graphics_pso(this, desc, riid, ppv)
        }
    } else {
        real::dev_create_graphics_pso(this, desc, riid, ppv)
    };
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        note_base(*ppv, desc, used_dynamic);
    }
    hr
}

/// size of D3D12_GRAPHICS_PIPELINE_STATE_DESC
const GFX_DESC: usize = 0x290;
/// VS, PS, DS, HS, GS { bytecode; length } in the desc
const STAGES: [usize; 5] = [0x08, 0x18, 0x28, 0x38, 0x48];
const SO_ENTRIES: usize = 0x60;
const INPUT_LAYOUT: usize = 0x228;
const CACHED_PSO: usize = 0x278;
/// Flags field in D3D12_GRAPHICS_PIPELINE_STATE_DESC
const PSO_FLAGS: usize = 0x288;
const FLAG_DYNAMIC_DEPTH_BIAS: u32 = D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS.0 as u32;
/// RasterizerState.{DepthBias, DepthBiasClamp, SlopeScaledDepthBias} in D3D12_GRAPHICS_PIPELINE_STATE_DESC
const RASTERIZER_DEPTH_BIAS: usize = 0x1D0;
const RASTERIZER_DEPTH_BIAS_CLAMP: usize = 0x1D4;
const RASTERIZER_SLOPE_SCALED_DEPTH_BIAS: usize = 0x1D8;
/// Maximum cached depth-bias PSO variants across all base pipelines
const MAX_DEPTH_BIAS_PSOS: usize = 1024;
/// CreateDerivedPipelineStateX patch: { u32 part; payload } at this stride (the largest
/// payload is the blend state)
const PATCH: usize = 0x14C;

/// Dynamic depth-bias override from `RSSetDepthBiasX`, stored as canonical f32 bits for hashing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub(super) struct DepthBias {
    pub bias: u32,
    pub clamp: u32,
    pub slope: u32,
}

impl DepthBias {
    pub(super) const ZERO: Self = Self { bias: 0, clamp: 0, slope: 0 };

    fn canon(v: f32) -> u32 {
        if v == 0.0 {
            0.0f32.to_bits()
        } else {
            v.to_bits()
        }
    }

    pub(super) fn new(bias: f32, clamp: f32, slope: f32) -> Self {
        Self { bias: Self::canon(bias), clamp: Self::canon(clamp), slope: Self::canon(slope) }
    }

    pub(super) fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    pub(super) fn depth_bias(self) -> f32 {
        f32::from_bits(self.bias)
    }

    pub(super) fn depth_bias_clamp(self) -> f32 {
        f32::from_bits(self.clamp)
    }

    pub(super) fn slope_scaled_depth_bias(self) -> f32 {
        f32::from_bits(self.slope)
    }

    pub(super) fn from_desc(d: &[u8; GFX_DESC]) -> Self {
        let bias = i32::from_le_bytes(d[RASTERIZER_DEPTH_BIAS..RASTERIZER_DEPTH_BIAS + 4].try_into().unwrap()) as f32;
        let clamp = f32::from_le_bytes(d[RASTERIZER_DEPTH_BIAS_CLAMP..RASTERIZER_DEPTH_BIAS_CLAMP + 4].try_into().unwrap());
        let slope = f32::from_le_bytes(d[RASTERIZER_SLOPE_SCALED_DEPTH_BIAS..RASTERIZER_SLOPE_SCALED_DEPTH_BIAS + 4].try_into().unwrap());
        Self::new(bias, clamp, slope)
    }

    pub(super) fn patch_desc(self, d: &mut [u8; GFX_DESC]) {
        let bias_i32 = self.depth_bias().round() as i32;
        d[RASTERIZER_DEPTH_BIAS..RASTERIZER_DEPTH_BIAS + 4].copy_from_slice(&bias_i32.to_le_bytes());
        d[RASTERIZER_DEPTH_BIAS_CLAMP..RASTERIZER_DEPTH_BIAS_CLAMP + 4].copy_from_slice(&self.clamp.to_le_bytes());
        d[RASTERIZER_SLOPE_SCALED_DEPTH_BIAS..RASTERIZER_SLOPE_SCALED_DEPTH_BIAS + 4].copy_from_slice(&self.slope.to_le_bytes());
    }
}

fn enable_dynamic_depth_bias(d: &mut [u8; GFX_DESC]) {
    let flags = rd32(d, PSO_FLAGS) | FLAG_DYNAMIC_DEPTH_BIAS;
    d[PSO_FLAGS..PSO_FLAGS + 4].copy_from_slice(&flags.to_le_bytes());
}

fn clear_dynamic_depth_bias(d: &mut [u8; GFX_DESC]) {
    let flags = rd32(d, PSO_FLAGS) & !FLAG_DYNAMIC_DEPTH_BIAS;
    d[PSO_FLAGS..PSO_FLAGS + 4].copy_from_slice(&flags.to_le_bytes());
}

/// A graphics pipeline desc as bytes, aligned like the struct for the runtime's reads.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct GfxDesc([u8; GFX_DESC]);

/// A title graphics pipeline's description, kept for CreateDerivedPipelineStateX and the
/// patched-PSO fallback of RSSetDepthBiasX. The input layout is copied; the shaders stay the
/// title's and are checked by hash when a pipeline is derived from it.
struct Base {
    desc: GfxDesc,
    shaders: [u64; 5],
    /// Whether this PSO was created with `D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS`.
    dynamic_depth_bias: bool,
    /// Original static `RasterizerState.{DepthBias, DepthBiasClamp, SlopeScaledDepthBias}`.
    static_bias: DepthBias,
    _elems: Vec<[u8; 32]>,
    _names: Vec<CString>,
}

// the desc points into `_elems` / `_names` and the title's shaders; nothing is mutated
unsafe impl Send for Base {}
unsafe impl Sync for Base {}

static BASES: Mutex<Option<HashMap<usize, Arc<Base>>>> = Mutex::new(None);
/// Cached derived PSOs for `RSSetDepthBiasX`: (base PSO, override) → owned derived PSO.
/// Released when the base PSO is released in `forget_base`.
static DEPTH_BIAS_PSOS: Mutex<Option<HashMap<(usize, DepthBias), usize>>> = Mutex::new(None);

fn rd32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}

fn rd64(d: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(d[o..o + 8].try_into().unwrap())
}

/// hash of each stage's bytecode (0: none); None when one is not readable
unsafe fn stage_hashes(d: &[u8; GFX_DESC]) -> Option<[u64; 5]> {
    let mut h = [0; 5];
    for (k, &o) in STAGES.iter().enumerate() {
        let (p, n) = (rd64(d, o) as usize, rd64(d, o + 8) as usize);
        if p != 0 && n != 0 {
            if !crate::gpu::shader::readable(p, n) {
                return None;
            }
            h[k] = crate::gpu::shader::fnv64(std::slice::from_raw_parts(p as *const u8, n));
        }
    }
    Some(h)
}

/// Remember a title graphics pipeline's description (not with stream output).
unsafe fn note_base(pso: *mut c_void, desc: *const c_void, dynamic_depth_bias: bool) {
    if pso.is_null() || desc.is_null() || !crate::gpu::shader::readable(desc as usize, GFX_DESC) {
        return;
    }
    let mut d = GfxDesc((desc as *const [u8; GFX_DESC]).read_unaligned());
    if rd32(&d.0, SO_ENTRIES) != 0 {
        return;
    }
    let Some(shaders) = stage_hashes(&d.0) else { return };
    let static_bias = DepthBias::from_desc(&d.0);
    let (p, n) = (rd64(&d.0, INPUT_LAYOUT) as usize, rd32(&d.0, INPUT_LAYOUT + 8) as usize);
    let (mut elems, mut names) = (Vec::new(), Vec::new());
    if n > 0 {
        if p == 0 || n > 64 || !crate::gpu::shader::readable(p, 32 * n) {
            return;
        }
        for i in 0..n {
            // D3D12_INPUT_ELEMENT_DESC { SemanticName @0; … }
            let mut e = ((p + 32 * i) as *const [u8; 32]).read_unaligned();
            let s = rd64(&e, 0) as *const c_char;
            let name = if s.is_null() { CString::default() } else { CStr::from_ptr(s).to_owned() };
            e[0..8].copy_from_slice(&(name.as_ptr() as u64).to_le_bytes());
            names.push(name);
            elems.push(e);
        }
    }
    let at = if elems.is_empty() { 0 } else { elems.as_ptr() as u64 };
    d.0[INPUT_LAYOUT..INPUT_LAYOUT + 8].copy_from_slice(&at.to_le_bytes());
    d.0[CACHED_PSO..CACHED_PSO + 16].fill(0);
    let base = Arc::new(Base { desc: d, shaders, dynamic_depth_bias, static_bias, _elems: elems, _names: names });
    BASES.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso as usize, base);
}

/// Depth-bias metadata for a tracked graphics PSO: `(created_with_dynamic_depth_bias, static_bias)`.
pub(super) fn base_depth_bias_info(pso: usize) -> Option<(bool, DepthBias)> {
    if pso == 0 {
        return None;
    }
    BASES.lock().unwrap().as_ref()?.get(&pso).map(|b| (b.dynamic_depth_bias, b.static_bias))
}

#[cfg(test)]
pub(super) fn depth_bias_cache_len() -> usize {
    DEPTH_BIAS_PSOS.lock().unwrap().as_ref().map_or(0, HashMap::len)
}

#[cfg(test)]
pub(super) fn register_test_base(pso: usize, dynamic_depth_bias: bool, static_bias: DepthBias) {
    let base = Arc::new(Base {
        desc: GfxDesc([0u8; GFX_DESC]),
        shaders: [0; 5],
        dynamic_depth_bias,
        static_bias,
        _elems: Vec::new(),
        _names: Vec::new(),
    });
    BASES.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, base);
}

/// A pipeline released for good is no longer a base; its cached depth-bias variants go with it.
pub(super) fn forget_base(pso: usize) {
    if let Some(m) = BASES.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    let gone: Vec<usize> = {
        let mut g = DEPTH_BIAS_PSOS.lock().unwrap();
        match g.as_mut() {
            Some(m) => {
                let mut out = Vec::new();
                m.retain(|&(base, _), &mut derived| {
                    if base == pso {
                        out.push(derived);
                        false
                    } else {
                        true
                    }
                });
                out
            }
            None => Vec::new(),
        }
    };
    // release outside the lock: pso_release calls forget_pso -> forget_base on each derived PSO
    for derived in gone {
        super::pso_async::release(derived);
    }
}

/// Derive (or reuse from cache) a graphics PSO from `base` with patched rasterizer depth-bias
/// fields for `RSSetDepthBiasX`. Returns `None` if `base` is unknown, its shaders are gone,
/// the cache cap is reached, or creation fails.
pub(super) unsafe fn depth_bias_pso(dev: *mut c_void, base: usize, bias: DepthBias) -> Option<usize> {
    static N: AtomicU64 = AtomicU64::new(0);
    if dev.is_null() || base == 0 {
        return None;
    }
    if let Some(&cached) = DEPTH_BIAS_PSOS.lock().unwrap().as_ref().and_then(|m| m.get(&(base, bias))) {
        return Some(cached);
    }
    let b = BASES.lock().unwrap().as_ref().and_then(|m| m.get(&base).cloned())?;
    if stage_hashes(&b.desc.0) != Some(b.shaders) {
        if once(&N, 4) {
            crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – shaders of base pipeline {base:#x} are gone");
        }
        return None;
    }
    let mut d = b.desc;
    clear_dynamic_depth_bias(&mut d.0);
    bias.patch_desc(&mut d.0);
    let mut out: *mut c_void = std::ptr::null_mut();
    let hr = real::dev_create_graphics_pso(dev, d.0.as_ptr() as *const c_void, &IID_PIPELINE_STATE, &mut out);
    if hr < 0 || out.is_null() {
        if once(&N, 4) {
            crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – deriving pipeline from {base:#x} failed ({hr:#x})");
        }
        return None;
    }
    let derived = out as usize;
    let mut release_extra = 0usize;
    let result = {
        // re-check base liveness so we never leak a derived PSO for a concurrently released base
        let alive = BASES.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&base));
        let mut g = DEPTH_BIAS_PSOS.lock().unwrap();
        let map = g.get_or_insert_with(HashMap::new);
        if !alive {
            release_extra = derived;
            None
        } else if let Some(&existing) = map.get(&(base, bias)) {
            release_extra = derived;
            Some(existing)
        } else if map.len() >= MAX_DEPTH_BIAS_PSOS {
            if once(&N, 1) {
                crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – derived PSO cache full ({MAX_DEPTH_BIAS_PSOS})");
            }
            release_extra = derived;
            None
        } else {
            map.insert((base, bias), derived);
            Some(derived)
        }
    };
    if release_extra != 0 {
        super::pso_async::release(release_extra);
    }
    result
}

/// Apply one CreateDerivedPipelineStateX patch to a desc: part 0 = depth-stencil state, 1 =
/// rasterizer state, 2 = blend state, 3 = sample mask. Ok(false): part 5 (seen on Fortnite,
/// 0x30 bytes, meaning unknown) – not applied. Err: an unknown part.
fn apply_patch(d: &mut [u8; GFX_DESC], patch: &[u8; PATCH]) -> Result<bool, u32> {
    let (at, n) = match rd32(patch, 0) {
        0 => (0x1F0, 0x34),
        1 => (0x1C4, 0x2C),
        2 => (0x78, 0x148),
        3 => (0x1C0, 4),
        5 => return Ok(false),
        t => return Err(t),
    };
    d[at..at + n].copy_from_slice(&patch[4..4 + n]);
    Ok(true)
}

/// device #55 CreateDerivedPipelineStateX(base, patch count, patches, riid, pipeline): the
/// base pipeline's description with the patched parts, created as a new pipeline (itself a
/// base for further ones)
pub(super) unsafe extern "system" fn create_derived_pso_x(this: *mut c_void, base: *mut c_void, count: u32, patches: *const u8, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static N: AtomicU64 = AtomicU64::new(0);
    static UNMODELLED: AtomicU64 = AtomicU64::new(0);
    if ppv.is_null() {
        return E_INVALIDARG;
    }
    *ppv = std::ptr::null_mut();
    let found = BASES.lock().unwrap().as_ref().and_then(|m| m.get(&(base as usize)).cloned());
    let Some(b) = found else {
        if once(&N, 4) {
            crate::xlog_warn!("D3D12.X: CreateDerivedPipelineStateX – base pipeline {base:?} unknown (its creation failed, or it has stream output)");
        }
        return E_FAIL;
    };
    if stage_hashes(&b.desc.0) != Some(b.shaders) {
        if once(&N, 4) {
            crate::xlog_warn!("D3D12.X: CreateDerivedPipelineStateX – the shaders of base pipeline {base:?} are gone");
        }
        return E_FAIL;
    }
    let n = count as usize;
    if n > 64 || (n > 0 && (patches.is_null() || !crate::gpu::shader::readable(patches as usize, n * PATCH))) {
        return E_INVALIDARG;
    }
    let mut d = b.desc;
    for i in 0..n {
        let patch = &*(patches.add(i * PATCH) as *const [u8; PATCH]);
        match apply_patch(&mut d.0, patch) {
            Ok(true) => {}
            Ok(false) => {
                if once(&UNMODELLED, 1) {
                    crate::xlog_info!("D3D12.X: CreateDerivedPipelineStateX – patch part 5 not modelled (ignored)");
                }
            }
            Err(t) => {
                if once(&N, 4) {
                    crate::xlog_warn!("D3D12.X: CreateDerivedPipelineStateX – unknown patch part {t}");
                }
                return E_FAIL;
            }
        }
    }
    let desc = d.0.as_ptr() as *const c_void;
    create_graphics_pso_tracked(this, desc, riid, ppv)
}

/// device #86 CreatePipelineState(stream desc, riid, pipeline)
pub(super) unsafe extern "system" fn create_pipeline_state_x(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    real::dev_create_pipeline_state(this, desc, riid, ppv)
}

/// device #56 / #66 SerializePipelineStateX: not available in a release title
pub(super) unsafe extern "system" fn serialize_pso_x(_this: *mut c_void, _pso: *mut c_void) -> HRESULT {
    E_ACCESSDENIED
}

/// device #57 / #67 Deserialize{Graphics,Compute}PipelineStateX(state, riid, pipeline): the
/// serialised form is the console driver's; the title creates the pipeline from its desc
pub(super) unsafe extern "system" fn deserialize_pso_x(_this: *mut c_void, _state: *const u8, _riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static N: AtomicU64 = AtomicU64::new(0);
    if once(&N, 1) {
        crate::xlog_warn!("D3D12.X: DeserializePipelineStateX – console pipeline caches are not supported");
    }
    if !ppv.is_null() {
        *ppv = std::ptr::null_mut();
    }
    E_NOTIMPL
}

/// device #52 SetDebugFlagsX(flags)
pub(super) unsafe extern "system" fn set_debug_flags_x(_this: *mut c_void, _flags: u32) -> HRESULT {
    S_OK
}

static PAGE_POOLS: Mutex<Vec<(u64, u32)>> = Mutex::new(Vec::new());

/// device #74 RegisterPagePoolX(address, pages, index out)
pub(super) unsafe extern "system" fn register_page_pool_x(_this: *mut c_void, va: u64, pages: u32, out: *mut u32) -> HRESULT {
    if va == 0 || pages == 0 {
        return E_INVALIDARG;
    }
    let mut p = PAGE_POOLS.lock().unwrap();
    let i = p.iter().position(|&e| e == (va, pages)).unwrap_or_else(|| {
        p.push((va, pages));
        p.len() - 1
    });
    if !out.is_null() {
        *out = i as u32;
    }
    S_OK
}

/// device #75 UnregisterPagePoolX(index)
pub(super) unsafe extern "system" fn unregister_page_pool_x(_this: *mut c_void, index: u64) -> HRESULT {
    match PAGE_POOLS.lock().unwrap().get_mut(index as usize) {
        Some(e) if e.1 != 0 => {
            *e = (0, 0);
            S_OK
        }
        _ => E_FAIL,
    }
}

/// device #61 CreateSamplerX(desc, destination): the title uses it for anisotropic samplers;
/// the console desc (0x6C bytes) carries the PC fields at other offsets (+0x18 address U / V /
/// W, +0x24 / +0x28 min / max LOD, +0x2C bias, +0x34 max anisotropy, +0x40 comparison); its
/// further anisotropy fields are not modelled
pub(super) unsafe extern "system" fn create_sampler_x(this: *mut c_void, x: *const u8, dest: u64) {
    if x.is_null() || !crate::gpu::shader::readable(x as usize, 0x6C) {
        return;
    }
    let u = |o: usize| (x.add(o) as *const u32).read_unaligned();
    let d = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_ANISOTROPIC,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE(u(0x18) as i32),
        AddressV: D3D12_TEXTURE_ADDRESS_MODE(u(0x1C) as i32),
        AddressW: D3D12_TEXTURE_ADDRESS_MODE(u(0x20) as i32),
        MipLODBias: f32::from_bits(u(0x2C)),
        MaxAnisotropy: u(0x34).clamp(1, 16),
        ComparisonFunc: D3D12_COMPARISON_FUNC(u(0x40) as i32),
        BorderColor: [0.0; 4],
        MinLOD: f32::from_bits(u(0x24)),
        MaxLOD: f32::from_bits(u(0x28)),
    };
    real::dev_create_sampler_x(this, &d, dest);
}

/// the console resource desc (0x38 bytes, D3D12_RESOURCE_DESC) as a PC one: its tile layouts
/// and console flag bits mean nothing to a PC driver
pub fn pc_resource_desc(x: &[u8; 0x38]) -> [u8; 0x38] {
    let mut d = *x;
    let dim = u32::from_le_bytes(d[0..4].try_into().unwrap());
    let mut flags = u32::from_le_bytes(d[0x30..0x34].try_into().unwrap());
    let format = u32::from_le_bytes(d[0x20..0x24].try_into().unwrap());
    // the console's D16_UNORM_S8_UINT family (118 – 120): the PC's D24S8 one, typeless so its
    // depth and stencil views both fit
    if matches!(format, 118..=120) {
        d[0x20..0x24].copy_from_slice(&44u32.to_le_bytes());
    }
    // a depth buffer in a typed depth format: typeless, as the console's shaders read it too
    // (R32_FLOAT of a D32_FLOAT buffer – on the PC only a typeless resource has both views)
    if flags & 2 != 0 {
        if let Some(t) = depth_typeless(format) {
            d[0x20..0x24].copy_from_slice(&t.to_le_bytes());
        }
    }
    // the console's R10G10B10_7E3_A2_FLOAT / _6E4_A2_FLOAT: the PC has none – half floats hold them
    if matches!(format, 116 | 117) {
        d[0x20..0x24].copy_from_slice(&10u32.to_le_bytes());
    }
    // B5G5R5A1 / B4G4R4A4 targets: not ones a PC driver takes for rendering together with
    // unordered access (a retail game's 2560x1440 one, flags 0x5: E_INVALIDARG) – R8G8B8A8, the
    // same values with more bits; the GPU's own (nothing of the CPU's is laid out for them)
    if matches!(format, 86 | 115) && flags & 5 != 0 {
        d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
    }
    // a depth buffer the console also lets shaders write (a retail game's: ALLOW_DEPTH_STENCIL |
    // ALLOW_UNORDERED_ACCESS): the PC refuses the pair – the depth buffer stays
    if flags & 2 != 0 {
        flags &= !4;
    }
    d[8..16].fill(0); // alignment: the driver's
    let layout = if dim == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32 { D3D12_TEXTURE_LAYOUT_ROW_MAJOR.0 } else { D3D12_TEXTURE_LAYOUT_UNKNOWN.0 };
    d[0x2C..0x30].copy_from_slice(&(layout as u32).to_le_bytes());
    d[0x30..0x34].copy_from_slice(&(flags & 0xFF).to_le_bytes());
    d
}

/// The PC format of a view, pipeline or clear value in the console's D16_UNORM_S8_UINT family:
/// D24_UNORM_S8_UINT, R24_UNORM_X8_TYPELESS, X24_TYPELESS_G8_UINT (`pc_resource_desc` makes such
/// resources R24G8_TYPELESS).
pub fn pc_view_format(f: u32) -> u32 {
    match f {
        118 => 45,
        119 => 46,
        120 => 47,
        116 | 117 => 10,
        f => f,
    }
}

/// The typeless format of a depth format's family (None: not a depth format).
fn depth_typeless(f: u32) -> Option<u32> {
    match f {
        20 => Some(19), // D32_FLOAT_S8X24_UINT
        40 => Some(39), // D32_FLOAT
        45 => Some(44), // D24_UNORM_S8_UINT
        55 => Some(53), // D16_UNORM
        _ => None,
    }
}

/// The format of a render target / depth view or pipeline on the PC: `pc_view_format`, and a
/// B5G5R5A1 / B4G4R4A4 target is R8G8B8A8 (`pc_resource_desc`).
pub fn pc_target_format(f: u32) -> u32 {
    match f {
        86 | 115 => 28,
        f => pc_view_format(f),
    }
}

/// The format a shader's view of a resource whose PC format is `resource` has on the PC (`view`
/// 0: the resource's own). The console names depth buffers' views plainly (R16_UNORM of a D16S8
/// buffer, R32_FLOAT of a D32 one, even the depth format itself); the PC wants its family's
/// readable formats and removes the device for anything else (0x887A0001).
pub fn pc_shader_view_format(resource: u32, view: u32) -> u32 {
    let view = pc_view_format(view);
    match (resource, view) {
        // R32G8X24_TYPELESS: depth 21, stencil 22
        (19..=22, 0 | 19 | 20 | 21 | 39 | 40 | 41) => 21,
        (19..=22, 22 | 61 | 62) => 22,
        // R24G8_TYPELESS: depth 46, stencil 47
        (44..=47, 0 | 41 | 44 | 45 | 46 | 53 | 54 | 55 | 56) => 46,
        (44..=47, 47 | 61 | 62) => 47,
        // R32_TYPELESS, R16_TYPELESS
        (39 | 40, 0 | 39 | 40) => 41,
        (53 | 55, 0 | 53 | 55) => 56,
        // a target made R8G8B8A8 (`pc_resource_desc`)
        (28, 86 | 115) => 28,
        (_, v) => v,
    }
}

/// Whether a shader view in `pc_shader_view_format`'s format reads the stencil (plane 1 on the PC).
pub fn stencil_view(resource: u32, view: u32) -> bool {
    matches!((resource, view), (19..=22, 22) | (44..=47, 47))
}

/// A clear value (D3D12_CLEAR_VALUE: format first) with `pc_view_format`, copied into `buf`.
pub(super) unsafe fn pc_clear(clear: *const c_void, buf: &mut [u8; 24]) -> *const c_void {
    if clear.is_null() || !crate::gpu::shader::readable(clear as usize, 24) {
        return clear;
    }
    std::ptr::copy_nonoverlapping(clear as *const u8, buf.as_mut_ptr(), 24);
    let f = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    buf[0..4].copy_from_slice(&pc_target_format(f).to_le_bytes());
    buf.as_ptr() as *const c_void
}

/// device #25 GetResourceAllocationInfo(visible mask, count, descs) → {size, alignment},
/// returned through a hidden pointer. Called the C way – (return buffer, device, …), XDefiant –
/// or the C++ way – (device, return buffer, …): whichever argument is the device is `this`.
/// Textures get the console's size (XG: what the title then places them in); buffers and
/// what XG does not model the PC's.
pub(super) unsafe extern "system" fn get_resource_allocation_info_x(a: *mut c_void, b: *mut c_void, mask: u32, n: u32, descs: *const [u8; 0x38]) -> *mut [u64; 2] {
    let (this, ret) = if real::this_first(a, b) { (a, b as *mut [u64; 2]) } else { (b, a as *mut [u64; 2]) };
    type F = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
    let pc: F = std::mem::transmute(real::orig_fn(this, 25));
    let (mut size, mut align) = (0u64, 0u64);
    for k in 0..n as usize {
        let x = if descs.is_null() { [0u8; 0x38] } else { descs.add(k).read_unaligned() };
        let (s, a) = match crate::hle::xg::Texture::new(&x).ok().filter(|_| u32::from_le_bytes(x[0..4].try_into().unwrap()) != D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32) {
            Some(t) => (t.size(), t.alignment()),
            None => {
                let d = pc_resource_desc(&x);
                let mut r = [0u64; 2];
                pc(this, &mut r, mask, 1, &d);
                (r[0], r[1])
            }
        };
        if s == u64::MAX || a == 0 {
            size = u64::MAX;
            break;
        }
        size = size.div_ceil(a) * a + s;
        align = align.max(a);
    }
    if !ret.is_null() {
        *ret = [size, align];
    }
    ret
}

/// device #108 (a retail game) GetResourceAllocationInfo2(visible mask, count, D3D12_RESOURCE_DESC1s,
/// D3D12_RESOURCE_ALLOCATION_INFO1* each) -> {size, alignment} of them all, returned like #25's;
/// each resource's {offset, alignment, size} goes to the last argument when there is one.
pub(super) unsafe extern "system" fn get_resource_allocation_info2_x(a: *mut c_void, b: *mut c_void, mask: u32, n: u32, descs: *const [u8; 0x48], infos: *mut [u64; 3]) -> *mut [u64; 2] {
    real::note_newer_device_method(108);
    let ret = if real::this_first(a, b) { b } else { a } as *mut [u64; 2];
    let (mut size, mut align) = (0u64, 0u64);
    for k in 0..n as usize {
        let mut r = [0u64; 2];
        let d = if descs.is_null() { std::ptr::null() } else { descs.add(k) as *const [u8; 0x38] };
        let (x, y) = if real::this_first(a, b) { (a, &mut r as *mut _ as *mut c_void) } else { (&mut r as *mut _ as *mut c_void, b) };
        get_resource_allocation_info_x(x, y, mask, 1, d);
        if r[0] == u64::MAX || r[1] == 0 {
            size = u64::MAX;
            break;
        }
        let offset = size.next_multiple_of(r[1]);
        if !infos.is_null() {
            *infos.add(k) = [offset, r[1], r[0]];
        }
        size = offset + r[0];
        align = align.max(r[1]);
    }
    if !ret.is_null() {
        *ret = [size, align];
    }
    ret
}

/// device #26 GetCustomHeapProperties(node mask, heap type) -> D3D12_HEAP_PROPERTIES (20 bytes),
/// returned through a hidden pointer - like #25 called the C way (a retail game: return buffer, device,
/// ...) or the C++ way.
pub(super) unsafe extern "system" fn get_custom_heap_properties_x(a: *mut c_void, b: *mut c_void, mask: u32, ty: u32) -> *mut [u32; 5] {
    let (this, ret) = if real::this_first(a, b) { (a, b as *mut [u32; 5]) } else { (b, a as *mut [u32; 5]) };
    type F = unsafe extern "system" fn(*mut c_void, *mut [u32; 5], u32, u32) -> *mut [u32; 5];
    let pc: F = std::mem::transmute(real::orig_fn(this, 26));
    let mut r = [0u32; 5];
    pc(this, &mut r, mask, ty);
    if !ret.is_null() {
        *ret = r;
    }
    ret
}

/// Resources the title placed at an address an earlier one was placed at: on the console they
/// share that memory (aliasing – transient render targets reuse it). Logged (first 16).
fn note_alias(va: usize, x: &[u8; 0x38]) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashMap<usize, String>>> = std::sync::Mutex::new(None);
    static N: AtomicU64 = AtomicU64::new(0);
    let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
    let what = format!("dimension {} {}x{} format {} flags {:#x}", u(0), u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x30));
    let mut g = SEEN.lock().unwrap();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    if let Some(prev) = m.insert(va, what.clone()) {
        if N.fetch_add(1, Ordering::Relaxed) < 16 {
            crate::xlog_info!("D3D12.X: CreatePlacedResourceX at {va:#x} ({what}) – aliases an earlier resource there ({prev})");
        }
    }
}

/// device #44 CreatePlacedResourceX(address, desc, initial state, clear value, riid, resource):
/// a resource in title memory. A buffer at a 64 KiB-aligned address in unified memory is that
/// memory (placed in its UMA heap); a texture gets a PC texture kept in step with the memory
/// (`title_textures`: detiled as the console lays it out, uploaded again after CPU writes).
pub(super) unsafe extern "system" fn create_placed_resource_x(this: *mut c_void, va: u64, desc: *const u8, state: u32, clear: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static N: AtomicU64 = AtomicU64::new(0);
    if desc.is_null() || ppv.is_null() || !crate::gpu::shader::readable(desc as usize, 0x38) {
        return E_INVALIDARG;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let d = pc_resource_desc(&x);
    let buffer = u32::from_le_bytes(d[0..4].try_into().unwrap()) == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
    note_alias(va as usize, &x);
    // console-only state bits (Unicorn Overlord's 4K display target: 0x100000) are not the
    // PC's: the PC refused the resource (E_INVALIDARG) and the title threw
    let mut state = state & real::PC_RESOURCE_STATES;
    // (a depth buffer placed for unordered access – no longer one on the PC: written as depth)
    if u32::from_le_bytes(x[0x30..0x34].try_into().unwrap()) & 2 != 0 && state & 0x8 != 0 {
        state = (state & !0x8) | 0x10;
    }
    let mut clear_buf = [0u8; 24];
    let clear = pc_clear(clear, &mut clear_buf);
    if buffer {
        if let Some(hr) = real::place_at_title_address(this, va as usize, &d, state, clear, riid, ppv) {
            return hr;
        }
    }
    // render targets, depth buffers, UAV textures: in video memory, aliased by address (`vram`);
    // sampled textures: on the same shared 64 KiB pages when tiled resources allow, else committed
    let flags = u32::from_le_bytes(d[0x30..0x34].try_into().unwrap());
    let mut placed_hr = None;
    if !buffer {
        let pc = std::ptr::read_unaligned(d.as_ptr() as *const D3D12_RESOURCE_DESC);
        let console_bytes = crate::hle::xg::Texture::new(&x).ok().map(|t| t.size()).filter(|&s| s != u64::MAX && s != 0);
        if super::vram::gpu_owned(flags) {
            if let Some(hr) = super::vram::place(this, va as usize, &pc, state, clear, console_bytes, riid, ppv) {
                if hr >= 0 && !(*ppv).is_null() {
                    if let Some(bytes) = console_bytes {
                        super::title_textures::retire_overlapping(va as usize, bytes as usize);
                    }
                    real::note_title_address(*ppv as usize, va as usize);
                }
                return hr;
            }
        } else {
            placed_hr = super::vram::place_pages(this, va as usize, &pc, state, clear, console_bytes, riid, ppv);
        }
    }
    let hr = match placed_hr {
        Some(hr) => hr,
        None => {
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            real::dev_create_committed_resource(this, &heap as *const _ as usize, 0, d.as_ptr() as usize, state as usize, clear as usize, riid, ppv)
        }
    };
    if hr >= 0 && !buffer && !(*ppv).is_null() {
        static M: AtomicU64 = AtomicU64::new(0);
        match super::title_textures::register(this, *ppv, va as usize, &x, D3D12_RESOURCE_STATES(state as i32)) {
            Ok(()) => {
                real::note_title_address(*ppv as usize, va as usize);
                return hr;
            }
            Err(e) if once(&M, 4) => {
                let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
                crate::xlog_warn!(
                    "D3D12.X: CreatePlacedResourceX texture at {va:#x} (dimension {} {}x{}x{}, {} mips, format {}, {} samples, layout {:#x}, flags {:#x}): not kept in step with the memory ({e})",
                    u(0), u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u16::from_le_bytes(x[0x1C..0x1E].try_into().unwrap()),
                    u16::from_le_bytes(x[0x1E..0x20].try_into().unwrap()), u(0x20), u(0x24), u(0x2C), u(0x30)
                )
            }
            Err(_) => {}
        }
    }
    if hr >= 0 && !(*ppv).is_null() {
        real::note_title_address(*ppv as usize, va as usize);
    }
    if hr < 0 || once(&N, 4) {
        let w = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        crate::xlog_info!(
            "D3D12.X: CreatePlacedResourceX at {va:#x}: dimension {} {}x{} format {} flags {:#x} – a resource of its own ({hr:#x})",
            w(0),
            u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()),
            w(0x18),
            w(0x20),
            w(0x30)
        );
    }
    hr
}

/// device #133 (a retail game): CreatePlacedResourceX with a barrier layout (D3D12_BARRIER_LAYOUT, seen:
/// 0xFFFFFFFF = undefined for a buffer) in the initial state's place.
pub(super) unsafe extern "system" fn create_placed_resource_layout_x(this: *mut c_void, va: u64, desc: *const u8, layout: u32, clear: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    real::note_newer_device_method(133);
    create_placed_resource_x(this, va, desc, layout_state(layout), clear, riid, ppv)
}

/// The resource state (D3D12_RESOURCE_STATES) of a D3D12_BARRIER_LAYOUT (undefined: COMMON).
pub(super) fn layout_state(layout: u32) -> u32 {
    match layout {
        1 => 0xAC3, // generic read
        2 => 0x4,   // render target
        3 => 0x8,   // unordered access
        4 => 0x10,  // depth write
        5 => 0x20,  // depth read
        6 => 0xC0,  // shader resource (all stages)
        7 => 0x800, // copy source
        8 => 0x400, // copy dest
        9 => 0x2000, // resolve source
        10 => 0x1000, // resolve dest
        _ => 0,
    }
}

/// device #38 GetCopyableFootprints: a console resource desc the PC rejects (total stays
/// UINT64_MAX) is asked again as the PC desc XCL creates for it
#[allow(clippy::too_many_arguments)]
pub(super) unsafe extern "system" fn get_copyable_footprints_x(
    this: *mut c_void,
    desc: *const u8,
    first: u32,
    num: u32,
    base: u64,
    layouts: *mut c_void,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    type F = unsafe extern "system" fn(*mut c_void, *const u8, u32, u32, u64, *mut c_void, *mut u32, *mut u64, *mut u64);
    let f: F = std::mem::transmute(real::orig_fn(this, 38));
    if desc.is_null() || !crate::gpu::shader::readable(desc as usize, 0x38) {
        return f(this, desc, first, num, base, layouts, rows, row_sizes, total);
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let mut probe = u64::MAX;
    let t = if total.is_null() { &mut probe as *mut u64 } else { total };
    *t = u64::MAX;
    f(this, x.as_ptr(), first, num, base, layouts, rows, row_sizes, t);
    if *t == u64::MAX {
        let d = pc_resource_desc(&x);
        f(this, d.as_ptr(), first, num, base, layouts, rows, row_sizes, total);
    }
}

/// device #41 CreateCommandSignature(desc, root signature, riid, signature): the console's
/// DispatchX argument (105) is a PC Dispatch; the root signature only goes with arguments
/// that need one (constants, CBV / SRV / UAV)
pub(super) unsafe extern "system" fn create_command_signature_x(this: *mut c_void, desc: *const u8, root: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if desc.is_null() || ppv.is_null() {
        return E_INVALIDARG;
    }
    let mut pc = [0u8; 24];
    std::ptr::copy_nonoverlapping(desc, pc.as_mut_ptr(), 24);
    let n = u32::from_le_bytes(pc[4..8].try_into().unwrap()) as usize;
    let args = u64::from_le_bytes(pc[8..16].try_into().unwrap()) as *const [u32; 4];
    let mut list: Vec<[u32; 4]> = if args.is_null() || n > 64 { Vec::new() } else { (0..n).map(|i| args.add(i).read_unaligned()).collect() };
    for a in list.iter_mut() {
        if a[0] == XBOX_ARG_DISPATCH_X {
            a[0] = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.0 as u32;
        }
    }
    let needs_root = list.iter().any(|a| (5..=8).contains(&a[0]));
    pc[8..16].copy_from_slice(&(if list.is_empty() { 0 } else { list.as_ptr() as u64 }).to_le_bytes());
    let hr = real::dev_create_command_signature(this, pc.as_ptr() as usize, if needs_root { root } else { 0 }, riid, ppv);
    if hr >= 0 && !ppv.is_null() && needs_root {
        super::bind::note_root_changing_signature(*ppv as usize);
    }
    hr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_types_and_descs_become_pc_ones() {
        assert_eq!(pc_list_type(XBOX_TYPE_DMA), D3D12_COMMAND_LIST_TYPE_COPY.0 as u32);
        assert_eq!(pc_list_type(2), 2);
        assert_eq!(pc_queue_desc([XBOX_TYPE_DMA, 1, 0, 0x10]), [3, 100, 0, 1]);
        assert_eq!(pc_queue_desc([0, 10000, 1, 0]), [0, 10000, 1, 0]);
        let mut x = [0u8; 0x38];
        x[0] = 1; // buffer
        x[8] = 0x40; // alignment
        x[0x2C] = 0x22; // console layout
        x[0x30..0x34].copy_from_slice(&0x1234u32.to_le_bytes());
        let d = pc_resource_desc(&x);
        assert_eq!((d[8], u32::from_le_bytes(d[0x2C..0x30].try_into().unwrap()), u32::from_le_bytes(d[0x30..0x34].try_into().unwrap())), (0, 1, 0x34));
        x[0] = 3; // texture 2D
        assert_eq!(u32::from_le_bytes(pc_resource_desc(&x)[0x2C..0x30].try_into().unwrap()), 0);
    }

    /// Console depth buffers become PC ones a shader can view: typeless, the D16S8 family as
    /// D24S8, no UAV flag; views get the family's readable formats (stencil from plane 1).
    #[test]
    fn console_depth_buffers_and_their_views_are_the_pcs() {
        let mut x = [0u8; 0x38];
        x[0] = 3; // texture 2D
        let desc = |x: &[u8; 0x38]| {
            let d = pc_resource_desc(x);
            (u32::from_le_bytes(d[0x20..0x24].try_into().unwrap()), u32::from_le_bytes(d[0x30..0x34].try_into().unwrap()))
        };
        x[0x20] = 118; // D16_UNORM_S8_UINT, depth + UAV + console bits
        x[0x30..0x34].copy_from_slice(&0x8030_0006u32.to_le_bytes());
        assert_eq!(desc(&x), (44, 0x2));
        x[0x20] = 40; // D32_FLOAT depth buffer: typeless
        x[0x30..0x34].copy_from_slice(&0x2u32.to_le_bytes());
        assert_eq!(desc(&x), (39, 0x2));
        x[0x30..0x34].copy_from_slice(&0x0u32.to_le_bytes()); // not a depth buffer: as it is
        assert_eq!(desc(&x), (40, 0));
        x[0x20] = 86; // B5G5R5A1 target that is also a UAV
        x[0x30..0x34].copy_from_slice(&0x5u32.to_le_bytes());
        assert_eq!(desc(&x), (28, 0x5));
        // views: a D24S8 format or R16_UNORM of the D16S8 buffer read depth; stencil formats plane 1
        assert_eq!(pc_shader_view_format(44, 45), 46);
        assert_eq!(pc_shader_view_format(44, 56), 46);
        assert_eq!(pc_shader_view_format(44, 118), 46);
        assert_eq!(pc_shader_view_format(44, 120), 47);
        assert!(stencil_view(44, 47) && !stencil_view(44, 46));
        assert_eq!(pc_shader_view_format(39, 0), 41);
        assert_eq!(pc_shader_view_format(19, 62), 22);
        assert_eq!(pc_shader_view_format(28, 86), 28);
        assert_eq!(pc_shader_view_format(86, 86), 86, "a B5G5R5A1 texture that stayed one");
        assert_eq!((pc_target_format(86), pc_target_format(118), pc_view_format(117)), (28, 45, 10));
    }

    #[test]
    fn derived_pipeline_patches_land_on_the_desc_parts() {
        use std::mem::{offset_of, size_of};
        type G = D3D12_GRAPHICS_PIPELINE_STATE_DESC;
        assert_eq!(size_of::<G>(), GFX_DESC);
        assert_eq!(STAGES, [offset_of!(G, VS), offset_of!(G, PS), offset_of!(G, DS), offset_of!(G, HS), offset_of!(G, GS)]);
        assert_eq!(SO_ENTRIES, offset_of!(G, StreamOutput) + offset_of!(D3D12_STREAM_OUTPUT_DESC, NumEntries));
        assert_eq!((INPUT_LAYOUT, CACHED_PSO), (offset_of!(G, InputLayout), offset_of!(G, CachedPSO)));
        let parts = [
            (0, offset_of!(G, DepthStencilState), size_of::<D3D12_DEPTH_STENCIL_DESC>()),
            (1, offset_of!(G, RasterizerState), size_of::<D3D12_RASTERIZER_DESC>()),
            (2, offset_of!(G, BlendState), size_of::<D3D12_BLEND_DESC>()),
            (3, offset_of!(G, SampleMask), 4),
        ];
        for (tag, at, n) in parts {
            let mut d = [0u8; GFX_DESC];
            let mut patch = [0xAAu8; PATCH];
            patch[0..4].copy_from_slice(&(tag as u32).to_le_bytes());
            assert_eq!(apply_patch(&mut d, &patch), Ok(true));
            assert!(d[at..at + n].iter().all(|&b| b == 0xAA), "part {tag}");
            assert_eq!(d.iter().filter(|&&b| b == 0xAA).count(), n, "part {tag} writes only its {n} bytes");
        }
        let mut d = [0u8; GFX_DESC];
        let mut patch = [0u8; PATCH];
        patch[0] = 5;
        assert_eq!(apply_patch(&mut d, &patch), Ok(false));
        patch[0] = 4;
        assert_eq!(apply_patch(&mut d, &patch), Err(4));
    }

    #[test]
    fn depth_bias_patches_rasterizer_fields() {
        use std::mem::offset_of;
        type G = D3D12_GRAPHICS_PIPELINE_STATE_DESC;
        let rast = offset_of!(G, RasterizerState);
        assert_eq!(RASTERIZER_DEPTH_BIAS, rast + offset_of!(D3D12_RASTERIZER_DESC, DepthBias));
        assert_eq!(RASTERIZER_DEPTH_BIAS_CLAMP, rast + offset_of!(D3D12_RASTERIZER_DESC, DepthBiasClamp));
        assert_eq!(RASTERIZER_SLOPE_SCALED_DEPTH_BIAS, rast + offset_of!(D3D12_RASTERIZER_DESC, SlopeScaledDepthBias));
        assert_eq!(PSO_FLAGS, offset_of!(G, Flags));
        assert_eq!(FLAG_DYNAMIC_DEPTH_BIAS, 4);

        // -0.0 canonicalizes to +0.0 so zero checks and cache keys agree
        let z = DepthBias::new(-0.0, 0.0, -0.0);
        assert!(z.is_zero());
        assert_eq!(z, DepthBias::new(0.0, 0.0, 0.0));

        let b = DepthBias::new(-125.6, 0.05, 1.5);
        assert!(!b.is_zero());
        assert_eq!(b.depth_bias(), -125.6);
        assert_eq!(b.depth_bias_clamp(), 0.05);
        assert_eq!(b.slope_scaled_depth_bias(), 1.5);
        let mut d = [0x55u8; GFX_DESC];
        b.patch_desc(&mut d);
        assert_eq!(i32::from_le_bytes(d[RASTERIZER_DEPTH_BIAS..RASTERIZER_DEPTH_BIAS + 4].try_into().unwrap()), -126);
        assert_eq!(f32::from_le_bytes(d[RASTERIZER_DEPTH_BIAS_CLAMP..RASTERIZER_DEPTH_BIAS_CLAMP + 4].try_into().unwrap()), 0.05);
        assert_eq!(f32::from_le_bytes(d[RASTERIZER_SLOPE_SCALED_DEPTH_BIAS..RASTERIZER_SLOPE_SCALED_DEPTH_BIAS + 4].try_into().unwrap()), 1.5);
        // from_desc recovers the baked rasterizer fields
        assert_eq!(DepthBias::from_desc(&d), DepthBias::new(-126.0, 0.05, 1.5));
        // untouched bytes before and after the 12-byte depth-bias slice
        assert!(d[..RASTERIZER_DEPTH_BIAS].iter().all(|&x| x == 0x55));
        assert!(d[RASTERIZER_SLOPE_SCALED_DEPTH_BIAS + 4..].iter().all(|&x| x == 0x55));

        // enable / clear D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS touches only bit 2 of Flags
        let mut f = [0u8; GFX_DESC];
        f[PSO_FLAGS..PSO_FLAGS + 4].copy_from_slice(&1u32.to_le_bytes());
        enable_dynamic_depth_bias(&mut f);
        assert_eq!(rd32(&f, PSO_FLAGS), 1 | FLAG_DYNAMIC_DEPTH_BIAS);
        clear_dynamic_depth_bias(&mut f);
        assert_eq!(rd32(&f, PSO_FLAGS), 1);
    }

    #[test]
    fn page_pools_register_once_and_unregister() {
        unsafe {
            let mut i = u32::MAX;
            assert_eq!(register_page_pool_x(std::ptr::null_mut(), 0x1_0000_0000, 16, &mut i), S_OK);
            let mut j = u32::MAX;
            assert_eq!(register_page_pool_x(std::ptr::null_mut(), 0x1_0000_0000, 16, &mut j), S_OK);
            assert_eq!(i, j);
            assert_eq!(register_page_pool_x(std::ptr::null_mut(), 0, 16, &mut j), E_INVALIDARG);
            assert_eq!(unregister_page_pool_x(std::ptr::null_mut(), i as u64), S_OK);
            assert_eq!(unregister_page_pool_x(std::ptr::null_mut(), i as u64), E_FAIL);
        }
    }
}
