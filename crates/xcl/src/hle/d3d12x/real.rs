//! Real D3D12.X backend: the title gets genuine D3D12 objects from the PC driver.
//!
//! Every object handed to the title is "adopted": its vtable pointer is replaced with
//! an extended copy:
//!
//! ```text
//! ext[-1]                 original vtable (for calling originals)
//! ext[0 .. public)        original PC methods (a few overridden, see below)
//! ext[public .. internal) guard thunks: runtime/DXGI calls → original internal virtual,
//!                         calls from the game image → logged as console-only method
//! ext[internal .. +64)    logging stubs (console-only methods past the end)
//! ```
//!
//! Overrides: QueryInterface (logs unknown IIDs), every Create* on the device (adopts
//! the new object, retries with the PC IID when the console one is unknown, captures
//! shaders, builds fallback pipeline states when console shaders are rejected),
//! CreateRenderTargetView / OMSetRenderTargets / ExecuteCommandLists (presentation).

#![allow(non_snake_case)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::Serialize;
use windows::core::{Interface, PCSTR};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows_sys::core::GUID;

use super::layout::{self, Iface};
use crate::stubs;
use crate::util::{HRESULT, E_INVALIDARG, E_NOINTERFACE, E_OUTOFMEMORY, S_FALSE, S_OK};

const DLL: &str = "d3d12_xs.dll";
const EXTRA_SLOTS: usize = 64;
const MAX_SCAN: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    Device,
    CommandQueue,
    CommandAllocator,
    CommandList,
    /// the console's DMA (copy) list, ID3D12XboxDmaCommandList: a PC list with the console's
    /// shorter vtable (`console_remap`)
    DmaList,
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
}

impl Kind {
    pub fn base(self) -> &'static str {
        match self {
            Kind::Device => "ID3D12Device",
            Kind::CommandQueue => "ID3D12CommandQueue",
            Kind::CommandAllocator => "ID3D12CommandAllocator",
            Kind::CommandList | Kind::DmaList => "ID3D12GraphicsCommandList",
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
        }
    }
}

// ---------------------------------------------------------------------------------------
// statistics for the report
// ---------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize)]
pub struct ClassStat {
    pub interface: String,
    pub public_methods: usize,
    pub internal_slots: usize,
}

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize)]
pub struct D3d12Stats {
    pub backend: String,
    pub adapter: String,
    pub classes: Vec<ClassStat>,
    pub unknown_iids: Vec<String>,
    pub fallback_pipelines: u64,
    pub failed_pipelines: u64,
    pub fallback_root_signatures: u64,
    pub presents: u64,
    pub frame_boundaries: u64,
    /// Optional content evidence from the existing frame-dump readback; no new GPU readback.
    #[serde(default)]
    pub last_frame_dump: Option<super::present::FrameDumpSample>,
    /// unified memory region opened as a GPU heap (`gpu::uma`): size, bytes in use, buffer VA
    #[serde(default)]
    pub uma_mb: u64,
    #[serde(default)]
    pub uma_used_kb: u64,
    #[serde(default)]
    pub uma_gpu_va: String,
    #[serde(default)]
    pub uma_error: String,
    /// whether the GPU sees CPU writes to the coherent windows (`check_uma_import`)
    #[serde(default)]
    pub uma_import_check: String,
    /// graphics pipelines built from recompiled console shaders (XCL shader ABI root signature)
    #[serde(default)]
    pub translated_pipelines: u64,
    /// why console shaders could not be used (first messages)
    #[serde(default)]
    pub translation_errors: Vec<String>,
    /// pipelines built in the background behind a stand-in: built, failed, still building
    #[serde(default)]
    pub background_pipelines: (usize, usize, usize),
    /// DXIL cache and compiler statistics.
    #[serde(default)]
    pub dxil_cache: crate::gpu::recomp::dxc::CacheStats,
}

static ADAPTER: Mutex<String> = Mutex::new(String::new());
static UNKNOWN_IIDS: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
static FALLBACK_PSO: AtomicU64 = AtomicU64::new(0);
static FAILED_PSO: AtomicU64 = AtomicU64::new(0);
static FALLBACK_RS: AtomicU64 = AtomicU64::new(0);
static TRANSLATED_PSO: AtomicU64 = AtomicU64::new(0);
static TRANSLATE_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// XCL shader ABI root signature on the backend device (raw, kept for the process lifetime)
static ABI_RS: AtomicUsize = AtomicUsize::new(0);

/// XCL's shader ABI root signature, once a translated pipeline created it.
pub fn abi_rs_cached() -> Option<*mut c_void> {
    match ABI_RS.load(Ordering::Acquire) {
        0 => None,
        p => Some(p as *mut c_void),
    }
}

thread_local! {
    /// the last compute pipeline created on this thread runs a translated console shader
    static LAST_COMPUTE_TRANSLATED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// … with this root signature (static samplers; 0 = the plain ABI one)
    static LAST_COMPUTE_RS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Root signature of the pipelines built from translated console shaders (bind it, then the
/// XCL ABI root UAVs io / res / mem, to draw with them).
pub fn abi_root_signature() -> Option<ID3D12RootSignature> {
    let raw = ABI_RS.load(Ordering::Acquire) as *mut c_void;
    if raw.is_null() {
        return None;
    }
    unsafe { ID3D12RootSignature::from_raw_borrowed(&raw).cloned() }
}
/// Each UMA window as one D3D12 buffer (placed in a heap opened over the process memory):
/// `mem`, `mem1`.. of recompiled shaders. (buffer, GPU VA) per window, kept for the process
/// lifetime.
static UMA_BUFFER: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
/// the committed UMA windows opened as D3D12 heaps (indexed by window `k`; `None` for sparse windows)
static UMA_HEAPS: Mutex<Vec<Option<usize>>> = Mutex::new(Vec::new());
/// console buffers placed in unified memory: resource → (address, bytes)
/// address, bytes, whether XCL allocated the memory (not: the title's own, CreatePlacedResourceX)
type UmaBuffer = (usize, usize, bool);
static UMA_BUFFERS: Mutex<Option<HashMap<usize, UmaBuffer>>> = Mutex::new(None);

/// Where a console-created buffer lives in unified memory (its console GPU address).
pub fn console_buffer(res: usize) -> Option<(usize, usize)> {
    UMA_BUFFERS.lock().unwrap().as_ref()?.get(&res).map(|&(p, n, _)| (p, n))
}

/// Number of live console buffers in unified memory.
pub fn console_buffers() -> usize {
    UMA_BUFFERS.lock().unwrap().as_ref().map_or(0, |m| m.len())
}
static UMA_ERROR: Mutex<String> = Mutex::new(String::new());

/// GPU virtual address of the unified memory buffer of window 0 (0: not available).
pub fn uma_gpu_va() -> u64 {
    uma_window_vas().first().copied().unwrap_or(0)
}

/// GPU virtual addresses of the UMA window buffers (`u2`, `u3`.. of the shader ABI).
pub fn uma_window_vas() -> Vec<u64> {
    UMA_BUFFER.lock().map(|g| g.iter().map(|b| b.1).collect()).unwrap_or_default()
}

/// The window buffer holding `len` bytes at UMA address `p`, and their offset in it.
pub(super) fn uma_window_buffer(p: usize, len: usize) -> Option<(usize, u64)> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(p)?;
    let (base, size) = u.window(k);
    if p - base + len > size {
        return None;
    }
    Some((UMA_BUFFER.lock().ok()?.get(k)?.0, (p - base) as u64))
}

/// GPU virtual address of a UMA address (None: not in UMA or not open on the GPU).
pub fn uma_va_of(p: usize) -> Option<u64> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(p)?;
    Some(*uma_window_vas().get(k)? + (p - u.window(k).0) as u64)
}

/// What the start-up check of the coherent UMA windows found (`check_uma_import`): empty – not
/// run, "ok …", "off (XCL_UMA_CHECK=0)", or why the GPU does not see the CPU's writes.
static UMA_IMPORT_CHECK: Mutex<String> = Mutex::new(String::new());

/// A word the CPU writes into each coherent window, read back by a GPU copy. The windows are
/// imported as heaps (`OpenExistingHeapFromAddress`), and a runtime that cannot import them may
/// still succeed with memory of its own: vkd3d-proton under a Wine without `userfaultfd` write
/// watches (#175). The GPU then reads zeros where the title wrote, and nothing else says so.
/// Off with `XCL_UMA_CHECK=0`.
unsafe fn check_uma_import(device: &ID3D12Device, u: &crate::gpu::uma::Uma, bufs: &[(usize, u64)]) -> Result<String, String> {
    let t0 = std::time::Instant::now();
    let wins: Vec<usize> = (0..bufs.len()).filter(|&k| u.is_committed_window(k) && bufs[k].0 != 0).collect();
    if wins.is_empty() {
        return Ok("no coherent window".into());
    }
    let queue: ID3D12CommandQueue =
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| format!("check not run: queue: {e}"))?;
    let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| format!("check not run: allocator: {e}"))?;
    let list: ID3D12GraphicsCommandList =
        device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| format!("check not run: list: {e}"))?;
    let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| format!("check not run: fence: {e}"))?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut readback: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback)
        .map_err(|e| format!("check not run: readback buffer: {e}"))?;
    let readback = readback.ok_or("check not run: readback buffer")?;
    // a block of the window's own (title data is not touched), one word per window
    let mut probes = Vec::new();
    for (i, &k) in wins.iter().take(64).enumerate() {
        let Some(p) = u.alloc_in(k, 4, 0) else { continue };
        let canary = 0x5843_4C00 | k as u32; // "XCL" + the window
        std::ptr::write_volatile(p as *mut u32, canary);
        if let Some(src) = ID3D12Resource::from_raw_borrowed(&(bufs[k].0 as *mut c_void)) {
            list.CopyBufferRegion(&readback, 4 * i as u64, src, (p as usize - u.window(k).0) as u64, 4);
        }
        probes.push((k, i, p as usize, canary));
    }
    let run = || -> Result<Vec<u32>, String> {
        list.Close().map_err(|e| format!("check not run: Close: {e}"))?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(|e| e.to_string())?)]);
        queue.Signal(&fence, 1).map_err(|e| format!("check not run: Signal: {e}"))?;
        crate::gpu::fence::wait(&fence, 1, device, None, 2000).map_err(|e| format!("check not run: {e}"))?;
        let mut m = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut m)).map_err(|e| format!("check not run: Map: {e}"))?;
        let got = std::slice::from_raw_parts(m as *const u32, 64).to_vec();
        readback.Unmap(0, None);
        Ok(got)
    };
    let got = match run() {
        Ok(got) => got,
        Err(e) => {
            // The copy may still be running: what it uses stays (a few KiB, once). Released under
            // the GPU, vkd3d-proton frees it at once and the GPU faults (RADV: page faults, then a
            // ring timeout) – a slow first use of the imported window is enough to get here.
            std::mem::forget((queue, alloc, list, fence, readback));
            return Err(e);
        }
    };
    for &(_, _, p, _) in &probes {
        u.free(p);
    }
    let wrong: Vec<String> =
        probes.iter().filter(|&&(_, i, _, c)| got[i] != c).map(|&(k, i, _, c)| format!("window {k}: wrote {c:#x}, the GPU read {:#x}", got[i])).collect();
    if !wrong.is_empty() {
        return Err(format!(
            "the GPU does not see what the CPU writes to unified memory ({}): the host-memory import fell back to memory of its own (Wine without userfaultfd write watches?); shaders will read zeros or stale data where the title wrote",
            wrong.join("; ")
        ));
    }
    Ok(format!("ok, {} window(s), {} µs", probes.len(), t0.elapsed().as_micros()))
}

unsafe fn open_uma(device: &ID3D12Device) {
    let Some(u) = crate::gpu::uma::get() else {
        *UMA_ERROR.lock().unwrap() = "no UMA region".into();
        return;
    };
    let open = |k: usize| -> Result<(usize, u64, usize), String> {
        let (base, size) = u.window(k);
        let dev3: ID3D12Device3 = device.cast().map_err(|e| format!("ID3D12Device3: {e}"))?;
        let heap: ID3D12Heap = dev3.OpenExistingHeapFromAddress(base as *const c_void).map_err(|e| format!("OpenExistingHeapFromAddress (window {k}): {e}"))?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        device.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("CreatePlacedResource: {e}"))?;
        let buf = buf.ok_or("CreatePlacedResource")?;
        let va = buf.GetGPUVirtualAddress();
        Ok((buf.into_raw() as usize, va, heap.into_raw() as usize))
    };
    let n = u.shader_windows();
    let mut bufs: Vec<(usize, u64)> = vec![(0, 0); n];
    let mut heaps: Vec<Option<usize>> = vec![None; n];
    for k in 0..n {
        if !u.is_committed_window(k) {
            continue;
        }
        match open(k) {
            Ok((buf_raw, va, heap_raw)) => {
                bufs[k] = (buf_raw, va);
                heaps[k] = Some(heap_raw);
            }
            Err(e) => {
                crate::xlog_warn!("D3D12.X: UMA not available to the GPU: {e}");
                *UMA_ERROR.lock().unwrap() = e;
                return;
            }
        }
    }
    let check = if std::env::var("XCL_UMA_CHECK").is_ok_and(|v| v == "0") {
        Ok("off (XCL_UMA_CHECK=0)".into())
    } else {
        check_uma_import(device, u, &bufs)
    };
    match &check {
        Ok(s) => crate::xlog_info!("D3D12.X: UMA import check: {s}"),
        Err(e) => crate::xlog_error!("D3D12.X: UMA import check: {e}"),
    }
    *UMA_IMPORT_CHECK.lock().unwrap() = check.unwrap_or_else(|e| e);
    let sparse = super::sparse::init(device);
    for (k, buf_raw, va) in &sparse {
        if let Some(slot) = bufs.get_mut(*k) {
            *slot = (*buf_raw, *va);
        }
    }
    let w0_va = bufs.first().map_or(0, |b| b.1);
    crate::xlog_info!(
        "D3D12.X: UMA opened ({} sparse VRAM window(s), {} MB coherent window), window 0 buffer at VA {:#x}",
        sparse.len(),
        u.size() >> 20,
        w0_va
    );
    *UMA_HEAPS.lock().unwrap() = heaps;
    // shaders take the windows from the shared descriptor heap (`pso::heap_windows`)
    if n > 1 && crate::gpu::recomp::pso::heap_windows() {
        let views: Vec<(usize, usize)> = bufs.iter().enumerate().map(|(k, b)| (b.0, u.window(k).1)).collect();
        if super::console_heap::set_window_views(device, &views) {
            crate::xlog_info!("D3D12.X: memory windows from the shared descriptor heap (no per-access switch)");
        }
    }
    *UMA_BUFFER.lock().unwrap() = bufs;
}

pub fn stats() -> D3d12Stats {
    let classes = CLASSES
        .lock()
        .map(|g| {
            g.iter()
                .map(|c| ClassStat { interface: c.iface.name.to_string(), public_methods: c.public, internal_slots: c.scanned - c.public })
                .collect()
        })
        .unwrap_or_default();
    let (presents, boundaries) = super::present::counters();
    D3d12Stats {
        backend: format!("{:?}", super::backend()).to_lowercase(),
        adapter: ADAPTER.lock().map(|a| a.clone()).unwrap_or_default(),
        classes,
        unknown_iids: UNKNOWN_IIDS.lock().map(|s| s.iter().cloned().collect()).unwrap_or_default(),
        fallback_pipelines: FALLBACK_PSO.load(Ordering::Relaxed),
        failed_pipelines: FAILED_PSO.load(Ordering::Relaxed),
        fallback_root_signatures: FALLBACK_RS.load(Ordering::Relaxed),
        presents,
        frame_boundaries: boundaries,
        last_frame_dump: super::present::last_frame_dump(),
        uma_mb: crate::gpu::uma::get().map(|u| (u.size() >> 20) as u64).unwrap_or(0),
        uma_used_kb: crate::gpu::uma::get().map(|u| (u.used() >> 10) as u64).unwrap_or(0),
        uma_gpu_va: match uma_gpu_va() {
            0 => String::new(),
            v => format!("{v:#x}"),
        },
        uma_error: UMA_ERROR.lock().map(|e| e.clone()).unwrap_or_default(),
        uma_import_check: UMA_IMPORT_CHECK.lock().map(|e| e.clone()).unwrap_or_default(),
        translated_pipelines: TRANSLATED_PSO.load(Ordering::Relaxed),
        translation_errors: TRANSLATE_ERRORS.lock().map(|e| e.clone()).unwrap_or_default(),
        background_pipelines: super::pso_async::counts(),
        dxil_cache: crate::gpu::recomp::dxc::cache_stats(),
    }
}

// ---------------------------------------------------------------------------------------
// device creation
// ---------------------------------------------------------------------------------------

static DEVICE: AtomicUsize = AtomicUsize::new(0);

pub fn device_raw() -> *mut c_void {
    DEVICE.load(Ordering::Acquire) as *mut c_void
}

/// The process's video memory on the device's adapter: (local, non-local) bytes in use.
pub fn video_memory() -> Option<(u64, u64)> {
    use windows::Win32::Graphics::Dxgi::*;
    let raw = device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?;
    unsafe {
        let luid = dev.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(luid).ok()?;
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local).ok()?;
        adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut non_local).ok()?;
        Some((local.CurrentUsage, non_local.CurrentUsage))
    }
}

/// The device runs only 32-lane waves: wave64 compute, mesh and geometry programs need the
/// split form (`Options::split_wave64`). Unknown device: assume so.
fn wave32_only(device: *mut c_void) -> bool {
    if device.is_null() {
        return true;
    }
    unsafe {
        let Some(dev) = ID3D12Device::from_raw_borrowed(&device) else { return true };
        crate::gpu::recomp::pso::wave32_only_for_device(dev)
    }
}

/// A wave64 VS / PS may run in 32-lane waves here (`pso::graphics_wave32`: the driver picks
/// the wave size of graphics shaders): the split flag for VS / PS translation. Unknown
/// device: assume so.
fn graphics_wave32(device: *mut c_void) -> bool {
    if device.is_null() {
        return true;
    }
    unsafe {
        let Some(dev) = ID3D12Device::from_raw_borrowed(&device) else { return true };
        crate::gpu::recomp::pso::graphics_wave32_for_device(dev)
    }
}

/// Create the process-wide D3D12 device (the adapter `d3d12_adapter` / `XCL_ADAPTER` chooses,
/// by default the high-performance GPU → WARP); once – later calls
/// return the existing device's adapter.
pub fn init() -> Result<String, String> {
    if !device_raw().is_null() {
        return Ok(ADAPTER.lock().map(|a| a.clone()).unwrap_or_default());
    }
    unsafe {
        let cfg = crate::state::config();
        match crate::gpu::agility::state() {
            crate::gpu::agility::State::Loaded { version, dir, .. } => {
                crate::xlog_info!("D3D12: Agility SDK v{version} ({})", dir.display());
            }
            crate::gpu::agility::State::Unavailable(reason) => {
                crate::xlog_warn!("D3D12: Agility SDK unavailable ({reason}) – using inbox D3D12");
            }
            crate::gpu::agility::State::NotFound => {}
        }
        let debug = cfg.d3d12_debug || std::env::var_os("XCL_D3D12_DEBUG").is_some();
        if debug {
            match crate::gpu::agility::configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug) {
                Ok(d) => {
                    d.EnableDebugLayer();
                    crate::xlog_info!("D3D12: debug layer enabled");
                }
                Err(e) => {
                    crate::xlog_warn!("D3D12: debug layer requested but unavailable: {e}");
                }
            }
        }
        // `XCL_DRED=1`: on a device removal the faulting address and the last GPU operations
        // (logged by `log_device_removed`); no debug layer needed. Opt-in: with it XDefiant
        // stalled in its first seconds twice.
        if std::env::var_os("XCL_DRED").is_some() {
            match crate::gpu::agility::configuration_interface::<ID3D12DeviceRemovedExtendedDataSettings>(
                &CLSID_D3D12DeviceRemovedExtendedData,
            ) {
                Ok(d) => {
                    d.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    d.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                }
                Err(e) => {
                    crate::xlog_warn!("D3D12: DRED requested but unavailable: {e}");
                }
            }
        }
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI: {e}"))?;
        let mut device: Option<ID3D12Device> = None;
        let mut name = String::new();
        let sel = crate::gpu::adapter::Selector::from_env_or(&cfg.d3d12_adapter);
        if sel != crate::gpu::adapter::Selector::Warp {
            let mut list = crate::gpu::adapter::candidates(&factory, &sel);
            if list.is_empty() {
                crate::xlog_warn!("D3D12: {} – using auto", crate::gpu::adapter::no_match(&factory, &sel));
                list = crate::gpu::adapter::candidates(&factory, &crate::gpu::adapter::Selector::Auto);
            }
            for (adapter, desc) in list {
                if let Ok(dev) = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0) {
                    device = Some(dev);
                    name = format!("{} ({} MiB VRAM)", crate::gpu::adapter::name(&desc), desc.DedicatedVideoMemory >> 20);
                    break;
                }
            }
        }
        let device = match device {
            Some(d) => d,
            None => {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().map_err(|e| format!("WARP: {e}"))?;
                let d = crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_11_0)
                    .map_err(|e| format!("D3D12 device (WARP): {e}"))?;
                name = "WARP (software renderer)".into();
                d
            }
        };
        if debug {
            // the debug layer's messages into XCL's log (they otherwise go to a debugger only)
            if let Ok(q) = device.cast::<ID3D12InfoQueue1>() {
                let mut cookie = 0u32;
                let _ = q.RegisterMessageCallback(Some(debug_message), D3D12_MESSAGE_CALLBACK_FLAG_NONE, std::ptr::null_mut(), &mut cookie);
            }
        }
        open_uma(&device);
        super::device_x::probe_dynamic_depth_bias(&device);
        let raw = device.into_raw(); // one reference kept for the process lifetime
        adopt(raw, Kind::Device);
        DEVICE.store(raw as usize, Ordering::Release);
        *ADAPTER.lock().unwrap() = name.clone();
        Ok(name)
    }
}

/// A debug layer message: errors and warnings into the log – the first 8 of each message id
/// (3 000 in all). One cap for all of them (200) was used up in seconds by two per-frame state
/// errors of a retail game, long before the message that came with its device removal.
unsafe extern "system" fn debug_message(_category: D3D12_MESSAGE_CATEGORY, severity: D3D12_MESSAGE_SEVERITY, id: D3D12_MESSAGE_ID, text: windows::core::PCSTR, _context: *mut c_void) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static PER_ID: Mutex<Option<HashMap<i32, u32>>> = Mutex::new(None);
    if severity.0 > D3D12_MESSAGE_SEVERITY_WARNING.0 {
        return;
    }
    {
        let mut g = PER_ID.lock().unwrap();
        let n = g.get_or_insert_with(HashMap::new).entry(id.0).or_insert(0);
        *n += 1;
        if *n > 8 {
            return;
        }
    }
    if N.fetch_add(1, Ordering::Relaxed) >= 3000 {
        return;
    }
    let t = if text.is_null() { String::new() } else { String::from_utf8_lossy(text.as_bytes()).into_owned() };
    crate::xlog_warn!("D3D12 debug layer ({}, id {}): {t}", if severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 { "error" } else { "warning" }, id.0);
}

type QiFn = unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT;
type RefFn = unsafe extern "system" fn(*mut c_void) -> u32;

pub unsafe fn create_device(riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return S_FALSE;
    }
    let dev = device_raw();
    if dev.is_null() {
        return E_INVALIDARG;
    }
    if !riid.is_null() {
        let qi: QiFn = std::mem::transmute(*vt(dev));
        let hr = qi(dev, riid as *const GUID, ppv);
        if hr >= 0 {
            crate::xlog_info!("D3D12.X: device {:p} for the title", *ppv);
            if std::env::var_os("XCL_WATCH_DEVICE").is_some() {
                crate::eh::watch_writes(*ppv as usize);
            }
            return hr;
        }
    }
    let add: RefFn = std::mem::transmute(*vt(dev).add(1));
    add(dev);
    *ppv = dev;
    S_OK
}

// ---------------------------------------------------------------------------------------
// vtable adoption
// ---------------------------------------------------------------------------------------

struct Class {
    orig: usize,
    kind: Kind,
    ext: usize,
    iface: &'static Iface,
    public: usize,
    scanned: usize,
}

static CLASSES: Mutex<Vec<Class>> = Mutex::new(Vec::new());
static EXTS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

unsafe fn vt(obj: *mut c_void) -> *const usize {
    *(obj as *const *const usize)
}

/// Original function of `slot` for an adopted object.
pub unsafe fn orig_fn(this: *mut c_void, slot: usize) -> usize {
    let ext = vt(this);
    let orig = *ext.sub(1) as *const usize;
    *orig.add(slot)
}

fn module_of(addr: usize) -> usize {
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };
    let mut h = std::ptr::null_mut();
    unsafe {
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, addr as *const u16, &mut h);
    }
    h as usize
}

/// Whether `obj` is an object XCL adopted (its vtable is one of XCL's).
pub(super) unsafe fn is_adopted(obj: *mut c_void) -> bool {
    !obj.is_null() && crate::gpu::shader::readable(obj as usize, 8) && is_ext(*(obj as *const usize))
}

/// A struct-returning method's first two arguments: true when `a` is the object (the C++ way:
/// this, return buffer, …), false when `b` is (the C way: return buffer, this, …). A return
/// buffer on the stack can hold a stale copy of an object's vtable pointer and look adopted
/// itself: a retail game's call of GetResourceAllocationInfo the C way then had its result
/// written over the device's vtable pointer (the next call read slot 25 of "vtable" 0x1000). So
/// when both look adopted, the one on this thread's stack – or the one that is not the device –
/// is the buffer.
pub(super) unsafe fn this_first(a: *mut c_void, b: *mut c_void) -> bool {
    match (is_adopted(a), is_adopted(b)) {
        (true, false) => true,
        (false, true) => false,
        (true, true) => {
            let (mut lo, mut hi) = (0usize, 0usize);
            windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut lo, &mut hi);
            let on_stack = |p: *mut c_void| (lo..hi).contains(&(p as usize));
            let dev = device_raw();
            if on_stack(a) != on_stack(b) {
                on_stack(b)
            } else if a == dev || b == dev {
                a == dev
            } else {
                true
            }
        }
        (false, false) => false,
    }
}

fn is_ext(v: usize) -> bool {
    EXTS.lock().map(|g| g.as_ref().map(|s| s.contains(&v)).unwrap_or(false)).unwrap_or(false)
}

fn kind_of(ext: usize) -> Option<Kind> {
    CLASSES.lock().unwrap().iter().find(|c| c.ext == ext).map(|c| c.kind)
}

/// If a native D3D12 method (such as `Close` or `Reset` on Agility SDK 1.620+, or copy-list
/// `Reset`) replaced `this`'s vtable pointer with another native runtime table, adopt the
/// new table under the same `Kind` so XCL's hooks stay installed while forwarding to the
/// runtime's current state table.
pub(super) unsafe fn preserve_vtable(this: *mut c_void, prev_ext: usize) {
    let cur = *(this as *const usize);
    if cur == prev_ext || is_ext(cur) {
        return;
    }
    if let Some(kind) = kind_of(prev_ext) {
        adopt(this, kind);
    } else {
        *(this as *mut usize) = prev_ext;
    }
}

/// An adopted object gets the extended vtable of another kind (a DMA list: created as a PC
/// list, it needs the console's DMA layout).
pub(super) unsafe fn readopt(obj: *mut c_void, kind: Kind) {
    if obj.is_null() || !is_ext(vt(obj) as usize) {
        return adopt(obj, kind);
    }
    let orig = *vt(obj).sub(1);
    *(obj as *mut usize) = class_for(obj, orig, kind);
}

/// Give a genuine D3D12 object the extended vtable for `kind`.
pub unsafe fn adopt(obj: *mut c_void, kind: Kind) {
    if obj.is_null() {
        return;
    }
    let v = vt(obj) as usize;
    if is_ext(v) {
        return;
    }
    let ext = class_for(obj, v, kind);
    *(obj as *mut usize) = ext;
}

unsafe fn class_for(obj: *mut c_void, orig: usize, kind: Kind) -> usize {
    if let Some(c) = CLASSES.lock().unwrap().iter().find(|c| c.orig == orig && c.kind == kind) {
        return c.ext;
    }
    let ot = orig as *const usize;
    // most derived PC interface this object implements
    let chain = layout::chain(kind.base());
    let qi: QiFn = std::mem::transmute(*ot);
    let release: RefFn = std::mem::transmute(*ot.add(2));
    let mut iface = *chain.last().expect("layout chain");
    for i in &chain {
        let iid = layout::parse_guid(i.iid);
        let mut out = std::ptr::null_mut();
        if qi(obj, &iid, &mut out) >= 0 && !out.is_null() {
            release(out);
            iface = i;
            break;
        }
    }
    let public = iface.methods.len();
    let m0 = module_of(*ot);
    let mut scanned = 0;
    while scanned < public + MAX_SCAN && module_of(*ot.add(scanned)) == m0 {
        scanned += 1;
    }
    let scanned = scanned.max(public);
    let max_console = console_slots(kind)
        .iter()
        .map(|e| e.0 + 1)
        .chain(console_remap(kind).iter().map(|e| e.0 + 1))
        .max()
        .unwrap_or(0);
    let total = (scanned + EXTRA_SLOTS).max(max_console);
    let mut v: Vec<usize> = Vec::with_capacity(total + 1);
    v.push(orig);
    for i in 0..scanned {
        v.push(*ot.add(i));
    }
    let hook = if kind == Kind::CommandQueue { super::present::on_queue_console_method as usize } else { 0 };
    for i in public..scanned {
        let info = stubs::new_info(DLL, &format!("{}::#{i} (not on PC)", iface.name), 0);
        info.hook.store(hook, Ordering::Release);
        v[1 + i] = super::guard::make_guard(v[1 + i], info);
    }
    for i in scanned..total {
        let (addr, info) = stubs::make_stub_info(DLL, &format!("{}::#{i} (not on PC)", iface.name), 0);
        info.hook.store(hook, Ordering::Release);
        v.push(addr);
    }
    for (method, f) in overrides(kind) {
        if let Some(s) = layout::slot(iface, method) {
            if s < public {
                v[1 + s] = f;
            }
        }
    }
    // the console's layout of the same PC methods (title calls only; the runtime keeps PC's)
    let pc = v.clone();
    for (x, p) in console_remap(kind) {
        if x < public {
            v[1 + x] = super::guard::make_split(pc[1 + p], pc[1 + x]);
        } else if x < scanned {
            v[1 + x] = super::guard::make_split(pc[1 + p], *ot.add(x));
        } else if x < total {
            v[1 + x] = pc[1 + p];
        }
    }
    // slots where the console's method differs from the PC one at the same index
    for (slot, f) in console_slots(kind) {
        if slot < public {
            v[1 + slot] = super::guard::make_split(f, v[1 + slot]);
        } else if slot < scanned {
            // an internal method of the runtime there (e.g. queue #30 = PresentX): the title
            // gets the console method, the runtime keeps its own
            v[1 + slot] = super::guard::make_split(f, *ot.add(slot));
        } else if slot < total {
            v[1 + slot] = f;
        }
    }
    // the other console extensions (device, command list): a title call is logged and answered
    // S_OK, not run as the PC method at that index
    if let Some(first) = console_extensions(kind) {
        let own: HashSet<usize> = console_slots(kind).iter().map(|e| e.0).chain(console_remap(kind).iter().map(|e| e.0)).collect();
        for i in (first..public).filter(|i| !own.contains(i)) {
            let (stub, _) = stubs::make_stub_info(DLL, &format!("{}::#{i} (console extension)", iface.name), 0);
            v[1 + i] = super::guard::make_split(stub, v[1 + i]);
        }
    }
    // methods the newer console layout keeps at other slots: picked at each call
    if kind == Kind::CommandList {
        let older_handlers = console_slots(kind);
        for (newer, older) in NEWER_LIST_SLOTS {
            let Some(&(_, f)) = older_handlers.iter().find(|e| e.0 == older) else { continue };
            if newer >= total {
                continue;
            }
            let pc = if newer < scanned { *ot.add(newer) } else { v[1 + newer] };
            v[1 + newer] = super::guard::make_layout_split(super::guard::LayoutTargets { older: v[1 + newer], newer: f, pc, newer_layout: &QUEUE_SHIFTED });
        }
    }
    // XCL_D3D12_TRACE: every public method called from console code is logged
    if std::env::var_os("XCL_D3D12_TRACE").is_some() {
        for i in 0..public {
            let name = format!("{}::{} (#{i})", iface.name, iface.methods.get(i).copied().unwrap_or("?"));
            v[1 + i] = super::guard::make_tracer(v[1 + i], name);
        }
    }
    let leaked: &'static [usize] = Box::leak(v.into_boxed_slice());
    let ext = leaked.as_ptr() as usize + 8;
    EXTS.lock().unwrap().get_or_insert_with(HashSet::new).insert(ext);
    crate::xlog_info!(
        "D3D12.X: {} – {public} PC methods, {} internal slots under the guard, +{EXTRA_SLOTS} stubs ({kind:?}, PC vtable {})",
        iface.name,
        scanned - public,
        crate::modules::describe_address(orig)
    );
    CLASSES.lock().unwrap().push(Class { orig, kind, ext, iface, public, scanned });
    ext
}

/// (console slot, PC slot) of PC methods that sit elsewhere in the console's vtable.
/// ID3D12GraphicsCommandList (from the Fortnite RHI call sites, matching XCL's own evidence
/// – FSR4 calls ClearUnorderedAccessViewUint at #53): 0–27 and 56–59 as on PC; the query
/// methods come right after ExecuteBundle (28–31 = PC 52–55: BeginQuery, EndQuery,
/// ResolveQueryData, SetPredication), so SetDescriptorHeaps … DiscardResource move up by 4
/// (32–55 = PC 28–51); ID3D12GraphicsCommandList1 is at 104–109 (PC 60–65), the shading-rate
/// methods at 124 / 125 (PC 77 / 78). The console-only methods in between: `list_x`.
fn console_remap(kind: Kind) -> Vec<(usize, usize)> {
    match kind {
        Kind::CommandList => (28..=31).map(|x| (x, x + 24)).chain((32..=55).map(|x| (x, x - 4))).chain((104..=109).map(|x| (x, x - 44))).chain([(124, 77), (125, 78)]).collect(),
        // the DMA list: Close / Reset, then the copy methods (XDefiant calls #12 with
        // CopyTextureRegion's arguments): 11–14 = PC 15–18 (CopyBufferRegion, CopyTextureRegion,
        // CopyResource, CopyTiles); what follows is not known yet (`console_extensions`)
        Kind::DmaList => (11..=14).map(|x| (x, x + 4)).collect(),
        _ => Vec::new(),
    }
}

/// console D3D12_DESCRIPTOR_HEAP_TYPE also used for CBV / SRV / UAV heaps
const XBOX_EXTRA_CBV_SRV_UAV: u32 = 4;

/// Map a console `D3D12_DESCRIPTOR_HEAP_TYPE` to its PC equivalent.
#[inline]
fn pc_heap_type(ty: u32) -> u32 {
    if ty == XBOX_EXTRA_CBV_SRV_UAV {
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32
    } else {
        ty
    }
}

/// Where a class's console extensions start: the device's from #44 (`device_x`), the command
/// list's from #60 (`list_x`).
fn console_extensions(kind: Kind) -> Option<usize> {
    match kind {
        Kind::Device => Some(44),
        Kind::CommandList => Some(60),
        Kind::DmaList => Some(15),
        _ => None,
    }
}

/// Console vtable slots that differ from the PC layout (found on console code calling them):
/// * device #62: `CreateComputePipelineState` with two extra arguments (XboxBvhBuilder.dll
///   passes a D3D12_COMPUTE_PIPELINE_STATE_DESC, NULL, NULL, riid, ppv); PC #62 = CreateStateObject.
/// * device #76: `CreateCommittedResource` (heap properties, flags, desc, state, clear value,
///   riid, ppv) as on PC #27 (PC #76 = CreateCommittedResource3); the console resource desc
///   carries more after the 52 PC bytes (amd_fidelityfx_upscaler_xbox.dll writes 3 at +52).
/// * descriptor heap #9 / #10: the heap start handles come back in rax on the console.
/// * resource #11 GetGPUVirtualAddress: console buffers live in unified memory; their console
///   address (= CPU address, low 32 bits = offset in `mem`) is what console code gets.
/// * device #14 / #15 / #17 / #18 / #19 / #22 / #23 / #24 (CreateDescriptorHeap,
///   GetDescriptorHandleIncrementSize, CreateConstantBufferView, CreateShaderResourceView,
///   CreateUnorderedAccessView, CreateSampler, CopyDescriptors, CopyDescriptorsSimple):
///   CBV/SRV/UAV and sampler heaps of console code get 32 / 16-byte slots in unified memory
///   (`console_heap`). #14–#19 seen from console code with the PC layout; #17 and #22–#24
///   assumed to follow it (each checks its arguments and otherwise goes to the PC method).
/// * device #12 `CreateCommandList`: console list types; with console lists on, console code
///   gets a console-shaped list (D3D12.X inline code writes into it), XCL's own code a PC one.
/// * the device's own methods (`device_x`, `frame`), the command list's (`list_x`), queue #30
///   PresentX (`present`).
fn console_slots(kind: Kind) -> Vec<(usize, usize)> {
    use super::device_x as x;
    match kind {
        Kind::Device => {
            vec![
                (0, dev_qi_x as usize),
                (8, x::create_command_queue_x as usize),
                (25, x::get_resource_allocation_info_x as usize),
                (26, x::get_custom_heap_properties_x as usize),
                (29, dev_create_placed_resource_x as usize),
                // CreateCommittedResource at its PC index too (XDefiant): console code uses the
                // buffer's GPU address as a CPU pointer – it goes into unified memory
                (27, dev_create_committed_resource_x as usize),
                (9, x::create_command_allocator_x as usize),
                (10, x::create_graphics_pso_pc as usize),
                (12, x::create_command_list_pc as usize),
                (14, dev_create_descriptor_heap_x as usize),
                (15, dev_increment_size_x as usize),
                (17, dev_create_cbv_x as usize),
                (18, dev_create_srv_x as usize),
                (21, dev_create_dsv_x as usize),
                (19, dev_create_uav_x as usize),
                (22, dev_create_sampler_x as usize),
                (23, dev_copy_descriptors_x as usize),
                (24, dev_copy_descriptors_simple_x as usize),
                (38, x::get_copyable_footprints_x as usize),
                (41, x::create_command_signature_x as usize),
                (44, x::create_placed_resource_x as usize),
                (48, x::create_command_queue_x as usize),
                (52, x::set_debug_flags_x as usize),
                (55, x::create_derived_pso_x as usize),
                (56, x::serialize_pso_x as usize),
                (57, x::deserialize_pso_x as usize),
                (60, x::create_graphics_pso_x as usize),
                (61, x::create_sampler_x as usize),
                (62, dev_create_compute_pso_x as usize),
                (66, x::serialize_pso_x as usize),
                (67, x::deserialize_pso_x as usize),
                (70, dev_create_srv_at_x as usize),
                (71, dev_create_uav_at_x as usize),
                (73, x::create_command_list_x as usize),
                (74, x::register_page_pool_x as usize),
                (75, x::unregister_page_pool_x as usize),
                (76, dev_create_committed_resource_x as usize),
                // frame pacing (SetFrameIntervalX, ScheduleFrameEventX, WaitFrameEventX,
                // GetFrameStatisticsX)
                (78, super::frame::set_frame_interval_x as usize),
                (79, super::frame::schedule_frame_event_x as usize),
                (80, super::frame::wait_frame_event_x as usize),
                (81, super::frame::frame_statistics_x as usize),
                (86, x::create_pipeline_state_x as usize),
                (108, x::get_resource_allocation_info2_x as usize),
                (127, dev_create_committed_resource_layout_x as usize),
                (133, x::create_placed_resource_layout_x as usize),
            ]
        }
        Kind::DescriptorHeap => vec![(9, heap_cpu_start_x as usize), (10, heap_gpu_start_x as usize)],
        Kind::Resource => vec![(11, res_gpu_va_x as usize)],
        Kind::Fence => vec![(12, super::fence_x::completed_value_address as usize)],
        // the title's calls from #14 on: Signal / Wait / … / SignalX / PresentX at the slots of
        // its runtime's layout (`queue_dispatch`)
        Kind::CommandQueue => QUEUE_SLOTS.iter().enumerate().map(|(k, &f)| (QUEUE_FIRST + k, f as usize)).collect(),
        Kind::DmaList => {
            use super::list_x as l;
            vec![(9, l::dma_close_x as usize), (10, l::reset_x as usize), (11, l::copy_buffer_region_x as usize), (12, l::copy_texture_region_x as usize), (13, l::copy_resource_x as usize)]
        }
        Kind::CommandList => {
            use super::bind as b;
            use super::list_x as l;
            vec![
                // compute root arguments of console code: kept for translated dispatches
                (12, b::draw_instanced as usize),
                (15, l::copy_buffer_region_x as usize),
                (16, l::copy_texture_region_x as usize),
                (17, l::copy_resource_x as usize),
                (19, l::resolve_subresource_x as usize),
                (51, l::clear_depth_stencil_x as usize),
                (108, l::resolve_subresource_region_x as usize),
                (13, b::draw_indexed_instanced as usize),
                (14, b::dispatch as usize),
                (48, l::ia_set_vertex_buffers_x as usize),
                // state a PC Reset clears, kept for KickoffX
                (20, l::ia_set_topology_x as usize),
                (21, l::rs_set_viewports_x as usize),
                (22, l::rs_set_scissors_x as usize),
                (23, l::om_set_blend_factor_x as usize),
                (24, l::om_set_stencil_ref_x as usize),
                (32, l::set_descriptor_heaps_x as usize),
                (59, b::execute_indirect as usize),
                (83, l::copy_texture_region_flags_x as usize),
                (34, b::set_graphics_root_signature as usize),
                (36, b::set_graphics_root_table as usize),
                (38, b::set_graphics_root_constant as usize),
                (40, b::set_graphics_root_constants as usize),
                (42, b::set_graphics_root_descriptor as usize),
                (44, b::set_graphics_root_descriptor as usize),
                (46, b::set_graphics_root_descriptor as usize),
                (26, l::resource_barrier_x as usize),
                (25, b::set_pipeline_state as usize),
                (33, b::set_compute_root_signature as usize),
                (35, b::set_compute_root_table as usize),
                (37, b::set_compute_root_constant as usize),
                (39, b::set_compute_root_constants as usize),
                (41, b::set_compute_root_descriptor as usize),
                (43, b::set_compute_root_descriptor as usize),
                (45, b::set_compute_root_descriptor as usize),
                (10, l::reset_x as usize),
                (28, l::begin_query_x as usize),
                (29, l::end_query_x as usize),
                (30, l::resolve_query_data_x as usize),
                (47, l::ia_set_index_buffer_x as usize),
                (60, l::copy_memory_x as usize),
                (61, l::fill32_x as usize),
                (62, l::fill64_x as usize),
                (63, l::fill128_x as usize),
                (68, l::set_depth_bounds_x as usize),
                (69, l::write32_bottom_x as usize),
                (70, l::write64_bottom_x as usize),
                (71, l::wait32_x as usize),
                (72, l::wait64_x as usize),
                (78, l::dispatch_x as usize),
                (81, l::draw_indexed_x as usize),
                (84, l::copy_resource_flags_x as usize),
                (85, l::set_predication_buffer_x as usize),
                (88, l::write32_top_x as usize),
                (89, l::write64_top_x as usize),
                (90, l::write32_bottom_of_shader_x as usize),
                (93, l::kickoff_x as usize),
                (94, l::set_depth_bias_x as usize),
                // console compute/BVH builder extensions (`XboxBvhBuilder.dll`)
                (146, l::uav_barrier_x as usize),
                (148, b::allocate_upload_memory as usize),
            ]
        }
        _ => vec![],
    }
}

/// descriptor heap #9 / #10 GetCPU / GetGPUDescriptorHandleForHeapStart: console code
/// takes the handle in rax (only `this` passed); the PC methods return it through a hidden
/// pointer
unsafe extern "system" fn heap_cpu_start_x(this: *mut c_void) -> u64 {
    if let Some(p) = super::console_heap::start(this as usize) {
        return p;
    }
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 = std::mem::transmute(orig_fn(this, 9));
    let mut h = 0u64;
    f(this, &mut h);
    h
}

unsafe extern "system" fn heap_gpu_start_x(this: *mut c_void) -> u64 {
    if let Some(p) = super::console_heap::start(this as usize) {
        return p;
    }
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 = std::mem::transmute(orig_fn(this, 10));
    let mut h = 0u64;
    f(this, &mut h);
    h
}

unsafe extern "system" fn dev_create_descriptor_heap_x(this: *mut c_void, desc: *const [u32; 4], riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut d = if desc.is_null() { [u32::MAX; 4] } else { *desc };
    d[0] = pc_heap_type(d[0]);
    let desc = if desc.is_null() { desc } else { &d as *const [u32; 4] };
    let sampler = d[0] == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32;
    if d[0] != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32 && !sampler {
        return dev_create_descriptor_heap(this, desc as usize, riid, ppv);
    }
    // views: an SRV / UAV pair per console slot (in the shared shader-visible heap, the
    // title's own a CPU-only mirror – `console_heap`); samplers: one per slot
    let pc_n = if sampler { d[1] } else { d[1].saturating_mul(2) };
    let visible = d[2] & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32 != 0;
    let shared = match ID3D12Device::from_raw_borrowed(&this) {
        Some(dev) if !sampler && !desc.is_null() => super::console_heap::shared_heap(dev),
        _ => false,
    };
    let flags = if shared { d[2] & !(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32) } else { d[2] };
    let pc = [d[0], pc_n, flags, d[3]];
    let hr = dev_create_descriptor_heap(this, pc.as_ptr() as usize, riid, ppv);
    if hr >= 0 {
        let raw = *ppv;
        let heap = ID3D12DescriptorHeap::from_raw_borrowed(&raw).unwrap();
        let dev = ID3D12Device::from_raw_borrowed(&this).unwrap();
        let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let gpu = if flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32 != 0 { heap.GetGPUDescriptorHandleForHeapStart().ptr } else { 0 };
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE(d[0] as i32));
        if sampler {
            // an S# of zeros – a slot nothing was written to – is a sampler on the console: point
            // filtering, wrap, mip 0 only. Every PC slot starts as that one, so a table entry the
            // title left empty samples as there (an unwritten PC descriptor is undefined).
            let zero = D3D12_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: 0.0,
            };
            for k in 0..d[1] as u64 {
                dev.CreateSampler(&zero, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + k * inc as u64) as usize });
            }
        }
        if !super::console_heap::register(raw as usize, d[1], cpu, gpu, inc, sampler, shared, visible) {
            crate::xlog_warn!("console descriptor heap: no room in UMA for {} slots", d[1]);
        }
    }
    hr
}

unsafe extern "system" fn dev_increment_size_x(this: *mut c_void, ty: u32) -> u32 {
    let ty = pc_heap_type(ty);
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32 {
        return super::console_heap::SLOT;
    }
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32 {
        return super::console_heap::SAMPLER_SLOT;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32) -> u32 = std::mem::transmute(orig_fn(this, 15));
    f(this, ty)
}

/// What a console view slot describes: (GFX10 format, image type, width, height, depth, mips)
/// from the view (or the resource when the view is NULL); None for buffers.
unsafe fn view_shape(res: &ID3D12Resource, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, dim: i32, array: u32, mips: u32) -> Option<(u32, crate::gpu::desc::ImageType, u32, u32, u32, u32)> {
    use crate::gpu::desc::ImageType as T;
    let r = res.GetDesc();
    let f = if format == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN { r.Format } else { format };
    let fmt = crate::gpu::desc::gfx10_format(f).unwrap_or(0);
    let mips = if mips == 0 || mips == u32::MAX { r.MipLevels as u32 } else { mips };
    let (w, h, d) = (r.Width as u32, r.Height, r.DepthOrArraySize as u32);
    let ty = match dim {
        2 => T::D1,
        3 => T::D1Array,
        4 => T::D2,
        5 => T::D2Array,
        6 => T::D2Msaa,
        7 => T::D2MsaaArray,
        8 => T::D3,
        9 | 10 => T::Cube,
        _ => return None,
    };
    let depth = match ty {
        T::D3 => d,
        T::D1Array | T::D2Array | T::D2MsaaArray => array.max(1),
        // a cube view: its faces (6 per cube)
        T::Cube => array.max(6),
        _ => 1,
    };
    Some((fmt, ty, w, h, depth, mips))
}

fn resource_dim_to_view(r: &D3D12_RESOURCE_DESC) -> i32 {
    match r.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            if r.DepthOrArraySize > 1 {
                3
            } else {
                2
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => match (r.SampleDesc.Count > 1, r.DepthOrArraySize > 1) {
            (true, true) => 7,
            (true, false) => 6,
            (false, true) => 5,
            (false, false) => 4,
        },
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => 8,
        _ => 1,
    }
}

static NON_UMA_BUFFER_VIEWS: AtomicU64 = AtomicU64::new(0);

/// V# of a buffer view (raw: 4-byte elements, num_records in bytes; structured / typed:
/// stride = element size, num_records in elements); empty when the buffer is not in UMA.
fn buffer_view(res: usize, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, first: u64, n: u32, stride: u32, raw: bool) -> [u32; 4] {
    let Some((at, bytes)) = console_buffer(res) else {
        note_buffer_view();
        return [0; 4];
    };
    let fmt = crate::gpu::desc::gfx10_format(format).unwrap_or(22);
    let (stride, elem) = if raw {
        (0, 4)
    } else if stride > 0 {
        (stride, stride)
    } else {
        let e = crate::gpu::desc::format_bytes(fmt);
        (e, e)
    };
    let base = at as u64 + first * elem as u64;
    let room = (bytes as u64).saturating_sub(first * elem as u64);
    let records = if stride == 0 { (n as u64 * 4).min(room) } else { (n as u64).min(room / stride as u64) };
    crate::gpu::desc::buffer_words(base, stride, records as u32, fmt)
}

/// Console buffer → a placed resource in the unified memory heap, so its console address is
/// what the recompiled shaders read through `mem`. None: not placed (fall back to committed).
/// A committed texture of console code: on the console its memory is unified memory too –
/// titles write it through its GPU address (XDefiant clears one with memset). It gets memory
/// of the console's size (XG; the PC's where XG has no model) and is created as a texture at
/// that address (`device_x::create_placed_resource_x`: kept in step with the memory where XG
/// lays it out); GetGPUVirtualAddress returns the address. The memory is the coherent
/// window's, as long as it keeps `COHERENT_KEPT` free; beyond that the texture is a plain PC
/// one (None). Unicorn Overlord's 2.2 GB of committed textures filled the 512 MB window in its
/// first second, leaving no room for readback heaps. (In the sparse windows instead, XDefiant
/// rendered black – not understood yet.)
/// bytes of the coherent window committed textures leave free: readback / write-back heaps
/// and buffers, XCL's blocks outside its arena pool
const COHERENT_KEPT: usize = 128 << 20;

unsafe fn place_console_texture(this: *mut c_void, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    if ppv.is_null() || !crate::gpu::shader::readable(desc, 0x38) {
        return None;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    if !(2..=4).contains(&u32::from_le_bytes(x[0..4].try_into().unwrap())) {
        return None;
    }
    let (size, align) = match crate::hle::xg::Texture::new(&x) {
        Ok(t) if t.size() > 0 => (t.size(), t.alignment()),
        _ => {
            type F = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
            let pc: F = std::mem::transmute(orig_fn(this, 25));
            let mut r = [0u64; 2];
            pc(this, &mut r, 0, 1, &super::device_x::pc_resource_desc(&x));
            (r[0], r[1])
        }
    };
    if size == 0 || size == u64::MAX {
        return None;
    }
    let align = (align as usize).max(0x1_0000);
    let bytes = (size as usize).div_ceil(align) * align;
    if u.size().saturating_sub(u.used()) < bytes + COHERENT_KEPT {
        return None;
    }
    let p = u.alloc(bytes, align)? as usize;
    let hr = super::device_x::create_placed_resource_x(this, p as u64, desc as *const u8, state as u32, clear as *const c_void, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        u.free(p);
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, size as usize, true));
    Some(hr)
}

/// Largest single `D3D12_HEAP_TYPE_READBACK` allocation kept on the coherent window; larger
/// ones spill to sparse VRAM so one big readback heap/buffer cannot exhaust coherent headroom.
const COHERENT_READBACK_MAX: usize = 64 << 20;
/// Largest single `D3D12_HEAP_TYPE_CUSTOM` + `WRITE_BACK` `COPY_DEST` buffer kept on coherent RAM.
const COHERENT_CUSTOM_WB_MAX: usize = 16 << 20;

/// bytes of the coherent window readback heaps and buffers leave free (XCL's blocks outside its
/// arena pool)
const COHERENT_FLOOR: usize = 16 << 20;

/// Whether a readback heap / buffer of `rounded` bytes may take coherent memory. The room
/// committed textures leave (`COHERENT_KEPT`) is kept for these: with the textures at that
/// limit (Unicorn Overlord's), a second 128 MB of headroom sent every readback buffer to the
/// sparse windows.
fn has_coherent_headroom(u: &crate::gpu::uma::Uma, rounded: usize) -> bool {
    u.size().saturating_sub(u.used()) >= COHERENT_FLOOR.min(u.size() / 8).saturating_add(rounded)
}

/// Sparse-backed buffers whose `Map(0, NULL, ...)` should read back their full range from VRAM
/// (readback buffers that spilled to sparse VRAM or `COPY_DEST` buffers in `CUSTOM + WRITE_BACK`).
static SPARSE_READBACK: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

fn note_sparse_readback(res: usize) {
    SPARSE_READBACK.lock().unwrap().get_or_insert_with(HashSet::new).insert(res);
}

/// Whether `props` asks for CPU-cached readback memory. On Xbox UMA, `D3D12_HEAP_TYPE_CUSTOM`
/// with `D3D12_CPU_PAGE_PROPERTY_WRITE_BACK` is the general CPU-cached graphics pool (used for
/// hundreds of MB of VBs/IBs/CBs/UAVs), so only explicit `D3D12_HEAP_TYPE_READBACK` heaps — or
/// small non-UAV `COPY_DEST` buffers — stay in the 512 MB coherent window.
fn is_coherent_readback_buffer(props: &D3D12_HEAP_PROPERTIES, state: u32, flags: D3D12_RESOURCE_FLAGS, bytes: usize) -> bool {
    if props.Type == D3D12_HEAP_TYPE_READBACK {
        return bytes <= COHERENT_READBACK_MAX;
    }
    props.Type == D3D12_HEAP_TYPE_CUSTOM
        && props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
        && state == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32
        && (flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0
        && bytes <= COHERENT_CUSTOM_WB_MAX
}

fn is_cpu_readback_buffer(props: &D3D12_HEAP_PROPERTIES, state: u32, flags: D3D12_RESOURCE_FLAGS) -> bool {
    props.Type == D3D12_HEAP_TYPE_READBACK
        || (props.Type == D3D12_HEAP_TYPE_CUSTOM
            && props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
            && state == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32
            && (flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0)
}

unsafe fn place_console_buffer_sparse(this: *mut c_void, d: &D3D12_RESOURCE_DESC, bytes: usize, state: u32, readback_on_null_map: bool, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let (p, _) = super::sparse::alloc(bytes, 0x1_0000)?;
    if let Some((hr, delta)) = super::sparse::place_buffer(this, p, d, state, riid, ppv) {
        UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, bytes, true));
        if delta != 0 {
            BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
        }
        if readback_on_null_map {
            note_sparse_readback(*ppv as usize);
        }
        return Some(hr);
    }
    super::sparse::free(p);
    None
}

/// Sanitize a console buffer description and initial state for placement in a coherent
/// cross-adapter UMA heap (`OpenExistingHeapFromAddress`).
fn sanitize_coherent_buffer(desc: &D3D12_RESOURCE_DESC, state: usize) -> (D3D12_RESOURCE_DESC, u32) {
    let mut d = *desc;
    d.Flags = D3D12_RESOURCE_FLAGS(d.Flags.0 & 0xFF) | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Alignment = 0;
    (d, state as u32 & PC_RESOURCE_STATES)
}

unsafe fn place_console_buffer(this: *mut c_void, props: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    if UMA_BUFFER.lock().unwrap().is_empty() || ppv.is_null() {
        return None;
    }
    let d: D3D12_RESOURCE_DESC = std::ptr::read(desc as *const D3D12_RESOURCE_DESC);
    if d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || d.Width == 0 {
        return None;
    }
    let bytes = d.Width as usize;
    let st = state as u32 & PC_RESOURCE_STATES;
    let heap_props = (props != 0 && crate::gpu::shader::readable(props, std::mem::size_of::<D3D12_HEAP_PROPERTIES>()))
        .then(|| std::ptr::read_unaligned(props as *const D3D12_HEAP_PROPERTIES));
    let rounded = (bytes + 0xFFFF) & !0xFFFF;
    let prefer_coherent = heap_props.as_ref().is_some_and(|hp| is_coherent_readback_buffer(hp, st, d.Flags, bytes))
        && has_coherent_headroom(u, rounded);
    let readback_on_null_map = heap_props.as_ref().is_some_and(|hp| is_cpu_readback_buffer(hp, st, d.Flags));
    // Non-readback (and general CUSTOM+WRITE_BACK) committed console buffers live in sparse VRAM
    // windows; small genuine readback buffers prefer coherent committed RAM (spilling to sparse if full).
    if !prefer_coherent && u.has_sparse() {
        if let Some(hr) = place_console_buffer_sparse(this, &d, bytes, st, readback_on_null_map, riid, ppv) {
            return Some(hr);
        }
    }
    let Some(p) = u.alloc(rounded, 0x1_0000).map(|a| a as usize) else {
        if prefer_coherent && u.has_sparse() {
            return place_console_buffer_sparse(this, &d, bytes, st, readback_on_null_map, riid, ppv);
        }
        return None;
    };
    // the heap from OpenExistingHeapFromAddress is cross-adapter: row-major + sanitized flags & state
    let (d, st) = sanitize_coherent_buffer(&d, state);
    let k = u.window_of(p)?;
    let heap = (*UMA_HEAPS.lock().unwrap().get(k)?)?;
    let hr = dev_create_placed_resource(this, heap, p - u.window(k).0, &d as *const _ as usize, st as usize, clear, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        u.free(p);
        static ONCE: AtomicU64 = AtomicU64::new(0);
        if ONCE.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!("console buffer in UMA: CreatePlacedResource {hr:#x} – keeping a separate resource (the shader will not see it)");
        }
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, bytes, true));
    // Map must hand out the same memory the shaders read (checked once)
    static CHECKED: AtomicU64 = AtomicU64::new(0);
    if CHECKED.fetch_add(1, Ordering::Relaxed) == 0 {
        let raw = *ppv;
        if let Some(r) = ID3D12Resource::from_raw_borrowed(&raw) {
            let mut m: *mut c_void = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut m)).is_ok() {
                r.Unmap(0, None);
                if m as usize != p {
                    crate::xlog_warn!("console buffer in UMA: Map returns {m:?}, not {p:#x} – CPU writes will not reach the console address");
                } else {
                    crate::xlog_info!("console buffers in UMA: Map = console address ({p:#x})");
                }
            }
        }
    }
    Some(hr)
}

/// A buffer the title places at its own address (CreatePlacedResourceX): at a placement
/// boundary (64 KiB) in unified memory it is that memory, placed in the window's heap; the
/// title's allocation stays the title's.
pub(super) unsafe fn place_at_title_address(this: *mut c_void, va: usize, desc: &[u8; 0x38], state: u32, clear: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(va)?;
    let (base, size) = u.window(k);
    let mut d: D3D12_RESOURCE_DESC = std::ptr::read_unaligned(desc.as_ptr() as *const D3D12_RESOURCE_DESC);
    let bytes = d.Width as usize;
    // a placed PC buffer starts at 64 KiB: one at another address starts below it and XCL adds
    // the difference where PC methods take an offset into it (`buffer_delta`)
    let start = va & !0xFFFF;
    let delta = va - start;
    if bytes == 0 || va - base + bytes > size {
        return None;
    }
    // the title's own memory (a sparse window): a resource over its video-memory tiles
    if u.is_sparse(va) {
        let (hr, delta) = super::sparse::place_buffer(this, va, &d, state, riid, ppv)?;
        UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (va, bytes, false));
        if delta != 0 {
            BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
        }
        return Some(hr);
    }
    let heap = (*UMA_HEAPS.lock().unwrap().get(k)?)?;
    d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Alignment = 0;
    d.Width = (delta + bytes) as u64;
    let hr = dev_create_placed_resource(this, heap, start - base, &d as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (va, bytes, false));
    if delta != 0 {
        BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
    }
    Some(hr)
}

/// buffers placed below their console address: resource → bytes before it
static BUFFER_DELTA: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// Where offset 0 of a console buffer is in its PC resource (0 for most).
pub fn buffer_delta(res: usize) -> u64 {
    BUFFER_DELTA.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()).unwrap_or(0)
}

#[cfg(test)]
pub(super) fn set_test_buffer_delta(res: usize, delta: Option<u64>) {
    let mut g = BUFFER_DELTA.lock().unwrap();
    match delta {
        Some(d) => {
            g.get_or_insert_with(HashMap::new).insert(res, d);
        }
        None => {
            if let Some(m) = g.as_mut() {
                m.remove(&res);
            }
        }
    }
}

/// Whether the PC's Map works on `res`: not a reserved resource, and its heap has CPU pages.
unsafe fn pc_can_map(res: *mut c_void) -> bool {
    !is_reserved(res) && pc_mappable(heap_properties(res))
}

/// The heap `res` lives in (not to be asked of a reserved resource).
unsafe fn heap_properties(res: *mut c_void) -> Option<D3D12_HEAP_PROPERTIES> {
    let r = ID3D12Resource::from_raw_borrowed(&res)?;
    let mut p = D3D12_HEAP_PROPERTIES::default();
    r.GetHeapProperties(Some(&mut p), None).ok().map(|_| p)
}

/// Whether the PC can map a resource in a heap with `props`: not a reserved one (no heap), not
/// one in a DEFAULT heap, not one in a custom heap without CPU pages.
fn pc_mappable(props: Option<D3D12_HEAP_PROPERTIES>) -> bool {
    match props {
        None => false,
        Some(p) if p.Type == D3D12_HEAP_TYPE_DEFAULT => false,
        Some(p) if p.Type == D3D12_HEAP_TYPE_CUSTOM => p.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
        Some(_) => true,
    }
}

/// resource Release (every caller): a console buffer's unified memory goes with it
/// Map: on the console every resource is CPU-visible memory. One in unified memory that the
/// PC cannot map (a texture XCL keeps in step with it) gives its unified-memory address – what
/// the console gives; the CPU's writes there reach the texture (`title_textures`).
unsafe extern "system" fn res_map(this: *mut c_void, sub: u32, range: *const c_void, data: *mut *mut c_void) -> HRESULT {
    // a resource in unified memory: its console address (no PC mapping – the PC's would be
    // the start of the PC resource, before the address for a buffer placed below it)
    if let Some((p, bytes)) = console_buffer(this as usize) {
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(p)) {
            if !range.is_null() {
                let r = *(range as *const D3D12_RANGE);
                if r.End > r.Begin {
                    let begin = r.Begin.min(bytes);
                    let end = r.End.min(bytes);
                    if end > begin {
                        super::sparse::readback_now(p + begin, end - begin);
                    }
                }
            } else if SPARSE_READBACK.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(this as usize))) {
                super::sparse::readback_now(p, bytes);
            }
        }
        if !data.is_null() {
            *data = p as *mut c_void;
        }
        UNMAPPABLE.lock().unwrap().get_or_insert_with(HashSet::new).insert(this as usize);
        return S_OK;
    }
    // the PC's Map only on a resource it can map: on a reserved one or one without CPU pages it is
    // an invalid call (debug layer 824 / 822) that the runtime answers with E_INVALIDARG
    let hr = if pc_can_map(this) {
        let f: unsafe extern "system" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> HRESULT = std::mem::transmute(orig_fn(this, slot_of(this, "Map")));
        f(this, sub, range, data)
    } else {
        E_INVALIDARG
    };
    if hr >= 0 {
        return hr;
    }
    match super::title_textures::address(this as usize).or_else(|| TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).copied())) {
        Some(p) => {
            if !data.is_null() {
                *data = p as *mut c_void;
            }
            UNMAPPABLE.lock().unwrap().get_or_insert_with(HashSet::new).insert(this as usize);
            S_OK
        }
        None => {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                let d = ID3D12Resource::from_raw_borrowed(&this).map(|r| r.GetDesc());
                crate::xlog_warn!("D3D12.X: Map of a resource not in unified memory → {:#x} ({:?})", hr as u32, d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Flags.0)));
            }
            hr
        }
    }
}

/// WriteToSubresource(subresource, box, data, row pitch, depth pitch): the CPU writing texels.
/// Every console resource is CPU-visible memory, a PC texture only in a CPU-visible heap.
unsafe extern "system" fn res_write_to_subresource(this: *mut c_void, sub: u32, dst_box: *const D3D12_BOX, data: *const c_void, row_pitch: u32, depth_pitch: u32) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_BOX, *const c_void, u32, u32) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "WriteToSubresource")));
    let hr = f(this, sub, dst_box, data, row_pitch, depth_pitch);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static FAILED: AtomicU64 = AtomicU64::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    if hr < 0 && FAILED.fetch_add(1, Ordering::Relaxed) < 8 {
        let d = ID3D12Resource::from_raw_borrowed(&this).map(|r| r.GetDesc());
        let b = dst_box.as_ref().map(|b| (b.left, b.top, b.right, b.bottom));
        crate::xlog_warn!(
            "D3D12.X: WriteToSubresource({sub}, box {b:?}, row pitch {row_pitch}) of {:?} → {:#x} (call #{n})",
            d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Layout.0, d.Flags.0)),
            hr as u32
        );
    }
    hr
}

/// resources the title placed at its own address that got a PC resource of their own: on the
/// console Map gives that address (the CPU's view of the resource)
static TITLE_ADDRESS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// The title address a resource stands for (buffers in unified memory, placed resources).
pub(super) fn title_address(res: usize) -> Option<usize> {
    console_buffer(res).map(|b| b.0).or_else(|| TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()))
}

pub(super) fn note_title_address(res: usize, va: usize) {
    TITLE_ADDRESS.lock().unwrap().get_or_insert_with(HashMap::new).insert(res, va);
}

/// resources whose Map gave their unified-memory address (no PC mapping to undo)
static UNMAPPABLE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

unsafe extern "system" fn res_unmap(this: *mut c_void, sub: u32, range: *const c_void) {
    if UNMAPPABLE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(this as usize))) {
        return;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32, *const c_void) = std::mem::transmute(orig_fn(this, slot_of(this, "Unmap")));
    f(this, sub, range)
}

unsafe extern "system" fn res_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    // The last reference: kept until the queues are past the work in flight (`late_release`).
    // (Two last-but-one releases at once can miss each other here; the object then goes at
    // once, as before.)
    let mut held = false;
    let n = if super::late_release::enabled() {
        let add: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 1));
        if add(this) == 2 {
            f(this);
            held = true;
            0
        } else {
            f(this);
            f(this)
        }
    } else {
        f(this)
    };
    if n == 0 {
        if let Some(s) = UNMAPPABLE.lock().unwrap().as_mut() {
            s.remove(&(this as usize));
        }
        if let Some(s) = SPARSE_READBACK.lock().unwrap().as_mut() {
            s.remove(&(this as usize));
        }
        if let Some(s) = RESERVED.lock().unwrap().as_mut() {
            s.remove(&(this as usize));
        }
        if let Some(m) = BUFFER_DELTA.lock().unwrap().as_mut() {
            m.remove(&(this as usize));
        }
        if let Some(m) = TITLE_ADDRESS.lock().unwrap().as_mut() {
            m.remove(&(this as usize));
        }
        super::sparse::forget_buffer(this as usize);
        super::vram::forget(this as usize);
        super::title_textures::forget(this as usize);
        let gone = UMA_BUFFERS.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize)));
        if let (Some((p, _, true)), Some(u)) = (gone, crate::gpu::uma::get()) {
            if u.is_sparse(p) {
                super::sparse::free(p);
            } else {
                u.free(p);
            }
        }
    }
    if held {
        super::late_release::hold(this as usize);
    }
    n
}

/// A stand-in pipeline released for good takes its background-built real pipeline with it.
unsafe extern "system" fn pso_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        forget_pso(this as usize);
        super::pso_async::release(super::pso_async::released(this as usize));
    }
    n
}

/// A pipeline state is gone: drop what XCL keeps by its address (a later pipeline may get the
/// same address and must not inherit its root signature or compute pass).
/// compute pipelines that run a translated console shader
static TRANSLATED_COMPUTE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// graphics pipelines built from translated console shaders
static TRANSLATED_GRAPHICS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether `pso` is a graphics pipeline of translated console shaders.
/// `XCL_TRACE_FRAME`: the console VS and PS (their dump names' hashes) of a translated PSO.
static PSO_SHADERS: Mutex<Option<HashMap<usize, (u64, u64)>>> = Mutex::new(None);

/// translated pipeline → its input layout as text (`XCL_TRACE_FRAME`)
static PSO_LAYOUTS: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);

pub fn pso_layout(pso: usize) -> Option<String> {
    PSO_LAYOUTS.lock().unwrap().as_ref()?.get(&pso).cloned()
}

pub fn pso_shaders(pso: usize) -> Option<(u64, u64)> {
    PSO_SHADERS.lock().unwrap().as_ref()?.get(&pso).copied()
}

pub fn is_translated_graphics(pso: usize) -> bool {
    pso != 0 && TRANSLATED_GRAPHICS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// graphics pipelines given the generated rect list GS (`rect_list_gs`)
static RECT_LIST_GS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether `pso` (as `is_translated_graphics` takes it: a stand-in resolved) draws rect lists
/// whole with the generated GS. Its draws must not be expanded to two triangles a rect as well:
/// each half is three corners of the rect and the GS would complete both.
pub fn has_rect_list_gs(pso: usize) -> bool {
    pso != 0 && RECT_LIST_GS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// Note `pso`, made from `copy`, if it kept the generated rect list GS (`rect_gs`; a pipeline
/// made again without it, `retry_without_rect_gs`, did not).
fn note_rect_list_gs(pso: usize, copy: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, rect_gs: bool) {
    if pso != 0 && rect_gs && !copy.GS.pShaderBytecode.is_null() {
        RECT_LIST_GS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

/// Whether `pso` runs a translated console compute shader.
pub fn is_translated_compute(pso: usize) -> bool {
    pso != 0 && TRANSLATED_COMPUTE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// compute pipelines the title made from a console shader that runs untranslated: a background
/// build's stand-in (until it is resolved) and the fallback of a console shader that failed
static CONSOLE_COMPUTE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

unsafe fn note_console_compute(hr: HRESULT, ppv: *mut *mut c_void) {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        CONSOLE_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    }
}

/// Whether `pso` was made from a console compute shader (translated or not); a native PC
/// compute pipeline is not.
pub fn is_console_compute(pso: usize) -> bool {
    pso != 0 && CONSOLE_COMPUTE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

#[cfg(test)]
pub(super) fn register_console_compute_for_test(pso: usize) {
    if pso != 0 {
        CONSOLE_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn register_translated_compute_for_test(pso: usize) {
    if pso != 0 {
        TRANSLATED_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn register_rect_list_gs_for_test(pso: usize) {
    if pso != 0 {
        RECT_LIST_GS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn register_translated_graphics_for_test(pso: usize) {
    if pso != 0 {
        TRANSLATED_GRAPHICS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn forget_pso_for_test(pso: usize) {
    forget_pso(pso);
}

fn forget_pso(pso: usize) {
    if let Some(s) = TRANSLATED_COMPUTE.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(s) = CONSOLE_COMPUTE.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(s) = TRANSLATED_GRAPHICS.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(s) = RECT_LIST_GS.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(m) = PSO_RS.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    if let Some(m) = PSO_USER_SGPRS.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    super::device_x::forget_base(pso);
    // the compute pass goes with the last user of its entry (`Drop for TwoPassPso`)
    let gone = TWO_PASS.lock().unwrap().as_mut().and_then(|m| m.remove(&pso));
    drop(gone);
}

/// resource #11 GetGPUVirtualAddress for console code
unsafe extern "system" fn res_gpu_va_x(this: *mut c_void) -> u64 {
    if let Some((p, _)) = console_buffer(this as usize) {
        return p as u64;
    }
    let f: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(orig_fn(this, 11));
    // a texture has no GPU address on the PC (0) – on the console every resource has one, and
    // titles use a texture's as the CPU's pointer to it (a retail game clears a 256x256 R32F one
    // through it: with 0 it wrote 256 KB at address 0): the title's address it was placed at
    match f(this) {
        0 => {
            let placed = TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).copied());
            placed.or_else(|| texture_memory(this)).unwrap_or(0) as u64
        }
        va => va,
    }
}

/// Memory for a texture the title did not place itself (a committed one) asking for its
/// address – on the console every texture has one (see `res_gpu_va_x`). Address space for the
/// texels of the PC description (+ a third for mips and 64 KiB of padding), its pages committed
/// as the title writes them (`kernelx::lazy`): the clears of GPU-written targets seen so far
/// cost nothing until done. What the title writes there is not uploaded to the texture, and the
/// memory stays with the address (not freed with the resource).
unsafe fn texture_memory(res: *mut c_void) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESERVE, PAGE_READWRITE};
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap();
    if let Some(p) = TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(res as usize)).copied()) {
        return Some(p);
    }
    let d = ID3D12Resource::from_raw_borrowed(&res)?.GetDesc();
    if d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return None;
    }
    let bits = crate::gpu::desc::gfx10_format(d.Format).map_or(128, |f| crate::gpu::desc::format_bytes(f).max(1) as u64 * 8);
    let texels = d.Width.checked_mul(d.Height as u64)?.checked_mul(d.DepthOrArraySize.max(1) as u64)?;
    let bytes = (texels.checked_mul(bits)? / 8 * 4 / 3 + 0x1_0000).next_multiple_of(0x1_0000);
    if bytes > 1 << 30 {
        return None;
    }
    let p = VirtualAlloc(std::ptr::null(), bytes as usize, MEM_RESERVE, PAGE_READWRITE) as usize;
    if p == 0 {
        return None;
    }
    for page in (p..p + bytes as usize).step_by(0x1_0000) {
        crate::hle::kernelx::lazy::mark(page);
    }
    note_title_address(res as usize, p);
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_info!(
            "D3D12.X: GetGPUVirtualAddress of a committed {}x{} texture (format {}, flags {:#x}): {bytes:#x} bytes at {p:#x} for the title to write through (not uploaded)",
            d.Width,
            d.Height,
            d.Format.0,
            d.Flags.0
        );
    }
    Some(p)
}

fn note_buffer_view() {
    if NON_UMA_BUFFER_VIEWS.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_warn!("buffer view in the console heap: the buffer is not in UMA – the shader gets an empty V# (num_records 0)");
    }
}

/// #17 CreateConstantBufferView: a V# of float4 rows at the buffer's console address
/// (`desc::cbv_words`; unified memory, anything else gets an empty V#). No PC CBV: translated
/// shaders read buffers through `mem`.
unsafe extern "system" fn dev_create_cbv_x(this: *mut c_void, desc: *const D3D12_CONSTANT_BUFFER_VIEW_DESC, cpu: u64) {
    let Some(slot) = super::console_heap::slot(cpu).filter(|s| !s.sampler) else {
        let f: unsafe extern "system" fn(*mut c_void, *const D3D12_CONSTANT_BUFFER_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 17));
        return f(this, desc, cpu);
    };
    let (va, size) = desc.as_ref().map_or((0, 0), |d| (d.BufferLocation, d.SizeInBytes));
    let in_uma = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    if !in_uma {
        note_buffer_view();
    }
    let (va, size) = if in_uma { (va, size) } else { (0, 0) };
    super::console_heap::write(&slot, &crate::gpu::desc::cbv_words(va, size));
}

/// #22 CreateSampler: the PC sampler at the slot's index, the S# in XCL ABI form (word 3
/// [11:0] = sampler heap index; the state lives in the PC sampler).
pub(super) unsafe extern "system" fn dev_create_sampler_x(this: *mut c_void, desc: *const D3D12_SAMPLER_DESC, cpu: u64) {
    let f: unsafe extern "system" fn(*mut c_void, *const D3D12_SAMPLER_DESC, u64) = std::mem::transmute(orig_fn(this, 22));
    let Some(slot) = super::console_heap::slot(cpu).filter(|s| s.sampler) else { return f(this, desc, cpu) };
    f(this, desc, slot.pc_srv);
    super::console_heap::write(&slot, &[0, 0, 0, slot.index & 0xFFF]);
}

/// Copy `n` console slots from `src` to `dst` (both in console heaps of one kind): the PC
/// descriptors (2 per view slot) and the slots. False: not two console ranges.
unsafe fn copy_console_slots(this: *mut c_void, n: u32, dst: u64, src: u64, ty: u32) -> bool {
    use super::console_heap::{copy_words, slot, SAMPLER_SLOT, SLOT};
    let (Some(d0), Some(s0)) = (slot(dst), slot(src)) else { return false };
    if d0.sampler != s0.sampler {
        return false;
    }
    let ty = pc_heap_type(ty);
    let f: unsafe extern "system" fn(*mut c_void, u32, u64, u64, u32) = std::mem::transmute(orig_fn(this, 24));
    f(this, if d0.sampler { n } else { 2 * n }, d0.pc_srv, s0.pc_srv, ty);
    if !d0.sampler {
        super::console_heap::publish_range(&d0, n);
    }
    let step = if d0.sampler { SAMPLER_SLOT } else { SLOT } as u64;
    for k in 0..n as u64 {
        if let (Some(d), Some(s)) = (slot(dst + k * step), slot(src + k * step)) {
            copy_words(&s, &d);
        }
    }
    true
}

/// `XCL_TRACE_FRAME`: the device's descriptor copies (first few, and a count)
fn note_copy_call(what: &str, n: u32, dst: u64, src: u64) {
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if super::present::trace_frame() && (k < 4 || k.is_power_of_two()) {
        crate::xlog_info!("frame trace: {what} #{k}: {n} from {src:#x} to {dst:#x}");
    }
}

static MIXED_COPIES: AtomicU64 = AtomicU64::new(0);

fn note_mixed_copy() {
    if MIXED_COPIES.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_warn!("CopyDescriptors (console): range outside the XCL console heaps – copying on the PC only");
    }
}

/// #24 CopyDescriptorsSimple(n, dst, src, type)
unsafe extern "system" fn dev_copy_descriptors_simple_x(this: *mut c_void, n: u32, dst: u64, src: u64, ty: u32) {
    let ty = pc_heap_type(ty);
    note_copy_call("CopyDescriptorsSimple", n, dst, src);
    if copy_console_slots(this, n, dst, src, ty) {
        return;
    }
    if super::console_heap::slot(dst).is_some() || super::console_heap::slot(src).is_some() {
        note_mixed_copy();
        return;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32, u64, u64, u32) = std::mem::transmute(orig_fn(this, 24));
    f(this, n, dst, src, ty);
}

/// #23 CopyDescriptors(dst ranges, src ranges, type): range by range over console slots.
#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn dev_copy_descriptors_x(this: *mut c_void, nd: u32, dsts: *const u64, dsizes: *const u32, ns: u32, srcs: *const u64, ssizes: *const u32, ty: u32) {
    let ty = pc_heap_type(ty);
    note_copy_call("CopyDescriptors", nd, if dsts.is_null() { 0 } else { *dsts }, if srcs.is_null() { 0 } else { *srcs });
    use super::console_heap::{slot, SAMPLER_SLOT, SLOT};
    let console = !dsts.is_null() && !srcs.is_null() && nd > 0 && slot(*dsts).is_some();
    if !console {
        if !srcs.is_null() && ns > 0 && slot(*srcs).is_some() {
            note_mixed_copy();
            return;
        }
        type F = unsafe extern "system" fn(*mut c_void, u32, *const u64, *const u32, u32, *const u64, *const u32, u32);
        let f: F = std::mem::transmute(orig_fn(this, 23));
        return f(this, nd, dsts, dsizes, ns, srcs, ssizes, ty);
    }
    // every descriptor as (dst, src) handle, ranges walked in step
    let step = if ty == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32 { SAMPLER_SLOT } else { SLOT } as u64;
    let expand = |starts: *const u64, sizes: *const u32, n: u32| -> Vec<u64> {
        (0..n as usize).flat_map(|r| {
            let (h, len) = (*starts.add(r), if sizes.is_null() { 1 } else { *sizes.add(r) });
            (0..len as u64).map(move |k| h + k * step)
        })
        .collect()
    };
    let (d, s) = (expand(dsts, dsizes, nd), expand(srcs, ssizes, ns));
    for (a, b) in d.iter().zip(s.iter()) {
        if !copy_console_slots(this, 1, *a, *b, ty) {
            note_mixed_copy();
        }
    }
}

/// descriptor heap Release (every caller): a console heap's slots go with it
unsafe extern "system" fn heap_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        super::console_heap::unregister(this as usize);
    }
    n
}

/// #21 CreateDepthStencilView: the console's view formats as the PC's (`pc_view_format`).
unsafe extern "system" fn dev_create_dsv_x(this: *mut c_void, res: *mut c_void, desc: *const D3D12_DEPTH_STENCIL_VIEW_DESC, cpu: u64) {
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_DEPTH_STENCIL_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 21));
    let mut copy = desc.as_ref().copied();
    if let Some(c) = copy.as_mut() {
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_target_format(c.Format.0 as u32) as i32);
    }
    // no description: the resource's format – which a depth buffer made typeless
    // (`device_x::pc_resource_desc`) is not one a view can have
    if copy.is_none() {
        copy = ID3D12Resource::from_raw_borrowed(&res).and_then(|r| default_dsv(&r.GetDesc()));
    }
    f(this, res, copy.as_ref().map_or(desc, |c| c as *const _), cpu)
}

/// The view a DSV without a description has, for a depth buffer `pc_resource_desc` made
/// typeless (None: the resource's own format is a depth one).
fn default_dsv(d: &D3D12_RESOURCE_DESC) -> Option<D3D12_DEPTH_STENCIL_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    let format = match d.Format.0 {
        19 => 20,
        39 => 40,
        44 => 45,
        53 => 55,
        _ => return None,
    };
    let mut v = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT(format), ..Default::default() };
    if d.DepthOrArraySize > 1 {
        v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_DSV { MipSlice: 0, FirstArraySlice: 0, ArraySize: d.DepthOrArraySize as u32 };
    } else {
        v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    }
    Some(v)
}

/// A shader view `v` (None: the resource's default one) of `rd` with the PC's format names
/// (`device_x::pc_shader_view_format`) – None: as it is.
fn pc_srv(rd: &D3D12_RESOURCE_DESC, v: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    if rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return None;
    }
    let rf = rd.Format.0 as u32;
    let mut out = match v {
        Some(v) => *v,
        None => {
            // only a typeless depth buffer needs a description of its own
            if !matches!(rf, 19 | 39 | 44 | 53) {
                return None;
            }
            let mut v = D3D12_SHADER_RESOURCE_VIEW_DESC { Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
            let slices = u32::from(rd.DepthOrArraySize);
            // a multisampled depth buffer has only the MS view dimensions
            match (rd.SampleDesc.Count > 1, slices > 1) {
                (true, true) => {
                    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                    v.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: 0, ArraySize: slices };
                }
                (true, false) => v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS,
                (false, true) => {
                    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                    v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: u32::MAX, ArraySize: slices, ..Default::default() };
                }
                (false, false) => {
                    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    v.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: u32::MAX, ..Default::default() };
                }
            }
            v
        }
    };
    if out.ViewDimension == D3D12_SRV_DIMENSION_BUFFER || out.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE {
        return None;
    }
    let f = super::device_x::pc_shader_view_format(rf, out.Format.0 as u32);
    let stencil = super::device_x::stencil_view(rf, f);
    if v.is_some() && f == out.Format.0 as u32 && !stencil {
        return None;
    }
    out.Format = DXGI_FORMAT(f as i32);
    // the PC reads stencil from plane 1; the console's view of it is plane 0 in a stencil format
    if stencil {
        if out.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D {
            out.Anonymous.Texture2D.PlaneSlice = 1;
        } else if out.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
            out.Anonymous.Texture2DArray.PlaneSlice = 1;
        }
    }
    Some(out)
}

unsafe extern "system" fn dev_create_srv_x(this: *mut c_void, res: *mut c_void, desc: *const D3D12_SHADER_RESOURCE_VIEW_DESC, cpu: u64) {
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 18));
    let mut copy = desc.as_ref().copied();
    if let Some(c) = copy.as_mut() {
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_view_format(c.Format.0 as u32) as i32);
    }
    // a depth buffer's view in the formats the PC has for it (a D24S8 format itself, R16_UNORM of
    // a D16S8 buffer …: an invalid call, the device removed)
    if let Some(v) = ID3D12Resource::from_raw_borrowed(&res).and_then(|r| pc_srv(&r.GetDesc(), copy.as_ref())) {
        copy = Some(v);
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _);
    let Some(slot) = super::console_heap::slot(cpu) else { return f(this, res, desc, cpu) };
    // Raytracing acceleration structure SRV (pResource == NULL, Location == console GPU VA):
    // translated shaders read BVH nodes from UMA (`bvh_intersect`), never from a PC DXR SRV,
    // so write the RDNA2 BVH descriptor directly and skip forwarding the console VA to PC.
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE) {
        let va = v.Anonymous.RaytracingAccelerationStructure.Location;
        let in_uma = va != 0 && crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
        if in_uma {
            let w = crate::gpu::desc::bvh_words(va, true);
            return super::console_heap::write(&slot, &[w[0], w[1], w[2], w[3], 0, 0, 0, 0]);
        }
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: CreateShaderResourceView(RAYTRACING_ACCELERATION_STRUCTURE) at {va:#x} is not in unified memory – writing empty BVH descriptor");
        }
        return super::console_heap::write(&slot, &[0; 8]);
    }
    // translated shaders declare every cube T# as a TextureCubeArray: a title's TEXTURECUBE
    // view becomes the cube array of that one cube (a dimension mismatch is undefined in D3D12)
    let cube = desc.as_ref().and_then(cube_as_cube_array);
    let desc = cube.as_ref().map_or(desc, |v| v as *const _);
    f(this, res, desc, slot.pc_srv);
    super::console_heap::publish(&slot, false);
    super::console_heap::note_resource(&slot, res as usize);
    if super::present::trace_frame() && super::title_textures::address(res as usize).is_some() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 24 {
            let rd = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc());
            let v = desc.as_ref().map(|v| (v.Format.0, v.ViewDimension.0, v.Shader4ComponentMapping, v.Anonymous.Texture2D.MostDetailedMip, v.Anonymous.Texture2D.MipLevels, v.Anonymous.Texture2D.ResourceMinLODClamp));
            crate::xlog_info!(
                "frame trace: SRV of title texture {res:p} {:?}: view (format, dimension, mapping, mip, mips, clamp) {v:x?}",
                rd.map(|d| (d.Width, d.Height, d.Format.0, d.MipLevels, d.Flags.0))
            );
        }
    }
    let Some(r) = ID3D12Resource::from_raw_borrowed(&res) else {
        return super::console_heap::write(&slot, &[0; 8]);
    };
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        return super::console_heap::write(&slot, &buffer_view(res as usize, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw));
    }
    let (format, dim, array, mips) = match desc.as_ref() {
        Some(v) => {
            let a = &v.Anonymous;
            let (array, mips) = match v.ViewDimension.0 {
                2 => (1, a.Texture1D.MipLevels),
                3 => (a.Texture1DArray.ArraySize, a.Texture1DArray.MipLevels),
                4 => (1, a.Texture2D.MipLevels),
                5 => (a.Texture2DArray.ArraySize, a.Texture2DArray.MipLevels),
                7 => (a.Texture2DMSArray.ArraySize, 1),
                8 => (1, a.Texture3D.MipLevels),
                9 => (6, a.TextureCube.MipLevels),
                10 => (a.TextureCubeArray.NumCubes.max(1) * 6, a.TextureCubeArray.MipLevels),
                _ => (1, 1),
            };
            (v.Format, v.ViewDimension.0, array, mips)
        }
        None => {
            let d = r.GetDesc();
            (d.Format, resource_dim_to_view(&d), d.DepthOrArraySize as u32, d.MipLevels as u32)
        }
    };
    match view_shape(r, format, dim, array, mips) {
        Some((fmt, ty, w, h, d, m)) => super::console_heap::write(&slot, &crate::gpu::desc::xcl_image_for_view(slot.index, fmt, ty, w, h, d, m)),
        None => {
            note_buffer_view();
            super::console_heap::write(&slot, &[0; 8]);
        }
    }
}

/// A TEXTURECUBE view as the TEXTURECUBEARRAY of its one cube (None: any other view).
fn cube_as_cube_array(v: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    if v.ViewDimension != D3D12_SRV_DIMENSION_TEXTURECUBE {
        return None;
    }
    // SAFETY: the union member the view dimension names
    let c = unsafe { v.Anonymous.TextureCube };
    Some(D3D12_SHADER_RESOURCE_VIEW_DESC {
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                MostDetailedMip: c.MostDetailedMip,
                MipLevels: c.MipLevels,
                First2DArrayFace: 0,
                NumCubes: 1,
                ResourceMinLODClamp: c.ResourceMinLODClamp,
            },
        },
        ..*v
    })
}

/// The texture a view by address describes: `desc` (a resource desc) placed at `va`, made once
/// per address and desc – titles make views of the same memory again and again.
unsafe fn resource_at(this: *mut c_void, desc: *const u8, va: u64) -> *mut c_void {
    type Key = (u64, [u8; 0x38]);
    static MADE: Mutex<Option<HashMap<Key, usize>>> = Mutex::new(None);
    if desc.is_null() || va == 0 || !crate::gpu::shader::readable(desc as usize, 0x38) {
        return std::ptr::null_mut();
    }
    // the PC part of the desc; the flags (@0x30) are its last dword, padding follows
    let mut x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    x[0x34..].fill(0);
    if let Some(&r) = MADE.lock().unwrap().get_or_insert_with(HashMap::new).get(&(va, x)) {
        return r as *mut c_void;
    }
    let mut r = std::ptr::null_mut();
    let hr = super::device_x::create_placed_resource_x(this, va, x.as_ptr(), 0, std::ptr::null(), &ID3D12Resource::IID as *const _ as *const GUID, &mut r);
    if hr < 0 || r.is_null() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            crate::xlog_warn!("D3D12.X: view at {va:#x}: the resource cannot be made ({hr:#x}); desc {x:02x?}");
        }
        r = std::ptr::null_mut();
    }
    MADE.lock().unwrap().get_or_insert_with(HashMap::new).insert((va, x), r as usize);
    r
}

/// device #70 (a retail game): a shader resource view of memory, without a resource object:
/// (resource desc, { the PC's SRV desc; GPU address @0x28; … }, destination).
unsafe extern "system" fn dev_create_srv_at_x(this: *mut c_void, rdesc: *const u8, view: *const u8, cpu: u64) {
    if view.is_null() || !crate::gpu::shader::readable(view as usize, 0x40) {
        return;
    }
    let va = (view.add(0x28) as *const u64).read_unaligned();
    let v = (view as *const D3D12_SHADER_RESOURCE_VIEW_DESC).read_unaligned();
    if v.ViewDimension == D3D12_SRV_DIMENSION_BUFFER {
        let Some(slot) = super::console_heap::slot(cpu) else { return };
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        let words = buffer_words_at(va, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw);
        return super::console_heap::write(&slot, &words);
    }
    dev_create_srv_x(this, resource_at(this, rdesc, va), &v, cpu);
}

/// device #71: the same for an unordered access view ({ the PC's UAV desc; GPU address @0x28;
/// the counter's address @0x30; … }).
unsafe extern "system" fn dev_create_uav_at_x(this: *mut c_void, rdesc: *const u8, view: *const u8, cpu: u64) {
    if view.is_null() || !crate::gpu::shader::readable(view as usize, 0x40) {
        return;
    }
    let va = (view.add(0x28) as *const u64).read_unaligned();
    let v = (view as *const D3D12_UNORDERED_ACCESS_VIEW_DESC).read_unaligned();
    if v.ViewDimension == D3D12_UAV_DIMENSION_BUFFER {
        let Some(slot) = super::console_heap::slot(cpu) else { return };
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        let words = buffer_words_at(va, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw);
        return super::console_heap::write(&slot, &words);
    }
    dev_create_uav_x(this, resource_at(this, rdesc, va), std::ptr::null_mut(), &v, cpu);
}

/// V# of `n` elements from element `first` of the buffer at `va`.
fn buffer_words_at(va: u64, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, first: u64, n: u32, stride: u32, raw: bool) -> [u32; 4] {
    let fmt = crate::gpu::desc::gfx10_format(format).unwrap_or(22);
    let (stride, elem) = match (raw, stride) {
        (true, _) => (0, 4),
        (false, 0) => {
            let e = crate::gpu::desc::format_bytes(fmt);
            (e, e)
        }
        (false, s) => (s, s),
    };
    let records = if stride == 0 { n.saturating_mul(4) } else { n };
    crate::gpu::desc::buffer_words(va + first * elem as u64, stride, records, fmt)
}

unsafe extern "system" fn dev_create_uav_x(this: *mut c_void, res: *mut c_void, counter: *mut c_void, desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, cpu: u64) {
    // the view's format as the PC's for the resource (B5G5R5A1 of a target made R8G8B8A8,
    // `device_x::pc_resource_desc`: an invalid call, the device removed)
    let mut copy = desc.as_ref().copied();
    if let (Some(c), Some(r)) = (copy.as_mut(), ID3D12Resource::from_raw_borrowed(&res)) {
        let rf = r.GetDesc().Format.0 as u32;
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_shader_view_format(rf, c.Format.0 as u32) as i32);
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _);
    super::present::note_use(res as usize, super::present::USE_UAV);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const D3D12_UNORDERED_ACCESS_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 19));
    if super::present::trace_frame() {
        let d = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc());
        if d.is_some_and(|d| d.Width >= 1920 && d.Flags.0 & 1 != 0) {
            let s = super::console_heap::slot(cpu);
            crate::xlog_info!("frame trace: UAV of render target {res:p} {:?} at {cpu:#x}: console slot {:?}, view {:?}", d.map(|d| (d.Width, d.Height, d.Format.0)), s.as_ref().map(|s| (s.index, s.pc_uav, s.at)), desc.as_ref().map(|v| (v.Format.0, v.ViewDimension.0)));
        }
    }
    // a resource without ALLOW_UNORDERED_ACCESS (a console depth buffer that also had it: the PC
    // refuses the pair, `device_x::pc_resource_desc`): a null view – a UAV of it is an invalid
    // call, and the device is removed (0x887A0001)
    let no_uav = ID3D12Resource::from_raw_borrowed(&res).is_some_and(|r| r.GetDesc().Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0);
    let null_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
    let pc_uav = |at: u64| {
        if no_uav {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: a UAV of a resource that cannot have one on the PC (a depth buffer) – a null view");
            }
            f(this, std::ptr::null_mut(), std::ptr::null_mut(), &null_desc, at)
        } else {
            f(this, res, counter, desc, at)
        }
    };
    let Some(slot) = super::console_heap::slot(cpu) else { return pc_uav(cpu) };
    pc_uav(slot.pc_uav);
    super::console_heap::publish(&slot, true);
    super::console_heap::note_resource(&slot, res as usize);
    let Some(r) = ID3D12Resource::from_raw_borrowed(&res) else {
        return super::console_heap::write(&slot, &[0; 8]);
    };
    let rd = r.GetDesc();
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_UAV_DIMENSION_BUFFER) {
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        return super::console_heap::write(&slot, &buffer_view(res as usize, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw));
    }
    // UAV dimensions: 1 buffer, 2 tex1d, 3 tex1d array, 4 tex2d, 5 tex2d array, 8 tex3d
    let (format, dim, array, mip) = match desc.as_ref() {
        Some(v) => {
            let a = &v.Anonymous;
            let (array, mip) = match v.ViewDimension.0 {
                2 => (1, a.Texture1D.MipSlice),
                3 => (a.Texture1DArray.ArraySize, a.Texture1DArray.MipSlice),
                4 => (1, a.Texture2D.MipSlice),
                5 => (a.Texture2DArray.ArraySize, a.Texture2DArray.MipSlice),
                8 => (a.Texture3D.WSize, a.Texture3D.MipSlice),
                _ => (1, 0),
            };
            (v.Format, v.ViewDimension.0, array, mip)
        }
        None => (rd.Format, resource_dim_to_view(&rd), rd.DepthOrArraySize as u32, 0),
    };
    let Some((fmt, ty, w, h, d, _)) = view_shape(r, format, dim, array, 1) else {
        note_buffer_view();
        return super::console_heap::write(&slot, &[0; 8]);
    };
    // RDNA images have no SRV / UAV distinction: console shaders may read this slot too
    if dim == 4 {
        let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: if format == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN { rd.Format } else { format },
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: mip, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        };
        if let Some(v) = pc_srv(&rd, Some(&srv)) {
            srv = v;
        }
        let g: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 18));
        g(this, res, &srv, slot.pc_srv);
        super::console_heap::publish(&slot, false);
    }
    let (w, h) = ((w >> mip).max(1), (h >> mip).max(1));
    super::console_heap::write(&slot, &crate::gpu::desc::xcl_image_for_view(slot.index, fmt, ty, w, h, d, 1));
}

/// device #127 (a retail game): CreateCommittedResource with a barrier layout in the initial state's place
/// – the arguments of the PC's CreateCommittedResource3 (heap properties, flags, desc, layout,
/// clear value, protected session, castable formats, riid, resource).
#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn dev_create_committed_resource_layout_x(
    this: *mut c_void,
    props: usize,
    flags: u32,
    desc: usize,
    layout: u32,
    clear: usize,
    _protected: usize,
    _castable: u32,
    _formats: usize,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    note_newer_device_method(127);
    let state = super::device_x::layout_state(layout) as usize;
    if flags & D3D12_HEAP_FLAG_ALLOW_DISPLAY.0 as u32 != 0 && desc != 0 && crate::gpu::shader::readable(desc, 0x38) {
        return create_display_target(this, props, desc, state, clear, riid, ppv);
    }
    dev_create_committed_resource_x(this, props, flags as usize, desc, state, clear, riid, ppv)
}

/// A display target: its console memory comes from the title's allocation hook, as the console
/// runtime's does (a retail game gives its back buffers one block and asserts that it is used up); on the
/// PC it is a resource of its own.
unsafe fn create_display_target(this: *mut c_void, props: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let d = super::device_x::pc_resource_desc(&x);
    let hr = dev_create_committed_resource(this, props, 0, d.as_ptr() as usize, state, clear, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        crate::xlog_warn!("D3D12.X: display target: {hr:#x}; desc {x:02x?}");
        return hr;
    }
    let size = crate::hle::xg::Texture::new(&x).map_or(0, |t| t.size()) as usize;
    let at = if size > 0 { crate::hle::kernelx::runtime_alloc(size, crate::hle::kernelx::D3D_DISPLAY_TARGET) } else { None };
    if let Some(p) = at {
        note_title_address(*ppv as usize, p);
    }
    crate::xlog_info!(
        "D3D12.X: display target {}x{} format {}: {size:#x} bytes of the title's memory at {:#x}",
        u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()),
        u32::from_le_bytes(x[0x18..0x1C].try_into().unwrap()),
        u32::from_le_bytes(x[0x20..0x24].try_into().unwrap()),
        at.unwrap_or(0)
    );
    hr
}

unsafe extern "system" fn dev_create_committed_resource_x(this: *mut c_void, props: usize, flags: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    if desc != 0 && std::env::var_os("XCL_D3D12_TRACE").is_some() && SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
        let w = |o: usize| *((desc + o) as *const u32);
        crate::xlog_info!(
            "CreateCommittedResource (console, slot 76): dimension {} {}x{} format {} flags {:#x}; console fields after the PC desc: +52 {:#x} +56 {:#x}",
            w(0),
            *((desc + 16) as *const u64),
            w(24),
            w(32),
            w(48),
            w(52),
            w(56)
        );
        crate::xlog_info!("  desc (dwords): {:x?}", (0..16).map(|k| w(4 * k)).collect::<Vec<_>>());
    }
    // console texture layouts (seen: 0x101, with 3 at +52) are not PC ones: the driver picks
    // its own (UNKNOWN); the console layout matters only where console code reads the memory
    let mut pc_desc = [0u8; 56];
    let console_desc = desc;
    let mut desc = desc;
    if desc != 0 {
        std::ptr::copy_nonoverlapping(desc as *const u8, pc_desc.as_mut_ptr(), 56);
        let layout = u32::from_le_bytes(pc_desc[44..48].try_into().unwrap());
        if layout > 3 {
            pc_desc[44..48].copy_from_slice(&0u32.to_le_bytes());
            pc_desc[52..56].copy_from_slice(&0u32.to_le_bytes());
            desc = pc_desc.as_ptr() as usize;
        }
    }
    if desc != 0 {
        if let Some(hr) = place_console_buffer(this, props, desc, state, clear, riid, ppv) {
            return hr;
        }
        if let Some(hr) = place_console_texture(this, console_desc, state, clear, riid, ppv) {
            return hr;
        }
    }
    // not placed in unified memory (e.g. its coherent window kept for readback heaps): a PC
    // resource of the PC's description – the console's flag bits (0x1000000, 0x300000 …) and
    // alignments made the PC refuse Fortnite's 3264x1836 scene targets (E_INVALIDARG, a crash)
    let pc_full = (console_desc != 0 && crate::gpu::shader::readable(console_desc, 0x38)).then(|| super::device_x::pc_resource_desc(&(console_desc as *const [u8; 0x38]).read_unaligned()));
    let desc = pc_full.as_ref().map_or(desc, |d| d.as_ptr() as usize);
    let mut clear_buf = [0u8; 24];
    let clear = super::device_x::pc_clear(clear as *const c_void, &mut clear_buf) as usize;
    // console-only state bits (e.g. 0x100000 on a display target) are not the PC's; a depth
    // buffer the PC description took ALLOW_UNORDERED_ACCESS from does not start as a UAV
    let st = pc_full.as_ref().map_or(state as u32 & PC_RESOURCE_STATES, |d| super::device_x::pc_initial_state(u32::from_le_bytes(d[0x30..0x34].try_into().unwrap()), state as u32));
    let hr = dev_create_committed_resource(this, props, flags, desc, st as usize, clear, riid, ppv);
    if hr < 0 {
        let pw = |o: usize| if props == 0 { 0 } else { *((props + o) as *const u32) };
        crate::xlog_warn!(
            "CreateCommittedResource (console): {hr:#x}; heap type {} cpu {} pool {}, heap flags {flags:#x}, state {state:#x}",
            pw(0),
            pw(4),
            pw(8)
        );
    }
    hr
}

pub(super) unsafe extern "system" fn dev_create_command_list_x(this: *mut c_void, mask: usize, ty: usize, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let pc_pso = super::console_list::pc_pso(pso as *mut c_void) as usize;
    let hr = dev_create_command_list(this, mask, ty, alloc, pc_pso, riid, ppv);
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        *ppv = super::console_list::create_with_allocator(*ppv, alloc as *mut c_void);
    }
    hr
}

unsafe extern "system" fn dev_create_compute_pso_x(this: *mut c_void, desc: *const c_void, a: usize, b: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if a != 0 || b != 0 {
        crate::xlog_warn!("CreateComputePipelineState (console, slot 62): unknown parameters {a:#x} {b:#x} ignored");
    }
    let hr = dev_create_compute_pso(this, desc, riid, ppv);
    // console code writes into pipeline objects (D3D12.X inline code): give it a console-shaped one
    if hr >= 0 && super::console_list::enabled() && !ppv.is_null() && !(*ppv).is_null() {
        *ppv = super::console_list::wrap_pso(*ppv, LAST_COMPUTE_TRANSLATED.with(|c| c.get()), LAST_COMPUTE_RS.with(|c| c.replace(0)));
    }
    hr
}

fn overrides(kind: Kind) -> Vec<(&'static str, usize)> {
    let mut v: Vec<(&'static str, usize)> = vec![("QueryInterface", qi_override as usize)];
    match kind {
        Kind::Device => v.extend_from_slice(&[
            ("CreateCommandQueue", dev_create_command_queue as usize),
            ("CheckFeatureSupport", dev_check_feature_support as usize),
            ("CreateCommandAllocator", dev_create_command_allocator as usize),
            ("CreateGraphicsPipelineState", dev_create_graphics_pso as usize),
            ("CreateComputePipelineState", dev_create_compute_pso as usize),
            ("CreateCommandList", dev_create_command_list as usize),
            ("CreateDescriptorHeap", dev_create_descriptor_heap as usize),
            ("CreateRootSignature", dev_create_root_signature as usize),
            ("CreateRenderTargetView", dev_create_rtv as usize),
            ("CreateCommittedResource", dev_create_committed_resource as usize),
            ("CreateHeap", dev_create_heap as usize),
            ("CreatePlacedResource", dev_create_placed_resource as usize),
            ("CreateReservedResource", dev_create_reserved_resource as usize),
            ("CreateFence", dev_create_fence as usize),
            ("CreateQueryHeap", dev_create_query_heap as usize),
            ("CreateCommandSignature", dev_create_command_signature as usize),
            ("CreatePipelineLibrary", dev_create_pipeline_library as usize),
            ("CreatePipelineState", dev_create_pipeline_state as usize),
            ("CreateCommandList1", dev_create_command_list1 as usize),
            ("CreateCommittedResource1", dev_create_committed_resource1 as usize),
            ("CreateHeap1", dev_create_heap1 as usize),
            ("CreateReservedResource1", dev_create_reserved_resource1 as usize),
            ("CreateStateObject", dev_create_state_object as usize),
            ("CreateCommittedResource2", dev_create_committed_resource2 as usize),
            ("CreatePlacedResource1", dev_create_placed_resource1 as usize),
            ("CreateShaderCacheSession", dev_create_shader_cache_session as usize),
            ("CreateCommandQueue1", dev_create_command_queue1 as usize),
        ]),
        Kind::CommandList => v.extend_from_slice(&[
            ("Release", cl_release as usize),
            ("Close", cl_close as usize),
            ("Reset", cl_reset as usize),
            ("ClearState", cl_clear_state as usize),
            ("DrawInstanced", cl_draw_instanced as usize),
            ("DrawIndexedInstanced", cl_draw_indexed_instanced as usize),
            ("SetPipelineState", cl_set_pipeline_state as usize),
            ("OMSetRenderTargets", cl_om_set_render_targets as usize),
            ("ClearRenderTargetView", cl_clear_rtv as usize),
            ("ExecuteIndirect", cl_execute_indirect as usize),
        ]),
        Kind::DmaList => v.extend_from_slice(&[
            ("Release", cl_release as usize),
            ("Close", cl_close as usize),
            ("Reset", cl_reset as usize),
        ]),
        Kind::CommandQueue => v.extend_from_slice(&[("Release", q_release as usize), ("ExecuteCommandLists", q_execute_command_lists as usize)]),
        Kind::CommandAllocator => v.extend_from_slice(&[("Release", alloc_release as usize), ("Reset", alloc_reset as usize)]),
        Kind::Resource => v.extend_from_slice(&[
            ("Release", res_release as usize),
            ("Map", res_map as usize),
            ("Unmap", res_unmap as usize),
            ("WriteToSubresource", res_write_to_subresource as usize),
        ]),
        Kind::PipelineState => v.push(("Release", pso_release as usize)),
        Kind::DescriptorHeap => v.push(("Release", heap_release as usize)),
        Kind::Heap => v.push(("Release", mem_heap_release as usize)),
        Kind::Fence => v.push(("Signal", fence_signal as usize)),
        _ => {}
    }
    v
}

/// Slot of `method` in the class of an adopted object.
unsafe fn slot_of(this: *mut c_void, method: &str) -> usize {
    let ext = vt(this) as usize;
    let g = CLASSES.lock().unwrap();
    let c = g.iter().find(|c| c.ext == ext).expect("adopted class");
    layout::slot(c.iface, method).expect("method in layout")
}

unsafe extern "system" fn qi_override(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let f: QiFn = std::mem::transmute(orig_fn(this, 0));
    let hr = f(this, riid, ppv);
    if hr == E_NOINTERFACE && !riid.is_null() {
        let s = layout::guid_string(&*riid);
        if UNKNOWN_IIDS.lock().unwrap().insert(s.clone()) {
            crate::xlog_warn!("D3D12.X: QueryInterface({s}) – interface unknown on the PC (console-only?)");
        }
    }
    hr
}

/// device #0 QueryInterface for console code: the console device is also an IDXGIDevice
/// (`dxgi_device`). Only the title gets it – DXGI asks a D3D12 device the same while creating
/// a swap chain and must keep getting the PC answer.
unsafe extern "system" fn dev_qi_x(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if !riid.is_null() && !ppv.is_null() && super::dxgi_device::answers(&std::mem::transmute::<GUID, windows::core::GUID>(*riid)) {
        *ppv = super::dxgi_device::create(this);
        return S_OK;
    }
    qi_override(this, riid, ppv)
}

fn pc_iid(kind: Kind) -> GUID {
    layout::parse_guid(layout::find(kind.base()).map(|i| i.iid).unwrap_or(""))
}

macro_rules! as_usize {
    ($x:ident) => {
        usize
    };
}

/// Close: a failure is logged with what was recorded (titles stop on it; the PC runtime
/// refuses some calls only here). Newer D3D12 runtimes (e.g. Agility SDK 1.620+) swap the
/// command list's vtable pointer on Close; adopt the runtime's new state vtable so
/// subsequent Reset and console calls dispatch through the current native table.
unsafe extern "system" fn cl_close(this: *mut c_void) -> HRESULT {
    super::list_x::before_close(this);
    let ext = *(this as *const usize);
    let f: RefFn = std::mem::transmute(orig_fn(this, slot_of(this, "Close")));
    let hr = f(this) as HRESULT;
    preserve_vtable(this, ext);
    if hr < 0 {
        let ty: RefFn = std::mem::transmute(orig_fn(this, slot_of(this, "GetType")));
        crate::xlog_warn!("D3D12.X: command list {this:p} (type {}) Close → {:#x}", ty(this), hr as u32);
    }
    hr
}

/// Reset (PC path): re-adopt the runtime's new open-state vtable if Reset replaced it.
unsafe extern "system" fn cl_reset(this: *mut c_void, allocator: *mut c_void, initial_state: *mut c_void) -> HRESULT {
    let ext = *(this as *const usize);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> HRESULT =
        std::mem::transmute(orig_fn(this, slot_of(this, "Reset")));
    let resolved = super::pso_async::resolve_for_bind(initial_state as usize) as *mut c_void;
    let hr = f(this, allocator, resolved);
    preserve_vtable(this, ext);
    if hr >= 0 {
        super::list_x::on_pc_reset(this);
    }
    hr
}

/// The PC size of a CheckFeatureSupport structure.
fn pc_feature_size(feature: i32) -> Option<usize> {
    use std::mem::size_of;
    Some(match D3D12_FEATURE(feature) {
        D3D12_FEATURE_D3D12_OPTIONS => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>(),
        D3D12_FEATURE_ARCHITECTURE => size_of::<D3D12_FEATURE_DATA_ARCHITECTURE>(),
        D3D12_FEATURE_FEATURE_LEVELS => size_of::<D3D12_FEATURE_DATA_FEATURE_LEVELS>(),
        D3D12_FEATURE_FORMAT_SUPPORT => size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>(),
        D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS => size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>(),
        D3D12_FEATURE_FORMAT_INFO => size_of::<D3D12_FEATURE_DATA_FORMAT_INFO>(),
        D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT => size_of::<D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT>(),
        D3D12_FEATURE_SHADER_MODEL => size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>(),
        D3D12_FEATURE_D3D12_OPTIONS1 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>(),
        D3D12_FEATURE_ROOT_SIGNATURE => size_of::<D3D12_FEATURE_DATA_ROOT_SIGNATURE>(),
        D3D12_FEATURE_ARCHITECTURE1 => size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>(),
        D3D12_FEATURE_D3D12_OPTIONS2 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS2>(),
        D3D12_FEATURE_SHADER_CACHE => size_of::<D3D12_FEATURE_DATA_SHADER_CACHE>(),
        D3D12_FEATURE_D3D12_OPTIONS3 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS3>(),
        D3D12_FEATURE_EXISTING_HEAPS => size_of::<D3D12_FEATURE_DATA_EXISTING_HEAPS>(),
        D3D12_FEATURE_D3D12_OPTIONS4 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS4>(),
        D3D12_FEATURE_D3D12_OPTIONS5 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>(),
        D3D12_FEATURE_D3D12_OPTIONS6 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS6>(),
        D3D12_FEATURE_D3D12_OPTIONS7 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>(),
        D3D12_FEATURE_D3D12_OPTIONS8 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS8>(),
        D3D12_FEATURE_D3D12_OPTIONS9 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS9>(),
        D3D12_FEATURE_D3D12_OPTIONS10 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS10>(),
        D3D12_FEATURE_D3D12_OPTIONS11 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>(),
        D3D12_FEATURE_D3D12_OPTIONS12 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>(),
        D3D12_FEATURE_D3D12_OPTIONS13 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS13>(),
        _ => return None,
    })
}

/// Apply `config.raytracing` / `XCL_RAYTRACING` override (if any) to a successful
/// `CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5)` result and log the reported tier once.
unsafe fn adjust_raytracing_options5(feature: i32, data: *mut u8, size: u32) {
    if D3D12_FEATURE(feature) != D3D12_FEATURE_D3D12_OPTIONS5 || data.is_null() || (size as usize) < std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() {
        return;
    }
    let opts = &mut *(data as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS5);
    let host = opts.RaytracingTier.0;
    if let Some(tier) = crate::state::config().raytracing_tier_override() {
        opts.RaytracingTier = D3D12_RAYTRACING_TIER(tier as i32);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: CheckFeatureSupport(D3D12_OPTIONS5) → RaytracingTier {} (host {})", opts.RaytracingTier.0, host);
    }
}

/// CheckFeatureSupport(feature, data, size): console structures can be larger than the PC's
/// (XDefiant: D3D12_OPTIONS with 0x40 bytes, 0x3C on the PC – and the title stops on the
/// error). A size the PC refuses is retried with the PC's, the console's extra bytes zero.
///
/// A query about one of the console's own formats (116 – 120, `device_x::pc_view_format`) asks
/// the PC about the format the resource gets there, and the caller's format field is put back.
/// A retail title asked FORMAT_SUPPORT of D16_UNORM_S8_UINT (118), the PC called it unsupported,
/// and the title stopped on its assertion.
unsafe extern "system" fn dev_check_feature_support(this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let f: CheckFeatureSupportFn = std::mem::transmute(orig_fn(this, slot_of(this, "CheckFeatureSupport")));
    check_feature_support_x(f, this, feature, data, size)
}

type CheckFeatureSupportFn = unsafe extern "system" fn(*mut c_void, i32, *mut u8, u32) -> HRESULT;

/// `dev_check_feature_support` with the PC device's own CheckFeatureSupport `f`.
pub(super) unsafe fn check_feature_support_x(f: CheckFeatureSupportFn, this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let console = console_format_query(feature, data, size);
    let hr = check_feature_support_pc(f, this, feature, data, size);
    if let Some(format) = console {
        (data as *mut u32).write_unaligned(format);
    }
    hr
}

/// The features whose data starts with a DXGI_FORMAT the caller asks about.
fn feature_takes_format(feature: i32) -> bool {
    matches!(
        D3D12_FEATURE(feature),
        D3D12_FEATURE_FORMAT_SUPPORT | D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS | D3D12_FEATURE_FORMAT_INFO | D3D12_FEATURE_PLACED_RESOURCE_SUPPORT_INFO
    )
}

/// For a format query (`feature_takes_format`) about a console-only format: its PC format written
/// into `data` in its place, and the console's returned (to be put back after the call).
unsafe fn console_format_query(feature: i32, data: *mut u8, size: u32) -> Option<u32> {
    if !feature_takes_format(feature) || data.is_null() || size < 4 {
        return None;
    }
    let format = (data as *const u32).read_unaligned();
    let pc = super::device_x::pc_view_format(format);
    (pc != format).then(|| {
        (data as *mut u32).write_unaligned(pc);
        format
    })
}

unsafe fn check_feature_support_pc(f: CheckFeatureSupportFn, this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let hr = f(this, feature, data, size);
    if hr >= 0 {
        adjust_raytracing_options5(feature, data, size);
        return hr;
    }
    let Some(pc) = pc_feature_size(feature).filter(|&pc| !data.is_null() && pc != size as usize) else { return hr };
    let n = pc.min(size as usize);
    let mut tmp = vec![0u8; pc];
    std::ptr::copy_nonoverlapping(data, tmp.as_mut_ptr(), n); // input fields (format, sample count …)
    let hr2 = f(this, feature, tmp.as_mut_ptr(), pc as u32);
    if hr2 >= 0 {
        std::ptr::write_bytes(data, 0, size as usize);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), data, n);
        adjust_raytracing_options5(feature, data, size);
    }
    static SEEN: Mutex<Vec<i32>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if !seen.contains(&feature) {
        seen.push(feature);
        crate::xlog_info!("D3D12.X: CheckFeatureSupport({feature}) with the console's {size} bytes, the PC's {pc} → {:#x}", hr2 as u32);
    }
    hr2
}

/// Device creation methods ending in (REFIID riid, void** ppv).
macro_rules! creator {
    ($name:ident, $method:literal, $kind:expr, [$($a:ident),*]) => {
        pub(super) unsafe extern "system" fn $name(this: *mut c_void, $($a: usize,)* riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
            type F = unsafe extern "system" fn(*mut c_void, $(as_usize!($a),)* *const GUID, *mut *mut c_void) -> HRESULT;
            let f: F = std::mem::transmute(orig_fn(this, slot_of(this, $method)));
            let mut hr = f(this, $($a,)* riid, ppv);
            if hr == E_NOINTERFACE {
                let iid = pc_iid($kind);
                hr = f(this, $($a,)* &iid, ppv);
                // an interface the PC does not have (a console one): which, once per method
                static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
                let asked = riid.as_ref().map(super::layout::guid_string).unwrap_or_default();
                let new = { let mut g = SEEN.lock().unwrap(); let n = !g.contains(&asked); if n { g.push(asked.clone()); } n };
                if new {
                    crate::xlog_info!("D3D12.X: {} for interface {asked} (not on the PC) – made as {}", $method, super::layout::guid_string(&iid));
                }
            }
            if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
                adopt(*ppv, $kind);
            }
            hr
        }
    };
}

creator!(dev_create_command_queue, "CreateCommandQueue", Kind::CommandQueue, [a1]);
creator!(dev_create_command_queue1, "CreateCommandQueue1", Kind::CommandQueue, [a1, a2]);
creator!(dev_create_command_allocator, "CreateCommandAllocator", Kind::CommandAllocator, [a1]);
creator!(dev_create_command_list, "CreateCommandList", Kind::CommandList, [a1, a2, a3, a4]);
creator!(dev_create_command_list1, "CreateCommandList1", Kind::CommandList, [a1, a2, a3]);

/// Create a raw adopted PC `ID3D12GraphicsCommandList` without wrapping it in `title_list`
/// (used for deferred mid-list split segments on the same allocator).
pub(super) unsafe fn create_raw_pc_command_list(
    dev: *mut c_void,
    ty: D3D12_COMMAND_LIST_TYPE,
    alloc: usize,
    init_pso: usize,
) -> Option<*mut c_void> {
    if dev.is_null() || alloc == 0 {
        return None;
    }
    let iid = layout::parse_guid(layout::find("ID3D12GraphicsCommandList")?.iid);
    let mut out: *mut c_void = std::ptr::null_mut();
    let hr = if is_ext(vt(dev) as usize) {
        dev_create_command_list(dev, 0, ty.0 as usize, alloc, init_pso, &iid, &mut out)
    } else {
        type F = unsafe extern "system" fn(*mut c_void, u32, D3D12_COMMAND_LIST_TYPE, usize, usize, *const GUID, *mut *mut c_void) -> HRESULT;
        let slot = layout::slot(layout::find("ID3D12Device")?, "CreateCommandList")?;
        let f: F = std::mem::transmute(*vt(dev).add(slot));
        let hr = f(dev, 0, ty, alloc, init_pso, &iid, &mut out);
        if hr >= 0 && !out.is_null() {
            adopt(out, Kind::CommandList);
        }
        hr
    };
    (hr >= 0 && !out.is_null()).then_some(out)
}
creator!(dev_create_descriptor_heap, "CreateDescriptorHeap", Kind::DescriptorHeap, [a1]);
creator!(dev_create_committed_resource, "CreateCommittedResource", Kind::Resource, [a1, a2, a3, a4, a5]);
creator!(dev_create_committed_resource1, "CreateCommittedResource1", Kind::Resource, [a1, a2, a3, a4, a5, a6]);
creator!(dev_create_committed_resource2, "CreateCommittedResource2", Kind::Resource, [a1, a2, a3, a4, a5, a6]);
creator!(dev_create_placed_resource, "CreatePlacedResource", Kind::Resource, [a1, a2, a3, a4, a5]);

/// D3D12_RESOURCE_STATES the PC knows (as `list_x::PC_STATES`)
pub(super) const PC_RESOURCE_STATES: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0xF_0000 | 0x20_0000 | 0x40_0000 | 0x80_0000 | 0x100_0000;

/// device #29 CreatePlacedResource(heap, offset, desc, state, clear, riid, resource) for console
/// code: a description the PC refuses (console layouts and flags – Fortnite's buffer in a
/// buffers-only heap) is made the PC's way (`device_x::pc_resource_desc`, the console's state
/// bits dropped). Logged with the console's values.
pub(super) unsafe extern "system" fn dev_create_placed_resource_x(this: *mut c_void, heap: usize, offset: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    // the clear value in the PC's format of the resource (D3D12 wants the two to match)
    let mut clear_buf = [0u8; 24];
    let clear = super::device_x::pc_clear(clear as *const c_void, &mut clear_buf) as usize;
    // a buffer in a heap that has unified memory (`note_heap_uma`): that memory at its offset
    if desc != 0 && crate::gpu::shader::readable(desc, 0x38) {
        let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
        let is_buffer = u32::from_le_bytes(x[0..4].try_into().unwrap()) == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
        if let Some((block, cpu_readback)) = is_buffer.then(|| heap_block(heap)).flatten() {
            let d = super::device_x::pc_resource_desc(&x);
            let st = state as u32 & PC_RESOURCE_STATES;
            if let Some(hr) = place_at_title_address(this, block + offset, &d, st, clear as *const c_void, riid, ppv) {
                if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() && cpu_readback && crate::gpu::uma::get().is_some_and(|u| u.is_sparse(block + offset)) {
                    let flags = u32::from_le_bytes(d[0x30..0x34].try_into().unwrap());
                    if st == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32 && (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32) == 0 {
                        note_sparse_readback(*ppv as usize);
                    }
                }
                return hr;
            }
        }
    }
    if let Some(hr) = own_memory_if_larger_on_the_pc(this, heap, offset, desc, state, clear, riid, ppv) {
        return hr;
    }
    let hr = dev_create_placed_resource(this, heap, offset, desc, state, clear, riid, ppv);
    if hr >= 0 || desc == 0 || !crate::gpu::shader::readable(desc, 0x38) {
        return hr;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let d = super::device_x::pc_resource_desc(&x);
    let st = super::device_x::pc_initial_state(u32::from_le_bytes(x[0x30..0x34].try_into().unwrap()), state as u32);
    let hr2 = dev_create_placed_resource(this, heap, offset, d.as_ptr() as usize, st as usize, clear, riid, ppv);
    let heap_desc = pc_heap_desc(heap);
    let pc_size = pc_allocation_size(this, &d);
    // A texture the title placed by the console's size (`get_resource_allocation_info_x`
    // answers with XG's) can be larger on the PC and run past the end of its heap, which the
    // PC refuses – Minecraft then ended (0xC0000409). It gets memory of its own, of the
    // heap's kind; the console's aliasing of that one placement is lost.
    let is_texture = u32::from_le_bytes(x[0..4].try_into().unwrap()) != D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
    let hr3 = match heap_desc {
        Some(h) if hr2 < 0 && is_texture => Some(dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, st as usize, clear, riid, ppv)),
        _ => None,
    };
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        let sizes = format!(
            "heap {:#x} bytes, PC size {:#x}{}",
            heap_desc.map_or(0, |h| h.SizeInBytes),
            pc_size,
            if heap_desc.is_some_and(|h| offset as u64 + pc_size > h.SizeInBytes) { " – past the heap's end" } else { "" }
        );
        let own = hr3.map_or(String::new(), |h| format!("; memory of its own: {h:#x}"));
        crate::xlog_info!(
            "D3D12.X: CreatePlacedResource(dimension {}, {}x{}, format {}, layout {:#x}, flags {:#x}, state {state:#x}) at {offset:#x} refused ({hr:#x}); the PC's way: {hr2:#x} ({sizes}){own}",
            u(0), u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x2C), u(0x30)
        );
    }
    hr3.unwrap_or(hr2)
}

/// A texture the title placed in a heap by the console's size (`get_resource_allocation_info_x`
/// answers with XG's), when the PC's is larger: placed at the console's offset it would cover
/// the start of whatever the title placed after it, and each would write over the other.
/// Minecraft's logo (1937x333, 0x2A0000 bytes on the console, 0x2E0000 on the PC) ran into the
/// text and 16x16 textures packed after it – magenta / black squares and a white speck along
/// its bottom edge. Such a texture gets memory of its own, of the heap's kind; the console's
/// aliasing of that one placement is lost. None: placed as the title asked.
#[allow(clippy::too_many_arguments)]
unsafe fn own_memory_if_larger_on_the_pc(this: *mut c_void, heap: usize, offset: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    if desc == 0 || !crate::gpu::shader::readable(desc, 0x38) {
        return None;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    if u32::from_le_bytes(x[0..4].try_into().unwrap()) == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32 {
        return None;
    }
    let console = crate::hle::xg::Texture::new(&x).ok()?.size();
    let d = super::device_x::pc_resource_desc(&x);
    let pc = pc_allocation_size(this, &d);
    if console == u64::MAX || pc == u64::MAX || pc <= console {
        return None;
    }
    let h = pc_heap_desc(heap)?;
    let st = super::device_x::pc_initial_state(u32::from_le_bytes(x[0x30..0x34].try_into().unwrap()), state as u32);
    let mut hr = dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, st as usize, clear, riid, ppv);
    if hr < 0 && st != 0 {
        // as in the past-the-end fallback: a state the PC refuses for this desc (e.g. UAV
        // without the UAV flag) – COMMON rather than falling back to the overlapping placement
        hr = dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, 0, clear, riid, ppv);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        crate::xlog_info!(
            "D3D12.X: CreatePlacedResource({}x{}, format {}, flags {:#x}) at {offset:#x}: {pc:#x} bytes on the PC, {console:#x} on the console – memory of its own ({hr:#x})",
            u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x30)
        );
    }
    (hr >= 0).then_some(hr)
}

/// `XCL_DEBUG_NO_STENCIL` / `XCL_DEBUG_NO_CULL` `=hash,…` (pixel shader dump hashes, hex):
/// pipelines with those pixel shaders draw without their stencil test / culling – diagnostics
/// for draws the fixed-function stages reject.
fn debug_pso_switch(var: &str, ps: u64) -> bool {
    ps != 0 && std::env::var(var).is_ok_and(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).any(|h| h == ps))
}

/// The PC description of a heap XCL adopted (ID3D12Heap::GetDesc, slot 8).
unsafe fn pc_heap_desc(heap: usize) -> Option<D3D12_HEAP_DESC> {
    if !is_adopted(heap as *mut c_void) {
        return None;
    }
    type F = unsafe extern "system" fn(*mut c_void, *mut D3D12_HEAP_DESC) -> *mut D3D12_HEAP_DESC;
    let f: F = std::mem::transmute(orig_fn(heap as *mut c_void, 8));
    let mut h = D3D12_HEAP_DESC::default();
    f(heap as *mut c_void, &mut h);
    Some(h)
}

/// The PC driver's size for a (PC) resource description (GetResourceAllocationInfo, slot 25).
unsafe fn pc_allocation_size(device: *mut c_void, d: &[u8; 0x38]) -> u64 {
    type F = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
    let f: F = std::mem::transmute(orig_fn(device, 25));
    let mut r = [0u64; 2];
    f(device, &mut r, 0, 1, d);
    r[0]
}
creator!(dev_create_placed_resource1, "CreatePlacedResource1", Kind::Resource, [a1, a2, a3, a4, a5]);
creator!(pc_create_reserved_resource, "CreateReservedResource", Kind::Resource, [a1, a2, a3]);
creator!(pc_create_reserved_resource1, "CreateReservedResource1", Kind::Resource, [a1, a2, a3, a4]);

/// Resources made with CreateReservedResource(1), with their description: they have no heap, so
/// `res_map` asks none of them (GetHeapProperties on one is an invalid call too, debug layer
/// 901). The description guards against an address a later resource reuses without the last
/// Release having been seen.
static RESERVED: Mutex<Option<HashMap<usize, D3D12_RESOURCE_DESC>>> = Mutex::new(None);

unsafe fn desc_of(res: *mut c_void) -> Option<D3D12_RESOURCE_DESC> {
    ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc())
}

unsafe fn note_reserved(hr: HRESULT, ppv: *mut *mut c_void) -> HRESULT {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        if let Some(d) = desc_of(*ppv) {
            RESERVED.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, d);
        }
    }
    hr
}

unsafe fn is_reserved(res: *mut c_void) -> bool {
    let kept = RESERVED.lock().unwrap().as_ref().and_then(|m| m.get(&(res as usize)).copied());
    kept.is_some_and(|k| desc_of(res) == Some(k))
}

/// CreateReservedResource
pub(super) unsafe extern "system" fn dev_create_reserved_resource(this: *mut c_void, a1: usize, a2: usize, a3: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    note_reserved(pc_create_reserved_resource(this, a1, a2, a3, riid, ppv), ppv)
}

/// CreateReservedResource1
pub(super) unsafe extern "system" fn dev_create_reserved_resource1(this: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    note_reserved(pc_create_reserved_resource1(this, a1, a2, a3, a4, riid, ppv), ppv)
}
creator!(dev_create_heap_pc, "CreateHeap", Kind::Heap, [a1]);

/// D3D12_HEAP_FLAGS the PC knows: SHARED 0x1, DENY_BUFFERS 0x4, ALLOW_DISPLAY 0x8,
/// SHARED_CROSS_ADAPTER 0x20, DENY_RT_DS_TEXTURES 0x40, DENY_NON_RT_DS_TEXTURES 0x80, then
/// 0x100 … 0x2000 (HARDWARE_PROTECTED … TOOLS_USE_MANUAL_WRITE_TRACKING); 0x2 and 0x10 are
/// the console's (Fortnite: 0xC2)
const PC_HEAP_FLAGS: u32 = 0x3FED;

/// CreateHeap(desc, riid, heap): a console description the PC refuses (console flags in the
/// high bits, alignments other than 64 KiB / 4 MiB, console heap types – Fortnite's distance
/// field atlas heap) is made the PC's way: known flags only, the alignment the PC has that
/// covers it, an unknown type as DEFAULT. Logged with the console's values.
pub(super) unsafe extern "system" fn dev_create_heap(this: *mut c_void, desc: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = dev_create_heap_pc(this, desc, riid, ppv);
    if hr >= 0 {
        note_heap_uma(desc, ppv);
    }
    if hr >= 0 || desc == 0 || !crate::gpu::shader::readable(desc, std::mem::size_of::<D3D12_HEAP_DESC>()) {
        return hr;
    }
    let mut d: D3D12_HEAP_DESC = std::ptr::read_unaligned(desc as *const D3D12_HEAP_DESC);
    let orig = (d.SizeInBytes, d.Properties.Type.0, d.Alignment, d.Flags.0);
    d.Flags = D3D12_HEAP_FLAGS(d.Flags.0 & PC_HEAP_FLAGS as i32);
    d.Alignment = if d.Alignment <= 0x1_0000 { 0 } else { D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT as u64 };
    if !(1..=5).contains(&d.Properties.Type.0) {
        d.Properties = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    }
    d.SizeInBytes = d.SizeInBytes.div_ceil(0x1_0000) * 0x1_0000;
    let hr2 = dev_create_heap_pc(this, &d as *const _ as usize, riid, ppv);
    if hr2 >= 0 {
        note_heap_uma(desc, ppv);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_info!(
            "D3D12.X: CreateHeap({:#x} bytes, type {}, alignment {:#x}, flags {:#x}) refused ({hr:#x}); as type {}, alignment {:#x}, flags {:#x}: {hr2:#x}",
            orig.0, orig.1, orig.2, orig.3, d.Properties.Type.0, d.Alignment, d.Flags.0
        );
    }
    hr2
}
creator!(dev_create_heap1_pc, "CreateHeap1", Kind::Heap, [a1, a2]);

pub(super) unsafe extern "system" fn dev_create_heap1(this: *mut c_void, desc: usize, session: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = dev_create_heap1_pc(this, desc, session, riid, ppv);
    if hr >= 0 {
        note_heap_uma(desc, ppv);
    }
    hr
}

struct HeapUma {
    size: usize,
    align: usize,
    /// Small explicit `D3D12_HEAP_TYPE_READBACK` heap: prefers coherent Window 3 RAM.
    prefer_coherent: bool,
    /// `READBACK` or `CUSTOM + WRITE_BACK`: if backed by sparse VRAM, `COPY_DEST` buffers placed
    /// in it read back on `Map(0, NULL, ...)`.
    cpu_readback: bool,
    block: Option<usize>,
}

/// heap → its backing block (allocated lazily on the first placed buffer; sparse VRAM for
/// DEFAULT/UPLOAD/CUSTOM heaps, coherent committed window for small explicit READBACK heaps)
static HEAP_UMA: Mutex<Option<HashMap<usize, HeapUma>>> = Mutex::new(None);

/// A heap the title made that may hold buffers: a block of unified memory as large stands for
/// it. On the console every heap is memory the CPU and the GPU address alike, and console code
/// reads buffers (root descriptors, V#s) from that memory – a PC heap's buffer has a GPU
/// address only (Minecraft's constant and vertex buffers: a heap of its own; its UI drew
/// nothing – the constants read as zero). Buffers placed in the heap are placed in the block
/// (`dev_create_placed_resource_x`); textures stay in the PC heap. The block is taken when the
/// first buffer is placed (most heaps hold textures only: a block for each filled the 2 GiB of
/// unified memory, and draws then found no room for their arguments – the image flickered);
/// none when unified memory is full (the PC heap as before).
unsafe fn note_heap_uma(desc: usize, ppv: *mut *mut c_void) {
    const DENY_BUFFERS: i32 = 0x4;
    if super::title_textures::is_own_work() || desc == 0 || ppv.is_null() || (*ppv).is_null() || !crate::gpu::shader::readable(desc, std::mem::size_of::<D3D12_HEAP_DESC>()) {
        return;
    }
    let d: D3D12_HEAP_DESC = std::ptr::read_unaligned(desc as *const D3D12_HEAP_DESC);
    if d.Flags.0 & DENY_BUFFERS != 0 || d.SizeInBytes == 0 || UMA_BUFFER.lock().unwrap().is_empty() {
        return;
    }
    let size = d.SizeInBytes as usize;
    let is_rb = d.Properties.Type == D3D12_HEAP_TYPE_READBACK;
    let is_custom_wb = d.Properties.Type == D3D12_HEAP_TYPE_CUSTOM && d.Properties.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    HEAP_UMA.lock().unwrap().get_or_insert_with(HashMap::new).insert(
        *ppv as usize,
        HeapUma {
            size,
            align: (d.Alignment as usize).max(0x1_0000),
            prefer_coherent: is_rb && size <= COHERENT_READBACK_MAX,
            cpu_readback: is_rb || is_custom_wb,
            block: None,
        },
    );
}

/// The block of unified memory standing for `heap` and whether it may hold CPU-readback buffers
/// (taken now if it has none yet; None: not a heap that takes one, or unified memory is full).
fn heap_block(heap: usize) -> Option<(usize, bool)> {
    let (size, align, prefer_coherent, cpu_readback) = {
        let g = HEAP_UMA.lock().unwrap();
        let e = g.as_ref()?.get(&heap)?;
        if let Some(p) = e.block {
            return Some((p, e.cpu_readback));
        }
        (e.size, e.align, e.prefer_coherent, e.cpu_readback)
    };
    let u = crate::gpu::uma::get()?;
    let rounded = size.div_ceil(0x1_0000) * 0x1_0000;
    let use_coherent = prefer_coherent && has_coherent_headroom(u, rounded);
    let p = if !use_coherent && u.has_sparse() {
        unsafe { super::sparse::alloc(size, align) }
            .map(|(a, _)| a)
            .or_else(|| u.alloc(rounded, align).map(|a| a as usize))
    } else {
        u.alloc(rounded, align)
            .map(|a| a as usize)
            .or_else(|| u.has_sparse().then(|| unsafe { super::sparse::alloc(size, align) }.map(|(a, _)| a)).flatten())
    };
    let Some(p) = p else {
        static FULL: AtomicU64 = AtomicU64::new(0);
        if FULL.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!(
                "D3D12.X: heap of {size:#x} bytes: unified memory is full ({} MB sparse, {} MB coherent used) – its buffers stay PC-only",
                u.sparse_used() >> 20,
                u.used() >> 20
            );
        }
        return None;
    };
    let mut g = HEAP_UMA.lock().unwrap();
    let Some(e) = g.as_mut().and_then(|m| m.get_mut(&heap)) else {
        drop(g);
        if u.is_sparse(p) {
            unsafe { super::sparse::free(p) };
        } else {
            u.free(p);
        }
        return None;
    };
    if let Some(existing) = e.block {
        let cr = e.cpu_readback;
        drop(g);
        if u.is_sparse(p) {
            unsafe { super::sparse::free(p) };
        } else {
            u.free(p);
        }
        return Some((existing, cr));
    }
    e.block = Some(p);
    drop(g);
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 4 {
        let kind = if u.is_sparse(p) { "sparse VRAM" } else { "coherent RAM" };
        crate::xlog_info!("D3D12.X: heap {heap:#x} ({size:#x} bytes) holds buffers: in {kind} at {p:#x}");
    }
    Some((p, cpu_readback))
}

/// ID3D12Heap::Release: its block of unified memory goes with it.
unsafe extern "system" fn mem_heap_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        let gone = HEAP_UMA.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize))).and_then(|e| e.block);
        if let (Some(p), Some(u)) = (gone, crate::gpu::uma::get()) {
            if u.is_sparse(p) {
                super::sparse::free(p);
            } else {
                u.free(p);
            }
        }
    }
    n
}
creator!(dev_create_fence_pc, "CreateFence", Kind::Fence, [a1, a2]);

/// fences created through the device (AddRef'd: they are released at the end only)
/// (a read-write lock: `is_fence` reads it on every queue call, and `end_title` reads it with
/// the title's threads suspended – a reader that holds it then blocks no other reader)
static FENCES: std::sync::RwLock<Vec<usize>> = std::sync::RwLock::new(Vec::new());

/// CreateFence: as on the PC; the fence is remembered for `end_title`
unsafe extern "system" fn dev_create_fence(this: *mut c_void, value: usize, flags: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = dev_create_fence_pc(this, value, flags, riid, ppv);
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        let add: RefFn = std::mem::transmute(*vt(*ppv).add(1));
        add(*ppv);
        let n = {
            let mut g = FENCES.write().unwrap();
            g.push(*ppv as usize);
            g.len()
        };
        // `XCL_WATCH_FENCE=n`: writes over the n-th fence's first 8 bytes (its vtable) are logged
        if std::env::var("XCL_WATCH_FENCE").ok().and_then(|v| v.parse::<usize>().ok()) == Some(n) {
            crate::xlog_info!("D3D12.X: fence #{n} {:p} (initial value {value:#x}, flags {flags:#x}) watched", *ppv);
            crate::eh::watch_writes(*ppv as usize);
        }
    }
    hr
}

/// End the title on purpose without leaving the process stuck in the driver: every fence it
/// created is signalled to the largest value (GPU waits on values the title would have signalled
/// later complete), the queues drain, and the process ends without running DLL detach code.
pub fn end_title(code: u32) -> ! {
    unsafe {
        // a last resort: the process ends within 15 s whatever happens below (a suspended
        // thread may hold a lock the drain needs)
        let (tx, rx) = std::sync::mpsc::channel();
        let _ = std::thread::Builder::new().name("xcl-end-watchdog".into()).spawn(move || {
            let _ = tx.send(windows_sys::Win32::System::Threading::GetCurrentThreadId());
            std::thread::sleep(std::time::Duration::from_secs(15));
            windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
        });
        let watchdog = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap_or(0);
        crate::log::flush();
        // the title's threads stop first: with its fences at the maximum a running title thinks
        // the GPU is done and resets command allocators the GPU still runs (Fortnite: a device
        // hang at the end of a run)
        suspend_other_threads(watchdog);
        for &f in FENCES.read().unwrap_or_else(|e| e.into_inner()).iter() {
            let fr = f as *mut c_void;
            if let Some(f) = ID3D12Fence::from_raw_borrowed(&fr) {
                let _ = f.Signal(u64::MAX);
            }
        }
        drain_gpu();
        crate::log::flush();
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
    }
    std::process::abort()
}

/// Every thread of the process but this one and `keep` (XCL's end watchdog), suspended – twice
/// over: the title may start threads meanwhile.
unsafe fn suspend_other_threads(keep: u32) {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, SuspendThread, THREAD_SUSPEND_RESUME};
    let (pid, me) = (GetCurrentProcessId(), GetCurrentThreadId());
    let mut done: Vec<u32> = Vec::with_capacity(1024);
    for _ in 0..2 {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return;
        }
        let mut te: THREADENTRY32 = std::mem::zeroed();
        te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut te) != 0;
        while ok {
            let tid = te.th32ThreadID;
            if te.th32OwnerProcessID == pid && tid != me && tid != keep && !done.contains(&tid) {
                let h = OpenThread(THREAD_SUSPEND_RESUME, 0, tid);
                if !h.is_null() {
                    SuspendThread(h);
                    CloseHandle(h);
                }
                if done.len() < done.capacity() {
                    done.push(tid);
                }
            }
            ok = Thread32Next(snap, &mut te) != 0;
        }
        CloseHandle(snap);
    }
}
creator!(dev_create_query_heap, "CreateQueryHeap", Kind::QueryHeap, [a1]);
creator!(dev_create_command_signature, "CreateCommandSignature", Kind::CommandSignature, [a1, a2]);
creator!(dev_create_pipeline_library, "CreatePipelineLibrary", Kind::PipelineLibrary, [a1, a2]);
creator!(dev_create_state_object, "CreateStateObject", Kind::StateObject, [a1]);

unsafe extern "system" fn dev_create_shader_cache_session(
    this: *mut c_void,
    desc: *const D3D12_SHADER_CACHE_SESSION_DESC,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, *const D3D12_SHADER_CACHE_SESSION_DESC, *const GUID, *mut *mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CreateShaderCacheSession")));
    let hr = f(this, desc, riid, ppv);
    if hr >= 0 {
        return hr;
    }
    let fb = super::shader_cache_session::create_fallback(this, desc, riid as *const windows::core::GUID, ppv);
    if fb >= 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            crate::xlog_info!("D3D12.X: CreateShaderCacheSession failed on the PC runtime ({hr:#x}) – using an in-memory session");
        });
        fb
    } else {
        hr
    }
}

// ---------------------------------------------------------------------------------------
// pipeline states: capture + fallback
// ---------------------------------------------------------------------------------------

struct Dummy {
    vs: Vec<u8>,
    ps: Vec<u8>,
    cs: Vec<u8>,
}

/// Stand-in mesh shader and pixel shader (DXIL, as mesh pipelines need): one degenerate
/// triangle (DXC wants every output written), nothing drawn.
fn dummy_mesh() -> Option<&'static (Vec<u8>, Vec<u8>)> {
    static D: OnceLock<Option<(Vec<u8>, Vec<u8>)>> = OnceLock::new();
    D.get_or_init(|| {
        use crate::gpu::recomp::dxc::compile;
        let ms = "struct V { float4 p : SV_Position; };\n[outputtopology(\"triangle\")]\n[numthreads(1, 1, 1)]\nvoid main(out vertices V v[1], out indices uint3 t[1]) { SetMeshOutputCounts(1, 1); v[0].p = float4(0, 0, 0, 1); t[0] = uint3(0, 0, 0); }\n";
        let ps = "float4 main() : SV_Target { return float4(1, 0, 1, 1); }\n";
        match (compile(ms, "main", "ms_6_5", &[]), compile(ps, "main", "ps_6_0", &[])) {
            (Ok(m), Ok(p)) => Some((m, p)),
            (m, p) => {
                crate::xlog_warn!("D3D12.X: stand-in mesh shaders: {:?} {:?}", m.err(), p.err());
                None
            }
        }
    })
    .as_ref()
}

fn compile(src: &str, target: &str) -> Vec<u8> {
    unsafe {
        let mut blob = None;
        let t = format!("{target}\0");
        let r = D3DCompile(
            src.as_ptr() as *const c_void,
            src.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(b"main\0".as_ptr()),
            PCSTR(t.as_ptr()),
            0,
            0,
            &mut blob,
            None,
        );
        match (r, blob) {
            (Ok(()), Some(b)) => std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec(),
            _ => Vec::new(),
        }
    }
}

/// Stand-in shaders used when the PC driver rejects a console shader.
fn dummy() -> &'static Dummy {
    static D: OnceLock<Dummy> = OnceLock::new();
    D.get_or_init(|| Dummy {
        vs: compile("float4 main(uint id : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }", "vs_5_0"),
        ps: compile("float4 main() : SV_Target { return float4(1, 0, 1, 1); }", "ps_5_0"),
        cs: compile("[numthreads(1, 1, 1)] void main() {}", "cs_5_0"),
    })
}

#[cfg(test)]
pub(super) fn dummy_cs_bytecode() -> &'static [u8] {
    &dummy().cs
}

fn bytecode(b: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as *const c_void, BytecodeLength: b.len() }
}

fn note_fallback(what: &str, hr: HRESULT, ok: bool) {
    let n = if ok { FALLBACK_PSO.fetch_add(1, Ordering::Relaxed) } else { FAILED_PSO.fetch_add(1, Ordering::Relaxed) };
    if n < 5 || n % 1000 == 0 {
        if ok {
            crate::xlog_warn!("D3D12.X: {what} rejected by the driver ({hr:#x}) – fallback shaders used (#{})", n + 1);
        } else {
            crate::xlog_error!("D3D12.X: {what} rejected ({hr:#x}), the fallback version too (#{})", n + 1);
        }
    }
}

type PsoFn = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;

unsafe fn finish_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut hr = driver_pso(f, this, desc, riid, ppv);
    if hr == E_NOINTERFACE {
        let iid = pc_iid(Kind::PipelineState);
        hr = driver_pso(f, this, desc, &iid, ppv);
    }
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        adopt(*ppv, Kind::PipelineState);
    }
    hr
}

thread_local! {
    /// the driver ran out of memory on a pipeline with heap windows (`with_switch_fallback`)
    static HEAP_WINDOW_OOM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// heap-window driver calls still to fail with E_OUTOFMEMORY (`inject_heap_window_oom`)
static INJECTED_OOM: AtomicU64 = AtomicU64::new(0);
/// pipelines built in the switch form after the driver refused them with heap windows
static SWITCH_FORM_RETRIES: AtomicU64 = AtomicU64::new(0);

/// Tests: the next `n` driver pipeline calls with heap windows fail with E_OUTOFMEMORY without
/// reaching the driver (0 stops it), as NVIDIA 616.92 refuses some big retail programs (#156).
#[doc(hidden)]
pub fn inject_heap_window_oom(n: u64) {
    INJECTED_OOM.store(n, Ordering::SeqCst);
}

/// Pipelines built in the switch form because the driver refused them with heap windows.
pub fn switch_form_retries() -> u64 {
    SWITCH_FORM_RETRIES.load(Ordering::Relaxed)
}

/// A driver pipeline call; E_OUTOFMEMORY while this thread translates with heap windows is
/// noted for `with_switch_fallback`.
unsafe fn driver_pso(f: PsoFn, this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let heap = crate::gpu::recomp::pso::mem_windows() & crate::gpu::recomp::HEAP_WINDOWS != 0;
    let injected = heap && INJECTED_OOM.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1)).is_ok();
    let hr = if injected { E_OUTOFMEMORY } else { f(this, desc, riid, ppv) };
    if hr == E_OUTOFMEMORY && heap {
        HEAP_WINDOW_OOM.with(|o| o.set(true));
    }
    hr
}

/// Build a translated pipeline (`build`: a `translated_*` path); when the driver ran out of
/// memory on it with heap windows, translate it again in the switch form and build that.
/// NVIDIA 616.92 refuses some big programs whose heap index depends on the window with
/// E_OUTOFMEMORY in under a second, and builds the same programs in the switch form (#156).
fn with_switch_fallback<R>(mut build: impl FnMut() -> R) -> R {
    HEAP_WINDOW_OOM.with(|o| o.set(false));
    let r = build();
    if !HEAP_WINDOW_OOM.with(|o| o.replace(false)) {
        return r;
    }
    if crate::gpu::uma::get().is_some_and(|u| u.shader_windows() > crate::gpu::uma::MAX_WINDOWS) {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            crate::xlog_warn!("D3D12.X: the driver refused an expanded-window pipeline; switch-form retry requires XCL_UMA_WINDOWS=4");
        });
        return r;
    }
    let n = SWITCH_FORM_RETRIES.fetch_add(1, Ordering::Relaxed);
    if n < 5 || n % 100 == 0 {
        crate::xlog_warn!("D3D12.X: the driver ran out of memory on a pipeline with heap windows – building it in the switch form ({} so far)", n + 1);
    }
    crate::gpu::recomp::pso::in_switch_form(build)
}

thread_local! {
    /// translation errors noted during a cached-only attempt (`attempt`)
    static ATTEMPT_NOTES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run a translated-pipeline path without compiling (`dxc::cached_only`): (its result, whether
/// a shader was not compiled yet). Errors it noted count only when nothing was missing.
fn attempt<R>(f: impl FnOnce() -> R) -> (R, bool) {
    let (r, missed) = crate::gpu::recomp::dxc::cached_only(f);
    let notes = ATTEMPT_NOTES.with(|n| std::mem::take(&mut *n.borrow_mut()));
    if !missed {
        notes.into_iter().for_each(note_translation_error);
    }
    (r, missed)
}

fn note_translation_error(e: String) {
    // a refusal `with_switch_fallback` builds again in the switch form (it says so itself)
    if HEAP_WINDOW_OOM.with(|o| o.get()) {
        return;
    }
    if crate::gpu::recomp::dxc::in_cached_only() {
        ATTEMPT_NOTES.with(|n| n.borrow_mut().push(e));
        return;
    }
    // each different reason once (the first five only hid a retail game's "mesh pipeline with
    // AS" behind five earlier ones)
    let mut g = TRANSLATE_ERRORS.lock().unwrap();
    if g.len() < 40 && !g.contains(&e) {
        crate::xlog_warn!("D3D12.X: console shaders in the PSO without a translation: {e}");
        g.push(e);
    }
}

unsafe fn shader_bytes<'a>(b: &D3D12_SHADER_BYTECODE) -> Option<&'a [u8]> {
    let (p, n) = (b.pShaderBytecode as usize, b.BytecodeLength);
    (p != 0 && n > 0 && crate::gpu::shader::readable(p, n)).then(|| std::slice::from_raw_parts(p as *const u8, n))
}

/// Root signature of every pipeline built from translated console shaders (static samplers
/// make it per pipeline): the draw binds it instead of the title's.
static PSO_RS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

fn note_pso_rs(pso: *mut c_void, rs: *mut c_void) {
    if !pso.is_null() {
        PSO_RS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso as usize, rs as usize);
    }
}

/// user SGPRs of the console CS of a translated compute pipeline
static PSO_USER_SGPRS: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);

/// How many user SGPRs the console compute shader of a translated pipeline has (None: not
/// known). What is above the driver's four holds root arguments (`console_list::inline_for`).
pub fn pso_user_sgprs(pso: usize) -> Option<u32> {
    let pso = super::pso_async::resolve(pso);
    PSO_USER_SGPRS.lock().unwrap().as_ref()?.get(&pso).copied()
}

/// The root signature a translated pipeline state was created with.
pub fn pso_root_signature(pso: usize) -> Option<usize> {
    let pso = super::pso_async::resolve(pso);
    PSO_RS.lock().unwrap().as_ref()?.get(&pso).copied()
}

/// ABI root signatures with static samplers, by sampler set (owned here, never released).
static ABI_RS_SAMPLERS: Mutex<Option<HashMap<Vec<u32>, usize>>> = Mutex::new(None);

/// The XCL ABI root signature with a pipeline's static samplers (the plain one without).
unsafe fn abi_rs_for(this: *mut c_void, samplers: &[crate::gpu::recomp::pso::StaticSampler]) -> Option<*mut c_void> {
    if samplers.is_empty() {
        return abi_rs_raw(this);
    }
    let key: Vec<u32> = samplers.iter().flat_map(|(w, c)| w.iter().copied().chain([*c as u32])).collect();
    let mut g = ABI_RS_SAMPLERS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(&rs) = map.get(&key) {
        return Some(rs as *mut c_void);
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    match crate::gpu::recomp::pso::abi_root_signature_with(dev, true, samplers) {
        Ok(rs) => {
            let raw = rs.into_raw();
            map.insert(key, raw as usize);
            Some(raw)
        }
        Err(e) => {
            note_translation_error(format!("XCL ABI root signature with static samplers: {e}"));
            None
        }
    }
}

/// XCL ABI root signature, created once on the backend device.
unsafe fn abi_rs_raw(this: *mut c_void) -> Option<*mut c_void> {
    let cur = ABI_RS.load(Ordering::Acquire);
    if cur != 0 {
        return Some(cur as *mut c_void);
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    match crate::gpu::recomp::pso::abi_root_signature(dev, true) {
        Ok(rs) => {
            let raw = rs.into_raw();
            match ABI_RS.compare_exchange(0, raw as usize, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => Some(raw),
                Err(other) => {
                    drop(ID3D12RootSignature::from_raw(raw));
                    Some(other as *mut c_void)
                }
            }
        }
        Err(e) => {
            note_translation_error(format!("XCL ABI root signature: {e}"));
            None
        }
    }
}

/// A graphics pipeline whose VS (and PS) are console shaders: recompiled VS/PS + the XCL
/// shader ABI root signature. None: not console shaders, or not translatable (the caller
/// goes on with the title's description and the stand-in fallback).
/// A compute pipeline with a console CS: `None` = not a console shader, `Some(Err)` = console
/// shader the recompiler could not take, `Some(Ok(hr))` = translated pipeline created (or the
/// driver's verdict on it).
/// DXIL size from which a pipeline's driver compile is "huge" (default 256 KB, `XCL_HUGE_PSO_BYTES`):
/// a compute one goes to the background, and any of them built on an `xcl-pso*` worker waits
/// for one of the `XCL_HUGE_PSO_CONCURRENCY` slots. NVIDIA's compiler (nvgpucomp64) takes
/// minutes and ~1 GB for a 1 – 2 MB program; ~20 of them at once (one per worker) were slower
/// in total than 1 – 2 at a time. Small programs (ms each) never wait.
fn huge_pso_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_count(std::env::var("XCL_HUGE_PSO_BYTES").ok().as_deref(), 256 << 10, 1, usize::MAX))
}

/// Huge driver compiles at once on the workers (default 1, `XCL_HUGE_PSO_CONCURRENCY`).
fn huge_pso_concurrency() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_count(std::env::var("XCL_HUGE_PSO_CONCURRENCY").ok().as_deref(), 1, 1, 64))
}

/// A count from the environment: `default` when unset or not a number, clamped to the
/// inclusive range `min..=max`.
fn env_count(v: Option<&str>, default: usize, min: usize, max: usize) -> usize {
    v.and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(default).clamp(min, max)
}

/// A counting semaphore: background driver compiles of big programs at once (`most` 0: the
/// `huge_pso_concurrency` of the environment).
struct Slots {
    n: Mutex<usize>,
    cv: std::sync::Condvar,
    most: usize,
}

impl Slots {
    fn most(&self) -> usize {
        if self.most == 0 { huge_pso_concurrency() } else { self.most }
    }
}

struct SlotGuard(&'static Slots);

impl Drop for SlotGuard {
    fn drop(&mut self) {
        *self.0.n.lock().unwrap() -= 1;
        self.0.cv.notify_one();
    }
}

static BIG_COMPILES: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 0 };
const GIANT_DRIVER_COMPILE: usize = 4 << 20;
static GIANT_COMPILES: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 1 };

fn compile_slots(bytes: usize) -> &'static Slots {
    if bytes >= GIANT_DRIVER_COMPILE { &GIANT_COMPILES } else { &BIG_COMPILES }
}

/// A driver compile of `bytes` of DXIL through `build`: a huge one on an `xcl-pso*` worker
/// waits for its slot first (`big_compile_turn`; not a title thread's, nor one the title waits
/// for). Logs its time, size and whether it was gated.
fn driver_compile<R>(kind: &str, bytes: usize, gate: bool, build: impl FnOnce() -> R) -> R {
    driver_compile_identified(kind, bytes, gate, None, build).0
}

/// `XCL_PSO_TRACE=1`: every driver compile is logged, not only huge ones.
fn pso_trace() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("XCL_PSO_TRACE").is_ok_and(|v| v.trim() == "1"))
}

/// `driver_compile`, with the console record's hash (`fnv64` of the shader container, as in
/// `[SHADER]` lines and `xcl shaders --measure-pso`) and whether a driver blob was supplied.
/// Also returns the time spent in the driver alone (not waiting for a slot).
fn driver_compile_identified<R>(kind: &str, bytes: usize, gate: bool, record: Option<(u64, bool)>, build: impl FnOnce() -> R) -> (R, std::time::Duration) {
    let huge = gate && bytes >= huge_pso_bytes() && std::thread::current().name().is_some_and(|n| n.starts_with("xcl-pso"));
    let waited = std::time::Instant::now();
    let slot = huge.then(|| big_compile_turn(compile_slots(bytes), || big_compile_commit_free(big_compile_headroom(bytes)), super::pso_async::build_is_wanted)).flatten();
    let waited = waited.elapsed();
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    let log = pso_trace() || bytes >= huge_pso_bytes();
    if log {
        let record = record.map_or_else(String::new, |(hash, cached)| {
            format!(", record {hash:016x}, driver blob {}", if cached { "supplied" } else { "absent" })
        });
        crate::xlog_info!("D3D12.X: driver compile start #{sequence}: {kind} PSO, {bytes} B DXIL{record}");
    }
    let started = std::time::Instant::now();
    let r = build();
    let took = started.elapsed();
    if log {
        crate::xlog_info!("D3D12.X: driver compile finish #{sequence}: {kind} PSO, {bytes} B DXIL, {} ms{}", took.as_millis(), gate_note(bytes >= huge_pso_bytes(), huge, slot.is_some(), waited));
    }
    (r, took)
}

/// How a logged driver compile went through the huge-pipeline gate.
fn gate_note(huge_size: bool, gated: bool, had_slot: bool, waited: std::time::Duration) -> String {
    match (huge_size, gated, had_slot) {
        (false, _, _) => String::new(),
        (true, true, true) => format!(", gated (waited {} ms)", waited.as_millis()),
        (true, true, false) => ", huge, wanted at once (no slot)".to_string(),
        (true, false, _) => ", huge, ungated here (a title thread's, cached, or gated by its caller)".to_string(),
    }
}

/// A cached-blob attempt should be quick and must not take a cold-compile slot. Its retry
/// without the rejected blob is a cold compile and does take one.
fn gate_compute_driver_compile(has_cached_blob: bool) -> bool {
    !has_cached_blob
}

/// Before a big driver compile in the background: its turn – at most the configured number at
/// once, and the commit headroom it asks for (`commit_free`, from `big_compile_headroom`; the
/// title loading next to it must not run out). Unless one of the title's threads waits for the
/// pipeline (`wanted`): then at once and without a slot (None), as when that thread builds it
/// itself. Holding such a build back holds the title back: on a machine with less than the
/// headroom free throughout a retail title's world loading, the first one would not go on and
/// keeps the slot, until the title's watchdog ends it.
fn big_compile_turn(slots: &'static Slots, commit_free: impl Fn() -> bool, wanted: impl Fn() -> bool) -> Option<SlotGuard> {
    const LOOK: std::time::Duration = std::time::Duration::from_millis(50);
    let slot = {
        let mut n = slots.n.lock().unwrap();
        loop {
            if wanted() {
                return None;
            }
            if *n < slots.most() {
                break;
            }
            n = slots.cv.wait_timeout(n, LOOK).unwrap().0;
        }
        *n += 1;
        SlotGuard(slots)
    };
    while !commit_free() {
        if wanted() {
            return None;
        }
        std::thread::sleep(LOOK);
    }
    Some(slot)
}

/// Commit headroom a big driver compile waits for: 3 GB for the class that starts at
/// `huge_pso_bytes()` (each takes ~1.5 GB in the driver), and ~512× the DXIL size above 6 MiB
/// of it — the driver's compile of a 29.8 MB program grew past 14 GB free before it was stopped
/// (#1269), and programs of ~1 MB took several GB each on an 8 GB card (same issue). The floor
/// leaves every program up to 6 MiB of DXIL waiting for exactly what it waited for before.
fn big_compile_headroom(dxil_len: usize) -> u64 {
    (3u64 << 30).max(dxil_len as u64 * 512)
}

/// That much commit is free (true when that is not known).
fn big_compile_commit_free(need: u64) -> bool {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    unsafe { GlobalMemoryStatusEx(&mut m) == 0 || m.ullAvailPageFile >= need }
}

#[cfg(test)]
mod big_compile_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn a_wanted_build_waits_for_neither_the_slot_nor_the_commit() {
        static S: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 1 };
        let held = big_compile_turn(&S, || true, || false).expect("a free slot");
        assert_eq!(*S.n.lock().unwrap(), 1);
        // the slot is taken: a wanted build goes on without one, at once
        assert!(big_compile_turn(&S, || true, || true).is_none());
        // … and one that becomes wanted while it waits for the slot goes on then
        static WANTED: AtomicBool = AtomicBool::new(false);
        let t = std::thread::spawn(|| big_compile_turn(&S, || true, || WANTED.load(Ordering::Acquire)).is_none());
        std::thread::sleep(Duration::from_millis(120));
        assert!(!t.is_finished(), "not wanted yet: it waits for its turn");
        WANTED.store(true, Ordering::Release);
        assert!(t.join().unwrap());
        drop(held);
        assert_eq!(*S.n.lock().unwrap(), 0);
        // without the commit: not wanted, it waits; wanted, it goes on and gives the slot back
        static WANTED2: AtomicBool = AtomicBool::new(false);
        let t0 = Instant::now();
        let t = std::thread::spawn(|| big_compile_turn(&S, || false, || WANTED2.load(Ordering::Acquire)).is_none());
        std::thread::sleep(Duration::from_millis(120));
        assert!(!t.is_finished(), "no commit free: it waits");
        WANTED2.store(true, Ordering::Release);
        assert!(t.join().unwrap() && t0.elapsed() < Duration::from_secs(5));
        assert_eq!(*S.n.lock().unwrap(), 0, "the slot went back");
    }

    #[test]
    fn env_counts_parse() {
        assert_eq!(env_count(None, 2, 1, 64), 2);
        assert_eq!(env_count(Some(" 4 "), 2, 1, 64), 4);
        assert_eq!(env_count(Some("0"), 2, 1, 64), 1, "at least one slot: no deadlock");
        assert_eq!(env_count(Some("lots"), 2, 1, 64), 2);
        assert_eq!(env_count(Some("1000"), 2, 1, 64), 64, "an accidental value cannot defeat the cap");
        assert_eq!(env_count(Some("524288"), 256 << 10, 1, usize::MAX), 512 << 10);
    }

    #[test]
    fn only_huge_compiles_get_a_gate_note() {
        let ms = Duration::from_millis(5);
        assert_eq!(gate_note(false, false, false, ms), "", "a small compile (XCL_PSO_TRACE) is not called huge");
        assert_eq!(gate_note(true, true, true, ms), ", gated (waited 5 ms)");
        assert!(gate_note(true, true, false, ms).contains("wanted at once"));
        assert!(gate_note(true, false, false, ms).contains("ungated here"));
    }

    #[test]
    fn giant_driver_compiles_have_their_own_lane() {
        assert!(std::ptr::eq(compile_slots(GIANT_DRIVER_COMPILE - 1), &BIG_COMPILES));
        assert!(std::ptr::eq(compile_slots(GIANT_DRIVER_COMPILE), &GIANT_COMPILES));
    }

    #[test]
    fn only_a_cold_compute_compile_takes_a_slot() {
        assert!(!gate_compute_driver_compile(true), "the cached-blob attempt bypasses the gate");
        assert!(gate_compute_driver_compile(false), "a cold attempt, including a retry, is gated");
    }

    #[test]
    fn the_concurrency_limit_holds_and_all_finish() {
        use std::sync::atomic::AtomicUsize;
        static S: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 2 };
        static IN: AtomicUsize = AtomicUsize::new(0);
        static PEAK: AtomicUsize = AtomicUsize::new(0);
        // the two holders of the slots meet here: both are inside at once, deterministically
        static BOTH: std::sync::Barrier = std::sync::Barrier::new(2);
        let t: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    let _g = big_compile_turn(&S, || true, || false).expect("a turn");
                    let now = IN.fetch_add(1, Ordering::SeqCst) + 1;
                    PEAK.fetch_max(now, Ordering::SeqCst);
                    BOTH.wait();
                    IN.fetch_sub(1, Ordering::SeqCst);
                    BOTH.wait();
                })
            })
            .collect();
        let t0 = Instant::now();
        t.into_iter().for_each(|t| t.join().unwrap());
        assert!(t0.elapsed() < Duration::from_secs(10), "no deadlock");
        assert_eq!(PEAK.load(Ordering::SeqCst), 2);
        assert_eq!(*S.n.lock().unwrap(), 0);
    }

    #[test]
    fn small_and_title_thread_compiles_bypass_the_gate() {
        // every slot taken: a small compile on a worker, and a huge one on an unnamed (title)
        // thread, both go on at once
        // (both lanes: a huge compile of any size would block on a gate it should bypass)
        let mut held: Vec<_> = (0..huge_pso_concurrency()).map(|_| big_compile_turn(&BIG_COMPILES, || true, || false).unwrap()).collect();
        held.push(big_compile_turn(&GIANT_COMPILES, || true, || false).unwrap());
        let small = std::thread::Builder::new()
            .name("xcl-pso-test".into())
            .spawn(|| driver_compile("test", huge_pso_bytes() - 1, true, || 7))
            .unwrap();
        let t0 = Instant::now();
        assert_eq!(small.join().unwrap(), 7);
        let big = huge_pso_bytes().max(GIANT_DRIVER_COMPILE - 1);
        assert_eq!(std::thread::spawn(move || driver_compile("test", big, true, || 8)).join().unwrap(), 8);
        assert_eq!(std::thread::spawn(|| driver_compile("test", usize::MAX / 2, true, || 8)).join().unwrap(), 8);
        assert!(t0.elapsed() < Duration::from_secs(2));
        drop(held);
    }

    #[test]
    fn headroom_is_the_3_gb_floor_until_512_times_the_dxil_exceeds_it() {
        // the class from huge_pso_bytes() up to 6 MiB of DXIL: the floor, unchanged
        assert_eq!(big_compile_headroom(huge_pso_bytes()), 3 << 30);
        assert_eq!(big_compile_headroom(6 << 20), 3 << 30);
        // above it the wait scales with the program: the 29.8 MB one that grew past 14 GB of
        // free commit (#1269) waits for 15 GB
        assert_eq!(big_compile_headroom(8 << 20), 4 << 30);
        assert_eq!(big_compile_headroom(29_800_000), 29_800_000u64 * 512);
    }

    #[test]
    fn unwanted_builds_go_one_at_a_time_when_the_commit_is_there() {
        static S: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 1 };
        let first = big_compile_turn(&S, || true, || false).expect("its turn");
        let t = std::thread::spawn(|| big_compile_turn(&S, || true, || false).is_some());
        std::thread::sleep(Duration::from_millis(120));
        assert!(!t.is_finished(), "one at a time");
        drop(first);
        assert!(t.join().unwrap(), "the next one's turn");
        assert_eq!(*S.n.lock().unwrap(), 0);
    }
}

unsafe fn translated_compute_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<Result<HRESULT, ()>> {
    with_switch_fallback(|| translated_compute_pso_once(this, f, desc, riid, ppv))
}

unsafe fn translated_compute_pso_once(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<Result<HRESULT, ()>> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    if !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let d = &*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let cs = shader_bytes(&d.CS)?;
    if !is_console(cs) {
        return None;
    }
    // made when first bound (`pso_async::lazy`): not even translated now
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return Some(Err(()));
    }
    let compiled = match console_to_dxil_for(cs, Stage::Compute, 0, wave32_only(this)) {
        Ok(b) => b,
        Err(e) => {
            note_translation_error(format!("CS: {e}"));
            return Some(Err(()));
        }
    };
    let dxil = compiled.dxil;
    // a big program: the driver's compile takes it seconds to a minute (a retail game's 1 MB
    // ray-tracing resolve programs ~30 s each, ~600 of them): built in the background like one
    // whose DXIL is missing – the title's loading does not wait for it, its first bind does
    let big = dxil.len() >= huge_pso_bytes();
    if big && crate::gpu::recomp::dxc::defer() {
        return Some(Err(()));
    }
    let Some(rs) = abi_rs_for(this, &compiled.samplers) else { return Some(Err(())) };
    LAST_COMPUTE_RS.with(|c| c.set(rs as usize));
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(d);
    copy.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    copy.CS = bytecode(&dxil);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    // A program built again gets the driver's own result of its first build: a retail title
    // made 14 big compute pipelines twice while loading, 13 - 30 s each the second time.
    let key = compute_blob_key(&dxil, rs as usize, copy.Flags.0);
    let mut cached = compute_blob(key);
    // … and one built in an earlier run gets the blob kept on disk then (`driver_cache`)
    let disk = driver_cache::path(&dxil, copy.Flags.0);
    let from_disk = cached.is_none();
    if from_disk {
        cached = disk.as_deref().and_then(driver_cache::load).map(std::sync::Arc::new);
    }
    if let Some(b) = &cached {
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as *const c_void, CachedBlobSizeInBytes: b.len() };
    }
    // in the background, a few at a time and within the commit headroom the program asks for
    let record = crate::gpu::shader::fnv64(cs);
    let (mut hr, mut in_driver) = driver_compile_identified(
        "compute",
        dxil.len(),
        gate_compute_driver_compile(cached.is_some()),
        Some((record, cached.is_some())),
        || finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv),
    );
    if hr < 0 && cached.is_some() {
        // a blob the driver does not take (another root signature at the same address…): compiled
        forget_compute_blob(key);
        if from_disk {
            driver_cache::note(false, disk.as_deref());
        }
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        let (retried, took) = driver_compile_identified(
            "compute retry",
            dxil.len(),
            gate_compute_driver_compile(false),
            Some((record, false)),
            || finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv),
        );
        hr = retried;
        in_driver += took;
    }
    let reused = hr >= 0 && cached.is_some() && copy.CachedPSO.CachedBlobSizeInBytes != 0;
    if hr >= 0 && !reused {
        keep_compute_blob(key, *ppv, dxil.len());
        if let (Some(p), Some(b)) = (disk.as_deref(), compute_blob(key)) {
            driver_cache::keep(p, &b, in_driver);
        }
    } else if reused && from_disk {
        driver_cache::note(true, None);
    }
    if hr < 0 {
        // A huge program's verdict in the log itself: `note_translation_error` goes to the
        // report, or to the notes of a cached-only attempt – a retail title's 13.8 MB program
        // took an AMD driver 15 – 23 s and ~15 GB to refuse with E_OUTOFMEMORY in every run, and
        // the log showed only its `driver compile finish` and the dispatches skipped after it.
        if dxil.len() >= huge_pso_bytes() {
            // (the record hash is the console program's, as in `driver compile start` and
            // `xcl shaders --measure-pso`; the driver-cache name is the translated DXIL's)
            crate::xlog_warn!(
                "D3D12.X: the driver refused the compute PSO of record {record:016x} ({} B DXIL, driver cache {}) with {hr:#x} after {} ms in the driver: its dispatches are skipped",
                dxil.len(),
                driver_cache::name(&dxil, copy.Flags.0),
                in_driver.as_millis()
            );
        }
        note_translation_error(format!("the driver rejected the compute PSO with the translated CS ({hr:#x})"));
        return Some(Err(()));
    }
    TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    note_pso_rs(*ppv, rs);
    if let Some(n) = crate::gpu::recomp::pso::compute_user_sgprs(cs) {
        PSO_USER_SGPRS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, n);
    }
    note_translated_compute(hr, ppv);
    crate::xlog_info!(
        "D3D12.X: compute PSO with the translated console CS ({} B DXIL, {} ms{})",
        dxil.len(),
        in_driver.as_millis(),
        if reused { ", the driver's blob of an earlier build" } else { "" }
    );
    Some(Ok(hr))
}

/// The blobs of `COMPUTE_BLOBS`, kept on disk between runs.
///
/// `COMPUTE_BLOBS` saves the second build of a program in one run; the first build of every
/// run still goes through the driver's compiler. On an NVIDIA GPU that is 5 – 27 s for each of
/// a retail title's larger compute programs when its world loads: the title dispatches without
/// them meanwhile and is ended before all are built, so the next run starts the same compiles
/// again. A blob whose build took `KEEP` or longer is written under the hash of its DXIL and
/// its flags (`<DXIL cache>\driver\`) and read when a later run has none in memory; the
/// driver refuses the blob of another driver, adapter or root signature (the file is dropped
/// and the program compiled as before). `XCL_PSO_DISK_CACHE=0`: off.
pub(crate) mod driver_cache {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// builds shorter than this are not worth a file
    const KEEP: std::time::Duration = std::time::Duration::from_millis(250);
    static HITS: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    static STORED: AtomicU64 = AtomicU64::new(0);

    /// The file of the pipeline made from `dxil` with `flags`.
    pub fn name(dxil: &[u8], flags: i32) -> String {
        format!("{:016x}-{:x}-{:x}.pso", crate::gpu::shader::fnv64(dxil), dxil.len(), flags)
    }

    /// The file of a translated graphics pipeline keyed by its full state + shader digest.
    pub fn graphics_name(hash: u64, bytes: usize) -> String {
        format!("gfx-{hash:016x}-{bytes:x}.pso")
    }

    fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("XCL_PSO_DISK_CACHE").map_or(true, |v| v.trim() != "0"))
    }

    pub fn path(dxil: &[u8], flags: i32) -> Option<PathBuf> {
        enabled().then(|| crate::gpu::recomp::dxc::cache_dir().join("driver").join(name(dxil, flags)))
    }

    pub fn graphics_path(hash: u64, bytes: usize) -> Option<PathBuf> {
        enabled().then(|| crate::gpu::recomp::dxc::cache_dir().join("driver").join(graphics_name(hash, bytes)))
    }

    pub fn load(p: &Path) -> Option<Vec<u8>> {
        std::fs::read(p).ok().filter(|b| !b.is_empty())
    }

    /// A pipeline was made from a kept blob (`ok`), or the driver refused the blob: dropped.
    pub fn note(ok: bool, p: Option<&Path>) {
        if ok {
            let n = HITS.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: {n} compute pipelines made from the driver's blob of an earlier run (kept on disk)");
            }
        } else {
            if let Some(p) = p {
                let _ = std::fs::remove_file(p);
            }
            if STALE.fetch_add(1, Ordering::Relaxed) == 0 {
                crate::xlog_info!("D3D12.X: a driver blob kept on disk was refused (another driver or adapter?) – such pipelines are compiled again");
            }
        }
    }

    fn create_staging_file(p: &Path) -> std::io::Result<(PathBuf, std::fs::File)> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        create_staging_file_with(p, &NEXT)
    }

    fn create_staging_file_with(p: &Path, next: &AtomicU64) -> std::io::Result<(PathBuf, std::fs::File)> {
        loop {
            let tmp = p.with_extension(format!("tmp{}-{}", std::process::id(), next.fetch_add(1, Ordering::Relaxed)));
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp) {
                Ok(file) => return Ok((tmp, file)),
                // A prior process with this PID may have left a file behind. It is not ours.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// `bytes` to `p`, under another name first: a run that is cut short leaves no half file.
    pub fn write(p: &Path, bytes: &[u8]) -> bool {
        use std::io::Write;
        if bytes.is_empty() {
            return false;
        }
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let Ok((tmp, mut file)) = create_staging_file(p) else { return false };
        let written = file.write_all(bytes).is_ok();
        drop(file);
        let ok = written && std::fs::rename(&tmp, p).is_ok();
        if !ok {
            let _ = std::fs::remove_file(&tmp);
        }
        ok
    }

    /// Keep `blob`, of a build that took `took`, for the next run.
    pub fn keep(p: &Path, blob: &[u8], took: std::time::Duration) {
        if took >= KEEP && write(p, blob) {
            let n = STORED.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: {n} driver blobs kept on disk for the next run (this one: {} ms, {} KB)", took.as_millis(), blob.len() >> 10);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        struct Directory(PathBuf);
        impl Drop for Directory {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }

        #[test]
        fn a_file_is_named_after_its_dxil_and_flags() {
            assert_eq!(name(b"abc", 0), name(b"abc", 0));
            assert_ne!(name(b"abc", 0), name(b"abd", 0));
            assert_ne!(name(b"abc", 0), name(b"abc", 1));
            // (the length is in the name: two programs of one hash and different lengths differ)
            assert!(name(&[0u8; 0x1234], 2).ends_with("-1234-2.pso"));
        }

        #[test]
        fn a_kept_blob_is_read_back_and_a_refused_one_is_dropped() {
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-test-{}", std::process::id()));
            let p = dir.join("driver").join(name(b"program", 0));
            assert_eq!(load(&p), None, "nothing kept yet");
            assert!(!write(&p, &[]), "an empty blob is not kept");
            keep(&p, &[1, 2, 3], std::time::Duration::from_millis(10));
            assert_eq!(load(&p), None, "a build of milliseconds is not kept");
            keep(&p, &[1, 2, 3], std::time::Duration::from_secs(1));
            assert_eq!(load(&p).as_deref(), Some(&[1u8, 2, 3][..]));
            // no half file under the temporary name
            assert_eq!(std::fs::read_dir(p.parent().unwrap()).unwrap().count(), 1);
            // the driver took the blob: it stays; it refused it: gone, the program is compiled again
            note(true, Some(&p));
            assert!(p.exists());
            note(false, Some(&p));
            assert_eq!(load(&p), None);
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn an_overlapping_writer_cannot_modify_the_published_blob() {
            use std::io::Write;
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-overlap-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let p = dir.join("program.pso");
            // Another writer, paused after the production staging-file creation step.
            let (_other_tmp, mut other) = create_staging_file(&p).unwrap();
            let complete = b"complete second blob";
            assert!(write(&p, complete));
            assert_eq!(load(&p).as_deref(), Some(complete.as_slice()));
            // Resuming the older writer must not change a file already published by this
            // invocation, even when the older payload is shorter than the newer one.
            other.write_all(b"old").unwrap();
            assert_eq!(load(&p).as_deref(), Some(complete.as_slice()), "published bytes must not alias another writer's staging file");
        }

        #[test]
        fn an_abandoned_staging_file_is_skipped_and_preserved() {
            use std::io::Write;
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-stale-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let p = dir.join("program.pso");
            let occupied = p.with_extension(format!("tmp{}-0", std::process::id()));
            std::fs::write(&occupied, b"not ours").unwrap();
            let next = AtomicU64::new(0);
            let (tmp, mut file) = create_staging_file_with(&p, &next).unwrap();
            assert_eq!(next.load(Ordering::Relaxed), 2, "the occupied candidate was passed over");
            assert_ne!(tmp, occupied);
            file.write_all(b"ours").unwrap();
            drop(file);
            std::fs::rename(&tmp, &p).unwrap();
            assert_eq!(load(&p).as_deref(), Some(b"ours".as_slice()));
            assert_eq!(std::fs::read(&occupied).unwrap(), b"not ours");
        }
    }
}

/// The driver's compiled form of translated compute programs (`GetCachedBlob`),
/// by (DXIL hash, root signature, flags), up to `COMPUTE_BLOBS_MAX` bytes in all.
static COMPUTE_BLOBS: Mutex<Option<(HashMap<(u64, usize, i32), std::sync::Arc<Vec<u8>>>, usize)>> = Mutex::new(None);
const COMPUTE_BLOBS_MAX: usize = 512 << 20;
/// smaller programs compile in milliseconds: not kept
const COMPUTE_BLOB_MIN_DXIL: usize = 16 << 10;

fn compute_blob_key(dxil: &[u8], rs: usize, flags: i32) -> (u64, usize, i32) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    dxil.hash(&mut h);
    (h.finish(), rs, flags)
}

fn compute_blob(key: (u64, usize, i32)) -> Option<std::sync::Arc<Vec<u8>>> {
    COMPUTE_BLOBS.lock().unwrap().as_ref().and_then(|(m, _)| m.get(&key).cloned())
}

fn forget_compute_blob(key: (u64, usize, i32)) {
    if let Some((m, total)) = COMPUTE_BLOBS.lock().unwrap().as_mut() {
        if let Some(b) = m.remove(&key) {
            *total -= b.len();
        }
    }
}

#[cfg(test)]
mod compute_blob_tests {
    #[test]
    fn a_blob_is_found_only_for_the_same_program_root_signature_and_flags_and_is_forgotten() {
        let dxil = vec![7u8; 1000];
        let key = super::compute_blob_key(&dxil, 0x1000, 0);
        assert_eq!(key, super::compute_blob_key(&dxil.clone(), 0x1000, 0));
        assert_ne!(key, super::compute_blob_key(&dxil, 0x2000, 0));
        assert_ne!(key, super::compute_blob_key(&dxil, 0x1000, 1));
        let mut other = dxil.clone();
        other[999] = 8;
        assert_ne!(key, super::compute_blob_key(&other, 0x1000, 0));
        {
            let mut g = super::COMPUTE_BLOBS.lock().unwrap();
            let (m, total) = g.get_or_insert_with(|| (super::HashMap::new(), 0));
            m.insert(key, std::sync::Arc::new(vec![1, 2, 3]));
            *total += 3;
        }
        assert_eq!(super::compute_blob(key).as_deref().map(|v| v.len()), Some(3));
        let before = super::COMPUTE_BLOBS.lock().unwrap().as_ref().unwrap().1;
        super::forget_compute_blob(key);
        assert!(super::compute_blob(key).is_none());
        assert_eq!(super::COMPUTE_BLOBS.lock().unwrap().as_ref().unwrap().1, before - 3);
    }
}

/// Read `GetCachedBlob` from either an adopted or raw PC `ID3D12PipelineState`.
unsafe fn read_pso_cached_blob(pso: *mut c_void) -> Option<Vec<u8>> {
    if pso.is_null() {
        return None;
    }
    let raw = if is_adopted(pso) {
        type GetBlob = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT;
        let get: GetBlob = std::mem::transmute(orig_fn(pso, slot_of(pso, "GetCachedBlob")));
        let mut out = std::ptr::null_mut();
        if get(pso, &mut out) < 0 || out.is_null() {
            return None;
        }
        out
    } else {
        let p = ID3D12PipelineState::from_raw_borrowed(&pso)?;
        let blob = p.GetCachedBlob().ok()?;
        return Some(std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()).to_vec());
    };
    let blob = ID3DBlob::from_raw(raw);
    let bytes = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()).to_vec();
    (!bytes.is_empty()).then_some(bytes)
}

/// The driver's blob of the PSO just built (works on both adopted title PSOs and raw PC compute
/// passes owned by `TwoPassPso`).
unsafe fn keep_compute_blob(key: (u64, usize, i32), pso: *mut c_void, dxil_len: usize) {
    if dxil_len < COMPUTE_BLOB_MIN_DXIL || pso.is_null() || compute_blob(key).is_some() {
        return;
    }
    let Some(bytes) = read_pso_cached_blob(pso) else { return };
    let mut g = COMPUTE_BLOBS.lock().unwrap();
    let (m, total) = g.get_or_insert_with(|| (HashMap::new(), 0));
    if bytes.is_empty() || *total + bytes.len() > COMPUTE_BLOBS_MAX {
        return;
    }
    *total += bytes.len();
    m.insert(key, std::sync::Arc::new(bytes));
}

/// The driver's compiled form of translated graphics pipelines (`GetCachedBlob`),
/// keyed by `(full_state_and_shader_hash, total_dxil_bytes)`, up to `GRAPHICS_BLOBS_MAX`.
static GRAPHICS_BLOBS: Mutex<Option<(HashMap<(u64, usize), std::sync::Arc<Vec<u8>>>, usize)>> = Mutex::new(None);
const GRAPHICS_BLOBS_MAX: usize = 256 << 20;

fn graphics_blob(key: (u64, usize)) -> Option<std::sync::Arc<Vec<u8>>> {
    GRAPHICS_BLOBS.lock().unwrap().as_ref().and_then(|(m, _)| m.get(&key).cloned())
}

fn forget_graphics_blob(key: (u64, usize)) {
    if let Some((m, total)) = GRAPHICS_BLOBS.lock().unwrap().as_mut() {
        if let Some(b) = m.remove(&key) {
            *total -= b.len();
        }
    }
}

unsafe fn keep_graphics_blob(key: (u64, usize), pso: *mut c_void, dxil_len: usize) {
    if dxil_len < COMPUTE_BLOB_MIN_DXIL || pso.is_null() || graphics_blob(key).is_some() {
        return;
    }
    let Some(bytes) = read_pso_cached_blob(pso) else { return };
    let mut g = GRAPHICS_BLOBS.lock().unwrap();
    let (m, total) = g.get_or_insert_with(|| (HashMap::new(), 0));
    if bytes.is_empty() || *total + bytes.len() > GRAPHICS_BLOBS_MAX {
        return;
    }
    *total += bytes.len();
    m.insert(key, std::sync::Arc::new(bytes));
}

/// Deterministic FNV-1a digest of every field in a translated PC graphics PSO (shaders, static
/// samplers, blend/rasterizer/depth-stencil state, input layout, render target / depth formats,
/// sample desc, topology, flags) for in-memory and on-disk driver blob caching.
unsafe fn graphics_blob_key(
    copy: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    vs_dxil: &[u8],
    ps_dxil: Option<&[u8]>,
    rect_gs: Option<&[u8]>,
    samplers: &[crate::gpu::recomp::pso::StaticSampler],
) -> (u64, usize) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let feed = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    let feed_u64 = |h: &mut u64, v: u64| feed(h, &v.to_le_bytes());
    let feed_u32 = |h: &mut u64, v: u32| feed(h, &v.to_le_bytes());
    let feed_u8 = |h: &mut u64, v: u8| feed(h, &[v]);

    feed_u64(&mut h, crate::gpu::shader::fnv64(vs_dxil));
    feed_u64(&mut h, vs_dxil.len() as u64);
    if let Some(ps) = ps_dxil {
        feed_u64(&mut h, crate::gpu::shader::fnv64(ps));
        feed_u64(&mut h, ps.len() as u64);
    } else {
        feed_u64(&mut h, 0);
    }
    if let Some(gs) = rect_gs {
        feed_u64(&mut h, crate::gpu::shader::fnv64(gs));
        feed_u64(&mut h, gs.len() as u64);
    } else {
        feed_u64(&mut h, 0);
    }
    feed_u64(&mut h, samplers.len() as u64);
    for (w, cmp) in samplers {
        for &x in w {
            feed_u32(&mut h, x);
        }
        feed_u8(&mut h, *cmp as u8);
    }
    // Hash BlendState, RasterizerState, and DepthStencilState field-by-field so struct
    // padding bytes (e.g. D3D12_RENDER_TARGET_BLEND_DESC tail padding and the 2 bytes after
    // DepthStencilState.StencilWriteMask) never cause nondeterministic cache keys.
    let bs = &copy.BlendState;
    feed_u32(&mut h, bs.AlphaToCoverageEnable.0 as u32);
    feed_u32(&mut h, bs.IndependentBlendEnable.0 as u32);
    for rt in &bs.RenderTarget {
        feed_u32(&mut h, rt.BlendEnable.0 as u32);
        feed_u32(&mut h, rt.LogicOpEnable.0 as u32);
        feed_u32(&mut h, rt.SrcBlend.0 as u32);
        feed_u32(&mut h, rt.DestBlend.0 as u32);
        feed_u32(&mut h, rt.BlendOp.0 as u32);
        feed_u32(&mut h, rt.SrcBlendAlpha.0 as u32);
        feed_u32(&mut h, rt.DestBlendAlpha.0 as u32);
        feed_u32(&mut h, rt.BlendOpAlpha.0 as u32);
        feed_u32(&mut h, rt.LogicOp.0 as u32);
        feed_u8(&mut h, rt.RenderTargetWriteMask);
    }
    feed_u32(&mut h, copy.SampleMask);
    let rs = &copy.RasterizerState;
    feed_u32(&mut h, rs.FillMode.0 as u32);
    feed_u32(&mut h, rs.CullMode.0 as u32);
    feed_u32(&mut h, rs.FrontCounterClockwise.0 as u32);
    feed_u32(&mut h, rs.DepthBias as u32);
    feed_u32(&mut h, rs.DepthBiasClamp.to_bits());
    feed_u32(&mut h, rs.SlopeScaledDepthBias.to_bits());
    feed_u32(&mut h, rs.DepthClipEnable.0 as u32);
    feed_u32(&mut h, rs.MultisampleEnable.0 as u32);
    feed_u32(&mut h, rs.AntialiasedLineEnable.0 as u32);
    feed_u32(&mut h, rs.ForcedSampleCount);
    feed_u32(&mut h, rs.ConservativeRaster.0 as u32);
    let ds = &copy.DepthStencilState;
    feed_u32(&mut h, ds.DepthEnable.0 as u32);
    feed_u32(&mut h, ds.DepthWriteMask.0 as u32);
    feed_u32(&mut h, ds.DepthFunc.0 as u32);
    feed_u32(&mut h, ds.StencilEnable.0 as u32);
    feed_u8(&mut h, ds.StencilReadMask);
    feed_u8(&mut h, ds.StencilWriteMask);
    for op in [&ds.FrontFace, &ds.BackFace] {
        feed_u32(&mut h, op.StencilFailOp.0 as u32);
        feed_u32(&mut h, op.StencilDepthFailOp.0 as u32);
        feed_u32(&mut h, op.StencilPassOp.0 as u32);
        feed_u32(&mut h, op.StencilFunc.0 as u32);
    }
    feed_u32(&mut h, copy.InputLayout.NumElements);
    if copy.InputLayout.NumElements > 0 && !copy.InputLayout.pInputElementDescs.is_null() {
        let elems = std::slice::from_raw_parts(copy.InputLayout.pInputElementDescs, copy.InputLayout.NumElements as usize);
        for e in elems {
            if !e.SemanticName.0.is_null() {
                feed(&mut h, std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char).to_bytes_with_nul());
            } else {
                feed(&mut h, &[0]);
            }
            feed_u32(&mut h, e.SemanticIndex);
            feed_u32(&mut h, e.Format.0 as u32);
            feed_u32(&mut h, e.InputSlot);
            feed_u32(&mut h, e.AlignedByteOffset);
            feed_u32(&mut h, e.InputSlotClass.0 as u32);
            feed_u32(&mut h, e.InstanceDataStepRate);
        }
    }
    feed_u32(&mut h, copy.IBStripCutValue.0 as u32);
    feed_u32(&mut h, copy.PrimitiveTopologyType.0 as u32);
    feed_u32(&mut h, copy.NumRenderTargets);
    for f in &copy.RTVFormats {
        feed_u32(&mut h, f.0 as u32);
    }
    feed_u32(&mut h, copy.DSVFormat.0 as u32);
    feed_u32(&mut h, copy.SampleDesc.Count);
    feed_u32(&mut h, copy.SampleDesc.Quality);
    feed_u32(&mut h, copy.NodeMask);
    feed_u32(&mut h, copy.Flags.0 as u32);
    let total_bytes = vs_dxil.len() + ps_dxil.map_or(0, <[u8]>::len) + rect_gs.map_or(0, <[u8]>::len);
    (h, total_bytes)
}

#[cfg(test)]
mod graphics_blob_tests {
    use super::*;

    #[test]
    fn graphics_blob_key_tracks_shaders_samplers_and_state() {
        unsafe {
            let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
            d.NumRenderTargets = 1;
            d.RTVFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
            d.SampleDesc.Count = 1;
            let vs = b"vs-dxil-1";
            let ps = b"ps-dxil-1";
            let k0 = graphics_blob_key(&d, vs, Some(ps), None, &[]);
            assert_eq!(k0, graphics_blob_key(&d, vs, Some(ps), None, &[]));
            assert_ne!(k0, graphics_blob_key(&d, b"vs-dxil-2", Some(ps), None, &[]));
            assert_ne!(k0, graphics_blob_key(&d, vs, None, None, &[]));
            assert_ne!(k0, graphics_blob_key(&d, vs, Some(ps), Some(b"gs"), &[]));
            assert_ne!(k0, graphics_blob_key(&d, vs, Some(ps), None, &[([1, 2, 3, 4], false)]));
            let mut d2 = d;
            d2.RTVFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;
            assert_ne!(k0, graphics_blob_key(&d2, vs, Some(ps), None, &[]));
        }
    }

    #[test]
    fn graphics_blob_key_ignores_struct_padding_bytes() {
        unsafe {
            let mut a: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::mem::zeroed();
            let mut b: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::mem::zeroed();
            // Fill every byte of DepthStencilState and BlendState with distinct poison patterns,
            // then populate every semantic field identically so only padding bytes differ.
            std::ptr::write_bytes(&mut a.DepthStencilState as *mut _ as *mut u8, 0xAA, std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC>());
            std::ptr::write_bytes(&mut b.DepthStencilState as *mut _ as *mut u8, 0x55, std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC>());
            std::ptr::write_bytes(&mut a.BlendState as *mut _ as *mut u8, 0xAA, std::mem::size_of::<D3D12_BLEND_DESC>());
            std::ptr::write_bytes(&mut b.BlendState as *mut _ as *mut u8, 0x55, std::mem::size_of::<D3D12_BLEND_DESC>());

            for d in [&mut a, &mut b] {
                d.DepthStencilState.DepthEnable = true.into();
                d.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
                d.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
                d.DepthStencilState.StencilEnable = false.into();
                d.DepthStencilState.StencilReadMask = 0xFF;
                d.DepthStencilState.StencilWriteMask = 0xFF;
                d.DepthStencilState.FrontFace = D3D12_DEPTH_STENCILOP_DESC::default();
                d.DepthStencilState.BackFace = D3D12_DEPTH_STENCILOP_DESC::default();
                d.BlendState.AlphaToCoverageEnable = false.into();
                d.BlendState.IndependentBlendEnable = false.into();
                for rt in &mut d.BlendState.RenderTarget {
                    rt.BlendEnable = false.into();
                    rt.LogicOpEnable = false.into();
                    rt.SrcBlend = D3D12_BLEND_ONE;
                    rt.DestBlend = D3D12_BLEND_ZERO;
                    rt.BlendOp = D3D12_BLEND_OP_ADD;
                    rt.SrcBlendAlpha = D3D12_BLEND_ONE;
                    rt.DestBlendAlpha = D3D12_BLEND_ZERO;
                    rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                    rt.LogicOp = D3D12_LOGIC_OP_NOOP;
                    rt.RenderTargetWriteMask = 0xF;
                }
            }
            let ka = graphics_blob_key(&a, b"vs", Some(b"ps"), None, &[]);
            let kb = graphics_blob_key(&b, b"vs", Some(b"ps"), None, &[]);
            assert_eq!(ka, kb, "padding bytes must not affect graphics_blob_key");

            b.DepthStencilState.StencilWriteMask = 0x7F;
            assert_ne!(ka, graphics_blob_key(&b, b"vs", Some(b"ps"), None, &[]));
        }
    }

    #[test]
    fn stale_cached_graphics_blob_preserves_rect_list_gs_on_retry() {
        const E_INVALIDARG: HRESULT = -2147024809;
        let gs = [1u8; 16];
        let blob = [0xABu8; 32];
        let mut copy = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        copy.GS = bytecode(&gs);
        let mut stale_evicted = 0usize;
        let mut attempts = Vec::new();
        let build = create_graphics_pso_cached(
            &mut copy,
            Some(&blob),
            true,
            || stale_evicted += 1,
            |label, has_cached, f| {
                let hr = f();
                attempts.push((label.to_string(), has_cached, hr));
                (hr, std::time::Duration::from_millis(1))
            },
            |c| {
                // Driver rejects the stale CachedPSO, and accepts the fresh compile with rect GS intact.
                if !c.CachedPSO.pCachedBlob.is_null() {
                    E_INVALIDARG
                } else if !c.GS.pShaderBytecode.is_null() {
                    0
                } else {
                    E_INVALIDARG
                }
            },
        );
        assert_eq!(build.hr, 0);
        assert_eq!(stale_evicted, 1);
        assert!(!build.reused);
        assert!(build.cache_eligible, "fresh compile with rect GS intact should be cached");
        assert!(!copy.GS.pShaderBytecode.is_null(), "stale cached blob must not strip rect GS");
        assert!(copy.CachedPSO.pCachedBlob.is_null());
        assert_eq!(attempts.len(), 2);
        assert_eq!((attempts[0].0.as_str(), attempts[0].1, attempts[0].2), ("graphics", true, E_INVALIDARG));
        assert_eq!((attempts[1].0.as_str(), attempts[1].1, attempts[1].2), ("graphics retry", false, 0));
    }

    #[test]
    fn rect_list_gs_fallback_does_not_cache_under_with_gs_key() {
        const E_INVALIDARG: HRESULT = -2147024809;
        let gs = [2u8; 16];
        let mut copy = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        copy.GS = bytecode(&gs);
        let mut calls = 0usize;
        let build = create_graphics_pso_cached(
            &mut copy,
            None,
            true,
            || panic!("no cached blob to evict"),
            |_, _, f| (f(), std::time::Duration::from_millis(1)),
            |c| {
                calls += 1;
                if c.GS.pShaderBytecode.is_null() { 0 } else { E_INVALIDARG }
            },
        );
        assert_eq!(build.hr, 0);
        assert_eq!(calls, 2);
        assert!(copy.GS.pShaderBytecode.is_null());
        assert!(!build.reused);
        assert!(!build.cache_eligible, "fallback no-GS PSO must not overwrite the with-GS cache key");
    }

    #[test]
    fn valid_cached_graphics_blob_reuses_without_retry() {
        let gs = [3u8; 16];
        let blob = [0xCDu8; 32];
        let mut copy = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        copy.GS = bytecode(&gs);
        let mut calls = 0usize;
        let build = create_graphics_pso_cached(
            &mut copy,
            Some(&blob),
            true,
            || panic!("valid cached blob must not be evicted"),
            |_, has_cached, f| {
                assert!(has_cached);
                (f(), std::time::Duration::from_millis(1))
            },
            |c| {
                calls += 1;
                assert!(!c.CachedPSO.pCachedBlob.is_null());
                assert!(!c.GS.pShaderBytecode.is_null());
                0
            },
        );
        assert_eq!(build.hr, 0);
        assert_eq!(calls, 1);
        assert!(build.reused);
        assert!(!build.cache_eligible);
        assert!(!copy.GS.pShaderBytecode.is_null());
    }
}

struct GraphicsPsoBuild {
    hr: HRESULT,
    in_driver: std::time::Duration,
    reused: bool,
    cache_eligible: bool,
}

fn create_graphics_pso_cached(
    copy: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    cached: Option<&[u8]>,
    rect_gs: bool,
    mut on_stale_cache: impl FnMut(),
    mut run: impl FnMut(&str, bool, &mut dyn FnMut() -> HRESULT) -> (HRESULT, std::time::Duration),
    mut create: impl FnMut(&D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> HRESULT,
) -> GraphicsPsoBuild {
    if let Some(b) = cached {
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as *const c_void, CachedBlobSizeInBytes: b.len() };
        // Do not strip `rect_gs` while a cached blob is still attached: a stale blob must be
        // discarded first so the uncached retry compiles with `copy.GS` intact.
        let (hr, in_driver) = run("graphics", true, &mut || create(copy));
        if hr >= 0 {
            return GraphicsPsoBuild { hr, in_driver, reused: true, cache_eligible: false };
        }
        on_stale_cache();
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        let (retried, took) = run("graphics retry", false, &mut || retry_without_rect_gs(copy, rect_gs, &mut create));
        let dropped_rect_gs = rect_gs && copy.GS.pShaderBytecode.is_null();
        return GraphicsPsoBuild {
            hr: retried,
            in_driver: in_driver + took,
            reused: false,
            cache_eligible: retried >= 0 && !dropped_rect_gs,
        };
    }
    let (hr, in_driver) = run("graphics", false, &mut || retry_without_rect_gs(copy, rect_gs, &mut create));
    let dropped_rect_gs = rect_gs && copy.GS.pShaderBytecode.is_null();
    GraphicsPsoBuild {
        hr,
        in_driver,
        reused: false,
        cache_eligible: hr >= 0 && !dropped_rect_gs,
    }
}

unsafe fn translated_graphics_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    with_switch_fallback(|| translated_graphics_pso_once(this, f, desc, riid, ppv))
}

unsafe fn translated_graphics_pso_once(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    if !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let d = &*(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    let vs = shader_bytes(&d.VS)?;
    if !is_console(vs) {
        return None;
    }
    if shader_bytes(&d.HS).is_some() || shader_bytes(&d.DS).is_some() {
        note_translation_error("PSO with HS/DS – not supported yet".into());
        return None;
    }
    // made when first used (`pso_async::lazy`): not even translated now (covers both VS+PS and GS)
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return None;
    }
    if let Some(gs) = shader_bytes(&d.GS).filter(|g| is_console(g)) {
        let _ = f;
        return Some(translated_gs_pso(this, d, vs, gs, riid, ppv));
    }
    // a GPU that can run 32-lane waves may run a wave64 VS / PS in waves of 32 (the driver
    // picks; RDNA2 does for pixel shaders): the lanes 32..63 it reads come from `upper_lanes`
    // (split_wave64)
    let split = graphics_wave32(this);
    let vs_c = match console_to_dxil_for(vs, Stage::Vertex, 0, split) {
        Ok(b) => b,
        Err(e) => {
            note_translation_error(format!("VS: {e}"));
            return None;
        }
    };
    let mut samplers = vs_c.samplers;
    let vs_dxil = vs_c.dxil;
    let ps_dxil = match shader_bytes(&d.PS) {
        Some(p) if is_console(p) => match console_to_dxil_for(p, Stage::Pixel, samplers.len() as u32, split) {
            Ok(b) => {
                samplers.extend(b.samplers);
                Some(b.dxil)
            }
            Err(e) => {
                note_translation_error(format!("PS: {e}"));
                return None;
            }
        },
        Some(p) => Some(p.to_vec()),
        None => None,
    };
    let rs = abi_rs_for(this, &samplers)?;
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(d);
    copy.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    copy.VS = bytecode(&vs_dxil);
    copy.PS = ps_dxil.as_deref().map(bytecode).unwrap_or_default();
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    // a rect list pipeline: the fourth corner of each rect comes from a generated GS; without
    // it each rect is drawn as the triangle of its three vertices
    let rect_gs = if d.PrimitiveTopologyType.0 == RECT_LIST_TYPE && copy.GS.pShaderBytecode.is_null() {
        rect_list_gs(vs, shader_bytes(&d.PS)).map_err(|e| note_translation_error(format!("rect list drawn as triangles: {e}"))).ok()
    } else {
        None
    };
    if let Some(gs) = &rect_gs {
        copy.GS = bytecode(gs);
    }
    pc_topology_type(&mut copy);
    let ps_hash = shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64);
    if debug_pso_switch("XCL_DEBUG_NO_STENCIL", ps_hash) {
        copy.DepthStencilState.StencilEnable = false.into();
    }
    if debug_pso_switch("XCL_DEBUG_NO_CULL", ps_hash) {
        copy.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    }
    let bytes = vs_dxil.len() + ps_dxil.as_ref().map_or(0, Vec::len) + rect_gs.as_ref().map_or(0, Vec::len);
    let gkey = graphics_blob_key(&copy, &vs_dxil, ps_dxil.as_deref(), rect_gs.as_deref(), &samplers);
    let mut cached = graphics_blob(gkey);
    let disk = driver_cache::graphics_path(gkey.0, gkey.1);
    let from_disk = cached.is_none();
    if from_disk {
        cached = disk.as_deref().and_then(driver_cache::load).map(std::sync::Arc::new);
    }
    let build = create_graphics_pso_cached(
        &mut copy,
        cached.as_deref().map(Vec::as_slice),
        rect_gs.is_some(),
        || {
            forget_graphics_blob(gkey);
            if from_disk {
                driver_cache::note(false, disk.as_deref());
            }
        },
        |label, has_cached, f| driver_compile_identified(label, bytes, gate_compute_driver_compile(has_cached), Some((ps_hash, has_cached)), f),
        |c| finish_pso(this, f, c as *const _ as *const c_void, riid, ppv),
    );
    let hr = build.hr;
    if build.cache_eligible {
        keep_graphics_blob(gkey, *ppv, bytes);
        if let (Some(p), Some(b)) = (disk.as_deref(), graphics_blob(gkey)) {
            driver_cache::keep(p, &b, build.in_driver);
        }
    } else if build.reused && from_disk {
        driver_cache::note(true, None);
    }
    if hr < 0 {
        note_translation_error(format!(
            "the driver rejected the PSO with the translated shaders ({hr:#x}; topology type {}, {} render targets, DSV format {}, HS {}, GS {})",
            copy.PrimitiveTopologyType.0,
            copy.NumRenderTargets,
            copy.DSVFormat.0,
            !copy.HS.pShaderBytecode.is_null(),
            !copy.GS.pShaderBytecode.is_null()
        ));
        return None;
    }
    let n = TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    note_pso_rs(*ppv, rs);
    TRANSLATED_GRAPHICS.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    note_rect_list_gs(*ppv as usize, &copy, rect_gs.is_some());
    if super::present::trace_frame() {
        let ps = shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64);
        PSO_SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (crate::gpu::shader::fnv64(vs), ps));
        // the input layout (semantic index, format, slot, offset) – what the vertices are read as
        let il = &d.InputLayout;
        let elems: Vec<String> = if il.NumElements > 0 && !il.pInputElementDescs.is_null() {
            std::slice::from_raw_parts(il.pInputElementDescs, il.NumElements as usize)
                .iter()
                .map(|e| format!("{}{} fmt {} slot {} @{}", e.SemanticName.to_string().unwrap_or_default(), e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset as i32))
                .collect()
        } else {
            Vec::new()
        };
        PSO_LAYOUTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, elems.join(", "));
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 400 {
            let b = &d.BlendState.RenderTarget[0];
            let z = &d.DepthStencilState;
            crate::xlog_info!(
                "frame trace: PSO vs {:016x} ps {ps:016x}: RTs {} {:?} DSV {} samples {}, depth (enable {}, write {}, func {}), stencil {}, blend0 (enable {}, src {} dst {} op {}, alpha src {} dst {}, mask {:#x}), cull {} fill {} depth clip {}, sample mask {:#x}",
                crate::gpu::shader::fnv64(vs),
                d.NumRenderTargets,
                &d.RTVFormats[..d.NumRenderTargets.min(8) as usize].iter().map(|f| f.0).collect::<Vec<_>>(),
                d.DSVFormat.0,
                d.SampleDesc.Count,
                z.DepthEnable.as_bool(),
                z.DepthWriteMask.0,
                z.DepthFunc.0,
                z.StencilEnable.as_bool(),
                b.BlendEnable.as_bool(),
                b.SrcBlend.0,
                b.DestBlend.0,
                b.BlendOp.0,
                b.SrcBlendAlpha.0,
                b.DestBlendAlpha.0,
                b.RenderTargetWriteMask,
                d.RasterizerState.CullMode.0,
                d.RasterizerState.FillMode.0,
                d.RasterizerState.DepthClipEnable.as_bool(),
                d.SampleMask
            );
            if z.StencilEnable.as_bool() {
                let f = |o: &D3D12_DEPTH_STENCILOP_DESC| (o.StencilFunc.0, o.StencilPassOp.0, o.StencilFailOp.0, o.StencilDepthFailOp.0);
                crate::xlog_info!(
                    "frame trace: … its stencil: read mask {:#x} write mask {:#x}, front (func, pass, fail, depth fail) {:?}, back {:?}",
                    z.StencilReadMask,
                    z.StencilWriteMask,
                    f(&z.FrontFace),
                    f(&z.BackFace)
                );
            }
        }
    }
    if n < 5 || n % 1000 == 0 {
        crate::xlog_info!("D3D12.X: graphics PSO from translated console shaders (#{})", n + 1);
    }
    Some(hr)
}

// ---------------------------------------------------------------------------------------
// non-blocking pipeline creation (see `pso_async`)
// ---------------------------------------------------------------------------------------

/// A pipeline description that owns what it points to (shader blobs, input layout and its
/// semantic names), so a worker can build the pipeline after the title's memory is gone.
/// The root signature pointer is kept as is (the translated paths replace it).
#[allow(dead_code)] // the vectors only keep the copies alive
enum OwnedDesc {
    Compute(Box<D3D12_COMPUTE_PIPELINE_STATE_DESC>, Vec<Vec<u8>>),
    Graphics(Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC>, Vec<Vec<u8>>, Vec<D3D12_INPUT_ELEMENT_DESC>, Vec<std::ffi::CString>),
    Stream(Box<[usize; 2]>, Vec<u8>, Vec<Vec<u8>>),
}

unsafe impl Send for OwnedDesc {}

/// A hash of a pipeline's shaders in their order (never 0): what `pso_async` knows a used
/// pipeline by in the next run.
pub(crate) fn shaders_key(blobs: &[Vec<u8>]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in blobs {
        h = (h ^ crate::gpu::shader::fnv64(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h | 1
}

#[cfg(test)]
mod shaders_key_tests {
    #[test]
    fn a_key_follows_the_shaders_and_their_order_and_is_never_zero() {
        let (a, b) = (vec![1u8, 2, 3], vec![4u8, 5]);
        let key = super::shaders_key(&[a.clone(), b.clone()]);
        assert_eq!(key, super::shaders_key(&[a.clone(), b.clone()]));
        assert_ne!(key, super::shaders_key(&[b.clone(), a.clone()]));
        assert_ne!(key, super::shaders_key(&[a.clone()]));
        assert_ne!(key, super::shaders_key(&[a, vec![4u8, 6]]));
        assert_ne!(super::shaders_key(&[]), 0);
    }
}

impl OwnedDesc {
    fn shaders_key(&self) -> u64 {
        shaders_key(match self {
            OwnedDesc::Compute(_, b) => b,
            OwnedDesc::Graphics(_, b, ..) => b,
            OwnedDesc::Stream(_, _, b) => b,
        })
    }

    fn ptr(&self) -> *const c_void {
        match self {
            OwnedDesc::Compute(d, _) => &**d as *const _ as *const c_void,
            OwnedDesc::Graphics(d, ..) => &**d as *const _ as *const c_void,
            OwnedDesc::Stream(d, ..) => &**d as *const _ as *const c_void,
        }
    }
}

/// Copy a shader blob into `blobs` and point `b` at the copy.
unsafe fn own_shader(b: &mut D3D12_SHADER_BYTECODE, blobs: &mut Vec<Vec<u8>>) {
    if let Some(bytes) = shader_bytes(b) {
        blobs.push(bytes.to_vec());
        let v = blobs.last().unwrap();
        *b = bytecode(v);
    } else {
        *b = D3D12_SHADER_BYTECODE::default();
    }
}

unsafe fn own_compute(desc: *const c_void) -> Option<OwnedDesc> {
    let mut d: Box<D3D12_COMPUTE_PIPELINE_STATE_DESC> = Box::new(std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC));
    let mut blobs = Vec::new();
    own_shader(&mut d.CS, &mut blobs);
    d.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    Some(OwnedDesc::Compute(d, blobs))
}

/// None with stream output (not copied; such pipelines are built in the call).
unsafe fn own_graphics(desc: *const c_void) -> Option<OwnedDesc> {
    let mut d: Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC> = Box::new(std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    if d.StreamOutput.NumEntries > 0 {
        return None;
    }
    let mut blobs = Vec::new();
    for b in [&mut d.VS, &mut d.PS, &mut d.DS, &mut d.HS, &mut d.GS] {
        own_shader(b, &mut blobs);
    }
    let n = d.InputLayout.NumElements as usize;
    let mut elems = Vec::new();
    let mut names = Vec::new();
    if n > 0 && !d.InputLayout.pInputElementDescs.is_null() {
        elems = std::slice::from_raw_parts(d.InputLayout.pInputElementDescs, n).to_vec();
        for e in &mut elems {
            names.push(std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char).to_owned());
            e.SemanticName = PCSTR(names.last().unwrap().as_ptr() as *const u8);
        }
    }
    d.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: if elems.is_empty() { std::ptr::null() } else { elems.as_ptr() }, NumElements: elems.len() as u32 };
    d.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    Some(OwnedDesc::Graphics(d, blobs, elems, names))
}

/// A pipeline state stream with its shaders copied. None when another subobject points at
/// title memory (input layout, stream output, view instancing).
unsafe fn own_stream(desc: *const c_void) -> Option<OwnedDesc> {
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let mut buf = std::slice::from_raw_parts(base as *const u8, size).to_vec();
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut off = 0usize;
    while off + 8 <= size {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let at = off + if palign == 8 { 8 } else { 4 };
        let word = |buf: &[u8], k: usize| u64::from_le_bytes(buf[at + 8 * k..at + 8 * k + 8].try_into().unwrap());
        match ty {
            1..=6 | 24 | 25 => {
                let (p, n) = (word(&buf, 0) as usize, word(&buf, 1) as usize);
                if p != 0 && n != 0 && crate::gpu::shader::readable(p, n) {
                    blobs.push(std::slice::from_raw_parts(p as *const u8, n).to_vec());
                    buf[at..at + 8].copy_from_slice(&(blobs.last().unwrap().as_ptr() as u64).to_le_bytes());
                }
            }
            7 | 12 | 22 if buf[at..at + psize].iter().any(|&b| b != 0) => return None,
            19 => buf[at..at + 16].fill(0),
            _ => {}
        }
        off += ((at - off) + psize + 7) & !7;
    }
    let d = Box::new([buf.len(), buf.as_ptr() as usize]);
    Some(OwnedDesc::Stream(d, buf, blobs))
}

/// Raw pointers a worker needs (the device lives for the process).
#[derive(Clone, Copy)]
struct Dev(usize, usize);
unsafe impl Send for Dev {}

/// The stand-in for a pipeline with dummy shaders (draws / computes nothing).
unsafe fn standin_graphics(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let d = dummy();
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    copy.VS = bytecode(&d.vs);
    copy.PS = bytecode(&d.ps);
    copy.DS = D3D12_SHADER_BYTECODE::default();
    copy.HS = D3D12_SHADER_BYTECODE::default();
    copy.GS = D3D12_SHADER_BYTECODE::default();
    copy.StreamOutput = std::mem::zeroed();
    copy.InputLayout = D3D12_INPUT_LAYOUT_DESC::default();
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    pc_topology_type(&mut copy);
    finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv)
}

unsafe fn standin_compute(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    copy.CS = bytecode(&dummy().cs);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv)
}

unsafe fn standin_stream(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let Some((buf, _)) = fallback_stream(desc as *const u8) else { return E_INVALIDARG };
    let new_desc: [usize; 2] = [buf.len(), buf.as_ptr() as usize];
    finish_pso(this, f, new_desc.as_ptr() as *const c_void, riid, ppv)
}

/// Hand the title `standin`'s result now and build the real pipeline with `build` on a worker.
unsafe fn go_background(
    this: *mut c_void,
    f: PsoFn,
    desc: *const c_void,
    riid: *const GUID,
    ppv: *mut *mut c_void,
    compute: bool,
    owned: OwnedDesc,
    standin: unsafe fn(*mut c_void, PsoFn, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT,
    build: unsafe fn(Dev, &OwnedDesc, *mut *mut c_void) -> bool,
) -> Option<HRESULT> {
    let hr = standin(this, f, desc, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        static ONCE: AtomicU64 = AtomicU64::new(0);
        if ONCE.fetch_add(1, Ordering::Relaxed) < 3 {
            crate::xlog_warn!("D3D12.X: no stand-in pipeline ({hr:#x}) – building this one in the call");
        }
        return None;
    }
    let dev = Dev(this as usize, f as usize);
    let key = owned.shaders_key();
    let job: super::pso_async::Job = Box::new(move || {
        let mut out: *mut c_void = std::ptr::null_mut();
        if unsafe { build(dev, &owned, &mut out) } && !out.is_null() {
            out as usize
        } else {
            0
        }
    });
    if super::pso_async::lazy() {
        super::pso_async::submit_lazy(*ppv as usize, compute, job, key);
    } else {
        super::pso_async::submit(*ppv as usize, compute, job);
    }
    Some(hr)
}

unsafe fn build_graphics(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_graphics_pso(dev.0 as *mut c_void, std::mem::transmute::<usize, PsoFn>(dev.1), d.ptr(), &iid, out), Some(hr) if hr >= 0)
}

unsafe fn build_compute(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_compute_pso(dev.0 as *mut c_void, std::mem::transmute::<usize, PsoFn>(dev.1), d.ptr(), &iid, out), Some(Ok(hr)) if hr >= 0)
}

unsafe fn build_stream(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_mesh_stream(dev.0 as *mut c_void, d.ptr(), &iid, out), Some(hr) if hr >= 0)
}

pub(super) unsafe extern "system" fn dev_create_graphics_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    graphics_pso(this, desc, riid, ppv, false)
}

/// `dev_create_graphics_pso`; `stream`: the desc is a pipeline stream's (`stream_pipeline`).
unsafe fn graphics_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void, stream: bool) -> HRESULT {
    super::null::capture_graphics_desc(desc as *const u8);
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateGraphicsPipelineState")));
    let (translated, missed) = if super::pso_async::enabled() { attempt(|| translated_graphics_pso(this, f, desc, riid, ppv)) } else { (translated_graphics_pso(this, f, desc, riid, ppv), false) };
    if missed {
        if let Some(hr) = own_graphics(desc).and_then(|o| go_background(this, f, desc, riid, ppv, false, o, standin_graphics, build_graphics)) {
            return hr;
        }
        if let Some(hr) = translated_graphics_pso(this, f, desc, riid, ppv) {
            return hr;
        }
    } else if let Some(hr) = translated {
        return hr;
    }
    let pc = untranslated_stream_desc(desc, stream);
    let hr = finish_pso(this, f, pc.as_ref().map_or(desc, |g| g as *const _ as *const c_void), riid, ppv);
    if hr >= 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return hr;
    }
    let d = dummy();
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    copy.VS = bytecode(&d.vs);
    copy.PS = bytecode(&d.ps);
    copy.DS = D3D12_SHADER_BYTECODE::default();
    copy.HS = D3D12_SHADER_BYTECODE::default();
    copy.GS = D3D12_SHADER_BYTECODE::default();
    copy.StreamOutput = std::mem::zeroed();
    copy.InputLayout = D3D12_INPUT_LAYOUT_DESC::default();
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    pc_topology_type(&mut copy);
    let hr2 = finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv);
    note_fallback("graphics PSO", hr, hr2 >= 0);
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

unsafe fn note_translated_compute(hr: HRESULT, ppv: *mut *mut c_void) {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        TRANSLATED_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    }
}

unsafe extern "system" fn dev_create_compute_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = create_compute_pso(this, desc, riid, ppv);
    if hr >= 0 && !ppv.is_null() {
        super::culprit::note_compute(*ppv as usize, desc);
    }
    hr
}

unsafe fn create_compute_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    super::null::capture_compute_desc(desc as *const u8);
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateComputePipelineState")));
    let (mut console, missed) = if super::pso_async::enabled() { attempt(|| translated_compute_pso(this, f, desc, riid, ppv)) } else { (translated_compute_pso(this, f, desc, riid, ppv), false) };
    if missed {
        if let Some(hr) = own_compute(desc).and_then(|o| go_background(this, f, desc, riid, ppv, true, o, standin_compute, build_compute)) {
            // a console program the worker is translating (its root signature comes with it);
            // the real PSO registers itself in TRANSLATED_COMPUTE once build_compute succeeds
            // (if build_compute fails, the stand-in stays unregistered so Dispatch /
            // ExecuteIndirect never bind XCL's ABI root signature onto it)
            LAST_COMPUTE_TRANSLATED.with(|c| c.set(true));
            LAST_COMPUTE_RS.with(|c| c.set(0));
            note_console_compute(hr, ppv);
            return hr;
        }
        console = translated_compute_pso(this, f, desc, riid, ppv);
    }
    LAST_COMPUTE_TRANSLATED.with(|c| c.set(matches!(console, Some(Ok(hr)) if hr >= 0)));
    if let Some(Ok(hr)) = console {
        return hr;
    }
    // a console shader the recompiler could not take must not reach the driver: AMD's shader
    // compiler crashes on the stripped DXIL (NOBC bitcode) instead of rejecting it
    let hr = if console.is_some() { E_INVALIDARG } else { finish_pso(this, f, desc, riid, ppv) };
    if hr >= 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return hr;
    }
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    copy.CS = bytecode(&dummy().cs);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    let hr2 = finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv);
    note_fallback("compute PSO", hr, hr2 >= 0);
    if console.is_some() {
        note_console_compute(hr2, ppv);
    }
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

// ---------------------------------------------------------------------------------------
// console geometry stages in two passes (MS / NGG, legacy GS): see `pso::TwoPass`
// ---------------------------------------------------------------------------------------

/// What XCL keeps for a mesh pipeline built from console MS / GS code: the title holds the
/// mesh pipeline state (pass-through MS + PS); the draw first runs `compute` (the console
/// program, one record per thread group into `res`), then the mesh pipeline over the records.
pub struct TwoPassPso {
    /// compute pipeline state, owned here, and its root signature (static samplers)
    pub compute: usize,
    pub compute_rs: usize,
    /// root signature of the mesh pipeline (the title's PSO; its PS's static samplers)
    pub mesh_rs: usize,
    pub record: crate::gpu::recomp::MsInterface,
    /// DispatchMesh groups per record (see `pso::TwoPass`)
    pub mesh_groups_per_record: u32,
    /// "MS" or "GS"
    pub kind: &'static str,
}

impl Drop for TwoPassPso {
    fn drop(&mut self) {
        // the compute pipeline state is XCL's (the root signatures are cached or the title's)
        super::pso_async::release(self.compute);
    }
}

static TWO_PASS: Mutex<Option<HashMap<usize, std::sync::Arc<TwoPassPso>>>> = Mutex::new(None);

/// The compute pass of a pipeline state XCL built from console MS / GS code.
pub fn two_pass_of(pso: usize) -> Option<std::sync::Arc<TwoPassPso>> {
    let pso = super::pso_async::resolve(pso);
    TWO_PASS.lock().unwrap().as_ref()?.get(&pso).cloned()
}

pub fn two_pass_count() -> usize {
    TWO_PASS.lock().unwrap().as_ref().map_or(0, |m| m.len())
}

/// Subobject types whose payload is a `D3D12_SHADER_BYTECODE`.
const SHADER_SUBOBJECTS: [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE; 8] = [
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
];

/// A pipeline state stream being written: subobjects aligned as D3D12 expects
/// (type, payload at 4 or 8, size rounded to 8).
#[derive(Default)]
struct StreamOut(Vec<u8>, usize);

impl StreamOut {
    fn raw(&mut self, ty: u32, payload: &[u8], align8: bool) {
        let at = self.0.len();
        self.0.extend_from_slice(&ty.to_le_bytes());
        self.0.resize(at + if align8 { 8 } else { 4 }, 0);
        self.0.extend_from_slice(payload);
        if SHADER_SUBOBJECTS.iter().any(|t| t.0 as u32 == ty) {
            // a D3D12_SHADER_BYTECODE: pointer, then length
            self.1 += payload.get(8..16).and_then(|b| b.try_into().ok()).map_or(0, usize::from_le_bytes);
        }
        let end = (self.0.len() + 7) & !7;
        self.0.resize(end, 0);
    }
    fn put<T>(&mut self, ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, v: &T) {
        let b = unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) };
        self.raw(ty.0 as u32, b, std::mem::align_of::<T>() == 8);
    }
    fn shader(&mut self, ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, code: &[u8]) {
        self.put(ty, &D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as *const c_void, BytecodeLength: code.len() });
    }
    /// Bytes of shader code referenced by this stream. A shader subobject contains only a
    /// pointer and length; `self.0.len()` therefore does not include the code itself.
    fn shader_bytes(&self) -> usize {
        self.1
    }
}

/// The subobjects of a stream: (type, payload).
fn stream_subobjects(buf: &[u8]) -> Option<Vec<(u32, &[u8])>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let poff = if palign == 8 { 8 } else { 4 };
        out.push((ty, buf.get(off + poff..off + poff + psize)?));
        off += (poff + psize + 7) & !7;
    }
    Some(out)
}

/// The stream's subobjects up to the first of a type XCL does not know (logged): a console
/// shader in front of it must still be found.
fn known_subobjects(buf: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let Some((psize, palign)) = super::null::subobject_payload(ty) else {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: pipeline stream: unknown subobject type {ty} at {off:#x} of {:#x} - the rest is not read", buf.len());
            }
            break;
        };
        let poff = if palign == 8 { 8 } else { 4 };
        let Some(p) = buf.get(off + poff..off + poff + psize) else { break };
        out.push((ty, p));
        off += (poff + psize + 7) & !7;
    }
    out
}

unsafe fn payload_shader<'a>(p: &[u8]) -> Option<&'a [u8]> {
    let b = D3D12_SHADER_BYTECODE { pShaderBytecode: u64::from_le_bytes(p[0..8].try_into().ok()?) as usize as *const c_void, BytecodeLength: u64::from_le_bytes(p[8..16].try_into().ok()?) as usize };
    shader_bytes(&b)
}

/// The passthrough mesh shader emits triangles for both translated MS and GS pipelines.
fn two_pass_mesh_stream(rs: *mut c_void, ms: &[u8], ps: Option<&[u8]>) -> StreamOut {
    let mut out = StreamOut::default();
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs);
    out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
    if let Some(ps) = ps {
        out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);
    }
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    out
}

#[cfg(test)]
#[test]
fn passthrough_mesh_stream_accepts_a_triangle_driver_blob() {
    use crate::gpu::recomp::{dxc, runner};
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    if super::backend() != super::Backend::Real {
        return runner::skip_test("no GPU");
    }
    let raw = device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap();
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    if unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS7).cast(), std::mem::size_of_val(&options) as u32) }.is_err()
        || options.MeshShaderTier.0 < D3D12_MESH_SHADER_TIER_1.0
    {
        return runner::skip_test("no mesh shaders");
    }
    let (ms, _) = dummy_mesh().expect("compile a synthetic triangle mesh shader");
    unsafe {
        let mut sig = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut sig, None).unwrap();
        let sig = sig.unwrap();
        let rs: ID3D12RootSignature = dev.CreateRootSignature(
            0, std::slice::from_raw_parts(sig.GetBufferPointer() as *const u8, sig.GetBufferSize()),
        ).unwrap();
        let create: PsoFn = std::mem::transmute(orig_fn(raw, slot_of(raw, "CreatePipelineState")));
        let build = |stream: &StreamOut| {
            let desc = [stream.0.len(), stream.0.as_ptr() as usize];
            let mut pso = std::ptr::null_mut();
            let hr = create(raw, desc.as_ptr().cast(), &pc_iid(Kind::PipelineState), &mut pso);
            assert!(hr >= 0 && !pso.is_null(), "mesh PSO creation failed ({hr:#x})");
            ID3D12PipelineState::from_raw(pso)
        };

        let mut reference = StreamOut::default();
        reference.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs.as_raw());
        reference.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
        reference.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
        let blob = build(&reference).GetCachedBlob().unwrap();
        let mut translated = two_pass_mesh_stream(rs.as_raw(), ms, None);
        let subobjects = stream_subobjects(&translated.0).unwrap();
        let topology = subobjects.iter().find(|(kind, _)| *kind == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY.0 as u32);
        assert_eq!(topology.map(|(_, payload)| *payload), Some(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0.to_le_bytes().as_slice()));
        translated.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &D3D12_CACHED_PIPELINE_STATE {
            pCachedBlob: blob.GetBufferPointer(),
            CachedBlobSizeInBytes: blob.GetBufferSize(),
        });
        build(&translated);
    }
}

#[allow(dead_code)] // Consumed by the stacked driver-cache integration.
mod two_pass_identity;

/// The console PS of a two-pass pipeline (a PC PS passes as it is).
/// The PS of a two-pass pipeline (DXIL, static samplers – a PC PS passes as it is).
fn two_pass_ps(ps: Option<&[u8]>, split_wave64: bool) -> Result<(Option<Vec<u8>>, Vec<crate::gpu::recomp::pso::StaticSampler>), String> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    match ps {
        Some(p) if is_console(p) => console_to_dxil_for(p, Stage::Pixel, 0, split_wave64).map(|c| (Some(c.dxil), c.samplers)).map_err(|e| format!("PS: {e}")),
        Some(p) => Ok((Some(p.to_vec()), Vec::new())),
        None => Ok((None, Vec::new())),
    }
}

/// Create the mesh pipeline (`stream`, RS and shaders already in it) and the compute pass;
/// the title gets the mesh pipeline state.
#[allow(clippy::too_many_arguments)]
unsafe fn finish_two_pass(this: *mut c_void, stream: &StreamOut, tp: crate::gpu::recomp::pso::TwoPass, mesh_rs: *mut c_void, kind: &'static str, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let Some(rs) = abi_rs_for(this, &tp.samplers) else { return E_INVALIDARG };
    let fs: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
    let desc: [usize; 2] = [stream.0.len(), stream.0.as_ptr() as usize];
    // the mesh pass and its compute pass: one turn for both
    let bytes = stream.shader_bytes() + tp.cs.len();
    let gate = bytes >= huge_pso_bytes() && std::thread::current().name().is_some_and(|n| n.starts_with("xcl-pso"));
    let waited = std::time::Instant::now();
    let _slot = gate.then(|| big_compile_turn(compile_slots(bytes), || big_compile_commit_free(big_compile_headroom(bytes)), super::pso_async::build_is_wanted)).flatten();
    if gate {
        crate::xlog_info!("D3D12.X: two-pass {kind} PSO ({bytes} B DXIL) gated: {}, waited {} ms", _slot.is_some(), waited.elapsed().as_millis());
    }
    let hr = driver_compile(kind, stream.shader_bytes(), false, || finish_pso(this, fs, desc.as_ptr() as *const c_void, riid, ppv));
    if hr < 0 {
        note_translation_error(format!("the driver rejected the mesh pipeline from the console {kind} ({hr:#x})"));
        return E_INVALIDARG;
    }
    let fc: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateComputePipelineState")));
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
    cdesc.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    cdesc.CS = bytecode(&tp.cs);
    // Reuse the compiled driver blob of the compute pass across all `(MS/GS, PS)` pipelines that
    // share `tp.cs` (in memory and on disk), matching `translated_compute_pso_once`.
    let ckey = compute_blob_key(&tp.cs, rs as usize, 0);
    let mut ccached = compute_blob(ckey);
    let cdisk = driver_cache::path(&tp.cs, 0);
    let cfrom_disk = ccached.is_none();
    if cfrom_disk {
        ccached = cdisk.as_deref().and_then(driver_cache::load).map(std::sync::Arc::new);
    }
    if let Some(b) = &ccached {
        cdesc.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as *const c_void, CachedBlobSizeInBytes: b.len() };
    }
    let iid = pc_iid(Kind::PipelineState);
    let mut compute: *mut c_void = std::ptr::null_mut();
    let (mut hc, mut cin_driver) = driver_compile_identified("compute pass", tp.cs.len(), false, None, || {
        driver_pso(fc, this, &cdesc as *const _ as *const c_void, &iid, &mut compute)
    });
    if hc < 0 && ccached.is_some() {
        forget_compute_blob(ckey);
        if cfrom_disk {
            driver_cache::note(false, cdisk.as_deref());
        }
        cdesc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        compute = std::ptr::null_mut();
        let (retried, took) = driver_compile_identified("compute pass retry", tp.cs.len(), false, None, || {
            driver_pso(fc, this, &cdesc as *const _ as *const c_void, &iid, &mut compute)
        });
        hc = retried;
        cin_driver += took;
    }
    let creused = hc >= 0 && ccached.is_some() && cdesc.CachedPSO.CachedBlobSizeInBytes != 0;
    if hc >= 0 && !creused && !compute.is_null() {
        keep_compute_blob(ckey, compute, tp.cs.len());
        if let (Some(p), Some(b)) = (cdisk.as_deref(), compute_blob(ckey)) {
            driver_cache::keep(p, &b, cin_driver);
        }
    } else if creused && cfrom_disk {
        driver_cache::note(true, None);
    }
    drop(_slot);
    if hc < 0 || compute.is_null() {
        note_translation_error(format!("the driver rejected the compute pass of the console {kind} ({hc:#x})"));
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(*ppv as *const *const usize)).add(2));
        release(*ppv);
        *ppv = std::ptr::null_mut();
        return E_INVALIDARG;
    }
    note_pso_rs(*ppv, mesh_rs);
    TWO_PASS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, std::sync::Arc::new(TwoPassPso { compute: compute as usize, compute_rs: rs as usize, mesh_rs: mesh_rs as usize, record: tp.record, mesh_groups_per_record: tp.mesh_groups_per_record, kind }));
    let n = TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    if n < 5 || n % 1000 == 0 {
        crate::xlog_info!("D3D12.X: pipeline from the console {kind} in two passes (compute {} B + mesh {} B DXIL)", tp.cs.len(), tp.ms.len());
    }
    hr
}

/// CreatePipelineState with a console MS: Some(hr) when it was a console mesh pipeline
/// (translated, or E_INVALIDARG – console bytecode must not reach the driver).
unsafe fn translated_mesh_stream(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    with_switch_fallback(|| translated_mesh_stream_once(this, desc, riid, ppv))
}

unsafe fn translated_mesh_stream_once(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    use crate::gpu::recomp::pso::{console_mesh, is_console};
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return None;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let subs = stream_subobjects(std::slice::from_raw_parts(base as *const u8, size))?;
    let as_slot = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS.0 as u32).and_then(|s| payload_shader(s.1));
    let ms_slot = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32).and_then(|s| payload_shader(s.1));
    // A console MS container in the AS slot and no MS subobject (a retail game's minimap
    // pipelines): refused. Translated as the mesh shader they are, the title rendered nothing
    // from then on (a black screen every run); refused, it goes on to its menus. What it
    // expects of them is not known yet.
    if let (None, Some(a)) = (ms_slot, as_slot) {
        if is_console(a) && crate::gpu::xbox::parse(a).is_some_and(|sh| sh.stage == "MS") {
            note_translation_error(format!("mesh pipeline with its MS {:016x} in the AS slot – refused", crate::gpu::shader::fnv64(a)));
            return Some(E_INVALIDARG);
        }
    }
    let ms = ms_slot?;
    if !is_console(ms) {
        return None;
    }
    if let Some(a) = as_slot {
        let types: Vec<u32> = subs.iter().map(|s| s.0).collect();
        note_translation_error(format!(
            "mesh pipeline with AS {:016x} (MS {:016x}) – not supported yet; subobjects {types:?}",
            crate::gpu::shader::fnv64(a),
            crate::gpu::shader::fnv64(ms)
        ));
        return Some(E_INVALIDARG);
    }
    // made when first bound (`pso_async::lazy`): not even translated now
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return None;
    }
    let ps = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS.0 as u32).and_then(|s| payload_shader(s.1));
    let (tp, (ps, ps_samplers)) = match (console_mesh(ms, ps, wave32_only(this)), two_pass_ps(ps, graphics_wave32(this))) {
        (Ok(tp), Ok(ps)) => (tp, ps),
        (Err(e), _) => {
            note_translation_error(format!("MS: {e}"));
            return Some(E_INVALIDARG);
        }
        (_, Err(e)) => {
            note_translation_error(e);
            return Some(E_INVALIDARG);
        }
    };
    let Some(rs) = abi_rs_for(this, &ps_samplers) else { return Some(E_INVALIDARG) };
    // the title's state (blend, rasteriser, depth, formats …) with XCL's RS and shaders
    let mut out = two_pass_mesh_stream(rs, &tp.ms, ps.as_deref());
    for (ty, payload) in &subs {
        let skip = [0, 2, 14, 19, 24, 25].contains(ty); // RS, PS, topology, cached PSO, AS, MS
        if !skip {
            let (_, palign) = super::null::subobject_payload(*ty).unwrap_or((0, 4));
            // RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT: the console's formats as the PC's
            if (*ty == 15 && payload.len() >= 32) || (*ty == 16 && payload.len() >= 4) {
                let mut p = payload.to_vec();
                for k in 0..if *ty == 15 { 8 } else { 1 } {
                    let f = super::device_x::pc_target_format(u32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap()));
                    p[4 * k..4 * k + 4].copy_from_slice(&f.to_le_bytes());
                }
                out.raw(*ty, &p, palign == 8);
                continue;
            }
            out.raw(*ty, payload, palign == 8);
        }
    }
    Some(finish_two_pass(this, &out, tp, rs, "MS", riid, ppv))
}

/// A graphics PSO with a console VS + GS: the GS pipeline in two passes, as a mesh pipeline
/// with the description's state.
unsafe fn translated_gs_pso(this: *mut c_void, d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, vs: &[u8], gs: &[u8], riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let (tp, (ps, ps_samplers)) = match (
        crate::gpu::recomp::pso::console_geometry(vs, gs, shader_bytes(&d.PS), wave32_only(this)),
        two_pass_ps(shader_bytes(&d.PS), graphics_wave32(this)),
    ) {
        (Ok(tp), Ok(ps)) => (tp, ps),
        (Err(e), _) => {
            // what the containers hold: each FSSM's hardware stages and its programs
            let describe = |blob: &[u8]| -> String {
                crate::gpu::xbox::parse(blob).map_or("not a console container".into(), |sh| {
                    sh.fssm
                        .iter()
                        .map(|f| {
                            format!(
                                "[{} | {} programs {:?}, version {:#x}, {} B, code {:?} (from header {}), error {:?}]",
                                f.stages.iter().map(|x| x.label()).collect::<Vec<_>>().join(", "),
                                f.programs.len(),
                                f.programs.iter().map(|p| p.len).collect::<Vec<_>>(),
                                f.version,
                                f.data.len(),
                                f.code,
                                f.code_from_header,
                                f.error
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            };
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: VS+GS pipeline not translated ({e}); VS {}; GS {}", describe(vs), describe(gs));
                if let Ok(dir) = std::env::var("XCL_DUMP_GS") {
                    let k = N.load(Ordering::Relaxed);
                    let _ = std::fs::write(format!("{dir}/vs{k}.bin"), vs);
                    let _ = std::fs::write(format!("{dir}/gs{k}.bin"), gs);
                }
            }
            note_translation_error(format!("VS+GS: {e}"));
            return E_INVALIDARG;
        }
        (_, Err(e)) => {
            note_translation_error(e);
            return E_INVALIDARG;
        }
    };
    let Some(rs) = abi_rs_for(this, &ps_samplers) else { return E_INVALIDARG };
    let mut out = two_pass_mesh_stream(rs, &tp.ms, ps.as_deref());
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &d.BlendState);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &d.SampleMask);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &d.RasterizerState);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, &d.DepthStencilState);
    let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: d.NumRenderTargets, ..Default::default() };
    rt.RTFormats = d.RTVFormats;
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &d.DSVFormat);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &d.SampleDesc);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, &d.NodeMask);
    if d.Flags.0 != 0 {
        out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS, &d.Flags);
    }
    finish_two_pass(this, &out, tp, rs, "GS", riid, ppv)
}

/// Rewrite a pipeline state stream with stand-in shaders. None if not possible.
unsafe fn fallback_stream(desc: *const u8) -> Option<(Vec<u8>, [usize; 2])> {
    let size = *(desc as *const usize);
    let base = *(desc.add(8) as *const usize);
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let mut buf = std::slice::from_raw_parts(base as *const u8, size).to_vec();
    let d = dummy();
    let mesh = stream_subobjects(&buf).is_some_and(|subs| subs.iter().any(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32));
    let dm = if mesh { Some(dummy_mesh()?) } else { None };
    let mut off = 0usize;
    while off + 8 <= size {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let poff = if palign == 8 { 8 } else { 4 };
        let at = off + poff;
        if at.checked_add(psize)? > size {
            return None; // SizeInBytes ends inside the payload
        }
        let put = |buf: &mut Vec<u8>, b: &[u8]| {
            // no shader: a null pointer – an empty slice's is dangling (0x1), which the driver
            // refused (a retail game's mesh stand-ins, E_INVALIDARG)
            let ptr = if b.is_empty() { 0 } else { b.as_ptr() as u64 };
            buf[at..at + 8].copy_from_slice(&ptr.to_le_bytes());
            buf[at + 8..at + 16].copy_from_slice(&(b.len() as u64).to_le_bytes());
        };
        match ty {
            1 => put(&mut buf, &d.vs),
            2 => put(&mut buf, dm.map_or(&d.ps[..], |m| &m.1[..])),
            6 => put(&mut buf, &d.cs),
            3 | 4 | 5 | 24 => put(&mut buf, &[]),
            25 => put(&mut buf, &dm?.0),
            12 | 19 => buf[at..at + 16].fill(0), // input layout, cached PSO
            7 => buf[at..at + 32].fill(0),        // stream output
            _ => {}
        }
        off += (poff + psize + 7) & !7;
    }
    Some((buf, [size, 0]))
}

/// The console's pipeline topology type of rect lists (draws of topology 17, `list_x::RECT_LIST`).
const RECT_LIST_TYPE: i32 = 19;

/// `create(copy)`, and once more without the generated rect list GS (`rect_gs`) when the driver
/// refuses the pipeline with it: each rect is then drawn as the triangle of its three vertices,
/// as without the GS, rather than not at all (the untranslated fallback's dummy shaders).
fn retry_without_rect_gs(copy: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC, rect_gs: bool, mut create: impl FnMut(&D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> HRESULT) -> HRESULT {
    let hr = create(copy);
    if hr >= 0 || !rect_gs {
        return hr;
    }
    note_translation_error(format!("the driver rejected the PSO with the rect list GS ({hr:#x}): rect list drawn as triangles"));
    copy.GS = D3D12_SHADER_BYTECODE::default();
    create(copy)
}

/// What an untranslated pipeline from a stream is made from: a rect list's (type 19, which
/// `stream_pipeline` keeps for `translated_graphics_pso`) with the PC's topology type, as the
/// stream path made it before; None: `desc` as it is.
unsafe fn untranslated_stream_desc(desc: *const c_void, stream: bool) -> Option<D3D12_GRAPHICS_PIPELINE_STATE_DESC> {
    if !stream || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let mut g = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    if g.PrimitiveTopologyType.0 != RECT_LIST_TYPE {
        return None;
    }
    pc_topology_type(&mut g);
    Some(g)
}

/// The GS that completes a rect list's rects behind the console VS `vs` and the PS `ps`; refused
/// for a PS that is not a console one (its inputs' modes are not known).
fn rect_list_gs(vs: &[u8], ps: Option<&[u8]>) -> Result<Vec<u8>, String> {
    use crate::gpu::recomp::pso;
    if ps.is_some_and(|p| !pso::is_console(p)) {
        return Err("a PC pixel shader".into());
    }
    pso::rect_list_gs(vs)
}

/// A topology type the PC has: the console's for rect lists (19, with draws of topology 17 –
/// `list_x::RECT_LIST`) is TRIANGLE, as those draws become triangle lists (the rects' fourth
/// corners come from `pso::rect_list_gs`, added by `translated_graphics_pso`); UNDEFINED (a stream
/// without one) is TRIANGLE, or PATCH with a hull shader – the PC refuses both with a VS.
fn pc_topology_type(g: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC) {
    // (and the console's D16S8 depth format the PC's: `device_x::pc_view_format`)
    g.DSVFormat = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_view_format(g.DSVFormat.0 as u32) as i32);
    for f in g.RTVFormats.iter_mut() {
        *f = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_target_format(f.0 as u32) as i32);
    }
    if !(1..=4).contains(&g.PrimitiveTopologyType.0) {
        g.PrimitiveTopologyType = if g.HS.pShaderBytecode.is_null() { D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE } else { D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH };
    }
}

enum StreamPipeline {
    Graphics(Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC>),
    Compute(Box<D3D12_COMPUTE_PIPELINE_STATE_DESC>),
    /// neither with console shaders (or not readable)
    Unknown,
}

/// A pipeline stream with a console VS (+ PS / GS) or CS as the plain pipeline desc, which is
/// translated like any other (`dev_create_graphics_pso`). The stream itself must not reach the
/// driver: AMD's shader compiler fails on the console's stripped DXIL and takes the device with
/// it (0x887A0020 at the retail game's first graphics pipeline). Depth bounds, view instancing and the
/// float depth bias' fraction have no place in the plain desc.
unsafe fn stream_pipeline(desc: *const c_void) -> StreamPipeline {
    use crate::gpu::recomp::pso::is_console;
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return StreamPipeline::Unknown;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return StreamPipeline::Unknown;
    }
    let subs = known_subobjects(std::slice::from_raw_parts(base as *const u8, size));
    let console = |ty: u32| subs.iter().find(|s| s.0 == ty).and_then(|s| payload_shader(s.1)).is_some_and(is_console);
    if console(6) {
        let mut c: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::mem::zeroed();
        for (ty, p) in &subs {
            match ty {
                0 => put(&mut c.pRootSignature, p),
                6 => put(&mut c.CS, p),
                18 => put(&mut c.NodeMask, p),
                _ => {}
            }
        }
        return StreamPipeline::Compute(Box::new(c));
    }
    if !console(1) {
        return StreamPipeline::Unknown;
    }
    let mut g: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::mem::zeroed();
    // what a stream leaves out has the PC's defaults
    g.SampleMask = u32::MAX;
    g.SampleDesc.Count = 1;
    g.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    g.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    g.RasterizerState.DepthClipEnable = true.into();
    for rt in &mut g.BlendState.RenderTarget {
        rt.SrcBlend = D3D12_BLEND_ONE;
        rt.DestBlend = D3D12_BLEND_ZERO;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = 15;
    }
    let keep = D3D12_DEPTH_STENCILOP_DESC { StencilFailOp: D3D12_STENCIL_OP_KEEP, StencilDepthFailOp: D3D12_STENCIL_OP_KEEP, StencilPassOp: D3D12_STENCIL_OP_KEEP, StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS };
    g.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: true.into(),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D12_COMPARISON_FUNC_LESS,
        StencilEnable: false.into(),
        StencilReadMask: 0xFF,
        StencilWriteMask: 0xFF,
        FrontFace: keep,
        BackFace: keep,
    };
    let mut has_depth_stencil = false;
    for (ty, p) in &subs {
        match ty {
            0 => put(&mut g.pRootSignature, p),
            1 => put(&mut g.VS, p),
            2 => put(&mut g.PS, p),
            3 => put(&mut g.DS, p),
            4 => put(&mut g.HS, p),
            5 => put(&mut g.GS, p),
            7 => put(&mut g.StreamOutput, p),
            8 => put(&mut g.BlendState, p),
            9 => put(&mut g.SampleMask, p),
            10 => put(&mut g.RasterizerState, p),
            // DEPTH_STENCIL; DEPTH_STENCIL1 is that and the depth bounds test
            11 | 21 => {
                has_depth_stencil = true;
                put(&mut g.DepthStencilState, p);
            }
            12 => put(&mut g.InputLayout, p),
            13 => put(&mut g.IBStripCutValue, p),
            14 => put(&mut g.PrimitiveTopologyType, p),
            15 => {
                let mut rt: D3D12_RT_FORMAT_ARRAY = std::mem::zeroed();
                put(&mut rt, p);
                g.RTVFormats = rt.RTFormats;
                g.NumRenderTargets = rt.NumRenderTargets;
            }
            16 => put(&mut g.DSVFormat, p),
            17 => put(&mut g.SampleDesc, p),
            18 => put(&mut g.NodeMask, p),
            20 => put(&mut g.Flags, p),
            26 => {
                has_depth_stencil = true;
                let mut d: D3D12_DEPTH_STENCIL_DESC2 = std::mem::zeroed();
                put(&mut d, p);
                let face = |f: &D3D12_DEPTH_STENCILOP_DESC1| D3D12_DEPTH_STENCILOP_DESC { StencilFailOp: f.StencilFailOp, StencilDepthFailOp: f.StencilDepthFailOp, StencilPassOp: f.StencilPassOp, StencilFunc: f.StencilFunc };
                g.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: d.DepthEnable,
                    DepthWriteMask: d.DepthWriteMask,
                    DepthFunc: d.DepthFunc,
                    StencilEnable: d.StencilEnable,
                    StencilReadMask: d.FrontFace.StencilReadMask,
                    StencilWriteMask: d.FrontFace.StencilWriteMask,
                    FrontFace: face(&d.FrontFace),
                    BackFace: face(&d.BackFace),
                };
            }
            27 | 28 => {
                // RASTERIZER1 / 2: fill, cull, winding, depth bias (float), clamp, slope, depth clip
                let mut d: D3D12_RASTERIZER_DESC1 = std::mem::zeroed();
                put(&mut d, p);
                g.RasterizerState = D3D12_RASTERIZER_DESC {
                    FillMode: d.FillMode,
                    CullMode: d.CullMode,
                    FrontCounterClockwise: d.FrontCounterClockwise,
                    DepthBias: d.DepthBias as i32,
                    DepthBiasClamp: d.DepthBiasClamp,
                    SlopeScaledDepthBias: d.SlopeScaledDepthBias,
                    DepthClipEnable: d.DepthClipEnable,
                    ..Default::default()
                };
            }
            _ => {}
        }
    }
    // D3D12 pipeline state streams default DepthEnable to FALSE when no depth-stencil
    // subobject is supplied and DSVFormat is DXGI_FORMAT_UNKNOWN (0).
    if !has_depth_stencil && g.DSVFormat.0 == 0 {
        g.DepthStencilState.DepthEnable = false.into();
    }
    // a rect list stays one for translated_graphics_pso, which gives it its GS (untranslated, it
    // becomes the PC's in graphics_pso)
    if g.PrimitiveTopologyType.0 != RECT_LIST_TYPE {
        pc_topology_type(&mut g);
    }
    StreamPipeline::Graphics(Box::new(g))
}

/// The bytes of a stream payload as a desc field (no references counted: the desc only borrows).
unsafe fn put<T>(to: &mut T, from: &[u8]) {
    let n = std::mem::size_of::<T>().min(from.len());
    std::ptr::copy_nonoverlapping(from.as_ptr(), to as *mut T as *mut u8, n);
}

pub(super) unsafe extern "system" fn dev_create_pipeline_state(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    super::null::capture_stream_desc(desc as *const u8);
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
    let (translated, missed) = if super::pso_async::enabled() { attempt(|| translated_mesh_stream(this, desc, riid, ppv)) } else { (translated_mesh_stream(this, desc, riid, ppv), false) };
    if missed {
        if let Some(hr) = own_stream(desc).and_then(|o| go_background(this, f, desc, riid, ppv, false, o, standin_stream, build_stream)) {
            return hr;
        }
        if let Some(hr) = translated_mesh_stream(this, desc, riid, ppv) {
            return hr;
        }
    } else if let Some(hr) = translated {
        return hr;
    }
    match stream_pipeline(desc) {
        StreamPipeline::Graphics(d) => return graphics_pso(this, &*d as *const _ as *const c_void, riid, ppv, true),
        StreamPipeline::Compute(d) => return dev_create_compute_pso(this, &*d as *const _ as *const c_void, riid, ppv),
        StreamPipeline::Unknown => {}
    }
    let hr = finish_pso(this, f, desc, riid, ppv);
    if hr >= 0 || !crate::gpu::shader::readable(desc as usize, 16) {
        return hr;
    }
    let Some((buf, _)) = fallback_stream(desc as *const u8) else {
        note_fallback("PSO stream", hr, false);
        return hr;
    };
    let new_desc: [usize; 2] = [buf.len(), buf.as_ptr() as usize];
    let hr2 = finish_pso(this, f, new_desc.as_ptr() as *const c_void, riid, ppv);
    note_fallback("PSO stream", hr, hr2 >= 0);
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

unsafe extern "system" fn dev_create_root_signature(this: *mut c_void, mask: u32, blob: *const c_void, len: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    crate::gpu::shader::capture("rootsig", blob as usize, len);
    type F = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const GUID, *mut *mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CreateRootSignature")));
    let mut hr = f(this, mask, blob, len, riid, ppv);
    if hr == E_NOINTERFACE {
        let iid = pc_iid(Kind::RootSignature);
        hr = f(this, mask, blob, len, &iid, ppv);
    }
    if hr < 0 {
        // console root signature rejected → empty one so the title keeps going
        let desc = D3D12_ROOT_SIGNATURE_DESC { Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, ..Default::default() };
        let mut ser = None;
        if D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).is_ok() {
            if let Some(b) = ser {
                let iid = pc_iid(Kind::RootSignature);
                let hr2 = f(this, mask, b.GetBufferPointer(), b.GetBufferSize(), &iid, ppv);
                if hr2 >= 0 {
                    let n = FALLBACK_RS.fetch_add(1, Ordering::Relaxed);
                    if n < 5 {
                        crate::xlog_warn!("D3D12.X: root signature rejected ({hr:#x}) – empty fallback (#{})", n + 1);
                    }
                    hr = hr2;
                }
            }
        }
    }
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        adopt(*ppv, Kind::RootSignature);
        if !blob.is_null() {
            super::console_list::note_root_signature(*ppv as usize, std::slice::from_raw_parts(blob as *const u8, len));
        }
    }
    hr
}

// ---------------------------------------------------------------------------------------
// presentation tracking
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn dev_create_rtv(this: *mut c_void, res: *mut c_void, desc: *const c_void, handle: usize) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, usize);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CreateRenderTargetView")));
    // a view in a console-only format, or of a target made R8G8B8A8 (`device_x::pc_resource_desc`)
    let mut copy = (!desc.is_null()).then(|| (desc as *const D3D12_RENDER_TARGET_VIEW_DESC).read_unaligned());
    if let Some(c) = copy.as_mut() {
        let rf = ID3D12Resource::from_raw_borrowed(&res).map_or(0, |r| r.GetDesc().Format.0 as u32);
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_shader_view_format(rf, c.Format.0 as u32) as i32);
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _ as *const c_void);
    f(this, res, desc, handle);
    super::present::note_rtv(handle, res);
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 12 {
            let d = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc()).map(|d| (d.Width, d.Height, d.Format.0));
            crate::xlog_info!("frame trace: CreateRenderTargetView {handle:#x} → {res:p} {d:?}");
        }
    }
}

/// ClearRenderTargetView: as on the PC (`XCL_TRACE_FRAME`: the colours logged)
unsafe extern "system" fn cl_clear_rtv(this: *mut c_void, rtv: usize, color: *const [f32; 4], n: u32, rects: *const c_void) {
    type F = unsafe extern "system" fn(*mut c_void, usize, *const [f32; 4], u32, *const c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ClearRenderTargetView")));
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 16 {
            crate::xlog_info!("frame trace: ClearRenderTargetView {rtv:#x} → {:?} colour {:?}, {n} rects", super::present::rtv_resource(rtv), color.as_ref());
        }
        if let Some(r) = super::present::rtv_resource(rtv) {
            super::present::note_use(r, super::present::USE_CLEAR);
        }
    }
    f(this, rtv, color, n, rects)
}

unsafe extern "system" fn cl_om_set_render_targets(this: *mut c_void, n: u32, rtvs: *const usize, single: i32, dsv: *const usize) {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "OMSetRenderTargets")));
    f(this, n, rtvs, single, dsv);
    super::list_x::note_render_targets(this, n, rtvs, single, dsv);
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 12 {
            let first = if n > 0 && !rtvs.is_null() { *rtvs } else { 0 };
            crate::xlog_info!("frame trace: OMSetRenderTargets({n}, first {first:#x} → {:?}, single {single}, dsv {:?})", super::present::rtv_resource(first), dsv.as_ref().copied());
        }
    }
    if n > 0 && !rtvs.is_null() {
        super::present::note_rt0(this as usize, *rtvs);
        if super::present::trace_frame() {
            for k in 0..n as usize {
                let h = if single != 0 { *rtvs + k * 32 } else { *rtvs.add(k) };
                if let Some(r) = super::present::rtv_resource(h) {
                    super::present::note_use(r, super::present::USE_RT);
                }
            }
        }
    }
}

/// A stand-in pipeline is bound as its real pipeline once that is built (compute: waited for).
/// SetPipelineState on the PC list (a pipeline still building in the background: its stand-in)
pub(super) unsafe fn bind_pipeline(this: *mut c_void, pso: *mut c_void) {
    cl_set_pipeline_state(this, pso)
}

unsafe extern "system" fn cl_set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "SetPipelineState")));
    f(this, super::pso_async::resolve_for_bind(pso as usize) as *mut c_void);
    super::list_x::on_set_pipeline_state(this, pso as usize);
}

unsafe extern "system" fn cl_clear_state(this: *mut c_void, pso: *mut c_void) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ClearState")));
    f(this, super::pso_async::resolve_for_bind(pso as usize) as *mut c_void);
    super::list_x::on_clear_state(this, pso as usize);
    super::bind::clear_state(this, pso as usize);
}

unsafe extern "system" fn cl_draw_instanced(this: *mut c_void, vcount: u32, icount: u32, sv: u32, si: u32) {
    super::list_x::prepare_draw(this);
    type F = unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "DrawInstanced")));
    f(this, vcount, icount, sv, si);
}

unsafe extern "system" fn cl_draw_indexed_instanced(this: *mut c_void, icount: u32, inst: u32, si: u32, bv: i32, sinst: u32) {
    super::list_x::prepare_draw(this);
    type F = unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "DrawIndexedInstanced")));
    f(this, icount, inst, si, bv, sinst);
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn cl_execute_indirect(
    this: *mut c_void,
    sig: *mut c_void,
    max_count: u32,
    arg_buf: *mut c_void,
    arg_off: u64,
    cnt_buf: *mut c_void,
    cnt_off: u64,
) {
    super::list_x::prepare_draw(this);
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut c_void, u64, *mut c_void, u64);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ExecuteIndirect")));
    f(this, sig, max_count, arg_buf, arg_off, cnt_buf, cnt_off);
}

unsafe extern "system" fn cl_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        super::list_x::on_release(this as usize);
        super::bind::forget(this);
    }
    n
}

unsafe extern "system" fn q_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        super::list_x::on_queue_release(this as usize);
    }
    n
}

unsafe extern "system" fn alloc_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = f(this);
    if n == 0 {
        super::alloc_pool::forget(this as usize);
        super::list_x::on_allocator_release(this as usize);
    }
    n
}

/// Reset the (adopted) PC allocator `raw` itself.
pub(super) unsafe fn reset_pc_allocator(raw: *mut c_void) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(raw, slot_of(raw, "Reset")));
    f(raw)
}

unsafe extern "system" fn alloc_reset(this: *mut c_void) -> HRESULT {
    // (lists may still hold recordings on it that the title submits again: `alloc_pool`)
    let hr = match super::alloc_pool::title_reset(this) {
        Some(hr) => hr,
        None => reset_pc_allocator(this),
    };
    if hr >= 0 {
        // the GPU is done with what lists recorded from this allocator used
        super::console_list::allocator_reset(this);
    }
    hr
}

// ---------------------------------------------------------------------------------------
// command queue layouts
// ---------------------------------------------------------------------------------------
//
// Each console title runs on the game OS it ships with, and the D3D12 runtime there sets the
// queue's vtable. Two layouts are known, with the same D3D12 SDK version (0x20004) and the
// same slots up to ExecuteCommandLists (#10):
// * the older (XDefiant, Minecraft, Fortnite): the PC's #11–#18 (SetMarker … Signal #14, Wait
//   #15, GetTimestampFrequency #16, GetClockCalibration #17, GetDesc #18), SignalX #23,
//   PresentX #30;
// * a newer one (Unicorn Overlord): two more methods before Signal – Signal #16, Wait #17,
//   GetTimestampFrequency #18, GetClockCalibration #19, GetDesc #20, SignalX #25, PresentX #32.
// Nothing the title passes at device creation tells them apart, but its calls do: Signal and
// Wait take a fence, the older layout's #16 / #17 (GetTimestampFrequency / GetClockCalibration)
// never do. A title calling #16 / #17 with one of its fences uses the newer layout, and the
// slots that differ follow that from then on.
//
// The same runtime moves the command list's console methods too: Unicorn Overlord records its
// texture uploads with CopyTextureRegionX at #85 (the older layout's #83; its #85 is
// SetPredicationBufferX, which dropped them – every texture stayed zero). From #85 on the
// list's console methods sit two slots later (`NEWER_LIST_SLOTS`), and from #72 on as well: a retail game
// calls Write64BitValueBottomOfPipeX at #72 and Wait64BitValueX at #74, Write32BitValueBottomOfPipeX
// at #71 (address, value, 0, 0) and ExecuteIndirect at #61 (signature, 1, argument buffer, offset,
// NULL, 0) – sent to FillMemoryWith32BitValueX, the title's indirect draws and dispatches never ran.
// So the shift starts at #59 or below; where is not known. The newer layout's #139 is a Reset with one argument more (the same
// title: (allocator, NULL, 0) on each list it takes from its pool before recording – as a stub
// the list stayed closed, the debug layer saw every call on it refused, and Close failed).

/// the title's runtime has the newer layouts (queue: two methods before Signal; command list:
/// console methods from CopyTextureRegionX on two slots later)
pub(super) static QUEUE_SHIFTED: AtomicBool = AtomicBool::new(false);

/// the title's runtime has the older queue layout: it signalled / waited on a fence at #14 / #15
/// (on the newer layout those slots are other methods)
static QUEUE_OLDER: AtomicBool = AtomicBool::new(false);

/// command list slots of the newer layout → the older layout's slot of the same method
const NEWER_LIST_SLOTS: [(usize, usize); 21] = [(61, 59), (62, 60), (63, 61), (64, 62), (65, 63), (70, 68), (71, 69), (72, 70), (73, 71), (74, 72), (80, 78), (83, 81), (85, 83), (86, 84), (87, 85), (90, 88), (91, 89), (92, 90), (95, 93), (96, 94), (139, 10)];

/// The title called device method `slot`, which only runtimes with the newer layouts have (a retail game
/// records its first lists before any queue call could tell).
pub(super) fn note_newer_device_method(slot: usize) {
    if !QUEUE_OLDER.load(Ordering::Acquire) && !QUEUE_SHIFTED.swap(true, Ordering::AcqRel) {
        crate::xlog_info!("D3D12.X: the title's runtime has the newer console layouts (device method #{slot} called)");
    }
}

/// Whether `p` is a fence the title created.
fn is_fence(p: usize) -> bool {
    p != 0 && FENCES.read().unwrap_or_else(|e| e.into_inner()).contains(&p)
}

/// first queue slot `queue_dispatch` handles
const QUEUE_FIRST: usize = 14;

/// A title's call of queue slot `slot` (arguments as integers; Signal / Wait / SignalX /
/// PresentX / GetTimestampFrequency / GetClockCalibration / GetDesc all fit).
unsafe fn queue_dispatch(slot: usize, this: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize) -> usize {
    const PC_SIGNAL: usize = 14;
    const PC_WAIT: usize = 15;
    const PC_TIMESTAMP_FREQUENCY: usize = 16;
    const PC_CLOCK_CALIBRATION: usize = 17;
    const PC_GET_DESC: usize = 18;
    if (slot == 16 || slot == 17) && is_fence(a1) && !QUEUE_SHIFTED.swap(true, Ordering::AcqRel) {
        crate::xlog_info!("D3D12.X: the title's runtime has the newer console layouts (queue: Signal #16, PresentX #32; command list: CopyTextureRegionX #85 – a fence passed to queue #{slot})");
    }
    let shifted = QUEUE_SHIFTED.load(Ordering::Acquire);
    if (slot == 14 || slot == 15) && !shifted && is_fence(a1) {
        QUEUE_OLDER.store(true, Ordering::Release);
    }
    // the method in the older layout's numbering
    let older = if shifted && slot >= 16 { slot - 2 } else if shifted { usize::MAX } else { slot };
    let pc = |s: usize| orig_fn(this, s);
    match older {
        // Signal / Wait take a fence: anything else is not one of them
        14 | 15 if is_fence(a1) => {
            if older == 14 {
                super::list_x::break_cycles_before_title_signal(this as usize, a1, a2 as u64);
            }
            let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 = std::mem::transmute(pc(if older == 14 { PC_SIGNAL } else { PC_WAIT }));
            let hr = f(this, a1, a2);
            if hr >= 0 {
                if older == 14 {
                    super::fence_x::note_queue_signal(a1, a2 as u64);
                    super::list_x::note_title_queue_signal(this as usize, a1, a2 as u64);
                } else {
                    super::fence_x::note_queue_wait(a1, a2 as u64);
                    super::list_x::note_title_queue_wait(this as usize, a1, a2 as u64);
                }
            }
            hr as u32 as usize
        }
        16 => {
            let f: unsafe extern "system" fn(*mut c_void, usize) -> i32 = std::mem::transmute(pc(PC_TIMESTAMP_FREQUENCY));
            f(this, a1) as u32 as usize
        }
        17 => {
            let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 = std::mem::transmute(pc(PC_CLOCK_CALIBRATION));
            f(this, a1, a2) as u32 as usize
        }
        // #18 before either layout is known: the older layout's GetDesc (a 16-byte structure) or
        // the newer one's GetTimestampFrequency (a UINT64). Only the first 8 bytes of the
        // description are written (type and priority), so an 8-byte out-param is not overrun.
        18 if !shifted && !QUEUE_OLDER.load(Ordering::Acquire) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!("D3D12.X: queue #18 before the title's first Signal / Wait: the queue layout is not known yet, GetDesc writes only its first 8 bytes"));
            let f: unsafe extern "system" fn(*mut c_void, usize) -> usize = std::mem::transmute(pc(PC_GET_DESC));
            let mut desc = [0u8; 16];
            f(this, desc.as_mut_ptr() as usize);
            if a1 != 0 {
                std::ptr::copy_nonoverlapping(desc.as_ptr(), a1 as *mut u8, 8);
            }
            a1
        }
        // GetDesc returns its structure through the hidden pointer, and that pointer
        18 => {
            let f: unsafe extern "system" fn(*mut c_void, usize) -> usize = std::mem::transmute(pc(PC_GET_DESC));
            f(this, a1)
        }
        23 => q_signal_x(this, a1 as *mut c_void, a2 as u64, a3 as u32) as u32 as usize,
        30 => super::present::present_x(this, a1 as u32, a2 as *const u8, a3 as *const u8) as u32 as usize,
        // the newer layout's #35: ExecuteCommandListsX(count, lists, a value, flags) – a retail
        // game submits all its command lists this way (flags 3; 0 for the one it recorded at
        // start); as a stub nothing the title recorded ran on the GPU but its first two frames
        33 if shifted => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_info!("D3D12.X: queue #35 ExecuteCommandListsX ({} lists, value {a3:#x}, flags {a4:#x})", a1));
            q_execute_command_lists(this, a1 as u32, a2 as *const *mut c_void);
            0
        }
        _ => {
            // a console method not modelled: logged once, S_OK (as the stubs it replaces; frame
            // pacing on queue methods keeps working for titles without PresentX)
            static INFOS: Mutex<Vec<(usize, &'static stubs::StubInfo)>> = Mutex::new(Vec::new());
            let info = {
                let mut g = INFOS.lock().unwrap();
                match g.iter().find(|e| e.0 == slot) {
                    Some(e) => e.1,
                    None => {
                        let i = stubs::new_info(DLL, &format!("ID3D12CommandQueue::#{slot} ({} layout)", if shifted { "newer" } else { "older" }), 0);
                        i.hook.store(super::present::on_queue_console_method as usize, Ordering::Release);
                        g.push((slot, i));
                        i
                    }
                }
            };
            let _ = a4;
            stubs::record_call(info, 0);
            0
        }
    }
}

macro_rules! queue_slots {
    ($($slot:literal),*) => {
        /// the title's entry for each queue slot from `QUEUE_FIRST`
        static QUEUE_SLOTS: &[unsafe extern "system" fn(*mut c_void, usize, usize, usize, usize) -> usize] = &[$({
            unsafe extern "system" fn f(this: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize) -> usize {
                queue_dispatch($slot, this, a1, a2, a3, a4)
            }
            f
        }),*];
    };
}
queue_slots!(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35);

/// command queue #23 SignalX(fence, value, flush flags): Signal with the console's cache flush
/// options – the PC's Signal makes the queue's writes visible itself. (A stub that signalled
/// nothing left Fortnite's render fences pending: its render thread waited, textures were
/// never released, and the garbage collector ended the game after 10 s.)
unsafe extern "system" fn q_signal_x(this: *mut c_void, fence: *mut c_void, value: u64, _flags: u32) -> HRESULT {
    const PC_SIGNAL: usize = 14;
    super::list_x::break_cycles_before_title_signal(this as usize, fence as usize, value);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT = std::mem::transmute(orig_fn(this, PC_SIGNAL));
    let hr = f(this, fence, value);
    if hr >= 0 {
        super::fence_x::note_queue_signal(fence as usize, value);
        super::list_x::note_title_queue_signal(this as usize, fence as usize, value);
    }
    hr
}

/// Fence::Signal (CPU)
unsafe extern "system" fn fence_signal(this: *mut c_void, value: u64) -> HRESULT {
    let f: unsafe extern "system" fn(*mut c_void, u64) -> HRESULT = std::mem::transmute(orig_fn(this, slot_of(this, "Signal")));
    let hr = f(this, value);
    super::fence_x::note_cpu_signal(this as usize);
    hr
}

unsafe extern "system" fn q_execute_command_lists(this: *mut c_void, n: u32, lists: *const *mut c_void) {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "ExecuteCommandLists")));
    if n == 0 || lists.is_null() {
        f(this, n, lists);
        stop_if_device_removed();
        return;
    }
    // textures in title memory the CPU wrote since: uploaded again first
    super::title_textures::refresh(this);
    // console-shaped lists run as the PC lists they were translated into (split title lists expand
    // into all of their recorded PC segments in exact chronological order)
    let mut mapped: Vec<*mut c_void> = Vec::with_capacity(n as usize);
    for &l in std::slice::from_raw_parts(lists, n as usize) {
        if let Some(pc) = super::console_list::pc_list(l) {
            mapped.push(pc);
        } else if !super::title_list::push_segments(l, &mut mapped) {
            mapped.push(l);
        }
    }
    super::list_x::note_queue_for_lists(this as usize, &mapped);
    super::list_x::execute_with_waits_on(this, &mapped, |chunk| {
        f(this, chunk.len() as u32, chunk.as_ptr());
        super::list_x::copy_scratch_submitted(this, chunk);
        super::alloc_pool::executed(this, chunk);
        super::list_x::signal_write64_fences(this, chunk);
        super::list_x::readback_sparse_writes(this, chunk);
        super::list_x::cpu_writes_after(this, chunk);
    });
    // the next uploads into textures kept in step start from the states these lists leave
    super::title_textures::submitted(&mapped);
    super::present::note_execute(this, &mapped);
    stop_if_device_removed();
}

/// Wait (up to 10 s) until the GPU has finished the work submitted to the queues console code
/// used (before the process ends on purpose).
pub fn drain_gpu() {
    use crate::gpu::fence::{self, FenceError};

    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    unsafe {
        // a fence per queue (they finish in any order)
        let mut fences = Vec::new();
        for q in super::present::known_queues() {
            let qr = q as *mut c_void;
            let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&qr) else { continue };
            let submit = || {
                let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .map_err(|e| fence::api_error(dev, "CreateFence", e))?;
                q.Signal(&f, 1).map_err(|e| fence::api_error(dev, "Signal", e))?;
                Ok(f)
            };
            match submit() {
                Ok(f) => fences.push(f),
                Err(e) => {
                    crate::xlog_error!("D3D12 queue drain: {e}");
                    if matches!(e, FenceError::Removed(_)) {
                        return;
                    }
                }
            }
        }
        let start = std::time::Instant::now();
        loop {
            let mut pending = false;
            for f in &fences {
                match fence::completed_value(f, dev) {
                    Ok(value) => pending |= value < 1,
                    Err(e) => {
                        crate::xlog_error!("D3D12 queue drain: {e}");
                        return;
                    }
                }
            }
            if !pending {
                return;
            }
            if start.elapsed().as_secs() >= 10 {
                crate::xlog_warn!("D3D12 queue drain: GPU fences still pending after 10 s");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

/// DRED gives us a linked list in driver order. Keep the diagnostic bounded even on a
/// long chain, with room in the log for work that the GPU had already started.
unsafe fn report_dred_lists(head: *const D3D12_AUTO_BREADCRUMB_NODE, mut report: impl FnMut(&D3D12_AUTO_BREADCRUMB_NODE, usize)) {
    const SCAN_LIMIT: usize = 4096;
    const REPORT_LIMIT: usize = 64;
    let nodes = || std::iter::successors(head.as_ref(), |node| node.pNext.as_ref()).take(SCAN_LIMIT);
    let mut reported = 0;
    for started in [true, false] {
        for node in nodes() {
            let done = node.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
            if done < node.BreadcrumbCount && (done != 0) == started {
                report(node, done as usize);
                reported += 1;
                if reported == REPORT_LIMIT { return; }
            }
        }
    }
}

#[cfg(test)]
mod dred_tests {
    use super::*;

    #[test]
    fn a_late_partially_executed_list_is_reported_before_unstarted_lists() {
        let progress = [0u32, 1, 2];
        let mut nodes = vec![D3D12_AUTO_BREADCRUMB_NODE::default(); 606];
        for (i, node) in nodes.iter_mut().enumerate() {
            node.BreadcrumbCount = 2;
            node.pLastBreadcrumbValue = if i < 193 { &progress[0] } else { &progress[2] };
        }
        nodes[605].pLastBreadcrumbValue = &progress[1];
        for i in 1..nodes.len() {
            nodes[i - 1].pNext = &nodes[i];
        }
        let mut reported = Vec::new();
        unsafe { report_dred_lists(nodes.as_ptr(), |node, done| reported.push((std::ptr::from_ref(node), done))) };
        assert_eq!(reported.first(), Some(&(std::ptr::from_ref(&nodes[605]), 1)));
        assert_eq!(reported.len(), 64);
        for (i, &(node, done)) in reported[1..].iter().enumerate() {
            assert_eq!((node, done), (std::ptr::from_ref(&nodes[i]), 0));
        }
    }
}

/// If the device was removed: the reason, DRED's page fault (address, the allocations around
/// it) and the operations each command list got to – into the log, once.
pub fn log_device_removed() {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    let Err(e) = (unsafe { dev.GetDeviceRemovedReason() }) else { return };
    if DONE.swap(true, Ordering::AcqRel) {
        return;
    }
    crate::xlog_error!("D3D12 device removed: {:#x} {}", e.code().0 as u32, e.message());
    unsafe {
        let Ok(dred) = dev.cast::<ID3D12DeviceRemovedExtendedData>() else { return };
        if let Ok(pf) = dred.GetPageFaultAllocationOutput() {
            let names = |mut n: *const D3D12_DRED_ALLOCATION_NODE| {
                let mut v = Vec::new();
                while !n.is_null() && v.len() < 8 {
                    let a = &*n;
                    let name = if a.ObjectNameA.is_null() { String::new() } else { std::ffi::CStr::from_ptr(a.ObjectNameA as *const std::ffi::c_char).to_string_lossy().into_owned() };
                    v.push(format!("type {} {}", a.AllocationType.0, name));
                    n = a.pNext;
                }
                v.join(", ")
            };
            let va = pf.PageFaultVA;
            let window = uma_window_vas().iter().position(|&w| va >= w && va < w + 0x1_0000_0000).map(|k| format!(" = window {k} + {:#x}", va - uma_window_vas()[k])).unwrap_or_default();
            crate::xlog_error!("DRED page fault at GPU VA {va:#x}{window}; allocations there: [{}]; recently freed: [{}]", names(pf.pHeadExistingAllocationNode), names(pf.pHeadRecentFreedAllocationNode));
        }
        if let Ok(bc) = dred.GetAutoBreadcrumbsOutput() {
            report_dred_lists(bc.pHeadAutoBreadcrumbNode, |node, done| {
                let count = node.BreadcrumbCount as usize;
                let ops: Vec<String> = (done.saturating_sub(3)..(done + 3).min(count)).map(|i| format!("{}{}", if i == done { ">" } else { "" }, (*node.pCommandHistory.add(i)).0)).collect();
                crate::xlog_error!("DRED: list {:p} on queue {:p}: {done} of {count} operations done, around: {}", node.pCommandList.as_ref().map_or(std::ptr::null_mut(), |l| l.as_raw()), node.pCommandQueue.as_ref().map_or(std::ptr::null_mut(), |q| q.as_raw()), ops.join(" "));
            });
        }
    }
}

/// The GPU hung or the device was removed: the title is stopped at once (a title that goes on
/// submitting work to a hung device can keep the driver from recovering), with the reason in
/// the log and the report.
pub fn stop_if_device_removed() {
    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    if let Err(e) = unsafe { dev.GetDeviceRemovedReason() } {
        log_device_removed();
        let msg = format!("the D3D12 device was removed ({:#x}: {}) – the title is stopped", e.code().0 as u32, e.message());
        crate::xlog_error!("{msg}");
        crate::report::write_final(Some(msg));
        crate::log::flush();
        unsafe {
            windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), e.code().0 as u32);
        }
    }
}

#[cfg(test)]
mod rect_list_pso_tests {
    use super::*;

    const REFUSED: HRESULT = 0x8007_0057u32 as i32;

    fn with_gs(gs: &[u8]) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC { GS: D3D12_SHADER_BYTECODE { pShaderBytecode: gs.as_ptr() as _, BytecodeLength: gs.len() }, ..Default::default() }
    }

    /// A pipeline the driver refuses with the rect list GS is made again without it, once; one
    /// it accepts keeps the GS, and a refused pipeline without a rect list GS is not retried.
    #[test]
    fn a_refused_rect_list_gs_is_retried_without_it() {
        let gs = [0u8; 4];
        let mut copy = with_gs(&gs);
        let mut calls = Vec::new();
        let hr = retry_without_rect_gs(&mut copy, true, |c| {
            calls.push(c.GS.pShaderBytecode.is_null());
            if c.GS.pShaderBytecode.is_null() {
                0
            } else {
                REFUSED
            }
        });
        assert_eq!((hr, calls), (0, vec![false, true]));
        let mut copy = with_gs(&gs);
        let mut n = 0;
        assert_eq!(
            retry_without_rect_gs(&mut copy, true, |_| {
                n += 1;
                0
            }),
            0
        );
        assert!(n == 1 && !copy.GS.pShaderBytecode.is_null());
        let mut n = 0;
        assert_eq!(
            retry_without_rect_gs(&mut copy, false, |_| {
                n += 1;
                REFUSED
            }),
            REFUSED
        );
        assert!(n == 1 && !copy.GS.pShaderBytecode.is_null());
    }

    /// `has_rect_list_gs`: a pipeline made with the rect list GS, not one made again without it
    /// (or never given it), and not once its address is forgotten.
    #[test]
    fn pipelines_with_the_rect_list_gs_are_known_until_forgotten() {
        let gs = [0u8; 4];
        let (kept, dropped, none) = (0x7E57_1000usize, 0x7E57_2000usize, 0x7E57_3000usize);
        let mut copy = with_gs(&gs);
        note_rect_list_gs(kept, &copy, true);
        assert!(retry_without_rect_gs(&mut copy, true, |c| if c.GS.pShaderBytecode.is_null() { 0 } else { REFUSED }) == 0);
        note_rect_list_gs(dropped, &copy, true);
        note_rect_list_gs(none, &with_gs(&gs), false);
        assert!(has_rect_list_gs(kept) && !has_rect_list_gs(dropped) && !has_rect_list_gs(none) && !has_rect_list_gs(0));
        forget_pso(kept);
        assert!(!has_rect_list_gs(kept));
    }

    /// An untranslated rect list pipeline from a stream gets the PC's topology type (TRIANGLE),
    /// as at 956a476; other types, and a direct CreateGraphicsPipelineState, are left as they are.
    #[test]
    fn untranslated_stream_rect_lists_get_the_pcs_topology_type() {
        let at = |t: i32| D3D12_GRAPHICS_PIPELINE_STATE_DESC { PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE(t), ..Default::default() };
        let desc = |g: &D3D12_GRAPHICS_PIPELINE_STATE_DESC| g as *const _ as *const c_void;
        let (rect, tri) = (at(RECT_LIST_TYPE), at(3));
        unsafe {
            let g = untranslated_stream_desc(desc(&rect), true).expect("a stream's rect list");
            assert_eq!(g.PrimitiveTopologyType, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
            assert!(untranslated_stream_desc(desc(&tri), true).is_none());
            assert!(untranslated_stream_desc(desc(&rect), false).is_none());
        }
    }
}

#[cfg(test)]
mod fallback_stream_tests {
    use super::*;

    #[test]
    fn a_stream_counts_referenced_shader_code_not_its_descriptors() {
        let mut stream = StreamOut::default();
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &[0; 11]);
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &[0; 7]);
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
        assert_eq!(stream.shader_bytes(), 18);
        assert_ne!(stream.0.len(), stream.shader_bytes(), "the stream stores bytecode descriptors, not code");
    }

    /// A stream whose SizeInBytes ends inside its last subobject's payload (a VS subobject with
    /// 8 of its 16 bytecode bytes) is not rewritten: None, not a range panic in the device hook
    /// (which aborts the title). The whole subobject gets the stand-in VS.
    #[test]
    fn a_stream_ending_inside_a_payload_is_not_rewritten() {
        let vs = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS.0 as u64;
        let stream = [vs, 0x1234, 0x10];
        unsafe {
            let cut: [usize; 2] = [16, stream.as_ptr() as usize];
            assert!(fallback_stream(cut.as_ptr() as *const u8).is_none());
            let whole: [usize; 2] = [24, stream.as_ptr() as usize];
            let (buf, size) = fallback_stream(whole.as_ptr() as *const u8).expect("a whole VS subobject");
            assert_eq!((buf.len(), size), (24, [24, 0]));
            assert_eq!(u64::from_le_bytes(buf[16..24].try_into().unwrap()), dummy().vs.len() as u64);
        }
    }
}

#[cfg(test)]
mod this_first_tests {
    use super::*;

    /// `this_first`: either argument order, and a return buffer whose first word is a stale copy
    /// of the object's vtable pointer (7a033c8: the retail game's C-way call wrote its result
    /// over the device). Synthetic objects; no GPU and no title.
    #[test]
    fn a_return_buffer_with_a_stale_vtable_copy_is_not_the_object() {
        let pc = [0usize; 32];
        let ext = [pc.as_ptr() as usize, 0];
        let vtbl = ext.as_ptr() as usize + 8;
        EXTS.lock().unwrap().get_or_insert_with(HashSet::new).insert(vtbl);
        let plain = 0x1234_5678usize;
        let p = &plain as *const usize as *mut c_void;
        // an object off the stack (as a real device is), and a stack buffer holding a stale copy
        // of its vtable pointer - the retail game's case
        let obj = Box::into_raw(Box::new([vtbl])) as *mut c_void;
        let buf = [vtbl];
        let b = buf.as_ptr() as *mut c_void;
        unsafe {
            assert!(is_adopted(obj) && is_adopted(b), "both look adopted");
            // one adopted: that one is the object, either position
            assert!(this_first(obj, p));
            assert!(!this_first(p, obj));
            // neither adopted: unchanged (the C way is assumed)
            assert!(!this_first(p, p) && !this_first(std::ptr::null_mut(), std::ptr::null_mut()));
            // one null argument: the other is taken as `this`, either position
            assert!(this_first(obj, std::ptr::null_mut()));
            assert!(!this_first(std::ptr::null_mut(), obj));
            // both adopted, one on this thread's stack: the stack one is the buffer, so the heap
            // one is `this` in either order - what 7a033c8 fixed
            assert!(this_first(obj, b));
            assert!(!this_first(b, obj));
            // both on the stack, neither the device: `a` (the C++ way)
            let stale = [vtbl];
            let s = stale.as_ptr() as *mut c_void;
            assert!(this_first(b, s) && this_first(s, b));
            drop(Box::from_raw(obj as *mut [usize; 1]));
        }
        if let Some(g) = EXTS.lock().unwrap().as_mut() {
            g.remove(&vtbl);
        }
    }
}

#[cfg(test)]
mod cache_report_tests {
    use super::*;

    #[test]
    fn reports_without_dxil_cache_statistics_still_deserialize() {
        let mut old = serde_json::to_value(D3d12Stats::default()).unwrap();
        old.as_object_mut().unwrap().remove("dxil_cache");
        let decoded: D3d12Stats = serde_json::from_value(old).unwrap();
        assert_eq!(
            decoded.dxil_cache,
            crate::gpu::recomp::dxc::CacheStats::default()
        );
    }
}

#[cfg(test)]
mod readback_routing_tests {
    use super::*;

    #[test]
    fn custom_write_back_only_stays_coherent_for_small_copy_dest_buffers() {
        let rb = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let custom_wb = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let copy_dest = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let gen_read = D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32;
        let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let none = D3D12_RESOURCE_FLAG_NONE;

        assert!(is_coherent_readback_buffer(&rb, copy_dest, none, 4 << 20));
        assert!(!is_coherent_readback_buffer(&rb, copy_dest, none, COHERENT_READBACK_MAX + 0x1_0000));
        assert!(is_cpu_readback_buffer(&rb, copy_dest, none));

        // General CUSTOM+WRITE_BACK buffers route to sparse VRAM, not coherent Window 3.
        assert!(!is_coherent_readback_buffer(&custom_wb, gen_read, none, 4 << 20));
        assert!(!is_cpu_readback_buffer(&custom_wb, gen_read, none));
        assert!(!is_coherent_readback_buffer(&custom_wb, copy_dest, uav, 4 << 20));
        assert!(!is_cpu_readback_buffer(&custom_wb, copy_dest, uav));

        // Small non-UAV COPY_DEST CUSTOM+WRITE_BACK buffers may use coherent Window 3; larger ones spill.
        assert!(is_coherent_readback_buffer(&custom_wb, copy_dest, none, 4 << 20));
        assert!(!is_coherent_readback_buffer(&custom_wb, copy_dest, none, COHERENT_CUSTOM_WB_MAX + 0x1_0000));
        assert!(is_cpu_readback_buffer(&custom_wb, copy_dest, none));
    }

    /// Bug #4: `place_console_buffer` coherent fallback must strip console-only resource flags
    /// and state bits before creating the cross-adapter placed resource.
    #[test]
    fn coherent_buffer_sanitizes_console_flags_and_state() {
        let raw_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0x10_0000,
            Width: 0x1_0000,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT(0x101), // console layout
            Flags: D3D12_RESOURCE_FLAGS(0x0130_0000 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0),
            ..Default::default()
        };
        let console_state = 0x0010_0000usize | (D3D12_RESOURCE_STATE_COPY_DEST.0 as usize);

        let (clean, st) = sanitize_coherent_buffer(&raw_desc, console_state);
        assert_eq!(clean.Alignment, 0);
        assert_eq!(clean.Layout, D3D12_TEXTURE_LAYOUT_ROW_MAJOR);
        assert_eq!(
            clean.Flags,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER
        );
        assert_eq!(st, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);

        // Verify against a real D3D12 device: raw console flags/state are rejected, sanitized succeeds.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            if let Ok(dev3) = dev.cast::<ID3D12Device3>() {
                use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
                let mem = VirtualAlloc(std::ptr::null(), 0x1_0000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                assert!(!mem.is_null());
                if let Ok(heap) = dev3.OpenExistingHeapFromAddress::<ID3D12Heap>(mem) {
                    let mut unsanitized = raw_desc;
                    unsanitized.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
                    unsanitized.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                    unsanitized.Alignment = 0;
                    let mut r_bad: Option<ID3D12Resource> = None;
                    assert!(
                        dev.CreatePlacedResource(&heap, 0, &unsanitized, D3D12_RESOURCE_STATES(console_state as i32), None, &mut r_bad).is_err(),
                        "PC D3D12 must reject unsanitized console flags/state"
                    );
                    let mut r_ok: Option<ID3D12Resource> = None;
                    dev.CreatePlacedResource(&heap, 0, &clean, D3D12_RESOURCE_STATES(st as i32), None, &mut r_ok)
                        .expect("sanitized coherent buffer must succeed on PC D3D12");
                    assert!(r_ok.is_some());
                    drop(r_ok);
                    drop(heap);
                }
                assert_ne!(VirtualFree(mem, 0, MEM_RELEASE), 0);
            }
        }
    }

    /// Bug #5: `CopyDescriptors` and `CopyDescriptorsSimple` must remap `XBOX_EXTRA_CBV_SRV_UAV`
    /// (`4`) to `D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV` (`0`) on both console and PC paths.
    #[test]
    fn copy_descriptors_remaps_xbox_extra_cbv_srv_uav_to_pc_enum() {
        use std::cell::RefCell;

        assert_eq!(pc_heap_type(XBOX_EXTRA_CBV_SRV_UAV), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        assert_eq!(pc_heap_type(0), 0);
        assert_eq!(pc_heap_type(1), 1);
        assert_eq!(pc_heap_type(2), 2);
        assert_eq!(pc_heap_type(3), 3);

        thread_local! {
            static SIMPLE_TYPES: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
            static RANGE_TYPES: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "system" fn fake_copy_simple(_this: *mut c_void, _n: u32, _dst: u64, _src: u64, ty: u32) {
            SIMPLE_TYPES.with(|v| v.borrow_mut().push(ty));
        }
        #[allow(clippy::too_many_arguments)]
        unsafe extern "system" fn fake_copy_ranges(
            _this: *mut c_void,
            _nd: u32,
            _dsts: *const u64,
            _dsizes: *const u32,
            _ns: u32,
            _srcs: *const u64,
            _ssizes: *const u32,
            ty: u32,
        ) {
            RANGE_TYPES.with(|v| v.borrow_mut().push(ty));
        }

        #[repr(C)]
        struct FakeDev {
            vt: *const usize,
        }

        let mut orig_vt = [0usize; 32];
        orig_vt[23] = fake_copy_ranges as usize;
        orig_vt[24] = fake_copy_simple as usize;
        let ext_vt = [orig_vt.as_ptr() as usize, 0usize];
        let mut dev = FakeDev {
            // SAFETY: `ext_vt` has 2 entries; `.add(1)` puts `orig_vt` at `ext[-1]`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
        };
        let dev_ptr = &mut dev as *mut FakeDev as *mut c_void;

        SIMPLE_TYPES.with(|v| v.borrow_mut().clear());
        RANGE_TYPES.with(|v| v.borrow_mut().clear());

        unsafe {
            // Non-console PC fallback paths
            dev_copy_descriptors_simple_x(dev_ptr, 1, 0x1000, 0x2000, XBOX_EXTRA_CBV_SRV_UAV);
            dev_copy_descriptors_simple_x(dev_ptr, 1, 0x1000, 0x2000, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32);
            let (d, s, sz) = ([0x1000u64], [0x2000u64], [1u32]);
            dev_copy_descriptors_x(dev_ptr, 1, d.as_ptr(), sz.as_ptr(), 1, s.as_ptr(), sz.as_ptr(), XBOX_EXTRA_CBV_SRV_UAV);
            dev_copy_descriptors_x(dev_ptr, 1, d.as_ptr(), sz.as_ptr(), 1, s.as_ptr(), sz.as_ptr(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV.0 as u32);

            // Console-slot paths (`copy_console_slots` invoked from both Simple and Ranges)
            let heap_obj = 0x5555_DEADusize;
            if crate::gpu::uma::get().is_some()
                && super::super::console_heap::register(heap_obj, 4, 0x10_0000, 0, 32, false, false, false)
            {
                let start = super::super::console_heap::start(heap_obj).unwrap();
                let dst_h = start + super::super::console_heap::SLOT as u64;
                dev_copy_descriptors_simple_x(dev_ptr, 1, dst_h, start, XBOX_EXTRA_CBV_SRV_UAV);
                let (cd, cs, csz) = ([dst_h], [start], [1u32]);
                dev_copy_descriptors_x(dev_ptr, 1, cd.as_ptr(), csz.as_ptr(), 1, cs.as_ptr(), csz.as_ptr(), XBOX_EXTRA_CBV_SRV_UAV);
                super::super::console_heap::unregister(heap_obj);

                SIMPLE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 1, 0, 0]));
            } else {
                SIMPLE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 1]));
            }
            RANGE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 2]));
        }
    }

    /// Bug #6: `Kind::DmaList` must install `cl_release` so releasing a DMA command list removes
    /// its entry from `list_x::LISTS`.
    #[test]
    fn dma_list_release_cleans_up_list_state() {
        assert!(
            overrides(Kind::DmaList).iter().any(|&(name, f)| name == "Release" && f == cl_release as usize),
            "Kind::DmaList must override Release with cl_release"
        );

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).expect("copy alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &alloc, None).expect("copy list");
            let raw = list.into_raw();
            let key = raw as usize;

            // Mirror `device_x::create_list`: adopt as CommandList, track via `list_x::on_create`,
            // then re-adopt as `Kind::DmaList`.
            adopt(raw, Kind::CommandList);
            super::super::list_x::on_create(raw, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COPY);
            readopt(raw, Kind::DmaList);
            assert!(super::super::list_x::has_list_state(key), "DMA list must be tracked in list_x::LISTS");

            // Invoke Release through the adopted vtable slot 2.
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt(raw).add(2));
            assert_eq!(release(raw), 0);
            assert!(!super::super::list_x::has_list_state(key), "releasing DMA list must remove its list_x::LISTS entry");
        }
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;

    // a fake adopted queue: the extension vtable's word before slot 0 points at the PC table
    unsafe extern "system" fn get_desc(_: *mut c_void, out: usize) -> usize {
        let d = [1u32, 2, 3, 4];
        std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, out as *mut u8, 16);
        out
    }
    unsafe extern "system" fn signal(_: *mut c_void, _: usize, _: usize) -> i32 {
        0
    }

    /// Queue #18 before the layout is known writes 8 bytes; after a Signal at #14 (the older
    /// layout) it is GetDesc with all 16.
    #[test]
    fn queue_slot_18_before_the_layout_is_known_writes_8_bytes() {
        let mut pc = [0usize; 32];
        pc[14] = signal as usize;
        pc[18] = get_desc as usize;
        let ext = [pc.as_ptr() as usize, 0];
        let vtbl = ext.as_ptr() as usize + 8;
        let mut obj = [vtbl];
        let this = obj.as_mut_ptr() as *mut c_void;
        let fence = 0x5EED_0000usize;
        QUEUE_SHIFTED.store(false, Ordering::Release);
        QUEUE_OLDER.store(false, Ordering::Release);
        unsafe {
            let mut out = [0xAAAA_AAAAu32; 4];
            let r = queue_dispatch(18, this, out.as_mut_ptr() as usize, 0, 0, 0);
            assert_eq!(r, out.as_ptr() as usize);
            assert_eq!(out, [1, 2, 0xAAAA_AAAA, 0xAAAA_AAAA], "only the first 8 bytes");
            FENCES.write().unwrap().push(fence);
            queue_dispatch(14, this, fence, 1, 0, 0);
            FENCES.write().unwrap().retain(|&f| f != fence);
            assert!(QUEUE_OLDER.load(Ordering::Acquire) && !QUEUE_SHIFTED.load(Ordering::Acquire));
            let mut out = [0xAAAA_AAAAu32; 4];
            queue_dispatch(18, this, out.as_mut_ptr() as usize, 0, 0, 0);
            assert_eq!(out, [1, 2, 3, 4], "older layout: the whole description");
        }
        QUEUE_OLDER.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod srv_tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC1_UNORM_SRGB;

    #[test]
    fn a_cube_view_becomes_the_cube_array_of_one_cube() {
        let cube = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_BC1_UNORM_SRGB,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: 0x1688,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV { MostDetailedMip: 2, MipLevels: 3, ResourceMinLODClamp: 1.5 },
            },
        };
        let v = cube_as_cube_array(&cube).expect("a cube view is rewritten");
        assert_eq!(v.ViewDimension, D3D12_SRV_DIMENSION_TEXTURECUBEARRAY);
        assert_eq!((v.Format, v.Shader4ComponentMapping), (DXGI_FORMAT_BC1_UNORM_SRGB, 0x1688), "format and mapping kept");
        let a = unsafe { v.Anonymous.TextureCubeArray };
        assert_eq!((a.MostDetailedMip, a.MipLevels, a.First2DArrayFace, a.NumCubes), (2, 3, 0, 1));
        assert_eq!(a.ResourceMinLODClamp, 1.5);

        // a cube array and every other view are forwarded as the title made them
        let mut array = cube;
        array.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        array.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV { NumCubes: 2, ..Default::default() };
        assert!(cube_as_cube_array(&array).is_none());
        let mut d2 = cube;
        d2.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        assert!(cube_as_cube_array(&d2).is_none());
    }

    #[test]
    fn a_default_view_of_a_depth_buffer_matches_its_sample_count_and_array_size() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_R24G8_TYPELESS, DXGI_SAMPLE_DESC};
        for (samples, slices, dimension, dim) in [
            (1, 1, D3D12_SRV_DIMENSION_TEXTURE2D, 4),
            (1, 3, D3D12_SRV_DIMENSION_TEXTURE2DARRAY, 5),
            (4, 1, D3D12_SRV_DIMENSION_TEXTURE2DMS, 6),
            (4, 3, D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY, 7),
        ] {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                DepthOrArraySize: slices,
                MipLevels: 1,
                Format: DXGI_FORMAT_R24G8_TYPELESS,
                SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 },
                ..Default::default()
            };
            let v = pc_srv(&desc, None).expect("a typeless depth buffer gets a view of its own");
            assert_eq!(v.Format, DXGI_FORMAT_R24_UNORM_X8_TYPELESS);
            assert_eq!(v.ViewDimension, dimension, "{samples} samples, {slices} slices");
            let a = &v.Anonymous;
            match dimension {
                D3D12_SRV_DIMENSION_TEXTURE2D => assert_eq!(unsafe { a.Texture2D.MipLevels }, u32::MAX),
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                    assert_eq!(unsafe { (a.Texture2DArray.FirstArraySlice, a.Texture2DArray.ArraySize, a.Texture2DArray.MipLevels) }, (0, u32::from(slices), u32::MAX))
                }
                D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => {
                    assert_eq!(unsafe { (a.Texture2DMSArray.FirstArraySlice, a.Texture2DMSArray.ArraySize) }, (0, u32::from(slices)))
                }
                _ => {}
            }
            // the console view (T#) of the resource without a description
            assert_eq!(resource_dim_to_view(&desc), dim, "{samples} samples, {slices} slices");
            assert_eq!(v.ViewDimension.0, dim);
        }
    }
}

#[cfg(test)]
mod console_format_query_tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    #[test]
    fn console_formats_are_asked_as_the_pcs() {
        for (console, pc) in [(116u32, 10u32), (117, 10), (118, 45), (119, 46), (120, 47)] {
            let mut d = [0u32; 3];
            d[0] = console;
            let back = unsafe { console_format_query(D3D12_FEATURE_FORMAT_SUPPORT.0, d.as_mut_ptr() as *mut u8, 12) };
            assert_eq!((back, d[0]), (Some(console), pc));
        }
        // a PC format, or a feature without one: untouched
        let mut d = [45u32, 0, 0];
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_FORMAT_SUPPORT.0, d.as_mut_ptr() as *mut u8, 12) }, None);
        let mut d = [118u32; 16];
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_D3D12_OPTIONS.0, d.as_mut_ptr() as *mut u8, 64) }, None);
        assert_eq!(d[0], 118);
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_FORMAT_INFO.0, std::ptr::null_mut(), 8) }, None);
    }

    /// A hardware device (None: no D3D12 GPU here – skipped).
    fn device() -> Option<ID3D12Device> {
        let mut dev: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev) }.ok()?;
        dev
    }

    /// CheckFeatureSupport of the PC device as `dev_check_feature_support` reaches it (vtable #13).
    fn pc_check(dev: &ID3D12Device) -> CheckFeatureSupportFn {
        unsafe { std::mem::transmute(*(*(dev.as_raw() as *const *const usize)).add(13)) }
    }

    /// GPU: the queries a title makes of D16_UNORM_S8_UINT (118) and its family succeed with the
    /// PC format's answers, and the caller's format field is the console's again. Asked as is, the
    /// PC has no such format.
    #[test]
    fn gpu_console_depth_formats_are_supported() {
        let Some(dev) = device() else { return };
        let (this, f) = (dev.as_raw(), pc_check(&dev));
        // the gap: the PC's own answer (unless another test adopted a device, and the vtable it
        // shares with this one already holds `dev_check_feature_support`)
        if f as usize != dev_check_feature_support as usize {
            let mut raw = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(118), ..Default::default() };
            let hr = unsafe { f(this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut raw as *mut _ as *mut u8, size_of_val(&raw) as u32) };
            assert!(hr < 0 || raw.Support1.0 == 0, "the PC knows format 118 ({hr:#x}, {:#x})", raw.Support1.0);
        }

        for (format, need) in [
            (118, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL),
            (119, D3D12_FORMAT_SUPPORT1_SHADER_LOAD),
            (120, D3D12_FORMAT_SUPPORT1_SHADER_LOAD),
            (116, D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
            (117, D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
        ] {
            let mut s = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(format), ..Default::default() };
            let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut s as *mut _ as *mut u8, size_of_val(&s) as u32) };
            assert!(hr >= 0, "FORMAT_SUPPORT {format}: {hr:#x}");
            assert_eq!(s.Format.0, format as i32, "the caller's format");
            assert!(s.Support1.0 & need.0 != 0, "FORMAT_SUPPORT {format}: {:#x}", s.Support1.0);
        }

        let mut ms = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: DXGI_FORMAT(118), SampleCount: 4, ..Default::default() };
        let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS.0, &mut ms as *mut _ as *mut u8, size_of_val(&ms) as u32) };
        assert!(hr >= 0 && ms.NumQualityLevels >= 1, "MULTISAMPLE_QUALITY_LEVELS 118 ×4: {hr:#x} {}", ms.NumQualityLevels);
        assert_eq!((ms.Format.0, ms.SampleCount), (118, 4));

        let mut info = D3D12_FEATURE_DATA_FORMAT_INFO { Format: DXGI_FORMAT(118), PlaneCount: 0 };
        let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_INFO.0, &mut info as *mut _ as *mut u8, size_of_val(&info) as u32) };
        assert!(hr >= 0, "FORMAT_INFO 118: {hr:#x}");
        assert_eq!((info.Format.0, info.PlaneCount), (118, 2));

        // a PC format is asked as it is
        let mut s = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ..Default::default() };
        assert!(unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut s as *mut _ as *mut u8, size_of_val(&s) as u32) } >= 0);
        assert_eq!(s.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
    }

    /// GPU: allocation info of the console's 116 – 120 textures is a real size – the console's
    /// (XG), and the PC's for the resource made of the desc (`pc_resource_desc`), both sane.
    #[test]
    fn gpu_console_format_allocation_info_is_sane() {
        let Some(dev) = device() else { return };
        // (bytes a pixel at least, as XG lays them out: 119 alone is a plain 2-byte format, 120 alone
        // the stencil plane)
        for (format, flags, bytes) in [(118u32, 2u32, 3u64), (118, 0, 3), (119, 0, 2), (120, 0, 1), (116, 1, 4), (117, 1, 4)] {
            let mut x = [0u8; 0x38];
            x[0..4].copy_from_slice(&3u32.to_le_bytes());
            x[0x10..0x18].copy_from_slice(&1920u64.to_le_bytes());
            x[0x18..0x1C].copy_from_slice(&1080u32.to_le_bytes());
            x[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            x[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            x[0x20..0x24].copy_from_slice(&format.to_le_bytes());
            x[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            x[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
            let t = crate::hle::xg::Texture::new(&x).unwrap_or_else(|hr| panic!("XG {format}: {hr:#x}"));
            let (size, align) = (t.size(), t.alignment());
            assert!(size >= 1920 * 1080 * bytes && size < 1 << 30 && align.is_power_of_two(), "XG {format}: {size:#x} {align:#x}");
            let d: D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&x).as_ptr() as *const _) };
            let pc = unsafe { dev.GetResourceAllocationInfo(0, &[d]) };
            assert!(pc.SizeInBytes != u64::MAX && pc.SizeInBytes >= 1920 * 1080 * 4 && pc.Alignment > 0, "PC {format} as {}: {:#x}", d.Format.0, pc.SizeInBytes);
        }
    }

    /// GPU: CreatePlacedResource (#29) of a D16_UNORM_S8_UINT (118) depth target with an
    /// optimized clear value of that format: the desc becomes the PC's (`pc_resource_desc`), and
    /// the clear value with it (`pc_clear`) – one still in 118 the PC refuses.
    #[test]
    fn gpu_placed_console_depth_target_with_clear_value() {
        let Some(dev) = device() else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        unsafe {
            adopt(dev.as_raw(), Kind::Device);
            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: 16 << 20,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            dev.CreateHeap(&heap_desc, &mut heap).expect("RT/DS heap");
            let heap = heap.unwrap();
            let mut x = [0u8; 0x38];
            x[0..4].copy_from_slice(&3u32.to_le_bytes());
            x[0x10..0x18].copy_from_slice(&256u64.to_le_bytes());
            x[0x18..0x1C].copy_from_slice(&256u32.to_le_bytes());
            x[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            x[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            x[0x20..0x24].copy_from_slice(&118u32.to_le_bytes());
            x[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            x[0x30..0x34].copy_from_slice(&(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 as u32).to_le_bytes());
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT(118), Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const GUID;
            let state = D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as usize;
            let mut p = std::ptr::null_mut();
            let hr = dev_create_placed_resource_x(dev.as_raw(), heap.as_raw() as usize, 0, x.as_ptr() as usize, state, &clear as *const _ as usize, riid, &mut p);
            assert!(hr >= 0 && !p.is_null(), "D16S8 target with a 118 clear value: {hr:#x}");
            let r = ID3D12Resource::from_raw(p);
            assert_eq!(r.GetDesc().Format.0, 44, "made R24G8_TYPELESS");
        }
    }
}

#[cfg(test)]
mod map_tests {
    use super::*;

    /// A title's Map reaches the PC's only for a resource the PC can map: a buffer in an upload,
    /// readback or CPU-visible custom heap. One in a DEFAULT heap, a custom heap without CPU pages
    /// or a reserved one gets its title address instead (the PC's Map on it is an invalid call).
    #[test]
    fn only_resources_with_cpu_pages_go_to_the_pc_map() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return; // no D3D12 adapter
            }
            let dev = dev.unwrap();
            let buffer = |props: D3D12_HEAP_PROPERTIES, state: D3D12_RESOURCE_STATES| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 0x1_0000,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).expect("buffer");
                r.unwrap()
            };
            let heap = |t| D3D12_HEAP_PROPERTIES { Type: t, ..Default::default() };
            let mappable = |r: &ID3D12Resource| pc_can_map(r.as_raw());

            assert!(mappable(&buffer(heap(D3D12_HEAP_TYPE_UPLOAD), D3D12_RESOURCE_STATE_GENERIC_READ)));
            assert!(mappable(&buffer(heap(D3D12_HEAP_TYPE_READBACK), D3D12_RESOURCE_STATE_COPY_DEST)));
            let default = buffer(heap(D3D12_HEAP_TYPE_DEFAULT), D3D12_RESOURCE_STATE_COMMON);
            assert!(!mappable(&default));
            // (what the PC answers there – the call this keeps from being made)
            let mut p = std::ptr::null_mut();
            assert!(default.Map(0, None, Some(&mut p)).is_err(), "a DEFAULT-heap buffer is not mappable on the PC");

            let custom = |page| D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: page, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            assert!(mappable(&buffer(custom(D3D12_CPU_PAGE_PROPERTY_WRITE_BACK), D3D12_RESOURCE_STATE_COMMON)));
            assert!(!mappable(&buffer(custom(D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE), D3D12_RESOURCE_STATE_COMMON)));

            // a reserved resource has no heap
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 256,
                Height: 256,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                ..Default::default()
            };
            let mut reserved: Option<ID3D12Resource> = None;
            if dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).is_ok() {
                let r = reserved.unwrap();
                let mut raw = r.as_raw();
                // what dev_create_reserved_resource does with what the PC made
                note_reserved(S_OK, &mut raw);
                assert!(is_reserved(raw) && !pc_can_map(raw), "a reserved resource is not asked for its heap");
                // a buffer at an address a reserved resource left without its last Release seen
                // is not taken for it (its description differs)
                let upload = buffer(heap(D3D12_HEAP_TYPE_UPLOAD), D3D12_RESOURCE_STATE_GENERIC_READ);
                let stale = desc_of(r.as_raw()).unwrap();
                RESERVED.lock().unwrap().as_mut().unwrap().insert(upload.as_raw() as usize, stale);
                assert!(!is_reserved(upload.as_raw()) && mappable(&upload));
                let mut g = RESERVED.lock().unwrap();
                let m = g.as_mut().unwrap();
                m.remove(&(raw as usize));
                m.remove(&(upload.as_raw() as usize));
            }
        }
    }

    #[test]
    fn stream_pipeline_defaults_depth_enable_from_dsv_format() {
        let console_vs = crate::gpu::xbox::tests::sample(
            1,
            &[crate::gpu::xbox::tests::fssm(&crate::gpu::xbox::tests::CODE, &[])],
        );
        let push_sub = |buf: &mut Vec<u8>, ty: u32, payload: &[u8], align: usize| {
            buf.extend_from_slice(&ty.to_le_bytes());
            if align == 8 {
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
            buf.extend_from_slice(payload);
            while buf.len() % 8 != 0 {
                buf.push(0);
            }
        };
        let parse = |dsv_format: Option<u32>, explicit_depth: Option<bool>| -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            let mut stream = Vec::new();
            let mut vs_bc = [0u8; 16];
            vs_bc[0..8].copy_from_slice(&(console_vs.as_ptr() as usize).to_le_bytes());
            vs_bc[8..16].copy_from_slice(&console_vs.len().to_le_bytes());
            push_sub(&mut stream, 1, &vs_bc, 8);
            if let Some(fmt) = dsv_format {
                push_sub(&mut stream, 16, &fmt.to_le_bytes(), 4);
            }
            if let Some(en) = explicit_depth {
                let ds = D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: en.into(),
                    ..Default::default()
                };
                let bytes = unsafe {
                    std::slice::from_raw_parts(&ds as *const _ as *const u8, std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC>())
                };
                push_sub(&mut stream, 11, bytes, 4);
            }
            #[repr(C)]
            struct StreamDesc {
                size: usize,
                ptr: usize,
            }
            let desc = StreamDesc { size: stream.len(), ptr: stream.as_ptr() as usize };
            match unsafe { stream_pipeline(&desc as *const _ as *const c_void) } {
                StreamPipeline::Graphics(g) => *g,
                _ => panic!("expected graphics stream pipeline"),
            }
        };

        // Omitted DEPTH_STENCIL + omitted/UNKNOWN DSVFormat -> DepthEnable defaults to false.
        assert!(!parse(None, None).DepthStencilState.DepthEnable.as_bool());
        assert!(!parse(Some(0), None).DepthStencilState.DepthEnable.as_bool());
        // Omitted DEPTH_STENCIL + non-UNKNOWN DSVFormat (D32_FLOAT = 40) -> DepthEnable defaults to true.
        assert!(parse(Some(40), None).DepthStencilState.DepthEnable.as_bool());
        // Explicit DEPTH_STENCIL subobject is always preserved regardless of DSVFormat.
        assert!(parse(None, Some(true)).DepthStencilState.DepthEnable.as_bool());
        assert!(!parse(Some(40), Some(false)).DepthStencilState.DepthEnable.as_bool());
    }
}
