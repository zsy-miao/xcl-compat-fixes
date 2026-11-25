//! kernelx.dll – the console flavour of kernel32/kernelbase.
//!
//! Only functions that differ from Windows are implemented here; everything else falls
//! back to the real KERNELBASE/KERNEL32 export of the same name (see resolver).
//! Signatures and constants: public GDK docs + WinDurango (LGPL-3.0) as reference only (no code taken).

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use windows_sys::Win32::Foundation as fnd;
use windows_sys::Win32::System::LibraryLoader as ll;
use windows_sys::Win32::System::Memory as mem;
use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
use windows_sys::Win32::System::Threading as thr;

use crate::modules;
use crate::pe::ImportName;
use crate::resolver;
use crate::util::{dll_key, from_ansi_ptr, from_wide_ptr, write_ansi_out, write_wide_out};

type HANDLE = *mut c_void;
type BOOL = i32;

/// The title's own unhandled-exception filter goes behind XCL's (`eh`): XCL's crash report
/// first, then the title's handler (its crash reporter) as on the console.
pub unsafe extern "system" fn SetUnhandledExceptionFilter(f: usize) -> usize {
    crate::eh::set_title_filter(f)
}

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![
        // registry: the console mode the GDK thunks read (XSystemGetDeviceType)
        RegOpenKeyW,
        RegOpenKeyExW,
        RegQueryValueExW,
        RegCloseKey,
        // memory
        VirtualAlloc,
        VirtualAllocEx,
        VirtualFree,
        VirtualFreeEx,
        VirtualProtect,
        VirtualProtectEx,
        VirtualQuery,
        VirtualQueryEx,
        XMemAlloc,
        XMemFree,
        XMemAllocDefault,
        XMemFreeDefault,
        XMemSetAllocationHooks,
        XMemVirtualQuery,
        XMemVirtualAlloc,
        XMemAllocatePhysicalPages,
        XMemMapPhysicalPages,
        XMemMapPhysicalPagesScatter,
        XMemFreePhysicalPages,
        XMemTransferMemory,
        XMemGetWorkingSetStatistics,
        XMemIsToolsMemAvailable,
        TitleMemoryStatus,
        JobTitleMemoryStatus,
        ToolingMemoryStatus,
        AllocateTitlePhysicalPages,
        FreeTitlePhysicalPages,
        MapTitlePhysicalPages,
        // app state (minappmodelx): a title is never suspended or constrained here
        RegisterAppStateChangeNotification,
        UnregisterAppStateChangeNotification,
        RegisterAppConstrainedChangeNotification,
        UnregisterAppConstrainedChangeNotification,
        // threads / system
        SetUnhandledExceptionFilter,
        Sleep,
        SetThreadName,
        GetThreadName,
        SetThreadpoolAffinityMask,
        SetThreadAffinityMask,
        SetThreadGroupAffinity,
        QueryProcessorSchedulingStatistics,
        GetConsoleType,
        OutputDebugStringA,
        OutputDebugStringW,
        GetSystemOSVersion,
        // modules
        LoadLibraryW,
        LoadLibraryA,
        LoadLibraryExW,
        LoadLibraryExA,
        LoadPackagedLibrary,
        GetProcAddress,
        GetModuleHandleW,
        GetModuleHandleA,
        GetModuleHandleExW,
        GetModuleHandleExA,
        GetModuleFileNameW,
        GetModuleFileNameA,
        FreeLibrary,
        FreeLibraryWhenCallbackReturns,
        DisableThreadLibraryCalls,
    ]
}

// ---------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------

pub const MEM_GRAPHICS: u32 = 0x1000_0000;
mod placeholders;
pub mod lazy;
pub const MEM_TITLE: u32 = 0x4000_0000;
const STD_ALLOC_MASK: u32 = 0x1000 | 0x2000 | 0x8_0000 | 0x10_0000 | 0x20_0000 | 0x100_0000;
const STD_PROTECT_MASK: u32 = 0xFF | 0x100 | 0x200 | 0x400;

static WARNED_FLAGS: AtomicU32 = AtomicU32::new(0);

pub fn sanitize_alloc_type(t: u32) -> u32 {
    let s = t & STD_ALLOC_MASK;
    let dropped = t & !STD_ALLOC_MASK;
    if dropped != 0 && WARNED_FLAGS.fetch_or(dropped, Ordering::Relaxed) & dropped != dropped {
        crate::xlog_debug!("VirtualAlloc: dropping console allocation flags {dropped:#x}");
    }
    if s == 0 {
        0x1000 | 0x2000
    } else {
        s
    }
}

pub fn sanitize_protect(p: u32) -> u32 {
    let s = p & STD_PROTECT_MASK;
    if s & 0xFF == 0 {
        (s & !0xFF) | mem::PAGE_READWRITE
    } else {
        s
    }
}

/// Graphics address space the title reserved (no commit) that unified memory had no room for
/// – where it maps physical pages (a texture streaming pool): base → size. Write-watched, like
/// unified memory, so textures in it follow CPU writes (`d3d12x::title_textures`).
static GRAPHICS_RESERVATIONS: std::sync::Mutex<std::collections::BTreeMap<usize, usize>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
static GRAPHICS_RESERVATIONS_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Generation counter incremented whenever `GRAPHICS_RESERVATIONS` changes.
pub fn graphics_reservations_gen() -> u64 {
    GRAPHICS_RESERVATIONS_GEN.load(Ordering::Acquire)
}

/// The title's graphics reservations (base, size).
pub fn graphics_reservations() -> Vec<(usize, usize)> {
    GRAPHICS_RESERVATIONS.lock().unwrap().iter().map(|(&b, &s)| (b, s)).collect()
}

/// Whether `p` lies in one of the title's graphics reservations.
pub fn in_graphics_reservation(p: usize) -> bool {
    GRAPHICS_RESERVATIONS.lock().unwrap().range(..=p).next_back().is_some_and(|(&b, &s)| p < b + s)
}

const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// GPU views of the title's CPU memory. A retail game reserves 24 GB of 64 KiB pages twice
/// in a row: once readable by the CPU (PAGE_READWRITE), once with PAGE_NOACCESS |
/// PAGE_GRAPHICS_NOACCESS. It maps physical pages only into the first, yet places GPU
/// resources at the same offset in the second (its Bink video planes: the decoder writes the
/// first view, the video shader samples textures in the second). On the console the second
/// reservation is a GPU-only view of the first; here it is separate, empty memory – the
/// video was its colour of zero luma and chroma, a green screen. `source` names the CPU view
/// behind an address of such a GPU view, which `d3d12x::title_textures` reads and watches.
pub mod mirrors {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    /// MEM_LARGE_PAGES | MEM_PHYSICAL: the console's 64 KiB pages
    const PAGES_64K: u32 = 0x2040_0000;
    /// PAGE_GRAPHICS_NOACCESS
    const GRAPHICS_NOACCESS: u32 = 0x800;

    struct State {
        /// CPU-readable reservations of 64 KiB pages, oldest first: (base, size)
        cpu: Vec<(usize, usize)>,
        /// GPU view base → (size, CPU view base)
        gpu: BTreeMap<usize, (usize, usize)>,
    }
    static STATE: Mutex<State> = Mutex::new(State { cpu: Vec::new(), gpu: BTreeMap::new() });

    /// Whether `(t, p)` is a CPU-inaccessible GPU-view alias reservation (`MEM_64K_PAGES`
    /// reserve with `PAGE_GRAPHICS_NOACCESS | PAGE_NOACCESS`).
    pub fn is_gpu_view_reservation(t: u32, p: u32) -> bool {
        t & PAGES_64K == PAGES_64K && t & 0x1000 == 0 && p & GRAPHICS_NOACCESS != 0 && p & 0xFF == super::mem::PAGE_NOACCESS
    }

    /// A reservation of `size` bytes at `base` with allocation type `t` and protection `p`.
    pub fn reserved(base: usize, size: usize, t: u32, p: u32) {
        if t & PAGES_64K != PAGES_64K || t & 0x1000 != 0 {
            return;
        }
        let mut s = STATE.lock().unwrap();
        if p & GRAPHICS_NOACCESS != 0 && p & 0xFF == super::mem::PAGE_NOACCESS {
            // the most recent CPU view of the same size without a GPU view yet
            let taken: Vec<usize> = s.gpu.values().map(|&(_, c)| c).collect();
            if let Some(&(cpu, _)) = s.cpu.iter().rev().find(|&&(b, n)| n == size && !taken.contains(&b)) {
                s.gpu.insert(base, (size, cpu));
                crate::xlog_info!("kernelx: reservation {base:#x} ({size:#x} bytes, no CPU access) is the GPU view of {cpu:#x}");
            }
        } else if p & 0xFF != super::mem::PAGE_NOACCESS {
            s.cpu.push((base, size));
        }
    }

    /// The reservation at `base` was released.
    pub fn released(base: usize) {
        let mut s = STATE.lock().unwrap();
        s.cpu.retain(|&(b, _)| b != base);
        s.gpu.remove(&base);
        s.gpu.retain(|_, &mut (_, c)| c != base);
    }

    pub fn source(va: usize) -> Option<usize> {
        let s = STATE.lock().unwrap();
        let (&gpu, &(size, cpu)) = s.gpu.range(..=va).next_back()?;
        let offset = va.checked_sub(gpu)?;
        (offset < size).then(|| cpu.checked_add(offset)).flatten()
    }
}

/// Graphics memory (MEM_GRAPHICS, committed, no fixed address) comes from the unified
/// memory region the GPU uses directly (`gpu::uma`); a graphics reservation (no commit) is
/// write-watched address space of Windows; everything else from Windows.
/// The console's MEM_64K_PAGES (MEM_LARGE_PAGES | MEM_PHYSICAL): title memory of 64 KiB
/// pages – GPU-visible on the console like all title memory.
const MEM_64K_PAGES: u32 = 0x2040_0000;
/// MEM_LARGE_PAGES: the console's 2 MiB pages of title memory (alone, or in MEM_64K_PAGES)
const MEM_LARGE_PAGES_X: u32 = 0x2000_0000;
/// a protection bit of the console's title memory the GPU uses (GTA VI's index and vertex data:
/// 0x22004; its 2.2 GB block of other data: 0x2404)
const PAGE_GPU_X: u32 = 0x2_0000;
/// PAGE_GRAPHICS_NOACCESS through PAGE_GRAPHICS_NOCACHE, as upstream PR #252.
const PAGE_GRAPHICS_ANY: u32 = 0x7_F800;

/// Commit `[a, a + n)` of a sparse window (the title's reservations there) and give it video
/// memory (`d3d12x::sparse`).
unsafe fn commit_sparse(a: usize, n: usize, p: u32) -> *mut c_void {
    let q = mem::VirtualAlloc(a as *const c_void, n, mem::MEM_COMMIT, sanitize_protect(p));
    if !q.is_null() {
        crate::hle::d3d12x::sparse::committed(q as usize, n);
    }
    q
}

pub unsafe extern "system" fn VirtualAlloc(addr: *const c_void, size: usize, t: u32, p: u32) -> *mut c_void {
    if let Some(q) = placeholders::alloc(addr, size, t, sanitize_protect(p)) {
        trace_mem("VirtualAlloc (placeholder)", addr as usize, size, t as u64, p, q as usize);
        return q;
    }
    let uma = crate::gpu::uma::get();
    // committing inside one of the title's reservations in a sparse window
    if !addr.is_null() && t & 0x1000 != 0 && uma.is_some_and(|u| u.is_sparse(addr as usize)) {
        let q = commit_sparse(addr as usize, size, p);
        trace_mem("VirtualAlloc (sparse commit)", addr as usize, size, t as u64, p, q as usize);
        return q;
    }
    // the title's own reservations of 64 KiB or large pages: in a sparse window, so the GPU sees
    // them. CPU-inaccessible GPU-view mirror reservations stay outside (`mirrors::source`).
    // (GTA VI keeps index and vertex data in a 1.2 GB VirtualAlloc of large pages; outside
    // the windows the GPU faulted on it – a driver timeout)
    let large = t & MEM_64K_PAGES != MEM_64K_PAGES && t & MEM_LARGE_PAGES_X != 0 && p & PAGE_GPU_X != 0;
    if addr.is_null() && t & 0x2000 != 0 && (t & MEM_64K_PAGES == MEM_64K_PAGES || large) && !mirrors::is_gpu_view_reservation(t, p) {
        // (large pages: from the highest window down – the title's physical pages, gigabytes
        // in one piece, keep the lower ones)
        let top = |u: &crate::gpu::uma::Uma| (1..crate::gpu::uma::MAX_WINDOWS).rev().find_map(|k| u.sparse_alloc_in(k, size, crate::gpu::uma::GRAIN).map(|(q, _)| q));
        if let Some(q) = uma.and_then(|u| if large { top(u).or_else(|| u.sparse_reserve(size)) } else { u.sparse_reserve(size) }) {
            if t & 0x1000 != 0 && commit_sparse(q, size, p).is_null() {
                uma.map(|u| u.sparse_release(q));
                return std::ptr::null_mut();
            }
            mirrors::reserved(q, size, t, p);
            trace_mem("VirtualAlloc (64K pages, sparse)", 0, size, t as u64, p, q);
            return q as *mut c_void;
        }
    }
    if addr.is_null() && t & MEM_GRAPHICS != 0 && t & 0x1000 != 0 {
        let align = if t & 0x8000_0000 != 0 {
            4 << 20
        } else if t & 0x2000_0000 != 0 && t & 0x0040_0000 == 0 {
            2 << 20
        } else {
            crate::gpu::uma::GRAIN
        };
        let rounded = size.max(1).div_ceil(align) * align;
        if let Some((q, _)) = crate::hle::d3d12x::sparse::alloc(rounded, align) {
            trace_mem("VirtualAlloc (graphics, sparse VRAM)", 0, rounded, t as u64, p, q);
            return q as *mut c_void;
        }
        if let Some(q) = uma.and_then(|u| u.alloc(rounded, align)) {
            trace_mem("VirtualAlloc (graphics, UMA)", 0, rounded, t as u64, p, q as usize);
            return q as *mut c_void;
        }
    }
    if addr.is_null() && t & MEM_GRAPHICS != 0 && t & 0x2000 != 0 && !mirrors::is_gpu_view_reservation(t, p) {
        // on the console all graphics memory is GPU-visible, reserved address space too (titles
        // reserve a range, then commit or map pages and place resources in it): prefer the
        // sparse VRAM windows first so reservations consume zero system RAM until committed
        let reserved = if t & 0x8000_0000 != 0 {
            let rounded = size.max(1).div_ceil(4 << 20) * (4 << 20);
            uma.and_then(|u| u.sparse_alloc(rounded, 4 << 20)).map(|(q, _)| q)
        } else {
            uma.and_then(|u| u.sparse_reserve(size))
        };
        if let Some(q) = reserved {
            mirrors::reserved(q, size, t, p);
            trace_mem("VirtualAlloc (graphics reserve, sparse)", 0, size, t as u64, p, q);
            return q as *mut c_void;
        }
        if let Some(q) = uma.and_then(|u| u.reserve(size, 0x1_0000)) {
            mirrors::reserved(q as usize, size, t, p);
            trace_mem("VirtualAlloc (graphics reserve, UMA)", 0, size, t as u64, p, q as usize);
            return q as *mut c_void;
        }
        let q = mem::VirtualAlloc(addr, size, mem::MEM_RESERVE | MEM_WRITE_WATCH, sanitize_protect(p));
        if !q.is_null() {
            reserved_outside("graphics reservation", size, t, q as usize);
            GRAPHICS_RESERVATIONS.lock().unwrap().insert(q as usize, size);
            GRAPHICS_RESERVATIONS_GEN.fetch_add(1, Ordering::Release);
            mirrors::reserved(q as usize, size, t, p);
            return q;
        }
    }
    // `XCL_SPARSE_RESERVE_MB=n`: a plain reservation (no commit, no address) of n MB or more goes
    // to the sparse windows too – on the console all title memory is GPU-visible, and a title
    // may map physical pages into a reservation without MEM_GRAPHICS / MEM_64K_PAGES. Off by
    // default: the Stage 0 warnings below show whether a title needs it.
    if addr.is_null() && t & 0x2000 != 0 && t & 0x1000 == 0 && size >= sparse_reserve_min() {
        if let Some(q) = uma.and_then(|u| u.sparse_reserve(size)) {
            trace_mem("VirtualAlloc (reservation, sparse)", 0, size, t as u64, p, q);
            return q as *mut c_void;
        }
    }
    // the title's GPU heaps (committed with a PAGE_GRAPHICS_* protection, write-combined: memory
    // the CPU writes and the GPU reads, as a PC game's upload heaps): in the coherent window, which
    // the GPU reads where it is – one copy (a retail game's 4.4 GB; as plain memory its shaders
    // read zeros, in the sparse windows video memory would have doubled them). The window is as
    // large as what the title asked for last time (`uma::coherent_hint`).
    if addr.is_null() && t & 0x1000 != 0 && p & PAGE_GRAPHICS_ANY != 0 && size <= crate::gpu::uma::MAX_SIZE && std::env::var("XCL_COHERENT_HEAPS").map_or(true, |v| v != "0") {
        static ASKED: AtomicUsize = AtomicUsize::new(0);
        let asked = ASKED.fetch_add(size, Ordering::Relaxed) + size;
        crate::gpu::uma::note_coherent_need(asked);
        if let Some(u) = uma {
            let align = if t & 0x2000_0000 != 0 && t & 0x0040_0000 == 0 { 2 << 20 } else { crate::gpu::uma::GRAIN };
            // what XCL keeps of the window for itself (readback heaps; its per-draw arguments have
            // a pool of their own)
            const KEEP: usize = 96 << 20;
            let fits = u.size().saturating_sub(u.used()) >= size + KEEP;
            if let Some(q) = fits.then(|| u.alloc(size, align)).flatten() {
                trace_mem("VirtualAlloc (graphics, coherent window)", 0, size, t as u64, p, q as usize);
                return q as *mut c_void;
            }
        }
    }
    // GPU-visible title memory the windows above had no room for (a retail game: a 24 GB
    // reservation of 64 KiB pages, 2.2 GB of large pages with a graphics protection) is plain
    // memory here. Textures the title places in it are kept in step with it
    // (`d3d12x::title_textures`), which needs its CPU writes to be seen: it is write-watched
    // like a graphics reservation. Without that such a texture was uploaded once, empty, and
    // never again (the game's glyph atlas: its menus lost their x-height letters).
    if gpu_visible_fallback(addr, t, p) {
        let q = reserve_watched(size, t, p);
        if !q.is_null() {
            mirrors::reserved(q as usize, size, t, p);
            reserved_outside("GPU-visible reservation", size, t, q as usize);
            trace_mem("VirtualAlloc (GPU-visible, write-watched)", addr as usize, size, t as u64, p, q as usize);
            return q;
        }
    }
    let q = mem::VirtualAlloc(addr, size, sanitize_alloc_type(t), sanitize_protect(p));
    trace_mem("VirtualAlloc", addr as usize, size, t as u64, p, q as usize);
    if !q.is_null() && addr.is_null() && t & 0x2000 != 0 && (t & MEM_64K_PAGES == MEM_64K_PAGES || size >= 64 << 20) {
        reserved_outside("reservation", size, t, q as usize);
    }
    q
}

/// Whether a VirtualAlloc that reached the plain-memory fallback is GPU-visible title memory
/// (a reservation with MEM_GRAPHICS, MEM_LARGE_PAGES / MEM_64K_PAGES or a PAGE_GRAPHICS_*
/// protection, at no fixed address), which is reserved write-watched (`reserve_watched`).
fn gpu_visible_fallback(addr: *const c_void, t: u32, p: u32) -> bool {
    addr.is_null() && t & 0x2000 != 0 && (t & (MEM_GRAPHICS | 0x2000_0000) != 0 || p & PAGE_GRAPHICS_ANY != 0)
}

/// Plain memory for GPU-visible title memory, write-watched and recorded with the graphics
/// reservations (whose ranges `title_textures` watches; `VirtualFree` forgets it). Null if
/// Windows refuses it.
unsafe fn reserve_watched(size: usize, t: u32, p: u32) -> *mut c_void {
    let q = mem::VirtualAlloc(std::ptr::null(), size, sanitize_alloc_type(t) | MEM_WRITE_WATCH, sanitize_protect(p));
    if !q.is_null() {
        GRAPHICS_RESERVATIONS.lock().unwrap().insert(q as usize, size);
        GRAPHICS_RESERVATIONS_GEN.fetch_add(1, Ordering::Release);
    }
    q
}

/// `XCL_TRACE_MEM=1`: the title's allocations of 64 KiB or more (the first 60; `XCL_TRACE_MEM=n`:
/// the first n when n > 1)
/// `XCL_SPARSE_RESERVE_MB` in bytes (usize::MAX: unset – plain reservations stay Windows memory).
fn sparse_reserve_min() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| std::env::var("XCL_SPARSE_RESERVE_MB").ok().and_then(|v| v.parse::<usize>().ok()).map_or(usize::MAX, |mb| mb.max(1) << 20))
}

/// Whether `XCL_TRACE_MEM` is set.
fn trace_mem_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_TRACE_MEM").is_some())
}

/// Bytes of the title's reservations that went to plain Windows memory, outside the windows
/// the GPU sees (graphics reservations, 64 KiB-page reservations, anything of 64 MiB or more).
static RESERVED_OUTSIDE: AtomicUsize = AtomicUsize::new(0);

/// Stage 0 of the memory diagnosis: a title reservation the GPU will not see. Always logged
/// once at 1 GB, every one with `XCL_TRACE_MEM`.
fn reserved_outside(what: &str, size: usize, flags: u32, got: usize) {
    let total = RESERVED_OUTSIDE.fetch_add(size, Ordering::Relaxed) + size;
    if trace_mem_on() || (total >= 1 << 30 && total - size < 1 << 30) {
        crate::xlog_warn!("mem: {what} of {} MB (flags {flags:#x}) at {got:#x} is outside the GPU's windows – shaders read zeros there ({} MB so far)", size >> 20, total >> 20);
    }
}

fn trace_mem(what: &str, addr: usize, size: usize, flags: u64, protect: u32, got: usize) {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    static N: AtomicUsize = AtomicUsize::new(0);
    // the call being traced may have just set the title's last error: reading XCL_TRACE_MEM
    // (the first trace; 203 ERROR_ENVVAR_NOT_FOUND when unset) or logging must not change it
    keep_last_error(|| {
        let limit = *LIMIT.get_or_init(|| match std::env::var("XCL_TRACE_MEM") {
            Ok(v) => v.parse().ok().filter(|&n| n > 1).unwrap_or(60),
            Err(_) => 0,
        });
        if size >= 0x1_0000 && N.fetch_add(1, Ordering::Relaxed) < limit {
            crate::xlog_info!("mem: {what}({addr:#x}, {size:#x}, flags {flags:#x}, protect {protect:#x}) → {got:#x}");
        }
    })
}

/// `f` with the thread's last error as it was before.
fn keep_last_error<R>(f: impl FnOnce() -> R) -> R {
    let e = unsafe { fnd::GetLastError() };
    let r = f();
    unsafe { fnd::SetLastError(e) };
    r
}

/// Blocks of the unified memory region go back to it (MEM_RELEASE) or stay (MEM_DECOMMIT
/// zeroes them); others to Windows.
pub unsafe extern "system" fn VirtualFree(addr: *mut c_void, size: usize, t: u32) -> BOOL {
    if let Some(ok) = placeholders::free(addr, size, t) {
        return ok;
    }
    // the title's reservations in a sparse window: decommit (and on release give the address
    // space back); their video memory goes with it
    if let Some(u) = crate::gpu::uma::get().filter(|u| u.is_sparse(addr as usize)) {
        let a = addr as usize;
        let release = t & mem::MEM_RELEASE != 0;
        let decommit = t & mem::MEM_DECOMMIT != 0;
        if release == decommit || (release && size != 0) {
            fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
            return 0;
        }
        let n = if release || size == 0 {
            match u.sparse_reservation(a) {
                Some((start, len)) if start == a => len,
                _ => {
                    fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
                    return 0;
                }
            }
        } else {
            match u.sparse_reservation(a) {
                Some((start, len)) if a.checked_add(size).is_some_and(|end| end <= start + len) => size,
                _ => {
                    fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
                    return 0;
                }
            }
        };
        if n > 0 {
            // sparse first: under its lock, so no copy of CPU writes to video memory reads the
            // pages while they go (the submit thread's copy faulted on them)
            crate::hle::d3d12x::sparse::decommitted(a, n);
            if mem::VirtualFree(addr, n, mem::MEM_DECOMMIT) == 0 {
                return 0;
            }
        }
        if release {
            u.sparse_release(a);
            mirrors::released(a);
        }
        return 1;
    }
    if let Some(u) = crate::gpu::uma::get().filter(|u| u.contains(addr as usize)) {
        if t & 0x8000 != 0 {
            return u.free(addr as usize) as BOOL;
        }
        if t & 0x4000 != 0 && size > 0 {
            std::ptr::write_bytes(addr as *mut u8, 0, size);
        }
        return 1;
    }
    let ok = mem::VirtualFree(addr, size, t);
    if ok != 0 && t & mem::MEM_RELEASE != 0 {
        lazy::forget_released(addr as usize);
        mirrors::released(addr as usize);
    }
    if ok != 0 && t & 0x8000 != 0 && GRAPHICS_RESERVATIONS.lock().unwrap().remove(&(addr as usize)).is_some() {
        GRAPHICS_RESERVATIONS_GEN.fetch_add(1, Ordering::Release);
    }
    ok
}

pub unsafe extern "system" fn VirtualFreeEx(proc_: HANDLE, addr: *mut c_void, size: usize, t: u32) -> BOOL {
    // This also clears lazy-page metadata for plain reservations in this process.
    if proc_ == thr::GetCurrentProcess() || thr::GetProcessId(proc_) == thr::GetCurrentProcessId() {
        return VirtualFree(addr, size, t);
    }
    mem::VirtualFreeEx(proc_, addr, size, t)
}

pub unsafe extern "system" fn VirtualAllocEx(proc_: HANDLE, addr: *const c_void, size: usize, t: u32, p: u32) -> *mut c_void {
    // this process: the title's memory, routed as `VirtualAlloc` (GPU-visible where that puts it)
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetProcessId};
    if proc_ as isize == -1 || GetProcessId(proc_) == GetCurrentProcessId() {
        return VirtualAlloc(addr, size, t, p);
    }
    mem::VirtualAllocEx(proc_, addr, size, sanitize_alloc_type(t), sanitize_protect(p))
}

pub unsafe extern "system" fn VirtualProtect(addr: *const c_void, size: usize, p: u32, old: *mut u32) -> BOOL {
    mem::VirtualProtect(addr, size, sanitize_protect(p), old)
}

pub unsafe extern "system" fn VirtualProtectEx(proc_: HANDLE, addr: *const c_void, size: usize, p: u32, old: *mut u32) -> BOOL {
    mem::VirtualProtectEx(proc_, addr, size, sanitize_protect(p), old)
}

/// XALLOC_ATTRIBUTES bit fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XAllocAttrs {
    pub object_type: u32,
    pub page_size: u32,
    pub allocator_id: u32,
    pub alignment: u32,
    pub memory_type: u32,
}

impl XAllocAttrs {
    pub fn decode(a: u64) -> Self {
        XAllocAttrs {
            object_type: (a & 0x3FFF) as u32,
            page_size: ((a >> 14) & 0x3) as u32,
            allocator_id: ((a >> 16) & 0xFF) as u32,
            alignment: ((a >> 24) & 0x1F) as u32,
            memory_type: ((a >> 29) & 0xF) as u32,
        }
    }
}

unsafe fn aligned_heap_alloc(size: usize, align: usize) -> *mut c_void {
    let Some(total) = size.checked_add(align).and_then(|n| n.checked_add(16)) else {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return std::ptr::null_mut();
    };
    let raw = mem::HeapAlloc(mem::GetProcessHeap(), mem::HEAP_ZERO_MEMORY, total) as usize;
    if raw == 0 {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return std::ptr::null_mut();
    }
    let aligned = (raw + 16 + align - 1) & !(align - 1);
    *((aligned - 8) as *mut usize) = raw;
    aligned as *mut c_void
}

unsafe fn aligned_heap_free(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    let raw = *((p as usize - 8) as *const usize);
    mem::HeapFree(mem::GetProcessHeap(), 0, raw as *const c_void);
}

/// Rounded byte size and alignment for an XMem graphics allocation honoring `page_size`
/// (`2` = `XALLOC_PAGESIZE_4MB`, otherwise 64 KiB sparse tile grain) and `alignment`.
fn xmem_graphics_layout(size: usize, a: XAllocAttrs) -> Option<(usize, usize)> {
    let page_grain = match a.page_size {
        2 => 4 << 20, // XALLOC_PAGESIZE_4MB
        _ => crate::gpu::uma::GRAIN,
    };
    let align = (1usize << a.alignment.min(usize::BITS - 2)).max(page_grain);
    let rounded = size.max(1).checked_next_multiple_of(page_grain)?;
    Some((rounded, align))
}

pub unsafe extern "system" fn XMemAllocDefault(size: usize, attrs: u64) -> *mut c_void {
    let a = XAllocAttrs::decode(attrs);
    if a.memory_type == 0 {
        let q = aligned_heap_alloc(size, 1usize << a.alignment.clamp(4, usize::BITS - 2));
        trace_mem("XMemAlloc", 0, size, attrs, 0, q as usize);
        return q;
    }
    let Some((rounded, align)) = xmem_graphics_layout(size, a) else {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return std::ptr::null_mut();
    };
    // Graphics memory types live in the sparse VRAM pool (fallback to coherent UMA, then Windows).
    if let Some((q, _)) = crate::hle::d3d12x::sparse::alloc(rounded, align) {
        trace_mem("XMemAlloc (graphics, sparse VRAM)", 0, rounded, attrs, 0, q);
        return q as *mut c_void;
    }
    if let Some(q) = crate::gpu::uma::get().and_then(|u| u.alloc(rounded, align)) {
        trace_mem("XMemAlloc (graphics, UMA)", 0, rounded, attrs, 0, q as usize);
        return q as *mut c_void;
    }
    mem::VirtualAlloc(std::ptr::null(), rounded, mem::MEM_COMMIT | mem::MEM_RESERVE, mem::PAGE_READWRITE)
}

pub unsafe extern "system" fn XMemFreeDefault(p: *mut c_void, attrs: u64) {
    if p.is_null() {
        return;
    }
    if XAllocAttrs::decode(attrs).memory_type == 0 {
        aligned_heap_free(p);
    } else if !crate::hle::d3d12x::sparse::free(p as usize) && !crate::gpu::uma::get().is_some_and(|u| u.free(p as usize)) {
        mem::VirtualFree(p, 0, mem::MEM_RELEASE);
    }
}

type XMemAllocFn = unsafe extern "system" fn(usize, u64) -> *mut c_void;
type XMemFreeFn = unsafe extern "system" fn(*mut c_void, u64);
static XMEM_ALLOC_HOOK: AtomicUsize = AtomicUsize::new(0);
static XMEM_FREE_HOOK: AtomicUsize = AtomicUsize::new(0);

pub unsafe extern "system" fn XMemAlloc(size: usize, attrs: u64) -> *mut c_void {
    match XMEM_ALLOC_HOOK.load(Ordering::Acquire) {
        0 => XMemAllocDefault(size, attrs),
        f => std::mem::transmute::<usize, XMemAllocFn>(f)(size, attrs),
    }
}

pub unsafe extern "system" fn XMemFree(p: *mut c_void, attrs: u64) {
    match XMEM_FREE_HOOK.load(Ordering::Acquire) {
        0 => XMemFreeDefault(p, attrs),
        f => std::mem::transmute::<usize, XMemFreeFn>(f)(p, attrs),
    }
}

/// The exe's exported data `__XMemAllocationHooks__` {alloc, free, …}: the allocator functions
/// of a title that replaces XMem's, which the console's runtime libraries call for the memory
/// they need (`runtime_alloc`). Titles see there what the runtime allocates – and may supply the
/// memory themselves.
static TITLE_HOOKS: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];

pub unsafe fn adopt_title_hooks(main: &crate::modules::MappedModule) {
    let Some(crate::pe::ExportTarget::Rva(rva)) = main.exports.by_name.get("__XMemAllocationHooks__") else { return };
    let p = (main.base + *rva as usize) as *const usize;
    let (alloc, free) = (p.read_unaligned(), p.add(1).read_unaligned());
    let inside = |f: usize| f > main.base && f < main.base + main.size;
    if inside(alloc) && inside(free) {
        TITLE_HOOKS[1].store(free, Ordering::Release);
        TITLE_HOOKS[0].store(alloc, Ordering::Release);
        crate::xlog_info!("XMem: the title has allocation hooks ({}, {})", crate::modules::describe_address(alloc), crate::modules::describe_address(free));
    }
}

/// XALLOC_ATTRIBUTES of a descriptor heap's memory: object type 4, 64 KiB pages and alignment,
/// allocator 0x80 (the D3D runtime), graphics memory. The retail game's heap class (its alloc hook for
/// allocator 0x80 compares object type and size) gives the memory of its heaps this way and
/// asserts that they start there.
pub const D3D_DESCRIPTOR_HEAP: u64 = 4 | 1 << 14 | 0x80 << 16 | 16 << 24 | 1 << 29;

/// XALLOC_ATTRIBUTES of a display target's memory: 4 MiB pages (the retail game's hook for its back buffers
/// takes every request with them), allocator 0x80, graphics memory.
pub const D3D_DISPLAY_TARGET: u64 = 2 << 14 | 0x80 << 16 | 16 << 24 | 1 << 29;

/// What the title's allocation hook gives a console runtime library (None: no hook, or null).
pub unsafe fn runtime_alloc(size: usize, attrs: u64) -> Option<usize> {
    let f = TITLE_HOOKS[0].load(Ordering::Acquire);
    if f == 0 {
        return None;
    }
    let p = std::mem::transmute::<usize, XMemAllocFn>(f)(size, attrs) as usize;
    (p != 0).then_some(p)
}

pub unsafe fn runtime_free(p: usize, attrs: u64) {
    let f = TITLE_HOOKS[1].load(Ordering::Acquire);
    if f != 0 {
        std::mem::transmute::<usize, XMemFreeFn>(f)(p as *mut c_void, attrs);
    }
}

/// `void* XMemVirtualAlloc(void* address, SIZE_T size, DWORD type, XALLOC_ATTRIBUTES attributes,
/// DWORD protect)`: VirtualAlloc with an allocator's attributes – a graphics memory type is
/// GPU-visible memory (as MEM_GRAPHICS: sparse VRAM pool).
pub unsafe extern "system" fn XMemVirtualAlloc(addr: *const c_void, size: usize, t: u32, attrs: u64, p: u32) -> *mut c_void {
    let a = XAllocAttrs::decode(attrs);
    let mut t = if a.memory_type != 0 { t | MEM_GRAPHICS } else { t };
    if a.memory_type != 0 && a.page_size == 2 {
        t |= 0x8000_0000;
    }
    VirtualAlloc(addr, size, t, p)
}

/// Fast-path `VirtualQuery` / `VirtualQueryEx` / `XMemVirtualQuery` for a committed coherent
/// UMA window: `OpenExistingHeapFromAddress` makes a kernel `VirtualQuery` on a 4 GiB coherent
/// window walk the whole region (~75–80 ms, holding `MmLockAddressSpace`), stalling
/// `GetWriteWatch` on other threads. A committed coherent window is `PAGE_READWRITE` as a whole.
unsafe fn query_committed_window(address: *const c_void, mbi: *mut mem::MEMORY_BASIC_INFORMATION, len: usize) -> Option<usize> {
    let need = std::mem::size_of::<mem::MEMORY_BASIC_INFORMATION>();
    if mbi.is_null() || len < need {
        return None;
    }
    let addr = address as usize;
    let u = crate::gpu::uma::get()?;
    if !u.in_committed_window(addr, 1) {
        return None;
    }
    let k = u.window_of(addr)?;
    let (win_base, win_len) = u.window(k);
    let page_base = addr & !0xFFF;
    mbi.write_unaligned(mem::MEMORY_BASIC_INFORMATION {
        BaseAddress: page_base as *mut c_void,
        AllocationBase: win_base as *mut c_void,
        AllocationProtect: mem::PAGE_READWRITE,
        PartitionId: 0,
        RegionSize: (win_base + win_len).saturating_sub(page_base),
        State: mem::MEM_COMMIT,
        Protect: mem::PAGE_READWRITE,
        Type: mem::MEM_PRIVATE,
    });
    Some(need)
}

fn is_current_process(process: HANDLE) -> bool {
    process == (-1isize as HANDLE) || process == unsafe { thr::GetCurrentProcess() }
}

pub unsafe extern "system" fn VirtualQuery(address: *const c_void, mbi: *mut mem::MEMORY_BASIC_INFORMATION, len: usize) -> usize {
    if let Some(n) = query_committed_window(address, mbi, len) {
        return n;
    }
    mem::VirtualQuery(address, mbi, len)
}

pub unsafe extern "system" fn VirtualQueryEx(process: HANDLE, address: *const c_void, mbi: *mut mem::MEMORY_BASIC_INFORMATION, len: usize) -> usize {
    if is_current_process(process) {
        if let Some(n) = query_committed_window(address, mbi, len) {
            return n;
        }
    }
    mem::VirtualQueryEx(process, address, mbi, len)
}

/// `BOOL XMemVirtualQuery(HANDLE process, const void* address, XMEM_BASIC_INFORMATION* info)`
/// (signature from XboxBvhBuilder.dll's call sites): MEMORY_BASIC_INFORMATION plus a field
/// at +48 whose bits 0-1 the caller requires for "CPU-readable memory" – set to 1 (CPU
/// cached) for committed readable memory. The caller zeroes the rest.
pub unsafe extern "system" fn XMemVirtualQuery(process: HANDLE, address: *const c_void, info: *mut c_void) -> BOOL {
    if info.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let mbi = info as *mut mem::MEMORY_BASIC_INFORMATION;
    if is_current_process(process) && query_committed_window(address, mbi, std::mem::size_of::<mem::MEMORY_BASIC_INFORMATION>()).is_some() {
        *(info as *mut u8).add(48).cast::<u32>() = 1;
        return 1;
    }
    if mem::VirtualQueryEx(process, address, mbi, std::mem::size_of::<mem::MEMORY_BASIC_INFORMATION>()) == 0 {
        return 0;
    }
    const READABLE: u32 = mem::PAGE_READONLY
        | mem::PAGE_READWRITE
        | mem::PAGE_WRITECOPY
        | mem::PAGE_EXECUTE_READ
        | mem::PAGE_EXECUTE_READWRITE
        | mem::PAGE_EXECUTE_WRITECOPY;
    let protect = (*mbi).Protect;
    let cpu = (*mbi).State == mem::MEM_COMMIT && (protect & mem::PAGE_GUARD) == 0 && (protect & READABLE) != 0;
    *(info as *mut u8).add(48).cast::<u32>() = cpu as u32;
    1
}

/// `BOOL XMemGetWorkingSetStatistics(UINT32 allocator, void* stats)` (as FSR's upscaler calls
/// it: allocator 0x20000001, 128-byte buffer, reads +8 only on success). No statistics:
/// FALSE, which the caller reports as 0 bytes.
/// `BOOL XMemGetWorkingSetStatistics(DWORD flags, XMEM_WORKING_SET_STATISTICS* stats)`: +0 the
/// title's memory limit, +8 what it uses – engines size their pools and texture streaming from
/// these (Fortnite: flags 0x10000001). Only those two are written: the structure's size
/// differs between GDK versions (Fortnite's exe passes 0x78 bytes, its EOSSDK – older – has
/// at most 0x50 bytes before its stack cookie, which zeroing 0x78 bytes destroyed: the DLL's
/// static initializer failed its cookie check, 0xC0000409).
pub unsafe extern "system" fn XMemGetWorkingSetStatistics(_flags: u32, stats: *mut u8) -> BOOL {
    if stats.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let limit = crate::state::config().title_memory_budget_mb() * 1024 * 1024;
    (stats as *mut u64).write_unaligned(limit);
    (stats.add(8) as *mut u64).write_unaligned(process_private_bytes().min(limit));
    1
}

/// `BOOL XMemTransferMemory(direction, size)`: moving budget between the title and the tools
/// pools – there is one budget here
pub unsafe extern "system" fn XMemTransferMemory(_direction: u32, _size: usize) -> BOOL {
    1
}

/// The title's debug output (console builds print a lot there) goes to XCL's log as well.
pub unsafe extern "system" fn OutputDebugStringA(s: *const u8) {
    if let Some(m) = from_ansi_ptr(s) {
        crate::xlog_info!("[debug] {}", m.trim_end());
    }
}

pub unsafe extern "system" fn OutputDebugStringW(s: *const u16) {
    if let Some(m) = from_wide_ptr(s) {
        crate::xlog_info!("[debug] {}", m.trim_end());
    }
}

pub unsafe extern "system" fn XMemSetAllocationHooks(alloc: usize, free: usize) {
    if alloc == 0 {
        XMEM_ALLOC_HOOK.store(0, Ordering::Release);
        XMEM_FREE_HOOK.store(0, Ordering::Release);
    } else {
        XMEM_ALLOC_HOOK.store(alloc, Ordering::Release);
        XMEM_FREE_HOOK.store(free, Ordering::Release);
    }
}

#[repr(C)]
pub struct TitleMemoryStatusT {
    pub dw_length: u32,
    pub dw_reserved: u32,
    pub total: u64,
    pub avail: u64,
    pub legacy_used: u64,
    pub legacy_peak: u64,
    pub legacy_avail: u64,
    pub title_used: u64,
    pub title_avail: u64,
    pub legacy_page_table_used: u64,
    pub title_page_table_used: u64,
}

fn process_private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        if K32GetProcessMemoryInfo(thr::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb) != 0 {
            // Wine fills the fields before PrivateUsage only; PagefileUsage is the same number
            // on Windows (the commit charge)
            if c.PrivateUsage != 0 { c.PrivateUsage as u64 } else { c.PagefileUsage as u64 }
        } else {
            0
        }
    }
}

pub unsafe extern "system" fn TitleMemoryStatus(buf: *mut TitleMemoryStatusT) -> BOOL {
    if buf.is_null() || (*buf).dw_length < std::mem::size_of::<TitleMemoryStatusT>() as u32 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let total = crate::state::config().title_memory_budget_mb() * 1024 * 1024;
    let used = process_private_bytes().min(total);
    let b = &mut *buf;
    b.total = total;
    b.avail = total - used;
    b.legacy_used = 0;
    b.legacy_peak = 0;
    b.legacy_avail = 0;
    b.title_used = used;
    b.title_avail = total - used;
    b.legacy_page_table_used = 0;
    b.title_page_table_used = 0;
    1
}

pub unsafe extern "system" fn JobTitleMemoryStatus(buf: *mut TitleMemoryStatusT) -> BOOL {
    TitleMemoryStatus(buf)
}

pub unsafe extern "system" fn ToolingMemoryStatus(buf: *mut u32) -> BOOL {
    if buf.is_null() || *buf < 40 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    std::ptr::write_bytes((buf as *mut u8).add(8), 0, 32);
    1
}

/// `BOOL XMemIsToolsMemAvailable()`: whether tooling (debug) memory is available on the console.
/// Returns FALSE (0) in retail title execution.
pub unsafe extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

/// Title physical pages (64 KiB). An allocation's pages have a *home*: CPU memory of their own
/// (a reservation per allocation, committed only while a page is there), which holds a page
/// while it is mapped nowhere. Homes take no GPU address space: a retail game allocates ~5.7 GB
/// of pages and maps them into its own 6 GB reservation, and with homes in the sparse windows
/// (as before) the two did not fit the windows' 11.5 GiB together – the reservation went to
/// plain Windows memory and every page mapped there was invisible to shaders (empty V#s).
/// A page's number counts 4 KiB frames (`page_number`; 0 means "none": unmap) of its own index
/// space. Mapping pages at an address – the title's reservation, in any order – commits it and
/// moves their data there (with video-memory tiles in a sparse window), and the home gives its
/// memory back meanwhile (`move_home_away`); unmapping them (or mapping them elsewhere) moves it
/// back home: a page is in one place at a time, keeps its contents and is committed once.
/// Mapping at no chosen address puts the pages at a sparse-window range of their own
/// (`AUTO_MAPPED`: GPU-visible), given back when they are unmapped there. A page never used
/// reads as zeros.
const PHYS_PAGE: usize = 0x1_0000;

/// Size of one console physical page (64 KiB).
pub const PHYS_PAGE_SIZE: usize = PHYS_PAGE;

/// An allocation of physical pages: its pages' indices `first..first + count`.
struct PhysBlock {
    count: usize,
    /// pages still allocated
    live: usize,
    /// the pages' home (CPU memory, `count` pages reserved)
    home: usize,
}

/// first page index → block; and the page indices free again (first, count)
struct PhysPages {
    blocks: std::collections::BTreeMap<usize, PhysBlock>,
    next: usize,
    free: Vec<(usize, usize)>,
}

static PHYS_BLOCKS: std::sync::Mutex<PhysPages> = std::sync::Mutex::new(PhysPages { blocks: std::collections::BTreeMap::new(), next: 0, free: Vec::new() });

/// Sparse-window ranges `MapTitlePhysicalPages` chose (no address given): address → pages.
static AUTO_MAPPED: std::sync::Mutex<std::collections::BTreeMap<usize, usize>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Bytes of physical pages mapped where the GPU does not see them (outside the windows): on the
/// console all title memory is GPU-visible, so shaders read zeros there (logged with `XCL_TRACE_MEM`).
static PAGES_OUTSIDE: AtomicUsize = AtomicUsize::new(0);

/// Pages mapped away from home: address → page, and page → address.
#[derive(Default)]
struct Away {
    at: std::collections::BTreeMap<usize, usize>,
    page: std::collections::HashMap<usize, usize>,
    /// homes in a sparse window whose pages went away in this call, given back at its end in
    /// runs (`give_back_homes`): one page at a time, a retail game's start took 20 s
    gone: Vec<usize>,
}
static AWAY: std::sync::Mutex<Option<Away>> = std::sync::Mutex::new(None);

/// A live physical mapping exists at this 64 KiB page even before lazy backing is
/// touched/committed. Consult the mapping table, not the host virtual-memory state.
pub fn physical_page_at(va: usize) -> bool {
    AWAY.lock().unwrap().as_ref().is_some_and(|a| a.at.contains_key(&(va & !(PHYS_PAGE - 1))))
}

static GPU_VIEW_MAP_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn gpu_view_map_gen() -> u64 {
    GPU_VIEW_MAP_GEN.load(Ordering::Acquire)
}

fn note_mapping_at(va: usize) {
    if mirrors::source(va).is_some() {
        GPU_VIEW_MAP_GEN.fetch_add(1, Ordering::Release);
    }
}

/// Page numbers count 4 KiB frames, as on the console: a 64 KiB page's is a multiple of 16 (a retail game
/// packs them shifted right by 4). Index 0 of unified memory is page 16 - 0 means "no page".
const FRAMES: usize = PHYS_PAGE >> 12;

fn page_number(index: usize) -> usize {
    (index + 1) * FRAMES
}

fn page_index(page: usize) -> Option<usize> {
    page.is_multiple_of(FRAMES).then(|| (page / FRAMES).checked_sub(1)).flatten()
}

/// The home address of page `page` if it belongs to a live allocation.
fn page_home(page: usize) -> Option<usize> {
    let index = page_index(page)?;
    let g = PHYS_BLOCKS.lock().ok()?;
    let (&first, b) = g.blocks.range(..=index).next_back()?;
    (b.live > 0 && index < first.checked_add(b.count)?).then(|| b.home.checked_add((index - first).checked_mul(PHYS_PAGE)?)).flatten()
}

/// Resolve the live virtual addresses of the physical pages covering `offset..offset+len`.
/// Caller must hold `AWAY`.
unsafe fn resolve_physical_pages(away: Option<&Away>, pages: &[usize], offset: usize, len: usize) -> Option<Vec<usize>> {
    if offset >= PHYS_PAGE {
        return None;
    }
    let span = offset.checked_add(len)?;
    let need = span.div_ceil(PHYS_PAGE);
    if pages.len() < need {
        return None;
    }
    let mut addrs = Vec::with_capacity(need);
    for &p in &pages[..need] {
        let home = page_home(p)?;
        let addr = match away.and_then(|a| a.page.get(&p).copied()) {
            Some(at) => at,
            None => home_again(home)?,
        };
        addrs.push(addr);
    }
    Some(addrs)
}

/// Copy `dst.len()` bytes starting at `offset` within `pages` (each a 64 KiB physical page
/// number from `AllocateTitlePhysicalPages`). Returns `false` if `offset >= PHYS_PAGE_SIZE`,
/// `pages` is too short, or any page is not a live allocation.
pub unsafe fn read_physical_pages(pages: &[usize], offset: usize, dst: &mut [u8]) -> bool {
    if dst.is_empty() {
        return offset < PHYS_PAGE;
    }
    let g = AWAY.lock().unwrap();
    let Some(addrs) = resolve_physical_pages(g.as_ref(), pages, offset, dst.len()) else { return false };
    let mut done = 0;
    let mut page_off = offset;
    for addr in addrs {
        let chunk = (PHYS_PAGE - page_off).min(dst.len() - done);
        std::ptr::copy_nonoverlapping((addr + page_off) as *const u8, dst.as_mut_ptr().add(done), chunk);
        done += chunk;
        page_off = 0;
    }
    true
}

/// Copy `src` into `pages` starting at `offset` within the first page. Validates every page
/// before writing so the destination gets all bytes or nothing.
pub unsafe fn write_physical_pages(pages: &[usize], offset: usize, src: &[u8]) -> bool {
    if src.is_empty() {
        return offset < PHYS_PAGE;
    }
    let g = AWAY.lock().unwrap();
    let Some(addrs) = resolve_physical_pages(g.as_ref(), pages, offset, src.len()) else { return false };
    let mut done = 0;
    let mut page_off = offset;
    for addr in addrs {
        let chunk = (PHYS_PAGE - page_off).min(src.len() - done);
        std::ptr::copy_nonoverlapping(src.as_ptr().add(done), (addr + page_off) as *mut u8, chunk);
        done += chunk;
        page_off = 0;
    }
    true
}

/// A page's data went from `home` to another address: a home in a sparse window is
/// decommitted at the end of the call or every 256 MB (`give_back_homes`; its GPU tiles go too: the GPU does
/// not see the page while it is away). True if homes were given back now.
unsafe fn move_home_away(away: &mut Away, home: usize) -> bool {
    away.gone.push(home);
    // (256 MB at a time: a call that maps gigabytes would otherwise hold them twice until
    // it ends)
    if away.gone.len() >= 4096 {
        give_back_homes(away);
        return true;
    }
    false
}

/// The homes `move_home_away` collected, decommitted in runs of neighbouring pages.
unsafe fn give_back_homes(away: &mut Away) {
    let mut homes = std::mem::take(&mut away.gone);
    homes.sort_unstable();
    homes.dedup();
    let mut i = 0;
    while i < homes.len() {
        let start = homes[i];
        let mut end = start + PHYS_PAGE;
        i += 1;
        while i < homes.len() && homes[i] == end {
            end += PHYS_PAGE;
            i += 1;
        }
        mem::VirtualFree(start as *mut c_void, end - start, mem::MEM_DECOMMIT);
    }
}

/// The page's home, committed again if `move_home_away` gave it back (None: it could not be).
unsafe fn home_again(home: usize) -> Option<usize> {
    if !committed_page(home) && mem::VirtualAlloc(home as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() {
        crate::xlog_warn!("physical page: its home {home:#x} could not be committed again");
        return None;
    }
    Some(home)
}

#[cfg(test)]
/// Whether all of `[a, a + len)` is committed: one VirtualQuery per run of pages in one state
/// (page by page each asked walked the rest of the run).
fn committed_range(a: usize, len: usize) -> bool {
    if len > 0 && crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(a, len)) {
        return true;
    }
    let mut p = a;
    while p < a + len {
        let mut m: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { mem::VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 || m.State != mem::MEM_COMMIT {
            return false;
        }
        p = m.BaseAddress as usize + m.RegionSize;
    }
    true
}

fn committed_page(a: usize) -> bool {
    committed_page_in(a, &mut StateRun::default())
}

/// The run of pages around an address that one VirtualQuery found in one state. `map_pages`
/// asks about the pages of a range one after the other and nothing it does in between (a page
/// left lazy) changes that state: a run answers them without asking again (VirtualQuery walks
/// the run to its end, and under Wine 98,304 fresh pages took 13 s).
#[derive(Default)]
struct StateRun {
    start: usize,
    end: usize,
    committed: bool,
    reserved: bool,
}

/// `committed_page`, remembering in `run` the state of the pages around `a`.
fn committed_page_in(a: usize, run: &mut StateRun) -> bool {
    if (run.start..run.end).contains(&a) {
        return run.committed;
    }
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(a, 1)) {
        return true;
    }
    // A page in the working set is committed: asking for that is one entry, where VirtualQuery
    // walks the page table to the end of the committed range the page is in (gigabytes of
    // homes: 0.25 ms a page, and a scatter of pages from their homes grew with its square).
    let mut ws = windows_sys::Win32::System::ProcessStatus::PSAPI_WORKING_SET_EX_INFORMATION { VirtualAddress: a as *mut c_void, ..Default::default() };
    if unsafe { windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx(thr::GetCurrentProcess(), &mut ws as *mut _ as *mut c_void, std::mem::size_of_val(&ws) as u32) } != 0
        && unsafe { ws.VirtualAttributes.Flags } & 1 != 0
    {
        return true;
    }
    #[cfg(test)]
    VIRTUAL_QUERIES.with(|n| n.set(n.get() + 1));
    let mut m: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { mem::VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 {
        return false;
    }
    let start = m.BaseAddress as usize;
    *run = StateRun { start, end: start + m.RegionSize, committed: m.State == mem::MEM_COMMIT, reserved: m.State == mem::MEM_RESERVE };
    run.committed
}

#[cfg(test)]
thread_local! {
    /// The VirtualQuery calls `committed_page_in` made on this thread.
    static VIRTUAL_QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Deferral must not turn an invalid address (or a partially committed page) into
/// a successful physical mapping. Share #556's query, but require the whole page reserved.
fn reserved_page_in(a: usize, run: &mut StateRun) -> bool {
    !committed_page_in(a, run) && run.reserved && (run.start..run.end).contains(&a)
        && a.checked_add(PHYS_PAGE).is_some_and(|end| end <= run.end)
}

/// Copy what is mapped at `at` back home and unmap it (Windows memory is decommitted).
unsafe fn unmap_away(away: &mut Away, at: usize) -> bool {
    unmap_away_preserving(away, at, &(0..0))
}

/// Homes being restored in this call remain committed even if another page was mapped there.
/// True if memory was committed or decommitted, invalidating the caller's query runs.
unsafe fn unmap_away_preserving(away: &mut Away, at: usize, restored: &std::ops::Range<usize>) -> bool {
    if let Some(page) = away.at.remove(&at) {
        away.page.remove(&page);
        note_mapping_at(at);
        // Never touched: the page still consists of zeros at its uncommitted home.
        if lazy::unmark(at) {
            return false;
        }
        // (a home this call has not given back yet is still committed)
        let pending = away.gone.len();
        away.gone.retain(|&h| Some(h) != page_home(page));
        let own = page_home(page);
        let home = if away.gone.len() < pending || own.is_some_and(|h| restored.contains(&h)) { own } else { own.and_then(|h| home_again(h)) };
        if let Some(home) = home {
            std::ptr::copy_nonoverlapping(at as *const u8, home as *mut u8, PHYS_PAGE);
            #[cfg(test)]
            tests::HOME_COPIES.with(|c| c.set(c.get() + 1));
        }
        // (an address among the homes the caller maps back is one of them: it stays)
        if !restored.contains(&at) {
            release_at(at);
        }
        true
    } else {
        false
    }
}

/// The memory at `at`, which a page mapped away from home left, decommitted (UMA memory that is
/// committed as a whole stays).
unsafe fn release_at(at: usize) {
    if crate::gpu::uma::get().is_some_and(|u| u.is_committed(at)) {
        return;
    }
    if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at)) {
        crate::hle::d3d12x::sparse::decommitted(at, PHYS_PAGE);
    }
    mem::VirtualFree(at as *mut c_void, PHYS_PAGE, mem::MEM_DECOMMIT);
}

/// A page mapped away from home (at `old`) that the title maps at `at`: what `old` holds copied
/// to `at`, and `old` given up – its home is not used. By way of its home a move was four
/// commits and decommits, each a tile mapping in a sparse window: a retail game moving pages one
/// at a time after its first movie spent 20 s in them and its watchdog ended it. False: the page
/// was never used at `old` (its home holds it, zeros) and is not mapped anywhere now.
unsafe fn move_away(away: &mut Away, old: usize, at: usize, page: usize, in_uma: bool) -> Result<bool, ()> {
    if lazy::unmark(old) {
        away.at.remove(&old);
        away.page.remove(&page);
        note_mapping_at(old);
        return Ok(false);
    }
    // (a destination that cannot be committed leaves the page where it was, mapped as before)
    if !in_uma && mem::VirtualAlloc(at as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() {
        return Err(());
    }
    away.at.remove(&old);
    away.page.remove(&page);
    std::ptr::copy_nonoverlapping(old as *const u8, at as *mut u8, PHYS_PAGE);
    if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at)) {
        crate::hle::d3d12x::sparse::committed(at, PHYS_PAGE);
    }
    release_at(old);
    away.at.insert(at, page);
    away.page.insert(page, at);
    note_mapping_at(old);
    note_mapping_at(at);
    Ok(true)
}

pub unsafe extern "system" fn AllocateTitlePhysicalPages(_p: HANDLE, _t: u32, num: *mut usize, arr: *mut usize) -> BOOL {
    if num.is_null() || arr.is_null() || *num == 0 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let n = *num;
    let Some(bytes) = n.checked_mul(PHYS_PAGE) else {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    };
    // the home: CPU address space only (committed while a page is there: `home_again`)
    let home = mem::VirtualAlloc(std::ptr::null(), bytes, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
    if home == 0 {
        crate::xlog_warn!("AllocateTitlePhysicalPages({n}): no address space for their home");
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    }
    let mut g = PHYS_BLOCKS.lock().unwrap();
    let first = match g.free.iter().position(|&(_, c)| c >= n) {
        Some(i) => {
            let (f, c) = g.free[i];
            if c == n {
                g.free.swap_remove(i);
            } else {
                g.free[i] = (f + n, c - n);
            }
            f
        }
        None => {
            g.next += n;
            g.next - n
        }
    };
    for i in 0..n {
        *arr.add(i) = page_number(first + i);
    }
    g.blocks.insert(first, PhysBlock { count: n, live: n, home });
    1
}

pub unsafe extern "system" fn FreeTitlePhysicalPages(_p: HANDLE, n: usize, arr: *mut usize) -> BOOL {
    if arr.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // freed pages mapped elsewhere: that address keeps whatever it holds, the mapping goes
    if let Some(away) = AWAY.lock().unwrap().as_mut() {
        for i in 0..n {
            if let Some(at) = away.page.remove(&*arr.add(i)) {
                away.at.remove(&at);
                note_mapping_at(at);
            }
        }
    }
    let mut g = PHYS_BLOCKS.lock().unwrap();
    for i in 0..n {
        let Some(page) = page_index(*arr.add(i)) else { continue };
        // the block holding this page
        let Some((&first, b)) = g.blocks.range(..=page).next_back() else { continue };
        if page >= first + b.count {
            continue;
        }
        let e = g.blocks.get_mut(&first).unwrap();
        e.live = e.live.saturating_sub(1);
        if e.live == 0 {
            let b = g.blocks.remove(&first).unwrap();
            mem::VirtualFree(b.home as *mut c_void, 0, mem::MEM_RELEASE);
            g.free.push((first, b.count));
        }
    }
    1
}

pub unsafe extern "system" fn MapTitlePhysicalPages(va: *mut c_void, n: usize, _t: u32, _p: u32, arr: *mut usize) -> *mut c_void {
    let mut g = AWAY.lock().unwrap();
    let away = g.get_or_insert_with(Away::default);
    let r = map_pages(away, va, n, arr);
    give_back_homes(away);
    r
}

/// `MapTitlePhysicalPages` with `AWAY` held; the homes it moves away are given back by the
/// caller (`give_back_homes`).
unsafe fn map_pages(away: &mut Away, va: *mut c_void, n: usize, arr: *mut usize) -> *mut c_void {
    if crate::gpu::uma::get().is_none() || n == 0 || (va.is_null() && arr.is_null()) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    let pages: Vec<usize> = if arr.is_null() { vec![0; n] } else { (0..n).map(|i| *arr.add(i)).collect() };
    if pages.iter().any(|&p| p != 0 && page_home(p).is_none()) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    // An explicit request for these pages' CPU homes must restore them in place, not
    // self-copy/decommit them through the ordinary away path. NULL still chooses a
    // GPU-visible range below, as required by the local physical-page ABI.
    let base = va as usize;
    let homes = if !va.is_null() && pages.iter().enumerate().all(|(i, &p)| {
        p != 0 && i.checked_mul(PHYS_PAGE).and_then(|offset| base.checked_add(offset)) == page_home(p)
    }) {
        n.checked_mul(PHYS_PAGE).and_then(|len| base.checked_add(len)).map(|end| base..end)
    } else {
        None
    };
    if let Some(homes) = homes {
        for home in (homes.start..homes.end).step_by(PHYS_PAGE) {
            if home_again(home).is_none() {
                fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
                return std::ptr::null_mut();
            }
        }
        away.gone.retain(|home| !homes.contains(home));
        for &p in &pages {
            if let Some(&at) = away.page.get(&p) {
                unmap_away_preserving(away, at, &homes);
            }
        }
        return va;
    }
    let va = if va.is_null() {
        // no address chosen: consecutive pages get a sparse-window range of their own
        let consecutive = pages[0] != 0 && (1..n).all(|i| pages[0].checked_add(i * FRAMES) == Some(pages[i]));
        if !consecutive {
            fnd::SetLastError(fnd::ERROR_NOT_SUPPORTED); // scattered pages need an address
            return std::ptr::null_mut();
        }
        let Some((at, _)) = crate::gpu::uma::get().and_then(|u| u.sparse_alloc(n * PHYS_PAGE, PHYS_PAGE)) else {
            crate::xlog_warn!("MapTitlePhysicalPages({n}): no room in the sparse windows");
            fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
            return std::ptr::null_mut();
        };
        AUTO_MAPPED.lock().unwrap().insert(at, n);
        at as *mut c_void
    } else {
        va
    };
    let base = va as usize;
    let visible = |a: usize| crate::gpu::uma::get().is_some_and(|u| u.is_sparse(a) || u.is_committed(a));
    if pages.iter().any(|&p| p != 0) && !visible(base) {
        let total = PAGES_OUTSIDE.fetch_add(n * PHYS_PAGE, Ordering::Relaxed) + n * PHYS_PAGE;
        if trace_mem_on() {
            crate::xlog_warn!("physical pages: {n} mapped at {base:#x}, outside the GPU's windows – shaders see zeros there ({} MB so far)", total >> 20);
        }
    }
    if !base.is_multiple_of(PHYS_PAGE) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    let in_uma = |a: usize| crate::gpu::uma::get().is_some_and(|u| u.is_committed(a));
    // the state of the addresses and homes asked about: dropped when `unmap_away` commits or
    // decommits and when `move_home_away` gives homes back. The rest of a step changes only an
    // address, or a home not asked about again (the addresses rise; a page mapped twice is
    // unmapped first) – bar an address that is another page's home (a home in plain memory):
    // that answer was wrong before the runs too.
    let (mut at_run, mut home_run) = (StateRun::default(), StateRun::default());
    for (i, &p) in pages.iter().enumerate() {
        let at = base + i * PHYS_PAGE;
        if unmap_away(away, at) {
            (at_run, home_run) = Default::default();
        }
        if p == 0 {
            continue;
        }
        if let Some(&old) = away.page.get(&p) {
            // (a move commits `at` and decommits `old`, which may lie in a run already answered:
            // both runs go – #556)
            match move_away(away, old, at, p, in_uma(at)) {
                Ok(true) => {
                    (at_run, home_run) = Default::default();
                    continue;
                }
                // a lazy page is only unmarked: memory is as it was, the runs stay
                Ok(false) => {}
                Err(()) => {
                    fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
                    return std::ptr::null_mut();
                }
            }
        }
        let fresh = !in_uma(at) && !committed_page_in(at, &mut at_run);
        let home = page_home(p).unwrap();
        // Only CPU-only reservations may defer backing. Sparse UMA needs its GPU tiles now.
        let plain = crate::gpu::uma::get().is_none_or(|u| !u.contains(at));
        if fresh && plain && reserved_page_in(at, &mut at_run) && !committed_page_in(home, &mut home_run) && lazy::mark(at) {
            away.at.insert(at, p);
            away.page.insert(p, at);
            note_mapping_at(at);
            continue;
        }
        if !in_uma(at) && mem::VirtualAlloc(at as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() {
            fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
            return std::ptr::null_mut();
        }
        if committed_page(home) {
            std::ptr::copy_nonoverlapping(home as *const u8, at as *mut u8, PHYS_PAGE);
            if move_home_away(away, home) {
                (at_run, home_run) = Default::default();
            }
        } else if !fresh {
            std::ptr::write_bytes(at as *mut u8, 0, PHYS_PAGE); // a page never used: zeros
        }
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at)) {
            crate::hle::d3d12x::sparse::committed(at, PHYS_PAGE);
        }
        away.at.insert(at, p);
        away.page.insert(p, at);
        note_mapping_at(at);
    }
    // a range chosen here (no address given), all of it unmapped: back to the sparse windows
    if pages.iter().all(|&p| p == 0) {
        let mut auto = AUTO_MAPPED.lock().unwrap();
        if auto.get(&base) == Some(&n) {
            auto.remove(&base);
            if let Some(u) = crate::gpu::uma::get() {
                u.sparse_release(base);
            }
        }
    }
    va
}

/// `BOOL XMemAllocatePhysicalPages(XALLOC_ATTRIBUTES, SIZE_T* count, ULONG64* pages)`: the
/// title physical pages above (64 KiB)
pub unsafe extern "system" fn XMemAllocatePhysicalPages(_attrs: u64, num: *mut usize, arr: *mut u64) -> BOOL {
    AllocateTitlePhysicalPages(std::mem::zeroed(), 0, num, arr as *mut usize)
}

/// `void* XMemMapPhysicalPages(void* address, SIZE_T count, ULONG64* pages)`
pub unsafe extern "system" fn XMemMapPhysicalPages(va: *mut c_void, n: usize, arr: *mut u64) -> *mut c_void {
    MapTitlePhysicalPages(va, n, 0, 0, arr as *mut usize)
}

/// `BOOL XMemMapPhysicalPagesScatter(void** addresses, SIZE_T count, ULONG64* pages)`: one 64 KiB
/// page at each address (page 0: the address is unmapped). No page array unmaps every address:
/// a retail game unmaps that way while it streams its world and stops on its own `int3` when the
/// call fails.
pub unsafe extern "system" fn XMemMapPhysicalPagesScatter(addresses: *const *mut c_void, n: usize, pages: *mut u64) -> BOOL {
    if addresses.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    if crate::gpu::uma::get().is_none() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let mut g = AWAY.lock().unwrap();
    let away = g.get_or_insert_with(Away::default);
    // no page array: page 0 (unmap) for every address
    let ok = (0..n).all(|i| {
        let mut none = 0usize;
        let page = if pages.is_null() { &mut none as *mut usize } else { pages.add(i) as *mut usize };
        !map_pages(away, *addresses.add(i), 1, page).is_null()
    });
    give_back_homes(away);
    ok as BOOL
}

/// `BOOL XMemFreePhysicalPages(SIZE_T count, ULONG64* pages)`
pub unsafe extern "system" fn XMemFreePhysicalPages(n: usize, arr: *mut u64) -> BOOL {
    FreeTitlePhysicalPages(std::mem::zeroed(), n, arr as *mut usize)
}

// ---------------------------------------------------------------------------------------
// App state (minappmodelx.dll, appnotify.h)
// ---------------------------------------------------------------------------------------

/// The registration handed out: the callbacks are never called (XCL does not suspend titles
/// or constrain them for the system UI).
static APP_NOTIFICATION: u8 = 0;

/// `ULONG RegisterAppStateChangeNotification(routine, context, PAPPSTATE_REGISTRATION* out)`
pub unsafe extern "system" fn RegisterAppStateChangeNotification(_routine: usize, _ctx: *mut c_void, out: *mut *const u8) -> u32 {
    if out.is_null() {
        return fnd::ERROR_INVALID_PARAMETER;
    }
    *out = &APP_NOTIFICATION;
    0
}

pub unsafe extern "system" fn UnregisterAppStateChangeNotification(_registration: *const u8) {}

/// `ULONG RegisterAppConstrainedChangeNotification(routine, context, PAPPCONSTRAIN_REGISTRATION* out)`
pub unsafe extern "system" fn RegisterAppConstrainedChangeNotification(routine: usize, ctx: *mut c_void, out: *mut *const u8) -> u32 {
    RegisterAppStateChangeNotification(routine, ctx, out)
}

pub unsafe extern "system" fn UnregisterAppConstrainedChangeNotification(_registration: *const u8) {}

// ---------------------------------------------------------------------------------------
// Threads / system info
// ---------------------------------------------------------------------------------------

fn hresult_to_win32(hr: i32) -> u32 {
    if (hr as u32 & 0xFFFF_0000) == 0x8007_0000 {
        hr as u32 & 0xFFFF
    } else {
        fnd::ERROR_INVALID_PARAMETER
    }
}

/// a sleep is taken in steps of at most this, so that time held from then on is not counted
const SLEEP_STEP_MS: u32 = 20;
/// how often a sleep looks whether the title is still held
const SLEEP_HELD_POLL_MS: u32 = 5;
/// the shortest sleep that is a wait for time: a shorter one is a poll and is passed on as it is
const SLEEP_HOLD_MIN_MS: u32 = 100;

/// Sleep(ms): `ms` of the title's time. The time XCL holds one of the title's threads in a bind
/// for a pipeline still being built (`pso_async::title_held`) is not the title's: a console
/// title never waits there. A retail title's own stall check – a thread that ticks once a second
/// with Sleep and ends the process after 30 ticks without a sign of its main thread – ended it
/// while the driver built the pipelines of its world (#505, #232). Only a sleep of
/// `SLEEP_HOLD_MIN_MS` or more is taken so: a thread that polls with short sleeps (streaming,
/// sound) goes on while a bind waits. `XCL_SLEEP_HOLD=<ms>` sets that length, `0`: the PC's
/// Sleep as it is.
pub unsafe extern "system" fn Sleep(ms: u32) {
    static MIN: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let Some(min) = *MIN.get_or_init(|| sleep_hold_min(std::env::var("XCL_SLEEP_HOLD").ok().as_deref())) else {
        return thr::Sleep(ms);
    };
    title_sleep(ms, min, |ms| thr::Sleep(ms), crate::hle::d3d12x::pso_async::title_held)
}

/// `XCL_SLEEP_HOLD`: the shortest sleep whose held time is not counted (`None`: no sleep).
fn sleep_hold_min(v: Option<&str>) -> Option<u32> {
    match v.map(str::trim) {
        None => Some(SLEEP_HOLD_MIN_MS),
        Some("0") => None,
        Some(v) => Some(v.parse().unwrap_or(SLEEP_HOLD_MIN_MS)),
    }
}

/// `ms` of time in which the title was not `held`, slept through `sleep` (a yield, an endless
/// sleep and one shorter than `min` as they are).
fn title_sleep(ms: u32, min: u32, sleep: impl Fn(u32), held: impl Fn() -> bool) {
    if ms == 0 || ms == thr::INFINITE || ms < min {
        return sleep(ms);
    }
    let mut left = std::time::Duration::from_millis(ms as u64);
    while !left.is_zero() {
        if held() {
            sleep(SLEEP_HELD_POLL_MS);
            continue;
        }
        let t = std::time::Instant::now();
        sleep((left.as_millis() as u32).clamp(1, SLEEP_STEP_MS));
        left = left.saturating_sub(t.elapsed());
    }
}

pub unsafe extern "system" fn SetThreadName(h: HANDLE, name: *const u16) -> BOOL {
    if let Some(n) = from_wide_ptr(name) {
        crate::xlog_debug!("SetThreadName({n})");
    }
    let hr = thr::SetThreadDescription(h, name);
    if hr < 0 {
        fnd::SetLastError(hresult_to_win32(hr));
        return 0;
    }
    1
}

pub unsafe extern "system" fn GetThreadName(h: HANDLE, buf: *mut u16, len: usize, ret_len: *mut usize) -> BOOL {
    if buf.is_null() && len > 0 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let mut p: *mut u16 = std::ptr::null_mut();
    let hr = thr::GetThreadDescription(h, &mut p);
    if hr < 0 || p.is_null() {
        fnd::SetLastError(hresult_to_win32(hr));
        return 0;
    }
    let s = from_wide_ptr(p).unwrap_or_default();
    fnd::LocalFree(p as *mut c_void);
    let n = s.encode_utf16().count();
    if !ret_len.is_null() {
        *ret_len = n;
    }
    if buf.is_null() || len <= n {
        fnd::SetLastError(fnd::ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    write_wide_out(&s, buf, len.min(u32::MAX as usize) as u32);
    1
}

pub unsafe extern "system" fn SetThreadpoolAffinityMask(_pool: *mut c_void, _mask: usize) -> BOOL {
    1
}

/// An affinity mask the PC takes. The title sees the PC's processors, so a mask made from them
/// passes unchanged. A console mask (its 7 cores / 14 threads) that names processors outside the
/// process's affinity – a PC with fewer processors, or a process limited to some – makes Windows
/// fail the call (ERROR_INVALID_PARAMETER) when none of its processors is there or any is outside
/// the process's mask: it is cut to the processors there are, and when none is left, each console
/// processor goes to the PC's processor at its index modulo their count.
pub fn fit_affinity_mask(mask: usize, allowed: usize) -> usize {
    if mask == 0 || allowed == 0 || mask & !allowed == 0 {
        return mask;
    }
    if mask & allowed != 0 {
        return mask & allowed;
    }
    let pcs: Vec<u32> = (0..usize::BITS).filter(|b| allowed >> b & 1 != 0).collect();
    (0..usize::BITS).filter(|b| mask >> b & 1 != 0).fold(0, |m, b| m | 1 << pcs[b as usize % pcs.len()])
}

/// The process's affinity mask; 0 (masks pass unchanged) with more than one processor group.
fn process_affinity() -> usize {
    unsafe {
        if thr::GetActiveProcessorGroupCount() != 1 {
            return 0;
        }
        let (mut process, mut system) = (0usize, 0usize);
        if thr::GetProcessAffinityMask(thr::GetCurrentProcess(), &mut process, &mut system) == 0 {
            return 0;
        }
        process
    }
}

fn fitted_affinity(mask: usize, what: &str) -> usize {
    static LOGGED: std::sync::Mutex<std::collections::BTreeSet<usize>> = std::sync::Mutex::new(std::collections::BTreeSet::new());
    let allowed = process_affinity();
    let fit = fit_affinity_mask(mask, allowed);
    if fit != mask && LOGGED.lock().unwrap_or_else(|e| e.into_inner()).insert(mask) {
        crate::xlog_info!("{what}: mask {mask:#x} names processors outside the process's {allowed:#x} – {fit:#x}");
    }
    fit
}

pub unsafe extern "system" fn SetThreadAffinityMask(h: HANDLE, mask: usize) -> usize {
    thr::SetThreadAffinityMask(h, fitted_affinity(mask, "SetThreadAffinityMask"))
}

pub unsafe extern "system" fn SetThreadGroupAffinity(h: HANDLE, a: *const GROUP_AFFINITY, prev: *mut GROUP_AFFINITY) -> BOOL {
    if a.is_null() || (*a).Group != 0 {
        return thr::SetThreadGroupAffinity(h, a, prev);
    }
    let mut fit = *a;
    fit.Mask = fitted_affinity(fit.Mask, "SetThreadGroupAffinity");
    thr::SetThreadGroupAffinity(h, &fit, prev)
}

pub unsafe extern "system" fn QueryProcessorSchedulingStatistics(out: *mut [u64; 3]) {
    if out.is_null() {
        return;
    }
    let mut idle = std::mem::zeroed::<fnd::FILETIME>();
    let mut kern = std::mem::zeroed::<fnd::FILETIME>();
    let mut user = std::mem::zeroed::<fnd::FILETIME>();
    thr::GetSystemTimes(&mut idle, &mut kern, &mut user);
    let ft = |f: fnd::FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
    let (i, k, u) = (ft(idle), ft(kern), ft(user));
    *out = [(k + u).saturating_sub(i), i, k + u];
}

pub unsafe extern "system" fn GetConsoleType() -> u32 {
    crate::state::config().console_type_raw
}

#[repr(C)]
pub struct SystemOsVersionInfo {
    pub major: u8,
    pub minor: u8,
    pub build: u16,
    pub revision: u16,
}

#[repr(C)]
struct OsVersionInfoW {
    size: u32,
    major: u32,
    minor: u32,
    build: u32,
    platform: u32,
    csd: [u16; 128],
}

#[link(name = "ntdll", kind = "raw-dylib")]
extern "system" {
    fn RtlGetVersion(info: *mut OsVersionInfoW) -> i32;
    fn NtQueryTimerResolution(coarsest: *mut u32, finest: *mut u32, current: *mut u32) -> i32;
    fn NtSetTimerResolution(desired: u32, set: u8, current: *mut u32) -> i32;
}

pub fn host_os_version() -> (u32, u32, u32) {
    unsafe {
        let mut v: OsVersionInfoW = std::mem::zeroed();
        v.size = std::mem::size_of::<OsVersionInfoW>() as u32;
        RtlGetVersion(&mut v);
        (v.major, v.minor, v.build)
    }
}

/// The finest timer Windows offers (0.5 ms) for the title: with the default 15.6 ms tick every
/// sleep rounds up (a retail game's audio mixer ran at a third of real time). Windows 11 keeps
/// it while the window is hidden only when told to.
pub fn fine_timer() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
    };
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        StateMask: 0,
    };
    let size = std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32;
    let kept = unsafe { SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state as *const _ as *const c_void, size) } != 0;
    // in 100 ns units; "finest" is the smallest period the system supports
    let (mut coarsest, mut finest, mut was) = (0u32, 0u32, 0u32);
    if unsafe { NtQueryTimerResolution(&mut coarsest, &mut finest, &mut was) } < 0 || finest == 0 {
        crate::xlog_warn!("timer: NtQueryTimerResolution failed – Windows' default resolution");
        return;
    }
    let mut now = 0u32;
    let status = unsafe { NtSetTimerResolution(finest, 1, &mut now) };
    crate::xlog_info!(
        "timer: {} (was {:.2} ms, now {:.2} ms; kept while the window is hidden: {})",
        if status >= 0 { "finest resolution for the title" } else { "raising the resolution failed" },
        was as f64 / 10_000.0,
        now as f64 / 10_000.0,
        if kept { "yes" } else { "no" }
    );
}

pub unsafe extern "system" fn GetSystemOSVersion(out: *mut SystemOsVersionInfo) {
    if out.is_null() {
        return;
    }
    let (major, minor, build) = host_os_version();
    *out = SystemOsVersionInfo { major: major as u8, minor: minor as u8, build: build as u16, revision: 0 };
}

// ---------------------------------------------------------------------------------------
// Registry: the console mode
// ---------------------------------------------------------------------------------------
//
// The GDK thunks a title links compute XSystemGetDeviceType from
// HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM, value ConsoleMode (REG_DWORD):
// 0x64 Series X (XboxScarlettAnaconda), 0x60 Series S (Lockhart), 0x68 devkit, 0x40 / 0x50 /
// 0x58 Xbox One S / X / devkit, no key = PC – read from the thunk in the game's XCurl.dll.
// Only that key is answered here; everything else goes to the real registry.

const OEM_KEY: &str = "software\\microsoft\\windows nt\\currentversion\\oem";
/// handle of the answered key (never reaches the real registry)
const OEM_HANDLE: usize = 0x5843_4C4F; // "XCLO"

fn console_mode() -> u32 {
    if crate::state::config().is_series_s() {
        0x60
    } else {
        0x64
    }
}

unsafe fn is_oem_key(root: usize, path: *const u16) -> bool {
    let hklm = windows_sys::Win32::System::Registry::HKEY_LOCAL_MACHINE as usize;
    root == hklm && from_wide_ptr(path).is_some_and(|p| p.trim_start_matches('\\').eq_ignore_ascii_case(OEM_KEY))
}

pub unsafe extern "system" fn RegOpenKeyExW(root: usize, path: *const u16, options: u32, sam: u32, out: *mut usize) -> u32 {
    if is_oem_key(root, path) && !out.is_null() {
        crate::xlog_debug!("registry: OEM key → console mode {:#x}", console_mode());
        *out = OEM_HANDLE;
        return 0;
    }
    windows_sys::Win32::System::Registry::RegOpenKeyExW(root as _, path, options, sam, out as _)
}

pub unsafe extern "system" fn RegOpenKeyW(root: usize, path: *const u16, out: *mut usize) -> u32 {
    if is_oem_key(root, path) && !out.is_null() {
        *out = OEM_HANDLE;
        return 0;
    }
    windows_sys::Win32::System::Registry::RegOpenKeyW(root as _, path, out as _)
}

pub unsafe extern "system" fn RegQueryValueExW(key: usize, name: *const u16, reserved: *mut u32, ty: *mut u32, data: *mut u8, size: *mut u32) -> u32 {
    if key != OEM_HANDLE {
        return windows_sys::Win32::System::Registry::RegQueryValueExW(key as _, name, reserved, ty, data, size);
    }
    const ERROR_FILE_NOT_FOUND: u32 = 2;
    const ERROR_MORE_DATA: u32 = 234;
    if !from_wide_ptr(name).is_some_and(|n| n.eq_ignore_ascii_case("ConsoleMode")) {
        return ERROR_FILE_NOT_FOUND;
    }
    if !ty.is_null() {
        *ty = 4; // REG_DWORD
    }
    if !data.is_null() {
        if size.is_null() || *size < 4 {
            if !size.is_null() {
                *size = 4;
            }
            return ERROR_MORE_DATA;
        }
        std::ptr::copy_nonoverlapping(console_mode().to_le_bytes().as_ptr(), data, 4);
    }
    if !size.is_null() {
        *size = 4;
    }
    0
}

pub unsafe extern "system" fn RegCloseKey(key: usize) -> u32 {
    if key == OEM_HANDLE {
        0
    } else {
        windows_sys::Win32::System::Registry::RegCloseKey(key as _)
    }
}

// ---------------------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------------------

const LOAD_AS_DATA: u32 = 0x2 | 0x20 | 0x40; // DATAFILE | IMAGE_RESOURCE | DATAFILE_EXCLUSIVE

fn load_library_impl(name: Option<String>) -> usize {
    let Some(name) = name else {
        unsafe { fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER) };
        return 0;
    };
    match resolver::load_library(&name) {
        Some(h) => {
            crate::xlog_debug!("LoadLibrary({name}) → {h:#x}");
            h
        }
        None => {
            crate::xlog_warn!("LoadLibrary({name}) → NULL");
            unsafe { fnd::SetLastError(fnd::ERROR_MOD_NOT_FOUND) };
            0
        }
    }
}

pub unsafe extern "system" fn LoadLibraryW(name: *const u16) -> usize {
    load_library_impl(from_wide_ptr(name))
}

pub unsafe extern "system" fn LoadLibraryA(name: *const u8) -> usize {
    load_library_impl(from_ansi_ptr(name))
}

unsafe fn load_as_data(name: &str, file: HANDLE, flags: u32) -> usize {
    let real = crate::vfs::get().and_then(|v| v.real_path_for_read(name)).unwrap_or_else(|| name.to_string());
    ll::LoadLibraryExW(crate::util::wide(&real).as_ptr(), file, flags) as usize
}

pub unsafe extern "system" fn LoadLibraryExW(name: *const u16, file: HANDLE, flags: u32) -> usize {
    if flags & LOAD_AS_DATA != 0 {
        let Some(name) = from_wide_ptr(name) else {
            fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
            return 0;
        };
        return load_as_data(&name, file, flags);
    }
    load_library_impl(from_wide_ptr(name))
}

pub unsafe extern "system" fn LoadLibraryExA(name: *const u8, file: HANDLE, flags: u32) -> usize {
    if flags & LOAD_AS_DATA != 0 {
        let Some(name) = from_ansi_ptr(name) else {
            fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
            return 0;
        };
        return load_as_data(&name, file, flags);
    }
    load_library_impl(from_ansi_ptr(name))
}

pub unsafe extern "system" fn LoadPackagedLibrary(name: *const u16, _reserved: u32) -> usize {
    load_library_impl(from_wide_ptr(name))
}

pub unsafe extern "system" fn GetProcAddress(h: usize, name: *const u8) -> usize {
    if name.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let what = if (name as usize) >> 16 == 0 {
        ImportName::Ordinal(name as usize as u16)
    } else {
        ImportName::Name { hint: 0, name: crate::util::from_cstr_ptr(name).unwrap_or_default() }
    };
    let h = if h == 0 { modules::main_module().map(|m| m.base).unwrap_or(0) } else { h };
    match resolver::get_proc_address(h, &what) {
        Some(a) => a,
        None => {
            fnd::SetLastError(fnd::ERROR_PROC_NOT_FOUND);
            0
        }
    }
}

fn module_handle_impl(name: &str) -> usize {
    module_handle_in(crate::hle::registry(), name)
}

fn module_handle_in(reg: &crate::hle::Registry, name: &str) -> usize {
    let key = dll_key(name);
    match reg.route(&key) {
        Some(crate::hle::Route::Hle(i)) => {
            let m = &reg.modules[*i];
            if m.aliases_are_system() && key != m.name {
                // Real Windows DLLs (kernel32 / kernelbase / advapi32 / Windows API sets) keep
                // their genuine HMODULE; console-only aliases (advapi32legacy, kernel32legacy,
                // minappmodelx) fall back to the HLE pseudo handle, matching LoadLibrary.
                // SAFETY: `wide(name)` produces a valid NUL-terminated UTF-16 string.
                let real = unsafe { ll::GetModuleHandleW(crate::util::wide(name).as_ptr()) as usize };
                if real != 0 {
                    return real;
                }
            }
            return modules::hle_handle(m.name);
        }
        Some(crate::hle::Route::Stub) => {
            // Return the pseudo handle only if the @stub DLL has already been loaded or imported,
            // preserving GetModuleHandle's non-loading probe semantics for optional DLLs.
            return modules::find_hle_handle(&key).unwrap_or(0);
        }
        Some(crate::hle::Route::System) => {
            // SAFETY: `wide(name)` produces a valid NUL-terminated UTF-16 string.
            return unsafe { ll::GetModuleHandleW(crate::util::wide(name).as_ptr()) as usize };
        }
        None => {}
    }
    if let Some(m) = modules::find_mapped_by_name(&key) {
        return m.base;
    }
    // SAFETY: `wide(name)` produces a valid NUL-terminated UTF-16 string.
    unsafe { ll::GetModuleHandleW(crate::util::wide(name).as_ptr()) as usize }
}

fn main_or_real_base() -> usize {
    modules::main_module()
        .map(|m| m.base)
        .unwrap_or_else(|| unsafe { ll::GetModuleHandleW(std::ptr::null()) as usize })
}

pub unsafe extern "system" fn GetModuleHandleW(name: *const u16) -> usize {
    match from_wide_ptr(name) {
        None => main_or_real_base(),
        Some(n) => {
            let h = module_handle_impl(&n);
            if h == 0 {
                fnd::SetLastError(fnd::ERROR_MOD_NOT_FOUND);
            }
            h
        }
    }
}

pub unsafe extern "system" fn GetModuleHandleA(name: *const u8) -> usize {
    match from_ansi_ptr(name) {
        None => main_or_real_base(),
        Some(n) => {
            let h = module_handle_impl(&n);
            if h == 0 {
                fnd::SetLastError(fnd::ERROR_MOD_NOT_FOUND);
            }
            h
        }
    }
}

const FROM_ADDRESS: u32 = 0x4;

unsafe fn module_handle_ex(flags: u32, name: Option<String>, addr: usize, out: *mut usize) -> BOOL {
    if out.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let h = if flags & FROM_ADDRESS != 0 {
        match modules::image_base_for_addr(addr) {
            Some(b) => b,
            None => {
                let mut r = std::ptr::null_mut();
                if ll::GetModuleHandleExW(flags, addr as *const u16, &mut r) != 0 {
                    r as usize
                } else {
                    0
                }
            }
        }
    } else {
        match name {
            None => main_or_real_base(),
            Some(n) => module_handle_impl(&n),
        }
    };
    *out = h;
    if h == 0 {
        fnd::SetLastError(fnd::ERROR_MOD_NOT_FOUND);
        0
    } else {
        1
    }
}

pub unsafe extern "system" fn GetModuleHandleExW(flags: u32, name: *const u16, out: *mut usize) -> BOOL {
    let n = if flags & FROM_ADDRESS != 0 { None } else { from_wide_ptr(name) };
    module_handle_ex(flags, n, name as usize, out)
}

pub unsafe extern "system" fn GetModuleHandleExA(flags: u32, name: *const u8, out: *mut usize) -> BOOL {
    let n = if flags & FROM_ADDRESS != 0 { None } else { from_ansi_ptr(name) };
    module_handle_ex(flags, n, name as usize, out)
}

fn module_path_for(h: usize) -> Option<String> {
    let main = modules::main_module();
    if h == 0 || main.as_ref().map(|m| m.base) == Some(h) {
        return main.map(|m| m.path.to_string_lossy().into_owned());
    }
    if let Some(m) = modules::find_mapped_by_base(h) {
        return Some(m.path.to_string_lossy().into_owned());
    }
    modules::hle_name_by_handle(h).map(|n| format!("C:\\Windows\\System32\\{n}"))
}

pub unsafe extern "system" fn GetModuleFileNameW(h: usize, buf: *mut u16, size: u32) -> u32 {
    match module_path_for(h) {
        Some(p) => write_wide_out(&p, buf, size),
        None => ll::GetModuleFileNameW(h as _, buf, size),
    }
}

pub unsafe extern "system" fn GetModuleFileNameA(h: usize, buf: *mut u8, size: u32) -> u32 {
    match module_path_for(h) {
        Some(p) => write_ansi_out(&p, buf, size),
        None => ll::GetModuleFileNameA(h as _, buf, size),
    }
}

pub unsafe extern "system" fn FreeLibrary(h: usize) -> BOOL {
    if modules::find_mapped_by_base(h).is_some() || modules::hle_name_by_handle(h).is_some() {
        return 1;
    }
    fnd::FreeLibrary(h as _)
}

/// A thread-pool callback asks for its module to be freed when it returns (the title's
/// msvcp140 does, for the reference it took on itself). A module XCL mapped is not in Windows'
/// loader – the pool's FreeLibrary fails and raises 0xC000070E in the worker (Minecraft: 20
/// workers at once, then a fail-fast). XCL's modules are never unloaded: nothing to do for them.
pub unsafe extern "system" fn FreeLibraryWhenCallbackReturns(instance: *mut c_void, h: usize) {
    let main = modules::main_module().is_some_and(|m| m.base == h);
    if main || modules::find_mapped_by_base(h).is_some() || modules::hle_name_by_handle(h).is_some() {
        return;
    }
    thr::FreeLibraryWhenCallbackReturns(instance as _, h as _)
}

pub unsafe extern "system" fn DisableThreadLibraryCalls(h: usize) -> BOOL {
    if let Some(m) = modules::find_mapped_by_base(h) {
        m.no_thread_calls.store(true, Ordering::Relaxed);
        return 1;
    }
    ll::DisableThreadLibraryCalls(h as _)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_sleep(ms: u32) {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }

    /// A title's Sleep does not count the time the title is held (a bind waiting for a pipeline
    /// still being built): it returns once that is over and its own time has gone by after it.
    #[test]
    fn a_sleep_does_not_count_the_time_the_title_is_held() {
        use std::sync::atomic::AtomicBool;
        let held = AtomicBool::new(true);
        let start = std::time::Instant::now();
        std::thread::scope(|s| {
            s.spawn(|| {
                real_sleep(300);
                held.store(false, Ordering::Release);
            });
            title_sleep(60, 60, real_sleep, || held.load(Ordering::Acquire));
        });
        let took = start.elapsed();
        assert!(took >= std::time::Duration::from_millis(355), "back after {took:?}: the 300 ms held were counted");
    }

    /// Time held from the middle of a sleep on is not counted either (but for the step under way).
    #[test]
    fn a_sleep_stops_counting_when_the_title_gets_held() {
        use std::sync::atomic::AtomicBool;
        let held = AtomicBool::new(false);
        let start = std::time::Instant::now();
        std::thread::scope(|s| {
            s.spawn(|| {
                real_sleep(50);
                held.store(true, Ordering::Release);
                real_sleep(300);
                held.store(false, Ordering::Release);
            });
            title_sleep(200, SLEEP_HOLD_MIN_MS, real_sleep, || held.load(Ordering::Acquire));
        });
        let took = start.elapsed();
        // 200 ms of its own, less at most one step counted while held, and the 300 ms held
        assert!(took >= std::time::Duration::from_millis(200 + 300 - SLEEP_STEP_MS as u64 - 15), "back after {took:?}");
    }

    /// With the title not held a sleep takes its time, in steps; a yield and an endless sleep
    /// are passed on as they are.
    #[test]
    fn a_sleep_of_a_title_not_held_takes_its_time() {
        let calls = std::cell::RefCell::new(Vec::new());
        let sleep = |ms: u32| {
            calls.borrow_mut().push(ms);
            if ms != 0 && ms != thr::INFINITE {
                real_sleep(ms);
            }
        };
        title_sleep(0, 1, sleep, || false);
        title_sleep(thr::INFINITE, 1, sleep, || false);
        assert_eq!(*calls.borrow(), [0, thr::INFINITE]);
        calls.borrow_mut().clear();
        let t = std::time::Instant::now();
        title_sleep(50, 1, sleep, || false);
        let took = t.elapsed();
        assert!(took >= std::time::Duration::from_millis(50) && took < std::time::Duration::from_millis(400), "took {took:?}");
        assert!(calls.borrow().iter().all(|&ms| (1..=SLEEP_STEP_MS).contains(&ms)), "{:?}", calls.borrow());
        // a sleep shorter than a step is one sleep, as before
        calls.borrow_mut().clear();
        title_sleep(1, 1, sleep, || false);
        assert_eq!(*calls.borrow(), [1]);
    }

    /// A sleep shorter than the shortest one held is a poll: it is passed on as it is, also
    /// while the title is held. `XCL_SLEEP_HOLD` sets that length, `0` takes no sleep.
    #[test]
    fn a_short_sleep_is_passed_on_while_the_title_is_held() {
        let calls = std::cell::RefCell::new(Vec::new());
        let sleep = |ms: u32| calls.borrow_mut().push(ms);
        title_sleep(SLEEP_HOLD_MIN_MS - 1, SLEEP_HOLD_MIN_MS, sleep, || true);
        assert_eq!(*calls.borrow(), [SLEEP_HOLD_MIN_MS - 1]);
        assert_eq!(sleep_hold_min(None), Some(SLEEP_HOLD_MIN_MS));
        assert_eq!(sleep_hold_min(Some("0")), None);
        assert_eq!(sleep_hold_min(Some(" 250 ")), Some(250));
        assert_eq!(sleep_hold_min(Some("on")), Some(SLEEP_HOLD_MIN_MS));
    }

    #[test]
    fn fine_timer_makes_short_sleeps_short() {
        // Windows' Sleep (std::thread::sleep has its own high-resolution timer)
        fine_timer();
        let t = std::time::Instant::now();
        for _ in 0..20 {
            unsafe { windows_sys::Win32::System::Threading::Sleep(1) };
        }
        let per = t.elapsed() / 20;
        assert!(per < std::time::Duration::from_millis(5), "Sleep(1) took {per:?}: the 15.6 ms tick");
    }

    #[test]
    fn commitment_probe_preserves_resident_cold_reserved_and_guarded_states() {
        unsafe {
            struct Reservation(*mut c_void);
            impl Drop for Reservation {
                fn drop(&mut self) { unsafe { mem::VirtualFree(self.0, 0, mem::MEM_RELEASE); } }
            }
            let allocation = Reservation(mem::VirtualAlloc(std::ptr::null(), 6 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE));
            assert!(!allocation.0.is_null());
            let r = allocation.0 as usize;
            let commit = |at: usize, protection| {
                assert_eq!(mem::VirtualAlloc(at as *const c_void, 4096, mem::MEM_COMMIT, protection) as usize, at);
            };
            let working_set = |at: usize| {
                let mut ws = windows_sys::Win32::System::ProcessStatus::PSAPI_WORKING_SET_EX_INFORMATION { VirtualAddress: at as *mut c_void, ..Default::default() };
                assert_ne!(windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx(thr::GetCurrentProcess(), &mut ws as *mut _ as *mut c_void, std::mem::size_of_val(&ws) as u32), 0);
                ws.VirtualAttributes.Flags & 1 != 0
            };
            commit(r, mem::PAGE_READWRITE);
            (r as *mut u32).write_volatile(0x12345678);
            assert!(working_set(r), "the touched page exercises the resident fast-positive");
            assert!(committed_page(r));
            assert_eq!((r as *const u32).read_volatile(), 0x12345678);
            let cold = r + PHYS_PAGE;
            commit(cold, mem::PAGE_READWRITE);
            assert!(!working_set(cold), "untouched commitment exercises the VirtualQuery fallback");
            assert!(committed_page(cold));
            let reserved = r + 2 * PHYS_PAGE;
            assert!(!committed_page(reserved));
            for (index, protection) in [(3, mem::PAGE_READWRITE | mem::PAGE_GUARD), (4, mem::PAGE_NOACCESS), (5, mem::PAGE_READONLY)] {
                let at = r + index * PHYS_PAGE;
                commit(at, protection);
                assert!(committed_page(at), "commitment does not imply readable/writable memory");
                let mut region: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(mem::VirtualQuery(at as *const c_void, &mut region, std::mem::size_of_val(&region)), 0);
                assert_eq!(region.State, mem::MEM_COMMIT);
                assert_eq!(region.Protect, protection, "metadata probe preserves protections and guard state");
            }
            drop(allocation);
            assert!(!committed_page(r), "a released page is not a positive working-set entry");
        }
    }

    #[test]
    fn physical_pages_reject_an_unreserved_plain_destination() {
        unsafe {
            crate::gpu::uma::get().expect("UMA must initialize for the physical-page API test");
            let mut count = 1usize;
            let mut pages = [0usize; 1];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut count, pages.as_mut_ptr()), 1);
            let destination = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE);
            assert!(!destination.is_null());
            assert_ne!(mem::VirtualFree(destination, 0, mem::MEM_RELEASE), 0);
            assert!(MapTitlePhysicalPages(destination, 1, 0, 0, pages.as_mut_ptr()).is_null());
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_ADDRESS);
            let away = AWAY.lock().unwrap();
            assert!(!away.as_ref().is_some_and(|a| a.at.contains_key(&(destination as usize)) || a.page.contains_key(&pages[0])));
            drop(away);
            assert!(!lazy::unmark(destination as usize), "invalid mapping left no deferred page");
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr()), 1);
        }
    }

    /// The real physical-page API defers only untouched CPU-only destinations, then
    /// preserves written contents through unmapping/remapping and untouched zeros.
    #[test]
    fn physical_pages_in_plain_reservations_are_lazy_and_round_trip() {
        unsafe {
            let u = crate::gpu::uma::get().expect("UMA must initialize for the physical-page API test");
            let destination = mem::VirtualAlloc(std::ptr::null(), 3 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert_ne!(destination, 0);
            assert!(!u.contains(destination), "the test must use a plain CPU reservation");
            let mut count = 3usize;
            let mut pages = [0usize; 3];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut count, pages.as_mut_ptr()), 1);
            assert_eq!(MapTitlePhysicalPages(destination as *mut c_void, 3, 0, 0, pages.as_mut_ptr()) as usize, destination);
            for i in 0..3 { assert!(!committed_page(destination + i * PHYS_PAGE), "mapped zeros stay reserved until used"); }
            assert_eq!(std::ptr::read_volatile(destination as *const u32), 0);
            std::ptr::write_volatile(destination as *mut u32, 0x55AA);
            lazy::ensure(destination + PHYS_PAGE + 4, 4);
            assert!(committed_page(destination) && committed_page(destination + PHYS_PAGE));
            std::ptr::write_volatile((destination + PHYS_PAGE + 4) as *mut u32, 0x7711);
            assert!(!committed_page(destination + 2 * PHYS_PAGE));
            assert_eq!(MapTitlePhysicalPages(destination as *mut c_void, 3, 0, 0, std::ptr::null_mut()) as usize, destination);
            assert!(!committed_page(page_home(pages[2]).unwrap()), "unmapping untouched zeros does not commit their home");
            assert_eq!(MapTitlePhysicalPages(destination as *mut c_void, 3, 0, 0, pages.as_mut_ptr()) as usize, destination);
            assert_eq!(std::ptr::read_volatile(destination as *const u32), 0x55AA);
            assert_eq!(std::ptr::read_volatile((destination + PHYS_PAGE + 4) as *const u32), 0x7711);
            assert!(!committed_page(destination + 2 * PHYS_PAGE));
            assert_eq!(std::ptr::read_volatile((destination + 2 * PHYS_PAGE) as *const u32), 0);
            assert_eq!(MapTitlePhysicalPages(destination as *mut c_void, 3, 0, 0, std::ptr::null_mut()) as usize, destination);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 3, pages.as_mut_ptr()), 1);
            assert_eq!(VirtualFree(destination as *mut c_void, 0, mem::MEM_RELEASE), 1);
        }
    }

    /// The GDK thunks' device type lookup: RegOpenKeyW(HKLM\…\OEM) + RegQueryValueExW(ConsoleMode).
    #[test]
    fn console_mode_registry_value() {
        unsafe {
            let hklm = windows_sys::Win32::System::Registry::HKEY_LOCAL_MACHINE as usize;
            let path = crate::util::wide("Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM");
            let mut k = 0usize;
            assert_eq!(RegOpenKeyW(hklm, path.as_ptr(), &mut k), 0);
            let (mut ty, mut v, mut n) = (0u32, 0u32, 4u32);
            let name = crate::util::wide("ConsoleMode");
            assert_eq!(RegQueryValueExW(k, name.as_ptr(), std::ptr::null_mut(), &mut ty, &mut v as *mut u32 as *mut u8, &mut n), 0);
            assert_eq!((ty, n), (4, 4));
            assert_eq!(v, console_mode());
            let other = crate::util::wide("Other");
            assert_eq!(RegQueryValueExW(k, other.as_ptr(), std::ptr::null_mut(), &mut ty, &mut v as *mut u32 as *mut u8, &mut n), 2);
            assert_eq!(RegCloseKey(k), 0);
            // any other key: the real registry
            let real = crate::util::wide("Software\\Microsoft\\Windows NT\\CurrentVersion");
            let mut r = 0usize;
            assert_eq!(RegOpenKeyExW(hklm, real.as_ptr(), 0, 0x20019, &mut r), 0);
            assert_ne!(r, OEM_HANDLE);
            assert_eq!(RegCloseKey(r), 0);
        }
    }

    #[test]
    fn console_alloc_flags_are_stripped() {
        assert_eq!(sanitize_alloc_type(MEM_GRAPHICS | 0x1000 | 0x2000), 0x3000);
        assert_eq!(sanitize_alloc_type(MEM_TITLE), 0x3000);
        assert_eq!(sanitize_protect(0x1000_0004), 0x04);
        assert_eq!(sanitize_protect(0x2000_0000), mem::PAGE_READWRITE);
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 0x10000, MEM_GRAPHICS | MEM_TITLE | 0x3000, 0x0400_0004);
            assert!(!p.is_null());
            *(p as *mut u64) = 0xABCD;
            // graphics memory: from the GPU-visible region (when it could be created)
            if let Some(u) = crate::gpu::uma::get() {
                assert!(u.contains(p as usize));
            }
            assert_eq!(VirtualFree(p, 0, mem::MEM_RELEASE), 1);
            let q = VirtualAlloc(std::ptr::null(), 0x10000, 0x3000, 0x04); // ordinary memory: Windows
            assert!(crate::gpu::uma::get().is_none_or(|u| !u.contains(q as usize)));
            assert_eq!(VirtualFree(q, 0, mem::MEM_RELEASE), 1);
        }
    }

    #[test]
    fn xmem_placeholder_split_replace_and_coalesce() {
        unsafe {
            // The title splits a placeholder, replaces each piece with a physical-page
            // reservation, then maps pages at those exact GPU-visible addresses.
            let grain = PHYS_PAGE;
            let base = XMemVirtualAlloc(std::ptr::null(), 3 * grain, 0x42000, 7, mem::PAGE_NOACCESS);
            assert!(!base.is_null());
            let split = VirtualFree(base, grain, 0x8002);
            if split == 0 {
                VirtualFree(base, 0, mem::MEM_RELEASE);
            }
            assert_eq!(split, 1, "split a placeholder without releasing its address space");
            let rest = base.byte_add(grain);
            assert_eq!(VirtualFree(rest, grain, 0x8002), 1);
            assert_eq!(XMemVirtualAlloc(base, grain, 0x20406000, 0x27, 0x1004), base);
            assert!(XMemVirtualAlloc(rest, 2 * grain, 0x20406000, 0x27, 0x1004).is_null(), "replacement must match one placeholder");
            if let Some(u) = crate::gpu::uma::get() {
                assert!(u.is_title_address(base as usize), "replacement stays GPU-visible");
            }
            assert_eq!(VirtualAlloc(base, grain, mem::MEM_COMMIT, mem::PAGE_READWRITE), base);
            *(base as *mut u32) = 0x12345678;
            assert_eq!(VirtualFree(base, grain, 0x8002), 1, "return a replacement to a placeholder");
            assert_eq!(VirtualFree(base, 3 * grain, 0x8001), 1, "coalesce adjacent placeholders");
            assert_eq!(VirtualFree(base, 0, mem::MEM_RELEASE), 1);
        }
    }

    #[test]
    fn xmem_alignment_and_zeroing() {
        let attrs = |align_log2: u64, mtype: u64| (align_log2 << 24) | (mtype << 29);
        assert_eq!(XAllocAttrs::decode(attrs(8, 0)).alignment, 8);
        unsafe {
            let p = XMemAlloc(1000, attrs(8, 0));
            assert!(!p.is_null());
            assert_eq!(p as usize % 256, 0);
            assert!(std::slice::from_raw_parts(p as *const u8, 1000).iter().all(|&b| b == 0));
            XMemFree(p, attrs(8, 0));
            let g = XMemAlloc(0x20000, attrs(0, 1)); // graphics memory → VirtualAlloc
            assert!(!g.is_null());
            assert_eq!(g as usize % 0x10000, 0);
            XMemFree(g, attrs(0, 1));
        }
    }

    #[test]
    fn xmem_virtual_query_marks_cpu_readable_memory() {
        unsafe {
            let mut info = [0u8; 64];
            let x = 5u32;
            let proc_ = thr::GetCurrentProcess();
            assert_eq!(XMemVirtualQuery(proc_, &x as *const u32 as _, info.as_mut_ptr() as _), 1);
            let mbi = &*(info.as_ptr() as *const mem::MEMORY_BASIC_INFORMATION);
            assert_eq!(mbi.State, mem::MEM_COMMIT);
            assert_eq!(u32::from_le_bytes(info[48..52].try_into().unwrap()) & 3, 1);
            // reserved, not committed: not CPU-readable
            let r = mem::VirtualAlloc(std::ptr::null(), 0x10000, mem::MEM_RESERVE, mem::PAGE_NOACCESS);
            let mut info = [0u8; 64];
            assert_eq!(XMemVirtualQuery(proc_, r, info.as_mut_ptr() as _), 1);
            assert_eq!(info[48] & 3, 0);
            mem::VirtualFree(r, 0, mem::MEM_RELEASE);
        }
    }

    /// Physical pages: homes in CPU memory (no GPU address space), mapped into the windows.
    #[test]
    fn physical_pages_have_cpu_homes_and_map_into_the_windows() {
        unsafe {
            let Some(u) = crate::gpu::uma::get() else { return };
            let mut n = 3usize;
            let mut pages = [0usize; 3];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            // 4 KiB frame numbers: multiples of 16, consecutive pages 16 apart
            assert!(pages[0].is_multiple_of(16) && pages[1] == pages[0] + 16 && pages[2] == pages[1] + 16);
            let home = page_home(pages[0]).unwrap();
            assert!(!u.contains(home), "a home takes no GPU address space");
            assert!(!committed_page(home), "and is only reserved until a page is there");
            // no address given: a range of the sparse windows (GPU-visible)
            let p = MapTitlePhysicalPages(std::ptr::null_mut(), 3, 0, 4, pages.as_mut_ptr());
            assert!(!p.is_null() && u.is_sparse(p as usize) && (p as usize).is_multiple_of(PHYS_PAGE));
            *(p as *mut u32).add(PHYS_PAGE / 4) = 0x55AA; // page 1
            // the same pages mapped again there: the same contents
            let q = MapTitlePhysicalPages(p, 3, 0, 4, pages.as_mut_ptr());
            assert_eq!(q, p);
            assert_eq!(*(p as *const u32).add(PHYS_PAGE / 4), 0x55AA);
            // a single page of the block maps to its own place, with its contents
            let r = MapTitlePhysicalPages(std::ptr::null_mut(), 1, 0, 4, pages.as_mut_ptr().add(1));
            assert!(u.is_sparse(r as usize));
            assert_eq!(*(r as *const u32), 0x55AA);
            // scattered pages with no address: refused
            let mut scattered = [pages[2], pages[0]];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, scattered.as_mut_ptr()).is_null());
            // … but one by one they go to addresses of their own
            let (range, _) = u.sparse_alloc(2 * PHYS_PAGE, PHYS_PAGE).unwrap();
            let at = [range, range + PHYS_PAGE].map(|a| a as *mut c_void);
            let mut ids = [pages[0] as u64, pages[2] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 2, ids.as_mut_ptr()), 1);
            ids[1] += 1;
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 2, ids.as_mut_ptr()), 0, "not a page number");
            // page numbers must belong to a live allocation
            let mut outside = [page_number(1 << 40)];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 1, 0, 4, outside.as_mut_ptr()).is_null());
            let mut wrapping = [usize::MAX];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, wrapping.as_mut_ptr()).is_null());
            // unmapping a range chosen here gives it back
            assert_eq!(MapTitlePhysicalPages(r, 1, 0, 4, std::ptr::null_mut()), r);
            assert!(!AUTO_MAPPED.lock().unwrap().contains_key(&(r as usize)));
            // freed: the block and its home are gone, its page numbers can be handed out again
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 3, pages.as_mut_ptr()), 1);
            assert!(!PHYS_BLOCKS.lock().unwrap().blocks.contains_key(&page_index(pages[0]).unwrap()));
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 3, 0, 4, pages.as_mut_ptr()).is_null());
            for a in at {
                VirtualFree(a, PHYS_PAGE, mem::MEM_DECOMMIT);
            }
            u.sparse_release(range);
        }
    }

    /// A page is committed once: its home gives its memory back while the page is mapped
    /// elsewhere and has the page's contents again when it comes back.
    #[test]
    fn a_page_away_from_home_is_committed_once() {
        unsafe {
            let Some(u) = crate::gpu::uma::get() else { return };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let home = page_home(pages[1]).unwrap();
            assert!(!committed_page(home), "a home is only reserved until it is used");
            // a page never used reads as zeros, and its home stays reserved
            let (r0, _) = u.sparse_alloc(PHYS_PAGE, PHYS_PAGE).unwrap();
            let mut never = [pages[0]];
            assert_eq!(MapTitlePhysicalPages(r0 as *mut c_void, 1, 0, 4, never.as_mut_ptr()) as usize, r0);
            assert_eq!(*(r0 as *const u32), 0, "a page never used reads as zeros");
            assert!(!committed_page(home - PHYS_PAGE), "and its home stays reserved");
            // written at one address, moved to another: the contents go with it, the first
            // address and the home are given back
            let (r1, _) = u.sparse_alloc(PHYS_PAGE, PHYS_PAGE).unwrap();
            let (r2, _) = u.sparse_alloc(PHYS_PAGE, PHYS_PAGE).unwrap();
            let mut one = [pages[1]];
            assert_eq!(MapTitlePhysicalPages(r1 as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r1);
            *(r1 as *mut u32) = 0x1234_5678;
            assert_eq!(MapTitlePhysicalPages(r2 as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r2);
            assert_eq!(*(r2 as *const u32), 0x1234_5678);
            assert!(!committed_page(r1), "the old address is decommitted");
            assert!(!committed_page(home), "the home is given back while the page is away");
            *(r2 as *mut u32) = 0x9ABC_DEF0;
            // unmapped: back home, committed, with what was written away
            assert_eq!(MapTitlePhysicalPages(r2 as *mut c_void, 1, 0, 4, std::ptr::null_mut()) as usize, r2);
            assert!(committed_page(home));
            assert_eq!(*(home as *const u32), 0x9ABC_DEF0);
            assert!(!committed_page(r2));
            let mut back = [0u8; 4];
            assert!(read_physical_pages(&one, 0, &mut back));
            assert_eq!(u32::from_le_bytes(back), 0x9ABC_DEF0);
            MapTitlePhysicalPages(r0 as *mut c_void, 1, 0, 4, std::ptr::null_mut());
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            for r in [r0, r1, r2] {
                u.sparse_release(r);
            }
        }
    }

    /// `XMemMapPhysicalPagesScatter` with no page array unmaps each address, as page 0 does (a
    /// retail game unmaps so while it streams; refused, it stopped on its own `int3`).
    #[test]
    fn scatter_without_pages_unmaps() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let mut n = 2;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let r = mem::VirtualAlloc(std::ptr::null(), 2 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert!(r != 0);
            let addrs = [r as *mut c_void, (r + PHYS_PAGE) as *mut c_void];
            let mut ids = [pages[0] as u64, pages[1] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, ids.as_mut_ptr()), 1);
            (r as *mut u32).write_volatile(7);
            ((r + PHYS_PAGE) as *mut u32).write_volatile(9);
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, std::ptr::null_mut()), 1, "no page array: unmapped, not refused");
            for a in addrs {
                assert!(!crate::gpu::shader::readable(a as usize, 4), "{a:p} still mapped");
            }
            // the pages kept what was written: mapped again, they read it
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, ids.as_mut_ptr()), 1);
            assert_eq!(((r as *const u32).read_volatile(), ((r + PHYS_PAGE) as *const u32).read_volatile()), (7, 9));
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, std::ptr::null_mut()), 1);
            // no addresses: the same answer with and without a page array
            let mut none = [0u64; 1];
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 0, std::ptr::null_mut()), XMemMapPhysicalPagesScatter(addrs.as_ptr(), 0, none.as_mut_ptr()));
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    #[test]
    fn state_runs_keep_whole_page_reservation_checks_without_extra_queries() {
        unsafe {
            const N: usize = 8;
            let r = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert_ne!(r, 0);
            let mut run = StateRun::default();
            let before = VIRTUAL_QUERIES.with(|q| q.get());
            for i in 0..N {
                assert!(!committed_page_in(r + i * PHYS_PAGE, &mut run));
                assert!(reserved_page_in(r + i * PHYS_PAGE, &mut run));
            }
            assert_eq!(VIRTUAL_QUERIES.with(|q| q.get()) - before, 1, "the fork's deferral guard shares the reserved-run query");
            assert!(!mem::VirtualAlloc((r + 0x1000) as *const c_void, 0x1000, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            // A native state change requires a fresh run, as the map loop's own changes do.
            run = StateRun::default();
            assert!(!reserved_page_in(r, &mut run), "a reserved prefix is not a whole reserved physical page");
            assert!(!reserved_page_in(r + 0x1000, &mut run), "a committed subpage is not deferred");
            assert!(reserved_page_in(r + PHYS_PAGE, &mut run), "a later whole reserved page remains eligible");
            assert_ne!(mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE), 0);
            assert!(!reserved_page_in(usize::MAX, &mut StateRun::default()), "an invalid address never becomes deferred");
        }
    }

    /// Fresh pages mapped into a reservation in one call are left lazy with a few VirtualQuery
    /// calls, not two per page (13 s under Wine for a title's 98,304 pages); a committed page
    /// among them is still told from the reserved ones.
    #[test]
    fn mapping_fresh_pages_does_not_query_each_page() {
        const N: usize = 64;
        unsafe {
            let mut n = N;
            let mut pages = [0usize; N];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let home = page_home(pages[0]).unwrap();
            assert!(!committed_page(home), "the fork's CPU-only home starts reserved");
            let r = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let used = r + 3 * PHYS_PAGE;
            assert!(!mem::VirtualAlloc(used as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            *(used as *mut u32) = 0xABAB_ABAB;
            let before = VIRTUAL_QUERIES.with(|q| q.get());
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, r);
            let queries = VIRTUAL_QUERIES.with(|q| q.get()) - before;
            assert!(queries <= 8, "{queries} VirtualQuery calls for {N} pages");
            assert_eq!(*(used as *const u32), 0, "a committed page is told from the reserved ones: it is not left lazy");
            assert!(!committed_page(r + 2 * PHYS_PAGE) && !committed_page(r + 4 * PHYS_PAGE), "the others are left lazy");
            assert!(!committed_page(page_home(pages[N - 1]).unwrap()), "and their homes stay reserved");
            assert_eq!(*((r + 4 * PHYS_PAGE) as *const u32), 0);
            // back home: the pages left lazy are unmarked (a freed one stays marked, and the address
            // is another test's later)
            assert!(!MapTitlePhysicalPages(std::ptr::null_mut(), N, 0, 4, pages.as_mut_ptr()).is_null());
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), N, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// The fork keeps fresh homes in CPU-only reservations. These #556 tests check native
    /// queries and CPU bytes; sparse tile side effects remain covered by sparse.rs's tests.
    unsafe fn fresh_pages(n: usize) -> Vec<usize> {
        let (mut num, mut pages) = (n, vec![0usize; n]);
        assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut num, pages.as_mut_ptr()), 1);
        assert!(!committed_page(page_home(pages[0]).unwrap()));
        pages
    }

    /// Pages mapped, half of them written, then all mapped again in one call – at another
    /// address or the same one: the written ones keep their contents. The first pages of the
    /// call are never-used (nothing to move), the ones after them are moved: what a run of
    /// the pages asked about before says is no longer true (the home they moved to is committed
    /// by now, the address they left decommitted).
    unsafe fn remap_keeps_what_was_written(same_address: bool) {
        const N: usize = 8;
        let mut pages = fresh_pages(N);
        let r1 = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
        let r2 = if same_address { r1 } else { mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize };
        assert_eq!(MapTitlePhysicalPages(r1 as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, r1);
        for i in N / 2..N {
            *((r1 + i * PHYS_PAGE) as *mut u32) = 0x100 + i as u32;
        }
        assert_eq!(MapTitlePhysicalPages(r2 as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, r2);
        for i in 0..N {
            let want = if i >= N / 2 { 0x100 + i as u32 } else { 0 };
            assert_eq!(*((r2 + i * PHYS_PAGE) as *const u32), want, "page {i} (same address: {same_address})");
        }
        // back home: the pages left lazy are unmarked (a freed one stays marked, and the address
        // is another test's later)
        assert!(!MapTitlePhysicalPages(std::ptr::null_mut(), N, 0, 4, pages.as_mut_ptr()).is_null());
        assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), N, pages.as_mut_ptr()), 1);
        mem::VirtualFree(r1 as *mut c_void, 0, mem::MEM_RELEASE);
        if !same_address {
            mem::VirtualFree(r2 as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    #[test]
    fn pages_mapped_again_at_another_address_keep_what_was_written() {
        unsafe { remap_keeps_what_was_written(false) }
    }

    #[test]
    fn pages_mapped_again_at_the_same_address_keep_what_was_written() {
        unsafe { remap_keeps_what_was_written(true) }
    }

    /// Pages never used, mapped again (they are only unmarked: memory is as it was) cost no
    /// more queries than the first time.
    #[test]
    fn mapping_lazy_pages_again_does_not_query_each_page() {
        const N: usize = 64;
        unsafe {
            let mut pages = fresh_pages(N);
            let r1 = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let r2 = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert_eq!(MapTitlePhysicalPages(r1 as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, r1);
            for to in [r1, r2] {
                let before = VIRTUAL_QUERIES.with(|q| q.get());
                assert_eq!(MapTitlePhysicalPages(to as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, to);
                let queries = VIRTUAL_QUERIES.with(|q| q.get()) - before;
                assert!(queries <= 8, "{queries} VirtualQuery calls for {N} pages mapped again");
            }
            // back home: the pages left lazy are unmarked (a freed one stays marked, and the address
            // is another test's later)
            assert!(!MapTitlePhysicalPages(std::ptr::null_mut(), N, 0, 4, pages.as_mut_ptr()).is_null());
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), N, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r1 as *mut c_void, 0, mem::MEM_RELEASE);
            mem::VirtualFree(r2 as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// `XMemMapPhysicalPagesScatter` against a model of physical pages: random scatters of
    /// pages to addresses (page 0: unmap), writes through the addresses and maps of the whole
    /// allocation at its home. After each step every page reads what was last written to it
    /// wherever it is mapped, a page never written reads zeros, and an address with no page is
    /// not committed.
    #[test]
    fn scatter_matches_a_model_of_physical_pages() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            const PAGES: usize = 24;
            const SLOTS: usize = 32;
            for seed in 1..=4u64 {
                let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let mut next = move || {
                    x ^= x >> 12;
                    x ^= x << 25;
                    x ^= x >> 27;
                    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
                };
                let mut n = PAGES;
                let mut pages = [0usize; PAGES];
                assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
                let r = mem::VirtualAlloc(std::ptr::null(), SLOTS * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
                assert!(r != 0);
                let slot = |s: usize| r + s * PHYS_PAGE;
                // model: what each page holds (first and last dword), where it is (None: home)
                let mut value = [(0u32, 0u32); PAGES];
                let mut at: [Option<usize>; PAGES] = [None; PAGES];
                let mut held: [Option<usize>; SLOTS] = [None; SLOTS];
                let mut home: Option<usize> = None;
                let read = |a: usize| ((a as *const u32).read_volatile(), ((a + PHYS_PAGE - 4) as *const u32).read_volatile());
                for step in 0..600 {
                    match next() % 8 {
                        // a scatter of 1–4 (slot, page or none)
                        0..=3 => {
                            let k = 1 + (next() % 4) as usize;
                            let mut addrs = Vec::new();
                            let mut ids = Vec::new();
                            for _ in 0..k {
                                let s = (next() % SLOTS as u64) as usize;
                                let p = if next() % 5 == 0 { None } else { Some((next() % PAGES as u64) as usize) };
                                addrs.push(slot(s) as *mut c_void);
                                ids.push(p.map_or(0, |p| pages[p] as u64));
                                // model, in call order: what the slot held goes home; the page leaves
                                // where it was
                                if let Some(q) = held[s].take() {
                                    at[q] = None;
                                }
                                if let Some(p) = p {
                                    if let Some(old) = at[p].take() {
                                        held[old] = None;
                                    }
                                    at[p] = Some(s);
                                    held[s] = Some(p);
                                }
                            }
                            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), k, ids.as_mut_ptr()), 1, "seed {seed} step {step}");
                        }
                        // a write through a mapped slot
                        4..=6 => {
                            let s = (next() % SLOTS as u64) as usize;
                            if let Some(p) = held[s] {
                                let v = ((next() as u32) | 1, (next() as u32) | 1);
                                (slot(s) as *mut u32).write_volatile(v.0);
                                ((slot(s) + PHYS_PAGE - 4) as *mut u32).write_volatile(v.1);
                                value[p] = v;
                            }
                        }
                        // the whole allocation at its home: every page leaves its slot
                        _ => {
                            let h = MapTitlePhysicalPages(std::ptr::null_mut(), PAGES, 0, 4, pages.as_mut_ptr()) as usize;
                            assert!(h != 0);
                            home = Some(h);
                            at = [None; PAGES];
                            held = [None; SLOTS];
                            for (p, &v) in value.iter().enumerate() {
                                assert_eq!(read(h + p * PHYS_PAGE), v, "seed {seed} step {step}: page {p} at home");
                            }
                        }
                    }
                    for s in 0..SLOTS {
                        match held[s] {
                            Some(p) => assert_eq!(read(slot(s)), value[p], "seed {seed} step {step}: page {p} at slot {s}"),
                            None => assert!(!committed_page(slot(s)), "seed {seed} step {step}: slot {s} has no page but is committed"),
                        }
                    }
                }
                let _ = home;
                assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), PAGES, pages.as_mut_ptr()), 1);
                mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
            }
        }
    }

    /// A home a page went back to in a call, then decommitted in the same call because another
    /// page was mapped at that address and unmapped again, is not taken as committed: mapping
    /// its page away does not read decommitted memory.
    #[test]
    fn a_home_decommitted_in_the_same_call_is_asked_again() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let (p, q) = (pages[0], pages[1]);
            let h = page_home(p).unwrap();
            let r = mem::VirtualAlloc(std::ptr::null(), 2 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let (s, t) = (r as *mut c_void, (r + PHYS_PAGE) as *mut c_void);
            let mut one = [p as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&s, 1, one.as_mut_ptr()), 1);
            *(s as *mut u32) = 7;
            // one call: p home (h holds it), q at h, h unmapped (decommitted), p to t
            let at = [s, h as *mut c_void, h as *mut c_void, t];
            let mut ids = [0, q as u64, 0, p as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 4, ids.as_mut_ptr()), 1);
            // (t reads: committed, or committed on first use)
            let _ = (t as *const u32).read_volatile();
            assert!(!committed_page(s as usize));
            let mut none = [0u64, 0];
            let both = [s, t];
            assert_eq!(XMemMapPhysicalPagesScatter(both.as_ptr(), 2, none.as_mut_ptr()), 1);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    thread_local! {
        /// pages `unmap_away` copied back home on this thread
        pub(super) static HOME_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// A page mapped away that the title maps somewhere else goes there straight, with what it
    /// holds, and not by way of its home; the address it left is given up.
    #[test]
    fn a_page_away_moves_on_without_going_home() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let mut n = 1usize;
            let mut pages = [0usize; 1];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let r = mem::VirtualAlloc(std::ptr::null(), 2 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let (a, b) = (r as *mut c_void, (r + PHYS_PAGE) as *mut c_void);
            let mut one = [pages[0] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&a, 1, one.as_mut_ptr()), 1);
            *(a as *mut u32) = 7;
            *((r + PHYS_PAGE - 4) as *mut u32) = 8;
            let before = HOME_COPIES.with(|c| c.get());
            assert_eq!(XMemMapPhysicalPagesScatter(&b, 1, one.as_mut_ptr()), 1);
            assert_eq!(HOME_COPIES.with(|c| c.get()), before, "the page went by way of its home");
            assert_eq!(*(b as *const u32), 7);
            assert_eq!(*((r + 2 * PHYS_PAGE - 4) as *const u32), 8);
            assert!(!committed_page(r), "the address the page left stays committed");
            let mut zero = [0u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&b, 1, zero.as_mut_ptr()), 1);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// A move whose destination cannot be committed fails with the page still mapped away where
    /// it was: its bytes and both lookups are kept.
    #[test]
    fn a_page_whose_move_fails_stays_mapped_where_it_was() {
        unsafe {
            let old = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize;
            *(old as *mut u32) = 7;
            // an address with nothing reserved: committing there fails
            let at = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            mem::VirtualFree(at as *mut c_void, 0, mem::MEM_RELEASE);
            let page = 0x7E57_0000usize;
            let mut away = Away { at: Default::default(), page: Default::default(), gone: Vec::new() };
            away.at.insert(old, page);
            away.page.insert(page, old);
            assert!(move_away(&mut away, old, at, page, false).is_err());
            assert_eq!(away.page.get(&page), Some(&old), "the page lost where it is");
            assert_eq!(away.at.get(&old), Some(&page));
            assert_eq!(*(old as *const u32), 7);
            mem::VirtualFree(old as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// A page mapped at another page's home, then the whole allocation mapped back home: the
    /// first page goes home and leaves that address committed, as the second one's home.
    #[test]
    fn a_page_away_at_another_home_goes_home_with_it() {
        assert_eq!(crate::hle::d3d12x::backend(), crate::hle::d3d12x::Backend::Real, "initialize UMA before exercising physical pages");
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let (q, p) = (pages[0], pages[1]);
            let h = page_home(p).unwrap();
            let r = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let s = r as *mut c_void;
            let mut one = [p as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&s, 1, one.as_mut_ptr()), 1);
            *(s as *mut u32) = 7;
            let at = h as *mut c_void;
            let mut one = [q as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&at, 1, one.as_mut_ptr()), 1);
            *(h as *mut u32) = 9;
            let expected_home = page_home(q).unwrap();
            let home = MapTitlePhysicalPages(expected_home as *mut c_void, 2, 0, 4, pages.as_mut_ptr()) as usize;
            assert_eq!(home, expected_home);
            assert_eq!(*(home as *const u32), 9);
            assert_eq!(*(h as *const u32), 7);
            assert!(!physical_page_at(home) && !physical_page_at(h), "restored homes are not away mappings");
            assert!(!committed_page(r));
            let mut bytes = [0u8; 4];
            for (page, expected) in [(q, 9), (p, 7)] {
                assert!(read_physical_pages(&[page], 0, &mut bytes));
                assert_eq!(u32::from_le_bytes(bytes), expected);
            }
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// Timing (`--ignored --nocapture`; `XCL_SCATTER_PAGES=4096,8192`): pages written at their
    /// home, scattered from there into a reservation in one call, scattered again in reverse,
    /// and mapped back home.
    #[test]
    #[ignore]
    fn scatter_timing() {
        let sizes: Vec<usize> = std::env::var("XCL_SCATTER_PAGES").ok().map_or(vec![4096, 8192, 16384], |v| v.split(',').map(|s| s.parse().unwrap()).collect());
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        for n in sizes {
            unsafe {
                let mut cnt = n;
                let mut pages = vec![0usize; n];
                assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut cnt, pages.as_mut_ptr()), 1);
                let h = MapTitlePhysicalPages(std::ptr::null_mut(), n, 0, 4, pages.as_mut_ptr()) as usize;
                assert!(h != 0);
                for i in 0..n {
                    *((h + i * PHYS_PAGE) as *mut u32) = i as u32 + 1;
                }
                let r = mem::VirtualAlloc(std::ptr::null(), n * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
                let addrs: Vec<*mut c_void> = (0..n).map(|i| (r + i * PHYS_PAGE) as *mut c_void).collect();
                let mut ids: Vec<u64> = pages.iter().map(|&p| p as u64).collect();
                let t = std::time::Instant::now();
                assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), n, ids.as_mut_ptr()), 1);
                let from_home = ms(t);
                ids.reverse();
                let t = std::time::Instant::now();
                assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), n, ids.as_mut_ptr()), 1);
                let reverse = ms(t);
                for i in (0..n).step_by(97) {
                    assert_eq!(*((r + i * PHYS_PAGE) as *const u32), (n - i) as u32);
                }
                let t = std::time::Instant::now();
                assert_eq!(MapTitlePhysicalPages(std::ptr::null_mut(), n, 0, 4, pages.as_mut_ptr()) as usize, h);
                let back = ms(t);
                for i in (0..n).step_by(97) {
                    assert_eq!(*((h + i * PHYS_PAGE) as *const u32), i as u32 + 1);
                }
                eprintln!("scatter_timing pages={n} from_home={from_home:.1}ms reverse={reverse:.1}ms back_home={back:.1}ms");
                FreeTitlePhysicalPages(std::ptr::null_mut(), n, pages.as_mut_ptr());
                mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
            }
        }
    }

    #[test]
    fn title_memory_status_reports_budget() {
        let mut s: TitleMemoryStatusT = unsafe { std::mem::zeroed() };
        s.dw_length = std::mem::size_of::<TitleMemoryStatusT>() as u32;
        assert_eq!(std::mem::size_of::<TitleMemoryStatusT>(), 80);
        assert_eq!(unsafe { TitleMemoryStatus(&mut s) }, 1);
        assert_eq!(s.total, 13_824 * 1024 * 1024);
        assert!(s.avail <= s.total);
        // the process's own memory counts as used (Wine leaves PrivateUsage 0)
        assert!(process_private_bytes() > 0 && s.avail < s.total);
        s.dw_length = 8;
        assert_eq!(unsafe { TitleMemoryStatus(&mut s) }, 0);
    }

    #[test]
    fn xmem_is_tools_mem_available_returns_false() {
        assert_eq!(unsafe { XMemIsToolsMemAvailable() }, 0);
    }

    #[test]
    fn get_proc_address_and_load_library_ex_reject_null_name() {
        unsafe {
            let k32 = ll::GetModuleHandleW(crate::util::wide("kernel32.dll").as_ptr()) as usize;
            assert_ne!(k32, 0);
            fnd::SetLastError(0);
            assert!(ll::GetProcAddress(k32 as _, std::ptr::null()).is_none());
            let win32_err = fnd::GetLastError();

            fnd::SetLastError(0);
            assert_eq!(GetProcAddress(k32, std::ptr::null()), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);
            // Windows' own answer; Wine's GetProcAddress says ERROR_PROC_NOT_FOUND instead
            if !crate::util::under_wine() {
                assert_eq!(win32_err, fnd::ERROR_INVALID_PARAMETER);
            }

            // Non-zero ordinal lookup still resolves when valid (ordinal #1 on kernel32).
            assert_ne!(GetProcAddress(k32, 1usize as *const u8), 0);

            fnd::SetLastError(0);
            assert_eq!(LoadLibraryExW(std::ptr::null(), std::ptr::null_mut(), LOAD_AS_DATA), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

            fnd::SetLastError(0);
            assert_eq!(LoadLibraryExA(std::ptr::null(), std::ptr::null_mut(), LOAD_AS_DATA), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);
        }
    }

    #[test]
    fn get_thread_name_sizing_truncation_and_errors() {
        std::thread::spawn(|| unsafe {
            let cur = thr::GetCurrentThread();

            // Empty thread name: capacity 0 fails with ERROR_INSUFFICIENT_BUFFER, capacity 1 writes NUL.
            let empty = crate::util::wide("");
            assert_eq!(SetThreadName(cur, empty.as_ptr()), 1);
            let mut ret = 99usize;
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(cur, std::ptr::null_mut(), 0, &mut ret), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(ret, 0);
            let mut nul_only = [0xBEEFu16; 1];
            assert_eq!(GetThreadName(cur, nul_only.as_mut_ptr(), 1, &mut ret), 1);
            assert_eq!((ret, nul_only[0]), (0, 0));

            let want = crate::util::wide("xcl-worker-1");
            assert_eq!(SetThreadName(cur, want.as_ptr()), 1);

            // Null buffer with non-zero length -> ERROR_INVALID_PARAMETER.
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(cur, std::ptr::null_mut(), 4, &mut ret), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

            // Size query (null buffer, len == 0) -> FALSE + ERROR_INSUFFICIENT_BUFFER with ret_len = 12.
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(cur, std::ptr::null_mut(), 0, &mut ret), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(ret, 12);

            // Buffer equal to char count (no room for NUL terminator) -> FALSE + ERROR_INSUFFICIENT_BUFFER.
            let mut exact = [0xBEEFu16; 12];
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(cur, exact.as_mut_ptr(), exact.len(), &mut ret), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(ret, 12);

            // Buffer with room for chars + NUL -> TRUE and exact match (both with and without ret_len).
            let mut full = [0u16; 13];
            assert_eq!(GetThreadName(cur, full.as_mut_ptr(), full.len(), &mut ret), 1);
            assert_eq!(ret, 12);
            assert_eq!(&full[..], &want[..]);
            full.fill(0);
            assert_eq!(GetThreadName(cur, full.as_mut_ptr(), full.len(), std::ptr::null_mut()), 1);
            assert_eq!(&full[..], &want[..]);

            // Invalid thread handle -> FALSE + Win32 error set.
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(std::ptr::null_mut(), full.as_mut_ptr(), full.len(), &mut ret), 0);
            assert_ne!(fnd::GetLastError(), 0);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn xmem_virtual_query_excludes_guard_pages() {
        unsafe {
            let proc_ = thr::GetCurrentProcess();
            let page = mem::VirtualAlloc(std::ptr::null(), 0x1000, mem::MEM_COMMIT | mem::MEM_RESERVE, mem::PAGE_READWRITE | mem::PAGE_GUARD);
            assert!(!page.is_null());
            let mut info = [0u64; 8];
            assert_eq!(XMemVirtualQuery(proc_, page, info.as_mut_ptr() as _), 1);
            assert_eq!((info[6] as u32) & 3, 0, "PAGE_GUARD page must not be marked CPU-readable");
            mem::VirtualFree(page, 0, mem::MEM_RELEASE);
        }
    }

    #[test]
    fn xmem_and_physical_page_overflows_fail_cleanly() {
        unsafe {
            // CPU heap allocation overflow -> null + ERROR_NOT_ENOUGH_MEMORY.
            fnd::SetLastError(0);
            assert!(XMemAllocDefault(usize::MAX, 0).is_null());
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);

            // Graphics layout rounding overflow (usize::MAX) and largest 64 KiB multiple
            // (which previously overflowed `size + GRAIN - 1` inside Uma::sparse_alloc / alloc_where).
            let gfx_attrs = 1u64 << 29;
            assert!(xmem_graphics_layout(usize::MAX, XAllocAttrs::decode(gfx_attrs)).is_none());
            fnd::SetLastError(0);
            assert!(XMemAllocDefault(usize::MAX, gfx_attrs).is_null());
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);

            let max_aligned = usize::MAX & !(crate::gpu::uma::GRAIN - 1);
            assert!(XMemAllocDefault(max_aligned, gfx_attrs).is_null());

            // Physical page byte-count overflow -> FALSE + ERROR_NOT_ENOUGH_MEMORY.
            let mut huge = usize::MAX;
            let mut page = 0usize;
            fnd::SetLastError(0);
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut huge, &mut page), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);
        }
    }

    /// A no-access graphics reservation of 64 KiB pages is the GPU view of the CPU-readable one
    /// of its size before it: textures there read the CPU view (`title_textures`).
    #[test]
    fn gpu_view_reservations_name_their_cpu_view() {
        // (addresses far above anything Windows hands out: only the bookkeeping is tested)
        let (a, b, size) = (0x7000_0000_0000usize, 0x7006_0000_0000usize, 0x6_0000_0000usize);
        let other = 0x7010_0000_0000usize;
        mirrors::reserved(a, size, 0x2040_2000, mem::PAGE_READWRITE);
        mirrors::reserved(other, 0x1000_0000, 0x2040_2000, mem::PAGE_READWRITE); // another size
        assert_eq!(mirrors::source(b + 0x1234_0000), None, "no GPU view yet");
        mirrors::reserved(b, size, 0x2040_2000, 0x801);
        assert_eq!(mirrors::source(b), Some(a));
        assert_eq!(mirrors::source(b + 0x4524_0000), Some(a + 0x4524_0000));
        assert_eq!(mirrors::source(b + size), None, "past its end");
        assert_eq!(mirrors::source(a + 0x10), None, "the CPU view is no GPU view");
        // committed memory or other protections are no GPU views
        let c = 0x7020_0000_0000usize;
        mirrors::reserved(c, size, 0x2040_3000, 0x801);
        assert_eq!(mirrors::source(c), None);
        mirrors::released(a);
        assert_eq!(mirrors::source(b), None, "released with its CPU view");
        mirrors::released(other);
    }

    /// GPU-visible title memory that reaches the plain-memory fallback (64 KiB pages, large
    /// pages, a graphics protection) is write-watched: a texture placed there sees the CPU's
    /// writes. The fallback is called directly: through `VirtualAlloc` these sizes land in a
    /// unified-memory window whenever one has room (once any test has made the windows).
    #[test]
    fn gpu_visible_fallback_memory_is_write_watched() {
        let null = std::ptr::null();
        // (MEM_64K_PAGES | MEM_RESERVE; MEM_LARGE_PAGES | MEM_RESERVE | MEM_COMMIT with a
        // PAGE_GRAPHICS_* protection; MEM_GRAPHICS | MEM_RESERVE | MEM_COMMIT)
        let gpu = [(0x2040_2000u32, mem::PAGE_READWRITE), (0x2000_3000, 0x2404), (MEM_GRAPHICS | 0x3000, mem::PAGE_READWRITE)];
        for &(t, p) in &gpu {
            assert!(gpu_visible_fallback(null, t, p), "flags {t:#x} protect {p:#x}");
        }
        // plain memory stays plain: CPU-only reservations, commits at a fixed address
        assert!(!gpu_visible_fallback(null, 0x3000, mem::PAGE_READWRITE));
        assert!(!gpu_visible_fallback(null, 0x1000, 0x2404), "a commit without a reservation");
        assert!(!gpu_visible_fallback(0x1_0000 as *const c_void, 0x2040_2000, mem::PAGE_READWRITE), "a fixed address");
        unsafe {
            for (t, p) in gpu {
                let size = 0x40_0000;
                let q = reserve_watched(size, t, p);
                assert!(!q.is_null(), "flags {t:#x}");
                let a = q as usize;
                assert!(in_graphics_reservation(a), "flags {t:#x}: a fallback allocation is watched");
                if t & 0x1000 == 0 {
                    assert!(!VirtualAlloc((a + 0x2_0000) as *const c_void, 0x1_0000, 0x1000, mem::PAGE_READWRITE).is_null());
                }
                let mut buf = [std::ptr::null_mut::<c_void>(); 16];
                let (mut n, mut gran) = (buf.len(), 0u32);
                assert_eq!(mem::GetWriteWatch(1, q, size, buf.as_mut_ptr(), &mut n, &mut gran), 0);
                ((a + 0x2_1234) as *mut u8).write_volatile(7);
                n = buf.len();
                assert_eq!(mem::GetWriteWatch(1, q, size, buf.as_mut_ptr(), &mut n, &mut gran), 0);
                assert_eq!(&buf[..n], &[(a + 0x2_1000) as *mut c_void], "flags {t:#x}: the written page is reported");
                assert_eq!(VirtualFree(q, 0, mem::MEM_RELEASE), 1);
                assert!(!in_graphics_reservation(a));
            }
        }
    }

    #[test]
    fn virtual_free_ex_and_sparse_validation() {
        unsafe {
            let cur = thr::GetCurrentProcess();

            // Failed MEM_RELEASE (non-zero size) must keep a tracked graphics reservation registered,
            // and VirtualFreeEx(GetCurrentProcess(), ...) must release & untrack it.
            let host = mem::VirtualAlloc(std::ptr::null(), 0x10000, mem::MEM_RESERVE, mem::PAGE_NOACCESS);
            assert!(!host.is_null());
            GRAPHICS_RESERVATIONS.lock().unwrap().insert(host as usize, 0x10000);
            assert!(in_graphics_reservation(host as usize));
            assert_eq!(VirtualFree(host, 0x1000, mem::MEM_RELEASE), 0);
            assert!(in_graphics_reservation(host as usize));
            assert_eq!(VirtualFreeEx(cur, host, 0, mem::MEM_RELEASE), 1);
            assert!(!in_graphics_reservation(host as usize));

            // Sparse UMA reservation: invalid flag combos / non-zero MEM_RELEASE fail without freeing;
            // VirtualFreeEx(GetCurrentProcess(), ...) releases cleanly.
            let sp = VirtualAlloc(std::ptr::null(), 0x10000, mem::MEM_RESERVE | MEM_GRAPHICS, mem::PAGE_READWRITE);
            if !sp.is_null() && crate::gpu::uma::get().is_some_and(|u| u.is_sparse(sp as usize)) {
                fnd::SetLastError(0);
                assert_eq!(VirtualFree(sp, 0x1000, mem::MEM_RELEASE), 0);
                assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

                fnd::SetLastError(0);
                assert_eq!(VirtualFree(sp, 0, mem::MEM_RELEASE | mem::MEM_DECOMMIT), 0);
                assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

                fnd::SetLastError(0);
                assert_eq!(VirtualFree(sp, 0, 0), 0);
                assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

                assert_eq!(VirtualFreeEx(cur, sp, 0, mem::MEM_RELEASE), 1);
                assert!(crate::gpu::uma::get().unwrap().sparse_reservation(sp as usize).is_none());
            } else if !sp.is_null() {
                assert_eq!(VirtualFreeEx(cur, sp, 0, mem::MEM_RELEASE), 1);
            }
        }
    }

    #[test]
    fn get_module_handle_matches_load_library_for_console_aliases_and_stubs() {
        unsafe {
            // Real Windows system DLL keeps its genuine Windows HMODULE.
            let real_k32 = ll::GetModuleHandleW(crate::util::wide("kernel32.dll").as_ptr()) as usize;
            assert_ne!(real_k32, 0);
            assert_eq!(GetModuleHandleW(crate::util::wide("kernel32.dll").as_ptr()), real_k32);
            assert_eq!(GetModuleHandleA(c"kernel32.dll".as_ptr() as *const u8), real_k32);

            // Console-only kernelx aliases return the kernelx HLE pseudo handle (matching LoadLibrary).
            let kx = modules::hle_handle(crate::hle::KERNELX);
            for dll in ["advapi32legacy.dll", "kernel32legacy.dll", "minappmodelx.dll"] {
                let w = crate::util::wide(dll);
                let h_load = LoadLibraryW(w.as_ptr());
                assert_eq!(h_load, kx, "LoadLibraryW({dll})");
                assert_eq!(GetModuleHandleW(w.as_ptr()), kx, "GetModuleHandleW({dll})");

                let c = std::ffi::CString::new(dll).unwrap();
                assert_eq!(GetModuleHandleA(c.as_ptr() as *const u8), kx, "GetModuleHandleA({dll})");

                let mut out = 0usize;
                assert_eq!(GetModuleHandleExW(0, w.as_ptr(), &mut out), 1, "GetModuleHandleExW({dll})");
                assert_eq!(out, kx);
            }

            // Unknown DLL still returns NULL + ERROR_MOD_NOT_FOUND.
            fnd::SetLastError(0);
            assert_eq!(GetModuleHandleW(crate::util::wide("no_such_optional_dll_xyz.dll").as_ptr()), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_MOD_NOT_FOUND);
        }

        // @stub DLLs: unloaded probe returns 0; once loaded (hle_handle created), returns that handle.
        let mut cfg = crate::config::Config::default();
        cfg.aliases.insert("unit_stub_probe.dll".into(), "@stub".into());
        let reg = crate::hle::build(&cfg);
        assert_eq!(module_handle_in(&reg, "unit_stub_probe.dll"), 0);
        let stub_h = modules::hle_handle("unit_stub_probe.dll");
        assert_ne!(stub_h, 0);
        assert_eq!(module_handle_in(&reg, "unit_stub_probe.dll"), stub_h);
    }


    #[test]
    fn affinity_masks_outside_the_process_are_fitted() {
        // a mask from the PC's processors is the title's choice
        assert_eq!(fit_affinity_mask(0x3F, 0xFF), 0x3F);
        assert_eq!(fit_affinity_mask(0, 0xF), 0);
        // the console's 7 cores on a 4-processor PC: the ones there are
        assert_eq!(fit_affinity_mask(0x7F, 0xF), 0xF);
        // only processors the PC lacks: dealt onto the PC's by index
        assert_eq!(fit_affinity_mask(1 << 6, 0xF), 1 << 2);
        assert_eq!(fit_affinity_mask(0x3000, 0xF), 0x3);
        assert_eq!(fit_affinity_mask(1 << 20, 0b1010_1010), 0b10);
    }


    #[test]
    fn set_thread_affinity_mask_takes_console_masks() {
        let allowed = process_affinity();
        if allowed == 0 || allowed == usize::MAX {
            println!("SKIPPED set_thread_affinity_mask_takes_console_masks: every processor bit is taken");
            return;
        }
        let missing = !allowed & allowed.wrapping_add(1);
        let first = allowed & allowed.wrapping_neg();
        std::thread::spawn(move || unsafe {
            let me = thr::GetCurrentThread();
            assert_ne!(SetThreadAffinityMask(me, missing), 0, "{missing:#x} in a {allowed:#x} process");
            assert_eq!(thr::SetThreadAffinityMask(me, allowed), fit_affinity_mask(missing, allowed));
            let mut a: GROUP_AFFINITY = std::mem::zeroed();
            a.Mask = missing | first;
            assert_eq!(SetThreadGroupAffinity(me, &a, std::ptr::null_mut()), 1);
        })
        .join()
        .unwrap();
    }


    #[test]
    fn tracing_keeps_the_last_error() {
        unsafe {
            fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
            keep_last_error(|| assert!(std::env::var_os("XCL_TEST_UNSET_8C1F").is_none()));
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);
            fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
            trace_mem("test", 0, 0x1_0000, 0, 0, 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);
        }
    }

    #[test]
    fn virtual_query_fastpath_covers_committed_coherent_window() {
        let Some(u) = crate::gpu::uma::get() else { return };
        let Some(k) = u.committed_window() else { return };
        let (base, len) = u.window(k);
        assert!(len >= 0x2000);
        let addr = (base + 0x1234) as *const c_void;
        unsafe {
            let mut mbi: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let n = VirtualQuery(addr, &mut mbi, std::mem::size_of_val(&mbi));
            assert_eq!(n, std::mem::size_of::<mem::MEMORY_BASIC_INFORMATION>());
            assert_eq!(mbi.BaseAddress as usize, base + 0x1000);
            assert_eq!(mbi.AllocationBase as usize, base);
            assert_eq!(mbi.RegionSize, len - 0x1000);
            assert_eq!(mbi.State, mem::MEM_COMMIT);
            assert_eq!(mbi.Protect, mem::PAGE_READWRITE);

            let mut mbi2: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let n2 = VirtualQueryEx(thr::GetCurrentProcess(), addr, &mut mbi2, std::mem::size_of_val(&mbi2));
            assert_eq!(n2, n);
            assert_eq!(mbi2.RegionSize, mbi.RegionSize);

            let mut xinfo = [0u8; 64];
            assert_eq!(XMemVirtualQuery(thr::GetCurrentProcess(), addr, xinfo.as_mut_ptr() as *mut c_void), 1);
            assert_eq!(u32::from_ne_bytes(xinfo[48..52].try_into().unwrap()), 1);
            assert!(committed_range(base + 0x1000, 0x1000));
            assert!(committed_page(base + 0x1000));
        }
    }

    #[test]
    fn sparse_64k_reservation_registers_gpu_view_mirror_and_releases_cleanly() {
        unsafe {
            // Reserve a 64 KiB-page CPU view (lands in sparse UMA when space is available) and
            // pair it with a PAGE_NOACCESS | PAGE_GRAPHICS_NOACCESS GPU view of the same size.
            let size = 4 << 20;
            let cpu = VirtualAlloc(std::ptr::null(), size, mem::MEM_RESERVE | MEM_64K_PAGES, mem::PAGE_READWRITE);
            assert!(!cpu.is_null());
            let gpu = VirtualAlloc(std::ptr::null(), size, mem::MEM_RESERVE | MEM_64K_PAGES, mem::PAGE_NOACCESS | 0x800);
            assert!(!gpu.is_null());
            assert_eq!(
                mirrors::source(gpu as usize + 0x2_0000),
                Some(cpu as usize + 0x2_0000),
                "GPU view must resolve to CPU view even when CPU view lives in sparse UMA"
            );
            // GPU-view alias must not consume sparse UMA VA.
            if let Some(u) = crate::gpu::uma::get() {
                assert!(!u.is_sparse(gpu as usize), "CPU-inaccessible GPU view should stay outside sparse UMA");
            }
            assert_eq!(VirtualFree(gpu, 0, mem::MEM_RELEASE), 1);
            assert_eq!(mirrors::source(gpu as usize + 0x2_0000), None);
            assert_eq!(VirtualFree(cpu, 0, mem::MEM_RELEASE), 1);
        }
    }
}
