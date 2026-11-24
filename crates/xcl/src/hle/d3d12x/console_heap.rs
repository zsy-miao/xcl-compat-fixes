//! Console descriptor heaps (CBV/SRV/UAV and samplers).
//!
//! On the console a descriptor is GPU-readable memory: 32 bytes per CBV/SRV/UAV slot (T# /
//! V#), 16 per sampler (S#), and a descriptor table in a root argument is a 32-bit GPU address
//! (shaders load it with the high half 0 – amd_fidelityfx_upscaler_xbox.dll, HANDOFF "FSR4").
//! XCL gives console code heaps whose slots live in unified memory (so the table addresses
//! console code computes from the heap start work as they are in the recompiler's `mem`
//! model) and keeps PC descriptors beside each: view slot k → an SRV and, next to it, a UAV;
//! sampler slot k → PC sampler k. A slot holds the descriptor in the XCL shader ABI
//! (`desc::xcl_image_words`: T# word 0 = the SRV's index in the shader-visible PC heap;
//! `desc::xcl_sampler_words`: S# word 3 [11:0] = k; V# with its unified-memory address).
//!
//! View descriptors are values on the console: titles create them in a CPU-only heap and copy
//! them into the shader-visible one as memory (the console header's inline
//! CopyDescriptorsSimple; XDefiant copies every texture that way), and a descriptor table may
//! be in any heap. So all console view heaps share one shader-visible PC heap (`Global`):
//! each gets a range there, T# word 0 is an index into it – a T# copied anywhere still names
//! its PC descriptor – and that heap is what lists bind. The title's own PC heap stays as a
//! CPU-only mirror (where views are made, copies come from and UAV clears take their CPU
//! handle). A heap gets its place in the shared heap a page of `PAGE_SLOTS` slots at a time,
//! when a slot of that page is first used: titles make large heaps and fill few slots
//! (Minecraft: eight view heaps of 73 727 slots – 147 454 PC descriptors each, more in all
//! than the 1 000 000 an NVIDIA shared heap holds – with a few hundred views in them). A heap
//! left out of the shared heap would be lost: lists bind only the shared heap for views.
//! Without the shared heap (it could not be made) a heap keeps its own PC heap: SRV at 2k,
//! UAV at 2k + 1.

use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// bytes of a CBV/SRV/UAV slot (T# / V#) and of a sampler slot (S#)
pub const SLOT: u32 = 32;
pub const SAMPLER_SLOT: u32 = 16;

/// Whose memory a heap's slots are in.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Memory {
    /// a sparse VRAM allocation in window 0
    Sparse,
    /// committed system RAM
    Uma,
    /// the title's (its allocation hook gave it: `kernelx::runtime_alloc`)
    Title,
}

struct Heap {
    obj: usize,
    shadow: usize,
    n: u32,
    pc_cpu: u64,
    /// 0: not shader visible
    pc_gpu: u64,
    inc: u32,
    sampler: bool,
    /// views in the shared heap: the index there of each page's first SRV (`UNMAPPED`: the
    /// page has no place yet); empty: the heap's views are not in the shared heap
    pages: Vec<u32>,
    /// shader visible as the title made it
    visible: bool,
    memory: Memory,
}

impl Heap {
    fn slot_bytes(&self) -> u32 {
        if self.sampler {
            SAMPLER_SLOT
        } else {
            SLOT
        }
    }
}

static HEAPS: Mutex<Vec<Heap>> = Mutex::new(Vec::new());

/// The shader-visible PC heap of all console view heaps.
struct Global {
    dev: ID3D12Device,
    heap: ID3D12DescriptorHeap,
    cpu: u64,
    gpu: u64,
    inc: u32,
    cap: u32,
    next: u32,
    /// released ranges (first, count)
    free: Vec<(u32, u32)>,
}

// D3D12 descriptor heaps and devices are free-threaded
unsafe impl Send for Global {}

static GLOBAL: Mutex<Option<Global>> = Mutex::new(None);
static GLOBAL_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// descriptors of the shared heap at most (the GPU may take more: OPTIONS19)
const GLOBAL_MAX: u32 = 1 << 21;

/// view slots per page of a heap's place in the shared heap (2 PC descriptors each)
pub const PAGE_SLOTS: u32 = 256;
/// a page with no place in the shared heap yet
const UNMAPPED: u32 = u32::MAX;

/// PC descriptors of page `p` of a heap of `n` slots: 2 per slot, and the last page only as
/// long as the heap (a heap of a few slots takes a few descriptors, not a whole page's).
fn page_descriptors(n: u32, p: u32) -> u32 {
    2 * n.saturating_sub(p * PAGE_SLOTS).clamp(1, PAGE_SLOTS)
}

/// A first-fit range of `count` in `free` (released ranges), else at `next` below `cap`.
fn take_range(free: &mut Vec<(u32, u32)>, next: &mut u32, cap: u32, count: u32) -> Option<u32> {
    if let Some(i) = free.iter().position(|&(_, n)| n >= count) {
        let (f, n) = free[i];
        if n == count {
            free.swap_remove(i);
        } else {
            free[i] = (f + count, n - count);
        }
        return Some(f);
    }
    if cap - *next < count {
        return None;
    }
    *next += count;
    Some(*next - count)
}

/// The shared shader-visible heap, made on first use. False: there is none (it could not be
/// made) – view heaps then keep their own.
pub unsafe fn shared_heap(dev: &ID3D12Device) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = GLOBAL.lock().unwrap();
    if g.is_none() {
        if GLOBAL_FAILED.load(Relaxed) {
            return false;
        }
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
        let max = if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &mut o as *mut _ as *mut std::ffi::c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.MaxViewDescriptorHeapSize > 0 {
            o.MaxViewDescriptorHeapSize
        } else {
            1_000_000
        };
        let cap = max.min(GLOBAL_MAX);
        let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: cap, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
        let heap: ID3D12DescriptorHeap = match dev.CreateDescriptorHeap(&d) {
            Ok(h) => h,
            Err(e) => {
                GLOBAL_FAILED.store(true, Relaxed);
                crate::xlog_warn!("console descriptor heaps: no shared shader-visible heap of {cap} ({e}) – each keeps its own");
                return false;
            }
        };
        crate::xlog_info!("console descriptor heaps: one shader-visible PC heap of {cap} view descriptors for all of them");
        *g = Some(Global {
            dev: dev.clone(),
            cpu: heap.GetCPUDescriptorHandleForHeapStart().ptr as u64,
            gpu: heap.GetGPUDescriptorHandleForHeapStart().ptr,
            inc: dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            heap,
            cap,
            // the memory windows' raw UAVs (`set_window_views`) come first
            next: crate::gpu::recomp::MEM_HEAP_BASE + crate::gpu::uma::MAX_WINDOWS as u32,
            free: Vec::new(),
        });
    }
    true
}

static WINDOW_VIEWS_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the memory windows could not be put in the shared heap (`set_window_views`):
/// translated shaders then switch over root UAVs (`pso::heap_windows`).
pub fn window_views_failed() -> bool {
    WINDOW_VIEWS_FAILED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The memory windows' buffers `(resource, bytes)` as raw UAVs at `MEM_HEAP_BASE + k` of the
/// shared heap, made now if need be (`Options::heap_windows`: shaders take window k from there).
/// A window without a buffer gets window 0's. False (and the shaders keep the switch): no
/// shared heap.
pub unsafe fn set_window_views(dev: &ID3D12Device, windows: &[(usize, usize)]) -> bool {
    let ok = !windows.is_empty() && windows[0].0 != 0 && shared_heap(dev) && {
        let (device, cpu, inc) = {
            let g = GLOBAL.lock().unwrap();
            let s = g.as_ref().unwrap();
            (s.dev.clone(), s.cpu, s.inc)
        };
        // The device may be adopted: its UAV hook re-enters descriptor bookkeeping.
        for k in 0..crate::gpu::uma::MAX_WINDOWS {
            let (raw, bytes) = windows.get(k).copied().filter(|w| w.0 != 0).unwrap_or(windows[0]);
            let raw = raw as *mut std::ffi::c_void;
            let Some(r) = ID3D12Resource::from_raw_borrowed(&raw) else { continue };
            let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + (crate::gpu::recomp::MEM_HEAP_BASE as u64 + k as u64) * inc as u64) as usize };
            device.CreateUnorderedAccessView(r, None::<&ID3D12Resource>, Some(&crate::gpu::recomp::runner::raw_window_uav(bytes as u64)), at);
        }
        true
    };
    if !ok {
        WINDOW_VIEWS_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
        crate::xlog_warn!("memory windows: not in the shared descriptor heap – translated shaders switch over the windows' root UAVs");
    }
    ok
}

/// `count` descriptors of the shared heap: the index of the first. None: no shared heap, or
/// no room left in it.
fn reserve_global(count: u32) -> Option<u32> {
    let mut g = GLOBAL.lock().unwrap();
    let s = g.as_mut()?;
    let (cap, mut next) = (s.cap, s.next);
    let r = take_range(&mut s.free, &mut next, cap, count);
    s.next = next;
    if r.is_none() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            crate::xlog_warn!("console descriptor heaps: the shared heap ({cap}) is full – shaders do not see the views of slots past it");
        }
    }
    r
}

/// The place in the shared heap of the page slot `k` of `h` is in – (index of the page's
/// first SRV there, the page's first slot) – given now if the page has none yet. None: the
/// heap is not in the shared heap, or that is full.
fn page_of(h: &mut Heap, k: u32) -> Option<(u32, u32)> {
    let p = (k / PAGE_SLOTS) as usize;
    let mut first = *h.pages.get(p)?;
    if first == UNMAPPED {
        first = reserve_global(page_descriptors(h.n, p as u32))?;
        h.pages[p] = first;
    }
    Some((first, p as u32 * PAGE_SLOTS))
}

/// The shared range starting at `first` (`count` descriptors) is free again.
pub fn release_global(first: u32, count: u32) {
    if let Some(s) = GLOBAL.lock().unwrap().as_mut() {
        s.free.push((first, count));
    }
}

/// The shader-visible PC heap lists bind for view descriptors (None: none shared).
pub fn global_heap() -> Option<ID3D12DescriptorHeap> {
    GLOBAL.lock().unwrap().as_ref().map(|s| s.heap.clone())
}

/// Whether `obj` (a title's view heap) has its descriptors in the shared heap.
pub fn in_global(obj: usize) -> bool {
    HEAPS.lock().unwrap().iter().any(|h| h.obj == obj && !h.pages.is_empty())
}

/// Memory for a heap of `bytes` from the title's allocation hook – where the title expects the
/// heap then. Only what descriptor tables can address (window 0) and is committed; anything else
/// goes back to the title.
unsafe fn title_memory(u: &crate::gpu::uma::Uma, bytes: usize) -> Option<usize> {
    use crate::hle::kernelx::{runtime_alloc, runtime_free, D3D_DESCRIPTOR_HEAP};
    let p = runtime_alloc(bytes, D3D_DESCRIPTOR_HEAP)?;
    let readable = crate::gpu::shader::readable(p, bytes);
    if readable && u.window_of(p) == Some(0) && u.window_of(p + bytes - 1) == Some(0) {
        return Some(p);
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
        crate::xlog_warn!("console descriptor heap: the title's memory {p:#x} ({bytes:#x} bytes) not used (window {:?}, committed {readable})", u.window_of(p));
    }
    runtime_free(p, D3D_DESCRIPTOR_HEAP);
    None
}

/// `obj` is a PC heap from `pc_cpu` / `pc_gpu` (increment `inc`; `pc_gpu` 0 when not shader
/// visible) with 2·`n` view descriptors or `n` samplers: give it `n` console slots in
/// unified memory. `shared`: its views go to the shared heap (`shared_heap`), a page at a
/// time as slots are used; `visible`: what the title asked for.
pub fn register(obj: usize, n: u32, pc_cpu: u64, pc_gpu: u64, inc: u32, sampler: bool, shared: bool, visible: bool) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let bytes = n.max(1) * if sampler { SAMPLER_SLOT } else { SLOT };
    // Descriptor tables are 32-bit console addresses (high dword 0): must live in window 0.
    // On the sparse-window path, `sparse::alloc_in(0, ...)` commits fresh zeroed pages so we
    // avoid touching unused pages with `write_bytes` (which would mark them dirty for write-watch).
    let (p, memory) = if let Some(p) = unsafe { title_memory(u, bytes as usize) } {
        (p as *mut u8, Memory::Title)
    } else if u.is_sparse_window(0) {
        let Some((addr, _)) = (unsafe { super::sparse::alloc_in(0, bytes as usize, 256) }) else {
            return false;
        };
        (addr as *mut u8, Memory::Sparse)
    } else {
        let Some(p) = u.alloc_in(0, bytes as usize, 256) else { return false };
        unsafe { std::ptr::write_bytes(p, 0, bytes as usize) };
        (p, Memory::Uma)
    };
    {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
            crate::xlog_info!(
                "console descriptor heap {obj:#x}: {n} {} slots at {} {:#x}, {}",
                if sampler { "sampler" } else { "view" },
                match memory {
                    Memory::Sparse => "sparse VRAM",
                    Memory::Uma => "UMA",
                    Memory::Title => "the title's memory",
                },
                p as usize - u.base(),
                if visible { "shader visible" } else { "CPU only" }
            );
        }
    }
    let pages = if shared && !sampler && GLOBAL.lock().unwrap().is_some() { vec![UNMAPPED; n.div_ceil(PAGE_SLOTS).max(1) as usize] } else { Vec::new() };
    HEAPS.lock().unwrap().push(Heap { obj, shadow: p as usize, n, pc_cpu, pc_gpu, inc, sampler, pages, visible, memory });
    true
}

/// The heap object was released for good: its slots go back to unified memory.
pub fn unregister(obj: usize) {
    let gone = {
        let mut heaps = HEAPS.lock().unwrap();
        heaps.iter().position(|h| h.obj == obj).map(|i| heaps.swap_remove(i))
    };
    if let Some(h) = gone {
        for (p, &f) in h.pages.iter().enumerate().filter(|&(_, &f)| f != UNMAPPED) {
            release_global(f, page_descriptors(h.n, p as u32));
        }
        match h.memory {
            Memory::Sparse => {
                unsafe { super::sparse::free(h.shadow) };
            }
            Memory::Uma => {
                if let Some(u) = crate::gpu::uma::get() {
                    u.free(h.shadow);
                }
            }
            Memory::Title => unsafe { crate::hle::kernelx::runtime_free(h.shadow, crate::hle::kernelx::D3D_DESCRIPTOR_HEAP) },
        }
    }
}

/// Start of the console slots of heap `obj` (CPU handle = GPU handle on the console).
pub fn start(obj: usize) -> Option<u64> {
    HEAPS.lock().unwrap().iter().find(|h| h.obj == obj).map(|h| h.shadow as u64)
}

/// A console slot: where its descriptor goes and its PC descriptors.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub at: usize,
    /// views: PC heap index of the SRV (the UAV is at + 1); samplers: of the sampler
    pub index: u32,
    /// PC CPU handles of the SRV / UAV where views are made (the title's heap)
    pub pc_srv: u64,
    pub pc_uav: u64,
    /// PC GPU handle of the UAV (0: heap not shader visible)
    pub pc_gpu_uav: u64,
    pub sampler: bool,
    /// CPU handle of the SRV in the shared heap (0: the heap is not in it)
    pub shared_srv: u64,
    /// the slot's number in its heap
    pub k: u32,
}

/// The slot behind a console CPU handle (None: not one of XCL's console heaps). A view slot
/// of a heap in the shared heap gets its page's place there now if the page has none yet.
pub fn slot(cpu: u64) -> Option<Slot> {
    let mut heaps = HEAPS.lock().unwrap();
    let h = heaps.iter_mut().find(|h| cpu >= h.shadow as u64 && cpu < (h.shadow + (h.n * h.slot_bytes()) as usize) as u64)?;
    let k = ((cpu - h.shadow as u64) / h.slot_bytes() as u64) as u32;
    if h.sampler {
        return Some(Slot { at: h.shadow + (k * SAMPLER_SLOT) as usize, index: k, pc_srv: h.pc_cpu + (k * h.inc) as u64, pc_uav: 0, pc_gpu_uav: 0, sampler: true, shared_srv: 0, k });
    }
    let uav = ((2 * k + 1) * h.inc) as u64;
    let shared = page_of(h, k).and_then(|(first, k0)| {
        let g = GLOBAL.lock().unwrap();
        let g = g.as_ref()?;
        let i = first + 2 * (k - k0);
        Some((i, g.gpu + (i as u64 + 1) * g.inc as u64, g.cpu + i as u64 * g.inc as u64))
    });
    let (index, pc_gpu_uav, shared_srv) = shared.unwrap_or((2 * k, if h.pc_gpu == 0 { 0 } else { h.pc_gpu + uav }, 0));
    Some(Slot { at: h.shadow + (k * SLOT) as usize, index, pc_srv: h.pc_cpu + (2 * k * h.inc) as u64, pc_uav: h.pc_cpu + uav, pc_gpu_uav, sampler: false, shared_srv, k })
}

/// The view just made in slot `s` (its SRV, or with `uav` its UAV) goes to the shared heap.
pub unsafe fn publish(s: &Slot, uav: bool) {
    if s.shared_srv == 0 {
        return;
    }
    if let Some(g) = GLOBAL.lock().unwrap().as_ref() {
        let (dst, src) = if uav { (s.shared_srv + g.inc as u64, s.pc_uav) } else { (s.shared_srv, s.pc_srv) };
        g.dev.CopyDescriptorsSimple(1, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: dst as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src as usize }, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
}

/// `n` view slots from `first` were copied (in the title's heap): they go to the shared heap,
/// a run per page (a heap's pages lie apart there).
pub unsafe fn publish_range(first: &Slot, n: u32) {
    let mut done = 0;
    while done < n {
        let s = if done == 0 { Some(*first) } else { slot((first.at + (done * SLOT) as usize) as u64) };
        let Some(s) = s else { return };
        let run = (PAGE_SLOTS - s.k % PAGE_SLOTS).min(n - done);
        if s.shared_srv != 0 {
            if let Some(g) = GLOBAL.lock().unwrap().as_ref() {
                g.dev.CopyDescriptorsSimple(2 * run, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.shared_srv as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_srv as usize }, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }
        }
        done += run;
    }
}

/// view slot address → the resource its view names (`XCL_TRACE_FRAME` diagnostics)
static RES: Mutex<Option<std::collections::HashMap<usize, usize>>> = Mutex::new(None);

/// The view in slot `s` names `res` (0: none).
pub fn note_resource(s: &Slot, res: usize) {
    if s.sampler || !super::present::trace_frame() {
        return;
    }
    let mut g = RES.lock().unwrap();
    let m = g.get_or_insert_with(Default::default);
    if res == 0 {
        m.remove(&s.at);
    } else {
        m.insert(s.at, res);
    }
}

/// The resource of the view slot at UMA address `at`.
pub fn resource_at(at: usize) -> Option<usize> {
    RES.lock().unwrap().as_ref()?.get(&at).copied()
}

pub fn write(s: &Slot, words: &[u32]) {
    let n = words.len().min(if s.sampler { 4 } else { 8 });
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), s.at as *mut u32, n) };
}

/// Copy a console slot's descriptor to another slot of the same kind, the PC heap index
/// in it (T# word 0, S# word 3 [11:0]) renumbered for the destination. Buffer V# carry
/// their address and stay as they are.
pub fn copy_words(src: &Slot, dst: &Slot) {
    let n = if src.sampler { 4 } else { 8 };
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(src.at as *const u32, w.as_mut_ptr(), n) };
    if src.sampler {
        w[3] = (w[3] & !0xFFF) | (dst.index & 0xFFF);
    } else if w[3] >> 28 >= 8 {
        w[0] = dst.index;
    }
    write(dst, &w[..n]);
    if !src.sampler {
        note_resource(dst, resource_at(src.at).unwrap_or(0));
    }
}

/// `XCL_TRACE_FRAME`: what the slots of each shader-visible view heap hold – images whose
/// PC index is the slot's own (made or copied through the device) or another (copied by the
/// title itself, as memory), buffers, empty.
pub fn census() {
    let heaps = HEAPS.lock().unwrap();
    // samplers: S# word 3 [11:0] is the PC index in the sampler heap the slot belongs to
    for h in heaps.iter().filter(|h| h.sampler) {
        let (mut empty, mut own, mut other) = (0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SAMPLER_SLOT) as usize) as *const u32, 4) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] & 0xFFF == k {
                own += 1;
            } else {
                other += 1;
                if examples.len() < 8 {
                    examples.push((k, w[3] & 0xFFF));
                }
            }
        }
        crate::xlog_info!(
            "frame trace: sampler heap {:#x} ({} slots, {}): {own} with their own index, {other} with another (slot, index: {examples:?}), {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
    for h in heaps.iter().filter(|h| !h.sampler) {
        let (mut empty, mut own, mut other, mut buffers) = (0u32, 0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        let own_index = |k: u32| match h.pages.get((k / PAGE_SLOTS) as usize) {
            Some(&f) if f != UNMAPPED => f + 2 * (k % PAGE_SLOTS),
            _ => 2 * k,
        };
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SLOT) as usize) as *const u32, 8) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] >> 28 >= 8 {
                if w[0] == own_index(k) {
                    own += 1;
                } else {
                    other += 1;
                    if examples.len() < 8 {
                        examples.push((k, w[0]));
                    }
                }
            } else {
                buffers += 1;
            }
        }
        crate::xlog_info!(
            "frame trace: view heap {:#x} ({} slots, {}): {own} images with their own PC index, {other} with another (slot, index: {examples:?}), {buffers} buffers, {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
}

/// The descriptor at a console GPU address (low 32 bits, as in a descriptor table), if it
/// is in one of the console heaps.
pub fn words_at(addr32: u32) -> Option<[u32; 8]> {
    let u = crate::gpu::uma::get()?;
    let s = slot((u.base() + addr32 as usize) as u64)?;
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(s.at as *const u32, w.as_mut_ptr(), if s.sampler { 4 } else { 8 }) };
    Some(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranges of the shared heap: released ones first (first fit, split), then new ones, and
    /// none past its size.
    #[test]
    fn shared_heap_ranges() {
        let (mut free, mut next) = (Vec::new(), 0u32);
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(0));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(40));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), None, "20 left");
        free.push((0, 40));
        assert_eq!(take_range(&mut free, &mut next, 100, 30), Some(0));
        assert_eq!(free, vec![(30, 10)]);
        assert_eq!(take_range(&mut free, &mut next, 100, 20), Some(80));
        assert_eq!(take_range(&mut free, &mut next, 100, 10), Some(30));
        assert!(free.is_empty());
    }

    #[test]
    fn window_views_do_not_hold_global_during_device_calls() {
        const CHILD: &str = "XCL_TEST_WINDOW_VIEWS_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::window_views_do_not_hold_global_during_device_calls", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        static UNLOCKED: AtomicBool = AtomicBool::new(true);
        static CALLS: AtomicU32 = AtomicU32::new(0);
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        unsafe extern "system" fn create_uav(_this: *mut c_void, _resource: *mut c_void, _counter: *mut c_void, _desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, _at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            // The adopted device hook can re-enter descriptor bookkeeping.
            UNLOCKED.fetch_and(GLOBAL.try_lock().is_ok(), Ordering::Relaxed);
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let mut vt = [0usize; 20];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        vt[19] = create_uav as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut heap = Object { vt: vt.as_ptr() };
        let mut resource = Object { vt: vt.as_ptr() };
        unsafe {
            let dev = ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void);
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: dev.clone(),
                heap: ID3D12DescriptorHeap::from_raw(&mut heap as *mut _ as *mut c_void),
                cpu: 0x1000, gpu: 0x2000, inc: 32, cap: 1024,
                next: crate::gpu::uma::MAX_WINDOWS as u32, free: Vec::new(),
            });
            assert!(set_window_views(&dev, &[(&mut resource as *mut _ as usize, 0x10000)]));
            assert_eq!(CALLS.load(Ordering::Relaxed), crate::gpu::uma::MAX_WINDOWS as u32);
            assert!(UNLOCKED.load(Ordering::Relaxed), "device callback ran while GLOBAL was locked");
            *GLOBAL.lock().unwrap() = None;
        }
    }

    #[test]
    fn slots_copy_and_release() {
        let Some(u) = crate::gpu::uma::get() else { return };
        let rel = |p: u64| (p as usize - u.base()) as u32;
        // views: 4 slots over a PC heap at 0x1000 (increment 32, GPU 0x5000); samplers: 2 at 0x9000
        assert!(register(0xA1, 4, 0x1000, 0x5000, 32, false, false, true));
        assert!(register(0xA2, 2, 0x9000, 0, 16, true, false, true));
        let (v, sm) = (start(0xA1).unwrap(), start(0xA2).unwrap());
        assert_eq!(u.window_of(v as usize), Some(0), "view heap must live in window 0 (32-bit table addressing)");
        assert_eq!(u.window_of(sm as usize), Some(0), "sampler heap must live in window 0");
        assert!(u.is_sparse(v as usize) && u.is_sparse(sm as usize), "descriptor heaps backed by sparse VRAM");
        let s1 = slot(v + 32).unwrap();
        assert_eq!((s1.index, s1.pc_srv, s1.pc_uav, s1.pc_gpu_uav), (2, 0x1040, 0x1060, 0x5060));
        // a 2D T# from slot 1 to slot 3: word 0 = the destination's SRV index
        write(&s1, &[2, 0x0470_0000, 0, 9 << 28, 0, 0, 0, 0]);
        let s3 = slot(v + 96).unwrap();
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], [6, 0x0470_0000, 0, 9 << 28]);
        // a buffer V# keeps its address
        write(&s1, &[0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[0], 0x1234);
        // a raytracing BVH SRSRC keeps its base address and box-sort flag verbatim
        let bvh = crate::gpu::desc::bvh_words(0x12_3456_7800, true);
        write(&s1, &[bvh[0], bvh[1], bvh[2], bvh[3], 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], bvh);
        // samplers: 16-byte slots, S# word 3 [11:0] = sampler index
        let (q0, q1) = (slot(sm).unwrap(), slot(sm + 16).unwrap());
        assert!(q1.sampler && q1.index == 1 && q1.pc_srv == 0x9010);
        write(&q0, &[1, 2, 3, 0xABC0_0000]);
        copy_words(&q0, &q1);
        assert_eq!(words_at(rel(sm + 16)).unwrap()[..4], [1, 2, 3, 0xABC0_0001]);
        assert!(slot(sm + 32).is_none(), "past the last sampler slot");
        unregister(0xA1);
        unregister(0xA2);
        assert!(start(0xA1).is_none() && slot(v).is_none() && slot(sm).is_none());
    }

    /// Minecraft's view heaps: eight of 73 727 slots, 147 454 PC descriptors each – more in all
    /// than an NVIDIA shared heap holds (1 000 000). Each gets a place there for the pages it
    /// uses; before, the seventh and eighth got none and their views were lost (flicker).
    #[test]
    fn large_heaps_get_shared_pages_as_used() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        if crate::gpu::uma::get().is_none() {
            return;
        }
        let dev = unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            dev.expect("D3D12 device")
        };
        assert!(unsafe { shared_heap(&dev) });
        const N: u32 = 73_727;
        let objs: Vec<usize> = (0..8).map(|i| 0xB000 + i).collect();
        for &o in &objs {
            assert!(register(o, N, 0x10_0000, 0, 32, false, true, true));
            assert!(in_global(o));
        }
        let mut seen = Vec::new();
        for &o in &objs {
            let v = start(o).unwrap();
            // slot 0, the last of page 0, the first of page 1 and the heap's last slot
            let s: Vec<Slot> = [0, PAGE_SLOTS - 1, PAGE_SLOTS, N - 1].iter().map(|&k| slot(v + (k * SLOT) as u64).unwrap()).collect();
            assert!(s.iter().all(|x| x.shared_srv != 0), "heap {o:#x}: a slot without a place in the shared heap");
            assert_eq!(s[1].index, s[0].index + 2 * (PAGE_SLOTS - 1), "a page is contiguous");
            assert_eq!(s[0].pc_srv, 0x10_0000, "views are still made in the title's heap");
            assert_eq!(s[3].pc_srv, 0x10_0000 + (2 * (N - 1) * 32) as u64);
            for x in &s {
                assert!(!seen.contains(&x.index), "two slots share the PC index {}", x.index);
                seen.push(x.index);
            }
            // the same slot again: the same place
            assert_eq!(slot(v).unwrap().index, s[0].index);
        }
        for &o in &objs {
            unregister(o);
        }
        // a heap smaller than a page takes 2 descriptors per slot, as it did when a heap's
        // whole range was reserved (a title with many small heaps must not fill the shared heap)
        assert_eq!((page_descriptors(4, 0), page_descriptors(N, 0), page_descriptors(N, N / PAGE_SLOTS)), (8, 2 * PAGE_SLOTS, 2 * (N % PAGE_SLOTS)));
        assert!(register(0xB100, 4, 0x10_0000, 0, 32, false, true, true) && register(0xB101, 4, 0x20_0000, 0, 32, false, true, true));
        let (a, b) = (slot(start(0xB100).unwrap()).unwrap(), slot(start(0xB101).unwrap()).unwrap());
        assert!(a.shared_srv != 0 && b.shared_srv != 0);
        assert_eq!(a.index.abs_diff(b.index), 8, "two heaps of 4 slots lie 8 descriptors apart");
        unregister(0xB100);
        unregister(0xB101);
    }
}
